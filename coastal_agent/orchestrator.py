"""Orchestrator — drives the incident lifecycle.

Per-tick (when given a `ScenarioRecord`):

  1. Write a `forecast` row (linked to active `incident_id` if one exists).
  2. If currently standby: check activation trigger. If it fires, open an
     incident.
  3. If active: write an `evaluation` row, compose a `brief` (LLM if a
     composer is wired in, stub otherwise), send an email at the right
     moment (activation / new_alert).
  4. If active: check stand-down. If it fires, close the incident and
     send a stand-down email.

Brief composition (M5):
  - When `composer` is None, writes the placeholder stub brief that
     downstream code (dashboard, email) already expects.
  - When `composer` is set (production path), runs the LLM tool-use
     loop via `LLMComposer.compose_brief` and persists the structured
     output. Kill switches inside the composer fall back to stub
     transparently.

Email is **mock-only** in this milestone — every send writes a `sent_emails`
row with `mode='mock'`. Real-SMTP swap is M7 (config flip, not a rewrite).

Dedup is **loose, per-incident**: an action ID emails at most once during
an incident. `monitor` never emails (it's the resting state, not an alert).
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable

from coastal_agent.brief import Brief, Recommendation
from coastal_agent.email_send import EmailSendError, EmailSender
from coastal_agent.llm import ComposeResult, LLMComposer
from coastal_agent.policy import confidence_signal, recommended_action_id
from coastal_agent.scenario import ScenarioRecord
from coastal_agent.tools import ToolContext
from coastal_agent.trigger import (
    TriggerConfig,
    TriggerEvaluation,
    evaluate_activation,
    evaluate_standdown,
)


log = __import__("logging").getLogger(__name__)


@dataclass
class IncidentState:
    """Mutable state for one active incident, owned by the Orchestrator."""

    incident_id: int
    case_study: str
    opened_at_tick: int
    opened_at_simulated: datetime
    seen_action_ids: set[str] = field(default_factory=set)
    consecutive_below_standdown: int = 0


@dataclass
class TickResult:
    """Per-tick summary of what the orchestrator did."""

    tick: int
    forecast_id: int
    evaluation_id: int | None
    brief_id: int | None
    incident_id: int | None
    events: list[str]


@dataclass
class ReplaySummary:
    """End-of-replay aggregate counts."""

    ticks_processed: int
    incidents_opened: int
    incidents_closed: int
    evaluations_written: int
    briefs_written: int
    emails_sent: int


class Orchestrator:
    """Drives the incident lifecycle for one case study against a stream of
    scenario records (replay mode) or live policy outputs (live mode).
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        trigger_config: TriggerConfig | None = None,
        composer: LLMComposer | None = None,
        vendor_dir: Path | None = None,
        corpus_dir: Path | None = None,
        live_data: bool = False,
        email_sender: EmailSender | None = None,
        email_to: list[str] | None = None,
        dashboard_base_url: str = "",
    ) -> None:
        self.conn = conn
        self.trigger_config = trigger_config or TriggerConfig()
        self.composer = composer
        self.vendor_dir = vendor_dir or Path("external/gnn_drl_ews_v003_seed2")
        self.corpus_dir = corpus_dir or Path("corpus")
        self.live_data = live_data
        self.email_sender = email_sender
        self.email_to = list(email_to) if email_to else []
        self.dashboard_base_url = dashboard_base_url.rstrip("/")
        self.active: IncidentState | None = None
        # Set by run() when a composer is present so fetch_forecast_detail
        # can look ahead. None when process_tick is called without context.
        self._upcoming_records: list[ScenarioRecord] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_tick(self, record: ScenarioRecord) -> TickResult:
        events: list[str] = []

        # Step 1 — forecast row (incident_id null until activation)
        forecast_id = self._write_forecast(record, self._current_incident_id())

        # Step 2 — activation check, only if currently standby
        if self.active is None:
            activation = evaluate_activation(record.forecast, self.trigger_config)
            if activation.fired:
                self.active = self._open_incident(record, activation)
                self._link_forecast_to_incident(forecast_id, self.active.incident_id)
                events.append("incident_opened")

        evaluation_id: int | None = None
        brief_id: int | None = None

        # Step 3 — process active tick
        if self.active is not None:
            evaluation_id = self._write_evaluation(
                record, forecast_id, self.active.incident_id
            )
            recommended = recommended_action_id(record.policy_output)
            brief_id = self._compose_and_write_brief(
                record, evaluation_id, self.active, recommended
            )
            events.append("brief_written")

            # Email decision — first non-monitor action of the incident => activation;
            # subsequent new distinct non-monitor actions => new_alert; monitor => silent.
            if recommended != "monitor":
                if not self.active.seen_action_ids:
                    self._send_email(
                        record,
                        moment="activation",
                        recommended=recommended,
                        brief_id=brief_id,
                    )
                    events.append("email_activation")
                elif recommended not in self.active.seen_action_ids:
                    self._send_email(
                        record,
                        moment="new_alert",
                        recommended=recommended,
                        brief_id=brief_id,
                    )
                    events.append("email_new_alert")
                self.active.seen_action_ids.add(recommended)

            # Step 4 — stand-down check
            standdown = evaluate_standdown(
                record.forecast,
                self.active.consecutive_below_standdown,
                self.trigger_config,
            )
            self.active.consecutive_below_standdown = int(
                standdown.values["consecutive_below_after_this_tick"]
            )
            if standdown.fired:
                self._close_incident(self.active.incident_id, record, standdown)
                self._send_email(
                    record,
                    moment="standdown",
                    recommended=None,
                    brief_id=None,
                )
                events.append("incident_closed")
                events.append("email_standdown")
                self.active = None

        return TickResult(
            tick=record.tick,
            forecast_id=forecast_id,
            evaluation_id=evaluation_id,
            brief_id=brief_id,
            incident_id=self._current_incident_id(),
            events=events,
        )

    def run(self, records: Iterable[ScenarioRecord]) -> ReplaySummary:
        """Process a stream of scenario records, return aggregate counts."""
        materialised = list(records)
        ticks = 0
        opens = closes = evals = briefs = emails = 0
        for idx, r in enumerate(materialised):
            self._upcoming_records = materialised[idx + 1 : idx + 25]
            tick_result = self.process_tick(r)
            ticks += 1
            if tick_result.evaluation_id is not None:
                evals += 1
            if tick_result.brief_id is not None:
                briefs += 1
            for ev in tick_result.events:
                if ev == "incident_opened":
                    opens += 1
                elif ev == "incident_closed":
                    closes += 1
                elif ev.startswith("email_"):
                    emails += 1
        return ReplaySummary(
            ticks_processed=ticks,
            incidents_opened=opens,
            incidents_closed=closes,
            evaluations_written=evals,
            briefs_written=briefs,
            emails_sent=emails,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _current_incident_id(self) -> int | None:
        return self.active.incident_id if self.active else None

    def _write_forecast(
        self, record: ScenarioRecord, incident_id: int | None
    ) -> int:
        cursor = self.conn.execute(
            "INSERT INTO forecasts(incident_id, source, location, raw_json, "
            "parsed_json, fetched_at) VALUES (?, 'replay', ?, ?, ?, ?)",
            (
                incident_id,
                self.trigger_config.case_study,
                record.forecast.model_dump_json(),
                record.forecast.model_dump_json(),
                record.simulated_time.isoformat(),
            ),
        )
        return int(cursor.lastrowid)

    def _link_forecast_to_incident(
        self, forecast_id: int, incident_id: int
    ) -> None:
        self.conn.execute(
            "UPDATE forecasts SET incident_id=? WHERE id=? AND incident_id IS NULL",
            (incident_id, forecast_id),
        )

    def _open_incident(
        self, record: ScenarioRecord, activation: TriggerEvaluation
    ) -> IncidentState:
        cursor = self.conn.execute(
            "INSERT INTO incidents(case_study, opened_at, trigger_condition, status) "
            "VALUES (?, ?, ?, 'active')",
            (
                self.trigger_config.case_study,
                record.simulated_time.isoformat(),
                json.dumps(
                    {
                        "rule": activation.rule,
                        "summary": activation.summary,
                        "values": activation.values,
                        "tick": record.tick,
                    }
                ),
            ),
        )
        return IncidentState(
            incident_id=int(cursor.lastrowid),
            case_study=self.trigger_config.case_study,
            opened_at_tick=record.tick,
            opened_at_simulated=record.simulated_time,
        )

    def _close_incident(
        self,
        incident_id: int,
        record: ScenarioRecord,
        standdown: TriggerEvaluation,
    ) -> None:
        self.conn.execute(
            "UPDATE incidents SET status='closed', closed_at=?, notes=? WHERE id=?",
            (
                record.simulated_time.isoformat(),
                json.dumps(
                    {
                        "standdown_summary": standdown.summary,
                        "standdown_values": standdown.values,
                        "closed_at_tick": record.tick,
                    }
                ),
                incident_id,
            ),
        )

    def _write_evaluation(
        self, record: ScenarioRecord, forecast_id: int, incident_id: int
    ) -> int:
        cursor = self.conn.execute(
            "INSERT INTO evaluations(incident_id, forecast_id, action_probs_json, "
            "value_estimate, confidence, storm_type, mask_json, evaluated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                incident_id,
                forecast_id,
                json.dumps(record.policy_output.action_probs),
                record.policy_output.value_estimate,
                max(record.policy_output.action_probs),
                record.policy_output.storm_type,
                json.dumps(record.mask),
                record.simulated_time.isoformat(),
            ),
        )
        return int(cursor.lastrowid)

    def _compose_and_write_brief(
        self,
        record: ScenarioRecord,
        evaluation_id: int,
        incident: IncidentState,
        recommended: str,
    ) -> int:
        if self.composer is not None:
            ctx = ToolContext(
                record=record,
                incident_id=incident.incident_id,
                case_study=incident.case_study,
                conn=self.conn,
                vendor_dir=self.vendor_dir,
                corpus_dir=self.corpus_dir,
                upcoming_records=list(self._upcoming_records),
                live_data=self.live_data,
            )
            result: ComposeResult = self.composer.compose_brief(
                record=record,
                recommended_action_id=recommended,
                case_study=incident.case_study,
                incident_id=incident.incident_id,
                opened_at=incident.opened_at_simulated,
                ctx=ctx,
            )
            brief_payload = result.brief.model_dump()
        else:
            brief_payload = {
                "incident_id": incident.incident_id,
                "tick": record.tick,
                "recommendation": {
                    "action_id": recommended,
                    "confidence_signal": confidence_signal(record.policy_output),
                    "rationale": "[stub — no composer wired in]",
                },
                "citations": [],
                "precondition_check": [],
                "concerns": [],
                "open_questions": [],
                "tool_calls": [],
            }

        cursor = self.conn.execute(
            "INSERT INTO briefs(incident_id, evaluation_id, tick, brief_json, composed_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                incident.incident_id,
                evaluation_id,
                record.tick,
                json.dumps(brief_payload),
                record.simulated_time.isoformat(),
            ),
        )
        return int(cursor.lastrowid)

    def _read_brief_payload(self, brief_id: int | None) -> dict | None:
        if brief_id is None:
            return None
        row = self.conn.execute(
            "SELECT brief_json FROM briefs WHERE id=?", (brief_id,)
        ).fetchone()
        if row is None:
            return None
        try:
            return json.loads(row["brief_json"])
        except json.JSONDecodeError:
            return None

    def _send_email(
        self,
        record: ScenarioRecord,
        moment: str,
        recommended: str | None,
        brief_id: int | None,
    ) -> int:
        assert self.active is not None
        incident_id = self.active.incident_id

        if moment == "activation":
            assert recommended is not None
            subject = (
                f"[INCIDENT {incident_id}] {self.active.case_study} — "
                f"surge {record.forecast.surge_cm:.0f} cm — incident opened — "
                f"first recommendation: {recommended}"
            )
        elif moment == "new_alert":
            assert recommended is not None
            subject = (
                f"[INCIDENT {incident_id}] {self.active.case_study} — "
                f"new recommendation: {recommended}"
            )
        elif moment == "standdown":
            subject = (
                f"[INCIDENT {incident_id}] {self.active.case_study} — "
                f"conditions clearing — incident closed"
            )
        else:
            raise ValueError(f"unknown email moment: {moment!r}")

        deeplink_path = (
            f"/incidents/{incident_id}/briefs/{brief_id}"
            if brief_id is not None
            else f"/incidents/{incident_id}"
        )
        deeplink = (
            f"{self.dashboard_base_url}{deeplink_path}"
            if self.dashboard_base_url
            else deeplink_path
        )
        body = self._render_email_body(
            record, moment, recommended, brief_id, deeplink,
        )

        # Decide mode and (when real) attempt delivery before recording.
        if self.email_sender is not None and self.email_to:
            mode = "real"
            try:
                self.email_sender.send(
                    subject=subject, body=body, to=list(self.email_to),
                )
                delivery_status: str | None = "sent"
            except EmailSendError as e:
                log.warning(
                    "real-mode email send failed (%s); recording with delivery_status=failed",
                    e,
                )
                delivery_status = f"failed:{e}"[:200]
        else:
            mode = "mock"
            delivery_status = None

        cursor = self.conn.execute(
            "INSERT INTO sent_emails(incident_id, brief_id, moment, subject, body, "
            "deeplink, mode, delivery_status, sent_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                incident_id,
                brief_id,
                moment,
                subject,
                body,
                deeplink,
                mode,
                delivery_status,
                record.simulated_time.isoformat(),
            ),
        )
        return int(cursor.lastrowid)

    def _render_email_body(
        self,
        record: ScenarioRecord,
        moment: str,
        recommended: str | None,
        brief_id: int | None,
        deeplink: str,
    ) -> str:
        forecast = record.forecast
        header = (
            f"Forecast: surge={forecast.surge_cm:.0f}cm, wind={forecast.wind_ms:.1f}m/s, "
            f"wave={forecast.wave_m:.2f}m  ({record.simulated_time.isoformat()})"
        )
        if moment == "standdown":
            return (
                f"{header}\n\nConditions clearing — incident closed.\n\n"
                f"Full timeline: {deeplink}"
            )

        brief = self._read_brief_payload(brief_id)
        if brief is None:
            return (
                f"{header}\n\nRecommended: {recommended}\n"
                f"View on the dashboard: {deeplink}"
            )
        rec = brief.get("recommendation", {}) or {}
        rationale = (rec.get("rationale") or "").strip()
        concerns = brief.get("concerns") or []
        n_citations = len(brief.get("citations") or [])
        lines = [
            header,
            "",
            f"Recommended: {rec.get('action_id', recommended)}  "
            f"(confidence: {rec.get('confidence_signal', '?')})",
        ]
        if rationale:
            lines.extend(["", rationale])
        if concerns:
            lines.append("")
            lines.append("Concerns:")
            for c in concerns[:5]:
                lines.append(f"  - {c}")
        if n_citations:
            lines.extend(["", f"Citations: {n_citations} (full text on dashboard)"])
        lines.extend(["", f"Open the brief: {deeplink}"])
        return "\n".join(lines)
