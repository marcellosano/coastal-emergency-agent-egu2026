"""Live-mode daemon — apscheduler driving the orchestrator on real data.

Per tick (hourly by default):

  1. Fetch real Open-Meteo forecast for Lido (`weather.fetch_lido_forecast`).
  2. Run live GAT-PPO inference (`policy.LivePolicy.build_scenario_record`).
  3. Hand the resulting ScenarioRecord to `Orchestrator.process_tick` —
     this is the same orchestrator the replay path uses; lifecycle,
     dedup, brief composition, email moments are all unchanged.
  4. Update the heartbeat row.

Failure isolation:
  - If the weather call fails, log + skip this tick (don't burn LLM
    spend on stale data and don't crash the daemon).
  - If the policy raises, log + skip this tick (the same).
  - If the orchestrator raises mid-tick, log + raise (the daemon is
    crash-restartable via systemd `Restart=on-failure`).
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger

from coastal_agent.config import Settings
from coastal_agent.email_send import (
    build_sender_from_settings,
    parse_recipients,
)
from coastal_agent.llm import LLMComposer, build_default_composer
from coastal_agent.orchestrator import Orchestrator
from coastal_agent.policy import LivePolicy
from coastal_agent.trigger import TriggerConfig
from coastal_agent.weather import WeatherFetchError, fetch_lido_forecast


log = logging.getLogger(__name__)


@dataclass
class Daemon:
    """One-process live agent: weather → policy → orchestrator → heartbeat."""

    conn: sqlite3.Connection
    orchestrator: Orchestrator
    policy: LivePolicy
    poll_interval_seconds: int
    case_study: str = "lido"

    @classmethod
    def from_settings(
        cls,
        conn: sqlite3.Connection,
        settings: Settings,
    ) -> "Daemon":
        composer: LLMComposer = build_default_composer(conn, settings)
        trigger_config = TriggerConfig(
            case_study="lido",
            activation_surge_cm=settings.trigger_activation_surge_cm,
            standdown_surge_cm=settings.trigger_standdown_surge_cm,
            standdown_consecutive_ticks=settings.trigger_standdown_consecutive_ticks,
        )
        email_sender = build_sender_from_settings(settings)
        email_to = parse_recipients(settings.email_to)
        orchestrator = Orchestrator(
            conn=conn,
            trigger_config=trigger_config,
            composer=composer,
            vendor_dir=Path(settings.vendor_dir),
            corpus_dir=Path(settings.corpus_dir),
            live_data=True,
            email_sender=email_sender,
            email_to=email_to,
            dashboard_base_url=settings.dashboard_base_url,
        )
        policy = LivePolicy(vendor_dir=Path(settings.vendor_dir))
        return cls(
            conn=conn,
            orchestrator=orchestrator,
            policy=policy,
            poll_interval_seconds=settings.poll_interval_seconds,
            case_study="lido",
        )

    # -- runtime ------------------------------------------------------

    def tick(self) -> None:
        """One scheduler tick. Bounded — every failure is caught and logged
        so the scheduler keeps firing on the next interval."""
        try:
            self._tick_inner()
        except Exception:
            log.exception("daemon tick failed; will retry at next interval")

    def _tick_inner(self) -> None:
        now = datetime.now(timezone.utc)
        log.info("tick start at %s", now.isoformat())

        # 1. Forecast.
        try:
            forecast_fetch = fetch_lido_forecast(now=now)
        except WeatherFetchError as e:
            log.warning("weather fetch failed: %s; skipping tick", e)
            return
        forecast = forecast_fetch.snapshot
        log.info(
            "forecast: surge=%.1fcm wind=%.1fm/s wave=%.2fm rain=%.1fmm",
            forecast.surge_cm, forecast.wind_ms, forecast.wave_m, forecast.rainfall_mm,
        )

        # 2. Tick number — monotonic across daemon restarts via heartbeat.
        tick_n = self._next_tick_number()

        # 3. Live policy inference.
        record = self.policy.build_scenario_record(
            forecast,
            tick=tick_n,
            simulated_time=now,
        )
        top_idx = max(
            range(len(record.policy_output.action_probs)),
            key=lambda i: record.policy_output.action_probs[i],
        )
        log.info(
            "policy: top action idx=%d prob=%.3f value=%.3f",
            top_idx,
            record.policy_output.action_probs[top_idx],
            record.policy_output.value_estimate,
        )

        # 4. Orchestrator — same path as replay; fires brief composer and
        #    incident lifecycle when the trigger says so.
        # Tools that fetch real data need ctx.live_data=True. We surface
        # that by setting the orchestrator's upcoming_records and a flag —
        # use the orchestrator's run() which materialises records, or pass
        # a single-record list. For one tick, set the flag manually.
        prev = self.orchestrator._upcoming_records  # noqa: SLF001 — internal coordination
        self.orchestrator._upcoming_records = []
        try:
            tick_result = self.orchestrator.process_tick(record)
        finally:
            self.orchestrator._upcoming_records = prev

        # 5. Heartbeat.
        self._update_heartbeat(now)
        log.info(
            "tick %d done: incident=%s events=%s",
            tick_n,
            tick_result.incident_id,
            tick_result.events,
        )

    def start_blocking(self) -> None:
        scheduler = BlockingScheduler(timezone="UTC")
        scheduler.add_job(
            self.tick,
            trigger=IntervalTrigger(seconds=self.poll_interval_seconds),
            id="coastal_agent_tick",
            next_run_time=datetime.now(timezone.utc),  # fire immediately
            coalesce=True,
            max_instances=1,
        )
        log.info(
            "daemon starting blocking scheduler; interval=%ds",
            self.poll_interval_seconds,
        )
        try:
            scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            log.info("daemon shutting down on interrupt")

    # -- internals ----------------------------------------------------

    def _next_tick_number(self) -> int:
        row = self.conn.execute(
            "SELECT tick_count FROM heartbeat WHERE id=1"
        ).fetchone()
        current = int(row["tick_count"]) if row is not None else 0
        return current

    def _update_heartbeat(self, now: datetime) -> None:
        self.conn.execute(
            "UPDATE heartbeat SET last_tick_at=?, tick_count=tick_count+1 "
            "WHERE id=1",
            (now.isoformat(),),
        )
