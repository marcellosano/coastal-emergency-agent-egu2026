"""Tests for the LLMComposer — fake OpenAI client, kill switches, retries.

The fake client is scripted: each call to chat.completions.create returns
the next pre-canned response. Tool-call rounds emit synthetic tool_calls
that the composer dispatches; the final round returns assistant content
with the JSON brief.

Tests are offline; no real network calls.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from coastal_agent.brief import Brief
from coastal_agent.db import connect, init_schema
from coastal_agent.llm import BriefBudget, LLMComposer
from coastal_agent.scenario import (
    ForecastSnapshot,
    GlobalState,
    PolicyOutput,
    ScenarioRecord,
)
from coastal_agent.tools import ToolContext

VENDOR_DIR = Path("external/gnn_drl_ews_v003_seed2")
HAS_PLAN = (VENDOR_DIR / "plans" / "lido.yaml").exists()


# ----------------------------------------------------------------
# Fake OpenAI client
# ----------------------------------------------------------------


def _fake_tool_call(call_id: str, name: str, args: dict) -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=json.dumps(args)),
    )


def _fake_response(content: str | None, tool_calls: list | None = None) -> SimpleNamespace:
    message = SimpleNamespace(content=content, tool_calls=tool_calls or None)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice])


class ScriptedClient:
    """Scripted OpenAI-compatible chat client.

    Pass a list of responses; each call to .chat.completions.create
    returns the next one. Records the last messages payload so tests
    can assert on prompt content if needed.
    """

    def __init__(self, responses: list[SimpleNamespace]) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

        outer = self

        class _Completions:
            def create(self, **kwargs):
                outer.calls.append(kwargs)
                if not outer._responses:
                    raise RuntimeError("ScriptedClient: no responses left")
                return outer._responses.pop(0)

        self.chat = SimpleNamespace(completions=_Completions())


# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------


def _record(tick: int = 7, surge_cm: float = 112.0) -> ScenarioRecord:
    return ScenarioRecord(
        tick=tick,
        simulated_time=datetime(2026, 11, 12, 8 + tick, 0),
        forecast=ForecastSnapshot(surge_cm=surge_cm, wind_ms=12.0, wave_m=1.5),
        state=GlobalState(forecast_tide=surge_cm, forecast_wind_wave=12.0,
                          resources=5.0),
        mask=[True, True, False, False, False, False, False],
        policy_output=PolicyOutput(
            action_probs=[0.2, 0.7, 0.05, 0.025, 0.025, 0.0, 0.0],
            value_estimate=-1.5,
            storm_type="tide",
        ),
    )


def _ctx(tmp_path: Path, record=None, incident_id: int = 1) -> ToolContext:
    conn = connect(tmp_path / "llm.db")
    init_schema(conn)
    return ToolContext(
        record=record or _record(),
        incident_id=incident_id,
        case_study="lido",
        conn=conn,
        vendor_dir=VENDOR_DIR,
        corpus_dir=Path("corpus"),
        upcoming_records=[],
    )


def _budget(tmp_path: Path, max_per_day: int = 50) -> BriefBudget:
    conn = connect(tmp_path / "budget.db")
    init_schema(conn)
    return BriefBudget(conn, max_per_day)


# ----------------------------------------------------------------
# Kill-switch and budget paths
# ----------------------------------------------------------------


def test_compose_brief_returns_stub_when_disabled(tmp_path: Path) -> None:
    composer = LLMComposer(
        client=ScriptedClient([]),
        model="anthropic/claude-sonnet-4-5",
        budget=_budget(tmp_path),
        enabled=False,
    )
    rec = _record()
    result = composer.compose_brief(
        record=rec,
        recommended_action_id="issue_alert",
        case_study="lido",
        incident_id=1,
        opened_at=rec.simulated_time,
        ctx=_ctx(tmp_path, record=rec),
    )
    assert result.is_stub is True
    assert result.reason == "kill_switch_disabled"
    assert result.brief.recommendation.action_id == "issue_alert"


def test_compose_brief_returns_stub_when_no_client(tmp_path: Path) -> None:
    composer = LLMComposer(
        client=None,
        model="anthropic/claude-sonnet-4-5",
        budget=_budget(tmp_path),
        enabled=True,
    )
    rec = _record()
    result = composer.compose_brief(
        record=rec,
        recommended_action_id="issue_alert",
        case_study="lido",
        incident_id=1,
        opened_at=rec.simulated_time,
        ctx=_ctx(tmp_path, record=rec),
    )
    assert result.is_stub is True
    assert result.reason == "no_llm_client"


def test_compose_brief_returns_stub_when_budget_exhausted(tmp_path: Path) -> None:
    composer = LLMComposer(
        client=ScriptedClient([]),
        model="anthropic/claude-sonnet-4-5",
        budget=_budget(tmp_path, max_per_day=0),
        enabled=True,
    )
    rec = _record()
    result = composer.compose_brief(
        record=rec,
        recommended_action_id="issue_alert",
        case_study="lido",
        incident_id=1,
        opened_at=rec.simulated_time,
        ctx=_ctx(tmp_path, record=rec),
    )
    assert result.is_stub is True
    assert result.reason == "daily_cap_reached"


# ----------------------------------------------------------------
# Budget tracker
# ----------------------------------------------------------------


def test_brief_budget_counts_only_non_stub_briefs(tmp_path: Path) -> None:
    conn = connect(tmp_path / "budget.db")
    init_schema(conn)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%dT12:00:00")
    incident_id = conn.execute(
        "INSERT INTO incidents(case_study, opened_at, trigger_condition, status) "
        "VALUES ('lido', ?, '{}', 'closed')",
        (today,),
    ).lastrowid

    real_brief = json.dumps({
        "incident_id": incident_id, "tick": 1,
        "recommendation": {"action_id": "issue_alert",
                           "confidence_signal": "high",
                           "rationale": "Real LLM rationale here."},
        "citations": [], "precondition_check": [],
        "concerns": [], "open_questions": [], "tool_calls": [],
    })
    stub_brief = json.dumps({
        "incident_id": incident_id, "tick": 2,
        "recommendation": {"action_id": "monitor",
                           "confidence_signal": "high",
                           "rationale": "[stub: kill_switch_disabled]"},
        "citations": [], "precondition_check": [],
        "concerns": [], "open_questions": [], "tool_calls": [],
    })
    conn.execute(
        "INSERT INTO briefs(incident_id, evaluation_id, tick, brief_json, composed_at) "
        "VALUES (?, NULL, 1, ?, ?)",
        (incident_id, real_brief, today),
    )
    conn.execute(
        "INSERT INTO briefs(incident_id, evaluation_id, tick, brief_json, composed_at) "
        "VALUES (?, NULL, 2, ?, ?)",
        (incident_id, stub_brief, today),
    )

    budget = BriefBudget(conn, max_per_day=2)
    # 1 real brief used → 1 remaining (stub doesn't count).
    assert budget.remaining_today() == 1
    assert budget.can_compose() is True


# ----------------------------------------------------------------
# Happy path with mocked tool round + final JSON
# ----------------------------------------------------------------


@pytest.mark.skipif(not HAS_PLAN, reason="vendored plan YAML not present")
def test_compose_brief_happy_path_two_round_tool_loop(tmp_path: Path) -> None:
    """Round 1: model calls get_plan_provision. Round 2: model returns
    final JSON brief. Composer dispatches the tool, validates the brief,
    and returns ComposeResult with is_stub=False."""

    final_payload = {
        "incident_id": 1,
        "tick": 7,
        "recommendation": {
            "action_id": "issue_alert",
            "confidence_signal": "high",
            "rationale": "Surge has crossed the alert threshold; PCE §4.1.4.9 directs the Sindaco to issue a public alert.",
        },
        "citations": [
            {
                "source_type": "plan_provision",
                "ref": "PCE_Comune_Venezia_v1_2008 §4.1.4.9 p41",
            }
        ],
        "precondition_check": [
            {
                "condition": "global.forecast_tide >= threshold.alert_tide",
                "satisfied": True,
                "evidence": "112cm >= 110cm",
            }
        ],
        "concerns": ["Lido vaporetto schedule may be disrupted."],
        "open_questions": [],
        "tool_calls": [],
    }

    responses = [
        _fake_response(
            content=None,
            tool_calls=[
                _fake_tool_call(
                    "call_1",
                    "get_plan_provision",
                    {"action_id": "issue_alert", "case_study": "lido"},
                ),
            ],
        ),
        _fake_response(content=json.dumps(final_payload)),
    ]
    client = ScriptedClient(responses)

    composer = LLMComposer(
        client=client,
        model="anthropic/claude-sonnet-4-5",
        budget=_budget(tmp_path),
        enabled=True,
    )
    rec = _record()
    result = composer.compose_brief(
        record=rec,
        recommended_action_id="issue_alert",
        case_study="lido",
        incident_id=1,
        opened_at=rec.simulated_time,
        ctx=_ctx(tmp_path, record=rec),
    )
    assert result.is_stub is False
    assert isinstance(result.brief, Brief)
    assert result.brief.recommendation.action_id == "issue_alert"
    assert len(result.brief.citations) == 1
    # Audit trail captures the one tool call.
    assert len(result.brief.tool_calls) == 1
    assert result.brief.tool_calls[0].name == "get_plan_provision"
    # Two LLM round-trips happened.
    assert len(client.calls) == 2


# ----------------------------------------------------------------
# Drift defenses: model can't override recommended action
# ----------------------------------------------------------------


def test_compose_brief_overrides_drifted_action_id(tmp_path: Path) -> None:
    """If the model writes a different action_id, the composer
    forces it back to the recommended_action_id."""
    drifted = {
        "incident_id": 1,
        "tick": 7,
        "recommendation": {
            "action_id": "full_evacuation",  # WRONG — model drifting
            "confidence_signal": "high",
            "rationale": "I think we should evacuate everyone now.",
        },
        "citations": [],
        "precondition_check": [],
        "concerns": [],
        "open_questions": [],
        "tool_calls": [],
    }
    client = ScriptedClient([_fake_response(content=json.dumps(drifted))])
    composer = LLMComposer(
        client=client,
        model="anthropic/claude-sonnet-4-5",
        budget=_budget(tmp_path),
        enabled=True,
    )
    rec = _record()
    result = composer.compose_brief(
        record=rec,
        recommended_action_id="issue_alert",
        case_study="lido",
        incident_id=1,
        opened_at=rec.simulated_time,
        ctx=_ctx(tmp_path, record=rec),
    )
    assert result.is_stub is False
    assert result.brief.recommendation.action_id == "issue_alert"  # forced


# ----------------------------------------------------------------
# Failure paths: parse error → stub fallback
# ----------------------------------------------------------------


def test_compose_brief_strips_markdown_code_fences(tmp_path: Path) -> None:
    """Claude wraps JSON output in ```json ... ``` and often prepends prose."""
    payload = {
        "incident_id": 1,
        "tick": 7,
        "recommendation": {
            "action_id": "issue_alert",
            "confidence_signal": "high",
            "rationale": "rationale text here.",
        },
        "citations": [],
        "precondition_check": [],
        "concerns": [],
        "open_questions": [],
        "tool_calls": [],
    }
    # Real-world shape: prose preamble, then fenced JSON, then maybe more prose.
    fenced = (
        "Corpus and history are empty. Composing the brief from plan + forecast.\n\n"
        "```json\n" + json.dumps(payload) + "\n```\n"
        "Done."
    )
    client = ScriptedClient([_fake_response(content=fenced)])
    composer = LLMComposer(
        client=client,
        model="anthropic/claude-sonnet-4.6",
        budget=_budget(tmp_path),
        enabled=True,
    )
    rec = _record()
    result = composer.compose_brief(
        record=rec,
        recommended_action_id="issue_alert",
        case_study="lido",
        incident_id=1,
        opened_at=rec.simulated_time,
        ctx=_ctx(tmp_path, record=rec),
    )
    assert result.is_stub is False
    assert result.brief.recommendation.rationale == "rationale text here."


def test_compose_brief_falls_back_when_json_invalid(tmp_path: Path) -> None:
    client = ScriptedClient(
        [_fake_response(content="this is not JSON, sorry")]
    )
    composer = LLMComposer(
        client=client,
        model="anthropic/claude-sonnet-4-5",
        budget=_budget(tmp_path),
        enabled=True,
    )
    rec = _record()
    result = composer.compose_brief(
        record=rec,
        recommended_action_id="issue_alert",
        case_study="lido",
        incident_id=1,
        opened_at=rec.simulated_time,
        ctx=_ctx(tmp_path, record=rec),
    )
    assert result.is_stub is True
    assert "compose_error" in (result.reason or "")


def test_compose_brief_falls_back_when_schema_invalid(tmp_path: Path) -> None:
    bad_payload = {
        "incident_id": 1,
        "tick": 7,
        "recommendation": {
            "action_id": "issue_alert",
            # missing confidence_signal AND rationale
        },
    }
    client = ScriptedClient([_fake_response(content=json.dumps(bad_payload))])
    composer = LLMComposer(
        client=client,
        model="anthropic/claude-sonnet-4-5",
        budget=_budget(tmp_path),
        enabled=True,
    )
    rec = _record()
    result = composer.compose_brief(
        record=rec,
        recommended_action_id="issue_alert",
        case_study="lido",
        incident_id=1,
        opened_at=rec.simulated_time,
        ctx=_ctx(tmp_path, record=rec),
    )
    assert result.is_stub is True


def test_compose_brief_loop_limit_falls_back(tmp_path: Path) -> None:
    """If the model keeps calling tools forever, composer aborts."""
    looping_responses = [
        _fake_response(
            content=None,
            tool_calls=[
                _fake_tool_call(
                    f"call_{i}", "fetch_live_sea_level", {}
                )
            ],
        )
        for i in range(20)
    ]
    client = ScriptedClient(looping_responses)
    composer = LLMComposer(
        client=client,
        model="anthropic/claude-sonnet-4-5",
        budget=_budget(tmp_path),
        enabled=True,
        max_tool_iterations=3,
    )
    rec = _record()
    result = composer.compose_brief(
        record=rec,
        recommended_action_id="issue_alert",
        case_study="lido",
        incident_id=1,
        opened_at=rec.simulated_time,
        ctx=_ctx(tmp_path, record=rec),
    )
    assert result.is_stub is True
    assert "compose_error" in (result.reason or "")
