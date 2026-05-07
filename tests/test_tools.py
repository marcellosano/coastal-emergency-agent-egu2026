"""Tests for the tool catalog + dispatcher."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from coastal_agent.db import connect, init_schema
from coastal_agent.scenario import (
    ForecastSnapshot,
    GlobalState,
    PolicyOutput,
    ScenarioRecord,
)
from coastal_agent.tools import (
    TOOL_NAMES,
    TOOL_SCHEMAS,
    ToolContext,
    ToolDispatcher,
    short_summary,
)

VENDOR_DIR = Path("external/gnn_drl_ews_v003_seed2")


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


def _ctx(tmp_path: Path, *, record=None, upcoming=None, incident_id: int = 1):
    conn = connect(tmp_path / "tools.db")
    init_schema(conn)
    return ToolContext(
        record=record or _record(),
        incident_id=incident_id,
        case_study="lido",
        conn=conn,
        vendor_dir=VENDOR_DIR,
        corpus_dir=Path("corpus"),
        upcoming_records=list(upcoming or []),
    )


# ----------------------------------------------------------------
# Schema sanity
# ----------------------------------------------------------------


def test_tool_schemas_are_well_formed() -> None:
    assert len(TOOL_SCHEMAS) == 6
    for spec in TOOL_SCHEMAS:
        assert spec["type"] == "function"
        fn = spec["function"]
        assert isinstance(fn["name"], str) and fn["name"]
        assert isinstance(fn["description"], str) and fn["description"]
        assert fn["parameters"]["type"] == "object"


def test_tool_names_match_dispatcher_handlers() -> None:
    expected = {
        "get_plan_provision",
        "verify_preconditions",
        "fetch_forecast_detail",
        "query_corpus",
        "recall_similar_incidents",
        "fetch_live_sea_level",
    }
    assert set(TOOL_NAMES) == expected


# ----------------------------------------------------------------
# get_plan_provision
# ----------------------------------------------------------------


@pytest.mark.skipif(
    not (VENDOR_DIR / "plans" / "lido.yaml").exists(),
    reason="vendored plan YAML not present",
)
def test_get_plan_provision_returns_pce_citation(tmp_path: Path) -> None:
    disp = ToolDispatcher(_ctx(tmp_path))
    result, _ = disp.call(
        "get_plan_provision",
        {"action_id": "issue_alert", "case_study": "lido"},
    )
    assert result["action_id"] == "issue_alert"
    assert "PCE_Comune_Venezia_v1_2008" in result["citation_ref"]
    assert "4.1.4.9" in result["citation_ref"]
    assert result["authority"] == "Sindaco"


@pytest.mark.skipif(
    not (VENDOR_DIR / "plans" / "lido.yaml").exists(),
    reason="vendored plan YAML not present",
)
def test_get_plan_provision_unknown_action(tmp_path: Path) -> None:
    disp = ToolDispatcher(_ctx(tmp_path))
    result, _ = disp.call(
        "get_plan_provision",
        {"action_id": "build_seawall", "case_study": "lido"},
    )
    assert "error" in result


def test_get_plan_provision_unknown_case_study(tmp_path: Path) -> None:
    disp = ToolDispatcher(_ctx(tmp_path))
    result, _ = disp.call(
        "get_plan_provision",
        {"action_id": "monitor", "case_study": "atlantis"},
    )
    assert "error" in result


# ----------------------------------------------------------------
# verify_preconditions
# ----------------------------------------------------------------


@pytest.mark.skipif(
    not (VENDOR_DIR / "plans" / "lido.yaml").exists(),
    reason="vendored plan YAML not present",
)
def test_verify_preconditions_issue_alert_threshold_satisfied(
    tmp_path: Path,
) -> None:
    disp = ToolDispatcher(_ctx(tmp_path, record=_record(surge_cm=120.0)))
    result, _ = disp.call(
        "verify_preconditions",
        {"action_id": "issue_alert", "case_study": "lido"},
    )
    sat = [c for c in result["checks"] if c["satisfied"] is True]
    assert any("alert_tide" in c["evidence"] for c in sat)


@pytest.mark.skipif(
    not (VENDOR_DIR / "plans" / "lido.yaml").exists(),
    reason="vendored plan YAML not present",
)
def test_verify_preconditions_issue_alert_threshold_unsatisfied(
    tmp_path: Path,
) -> None:
    disp = ToolDispatcher(_ctx(tmp_path, record=_record(surge_cm=80.0)))
    result, _ = disp.call(
        "verify_preconditions",
        {"action_id": "issue_alert", "case_study": "lido"},
    )
    unsat = [c for c in result["checks"] if c["satisfied"] is False]
    assert len(unsat) >= 1


# ----------------------------------------------------------------
# fetch_forecast_detail
# ----------------------------------------------------------------


def test_fetch_forecast_detail_offset_zero_returns_current(tmp_path: Path) -> None:
    disp = ToolDispatcher(_ctx(tmp_path))
    result, _ = disp.call("fetch_forecast_detail", {"tick_offset": 0})
    assert result["surge_cm"] == 112.0
    assert result["tick"] == 7


def test_fetch_forecast_detail_offset_n_returns_upcoming(tmp_path: Path) -> None:
    upcoming = [_record(tick=8, surge_cm=130.0), _record(tick=9, surge_cm=140.0)]
    disp = ToolDispatcher(_ctx(tmp_path, upcoming=upcoming))
    result, _ = disp.call("fetch_forecast_detail", {"tick_offset": 2})
    assert result["surge_cm"] == 140.0
    assert result["tick"] == 9


def test_fetch_forecast_detail_offset_beyond_horizon(tmp_path: Path) -> None:
    disp = ToolDispatcher(_ctx(tmp_path))
    result, _ = disp.call("fetch_forecast_detail", {"tick_offset": 5})
    assert "error" in result


# ----------------------------------------------------------------
# query_corpus & recall_similar_incidents (empty cases)
# ----------------------------------------------------------------


def test_query_corpus_empty_returns_note(tmp_path: Path) -> None:
    disp = ToolDispatcher(_ctx(tmp_path))
    result, _ = disp.call("query_corpus", {"query": "Sindaco alert"})
    assert result["results"] == []
    assert "corpus is empty" in result["note"]


def test_recall_similar_incidents_empty(tmp_path: Path) -> None:
    disp = ToolDispatcher(_ctx(tmp_path))
    result, _ = disp.call(
        "recall_similar_incidents", {"query": "Lido acqua alta"}
    )
    assert result["results"] == []


# ----------------------------------------------------------------
# fetch_live_sea_level
# ----------------------------------------------------------------


def test_fetch_live_sea_level_returns_stub(tmp_path: Path) -> None:
    disp = ToolDispatcher(_ctx(tmp_path))
    result, latency = disp.call("fetch_live_sea_level", {})
    assert result["gauge_id"] == "lido_diga_sud"
    assert result["value_cm"] == 112.0
    assert "url" in result
    assert latency >= 0


# ----------------------------------------------------------------
# Unknown tool + summary helper
# ----------------------------------------------------------------


def test_unknown_tool_returns_error(tmp_path: Path) -> None:
    disp = ToolDispatcher(_ctx(tmp_path))
    result, _ = disp.call("send_email_to_president", {})
    assert "error" in result


def test_short_summary_handles_each_tool() -> None:
    examples = {
        "get_plan_provision": {"action_id": "issue_alert",
                               "citation_ref": "PCE §4.1.4.9 p41"},
        "verify_preconditions": {"action_id": "issue_alert",
                                 "checks": [{"satisfied": True}, {"satisfied": False}]},
        "fetch_forecast_detail": {"tick_offset": 2, "surge_cm": 130, "wind_ms": 11.5},
        "query_corpus": {"query": "alert", "results": []},
        "recall_similar_incidents": {"query": "Lido", "results": []},
        "fetch_live_sea_level": {"value_cm": 112, "gauge_id": "lido_diga_sud"},
    }
    for name, result in examples.items():
        s = short_summary(name, result)
        assert isinstance(s, str) and len(s) > 0
