"""Tests for the orchestrator — end-to-end against the seeded scenario."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from coastal_agent.db import connect, init_schema
from coastal_agent.orchestrator import Orchestrator
from coastal_agent.scenario import iter_scenario, load_scenario
from coastal_agent.trigger import TriggerConfig

SEED_SCENARIO = Path("scenarios/lido_acqua_alta_01.jsonl")


# -----------------------------------------------------------------
# End-to-end against the seeded Lido scenario
# -----------------------------------------------------------------


def test_seed_scenario_runs_to_completion(tmp_path: Path) -> None:
    """Replay the full Lido acqua alta scenario; verify lifecycle counts."""
    conn = connect(tmp_path / "test.db")
    init_schema(conn)
    orch = Orchestrator(conn)

    summary = orch.run(iter_scenario(SEED_SCENARIO))

    # 20 ticks processed, 1 incident, 1 close.
    assert summary.ticks_processed == 20
    assert summary.incidents_opened == 1
    assert summary.incidents_closed == 1


def test_seed_scenario_writes_expected_db_rows(tmp_path: Path) -> None:
    """Verify the database state after a full replay."""
    conn = connect(tmp_path / "test.db")
    init_schema(conn)
    orch = Orchestrator(conn)
    orch.run(iter_scenario(SEED_SCENARIO))

    # All 20 ticks produce a forecast row.
    n_forecasts = conn.execute("SELECT COUNT(*) AS c FROM forecasts").fetchone()["c"]
    assert n_forecasts == 20

    # Exactly one incident, opened and closed.
    incidents = conn.execute(
        "SELECT id, status, opened_at, closed_at FROM incidents"
    ).fetchall()
    assert len(incidents) == 1
    inc = incidents[0]
    assert inc["status"] == "closed"
    assert inc["closed_at"] is not None

    # Active period: T+7 (activation) through T+19 (standdown). 13 ticks.
    n_evals = conn.execute("SELECT COUNT(*) AS c FROM evaluations").fetchone()["c"]
    assert n_evals == 13
    n_briefs = conn.execute("SELECT COUNT(*) AS c FROM briefs").fetchone()["c"]
    assert n_briefs == 13

    # 7 of those active ticks have non-monitor recommendations
    # (T+7..T+16 = 10 ticks). T+17..T+19 are 'monitor'.
    # Briefs for monitor are still written, just not emailed.
    monitor_briefs = conn.execute(
        "SELECT COUNT(*) AS c FROM briefs "
        "WHERE json_extract(brief_json, '$.recommendation.action_id') = 'monitor'"
    ).fetchone()["c"]
    assert monitor_briefs == 3  # T+17, T+18, T+19


def test_seed_scenario_email_count_matches_design(tmp_path: Path) -> None:
    """Per §3.5.D: activation + new_alert per distinct action + standdown.

    Storyline produces:
      - T+7  activation (issue_alert)
      - T+9  new_alert (deploy_sandbags)
      - T+10 new_alert (close_road)
      - T+11 new_alert (open_shelter)
      - T+12 new_alert (assisted_evacuation)
      - T+19 standdown
      ----
      6 emails total, all mock-mode.
    """
    conn = connect(tmp_path / "test.db")
    init_schema(conn)
    orch = Orchestrator(conn)
    orch.run(iter_scenario(SEED_SCENARIO))

    emails = conn.execute(
        "SELECT moment, mode, subject FROM sent_emails ORDER BY id"
    ).fetchall()
    assert len(emails) == 6
    moments = [e["moment"] for e in emails]
    assert moments == [
        "activation",
        "new_alert",
        "new_alert",
        "new_alert",
        "new_alert",
        "standdown",
    ]
    assert all(e["mode"] == "mock" for e in emails)


def test_seed_scenario_dedup_loose_per_incident(tmp_path: Path) -> None:
    """Same action recommended twice in one incident emails only once."""
    conn = connect(tmp_path / "test.db")
    init_schema(conn)
    orch = Orchestrator(conn)
    orch.run(iter_scenario(SEED_SCENARIO))

    # T+15 recommends open_shelter (same as T+11). Only T+11 emails;
    # T+15 is silently deduped.
    open_shelter_emails = conn.execute(
        "SELECT COUNT(*) AS c FROM sent_emails WHERE subject LIKE '%open_shelter%'"
    ).fetchone()["c"]
    assert open_shelter_emails == 1

    # close_road recommended at T+10 and T+16 — only one email.
    close_road_emails = conn.execute(
        "SELECT COUNT(*) AS c FROM sent_emails WHERE subject LIKE '%close_road%'"
    ).fetchone()["c"]
    assert close_road_emails == 1


def test_seed_scenario_monitor_never_emails(tmp_path: Path) -> None:
    """`monitor` is the resting state — no email when the policy recommends it."""
    conn = connect(tmp_path / "test.db")
    init_schema(conn)
    orch = Orchestrator(conn)
    orch.run(iter_scenario(SEED_SCENARIO))

    monitor_emails = conn.execute(
        "SELECT COUNT(*) AS c FROM sent_emails WHERE subject LIKE '%monitor%'"
    ).fetchone()["c"]
    assert monitor_emails == 0


def test_seed_scenario_brief_uniqueness(tmp_path: Path) -> None:
    """Each (incident_id, tick) pair has at most one brief — UNIQUE constraint."""
    conn = connect(tmp_path / "test.db")
    init_schema(conn)
    orch = Orchestrator(conn)
    orch.run(iter_scenario(SEED_SCENARIO))

    rows = conn.execute(
        "SELECT incident_id, tick, COUNT(*) AS c FROM briefs "
        "GROUP BY incident_id, tick HAVING c > 1"
    ).fetchall()
    assert rows == []


def test_seed_scenario_incident_trigger_condition(tmp_path: Path) -> None:
    """The opened incident records the activation rule + values."""
    conn = connect(tmp_path / "test.db")
    init_schema(conn)
    orch = Orchestrator(conn)
    orch.run(iter_scenario(SEED_SCENARIO))

    inc = conn.execute("SELECT trigger_condition FROM incidents WHERE id=1").fetchone()
    cond = json.loads(inc["trigger_condition"])
    assert cond["rule"] == "surge_above_activation"
    assert cond["tick"] == 7  # T+7 is when surge first crosses 110 cm
    assert cond["values"]["surge_cm"] == 112.0


# -----------------------------------------------------------------
# Quiet-day scenario — no incident
# -----------------------------------------------------------------


def test_quiet_scenario_does_not_open_incident(tmp_path: Path) -> None:
    """A scenario where surge never crosses threshold should produce no incident."""
    quiet_path = tmp_path / "quiet.jsonl"
    lines = []
    for tick in range(10):
        lines.append(
            '{"tick":' + str(tick) + ',"simulated_time":"2026-01-01T'
            + f"{tick:02d}" + ':00:00",'
            '"forecast":{"surge_cm":50,"wind_ms":3.0,"wave_m":0.5},'
            '"state":{"forecast_tide":50,"forecast_wind_wave":3.0},'
            '"mask":[true,false,false,false,false,false,false],'
            '"policy_output":{"action_probs":[1.0,0,0,0,0,0,0],"value_estimate":0.1}}'
        )
    quiet_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    conn = connect(tmp_path / "test.db")
    init_schema(conn)
    orch = Orchestrator(conn)
    summary = orch.run(load_scenario(quiet_path))

    assert summary.ticks_processed == 10
    assert summary.incidents_opened == 0
    assert summary.evaluations_written == 0  # no incident => no policy evaluations recorded
    assert summary.briefs_written == 0
    assert summary.emails_sent == 0
    # 10 forecast rows still written (standby was watching them)
    n_forecasts = conn.execute("SELECT COUNT(*) AS c FROM forecasts").fetchone()["c"]
    assert n_forecasts == 10
