"""Unit tests for the live-mode daemon (apscheduler tick).

Doesn't exercise live extras (no torch_geometric required); uses a
fake policy that returns a pre-canned LiveTickResult. Verifies the
tick loop wires forecast → policy → orchestrator → heartbeat correctly.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from coastal_agent.db import connect, init_schema
from coastal_agent.orchestrator import Orchestrator
from coastal_agent.policy import LiveTickResult
from coastal_agent.scenario import (
    ForecastSnapshot,
    GlobalState,
    PolicyOutput,
)
from coastal_agent.scheduler import Daemon
from coastal_agent.trigger import TriggerConfig
from coastal_agent.weather import ForecastFetch


class _FakePolicy:
    """LivePolicy stand-in for unit tests — no torch, no env."""

    def __init__(self, mask: list[bool] | None = None) -> None:
        self.mask = mask or [True, True, False, False, False, False, False]
        self.calls = 0

    def build_scenario_record(
        self,
        forecast,
        *,
        tick: int,
        simulated_time: datetime,
        **_: object,
    ):
        self.calls += 1
        from coastal_agent.scenario import ScenarioRecord
        po = PolicyOutput(
            action_probs=[0.2, 0.7, 0.05, 0.025, 0.025, 0.0, 0.0],
            value_estimate=-1.5,
            storm_type="tide",
        )
        gs = GlobalState(
            forecast_tide=forecast.surge_cm,
            forecast_wind_wave=forecast.wind_ms,
            storm_phase=0.0,
            time_remaining=24.0,
            resources=5.0,
        )
        return ScenarioRecord(
            tick=tick,
            simulated_time=simulated_time,
            forecast=forecast,
            state=gs,
            mask=self.mask,
            policy_output=po,
        )


def _build_daemon(tmp_path: Path, *, surge_cm: float = 60.0) -> Daemon:
    conn = connect(tmp_path / "daemon.db")
    init_schema(conn)
    orch = Orchestrator(conn=conn, trigger_config=TriggerConfig())
    return Daemon(
        conn=conn,
        orchestrator=orch,
        policy=_FakePolicy(),
        poll_interval_seconds=3600,
    )


def _patch_weather(surge_cm: float):
    """Patch fetch_lido_forecast in the scheduler module to return a canned snapshot."""
    snap = ForecastSnapshot(
        surge_cm=surge_cm, wind_ms=10.0, wave_m=1.2, rainfall_mm=0.0,
    )
    fetch = ForecastFetch(
        snapshot=snap,
        fetched_at=datetime.now(timezone.utc),
        source="open-meteo-test",
        raw={},
    )
    return patch("coastal_agent.scheduler.fetch_lido_forecast", return_value=fetch)


# ----------------------------------------------------------------


def test_one_tick_below_threshold_writes_forecast_no_incident(tmp_path: Path) -> None:
    daemon = _build_daemon(tmp_path)
    with _patch_weather(surge_cm=60.0):
        daemon.tick()

    n_forecasts = daemon.conn.execute(
        "SELECT COUNT(*) AS c FROM forecasts"
    ).fetchone()["c"]
    assert n_forecasts == 1

    n_incidents = daemon.conn.execute(
        "SELECT COUNT(*) AS c FROM incidents"
    ).fetchone()["c"]
    assert n_incidents == 0

    heartbeat = daemon.conn.execute(
        "SELECT tick_count, last_tick_at FROM heartbeat WHERE id=1"
    ).fetchone()
    assert heartbeat["tick_count"] == 1
    assert heartbeat["last_tick_at"] is not None


def test_one_tick_above_threshold_opens_incident(tmp_path: Path) -> None:
    daemon = _build_daemon(tmp_path)
    with _patch_weather(surge_cm=130.0):
        daemon.tick()

    n_incidents = daemon.conn.execute(
        "SELECT COUNT(*) AS c FROM incidents WHERE status='active'"
    ).fetchone()["c"]
    assert n_incidents == 1

    n_briefs = daemon.conn.execute(
        "SELECT COUNT(*) AS c FROM briefs"
    ).fetchone()["c"]
    assert n_briefs == 1


def test_tick_isolates_weather_failure(tmp_path: Path) -> None:
    """Weather error logs + skips; heartbeat NOT incremented."""
    daemon = _build_daemon(tmp_path)
    from coastal_agent.weather import WeatherFetchError

    with patch(
        "coastal_agent.scheduler.fetch_lido_forecast",
        side_effect=WeatherFetchError("boom"),
    ):
        daemon.tick()

    n_forecasts = daemon.conn.execute(
        "SELECT COUNT(*) AS c FROM forecasts"
    ).fetchone()["c"]
    assert n_forecasts == 0
    heartbeat = daemon.conn.execute(
        "SELECT tick_count FROM heartbeat WHERE id=1"
    ).fetchone()
    assert heartbeat["tick_count"] == 0


def test_tick_catches_unexpected_exception_in_orchestrator(tmp_path: Path) -> None:
    """A bug inside process_tick should not crash the daemon (systemd
    would Restart=on-failure, but we want the next tick to fire)."""
    daemon = _build_daemon(tmp_path)

    with _patch_weather(surge_cm=60.0):
        with patch.object(
            daemon.orchestrator, "process_tick", side_effect=RuntimeError("oops")
        ):
            daemon.tick()  # must not raise


def test_tick_increments_monotonic_counter(tmp_path: Path) -> None:
    daemon = _build_daemon(tmp_path)
    with _patch_weather(surge_cm=60.0):
        daemon.tick()
        daemon.tick()
        daemon.tick()
    heartbeat = daemon.conn.execute(
        "SELECT tick_count FROM heartbeat WHERE id=1"
    ).fetchone()
    assert heartbeat["tick_count"] == 3
