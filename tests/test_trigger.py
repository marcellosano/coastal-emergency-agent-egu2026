"""Tests for the rule-based trigger module."""

from __future__ import annotations

import pytest

from coastal_agent.scenario import ForecastSnapshot
from coastal_agent.trigger import (
    TriggerConfig,
    evaluate_activation,
    evaluate_standdown,
)


def _forecast(surge_cm: float = 60.0) -> ForecastSnapshot:
    return ForecastSnapshot(surge_cm=surge_cm, wind_ms=5.0, wave_m=0.8)


# -----------------------------------------------------------------
# Activation
# -----------------------------------------------------------------


def test_activation_fires_at_threshold() -> None:
    cfg = TriggerConfig(activation_surge_cm=110.0)
    e = evaluate_activation(_forecast(110.0), cfg)
    assert e.fired
    assert e.rule == "surge_above_activation"
    assert e.values["surge_cm"] == 110.0


def test_activation_fires_above_threshold() -> None:
    cfg = TriggerConfig(activation_surge_cm=110.0)
    assert evaluate_activation(_forecast(120.0), cfg).fired


def test_activation_doesnt_fire_below_threshold() -> None:
    cfg = TriggerConfig(activation_surge_cm=110.0)
    e = evaluate_activation(_forecast(108.0), cfg)
    assert not e.fired
    assert e.rule == "surge_below_activation"


def test_activation_threshold_configurable() -> None:
    cfg = TriggerConfig(activation_surge_cm=80.0)
    assert evaluate_activation(_forecast(85.0), cfg).fired
    cfg = TriggerConfig(activation_surge_cm=200.0)
    assert not evaluate_activation(_forecast(150.0), cfg).fired


# -----------------------------------------------------------------
# Stand-down
# -----------------------------------------------------------------


def test_standdown_requires_consecutive_below_threshold() -> None:
    """First below-threshold tick must NOT fire stand-down (counter = 1, needs 2)."""
    cfg = TriggerConfig(standdown_surge_cm=100.0, standdown_consecutive_ticks=2)
    e = evaluate_standdown(_forecast(85.0), consecutive_below_before=0, config=cfg)
    assert not e.fired
    assert e.values["consecutive_below_after_this_tick"] == 1


def test_standdown_fires_on_second_consecutive_below() -> None:
    cfg = TriggerConfig(standdown_surge_cm=100.0, standdown_consecutive_ticks=2)
    e = evaluate_standdown(_forecast(85.0), consecutive_below_before=1, config=cfg)
    assert e.fired
    assert e.values["consecutive_below_after_this_tick"] == 2


def test_standdown_resets_when_above() -> None:
    """Reset counter when surge climbs back above threshold."""
    cfg = TriggerConfig(standdown_surge_cm=100.0, standdown_consecutive_ticks=2)
    e = evaluate_standdown(_forecast(105.0), consecutive_below_before=1, config=cfg)
    assert not e.fired
    assert e.values["consecutive_below_after_this_tick"] == 0
    assert e.rule == "standdown_above_threshold"


def test_standdown_pending_summary_mentions_remaining_ticks() -> None:
    cfg = TriggerConfig(standdown_surge_cm=100.0, standdown_consecutive_ticks=3)
    e = evaluate_standdown(_forecast(80.0), consecutive_below_before=0, config=cfg)
    assert not e.fired
    assert e.rule == "standdown_pending"
    assert "needs 2 more" in e.summary


def test_standdown_consecutive_ticks_configurable() -> None:
    """One-tick stand-down (no hysteresis) for low-noise replay scenarios."""
    cfg = TriggerConfig(standdown_surge_cm=100.0, standdown_consecutive_ticks=1)
    e = evaluate_standdown(_forecast(80.0), consecutive_below_before=0, config=cfg)
    assert e.fired
    assert e.values["consecutive_below_after_this_tick"] == 1
