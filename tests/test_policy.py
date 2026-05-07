"""Tests for the replay-mode policy adapter."""

from __future__ import annotations

from pathlib import Path

import pytest

from coastal_agent.policy import (
    LivePolicy,
    ReplayPolicy,
    confidence_signal,
    recommended_action_id,
    recommended_action_index,
)
from coastal_agent.scenario import LIDO_ACTIONS, PolicyOutput

SEED_SCENARIO = Path("scenarios/lido_acqua_alta_01.jsonl")


# -----------------------------------------------------------------
# ReplayPolicy
# -----------------------------------------------------------------


def test_replay_policy_loads_scenario() -> None:
    policy = ReplayPolicy(SEED_SCENARIO)
    assert len(policy) == 20


def test_replay_policy_at_tick_returns_record() -> None:
    policy = ReplayPolicy(SEED_SCENARIO)
    r = policy.at_tick(0)
    assert r.tick == 0


def test_replay_policy_output_at_returns_policy_output() -> None:
    policy = ReplayPolicy(SEED_SCENARIO)
    out = policy.output_at(7)
    assert isinstance(out, PolicyOutput)
    assert sum(out.action_probs) == pytest.approx(1.0, abs=1e-3)


def test_replay_policy_out_of_range_raises() -> None:
    policy = ReplayPolicy(SEED_SCENARIO)
    with pytest.raises(IndexError):
        policy.at_tick(99)
    with pytest.raises(IndexError):
        policy.at_tick(-1)


# -----------------------------------------------------------------
# LivePolicy — gated on the optional `live` dependency group
# -----------------------------------------------------------------


def _has_live_deps() -> bool:
    try:
        import torch_geometric  # noqa: F401
        import geopandas  # noqa: F401
        import rasterio  # noqa: F401
        return True
    except ImportError:
        return False


def test_live_policy_raises_when_extras_missing() -> None:
    """On a build without the `live` extras, LivePolicy raises a clear
    ImportError pointing the operator at the install command."""
    if _has_live_deps():
        pytest.skip("live extras installed; this test exercises the missing-deps path")
    from pathlib import Path
    with pytest.raises((ImportError, FileNotFoundError)):
        LivePolicy(Path("external/gnn_drl_ews_v003_seed2"))


# -----------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------


def test_recommended_action_id_picks_argmax() -> None:
    out = PolicyOutput(
        action_probs=[0.10, 0.55, 0.15, 0.10, 0.05, 0.025, 0.025],
        value_estimate=-0.2,
    )
    assert recommended_action_id(out) == "issue_alert"
    assert recommended_action_index(out) == 1


def test_confidence_signal_categories() -> None:
    high = PolicyOutput(action_probs=[0.85, 0.10, 0.025, 0.015, 0.005, 0.0025, 0.0025], value_estimate=0.0)
    medium = PolicyOutput(action_probs=[0.50, 0.30, 0.10, 0.05, 0.04, 0.005, 0.005], value_estimate=0.0)
    split = PolicyOutput(action_probs=[0.45, 0.40, 0.10, 0.03, 0.01, 0.005, 0.005], value_estimate=0.0)
    low = PolicyOutput(action_probs=[0.30, 0.18, 0.16, 0.14, 0.12, 0.06, 0.04], value_estimate=0.0)
    assert confidence_signal(high) == "high"
    assert confidence_signal(medium) == "medium"
    assert confidence_signal(split) == "split"
    assert confidence_signal(low) == "low"


# -----------------------------------------------------------------
# Storyline-level smoke tests against the seeded scenario
# -----------------------------------------------------------------


def test_seed_scenario_storyline_arc() -> None:
    """Verify the seeded scenario follows the documented Lido acqua alta arc:
    monitor → issue_alert → deploy_sandbags → close_road → open_shelter
    → assisted_evacuation → recovery → monitor.
    """
    policy = ReplayPolicy(SEED_SCENARIO)
    actions = [recommended_action_id(policy.output_at(i)) for i in range(len(policy))]

    # Pre-trigger ticks all monitor.
    assert all(a == "monitor" for a in actions[:7])
    # First alert action lands at T+7.
    assert actions[7] == "issue_alert"
    # Peak action (assisted_evacuation) appears around T+12.
    assert "assisted_evacuation" in actions[12:15]
    # Final ticks back to monitor.
    assert all(a == "monitor" for a in actions[-3:])


def test_seed_scenario_value_estimates_dip_at_peak() -> None:
    """Value should be most negative around the storm peak (T+12-13)."""
    policy = ReplayPolicy(SEED_SCENARIO)
    values = [policy.output_at(i).value_estimate for i in range(len(policy))]
    peak_min = min(values[10:14])
    pre_storm = values[0]
    assert peak_min < pre_storm - 0.5  # value drops by at least 0.5 from baseline
