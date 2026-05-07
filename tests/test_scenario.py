"""Tests for the scenario record schema and JSONL loader."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from coastal_agent.scenario import (
    LIDO_ACTIONS,
    NUM_ACTIONS,
    GlobalState,
    PolicyOutput,
    ScenarioRecord,
    iter_scenario,
    load_scenario,
)

SEED_SCENARIO = Path("scenarios/lido_acqua_alta_01.jsonl")


# -----------------------------------------------------------------
# Constants / sanity
# -----------------------------------------------------------------


def test_action_set_size() -> None:
    assert NUM_ACTIONS == 7
    assert LIDO_ACTIONS[0] == "monitor"  # always-legal action lives at index 0


# -----------------------------------------------------------------
# PolicyOutput validation
# -----------------------------------------------------------------


def test_policy_output_accepts_normalised_probs() -> None:
    PolicyOutput(action_probs=[1.0, 0, 0, 0, 0, 0, 0], value_estimate=0.0)
    PolicyOutput(
        action_probs=[0.20, 0.55, 0.15, 0.06, 0.04, 0, 0],
        value_estimate=-0.25,
    )


def test_policy_output_rejects_unnormalised_probs() -> None:
    with pytest.raises(ValidationError, match="must sum to 1.0"):
        PolicyOutput(
            action_probs=[0.5, 0.5, 0.5, 0, 0, 0, 0],
            value_estimate=0.0,
        )


def test_policy_output_rejects_negative_prob() -> None:
    with pytest.raises(ValidationError, match=r"\[0, 1\]"):
        PolicyOutput(
            action_probs=[1.1, -0.1, 0, 0, 0, 0, 0],
            value_estimate=0.0,
        )


def test_policy_output_rejects_wrong_length() -> None:
    with pytest.raises(ValidationError):
        PolicyOutput(action_probs=[1.0, 0, 0, 0, 0, 0], value_estimate=0.0)


# -----------------------------------------------------------------
# ScenarioRecord validation
# -----------------------------------------------------------------


def _good_record(**overrides) -> dict:
    base = {
        "tick": 0,
        "simulated_time": "2026-11-12T08:00:00",
        "forecast": {"surge_cm": 60, "wind_ms": 5.0, "wave_m": 0.8},
        "state": {"forecast_tide": 60, "forecast_wind_wave": 5.0},
        "mask": [True, False, False, False, False, False, False],
        "policy_output": {"action_probs": [1.0, 0, 0, 0, 0, 0, 0], "value_estimate": 0.0},
    }
    base.update(overrides)
    return base


def test_scenario_record_accepts_good_input() -> None:
    ScenarioRecord.model_validate(_good_record())


def test_scenario_record_rejects_monitor_illegal_in_mask() -> None:
    bad_mask = [False, True, True, True, True, True, True]
    with pytest.raises(ValidationError, match="monitor"):
        ScenarioRecord.model_validate(_good_record(mask=bad_mask))


def test_scenario_record_rejects_negative_tick() -> None:
    with pytest.raises(ValidationError):
        ScenarioRecord.model_validate(_good_record(tick=-1))


# -----------------------------------------------------------------
# JSONL loader
# -----------------------------------------------------------------


def test_load_seeded_scenario() -> None:
    """The committed seed scenario should parse cleanly."""
    records = load_scenario(SEED_SCENARIO)
    assert len(records) == 20
    assert records[0].tick == 0
    assert records[-1].tick == 19


def test_load_scenario_validates_tick_sequence() -> None:
    records = load_scenario(SEED_SCENARIO)
    for i, r in enumerate(records):
        assert r.tick == i


def test_load_scenario_skips_comments_and_blank_lines(tmp_path: Path) -> None:
    p = tmp_path / "with_comments.jsonl"
    p.write_text(
        "// header comment\n"
        "\n"
        '{"tick":0,"simulated_time":"2026-01-01T00:00:00",'
        '"forecast":{"surge_cm":50,"wind_ms":3,"wave_m":0.5},'
        '"state":{"forecast_tide":50,"forecast_wind_wave":3},'
        '"mask":[true,false,false,false,false,false,false],'
        '"policy_output":{"action_probs":[1.0,0,0,0,0,0,0],"value_estimate":0.0}}\n'
        "// trailing comment\n",
        encoding="utf-8",
    )
    records = load_scenario(p)
    assert len(records) == 1


def test_load_scenario_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_scenario(tmp_path / "nonexistent.jsonl")


def test_load_scenario_rejects_malformed_json(tmp_path: Path) -> None:
    p = tmp_path / "bad.jsonl"
    p.write_text("{not valid json}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Failed to parse"):
        load_scenario(p)


def test_load_scenario_rejects_tick_gap(tmp_path: Path) -> None:
    p = tmp_path / "gap.jsonl"
    record_template = (
        '{{"tick":{tick},"simulated_time":"2026-01-01T0{tick}:00:00",'
        '"forecast":{{"surge_cm":50,"wind_ms":3,"wave_m":0.5}},'
        '"state":{{"forecast_tide":50,"forecast_wind_wave":3}},'
        '"mask":[true,false,false,false,false,false,false],'
        '"policy_output":{{"action_probs":[1.0,0,0,0,0,0,0],"value_estimate":0.0}}}}'
    )
    p.write_text(
        record_template.format(tick=0) + "\n" + record_template.format(tick=2) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="tick gap or repeat"):
        load_scenario(p)


def test_iter_scenario_streams_all_records() -> None:
    streamed = list(iter_scenario(SEED_SCENARIO))
    assert len(streamed) == 20
