"""Scenario records — paired (forecast, state, mask, policy_output) tuples
that the replay path consumes to drive end-to-end pipeline runs without
touching `torch_geometric` or the live env.

A scenario is a JSONL file: one `ScenarioRecord` per line, ordered by
`tick`. Each record represents one hour of an incident's evolution.

Records can come from two sources, both producing the same shape:
  - hand-crafted (for narrative-driven demo scenarios)
  - generated offline from the vendored gnn_drl_ews snapshot via
    `scripts/generate_scenario.py` (real-data path; not yet built)

The orchestrator's replay-mode reads these to step through an incident
deterministically. No GAT, no graph construction, no torch_geometric at
runtime. The vendored snapshot is referenced only for provenance and
for re-generating scenarios offline.

Action ID order (canonical, from `external/gnn_drl_ews_v003_seed2/plans/lido.yaml`):

    0 monitor
    1 issue_alert
    2 deploy_sandbags
    3 close_road
    4 open_shelter
    5 assisted_evacuation
    6 full_evacuation
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterator

from pydantic import BaseModel, Field, field_validator


LIDO_ACTIONS: tuple[str, ...] = (
    "monitor",
    "issue_alert",
    "deploy_sandbags",
    "close_road",
    "open_shelter",
    "assisted_evacuation",
    "full_evacuation",
)
NUM_ACTIONS = len(LIDO_ACTIONS)
PROB_SUM_TOLERANCE = 1e-3


class ForecastSnapshot(BaseModel):
    """Snapshot of forecast values at one tick."""

    surge_cm: float           # Adriatic-side storm-surge water level
    wind_ms: float            # wind speed
    wave_m: float             # significant wave height
    rainfall_mm: float = 0.0
    horizon_hours: int = 24


class GlobalState(BaseModel):
    """Plan-policy global features (mirrors gat_actor_critic.GLOBAL_FEATURES).

    storm_phase encodes the PCE phase (0=Attenzione, 1=Preallarme,
    2=Allarme, 3=Rientro). Stored as float for the GAT but conceptually
    discrete.
    """

    forecast_tide: float
    forecast_wind_wave: float
    storm_phase: float = 0.0
    time_remaining: float = 24.0
    resources: float = 5.0
    preparedness: float = 0.0


class PolicyOutput(BaseModel):
    """Cached policy output for a single tick. Replay mode reads these
    instead of running the GAT live."""

    action_probs: list[float] = Field(min_length=NUM_ACTIONS, max_length=NUM_ACTIONS)
    value_estimate: float
    storm_type: str = "tide"

    @field_validator("action_probs")
    @classmethod
    def _probs_sum_to_one(cls, v: list[float]) -> list[float]:
        s = sum(v)
        if abs(s - 1.0) > PROB_SUM_TOLERANCE:
            raise ValueError(f"action_probs must sum to 1.0; got {s:.4f}")
        if any(p < 0.0 or p > 1.0 for p in v):
            raise ValueError("each action probability must be in [0, 1]")
        return v


class ScenarioRecord(BaseModel):
    """One tick in a scenario JSONL."""

    tick: int = Field(ge=0)
    simulated_time: datetime
    forecast: ForecastSnapshot
    state: GlobalState
    mask: list[bool] = Field(min_length=NUM_ACTIONS, max_length=NUM_ACTIONS)
    policy_output: PolicyOutput

    @field_validator("mask")
    @classmethod
    def _monitor_always_legal(cls, v: list[bool]) -> list[bool]:
        if not v[0]:
            raise ValueError("monitor (action[0]) must always be legal in the mask")
        return v


def load_scenario(path: Path) -> list[ScenarioRecord]:
    """Parse and validate a JSONL scenario file. Returns records ordered by tick."""
    if not path.exists():
        raise FileNotFoundError(f"Scenario not found at {path}")
    records: list[ScenarioRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("//"):
                continue
            try:
                records.append(ScenarioRecord.model_validate_json(line))
            except Exception as e:
                raise ValueError(f"Failed to parse {path}:{line_no}: {e}") from e
    if not records:
        raise ValueError(f"Scenario {path} contains no records")
    for i, r in enumerate(records):
        if r.tick != i:
            raise ValueError(
                f"Scenario {path}: tick gap or repeat — record index {i} has tick={r.tick}, expected {i}"
            )
    return records


def iter_scenario(path: Path) -> Iterator[ScenarioRecord]:
    """Iterate scenario records lazily. Convenient for replay loops."""
    yield from load_scenario(path)
