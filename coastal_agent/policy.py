"""Policy adapter — produces `PolicyOutput` from either replay or live inference.

Two modes, both first-class in the architecture:

  - **replay** (the path tests use, and the path the EGU video can fall
    back to). Reads a pre-computed `PolicyOutput` from a scenario JSONL
    via `coastal_agent.scenario`. No `torch_geometric`, no env, no GAT
    instantiation. Deterministic, fast, demo-safe.

  - **live**. Loads the GAT from `external/gnn_drl_ews_v003_seed2/` and
    runs forward inference. Requires the optional `live` dependency
    group (`torch-geometric`, `geopandas`, `rasterio`, `gymnasium`,
    `osmnx`, `scikit-learn`). On the droplet that group is installed;
    on Windows dev it is not (default build). LivePolicy raises an
    informative ImportError on the missing deps; the orchestrator's
    constructor accepts None for composer/policy and the daemon decides.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from coastal_agent.scenario import (
    LIDO_ACTIONS,
    NUM_ACTIONS,
    ForecastSnapshot,
    GlobalState,
    PolicyOutput,
    ScenarioRecord,
    load_scenario,
)


if TYPE_CHECKING:
    import torch.nn as nn  # noqa: F401


from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class LiveTickResult:
    """LivePolicy.infer output — everything the daemon needs to build a
    ScenarioRecord (forecast comes from the daemon's weather call)."""

    policy_output: PolicyOutput
    state: GlobalState
    mask: list[bool]


class ReplayPolicy:
    """Reads pre-computed policy outputs from a scenario JSONL.

    Used by the orchestrator's replay-mode at runtime, and by tests.
    """

    def __init__(self, scenario_path: Path) -> None:
        self.scenario_path = scenario_path
        self.records: list[ScenarioRecord] = load_scenario(scenario_path)

    def __len__(self) -> int:
        return len(self.records)

    def at_tick(self, tick: int) -> ScenarioRecord:
        if tick < 0 or tick >= len(self.records):
            raise IndexError(
                f"Scenario has {len(self.records)} ticks; requested {tick}"
            )
        return self.records[tick]

    def output_at(self, tick: int) -> PolicyOutput:
        return self.at_tick(tick).policy_output


class LivePolicy:
    """Live GAT-PPO inference adapter.

    On `__init__`:
      - Path-injects `external/<vendored>/src` so `gnn_drl_ews` resolves.
      - Loads the applied training config; rewrites all relative paths
        (graph cache, hazards cache, demography cache, plan path,
        holdout CSV) to absolute paths under vendor_dir, so the env
        finds them regardless of cwd.
      - Constructs `EWSEnv(config)` and calls `reset_cached(seed=...)`
        once. First call builds the HeteroData graph from OSM gpkg +
        DEM + ISTAT census + ISPRA hazards (slow, ~seconds); subsequent
        resets deepcopy the cached graph.
      - Loads the checkpoint, instantiates `GATActorCritic` with
        the saved `policy_config`, loads weights, switches to eval mode.

    On `infer(forecast)`:
      - reset_cached() to get a fresh graph + state + mask.
      - Override forecast_tide / forecast_wind_wave / storm_phase /
        time_remaining with the live values.
      - Re-derive mask under the updated state (preconditions depend
        on tide thresholds).
      - Run `model.forward(graph, state, mask)`, unpack
        `(Categorical, value)`, return PolicyOutput.

    Requires the optional `live` dependency group; ImportError raised
    on a build without those deps with guidance.
    """

    def __init__(
        self,
        vendor_dir: Path,
        *,
        config_relpath: str = "runs/applied/lido_real.yaml",
        checkpoint_relpath: str = "runs/seed-2/checkpoint.pt",
        seed: int = 2,
    ) -> None:
        self.vendor_dir = Path(vendor_dir).resolve()
        self.seed = seed

        src_dir = self.vendor_dir / "src"
        if not src_dir.exists():
            raise FileNotFoundError(
                f"vendored package source not found at {src_dir}"
            )
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))

        try:
            import torch
            from gnn_drl_ews.config import load_config
            from gnn_drl_ews.env import EWSEnv
            from gnn_drl_ews.policy.gat_actor_critic import GATActorCritic
        except ImportError as e:
            raise ImportError(
                "LivePolicy requires the optional 'live' dependency group "
                "(torch-geometric, geopandas, rasterio, gymnasium, osmnx, "
                "scikit-learn). On the droplet, install via "
                "`uv pip install --no-cache-dir torch-geometric gymnasium "
                "osmnx geopandas rasterio scikit-learn pandas`. "
                f"Underlying ImportError: {e}"
            ) from e

        config_path = self.vendor_dir / config_relpath
        if not config_path.exists():
            raise FileNotFoundError(f"applied config not found: {config_path}")
        cfg = load_config(config_path)
        _absolutize_config_paths(cfg, self.vendor_dir)

        self.env = EWSEnv(cfg)
        self.env.reset_cached(seed=seed)

        ckpt_path = self.vendor_dir / checkpoint_relpath
        if not ckpt_path.exists():
            raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
        blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if not isinstance(blob, dict) or "policy_state_dict" not in blob:
            raise ValueError(
                f"checkpoint at {ckpt_path} missing 'policy_state_dict'"
            )
        pcfg = blob["policy_config"]
        self.model = GATActorCritic(
            node_feature_dims=pcfg["node_feature_dims"],
            edge_feature_dim=0,
            edge_types=pcfg["edge_types"],
            action_count=pcfg["action_count"],
            hidden_dim=pcfg.get("hidden_dim", 64),
            num_layers=pcfg.get("num_layers", 2),
        )
        self.model.load_state_dict(blob["policy_state_dict"])
        self.model.eval()
        self._torch = torch

    def infer(
        self,
        forecast: ForecastSnapshot,
        *,
        storm_phase: float = 0.0,
        time_remaining: float = 1.0,
        resources: float | None = None,
    ) -> LiveTickResult:
        """One-shot inference: forecast → (policy_output, state, mask)."""
        from gnn_drl_ews.plan.mask import compute_mask

        graph, state_dict, _ = self.env.reset_cached(seed=self.seed)
        state_dict["forecast_tide"] = float(forecast.surge_cm)
        state_dict["forecast_wind_wave"] = float(forecast.wind_ms)
        state_dict["storm_phase"] = float(storm_phase)
        state_dict["time_remaining"] = float(time_remaining)
        if resources is not None:
            state_dict["resources"] = float(resources)
        mask_tensor = compute_mask(graph, self.env.plan, state_dict)

        with self._torch.no_grad():
            dist, value = self.model.forward(graph, state_dict, mask_tensor)
        probs = dist.probs.detach().cpu().tolist()
        if len(probs) != NUM_ACTIONS:
            raise RuntimeError(
                f"action_count mismatch: model emits {len(probs)}, "
                f"scenario expects {NUM_ACTIONS}"
            )
        # Renormalise to compensate for any tiny floating-point drift —
        # Pydantic validates `sum(action_probs) == 1.0 ± 1e-3`.
        s = sum(probs)
        if s > 0:
            probs = [p / s for p in probs]

        po = PolicyOutput(
            action_probs=probs,
            value_estimate=float(value.item()),
            storm_type="tide",
        )
        gs = GlobalState(
            forecast_tide=float(state_dict.get("forecast_tide", 0.0)),
            forecast_wind_wave=float(state_dict.get("forecast_wind_wave", 0.0)),
            storm_phase=float(state_dict.get("storm_phase", 0.0)),
            time_remaining=float(state_dict.get("time_remaining", 24.0)),
            resources=float(state_dict.get("resources", 5.0)),
            preparedness=float(state_dict.get("preparedness", 0.0)),
        )
        mask_list = [bool(x) for x in mask_tensor.detach().cpu().tolist()]
        # Defensive: monitor must always remain legal — the orchestrator
        # crashes (Pydantic validator) if mask[0] is False.
        if not mask_list[0]:
            mask_list[0] = True
        return LiveTickResult(policy_output=po, state=gs, mask=mask_list)

    def build_scenario_record(
        self,
        forecast: ForecastSnapshot,
        *,
        tick: int,
        simulated_time: datetime,
        storm_phase: float = 0.0,
        time_remaining: float = 1.0,
        resources: float | None = None,
    ) -> ScenarioRecord:
        """Run one inference and pack into a ScenarioRecord the
        orchestrator can consume directly.
        """
        result = self.infer(
            forecast,
            storm_phase=storm_phase,
            time_remaining=time_remaining,
            resources=resources,
        )
        return ScenarioRecord(
            tick=tick,
            simulated_time=simulated_time,
            forecast=forecast,
            state=result.state,
            mask=result.mask,
            policy_output=result.policy_output,
        )


def _absolutize_config_paths(cfg: dict, vendor_dir: Path) -> None:
    """Rewrite relative paths in the training config to absolute paths
    under vendor_dir, so the env can find caches regardless of cwd.
    """
    for top, sub_keys in (
        ("graph", ["cache_dir"]),
        ("hazards", ["cache_dir"]),
        ("demography", ["cache_dir"]),
    ):
        sec = cfg.get(top, {})
        if isinstance(sec, dict):
            for sk in sub_keys:
                if sk in sec and not Path(str(sec[sk])).is_absolute():
                    sec[sk] = str((vendor_dir / sec[sk]).resolve())

    plan_sec = cfg.get("plan", {})
    if isinstance(plan_sec, dict) and "path" in plan_sec:
        if not Path(str(plan_sec["path"])).is_absolute():
            plan_sec["path"] = str((vendor_dir / plan_sec["path"]).resolve())

    haz_sec = cfg.get("hazards")
    if isinstance(haz_sec, dict):
        haz_excl = haz_sec.get("holdout_exclusions")
        if isinstance(haz_excl, dict) and "csv" in haz_excl:
            if not Path(str(haz_excl["csv"])).is_absolute():
                haz_excl["csv"] = str((vendor_dir / haz_excl["csv"]).resolve())


def recommended_action_id(output: PolicyOutput) -> str:
    """Return the action_id (string) with highest probability."""
    idx = max(range(NUM_ACTIONS), key=lambda i: output.action_probs[i])
    return LIDO_ACTIONS[idx]


def recommended_action_index(output: PolicyOutput) -> int:
    """Return the action index (int) with highest probability."""
    return max(range(NUM_ACTIONS), key=lambda i: output.action_probs[i])


def confidence_signal(
    output: PolicyOutput,
    threshold_high: float = 0.6,
    threshold_medium: float = 0.4,
    split_margin: float = 0.1,
) -> str:
    """Coarse confidence label derived from the action distribution.

    Returns one of: 'high' | 'medium' | 'low' | 'split'.
    'split' = top action is barely ahead of the runner-up; emit a hedge
    signal regardless of absolute probability.
    """
    sorted_probs = sorted(output.action_probs, reverse=True)
    top, second = sorted_probs[0], sorted_probs[1]
    if top - second < split_margin:
        return "split"
    if top >= threshold_high:
        return "high"
    if top >= threshold_medium:
        return "medium"
    return "low"
