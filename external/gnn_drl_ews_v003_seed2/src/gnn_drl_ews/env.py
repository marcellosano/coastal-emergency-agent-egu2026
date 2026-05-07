"""GNN_DRL_EWS environment.

The environment composes:
  - a graph (HeteroData) built from config
  - a hazard trajectory dict (e.g. {"tide": [T], "wind_wave": [T]})
  - a plan (action set + thresholds)

State exposed to the agent / mask:
  state = {
    "t": <int step>,
    "T": <int horizon>,
    "forecast_tide": <float, lookahead 1>,   # canonical name in mask DSL
    "forecast_wind_wave": <float, optional>,
    "storm_phase": <0..1>,
    "time_remaining": <0..1>,
    "resources": <float>,
    "preparedness": <0/1, set by issue_alert>,
    "flag_<x>": <set by global_flag_set effects>,
  }

`step(action_id)` applies the chosen plan action's effect, advances the
hazard, propagates water levels into nodes, and returns the new state +
reward + done. Random-policy agents and PPO agents alike consume the
observation = (graph, state, mask).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch_geometric.data import HeteroData

from .data import build_graph
from .graph.schema import EDGE_FEATURES, feature_index
from .hazards import generate_hazard
from .plan import Plan, load_plan
from .plan.effects import apply_action
from .plan.mask import compute_mask


# Mandate set + window K must mirror eval (`gnn_drl_ews.eval.plan_compliance`).
# A mandate is the subset of plan actions whose firing within K steps of the
# mask flipping false→true is what the eval metric scores. Keeping these as
# module-level constants ensures train-time shaping and test-time scoring
# agree on the trigger semantics.
COMPLIANCE_K = 2
MANDATE_ACTION_IDS = frozenset({"issue_alert", "full_evacuation"})


class EWSEnv:
    def __init__(self, config: dict):
        self.config = config
        self.plan: Plan = load_plan(config["plan"]["path"])
        self.T: int = int(config["hazards"]["T"])
        self.episode_steps: int = int(config["env"]["episode_steps"])
        self.reward_weights = config["reward"]["weights"]

        self.graph: HeteroData | None = None
        self.hazards: dict[str, np.ndarray] | None = None
        self.state: dict[str, Any] | None = None
        self.t: int = 0

        # Caches for reset_cached() — populated lazily on first call.
        self._cached_graph: HeteroData | None = None
        self._cached_hazards: dict[str, np.ndarray] | None = None

        # Mandate-action indices for the compliance reward shaping signal.
        # Restricted to actions that actually exist in the loaded plan.
        self._mandate_action_idx: dict[str, int] = {
            a.id: i
            for i, a in enumerate(self.plan.actions)
            if a.id in MANDATE_ACTION_IDS
        }
        # Per-episode compliance bookkeeping (reset on each episode start).
        self._mandate_state: dict[str, dict] = {}
        self._prev_obs_mask: torch.Tensor | None = None
        self._last_obs_mask: torch.Tensor | None = None

    # ------------------------------------------------------------------
    def reset(
        self,
        seed: int | None = None,
        *,
        cache_graph: bool = False,
    ) -> tuple[HeteroData, dict, torch.Tensor]:
        """Reset the environment.

        Args:
            seed: episode seed (controls hazard sampling for synthetic /
                uniform / severity modes).
            cache_graph: if True, build the graph once on first call, then
                deep-copy from the cache on subsequent resets. Use during
                training when the graph is deterministic from the OSM
                geometry (lido_real) — saves the per-episode build_graph
                cost. Hazards are always regenerated from the seed so
                training stochasticity is preserved.
        """
        import copy
        if seed is not None:
            self.config["seed"] = int(seed)
        if cache_graph:
            if self._cached_graph is None:
                self._cached_graph = build_graph(self.config)
            self.graph = copy.deepcopy(self._cached_graph)
        else:
            self.graph = build_graph(self.config)
        self.hazards = generate_hazard(self.config, self.T)
        return self._initialize_state()

    # ------------------------------------------------------------------
    def reset_cached(self, seed: int | None = None) -> tuple[HeteroData, dict, torch.Tensor]:
        """Reset reusing graph + hazards built on first call.

        Only safe when graph and hazards are deterministic across episodes —
        e.g., real-data named_event mode used by the eval harness. First call
        builds the graph (slow for OSM-backed configs); subsequent calls deep-
        copy the cached graph and reuse the hazard trajectory.
        """
        import copy
        if seed is not None:
            self.config["seed"] = int(seed)
        if self._cached_graph is None or self._cached_hazards is None:
            self._cached_graph = build_graph(self.config)
            self._cached_hazards = generate_hazard(self.config, self.T)
        self.graph = copy.deepcopy(self._cached_graph)
        self.hazards = self._cached_hazards
        return self._initialize_state()

    # ------------------------------------------------------------------
    def _initialize_state(self) -> tuple[HeteroData, dict, torch.Tensor]:
        assert self.graph is not None and self.hazards is not None
        self.t = 0
        self.state = {
            "t": 0,
            "T": self.T,
            "forecast_tide": float(self.hazards["tide"][min(1, self.T - 1)]),
            "storm_phase": 0.0,
            "time_remaining": 1.0,
            "resources": float(self.config["resources"]["initial_budget"]),
        }
        if "wind_wave" in self.hazards:
            self.state["forecast_wind_wave"] = float(self.hazards["wind_wave"][min(1, self.T - 1)])

        # Per-episode compliance bookkeeping resets here, not in __init__,
        # so multi-episode rollouts on the same env instance start clean.
        self._mandate_state = {}
        self._prev_obs_mask = None

        self._propagate_hazard_to_graph()
        mask = compute_mask(self.graph, self.plan, self.state)
        self._last_obs_mask = mask
        return self.graph, dict(self.state), mask

    # ------------------------------------------------------------------
    def step(self, action_idx: int) -> tuple[HeteroData, dict, float, bool, dict]:
        assert self.graph is not None and self.state is not None and self.hazards is not None
        action = self.plan.actions[action_idx]

        # Compliance signal computed BEFORE state changes — uses the mask the
        # agent actually saw when picking action_idx (self._last_obs_mask),
        # compared to the mask seen the previous call (self._prev_obs_mask).
        # See _compute_compliance_signal for trigger semantics.
        compliance_signal = self._compute_compliance_signal(action_idx)
        # Roll the obs-mask history forward for the next step's detection.
        self._prev_obs_mask = self._last_obs_mask

        prev_unsafe = self._unsafe_population_count()

        # Apply effect.
        self.graph, self.state, info = apply_action(action, self.graph, self.plan, self.state)

        # Advance hazard.
        self.t += 1
        self.state["t"] = self.t
        self.state["time_remaining"] = max(0.0, 1.0 - self.t / max(1, self.episode_steps))
        idx_now = min(self.t, self.T - 1)
        idx_fwd = min(self.t + 1, self.T - 1)
        self.state["forecast_tide"] = float(self.hazards["tide"][idx_fwd])
        if "wind_wave" in self.hazards:
            self.state["forecast_wind_wave"] = float(self.hazards["wind_wave"][idx_fwd])
        self.state["storm_phase"] = float(idx_now / max(1, self.T - 1))

        self._propagate_hazard_to_graph()

        new_unsafe = self._unsafe_population_count()
        damage = max(0.0, new_unsafe - prev_unsafe) if prev_unsafe is not None else 0.0
        reward, components = self._compute_reward(action, info, damage, new_unsafe)

        # Compliance shaping. Default weight is 0 in configs/default.yaml,
        # so legacy training is unaffected. When non-zero, training reward
        # gains a per-step ±1 signal aligned with the eval metric.
        w_comp = float(self.reward_weights.get("compliance", 0.0))
        reward += w_comp * compliance_signal
        components["compliance"] = compliance_signal
        info["reward_components"] = components

        done = self.t >= self.episode_steps
        mask = compute_mask(self.graph, self.plan, self.state)
        info["mask"] = mask
        # Cache for the next step's compliance detection.
        self._last_obs_mask = mask
        return self.graph, dict(self.state), reward, done, info

    # ------------------------------------------------------------------
    def _compute_compliance_signal(self, action_idx: int) -> float:
        """Detect mandate triggers and emit a ±1 shaping signal.

        Trigger semantics mirror `eval.plan_compliance.compliance`:
          - A mandate fires *on time* if the agent picks the mandate action
            within `COMPLIANCE_K` steps of its mask flipping false→true.
          - A mandate fires *late* (or not at all) once the K-step window
            has passed without the action being taken.

        Returns:
            +1.0 — mandate fired on time this step (one-shot per trigger).
            -1.0 — mandate window just closed without firing (one-shot per trigger).
             0.0 — no scoring event this step.
        """
        if not self._mandate_action_idx:
            return 0.0  # Plan has no mandates — no signal possible.
        cur_mask = self._last_obs_mask
        if cur_mask is None:
            return 0.0  # Defensive: reset() should have set this.

        signal = 0.0
        for mandate_id, m_idx in self._mandate_action_idx.items():
            was_legal_before = (
                bool(self._prev_obs_mask[m_idx])
                if self._prev_obs_mask is not None
                else False
            )
            is_legal_now = bool(cur_mask[m_idx])

            # New trigger: false→true transition, first time only.
            if not was_legal_before and is_legal_now and mandate_id not in self._mandate_state:
                self._mandate_state[mandate_id] = {
                    "trigger_step": self.t,
                    "fired": False,
                    "scored": False,
                }

            ms = self._mandate_state.get(mandate_id)
            if ms is None or ms["scored"]:
                continue

            window_age = self.t - ms["trigger_step"]
            action_id = self.plan.actions[action_idx].id

            if not ms["fired"] and action_id == mandate_id and window_age <= COMPLIANCE_K:
                ms["fired"] = True
                ms["scored"] = True
                signal += 1.0
            elif not ms["fired"] and window_age > COMPLIANCE_K:
                ms["scored"] = True
                signal -= 1.0
        return signal

    # ------------------------------------------------------------------
    def _propagate_hazard_to_graph(self) -> None:
        """Lift the current tide level onto exposed residential and road
        nodes so node.water_level reflects the hazard. Local barriers raise
        local_flood_threshold and partly absorb the level."""
        assert self.graph is not None and self.hazards is not None
        idx_now = min(self.t, self.T - 1)
        tide_now = float(self.hazards["tide"][idx_now])

        for nt in ("residential", "road"):
            if nt not in self.graph.node_types:
                continue
            wl_col = self._col(nt, "water_level")
            fwl_col = self._col(nt, "forecast_water_level")
            elev_col = self._col(nt, "elevation")
            local_th_col = self._col(nt, "local_flood_threshold")

            if wl_col is None or elev_col is None:
                continue

            elev = self.graph[nt].x[:, elev_col]
            # Effective inundation = max(tide - elev, 0), softened by local barrier.
            level = torch.clamp(tide_now - elev, min=0.0)
            if local_th_col is not None:
                # If a barrier raised local threshold, subtract that delta.
                base = float(self.config["thresholds"]["road_flood_default"])
                barrier = torch.clamp(self.graph[nt].x[:, local_th_col] - base, min=0.0)
                level = torch.clamp(level - barrier, min=0.0)
            self.graph[nt].x[:, wl_col] = level

            if fwl_col is not None:
                idx_fwd = min(self.t + 1, self.T - 1)
                tide_fwd = float(self.hazards["tide"][idx_fwd])
                fwd = torch.clamp(tide_fwd - elev, min=0.0)
                self.graph[nt].x[:, fwl_col] = fwd

    def _unsafe_population_count(self) -> float | None:
        assert self.graph is not None
        if "residential" not in self.graph.node_types:
            return None
        pop_col = self._col("residential", "population")
        evac_col = self._col("residential", "evacuated")
        wl_col = self._col("residential", "water_level")
        local_th_col = self._col("residential", "local_flood_threshold")
        if pop_col is None or wl_col is None:
            return None
        x = self.graph["residential"].x
        unsafe_mask = x[:, wl_col] > 0  # any inundation
        if local_th_col is not None:
            base = float(self.config["thresholds"]["road_flood_default"])
            barrier = torch.clamp(x[:, local_th_col] - base, min=0.0)
            unsafe_mask = (x[:, wl_col] > 0) & ((x[:, wl_col] + barrier) > 1e-3)
        if evac_col is not None:
            unsafe_mask = unsafe_mask & (x[:, evac_col] < 0.5)
        return float((x[unsafe_mask, pop_col]).sum().item())

    def _compute_reward(
        self,
        action,
        info: dict,
        damage: float,
        unsafe_pop: float | None,
    ) -> tuple[float, dict]:
        w = self.reward_weights
        # safety: negative of new unsafe population (normalised). Higher is
        # better.
        safety_raw = -(unsafe_pop or 0.0)

        # connectivity: fraction of road nodes still passable from a "spine"
        # perspective (passability > 0 on at least one outgoing edge).
        connectivity = self._connectivity_score()

        # damage already computed (positive number is bad).
        damage_term = -damage

        # resource cost (non-negative).
        resource_cost = float(action.resource_cost)

        scale_pop = 100.0  # normalise hundreds of people
        reward = (
            float(w["safety"]) * safety_raw / scale_pop
            + float(w["connectivity"]) * connectivity
            + float(w["damage"]) * damage_term / scale_pop
            - float(w["resource_cost"]) * resource_cost
        )
        return reward, {
            "safety": safety_raw,
            "connectivity": connectivity,
            "damage": damage,
            "resource_cost": resource_cost,
        }

    def _connectivity_score(self) -> float:
        assert self.graph is not None
        et = ("road", "connects", "road")
        if et not in self.graph.edge_types or self.graph[et].edge_index.shape[1] == 0:
            return 0.0
        pass_col = EDGE_FEATURES.index("passability")
        a = self.graph[et].edge_attr
        return float((a[:, pass_col] > 0).float().mean().item())

    def _col(self, node_type: str, attr: str) -> int | None:
        assert self.graph is not None
        if node_type not in self.graph.node_types:
            return None
        names = list(self.graph[node_type].feature_names)
        try:
            return feature_index(names, attr)
        except KeyError:
            return None
