"""Eval harness — runs a checkpoint on the autoresearch loop's holdout set.

Per AUTORESEARCH.md §1.4: this is the contract the autoresearch loop reads.
It is also what the post-loop final v1 sweep (Phase 5) uses for paper
tables. Keep the dict schema stable.

Each call evaluates ONE checkpoint over the 20 held-out windows
(`data/eval/lido_holdout.csv`), with K env-stochasticity seeds and N
rollouts per (window, seed) combination. Default 3 × 20 × 5 = 300
episodes.

The autoresearch loop's accept/reject rule (per AUTORESEARCH.md
Decision 3) reads `pooled_std` (over the K per-seed means) — this is
what makes the rule variance-aware.

Until M2 (PPO + GAT) lands, no real policy checkpoints exist. If the
checkpoint is missing, empty, or doesn't carry a recognised policy
state_dict, the harness falls back to a *random-masked* policy with a
one-shot warning. That keeps Phase 1 self-contained: scaffold + smoke
tests pass without depending on PPO infrastructure.
"""

from __future__ import annotations

import hashlib
import warnings
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import torch

from ..config import load_config
from ..env import EWSEnv


REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_HOLDOUT_CSV = REPO_ROOT / "data" / "eval" / "lido_holdout.csv"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "lido_eval_holdout.yaml"
DEFAULT_K_SEEDS = 3
DEFAULT_N_ROLLOUTS = 5


PolicyFn = Callable[[Any, dict, "torch.Tensor", np.random.Generator], int]


def run_eval(
    checkpoint_path: str | Path | None,
    config: dict | None = None,
    *,
    config_path: str | Path | None = None,
    holdout_csv: str | Path | None = None,
    k_seeds: int = DEFAULT_K_SEEDS,
    n_rollouts: int = DEFAULT_N_ROLLOUTS,
    policy_fn: Optional[PolicyFn] = None,
    current_best_return: float | None = None,
) -> dict:
    """Evaluate a checkpoint on the held-out window set.

    Args:
        checkpoint_path: torch checkpoint path. If None, missing, empty, or
            unrecognised, falls back to random-masked policy (with warning).
        config: pre-loaded config dict. If None, loads `config_path` or
            the default `configs/lido_eval_holdout.yaml`.
        config_path: path to a config YAML (used only if `config` is None).
        holdout_csv: path to the holdout CSV (cols: start_iso, peak_cm, bin).
            Defaults to `data/eval/lido_holdout.csv`.
        k_seeds: number of env-stochasticity seeds (axis k of per_seed).
        n_rollouts: rollouts per (window, k_seed) combination.
        policy_fn: optional callable
            (graph, state, mask, rng) -> action_idx. Bypasses checkpoint
            loading when provided. Used by tests and by the autoresearch
            loop to inject specific policy implementations.
        current_best_return: scalar reference for guardrail_return_ratio.
            None on the first iteration; ratio is np.nan in that case.

    Returns:
        Metric battery dict — see AUTORESEARCH.md §1.4 for the schema.
    """
    # ---- Resolve config + holdout list -----------------------------------
    if config is None:
        cfg_path = Path(config_path) if config_path else DEFAULT_CONFIG
        config = load_config(cfg_path)
    holdout_path = Path(holdout_csv) if holdout_csv else DEFAULT_HOLDOUT_CSV
    if not holdout_path.exists():
        raise FileNotFoundError(
            f"Holdout CSV not found: {holdout_path}. "
            f"Run `python scripts/build_holdout_set.py` first."
        )
    holdout = pd.read_csv(holdout_path)
    if "start_iso" not in holdout.columns:
        raise ValueError(f"Holdout CSV {holdout_path} missing 'start_iso' column.")

    # ---- Resolve policy --------------------------------------------------
    if policy_fn is None:
        policy_fn = _load_policy_or_fallback(checkpoint_path)

    # ---- Roll out: K seeds × W windows × N rollouts ---------------------
    metric_keys = ("plan_compliance", "return", "success_rate")
    per_seed_episodes: list[dict[str, list[float]]] = [
        {k: [] for k in metric_keys} for _ in range(k_seeds)
    ]

    # Import here so the test surface for the JSON-schema part of the
    # harness doesn't pay the import cost when only doing a stub run.
    from .plan_compliance import compliance as plan_compliance_fn

    # Build one env per unique window. Graph + hazards are deterministic
    # across the (k_seeds × n_rollouts) episodes for that window in
    # named_event mode, so we reuse via env.reset_cached() — avoids the
    # ~30 s OSM rebuild for every episode on lido_real.
    env_by_window: dict[str, EWSEnv] = {}

    for k in range(k_seeds):
        for w_idx, row in holdout.reset_index(drop=True).iterrows():
            window_iso = str(row["start_iso"])
            if window_iso not in env_by_window:
                env_by_window[window_iso] = EWSEnv(_config_for_window(config, window_iso))
            env = env_by_window[window_iso]
            for n in range(n_rollouts):
                env_seed = _compose_seed(k, window_iso, n)
                ep = _rollout_one_episode(
                    env=env,
                    env_seed=env_seed,
                    policy_fn=policy_fn,
                )
                ep["plan_compliance"] = plan_compliance_fn(ep["trajectory"], _plan_for(config))
                for key in metric_keys:
                    per_seed_episodes[k][key].append(float(ep[key]))

    # ---- Aggregate -------------------------------------------------------
    per_seed = {
        key: [float(np.mean(per_seed_episodes[k][key])) for k in range(k_seeds)]
        for key in metric_keys
    }
    mean = {key: float(np.mean(per_seed[key])) for key in metric_keys}
    pooled_std = {
        key: float(np.std(per_seed[key], ddof=0)) for key in metric_keys
    }

    if current_best_return is not None and current_best_return != 0.0:
        guardrail = float(mean["return"] / current_best_return)
    else:
        guardrail = float("nan")

    return {
        "primary": "plan_compliance",
        "per_seed": per_seed,
        "mean": mean,
        "pooled_std": pooled_std,
        "guardrail_return_ratio": guardrail,
        "n_windows": int(len(holdout)),
        "n_seeds": int(k_seeds),
        "n_rollouts_per_window": int(n_rollouts),
    }


# ---------------------------------------------------------------------------
# Per-episode rollout
# ---------------------------------------------------------------------------

def _rollout_one_episode(
    env: "EWSEnv",
    env_seed: int,
    policy_fn: PolicyFn,
) -> dict:
    """Run a single episode against the pre-built env, return metrics + trajectory.

    Uses env.reset_cached(seed=...) so the OSM-backed graph build runs once
    per window, not once per episode.
    """
    rng = np.random.default_rng(env_seed)
    graph, state, mask = env.reset_cached(seed=env_seed)

    trajectory: list[dict[str, Any]] = []
    cumulative_return = 0.0
    final_unsafe_pop = 0.0
    done = False
    safeguard = 0  # paranoia: env should always terminate at episode_steps

    while not done and safeguard < 10_000:
        action_idx = int(policy_fn(graph, state, mask, rng))
        action_id = env.plan.action_ids[action_idx]
        # Capture mask + action BEFORE stepping — that's the contract
        # plan_compliance reads.
        trajectory.append({
            "mask": np.asarray(mask, dtype=bool).tolist(),
            "action_id": action_id,
            "action_idx": action_idx,
        })
        graph, state, reward, done, info = env.step(action_idx)
        cumulative_return += float(reward)
        # Track unsafe population from reward_components.safety = -unsafe_pop.
        rc = info.get("reward_components", {})
        if "safety" in rc:
            final_unsafe_pop = float(-rc["safety"])
        mask = info.get("mask", mask)
        safeguard += 1

    return {
        "trajectory": trajectory,
        "return": cumulative_return,
        "success_rate": 1.0 if final_unsafe_pop <= 0.0 else 0.0,
    }


def _config_for_window(base_cfg: dict, window_iso: str) -> dict:
    """Shallow copy of base config, set named_event mode pointing at window_iso."""
    import copy
    cfg = copy.deepcopy(base_cfg)
    cfg.setdefault("hazards", {})
    cfg["hazards"]["mode"] = "named_event"
    cfg["hazards"]["named_event"] = window_iso
    return cfg


def _compose_seed(k: int, window_iso: str, n: int) -> int:
    """Deterministic env-seed from (k, window, n). Stable across runs."""
    h = hashlib.md5(f"{k}|{window_iso}|{n}".encode("utf-8")).digest()
    # Take low 31 bits to fit common int seed contracts.
    return int.from_bytes(h[:4], "big") & 0x7FFF_FFFF


def _plan_for(config: dict):
    """Lazy plan loader (cached on the config dict for re-use across calls)."""
    cache = config.setdefault("__cache__", {})
    if "plan" not in cache:
        from ..plan import load_plan
        cache["plan"] = load_plan(config["plan"]["path"])
    return cache["plan"]


# ---------------------------------------------------------------------------
# Policy resolution
# ---------------------------------------------------------------------------

_FALLBACK_WARNED = False


def _load_policy_or_fallback(checkpoint_path: str | Path | None) -> PolicyFn:
    """Load a real PPO checkpoint if possible; otherwise return random-masked.

    Recognises checkpoints written by `src/gnn_drl_ews/policy/ppo_trainer.py`
    (a torch.save dict with `policy_state_dict` + `policy_config`).
    Falls back to a uniformly-random-from-mask policy with a one-shot
    warning if loading fails — that path is what stub-eval mode and the
    very-first iteration use.
    """
    global _FALLBACK_WARNED

    if checkpoint_path is not None:
        path = Path(checkpoint_path)
        if path.exists() and path.stat().st_size > 0 and not str(path).endswith(".stub"):
            try:
                return _build_real_policy_fn(path)
            except Exception as e:
                warnings.warn(
                    f"run_eval: failed to load real policy from {path}: {e}; "
                    f"falling back to random-masked."
                )

    if not _FALLBACK_WARNED:
        warnings.warn(
            "run_eval: using random-masked policy (no real checkpoint or "
            "checkpoint is a stub). Expected for the autoresearch loop's "
            "first iteration before M2 produces real policies."
        )
        _FALLBACK_WARNED = True

    return _random_masked_policy


def _build_real_policy_fn(checkpoint_path: Path) -> PolicyFn:
    """Load a GATActorCritic from a torch checkpoint and wrap it as a PolicyFn."""
    from ..policy.gat_actor_critic import GATActorCritic
    blob = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(blob, dict) or "policy_state_dict" not in blob or "policy_config" not in blob:
        raise ValueError(
            f"checkpoint missing 'policy_state_dict' or 'policy_config' keys "
            f"(found: {list(blob.keys()) if isinstance(blob, dict) else type(blob).__name__})"
        )
    cfg = blob["policy_config"]
    policy = GATActorCritic(
        node_feature_dims=cfg["node_feature_dims"],
        edge_feature_dim=0,
        edge_types=cfg["edge_types"],
        action_count=cfg["action_count"],
        hidden_dim=cfg.get("hidden_dim", 64),
        num_layers=cfg.get("num_layers", 2),
    )
    policy.load_state_dict(blob["policy_state_dict"])
    policy.eval()

    def _policy_fn(graph, state, mask, rng) -> int:
        with torch.no_grad():
            dist, _value = policy(graph, state, mask)
            # Greedy at eval time so results are deterministic per seed.
            return int(dist.probs.argmax().item())

    return _policy_fn


def _random_masked_policy(graph, state, mask, rng: np.random.Generator) -> int:
    """Uniform sample over legal actions; falls back to action 0 if mask empty."""
    arr = np.asarray(mask, dtype=bool)
    legal = np.where(arr)[0]
    if len(legal) == 0:
        return 0
    return int(rng.choice(legal))
