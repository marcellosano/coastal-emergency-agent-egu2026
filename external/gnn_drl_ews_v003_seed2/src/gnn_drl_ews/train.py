"""CLI entry point for one PPO training seed.

Used by the autoresearch loop's `_train_k_seeds` (per AUTORESEARCH.md
Phase 2.1) and standalone for the post-loop final v1 sweep.

    python -m gnn_drl_ews.train \
        --config configs/lido_real.yaml \
        --seed 0 \
        --total-timesteps 100000 \
        --out-dir runs/exp1/seed-0/

Writes:
    <out-dir>/checkpoint.pt   torch state_dict + policy_config + metadata
    <out-dir>/metrics.json    seed, n_episodes, mean_episodic_return, etc.

Exit code: 0 on success, non-zero on failure.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import load_config
from .policy.ppo_trainer import train


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, type=Path,
                    help="Path to scenario config YAML (extends default.yaml).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--total-timesteps", type=int, default=None,
                    help="Override training.total_timesteps (loop uses truncated value).")
    ap.add_argument("--out-dir", required=True, type=Path,
                    help="Directory for checkpoint.pt + metrics.json.")
    ap.add_argument("--quiet", action="store_true",
                    help="Suppress periodic progress prints.")
    ap.add_argument("--no-cache-graph", action="store_true",
                    help="Disable graph-cache fast path (rebuild graph every "
                         "episode reset). For verification/comparison only.")
    args = ap.parse_args()

    if not args.config.exists():
        print(f"[train] FAIL: config not found: {args.config}", file=sys.stderr)
        return 2

    config = load_config(args.config)
    print(f"[train] config={args.config} seed={args.seed} "
          f"total_timesteps={args.total_timesteps or config['training']['total_timesteps']} "
          f"out={args.out_dir}", flush=True)

    outcome = train(
        config=config,
        output_dir=args.out_dir,
        seed=args.seed,
        total_timesteps_override=args.total_timesteps,
        quiet=args.quiet,
        cache_graph=not args.no_cache_graph,
    )

    print(
        f"[train] DONE seed={outcome.seed} "
        f"episodes={outcome.n_episodes} "
        f"mean_return={outcome.mean_episodic_return:+.2f} "
        f"final_return={outcome.final_episodic_return} "
        f"wall={outcome.wallclock_s:.1f}s "
        f"checkpoint={outcome.checkpoint_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
