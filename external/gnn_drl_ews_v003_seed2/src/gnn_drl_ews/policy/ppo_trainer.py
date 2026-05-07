"""CleanRL-style PPO trainer for plan-conditioned EWS — vectorised (M2.1).

Vectorisation: `num_envs` independent EWSEnv instances, batched HeteroData
forward via PyG's Batch.from_data_list + scatter-mean pooling. Rough
speedup over the v1 single-env implementation: ~5-8x on CPU. With
num_envs=1 the code degrades gracefully to single-env.

Entry point: `train(config, output_dir, ...)`. Used by `train.py` (the CLI
the autoresearch loop subprocesses) and by tests directly.

Hyperparameters consumed from `config["training"]`:
  total_timesteps  (int)        total env steps across ALL envs combined
  num_envs         (int, opt)   parallel envs (default 8)
  rollout_length   (int)        steps PER ENV per PPO update cycle
  minibatch_size   (int)        transitions per gradient minibatch
  epochs           (int)        passes over each rollout buffer
  gamma            (float)      discount
  gae_lambda       (float)      GAE smoothing
  clip_range       (float)      PPO clip epsilon
  lr               (float)      Adam learning rate
  entropy_coef     (float)      entropy bonus weight (program.md hard floor 0.01)
  value_coef       (float, opt) critic loss weight (default 0.5)
  max_grad_norm    (float, opt) gradient clip (default 0.5)

Architecture knobs from `config["policy"]`: hidden_dim, num_layers (heads
defaults to 2). Out of scope for the autoresearch agent per program.md
v1 — the loop's mutable tier is hyperparams only.
"""

from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Batch, HeteroData

from ..env import EWSEnv
from .gat_actor_critic import GATActorCritic, build_actor_critic


@dataclass
class TrainOutcome:
    """Returned by `train()` so callers can log without re-reading the checkpoint."""
    checkpoint_path: Path
    total_timesteps: int
    final_episodic_return: float | None
    mean_episodic_return: float
    n_episodes: int
    wallclock_s: float
    seed: int


@dataclass
class _RolloutBuffer:
    """Flat (rollout_length × num_envs) ordering: stride=num_envs.
    Index i*num_envs + e holds env e's transition at step i. Easy to recover
    per-env trajectories for GAE."""
    graphs: list[HeteroData] = field(default_factory=list)
    states: list[dict[str, Any]] = field(default_factory=list)
    masks: list[torch.Tensor] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    log_probs: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)

    def clear(self) -> None:
        self.graphs.clear(); self.states.clear(); self.masks.clear()
        self.actions.clear(); self.log_probs.clear(); self.values.clear()
        self.rewards.clear(); self.dones.clear()

    def __len__(self) -> int:
        return len(self.actions)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def train(
    config: dict,
    output_dir: Path,
    *,
    seed: int = 0,
    total_timesteps_override: int | None = None,
    quiet: bool = False,
    cache_graph: bool = True,
) -> TrainOutcome:
    """Train one PPO seed. Saves checkpoint.pt + metrics.json under output_dir."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t_cfg = config.get("training", {})
    p_cfg = config.get("policy", {})

    total_timesteps = int(total_timesteps_override or t_cfg.get("total_timesteps", 100_000))
    num_envs = max(1, int(t_cfg.get("num_envs", 8)))
    rollout_length = int(t_cfg.get("rollout_length", 256))
    minibatch_size = int(t_cfg.get("minibatch_size", 64))
    epochs = int(t_cfg.get("epochs", 4))
    gamma = float(t_cfg.get("gamma", 0.99))
    gae_lambda = float(t_cfg.get("gae_lambda", 0.95))
    clip_range = float(t_cfg.get("clip_range", 0.2))
    lr = float(t_cfg.get("lr", 3e-4))
    entropy_coef = max(0.01, float(t_cfg.get("entropy_coef", 0.01)))  # M2-D1 hygiene floor
    value_coef = float(t_cfg.get("value_coef", 0.5))
    max_grad_norm = float(t_cfg.get("max_grad_norm", 0.5))

    hidden_dim = int(p_cfg.get("hidden_dim", 64))
    num_layers = int(p_cfg.get("num_layers", 2))

    # Device. CUDA if available; envs themselves stay on CPU (they're
    # python loops over numpy + torch_geometric HeteroData), the policy +
    # rollout tensors move to GPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not quiet:
        print(f"[ppo] device={device}")

    # Spin up `num_envs` independent envs with distinct seeds. Use
    # cache_graph=True so the OSM-backed graph is built once per env and
    # deep-copied on subsequent episode resets — avoids paying the
    # ~3-5 s build_graph cost per episode (this dominates wall time on
    # lido_real and is why GPU sat idle in earlier sweeps).
    envs = [EWSEnv(copy.deepcopy(config)) for _ in range(num_envs)]
    obs = [envs[i].reset(seed=_env_seed(seed, i), cache_graph=cache_graph) for i in range(num_envs)]
    graphs = [g for g, _, _ in obs]
    states = [s for _, s, _ in obs]
    masks = torch.stack([m for _, _, m in obs]).to(device)  # [num_envs, action_count]

    policy = build_actor_critic(
        graphs[0], envs[0].plan.action_count,
        hidden_dim=hidden_dim, num_layers=num_layers,
    ).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    buf = _RolloutBuffer()
    episode_returns: list[float] = []
    current_episode_return = [0.0] * num_envs

    # Episode reset bookkeeping: each env keeps its own seed counter so resets
    # don't collide across envs.
    env_reset_counter = [0] * num_envs

    t0 = time.time()
    last_log = t0
    steps_per_env = total_timesteps // num_envs

    for step in range(steps_per_env):
        # Batched forward across num_envs graphs.
        with torch.no_grad():
            batch_g = Batch.from_data_list([g.clone() for g in graphs]).to(device)
            dist, values = policy.forward_batch(batch_g, states, masks)
            actions = dist.sample()                # [num_envs]
            log_probs = dist.log_prob(actions)     # [num_envs]

        # Buffer per-env (flat ordering: stride=num_envs).
        for e in range(num_envs):
            buf.graphs.append(graphs[e].clone())
            buf.states.append(dict(states[e]))
            buf.masks.append(masks[e].clone())
            buf.actions.append(int(actions[e].item()))
            buf.log_probs.append(float(log_probs[e].item()))
            buf.values.append(float(values[e].item()))

        # Step each env.
        next_graphs: list[HeteroData] = [None] * num_envs
        next_states: list[dict[str, Any]] = [None] * num_envs
        next_masks_l: list[torch.Tensor] = [None] * num_envs
        rewards = [0.0] * num_envs
        dones = [False] * num_envs
        for e in range(num_envs):
            ng, ns, r, d, info = envs[e].step(int(actions[e].item()))
            rewards[e] = float(r)
            dones[e] = bool(d)
            current_episode_return[e] += float(r)
            if d:
                episode_returns.append(current_episode_return[e])
                current_episode_return[e] = 0.0
                env_reset_counter[e] += 1
                ng, ns, info_mask = envs[e].reset(
                    seed=_env_seed(seed, e) + 100_000 * env_reset_counter[e],
                    cache_graph=cache_graph,
                )
                next_masks_l[e] = info_mask
            else:
                next_masks_l[e] = info.get("mask", masks[e])
            next_graphs[e] = ng
            next_states[e] = ns

        # Buffer rewards/dones for this step (one per env).
        for e in range(num_envs):
            buf.rewards.append(rewards[e])
            buf.dones.append(dones[e])

        graphs = next_graphs
        states = next_states
        masks = torch.stack(next_masks_l).to(device)

        # PPO update at end of rollout (per-env step counter).
        if (step + 1) % rollout_length == 0 and len(buf) >= num_envs * 2:
            with torch.no_grad():
                batch_g = Batch.from_data_list([g.clone() for g in graphs]).to(device)
                _, last_values = policy.forward_batch(batch_g, states, masks)
            _ppo_update(
                policy, optimizer, buf,
                last_values=[float(v.item()) for v in last_values],
                num_envs=num_envs,
                gamma=gamma, gae_lambda=gae_lambda,
                clip_range=clip_range, epochs=epochs,
                minibatch_size=minibatch_size,
                entropy_coef=entropy_coef, value_coef=value_coef,
                max_grad_norm=max_grad_norm,
                device=device,
            )
            buf.clear()

        if not quiet and (time.time() - last_log) > 30.0:
            recent = episode_returns[-10:] or current_episode_return
            print(
                f"[ppo] env_step={step+1}/{steps_per_env} "
                f"total_steps={(step+1)*num_envs}/{total_timesteps} "
                f"episodes={len(episode_returns)} "
                f"recent_return={np.mean(recent):.2f}",
                flush=True,
            )
            last_log = time.time()

    wallclock_s = time.time() - t0

    ckpt_path = output_dir / "checkpoint.pt"
    torch.save({
        "policy_state_dict": policy.state_dict(),
        "policy_config": {
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "action_count": envs[0].plan.action_count,
            "node_feature_dims": policy.node_feature_dims,
            "edge_types": policy.edge_types,
        },
        "total_timesteps": total_timesteps,
        "num_envs": num_envs,
        "seed": seed,
    }, ckpt_path)

    metrics = {
        "seed": seed,
        "total_timesteps": total_timesteps,
        "num_envs": num_envs,
        "n_episodes": len(episode_returns),
        "mean_episodic_return": float(np.mean(episode_returns)) if episode_returns else 0.0,
        "final_episodic_return": float(episode_returns[-1]) if episode_returns else None,
        "wallclock_s": wallclock_s,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    return TrainOutcome(
        checkpoint_path=ckpt_path,
        total_timesteps=total_timesteps,
        final_episodic_return=metrics["final_episodic_return"],
        mean_episodic_return=metrics["mean_episodic_return"],
        n_episodes=metrics["n_episodes"],
        wallclock_s=wallclock_s,
        seed=seed,
    )


def _env_seed(base_seed: int, env_idx: int) -> int:
    """Distinct deterministic seed per env. Stays positive within int32."""
    return (base_seed * 1009 + env_idx * 31) & 0x7FFF_FFFF


# ---------------------------------------------------------------------------
# PPO update + GAE
# ---------------------------------------------------------------------------

def _ppo_update(
    policy: GATActorCritic,
    optimizer: optim.Optimizer,
    buf: _RolloutBuffer,
    *,
    last_values: list[float],
    num_envs: int,
    gamma: float,
    gae_lambda: float,
    clip_range: float,
    epochs: int,
    minibatch_size: int,
    entropy_coef: float,
    value_coef: float,
    max_grad_norm: float,
    device: torch.device | None = None,
) -> None:
    if device is None:
        device = next(policy.parameters()).device
    n = len(buf)  # num_envs × steps
    steps_per_env = n // num_envs

    # Compute per-env GAE then flatten back to the buffer's stride layout.
    advantages_flat = [0.0] * n
    returns_flat = [0.0] * n
    for e in range(num_envs):
        # Per-env slices using stride=num_envs.
        env_rewards = [buf.rewards[i * num_envs + e] for i in range(steps_per_env)]
        env_values = [buf.values[i * num_envs + e] for i in range(steps_per_env)]
        env_dones = [buf.dones[i * num_envs + e] for i in range(steps_per_env)]
        env_advs, env_rets = _compute_gae(
            env_rewards, env_values, env_dones, last_values[e], gamma, gae_lambda,
        )
        for i in range(steps_per_env):
            advantages_flat[i * num_envs + e] = env_advs[i]
            returns_flat[i * num_envs + e] = env_rets[i]

    adv_t = torch.tensor(advantages_flat, dtype=torch.float32, device=device)
    ret_t = torch.tensor(returns_flat, dtype=torch.float32, device=device)
    old_log_prob_t = torch.tensor(buf.log_probs, dtype=torch.float32, device=device)
    action_t = torch.tensor(buf.actions, dtype=torch.long, device=device)

    # Normalise advantages (CleanRL convention).
    if adv_t.numel() > 1:
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

    indices = np.arange(n)
    for _ in range(epochs):
        np.random.shuffle(indices)
        for start in range(0, n, minibatch_size):
            mb_idx = indices[start:start + minibatch_size]
            if len(mb_idx) < 2:
                continue

            # Batched forward over the minibatch.
            mb_graphs = [buf.graphs[i] for i in mb_idx]
            mb_states = [buf.states[i] for i in mb_idx]
            mb_masks = torch.stack([buf.masks[i] for i in mb_idx]).to(device)
            mb_batch = Batch.from_data_list(mb_graphs).to(device)
            mb_actions = action_t[mb_idx]

            new_log_probs, new_values, entropies = policy.evaluate_batch(
                mb_batch, mb_states, mb_masks, mb_actions,
            )

            ratio = torch.exp(new_log_probs - old_log_prob_t[mb_idx])
            clipped = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
            actor_loss = -torch.min(
                ratio * adv_t[mb_idx],
                clipped * adv_t[mb_idx],
            ).mean()
            critic_loss = ((new_values - ret_t[mb_idx]) ** 2).mean()
            entropy_loss = -entropies.mean()

            loss = actor_loss + value_coef * critic_loss + entropy_coef * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()


def _compute_gae(
    rewards: list[float],
    values: list[float],
    dones: list[bool],
    last_value: float,
    gamma: float,
    gae_lambda: float,
) -> tuple[list[float], list[float]]:
    """GAE per-env trajectory."""
    n = len(rewards)
    advantages = [0.0] * n
    last_gae = 0.0
    for t in reversed(range(n)):
        next_value = last_value if t == n - 1 else values[t + 1]
        next_non_terminal = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        advantages[t] = last_gae
    returns = [a + v for a, v in zip(advantages, values)]
    return advantages, returns
