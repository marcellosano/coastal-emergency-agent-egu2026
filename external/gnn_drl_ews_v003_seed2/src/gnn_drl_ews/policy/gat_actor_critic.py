"""GAT actor-critic over the heterogeneous EWS graph.

Architecture (per AUTORESEARCH.md M2 design + program.md v1 rubric):

    HeteroData (5 node types, 6 edge types)
        |
        v
    HeteroConv(GATConv per edge type) x num_layers
        |
        v
    Per-node-type mean pool -> concat with global state features
        |
        v
    +-- Actor head -----+   +-- Critic head ------+
    | Linear -> logits  |   | Linear -> scalar V  |
    | masked softmax    |   |                     |
    +-------------------+   +---------------------+

The action space is the flat plan.action_count (one logit per plan action).
Node target selection for node-targeted actions is handled deterministically
by `plan/mask.py::select_target` after the policy picks an action — see
DECISIONS [M1-D1] for the rationale (avoids exploding the action space).

Action masking: illegal actions get logit -inf so they contribute zero
mass to the softmax. The categorical distribution then samples only from
legal actions.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATConv, HeteroConv

from ..graph.schema import NODE_TYPES


# Global state features the actor-critic concatenates onto the pooled graph
# embedding before the heads. Order is fixed for checkpoint stability.
GLOBAL_FEATURES = (
    "forecast_tide",
    "forecast_wind_wave",
    "storm_phase",
    "time_remaining",
    "resources",
    "preparedness",
)


class GATActorCritic(nn.Module):
    """GAT encoder + flat actor + scalar critic for plan-conditioned EWS PPO.

    Args:
        node_feature_dims: dict[node_type -> int]; per-node-type input feature width.
        edge_feature_dim: int; edge attribute width (passed through but unused at v1).
        edge_types: list of (src, rel, dst) tuples for the HeteroConv.
        action_count: number of plan actions (= softmax width).
        hidden_dim: GAT hidden width per layer.
        num_layers: number of stacked GAT layers.
        heads: GAT attention heads (concatenated).
        global_features: ordered list of global-state feature names to
            concat onto the pooled embedding.
    """

    def __init__(
        self,
        node_feature_dims: dict[str, int],
        edge_feature_dim: int,
        edge_types: list[tuple[str, str, str]],
        action_count: int,
        *,
        hidden_dim: int = 64,
        num_layers: int = 2,
        heads: int = 2,
        global_features: tuple[str, ...] = GLOBAL_FEATURES,
    ) -> None:
        super().__init__()
        self.node_types = tuple(sorted(node_feature_dims.keys()))
        self.edge_types = list(edge_types)
        self.action_count = int(action_count)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.heads = int(heads)
        self.global_features = tuple(global_features)
        self.node_feature_dims = dict(node_feature_dims)

        # Per-node-type input projections so all node types share the same
        # hidden_dim before message passing (HeteroConv requires this).
        self.input_proj = nn.ModuleDict({
            nt: nn.Linear(node_feature_dims[nt], hidden_dim)
            for nt in self.node_types
        })

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv(
                {
                    et: GATConv(
                        in_channels=hidden_dim,
                        out_channels=hidden_dim // heads,
                        heads=heads,
                        concat=True,
                        add_self_loops=False,
                    )
                    for et in edge_types
                },
                aggr="sum",
            )
            self.convs.append(conv)

        # Heads: input = mean-pooled embedding per node type (concatenated)
        # + global state vector.
        feature_width = hidden_dim * len(self.node_types) + len(self.global_features)
        self.actor = nn.Sequential(
            nn.Linear(feature_width, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_count),
        )
        self.critic = nn.Sequential(
            nn.Linear(feature_width, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    # ------------------------------------------------------------------
    def encode(self, graph: HeteroData) -> dict[str, torch.Tensor]:
        """Run the GAT stack; return {node_type: [n_nodes, hidden_dim]}.

        Source-only node types (depot, hazard_zone in the EWS schema) are
        never targets of any edge, so HeteroConv drops them from its
        output dict. Carry them forward unchanged between layers; the
        downstream pool concatenates ALL node-type means, so dropping any
        of them would shrink the pooled vector and break the head dims.
        """
        x_dict = {nt: self.input_proj[nt](graph[nt].x) for nt in self.node_types
                  if nt in graph.node_types}
        for conv in self.convs:
            updated = conv(x_dict, graph.edge_index_dict)
            updated = {nt: F.relu(x) for nt, x in updated.items()}
            # Carry forward node types that didn't get updated this layer.
            for nt in x_dict:
                if nt not in updated:
                    updated[nt] = x_dict[nt]
            x_dict = updated
        return x_dict

    # ------------------------------------------------------------------
    def _pool(self, x_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        for nt in self.node_types:
            if nt in x_dict and x_dict[nt].numel() > 0:
                parts.append(x_dict[nt].mean(dim=0))
            else:
                parts.append(torch.zeros(self.hidden_dim, dtype=torch.float32))
        return torch.cat(parts, dim=0)

    def _global_vector(self, state: dict[str, Any]) -> torch.Tensor:
        # Built on CPU; caller must .to(device) when concatenating with
        # encoder outputs (forward / forward_batch handle this).
        return torch.tensor(
            [float(state.get(name, 0.0)) for name in self.global_features],
            dtype=torch.float32,
        )

    # ------------------------------------------------------------------
    def forward(
        self,
        graph: HeteroData,
        state: dict[str, Any],
        mask: torch.Tensor,
    ) -> tuple[Categorical, torch.Tensor]:
        """Return (action distribution, value scalar)."""
        x_dict = self.encode(graph)
        pooled = self._pool(x_dict)
        gvec = self._global_vector(state).to(pooled.device)
        features = torch.cat([pooled, gvec], dim=0)

        logits = self.actor(features)
        value = self.critic(features).squeeze(-1)

        # Mask illegal actions: -inf logits -> zero softmax mass.
        m = mask.to(dtype=torch.bool)
        if not m.any():
            # Should never happen — mask.py always permits `monitor` — but
            # be defensive: fall back to uniform over all actions.
            m = torch.ones_like(m)
        masked_logits = logits.masked_fill(~m, float("-inf"))
        return Categorical(logits=masked_logits), value

    # ------------------------------------------------------------------
    def evaluate(
        self,
        graph: HeteroData,
        state: dict[str, Any],
        mask: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Re-evaluate a sampled action under the current policy.

        Returns (log_prob, value, entropy).
        """
        dist, value = self.forward(graph, state, mask)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, value, entropy

    # ------------------------------------------------------------------
    # Vectorised forward (M2.1) — operates on a Batch of B HeteroData
    # graphs collated via Batch.from_data_list. ~5-8x speedup vs calling
    # forward(graph, state, mask) in a Python loop.
    # ------------------------------------------------------------------

    def forward_batch(
        self,
        batch_graph: HeteroData,
        states: list[dict[str, Any]],
        masks: torch.Tensor,
    ) -> tuple[Categorical, torch.Tensor]:
        """Forward over B graphs at once. Returns (Categorical[B, A], values[B])."""
        B = len(states)
        x_dict = self.encode(batch_graph)
        pooled = self._batched_pool(x_dict, batch_graph, B)
        gvec = torch.stack([self._global_vector(s) for s in states]).to(pooled.device)  # [B, n_global]
        features = torch.cat([pooled, gvec], dim=1)                    # [B, feature_width]

        logits = self.actor(features)                                   # [B, action_count]
        values = self.critic(features).squeeze(-1)                      # [B]

        m = masks.to(dtype=torch.bool)
        if not m.any(dim=1).all():
            # Defensive: any all-False mask row falls back to all-True.
            no_legal = ~m.any(dim=1)
            m = m.clone()
            m[no_legal] = True
        masked_logits = logits.masked_fill(~m, float("-inf"))
        return Categorical(logits=masked_logits), values

    def evaluate_batch(
        self,
        batch_graph: HeteroData,
        states: list[dict[str, Any]],
        masks: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Re-evaluate B sampled actions in one forward. Returns (log_probs[B], values[B], entropy[B])."""
        dist, values = self.forward_batch(batch_graph, states, masks)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, values, entropy

    def _batched_pool(
        self,
        x_dict: dict[str, torch.Tensor],
        batch_graph: HeteroData,
        B: int,
    ) -> torch.Tensor:
        """Per-node-type mean pool across the batch dimension.

        For each node type, scatter_mean groups the per-node embeddings by
        the .batch index (which graph each node belongs to). Returns
        [B, hidden_dim * n_node_types].
        """
        from torch_geometric.utils import scatter
        parts: list[torch.Tensor] = []
        for nt in self.node_types:
            if nt in x_dict and x_dict[nt].numel() > 0:
                # batch index lives on the batched HeteroData per-node-type store.
                batch_idx = getattr(batch_graph[nt], "batch", None)
                if batch_idx is None:
                    # Single-graph case (B=1): node-type dim collapses to mean.
                    pooled = x_dict[nt].mean(dim=0, keepdim=True)
                    if pooled.shape[0] != B:
                        pooled = pooled.expand(B, -1)
                else:
                    pooled = scatter(
                        x_dict[nt], batch_idx, dim=0, dim_size=B, reduce="mean",
                    )
                parts.append(pooled)
            else:
                parts.append(torch.zeros(B, self.hidden_dim, dtype=torch.float32))
        return torch.cat(parts, dim=1)


# ---------------------------------------------------------------------------
# Convenience builder that reads dims from a sample env.reset() output.
# ---------------------------------------------------------------------------

def build_actor_critic(
    sample_graph: HeteroData,
    action_count: int,
    *,
    hidden_dim: int = 64,
    num_layers: int = 2,
    heads: int = 2,
) -> GATActorCritic:
    """Inspect a sample HeteroData to read per-node-type feature widths."""
    node_feature_dims = {
        nt: int(sample_graph[nt].x.shape[1])
        for nt in sample_graph.node_types
        if hasattr(sample_graph[nt], "x")
    }
    edge_types = list(sample_graph.edge_types)
    return GATActorCritic(
        node_feature_dims=node_feature_dims,
        edge_feature_dim=0,
        edge_types=edge_types,
        action_count=action_count,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        heads=heads,
    )
