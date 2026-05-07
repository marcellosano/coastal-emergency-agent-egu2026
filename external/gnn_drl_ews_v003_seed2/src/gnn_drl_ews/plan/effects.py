"""Plan effects: apply a chosen action's effect to the graph + state.

Effect types supported (matching plan/schema.yaml):
    - noop
    - global_flag_set
    - node_attribute_set        (set scalar OR add `delta`)
    - edge_attribute_set        (target = "passability" → flips edges incident to the chosen node)
    - global_action             (e.g. evacuate all_residential)
    - routing_change            (records a flag in state; used by reward)

Returns the (mutated) graph and updated state dict. Resource cost is debited
inside this function so the env loop stays simple.
"""

from __future__ import annotations

from typing import Any

import torch
from torch_geometric.data import HeteroData

from ..graph.schema import EDGE_FEATURES, feature_index
from .loader import Action, Plan
from .mask import select_target


def apply_action(
    action: Action,
    graph: HeteroData,
    plan: Plan,
    state: dict[str, Any],
    chosen_target: int | None = None,
) -> tuple[HeteroData, dict[str, Any], dict[str, Any]]:
    """Apply `action` and return (graph, state, info).

    `info` reports what happened — useful for the reward and for the
    light-scope HITL trace later (M4).
    """
    info: dict[str, Any] = {
        "action_id": action.id,
        "target_idx": None,
        "applied": True,
        "trace": {
            "preconditions": list(action.preconditions),
            "effect": dict(action.effect),
            "resource_cost": action.resource_cost,
        },
    }

    # Resource debit (mask already verified availability).
    if action.resource_cost > 0:
        state["resources"] = float(state.get("resources", 0.0)) - action.resource_cost

    eff = action.effect
    etype = eff.get("type", "noop")

    if etype == "noop":
        return graph, state, info

    if etype == "global_flag_set":
        target = eff["target"]
        value = eff.get("value", True)
        state[f"flag_{target}"] = value
        return graph, state, info

    if etype == "global_action":
        # e.g. evacuate all_residential
        if eff.get("target") == "all_residential" and eff.get("value") == "evacuated":
            evac_col = _col(graph, "residential", "evacuated")
            if evac_col is not None:
                graph["residential"].x[:, evac_col] = 1.0
        info["target_idx"] = -1
        return graph, state, info

    # Node- or edge-targeted effects need a chosen target.
    if chosen_target is None:
        chosen_target = select_target(action, graph, plan, state)
    if chosen_target is None:
        info["applied"] = False
        return graph, state, info
    info["target_idx"] = int(chosen_target)

    if etype == "node_attribute_set":
        nt = action.target_node_type
        attr = eff["target"]
        col = _col(graph, nt, attr)
        if col is None:
            info["applied"] = False
            return graph, state, info
        if "delta" in eff:
            graph[nt].x[chosen_target, col] = graph[nt].x[chosen_target, col] + float(eff["delta"])
        else:
            value = eff.get("value", True)
            scalar = 1.0 if value is True else 0.0 if value is False else float(value)
            graph[nt].x[chosen_target, col] = scalar
        return graph, state, info

    if etype == "edge_attribute_set":
        # Target is an edge attribute — flip every edge incident to
        # `chosen_target` of the action's node type. For close_road this
        # blanks passability on every edge touching that road node.
        attr = eff["target"]
        if attr not in EDGE_FEATURES:
            info["applied"] = False
            return graph, state, info
        col = EDGE_FEATURES.index(attr)
        scalar = float(eff.get("value", 0.0))
        nt = action.target_node_type
        for et in list(graph.edge_types):
            if nt not in (et[0], et[2]):
                continue
            ei = graph[et].edge_index
            ea = graph[et].edge_attr
            if ei.shape[1] == 0:
                continue
            mask = torch.zeros(ei.shape[1], dtype=torch.bool)
            if et[0] == nt:
                mask = mask | (ei[0] == chosen_target)
            if et[2] == nt:
                mask = mask | (ei[1] == chosen_target)
            ea[mask, col] = scalar
        return graph, state, info

    if etype == "routing_change":
        state["flag_rerouted"] = True
        return graph, state, info

    info["applied"] = False
    return graph, state, info


def _col(graph: HeteroData, node_type: str | None, attr: str) -> int | None:
    if node_type is None or node_type not in graph.node_types:
        return None
    names = list(graph[node_type].feature_names)
    try:
        return feature_index(names, attr)
    except KeyError:
        return None
