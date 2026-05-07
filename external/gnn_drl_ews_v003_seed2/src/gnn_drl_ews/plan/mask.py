"""Plan-derived action mask.

Contract (HANDOFF §6):
    compute_mask(graph, plan, state) -> torch.BoolTensor[action_count]

A plan action is *available* iff every precondition holds. For node-targeted
actions, "every precondition holds" is interpreted as "there exists at least
one node of `target_node_type` for which every node-level precondition holds
and every global precondition also holds." The mask thus answers: "could the
agent choose this action right now and have a legal application?" The
target-selection step lives in `plan/effects.py`.

The DSL is deliberately tiny — the patterns enumerated in `plans/schema.yaml`
plus a couple of named predicates ("exists path …", "exists alternative
passable edge"). Anything outside the grammar evaluates to False with a
warning, which is the safe direction for a mask.
"""

from __future__ import annotations

import re
import warnings
from typing import Any

import torch
from torch_geometric.data import HeteroData

from ..graph.schema import EDGE_FEATURES, EDGE_TYPES, feature_index
from .loader import Action, Plan

_NUMBER = r"-?\d+(?:\.\d+)?"

_RE_GLOBAL_THRESHOLD = re.compile(rf"^global\.(\w+)\s*(>=|<=|>|<|==)\s*threshold\.(\w+)$")
_RE_GLOBAL_NUMBER = re.compile(rf"^global\.(\w+)\s*(>=|<=|>|<|==)\s*({_NUMBER})$")
_RE_GLOBAL_RESOURCES = re.compile(rf"^global\.resources\s*(>=|<=|>|<|==)\s*(\d+)$")
_RE_NODE_NODE = re.compile(r"^node\.(\w+)\s*(>=|<=|>|<|==)\s*node\.(\w+)$")
_RE_NODE_BOOL = re.compile(r"^node\.(\w+)\s*==\s*(true|false)$", re.IGNORECASE)
_RE_NODE_NUMBER = re.compile(rf"^node\.(\w+)\s*(>=|<=|>|<|==)\s*({_NUMBER})$")


def _cmp(op: str, a: float, b: float) -> bool:
    return {
        ">=": a >= b,
        "<=": a <= b,
        ">": a > b,
        "<": a < b,
        "==": a == b,
    }[op]


def _global_value(state: dict, name: str) -> float | None:
    if name in state:
        return float(state[name])
    return None


def _node_value(graph: HeteroData, node_type: str, idx: int, attr: str) -> float | None:
    if node_type not in graph.node_types:
        return None
    names = list(graph[node_type].feature_names)
    try:
        col = feature_index(names, attr)
    except KeyError:
        return None
    return float(graph[node_type].x[idx, col].item())


def _eval_global_only(expr: str, state: dict, plan: Plan) -> bool:
    """Evaluate a precondition that does not reference `node.…`. Returns True
    if it cannot be parsed under the global lens (so we can fall back to the
    node-level evaluator for mixed expressions)."""
    expr = expr.strip()

    m = _RE_GLOBAL_THRESHOLD.match(expr)
    if m:
        feat, op, thresh_name = m.groups()
        v = _global_value(state, feat)
        if v is None or thresh_name not in plan.thresholds:
            return False
        return _cmp(op, v, float(plan.thresholds[thresh_name]))

    m = _RE_GLOBAL_RESOURCES.match(expr)
    if m:
        op, n = m.groups()
        v = float(state.get("resources", 0.0))
        return _cmp(op, v, float(n))

    m = _RE_GLOBAL_NUMBER.match(expr)
    if m:
        feat, op, num = m.groups()
        v = _global_value(state, feat)
        if v is None:
            return False
        return _cmp(op, v, float(num))

    if expr == "exists alternative passable edge":
        return _exists_alternative_passable_edge(state.get("__graph__"))

    return None  # not a pure-global expression


def _eval_node_expr(expr: str, graph: HeteroData, node_type: str | None, idx: int | None, state: dict) -> bool:
    expr = expr.strip()
    m = _RE_NODE_BOOL.match(expr)
    if m and node_type is not None and idx is not None:
        attr, val = m.groups()
        v = _node_value(graph, node_type, idx, attr)
        if v is None:
            return False
        target = 1.0 if val.lower() == "true" else 0.0
        return v == target

    m = _RE_NODE_NODE.match(expr)
    if m and node_type is not None and idx is not None:
        a, op, b = m.groups()
        va = _node_value(graph, node_type, idx, a)
        vb = _node_value(graph, node_type, idx, b)
        if va is None or vb is None:
            return False
        return _cmp(op, va, vb)

    m = _RE_NODE_NUMBER.match(expr)
    if m and node_type is not None and idx is not None:
        attr, op, num = m.groups()
        v = _node_value(graph, node_type, idx, attr)
        if v is None:
            return False
        return _cmp(op, v, float(num))

    if expr == "exists path(node → shelter, passable)" and node_type is not None and idx is not None:
        return _exists_passable_path_to_shelter(graph, node_type, idx)

    return False


def _exists_passable_path_to_shelter(graph: HeteroData, node_type: str, idx: int) -> bool:
    """Cheap M1 reachability: residential -> road (uses), then road chain
    (connects), then road -> shelter (leads_to). Treats an edge as passable
    if its `passability` attribute > 0."""
    if node_type != "residential":
        return False
    pass_col = EDGE_FEATURES.index("passability")

    # Residential -> first hop road set.
    et_uses = ("residential", "uses", "road")
    if et_uses not in graph.edge_types:
        return False
    e_uses = graph[et_uses].edge_index
    a_uses = graph[et_uses].edge_attr
    mask = (e_uses[0] == idx) & (a_uses[:, pass_col] > 0)
    seed_roads = set(int(x) for x in e_uses[1, mask].tolist())
    if not seed_roads:
        return False

    # BFS over road->road.
    visited_roads = set(seed_roads)
    frontier = list(seed_roads)
    et_rr = ("road", "connects", "road")
    if et_rr in graph.edge_types:
        e_rr = graph[et_rr].edge_index
        a_rr = graph[et_rr].edge_attr
        while frontier:
            nxt: list[int] = []
            for r in frontier:
                m_out = (e_rr[0] == r) & (a_rr[:, pass_col] > 0)
                for d in e_rr[1, m_out].tolist():
                    d = int(d)
                    if d not in visited_roads:
                        visited_roads.add(d)
                        nxt.append(d)
            frontier = nxt

    # Any road in visited set with a passable leads_to -> shelter?
    et_ls = ("road", "leads_to", "shelter")
    if et_ls not in graph.edge_types:
        return False
    e_ls = graph[et_ls].edge_index
    a_ls = graph[et_ls].edge_attr
    mask = a_ls[:, pass_col] > 0
    reachable_road_src = set(int(x) for x in e_ls[0, mask].tolist())
    return bool(visited_roads & reachable_road_src)


def _exists_alternative_passable_edge(graph: HeteroData | None) -> bool:
    if graph is None:
        return False
    pass_col = EDGE_FEATURES.index("passability")
    et_rr = ("road", "connects", "road")
    if et_rr not in graph.edge_types:
        return False
    a = graph[et_rr].edge_attr
    return bool(int((a[:, pass_col] > 0).sum().item()) >= 2)


def _action_available(action: Action, graph: HeteroData, plan: Plan, state: dict) -> tuple[bool, int | None]:
    """Return (available, chosen_target_idx). For non-targeted actions the
    second element is None."""
    state_for_eval = dict(state)
    state_for_eval["__graph__"] = graph

    if not action.is_node_targeted:
        for expr in action.preconditions:
            ok = _eval_global_only(expr, state_for_eval, plan)
            if ok is None:
                # Has node refs in a global-only context — always False.
                warnings.warn(
                    f"Action {action.id}: precondition {expr!r} references node state "
                    "but action is not node-targeted; treating as False."
                )
                return False, None
            if not ok:
                return False, None
        return True, None

    # Node-targeted: split global vs node preconditions.
    nt = action.target_node_type
    if nt not in graph.node_types or graph[nt].x.shape[0] == 0:
        return False, None

    global_preconds: list[str] = []
    node_preconds: list[str] = []
    for expr in action.preconditions:
        v = _eval_global_only(expr, state_for_eval, plan)
        if v is None:
            node_preconds.append(expr)
        else:
            global_preconds.append(expr)

    for expr in global_preconds:
        if not _eval_global_only(expr, state_for_eval, plan):
            return False, None

    n = graph[nt].x.shape[0]
    best_idx: int | None = None
    best_score = -1.0
    # Score eligible targets by water-level pressure (fall back to elevation
    # exposure). M2 will replace this with an actor head.
    pressure_col = _safe_col(graph, nt, "water_level")
    expo_col = _safe_col(graph, nt, "exposure")
    for i in range(n):
        ok = all(_eval_node_expr(expr, graph, nt, i, state_for_eval) for expr in node_preconds)
        if not ok:
            continue
        if pressure_col is not None:
            s = float(graph[nt].x[i, pressure_col].item())
        elif expo_col is not None:
            s = float(graph[nt].x[i, expo_col].item())
        else:
            s = 0.0
        if s > best_score:
            best_score = s
            best_idx = i
    return (best_idx is not None), best_idx


def _safe_col(graph: HeteroData, node_type: str, attr: str) -> int | None:
    if node_type not in graph.node_types:
        return None
    names = list(graph[node_type].feature_names)
    try:
        return feature_index(names, attr)
    except KeyError:
        return None


def compute_mask(graph: HeteroData, plan: Plan, state: dict) -> torch.Tensor:
    """Return a Bool tensor of shape [plan.action_count].

    On top of the YAML preconditions we enforce a universal rule: any action
    with `cost.resources > 0` is masked out if `state.resources < cost`.
    Plan YAMLs may forget to declare this (the placeholder `full_evacuation`
    does), but the policy still must not be allowed to overdraw the budget.
    """
    mask = torch.zeros(plan.action_count, dtype=torch.bool)
    resources = float(state.get("resources", 0.0))
    for i, action in enumerate(plan.actions):
        if action.resource_cost > 0 and resources < action.resource_cost:
            mask[i] = False
            continue
        ok, _ = _action_available(action, graph, plan, state)
        mask[i] = ok
    # Always permit `monitor` as a safety valve.
    for i, a in enumerate(plan.actions):
        if a.id == "monitor":
            mask[i] = True
            break
    return mask


def select_target(action: Action, graph: HeteroData, plan: Plan, state: dict) -> int | None:
    """Sibling helper used by env/effects to recover the chosen target."""
    _, idx = _action_available(action, graph, plan, state)
    return idx
