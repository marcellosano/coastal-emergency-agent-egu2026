"""Plan-action YAML loader.

A `Plan` is the runtime representation of a plan-action YAML. The loader
treats actions as ordered records — the index in `plan.actions` is the
discrete-action id used by the policy and by the action mask.

For node-targeted actions (those declaring `target_node_type`), the policy
later expands them into per-target sub-actions. For M1 the smoke test treats
each plan action as a single discrete action and lets the mask pick a target
when one is needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Action:
    id: str
    description: str = ""
    target_node_type: str | None = None
    preconditions: list[str] = field(default_factory=list)
    required_data: list[str] = field(default_factory=list)
    effect: dict[str, Any] = field(default_factory=dict)
    latency: int = 0
    cost: dict[str, int] = field(default_factory=dict)

    @property
    def is_node_targeted(self) -> bool:
        return self.target_node_type is not None

    @property
    def resource_cost(self) -> int:
        return int(self.cost.get("resources", 0))


@dataclass
class Plan:
    name: str
    hazards: list[dict[str, Any]]
    global_features: list[str]
    thresholds: dict[str, float]
    actions: list[Action]

    @property
    def action_count(self) -> int:
        return len(self.actions)

    @property
    def action_ids(self) -> list[str]:
        return [a.id for a in self.actions]

    def action_by_id(self, action_id: str) -> Action:
        for a in self.actions:
            if a.id == action_id:
                return a
        raise KeyError(f"Unknown action id: {action_id}")


def load_plan(path: str | Path) -> Plan:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Plan file {path} did not parse to a mapping")

    actions_raw = raw.get("actions", [])
    if not actions_raw:
        raise ValueError(f"Plan {path} declares no actions")

    actions: list[Action] = []
    seen_ids: set[str] = set()
    for entry in actions_raw:
        if "id" not in entry:
            raise ValueError(f"Action without id in {path}: {entry}")
        if entry["id"] in seen_ids:
            raise ValueError(f"Duplicate action id {entry['id']} in {path}")
        seen_ids.add(entry["id"])
        actions.append(
            Action(
                id=entry["id"],
                description=entry.get("description", ""),
                target_node_type=entry.get("target_node_type"),
                preconditions=list(entry.get("preconditions", []) or []),
                required_data=list(entry.get("required_data", []) or []),
                effect=dict(entry.get("effect", {}) or {}),
                latency=int(entry.get("latency", 0)),
                cost=dict(entry.get("cost", {}) or {}),
            )
        )

    return Plan(
        name=str(raw.get("plan", path.stem)),
        hazards=list(raw.get("hazards", []) or []),
        global_features=list(raw.get("global_features", []) or []),
        thresholds={k: float(v) for k, v in (raw.get("thresholds", {}) or {}).items()},
        actions=actions,
    )
