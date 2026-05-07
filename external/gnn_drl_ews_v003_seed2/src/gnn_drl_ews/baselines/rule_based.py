"""Rule-based PCE baselines for v1 paper deliverables.

Two policies live here, spanning the doctrine spectrum:

    1. PCE-strict (`build_rule_based_policy_fn`) — the literal
       mandates-only baseline. Fires only the 2 PCE actions with
       hard numerical thresholds (issue_alert, full_evacuation);
       monitors otherwise. Ignores the 5 discretionary PCE actions
       because doctrine provides only qualitative preconditions for
       those, no numerical triggers.

    2. PCE-extended heuristic (`build_pce_extended_heuristic_policy_fn`)
       — a principled extension that operationalises the discretionary
       actions on heuristic triggers derived from PCE qualitative
       criteria ("activate shelters in alert phase", etc.). The
       discretionary thresholds are explicitly arbitrary defaults; the
       paper's §X positions them as the strongest reasonable
       hand-engineered baseline against which to measure learned-policy
       value.

Both policies project through the action mask; if a recommended action
is illegal at the current step (precondition unsatisfied, e.g. lacking
resources), they fall back through a precedence chain to monitor.

Used by `scripts/sweep_v1.py` as the two structured baselines (random
-masked is the third). Per notes/BLOCKED.md [B3], the alert threshold
is operationally cited (110 cm); the full-evac threshold (150 cm) is a
parametric placeholder pending the 2002 Piano integrato alta marea
lookup. The §8 limitations paragraph covers the provenance.

Contract: each policy returns a `policy_fn` matching the
`Callable[[Any, dict, torch.Tensor, np.random.Generator], int]`
signature consumed by `eval.run_eval._rollout_one_episode`.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch

from ..plan.loader import Plan


def build_rule_based_policy_fn(plan: Plan) -> Callable:
    """Return a policy_fn closure that fires PCE-threshold actions.

    Reads `plan.thresholds` and `plan.action_ids` once at construction,
    so per-step cost is two scalar comparisons + one mask lookup.
    """
    action_ids: list[str] = list(plan.action_ids)
    aidx = {aid: i for i, aid in enumerate(action_ids)}

    if "issue_alert" not in aidx or "full_evacuation" not in aidx:
        raise ValueError(
            "rule_based baseline requires 'issue_alert' and 'full_evacuation' "
            "in plan.actions (PCE-grounded mandate set). Got: "
            f"{action_ids}"
        )
    if "alert_tide" not in plan.thresholds or "full_evac" not in plan.thresholds:
        raise ValueError(
            "rule_based baseline requires plan.thresholds['alert_tide'] and "
            "['full_evac']. Got thresholds: "
            f"{list(plan.thresholds.keys())}"
        )

    alert_threshold = float(plan.thresholds["alert_tide"])
    full_evac_threshold = float(plan.thresholds["full_evac"])

    monitor_idx = aidx.get("monitor")
    issue_alert_idx = aidx["issue_alert"]
    full_evac_idx = aidx["full_evacuation"]

    def policy_fn(graph: Any, state: dict, mask: torch.Tensor,
                  rng: np.random.Generator) -> int:
        # Mask may be torch.BoolTensor, np.ndarray, or list-like; coerce once.
        m = np.asarray(mask, dtype=bool).reshape(-1)
        forecast_tide = float(state.get("forecast_tide", 0.0))

        # Strict precedence: full_evac > alert > monitor > first-legal.
        if forecast_tide >= full_evac_threshold and m[full_evac_idx]:
            return full_evac_idx
        if forecast_tide >= alert_threshold and m[issue_alert_idx]:
            return issue_alert_idx
        if monitor_idx is not None and m[monitor_idx]:
            return monitor_idx
        # Fallback: first legal action. Should be rare given monitor has no preconditions.
        legal = np.flatnonzero(m)
        if legal.size == 0:
            raise RuntimeError("rule_based fallback: zero legal actions in mask")
        return int(legal[0])

    return policy_fn


# ---------------------------------------------------------------------------
# PCE-extended heuristic baseline
# ---------------------------------------------------------------------------

def build_pce_extended_heuristic_policy_fn(plan: Plan) -> Callable:
    """Return a policy_fn closure that uses ALL 7 PCE-grounded actions.

    Discretionary triggers are derived from PCE qualitative criteria
    (e.g., "activate shelters when alert phase is active"), with
    explicit numerical defaults documented in the paper.

    Precedence (highest first; first action with both heuristic-trigger
    satisfied AND mask bit True is chosen):

        1. full_evacuation        forecast_tide >= full_evac (mandate)
        2. issue_alert            forecast_tide >= alert AND not preparedness (mandate)
        3. close_road             mask bit set (precondition: any road inundated)
        4. assisted_evacuation    forecast_tide >= (alert+full_evac)/2
        5. open_shelter           forecast_tide >= alert
        6. deploy_sandbags        forecast_tide >= alert
        7. monitor                otherwise

    The "(alert + full_evac)/2" threshold for assisted_evacuation is
    the principled mid-alert proactive evacuation point, halfway between
    "raise public alert" and "order area-wide evacuation". This is the
    most defensible default for a qualitative PCE precondition.
    """
    action_ids: list[str] = list(plan.action_ids)
    aidx = {aid: i for i, aid in enumerate(action_ids)}

    if "issue_alert" not in aidx or "full_evacuation" not in aidx:
        raise ValueError(
            "pce_extended_heuristic baseline requires 'issue_alert' and "
            "'full_evacuation' in plan.actions. Got: " + str(action_ids)
        )
    if "alert_tide" not in plan.thresholds or "full_evac" not in plan.thresholds:
        raise ValueError(
            "pce_extended_heuristic baseline requires plan.thresholds['alert_tide'] "
            "and ['full_evac']. Got: " + str(list(plan.thresholds.keys()))
        )

    alert_threshold = float(plan.thresholds["alert_tide"])
    full_evac_threshold = float(plan.thresholds["full_evac"])
    proactive_evac_threshold = (alert_threshold + full_evac_threshold) / 2.0

    monitor_idx = aidx.get("monitor")
    issue_alert_idx = aidx["issue_alert"]
    full_evac_idx = aidx["full_evacuation"]
    deploy_sandbags_idx = aidx.get("deploy_sandbags")
    close_road_idx = aidx.get("close_road")
    open_shelter_idx = aidx.get("open_shelter")
    assisted_evac_idx = aidx.get("assisted_evacuation")

    def policy_fn(graph: Any, state: dict, mask: torch.Tensor,
                  rng: np.random.Generator) -> int:
        m = np.asarray(mask, dtype=bool).reshape(-1)
        forecast_tide = float(state.get("forecast_tide", 0.0))
        preparedness = bool(state.get("preparedness", False))

        # 1. full_evacuation (mandate)
        if forecast_tide >= full_evac_threshold and m[full_evac_idx]:
            return full_evac_idx

        # 2. issue_alert (mandate, one-shot)
        if forecast_tide >= alert_threshold and not preparedness and m[issue_alert_idx]:
            return issue_alert_idx

        # 3. close_road (immediate response: any road's water_level over flood threshold,
        #    encoded in the mask precondition)
        if close_road_idx is not None and m[close_road_idx]:
            return close_road_idx

        # 4. assisted_evacuation (proactive vulnerable evac at mid-alert)
        if (assisted_evac_idx is not None
                and forecast_tide >= proactive_evac_threshold
                and m[assisted_evac_idx]):
            return assisted_evac_idx

        # 5. open_shelter (proactive activation at alert phase)
        if (open_shelter_idx is not None
                and forecast_tide >= alert_threshold
                and m[open_shelter_idx]):
            return open_shelter_idx

        # 6. deploy_sandbags (proactive flood protection at alert phase)
        if (deploy_sandbags_idx is not None
                and forecast_tide >= alert_threshold
                and m[deploy_sandbags_idx]):
            return deploy_sandbags_idx

        # 7. monitor (default)
        if monitor_idx is not None and m[monitor_idx]:
            return monitor_idx

        # Fallback: first legal action.
        legal = np.flatnonzero(m)
        if legal.size == 0:
            raise RuntimeError("pce_extended_heuristic fallback: zero legal actions in mask")
        return int(legal[0])

    return policy_fn
