"""Plan-compliance metric for the autoresearch loop's eval harness.

Definition (per notes/AUTORESEARCH.md Decision 3 + Phase 1.2):

    compliance = mandates_fired_within_window / total_mandates

A *mandate* is triggered at step t when one of the MANDATE_ACTIONS
becomes legal at step t but was not legal at step t-1 (or at t=0 when
already legal). The mandate is *fired on time* if the corresponding
action_id appears in the trajectory at any step in
[t, t + COMPLIANCE_K + action.latency].

If no mandate is triggered in the episode (e.g., a low-severity window
where forecast tide never crosses the alert threshold), the metric
returns 1.0 (vacuously compliant).

Why this definition. The env hard-masks illegal actions at sample time
(plan/mask.py compute_mask), so "agent stayed in mask" is trivially
1.0 — useless as a metric. The compliance signal that matters for a
plan-conditioned policy is the inverse: did the agent act on the
plan's mandatory triggers, on time?

Mandate set rationale (PCE-grounded, hardcoded — the YAML schema does
not carry a `mandatory:` flag):
  - issue_alert      PCE §4.1.4.9 (Sindaco MUST alert when forecast
                     tide >= alert threshold).
  - full_evacuation  PCE §4.1.5.7 (Sindaco MUST order full evac when
                     forecast tide >= full_evac threshold; cedimento
                     difese a mare).

Other plan actions (monitor, deploy_sandbags, close_road, open_shelter,
assisted_evacuation) are PERMITTED under their preconditions but not
strictly mandated by the PCE. They contribute to mean episodic return,
not to plan-compliance.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from ..plan.loader import Plan


MANDATE_ACTIONS: frozenset[str] = frozenset({"issue_alert", "full_evacuation"})
COMPLIANCE_K: int = 2  # base window: observation cycles of perceived-then-acted latency


def compliance(
    trajectory: list[dict[str, Any]],
    plan: Plan,
    mandate_actions: Iterable[str] | None = None,
    k_steps: int = COMPLIANCE_K,
) -> float:
    """Compute the plan-compliance score for one episode trajectory.

    Each trajectory entry MUST contain:
        - "mask": iterable of bools, length plan.action_count, in plan.actions order.
                  May be a torch.BoolTensor, np.ndarray, or list[bool].
    Each entry SHOULD contain one of:
        - "action_id": str — the action_id chosen at this step.
        - "action_idx": int — position into plan.actions.
    Steps with neither contribute no-op (no mandate satisfied at that step).

    Args:
        trajectory: list of per-step records, length T.
        plan: loaded Plan object (for action_ids and per-action latency).
        mandate_actions: action_ids treated as mandates. Defaults to MANDATE_ACTIONS.
        k_steps: base compliance window (added to per-action latency).

    Returns:
        compliance score in [0, 1]. Returns 1.0 if no mandates were triggered.

    Raises:
        KeyError: if a trajectory step lacks a 'mask', or if a mandate is not
                  in plan.actions.
        ValueError: if a 'mask' has the wrong length.
    """
    if mandate_actions is None:
        mandate_actions = MANDATE_ACTIONS
    mandate_set = set(mandate_actions)

    if not trajectory:
        return 1.0

    action_ids = plan.action_ids
    action_idx_by_id = {aid: i for i, aid in enumerate(action_ids)}

    sorted_mandates = sorted(mandate_set)
    mandate_indices: list[int] = []
    for mid in sorted_mandates:
        if mid not in action_idx_by_id:
            raise KeyError(
                f"Mandate '{mid}' not found in plan.actions. "
                f"Plan actions: {action_ids}"
            )
        mandate_indices.append(action_idx_by_id[mid])

    step_action_ids = _normalise_action_ids(trajectory, action_ids)

    T = len(trajectory)
    mask_table = _extract_mandate_mask_table(trajectory, mandate_indices, len(action_ids))

    # Identify trigger steps per mandate: false->true transitions. Also count
    # a True at t=0 as a trigger (the mandate is active from the start).
    triggers: list[tuple[int, str]] = []
    for j, mid in enumerate(sorted_mandates):
        for t in range(T):
            now_legal = bool(mask_table[t, j])
            prev_legal = bool(mask_table[t - 1, j]) if t > 0 else False
            if now_legal and not prev_legal:
                triggers.append((t, mid))

    if not triggers:
        return 1.0

    n_satisfied = 0
    for trigger_t, mid in triggers:
        action = plan.action_by_id(mid)
        window_end = min(T - 1, trigger_t + k_steps + action.latency)
        for t_check in range(trigger_t, window_end + 1):
            if step_action_ids[t_check] == mid:
                n_satisfied += 1
                break

    return n_satisfied / len(triggers)


def _normalise_action_ids(trajectory: list[dict[str, Any]],
                          action_ids: list[str]) -> list[str | None]:
    """Coerce per-step action records to action_id strings (or None)."""
    out: list[str | None] = []
    for entry in trajectory:
        if "action_id" in entry and entry["action_id"] is not None:
            out.append(str(entry["action_id"]))
            continue
        if "action_idx" in entry and entry["action_idx"] is not None:
            ai = int(entry["action_idx"])
            out.append(action_ids[ai] if 0 <= ai < len(action_ids) else None)
            continue
        out.append(None)
    return out


def _extract_mandate_mask_table(
    trajectory: list[dict[str, Any]],
    mandate_indices: list[int],
    action_count: int,
) -> np.ndarray:
    """Return a (T, n_mandates) bool array of mandate-action legality per step."""
    T = len(trajectory)
    table = np.zeros((T, len(mandate_indices)), dtype=bool)
    for t, entry in enumerate(trajectory):
        if "mask" not in entry:
            raise KeyError(f"Trajectory step {t} is missing 'mask'.")
        m_arr = np.asarray(entry["mask"], dtype=bool).reshape(-1)
        if m_arr.shape[0] != action_count:
            raise ValueError(
                f"Trajectory step {t} mask has length {m_arr.shape[0]}; "
                f"expected plan.action_count={action_count}."
            )
        for j, mi in enumerate(mandate_indices):
            table[t, j] = bool(m_arr[mi])
    return table
