"""Rule-based trigger — decides when standby transitions to active and back.

The trigger is deliberately simple: a small set of inequality rules over the
forecast. The defaults below mirror the PCE Lido alert threshold (110 cm
surge); they are placeholders until Cowork supplies the doctrine-grounded
trigger spec (one of the four still-open §3.5 questions).

Two evaluations:
  - `evaluate_activation` — should standby become active right now?
  - `evaluate_standdown`  — should this active incident close right now?

Both are pure functions of inputs and config. The Orchestrator owns the
consecutive-below-threshold counter and threads it through `evaluate_standdown`
across ticks.
"""

from __future__ import annotations

from dataclasses import dataclass

from coastal_agent.scenario import ForecastSnapshot


@dataclass(frozen=True)
class TriggerConfig:
    """Per-case-study trigger thresholds. Defaults track the PCE Lido alert
    surge threshold; override per case study."""

    case_study: str = "lido"
    activation_surge_cm: float = 110.0
    standdown_surge_cm: float = 100.0
    standdown_consecutive_ticks: int = 2


@dataclass(frozen=True)
class TriggerEvaluation:
    """Pure description of a trigger check at one tick."""

    fired: bool
    rule: str
    summary: str
    values: dict[str, float]


def evaluate_activation(
    forecast: ForecastSnapshot,
    config: TriggerConfig,
) -> TriggerEvaluation:
    """Whether this tick's forecast should activate a new incident."""
    if forecast.surge_cm >= config.activation_surge_cm:
        return TriggerEvaluation(
            fired=True,
            rule="surge_above_activation",
            summary=(
                f"Forecast surge {forecast.surge_cm:.0f} cm meets or exceeds "
                f"activation threshold {config.activation_surge_cm:.0f} cm"
            ),
            values={
                "surge_cm": forecast.surge_cm,
                "activation_threshold_cm": config.activation_surge_cm,
            },
        )
    return TriggerEvaluation(
        fired=False,
        rule="surge_below_activation",
        summary=(
            f"Forecast surge {forecast.surge_cm:.0f} cm below "
            f"activation threshold {config.activation_surge_cm:.0f} cm"
        ),
        values={
            "surge_cm": forecast.surge_cm,
            "activation_threshold_cm": config.activation_surge_cm,
        },
    )


def evaluate_standdown(
    forecast: ForecastSnapshot,
    consecutive_below_before: int,
    config: TriggerConfig,
) -> TriggerEvaluation:
    """Whether stand-down should fire this tick.

    Stand-down requires the surge to be below `standdown_surge_cm` for
    `standdown_consecutive_ticks` consecutive ticks. The caller passes the
    counter state from the previous tick; this function returns the new
    count via `values["consecutive_below_after_this_tick"]` along with
    whether the threshold was reached.
    """
    is_below = forecast.surge_cm < config.standdown_surge_cm
    new_count = consecutive_below_before + 1 if is_below else 0
    fired = new_count >= config.standdown_consecutive_ticks

    if fired:
        rule = "standdown_persistent_below"
        summary = (
            f"Forecast surge {forecast.surge_cm:.0f} cm below standdown threshold "
            f"{config.standdown_surge_cm:.0f} cm for {new_count} consecutive ticks "
            f"(>= required {config.standdown_consecutive_ticks})"
        )
    elif is_below:
        rule = "standdown_pending"
        summary = (
            f"Forecast surge {forecast.surge_cm:.0f} cm below standdown threshold "
            f"{config.standdown_surge_cm:.0f} cm; counter at {new_count}, "
            f"needs {config.standdown_consecutive_ticks - new_count} more"
        )
    else:
        rule = "standdown_above_threshold"
        summary = (
            f"Forecast surge {forecast.surge_cm:.0f} cm above standdown threshold; "
            f"counter reset"
        )

    return TriggerEvaluation(
        fired=fired,
        rule=rule,
        summary=summary,
        values={
            "surge_cm": forecast.surge_cm,
            "standdown_threshold_cm": config.standdown_surge_cm,
            "consecutive_below_after_this_tick": new_count,
            "required_consecutive": config.standdown_consecutive_ticks,
        },
    )
