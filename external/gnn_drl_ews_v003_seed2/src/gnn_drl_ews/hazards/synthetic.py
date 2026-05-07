"""Synthetic hazard trajectories.

Tide: sinusoid + surge bump + noise; units = cm.
Wind/wave: similar shape, offset phase; units = m/s.
"""

from __future__ import annotations

import numpy as np


def generate_synthetic_hazard(config: dict, T: int) -> dict[str, np.ndarray]:
    seed = int(config.get("seed", 0))
    rng = np.random.default_rng(seed + 7919)  # decoupled from graph seed

    tide_cfg = config["hazards"]["tide"]
    tide = _sinusoid_with_surge(
        rng,
        T,
        base=float(tide_cfg["base_cm"]),
        amplitude=float(tide_cfg["surge_amplitude_cm"]),
        period=float(tide_cfg["period_steps"]),
        noise=float(tide_cfg["noise_cm"]),
    )

    out: dict[str, np.ndarray] = {"tide": tide.astype(np.float32)}

    if "wind_wave" in config["hazards"]:
        ww_cfg = config["hazards"]["wind_wave"]
        ww = _sinusoid_with_surge(
            rng,
            T,
            base=float(ww_cfg["base_ms"]),
            amplitude=float(ww_cfg["surge_amplitude_ms"]),
            period=float(ww_cfg["period_steps"]),
            noise=float(ww_cfg["noise_ms"]),
            phase_offset=np.pi / 4,
        )
        out["wind_wave"] = ww.astype(np.float32)

    return out


def _sinusoid_with_surge(
    rng: np.random.Generator,
    T: int,
    base: float,
    amplitude: float,
    period: float,
    noise: float,
    phase_offset: float = 0.0,
) -> np.ndarray:
    """Half-wave sinusoid centred near the middle of the episode + noise.
    Produces a single rise-and-fall hazard event suitable for an EW scenario."""
    t = np.arange(T, dtype=np.float64)
    centre = T / 2.0
    # Gaussian-bump surge to make 'storm phase' obvious to the agent.
    bump = np.exp(-((t - centre) ** 2) / (2.0 * (period / 2.0) ** 2))
    cycle = np.sin(2.0 * np.pi * t / max(period, 1.0) + phase_offset)
    series = base + amplitude * (0.6 * bump + 0.4 * (0.5 + 0.5 * cycle))
    series += rng.normal(0.0, noise, size=T)
    return series
