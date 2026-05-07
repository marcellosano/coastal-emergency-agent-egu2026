"""Real hazard sampling from the ISPRA Diga Sud Lido cache (open-sea tide
gauge on the south breakwater of Bocca di Porto del Lido, Adriatic side;
distinct from the Faro Rocchetta lagoon-side station — see
scripts/ingest_lido_hazards.py for provenance, BLOCKED.md [B3] for the
sensor-vs-inlet rationale).

Reads the unified hourly series produced by `scripts/ingest_lido_hazards.py`
and returns T-hour windows in the same shape as `hazards.synthetic`:

    {"tide": np.ndarray[T] in cm above ZMPS,
     "wind_wave": np.ndarray[T] in m/s}     (optional)

Sampling modes (config['hazards']['mode']):
    uniform        — pick a random T-hour window with no gaps in water level.
    named_event    — start window at config['hazards']['named_event'] (ISO date).
    severity       — pick uniformly from windows whose peak level >= min_peak_cm.

Wind: if a chosen window predates 2012 (no wind data), wind is filled with
the archive-mean wind speed and a one-shot warning is issued.

The cache is loaded lazily and memoised at module level — repeated env
resets only pay disk I/O once per process.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Module-level cache (loaded on first call).
# ---------------------------------------------------------------------------

_LOADED: dict[str, pd.DataFrame] = {}
_WIND_FALLBACK: float | None = None
_VALID_STARTS: dict[str, np.ndarray] = {}   # cache_dir -> array of valid window-start positions
_EXCLUDED_POSITIONS: dict[str, set[int]] = {}  # exclusion-config hash -> set of excluded hour positions


def generate_real_hazard(config: dict, T: int) -> dict[str, np.ndarray]:
    cache_dir = Path(config["hazards"].get("cache_dir") or
                     "data/cache/lido/hazards")
    mode = config["hazards"].get("mode", "uniform")
    seed = int(config.get("seed", 0))
    include_wind = bool(config["hazards"].get("include_wind", True))

    wl, wd = _load(cache_dir)
    valid_starts = _valid_window_starts(wl, T, str(cache_dir))

    # Holdout exclusion: training samplers (uniform, severity) skip windows
    # near any timestamp listed in config.hazards.holdout_exclusions, so the
    # autoresearch loop's eval set + the §6 named-event are never seen during
    # training. `named_event` mode bypasses (it is the explicit way to ask
    # for a specific window — e.g. the 22 Nov 2022 surge for the paper figure).
    if mode in ("uniform", "severity"):
        valid_starts = _filter_excluded_windows(valid_starts, wl, config["hazards"])

    if mode == "named_event":
        start_iso = config["hazards"].get("named_event")
        if not start_iso:
            raise ValueError("hazards.mode='named_event' requires hazards.named_event (ISO date)")
        start_idx = _locate_named(wl, start_iso)
    elif mode == "severity":
        min_peak = float(config["hazards"].get("min_peak_cm", 100.0))
        start_idx = _pick_severity(wl, valid_starts, T, min_peak, seed)
    else:  # uniform
        if len(valid_starts) == 0:
            raise RuntimeError(f"No gap-free {T}h windows in cache {cache_dir}")
        rng = np.random.default_rng(seed)
        start_idx = int(rng.choice(valid_starts))

    return _build_window(wl, wd, start_idx, T, include_wind)


# ---------------------------------------------------------------------------
# Cache loading
# ---------------------------------------------------------------------------

def _load(cache_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    key = str(cache_dir.resolve())
    if key in _LOADED:
        return _LOADED[key], _LOADED[key + ":wind"]
    wl_path = cache_dir / "water_level_hourly.csv"
    wd_path = cache_dir / "wind_hourly.csv"
    if not wl_path.exists():
        raise FileNotFoundError(
            f"Real-hazard cache not found at {wl_path}. "
            f"Run `python scripts/ingest_lido_hazards.py` first."
        )
    wl = pd.read_csv(wl_path, parse_dates=["timestamp_solar"])
    wl = wl.set_index("timestamp_solar").sort_index()
    # Re-index to a dense hourly grid; gaps become NaN so the window picker
    # can avoid them.
    full = pd.date_range(wl.index.min().floor("h"), wl.index.max().ceil("h"), freq="h")
    wl = wl.reindex(full)

    wd = pd.DataFrame(columns=["wind_speed_ms"])
    if wd_path.exists():
        wd = pd.read_csv(wd_path, parse_dates=["timestamp_solar"])
        wd = wd.set_index("timestamp_solar").sort_index()
        wd = wd.reindex(full)

    global _WIND_FALLBACK
    if _WIND_FALLBACK is None and "wind_speed_ms" in wd.columns:
        _WIND_FALLBACK = float(np.nanmean(wd["wind_speed_ms"].values))

    _LOADED[key] = wl
    _LOADED[key + ":wind"] = wd
    return wl, wd


def _valid_window_starts(wl: pd.DataFrame, T: int, cache_key: str) -> np.ndarray:
    """Hour positions where wl[start:start+T] has no NaN."""
    key = f"{cache_key}::T={T}"
    if key in _VALID_STARTS:
        return _VALID_STARTS[key]
    levels = wl["level_cm"].values
    # Rolling NaN check — `pd.Series.rolling(T).count()` counts non-NaN.
    valid = pd.Series(levels).rolling(T).count().values >= T
    starts = np.where(valid)[0] - (T - 1)
    starts = starts[starts >= 0]
    _VALID_STARTS[key] = starts
    return starts


def _filter_excluded_windows(
    valid_starts: np.ndarray,
    wl: pd.DataFrame,
    hazards_cfg: dict,
) -> np.ndarray:
    """Drop window starts within +/- buffer hours of any excluded timestamp.

    Reads `hazards.holdout_exclusions` from config:
        csv:           path (cwd-relative) to a CSV with a `start_iso` column
        named_events:  list of ISO timestamps
        buffer_hours:  +/- buffer applied to each excluded timestamp (default 48)

    Returns a (possibly shorter) array. If no exclusions configured, returns
    the input unchanged.
    """
    exc_cfg = hazards_cfg.get("holdout_exclusions")
    if not exc_cfg:
        return valid_starts

    # Cache excluded-position set per (config-hash + cache-shape) so repeated
    # env resets don't re-read the CSV or recompute the buffer math.
    cache_key = json.dumps(exc_cfg, sort_keys=True, default=str) + f"::n={len(wl)}"
    if cache_key not in _EXCLUDED_POSITIONS:
        _EXCLUDED_POSITIONS[cache_key] = _build_excluded_positions(exc_cfg, wl)
    excluded = _EXCLUDED_POSITIONS[cache_key]
    if not excluded:
        return valid_starts

    keep = np.array([int(s) not in excluded for s in valid_starts], dtype=bool)
    return valid_starts[keep]


def _build_excluded_positions(exc_cfg: dict, wl: pd.DataFrame) -> set[int]:
    """Convert the exclusion config to a set of hour-grid positions to exclude."""
    buffer_h = int(exc_cfg.get("buffer_hours", 48))

    excluded_iso: list[str] = []

    csv_ref = exc_cfg.get("csv")
    if csv_ref:
        csv_path = Path(csv_ref)
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                if "start_iso" in df.columns:
                    excluded_iso.extend(df["start_iso"].astype(str).tolist())
                else:
                    warnings.warn(
                        f"holdout_exclusions.csv {csv_path} has no 'start_iso' "
                        "column; ignoring CSV-side exclusions."
                    )
            except Exception as e:
                warnings.warn(f"failed to read holdout_exclusions.csv {csv_path}: {e}")
        else:
            warnings.warn(
                f"holdout_exclusions.csv not found: {csv_path}. "
                "Training will NOT exclude these windows. "
                "Run `python scripts/build_holdout_set.py` to populate."
            )

    for iso in (exc_cfg.get("named_events") or []):
        excluded_iso.append(str(iso))

    if not excluded_iso:
        return set()

    positions: set[int] = set()
    n = len(wl)
    for iso in excluded_iso:
        try:
            ts = pd.Timestamp(iso)
        except Exception:
            warnings.warn(f"holdout_exclusions: cannot parse timestamp {iso!r}; skipped.")
            continue
        # Find the nearest hour index for this timestamp; expand by +/- buffer_h.
        idx = int(wl.index.get_indexer([ts], method="nearest")[0])
        lo = max(0, idx - buffer_h)
        hi = min(n - 1, idx + buffer_h)
        positions.update(range(lo, hi + 1))
    return positions


# ---------------------------------------------------------------------------
# Mode-specific selection
# ---------------------------------------------------------------------------

def _locate_named(wl: pd.DataFrame, start_iso: str) -> int:
    ts = pd.Timestamp(start_iso)
    if ts not in wl.index:
        # Snap to nearest available index.
        i = wl.index.get_indexer([ts], method="nearest")[0]
        warnings.warn(f"named_event {start_iso} not exact; snapped to {wl.index[i]}")
        return int(i)
    return int(wl.index.get_indexer([ts])[0])


def _pick_severity(wl: pd.DataFrame, valid_starts: np.ndarray, T: int,
                   min_peak: float, seed: int) -> int:
    levels = wl["level_cm"].values
    # Compute window-max for every valid start.
    rolling_max = pd.Series(levels).rolling(T).max().values
    candidate_max = rolling_max[valid_starts + (T - 1)]
    keep = valid_starts[candidate_max >= min_peak]
    if len(keep) == 0:
        warnings.warn(
            f"No windows with peak >= {min_peak} cm; falling back to uniform."
        )
        keep = valid_starts
    rng = np.random.default_rng(seed)
    return int(rng.choice(keep))


# ---------------------------------------------------------------------------
# Window construction
# ---------------------------------------------------------------------------

def _build_window(wl: pd.DataFrame, wd: pd.DataFrame, start: int, T: int,
                  include_wind: bool) -> dict[str, np.ndarray]:
    end = start + T
    tide = wl["level_cm"].values[start:end].astype(np.float32)
    # Forward-fill any residual NaNs (shouldn't happen if start came from
    # _valid_window_starts, but be safe for named_event).
    if np.isnan(tide).any():
        tide = pd.Series(tide).ffill().bfill().fillna(0.0).values.astype(np.float32)

    out: dict[str, np.ndarray] = {"tide": tide}

    if include_wind and "wind_speed_ms" in wd.columns:
        wind = wd["wind_speed_ms"].values[start:end].astype(np.float32)
        if np.isnan(wind).all():
            wind = np.full(T, _WIND_FALLBACK or 4.0, dtype=np.float32)
        else:
            wind = pd.Series(wind).ffill().bfill().fillna(_WIND_FALLBACK or 4.0).values.astype(np.float32)
        out["wind_wave"] = wind

    # Stash the window timestamp on the array as an attribute for
    # downstream HITL traces. NumPy doesn't allow arbitrary attrs on
    # ndarrays so we attach via a sidecar dict in the returned dict.
    out["_meta"] = {
        "window_start": str(wl.index[start]),
        "window_end":   str(wl.index[end - 1]),
        "peak_cm":      float(np.max(tide)),
        "datum":        "ZMPS",
        "time":         "ora_solare (UTC+1, no DST)",
    }
    return out
