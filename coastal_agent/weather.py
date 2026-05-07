"""Real-data weather provider for live mode.

Open-Meteo Marine + standard forecast API (free, no auth). For Lido
(45.42°N, 12.38°E) the marine endpoint exposes `sea_level_height_msl`
(metres above MSL — i.e. storm surge) and wave height; the standard
forecast endpoint exposes wind speed and precipitation.

Picks the *current hour* from the hourly arrays and packs into the
existing `ForecastSnapshot` shape. The orchestrator is unchanged —
it consumes a `ScenarioRecord` either way.

Offline-testable: a custom `httpx.Client` can be injected by tests.
Network failure raises `WeatherFetchError`; the daemon logs and
preserves the last known forecast so a transient API blip doesn't
blow up an active incident.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import httpx

from coastal_agent.scenario import ForecastSnapshot


# Lido di Venezia — case-study coordinates.
LIDO_LAT = 45.42
LIDO_LON = 12.38

MARINE_BASE = "https://marine-api.open-meteo.com/v1/marine"
FORECAST_BASE = "https://api.open-meteo.com/v1/forecast"

DEFAULT_TIMEOUT_S = 10.0
DEFAULT_HORIZON_HOURS = 24


class WeatherFetchError(RuntimeError):
    """Raised when an upstream weather call fails or returns malformed data."""


@dataclass(frozen=True)
class ForecastFetch:
    """Wraps the snapshot with provenance for the audit trail."""

    snapshot: ForecastSnapshot
    fetched_at: datetime
    source: str
    raw: dict


def fetch_lido_forecast(
    *,
    client: httpx.Client | None = None,
    horizon_hours: int = DEFAULT_HORIZON_HOURS,
    now: datetime | None = None,
) -> ForecastFetch:
    """Fetch current hour's forecast for Lido di Venezia.

    Picks the hourly array index whose ISO timestamp is the latest
    hour at-or-before `now` (UTC). Defaults `now` to wall clock.
    """
    own_client = client is None
    if client is None:
        client = httpx.Client(timeout=DEFAULT_TIMEOUT_S)
    try:
        marine = _fetch_marine(client, horizon_hours)
        surface = _fetch_surface(client, horizon_hours)
    finally:
        if own_client:
            client.close()

    when = now or datetime.now(timezone.utc)
    idx = _pick_current_index(marine.get("hourly", {}).get("time", []), when)
    if idx is None:
        raise WeatherFetchError(
            "Open-Meteo response did not include a current-hour entry"
        )

    surge_m = _safe_float(marine["hourly"].get("sea_level_height_msl", []), idx)
    wave_m = _safe_float(marine["hourly"].get("wave_height", []), idx)
    wind_ms = _safe_float(surface["hourly"].get("wind_speed_10m", []), idx)
    rain_mm = _safe_float(surface["hourly"].get("precipitation", []), idx, default=0.0)

    snapshot = ForecastSnapshot(
        surge_cm=surge_m * 100.0,
        wind_ms=wind_ms,
        wave_m=wave_m,
        rainfall_mm=rain_mm,
        horizon_hours=horizon_hours,
    )
    return ForecastFetch(
        snapshot=snapshot,
        fetched_at=datetime.now(timezone.utc),
        source="open-meteo",
        raw={"marine": marine, "surface": surface, "picked_index": idx},
    )


# ---------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------


def _fetch_marine(client: httpx.Client, horizon_hours: int) -> dict:
    params = {
        "latitude": LIDO_LAT,
        "longitude": LIDO_LON,
        "hourly": "sea_level_height_msl,wave_height,wind_wave_height",
        "forecast_hours": horizon_hours,
        "timezone": "UTC",
    }
    return _get_json(client, MARINE_BASE, params)


def _fetch_surface(client: httpx.Client, horizon_hours: int) -> dict:
    params = {
        "latitude": LIDO_LAT,
        "longitude": LIDO_LON,
        "hourly": "wind_speed_10m,precipitation",
        "forecast_hours": horizon_hours,
        "timezone": "UTC",
        "wind_speed_unit": "ms",
    }
    return _get_json(client, FORECAST_BASE, params)


def _get_json(client: httpx.Client, url: str, params: dict) -> dict:
    try:
        r = client.get(url, params=params)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPError as e:
        raise WeatherFetchError(f"Open-Meteo {url}: {e}") from e
    except ValueError as e:
        raise WeatherFetchError(f"Open-Meteo {url}: invalid JSON: {e}") from e


def _pick_current_index(times: list[str], when: datetime) -> int | None:
    """Pick the index of the latest timestamp at-or-before `when`."""
    if not times:
        return None
    target = when.replace(minute=0, second=0, microsecond=0).astimezone(timezone.utc)
    best: int | None = None
    for i, raw in enumerate(times):
        try:
            t = _parse_iso(raw)
        except ValueError:
            continue
        if t <= target:
            best = i
        else:
            break
    return best if best is not None else 0


def _parse_iso(raw: str) -> datetime:
    """Open-Meteo emits 'YYYY-MM-DDTHH:MM' in the requested timezone."""
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    if len(raw) == 16:  # 'YYYY-MM-DDTHH:MM' — assume UTC since we requested it
        raw = raw + ":00+00:00"
    return datetime.fromisoformat(raw)


def _safe_float(arr: list, idx: int, default: float = 0.0) -> float:
    if idx < 0 or idx >= len(arr):
        return default
    v = arr[idx]
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default
