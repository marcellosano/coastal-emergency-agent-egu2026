"""Live sea-level observation for Lido di Venezia.

ISPRA's Rete Mareografica Nazionale serves Lido Diga Sud but the public
JSON endpoint isn't openly documented and varies. We support two paths:

  1. **ISPRA endpoint (if configured)** — `ISPRA_GAUGE_URL` env var set
     to a JSON URL whose response contains a `value_cm` and `observed_at`.
     The exact JSON shape is operator-supplied via env (`ISPRA_VALUE_FIELD`,
     `ISPRA_TIME_FIELD`); leaves room to plug in any specific endpoint
     without code changes.
  2. **Open-Meteo nowcast (fallback)** — same Marine API used for forecast,
     pick the *current* hour's `sea_level_height_msl`. Not strictly a gauge
     reading but the best instrumentally-grounded value we can fetch
     without auth, and it tracks the same physical signal.

If both fail, callers receive an `Observation` with `source='stub'` and
the supplied default — so the brief composer always has something to
cite. The `source` field flows into the LLM brief's citation_ref so
audiences see exactly where the number came from.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone

import httpx

from coastal_agent.weather import LIDO_LAT, LIDO_LON, MARINE_BASE


DEFAULT_TIMEOUT_S = 8.0


@dataclass(frozen=True)
class Observation:
    gauge_id: str
    value_cm: float
    observed_at: datetime
    source: str               # 'ispra' | 'open-meteo' | 'stub'
    url: str | None = None
    raw: dict | None = None


def fetch_lido_sea_level(
    *,
    client: httpx.Client | None = None,
    fallback_value_cm: float | None = None,
    gauge_id: str = "lido_diga_sud",
) -> Observation:
    own_client = client is None
    if client is None:
        client = httpx.Client(timeout=DEFAULT_TIMEOUT_S)
    try:
        ispra = _try_ispra(client, gauge_id)
        if ispra is not None:
            return ispra
        meteo = _try_open_meteo_nowcast(client, gauge_id)
        if meteo is not None:
            return meteo
    finally:
        if own_client:
            client.close()
    return _stub_observation(gauge_id, fallback_value_cm)


# ---------------------------------------------------------------------
# ISPRA path (env-configurable; off by default)
# ---------------------------------------------------------------------


def _try_ispra(client: httpx.Client, gauge_id: str) -> Observation | None:
    url = os.environ.get("ISPRA_GAUGE_URL", "").strip()
    if not url:
        return None
    value_field = os.environ.get("ISPRA_VALUE_FIELD", "value_cm")
    time_field = os.environ.get("ISPRA_TIME_FIELD", "observed_at")
    try:
        r = client.get(url)
        r.raise_for_status()
        data = r.json()
    except (httpx.HTTPError, ValueError, json.JSONDecodeError):
        return None

    payload = _extract_field_path(data, "ISPRA_PAYLOAD_PATH")
    if not isinstance(payload, dict):
        return None
    try:
        value_cm = float(payload[value_field])
        observed_at = _parse_iso(str(payload[time_field]))
    except (KeyError, ValueError, TypeError):
        return None

    return Observation(
        gauge_id=gauge_id,
        value_cm=value_cm,
        observed_at=observed_at,
        source="ispra",
        url=url,
        raw=payload if isinstance(payload, dict) else None,
    )


def _extract_field_path(data: object, env_var: str) -> object:
    """Optional dotted path through the JSON to reach the observation
    object (e.g. ISPRA_PAYLOAD_PATH=data.observations.0). Empty/unset → return data."""
    path = os.environ.get(env_var, "").strip()
    if not path:
        return data
    cur: object = data
    for token in path.split("."):
        if token.isdigit() and isinstance(cur, list):
            idx = int(token)
            if 0 <= idx < len(cur):
                cur = cur[idx]
            else:
                return None
        elif isinstance(cur, dict):
            cur = cur.get(token)
        else:
            return None
    return cur


# ---------------------------------------------------------------------
# Open-Meteo nowcast fallback
# ---------------------------------------------------------------------


def _try_open_meteo_nowcast(client: httpx.Client, gauge_id: str) -> Observation | None:
    params = {
        "latitude": LIDO_LAT,
        "longitude": LIDO_LON,
        "hourly": "sea_level_height_msl",
        "past_hours": 1,
        "forecast_hours": 1,
        "timezone": "UTC",
    }
    try:
        r = client.get(MARINE_BASE, params=params)
        r.raise_for_status()
        data = r.json()
    except (httpx.HTTPError, ValueError):
        return None

    times = data.get("hourly", {}).get("time", []) or []
    levels = data.get("hourly", {}).get("sea_level_height_msl", []) or []
    if not times or not levels:
        return None

    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    best_idx = 0
    for i, raw in enumerate(times):
        try:
            t = _parse_iso(raw)
        except ValueError:
            continue
        if t <= now:
            best_idx = i

    try:
        value_m = float(levels[best_idx])
    except (TypeError, ValueError, IndexError):
        return None
    try:
        observed_at = _parse_iso(times[best_idx])
    except ValueError:
        observed_at = now

    return Observation(
        gauge_id=gauge_id,
        value_cm=value_m * 100.0,
        observed_at=observed_at,
        source="open-meteo",
        url=f"{MARINE_BASE}?latitude={LIDO_LAT}&longitude={LIDO_LON}",
        raw=None,
    )


# ---------------------------------------------------------------------
# Last-resort stub
# ---------------------------------------------------------------------


def _stub_observation(gauge_id: str, fallback_value_cm: float | None) -> Observation:
    return Observation(
        gauge_id=gauge_id,
        value_cm=fallback_value_cm if fallback_value_cm is not None else 0.0,
        observed_at=datetime.now(timezone.utc),
        source="stub",
        url=None,
        raw=None,
    )


def _parse_iso(raw: str) -> datetime:
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    if len(raw) == 16:
        raw = raw + ":00+00:00"
    return datetime.fromisoformat(raw)
