"""Offline tests for the Open-Meteo weather provider."""

from __future__ import annotations

from datetime import datetime, timezone

import httpx
import pytest

from coastal_agent.weather import (
    FORECAST_BASE,
    MARINE_BASE,
    WeatherFetchError,
    fetch_lido_forecast,
)


def _marine_payload(times: list[str], surge_m: list[float]) -> dict:
    return {
        "hourly": {
            "time": times,
            "sea_level_height_msl": surge_m,
            "wave_height": [1.2] * len(times),
            "wind_wave_height": [0.4] * len(times),
        }
    }


def _surface_payload(times: list[str], wind_ms: list[float]) -> dict:
    return {
        "hourly": {
            "time": times,
            "wind_speed_10m": wind_ms,
            "precipitation": [0.0] * len(times),
        }
    }


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), timeout=5.0)


def test_fetch_lido_forecast_picks_current_hour() -> None:
    times = [
        "2026-11-12T13:00",
        "2026-11-12T14:00",
        "2026-11-12T15:00",
        "2026-11-12T16:00",
    ]
    surge_m = [0.95, 1.05, 1.12, 1.20]   # cm: 95, 105, 112, 120
    wind_ms = [10.0, 11.0, 12.5, 13.0]

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url).startswith(MARINE_BASE):
            return httpx.Response(200, json=_marine_payload(times, surge_m))
        if str(request.url).startswith(FORECAST_BASE):
            return httpx.Response(200, json=_surface_payload(times, wind_ms))
        return httpx.Response(404)

    fetch = fetch_lido_forecast(
        client=_client(handler),
        now=datetime(2026, 11, 12, 15, 17, tzinfo=timezone.utc),
    )
    snap = fetch.snapshot
    assert snap.surge_cm == pytest.approx(112.0)
    assert snap.wind_ms == pytest.approx(12.5)
    assert snap.wave_m == pytest.approx(1.2)
    assert fetch.source == "open-meteo"


def test_fetch_lido_forecast_handles_naive_iso_strings() -> None:
    """Open-Meteo emits 'YYYY-MM-DDTHH:MM' (no Z, no offset). Should parse."""
    times = ["2026-11-12T15:00"]
    handler_log = {}

    def handler(request):
        handler_log["url"] = str(request.url)
        if MARINE_BASE in str(request.url):
            return httpx.Response(200, json=_marine_payload(times, [1.10]))
        if FORECAST_BASE in str(request.url):
            return httpx.Response(200, json=_surface_payload(times, [9.0]))
        return httpx.Response(404)

    fetch = fetch_lido_forecast(
        client=_client(handler),
        now=datetime(2026, 11, 12, 15, 30, tzinfo=timezone.utc),
    )
    assert fetch.snapshot.surge_cm == pytest.approx(110.0)


def test_fetch_lido_forecast_raises_on_http_error() -> None:
    def handler(request):
        return httpx.Response(503)

    with pytest.raises(WeatherFetchError):
        fetch_lido_forecast(client=_client(handler))


def test_fetch_lido_forecast_raises_on_empty_hourly() -> None:
    def handler(request):
        return httpx.Response(200, json={"hourly": {"time": []}})

    with pytest.raises(WeatherFetchError):
        fetch_lido_forecast(client=_client(handler))


def test_fetch_lido_forecast_handles_null_value() -> None:
    """Open-Meteo can return null for the most recent hour if data is
    not yet available. The provider should fall through to the default."""
    times = ["2026-11-12T15:00"]

    def handler(request):
        if MARINE_BASE in str(request.url):
            return httpx.Response(
                200,
                json={
                    "hourly": {
                        "time": times,
                        "sea_level_height_msl": [None],
                        "wave_height": [None],
                        "wind_wave_height": [None],
                    }
                },
            )
        if FORECAST_BASE in str(request.url):
            return httpx.Response(200, json=_surface_payload(times, [None]))
        return httpx.Response(404)

    fetch = fetch_lido_forecast(
        client=_client(handler),
        now=datetime(2026, 11, 12, 15, 30, tzinfo=timezone.utc),
    )
    assert fetch.snapshot.surge_cm == 0.0
    assert fetch.snapshot.wind_ms == 0.0
