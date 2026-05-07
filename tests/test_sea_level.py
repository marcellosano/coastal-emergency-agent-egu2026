"""Offline tests for the Lido sea-level fetcher (ISPRA + Open-Meteo + stub)."""

from __future__ import annotations

import os
from datetime import datetime, timezone

import httpx
import pytest

from coastal_agent.sea_level import fetch_lido_sea_level


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), timeout=5.0)


# ----------------------------------------------------------------
# ISPRA path (env-configured)
# ----------------------------------------------------------------


def test_ispra_path_used_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ISPRA_GAUGE_URL", "https://fake.ispra.example/gauge")
    monkeypatch.setenv("ISPRA_VALUE_FIELD", "value_cm")
    monkeypatch.setenv("ISPRA_TIME_FIELD", "observed_at")
    monkeypatch.delenv("ISPRA_PAYLOAD_PATH", raising=False)

    def handler(request):
        if "fake.ispra.example" in str(request.url):
            return httpx.Response(200, json={
                "value_cm": 117.5,
                "observed_at": "2026-11-12T15:00:00Z",
            })
        return httpx.Response(404)

    obs = fetch_lido_sea_level(client=_client(handler))
    assert obs.source == "ispra"
    assert obs.value_cm == pytest.approx(117.5)
    assert obs.gauge_id == "lido_diga_sud"


def test_ispra_payload_path_resolves_nested_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ISPRA_GAUGE_URL", "https://fake.ispra.example/data")
    monkeypatch.setenv("ISPRA_PAYLOAD_PATH", "data.observations.0")

    def handler(request):
        return httpx.Response(200, json={
            "data": {
                "observations": [
                    {"value_cm": 105.0, "observed_at": "2026-11-12T15:00:00Z"}
                ]
            }
        })

    obs = fetch_lido_sea_level(client=_client(handler))
    assert obs.source == "ispra"
    assert obs.value_cm == 105.0


def test_ispra_falls_through_on_http_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ISPRA_GAUGE_URL", "https://fake.ispra.example/data")
    monkeypatch.delenv("ISPRA_PAYLOAD_PATH", raising=False)

    def handler(request):
        url = str(request.url)
        if "fake.ispra.example" in url:
            return httpx.Response(500)
        if "marine-api.open-meteo.com" in url:
            return httpx.Response(200, json={
                "hourly": {
                    "time": ["2026-11-12T15:00"],
                    "sea_level_height_msl": [1.10],
                }
            })
        return httpx.Response(404)

    obs = fetch_lido_sea_level(client=_client(handler))
    # ISPRA failed → fallback to Open-Meteo.
    assert obs.source == "open-meteo"
    assert obs.value_cm == pytest.approx(110.0)


# ----------------------------------------------------------------
# Open-Meteo nowcast path (default when no ISPRA env)
# ----------------------------------------------------------------


def test_open_meteo_path_used_when_no_ispra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ISPRA_GAUGE_URL", raising=False)

    def handler(request):
        if "marine-api.open-meteo.com" in str(request.url):
            return httpx.Response(200, json={
                "hourly": {
                    "time": ["2026-11-12T14:00", "2026-11-12T15:00"],
                    "sea_level_height_msl": [1.05, 1.13],
                }
            })
        return httpx.Response(404)

    obs = fetch_lido_sea_level(client=_client(handler))
    assert obs.source == "open-meteo"
    # Either picks 14:00 or 15:00 depending on now; both are valid nowcasts.
    assert 100.0 <= obs.value_cm <= 115.0


# ----------------------------------------------------------------
# Stub fallback path
# ----------------------------------------------------------------


def test_stub_used_when_all_endpoints_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ISPRA_GAUGE_URL", raising=False)

    def handler(request):
        return httpx.Response(503)

    obs = fetch_lido_sea_level(
        client=_client(handler),
        fallback_value_cm=87.5,
    )
    assert obs.source == "stub"
    assert obs.value_cm == 87.5


def test_stub_default_zero_when_no_fallback() -> None:
    monkeypatch_env = os.environ.copy()
    try:
        os.environ.pop("ISPRA_GAUGE_URL", None)

        def handler(request):
            return httpx.Response(503)

        obs = fetch_lido_sea_level(client=_client(handler))
        assert obs.source == "stub"
        assert obs.value_cm == 0.0
    finally:
        os.environ.clear()
        os.environ.update(monkeypatch_env)
