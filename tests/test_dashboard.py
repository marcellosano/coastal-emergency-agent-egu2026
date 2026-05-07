"""Smoke tests for the dashboard FastAPI app.

Builds a temp DB, seeds it with one closed incident plus a brief, then
hits the routes via FastAPI's TestClient. Verifies the HTML routes
return 200 and contain key content; verifies the JSON routes return
the right shape.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from coastal_agent.db import connect, init_schema


@pytest.fixture
def seeded_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> sqlite3.Connection:
    db_path = tmp_path / "dash.db"
    monkeypatch.setenv("STATE_DB_PATH", str(db_path))
    # Force coastal_agent.config.settings to re-read STATE_DB_PATH
    import importlib
    from coastal_agent import config as cfg_mod
    importlib.reload(cfg_mod)

    conn = connect(db_path)
    init_schema(conn)
    # Heartbeat (init_schema already inserts id=1; bump count + time)
    conn.execute(
        "UPDATE heartbeat SET tick_count=5, last_tick_at='2026-05-07T10:00:00Z' "
        "WHERE id=1"
    )
    # One closed incident
    conn.execute(
        "INSERT INTO incidents(id, case_study, opened_at, closed_at, "
        "trigger_condition, status) VALUES "
        "(1, 'lido', '2026-05-07T08:00:00Z', '2026-05-07T11:00:00Z', "
        "'{\"rule\":\"surge_above_activation\",\"summary\":\"surge=112cm >= 110cm\",\"tick\":7}', 'closed')"
    )
    # One forecast row tied to that incident
    fc_payload = json.dumps({
        "surge_cm": 112.0, "wind_ms": 12.0, "wave_m": 1.5,
        "rainfall_mm": 0.0, "horizon_hours": 24,
    })
    conn.execute(
        "INSERT INTO forecasts(id, incident_id, source, location, raw_json, "
        "parsed_json, fetched_at) VALUES "
        "(1, 1, 'open-meteo', 'lido', ?, ?, '2026-05-07T08:00:00Z')",
        (fc_payload, fc_payload),
    )
    # One evaluation
    conn.execute(
        "INSERT INTO evaluations(id, incident_id, forecast_id, action_probs_json, "
        "value_estimate, confidence, storm_type, mask_json, evaluated_at) VALUES "
        "(1, 1, 1, ?, -1.5, 0.7, 'tide', ?, '2026-05-07T08:00:00Z')",
        (
            json.dumps([0.2, 0.7, 0.05, 0.025, 0.025, 0.0, 0.0]),
            json.dumps([True, True, False, False, False, False, False]),
        ),
    )
    # One brief — full structured payload
    brief_payload = json.dumps({
        "incident_id": 1,
        "tick": 7,
        "recommendation": {
            "action_id": "issue_alert",
            "confidence_signal": "high",
            "rationale": "Surge crossed alert threshold; PCE §4.1.4.9 mandates Sindaco alert.",
        },
        "citations": [
            {"source_type": "plan_provision",
             "ref": "PCE_Comune_Venezia_v1_2008 §4.1.4.9 p41",
             "excerpt": "Sindaco issues public alert"}
        ],
        "precondition_check": [
            {"condition": "global.forecast_tide >= threshold.alert_tide",
             "satisfied": True, "evidence": "112cm >= 110cm"}
        ],
        "concerns": ["Wave height of 1.5m compounds inundation."],
        "open_questions": ["Has the Sindaco been formally notified?"],
        "tool_calls": [
            {"name": "get_plan_provision",
             "args": {"action_id": "issue_alert"},
             "result_summary": "PCE §4.1.4.9 p41",
             "latency_ms": 67}
        ],
    })
    conn.execute(
        "INSERT INTO briefs(id, incident_id, evaluation_id, tick, brief_json, composed_at) "
        "VALUES (1, 1, 1, 7, ?, '2026-05-07T08:00:00Z')",
        (brief_payload,),
    )
    # One sent_email
    conn.execute(
        "INSERT INTO sent_emails(id, incident_id, brief_id, moment, subject, body, "
        "deeplink, mode, sent_at) VALUES "
        "(1, 1, 1, 'activation', '[INCIDENT 1] lido — surge 112 cm', "
        "'<body>', '/incidents/1/briefs/1', 'mock', '2026-05-07T08:00:00Z')"
    )
    return conn


@pytest.fixture
def client(seeded_db) -> TestClient:
    # Late import so the monkeypatch'd STATE_DB_PATH is picked up.
    import importlib
    from dashboard import api as api_mod
    importlib.reload(api_mod)
    return TestClient(api_mod.app)


# ------------------------------------------------------------------
# HTML routes
# ------------------------------------------------------------------


def test_index_html_renders(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    body = r.text
    assert "Coastal Emergency Agent" in body
    # Standby (incident is closed)
    assert "Standby" in body
    assert "Incidents" in body
    assert "lido" in body
    # Latest forecast surfaces
    assert "112.0 cm" in body or "112 cm" in body


def test_incident_detail_html_renders(client: TestClient) -> None:
    r = client.get("/incidents/1")
    assert r.status_code == 200
    body = r.text
    assert "Incident #1" in body
    assert "issue_alert" in body
    assert "activation" in body  # email moment label


def test_brief_detail_html_renders(client: TestClient) -> None:
    r = client.get("/incidents/1/briefs/1")
    assert r.status_code == 200
    body = r.text
    assert "issue_alert" in body
    assert "PCE_Comune_Venezia_v1_2008" in body
    assert "Wave height" in body  # concern
    assert "Sindaco" in body      # open question
    # Policy distribution rendered
    assert "monitor" in body and "issue_alert" in body
    # Tool audit
    assert "get_plan_provision" in body


def test_incident_404(client: TestClient) -> None:
    r = client.get("/incidents/999")
    assert r.status_code == 404


def test_brief_404(client: TestClient) -> None:
    r = client.get("/incidents/1/briefs/999")
    assert r.status_code == 404


# ------------------------------------------------------------------
# JSON routes
# ------------------------------------------------------------------


def test_api_health(client: TestClient) -> None:
    r = client.get("/api/health")
    assert r.status_code == 200
    j = r.json()
    assert j["heartbeat"]["tick_count"] == 5
    assert j["recent_errors"] == []


def test_api_live(client: TestClient) -> None:
    r = client.get("/api/live")
    assert r.status_code == 200
    j = r.json()
    assert j["latest_forecast"]["surge_cm"] == 112.0
    assert j["active_incident"] is None  # incident is closed


def test_api_incidents(client: TestClient) -> None:
    r = client.get("/api/incidents")
    assert r.status_code == 200
    j = r.json()
    assert len(j["incidents"]) == 1
    assert j["incidents"][0]["status"] == "closed"


def test_api_brief(client: TestClient) -> None:
    r = client.get("/api/incidents/1/briefs/1")
    assert r.status_code == 200
    j = r.json()
    assert j["brief"]["recommendation"]["action_id"] == "issue_alert"
    assert len(j["brief"]["citations"]) == 1
