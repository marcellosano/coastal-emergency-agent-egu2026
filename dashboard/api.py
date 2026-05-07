"""FastAPI dashboard — HTML for humans, JSON API for tools.

Reads the same SQLite file the agent writes to. Never imports the
agent's model or scheduler.

Routes (HTML, mobile-first):
  GET /                                       — landing: standby/active banner,
                                                  latest forecast, incident list
  GET /incidents/{incident_id}                — incident timeline (briefs + emails)
  GET /incidents/{incident_id}/briefs/{brief_id}
                                              — single brief with citations,
                                                  preconditions, concerns,
                                                  policy distribution

Routes (JSON):
  GET /api/health                             — heartbeat + recent errors
  GET /api/live                               — latest forecast + active incident
  GET /api/incidents                          — list, newest first
  GET /api/incidents/{incident_id}            — incident + briefs + emails
  GET /api/incidents/{incident_id}/briefs/{brief_id}
                                              — single brief

The HTML routes deliberately accept the same `(incident_id, brief_id)`
shape that the orchestrator writes into `sent_emails.deeplink`, so
email click-throughs land directly on the brief page.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from coastal_agent.config import settings
from coastal_agent.db import connect, init_schema
from coastal_agent.scenario import LIDO_ACTIONS


app = FastAPI(title="Coastal Emergency Agent — Lido", version="0.2.0")

TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def _conn() -> sqlite3.Connection:
    conn = connect(settings.state_db_path)
    init_schema(conn)
    return conn


def _heartbeat_row(conn: sqlite3.Connection) -> dict | None:
    r = conn.execute(
        "SELECT last_tick_at, tick_count FROM heartbeat WHERE id=1"
    ).fetchone()
    return dict(r) if r else None


def _common_ctx(conn: sqlite3.Connection) -> dict:
    hb = _heartbeat_row(conn)
    return {
        "heartbeat": hb,
        "heartbeat_at": (hb or {}).get("last_tick_at"),
        "git_sha": "",
    }


def _latest_forecast_row(conn: sqlite3.Connection) -> dict | None:
    r = conn.execute(
        "SELECT raw_json, fetched_at FROM forecasts ORDER BY id DESC LIMIT 1"
    ).fetchone()
    if r is None:
        return None
    try:
        f = json.loads(r["raw_json"])
    except (TypeError, json.JSONDecodeError):
        return None
    return {
        "surge_cm": float(f.get("surge_cm", 0.0)),
        "wind_ms": float(f.get("wind_ms", 0.0)),
        "wave_m": float(f.get("wave_m", 0.0)),
        "rainfall_mm": float(f.get("rainfall_mm", 0.0)),
        "fetched_at": r["fetched_at"],
    }


def _incident_summary(conn: sqlite3.Connection, inc_id: int) -> dict:
    """Add brief_count + email_count to an incident row."""
    nb = conn.execute(
        "SELECT COUNT(*) AS c FROM briefs WHERE incident_id=?", (inc_id,)
    ).fetchone()
    ne = conn.execute(
        "SELECT COUNT(*) AS c FROM sent_emails WHERE incident_id=?", (inc_id,)
    ).fetchone()
    return {
        "brief_count": int(nb["c"]) if nb else 0,
        "email_count": int(ne["c"]) if ne else 0,
    }


def _brief_summary_row(row: sqlite3.Row) -> dict:
    """Extract recommendation fields from a brief's JSON for list views."""
    try:
        b = json.loads(row["brief_json"])
    except (TypeError, json.JSONDecodeError):
        b = {}
    rec = b.get("recommendation") or {}
    return {
        "id": row["id"],
        "tick": row["tick"],
        "composed_at": row["composed_at"],
        "action_id": rec.get("action_id", "?"),
        "confidence_signal": rec.get("confidence_signal"),
    }


# ---------------------------------------------------------------------
# HTML routes (the demo-facing surface)
# ---------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    conn = _conn()
    inc_rows = conn.execute(
        "SELECT id, case_study, opened_at, closed_at, status "
        "FROM incidents ORDER BY id DESC LIMIT 20"
    ).fetchall()
    incidents = []
    active_incident = None
    for r in inc_rows:
        rec = dict(r)
        rec.update(_incident_summary(conn, int(r["id"])))
        incidents.append(rec)
        if active_incident is None and r["status"] == "active":
            active_incident = rec

    ctx = _common_ctx(conn)
    ctx.update(
        {
            "request": request,
            "active_incident": active_incident,
            "incidents": incidents,
            "latest_forecast": _latest_forecast_row(conn),
        }
    )
    return templates.TemplateResponse(request, "index.html", ctx)


@app.get("/incidents/{incident_id}", response_class=HTMLResponse)
def incident_detail(request: Request, incident_id: int) -> HTMLResponse:
    conn = _conn()
    inc = conn.execute(
        "SELECT id, case_study, opened_at, closed_at, status, "
        "trigger_condition FROM incidents WHERE id=?",
        (incident_id,),
    ).fetchone()
    if inc is None:
        raise HTTPException(status_code=404, detail="incident not found")

    trigger: dict | None = None
    if inc["trigger_condition"]:
        try:
            trigger = json.loads(inc["trigger_condition"])
        except json.JSONDecodeError:
            trigger = None

    brief_rows = conn.execute(
        "SELECT id, tick, brief_json, composed_at FROM briefs "
        "WHERE incident_id=? ORDER BY tick ASC",
        (incident_id,),
    ).fetchall()
    briefs = [_brief_summary_row(r) for r in brief_rows]

    email_rows = conn.execute(
        "SELECT moment, mode, subject, sent_at, delivery_status "
        "FROM sent_emails WHERE incident_id=? ORDER BY id ASC",
        (incident_id,),
    ).fetchall()
    emails = [dict(r) for r in email_rows]

    ctx = _common_ctx(conn)
    ctx.update(
        {
            "request": request,
            "incident": dict(inc),
            "trigger": trigger,
            "briefs": briefs,
            "emails": emails,
        }
    )
    return templates.TemplateResponse(request, "incident.html", ctx)


@app.get(
    "/incidents/{incident_id}/briefs/{brief_id}",
    response_class=HTMLResponse,
)
def brief_detail(
    request: Request, incident_id: int, brief_id: int
) -> HTMLResponse:
    conn = _conn()
    inc = conn.execute(
        "SELECT id, case_study FROM incidents WHERE id=?", (incident_id,)
    ).fetchone()
    if inc is None:
        raise HTTPException(status_code=404, detail="incident not found")
    row = conn.execute(
        "SELECT id, tick, brief_json, composed_at, evaluation_id "
        "FROM briefs WHERE id=? AND incident_id=?",
        (brief_id, incident_id),
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="brief not found")
    try:
        brief = json.loads(row["brief_json"])
    except (TypeError, json.JSONDecodeError) as e:
        raise HTTPException(status_code=500, detail=f"corrupt brief: {e}")

    # Lift forecast + action_probs from the linked evaluation/forecast rows.
    forecast: dict | None = None
    action_probs: list[tuple[str, float]] = []
    if row["evaluation_id"] is not None:
        ev = conn.execute(
            "SELECT forecast_id, action_probs_json FROM evaluations WHERE id=?",
            (row["evaluation_id"],),
        ).fetchone()
        if ev is not None:
            try:
                probs = json.loads(ev["action_probs_json"])
                action_probs = list(zip(LIDO_ACTIONS, probs))
            except (TypeError, json.JSONDecodeError):
                pass
            f_row = conn.execute(
                "SELECT raw_json FROM forecasts WHERE id=?",
                (ev["forecast_id"],),
            ).fetchone()
            if f_row is not None:
                try:
                    forecast = json.loads(f_row["raw_json"])
                except (TypeError, json.JSONDecodeError):
                    pass

    ctx = _common_ctx(conn)
    ctx.update(
        {
            "request": request,
            "incident": {"id": int(inc["id"]), "case_study": inc["case_study"]},
            "brief": {
                "id": row["id"],
                "tick": row["tick"],
                "composed_at": row["composed_at"],
            },
            "rec": brief.get("recommendation") or {},
            "citations": brief.get("citations") or [],
            "precondition_check": brief.get("precondition_check") or [],
            "concerns": brief.get("concerns") or [],
            "open_questions": brief.get("open_questions") or [],
            "tool_calls": brief.get("tool_calls") or [],
            "forecast": forecast,
            "action_probs": action_probs,
        }
    )
    return templates.TemplateResponse(request, "brief.html", ctx)


# ---------------------------------------------------------------------
# JSON API
# ---------------------------------------------------------------------


@app.get("/api/health")
def api_health() -> dict:
    conn = _conn()
    hb = _heartbeat_row(conn)
    errors = conn.execute(
        "SELECT component, operation, error_message, occurred_at "
        "FROM audit_log WHERE status='error' "
        "ORDER BY occurred_at DESC LIMIT 20"
    ).fetchall()
    return {
        "heartbeat": hb,
        "recent_errors": [dict(r) for r in errors],
    }


@app.get("/api/live")
def api_live() -> dict:
    conn = _conn()
    inc = conn.execute(
        "SELECT id, case_study, opened_at, closed_at, status "
        "FROM incidents WHERE status='active' ORDER BY id DESC LIMIT 1"
    ).fetchone()
    return {
        "heartbeat": _heartbeat_row(conn),
        "latest_forecast": _latest_forecast_row(conn),
        "active_incident": dict(inc) if inc else None,
    }


@app.get("/api/incidents")
def api_incidents() -> dict:
    conn = _conn()
    rows = conn.execute(
        "SELECT id, case_study, opened_at, closed_at, status "
        "FROM incidents ORDER BY id DESC LIMIT 50"
    ).fetchall()
    return {"incidents": [dict(r) for r in rows]}


@app.get("/api/incidents/{incident_id}")
def api_incident(incident_id: int) -> dict:
    conn = _conn()
    inc = conn.execute(
        "SELECT id, case_study, opened_at, closed_at, status, "
        "trigger_condition, notes FROM incidents WHERE id=?",
        (incident_id,),
    ).fetchone()
    if inc is None:
        raise HTTPException(status_code=404, detail="incident not found")
    briefs = conn.execute(
        "SELECT id, tick, brief_json, composed_at FROM briefs "
        "WHERE incident_id=? ORDER BY tick ASC",
        (incident_id,),
    ).fetchall()
    emails = conn.execute(
        "SELECT id, moment, mode, subject, sent_at, delivery_status "
        "FROM sent_emails WHERE incident_id=? ORDER BY id ASC",
        (incident_id,),
    ).fetchall()
    return {
        "incident": dict(inc),
        "briefs": [
            {
                "id": r["id"],
                "tick": r["tick"],
                "composed_at": r["composed_at"],
                "brief": json.loads(r["brief_json"]),
            }
            for r in briefs
        ],
        "emails": [dict(r) for r in emails],
    }


@app.get("/api/incidents/{incident_id}/briefs/{brief_id}")
def api_brief(incident_id: int, brief_id: int) -> dict:
    conn = _conn()
    row = conn.execute(
        "SELECT id, tick, brief_json, composed_at FROM briefs "
        "WHERE id=? AND incident_id=?",
        (brief_id, incident_id),
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="brief not found")
    return {
        "id": row["id"],
        "tick": row["tick"],
        "composed_at": row["composed_at"],
        "brief": json.loads(row["brief_json"]),
    }
