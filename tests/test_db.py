"""Smoke tests for the database layer.

Verifies schema initialises, sqlite-vec loads, FTS5 wires up, the
heartbeat singleton is enforced, and `integrity_check` reports OK.
"""

from __future__ import annotations

from pathlib import Path

from coastal_agent.db import connect, init_schema, integrity_check


def test_schema_initialises(tmp_path: Path) -> None:
    conn = connect(tmp_path / "test.db")
    init_schema(conn)

    tables = {
        row["name"]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
    }
    expected = {
        "incidents", "forecasts", "evaluations", "briefs", "alerts",
        "sent_emails", "audit_log", "heartbeat", "documents", "chunks",
    }
    assert expected.issubset(tables)


def test_incident_lifecycle_columns(tmp_path: Path) -> None:
    """Verify incidents table has the expected columns and CHECK constraints."""
    conn = connect(tmp_path / "test.db")
    init_schema(conn)

    cols = {row["name"]: row for row in conn.execute("PRAGMA table_info(incidents)")}
    assert {"id", "case_study", "opened_at", "closed_at", "trigger_condition", "status"}.issubset(cols)

    # Insert an active incident, then close it.
    conn.execute(
        "INSERT INTO incidents(case_study, opened_at, trigger_condition, status) "
        "VALUES ('lido', datetime('now'), '{\"rule\": \"forecast_surge\", \"value\": 1.4}', 'active')"
    )
    conn.execute(
        "UPDATE incidents SET status='closed', closed_at=datetime('now') WHERE id=1"
    )
    row = conn.execute("SELECT status, closed_at FROM incidents WHERE id=1").fetchone()
    assert row["status"] == "closed"
    assert row["closed_at"] is not None


def test_briefs_unique_per_tick(tmp_path: Path) -> None:
    """Two briefs for the same (incident, tick) should be rejected."""
    import sqlite3 as _sqlite3
    conn = connect(tmp_path / "test.db")
    init_schema(conn)
    conn.execute(
        "INSERT INTO incidents(case_study, opened_at, trigger_condition, status) "
        "VALUES ('lido', datetime('now'), '{}', 'active')"
    )
    conn.execute(
        "INSERT INTO briefs(incident_id, tick, brief_json, composed_at) "
        "VALUES (1, 0, '{}', datetime('now'))"
    )
    try:
        conn.execute(
            "INSERT INTO briefs(incident_id, tick, brief_json, composed_at) "
            "VALUES (1, 0, '{}', datetime('now'))"
        )
        raise AssertionError("Expected IntegrityError on duplicate (incident_id, tick)")
    except _sqlite3.IntegrityError:
        pass


def test_sent_emails_mode_constraint(tmp_path: Path) -> None:
    """sent_emails.mode CHECK should reject anything other than 'real' or 'mock'."""
    import sqlite3 as _sqlite3
    conn = connect(tmp_path / "test.db")
    init_schema(conn)
    conn.execute(
        "INSERT INTO incidents(case_study, opened_at, trigger_condition, status) "
        "VALUES ('lido', datetime('now'), '{}', 'active')"
    )
    # Valid modes accepted
    for mode in ("real", "mock"):
        conn.execute(
            "INSERT INTO sent_emails(incident_id, moment, subject, body, deeplink, mode, sent_at) "
            "VALUES (1, 'activation', 's', 'b', 'd', ?, datetime('now'))",
            (mode,),
        )
    # Invalid mode rejected
    try:
        conn.execute(
            "INSERT INTO sent_emails(incident_id, moment, subject, body, deeplink, mode, sent_at) "
            "VALUES (1, 'activation', 's', 'b', 'd', 'pretend', datetime('now'))"
        )
        raise AssertionError("Expected IntegrityError on invalid mode")
    except _sqlite3.IntegrityError:
        pass


def test_sqlite_vec_loaded(tmp_path: Path) -> None:
    conn = connect(tmp_path / "test.db")
    version = conn.execute("SELECT vec_version()").fetchone()
    assert version is not None


def test_fts5_wired(tmp_path: Path) -> None:
    conn = connect(tmp_path / "test.db")
    init_schema(conn)
    conn.execute(
        "INSERT INTO documents(case_study, source, title, path, hash, indexed_at) "
        "VALUES ('test', 'test-src', 'doc', '/x', 'h1', datetime('now'))"
    )
    doc_id = conn.execute(
        "SELECT id FROM documents WHERE hash='h1'"
    ).fetchone()["id"]
    conn.execute(
        "INSERT INTO chunks(document_id, text, span_start, span_end) "
        "VALUES (?, 'high tide warning issued', 0, 24)",
        (doc_id,),
    )
    matches = conn.execute(
        "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH 'tide'"
    ).fetchall()
    assert len(matches) == 1


def test_heartbeat_singleton(tmp_path: Path) -> None:
    conn = connect(tmp_path / "test.db")
    init_schema(conn)
    init_schema(conn)  # idempotent — INSERT OR IGNORE keeps it a singleton
    count = conn.execute("SELECT COUNT(*) AS c FROM heartbeat").fetchone()["c"]
    assert count == 1


def test_integrity_check(tmp_path: Path) -> None:
    conn = connect(tmp_path / "test.db")
    init_schema(conn)
    assert integrity_check(conn) == "ok"
