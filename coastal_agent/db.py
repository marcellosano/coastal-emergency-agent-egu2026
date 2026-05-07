"""SQLite schema and connection management.

Single writer (the daemon), multiple readers (dashboard, ad-hoc tools).
WAL mode for concurrent reads. sqlite-vec loaded for chunk embeddings;
FTS5 mirror over `chunks.text` for keyword retrieval.

Schema follows the agreed §3.5 design (incident-shaped, orchestrator
produces structured JSON briefs, mock or real SMTP both first-class).
Re-indexing the corpus soft-deletes old chunks via `superseded_at` so
prior citations remain resolvable.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import sqlite_vec

SCHEMA = """
-- Incident-shaped state. An incident wraps a single storm event from
-- activation (trigger crossed bounds) to stand-down (conditions cleared).
CREATE TABLE IF NOT EXISTS incidents (
    id INTEGER PRIMARY KEY,
    case_study TEXT NOT NULL,                -- 'lido', 'seq', etc.
    opened_at TEXT NOT NULL,
    closed_at TEXT,
    trigger_condition TEXT NOT NULL,         -- JSON: which rule fired and why
    status TEXT NOT NULL CHECK (status IN ('active', 'closed', 'standdown_pending')),
    notes TEXT
);

CREATE TABLE IF NOT EXISTS forecasts (
    id INTEGER PRIMARY KEY,
    incident_id INTEGER REFERENCES incidents(id),
    source TEXT NOT NULL,
    location TEXT NOT NULL,
    raw_json TEXT NOT NULL,
    parsed_json TEXT NOT NULL,
    fetched_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS evaluations (
    id INTEGER PRIMARY KEY,
    incident_id INTEGER REFERENCES incidents(id),
    forecast_id INTEGER NOT NULL REFERENCES forecasts(id),
    action_probs_json TEXT NOT NULL,
    value_estimate REAL,
    confidence REAL,
    storm_type TEXT,
    mask_json TEXT,                          -- per-action legal/illegal flags
    evaluated_at TEXT NOT NULL
);

-- A brief is the LLM's structured-JSON output for one tick of an incident.
-- See DESIGN §3.5.B for the field schema.
CREATE TABLE IF NOT EXISTS briefs (
    id INTEGER PRIMARY KEY,
    incident_id INTEGER NOT NULL REFERENCES incidents(id),
    evaluation_id INTEGER REFERENCES evaluations(id),
    tick INTEGER NOT NULL,                   -- ordinal tick within incident
    brief_json TEXT NOT NULL,                -- full structured BriefOutput
    composed_at TEXT NOT NULL,
    UNIQUE(incident_id, tick)
);

CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY,
    incident_id INTEGER REFERENCES incidents(id),
    evaluation_id INTEGER REFERENCES evaluations(id),
    brief_id INTEGER REFERENCES briefs(id),
    dedup_key TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL CHECK (status IN ('pending', 'sent', 'acked', 'failed')),
    brief_text TEXT,
    citations_json TEXT,
    created_at TEXT NOT NULL,
    sent_at TEXT,
    acked_at TEXT
);

-- Email layer is mode-agnostic: real SMTP and mock both write here.
-- Real-SMTP mode also performs the network send and updates delivery_status.
-- Mock mode just persists what would have been sent.
CREATE TABLE IF NOT EXISTS sent_emails (
    id INTEGER PRIMARY KEY,
    incident_id INTEGER NOT NULL REFERENCES incidents(id),
    brief_id INTEGER REFERENCES briefs(id),
    moment TEXT NOT NULL CHECK (moment IN ('activation', 'new_alert', 'standdown')),
    subject TEXT NOT NULL,
    body TEXT NOT NULL,
    deeplink TEXT NOT NULL,
    mode TEXT NOT NULL CHECK (mode IN ('real', 'mock')),
    delivery_status TEXT,                    -- 'sent', 'failed', NULL for mock
    sent_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY,
    incident_id INTEGER REFERENCES incidents(id),
    component TEXT NOT NULL,
    operation TEXT NOT NULL,
    status TEXT NOT NULL,
    latency_ms INTEGER,
    error_message TEXT,
    occurred_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS heartbeat (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    last_tick_at TEXT NOT NULL,
    tick_count INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY,
    case_study TEXT NOT NULL,
    source TEXT,
    title TEXT,
    path TEXT NOT NULL,
    hash TEXT NOT NULL UNIQUE,
    indexed_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id),
    text TEXT NOT NULL,
    span_start INTEGER NOT NULL,
    span_end INTEGER NOT NULL,
    embedding BLOB,
    superseded_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_incidents_status ON incidents(status);
CREATE INDEX IF NOT EXISTS idx_incidents_opened ON incidents(opened_at);
CREATE INDEX IF NOT EXISTS idx_forecasts_incident ON forecasts(incident_id);
CREATE INDEX IF NOT EXISTS idx_evaluations_forecast ON evaluations(forecast_id);
CREATE INDEX IF NOT EXISTS idx_evaluations_incident ON evaluations(incident_id);
CREATE INDEX IF NOT EXISTS idx_briefs_incident ON briefs(incident_id);
CREATE INDEX IF NOT EXISTS idx_alerts_incident ON alerts(incident_id);
CREATE INDEX IF NOT EXISTS idx_sent_emails_incident ON sent_emails(incident_id);
CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_active ON chunks(document_id) WHERE superseded_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_audit_occurred ON audit_log(occurred_at);
"""

FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text,
    content='chunks',
    content_rowid='id'
);

CREATE TRIGGER IF NOT EXISTS chunks_fts_insert AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;

CREATE TRIGGER IF NOT EXISTS chunks_fts_delete AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;
"""


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # check_same_thread=False is safe here: the daemon serializes writes
    # via apscheduler max_instances=1, and WAL mode lets dashboard
    # readers run concurrently without blocking. Without this flag,
    # apscheduler's worker thread would refuse to use a connection
    # opened on the daemon's main thread.
    conn = sqlite3.connect(
        db_path, isolation_level=None, check_same_thread=False,
    )
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.row_factory = sqlite3.Row
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.executescript(FTS_SCHEMA)
    conn.execute(
        "INSERT OR IGNORE INTO heartbeat(id, last_tick_at, tick_count) "
        "VALUES (1, datetime('now'), 0)"
    )


def integrity_check(conn: sqlite3.Connection) -> str:
    row = conn.execute("PRAGMA integrity_check").fetchone()
    return row[0] if row else "unknown"


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    conn.execute("BEGIN")
    try:
        yield conn
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
