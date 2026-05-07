"""Forecast → model → SQLite orchestration.

Pulls a forecast (`weather.py`), runs it through the model (`model.py`),
writes raw forecast and evaluation rows to SQLite (`db.py`), then checks
whether the evaluation crosses the alerting threshold (handed to
`alerts.py`).
"""

from __future__ import annotations


def run_once() -> None:
    raise NotImplementedError("Phase 2")
