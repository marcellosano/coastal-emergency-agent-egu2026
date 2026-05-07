"""Daemon entrypoint.

Run as: `python -m coastal_agent.main` or via the systemd unit
`deploy/coastal-agent.service`. Loads `Settings` from environment
(populated by systemd `EnvironmentFile=/opt/coastal-agent.env`),
opens the SQLite database, builds an `LLMComposer` + `LivePolicy` +
`Orchestrator`, then enters the apscheduler blocking loop.

Live mode requires the optional `live` dependency group on this
machine — see `coastal_agent.policy.LivePolicy`.
"""

from __future__ import annotations

import logging

from coastal_agent.config import Settings
from coastal_agent.db import connect, init_schema, integrity_check
from coastal_agent.scheduler import Daemon


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    log = logging.getLogger(__name__)

    settings = Settings()
    log.info(
        "coastal-agent starting; db=%s vendor_dir=%s llm_enabled=%s",
        settings.state_db_path, settings.vendor_dir, settings.llm_enabled,
    )

    conn = connect(settings.state_db_path)
    init_schema(conn)
    result = integrity_check(conn)
    log.info("schema initialised; integrity_check=%s", result)
    if result != "ok":
        log.error("SQLite integrity check failed: %s", result)
        raise SystemExit(1)

    daemon = Daemon.from_settings(conn, settings)
    log.info(
        "daemon constructed; LivePolicy ready, composer enabled=%s, "
        "tools=6, poll_interval=%ds",
        daemon.orchestrator.composer is not None
        and daemon.orchestrator.composer.enabled,
        daemon.poll_interval_seconds,
    )

    daemon.start_blocking()


if __name__ == "__main__":
    main()
