"""One-shot diagnostic dump.

Runs a single scan cycle (read-only — no journal write) and prints the full
state, score breakdown, proximity, strike selection result, and entry quote
as JSON to stdout. Exits 0.

This is the "ssh into the running paper system" view we lacked when 0 trades
landed across two consecutive sessions. Wired into ``entrypoint.sh`` as
``MODE=diag`` so we can fire a Fargate one-off task whenever the scanner
goes quiet.

Usage:
    uv run zp-diag
    MODE=diag <docker> ./entrypoint.sh
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import date, datetime

from zeroday_paper.engine.scanner import diagnostic_snapshot
from zeroday_paper.logging_setup import configure_logging, get_logger


def _json_default(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, date):
        return obj.isoformat()
    return str(obj)


def main() -> None:
    configure_logging()
    log = get_logger(__name__)
    log.info("diag.boot")
    try:
        snapshot = asyncio.run(diagnostic_snapshot())
    except Exception as exc:
        log.exception("diag.failed", error=str(exc))
        sys.exit(1)
    # Log a single structured line for CloudWatch search, then dump to stdout
    # for human / curl convenience.
    log.info("diag.snapshot", **{k: snapshot.get(k) for k in ("threshold", "errors")})
    print(json.dumps(snapshot, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
