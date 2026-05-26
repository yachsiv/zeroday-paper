"""Run historical replay.

Usage:
    uv run zp-replay --days 365
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from zeroday_paper.engine.replay import run_replay
from zeroday_paper.logging_setup import configure_logging, get_logger


def main() -> None:
    configure_logging()
    log = get_logger(__name__)

    parser = argparse.ArgumentParser(description="zeroday-paper replay")
    parser.add_argument("--days", type=int, default=None, help="Days back to replay (default: config)")
    args = parser.parse_args()

    log.info("replay.start", days=args.days)
    try:
        progress = asyncio.run(run_replay(days_back=args.days))
    except KeyboardInterrupt:
        log.warning("replay.interrupted")
        sys.exit(130)

    log.info(
        "replay.done",
        days_processed=progress.days_processed,
        chains_fetched=progress.chains_fetched,
        states_scored=progress.states_scored,
        trades_written=progress.trades_written,
        errors=progress.errors,
    )


if __name__ == "__main__":
    main()
