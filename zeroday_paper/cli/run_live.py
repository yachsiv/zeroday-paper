"""Run the live 2-min scanner+monitor loop.

Usage:
    uv run zp-live
"""

from __future__ import annotations

import asyncio
import sys

from zeroday_paper.engine.scanner import run_live_loop
from zeroday_paper.logging_setup import configure_logging, get_logger


def main() -> None:
    configure_logging()
    log = get_logger(__name__)
    log.info("live.boot")
    try:
        asyncio.run(run_live_loop())
    except KeyboardInterrupt:
        log.warning("live.interrupted")
        sys.exit(130)


if __name__ == "__main__":
    main()
