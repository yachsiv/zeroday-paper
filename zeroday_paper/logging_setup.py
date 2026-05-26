"""Structured logging for zeroday-paper.

JSON output suitable for CloudWatch Logs. Console output for local dev.
"""

from __future__ import annotations

import logging
import os
import sys

import structlog

from zeroday_paper.config import settings


def configure_logging() -> None:
    """Configure structlog + stdlib logging for the process.

    Idempotent — safe to call multiple times.
    """
    level = getattr(logging, settings.logging.level.upper(), logging.INFO)
    use_json = settings.logging.format == "json" and not sys.stderr.isatty()

    timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True)
    shared_processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        timestamper,
        structlog.processors.StackInfoRenderer(),
    ]

    if use_json or os.getenv("AWS_EXECUTION_ENV"):
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
