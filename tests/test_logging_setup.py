"""Structlog configuration helpers."""

from __future__ import annotations

import structlog

from zeroday_paper.logging_setup import configure_logging, get_logger


def test_configure_logging_is_idempotent():
    configure_logging()
    configure_logging()


def test_get_logger_returns_bound_logger():
    configure_logging()
    log = get_logger("test")
    assert hasattr(log, "info")
    assert hasattr(log, "warning")
    # Doesn't raise
    log.info("hello", k=1)


def test_get_logger_default_name():
    log = get_logger()
    assert log is not None
