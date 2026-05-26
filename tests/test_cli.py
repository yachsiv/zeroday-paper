"""CLI entry-point smoke tests."""

from __future__ import annotations

import sys
from datetime import UTC, date
from unittest.mock import AsyncMock, MagicMock

import pytest

from zeroday_paper.cli import run_replay, run_report


# ---------------------------------------------------------------- run_replay


def test_run_replay_main_smoke(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["zp-replay", "--days", "0"])
    from zeroday_paper.engine.replay import ReplayProgress
    monkeypatch.setattr(run_replay, "run_replay", AsyncMock(return_value=ReplayProgress()))
    run_replay.main()


def test_run_replay_main_keyboard_interrupt(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["zp-replay"])

    async def boom(*a, **kw):
        raise KeyboardInterrupt()

    monkeypatch.setattr(run_replay, "run_replay", boom)
    with pytest.raises(SystemExit) as exc:
        run_replay.main()
    assert exc.value.code == 130


# ---------------------------------------------------------------- run_report


def test_run_report_main_no_discord(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["zp-report", "--no-discord", "--reports-dir", str(tmp_path)])
    monkeypatch.setattr(run_report, "build_markdown", lambda: "# fake report\n")
    monkeypatch.setattr(
        run_report,
        "write_report_files",
        lambda md, base_dir: (tmp_path / "r.md", tmp_path / "r.html"),
    )
    monkeypatch.setattr(run_report, "post_to_discord", lambda md: True)
    run_report.main()


def test_run_report_main_discord_path(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["zp-report", "--reports-dir", str(tmp_path)])
    monkeypatch.setattr(run_report, "build_markdown", lambda: "# fake report\n")
    monkeypatch.setattr(
        run_report,
        "write_report_files",
        lambda md, base_dir: (tmp_path / "r.md", tmp_path / "r.html"),
    )
    monkeypatch.setattr(run_report, "post_to_discord", lambda md: True)
    run_report.main()


# ---------------------------------------------------------------- run_live (smoke)


def test_run_live_main_smoke(monkeypatch):
    """run_live.main wraps run_live_loop in asyncio.run; we stub the coroutine."""
    from zeroday_paper.cli import run_live

    async def stub_loop():
        return None
    monkeypatch.setattr(run_live, "run_live_loop", stub_loop)
    monkeypatch.setattr(sys, "argv", ["zp-live"])
    run_live.main()


def test_run_live_main_keyboard_interrupt(monkeypatch):
    from zeroday_paper.cli import run_live

    async def boom():
        raise KeyboardInterrupt()
    monkeypatch.setattr(run_live, "run_live_loop", boom)
    monkeypatch.setattr(sys, "argv", ["zp-live"])
    with pytest.raises(SystemExit) as exc:
        run_live.main()
    assert exc.value.code == 130
