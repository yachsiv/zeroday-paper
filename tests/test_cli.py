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


# ---------------------------------------------------------------- run_diag


def test_run_diag_main_dumps_json(monkeypatch, capsys):
    """zp-diag must run one read-only cycle and print the snapshot as JSON."""
    from zeroday_paper.cli import run_diag

    fake_snapshot = {
        "asof_utc": "2026-05-27T14:00:00+00:00",
        "asof_et": "2026-05-27T10:00:00-04:00",
        "expiry": "2026-05-27",
        "threshold": 13,
        "errors": [],
        "chain": {"spot": 7519.0, "calls": 100, "puts": 100,
                  "with_delta": 0, "without_delta": 200, "atm_strike": 7520.0},
        "signals": {"source": "flashalpha", "regime": "positive_gamma"},
        "proximity": {"to_put_wall": 124.0, "to_call_wall": 681.0,
                      "to_flip": 119.0, "to_magnet": -1.0, "above_flip": True},
        "patterns_l1": [],
        "patterns_l2": [],
        "strategies": [
            {"strategy": "BULL_PUT_SPREAD", "score": 17, "breakdown": {"base": 10},
             "regime_ok": True, "notes": [], "clears_threshold": True,
             "selection": {"spread_found": False, "reasons": ["no_short_in_delta_band"],
                           "candidates_considered": 0}},
        ],
    }

    async def stub():
        return fake_snapshot
    monkeypatch.setattr(run_diag, "diagnostic_snapshot", stub)
    monkeypatch.setattr(sys, "argv", ["zp-diag"])
    run_diag.main()

    captured = capsys.readouterr()
    import json as _json
    parsed = _json.loads(captured.out)
    assert parsed["threshold"] == 13
    assert parsed["chain"]["spot"] == 7519.0
    assert parsed["strategies"][0]["clears_threshold"] is True


def test_run_diag_main_failure_exits_1(monkeypatch):
    """If the snapshot raises, the CLI must exit non-zero so the operator notices."""
    from zeroday_paper.cli import run_diag

    async def boom():
        raise RuntimeError("boom")
    monkeypatch.setattr(run_diag, "diagnostic_snapshot", boom)
    monkeypatch.setattr(sys, "argv", ["zp-diag"])
    with pytest.raises(SystemExit) as exc:
        run_diag.main()
    assert exc.value.code == 1
