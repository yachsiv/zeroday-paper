"""Status brief: build, render, post + CLI happy path.

Covers the required scenarios from the spec:

    - happy-path with seeded journal (entries + closed + open with marks)
    - empty journal renders "no trades yet"
    - weekend renders "session closed" / "Last session"
    - open positions render the table
    - pre-market renders the wait-until-open countdown
    - diag snapshot vs heartbeat-only "last cycle"
    - Discord post error paths
    - CLI smoke + --no-discord + diag failure swallow
"""

from __future__ import annotations

import sys
from datetime import UTC, date, datetime, timedelta
from unittest.mock import MagicMock

import pytest

from zeroday_paper.cli import run_status
from zeroday_paper.engine.journal import Journal
from zeroday_paper.engine.models import (
    PositionStatus,
    TradeOutcome,
    TradeTick,
)
from zeroday_paper.reporting import status_brief as sb

# --------------------------------------------------------------------------- helpers


def _et_dt(year: int, month: int, day: int, hour: int, minute: int) -> datetime:
    """Build a UTC datetime that represents (year-month-day hour:minute) in ET."""
    return datetime(year, month, day, hour, minute, tzinfo=sb.ET).astimezone(UTC)


@pytest.fixture
def seeded_journal_path(tmp_path, make_paper_trade):
    """Write a small journal: one closed winner, one open trade with a tick."""
    db = tmp_path / "status.duckdb"
    j = Journal(str(db))

    entry_ts = datetime(2026, 5, 29, 14, 0, tzinfo=UTC)  # 10:00 ET on a Friday
    # Closed winner
    pt_closed = make_paper_trade(
        trade_id="closed-1", entry_ts=entry_ts, source="live",
        short_strike=5800.0, long_strike=5775.0, pattern="P01",
    )
    # Open trade
    pt_open = make_paper_trade(
        trade_id="open-1",
        entry_ts=datetime(2026, 5, 29, 17, 30, tzinfo=UTC),     # 13:30 ET
        source="live",
        short_strike=5825.0, long_strike=5800.0, pattern="P02",
    )
    j.write_trade(pt_closed)
    j.write_trade(pt_open)

    j.write_outcome(TradeOutcome(
        trade_id=pt_closed.trade_id, exit_ts=entry_ts + timedelta(minutes=45),
        exit_status=PositionStatus.CLOSED_TARGET, exit_spot=5810.0,
        exit_cost_bid=0.4, exit_cost_mid=0.42, pnl_bid=60.0, pnl_mid=58.0,
        held_minutes=45, max_excursion_pct=0.65, min_excursion_pct=-0.1,
        exit_reason="profit_target_60",
    ))

    # Tick for the open position so the renderer has a mark + unrealized P&L
    j.write_tick(TradeTick(
        trade_id=pt_open.trade_id,
        ts=datetime(2026, 5, 29, 18, 0, tzinfo=UTC),
        spot=5820.0,
        short_bid=1.0, short_ask=1.1, long_bid=0.3, long_ask=0.4,
        exit_cost_bid=0.7, exit_cost_mid=0.65,
        pnl_bid=30.0, pnl_mid=40.0,
        pct_of_max_profit=0.3, status=PositionStatus.WINNING,
    ))

    j.heartbeat("scanner", status="ok")
    j.close()
    return str(db)


@pytest.fixture
def empty_journal_path(tmp_path):
    db = tmp_path / "empty.duckdb"
    j = Journal(str(db))
    j.close()
    return str(db)


# --------------------------------------------------------------------------- market clock


def test_market_state_open_during_session():
    now = _et_dt(2026, 5, 29, 11, 0)  # Fri 11:00 ET
    assert sb._market_state(now) == "OPEN"


def test_market_state_pre_market_before_open():
    now = _et_dt(2026, 5, 29, 8, 0)  # Fri 08:00 ET
    assert sb._market_state(now) == "PRE_MARKET"


def test_market_state_after_hours():
    now = _et_dt(2026, 5, 29, 17, 30)  # Fri 17:30 ET
    assert sb._market_state(now) == "AFTER_HOURS"


def test_market_state_weekend():
    now = _et_dt(2026, 5, 30, 12, 0)  # Saturday
    assert sb._market_state(now) == "WEEKEND"


def test_seconds_to_session_end_in_session():
    now = _et_dt(2026, 5, 29, 14, 30)  # 14:30 ET, end 15:45
    s = sb._seconds_to_session_end(now)
    assert s is not None
    assert 60 * 60 <= s <= 90 * 60


def test_seconds_to_session_end_after_close_returns_none():
    assert sb._seconds_to_session_end(_et_dt(2026, 5, 29, 17, 0)) is None


def test_seconds_to_session_end_weekend_returns_none():
    assert sb._seconds_to_session_end(_et_dt(2026, 5, 30, 11, 0)) is None


def test_seconds_to_session_start_pre_market():
    now = _et_dt(2026, 5, 29, 8, 0)
    s = sb._seconds_to_session_start(now)
    assert s is not None
    assert 60 * 60 <= s <= 2 * 60 * 60


def test_seconds_to_session_start_during_session_returns_none():
    assert sb._seconds_to_session_start(_et_dt(2026, 5, 29, 11, 0)) is None


# --------------------------------------------------------------------------- duration formatter


def test_fmt_duration_seconds():
    assert sb._fmt_duration(45) == "45s"


def test_fmt_duration_minutes():
    assert sb._fmt_duration(125) == "2m 5s"


def test_fmt_duration_round_minutes():
    assert sb._fmt_duration(120) == "2m"


def test_fmt_duration_hours():
    assert sb._fmt_duration(3 * 3600 + 30 * 60) == "3h 30m"


def test_fmt_duration_days():
    assert sb._fmt_duration(2 * 86400 + 5 * 3600) == "2d 5h"


# --------------------------------------------------------------------------- strategy shortener


def test_short_strategy_known():
    assert sb._short_strategy("BULL_PUT_SPREAD") == "BULL_PUT"
    assert sb._short_strategy("BEAR_CALL_SPREAD") == "BEAR_CALL"
    assert sb._short_strategy("IRON_CONDOR") == "IC"


def test_short_strategy_unknown_passthrough():
    assert sb._short_strategy("MYSTERY") == "MYSTERY"


# --------------------------------------------------------------------------- split_pattern_field


def test_split_pattern_field_handles_none_and_empty():
    assert sb._split_pattern_field(None) == []
    assert sb._split_pattern_field("") == []


def test_split_pattern_field_trims_and_dedupes_spaces():
    assert sb._split_pattern_field("P01, P02 ,P03,") == ["P01", "P02", "P03"]


# --------------------------------------------------------------------------- build_bundle paths


def test_build_bundle_happy_path(seeded_journal_path):
    now = _et_dt(2026, 5, 29, 14, 5)  # Fri in-session
    bundle = sb.build_bundle(
        journal_path=seeded_journal_path,
        revision=":10",
        task_id="task-abc1234567890",
        runtime_seconds=2.3,
        now_utc=now,
    )
    assert bundle.market_state == "OPEN"
    assert bundle.today.entries == 2
    assert bundle.today.closed == 1
    assert bundle.today.wins == 1
    assert bundle.today.losses == 0
    assert bundle.today.realized_pnl_bid == 60.0
    assert bundle.today.open_count == 1
    assert len(bundle.open_positions) == 1
    op = bundle.open_positions[0]
    assert op.trade_id_short == "open-1"[:8]
    assert op.current_mark_bid == 0.7
    assert op.unrealized_pnl_bid == 30.0
    assert op.distance_from_short == round(5820.0 - 5825.0, 2)
    assert bundle.patterns.l1_total == 2     # P01 + P02
    assert bundle.risk.cap == 6              # config default
    assert bundle.risk.seconds_to_session_end is not None
    assert bundle.runtime_cost_usd is not None and bundle.runtime_cost_usd > 0
    # Heartbeat fallback engages when no diag passed
    assert bundle.last_cycle is not None
    assert bundle.last_cycle.source == "heartbeat_only"


def test_build_bundle_no_data(empty_journal_path):
    now = _et_dt(2026, 5, 29, 14, 0)
    bundle = sb.build_bundle(
        journal_path=empty_journal_path,
        now_utc=now,
        runtime_seconds=0.5,
    )
    assert bundle.today.entries == 0
    assert bundle.today.open_count == 0
    assert bundle.open_positions == ()
    assert bundle.patterns.l1_total == 0
    assert bundle.last_cycle is None
    md = sb.render_brief(bundle)
    assert "no trades yet" in md
    assert "Open positions" in md
    assert "_None._" in md


def test_build_bundle_weekend_falls_back_to_last_session(seeded_journal_path):
    # Saturday — should walk back to Friday (the only session in the seed)
    now = _et_dt(2026, 5, 30, 12, 0)
    bundle = sb.build_bundle(journal_path=seeded_journal_path, now_utc=now)
    assert bundle.market_state == "WEEKEND"
    assert bundle.today.session_date == date(2026, 5, 29)
    assert bundle.today.is_today is False
    assert bundle.today.entries == 2
    md = sb.render_brief(bundle)
    assert "Last session" in md
    assert "weekend" in md.lower()


def test_build_bundle_weekend_empty_falls_back_to_today(empty_journal_path):
    now = _et_dt(2026, 5, 30, 12, 0)  # Saturday w/ empty journal
    bundle = sb.build_bundle(journal_path=empty_journal_path, now_utc=now)
    # No prior session → session_date stays today_et
    assert bundle.today.session_date == date(2026, 5, 30)
    md = sb.render_brief(bundle)
    assert "no trades yet" in md


def test_build_bundle_pre_market(empty_journal_path):
    now = _et_dt(2026, 5, 29, 8, 0)  # 08:00 ET Fri
    bundle = sb.build_bundle(journal_path=empty_journal_path, now_utc=now)
    assert bundle.market_state == "PRE_MARKET"
    md = sb.render_brief(bundle)
    assert "session opens at 09:30 ET in" in md


def test_build_bundle_uses_diag_snapshot_for_last_cycle(seeded_journal_path):
    diag = {
        "asof_utc": "2026-05-29T17:50:00+00:00",
        "signals": {"spot": 5820.5, "regime": "positive_gamma"},
        "strategies": [
            {"strategy": "BULL_PUT_SPREAD", "score": 14},
            {"strategy": "BEAR_CALL_SPREAD", "score": 9},
        ],
    }
    now = _et_dt(2026, 5, 29, 14, 5)
    bundle = sb.build_bundle(
        journal_path=seeded_journal_path,
        diag_snapshot=diag,
        now_utc=now,
        runtime_seconds=3.0,
    )
    assert bundle.last_cycle is not None
    assert bundle.last_cycle.source == "diag_snapshot"
    assert bundle.last_cycle.spot == 5820.5
    assert bundle.last_cycle.score_bull_put == 14
    assert bundle.last_cycle.score_bear_call == 9
    md = sb.render_brief(bundle)
    assert "scan.score" in md
    assert "BULL_PUT **14**" in md


def test_build_bundle_diag_snapshot_handles_malformed_asof(empty_journal_path):
    diag = {"asof_utc": "not-a-timestamp", "signals": {"spot": None}}
    bundle = sb.build_bundle(
        journal_path=empty_journal_path,
        diag_snapshot=diag,
        now_utc=_et_dt(2026, 5, 29, 11, 0),
    )
    assert bundle.last_cycle is not None
    assert bundle.last_cycle.cycle_ts is None
    md = sb.render_brief(bundle)
    assert "no cycle timestamp" in md


def test_estimate_runtime_cost_none_for_invalid():
    assert sb._estimate_runtime_cost(None) is None
    assert sb._estimate_runtime_cost(0) is None
    assert sb._estimate_runtime_cost(-1) is None


def test_estimate_runtime_cost_positive():
    cost = sb._estimate_runtime_cost(60.0)
    assert cost is not None and cost > 0


# --------------------------------------------------------------------------- render variants


def test_render_brief_open_positions_table(seeded_journal_path):
    bundle = sb.build_bundle(
        journal_path=seeded_journal_path,
        now_utc=_et_dt(2026, 5, 29, 14, 5),
        runtime_seconds=1.0,
    )
    md = sb.render_brief(bundle)
    assert "## Open positions" in md
    assert "| id | strategy" in md
    # Strikes
    assert "5825 / 5800" in md
    # Mark + unrealized
    assert "$0.70" in md
    assert "$+30.00" in md


def test_render_brief_after_hours_label(seeded_journal_path):
    now = _et_dt(2026, 5, 29, 16, 30)
    bundle = sb.build_bundle(journal_path=seeded_journal_path, now_utc=now)
    md = sb.render_brief(bundle)
    assert "after-hours" in md
    # session closed appears in risk clock
    assert "session closed" in md


def test_render_brief_includes_errors_in_footer(empty_journal_path):
    now = _et_dt(2026, 5, 29, 11, 0)
    bundle = sb.build_bundle(journal_path=empty_journal_path, now_utc=now)
    # Inject a fake error to exercise the footer path
    bundle_with_err = sb.StatusBundle(
        **{**bundle.__dict__, "errors": ("test_err:boom",)}
    )
    md = sb.render_brief(bundle_with_err)
    assert "test_err:boom" in md


def test_state_pretty_unknown_passthrough():
    assert sb._state_pretty("MARS") == "MARS"


# --------------------------------------------------------------------------- post_to_discord


def test_post_to_discord_no_webhook_returns_false(monkeypatch):
    monkeypatch.setattr(sb, "_resolve_webhook", lambda: None)
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
    assert sb.post_to_discord("hello") is False


def test_post_to_discord_happy_path(monkeypatch):
    captured = []

    def fake_post(url, json=None, timeout=None):
        captured.append({"url": url, "body": json})
        resp = MagicMock()
        resp.status_code = 204
        resp.text = ""
        return resp

    monkeypatch.setattr(sb.httpx, "post", fake_post)
    assert sb.post_to_discord("hello world", webhook_url="https://x.test/hook") is True
    assert captured[0]["url"] == "https://x.test/hook"
    assert captured[0]["body"]["content"] == "hello world"


def test_post_to_discord_non_2xx_returns_false(monkeypatch):
    def fake_post(url, json=None, timeout=None):
        resp = MagicMock()
        resp.status_code = 500
        resp.text = "boom"
        return resp

    monkeypatch.setattr(sb.httpx, "post", fake_post)
    assert sb.post_to_discord("x", webhook_url="https://x.test/hook") is False


def test_post_to_discord_exception_returns_false(monkeypatch):
    def fake_post(*a, **kw):
        raise RuntimeError("net down")

    monkeypatch.setattr(sb.httpx, "post", fake_post)
    assert sb.post_to_discord("x", webhook_url="https://x.test/hook") is False


def test_resolve_webhook_returns_env(monkeypatch):
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://hook.test/abc")
    assert sb._resolve_webhook() == "https://hook.test/abc"


def test_resolve_webhook_failure_returns_none(monkeypatch):
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)

    def boom(_key):
        raise RuntimeError("nope")

    monkeypatch.setattr(sb, "discord_webhook", boom)
    assert sb._resolve_webhook() is None


# --------------------------------------------------------------------------- CLI smoke


def test_run_status_main_smoke(monkeypatch, tmp_path):
    db = str(tmp_path / "cli.duckdb")
    Journal(db).close()

    async def stub_diag():
        return None

    monkeypatch.setattr(run_status, "_maybe_diag", stub_diag)
    monkeypatch.setattr(run_status, "_resolve_revision", lambda: ":10")
    monkeypatch.setattr(run_status, "_resolve_task_id", lambda: "abc12345")
    monkeypatch.setattr(
        run_status, "build_bundle",
        lambda **kw: sb.build_bundle(journal_path=db, **kw),
    )
    monkeypatch.setattr(run_status, "post_to_discord", lambda md: True)
    monkeypatch.setattr(sys, "argv", ["zp-status", "--no-discord", "--no-diag", "--print"])
    with pytest.raises(SystemExit) as exc:
        run_status.main()
    assert exc.value.code == 0


def test_run_status_main_discord_failure_returns_1(monkeypatch, tmp_path):
    db = str(tmp_path / "cli.duckdb")
    Journal(db).close()
    monkeypatch.setattr(
        run_status, "build_bundle",
        lambda **kw: sb.build_bundle(journal_path=db, **kw),
    )
    monkeypatch.setattr(run_status, "post_to_discord", lambda md: False)
    monkeypatch.setattr(sys, "argv", ["zp-status", "--no-diag"])
    with pytest.raises(SystemExit) as exc:
        run_status.main()
    assert exc.value.code == 1


def test_run_status_maybe_diag_swallows_errors(monkeypatch):
    """If diagnostic_snapshot raises, the CLI helper returns None instead of bubbling."""
    import asyncio

    async def boom():
        raise RuntimeError("polygon down")

    from zeroday_paper.engine import scanner as scanner_mod
    monkeypatch.setattr(scanner_mod, "diagnostic_snapshot", boom)
    out = asyncio.run(run_status._maybe_diag())
    assert out is None


def test_run_status_resolve_revision_from_env(monkeypatch):
    monkeypatch.setenv("ZP_REVISION", ":42")
    assert run_status._resolve_revision() == ":42"


def test_run_status_resolve_revision_missing(monkeypatch):
    monkeypatch.delenv("ZP_REVISION", raising=False)
    monkeypatch.delenv("ECS_TASK_DEFINITION_REVISION", raising=False)
    assert run_status._resolve_revision() is None


def test_run_status_resolve_task_id_from_env(monkeypatch):
    monkeypatch.setenv("ZP_TASK_ID", "fake-arn-1234")
    assert run_status._resolve_task_id() == "fake-arn-1234"
