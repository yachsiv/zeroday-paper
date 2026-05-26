"""Reporting aggregates over a real tmp DuckDB."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from zeroday_paper.engine.journal import Journal
from zeroday_paper.engine.models import PositionStatus, TradeOutcome
from zeroday_paper.reporting import stats


@pytest.fixture
def seeded_journal(tmp_path, make_paper_trade, monkeypatch):
    """A tmp DuckDB pre-loaded with trades+outcomes; settings.duckdb_path patched."""
    db = tmp_path / "stats.duckdb"
    j = Journal(str(db))

    # 3 trades on the same date, mixed outcomes
    base = datetime(2025, 5, 28, 14, 0, tzinfo=UTC)
    trades = [
        make_paper_trade(trade_id=f"t-{i}",
                         entry_ts=base + timedelta(minutes=i),
                         short_strike=5775.0 - i, long_strike=5750.0 - i,
                         source="live", pattern=("P02" if i % 2 == 0 else "P01"))
        for i in range(3)
    ]
    for t in trades:
        j.write_trade(t)
    # Outcome 1: TARGET win, P&L 50
    j.write_outcome(TradeOutcome(
        trade_id="t-0", exit_ts=base + timedelta(hours=1),
        exit_status=PositionStatus.CLOSED_TARGET, exit_spot=5800.0,
        exit_cost_bid=0.40, exit_cost_mid=0.42,
        pnl_bid=50.0, pnl_mid=55.0,
        held_minutes=60, max_excursion_pct=0.6, min_excursion_pct=-0.1,
        exit_reason="profit_target",
    ))
    # Outcome 2: STOP loss, P&L -100
    j.write_outcome(TradeOutcome(
        trade_id="t-1", exit_ts=base + timedelta(hours=2),
        exit_status=PositionStatus.CLOSED_STOP, exit_spot=5750.0,
        exit_cost_bid=3.0, exit_cost_mid=3.0,
        pnl_bid=-100.0, pnl_mid=-100.0,
        held_minutes=120, max_excursion_pct=0.2, min_excursion_pct=-2.5,
        exit_reason="stop_loss_2x",
    ))
    # Outcome 3: HARD close, P&L 20
    j.write_outcome(TradeOutcome(
        trade_id="t-2", exit_ts=base + timedelta(hours=3),
        exit_status=PositionStatus.CLOSED_HARD, exit_spot=5790.0,
        exit_cost_bid=0.7, exit_cost_mid=0.7,
        pnl_bid=20.0, pnl_mid=20.0,
        held_minutes=180, max_excursion_pct=0.3, min_excursion_pct=-0.1,
        exit_reason="hard_close",
    ))
    # Replay trade for source filtering
    j.write_trade(make_paper_trade(
        trade_id="t-replay-1",
        entry_ts=base, short_strike=5700.0, long_strike=5675.0,
        source="replay", pattern="P03",
    ))
    j.close()

    # Patch the `settings` reference inside the stats module so _connect()
    # opens the tmp DB instead of the production path.
    monkeypatch.setattr(stats, "settings", _patched_settings(str(db)))
    return db


def _patched_settings(path: str):
    """Lightweight settings shim with a duckdb_path property."""
    from zeroday_paper.config import settings as real_settings
    class _Shim:
        duckdb_path = path
        engine = real_settings.engine
        reporting = real_settings.reporting
        regime_gates = real_settings.regime_gates
        strikes = real_settings.strikes
        exits = real_settings.exits
        concurrency = real_settings.concurrency
        dedup = real_settings.dedup
        replay = real_settings.replay
        patterns = real_settings.patterns
        storage = real_settings.storage
        alarms = real_settings.alarms
        secrets = real_settings.secrets
        logging = real_settings.logging
    return _Shim()


# --------------------------------------------------------------- window_stats


def test_window_stats_aggregates_all_sources(seeded_journal):
    target = date(2025, 5, 28)
    w = stats.window_stats(start=target, end=target)
    assert w.trades == 4  # 3 live + 1 replay
    assert w.closed == 3
    assert w.wins == 1
    assert w.losses == 1
    assert 0 < w.win_rate <= 1.0
    assert w.total_pnl_bid == pytest.approx(-30.0)
    assert w.best_trade == pytest.approx(50.0)
    assert w.worst_trade == pytest.approx(-100.0)


def test_window_stats_filters_by_source_live(seeded_journal):
    target = date(2025, 5, 28)
    w = stats.window_stats(start=target, end=target, source="live")
    assert w.trades == 3


def test_window_stats_filters_by_source_replay(seeded_journal):
    target = date(2025, 5, 28)
    w = stats.window_stats(start=target, end=target, source="replay")
    assert w.trades == 1
    assert w.closed == 0  # the replay trade has no outcome
    assert w.win_rate == 0.0


def test_window_stats_window_label(seeded_journal):
    target = date(2025, 5, 28)
    w = stats.window_stats(start=target, end=target, label="custom-label")
    assert w.window == "custom-label"


def test_window_stats_empty_window(seeded_journal):
    far = date(2024, 1, 1)
    w = stats.window_stats(start=far, end=far)
    assert w.trades == 0
    assert w.win_rate == 0.0
    assert w.total_pnl_bid == 0.0


# ----------------------------------------------------------- pattern_breakdown


def test_pattern_breakdown(seeded_journal):
    target = date(2025, 5, 28)
    rows = stats.pattern_breakdown(start=target, end=target)
    # 3 live trades: t-0 P02, t-1 P01, t-2 P02 → 2 patterns (P02, P01)
    # Plus the replay row P03
    ids = {r.pattern_id for r in rows}
    assert "P02" in ids
    assert "P01" in ids
    assert "P03" in ids


def test_pattern_breakdown_empty_window(seeded_journal):
    far = date(2020, 1, 1)
    rows = stats.pattern_breakdown(start=far, end=far)
    assert rows == []


# ------------------------------------------------------------ regime_breakdown


def test_regime_breakdown(seeded_journal):
    target = date(2025, 5, 28)
    r = stats.regime_breakdown(start=target, end=target)
    assert "positive_gamma" in r
    assert r["positive_gamma"]["trades"] == 4


def test_regime_breakdown_empty(seeded_journal):
    far = date(2020, 1, 1)
    r = stats.regime_breakdown(start=far, end=far)
    assert r == {}


# ----------------------------------------------------------- helper aggregates


def test_today_window_stats(seeded_journal, monkeypatch):
    # Avoid depending on real "today" by passing the target
    w = stats.today_window_stats(date(2025, 5, 28))
    assert w.trades == 3  # live-only on that date


def test_all_time_window_stats_runs(seeded_journal):
    w = stats.all_time_window_stats()
    assert w is not None
    assert isinstance(w.trades, int)
