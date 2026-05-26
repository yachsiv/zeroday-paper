"""DuckDB journal: schema, idempotency, round-trips."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from zeroday_paper.engine.journal import Journal, SCHEMA_SQL, trade_id_for
from zeroday_paper.engine.models import (
    PositionStatus,
    StrategyType,
    TradeOutcome,
    TradeTick,
)


# ------------------------------------------------------------------- trade_id_for


def test_trade_id_for_deterministic():
    args = dict(
        entry_date=date(2025, 5, 28), entry_minute=600,
        short_strike=5775.0, long_strike=5750.0,
        strategy=StrategyType.BULL_PUT, source="live",
    )
    assert trade_id_for(**args) == trade_id_for(**args)


def test_trade_id_for_distinguishes_each_field():
    base = dict(
        entry_date=date(2025, 5, 28), entry_minute=600,
        short_strike=5775.0, long_strike=5750.0,
        strategy=StrategyType.BULL_PUT, source="live",
    )
    a = trade_id_for(**base)
    # Vary each field
    variants = [
        {**base, "entry_date": date(2025, 5, 29)},
        {**base, "entry_minute": 601},
        {**base, "short_strike": 5780.0},
        {**base, "long_strike": 5755.0},
        {**base, "strategy": StrategyType.BEAR_CALL},
        {**base, "source": "replay"},
    ]
    for v in variants:
        assert trade_id_for(**v) != a


def test_trade_id_for_24_char_hex():
    out = trade_id_for(
        entry_date=date(2025, 5, 28), entry_minute=600,
        short_strike=5775.0, long_strike=5750.0,
        strategy=StrategyType.BULL_PUT, source="live",
    )
    assert len(out) == 24
    int(out, 16)  # parseable as hex


# --------------------------------------------------------------- schema + lifecycle


def test_journal_schema_creates_tables(tmp_journal):
    cur = tmp_journal._conn.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main' ORDER BY table_name"
    )
    names = [r[0] for r in cur.fetchall()]
    assert "paper_trades" in names
    assert "paper_ticks" in names
    assert "paper_outcomes" in names
    assert "replay_cursor" in names
    assert "scanner_heartbeat" in names


def test_journal_schema_idempotent(tmp_path):
    # Re-instantiate journal twice on the same path
    p = str(tmp_path / "j.duckdb")
    j1 = Journal(p)
    j1.close()
    j2 = Journal(p)
    j2.close()


def test_journal_creates_parent_dir(tmp_path):
    nested = tmp_path / "nested" / "dir" / "j.duckdb"
    j = Journal(str(nested))
    j.close()
    assert nested.parent.exists()


# ------------------------------------------------------------------------- writes


def test_write_trade_returns_true_first_time(tmp_journal, make_paper_trade):
    pt = make_paper_trade()
    assert tmp_journal.write_trade(pt) is True


def test_write_trade_idempotent_duplicate_returns_false(tmp_journal, make_paper_trade):
    pt = make_paper_trade()
    assert tmp_journal.write_trade(pt) is True
    assert tmp_journal.write_trade(pt) is False
    # row count = 1
    cur = tmp_journal._conn.execute("SELECT COUNT(*) FROM paper_trades")
    assert cur.fetchone()[0] == 1


def test_write_trade_populates_all_columns(tmp_journal, make_paper_trade):
    pt = make_paper_trade(short_strike=5750.0, long_strike=5725.0, width=25.0)
    tmp_journal.write_trade(pt)
    row = tmp_journal._conn.execute(
        "SELECT trade_id, strategy, short_strike, long_strike, width, status, source FROM paper_trades"
    ).fetchone()
    assert row[0] == pt.trade_id
    assert row[1] == str(pt.strategy)
    assert row[2] == 5750.0
    assert row[3] == 5725.0
    assert row[4] == 25.0
    assert row[5] == "OPEN"
    assert row[6] == "replay"


def test_write_tick_upserts(tmp_journal, make_paper_trade):
    pt = make_paper_trade()
    tmp_journal.write_trade(pt)
    base_tick = TradeTick(
        trade_id=pt.trade_id, ts=pt.entry_ts, spot=5800.0,
        short_bid=0.3, short_ask=0.4, long_bid=0.1, long_ask=0.15,
        exit_cost_bid=0.30, exit_cost_mid=0.25, pnl_bid=10.0, pnl_mid=12.0,
        pct_of_max_profit=0.5, status=PositionStatus.WINNING,
    )
    tmp_journal.write_tick(base_tick)
    tmp_journal.write_tick(base_tick)  # same ts → INSERT OR REPLACE
    n = tmp_journal._conn.execute("SELECT COUNT(*) FROM paper_ticks").fetchone()[0]
    assert n == 1


def test_write_outcome_flips_status(tmp_journal, make_paper_trade):
    pt = make_paper_trade()
    tmp_journal.write_trade(pt)
    outcome = TradeOutcome(
        trade_id=pt.trade_id, exit_ts=pt.entry_ts, exit_status=PositionStatus.CLOSED_TARGET,
        exit_spot=5800.0, exit_cost_bid=0.50, exit_cost_mid=0.55,
        pnl_bid=50.0, pnl_mid=55.0, held_minutes=60,
        max_excursion_pct=0.6, min_excursion_pct=-0.2,
        exit_reason="profit_target",
    )
    tmp_journal.write_outcome(outcome)
    row = tmp_journal._conn.execute(
        "SELECT status FROM paper_trades WHERE trade_id = ?", [pt.trade_id]
    ).fetchone()
    assert row[0] == str(PositionStatus.CLOSED_TARGET)


def test_heartbeat_writes_row(tmp_journal):
    tmp_journal.heartbeat("scanner")
    r = tmp_journal._conn.execute(
        "SELECT component, last_status FROM scanner_heartbeat"
    ).fetchone()
    assert r[0] == "scanner"
    assert r[1] == "ok"


def test_heartbeat_overwrites_on_repeat(tmp_journal):
    tmp_journal.heartbeat("scanner", status="ok")
    tmp_journal.heartbeat("scanner", status="degraded")
    rows = tmp_journal._conn.execute(
        "SELECT last_status FROM scanner_heartbeat"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "degraded"


# -------------------------------------------------------------------------- reads


def test_open_positions_returns_only_active(tmp_journal, make_paper_trade):
    pt_open = make_paper_trade(trade_id="open-1")
    pt_closed = make_paper_trade(trade_id="closed-1", short_strike=5800.0, long_strike=5775.0)
    tmp_journal.write_trade(pt_open)
    tmp_journal.write_trade(pt_closed)

    # close the second
    outcome = TradeOutcome(
        trade_id=pt_closed.trade_id, exit_ts=pt_closed.entry_ts,
        exit_status=PositionStatus.CLOSED_TARGET, exit_spot=5800.0,
        exit_cost_bid=0.5, exit_cost_mid=0.55, pnl_bid=50.0, pnl_mid=55.0,
        held_minutes=60, max_excursion_pct=0.6, min_excursion_pct=-0.2,
        exit_reason="t",
    )
    tmp_journal.write_outcome(outcome)

    rows = tmp_journal.open_positions()
    ids = {r["trade_id"] for r in rows}
    assert "open-1" in ids
    assert "closed-1" not in ids


def test_count_today(tmp_journal, make_paper_trade):
    today = date(2025, 5, 28)
    pt1 = make_paper_trade(trade_id="a", entry_ts=datetime(2025, 5, 28, 14, 0, tzinfo=UTC), source="live")
    pt2 = make_paper_trade(trade_id="b", entry_ts=datetime(2025, 5, 28, 14, 5, tzinfo=UTC), source="live")
    pt3 = make_paper_trade(trade_id="c", entry_ts=datetime(2025, 5, 28, 14, 0, tzinfo=UTC), source="replay")
    pt4 = make_paper_trade(trade_id="d", entry_ts=datetime(2025, 5, 27, 14, 0, tzinfo=UTC), source="live")
    for pt in (pt1, pt2, pt3, pt4):
        tmp_journal.write_trade(pt)
    assert tmp_journal.count_today(today, "live") == 2
    assert tmp_journal.count_today(today, "replay") == 1


def test_count_today_zero_when_none(tmp_journal):
    assert tmp_journal.count_today(date(2024, 1, 1)) == 0


def test_replay_cursor_roundtrip(tmp_journal):
    tmp_journal.mark_chunk_done(1, date(2025, 5, 1), date(2025, 5, 30))
    tmp_journal.mark_chunk_done(2, date(2025, 4, 1), date(2025, 4, 30))
    done = tmp_journal.replay_chunks_done()
    assert len(done) == 2
    starts = {d[0] for d in done}
    assert date(2025, 5, 1) in starts
    assert date(2025, 4, 1) in starts


def test_journal_transaction_rolls_back(tmp_journal, make_paper_trade):
    pt = make_paper_trade()
    with pytest.raises(RuntimeError):
        with tmp_journal.transaction() as conn:
            tmp_journal._conn.execute(
                "INSERT INTO scanner_heartbeat(component, last_run_at, last_status) VALUES (?, ?, ?)",
                ["x", datetime(2025, 1, 1, tzinfo=UTC), "ok"],
            )
            raise RuntimeError("forced")
    # Row should have rolled back
    assert tmp_journal._conn.execute(
        "SELECT COUNT(*) FROM scanner_heartbeat"
    ).fetchone()[0] == 0


def test_journal_transaction_commit(tmp_journal):
    with tmp_journal.transaction() as conn:
        tmp_journal._conn.execute(
            "INSERT INTO scanner_heartbeat(component, last_run_at, last_status) VALUES (?, ?, ?)",
            ["x", datetime(2025, 1, 1, tzinfo=UTC), "ok"],
        )
    assert tmp_journal._conn.execute(
        "SELECT COUNT(*) FROM scanner_heartbeat"
    ).fetchone()[0] == 1
