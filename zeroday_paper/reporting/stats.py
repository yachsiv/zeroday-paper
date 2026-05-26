"""Aggregations over the paper-trade journal.

Pure read-only. Uses DuckDB SQL directly so we can reuse the same connection
opened by the Journal.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

import duckdb

from zeroday_paper.config import settings


@dataclass(frozen=True)
class WindowStats:
    window: str
    trades: int
    closed: int
    wins: int
    losses: int
    win_rate: float
    total_pnl_bid: float
    avg_pnl_bid: float
    median_pnl_bid: float
    expectancy: float
    best_trade: float
    worst_trade: float


@dataclass(frozen=True)
class PatternRow:
    pattern_id: str
    layer: str
    trades: int
    win_rate: float
    avg_pnl_bid: float


def _connect() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(settings.duckdb_path, read_only=True)


def window_stats(*, start: date, end: date, source: str | None = None, label: str | None = None) -> WindowStats:
    where = "entry_date BETWEEN ? AND ?"
    params: list[Any] = [start, end]
    if source is not None:
        where += " AND source = ?"
        params.append(source)

    with _connect() as conn:
        row = conn.execute(
            f"""
            SELECT
                COUNT(*) AS trades,
                COUNT(CASE WHEN status LIKE 'CLOSED%' THEN 1 END) AS closed,
                COUNT(CASE WHEN status = 'CLOSED_TARGET' THEN 1 END) AS wins,
                COUNT(CASE WHEN status IN ('CLOSED_STOP','CLOSED_THESIS') THEN 1 END) AS losses
            FROM paper_trades
            WHERE {where}
            """,
            params,
        ).fetchone()

        pnl_row = conn.execute(
            f"""
            SELECT
                COALESCE(SUM(o.pnl_bid),0)            AS total_pnl,
                COALESCE(AVG(o.pnl_bid),0)            AS avg_pnl,
                COALESCE(MEDIAN(o.pnl_bid),0)         AS median_pnl,
                COALESCE(MAX(o.pnl_bid),0)            AS best,
                COALESCE(MIN(o.pnl_bid),0)            AS worst
            FROM paper_outcomes o
            JOIN paper_trades t USING (trade_id)
            WHERE t.entry_date BETWEEN ? AND ?
              {"AND t.source = ?" if source is not None else ""}
            """,
            params,
        ).fetchone()

    trades = int(row[0]) if row else 0
    closed = int(row[1]) if row else 0
    wins = int(row[2]) if row else 0
    losses = int(row[3]) if row else 0
    win_rate = wins / closed if closed else 0.0

    total_pnl = float(pnl_row[0]) if pnl_row else 0.0
    avg_pnl = float(pnl_row[1]) if pnl_row else 0.0
    median_pnl = float(pnl_row[2]) if pnl_row else 0.0
    best = float(pnl_row[3]) if pnl_row else 0.0
    worst = float(pnl_row[4]) if pnl_row else 0.0

    expectancy = total_pnl / closed if closed else 0.0

    return WindowStats(
        window=label or f"{start}..{end}",
        trades=trades,
        closed=closed,
        wins=wins,
        losses=losses,
        win_rate=round(win_rate, 4),
        total_pnl_bid=round(total_pnl, 2),
        avg_pnl_bid=round(avg_pnl, 2),
        median_pnl_bid=round(median_pnl, 2),
        expectancy=round(expectancy, 2),
        best_trade=round(best, 2),
        worst_trade=round(worst, 2),
    )


def pattern_breakdown(*, start: date, end: date) -> list[PatternRow]:
    """Win rate per L1 pattern over the window."""
    with _connect() as conn:
        rows = conn.execute(
            """
            WITH expanded AS (
                SELECT
                    t.trade_id,
                    o.pnl_bid,
                    o.exit_status,
                    UNNEST(string_split(t.active_patterns_l1, ',')) AS pattern_id
                FROM paper_trades t
                LEFT JOIN paper_outcomes o USING (trade_id)
                WHERE t.entry_date BETWEEN ? AND ?
                  AND t.active_patterns_l1 IS NOT NULL
                  AND t.active_patterns_l1 <> ''
            )
            SELECT
                pattern_id,
                COUNT(*) AS trades,
                AVG(CASE WHEN exit_status = 'CLOSED_TARGET' THEN 1.0 ELSE 0.0 END) AS win_rate,
                COALESCE(AVG(pnl_bid),0) AS avg_pnl
            FROM expanded
            WHERE pattern_id <> ''
            GROUP BY pattern_id
            ORDER BY trades DESC
            """,
            [start, end],
        ).fetchall()

    return [
        PatternRow(
            pattern_id=r[0],
            layer="L1",
            trades=int(r[1]),
            win_rate=round(float(r[2]), 4),
            avg_pnl_bid=round(float(r[3]), 2),
        )
        for r in rows
    ]


def regime_breakdown(*, start: date, end: date) -> dict[str, dict[str, float]]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT
                t.gamma_regime,
                COUNT(*) AS trades,
                AVG(CASE WHEN o.exit_status = 'CLOSED_TARGET' THEN 1.0 ELSE 0.0 END) AS win_rate,
                COALESCE(SUM(o.pnl_bid),0) AS total_pnl
            FROM paper_trades t
            LEFT JOIN paper_outcomes o USING (trade_id)
            WHERE t.entry_date BETWEEN ? AND ?
            GROUP BY t.gamma_regime
            """,
            [start, end],
        ).fetchall()
    return {
        (r[0] or "unknown"): {
            "trades": int(r[1]),
            "win_rate": round(float(r[2] or 0.0), 4),
            "total_pnl": round(float(r[3] or 0.0), 2),
        }
        for r in rows
    }


def today_window_stats(today: date | None = None) -> WindowStats:
    t = today or datetime.utcnow().date()
    return window_stats(start=t, end=t, source="live", label=f"live-{t}")


def all_time_window_stats() -> WindowStats:
    today = datetime.utcnow().date()
    return window_stats(start=today - timedelta(days=400), end=today, label="all-time")
