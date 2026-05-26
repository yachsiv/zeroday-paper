"""DuckDB journal.

Three tables:

    paper_trades        — one row per entry. Immutable after insert.
    paper_ticks         — one row per polling cycle while position open.
    paper_outcomes      — one row per position close. Joined to paper_trades by trade_id.

Idempotent: every insert uses ON CONFLICT DO NOTHING on the natural key.

Why DuckDB (vs SQLite or Postgres):
    - Columnar storage → fast `pl.read_database` for analytics.
    - Single-file → trivial backup to S3.
    - Embedded → no extra Fargate task.
    - Reads + writes are safe within one process; we are single-writer by design.
"""

from __future__ import annotations

import hashlib
import json
import threading
from contextlib import contextmanager
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Iterator

import duckdb
import structlog

from zeroday_paper.config import settings
from zeroday_paper.engine.models import (
    PaperTrade,
    PositionStatus,
    StrategyType,
    TradeOutcome,
    TradeTick,
)

logger = structlog.get_logger(__name__)

SCHEMA_VERSION = 1


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_meta (
    schema_version INTEGER NOT NULL,
    applied_at     TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS paper_trades (
    trade_id          TEXT PRIMARY KEY,
    strategy          TEXT NOT NULL,
    entry_ts          TIMESTAMP NOT NULL,
    entry_date        DATE NOT NULL,
    entry_minute      INTEGER NOT NULL,
    expiry            DATE NOT NULL,
    source            TEXT NOT NULL,                 -- 'live' | 'replay'

    spot_at_entry     DOUBLE NOT NULL,
    short_strike      DOUBLE NOT NULL,
    long_strike       DOUBLE NOT NULL,
    short_contract    TEXT NOT NULL,
    long_contract     TEXT NOT NULL,
    width             DOUBLE NOT NULL,
    contracts         INTEGER NOT NULL,

    credit_mid        DOUBLE NOT NULL,
    credit_bid        DOUBLE NOT NULL,
    max_loss_bid      DOUBLE NOT NULL,
    credit_bid_ratio  DOUBLE NOT NULL,

    short_delta       DOUBLE,
    long_delta        DOUBLE,
    short_iv          DOUBLE,
    long_iv           DOUBLE,
    short_gamma       DOUBLE,
    short_theta       DOUBLE,
    short_vega        DOUBLE,
    short_oi          INTEGER,
    long_oi           INTEGER,
    short_volume      INTEGER,

    gamma_regime      TEXT,
    gamma_flip        DOUBLE,
    call_wall         DOUBLE,
    put_wall          DOUBLE,
    magnet_strike     DOUBLE,
    pin_score         DOUBLE,
    total_gex_b       DOUBLE,
    zero_dte_gex_share DOUBLE,

    vix_1d            DOUBLE,
    cboe_skew         DOUBLE,
    rr25              DOUBLE,

    active_patterns_l1  TEXT,
    active_patterns_l2  TEXT,
    patterns_score_bonus INTEGER,

    score             INTEGER NOT NULL,
    score_breakdown_json TEXT NOT NULL,
    notes             TEXT,

    status            TEXT NOT NULL DEFAULT 'OPEN'
);

CREATE INDEX IF NOT EXISTS idx_trades_entry_date ON paper_trades(entry_date);
CREATE INDEX IF NOT EXISTS idx_trades_status     ON paper_trades(status);
CREATE INDEX IF NOT EXISTS idx_trades_source     ON paper_trades(source);

CREATE TABLE IF NOT EXISTS paper_ticks (
    trade_id         TEXT NOT NULL,
    ts               TIMESTAMP NOT NULL,
    spot             DOUBLE,
    short_bid        DOUBLE, short_ask DOUBLE,
    long_bid         DOUBLE, long_ask  DOUBLE,
    exit_cost_bid    DOUBLE, exit_cost_mid DOUBLE,
    pnl_bid          DOUBLE, pnl_mid DOUBLE,
    pct_of_max_profit DOUBLE,
    status           TEXT,
    PRIMARY KEY (trade_id, ts)
);

CREATE INDEX IF NOT EXISTS idx_ticks_trade ON paper_ticks(trade_id);

CREATE TABLE IF NOT EXISTS paper_outcomes (
    trade_id         TEXT PRIMARY KEY,
    exit_ts          TIMESTAMP NOT NULL,
    exit_status      TEXT NOT NULL,
    exit_spot        DOUBLE,
    exit_cost_bid    DOUBLE,
    exit_cost_mid    DOUBLE,
    pnl_bid          DOUBLE,
    pnl_mid          DOUBLE,
    held_minutes     INTEGER,
    max_excursion_pct DOUBLE,
    min_excursion_pct DOUBLE,
    exit_reason      TEXT
);

CREATE TABLE IF NOT EXISTS replay_cursor (
    chunk_id         INTEGER PRIMARY KEY,
    start_date       DATE NOT NULL,
    end_date         DATE NOT NULL,
    completed_at     TIMESTAMP
);

CREATE TABLE IF NOT EXISTS scanner_heartbeat (
    component        TEXT PRIMARY KEY,
    last_run_at      TIMESTAMP NOT NULL,
    last_status      TEXT NOT NULL
);
"""


def trade_id_for(
    *,
    entry_date: date,
    entry_minute: int,
    short_strike: float,
    long_strike: float,
    strategy: StrategyType,
    source: str,
) -> str:
    """Deterministic trade id.

    Replay-safe (same inputs → same id). Restart-safe (won't double-log).
    """
    key = f"{entry_date.isoformat()}|{entry_minute}|{short_strike:.2f}|{long_strike:.2f}|{strategy}|{source}"
    return hashlib.sha256(key.encode()).hexdigest()[:24]


class Journal:
    """Single-writer DuckDB wrapper.

    The DuckDB Python client is not multi-thread safe for the same connection.
    We serialize writes with a process-level lock; reads can reopen briefly.
    """

    def __init__(self, db_path: str | None = None) -> None:
        path = db_path or settings.duckdb_path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._lock = threading.Lock()
        self._conn = duckdb.connect(self._path)
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.execute(SCHEMA_SQL)
            row = self._conn.execute("SELECT schema_version FROM schema_meta LIMIT 1").fetchone()
            if row is None:
                self._conn.execute(
                    "INSERT INTO schema_meta(schema_version, applied_at) VALUES (?, ?)",
                    (SCHEMA_VERSION, datetime.now(UTC)),
                )

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @contextmanager
    def transaction(self) -> Iterator[duckdb.DuckDBPyConnection]:
        with self._lock:
            self._conn.execute("BEGIN")
            try:
                yield self._conn
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    # ----------------------------------------------------------------- writes
    def write_trade(self, t: PaperTrade) -> bool:
        """Insert a paper trade. Returns True if newly inserted, False if duplicate."""
        entry_date = t.entry_ts.date()
        entry_minute = t.entry_ts.hour * 60 + t.entry_ts.minute

        with self._lock:
            existing = self._conn.execute(
                "SELECT 1 FROM paper_trades WHERE trade_id = ?", [t.trade_id]
            ).fetchone()
            if existing is not None:
                return False

            self._conn.execute(
                """
                INSERT INTO paper_trades VALUES (
                    ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?,
                    'OPEN'
                )
                """,
                [
                    t.trade_id, str(t.strategy), t.entry_ts, entry_date, entry_minute,
                    t.expiry, t.source,
                    t.spot_at_entry, t.short_strike, t.long_strike,
                    t.short_contract, t.long_contract, t.width, t.contracts,
                    t.credit_mid, t.credit_bid, t.max_loss_bid, t.credit_bid_ratio,
                    t.short_delta, t.long_delta, t.short_iv, t.long_iv,
                    t.short_gamma, t.short_theta, t.short_vega,
                    t.short_oi, t.long_oi, t.short_volume,
                    t.gamma_regime, t.gamma_flip, t.call_wall, t.put_wall,
                    t.magnet_strike, t.pin_score, t.total_gex_b, t.zero_dte_gex_share,
                    t.vix_1d, t.cboe_skew, t.rr25,
                    t.active_patterns_l1, t.active_patterns_l2, t.patterns_score_bonus,
                    t.score, t.score_breakdown_json, t.notes,
                ],
            )
            return True

    def write_tick(self, tick: TradeTick) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO paper_ticks VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                [
                    tick.trade_id, tick.ts, tick.spot,
                    tick.short_bid, tick.short_ask, tick.long_bid, tick.long_ask,
                    tick.exit_cost_bid, tick.exit_cost_mid,
                    tick.pnl_bid, tick.pnl_mid, tick.pct_of_max_profit,
                    str(tick.status),
                ],
            )

    def write_outcome(self, o: TradeOutcome) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO paper_outcomes VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                [
                    o.trade_id, o.exit_ts, str(o.exit_status), o.exit_spot,
                    o.exit_cost_bid, o.exit_cost_mid, o.pnl_bid, o.pnl_mid,
                    o.held_minutes, o.max_excursion_pct, o.min_excursion_pct, o.exit_reason,
                ],
            )
            self._conn.execute(
                "UPDATE paper_trades SET status = ? WHERE trade_id = ?",
                [str(o.exit_status), o.trade_id],
            )

    def heartbeat(self, component: str, status: str = "ok") -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO scanner_heartbeat(component, last_run_at, last_status)
                VALUES (?, ?, ?)
                """,
                [component, datetime.now(UTC), status],
            )

    # ----------------------------------------------------------------- reads
    def open_positions(self) -> list[dict[str, Any]]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM paper_trades WHERE status IN ('OPEN','WINNING','LOSING') ORDER BY entry_ts"
            )
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]

    def count_today(self, today: date, source: str = "live") -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM paper_trades WHERE entry_date = ? AND source = ?",
                [today, source],
            ).fetchone()
            return int(row[0]) if row else 0

    def replay_chunks_done(self) -> list[tuple[date, date]]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT start_date, end_date FROM replay_cursor WHERE completed_at IS NOT NULL ORDER BY chunk_id"
            )
            return [(r[0], r[1]) for r in cur.fetchall()]

    def mark_chunk_done(self, chunk_id: int, start: date, end: date) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO replay_cursor(chunk_id, start_date, end_date, completed_at)
                VALUES (?, ?, ?, ?)
                """,
                [chunk_id, start, end, datetime.now(UTC)],
            )
