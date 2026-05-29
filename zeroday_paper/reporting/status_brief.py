"""On-demand status brief.

Renders a one-shot snapshot of the paper trading system: today's entries,
realized + unrealized P&L, open positions with live marks, last scanner cycle,
pattern hits, and risk gauges. Stateless — does no I/O beyond reading the
DuckDB journal in **read-only** mode (a hard requirement so we never compete
for the write lock with the live scanner — that lock fight has manifested as
``Conflicting lock is held in PID 0`` errors in the past).

The renderer is split into:

    StatusBundle    — frozen dataclass with everything needed to render
    build_bundle()  — read-only DuckDB queries + optional diag snapshot
    render_brief()  — pure Markdown formatter; never raises
    post_to_discord — chunked HTTP post; mirrors daily_report

Designed so an empty journal, weekend, or pre-market clock all render cleanly
with explicit context rather than blanks.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any
from zoneinfo import ZoneInfo

import duckdb
import httpx
import structlog

from zeroday_paper.config import settings
from zeroday_paper.reporting.daily_report import _chunk_markdown
from zeroday_paper.secrets import discord_webhook

logger = structlog.get_logger(__name__)

ET = ZoneInfo(settings.engine.market_timezone)
UNAVAILABLE = "[UNAVAILABLE]"

# Rough Fargate ARM64 0.5vCPU + 1GB pricing (us-east-1, 2026 list price).
# Used only for the footer cost line so operators have a sense of how cheap
# these ad-hoc runs are. Off by a fraction of a cent either way is fine.
FARGATE_VCPU_PER_HOUR_USD = 0.04048
FARGATE_GB_PER_HOUR_USD = 0.004445


# --------------------------------------------------------------------------- types


@dataclass(frozen=True)
class TodaySummary:
    session_date: date
    is_today: bool                # False when we fell back to last completed session
    entries: int                  # live source only
    closed: int
    wins: int
    losses: int
    realized_pnl_bid: float
    open_count: int


@dataclass(frozen=True)
class OpenPositionRow:
    trade_id_short: str
    strategy: str
    short_strike: float
    long_strike: float
    entry_credit_bid: float
    current_mark_bid: float | None
    unrealized_pnl_bid: float | None
    distance_from_short: float | None  # spot - short_strike (signed)
    age_minutes: int


@dataclass(frozen=True)
class LastCycleSection:
    cycle_ts: datetime | None
    spot: float | None
    regime: str | None
    monitored: int | None
    exited: int | None
    score_bull_put: int | None
    score_bear_call: int | None
    source: str                   # "diag_snapshot" | "heartbeat_only" | "unavailable"


@dataclass(frozen=True)
class PatternHits:
    l1_total: int
    l2_total: int
    top_l1: tuple[tuple[str, int], ...]   # [(pattern_id, count)]
    top_l2: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class RiskGauges:
    today_total_live: int
    cap: int
    seconds_to_session_end: int | None    # None when not in session
    session_state_label: str              # e.g. "session closed", "weekend"
    vix_1d: float | None


@dataclass(frozen=True)
class StatusBundle:
    asof_utc: datetime
    asof_et: datetime
    revision: str | None
    task_id: str | None
    market_state: str                     # OPEN / PRE_MARKET / AFTER_HOURS / WEEKEND / HOLIDAY
    today: TodaySummary
    open_positions: tuple[OpenPositionRow, ...]
    last_cycle: LastCycleSection | None
    patterns: PatternHits
    risk: RiskGauges
    log_group: str
    runtime_seconds: float | None
    runtime_cost_usd: float | None
    journal_path: str
    errors: tuple[str, ...] = field(default_factory=tuple)


# --------------------------------------------------------------------------- IO helpers


@contextmanager
def _ro_conn(db_path: str) -> Iterator[duckdb.DuckDBPyConnection]:
    """Open a read-only DuckDB connection.

    The status path is the *only* code that reads the production journal while
    the live scanner holds the write lock. ``read_only=True`` lets DuckDB
    open the database file without contending for the writer lock — this is
    the explicit fix for the ``Conflicting lock is held in PID 0`` errors we
    observed mid-week when two writer tasks raced.
    """
    conn = duckdb.connect(db_path, read_only=True)
    try:
        yield conn
    finally:
        conn.close()


def _now_utc() -> datetime:
    return datetime.now(UTC)


def _today_et(now_utc: datetime | None = None) -> date:
    return (now_utc or _now_utc()).astimezone(ET).date()


def _is_weekend(d: date) -> bool:
    return d.weekday() >= 5


# --------------------------------------------------------------------------- market clock


def _market_state(now_utc: datetime) -> str:
    """Return one of OPEN / PRE_MARKET / AFTER_HOURS / WEEKEND.

    Note: we don't enumerate holidays here; the morning brief module owns the
    holiday list. The status post is intentionally lightweight — if today is a
    holiday the journal will simply show no entries and the operator can read
    the brief separately.
    """
    et = now_utc.astimezone(ET)
    if _is_weekend(et.date()):
        return "WEEKEND"
    t = et.time()
    if t < settings.engine.session_start:
        return "PRE_MARKET"
    if t > settings.engine.session_end:
        return "AFTER_HOURS"
    return "OPEN"


def _seconds_to_session_end(now_utc: datetime) -> int | None:
    et = now_utc.astimezone(ET)
    if _is_weekend(et.date()):
        return None
    end_dt = datetime.combine(et.date(), settings.engine.session_end, tzinfo=ET)
    delta = end_dt - et
    if delta.total_seconds() <= 0:
        return None
    return int(delta.total_seconds())


def _seconds_to_session_start(now_utc: datetime) -> int | None:
    et = now_utc.astimezone(ET)
    if _is_weekend(et.date()):
        return None
    start_dt = datetime.combine(et.date(), settings.engine.session_start, tzinfo=ET)
    delta = start_dt - et
    if delta.total_seconds() <= 0:
        return None
    return int(delta.total_seconds())


# --------------------------------------------------------------------------- journal queries


def _last_session_date(conn: duckdb.DuckDBPyConnection, on_or_before: date) -> date | None:
    """Most recent ``entry_date`` (live source) on or before ``on_or_before``."""
    row = conn.execute(
        """
        SELECT MAX(entry_date)
        FROM paper_trades
        WHERE source = 'live' AND entry_date <= ?
        """,
        [on_or_before],
    ).fetchone()
    return row[0] if row and row[0] else None


def _today_summary(
    conn: duckdb.DuckDBPyConnection,
    *,
    today_et: date,
    market_state: str,
) -> TodaySummary:
    """Build TodaySummary, falling back to last completed session when off-hours.

    Rules:
        - During OPEN or PRE_MARKET: ``today_et`` is the session date.
        - During AFTER_HOURS: still ``today_et`` (the day's session ran).
        - During WEEKEND: walk back to most recent live trade date.
    """
    if market_state == "WEEKEND":
        session_date = _last_session_date(conn, today_et) or today_et
        is_today = session_date == today_et
    else:
        session_date = today_et
        is_today = True

    row = conn.execute(
        """
        SELECT
            COUNT(*) AS entries,
            COUNT(CASE WHEN status LIKE 'CLOSED%' THEN 1 END) AS closed,
            COUNT(CASE WHEN status = 'CLOSED_TARGET' THEN 1 END) AS wins,
            COUNT(CASE WHEN status IN ('CLOSED_STOP','CLOSED_THESIS','CLOSED_HARD_CLOSE') THEN 1 END) AS losses,
            COUNT(CASE WHEN status IN ('OPEN','WINNING','LOSING') THEN 1 END) AS open_count
        FROM paper_trades
        WHERE entry_date = ? AND source = 'live'
        """,
        [session_date],
    ).fetchone()
    entries = int(row[0]) if row else 0
    closed = int(row[1]) if row else 0
    wins = int(row[2]) if row else 0
    losses = int(row[3]) if row else 0
    open_count = int(row[4]) if row else 0

    pnl_row = conn.execute(
        """
        SELECT COALESCE(SUM(o.pnl_bid), 0)
        FROM paper_outcomes o
        JOIN paper_trades t USING (trade_id)
        WHERE t.entry_date = ? AND t.source = 'live'
        """,
        [session_date],
    ).fetchone()
    realized = float(pnl_row[0]) if pnl_row else 0.0

    return TodaySummary(
        session_date=session_date,
        is_today=is_today,
        entries=entries,
        closed=closed,
        wins=wins,
        losses=losses,
        realized_pnl_bid=round(realized, 2),
        open_count=open_count,
    )


def _open_positions(
    conn: duckdb.DuckDBPyConnection,
    *,
    now_utc: datetime,
) -> list[OpenPositionRow]:
    """List open positions with the latest tick joined for mark + unrealized P&L."""
    rows = conn.execute(
        """
        WITH latest AS (
            SELECT
                tk.trade_id,
                tk.exit_cost_bid,
                tk.pnl_bid,
                tk.spot,
                tk.ts,
                ROW_NUMBER() OVER (PARTITION BY tk.trade_id ORDER BY tk.ts DESC) AS rn
            FROM paper_ticks tk
        )
        SELECT
            t.trade_id,
            t.strategy,
            t.short_strike,
            t.long_strike,
            t.credit_bid,
            t.entry_ts,
            l.exit_cost_bid,
            l.pnl_bid,
            l.spot
        FROM paper_trades t
        LEFT JOIN latest l ON l.trade_id = t.trade_id AND l.rn = 1
        WHERE t.status IN ('OPEN', 'WINNING', 'LOSING')
        ORDER BY t.entry_ts
        """,
    ).fetchall()

    out: list[OpenPositionRow] = []
    for r in rows:
        trade_id = r[0]
        strategy = str(r[1])
        short_k = float(r[2])
        long_k = float(r[3])
        credit_bid = float(r[4])
        entry_ts = r[5]
        exit_cost_bid = r[6]
        pnl_bid = r[7]
        spot = r[8]

        if entry_ts is None:
            age_minutes = 0
        else:
            # DuckDB returns naive timestamps from a TIMESTAMP column; we wrote
            # them as UTC-aware. Treat them as UTC for the age computation.
            entry_ts_utc = entry_ts.replace(tzinfo=UTC) if entry_ts.tzinfo is None else entry_ts
            age_minutes = max(0, int((now_utc - entry_ts_utc).total_seconds() / 60))

        distance = None
        if spot is not None:
            distance = round(float(spot) - short_k, 2)

        out.append(OpenPositionRow(
            trade_id_short=trade_id[:8],
            strategy=_short_strategy(strategy),
            short_strike=short_k,
            long_strike=long_k,
            entry_credit_bid=round(credit_bid, 2),
            current_mark_bid=round(float(exit_cost_bid), 2) if exit_cost_bid is not None else None,
            unrealized_pnl_bid=round(float(pnl_bid), 2) if pnl_bid is not None else None,
            distance_from_short=distance,
            age_minutes=age_minutes,
        ))
    return out


def _pattern_hits_for_session(
    conn: duckdb.DuckDBPyConnection,
    session_date: date,
) -> PatternHits:
    """Aggregate active_patterns_l1 / _l2 across the session's live entries."""
    rows = conn.execute(
        """
        SELECT active_patterns_l1, active_patterns_l2
        FROM paper_trades
        WHERE entry_date = ? AND source = 'live'
        """,
        [session_date],
    ).fetchall()

    l1_counts: dict[str, int] = {}
    l2_counts: dict[str, int] = {}
    for row in rows:
        for pid in _split_pattern_field(row[0]):
            l1_counts[pid] = l1_counts.get(pid, 0) + 1
        for pid in _split_pattern_field(row[1]):
            l2_counts[pid] = l2_counts.get(pid, 0) + 1

    return PatternHits(
        l1_total=sum(l1_counts.values()),
        l2_total=sum(l2_counts.values()),
        top_l1=tuple(sorted(l1_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:5]),
        top_l2=tuple(sorted(l2_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:5]),
    )


def _split_pattern_field(value: Any) -> list[str]:
    if not value:
        return []
    return [p.strip() for p in str(value).split(",") if p.strip()]


def _heartbeat_cycle(
    conn: duckdb.DuckDBPyConnection,
) -> tuple[datetime | None, str | None]:
    """Return (last_run_at, last_status) for the scanner heartbeat row.

    This is the fallback when an on-demand diagnostic_snapshot fails or is
    skipped. It at least proves the scanner has been writing, which is the
    one thing we always want visible in the status post.
    """
    row = conn.execute(
        """
        SELECT last_run_at, last_status
        FROM scanner_heartbeat
        WHERE component = 'scanner'
        """,
    ).fetchone()
    if row is None:
        return None, None
    ts = row[0]
    if ts is not None and ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    return ts, str(row[1]) if row[1] else None


def _vix_1d_from_journal(conn: duckdb.DuckDBPyConnection) -> float | None:
    """Pull the most recently logged VIX1D off an open or recent trade.

    The status path doesn't fetch CBOE directly to keep latency low; the most
    recent paper_trade or tick has VIX1D recorded by the live scanner, so we
    surface that as a "last seen" value. It's a snapshot, not real-time.
    """
    row = conn.execute(
        """
        SELECT vix_1d
        FROM paper_trades
        WHERE vix_1d IS NOT NULL
        ORDER BY entry_ts DESC
        LIMIT 1
        """,
    ).fetchone()
    if row is None or row[0] is None:
        return None
    return float(row[0])


# --------------------------------------------------------------------------- diag snapshot adapter


def _last_cycle_from_diag(diag: dict[str, Any]) -> LastCycleSection:
    """Map the ``diagnostic_snapshot()`` dict into a LastCycleSection.

    The diag dict shape is fixed by ``zeroday_paper.engine.scanner``. We pull
    only the few fields we actually render so renderer stays stable even if
    diagnostic_snapshot grows new keys.
    """
    signals = diag.get("signals") or {}
    score_bp = score_bc = None
    for s in diag.get("strategies", []) or []:
        if s.get("strategy") == "BULL_PUT_SPREAD":
            score_bp = int(s.get("score")) if s.get("score") is not None else None
        elif s.get("strategy") == "BEAR_CALL_SPREAD":
            score_bc = int(s.get("score")) if s.get("score") is not None else None

    asof_raw = diag.get("asof_utc")
    asof_dt: datetime | None
    if isinstance(asof_raw, str):
        try:
            asof_dt = datetime.fromisoformat(asof_raw)
        except ValueError:
            asof_dt = None
    elif isinstance(asof_raw, datetime):
        asof_dt = asof_raw
    else:
        asof_dt = None

    return LastCycleSection(
        cycle_ts=asof_dt,
        spot=float(signals.get("spot")) if signals.get("spot") is not None else None,
        regime=signals.get("regime"),
        monitored=None,           # diag snapshot doesn't run monitor
        exited=None,
        score_bull_put=score_bp,
        score_bear_call=score_bc,
        source="diag_snapshot",
    )


def _last_cycle_from_heartbeat(
    cycle_ts: datetime | None, status: str | None
) -> LastCycleSection:
    return LastCycleSection(
        cycle_ts=cycle_ts,
        spot=None,
        regime=status,
        monitored=None,
        exited=None,
        score_bull_put=None,
        score_bear_call=None,
        source="heartbeat_only",
    )


# --------------------------------------------------------------------------- public API


def build_bundle(
    *,
    journal_path: str | None = None,
    diag_snapshot: dict[str, Any] | None = None,
    revision: str | None = None,
    task_id: str | None = None,
    runtime_seconds: float | None = None,
    now_utc: datetime | None = None,
) -> StatusBundle:
    """Read the journal and assemble a StatusBundle.

    Pure read path. The optional ``diag_snapshot`` arg lets the CLI pass in a
    one-shot ``diagnostic_snapshot()`` result without this module needing to
    know about Polygon/CBOE clients. If absent, we fall back to the heartbeat
    row in the journal.
    """
    now = now_utc or _now_utc()
    asof_et = now.astimezone(ET)
    today_et = asof_et.date()
    market_state = _market_state(now)
    db_path = journal_path or settings.duckdb_path
    errors: list[str] = []

    try:
        with _ro_conn(db_path) as conn:
            today = _today_summary(conn, today_et=today_et, market_state=market_state)
            positions = tuple(_open_positions(conn, now_utc=now))
            patterns = _pattern_hits_for_session(conn, today.session_date)
            heartbeat_ts, heartbeat_status = _heartbeat_cycle(conn)
            vix_1d = _vix_1d_from_journal(conn)
            today_total_live = today.entries
    except Exception as exc:        # pragma: no cover — defensive
        logger.warning("status.journal_read_failed", error=str(exc))
        errors.append(f"journal_read_failed:{exc}")
        today = TodaySummary(
            session_date=today_et, is_today=True,
            entries=0, closed=0, wins=0, losses=0,
            realized_pnl_bid=0.0, open_count=0,
        )
        positions = ()
        patterns = PatternHits(0, 0, (), ())
        heartbeat_ts, heartbeat_status = None, None
        vix_1d = None
        today_total_live = 0

    if diag_snapshot:
        last_cycle: LastCycleSection | None = _last_cycle_from_diag(diag_snapshot)
    elif heartbeat_ts is not None:
        last_cycle = _last_cycle_from_heartbeat(heartbeat_ts, heartbeat_status)
    else:
        last_cycle = None

    seconds_to_end = _seconds_to_session_end(now)
    if market_state == "WEEKEND":
        session_label = "weekend"
    elif market_state == "AFTER_HOURS":
        session_label = "session closed"
    elif market_state == "PRE_MARKET":
        session_label = "pre-market"
    else:
        session_label = "in session"

    risk = RiskGauges(
        today_total_live=today_total_live,
        cap=settings.concurrency.max_concurrent_total,
        seconds_to_session_end=seconds_to_end,
        session_state_label=session_label,
        vix_1d=vix_1d,
    )

    cost = _estimate_runtime_cost(runtime_seconds)

    return StatusBundle(
        asof_utc=now,
        asof_et=asof_et,
        revision=revision,
        task_id=task_id,
        market_state=market_state,
        today=today,
        open_positions=positions,
        last_cycle=last_cycle,
        patterns=patterns,
        risk=risk,
        log_group="/zeroday-paper/tasks",
        runtime_seconds=runtime_seconds,
        runtime_cost_usd=cost,
        journal_path=db_path,
        errors=tuple(errors),
    )


def _estimate_runtime_cost(runtime_seconds: float | None) -> float | None:
    """Back-of-envelope Fargate cost for the task definition's 0.5 vCPU / 1GB shape."""
    if runtime_seconds is None or runtime_seconds <= 0:
        return None
    hours = runtime_seconds / 3600.0
    vcpu = 0.5
    mem_gb = 1.0
    return round(hours * (vcpu * FARGATE_VCPU_PER_HOUR_USD + mem_gb * FARGATE_GB_PER_HOUR_USD), 5)


# --------------------------------------------------------------------------- rendering


def render_brief(bundle: StatusBundle) -> str:
    """Render the StatusBundle as Discord-friendly markdown. Never raises."""
    lines: list[str] = []
    et_ts = bundle.asof_et.strftime("%a %Y-%m-%d %H:%M:%S %Z")
    revision = bundle.revision or "unknown"
    task = bundle.task_id or "ad-hoc"
    state = _state_pretty(bundle.market_state)

    lines.append(f"# Paper status — {et_ts}")
    lines.append(f"_rev `{revision}` · task `{task[:12]}` · {state}_")
    lines.append("")

    lines.append(_render_today(bundle))
    lines.append("")

    lines.append(_render_open_positions(bundle))
    lines.append("")

    lines.append(_render_last_cycle(bundle))
    lines.append("")

    lines.append(_render_patterns(bundle))
    lines.append("")

    lines.append(_render_risk(bundle))
    lines.append("")

    lines.append(_render_footer(bundle))
    return "\n".join(lines).rstrip() + "\n"


def _state_pretty(state: str) -> str:
    return {
        "OPEN": "session OPEN",
        "PRE_MARKET": "pre-market",
        "AFTER_HOURS": "after-hours",
        "WEEKEND": "weekend",
        "HOLIDAY": "holiday",
    }.get(state, state)


def _render_today(bundle: StatusBundle) -> str:
    t = bundle.today
    if t.entries == 0 and t.open_count == 0:
        if bundle.market_state == "PRE_MARKET":
            wait = _seconds_to_session_start(bundle.asof_utc)
            if wait is not None:
                return (
                    f"## Today ({t.session_date.isoformat()})\n"
                    f"- session opens at 09:30 ET in {_fmt_duration(wait)}\n"
                    f"- entries today: 0\n"
                    f"- no open positions"
                )
        if bundle.market_state == "WEEKEND":
            return (
                f"## Last session ({t.session_date.isoformat()})\n"
                "- no trades yet"
                if not t.is_today
                else "## Today\n- no trades yet"
            )
        return f"## Today ({t.session_date.isoformat()})\n- no trades yet"

    header = (
        f"## Today ({t.session_date.isoformat()})"
        if t.is_today
        else f"## Last session ({t.session_date.isoformat()})"
    )
    return (
        f"{header}\n"
        f"- Entries: **{t.entries}**\n"
        f"- Closed: **{t.closed}** (W {t.wins} / L {t.losses})\n"
        f"- Realized P&L (bid): **${t.realized_pnl_bid:,.2f}**\n"
        f"- Open positions: **{t.open_count}**"
    )


def _render_open_positions(bundle: StatusBundle) -> str:
    if not bundle.open_positions:
        return "## Open positions\n_None._"
    lines = ["## Open positions",
             "| id | strategy | short / long | credit | mark | unrl P&L | dist | age |",
             "|---|---|---|---|---|---|---|---|"]
    for p in bundle.open_positions:
        strikes = f"{p.short_strike:.0f} / {p.long_strike:.0f}"
        mark = f"${p.current_mark_bid:.2f}" if p.current_mark_bid is not None else "—"
        upnl = f"${p.unrealized_pnl_bid:+,.2f}" if p.unrealized_pnl_bid is not None else "—"
        dist = f"{p.distance_from_short:+.1f}" if p.distance_from_short is not None else "—"
        lines.append(
            f"| `{p.trade_id_short}` | {p.strategy} | {strikes} | "
            f"${p.entry_credit_bid:.2f} | {mark} | {upnl} | {dist} | {p.age_minutes}m |"
        )
    return "\n".join(lines)


def _render_last_cycle(bundle: StatusBundle) -> str:
    lc = bundle.last_cycle
    if lc is None:
        return "## Last scanner cycle\n- _no heartbeat / diag data_"
    if lc.cycle_ts is None:
        return "## Last scanner cycle\n- _no cycle timestamp_"

    age_s = max(0, int((bundle.asof_utc - lc.cycle_ts).total_seconds()))
    age_str = _fmt_duration(age_s)
    src = "diag snapshot" if lc.source == "diag_snapshot" else "scanner heartbeat"

    bits = [
        f"## Last scanner cycle ({src})",
        f"- ts: {lc.cycle_ts.astimezone(ET).strftime('%H:%M:%S ET')} ({age_str} ago)",
    ]
    if lc.spot is not None:
        bits.append(f"- spot: **{lc.spot:.2f}**")
    if lc.regime is not None:
        bits.append(f"- regime / status: **{lc.regime}**")
    if lc.score_bull_put is not None or lc.score_bear_call is not None:
        bp = "—" if lc.score_bull_put is None else str(lc.score_bull_put)
        bc = "—" if lc.score_bear_call is None else str(lc.score_bear_call)
        bits.append(
            f"- scan.score: BULL_PUT **{bp}**, BEAR_CALL **{bc}** "
            f"(threshold {settings.engine.score_threshold})"
        )
    return "\n".join(bits)


def _render_patterns(bundle: StatusBundle) -> str:
    p = bundle.patterns
    if p.l1_total == 0 and p.l2_total == 0:
        return "## Pattern hits today\n- _none recorded on today's entries_"
    lines = [
        "## Pattern hits today",
        f"- L1 hits: **{p.l1_total}** / L2 hits: **{p.l2_total}**",
    ]
    if p.top_l1:
        lines.append("- top L1: " + ", ".join(f"`{k}` x{v}" for k, v in p.top_l1))
    if p.top_l2:
        lines.append("- top L2: " + ", ".join(f"`{k}` x{v}" for k, v in p.top_l2))
    return "\n".join(lines)


def _render_risk(bundle: StatusBundle) -> str:
    r = bundle.risk
    if r.seconds_to_session_end is not None:
        clock = f"{_fmt_duration(r.seconds_to_session_end)} to session end"
    else:
        clock = r.session_state_label
    vix = f"{r.vix_1d:.2f}" if r.vix_1d is not None else UNAVAILABLE
    return (
        "## Risk gauges\n"
        f"- today_total: **{r.today_total_live} / {r.cap}**\n"
        f"- clock: **{clock}**\n"
        f"- VIX1D (last journaled): **{vix}**"
    )


def _render_footer(bundle: StatusBundle) -> str:
    cost = (
        f"~${bundle.runtime_cost_usd:.4f}"
        if bundle.runtime_cost_usd is not None
        else "n/a"
    )
    rt = (
        f"{bundle.runtime_seconds:.1f}s"
        if bundle.runtime_seconds is not None
        else "n/a"
    )
    bits = [
        "---",
        f"_logs: `{bundle.log_group}` · journal: `{bundle.journal_path}` · runtime: {rt} (est. {cost})_",
    ]
    if bundle.errors:
        bits.append("_errors: " + "; ".join(bundle.errors) + "_")
    return "\n".join(bits)


def _fmt_duration(seconds: int) -> str:
    if seconds < 60:
        return f"{seconds}s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {secs}s" if secs else f"{minutes}m"
    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        return f"{hours}h {minutes}m"
    days, hours = divmod(hours, 24)
    return f"{days}d {hours}h"


def _short_strategy(strategy: str) -> str:
    return {
        "BULL_PUT_SPREAD": "BULL_PUT",
        "BEAR_CALL_SPREAD": "BEAR_CALL",
        "IRON_CONDOR": "IC",
    }.get(strategy, strategy)


# --------------------------------------------------------------------------- Discord post


def post_to_discord(markdown_text: str, *, webhook_url: str | None = None) -> bool:
    """Post status markdown to the status webhook. Chunks at 1900 chars.

    Mirrors ``daily_report.post_to_discord`` so the chunking rule + error
    handling stay consistent across all three webhook surfaces.
    """
    url = webhook_url or _resolve_webhook()
    if not url:
        logger.warning("status.no_webhook")
        return False
    chunks = _chunk_markdown(markdown_text, max_chars=1900)
    ok = True
    for chunk in chunks:
        try:
            r = httpx.post(url, json={"content": chunk}, timeout=10.0)
            if r.status_code not in (200, 204):
                logger.warning("status.discord_status", status=r.status_code, body=r.text[:200])
                ok = False
        except Exception as exc:
            logger.warning("status.discord_error", error=str(exc))
            ok = False
    return ok


def _resolve_webhook() -> str | None:
    try:
        return discord_webhook(settings.reporting.status_discord_webhook_secret_key)
    except Exception as exc:
        logger.warning("status.webhook_lookup_failed", error=str(exc))
        return None
