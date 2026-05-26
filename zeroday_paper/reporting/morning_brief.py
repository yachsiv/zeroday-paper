"""Pre-market morning brief composer.

Fires at 08:00 ET on weekdays (EventBridge → ECS task, ``MODE=morning``). Pulls
a comprehensive market snapshot in parallel via ``asyncio.gather``, renders a
single Markdown brief, and posts to the ``webhook_morning_brief`` Discord
channel. Any source failure is logged + marked ``[UNAVAILABLE]`` — the brief
itself never raises.

Sections, in render order:
    1. Today's session metadata     (0DTE / holiday flag)
    2. Volatility context           (CBOE: VIX1D, SKEW + gate verdict)
    3. Yesterday's key levels       (Polygon chain → signals_from_chain)
    4. ES futures overnight         (Polygon SPY proxy → implied SPX open)
    5. Expected 1SD daily move      (Polygon ATM 0DTE straddle * 0.85)
    6. Economic calendar            (Perplexity → Tier-1 SKIP flag)
    7. Earnings                     (Perplexity top S&P 500)
    8. Global markets sentiment     (Perplexity Asia/Europe)
    9. Recommended posture          (rule-based on above)

Discord limit is 2000 chars/post but markdown formatting often pushes us
past. We chunk at 1900 chars on line boundaries via the existing helper.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import httpx
import structlog

from zeroday_paper.config import settings
from zeroday_paper.data.cboe_client import CboeClient, VolSnapshot
from zeroday_paper.data.flashalpha_client import MarketSignals, signals_from_chain
from zeroday_paper.data.perplexity_client import PerplexityClient
from zeroday_paper.data.polygon_client import (
    ChainSnapshot,
    PolygonClient,
    next_spx_expiry,
)
from zeroday_paper.engine.replay import US_HOLIDAYS
from zeroday_paper.reporting.daily_report import _chunk_markdown
from zeroday_paper.secrets import discord_webhook

logger = structlog.get_logger(__name__)

ET = ZoneInfo(settings.engine.market_timezone)

UNAVAILABLE = "[UNAVAILABLE]"

# ATM 0DTE straddle multiplier. The textbook 1SD daily move from a 0DTE
# straddle is roughly 0.85 * (call_mid + put_mid). Source: Sinclair, Option
# Volatility Trading; corroborated by SpotGamma's daily-expected-move docs.
ATM_STRADDLE_DAILY_SD_MULTIPLIER = 0.85

# Polygon-side SPY → SPX implied open conversion. Uses yesterday's
# `prev_spx_close ≈ prev_spy_close * 10` as a fallback if no basis available.
SPY_SPX_RATIO = 10.0

# Window in ET for Tier-1 events that trigger the SKIP flag (09:30-14:00).
TIER_ONE_WINDOW_START = time(9, 30)
TIER_ONE_WINDOW_END = time(14, 0)


# --------------------------------------------------------------------------- types


@dataclass(frozen=True)
class SessionMeta:
    today_et: date
    is_0dte_day: bool
    is_holiday: bool


@dataclass(frozen=True)
class VolContext:
    vols: VolSnapshot | None
    would_skip: bool
    skip_reasons: tuple[str, ...]


@dataclass(frozen=True)
class LevelsSection:
    signals: MarketSignals | None
    spot_used: float | None


@dataclass(frozen=True)
class OvernightSection:
    source: str                       # "ES" | "SPY" | "unavailable"
    overnight_high: float | None
    overnight_low: float | None
    overnight_last: float | None
    change_pts: float | None
    change_pct: float | None
    implied_spx_open: float | None
    implied_spx_change_pct: float | None
    prev_spx_close: float | None


@dataclass(frozen=True)
class ExpectedMoveSection:
    atm_strike: float | None
    atm_call_mid: float | None
    atm_put_mid: float | None
    expected_move_pts: float | None
    expected_move_pct: float | None
    range_low: float | None
    range_high: float | None
    spot: float | None


@dataclass(frozen=True)
class EconomicEvent:
    name: str
    time_et: str
    is_tier_one: bool


@dataclass(frozen=True)
class CalendarSection:
    events: tuple[EconomicEvent, ...]
    skip_today: bool
    skip_reason: str | None


@dataclass(frozen=True)
class EarningsSection:
    items: tuple[str, ...]                # rendered "TICKER (pre-open|after-close)" strings


@dataclass(frozen=True)
class GlobalSection:
    nikkei: str | None
    hang_seng: str | None
    dax: str | None
    ftse: str | None
    sentiment: str | None                 # "Risk On" | "Risk Off" | "Neutral"
    notes: str | None                     # one-sentence summary


@dataclass(frozen=True)
class BriefBundle:
    """All sections, populated or marked None on failure."""
    meta: SessionMeta
    vol: VolContext | None
    levels: LevelsSection | None
    overnight: OvernightSection | None
    expected_move: ExpectedMoveSection | None
    calendar: CalendarSection | None
    earnings: EarningsSection | None
    global_markets: GlobalSection | None
    failures: dict[str, str] = field(default_factory=dict)


# --------------------------------------------------------------------------- public API


def today_et() -> date:
    return datetime.now(UTC).astimezone(ET).date()


async def build_bundle(*, today: date | None = None) -> BriefBundle:
    """Fetch every section concurrently and return a fully-populated bundle.

    Each section gets its own try/except so one failure can't kill the whole
    brief. The orchestrator records the failure reason in ``failures`` so the
    renderer can show ``[UNAVAILABLE]`` markers with context.

    All three external clients are opened once in a single outer async-with
    block to avoid races between parallel tasks sharing the same instance.
    """
    day = today or today_et()
    meta = SessionMeta(
        today_et=day,
        is_0dte_day=day.weekday() in (0, 2, 4),
        is_holiday=day in US_HOLIDAYS,
    )
    failures: dict[str, str] = {}

    if meta.is_holiday:
        return BriefBundle(
            meta=meta, vol=None, levels=None, overnight=None,
            expected_move=None, calendar=None, earnings=None,
            global_markets=None, failures=failures,
        )

    expiry_today = next_spx_expiry(day)

    async with AsyncExitStack() as stack:
        polygon = await _maybe_enter(stack, PolygonClient, "polygon", failures)
        cboe = await _maybe_enter(stack, CboeClient, "vol", failures)
        perplexity = await _maybe_enter(stack, PerplexityClient, "perplexity", failures)

        async def _vol() -> VolContext | None:
            if cboe is None:
                return None
            try:
                vols = await cboe.get_live_snapshot()
                return _evaluate_vol(vols)
            except Exception as exc:
                failures["vol"] = str(exc)
                logger.warning("morning.vol_failed", error=str(exc))
                return None

        async def _polygon_sections() -> tuple[
            LevelsSection | None, OvernightSection | None, ExpectedMoveSection | None
        ]:
            if polygon is None:
                return None, None, None
            try:
                chain = await polygon.get_chain_snapshot(expiry_today)
            except Exception as exc:
                failures["polygon"] = str(exc)
                logger.warning("morning.polygon_failed", error=str(exc))
                return None, None, None
            levels = _levels_from_chain(chain)
            emove = _expected_move(chain)
            try:
                overnight = await _build_overnight(polygon, chain)
            except Exception as exc:
                failures["overnight"] = str(exc)
                logger.warning("morning.overnight_failed", error=str(exc))
                overnight = None
            return levels, overnight, emove

        async def _calendar() -> CalendarSection | None:
            if perplexity is None:
                return None
            try:
                return await _build_calendar(perplexity, day)
            except Exception as exc:
                failures["calendar"] = str(exc)
                logger.warning("morning.calendar_failed", error=str(exc))
                return None

        async def _earnings() -> EarningsSection | None:
            if perplexity is None:
                return None
            try:
                return await _build_earnings(perplexity, day)
            except Exception as exc:
                failures["earnings"] = str(exc)
                logger.warning("morning.earnings_failed", error=str(exc))
                return None

        async def _global() -> GlobalSection | None:
            if perplexity is None:
                return None
            try:
                return await _build_global(perplexity, day)
            except Exception as exc:
                failures["global"] = str(exc)
                logger.warning("morning.global_failed", error=str(exc))
                return None

        vol, polygon_tuple, calendar, earnings, global_ = await asyncio.gather(
            _vol(), _polygon_sections(), _calendar(), _earnings(), _global(),
        )

    levels, overnight, expected_move = polygon_tuple

    return BriefBundle(
        meta=meta, vol=vol, levels=levels, overnight=overnight,
        expected_move=expected_move, calendar=calendar,
        earnings=earnings, global_markets=global_, failures=failures,
    )


async def _maybe_enter(stack: AsyncExitStack, factory, name: str, failures: dict) -> Any:
    """Open an async context manager and record failure rather than propagating.

    The orchestrator uses one shared stack so all clients get closed at the
    end of ``build_bundle``. If a client cannot be constructed (missing API
    key) or its ``__aenter__`` raises (e.g. boto3 secret lookup failure), we
    capture the failure and return ``None`` so dependent sections render
    ``[UNAVAILABLE]`` instead of dragging the whole brief down.
    """
    try:
        client = factory()
        return await stack.enter_async_context(client)
    except Exception as exc:
        failures[f"{name}_open"] = str(exc)
        logger.warning(f"morning.{name}_client_open_failed", error=str(exc))
        return None


def render_brief(bundle: BriefBundle) -> str:
    """Render the brief as Markdown. Never raises."""
    day = bundle.meta.today_et
    pretty_day = day.strftime("%a %Y-%m-%d")

    if bundle.meta.is_holiday:
        return f"# Pre-Market Brief — {pretty_day}\n\nMarket closed today (holiday). No brief."

    lines: list[str] = []
    lines.append(f"# Pre-Market Brief — {pretty_day}")
    lines.append("")

    skip_today = bool(bundle.calendar and bundle.calendar.skip_today)
    if skip_today:
        lines.append(f"> **SKIP_TODAY=True** — {bundle.calendar.skip_reason}")
        lines.append("")

    # Section 1 — session meta
    lines.append("## Session metadata")
    lines.append(f"- 0DTE expiry day: **{'Y' if bundle.meta.is_0dte_day else 'N'}**")
    lines.append(f"- Holiday: **{'Y' if bundle.meta.is_holiday else 'N'}**")
    lines.append("")

    # Section 2 — volatility
    lines.append("## Volatility context")
    if bundle.vol is None or bundle.vol.vols is None:
        lines.append(f"- {UNAVAILABLE}")
    else:
        v = bundle.vol.vols
        # We don't fetch VIX30 separately here; CBOE's free feed gives us
        # VIX1D + SKEW which are the two inputs to the regime gates.
        lines.append(f"- VIX1D close: **{_fmt(v.vix_1d, UNAVAILABLE)}**")
        lines.append(f"- CBOE SKEW close: **{_fmt(v.cboe_skew, UNAVAILABLE)}**")
        verdict = "WOULD_SKIP" if bundle.vol.would_skip else "OK"
        lines.append(f"- Regime-gate verdict: **{verdict}**" + (
            f" ({'; '.join(bundle.vol.skip_reasons)})" if bundle.vol.skip_reasons else ""
        ))
    lines.append("")

    # Section 3 — yesterday's levels
    lines.append("## Yesterday's key levels (today's 0DTE chain)")
    if bundle.levels is None or bundle.levels.signals is None:
        lines.append(f"- {UNAVAILABLE}")
    else:
        s = bundle.levels.signals
        lines.append(f"- Gamma flip: **{_fmt_strike(s.gamma_flip)}**")
        lines.append(f"- Call wall: **{_fmt_strike(s.call_wall)}** | Put wall: **{_fmt_strike(s.put_wall)}**")
        lines.append(f"- Magnet strike: **{_fmt_strike(s.magnet_strike)}** | Max pain: **{_fmt_strike(s.max_pain)}**")
        if s.total_gex is not None:
            lines.append(f"- Total dealer GEX: **${s.total_gex:+.2f}B**")
        else:
            lines.append(f"- Total dealer GEX: {UNAVAILABLE}")
        lines.append(f"- Gamma regime: **{s.gamma_regime.replace('_', ' ')}**")
    lines.append("")

    # Section 4 — ES futures overnight
    lines.append("## ES futures overnight & implied open")
    if bundle.overnight is None or bundle.overnight.source == "unavailable":
        lines.append(f"- {UNAVAILABLE}")
    else:
        ov = bundle.overnight
        lines.append(f"- Source: **{ov.source}** (ES futures preferred; SPY x10 fallback)")
        lines.append(
            f"- Overnight: high {_fmt(ov.overnight_high)}, low {_fmt(ov.overnight_low)}, "
            f"last {_fmt(ov.overnight_last)}"
        )
        if ov.change_pts is not None and ov.change_pct is not None:
            lines.append(f"- Change: **{ov.change_pts:+.2f} pts ({ov.change_pct:+.2f}%)**")
        if ov.implied_spx_open is not None and ov.implied_spx_change_pct is not None:
            lines.append(
                f"- Implied SPX open: **{ov.implied_spx_open:.0f}** "
                f"({ov.implied_spx_change_pct:+.2f}% vs prior close {_fmt_strike(ov.prev_spx_close)})"
            )
    lines.append("")

    # Section 5 — expected move
    lines.append("## Expected 1SD daily move (0DTE ATM straddle x 0.85)")
    if bundle.expected_move is None or bundle.expected_move.expected_move_pts is None:
        lines.append(f"- {UNAVAILABLE}")
    else:
        em = bundle.expected_move
        lines.append(
            f"- ATM strike: **{_fmt_strike(em.atm_strike)}** (spot ~{_fmt_strike(em.spot)})"
        )
        lines.append(
            f"- ATM call mid: {_fmt(em.atm_call_mid)}, ATM put mid: {_fmt(em.atm_put_mid)}"
        )
        lines.append(
            f"- Expected move: **±{em.expected_move_pts:.1f} pts** "
            f"(±{(em.expected_move_pct or 0.0):.2f}%)"
        )
        lines.append(
            f"- Implied range: **{_fmt_strike(em.range_low)} → {_fmt_strike(em.range_high)}**"
        )
    lines.append("")

    # Section 6 — economic calendar
    lines.append("## Economic calendar (US, today)")
    if bundle.calendar is None:
        lines.append(f"- {UNAVAILABLE}")
    elif not bundle.calendar.events:
        lines.append("- No scheduled releases.")
    else:
        for ev in bundle.calendar.events:
            tier_marker = " **(Tier-1)**" if ev.is_tier_one else ""
            lines.append(f"- {ev.time_et} ET — {ev.name}{tier_marker}")
    lines.append("")

    # Section 7 — earnings
    lines.append("## Earnings (top S&P 500)")
    if bundle.earnings is None:
        lines.append(f"- {UNAVAILABLE}")
    elif not bundle.earnings.items:
        lines.append("- No major reports.")
    else:
        for it in bundle.earnings.items:
            lines.append(f"- {it}")
    lines.append("")

    # Section 8 — global markets
    lines.append("## Global markets sentiment")
    if bundle.global_markets is None:
        lines.append(f"- {UNAVAILABLE}")
    else:
        g = bundle.global_markets
        lines.append(f"- Asia: Nikkei {_or_unk(g.nikkei)} | Hang Seng {_or_unk(g.hang_seng)}")
        lines.append(f"- Europe (still trading): DAX {_or_unk(g.dax)} | FTSE {_or_unk(g.ftse)}")
        if g.sentiment:
            lines.append(f"- Verdict: **{g.sentiment}**")
        if g.notes:
            lines.append(f"- _{g.notes}_")
    lines.append("")

    # Section 9 — recommended posture
    lines.append("## Recommended posture")
    lines.append(_recommended_posture(bundle))

    return "\n".join(lines)


def _recommended_posture(bundle: BriefBundle) -> str:
    if bundle.meta.is_holiday:
        return "Market closed."
    if bundle.calendar and bundle.calendar.skip_today:
        return f"**Stand down** — {bundle.calendar.skip_reason}."
    if bundle.vol and bundle.vol.would_skip:
        return (
            "**High-volatility regime** — paper trades will be regime-blocked "
            f"at session start ({'; '.join(bundle.vol.skip_reasons)})."
        )
    levels = bundle.levels.signals if bundle.levels else None
    parts = [f"Standard scan. Score threshold {settings.engine.score_threshold}."]
    if levels is not None:
        watch = []
        if levels.call_wall is not None:
            watch.append(f"call_wall {levels.call_wall:.0f} resistance")
        if levels.put_wall is not None:
            watch.append(f"put_wall {levels.put_wall:.0f} support")
        if levels.gamma_flip is not None:
            watch.append(f"gamma flip {levels.gamma_flip:.0f}")
        if watch:
            parts.append("Watch: " + " / ".join(watch) + ".")
    return " ".join(parts)


# --------------------------------------------------------------------------- section builders


def _evaluate_vol(vols: VolSnapshot) -> VolContext:
    reasons: list[str] = []
    g = settings.regime_gates
    if vols.vix_1d is not None and vols.vix_1d > g.vix_1d_skip_threshold:
        reasons.append(f"VIX1D {vols.vix_1d:.1f} > {g.vix_1d_skip_threshold}")
    if vols.cboe_skew is not None and vols.cboe_skew > g.cboe_skew_skip_threshold:
        reasons.append(f"SKEW {vols.cboe_skew:.1f} > {g.cboe_skew_skip_threshold}")
    return VolContext(vols=vols, would_skip=bool(reasons), skip_reasons=tuple(reasons))


def _levels_from_chain(chain: ChainSnapshot) -> LevelsSection:
    sig = signals_from_chain(chain)
    return LevelsSection(signals=sig, spot_used=chain.spot or None)


def _expected_move(chain: ChainSnapshot) -> ExpectedMoveSection | None:
    """ATM 0DTE straddle * 0.85 -> 1SD pts.

    Note: at 08:00 ET the SPX session isn't open, so quotes are stale prior-day
    closes. Polygon snapshots still return last_quote bid/ask which is what we
    want — it represents the market's last expectation of today's IV.
    """
    spot = chain.spot
    if spot <= 0:
        return None
    atm = round(spot / 5.0) * 5.0
    call = next((q for q in chain.calls if q.strike == atm), None)
    put = next((q for q in chain.puts if q.strike == atm), None)
    if call is None or put is None:
        return None
    call_mid = call.mid if call.mid > 0 else None
    put_mid = put.mid if put.mid > 0 else None
    if call_mid is None or put_mid is None:
        return None
    em_pts = (call_mid + put_mid) * ATM_STRADDLE_DAILY_SD_MULTIPLIER
    em_pct = (em_pts / spot * 100.0) if spot else None
    return ExpectedMoveSection(
        atm_strike=atm,
        atm_call_mid=call_mid,
        atm_put_mid=put_mid,
        expected_move_pts=round(em_pts, 1),
        expected_move_pct=round(em_pct, 3) if em_pct is not None else None,
        range_low=round(spot - em_pts, 0),
        range_high=round(spot + em_pts, 0),
        spot=spot,
    )


async def _build_overnight(polygon: PolygonClient, chain: ChainSnapshot) -> OvernightSection:
    """Fetch SPY prev-day + overnight minute bars to imply SPX open.

    ES futures path: Polygon entitlement isn't reliable; spec says fall back
    to SPY. We compute the SPY->SPX basis from yesterday's closes (SPY close
    * 10 vs derived SPX spot from the chain).
    """
    spy_prev = await polygon.get_prev_day_aggregate("SPY")
    if not spy_prev:
        return OvernightSection(
            source="unavailable",
            overnight_high=None, overnight_low=None, overnight_last=None,
            change_pts=None, change_pct=None,
            implied_spx_open=None, implied_spx_change_pct=None,
            prev_spx_close=None,
        )

    # Pull last ~16 hours of SPY minute bars (post-close yesterday → now).
    now = datetime.now(UTC)
    start = now - timedelta(hours=16)
    try:
        bars = await polygon.get_minute_bars_range("SPY", start, now)
    except Exception as exc:
        logger.debug("morning.spy_minute_failed", error=str(exc))
        bars = []

    spy_prev_close = spy_prev["close"]
    if bars:
        highs = [b["high"] for b in bars if b["high"] > 0]
        lows = [b["low"] for b in bars if b["low"] > 0]
        last = bars[-1]["close"]
        overnight_high = max(highs) if highs else None
        overnight_low = min(lows) if lows else None
    else:
        overnight_high = overnight_low = None
        last = spy_prev_close  # no overnight data → assume flat

    change_pts = last - spy_prev_close
    change_pct = (change_pts / spy_prev_close * 100.0) if spy_prev_close else None

    # SPX basis: prefer chain-derived spot (yesterday's close-of-day spot from
    # put-call parity) and the SPY closing print. basis = spx_close - spy*10.
    derived_spx_close = chain.spot if chain.spot > 0 else None
    if derived_spx_close is not None:
        basis = derived_spx_close - spy_prev_close * SPY_SPX_RATIO
    else:
        basis = 0.0
    implied_spx_open = last * SPY_SPX_RATIO + basis
    prev_spx_close = (derived_spx_close
                      if derived_spx_close is not None
                      else spy_prev_close * SPY_SPX_RATIO)
    implied_spx_change = implied_spx_open - prev_spx_close
    implied_spx_change_pct = (
        implied_spx_change / prev_spx_close * 100.0 if prev_spx_close else None
    )
    return OvernightSection(
        source="SPY",
        overnight_high=overnight_high,
        overnight_low=overnight_low,
        overnight_last=last,
        change_pts=round(change_pts, 2),
        change_pct=round(change_pct, 3) if change_pct is not None else None,
        implied_spx_open=round(implied_spx_open, 0),
        implied_spx_change_pct=round(implied_spx_change_pct, 3)
            if implied_spx_change_pct is not None else None,
        prev_spx_close=prev_spx_close,
    )


# --------------------------------------------------------------------------- Perplexity sections


async def _build_calendar(client: PerplexityClient, day: date) -> CalendarSection:
    iso = day.isoformat()
    pretty = day.strftime("%A %B %-d, %Y")
    prompt = (
        f"List today's scheduled US economic data releases for {pretty} ({iso}). "
        "Include CPI, PPI, PCE, NFP, JOLTS, retail sales, FOMC minutes/decisions, "
        "and any Fed speaker remarks. For each, give the scheduled time in US "
        "Eastern Time (HH:MM in 24h format) and the official release name. "
        "Return ONLY valid JSON in this exact shape, no commentary: "
        '{"events": [{"name": "<release>", "time_et": "HH:MM"}]}. '
        "If nothing is scheduled today, return {\"events\": []}."
    )
    content = await client.ask(prompt)
    parsed = _safe_json(content)
    raw_events = parsed.get("events", []) if isinstance(parsed, dict) else []

    events: list[EconomicEvent] = []
    skip_reason: str | None = None
    for raw in raw_events:
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("name") or "").strip()
        time_et = str(raw.get("time_et") or "").strip()
        if not name or not time_et:
            continue
        tier = _is_tier_one(name)
        events.append(EconomicEvent(name=name, time_et=time_et, is_tier_one=tier))
        if tier and skip_reason is None and _within_tier_one_window(time_et):
            skip_reason = f"Tier-1 event at {time_et} ET — {name}"

    return CalendarSection(
        events=tuple(events),
        skip_today=skip_reason is not None,
        skip_reason=skip_reason,
    )


async def _build_earnings(client: PerplexityClient, day: date) -> EarningsSection:
    iso = day.isoformat()
    pretty = day.strftime("%A %B %-d, %Y")
    prompt = (
        f"List the top 5 largest S&P 500 companies by market cap reporting "
        f"earnings on {pretty} ({iso}). For each, indicate whether the report "
        "is pre-market (BMO) or after-close (AMC). Return ONLY valid JSON: "
        '{"earnings": [{"ticker": "AAPL", "timing": "BMO|AMC"}]}. '
        "If none, return {\"earnings\": []}."
    )
    content = await client.ask(prompt)
    parsed = _safe_json(content)
    raw = parsed.get("earnings", []) if isinstance(parsed, dict) else []
    items: list[str] = []
    for r in raw:
        if not isinstance(r, dict):
            continue
        tic = str(r.get("ticker") or "").strip().upper()
        timing = str(r.get("timing") or "").strip().upper()
        if not tic:
            continue
        pretty_timing = "pre-open" if "B" in timing else "after-close" if "A" in timing else timing.lower()
        items.append(f"{tic} ({pretty_timing})")
    return EarningsSection(items=tuple(items))


async def _build_global(client: PerplexityClient, day: date) -> GlobalSection:
    iso = day.isoformat()
    pretty = day.strftime("%A %B %-d, %Y")
    prompt = (
        f"Today is {pretty} ({iso}). Give a one-sentence summary of how Asian "
        "markets closed overnight (Nikkei 225, Hang Seng) and where European "
        "markets are currently trading (DAX, FTSE 100 — both still open at "
        "08:00 ET). For each index, summarize direction as one of: up, down, "
        "flat. End with an overall sentiment verdict: Risk On, Risk Off, or "
        "Neutral. Return ONLY valid JSON: "
        '{"nikkei": "up|down|flat", "hang_seng": "up|down|flat", '
        '"dax": "up|down|flat", "ftse": "up|down|flat", '
        '"sentiment": "Risk On|Risk Off|Neutral", "notes": "<one sentence>"}.'
    )
    content = await client.ask(prompt)
    parsed = _safe_json(content)
    if not isinstance(parsed, dict):
        return GlobalSection(None, None, None, None, None, None)
    return GlobalSection(
        nikkei=_norm_dir(parsed.get("nikkei")),
        hang_seng=_norm_dir(parsed.get("hang_seng")),
        dax=_norm_dir(parsed.get("dax")),
        ftse=_norm_dir(parsed.get("ftse")),
        sentiment=_norm_sentiment(parsed.get("sentiment")),
        notes=(str(parsed.get("notes")).strip() if parsed.get("notes") else None),
    )


# --------------------------------------------------------------------------- helpers


def _safe_json(content: str) -> dict[str, Any]:
    """Parse Perplexity content as JSON, tolerating leading markdown fences."""
    text = content.strip()
    if text.startswith("```"):
        # Strip ```json ... ``` fences
        text = text.strip("`").lstrip("json").strip()
    # Drop leading prose by finding the first '{'.
    first = text.find("{")
    last = text.rfind("}")
    if first == -1 or last == -1 or last <= first:
        return {}
    candidate = text[first:last + 1]
    try:
        return json.loads(candidate)
    except (ValueError, json.JSONDecodeError):
        return {}


def _is_tier_one(name: str) -> bool:
    upper = name.upper()
    return any(tag.upper() in upper for tag in settings.morning_brief.tier_one_events)


def _within_tier_one_window(time_et: str) -> bool:
    """time_et like '08:30' or '14:00'."""
    try:
        hh, mm = time_et.split(":")
        t = time(int(hh), int(mm))
    except (ValueError, AttributeError):
        return False
    return TIER_ONE_WINDOW_START <= t <= TIER_ONE_WINDOW_END


def _norm_dir(raw: Any) -> str | None:
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if s.startswith("up") or s == "higher":
        return "up"
    if s.startswith("down") or s == "lower":
        return "down"
    if s.startswith("flat") or s == "mixed":
        return "flat"
    return None


def _norm_sentiment(raw: Any) -> str | None:
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if "off" in s:
        return "Risk Off"
    if "on" in s:
        return "Risk On"
    if "neutral" in s or "mixed" in s:
        return "Neutral"
    return None


def _fmt(v: float | None, missing: str = "—") -> str:
    if v is None:
        return missing
    return f"{v:.2f}"


def _fmt_strike(v: float | None, missing: str = "—") -> str:
    if v is None:
        return missing
    return f"{v:.0f}"


def _or_unk(v: str | None) -> str:
    return v if v else UNAVAILABLE


# --------------------------------------------------------------------------- Discord post


def post_to_discord(markdown_text: str, *, webhook_url: str | None = None) -> bool:
    """Post brief Markdown to the morning-brief webhook. Chunks at 1900 chars.

    Mirrors ``zeroday_paper.reporting.daily_report.post_to_discord`` so the
    chunking logic stays consistent.
    """
    url = webhook_url or _resolve_webhook()
    if not url:
        logger.warning("morning.no_webhook")
        return False
    chunks = _chunk_markdown(markdown_text, max_chars=1900)
    ok = True
    for chunk in chunks:
        try:
            r = httpx.post(url, json={"content": chunk}, timeout=10.0)
            if r.status_code not in (200, 204):
                logger.warning("morning.discord_status", status=r.status_code, body=r.text[:200])
                ok = False
        except Exception as exc:
            logger.warning("morning.discord_error", error=str(exc))
            ok = False
    return ok


def _resolve_webhook() -> str | None:
    try:
        return discord_webhook(settings.reporting.morning_brief_discord_webhook_secret_key)
    except Exception as exc:
        logger.warning("morning.webhook_lookup_failed", error=str(exc))
        return None
