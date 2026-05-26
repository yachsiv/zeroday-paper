"""365-day historical replay.

Iterates backward from yesterday, chunked by `chunk_size_days` to support
resumable execution. Each chunk:

    1. Picks SPX 0DTE expirations in the window (Mon/Wed/Fri).
    2. For each expiry, replays at 2-min cadence between 09:30 and 13:30 ET.
    3. Self-computes GEX from the historical chain (no FlashAlpha).
    4. Calls the full pipeline (Layer 1 patterns, scorer, strike selector).
    5. Journals trades with source="replay".

Skips weekends + US market holidays via a hardcoded list (sufficient for 365d).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from typing import Iterable
from zoneinfo import ZoneInfo

import structlog

from zeroday_paper.config import settings
from zeroday_paper.data.cboe_client import CboeClient
from zeroday_paper.data.flashalpha_client import signals_from_chain
from zeroday_paper.data.polygon_client import PolygonClient, next_spx_expiry
from zeroday_paper.engine.gex_levels import compute_rr25
from zeroday_paper.engine.gex_patterns import classify_layer1
from zeroday_paper.engine.journal import Journal, trade_id_for
from zeroday_paper.engine.models import (
    MarketState,
    PaperTrade,
    PatternHit,
    StrategyType,
)
from zeroday_paper.engine.pricing import entry_quote
from zeroday_paper.engine.score import score_state
from zeroday_paper.engine.strike_select import select_for

logger = structlog.get_logger(__name__)
ET = ZoneInfo(settings.engine.market_timezone)


# US market holidays 2024-2026 (closed days that fall on M/W/F SPX 0DTE candidates).
US_HOLIDAYS: frozenset[date] = frozenset({
    date(2024, 1, 1), date(2024, 1, 15), date(2024, 2, 19), date(2024, 3, 29),
    date(2024, 5, 27), date(2024, 6, 19), date(2024, 7, 4), date(2024, 9, 2),
    date(2024, 11, 28), date(2024, 12, 25),
    date(2025, 1, 1), date(2025, 1, 20), date(2025, 2, 17), date(2025, 4, 18),
    date(2025, 5, 26), date(2025, 6, 19), date(2025, 7, 4), date(2025, 9, 1),
    date(2025, 11, 27), date(2025, 12, 25),
    date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16), date(2026, 4, 3),
    date(2026, 5, 25), date(2026, 6, 19), date(2026, 7, 3), date(2026, 9, 7),
    date(2026, 11, 26), date(2026, 12, 25),
})


@dataclass
class ReplayProgress:
    days_processed: int = 0
    chains_fetched: int = 0
    states_scored: int = 0
    trades_written: int = 0
    errors: int = 0


def iter_replay_dates(start: date, days_back: int) -> Iterable[date]:
    """0DTE candidate dates within `days_back`, newest first."""
    d = start
    end = start - timedelta(days=days_back)
    while d >= end:
        if d.weekday() in (0, 2, 4) and d not in US_HOLIDAYS:  # Mon/Wed/Fri
            yield d
        d -= timedelta(days=1)


def iter_scan_times_for(d: date) -> Iterable[datetime]:
    """2-min cadence between 09:35 and 13:30 ET on `d`."""
    start = datetime.combine(d, time(9, 35), ET)
    cutoff = datetime.combine(d, settings.engine.new_entry_cutoff, ET)
    step = timedelta(seconds=settings.engine.poll_interval_seconds)
    t = start
    while t <= cutoff:
        yield t.astimezone(UTC)
        t += step


async def replay_one_day(
    d: date,
    *,
    polygon: PolygonClient,
    cboe: CboeClient,
    journal: Journal,
    progress: ReplayProgress,
) -> None:
    """Replay one 0DTE expiry for date `d`."""
    try:
        chain_anchor = await polygon.get_chain_snapshot(d)
        progress.chains_fetched += 1
    except Exception as exc:
        logger.warning("replay.chain_anchor_failed", date=str(d), error=str(exc))
        progress.errors += 1
        return

    try:
        vols = await cboe.get_historical_snapshot(d)
    except Exception as exc:
        logger.warning("replay.vols_failed", date=str(d), error=str(exc))
        vols = (await cboe.get_live_snapshot())  # last-resort fallback

    prior_state: MarketState | None = None

    for asof in iter_scan_times_for(d):
        try:
            chain = await polygon.get_chain_snapshot_at(d, asof)
        except Exception as exc:
            logger.debug("replay.snapshot_at_failed", asof=str(asof), error=str(exc))
            progress.errors += 1
            continue
        progress.chains_fetched += 1

        signals = signals_from_chain(chain)
        state = MarketState(asof=asof, chain=chain, signals=signals, vols=vols, spot=chain.spot)

        l1_matches = classify_layer1(state, prior_state)
        l1_hits = [
            PatternHit(
                pattern_id=m.pattern_id, name=m.name, direction=m.direction,
                confidence=str(m.confidence), layer="L1_RULES", score_bonus=m.score_bonus,
            )
            for m in l1_matches
        ]

        for strategy in (StrategyType.BULL_PUT, StrategyType.BEAR_CALL):
            score = score_state(state, strategy, l1_hits=l1_hits)
            progress.states_scored += 1
            if score.total < settings.engine.score_threshold:
                continue

            sel = select_for(strategy, state)
            if sel.spread is None:
                continue

            eq = entry_quote(sel.spread)
            if not eq.is_acceptable:
                continue

            t_id = trade_id_for(
                entry_date=asof.astimezone(ET).date(),
                entry_minute=asof.astimezone(ET).hour * 60 + asof.astimezone(ET).minute,
                short_strike=sel.spread.short_leg.strike,
                long_strike=sel.spread.long_leg.strike,
                strategy=strategy,
                source="replay",
            )

            t = PaperTrade(
                trade_id=t_id, strategy=strategy, entry_ts=asof, expiry=d,
                spot_at_entry=state.spot,
                short_strike=sel.spread.short_leg.strike,
                long_strike=sel.spread.long_leg.strike,
                short_contract=sel.spread.short_leg.contract,
                long_contract=sel.spread.long_leg.contract,
                width=sel.spread.width, contracts=1,
                credit_mid=eq.credit_mid, credit_bid=eq.credit_bid,
                max_loss_bid=eq.max_loss_bid, credit_bid_ratio=eq.credit_ratio,
                short_delta=sel.spread.short_leg.delta,
                long_delta=sel.spread.long_leg.delta,
                short_iv=sel.spread.short_leg.iv, long_iv=sel.spread.long_leg.iv,
                short_gamma=sel.spread.short_leg.gamma,
                short_theta=sel.spread.short_leg.theta,
                short_vega=sel.spread.short_leg.vega,
                short_oi=sel.spread.short_leg.open_interest,
                long_oi=sel.spread.long_leg.open_interest,
                short_volume=sel.spread.short_leg.volume,
                gamma_regime=state.signals.gamma_regime,
                gamma_flip=state.signals.gamma_flip,
                call_wall=state.signals.call_wall, put_wall=state.signals.put_wall,
                magnet_strike=state.signals.magnet_strike, pin_score=state.signals.pin_score,
                total_gex_b=state.signals.total_gex,
                zero_dte_gex_share=state.signals.zero_dte_gex_share,
                vix_1d=vols.vix_1d, cboe_skew=vols.cboe_skew,
                rr25=compute_rr25(state),
                active_patterns_l1=",".join(h.pattern_id for h in l1_hits),
                active_patterns_l2="",
                patterns_score_bonus=sum(h.score_bonus for h in l1_hits),
                score=score.total,
                score_breakdown_json=str(score.breakdown),
                source="replay",
                notes=";".join(score.notes),
            )

            if journal.write_trade(t):
                progress.trades_written += 1

        prior_state = state

    progress.days_processed += 1
    logger.info(
        "replay.day_complete",
        date=str(d),
        trades=progress.trades_written,
        chains=progress.chains_fetched,
        errors=progress.errors,
    )


async def run_replay(*, days_back: int | None = None) -> ReplayProgress:
    """Top-level replay entry point. Resumable via replay_cursor table."""
    settings_days = days_back if days_back is not None else settings.replay.days_back
    today = datetime.now(UTC).astimezone(ET).date()
    start = today - timedelta(days=1)

    journal = Journal()
    done_chunks = {(a, b) for a, b in journal.replay_chunks_done()}
    progress = ReplayProgress()

    candidate_dates = list(iter_replay_dates(start, settings_days))
    chunk_size = settings.replay.chunk_size_days

    chunk_id = 0
    async with PolygonClient() as polygon, CboeClient() as cboe:
        for i in range(0, len(candidate_dates), chunk_size):
            chunk = candidate_dates[i:i + chunk_size]
            if not chunk:
                continue
            chunk_id += 1
            chunk_end = chunk[0]      # newest in chunk
            chunk_start = chunk[-1]   # oldest in chunk
            if (chunk_start, chunk_end) in done_chunks:
                logger.info("replay.chunk_skip", chunk_id=chunk_id, start=str(chunk_start), end=str(chunk_end))
                continue
            for d in chunk:
                await replay_one_day(d, polygon=polygon, cboe=cboe, journal=journal, progress=progress)
                journal.heartbeat("replay")
            journal.mark_chunk_done(chunk_id, chunk_start, chunk_end)
            logger.info("replay.chunk_complete", chunk_id=chunk_id, trades=progress.trades_written)

    journal.close()
    return progress
