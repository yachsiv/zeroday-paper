"""Live scanner + monitor (2-min loop).

One cycle:

    1. Fetch (chain + signals + vols) once.
    2. Monitor: for each open position, compute fresh exit quote, decide hold/close.
    3. Scan: score each strategy, select strikes, write new trades subject to
       dedup + concurrency caps.

Designed to run inside an ECS Fargate task. Single-process, single-writer to DuckDB.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, time, timedelta
from zoneinfo import ZoneInfo

import structlog

from zeroday_paper.config import settings
from zeroday_paper.data.cboe_client import CboeClient
from zeroday_paper.data.flashalpha_client import (
    FlashAlphaClient,
    MarketSignals,
    signals_from_chain,
)
from zeroday_paper.data.polygon_client import PolygonClient, next_spx_expiry
from zeroday_paper.engine.gex_levels import compute_rr25
from zeroday_paper.engine.gex_patterns import classify_layer1
from zeroday_paper.engine.gex_patterns_llm import classify_layer2
from zeroday_paper.engine.journal import Journal, trade_id_for
from zeroday_paper.engine.models import (
    MarketState,
    PaperTrade,
    PatternHit,
    PositionStatus,
    Spread,
    StrategyType,
    TradeOutcome,
    TradeTick,
)
from zeroday_paper.engine.pricing import entry_quote, exit_quote
from zeroday_paper.engine.score import score_state
from zeroday_paper.engine.state import decide_exit
from zeroday_paper.engine.strike_select import select_for
from zeroday_paper.metrics import emit as emit_metric

logger = structlog.get_logger(__name__)
ET = ZoneInfo(settings.engine.market_timezone)


# -----------------------------------------------------------------------------
# Cycle
# -----------------------------------------------------------------------------


async def run_one_cycle(journal: Journal) -> dict[str, int]:
    """Single scan+monitor cycle. Returns a small status dict."""
    now = datetime.now(UTC)
    expiry = next_spx_expiry(now.astimezone(ET).date())

    stats = {"new_trades": 0, "monitored": 0, "exited": 0, "errors": 0}

    async with PolygonClient() as polygon, CboeClient() as cboe:
        # Vols are cheap; fetch unconditionally.
        try:
            vols = await cboe.get_live_snapshot()
        except Exception as exc:
            logger.warning("cycle.vols_failed", error=str(exc))
            stats["errors"] += 1
            return stats

        try:
            chain = await polygon.get_chain_snapshot(expiry)
        except Exception as exc:
            logger.warning("cycle.chain_failed", error=str(exc))
            stats["errors"] += 1
            return stats

        # Signals: FlashAlpha preferred, fall back to self-computed.
        signals: MarketSignals
        try:
            async with FlashAlphaClient() as fa:
                signals = await fa.get_signals("SPX")
                if signals.spot <= 0:
                    signals = signals_from_chain(chain)
        except Exception as exc:
            logger.warning("cycle.flashalpha_failed", error=str(exc))
            signals = signals_from_chain(chain)

        if chain.spot <= 0:
            logger.warning("cycle.spot_unresolved", calls=len(chain.calls), puts=len(chain.puts))
            stats["errors"] += 1
            return stats

        state = MarketState(asof=now, chain=chain, signals=signals, vols=vols, spot=chain.spot)

        # ----- Monitor open positions ---------------------------------------
        monitored, exited = await _monitor_open_positions(journal, state, polygon)
        stats["monitored"] = monitored
        stats["exited"] = exited

        # ----- Scan for new entries -----------------------------------------
        scanned = await _scan_for_entries(journal, state)
        stats["new_trades"] = scanned

    journal.heartbeat("scanner")
    emit_metric("cycle.complete", 1.0)
    if stats["new_trades"]:
        emit_metric("trade.written", float(stats["new_trades"]))
    if stats["exited"]:
        emit_metric("position.exit", float(stats["exited"]))
    if stats["errors"]:
        emit_metric("cycle.errors", float(stats["errors"]))
    logger.info("cycle.complete", **stats, spot=state.spot, regime=state.signals.gamma_regime)
    return stats


# -----------------------------------------------------------------------------
# Monitor
# -----------------------------------------------------------------------------


async def _monitor_open_positions(
    journal: Journal,
    state: MarketState,
    polygon: PolygonClient,
) -> tuple[int, int]:
    open_rows = journal.open_positions()
    if not open_rows:
        return 0, 0

    by_contract = {q.contract: q for q in state.chain.calls + state.chain.puts}
    exited = 0

    for row in open_rows:
        short = by_contract.get(row["short_contract"])
        long = by_contract.get(row["long_contract"])
        if short is None or long is None:
            logger.debug("monitor.contract_missing", trade_id=row["trade_id"])
            continue

        # Reconstruct a Spread shell for pricing.
        strategy = StrategyType(row["strategy"])
        spread = Spread(strategy=strategy, short_leg=short, long_leg=long, contracts=row["contracts"])
        eq = exit_quote(spread, entry_credit_bid=row["credit_bid"])

        decision = decide_exit(
            now_utc=state.asof,
            entry_credit_bid=row["credit_bid"],
            exit_cost_bid=eq.cost_bid,
            pct_of_max_profit=eq.pct_of_max_profit_bid,
        )

        tick = TradeTick(
            trade_id=row["trade_id"], ts=state.asof, spot=state.spot,
            short_bid=short.bid, short_ask=short.ask,
            long_bid=long.bid, long_ask=long.ask,
            exit_cost_bid=eq.cost_bid, exit_cost_mid=eq.cost_mid,
            pnl_bid=eq.pnl_bid, pnl_mid=eq.pnl_mid,
            pct_of_max_profit=eq.pct_of_max_profit_bid,
            status=decision.next_status,
        )
        journal.write_tick(tick)

        if decision.should_exit:
            held = int((state.asof - row["entry_ts"].replace(tzinfo=UTC)).total_seconds() / 60)
            outcome = TradeOutcome(
                trade_id=row["trade_id"],
                exit_ts=state.asof,
                exit_status=decision.next_status,
                exit_spot=state.spot,
                exit_cost_bid=eq.cost_bid,
                exit_cost_mid=eq.cost_mid,
                pnl_bid=eq.pnl_bid,
                pnl_mid=eq.pnl_mid,
                held_minutes=held,
                max_excursion_pct=eq.pct_of_max_profit_bid,
                min_excursion_pct=eq.pct_of_max_profit_bid,
                exit_reason=decision.reason,
            )
            journal.write_outcome(outcome)
            exited += 1
            logger.info("monitor.exit", trade_id=row["trade_id"], pnl_bid=eq.pnl_bid, reason=decision.reason)

    return len(open_rows), exited


# -----------------------------------------------------------------------------
# Scan
# -----------------------------------------------------------------------------


async def _scan_for_entries(journal: Journal, state: MarketState) -> int:
    """Score all strategies, write any that clear threshold + caps."""
    l1_matches = classify_layer1(state)
    l1_hits = [
        PatternHit(
            pattern_id=m.pattern_id, name=m.name, direction=m.direction,
            confidence=str(m.confidence), layer="L1_RULES", score_bonus=m.score_bonus,
        )
        for m in l1_matches
    ]

    l2_hits: list[PatternHit] = []
    if settings.patterns.layer_2_llm_enabled:
        try:
            l2_matches = await classify_layer2(state)
            l2_hits = [
                PatternHit(
                    pattern_id=m.pattern_id, name=m.name, direction=m.direction,
                    confidence=str(m.confidence), layer="L2_LLM", score_bonus=m.score_bonus,
                )
                for m in l2_matches
            ]
        except Exception as exc:
            logger.warning("scan.layer2_failed", error=str(exc))

    today = state.asof.astimezone(ET).date()
    today_total = journal.count_today(today, source="live")

    written = 0
    for strategy in (StrategyType.BULL_PUT, StrategyType.BEAR_CALL):
        if today_total + written >= settings.concurrency.max_concurrent_total:
            break

        score = score_state(state, strategy, l1_hits=l1_hits, l2_hits=l2_hits)
        if score.total < settings.engine.score_threshold:
            continue

        sel = select_for(strategy, state)
        if sel.spread is None:
            continue

        eq = entry_quote(sel.spread)
        if not eq.is_acceptable:
            continue

        entry_minute = state.asof.astimezone(ET).hour * 60 + state.asof.astimezone(ET).minute
        t_id = trade_id_for(
            entry_date=today,
            entry_minute=entry_minute,
            short_strike=sel.spread.short_leg.strike,
            long_strike=sel.spread.long_leg.strike,
            strategy=strategy,
            source="live",
        )

        t = PaperTrade(
            trade_id=t_id, strategy=strategy, entry_ts=state.asof, expiry=sel.spread.expiry,
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
            vix_1d=state.vols.vix_1d, cboe_skew=state.vols.cboe_skew,
            rr25=compute_rr25(state),
            active_patterns_l1=",".join(h.pattern_id for h in l1_hits),
            active_patterns_l2=",".join(h.pattern_id for h in l2_hits),
            patterns_score_bonus=sum(h.score_bonus for h in l1_hits + l2_hits),
            score=score.total,
            score_breakdown_json=json.dumps(score.breakdown),
            source="live",
            notes=";".join(score.notes),
        )
        if journal.write_trade(t):
            written += 1
            logger.info(
                "scan.entry",
                strategy=str(strategy), score=score.total,
                credit_bid=eq.credit_bid, short=t.short_strike, long=t.long_strike,
                patterns_l1=t.active_patterns_l1, patterns_l2=t.active_patterns_l2,
            )

    return written


# -----------------------------------------------------------------------------
# Live loop
# -----------------------------------------------------------------------------


def is_market_hours_et(now_utc: datetime | None = None) -> bool:
    now = (now_utc or datetime.now(UTC)).astimezone(ET)
    if now.weekday() >= 5:
        return False
    return settings.engine.session_start <= now.time() <= settings.engine.session_end


async def run_live_loop() -> None:
    """Live scanner loop.

    Behavior:
      - Pre-session weekday boot (e.g. cron fires at 09:28 ET): sleep until
        session_start, then begin scanning. Sleep cap = 30 min so we don't go
        completely dark even if the clock jumps unexpectedly.
      - Inside the session: poll every interval_s, run one cycle, repeat.
      - Past session_end: log and exit (Fargate task naturally stops).
      - Weekend / holiday: log and exit.
    """
    journal = Journal()
    interval = settings.engine.poll_interval_seconds
    logger.info("live.start", interval_s=interval, threshold=settings.engine.score_threshold)
    try:
        while True:
            now_et = datetime.now(UTC).astimezone(ET)
            if now_et.weekday() >= 5:
                logger.info("live.weekend_exit")
                return
            t = now_et.time()
            if t < settings.engine.session_start:
                target = datetime.combine(
                    now_et.date(), settings.engine.session_start, tzinfo=ET
                )
                wait_s = max(0.0, (target - now_et).total_seconds())
                logger.info("live.pre_session_wait", seconds=int(wait_s))
                journal.heartbeat("scanner", status="waiting_for_session")
                await asyncio.sleep(min(wait_s + 5, 1800))
                continue
            if t > settings.engine.session_end:
                logger.info("live.after_session_exit")
                return
            try:
                await run_one_cycle(journal)
            except Exception as exc:
                logger.exception("live.cycle_error", error=str(exc))
                journal.heartbeat("scanner", status=f"error:{exc}")
            await asyncio.sleep(interval)
    finally:
        journal.close()
