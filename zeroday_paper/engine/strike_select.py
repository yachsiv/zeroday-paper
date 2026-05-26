"""4-stage strike selector.

Stage 1: Geometric  — short delta in [target-tol, target+tol], width within range.
Stage 2: Quality    — OI, bid/ask spread, both legs tradable.
Stage 3: Volatility — adjust width by realized vol band (uses VIX1D as proxy).
Stage 4: Rank       — among survivors, pick highest credit_ratio.

Returns at most one Spread per strategy (or None if no candidate passes).
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog

from zeroday_paper.config import settings
from zeroday_paper.data.polygon_client import ChainSnapshot, OptionQuote
from zeroday_paper.engine.models import MarketState, Spread, StrategyType
from zeroday_paper.engine.pricing import entry_quote

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class SelectionResult:
    spread: Spread | None
    reasons: list[str]
    candidates_considered: int


def select_for(strategy: StrategyType, state: MarketState) -> SelectionResult:
    """Return the best Spread for the requested strategy, or None."""
    chain = state.chain
    vix_1d = state.vols.vix_1d

    if strategy == StrategyType.BULL_PUT:
        return _select_credit_spread(chain.puts, state, side="put", vix_1d=vix_1d)
    if strategy == StrategyType.BEAR_CALL:
        return _select_credit_spread(chain.calls, state, side="call", vix_1d=vix_1d)
    if strategy == StrategyType.IRON_CONDOR:
        return _select_iron_condor(state, vix_1d=vix_1d)
    return SelectionResult(None, [f"unsupported_strategy:{strategy}"], 0)


def _select_credit_spread(
    quotes: list[OptionQuote],
    state: MarketState,
    *,
    side: str,
    vix_1d: float | None,
) -> SelectionResult:
    cfg = settings.strikes
    target = cfg.default_short_delta_target
    tol = cfg.default_short_delta_tolerance
    width = _adjusted_width(cfg.default_spread_width, vix_1d)

    # Stage 1: short delta candidates (delta is negative for puts; abs comparison)
    shorts_s1: list[OptionQuote] = []
    for q in quotes:
        if q.delta is None or q.strike <= 0:
            continue
        d = abs(q.delta)
        if (target - tol) <= d <= (target + tol):
            shorts_s1.append(q)
    if not shorts_s1:
        return SelectionResult(None, ["no_short_in_delta_band"], 0)

    # Stage 2: quality gates on short leg
    shorts_s2 = [q for q in shorts_s1 if _is_quality(q, cfg.min_short_oi, cfg.max_bid_ask_spread_dollars)]
    if not shorts_s2:
        return SelectionResult(None, ["all_shorts_failed_quality"], len(shorts_s1))

    strategy = StrategyType.BULL_PUT if side == "put" else StrategyType.BEAR_CALL

    # For each short, find a matching long at +width (call) or -width (put)
    candidates: list[tuple[Spread, float]] = []
    by_strike = {q.strike: q for q in quotes}
    for s in shorts_s2:
        target_long_strike = s.strike - width if side == "put" else s.strike + width
        long = _closest_strike(by_strike, target_long_strike)
        if long is None:
            continue
        if not _is_quality(long, cfg.min_long_oi, cfg.max_bid_ask_spread_dollars):
            continue
        spread = Spread(strategy=strategy, short_leg=s, long_leg=long, contracts=1)
        eq = entry_quote(spread)
        if not eq.is_acceptable:
            continue
        candidates.append((spread, eq.credit_ratio))

    if not candidates:
        return SelectionResult(None, ["no_pair_with_acceptable_entry"], len(shorts_s2))

    candidates.sort(key=lambda c: c[1], reverse=True)
    best, ratio = candidates[0]
    return SelectionResult(
        spread=best,
        reasons=[f"selected ratio={ratio:.3f}"],
        candidates_considered=len(candidates),
    )


def _select_iron_condor(state: MarketState, *, vix_1d: float | None) -> SelectionResult:
    """Iron condor = bull put + bear call, both at the same delta target.

    Selects independently and combines. v1 returns the put leg only marked as
    BULL_PUT; the scanner handles the dual-leg case by calling _select_credit_spread
    twice for the iron-condor strategy class. Kept simple for v1 — for paper
    trading we treat ICs as two separate spreads journaled with shared metadata.
    """
    # v1 simplification: emit the put spread; scanner will optionally emit the call too.
    return _select_credit_spread(state.chain.puts, state, side="put", vix_1d=vix_1d)


def _adjusted_width(base_width: int, vix_1d: float | None) -> int:
    if vix_1d is None:
        return base_width
    if vix_1d > 20:
        return base_width + 10
    if vix_1d < 12:
        return max(15, base_width - 5)
    return base_width


def _is_quality(q: OptionQuote, min_oi: int, max_spread: float) -> bool:
    if q.open_interest < min_oi:
        return False
    if not q.is_tradable:
        return False
    if q.bid_ask_spread > max_spread:
        return False
    return True


def _closest_strike(by_strike: dict[float, OptionQuote], target: float) -> OptionQuote | None:
    if not by_strike:
        return None
    strikes = sorted(by_strike.keys())
    closest = min(strikes, key=lambda k: abs(k - target))
    if abs(closest - target) > 25:
        return None
    return by_strike.get(closest)
