"""Bid/ask pricing helpers.

Single rule, applied everywhere: paper trades use realistic fills.

    Entry credit (we are selling the spread, so we get the worse of bid):
        credit_bid = short.bid - long.ask

    Exit cost (we are buying it back):
        exit_cost_bid = short.ask - long.bid

Mid prices are computed too for diagnostic comparison but **never** drive
exit decisions or stored P&L.
"""

from __future__ import annotations

from dataclasses import dataclass

from zeroday_paper.data.polygon_client import OptionQuote
from zeroday_paper.engine.models import Spread


@dataclass(frozen=True)
class EntryQuote:
    credit_bid: float
    credit_mid: float
    max_loss_bid: float
    credit_ratio: float          # credit / max_loss
    is_acceptable: bool
    reasons: list[str]


@dataclass(frozen=True)
class ExitQuote:
    cost_bid: float
    cost_mid: float
    pnl_bid: float
    pnl_mid: float
    pct_of_max_profit_bid: float


def entry_quote(spread: Spread, *, min_credit: float = 0.10, max_bid_ask: float = 2.0) -> EntryQuote:
    """Compute realistic entry credit + acceptance flags."""
    short, long = spread.short_leg, spread.long_leg

    credit_bid = round(short.bid - long.ask, 4)
    credit_mid = round(short.mid - long.mid, 4)
    width = spread.width
    max_loss_bid = round(max(0.0, width - credit_bid), 4)
    credit_ratio = credit_bid / width if width > 0 else 0.0

    reasons: list[str] = []
    if credit_bid < min_credit:
        reasons.append(f"credit_bid {credit_bid} < {min_credit}")
    if not short.is_tradable:
        reasons.append(f"short illiquid: bid={short.bid} ask={short.ask}")
    if not long.is_tradable:
        reasons.append(f"long illiquid: bid={long.bid} ask={long.ask}")
    if short.bid_ask_spread > max_bid_ask:
        reasons.append(f"short ba_spread {short.bid_ask_spread} > {max_bid_ask}")
    if long.bid_ask_spread > max_bid_ask:
        reasons.append(f"long ba_spread {long.bid_ask_spread} > {max_bid_ask}")
    if max_loss_bid <= 0:
        reasons.append("max_loss <= 0 (impossible)")

    return EntryQuote(
        credit_bid=credit_bid,
        credit_mid=credit_mid,
        max_loss_bid=max_loss_bid,
        credit_ratio=credit_ratio,
        is_acceptable=not reasons and credit_bid > 0,
        reasons=reasons,
    )


def exit_quote(spread: Spread, *, entry_credit_bid: float) -> ExitQuote:
    """Compute realistic close cost + P&L vs the original credit."""
    short, long = spread.short_leg, spread.long_leg

    cost_bid = round(short.ask - long.bid, 4)        # worse side: buy short, sell long
    cost_mid = round(short.mid - long.mid, 4)
    pnl_bid = round((entry_credit_bid - cost_bid) * 100.0 * spread.contracts, 4)
    pnl_mid = round((entry_credit_bid - cost_mid) * 100.0 * spread.contracts, 4)

    pct_max_profit = (entry_credit_bid - cost_bid) / entry_credit_bid if entry_credit_bid > 0 else 0.0
    return ExitQuote(
        cost_bid=cost_bid,
        cost_mid=cost_mid,
        pnl_bid=pnl_bid,
        pnl_mid=pnl_mid,
        pct_of_max_profit_bid=round(pct_max_profit, 4),
    )


def realized_at_expiry(spread: Spread, *, entry_credit_bid: float, settlement_spot: float) -> float:
    """Final P&L if the spread expires.

    Bull put: max loss if spot < long_strike. Max profit if spot > short_strike.
    Bear call: mirror.
    """
    from zeroday_paper.engine.models import StrategyType

    short_k = spread.short_leg.strike
    long_k = spread.long_leg.strike
    width = spread.width

    if spread.strategy == StrategyType.BULL_PUT:
        if settlement_spot >= short_k:
            intrinsic_cost = 0.0
        elif settlement_spot <= long_k:
            intrinsic_cost = width
        else:
            intrinsic_cost = short_k - settlement_spot
    elif spread.strategy == StrategyType.BEAR_CALL:
        if settlement_spot <= short_k:
            intrinsic_cost = 0.0
        elif settlement_spot >= long_k:
            intrinsic_cost = width
        else:
            intrinsic_cost = settlement_spot - short_k
    else:
        raise NotImplementedError(f"realized_at_expiry not implemented for {spread.strategy}")

    pnl = (entry_credit_bid - intrinsic_cost) * 100.0 * spread.contracts
    return round(pnl, 4)
