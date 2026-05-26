"""Position state machine.

Given an open paper trade + a fresh ExitQuote + a clock, decide whether to
HOLD or transition to one of the CLOSED_* states.

Pure function — no I/O. The scanner/monitor calls this and persists the result.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from zoneinfo import ZoneInfo

from zeroday_paper.config import settings
from zeroday_paper.engine.models import PositionStatus

ET = ZoneInfo(settings.engine.market_timezone)


@dataclass(frozen=True)
class ExitDecision:
    next_status: PositionStatus
    should_exit: bool
    reason: str


def profit_target_for(now_et: datetime) -> float:
    """Time-decayed profit target. Tighter as the day progresses."""
    t = now_et.time()
    e = settings.exits
    if t < time(11, 0):
        return e.profit_target_pre_11am
    if t < time(13, 0):
        return e.profit_target_11_to_1pm
    if t < time(14, 0):
        return e.profit_target_1_to_2pm
    return e.profit_target_after_2pm


def decide_exit(
    *,
    now_utc: datetime,
    entry_credit_bid: float,
    exit_cost_bid: float,
    pct_of_max_profit: float,
    thesis_invalidated: bool = False,
) -> ExitDecision:
    """Pure transition: returns next status + whether to close."""
    now_et = now_utc.astimezone(ET)

    if now_et.time() >= settings.exits.hard_close:
        return ExitDecision(PositionStatus.CLOSED_HARD, True, "hard_close_15_45_ET")

    if thesis_invalidated:
        return ExitDecision(
            PositionStatus.CLOSED_THESIS, True, "thesis_invalidated"
        )

    target = profit_target_for(now_et)
    if pct_of_max_profit >= target:
        return ExitDecision(
            PositionStatus.CLOSED_TARGET,
            True,
            f"profit_target_hit_{int(target*100)}pct",
        )

    loss_multiple = settings.exits.stop_loss_multiple
    if exit_cost_bid >= entry_credit_bid * (1.0 + loss_multiple):
        return ExitDecision(
            PositionStatus.CLOSED_STOP,
            True,
            f"stop_loss_{loss_multiple}x_credit",
        )

    if pct_of_max_profit > 0.0:
        return ExitDecision(PositionStatus.WINNING, False, "holding_in_profit")
    return ExitDecision(PositionStatus.LOSING, False, "holding_in_drawdown")
