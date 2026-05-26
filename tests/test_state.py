"""Position state machine: profit targets, stop-loss, hard close, thesis."""

from __future__ import annotations

from datetime import UTC, datetime, time
from zoneinfo import ZoneInfo

import pytest

from zeroday_paper.config import settings
from zeroday_paper.engine.models import PositionStatus
from zeroday_paper.engine.state import ExitDecision, decide_exit, profit_target_for

ET = ZoneInfo("America/New_York")


def _et(hour: int, minute: int) -> datetime:
    return datetime(2025, 5, 28, hour, minute, tzinfo=ET).astimezone(UTC)


# ---------------------------------------------------------------- profit_target_for


@pytest.mark.parametrize(
    "h,m,expected",
    [
        (9, 45, settings.exits.profit_target_pre_11am),
        (10, 59, settings.exits.profit_target_pre_11am),
        (11, 0, settings.exits.profit_target_11_to_1pm),
        (12, 30, settings.exits.profit_target_11_to_1pm),
        (13, 0, settings.exits.profit_target_1_to_2pm),
        (13, 59, settings.exits.profit_target_1_to_2pm),
        (14, 0, settings.exits.profit_target_after_2pm),
        (15, 30, settings.exits.profit_target_after_2pm),
    ],
)
def test_profit_target_buckets(h, m, expected):
    now_et = datetime(2025, 5, 28, h, m, tzinfo=ET)
    assert profit_target_for(now_et) == expected


# ----------------------------------------------------------------- decide_exit


def test_decide_exit_hard_close_fires_at_or_after_15_45():
    d = decide_exit(
        now_utc=_et(15, 45),
        entry_credit_bid=1.0, exit_cost_bid=0.5,
        pct_of_max_profit=0.5,
    )
    assert d.should_exit is True
    assert d.next_status == PositionStatus.CLOSED_HARD
    assert "hard_close" in d.reason


def test_decide_exit_hard_close_overrides_thesis():
    d = decide_exit(
        now_utc=_et(15, 50),
        entry_credit_bid=1.0, exit_cost_bid=10.0,
        pct_of_max_profit=-9.0,
        thesis_invalidated=True,
    )
    # Hard close evaluated first
    assert d.next_status == PositionStatus.CLOSED_HARD


def test_decide_exit_thesis_invalidated_before_hard_close():
    d = decide_exit(
        now_utc=_et(12, 0),
        entry_credit_bid=1.0, exit_cost_bid=0.7,
        pct_of_max_profit=0.30,
        thesis_invalidated=True,
    )
    assert d.next_status == PositionStatus.CLOSED_THESIS
    assert d.should_exit is True


def test_decide_exit_profit_target_pre_11am():
    # target before 11 = 0.75
    d = decide_exit(
        now_utc=_et(10, 30),
        entry_credit_bid=1.0, exit_cost_bid=0.20,
        pct_of_max_profit=0.80,
    )
    assert d.next_status == PositionStatus.CLOSED_TARGET
    assert d.should_exit is True
    assert "75pct" in d.reason


def test_decide_exit_profit_target_after_2pm_uses_lower_target():
    # after 14:00 target = 0.40
    d = decide_exit(
        now_utc=_et(14, 30),
        entry_credit_bid=1.0, exit_cost_bid=0.55,
        pct_of_max_profit=0.45,
    )
    assert d.next_status == PositionStatus.CLOSED_TARGET
    assert "40pct" in d.reason


def test_decide_exit_stop_loss_triggered_at_2x():
    # cost = 2.0 * entry_credit (=1.0) * (1+2.0) = 3.0  → trigger
    d = decide_exit(
        now_utc=_et(11, 30),
        entry_credit_bid=1.0,
        exit_cost_bid=3.0,
        pct_of_max_profit=-2.0,
    )
    assert d.next_status == PositionStatus.CLOSED_STOP
    assert "stop_loss" in d.reason


def test_decide_exit_holding_in_profit():
    d = decide_exit(
        now_utc=_et(11, 30),
        entry_credit_bid=1.0, exit_cost_bid=0.80,
        pct_of_max_profit=0.20,
    )
    assert d.should_exit is False
    assert d.next_status == PositionStatus.WINNING


def test_decide_exit_holding_in_drawdown():
    d = decide_exit(
        now_utc=_et(11, 30),
        entry_credit_bid=1.0, exit_cost_bid=1.20,
        pct_of_max_profit=-0.2,
    )
    assert d.should_exit is False
    assert d.next_status == PositionStatus.LOSING


def test_decide_exit_zero_pct_classified_as_drawdown():
    d = decide_exit(
        now_utc=_et(11, 30),
        entry_credit_bid=1.0, exit_cost_bid=1.0,
        pct_of_max_profit=0.0,
    )
    assert d.next_status == PositionStatus.LOSING


def test_decide_exit_returns_namedtuple_like_object():
    d = decide_exit(
        now_utc=_et(11, 30),
        entry_credit_bid=1.0, exit_cost_bid=0.5,
        pct_of_max_profit=0.5,
    )
    assert isinstance(d, ExitDecision)
    assert isinstance(d.next_status, PositionStatus)


# Regression: total transition function — every input gets a decision
def test_decide_exit_is_total_over_sample_grid():
    sample_times = [(10, 0), (11, 30), (13, 30), (14, 30), (15, 30), (15, 45), (15, 50)]
    sample_pnls = [-0.5, 0.0, 0.4, 0.8]
    for h, m in sample_times:
        for pct in sample_pnls:
            cost = 1.0 - pct
            d = decide_exit(
                now_utc=_et(h, m),
                entry_credit_bid=1.0, exit_cost_bid=cost,
                pct_of_max_profit=pct,
            )
            assert isinstance(d, ExitDecision)
            assert d.next_status in PositionStatus
