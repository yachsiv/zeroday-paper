"""Engine model dataclasses + enums."""

from __future__ import annotations

from datetime import date

import pytest

from zeroday_paper.engine.models import (
    MarketState,
    PaperTrade,
    PositionStatus,
    Spread,
    StrategyType,
    TradeOutcome,
    TradeTick,
)


def test_spread_width_absolute(make_quote):
    short = make_quote(strike=5775, right="P")
    long = make_quote(strike=5750, right="P")
    s = Spread(strategy=StrategyType.BULL_PUT, short_leg=short, long_leg=long)
    assert s.width == 25.0


def test_spread_width_handles_reversed_strikes_for_bear_call(make_quote):
    short = make_quote(strike=5825, right="C")
    long = make_quote(strike=5850, right="C")
    s = Spread(strategy=StrategyType.BEAR_CALL, short_leg=short, long_leg=long)
    assert s.width == 25.0


def test_spread_expiry_taken_from_short_leg(make_quote):
    short = make_quote(strike=5775, right="P", expiry=date(2025, 6, 16))
    long = make_quote(strike=5750, right="P", expiry=date(2025, 6, 16))
    s = Spread(strategy=StrategyType.BULL_PUT, short_leg=short, long_leg=long)
    assert s.expiry == date(2025, 6, 16)


def test_market_state_is_replay_flag(make_state, make_signals):
    state_live = make_state(signals=make_signals(source="flashalpha"))
    state_replay = make_state(signals=make_signals(source="self_computed"))
    state_replay_empty = make_state(signals=make_signals(source="self_computed_empty"))
    assert state_live.is_replay is False
    assert state_replay.is_replay is True
    assert state_replay_empty.is_replay is True


@pytest.mark.parametrize("strat", list(StrategyType))
def test_strategy_type_values(strat):
    assert isinstance(strat.value, str)


@pytest.mark.parametrize("status", list(PositionStatus))
def test_position_status_values(status):
    assert isinstance(status.value, str)
    assert "CLOSED" in status.value or status.value in {"OPEN", "WINNING", "LOSING", "ERROR"}


def test_paper_trade_dataclass_constructs(make_paper_trade):
    pt = make_paper_trade()
    assert isinstance(pt, PaperTrade)
    assert pt.trade_id == "trade_abc123"


def test_trade_outcome_dataclass(make_paper_trade):
    pt = make_paper_trade()
    o = TradeOutcome(
        trade_id=pt.trade_id, exit_ts=pt.entry_ts,
        exit_status=PositionStatus.CLOSED_TARGET,
        exit_spot=5800.0, exit_cost_bid=0.5, exit_cost_mid=0.55,
        pnl_bid=50.0, pnl_mid=55.0, held_minutes=60,
        max_excursion_pct=0.6, min_excursion_pct=-0.2,
        exit_reason="profit_target",
    )
    assert o.exit_status == PositionStatus.CLOSED_TARGET


def test_trade_tick_dataclass(make_paper_trade):
    pt = make_paper_trade()
    t = TradeTick(
        trade_id=pt.trade_id, ts=pt.entry_ts, spot=5800.0,
        short_bid=0.3, short_ask=0.4, long_bid=0.1, long_ask=0.15,
        exit_cost_bid=0.30, exit_cost_mid=0.25, pnl_bid=10.0, pnl_mid=12.0,
        pct_of_max_profit=0.5, status=PositionStatus.WINNING,
    )
    assert t.status == PositionStatus.WINNING
