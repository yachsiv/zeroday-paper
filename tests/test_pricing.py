"""Pricing math: entry, exit, settlement P&L invariants."""

from __future__ import annotations

import pytest

from zeroday_paper.engine.models import Spread, StrategyType
from zeroday_paper.engine.pricing import entry_quote, exit_quote, realized_at_expiry


def _bull_put(make_quote, *, short_strike=5775, long_strike=5750,
              short_bid=1.20, short_ask=1.40, long_bid=0.60, long_ask=0.80,
              contracts=1):
    short = make_quote(strike=short_strike, right="P",
                       bid=short_bid, ask=short_ask, delta=-0.18)
    long = make_quote(strike=long_strike, right="P",
                      bid=long_bid, ask=long_ask, delta=-0.10)
    return Spread(strategy=StrategyType.BULL_PUT, short_leg=short, long_leg=long, contracts=contracts)


def _bear_call(make_quote, *, short_strike=5825, long_strike=5850,
               short_bid=1.20, short_ask=1.40, long_bid=0.60, long_ask=0.80,
               contracts=1):
    short = make_quote(strike=short_strike, right="C",
                       bid=short_bid, ask=short_ask, delta=0.18)
    long = make_quote(strike=long_strike, right="C",
                      bid=long_bid, ask=long_ask, delta=0.10)
    return Spread(strategy=StrategyType.BEAR_CALL, short_leg=short, long_leg=long, contracts=contracts)


# --------------------------------------------------------------------- entry_quote


def test_entry_quote_bull_put_credit_math(make_quote):
    spread = _bull_put(make_quote)
    eq = entry_quote(spread)
    # Invariant: credit_bid == short.bid - long.ask
    assert eq.credit_bid == pytest.approx(1.20 - 0.80)
    # max_loss = width - credit_bid
    assert eq.max_loss_bid == pytest.approx(25.0 - 0.40)
    assert eq.credit_ratio == pytest.approx(0.40 / 25.0)
    assert eq.is_acceptable is True
    assert eq.reasons == []


def test_entry_quote_bear_call_credit_math(make_quote):
    spread = _bear_call(make_quote)
    eq = entry_quote(spread)
    assert eq.credit_bid == pytest.approx(0.40)
    assert eq.max_loss_bid == pytest.approx(24.60)
    assert eq.is_acceptable is True


def test_entry_quote_credit_mid_uses_midprices(make_quote):
    spread = _bull_put(make_quote)
    eq = entry_quote(spread)
    expected_mid = (1.20 + 1.40) / 2 - (0.60 + 0.80) / 2
    assert eq.credit_mid == pytest.approx(expected_mid)


def test_entry_quote_rejects_low_credit(make_quote):
    # tighten spread until credit_bid < min_credit
    spread = _bull_put(make_quote, short_bid=0.85, short_ask=0.95,
                      long_bid=0.80, long_ask=0.90)
    eq = entry_quote(spread, min_credit=0.10)
    # credit_bid = 0.85 - 0.90 = -0.05 → < min_credit
    assert eq.is_acceptable is False
    assert any("credit_bid" in r for r in eq.reasons)


def test_entry_quote_rejects_illiquid_short_leg(make_quote):
    spread = _bull_put(make_quote, short_bid=0.0, short_ask=1.40)
    eq = entry_quote(spread)
    assert eq.is_acceptable is False
    assert any("short illiquid" in r for r in eq.reasons)


def test_entry_quote_rejects_illiquid_long_leg(make_quote):
    spread = _bull_put(make_quote, long_bid=0.0, long_ask=1.0)
    eq = entry_quote(spread)
    assert eq.is_acceptable is False
    assert any("long illiquid" in r for r in eq.reasons)


def test_entry_quote_rejects_wide_short_bid_ask(make_quote):
    spread = _bull_put(make_quote, short_bid=0.50, short_ask=4.0)
    eq = entry_quote(spread, max_bid_ask=2.0)
    assert eq.is_acceptable is False
    assert any("short ba_spread" in r for r in eq.reasons)


def test_entry_quote_rejects_wide_long_bid_ask(make_quote):
    spread = _bull_put(make_quote, long_bid=0.50, long_ask=4.0)
    eq = entry_quote(spread, max_bid_ask=2.0)
    assert eq.is_acceptable is False
    assert any("long ba_spread" in r for r in eq.reasons)


def test_entry_quote_zero_credit_not_acceptable(make_quote):
    # bid == ask on both legs → credit_bid = short.bid - long.ask = exactly 0
    spread = _bull_put(make_quote, short_bid=1.0, short_ask=1.10,
                      long_bid=0.95, long_ask=1.0)
    eq = entry_quote(spread, min_credit=0.01)
    # credit_bid = 1.0 - 1.0 = 0
    assert eq.credit_bid == 0
    assert eq.is_acceptable is False  # credit_bid > 0 fails


def test_entry_quote_zero_width_handled(make_quote):
    short = make_quote(strike=5775, right="P", bid=1.0, ask=1.2)
    long = make_quote(strike=5775, right="P", bid=0.50, ask=0.60)
    spread = Spread(strategy=StrategyType.BULL_PUT, short_leg=short, long_leg=long, contracts=1)
    eq = entry_quote(spread)
    assert eq.credit_ratio == 0.0


# ----------------------------------------------------------------------- exit_quote


def test_exit_quote_cost_and_pnl_math(make_quote):
    spread = _bull_put(make_quote, short_bid=0.20, short_ask=0.30,
                      long_bid=0.10, long_ask=0.15)
    eq = exit_quote(spread, entry_credit_bid=0.40)
    # cost_bid = short.ask - long.bid = 0.30 - 0.10 = 0.20
    assert eq.cost_bid == pytest.approx(0.20)
    # pnl = (0.40 - 0.20) * 100 * 1 = 20.0
    assert eq.pnl_bid == pytest.approx(20.0)
    # pct = (0.40 - 0.20) / 0.40 = 0.5
    assert eq.pct_of_max_profit_bid == pytest.approx(0.5)


def test_exit_quote_loss_when_cost_above_credit(make_quote):
    spread = _bull_put(make_quote, short_bid=1.5, short_ask=2.0,
                      long_bid=0.30, long_ask=0.50)
    eq = exit_quote(spread, entry_credit_bid=0.40)
    # cost_bid = 2.0 - 0.30 = 1.70
    assert eq.cost_bid == pytest.approx(1.70)
    # pnl = (0.40 - 1.70) * 100 = -130.0
    assert eq.pnl_bid == pytest.approx(-130.0)
    assert eq.pct_of_max_profit_bid < 0


def test_exit_quote_scales_with_contracts(make_quote):
    spread = _bull_put(make_quote, short_bid=0.20, short_ask=0.30,
                      long_bid=0.10, long_ask=0.15, contracts=5)
    eq = exit_quote(spread, entry_credit_bid=0.40)
    assert eq.pnl_bid == pytest.approx(20.0 * 5)


def test_exit_quote_pct_of_max_profit_zero_when_no_credit(make_quote):
    spread = _bull_put(make_quote)
    eq = exit_quote(spread, entry_credit_bid=0.0)
    assert eq.pct_of_max_profit_bid == 0.0


# ------------------------------------------------------------- realized_at_expiry


def test_realized_at_expiry_bull_put_full_profit(make_quote):
    spread = _bull_put(make_quote)
    pnl = realized_at_expiry(spread, entry_credit_bid=0.40, settlement_spot=5900.0)
    # spot >> short_strike → intrinsic 0 → keep full credit
    assert pnl == pytest.approx(0.40 * 100)


def test_realized_at_expiry_bull_put_full_loss(make_quote):
    spread = _bull_put(make_quote)
    pnl = realized_at_expiry(spread, entry_credit_bid=0.40, settlement_spot=5700.0)
    # spot < long_strike → intrinsic = width → pnl = (0.40 - 25) * 100 = -2460
    assert pnl == pytest.approx((0.40 - 25.0) * 100)


def test_realized_at_expiry_bull_put_partial(make_quote):
    spread = _bull_put(make_quote)
    # spot = 5760 → between long_strike (5750) and short_strike (5775)
    # intrinsic = short_strike - spot = 15
    pnl = realized_at_expiry(spread, entry_credit_bid=0.40, settlement_spot=5760.0)
    assert pnl == pytest.approx((0.40 - 15.0) * 100)


def test_realized_at_expiry_bear_call_full_profit(make_quote):
    spread = _bear_call(make_quote)
    pnl = realized_at_expiry(spread, entry_credit_bid=0.40, settlement_spot=5700.0)
    # spot <= short_k → intrinsic 0 → full credit
    assert pnl == pytest.approx(0.40 * 100)


def test_realized_at_expiry_bear_call_full_loss(make_quote):
    spread = _bear_call(make_quote)
    pnl = realized_at_expiry(spread, entry_credit_bid=0.40, settlement_spot=5900.0)
    assert pnl == pytest.approx((0.40 - 25.0) * 100)


def test_realized_at_expiry_bear_call_partial(make_quote):
    spread = _bear_call(make_quote)
    # spot = 5835 → between short (5825) and long (5850), intrinsic = 10
    pnl = realized_at_expiry(spread, entry_credit_bid=0.40, settlement_spot=5835.0)
    assert pnl == pytest.approx((0.40 - 10.0) * 100)


def test_realized_at_expiry_iron_condor_not_implemented(make_quote):
    short = make_quote(strike=5800, right="P", bid=1.0, ask=1.2)
    long = make_quote(strike=5775, right="P", bid=0.5, ask=0.6)
    spread = Spread(strategy=StrategyType.IRON_CONDOR, short_leg=short, long_leg=long)
    with pytest.raises(NotImplementedError):
        realized_at_expiry(spread, entry_credit_bid=0.40, settlement_spot=5800.0)


def test_realized_at_expiry_scales_with_contracts(make_quote):
    spread = _bull_put(make_quote, contracts=3)
    pnl = realized_at_expiry(spread, entry_credit_bid=0.40, settlement_spot=5900.0)
    assert pnl == pytest.approx(0.40 * 100 * 3)
