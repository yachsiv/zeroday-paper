"""Strike selector + width adjustment."""

from __future__ import annotations

from datetime import date

import pytest

from zeroday_paper.data.polygon_client import ChainSnapshot, OptionQuote
from zeroday_paper.engine.models import StrategyType
from zeroday_paper.engine.strike_select import (
    SelectionResult,
    _adjusted_width,
    _closest_strike,
    _is_quality,
    _select_credit_spread,
    select_for,
)


# ------------------------------------------------------------------- adjusted_width


def test_adjusted_width_baseline_when_vix_none():
    assert _adjusted_width(25, None) == 25


def test_adjusted_width_widens_in_high_vix():
    assert _adjusted_width(25, 25.0) == 35


def test_adjusted_width_narrows_in_low_vix():
    assert _adjusted_width(25, 10.0) == 20
    assert _adjusted_width(20, 8.0) == 15  # min floor 15


def test_adjusted_width_unchanged_in_mid_vix():
    assert _adjusted_width(25, 15.0) == 25


# ----------------------------------------------------------------------- is_quality


def test_is_quality_passes_with_good_inputs(make_quote):
    q = make_quote(strike=5800, right="P", bid=1.0, ask=1.2, open_interest=500)
    assert _is_quality(q, min_oi=250, max_spread=2.0) is True


def test_is_quality_fails_low_oi(make_quote):
    q = make_quote(strike=5800, right="P", bid=1.0, ask=1.2, open_interest=100)
    assert _is_quality(q, min_oi=250, max_spread=2.0) is False


def test_is_quality_fails_wide_spread(make_quote):
    q = make_quote(strike=5800, right="P", bid=1.0, ask=4.5, open_interest=500)
    assert _is_quality(q, min_oi=250, max_spread=2.0) is False


def test_is_quality_fails_untradable(make_quote):
    q = make_quote(strike=5800, right="P", bid=0.0, ask=1.0)
    assert _is_quality(q, min_oi=250, max_spread=2.0) is False


# --------------------------------------------------------------------- closest_strike


def test_closest_strike_picks_nearest(make_quote):
    quotes = {5750.0: make_quote(strike=5750), 5775.0: make_quote(strike=5775)}
    # 5770 is closer to 5775 than 5750
    result = _closest_strike(quotes, 5770.0)
    assert result.strike == 5775.0


def test_closest_strike_rejects_when_too_far(make_quote):
    quotes = {5500.0: make_quote(strike=5500)}
    # target 5800, closest 5500, |diff| = 300 > 25
    assert _closest_strike(quotes, 5800.0) is None


def test_closest_strike_empty_dict_returns_none():
    assert _closest_strike({}, 5800.0) is None


def test_closest_strike_exact_match(make_quote):
    quotes = {5800.0: make_quote(strike=5800)}
    assert _closest_strike(quotes, 5800.0).strike == 5800.0


# ----------------------------------------------------------------- select_for full flow


def test_select_for_bull_put_returns_spread(make_state):
    state = make_state()
    result = select_for(StrategyType.BULL_PUT, state)
    assert result.spread is not None
    assert result.spread.strategy == StrategyType.BULL_PUT
    assert result.spread.short_leg.right == "P"
    assert result.candidates_considered >= 1


def test_select_for_bear_call_returns_spread(make_state):
    state = make_state()
    result = select_for(StrategyType.BEAR_CALL, state)
    assert result.spread is not None
    assert result.spread.strategy == StrategyType.BEAR_CALL
    assert result.spread.short_leg.right == "C"


def test_select_for_iron_condor_passes_through(make_state):
    state = make_state()
    result = select_for(StrategyType.IRON_CONDOR, state)
    # v1: emits put spread
    assert result.spread is not None
    assert result.spread.strategy == StrategyType.BULL_PUT


def test_select_for_unsupported_strategy_returns_none(make_state):
    state = make_state()
    # Construct a fake strategy
    class _Fake:
        value = "FAKE"
        def __str__(self):
            return "FAKE"
    result = select_for(_Fake(), state)
    assert result.spread is None
    assert any("unsupported" in r for r in result.reasons)


def test_select_for_falls_back_to_moneyness_when_delta_missing(make_chain, make_state):
    """Polygon free-tier returns greeks=None on many strikes. The selector must
    fall back to a moneyness-based picker so we still produce a candidate
    spread instead of silently returning no_short_in_delta_band.

    Regression for 2026-05-27 "0 paper trades for 2 sessions" incident."""
    chain = make_chain()
    no_delta_puts = [
        type(q)(
            contract=q.contract, strike=q.strike, right=q.right, expiry=q.expiry,
            bid=q.bid, ask=q.ask, mid=q.mid, last=q.last,
            iv=q.iv, delta=None, gamma=q.gamma, theta=q.theta, vega=q.vega,
            open_interest=q.open_interest, volume=q.volume,
        )
        for q in chain.puts
    ]
    no_delta_chain = type(chain)(
        fetched_at=chain.fetched_at, spot=chain.spot, expiry=chain.expiry,
        calls=chain.calls, puts=no_delta_puts,
    )
    state = make_state(chain=no_delta_chain)
    result = select_for(StrategyType.BULL_PUT, state)
    assert result.spread is not None, (
        f"moneyness fallback should pick a strike when delta is None; reasons={result.reasons}"
    )
    assert "moneyness_fallback" in " ".join(result.reasons)
    # Short must be strictly OTM (strike < spot) for a bull put.
    assert result.spread.short_leg.strike < state.spot


def test_select_for_returns_none_when_no_strikes_near_spot(make_chain, make_state):
    """If the chain has neither delta nor any strikes near spot, both selectors
    fail and we surface both reasons so the operator can see it."""
    # Chain is centered at 5800; put spot at 7000 so no strike is within the
    # moneyness band either.
    chain = make_chain(spot=5800.0)
    no_delta_puts = [
        type(q)(
            contract=q.contract, strike=q.strike, right=q.right, expiry=q.expiry,
            bid=q.bid, ask=q.ask, mid=q.mid, last=q.last,
            iv=q.iv, delta=None, gamma=q.gamma, theta=q.theta, vega=q.vega,
            open_interest=q.open_interest, volume=q.volume,
        )
        for q in chain.puts
    ]
    no_delta_chain = type(chain)(
        fetched_at=chain.fetched_at, spot=chain.spot, expiry=chain.expiry,
        calls=chain.calls, puts=no_delta_puts,
    )
    # Spot way above all the chain strikes → no put is within 35pt of target_strike
    state = make_state(spot=7000.0, chain=no_delta_chain)
    result = select_for(StrategyType.BULL_PUT, state)
    assert result.spread is None
    assert any("no_short_in_moneyness_band" in r for r in result.reasons)


def test_select_for_quality_gate_rejects_low_oi(make_chain, make_state, make_quote):
    chain = make_chain()
    # Make all puts have low OI
    bad_puts = [
        type(q)(
            contract=q.contract, strike=q.strike, right=q.right, expiry=q.expiry,
            bid=q.bid, ask=q.ask, mid=q.mid, last=q.last,
            iv=q.iv, delta=q.delta, gamma=q.gamma, theta=q.theta, vega=q.vega,
            open_interest=10,  # too low
            volume=q.volume,
        )
        for q in chain.puts
    ]
    bad_chain = type(chain)(
        fetched_at=chain.fetched_at, spot=chain.spot, expiry=chain.expiry,
        calls=chain.calls, puts=bad_puts,
    )
    state = make_state(chain=bad_chain)
    result = select_for(StrategyType.BULL_PUT, state)
    assert result.spread is None
    assert any("all_shorts_failed_quality" in r for r in result.reasons)


def test_select_for_ranks_by_credit_ratio(make_state):
    state = make_state()
    result = select_for(StrategyType.BULL_PUT, state)
    assert result.spread is not None
    # The first candidate should have highest credit_ratio
    assert "selected" in " ".join(result.reasons)


def test_select_for_width_adjusts_with_vix(make_state, make_vols, make_chain):
    # Force high vix → adjusted target width = base+10. Verify selector still returns
    # a spread (actual width depends on available strikes in the chain).
    chain = make_chain(n_strikes=41, step=5)  # broader chain to support wider targets
    state = make_state(vols=make_vols(vix_1d=25.0), chain=chain)
    result = select_for(StrategyType.BULL_PUT, state)
    assert result.spread is not None
    assert result.spread.width > 0
