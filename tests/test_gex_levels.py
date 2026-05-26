"""GEX level proximity helpers."""

from __future__ import annotations

import pytest

from zeroday_paper.engine.gex_levels import compute_proximity, compute_rr25, _iv_at_delta


def test_compute_proximity_basic(make_state, make_signals):
    state = make_state(
        spot=5800.0,
        signals=make_signals(
            spot=5800.0, gamma_flip=5790.0, call_wall=5850.0,
            put_wall=5750.0, magnet_strike=5800.0,
        ),
    )
    prox = compute_proximity(state)
    assert prox.to_gamma_flip == 10.0
    assert prox.to_call_wall == -50.0
    assert prox.to_put_wall == 50.0
    assert prox.to_magnet == 0.0
    assert prox.above_flip is True
    assert prox.below_call_wall is True
    assert prox.above_put_wall is True


def test_compute_proximity_below_flip(make_state, make_signals):
    state = make_state(
        spot=5780.0,
        signals=make_signals(spot=5780.0, gamma_flip=5790.0),
    )
    prox = compute_proximity(state)
    assert prox.above_flip is False


def test_compute_proximity_missing_levels(make_state, make_signals):
    state = make_state(
        signals=make_signals(
            gamma_flip=None, call_wall=None, put_wall=None, magnet_strike=None,
        ),
    )
    prox = compute_proximity(state)
    assert prox.to_gamma_flip is None
    assert prox.to_call_wall is None
    assert prox.to_put_wall is None
    assert prox.to_magnet is None
    assert prox.above_flip is None
    assert prox.below_call_wall is None
    assert prox.above_put_wall is None


def test_compute_rr25_insufficient_data(make_state, make_chain, make_quote):
    # Strip iv from all quotes
    chain = make_chain()
    new_puts = [
        type(q)(
            contract=q.contract, strike=q.strike, right=q.right, expiry=q.expiry,
            bid=q.bid, ask=q.ask, mid=q.mid, last=q.last,
            iv=None, delta=q.delta, gamma=q.gamma, theta=q.theta, vega=q.vega,
            open_interest=q.open_interest, volume=q.volume,
        )
        for q in chain.puts
    ]
    chain_no_iv = type(chain)(
        fetched_at=chain.fetched_at, spot=chain.spot, expiry=chain.expiry,
        calls=chain.calls, puts=new_puts,
    )
    state = make_state(chain=chain_no_iv)
    assert compute_rr25(state) is None


def test_compute_rr25_returns_number_when_sufficient(make_state):
    state = make_state()
    # Default chain has 21 strikes with iv=0.20 on both legs → rr25 = (0.20 - 0.20) * 100 = 0
    val = compute_rr25(state)
    assert val == pytest.approx(0.0)


def test_iv_at_delta_too_few_returns_none():
    assert _iv_at_delta([], 0.25, lambda q: q.delta) is None


def test_iv_at_delta_picks_closest(make_quote):
    quotes = [
        make_quote(strike=5800, delta=0.24, iv=0.21),
        make_quote(strike=5810, delta=0.30, iv=0.22),
        make_quote(strike=5790, delta=0.18, iv=0.19),
    ]
    assert _iv_at_delta(quotes, 0.25, lambda q: q.delta) == pytest.approx(0.21)
