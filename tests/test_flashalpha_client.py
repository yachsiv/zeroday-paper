"""FlashAlpha client + self-computed signals from chain."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Callable

import httpx
import pytest

from zeroday_paper.data import flashalpha_client as fa
from zeroday_paper.data.flashalpha_client import (
    FlashAlphaClient,
    FlashAlphaError,
    MarketSignals,
    _approx_gamma_flip,
    _approx_magnet,
    _approx_max_pain,
    _approx_pin_score,
    _empty_signals,
    _f,
    signals_from_chain,
)
from zeroday_paper.data.polygon_client import ChainSnapshot


def _install_transport(monkeypatch, handler):
    real_asyncclient = httpx.AsyncClient

    def fake_async_client(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_asyncclient(*args, **kwargs)

    monkeypatch.setattr(fa.httpx, "AsyncClient", fake_async_client)


# ----------------------------------------------------------------- _f helper


def test_f_none():
    assert _f(None) is None


def test_f_valid():
    assert _f("1.5") == 1.5
    assert _f(3) == 3.0


def test_f_invalid():
    assert _f("abc") is None
    assert _f({}) is None


# --------------------------------------------------------------- HTTP behavior


@pytest.mark.asyncio
async def test_flashalpha_get_signals_success(monkeypatch):
    def handler(req: httpx.Request) -> httpx.Response:
        path = req.url.path
        # Verify auth header
        assert req.headers["authorization"].startswith("Bearer ")
        if path == "/v1/zero_dte":
            return httpx.Response(200, json={
                "spot": 5800.0,
                "gamma_regime": "Positive Gamma",
                "magnet_strike": 5800.0,
                "pin_score": 70.0,
                "zero_dte_gex_share": 0.7,
                "remaining_1sd": 20.0,
                "full_day_1sd": 40.0,
                "hours_remaining": 4.0,
                "max_pain": 5805.0,
            })
        if path == "/v1/exposure_levels":
            return httpx.Response(200, json={
                "gamma_flip": 5790.0,
                "call_wall": 5850.0,
                "put_wall": 5750.0,
                "total_gex": 2.0,
            })
        return httpx.Response(404)

    _install_transport(monkeypatch, handler)
    async with FlashAlphaClient(api_key="test") as c:
        signals = await c.get_signals("SPX")

    assert signals.source == "flashalpha"
    assert signals.spot == 5800.0
    assert signals.gamma_regime == "positive_gamma"
    assert signals.gamma_flip == 5790.0
    assert signals.call_wall == 5850.0
    assert signals.put_wall == 5750.0
    assert signals.pin_score == 70.0
    assert signals.total_gex == 2.0
    assert signals.raw is not None
    assert "zero_dte" in signals.raw


@pytest.mark.asyncio
async def test_flashalpha_negative_gamma_parsed(monkeypatch):
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/v1/zero_dte":
            return httpx.Response(200, json={"spot": 5800, "gamma_regime": "Negative GEX"})
        return httpx.Response(200, json={})

    _install_transport(monkeypatch, handler)
    async with FlashAlphaClient(api_key="test") as c:
        signals = await c.get_signals()
    assert signals.gamma_regime == "negative_gamma"


@pytest.mark.asyncio
async def test_flashalpha_neutral_regime_when_unknown(monkeypatch):
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/v1/zero_dte":
            return httpx.Response(200, json={"spot": 5800, "gamma_regime": "?"})
        return httpx.Response(200, json={})

    _install_transport(monkeypatch, handler)
    async with FlashAlphaClient(api_key="test") as c:
        signals = await c.get_signals()
    assert signals.gamma_regime == "neutral"


@pytest.mark.asyncio
async def test_flashalpha_auth_error_on_401(monkeypatch):
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "bad key"})

    _install_transport(monkeypatch, handler)
    async with FlashAlphaClient(api_key="test") as c:
        with pytest.raises(FlashAlphaError):
            await c.get_signals()


@pytest.mark.asyncio
async def test_flashalpha_auth_error_on_403(monkeypatch):
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"error": "forbidden"})

    _install_transport(monkeypatch, handler)
    async with FlashAlphaClient(api_key="test") as c:
        with pytest.raises(FlashAlphaError):
            await c.get_signals()


@pytest.mark.asyncio
async def test_flashalpha_spot_fallback_to_underlying_price(monkeypatch):
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/v1/zero_dte":
            return httpx.Response(200, json={"underlying_price": 5810, "gamma_regime": "positive"})
        return httpx.Response(200, json={})

    _install_transport(monkeypatch, handler)
    async with FlashAlphaClient(api_key="test") as c:
        signals = await c.get_signals()
    assert signals.spot == 5810.0


# --------------------------------------------------------- signals_from_chain


def test_signals_from_chain_basic(make_chain):
    chain = make_chain(spot=5800.0, n_strikes=11)
    signals = signals_from_chain(chain)
    assert signals.source == "self_computed"
    assert signals.spot == 5800.0
    assert signals.gamma_regime in ("positive_gamma", "negative_gamma")
    assert signals.total_gex is not None
    assert signals.call_wall is not None
    assert signals.put_wall is not None
    assert signals.max_pain is not None
    assert signals.magnet_strike is not None
    assert signals.pin_score is not None
    assert signals.zero_dte_gex_share == 1.0


def test_signals_from_chain_zero_spot_returns_empty(make_chain):
    chain = make_chain()
    chain_zero = type(chain)(
        fetched_at=chain.fetched_at, spot=0.0, expiry=chain.expiry,
        calls=chain.calls, puts=chain.puts,
    )
    signals = signals_from_chain(chain_zero)
    assert signals.source == "self_computed_empty"
    assert signals.spot == 0.0
    assert signals.total_gex is None


def test_signals_from_chain_skips_quotes_with_no_gamma(make_chain):
    chain = make_chain()
    # Build a new chain with all gamma stripped → call_gex/put_gex empty
    blank_calls = [
        type(q)(
            contract=q.contract, strike=q.strike, right=q.right, expiry=q.expiry,
            bid=q.bid, ask=q.ask, mid=q.mid, last=q.last,
            iv=q.iv, delta=q.delta, gamma=None, theta=q.theta, vega=q.vega,
            open_interest=q.open_interest, volume=q.volume,
        )
        for q in chain.calls
    ]
    blank_puts = [
        type(q)(
            contract=q.contract, strike=q.strike, right=q.right, expiry=q.expiry,
            bid=q.bid, ask=q.ask, mid=q.mid, last=q.last,
            iv=q.iv, delta=q.delta, gamma=None, theta=q.theta, vega=q.vega,
            open_interest=q.open_interest, volume=q.volume,
        )
        for q in chain.puts
    ]
    chain_no_gamma = type(chain)(
        fetched_at=chain.fetched_at, spot=chain.spot, expiry=chain.expiry,
        calls=blank_calls, puts=blank_puts,
    )
    signals = signals_from_chain(chain_no_gamma)
    assert signals.total_gex == 0.0
    assert signals.call_wall is None  # empty call_gex dict
    assert signals.put_wall is None


def test_empty_signals_fields():
    es = _empty_signals("SPX", 0.0)
    assert es.source == "self_computed_empty"
    assert es.total_gex is None
    assert es.spot == 0.0
    assert es.gamma_regime == "neutral"


# ----------------------------------------------------------- approx helpers


def test_approx_gamma_flip_finds_sign_change():
    cgex = {5790.0: 1.0, 5800.0: 2.0, 5810.0: 1.0}
    pgex = {5790.0: 0.5, 5800.0: 3.0, 5810.0: 1.0}
    flip = _approx_gamma_flip(cgex, pgex)
    # net@5790 = 0.5, net@5800 = -1.0 → sign change at 5800
    assert flip == 5800.0


def test_approx_gamma_flip_returns_none_when_no_change():
    cgex = {5790.0: 1.0, 5800.0: 2.0}
    pgex = {5790.0: 0.5, 5800.0: 0.6}
    assert _approx_gamma_flip(cgex, pgex) is None


def test_approx_gamma_flip_returns_none_too_few_strikes():
    assert _approx_gamma_flip({5800.0: 1.0}, {}) is None


def test_approx_max_pain(make_chain):
    chain = make_chain(spot=5800.0, n_strikes=11)
    mp = _approx_max_pain(chain)
    assert mp is not None


def test_approx_max_pain_empty_chain():
    empty = ChainSnapshot(fetched_at=datetime(2025, 5, 28, tzinfo=UTC), spot=5800.0, expiry=None, calls=[], puts=[])
    assert _approx_max_pain(empty) is None


def test_approx_magnet(make_chain):
    chain = make_chain()
    m = _approx_magnet(chain)
    assert m is not None


def test_approx_magnet_zero_spot():
    chain = ChainSnapshot(fetched_at=datetime(2025, 5, 28, tzinfo=UTC), spot=0.0, expiry=None, calls=[], puts=[])
    assert _approx_magnet(chain) is None


def test_approx_magnet_no_oi_in_band(make_chain, make_quote):
    chain = make_chain()
    # Replace all calls/puts so none have strikes within ±50 of spot.
    far_calls = [make_quote(strike=5000, right="C", open_interest=100)]
    far_puts = [make_quote(strike=5000, right="P", open_interest=100)]
    chain2 = type(chain)(
        fetched_at=chain.fetched_at, spot=chain.spot, expiry=chain.expiry,
        calls=far_calls, puts=far_puts,
    )
    assert _approx_magnet(chain2) is None


def test_approx_pin_score():
    assert _approx_pin_score(5800.0, 5800.0) == 100.0
    # 5pt diff → 100 - 5*4 = 80
    assert _approx_pin_score(5805.0, 5800.0) == 80.0
    # Floor at 0
    assert _approx_pin_score(5900.0, 5800.0) == 0.0


def test_approx_pin_score_handles_none():
    assert _approx_pin_score(5800.0, None) is None


def test_approx_pin_score_zero_spot():
    assert _approx_pin_score(0.0, 5800.0) is None
