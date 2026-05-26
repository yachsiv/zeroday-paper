"""FlashAlpha client (SDK-wrapped) + self-computed signals from chain."""

from __future__ import annotations

import sys
import types
from datetime import UTC, datetime

import pytest

from zeroday_paper.data import flashalpha_client as fa
from zeroday_paper.data.flashalpha_client import (
    FlashAlphaClient,
    FlashAlphaError,
    _approx_gamma_flip,
    _approx_magnet,
    _approx_max_pain,
    _approx_pin_score,
    _empty_signals,
    _f,
    signals_from_chain,
)
from zeroday_paper.data.polygon_client import ChainSnapshot


# --------------------------------------------------------------------- SDK mock
#
# The production FlashAlphaClient now wraps the official ``flashalpha`` Python
# SDK (sync) via ``asyncio.to_thread``. We install a fake ``flashalpha`` module
# whose ``FlashAlpha`` factory returns a stub exposing ``zero_dte`` and
# ``exposure_levels`` — exactly the surface the client touches.


def _install_sdk_mock(monkeypatch, *, zero_dte=None, exposure_levels=None, raise_on=None):
    """Install a fake ``flashalpha.FlashAlpha`` with deterministic payloads.

    ``raise_on`` is one of ``None | "zero_dte" | "exposure_levels"``; matching
    method raises a fake SDK error instead of returning data.
    """
    zero_dte_payload = zero_dte if zero_dte is not None else {
        "underlying_price": 7519.12,
        "regime": {
            "label": "positive_gamma",
            "gamma_flip": 7518.71,
        },
        "pin_risk": {
            "magnet_strike": 7515,
            "pin_score": 85,
            "max_pain": 6960,
        },
        "expected_move": {
            "implied_1sd_dollars": 51.7,
            "remaining_1sd_dollars": 12.3,
        },
        "vol_context": {"vix": 18.3},
        "time_to_close_hours": 4.0,
        "exposures": {
            "pct_of_total_gex": 36.9,
            "total_chain_net_gex": 5.022e9,
        },
    }
    exposure_payload = exposure_levels if exposure_levels is not None else {
        "levels": {
            "gamma_flip": 7399.51,
            "call_wall": 8200,
            "put_wall": 7395,
            "zero_dte_magnet": 7515,
        },
    }

    class _StubFA:
        def __init__(self, api_key):
            self.api_key = api_key

        def zero_dte(self, symbol):
            if raise_on == "zero_dte":
                raise RuntimeError("sdk boom (zero_dte)")
            return zero_dte_payload

        def exposure_levels(self, symbol):
            if raise_on == "exposure_levels":
                raise RuntimeError("sdk boom (exposure_levels)")
            return exposure_payload

    fake_mod = types.ModuleType("flashalpha")
    fake_mod.FlashAlpha = _StubFA
    monkeypatch.setitem(sys.modules, "flashalpha", fake_mod)


# ----------------------------------------------------------------- _f helper


def test_f_none():
    assert _f(None) is None


def test_f_valid():
    assert _f("1.5") == 1.5
    assert _f(3) == 3.0


def test_f_invalid():
    assert _f("abc") is None
    assert _f({}) is None


# ---------------------------------------------------------------- SDK behavior


@pytest.mark.asyncio
async def test_flashalpha_get_signals_success(monkeypatch):
    _install_sdk_mock(monkeypatch)
    async with FlashAlphaClient(api_key="test") as c:
        signals = await c.get_signals("SPX")

    assert signals.source == "flashalpha"
    assert signals.spot == 7519.12
    assert signals.gamma_regime == "positive_gamma"
    # exposure_levels.gamma_flip preferred over regime.gamma_flip
    assert signals.gamma_flip == 7399.51
    assert signals.call_wall == 8200.0
    assert signals.put_wall == 7395.0
    assert signals.pin_score == 85.0
    assert signals.magnet_strike == 7515.0
    assert signals.max_pain == 6960.0
    # total_chain_net_gex (dollars) → billions
    assert signals.total_gex == pytest.approx(5.022, rel=1e-3)
    # pct_of_total_gex (0..100) → 0..1
    assert signals.zero_dte_gex_share == pytest.approx(0.369, rel=1e-3)
    assert signals.remaining_1sd == 12.3
    assert signals.full_day_1sd == 51.7
    assert signals.hours_remaining == 4.0
    assert signals.raw is not None
    assert "zero_dte" in signals.raw and "exposure_levels" in signals.raw


@pytest.mark.asyncio
async def test_flashalpha_negative_gamma_parsed(monkeypatch):
    _install_sdk_mock(monkeypatch, zero_dte={
        "underlying_price": 5800.0,
        "regime": {"label": "Negative_Gamma"},
    })
    async with FlashAlphaClient(api_key="test") as c:
        signals = await c.get_signals()
    assert signals.gamma_regime == "negative_gamma"


@pytest.mark.asyncio
async def test_flashalpha_neutral_regime_when_unknown(monkeypatch):
    _install_sdk_mock(monkeypatch, zero_dte={
        "underlying_price": 5800.0,
        "regime": {"label": "?"},
    })
    async with FlashAlphaClient(api_key="test") as c:
        signals = await c.get_signals()
    assert signals.gamma_regime == "neutral"


@pytest.mark.asyncio
async def test_flashalpha_zero_dte_failure_raises(monkeypatch):
    _install_sdk_mock(monkeypatch, raise_on="zero_dte")
    async with FlashAlphaClient(api_key="test") as c:
        with pytest.raises(FlashAlphaError):
            await c.get_signals()


@pytest.mark.asyncio
async def test_flashalpha_exposure_levels_failure_raises(monkeypatch):
    _install_sdk_mock(monkeypatch, raise_on="exposure_levels")
    async with FlashAlphaClient(api_key="test") as c:
        with pytest.raises(FlashAlphaError):
            await c.get_signals()


@pytest.mark.asyncio
async def test_flashalpha_spot_fallback_to_spot_key(monkeypatch):
    # SDK has historically used both `underlying_price` and `spot`; ensure we
    # tolerate the legacy key.
    _install_sdk_mock(monkeypatch, zero_dte={
        "spot": 5810.0,
        "regime": {"label": "positive_gamma"},
    })
    async with FlashAlphaClient(api_key="test") as c:
        signals = await c.get_signals()
    assert signals.spot == 5810.0


@pytest.mark.asyncio
async def test_flashalpha_missing_sdk_raises(monkeypatch):
    # Force the lazy ``from flashalpha import FlashAlpha`` to fail.
    monkeypatch.setitem(sys.modules, "flashalpha", None)
    with pytest.raises(FlashAlphaError):
        async with FlashAlphaClient(api_key="test"):
            pass


@pytest.mark.asyncio
async def test_flashalpha_falls_back_to_regime_flip_when_levels_blank(monkeypatch):
    _install_sdk_mock(monkeypatch, exposure_levels={"levels": {}})
    async with FlashAlphaClient(api_key="test") as c:
        signals = await c.get_signals()
    # Falls through to regime.gamma_flip from the canned zero_dte payload
    assert signals.gamma_flip == 7518.71


@pytest.mark.asyncio
async def test_flashalpha_handles_missing_exposures_block(monkeypatch):
    _install_sdk_mock(monkeypatch, zero_dte={
        "underlying_price": 5800.0,
        "regime": {"label": "positive_gamma"},
    })
    async with FlashAlphaClient(api_key="test") as c:
        signals = await c.get_signals()
    assert signals.total_gex is None
    assert signals.zero_dte_gex_share is None


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
    assert signals.call_wall is None
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
    far_calls = [make_quote(strike=5000, right="C", open_interest=100)]
    far_puts = [make_quote(strike=5000, right="P", open_interest=100)]
    chain2 = type(chain)(
        fetched_at=chain.fetched_at, spot=chain.spot, expiry=chain.expiry,
        calls=far_calls, puts=far_puts,
    )
    assert _approx_magnet(chain2) is None


def test_approx_pin_score():
    assert _approx_pin_score(5800.0, 5800.0) == 100.0
    assert _approx_pin_score(5805.0, 5800.0) == 80.0
    assert _approx_pin_score(5900.0, 5800.0) == 0.0


def test_approx_pin_score_handles_none():
    assert _approx_pin_score(5800.0, None) is None


def test_approx_pin_score_zero_spot():
    assert _approx_pin_score(0.0, 5800.0) is None


# ------------------------------------------------------- known-good auth flow


@pytest.mark.asyncio
async def test_flashalpha_uses_secret_when_no_explicit_key(monkeypatch):
    captured: dict[str, str] = {}

    class _StubFA:
        def __init__(self, api_key):
            captured["api_key"] = api_key

        def zero_dte(self, symbol):
            return {"underlying_price": 5800.0, "regime": {"label": "positive_gamma"}}

        def exposure_levels(self, symbol):
            return {"levels": {}}

    fake_mod = types.ModuleType("flashalpha")
    fake_mod.FlashAlpha = _StubFA
    monkeypatch.setitem(sys.modules, "flashalpha", fake_mod)
    monkeypatch.setattr(fa, "flashalpha_api_key", lambda: "from-secret-store")

    async with FlashAlphaClient() as c:
        await c.get_signals()

    assert captured["api_key"] == "from-secret-store"
