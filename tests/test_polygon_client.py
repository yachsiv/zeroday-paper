"""Polygon REST client: HTTP via httpx.MockTransport, pagination, parsing."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from typing import Callable

import httpx
import pytest

from zeroday_paper.data import polygon_client as pc
from zeroday_paper.data.polygon_client import (
    ChainSnapshot,
    OptionQuote,
    PolygonAuthError,
    PolygonClient,
    PolygonError,
    PolygonTransportError,
    _safe_float,
    next_spx_expiry,
)


def _install_transport(monkeypatch, handler: Callable[[httpx.Request], httpx.Response]):
    real_asyncclient = httpx.AsyncClient

    def fake_async_client(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_asyncclient(*args, **kwargs)

    monkeypatch.setattr(pc.httpx, "AsyncClient", fake_async_client)


# ---------------------------------------------------------------- pure helpers


def test_safe_float_none():
    assert _safe_float(None) is None


def test_safe_float_valid():
    assert _safe_float("1.5") == 1.5
    assert _safe_float(3) == 3.0


def test_safe_float_invalid():
    assert _safe_float("abc") is None
    assert _safe_float({}) is None


@pytest.mark.parametrize("d,expected_offset_days", [
    (date(2025, 5, 26), 0),   # Monday → Mon
    (date(2025, 5, 27), 1),   # Tuesday → Wed
    (date(2025, 5, 28), 0),   # Wednesday → Wed
    (date(2025, 5, 29), 1),   # Thursday → Fri
    (date(2025, 5, 30), 0),   # Friday → Fri
    (date(2025, 5, 31), 2),   # Saturday → Mon
    (date(2025, 6, 1), 1),    # Sunday → Mon
])
def test_next_spx_expiry(d, expected_offset_days):
    result = next_spx_expiry(d)
    assert (result - d).days == expected_offset_days
    assert result.weekday() in (0, 2, 4)


def test_next_spx_expiry_default_uses_today():
    out = next_spx_expiry()
    assert out.weekday() in (0, 2, 4)


def test_extract_next_url_strips_base():
    assert PolygonClient._extract_next_url("https://api.polygon.io/v3/foo?cursor=abc") == "/v3/foo?cursor=abc"


def test_extract_next_url_passes_relative():
    assert PolygonClient._extract_next_url("/v3/foo?cursor=abc") == "/v3/foo?cursor=abc"


def test_extract_next_url_returns_none_for_empty():
    assert PolygonClient._extract_next_url(None) is None
    assert PolygonClient._extract_next_url("") is None


def test_parse_snapshot_result_success():
    payload = {
        "details": {"ticker": "O:SPXW250528P05775000", "strike_price": 5775.0, "contract_type": "put"},
        "last_quote": {"bid": 1.20, "ask": 1.40},
        "last_trade": {"price": 1.30},
        "day": {"volume": 100},
        "open_interest": 500,
        "implied_volatility": 0.21,
        "greeks": {"delta": -0.18, "gamma": 0.01, "theta": -0.5, "vega": 0.2},
    }
    q = PolygonClient._parse_snapshot_result(payload, date(2025, 5, 28))
    assert q is not None
    assert q.strike == 5775.0
    assert q.right == "P"
    assert q.bid == 1.20
    assert q.ask == 1.40
    assert q.iv == 0.21
    assert q.delta == -0.18
    assert q.open_interest == 500


def test_parse_snapshot_result_handles_call_contract_type():
    payload = {
        "details": {"ticker": "O:X", "strike_price": 5800, "contract_type": "call"},
        "last_quote": {"bid": 1.0, "ask": 1.2},
    }
    q = PolygonClient._parse_snapshot_result(payload, date(2025, 5, 28))
    assert q.right == "C"


def test_parse_snapshot_result_missing_ticker_returns_none():
    payload = {"details": {"strike_price": 5800, "contract_type": "call"}}
    assert PolygonClient._parse_snapshot_result(payload, date(2025, 5, 28)) is None


def test_parse_snapshot_result_missing_strike_returns_none():
    payload = {"details": {"ticker": "X", "contract_type": "call"}}
    assert PolygonClient._parse_snapshot_result(payload, date(2025, 5, 28)) is None


def test_parse_snapshot_result_missing_right_returns_none():
    payload = {"details": {"ticker": "X", "strike_price": 5800}}
    assert PolygonClient._parse_snapshot_result(payload, date(2025, 5, 28)) is None


def test_parse_snapshot_result_handles_garbage():
    # Bad numeric values
    payload = {"details": {"ticker": "X", "strike_price": "not-a-number", "contract_type": "call"}}
    assert PolygonClient._parse_snapshot_result(payload, date(2025, 5, 28)) is None


def test_option_quote_bid_ask_spread(make_quote):
    q = make_quote(strike=5775, right="P", bid=1.0, ask=1.5)
    assert q.bid_ask_spread == pytest.approx(0.5)


def test_option_quote_is_tradable_true(make_quote):
    q = make_quote(strike=5775, right="P", bid=1.0, ask=1.5)
    assert q.is_tradable is True


def test_option_quote_is_tradable_false_no_bid(make_quote):
    q = make_quote(strike=5775, right="P", bid=0.0, ask=1.0)
    assert q.is_tradable is False


def test_option_quote_is_tradable_false_wide_spread(make_quote):
    q = make_quote(strike=5775, right="P", bid=1.0, ask=10.0)
    assert q.is_tradable is False


def test_chain_snapshot_total_quotes_and_atm_strike(make_chain):
    chain = make_chain(spot=5800.0, n_strikes=11, step=5)
    assert chain.total_quotes() == 22  # 11 calls + 11 puts
    assert chain.atm_strike() == 5800.0


def test_chain_snapshot_atm_rounds_to_5(make_chain):
    chain = make_chain(spot=5803.0, n_strikes=11)
    # ATM strike should be the nearest 5-pt round
    assert chain.atm_strike() == 5805.0


# -------------------------------------------------------- HTTP-level integration


@pytest.mark.asyncio
async def test_polygon_get_spx_spot_success(monkeypatch):
    def handler(req: httpx.Request) -> httpx.Response:
        if "/v3/snapshot/indices" in str(req.url):
            return httpx.Response(200, json={"results": [{"value": 5825.5}]})
        return httpx.Response(404)

    _install_transport(monkeypatch, handler)
    async with PolygonClient(api_key="x") as c:
        spot = await c.get_spx_spot()
        assert spot == pytest.approx(5825.5)


@pytest.mark.asyncio
async def test_polygon_get_spx_spot_session_fallback(monkeypatch):
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": [{"session": {"price": 5810.0}}]})

    _install_transport(monkeypatch, handler)
    async with PolygonClient(api_key="x") as c:
        spot = await c.get_spx_spot()
        assert spot == pytest.approx(5810.0)


@pytest.mark.asyncio
async def test_polygon_get_spx_spot_falls_back_to_chain_peek(monkeypatch):
    def handler(req: httpx.Request) -> httpx.Response:
        path = req.url.path
        if path == "/v3/snapshot/indices":
            return httpx.Response(200, json={"results": []})
        if path.startswith("/v3/snapshot/options/"):
            return httpx.Response(200, json={
                "results": [{"underlying_asset": {"price": 5811.0}}]
            })
        return httpx.Response(404)

    _install_transport(monkeypatch, handler)
    async with PolygonClient(api_key="x") as c:
        spot = await c.get_spx_spot()
        assert spot == 5811.0


@pytest.mark.asyncio
async def test_polygon_get_spx_spot_falls_back_to_spy_x10(monkeypatch):
    def handler(req: httpx.Request) -> httpx.Response:
        path = req.url.path
        if path == "/v3/snapshot/indices":
            return httpx.Response(403)
        if path.startswith("/v3/snapshot/options/"):
            return httpx.Response(200, json={"results": []})
        if path == "/v3/snapshot/stocks":
            return httpx.Response(200, json={"results": [{"value": 580.0}]})
        return httpx.Response(404)

    _install_transport(monkeypatch, handler)
    async with PolygonClient(api_key="x") as c:
        spot = await c.get_spx_spot()
        assert spot == 5800.0


@pytest.mark.asyncio
async def test_polygon_get_spx_spot_all_endpoints_empty_raises(monkeypatch):
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": []})

    _install_transport(monkeypatch, handler)
    async with PolygonClient(api_key="x") as c:
        with pytest.raises(PolygonError):
            await c.get_spx_spot()


@pytest.mark.asyncio
async def test_polygon_auth_error_propagates_from_get_minute_bar(monkeypatch):
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "bad key"})

    _install_transport(monkeypatch, handler)
    async with PolygonClient(api_key="x") as c:
        with pytest.raises(PolygonAuthError):
            await c.get_minute_bar("O:X", datetime(2025, 5, 28, 14, 30, tzinfo=UTC))


@pytest.mark.asyncio
async def test_polygon_auth_error_403_propagates(monkeypatch):
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"error": "forbidden"})

    _install_transport(monkeypatch, handler)
    async with PolygonClient(api_key="x") as c:
        with pytest.raises(PolygonAuthError):
            await c.get_minute_bar("O:X", datetime(2025, 5, 28, 14, 30, tzinfo=UTC))


@pytest.mark.asyncio
async def test_polygon_5xx_retries_then_succeeds(monkeypatch):
    calls = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 2:
            return httpx.Response(500, json={"error": "server"})
        return httpx.Response(200, json={"results": [{"o": 1, "h": 2, "l": 0.5, "c": 1.5, "v": 10}]})

    _install_transport(monkeypatch, handler)
    async with PolygonClient(api_key="x") as c:
        bar = await c.get_minute_bar("O:X", datetime(2025, 5, 28, 14, 30, tzinfo=UTC))
        assert bar is not None
        assert calls["n"] == 2


@pytest.mark.asyncio
async def test_polygon_5xx_exhausts_retries_raises(monkeypatch):
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "down"})

    _install_transport(monkeypatch, handler)
    async with PolygonClient(api_key="x") as c:
        with pytest.raises(httpx.HTTPStatusError):
            await c.get_minute_bar("O:X", datetime(2025, 5, 28, 14, 30, tzinfo=UTC))


@pytest.mark.asyncio
async def test_polygon_get_spx_spot_at_daily_close(monkeypatch):
    def handler(req: httpx.Request) -> httpx.Response:
        if "/v2/aggs/ticker/" in str(req.url):
            return httpx.Response(200, json={"results": [{"c": 5777.7}]})
        return httpx.Response(404)

    _install_transport(monkeypatch, handler)
    async with PolygonClient(api_key="x") as c:
        spot = await c.get_spx_spot_at(datetime(2025, 5, 28, tzinfo=UTC))
        assert spot == pytest.approx(5777.7)


@pytest.mark.asyncio
async def test_polygon_get_spx_spot_at_no_results_raises(monkeypatch):
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": []})

    _install_transport(monkeypatch, handler)
    async with PolygonClient(api_key="x") as c:
        with pytest.raises(PolygonError):
            await c.get_spx_spot_at(datetime(2025, 5, 28, tzinfo=UTC))


@pytest.mark.asyncio
async def test_polygon_get_spx_spot_at_falls_back_to_spy(monkeypatch):
    def handler(req: httpx.Request) -> httpx.Response:
        path = req.url.path
        if "I:SPX" in path:
            return httpx.Response(200, json={"results": []})
        if "SPY" in path:
            return httpx.Response(200, json={"results": [{"c": 580.0}]})
        return httpx.Response(404)

    _install_transport(monkeypatch, handler)
    async with PolygonClient(api_key="x") as c:
        spot = await c.get_spx_spot_at(datetime(2025, 5, 28, tzinfo=UTC))
        assert spot == 5800.0


@pytest.mark.asyncio
async def test_polygon_get_minute_bar(monkeypatch):
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "results": [{"o": 1.0, "h": 1.5, "l": 0.9, "c": 1.2, "v": 100}]
        })

    _install_transport(monkeypatch, handler)
    async with PolygonClient(api_key="x") as c:
        bar = await c.get_minute_bar("O:X", datetime(2025, 5, 28, 14, 30, tzinfo=UTC))
        assert bar is not None
        assert bar["close"] == 1.2
        assert bar["volume"] == 100.0


@pytest.mark.asyncio
async def test_polygon_get_minute_bar_no_results_returns_none(monkeypatch):
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": []})

    _install_transport(monkeypatch, handler)
    async with PolygonClient(api_key="x") as c:
        bar = await c.get_minute_bar("O:X", datetime(2025, 5, 28, 14, 30, tzinfo=UTC))
        assert bar is None


@pytest.mark.asyncio
async def test_polygon_get_minute_bar_naive_ts_assumes_utc(monkeypatch):
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": [{"o": 1, "h": 1, "l": 1, "c": 1, "v": 0}]})

    _install_transport(monkeypatch, handler)
    async with PolygonClient(api_key="x") as c:
        bar = await c.get_minute_bar("O:X", datetime(2025, 5, 28, 14, 30))
        assert bar is not None


@pytest.mark.asyncio
async def test_polygon_get_chain_snapshot_paginates(monkeypatch):
    pages = [
        {
            "results": [
                {
                    "details": {"ticker": "O:1", "strike_price": 5800, "contract_type": "call"},
                    "last_quote": {"bid": 1.0, "ask": 1.2},
                    "open_interest": 100, "day": {"volume": 50},
                    "greeks": {"delta": 0.5, "gamma": 0.01, "theta": -0.5, "vega": 0.2},
                    "implied_volatility": 0.21,
                },
            ],
            "next_url": "https://api.polygon.io/v3/snapshot/options/SPX?cursor=p2",
        },
        {
            "results": [
                {
                    "details": {"ticker": "O:2", "strike_price": 5800, "contract_type": "put"},
                    "last_quote": {"bid": 0.5, "ask": 0.7},
                    "open_interest": 200, "day": {"volume": 30},
                    "greeks": {"delta": -0.5, "gamma": 0.01, "theta": -0.5, "vega": 0.2},
                    "implied_volatility": 0.22,
                },
            ],
            "next_url": None,
        },
    ]
    state = {"page": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        if "indices" in str(req.url):
            return httpx.Response(200, json={"results": [{"value": 5800.0}]})
        if "/v3/snapshot/options/" in str(req.url):
            i = state["page"]
            state["page"] += 1
            return httpx.Response(200, json=pages[i])
        return httpx.Response(404)

    _install_transport(monkeypatch, handler)
    async with PolygonClient(api_key="x") as c:
        snap = await c.get_chain_snapshot(date(2025, 5, 28))
    assert snap.spot == 5800.0
    assert snap.total_quotes() == 2
    assert len(snap.calls) == 1
    assert len(snap.puts) == 1


@pytest.mark.asyncio
async def test_polygon_get_chain_snapshot_uses_spot_override(monkeypatch):
    def handler(req: httpx.Request) -> httpx.Response:
        if "/v3/snapshot/options" in str(req.url):
            return httpx.Response(200, json={"results": [], "next_url": None})
        return httpx.Response(404)

    _install_transport(monkeypatch, handler)
    async with PolygonClient(api_key="x") as c:
        # No call to get_spx_spot needed
        snap = await c.get_chain_snapshot(date(2025, 5, 28), spot_override=5900.0)
    assert snap.spot == 5900.0
    assert snap.calls == []
    assert snap.puts == []


@pytest.mark.asyncio
async def test_polygon_get_chain_snapshot_at_reconstructs_bars(monkeypatch):
    def handler(req: httpx.Request) -> httpx.Response:
        path = req.url.path
        if "/v2/aggs/ticker/I:SPX/" in path:
            return httpx.Response(200, json={"results": [{"c": 5800.0}]})
        if "/v3/snapshot/options/" in path:
            return httpx.Response(200, json={
                "results": [
                    {
                        "details": {"ticker": "O:Z", "strike_price": 5800, "contract_type": "call"},
                        "last_quote": {"bid": 1.0, "ask": 1.2},
                        "open_interest": 100, "day": {"volume": 50},
                        "greeks": {"delta": 0.5, "gamma": 0.01, "theta": -0.5, "vega": 0.2},
                        "implied_volatility": 0.21,
                    },
                ],
                "next_url": None,
            })
        if "/v2/aggs/ticker/O:Z/" in path:
            return httpx.Response(200, json={"results": [{"o": 2, "h": 3, "l": 2, "c": 2.5, "v": 10}]})
        return httpx.Response(404)

    _install_transport(monkeypatch, handler)
    async with PolygonClient(api_key="x") as c:
        snap = await c.get_chain_snapshot_at(
            date(2025, 5, 28), datetime(2025, 5, 28, 14, 30, tzinfo=UTC)
        )
    assert snap.spot == 5800.0
    assert len(snap.calls) == 1
    # The call should have been re-priced from the minute bar (close=2.5)
    assert snap.calls[0].mid == 2.5
    assert snap.calls[0].bid >= 0.0


@pytest.mark.asyncio
async def test_polygon_get_chain_snapshot_at_skips_missing_bar(monkeypatch):
    def handler(req: httpx.Request) -> httpx.Response:
        path = req.url.path
        if "/v2/aggs/ticker/I:SPX/" in path:
            return httpx.Response(200, json={"results": [{"c": 5800.0}]})
        if "/v3/snapshot/options/" in path:
            return httpx.Response(200, json={
                "results": [
                    {
                        "details": {"ticker": "O:Z", "strike_price": 5800, "contract_type": "call"},
                        "last_quote": {"bid": 1.0, "ask": 1.2},
                        "open_interest": 100, "day": {"volume": 50},
                        "greeks": {"delta": 0.5},
                        "implied_volatility": 0.21,
                    },
                ],
                "next_url": None,
            })
        # All minute bars return empty
        return httpx.Response(200, json={"results": []})

    _install_transport(monkeypatch, handler)
    async with PolygonClient(api_key="x") as c:
        snap = await c.get_chain_snapshot_at(
            date(2025, 5, 28), datetime(2025, 5, 28, 14, 30, tzinfo=UTC)
        )
    # contract preserved (not None) but bid/ask unchanged
    assert len(snap.calls) == 1
    assert snap.calls[0].mid == 1.1  # original mid from chain
