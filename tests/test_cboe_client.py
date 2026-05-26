"""CBOE free-tier client: JSON quotes + CSV history."""

from __future__ import annotations

from datetime import UTC, date, datetime

import httpx
import pytest

from zeroday_paper.data import cboe_client as cb
from zeroday_paper.data.cboe_client import (
    CboeClient,
    CboeError,
    VolSnapshot,
    _extract_quote_last,
    _parse_cboe_date,
)


def _install_transport(monkeypatch, handler):
    real_asyncclient = httpx.AsyncClient

    def fake_async_client(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_asyncclient(*args, **kwargs)

    monkeypatch.setattr(cb.httpx, "AsyncClient", fake_async_client)


# ------------------------------------------------------- _extract_quote_last


def test_extract_quote_last_prefers_last():
    payload = {"data": {"last": 14.5, "close": 13.0, "iv": 11.0}}
    assert _extract_quote_last(payload) == 14.5


def test_extract_quote_last_falls_back_to_current_price():
    payload = {"data": {"current_price": 12.5, "close": 13.0}}
    assert _extract_quote_last(payload) == 12.5


def test_extract_quote_last_falls_back_to_close():
    payload = {"data": {"close": 13.5}}
    assert _extract_quote_last(payload) == 13.5


def test_extract_quote_last_falls_back_to_iv():
    payload = {"data": {"iv": 22.0}}
    assert _extract_quote_last(payload) == 22.0


def test_extract_quote_last_returns_none_if_no_keys():
    assert _extract_quote_last({}) is None
    assert _extract_quote_last({"data": {}}) is None


def test_extract_quote_last_skips_invalid_floats():
    payload = {"data": {"last": "garbage", "close": 13.0}}
    assert _extract_quote_last(payload) == 13.0


# ------------------------------------------------------------ _parse_cboe_date


@pytest.mark.parametrize("raw,expected", [
    ("5/28/2025", date(2025, 5, 28)),
    ("2025-05-28", date(2025, 5, 28)),
    ("5/28/25", date(2025, 5, 28)),
    (" 5/28/2025 ", date(2025, 5, 28)),
])
def test_parse_cboe_date_formats(raw, expected):
    assert _parse_cboe_date(raw) == expected


def test_parse_cboe_date_invalid_raises():
    with pytest.raises(ValueError):
        _parse_cboe_date("not-a-date")


# ----------------------------------------------------------- live snapshot


@pytest.mark.asyncio
async def test_cboe_get_live_snapshot_success(monkeypatch):
    def handler(req: httpx.Request) -> httpx.Response:
        if "VIX1D" in str(req.url):
            return httpx.Response(200, json={"data": {"last": 14.2}})
        if "SKEW" in str(req.url):
            return httpx.Response(200, json={"data": {"last": 142.0}})
        return httpx.Response(404)

    _install_transport(monkeypatch, handler)
    async with CboeClient() as c:
        snap = await c.get_live_snapshot()
    assert snap.vix_1d == 14.2
    assert snap.cboe_skew == 142.0


@pytest.mark.asyncio
async def test_cboe_get_live_snapshot_partial_failure(monkeypatch):
    def handler(req: httpx.Request) -> httpx.Response:
        if "VIX1D" in str(req.url):
            return httpx.Response(500)
        return httpx.Response(200, json={"data": {"last": 130.0}})

    _install_transport(monkeypatch, handler)
    async with CboeClient() as c:
        snap = await c.get_live_snapshot()
    assert snap.vix_1d is None
    assert snap.cboe_skew == 130.0


# --------------------------------------------------------- historical CSV


@pytest.mark.asyncio
async def test_cboe_historical_snapshot_with_close_header(monkeypatch):
    csv_body = "DATE,OPEN,HIGH,LOW,CLOSE\n5/28/2025,14.0,14.5,13.5,14.2\n"
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=csv_body)

    _install_transport(monkeypatch, handler)
    async with CboeClient() as c:
        snap = await c.get_historical_snapshot(date(2025, 5, 28))
    assert snap.vix_1d == 14.2


@pytest.mark.asyncio
async def test_cboe_historical_snapshot_with_index_named_header(monkeypatch):
    csv_body = "DATE,VIX1D\n5/28/2025,15.5\n"
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=csv_body)

    _install_transport(monkeypatch, handler)
    async with CboeClient() as c:
        snap = await c.get_historical_snapshot(date(2025, 5, 28))
    assert snap.vix_1d == 15.5


@pytest.mark.asyncio
async def test_cboe_historical_snapshot_date_not_found_returns_none(monkeypatch):
    csv_body = "DATE,CLOSE\n5/27/2025,14.0\n"
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=csv_body)

    _install_transport(monkeypatch, handler)
    async with CboeClient() as c:
        snap = await c.get_historical_snapshot(date(2025, 5, 28))
    assert snap.vix_1d is None
    assert snap.cboe_skew is None


@pytest.mark.asyncio
async def test_cboe_historical_snapshot_no_date_column_returns_none(monkeypatch):
    csv_body = "OPEN,CLOSE\n14.0,14.5\n"
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=csv_body)

    _install_transport(monkeypatch, handler)
    async with CboeClient() as c:
        snap = await c.get_historical_snapshot(date(2025, 5, 28))
    assert snap.vix_1d is None


@pytest.mark.asyncio
async def test_cboe_historical_csv_fetch_failure_returns_none(monkeypatch):
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    _install_transport(monkeypatch, handler)
    async with CboeClient() as c:
        snap = await c.get_historical_snapshot(date(2025, 5, 28))
    assert snap.vix_1d is None
    assert snap.cboe_skew is None


@pytest.mark.asyncio
async def test_cboe_historical_handles_bad_date_row(monkeypatch):
    csv_body = "DATE,CLOSE\nGARBAGE,14.0\n5/28/2025,14.5\n"
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=csv_body)

    _install_transport(monkeypatch, handler)
    async with CboeClient() as c:
        snap = await c.get_historical_snapshot(date(2025, 5, 28))
    assert snap.vix_1d == 14.5


@pytest.mark.asyncio
async def test_cboe_historical_handles_short_row(monkeypatch):
    csv_body = "DATE,CLOSE\n5/28/2025\n5/29/2025,15.0\n"
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=csv_body)

    _install_transport(monkeypatch, handler)
    async with CboeClient() as c:
        snap = await c.get_historical_snapshot(date(2025, 5, 29))
    assert snap.vix_1d == 15.0


@pytest.mark.asyncio
async def test_cboe_historical_empty_csv_returns_none(monkeypatch):
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="")

    _install_transport(monkeypatch, handler)
    async with CboeClient() as c:
        snap = await c.get_historical_snapshot(date(2025, 5, 28))
    assert snap.vix_1d is None


@pytest.mark.asyncio
async def test_cboe_historical_invalid_close_value(monkeypatch):
    csv_body = "DATE,CLOSE\n5/28/2025,not-a-number\n"
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=csv_body)

    _install_transport(monkeypatch, handler)
    async with CboeClient() as c:
        snap = await c.get_historical_snapshot(date(2025, 5, 28))
    assert snap.vix_1d is None
