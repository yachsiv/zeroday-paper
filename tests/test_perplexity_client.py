"""Perplexity client: HTTP via httpx.MockTransport, retry, auth errors."""

from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest

from zeroday_paper.data import perplexity_client as pc
from zeroday_paper.data.perplexity_client import (
    PerplexityAuthError,
    PerplexityClient,
    PerplexityError,
    _extract_content,
)


def _install_transport(monkeypatch, handler: Callable[[httpx.Request], httpx.Response]):
    real_asyncclient = httpx.AsyncClient

    def fake_async_client(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_asyncclient(*args, **kwargs)

    monkeypatch.setattr(pc.httpx, "AsyncClient", fake_async_client)


# ----------------------------------------------------- pure helpers


def test_extract_content_success():
    env = {"choices": [{"message": {"role": "assistant", "content": "hello"}}]}
    assert _extract_content(env) == "hello"


def test_extract_content_missing_choices_raises():
    with pytest.raises(PerplexityError):
        _extract_content({"foo": "bar"})


def test_extract_content_empty_choices_raises():
    with pytest.raises(PerplexityError):
        _extract_content({"choices": []})


def test_extract_content_missing_message_raises():
    with pytest.raises(PerplexityError):
        _extract_content({"choices": [{"foo": "bar"}]})


def test_extract_content_non_string_raises():
    with pytest.raises(PerplexityError):
        _extract_content({"choices": [{"message": {"content": 42}}]})


# ----------------------------------------------------- HTTP integration


@pytest.mark.asyncio
async def test_ask_success(monkeypatch):
    captured = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured["method"] = req.method
        captured["path"] = req.url.path
        captured["auth"] = req.headers.get("authorization")
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"ok": true}'}}]},
        )

    _install_transport(monkeypatch, handler)
    async with PerplexityClient(api_key="abc") as c:
        out = await c.ask("hi")
    assert out == '{"ok": true}'
    assert captured["method"] == "POST"
    assert captured["path"] == "/chat/completions"
    assert captured["auth"] == "Bearer abc"


@pytest.mark.asyncio
async def test_ask_uses_passed_model(monkeypatch):
    captured = {}

    def handler(req: httpx.Request) -> httpx.Response:
        import json as _j
        captured["body"] = _j.loads(req.content.decode())
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "ok"}}]}
        )

    _install_transport(monkeypatch, handler)
    async with PerplexityClient(api_key="x") as c:
        await c.ask("query", model="sonar-pro")
    assert captured["body"]["model"] == "sonar-pro"
    assert captured["body"]["messages"][0]["content"] == "query"


@pytest.mark.asyncio
async def test_ask_auth_401(monkeypatch):
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "bad key"})

    _install_transport(monkeypatch, handler)
    async with PerplexityClient(api_key="x") as c:
        with pytest.raises(PerplexityAuthError):
            await c.ask("hi")


@pytest.mark.asyncio
async def test_ask_auth_403(monkeypatch):
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"error": "forbidden"})

    _install_transport(monkeypatch, handler)
    async with PerplexityClient(api_key="x") as c:
        with pytest.raises(PerplexityAuthError):
            await c.ask("hi")


@pytest.mark.asyncio
async def test_ask_rate_limited_429(monkeypatch):
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": "too many"})

    _install_transport(monkeypatch, handler)
    async with PerplexityClient(api_key="x") as c:
        with pytest.raises(PerplexityError):
            await c.ask("hi")


@pytest.mark.slow
@pytest.mark.asyncio
async def test_ask_5xx_retries_then_succeeds(monkeypatch):
    calls = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 2:
            return httpx.Response(500, json={"error": "server"})
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "ok"}}]}
        )

    _install_transport(monkeypatch, handler)
    async with PerplexityClient(api_key="x") as c:
        out = await c.ask("hi")
        assert out == "ok"
        assert calls["n"] == 2
