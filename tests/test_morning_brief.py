"""Pre-market brief: bundle building, rendering, Discord post, CLI."""

from __future__ import annotations

import json
import sys
from datetime import UTC, date, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from zeroday_paper.cli import run_morning
from zeroday_paper.config import settings
from zeroday_paper.engine.replay import US_HOLIDAYS
from zeroday_paper.reporting import morning_brief as mb

# --------------------------------------------------------------------------- builders


def _set_today(monkeypatch, d: date) -> None:
    monkeypatch.setattr(mb, "today_et", lambda: d)


class _FakeAsyncClient:
    """Generic async-context-manager stand-in. ``__aenter__`` returns self.

    Attach the methods/AsyncMock side-effects you want on the instance
    AFTER constructing.
    """

    def __init__(self, *, raise_on_open: bool = False):
        self._raise_on_open = raise_on_open

    async def __aenter__(self):
        if self._raise_on_open:
            raise RuntimeError("client open failed")
        return self

    async def __aexit__(self, *_):
        return None


def _stub_polygon(monkeypatch, *, chain, prev_aggregate=None, minute_bars=None,
                  raise_on_open: bool = False):
    fake = _FakeAsyncClient(raise_on_open=raise_on_open)
    fake.get_chain_snapshot = AsyncMock(return_value=chain)
    fake.get_prev_day_aggregate = AsyncMock(
        return_value=prev_aggregate or {
            "open": 580.0, "high": 582.0, "low": 578.0,
            "close": 580.5, "volume": 1.0e8, "timestamp": 0.0,
        }
    )
    fake.get_minute_bars_range = AsyncMock(
        return_value=minute_bars if minute_bars is not None else [
            {"open": 581.0, "high": 583.0, "low": 580.0, "close": 582.5,
             "volume": 1e5, "timestamp": 0.0},
        ]
    )
    monkeypatch.setattr(mb, "PolygonClient", lambda *a, **kw: fake)
    return fake


def _stub_cboe(monkeypatch, *, vols=None, raise_on_fetch: bool = False):
    fake = _FakeAsyncClient()
    if raise_on_fetch:
        fake.get_live_snapshot = AsyncMock(side_effect=RuntimeError("cboe down"))
    else:
        fake.get_live_snapshot = AsyncMock(return_value=vols)
    monkeypatch.setattr(mb, "CboeClient", lambda *a, **kw: fake)
    return fake


def _stub_perplexity(monkeypatch, *, responses=None, side_effect=None):
    fake = _FakeAsyncClient()
    if side_effect is not None:
        fake.ask = AsyncMock(side_effect=side_effect)
    else:
        fake.ask = AsyncMock(side_effect=lambda prompt, model="sonar-pro": next(responses))
    monkeypatch.setattr(mb, "PerplexityClient", lambda *a, **kw: fake)
    return fake


# --------------------------------------------------------------------------- _safe_json


def test_safe_json_plain():
    assert mb._safe_json('{"a": 1}') == {"a": 1}


def test_safe_json_with_fences():
    raw = "```json\n{\"a\": 1}\n```"
    assert mb._safe_json(raw) == {"a": 1}


def test_safe_json_with_prose_prefix():
    raw = "Sure, here is the JSON:\n{\"x\": 2}\nLet me know if more!"
    assert mb._safe_json(raw) == {"x": 2}


def test_safe_json_invalid_returns_empty():
    assert mb._safe_json("totally not json") == {}


def test_safe_json_malformed_braces():
    assert mb._safe_json("{nope") == {}


# --------------------------------------------------------------------------- _is_tier_one / window


def test_is_tier_one_matches_config():
    assert mb._is_tier_one("FOMC Rate Decision") is True
    assert mb._is_tier_one("CPI release") is True
    assert mb._is_tier_one("Initial Jobless Claims") is False


def test_within_tier_one_window_inside():
    assert mb._within_tier_one_window("10:00") is True
    assert mb._within_tier_one_window("09:30") is True
    assert mb._within_tier_one_window("14:00") is True


def test_within_tier_one_window_outside():
    assert mb._within_tier_one_window("08:30") is False
    assert mb._within_tier_one_window("15:00") is False


def test_within_tier_one_window_malformed():
    assert mb._within_tier_one_window("badtime") is False
    assert mb._within_tier_one_window("") is False


# --------------------------------------------------------------------------- direction normalizers


@pytest.mark.parametrize("raw,expected", [
    ("up", "up"), ("UP", "up"), ("higher", "up"),
    ("down", "down"), ("lower", "down"),
    ("flat", "flat"), ("mixed", "flat"),
    ("garbage", None), (None, None),
])
def test_norm_dir(raw, expected):
    assert mb._norm_dir(raw) == expected


@pytest.mark.parametrize("raw,expected", [
    ("Risk On", "Risk On"), ("risk off", "Risk Off"),
    ("neutral", "Neutral"), ("mixed", "Neutral"),
    ("unknown phrase", None), (None, None),
])
def test_norm_sentiment(raw, expected):
    assert mb._norm_sentiment(raw) == expected


# --------------------------------------------------------------------------- _evaluate_vol


def test_evaluate_vol_skip_on_high_vix(make_vols):
    vols = make_vols(vix_1d=30.0, cboe_skew=120.0)
    ctx = mb._evaluate_vol(vols)
    assert ctx.would_skip is True
    assert any("VIX1D" in r for r in ctx.skip_reasons)


def test_evaluate_vol_skip_on_high_skew(make_vols):
    vols = make_vols(vix_1d=12.0, cboe_skew=170.0)
    ctx = mb._evaluate_vol(vols)
    assert ctx.would_skip is True
    assert any("SKEW" in r for r in ctx.skip_reasons)


def test_evaluate_vol_passes_quiet(make_vols):
    vols = make_vols(vix_1d=12.0, cboe_skew=120.0)
    ctx = mb._evaluate_vol(vols)
    assert ctx.would_skip is False
    assert ctx.skip_reasons == ()


def test_evaluate_vol_skips_missing(make_vols):
    vols = make_vols(vix_1d=None, cboe_skew=None)
    ctx = mb._evaluate_vol(vols)
    assert ctx.would_skip is False


# --------------------------------------------------------------------------- _expected_move


def test_expected_move_happy_path(make_chain):
    chain = make_chain(spot=5800.0, n_strikes=21)
    em = mb._expected_move(chain)
    assert em is not None
    assert em.atm_strike == 5800.0
    assert em.expected_move_pts is not None and em.expected_move_pts > 0
    assert em.range_low < em.atm_strike < em.range_high


def test_expected_move_no_spot(make_chain):
    chain = make_chain(spot=0.0)
    assert mb._expected_move(chain) is None


def test_expected_move_no_atm_quote(make_chain, make_quote):
    # Build a chain with NO atm strike (skip 5800)
    spot = 5800.0
    calls = [make_quote(strike=5810.0, right="C", bid=2.0, ask=2.2)]
    puts = [make_quote(strike=5790.0, right="P", bid=2.0, ask=2.2)]
    from zeroday_paper.data.polygon_client import ChainSnapshot
    chain = ChainSnapshot(
        fetched_at=datetime(2025, 5, 28, 14, 30, tzinfo=UTC),
        spot=spot, expiry=date(2025, 5, 28),
        calls=calls, puts=puts,
    )
    assert mb._expected_move(chain) is None


# --------------------------------------------------------------------------- _build_overnight


@pytest.mark.asyncio
async def test_build_overnight_with_prev_and_bars(make_chain):
    chain = make_chain(spot=5800.0)
    polygon = MagicMock()
    polygon.get_prev_day_aggregate = AsyncMock(return_value={
        "open": 578.0, "high": 581.0, "low": 577.0, "close": 580.0,
        "volume": 1e8, "timestamp": 0.0,
    })
    polygon.get_minute_bars_range = AsyncMock(return_value=[
        {"open": 580.0, "high": 581.0, "low": 579.5, "close": 580.5,
         "volume": 1e5, "timestamp": 0.0},
        {"open": 580.5, "high": 583.0, "low": 580.0, "close": 582.0,
         "volume": 1e5, "timestamp": 0.0},
    ])
    out = await mb._build_overnight(polygon, chain)
    assert out.source == "SPY"
    assert out.overnight_high == 583.0
    assert out.overnight_low == 579.5
    assert out.overnight_last == 582.0
    assert out.change_pts is not None
    assert out.implied_spx_open is not None
    assert out.prev_spx_close is not None


@pytest.mark.asyncio
async def test_build_overnight_no_prev_day_returns_unavailable(make_chain):
    chain = make_chain(spot=5800.0)
    polygon = MagicMock()
    polygon.get_prev_day_aggregate = AsyncMock(return_value=None)
    polygon.get_minute_bars_range = AsyncMock(return_value=[])
    out = await mb._build_overnight(polygon, chain)
    assert out.source == "unavailable"
    assert out.implied_spx_open is None


@pytest.mark.asyncio
async def test_build_overnight_minutes_failure_still_renders(make_chain):
    chain = make_chain(spot=5800.0)
    polygon = MagicMock()
    polygon.get_prev_day_aggregate = AsyncMock(return_value={
        "open": 578.0, "high": 581.0, "low": 577.0, "close": 580.0,
        "volume": 1e8, "timestamp": 0.0,
    })
    polygon.get_minute_bars_range = AsyncMock(side_effect=RuntimeError("polygon 5xx"))
    out = await mb._build_overnight(polygon, chain)
    assert out.source == "SPY"
    # No minute bars → falls back to flat (last == prev close).
    assert out.overnight_high is None
    assert out.overnight_low is None
    assert out.overnight_last == 580.0


# --------------------------------------------------------------------------- _build_calendar


@pytest.mark.asyncio
async def test_build_calendar_parses_tier_one(monkeypatch):
    responses = iter([
        json.dumps({"events": [
            {"name": "CPI release", "time_et": "08:30"},
            {"name": "FOMC Statement", "time_et": "14:00"},
            {"name": "ISM Manufacturing", "time_et": "10:00"},
        ]}),
    ])
    perplexity = MagicMock()
    perplexity.ask = AsyncMock(side_effect=lambda prompt, model="sonar-pro": next(responses))
    out = await mb._build_calendar(perplexity, date(2026, 5, 27))
    assert out.skip_today is True
    assert out.skip_reason and "FOMC" in out.skip_reason
    assert any(e.is_tier_one for e in out.events)
    assert any(e.name.startswith("ISM") for e in out.events)


@pytest.mark.asyncio
async def test_build_calendar_no_events():
    perplexity = MagicMock()
    perplexity.ask = AsyncMock(return_value='{"events": []}')
    out = await mb._build_calendar(perplexity, date(2026, 5, 27))
    assert out.events == ()
    assert out.skip_today is False
    assert out.skip_reason is None


@pytest.mark.asyncio
async def test_build_calendar_drops_malformed_rows():
    perplexity = MagicMock()
    perplexity.ask = AsyncMock(return_value=json.dumps({"events": [
        {"name": "", "time_et": "08:30"},
        {"name": "Valid", "time_et": "10:00"},
        "not a dict",
        {"name": "CPI release", "time_et": ""},
    ]}))
    out = await mb._build_calendar(perplexity, date(2026, 5, 27))
    names = [e.name for e in out.events]
    assert names == ["Valid"]


@pytest.mark.asyncio
async def test_build_calendar_tier_one_outside_window_no_skip():
    perplexity = MagicMock()
    perplexity.ask = AsyncMock(return_value=json.dumps({"events": [
        {"name": "FOMC speech", "time_et": "18:30"},  # outside 09:30-14:00 window
    ]}))
    out = await mb._build_calendar(perplexity, date(2026, 5, 27))
    assert out.skip_today is False


# --------------------------------------------------------------------------- _build_earnings


@pytest.mark.asyncio
async def test_build_earnings_parses():
    perplexity = MagicMock()
    perplexity.ask = AsyncMock(return_value=json.dumps({"earnings": [
        {"ticker": "AAPL", "timing": "BMO"},
        {"ticker": "msft", "timing": "AMC"},
        {"ticker": "", "timing": "BMO"},
        "not a dict",
    ]}))
    out = await mb._build_earnings(perplexity, date(2026, 5, 27))
    assert any("AAPL (pre-open)" in s for s in out.items)
    assert any("MSFT (after-close)" in s for s in out.items)
    assert all("(" in s for s in out.items)
    assert len(out.items) == 2


@pytest.mark.asyncio
async def test_build_earnings_empty():
    perplexity = MagicMock()
    perplexity.ask = AsyncMock(return_value='{"earnings": []}')
    out = await mb._build_earnings(perplexity, date(2026, 5, 27))
    assert out.items == ()


# --------------------------------------------------------------------------- _build_global


@pytest.mark.asyncio
async def test_build_global_parses():
    perplexity = MagicMock()
    perplexity.ask = AsyncMock(return_value=json.dumps({
        "nikkei": "up", "hang_seng": "down", "dax": "flat", "ftse": "up",
        "sentiment": "Risk On", "notes": "Tech leads.",
    }))
    g = await mb._build_global(perplexity, date(2026, 5, 27))
    assert g.nikkei == "up" and g.hang_seng == "down"
    assert g.dax == "flat" and g.ftse == "up"
    assert g.sentiment == "Risk On"
    assert g.notes == "Tech leads."


@pytest.mark.asyncio
async def test_build_global_malformed_returns_none_fields():
    perplexity = MagicMock()
    perplexity.ask = AsyncMock(return_value="garbage")
    g = await mb._build_global(perplexity, date(2026, 5, 27))
    assert g.nikkei is None
    assert g.sentiment is None


# --------------------------------------------------------------------------- build_bundle + render


@pytest.mark.asyncio
async def test_build_bundle_holiday_skips_all(monkeypatch):
    """If today is in US_HOLIDAYS, no fetches happen and meta.is_holiday is True."""
    holiday = next(iter(US_HOLIDAYS))
    _set_today(monkeypatch, holiday)
    bundle = await mb.build_bundle()
    assert bundle.meta.is_holiday is True
    assert bundle.vol is None
    assert bundle.levels is None
    assert bundle.overnight is None
    md = mb.render_brief(bundle)
    assert "Market closed today (holiday)" in md


@pytest.mark.asyncio
async def test_build_bundle_happy_path(monkeypatch, make_chain, make_vols):
    today = date(2026, 5, 27)  # Wednesday (0DTE day)
    _set_today(monkeypatch, today)

    chain = make_chain(spot=5800.0, n_strikes=21, expiry=today)
    _stub_polygon(monkeypatch, chain=chain)
    _stub_cboe(monkeypatch, vols=make_vols(vix_1d=12.0, cboe_skew=130.0))

    responses = iter([
        json.dumps({"events": [{"name": "ISM Mfg", "time_et": "10:00"}]}),
        json.dumps({"earnings": [{"ticker": "NVDA", "timing": "AMC"}]}),
        json.dumps({
            "nikkei": "up", "hang_seng": "flat", "dax": "up", "ftse": "flat",
            "sentiment": "Risk On", "notes": "Asia firm."
        }),
    ])
    _stub_perplexity(monkeypatch, responses=responses)

    bundle = await mb.build_bundle()
    assert bundle.meta.is_0dte_day is True
    assert bundle.vol is not None and bundle.vol.would_skip is False
    assert bundle.levels is not None and bundle.levels.signals is not None
    assert bundle.overnight is not None and bundle.overnight.source == "SPY"
    assert bundle.expected_move is not None
    assert bundle.calendar is not None and bundle.calendar.skip_today is False
    assert bundle.earnings is not None and any("NVDA" in s for s in bundle.earnings.items)
    assert bundle.global_markets is not None and bundle.global_markets.sentiment == "Risk On"

    md = mb.render_brief(bundle)
    assert "# Pre-Market Brief" in md
    assert "## Session metadata" in md
    assert "## Volatility context" in md
    assert "## Yesterday's key levels" in md
    assert "## ES futures overnight" in md
    assert "## Expected 1SD daily move" in md
    assert "## Economic calendar" in md
    assert "## Earnings" in md
    assert "## Global markets" in md
    assert "## Recommended posture" in md
    assert "NVDA" in md
    assert "Standard scan" in md


@pytest.mark.asyncio
async def test_build_bundle_tier_one_skips_today(monkeypatch, make_chain, make_vols):
    today = date(2026, 5, 27)
    _set_today(monkeypatch, today)
    chain = make_chain(spot=5800.0, n_strikes=21, expiry=today)
    _stub_polygon(monkeypatch, chain=chain)
    _stub_cboe(monkeypatch, vols=make_vols(vix_1d=12.0, cboe_skew=130.0))

    responses = iter([
        json.dumps({"events": [
            {"name": "CPI release", "time_et": "08:30"},  # outside window
            {"name": "FOMC Statement", "time_et": "14:00"},  # in window
        ]}),
        json.dumps({"earnings": []}),
        json.dumps({"nikkei": "flat", "hang_seng": "flat", "dax": "flat",
                    "ftse": "flat", "sentiment": "Neutral", "notes": "quiet"}),
    ])
    _stub_perplexity(monkeypatch, responses=responses)

    bundle = await mb.build_bundle()
    assert bundle.calendar is not None
    assert bundle.calendar.skip_today is True
    md = mb.render_brief(bundle)
    assert "SKIP_TODAY=True" in md
    assert "Stand down" in md


@pytest.mark.asyncio
async def test_build_bundle_high_vol_blocks(monkeypatch, make_chain, make_vols):
    today = date(2026, 5, 27)
    _set_today(monkeypatch, today)
    chain = make_chain(spot=5800.0, n_strikes=21, expiry=today)
    _stub_polygon(monkeypatch, chain=chain)
    _stub_cboe(monkeypatch, vols=make_vols(vix_1d=35.0, cboe_skew=130.0))
    responses = iter([
        json.dumps({"events": []}),
        json.dumps({"earnings": []}),
        json.dumps({"nikkei": "flat", "hang_seng": "flat", "dax": "flat",
                    "ftse": "flat", "sentiment": "Neutral"}),
    ])
    _stub_perplexity(monkeypatch, responses=responses)

    bundle = await mb.build_bundle()
    assert bundle.vol is not None and bundle.vol.would_skip is True
    md = mb.render_brief(bundle)
    assert "WOULD_SKIP" in md
    assert "High-volatility regime" in md


@pytest.mark.asyncio
async def test_build_bundle_marks_perplexity_unavailable(monkeypatch, make_chain, make_vols):
    today = date(2026, 5, 27)
    _set_today(monkeypatch, today)
    chain = make_chain(spot=5800.0, n_strikes=21, expiry=today)
    _stub_polygon(monkeypatch, chain=chain)
    _stub_cboe(monkeypatch, vols=make_vols(vix_1d=12.0, cboe_skew=130.0))
    _stub_perplexity(monkeypatch, side_effect=RuntimeError("perplexity 429"))

    bundle = await mb.build_bundle()
    assert bundle.calendar is None
    assert bundle.earnings is None
    assert bundle.global_markets is None
    assert "calendar" in bundle.failures
    md = mb.render_brief(bundle)
    # Sections that failed render as [UNAVAILABLE], but the brief still produces.
    assert "[UNAVAILABLE]" in md
    assert "# Pre-Market Brief" in md


@pytest.mark.asyncio
async def test_build_bundle_marks_polygon_unavailable_on_chain_failure(
    monkeypatch, make_chain, make_vols
):
    today = date(2026, 5, 27)
    _set_today(monkeypatch, today)
    chain = make_chain(spot=5800.0, n_strikes=21, expiry=today)
    fake = _stub_polygon(monkeypatch, chain=chain)
    fake.get_chain_snapshot = AsyncMock(side_effect=RuntimeError("Polygon NOT_ENTITLED"))
    _stub_cboe(monkeypatch, vols=make_vols())
    _stub_perplexity(monkeypatch, responses=iter([
        json.dumps({"events": []}),
        json.dumps({"earnings": []}),
        json.dumps({"nikkei": "flat", "hang_seng": "flat", "dax": "flat",
                    "ftse": "flat", "sentiment": "Neutral"}),
    ]))
    bundle = await mb.build_bundle()
    assert bundle.levels is None
    assert bundle.overnight is None
    assert bundle.expected_move is None
    assert "polygon" in bundle.failures
    md = mb.render_brief(bundle)
    assert "[UNAVAILABLE]" in md


@pytest.mark.asyncio
async def test_build_bundle_marks_polygon_open_failure(monkeypatch, make_vols):
    today = date(2026, 5, 27)
    _set_today(monkeypatch, today)
    _stub_polygon(monkeypatch, chain=None, raise_on_open=True)
    _stub_cboe(monkeypatch, vols=make_vols())
    _stub_perplexity(monkeypatch, responses=iter([
        json.dumps({"events": []}),
        json.dumps({"earnings": []}),
        json.dumps({"nikkei": "flat", "hang_seng": "flat", "dax": "flat",
                    "ftse": "flat", "sentiment": "Neutral"}),
    ]))
    bundle = await mb.build_bundle()
    assert bundle.levels is None
    assert bundle.overnight is None
    assert bundle.expected_move is None
    assert "polygon_open" in bundle.failures
    md = mb.render_brief(bundle)
    assert "[UNAVAILABLE]" in md


@pytest.mark.asyncio
async def test_build_bundle_marks_cboe_unavailable(monkeypatch, make_chain):
    today = date(2026, 5, 27)
    _set_today(monkeypatch, today)
    chain = make_chain(spot=5800.0, n_strikes=21, expiry=today)
    _stub_polygon(monkeypatch, chain=chain)
    _stub_cboe(monkeypatch, raise_on_fetch=True)
    _stub_perplexity(monkeypatch, responses=iter([
        json.dumps({"events": []}),
        json.dumps({"earnings": []}),
        json.dumps({"nikkei": "flat", "hang_seng": "flat", "dax": "flat",
                    "ftse": "flat", "sentiment": "Neutral"}),
    ]))
    bundle = await mb.build_bundle()
    assert bundle.vol is None
    assert "vol" in bundle.failures


# --------------------------------------------------------------------------- recommended posture text


def test_recommended_posture_holiday():
    meta = mb.SessionMeta(today_et=date(2026, 1, 1), is_0dte_day=False, is_holiday=True)
    b = mb.BriefBundle(meta=meta, vol=None, levels=None, overnight=None,
                       expected_move=None, calendar=None, earnings=None,
                       global_markets=None)
    assert mb._recommended_posture(b) == "Market closed."


def test_recommended_posture_skip_today():
    meta = mb.SessionMeta(today_et=date(2026, 5, 27), is_0dte_day=True, is_holiday=False)
    cal = mb.CalendarSection(events=(), skip_today=True, skip_reason="Tier-1 at 14:00")
    b = mb.BriefBundle(meta=meta, vol=None, levels=None, overnight=None,
                       expected_move=None, calendar=cal, earnings=None,
                       global_markets=None)
    assert "Stand down" in mb._recommended_posture(b)


# --------------------------------------------------------------------------- Discord post


def test_post_to_discord_no_webhook_returns_false(monkeypatch):
    monkeypatch.setattr(mb, "_resolve_webhook", lambda: None)
    assert mb.post_to_discord("hello") is False


def test_post_to_discord_explicit_url(monkeypatch):
    captured = []

    def fake_post(url, json=None, timeout=None):
        captured.append(json["content"])
        r = MagicMock()
        r.status_code = 204
        r.text = ""
        return r

    monkeypatch.setattr(mb.httpx, "post", fake_post)
    out = mb.post_to_discord("hello world", webhook_url="https://hook")
    assert out is True
    assert captured[0] == "hello world"


def test_post_to_discord_non_2xx(monkeypatch):
    def fake_post(url, json=None, timeout=None):
        r = MagicMock()
        r.status_code = 500
        r.text = "boom"
        return r

    monkeypatch.setattr(mb.httpx, "post", fake_post)
    assert mb.post_to_discord("hi", webhook_url="https://x") is False


def test_post_to_discord_raises(monkeypatch):
    def fake_post(url, json=None, timeout=None):
        raise RuntimeError("conn refused")

    monkeypatch.setattr(mb.httpx, "post", fake_post)
    assert mb.post_to_discord("hi", webhook_url="https://x") is False


def test_post_to_discord_chunks_long_text(monkeypatch):
    posts = []

    def fake_post(url, json=None, timeout=None):
        posts.append(json["content"])
        r = MagicMock()
        r.status_code = 204
        return r

    monkeypatch.setattr(mb.httpx, "post", fake_post)
    long = "\n".join(f"row-{i}" for i in range(500))
    out = mb.post_to_discord(long, webhook_url="https://x")
    assert out is True
    assert len(posts) >= 2


def test_resolve_webhook_uses_morning_key(monkeypatch):
    captured = {}

    def fake_dw(key=None):
        captured["key"] = key
        return "https://hook.test/morning"

    monkeypatch.setattr(mb, "discord_webhook", fake_dw)
    assert mb._resolve_webhook() == "https://hook.test/morning"
    assert captured["key"] == settings.reporting.morning_brief_discord_webhook_secret_key


def test_resolve_webhook_failure_returns_none(monkeypatch):
    def fake_dw(key=None):
        raise RuntimeError("no secret")

    monkeypatch.setattr(mb, "discord_webhook", fake_dw)
    assert mb._resolve_webhook() is None


# --------------------------------------------------------------------------- CLI


def test_cli_no_discord(monkeypatch):
    async def fake_bundle(*a, **kw):
        meta = mb.SessionMeta(today_et=date(2026, 5, 27), is_0dte_day=True, is_holiday=False)
        return mb.BriefBundle(
            meta=meta, vol=None, levels=None, overnight=None,
            expected_move=None, calendar=None, earnings=None, global_markets=None,
        )

    posted = {"n": 0}

    def fake_post(md):
        posted["n"] += 1
        return True

    monkeypatch.setattr(run_morning, "build_bundle", fake_bundle)
    monkeypatch.setattr(run_morning, "post_to_discord", fake_post)
    monkeypatch.setattr(sys, "argv", ["zp-morning", "--no-discord"])
    with pytest.raises(SystemExit) as exc:
        run_morning.main()
    assert exc.value.code == 0
    assert posted["n"] == 0


def test_cli_discord_path(monkeypatch, capsys):
    async def fake_bundle(*a, **kw):
        meta = mb.SessionMeta(today_et=date(2026, 5, 27), is_0dte_day=True, is_holiday=False)
        return mb.BriefBundle(
            meta=meta, vol=None, levels=None, overnight=None,
            expected_move=None, calendar=None, earnings=None, global_markets=None,
        )

    monkeypatch.setattr(run_morning, "build_bundle", fake_bundle)
    monkeypatch.setattr(run_morning, "post_to_discord", lambda md: True)
    monkeypatch.setattr(sys, "argv", ["zp-morning", "--print"])
    with pytest.raises(SystemExit) as exc:
        run_morning.main()
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "# Pre-Market Brief" in out


def test_cli_discord_failure_exit_1(monkeypatch):
    async def fake_bundle(*a, **kw):
        meta = mb.SessionMeta(today_et=date(2026, 5, 27), is_0dte_day=True, is_holiday=False)
        return mb.BriefBundle(
            meta=meta, vol=None, levels=None, overnight=None,
            expected_move=None, calendar=None, earnings=None, global_markets=None,
        )

    monkeypatch.setattr(run_morning, "build_bundle", fake_bundle)
    monkeypatch.setattr(run_morning, "post_to_discord", lambda md: False)
    monkeypatch.setattr(sys, "argv", ["zp-morning"])
    with pytest.raises(SystemExit) as exc:
        run_morning.main()
    assert exc.value.code == 1


def test_cli_disabled_returns_zero(monkeypatch):
    """When `[morning_brief].enabled = false`, CLI exits 0 without building."""
    import dataclasses

    from zeroday_paper.config import MorningBriefConfig
    monkeypatch.setattr(sys, "argv", ["zp-morning"])

    fake = dataclasses.replace(
        run_morning.settings,
        morning_brief=MorningBriefConfig(enabled=False, tier_one_events=()),
    )
    monkeypatch.setattr(run_morning, "settings", fake)

    called = {"n": 0}

    async def fake_bundle(*a, **kw):
        called["n"] += 1
        raise AssertionError("should not be called when disabled")

    monkeypatch.setattr(run_morning, "build_bundle", fake_bundle)
    with pytest.raises(SystemExit) as exc:
        run_morning.main()
    assert exc.value.code == 0
    assert called["n"] == 0
