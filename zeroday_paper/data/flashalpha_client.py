"""FlashAlpha client (zero_dte + exposure_levels).

Wraps the `flashalpha` Python SDK if available; falls back to bare HTTP if not.
Surfaces a single `get_signals(symbol)` returning a `MarketSignals` snapshot used
by the scorer and pattern classifier.

For replay (no historical FlashAlpha endpoint in the free tier), `get_signals_at`
self-computes gamma flip / pin score from the Polygon chain. This keeps replay
leak-free.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import httpx
import structlog
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from zeroday_paper.data.polygon_client import ChainSnapshot
from zeroday_paper.secrets import flashalpha_api_key

logger = structlog.get_logger(__name__)

FLASHALPHA_BASE = "https://api.flashalpha.io"


class FlashAlphaError(RuntimeError):
    pass


@dataclass(frozen=True)
class MarketSignals:
    fetched_at: datetime
    source: str                          # "flashalpha" or "self_computed"
    symbol: str
    spot: float
    gamma_regime: str                    # "positive_gamma" | "negative_gamma" | "neutral"
    gamma_flip: float | None
    call_wall: float | None
    put_wall: float | None
    max_pain: float | None
    magnet_strike: float | None
    pin_score: float | None              # 0..100
    zero_dte_gex_share: float | None     # 0..1
    remaining_1sd: float | None
    full_day_1sd: float | None
    hours_remaining: float | None
    total_gex: float | None              # billions (signed)
    raw: dict[str, Any] | None = None


class FlashAlphaClient:
    """Async client for FlashAlpha REST API.

    The SDK is dropped in favor of a direct REST shim — keeps this repo's
    dependency surface minimal.
    """

    def __init__(self, api_key: str | None = None, timeout: float = 8.0):
        self._api_key = api_key or flashalpha_api_key()
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "FlashAlphaClient":
        self._client = httpx.AsyncClient(
            base_url=FLASHALPHA_BASE,
            timeout=self._timeout,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "User-Agent": "zeroday-paper/0.1",
            },
        )
        return self

    async def __aexit__(self, *_: Any) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        assert self._client is not None
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
            retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TransportError, httpx.TimeoutException)),
            reraise=True,
        ):
            with attempt:
                resp = await self._client.get(path, params=params or {})
                if resp.status_code in (401, 403):
                    raise FlashAlphaError(f"auth failed: {resp.status_code}")
                resp.raise_for_status()
                return resp.json()
        raise FlashAlphaError(f"unreachable for {path}")  # pragma: no cover

    async def get_signals(self, symbol: str = "SPX") -> MarketSignals:
        """Live signals via two calls: zero_dte + exposure_levels."""
        zd = await self._get("/v1/zero_dte", params={"symbol": symbol})
        ex = await self._get("/v1/exposure_levels", params={"symbol": symbol})

        regime_raw = (zd.get("gamma_regime") or "").lower()
        if "pos" in regime_raw:
            regime = "positive_gamma"
        elif "neg" in regime_raw:
            regime = "negative_gamma"
        else:
            regime = "neutral"

        return MarketSignals(
            fetched_at=datetime.now(UTC),
            source="flashalpha",
            symbol=symbol,
            spot=float(zd.get("spot") or zd.get("underlying_price") or 0.0),
            gamma_regime=regime,
            gamma_flip=_f(ex.get("gamma_flip") or zd.get("gamma_flip")),
            call_wall=_f(ex.get("call_wall")),
            put_wall=_f(ex.get("put_wall")),
            max_pain=_f(ex.get("max_pain") or zd.get("max_pain")),
            magnet_strike=_f(zd.get("magnet_strike")),
            pin_score=_f(zd.get("pin_score")),
            zero_dte_gex_share=_f(zd.get("zero_dte_gex_share")),
            remaining_1sd=_f(zd.get("remaining_1sd")),
            full_day_1sd=_f(zd.get("full_day_1sd")),
            hours_remaining=_f(zd.get("hours_remaining")),
            total_gex=_f(ex.get("total_gex")),
            raw={"zero_dte": zd, "exposure_levels": ex},
        )


def _f(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Self-computed GEX (fallback + replay)
# ---------------------------------------------------------------------------


def signals_from_chain(chain: ChainSnapshot, *, symbol: str = "SPX") -> MarketSignals:
    """Compute GEX-derived signals directly from a Polygon ChainSnapshot.

    Used when:
    - FlashAlpha is unavailable, OR
    - We are replaying (no historical FlashAlpha endpoint).

    Formula (dealer GEX, assuming dealers are short calls / long puts net):
        gex_per_contract = gamma * 100 * spot^2 * 0.01 * OI
        call_gex contributes +
        put_gex contributes -
    """
    spot = chain.spot
    if spot <= 0:
        return _empty_signals(symbol, spot)

    call_gex_by_strike: dict[float, float] = {}
    put_gex_by_strike: dict[float, float] = {}

    for q in chain.calls:
        if q.gamma is None or q.open_interest <= 0:
            continue
        gex = q.gamma * 100.0 * (spot ** 2) * 0.01 * q.open_interest
        call_gex_by_strike[q.strike] = call_gex_by_strike.get(q.strike, 0.0) + gex

    for q in chain.puts:
        if q.gamma is None or q.open_interest <= 0:
            continue
        gex = q.gamma * 100.0 * (spot ** 2) * 0.01 * q.open_interest
        put_gex_by_strike[q.strike] = put_gex_by_strike.get(q.strike, 0.0) + gex

    total_call_gex = sum(call_gex_by_strike.values())
    total_put_gex = sum(put_gex_by_strike.values())
    total_gex = (total_call_gex - total_put_gex) / 1e9   # billions

    regime = "positive_gamma" if total_gex >= 0 else "negative_gamma"

    call_wall: float | None = max(call_gex_by_strike, key=call_gex_by_strike.get, default=None) if call_gex_by_strike else None
    put_wall: float | None = max(put_gex_by_strike, key=put_gex_by_strike.get, default=None) if put_gex_by_strike else None

    gamma_flip = _approx_gamma_flip(call_gex_by_strike, put_gex_by_strike)

    max_pain = _approx_max_pain(chain)
    magnet_strike = _approx_magnet(chain)
    pin_score = _approx_pin_score(spot, magnet_strike)

    return MarketSignals(
        fetched_at=chain.fetched_at,
        source="self_computed",
        symbol=symbol,
        spot=spot,
        gamma_regime=regime,
        gamma_flip=gamma_flip,
        call_wall=call_wall,
        put_wall=put_wall,
        max_pain=max_pain,
        magnet_strike=magnet_strike,
        pin_score=pin_score,
        zero_dte_gex_share=1.0,
        remaining_1sd=None,
        full_day_1sd=None,
        hours_remaining=None,
        total_gex=total_gex,
    )


def _empty_signals(symbol: str, spot: float) -> MarketSignals:
    return MarketSignals(
        fetched_at=datetime.now(UTC),
        source="self_computed_empty",
        symbol=symbol,
        spot=spot,
        gamma_regime="neutral",
        gamma_flip=None,
        call_wall=None,
        put_wall=None,
        max_pain=None,
        magnet_strike=None,
        pin_score=None,
        zero_dte_gex_share=None,
        remaining_1sd=None,
        full_day_1sd=None,
        hours_remaining=None,
        total_gex=None,
    )


def _approx_gamma_flip(call_gex: dict[float, float], put_gex: dict[float, float]) -> float | None:
    """Strike where net dealer gamma changes sign (interpolate)."""
    strikes = sorted(set(call_gex) | set(put_gex))
    if len(strikes) < 2:
        return None
    last_net: float | None = None
    flip: float | None = None
    for k in strikes:
        net = call_gex.get(k, 0.0) - put_gex.get(k, 0.0)
        if last_net is not None and last_net * net < 0:
            flip = k
            break
        last_net = net
    return flip


def _approx_max_pain(chain: ChainSnapshot) -> float | None:
    """Strike with minimum total payout to option holders."""
    strikes = sorted({q.strike for q in chain.calls} | {q.strike for q in chain.puts})
    if not strikes:
        return None
    best_strike: float | None = None
    best_payout = float("inf")
    for k in strikes:
        call_payout = sum(max(0.0, k - q.strike) * q.open_interest for q in chain.calls)
        put_payout = sum(max(0.0, q.strike - k) * q.open_interest for q in chain.puts)
        total = call_payout + put_payout
        if total < best_payout:
            best_payout = total
            best_strike = k
    return best_strike


def _approx_magnet(chain: ChainSnapshot) -> float | None:
    """Strike with the highest combined call+put OI within ATM ±50 pts."""
    if chain.spot <= 0:
        return None
    by_strike: dict[float, int] = {}
    for q in chain.calls + chain.puts:
        if abs(q.strike - chain.spot) <= 50.0:
            by_strike[q.strike] = by_strike.get(q.strike, 0) + q.open_interest
    if not by_strike:
        return None
    return max(by_strike, key=by_strike.get)


def _approx_pin_score(spot: float, magnet: float | None) -> float | None:
    if magnet is None or spot <= 0:
        return None
    distance_pts = abs(spot - magnet)
    score = max(0.0, 100.0 - distance_pts * 4.0)
    return round(score, 2)
