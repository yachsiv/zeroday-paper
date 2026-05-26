"""FlashAlpha client (zero_dte + exposure_levels).

Wraps the official ``flashalpha`` Python SDK and exposes an ``async`` surface
matching the rest of the scanner. The previous bare-HTTP implementation
targeted a base URL (``api.flashalpha.io``) that does not exist; FlashAlpha
distributes the SDK as the only supported client. See the parent
``zeroday-trading`` repo (``agents/flashalpha_client.py``) for the canonical
field mapping.

For replay (no historical FlashAlpha endpoint in the free tier),
``signals_from_chain`` self-computes gamma flip / pin score from the Polygon
chain. The same path is used in live mode if FlashAlpha is unreachable or
the SDK is not installed, so the scanner never hard-fails on FA outages.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import structlog

from zeroday_paper.data.polygon_client import ChainSnapshot
from zeroday_paper.secrets import flashalpha_api_key

logger = structlog.get_logger(__name__)


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
    """Async wrapper around the synchronous ``flashalpha`` SDK.

    The SDK ships a single ``FlashAlpha(api_key)`` factory exposing
    ``zero_dte(symbol)`` and ``exposure_levels(symbol)``. We invoke both
    via ``asyncio.to_thread`` so the scanner's event loop stays unblocked.

    Construction never touches the network; the SDK is lazy-imported in
    ``__aenter__`` so unit tests that monkey-patch ``flashalpha`` keep
    working and a missing SDK raises ``FlashAlphaError`` (which the
    scanner already handles with a self-compute fallback).
    """

    def __init__(self, api_key: str | None = None, timeout: float = 8.0):
        self._api_key = api_key
        self._timeout = timeout
        self._fa: Any | None = None

    async def __aenter__(self) -> "FlashAlphaClient":
        if self._fa is not None:
            return self
        try:
            from flashalpha import FlashAlpha  # type: ignore[import-not-found]
        except ImportError as exc:
            raise FlashAlphaError(
                "flashalpha SDK not installed; run `uv sync` to install"
            ) from exc
        api_key = self._api_key or flashalpha_api_key()
        self._fa = FlashAlpha(api_key)
        return self

    async def __aexit__(self, *_: Any) -> None:
        self._fa = None

    async def _call(self, method: str, symbol: str) -> dict[str, Any]:
        if self._fa is None:  # pragma: no cover - defensive
            raise FlashAlphaError("client not initialised; use `async with`")
        fn = getattr(self._fa, method)
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(fn, symbol), timeout=self._timeout
            )
        except asyncio.TimeoutError as exc:
            raise FlashAlphaError(f"{method} timeout after {self._timeout}s") from exc
        except Exception as exc:
            raise FlashAlphaError(f"{method} failed: {exc}") from exc
        if not isinstance(result, dict):  # pragma: no cover - SDK contract
            raise FlashAlphaError(f"{method} returned {type(result).__name__}, expected dict")
        return result

    async def get_signals(self, symbol: str = "SPX") -> MarketSignals:
        """Live signals via two SDK calls: zero_dte + exposure_levels."""
        zd = await self._call("zero_dte", symbol)
        ex = await self._call("exposure_levels", symbol)

        regime = (zd.get("regime") or {}) if isinstance(zd.get("regime"), dict) else {}
        regime_label = str(regime.get("label") or zd.get("gamma_regime") or "").lower()
        if "pos" in regime_label:
            gamma_regime = "positive_gamma"
        elif "neg" in regime_label:
            gamma_regime = "negative_gamma"
        else:
            gamma_regime = "neutral"

        pin = zd.get("pin_risk") or {}
        em = zd.get("expected_move") or {}
        lvls = ex.get("levels") or {}
        exposures = zd.get("exposures") or {}

        spot = _f(zd.get("underlying_price") or zd.get("spot")) or 0.0

        # 0DTE gamma_flip from regime; broad-chain gamma_flip from exposure_levels.
        # Prefer the chain-wide flip for the scanner since the scorer compares
        # against current spot (zero_dte gamma_flip ≈ spot near close).
        gamma_flip = _f(lvls.get("gamma_flip")) or _f(regime.get("gamma_flip"))

        total_chain_gex_dollars = _f(exposures.get("total_chain_net_gex"))
        total_gex_b: float | None = None
        if total_chain_gex_dollars is not None:
            total_gex_b = total_chain_gex_dollars / 1e9

        zero_dte_share = _f(exposures.get("pct_of_total_gex"))
        if zero_dte_share is not None:
            zero_dte_share = zero_dte_share / 100.0

        return MarketSignals(
            fetched_at=datetime.now(UTC),
            source="flashalpha",
            symbol=symbol,
            spot=spot,
            gamma_regime=gamma_regime,
            gamma_flip=gamma_flip,
            call_wall=_f(lvls.get("call_wall")),
            put_wall=_f(lvls.get("put_wall")),
            max_pain=_f(pin.get("max_pain")),
            magnet_strike=_f(pin.get("magnet_strike") or lvls.get("zero_dte_magnet")),
            pin_score=_f(pin.get("pin_score")),
            zero_dte_gex_share=zero_dte_share,
            remaining_1sd=_f(em.get("remaining_1sd_dollars")),
            full_day_1sd=_f(em.get("implied_1sd_dollars")),
            hours_remaining=_f(zd.get("time_to_close_hours")),
            total_gex=total_gex_b,
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
