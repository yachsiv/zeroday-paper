"""Lean Polygon client for paper trading.

Three concerns only:
    1. SPX spot price (current or as-of timestamp).
    2. Full 0DTE option chain snapshot (current or historical).
    3. Single contract OHLCV at minute resolution (for exit pricing).

Everything else (WS, IV-rank, IV surface fits) is out-of-scope here.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Any

import httpx
import structlog
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from zeroday_paper.secrets import polygon_api_key

logger = structlog.get_logger(__name__)

POLYGON_BASE = "https://api.polygon.io"
SPX_TICKER = "I:SPX"           # SPX cash index ticker on Polygon
SPX_OPT_ROOT = "SPX"           # option underlying root


class PolygonError(RuntimeError):
    pass


class PolygonAuthError(PolygonError):
    pass


class PolygonTransportError(PolygonError):
    pass


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return 500 <= exc.response.status_code < 600
    return isinstance(exc, (httpx.TransportError, httpx.TimeoutException))


@dataclass(frozen=True)
class OptionQuote:
    contract: str          # OCC-style: O:SPXW250528P05800000
    strike: float
    right: str             # "C" or "P"
    expiry: date
    bid: float
    ask: float
    mid: float
    last: float
    iv: float | None
    delta: float | None
    gamma: float | None
    theta: float | None
    vega: float | None
    open_interest: int
    volume: int

    @property
    def bid_ask_spread(self) -> float:
        return self.ask - self.bid

    @property
    def is_tradable(self) -> bool:
        """Has both sides quoted and a sane spread."""
        return self.bid > 0.0 and self.ask > self.bid and self.bid_ask_spread < 5.0


@dataclass(frozen=True)
class ChainSnapshot:
    fetched_at: datetime
    spot: float
    expiry: date
    calls: list[OptionQuote] = field(default_factory=list)
    puts: list[OptionQuote] = field(default_factory=list)

    def total_quotes(self) -> int:
        return len(self.calls) + len(self.puts)

    def atm_strike(self) -> float:
        """Nearest 5-pt strike to spot."""
        return round(self.spot / 5.0) * 5.0


class PolygonClient:
    """Thin async wrapper around Polygon REST."""

    def __init__(self, api_key: str | None = None, timeout: float = 10.0):
        self._api_key = api_key or polygon_api_key()
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "PolygonClient":
        self._client = httpx.AsyncClient(
            base_url=POLYGON_BASE,
            timeout=self._timeout,
            params={"apiKey": self._api_key},
            headers={"User-Agent": "zeroday-paper/0.1"},
        )
        return self

    async def __aexit__(self, *_: Any) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        assert self._client is not None, "use as async context manager"
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
            retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TransportError, httpx.TimeoutException)),
            reraise=True,
        ):
            with attempt:
                resp = await self._client.get(path, params=params or {})
                if resp.status_code in (401, 403):
                    raise PolygonAuthError(f"Polygon rejected key: {resp.status_code}")
                resp.raise_for_status()
                return resp.json()
        raise PolygonTransportError(f"unreachable for {path}")  # pragma: no cover

    async def get_spx_spot(self) -> float:
        """Current SPX index spot price."""
        data = await self._get(f"/v3/snapshot/indices", params={"ticker.any_of": SPX_TICKER})
        results = data.get("results") or []
        if not results:
            raise PolygonError("no SPX snapshot result")
        value = results[0].get("value")
        if value is None:
            session = results[0].get("session") or {}
            value = session.get("price") or session.get("close")
        if value is None:
            raise PolygonError(f"SPX spot missing in snapshot: {results[0]}")
        return float(value)

    async def get_spx_spot_at(self, ts: datetime) -> float:
        """SPX spot as-of `ts` (replay-safe).

        Uses the daily aggregate close for the date. For intraday precision use
        get_minute_bar.
        """
        d = ts.date()
        data = await self._get(
            f"/v2/aggs/ticker/{SPX_TICKER}/range/1/day/{d.isoformat()}/{d.isoformat()}",
        )
        results = data.get("results") or []
        if not results:
            raise PolygonError(f"no SPX daily for {d}")
        return float(results[0]["c"])

    async def get_minute_bar(self, ticker: str, ts: datetime) -> dict[str, float] | None:
        """One-minute bar for `ticker` at the minute containing `ts`."""
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        from_ms = int(ts.replace(second=0, microsecond=0).timestamp() * 1000)
        to_ms = from_ms + 60_000 - 1
        data = await self._get(
            f"/v2/aggs/ticker/{ticker}/range/1/minute/{from_ms}/{to_ms}",
        )
        results = data.get("results") or []
        if not results:
            return None
        bar = results[0]
        return {
            "open": float(bar["o"]),
            "high": float(bar["h"]),
            "low": float(bar["l"]),
            "close": float(bar["c"]),
            "volume": float(bar.get("v", 0)),
        }

    async def get_chain_snapshot(self, expiry: date, *, spot_override: float | None = None) -> ChainSnapshot:
        """Full 0DTE chain snapshot for `expiry`.

        Walks the paginated `/v3/snapshot/options/{underlying}` endpoint, filters
        to the requested expiry, and returns calls + puts.
        """
        spot = spot_override if spot_override is not None else await self.get_spx_spot()

        all_quotes: list[OptionQuote] = []
        next_url: str | None = f"/v3/snapshot/options/{SPX_OPT_ROOT}"
        params: dict[str, Any] | None = {
            "expiration_date": expiry.isoformat(),
            "limit": 250,
        }

        page_count = 0
        while next_url is not None and page_count < 20:
            page_count += 1
            data = await self._get(next_url, params=params)
            params = None  # only first page uses params
            results = data.get("results") or []
            for r in results:
                q = self._parse_snapshot_result(r, expiry)
                if q is not None:
                    all_quotes.append(q)
            next_url = self._extract_next_url(data.get("next_url"))

        calls = [q for q in all_quotes if q.right == "C"]
        puts = [q for q in all_quotes if q.right == "P"]
        calls.sort(key=lambda q: q.strike)
        puts.sort(key=lambda q: q.strike)

        return ChainSnapshot(
            fetched_at=datetime.now(UTC),
            spot=spot,
            expiry=expiry,
            calls=calls,
            puts=puts,
        )

    @staticmethod
    def _extract_next_url(raw: str | None) -> str | None:
        if not raw:
            return None
        if raw.startswith(POLYGON_BASE):
            return raw[len(POLYGON_BASE):]
        return raw

    @staticmethod
    def _parse_snapshot_result(r: dict[str, Any], expiry: date) -> OptionQuote | None:
        try:
            details = r.get("details") or {}
            quote = r.get("last_quote") or {}
            last_trade = r.get("last_trade") or {}
            day = r.get("day") or {}
            greeks = r.get("greeks") or {}

            contract = details.get("ticker") or r.get("ticker")
            strike = details.get("strike_price")
            right_raw = (details.get("contract_type") or "").lower()
            if not contract or strike is None or not right_raw:
                return None
            right = "C" if right_raw.startswith("c") else "P"

            bid = float(quote.get("bid", 0.0) or 0.0)
            ask = float(quote.get("ask", 0.0) or 0.0)
            last = float(last_trade.get("price", 0.0) or 0.0)
            mid = (bid + ask) / 2.0 if (bid > 0 and ask > 0) else (last or 0.0)

            return OptionQuote(
                contract=str(contract),
                strike=float(strike),
                right=right,
                expiry=expiry,
                bid=bid,
                ask=ask,
                mid=mid,
                last=last,
                iv=_safe_float(r.get("implied_volatility")),
                delta=_safe_float(greeks.get("delta")),
                gamma=_safe_float(greeks.get("gamma")),
                theta=_safe_float(greeks.get("theta")),
                vega=_safe_float(greeks.get("vega")),
                open_interest=int(r.get("open_interest") or 0),
                volume=int(day.get("volume") or 0),
            )
        except (TypeError, ValueError, KeyError) as exc:
            logger.debug("polygon.snapshot.skip", error=str(exc), payload_keys=list(r.keys()))
            return None

    async def get_chain_snapshot_at(self, expiry: date, ts: datetime) -> ChainSnapshot:
        """Historical chain snapshot for `expiry` as-of timestamp `ts`.

        Replay-safe: looks up each contract's minute bar for the minute
        containing `ts` and reconstructs bid/ask/mid from the bar close.

        Trade-off: bar close is not a true quote. For paper-trading P&L it is
        a reasonable proxy when paired with bid/ask realism in the live path.
        """
        spot = await self.get_spx_spot_at(ts)

        chain_now = await self.get_chain_snapshot(expiry, spot_override=spot)
        contracts = chain_now.calls + chain_now.puts

        async def fetch_for(q: OptionQuote) -> OptionQuote:
            bar = await self.get_minute_bar(q.contract, ts)
            if bar is None:
                return q
            close = bar["close"]
            spread_est = max(0.05, close * 0.04)
            bid = max(0.0, close - spread_est / 2.0)
            ask = close + spread_est / 2.0
            return OptionQuote(
                contract=q.contract, strike=q.strike, right=q.right, expiry=q.expiry,
                bid=bid, ask=ask, mid=close, last=close,
                iv=q.iv, delta=q.delta, gamma=q.gamma, theta=q.theta, vega=q.vega,
                open_interest=q.open_interest, volume=int(bar.get("volume", 0)),
            )

        sem = asyncio.Semaphore(8)

        async def bounded(q: OptionQuote) -> OptionQuote:
            async with sem:
                return await fetch_for(q)

        priced = await asyncio.gather(*[bounded(c) for c in contracts])
        calls = sorted([q for q in priced if q.right == "C"], key=lambda q: q.strike)
        puts = sorted([q for q in priced if q.right == "P"], key=lambda q: q.strike)

        return ChainSnapshot(
            fetched_at=ts.astimezone(UTC),
            spot=spot,
            expiry=expiry,
            calls=calls,
            puts=puts,
        )


def _safe_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def next_spx_expiry(reference: date | None = None) -> date:
    """Next available SPX 0DTE expiry (M/W/F).

    SPX has Mon/Wed/Fri 0DTE expirations. This returns the same-day expiry if
    today is one of them, else the next available.
    """
    ref = reference or datetime.now(UTC).date()
    weekday = ref.weekday()  # 0=Mon, 4=Fri
    if weekday in (0, 2, 4):  # Mon, Wed, Fri
        return ref
    offsets = {1: 1, 3: 1, 5: 2, 6: 1}  # Tue->Wed, Thu->Fri, Sat->Mon, Sun->Mon
    return ref + timedelta(days=offsets[weekday])
