"""CBOE free-tier client.

Pulls VIX1D (1-day VIX) and CBOE SKEW Index. Both are delayed; both are public.
Used as regime-gate inputs (skip new entries when realized vol high or tail risk high).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import UTC, date, datetime
from io import StringIO
from typing import Any

import httpx
import structlog
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = structlog.get_logger(__name__)


class CboeError(RuntimeError):
    pass


# CBOE publishes free historical CSVs at these URLs.
VIX1D_HISTORICAL_CSV = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX1D_History.csv"
SKEW_HISTORICAL_CSV = "https://cdn.cboe.com/api/global/us_indices/daily_prices/SKEW_History.csv"
VIX1D_QUOTE_JSON = "https://cdn.cboe.com/api/global/delayed_quotes/quotes/_VIX1D.json"
SKEW_QUOTE_JSON = "https://cdn.cboe.com/api/global/delayed_quotes/quotes/_SKEW.json"


@dataclass(frozen=True)
class VolSnapshot:
    fetched_at: datetime
    vix_1d: float | None
    cboe_skew: float | None


class CboeClient:
    def __init__(self, timeout: float = 8.0):
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "CboeClient":
        self._client = httpx.AsyncClient(
            timeout=self._timeout,
            headers={"User-Agent": "zeroday-paper/0.1"},
        )
        return self

    async def __aexit__(self, *_: Any) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _get(self, url: str) -> httpx.Response:
        assert self._client is not None
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
            retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TransportError, httpx.TimeoutException)),
            reraise=True,
        ):
            with attempt:
                resp = await self._client.get(url)
                resp.raise_for_status()
                return resp
        raise CboeError(f"unreachable: {url}")  # pragma: no cover

    async def get_live_snapshot(self) -> VolSnapshot:
        """Latest delayed quote for VIX1D + SKEW."""
        vix_1d: float | None = None
        skew: float | None = None
        try:
            r = await self._get(VIX1D_QUOTE_JSON)
            data = r.json()
            vix_1d = _extract_quote_last(data)
        except Exception as exc:
            logger.warning("cboe.vix1d.failed", error=str(exc))
        try:
            r = await self._get(SKEW_QUOTE_JSON)
            data = r.json()
            skew = _extract_quote_last(data)
        except Exception as exc:
            logger.warning("cboe.skew.failed", error=str(exc))

        return VolSnapshot(fetched_at=datetime.now(UTC), vix_1d=vix_1d, cboe_skew=skew)

    async def get_historical_snapshot(self, target: date) -> VolSnapshot:
        """As-of close for `target` date — used by replay."""
        vix_1d = await self._lookup_csv_close(VIX1D_HISTORICAL_CSV, target)
        skew = await self._lookup_csv_close(SKEW_HISTORICAL_CSV, target)
        return VolSnapshot(
            fetched_at=datetime.combine(target, datetime.min.time(), tzinfo=UTC),
            vix_1d=vix_1d,
            cboe_skew=skew,
        )

    async def _lookup_csv_close(self, url: str, target: date) -> float | None:
        try:
            r = await self._get(url)
        except Exception as exc:
            logger.warning("cboe.csv.failed", url=url, error=str(exc))
            return None

        reader = csv.reader(StringIO(r.text))
        rows = list(reader)
        if len(rows) < 2:
            return None

        header = [h.strip().upper() for h in rows[0]]
        try:
            date_idx = next(i for i, h in enumerate(header) if "DATE" in h)
        except StopIteration:
            return None
        try:
            close_idx = next(i for i, h in enumerate(header) if h in ("CLOSE", "VIX1D", "SKEW"))
        except StopIteration:
            close_idx = len(header) - 1

        for row in rows[1:]:
            if not row or len(row) <= max(date_idx, close_idx):
                continue
            try:
                row_date = _parse_cboe_date(row[date_idx])
            except ValueError:
                continue
            if row_date == target:
                try:
                    return float(row[close_idx])
                except (TypeError, ValueError):
                    return None
        return None


def _extract_quote_last(payload: dict[str, Any]) -> float | None:
    data = payload.get("data") or {}
    for key in ("last", "current_price", "close", "iv"):
        val = data.get(key)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                continue
    return None


def _parse_cboe_date(raw: str) -> date:
    raw = raw.strip()
    for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%m/%d/%y"):
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"unparseable CBOE date: {raw}")
