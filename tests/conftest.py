"""Shared pytest fixtures + bootstrap.

Sets dummy credentials BEFORE importing zeroday_paper so that lazy secret
fetches do not attempt to hit boto3/Secrets Manager. Provides builders for
the core dataclasses + a tmp DuckDB Journal.
"""

from __future__ import annotations

import os

os.environ.setdefault("POLYGON_API_KEY", "test-polygon-key")
os.environ.setdefault("FLASHALPHA_API_KEY", "test-flashalpha-key")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-anthropic-key")
os.environ.setdefault("DISCORD_WEBHOOK_URL", "https://discord.test/webhook")
# Prevent any boto3/Secrets-Manager fallthrough during tests.
os.environ.setdefault("AWS_ACCESS_KEY_ID", "test-fake")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "test-fake")
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
# Belt-and-suspenders: signal metrics module to short-circuit if accidentally invoked
os.environ.setdefault("ZP_METRICS_DISABLED", "1")

from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
from typing import Callable

import pytest

from zeroday_paper.data.cboe_client import VolSnapshot
from zeroday_paper.data.flashalpha_client import MarketSignals
from zeroday_paper.data.polygon_client import ChainSnapshot, OptionQuote
from zeroday_paper.engine.journal import Journal
from zeroday_paper.engine.models import MarketState, PaperTrade, StrategyType

ET = ZoneInfo("America/New_York")


# --------------------------------------------------------------------------- builders


def _make_quote(
    *,
    strike: float,
    right: str = "P",
    bid: float = 1.0,
    ask: float = 1.2,
    delta: float | None = -0.18,
    gamma: float | None = 0.01,
    theta: float | None = -0.5,
    vega: float | None = 0.2,
    iv: float | None = 0.20,
    open_interest: int = 1000,
    volume: int = 500,
    expiry: date | None = None,
    contract: str | None = None,
) -> OptionQuote:
    """Lightweight OptionQuote builder for tests."""
    expiry = expiry or date(2025, 5, 28)
    mid = (bid + ask) / 2.0 if bid > 0 and ask > 0 else 0.0
    return OptionQuote(
        contract=contract or f"O:SPXW{expiry.strftime('%y%m%d')}{right}{int(strike*1000):08d}",
        strike=float(strike),
        right=right,
        expiry=expiry,
        bid=bid,
        ask=ask,
        mid=mid,
        last=mid,
        iv=iv,
        delta=delta,
        gamma=gamma,
        theta=theta,
        vega=vega,
        open_interest=open_interest,
        volume=volume,
    )


@pytest.fixture
def make_quote() -> Callable[..., OptionQuote]:
    return _make_quote


def _make_chain(
    *,
    spot: float = 5800.0,
    expiry: date | None = None,
    n_strikes: int = 21,
    step: int = 5,
) -> ChainSnapshot:
    """Build a small synthetic chain centered on `spot`."""
    expiry = expiry or date(2025, 5, 28)
    base = round(spot / step) * step
    calls = []
    puts = []
    for i in range(-(n_strikes // 2), n_strikes // 2 + 1):
        k = base + i * step
        # Approximate delta by distance from spot
        d_dist = (k - spot) / 100.0
        call_delta = max(0.01, min(0.99, 0.5 - d_dist))
        put_delta = max(-0.99, min(-0.01, -0.5 - d_dist))
        gamma = max(0.001, 0.02 - abs(d_dist) * 0.01)
        calls.append(_make_quote(
            strike=float(k), right="C",
            bid=max(0.05, 5.0 - d_dist * 5.0),
            ask=max(0.10, 5.2 - d_dist * 5.0),
            delta=call_delta, gamma=gamma, expiry=expiry,
            open_interest=1500, volume=500,
        ))
        puts.append(_make_quote(
            strike=float(k), right="P",
            bid=max(0.05, 5.0 + d_dist * 5.0),
            ask=max(0.10, 5.2 + d_dist * 5.0),
            delta=put_delta, gamma=gamma, expiry=expiry,
            open_interest=1500, volume=500,
        ))
    return ChainSnapshot(
        fetched_at=datetime(2025, 5, 28, 14, 30, tzinfo=UTC),
        spot=spot,
        expiry=expiry,
        calls=sorted(calls, key=lambda q: q.strike),
        puts=sorted(puts, key=lambda q: q.strike),
    )


@pytest.fixture
def make_chain() -> Callable[..., ChainSnapshot]:
    return _make_chain


def _make_signals(
    *,
    spot: float = 5800.0,
    gamma_regime: str = "positive_gamma",
    gamma_flip: float | None = 5790.0,
    call_wall: float | None = 5850.0,
    put_wall: float | None = 5750.0,
    max_pain: float | None = 5800.0,
    magnet_strike: float | None = 5800.0,
    pin_score: float | None = 70.0,
    total_gex: float | None = 2.0,
    source: str = "flashalpha",
) -> MarketSignals:
    return MarketSignals(
        fetched_at=datetime(2025, 5, 28, 14, 30, tzinfo=UTC),
        source=source,
        symbol="SPX",
        spot=spot,
        gamma_regime=gamma_regime,
        gamma_flip=gamma_flip,
        call_wall=call_wall,
        put_wall=put_wall,
        max_pain=max_pain,
        magnet_strike=magnet_strike,
        pin_score=pin_score,
        zero_dte_gex_share=0.7,
        remaining_1sd=20.0,
        full_day_1sd=40.0,
        hours_remaining=4.0,
        total_gex=total_gex,
    )


@pytest.fixture
def make_signals() -> Callable[..., MarketSignals]:
    return _make_signals


def _make_vols(*, vix_1d: float | None = 12.0, cboe_skew: float | None = 130.0) -> VolSnapshot:
    return VolSnapshot(
        fetched_at=datetime(2025, 5, 28, 14, 30, tzinfo=UTC),
        vix_1d=vix_1d,
        cboe_skew=cboe_skew,
    )


@pytest.fixture
def make_vols() -> Callable[..., VolSnapshot]:
    return _make_vols


def _et_to_utc(year: int, month: int, day: int, hour: int, minute: int) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=ET).astimezone(UTC)


@pytest.fixture
def et_to_utc() -> Callable[..., datetime]:
    return _et_to_utc


def _make_state(
    *,
    spot: float = 5800.0,
    asof: datetime | None = None,
    chain: ChainSnapshot | None = None,
    signals: MarketSignals | None = None,
    vols: VolSnapshot | None = None,
) -> MarketState:
    asof = asof or _et_to_utc(2025, 5, 28, 11, 30)
    chain = chain or _make_chain(spot=spot)
    signals = signals or _make_signals(spot=spot)
    vols = vols or _make_vols()
    return MarketState(asof=asof, chain=chain, signals=signals, vols=vols, spot=spot)


@pytest.fixture
def make_state() -> Callable[..., MarketState]:
    return _make_state


@pytest.fixture
def tmp_journal(tmp_path) -> Journal:
    path = tmp_path / "journal.duckdb"
    j = Journal(str(path))
    try:
        yield j
    finally:
        j.close()


@pytest.fixture(autouse=True)
def _silence_metrics(monkeypatch):
    """Make zeroday_paper.metrics.emit a no-op in all tests by default.

    Individual tests that need to assert metric emission can re-patch the
    function on their own module reference.
    """
    try:
        from zeroday_paper import metrics as _metrics
        monkeypatch.setattr(_metrics, "emit", lambda *a, **kw: None)
        # Also short-circuit client init so it never tries to reach boto3.
        monkeypatch.setattr(_metrics, "_get_client", lambda: None)
    except ImportError:
        pass
    yield


def _make_paper_trade(
    *,
    trade_id: str = "trade_abc123",
    strategy: StrategyType = StrategyType.BULL_PUT,
    entry_ts: datetime | None = None,
    expiry: date | None = None,
    short_strike: float = 5775.0,
    long_strike: float = 5750.0,
    width: float = 25.0,
    credit_bid: float = 1.0,
    credit_mid: float = 1.05,
    score: int = 18,
    source: str = "replay",
    pattern: str = "P02",
) -> PaperTrade:
    entry_ts = entry_ts or datetime(2025, 5, 28, 14, 30, tzinfo=UTC)
    expiry = expiry or entry_ts.date()
    return PaperTrade(
        trade_id=trade_id,
        strategy=strategy,
        entry_ts=entry_ts,
        expiry=expiry,
        spot_at_entry=5800.0,
        short_strike=short_strike,
        long_strike=long_strike,
        short_contract=f"O:SPXW{expiry.strftime('%y%m%d')}P{int(short_strike*1000):08d}",
        long_contract=f"O:SPXW{expiry.strftime('%y%m%d')}P{int(long_strike*1000):08d}",
        width=width,
        contracts=1,
        credit_mid=credit_mid,
        credit_bid=credit_bid,
        max_loss_bid=width - credit_bid,
        credit_bid_ratio=credit_bid / width if width > 0 else 0.0,
        short_delta=-0.18,
        long_delta=-0.12,
        short_iv=0.21,
        long_iv=0.22,
        short_gamma=0.01,
        short_theta=-0.5,
        short_vega=0.1,
        short_oi=1000,
        long_oi=800,
        short_volume=400,
        gamma_regime="positive_gamma",
        gamma_flip=5790.0,
        call_wall=5850.0,
        put_wall=5750.0,
        magnet_strike=5800.0,
        pin_score=70.0,
        total_gex_b=2.0,
        zero_dte_gex_share=0.8,
        vix_1d=12.0,
        cboe_skew=130.0,
        rr25=2.5,
        active_patterns_l1=pattern,
        active_patterns_l2="",
        patterns_score_bonus=1,
        score=score,
        score_breakdown_json='{"base":10}',
        source=source,
        notes="",
    )


@pytest.fixture
def make_paper_trade() -> Callable[..., PaperTrade]:
    return _make_paper_trade
