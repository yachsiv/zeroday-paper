"""Engine-wide dataclasses.

All types frozen + hashable where practical. Single source of truth for
scoring, journaling, reporting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from enum import StrEnum
from typing import Any

from zeroday_paper.data.cboe_client import VolSnapshot
from zeroday_paper.data.flashalpha_client import MarketSignals
from zeroday_paper.data.polygon_client import ChainSnapshot, OptionQuote


class StrategyType(StrEnum):
    BULL_PUT = "BULL_PUT_SPREAD"
    BEAR_CALL = "BEAR_CALL_SPREAD"
    IRON_CONDOR = "IRON_CONDOR"


class PositionStatus(StrEnum):
    OPEN = "OPEN"
    WINNING = "WINNING"
    LOSING = "LOSING"
    CLOSED_TARGET = "CLOSED_TARGET"
    CLOSED_STOP = "CLOSED_STOP"
    CLOSED_HARD = "CLOSED_HARD_CLOSE"
    CLOSED_THESIS = "CLOSED_THESIS_INVALIDATED"
    CLOSED_EXPIRY = "CLOSED_EXPIRY"
    ERROR = "ERROR"


@dataclass(frozen=True)
class Spread:
    """A two-leg credit spread (one short + one long)."""

    strategy: StrategyType
    short_leg: OptionQuote
    long_leg: OptionQuote
    contracts: int = 1

    @property
    def width(self) -> float:
        return abs(self.short_leg.strike - self.long_leg.strike)

    @property
    def expiry(self) -> date:
        return self.short_leg.expiry


@dataclass(frozen=True)
class MarketState:
    """Bundle: everything the scorer/strike-picker needs for one cycle."""

    asof: datetime
    chain: ChainSnapshot
    signals: MarketSignals
    vols: VolSnapshot
    spot: float

    @property
    def is_replay(self) -> bool:
        return self.signals.source.startswith("self_computed")


@dataclass(frozen=True)
class ScoreResult:
    total: int
    breakdown: dict[str, int]
    regime_ok: bool
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PatternHit:
    pattern_id: str
    name: str
    direction: str             # "BULLISH" / "BEARISH" / "NEUTRAL"
    confidence: str            # "HIGH" / "MEDIUM" / "LOW"
    layer: str                 # "L1_RULES" / "L2_LLM"
    score_bonus: int


@dataclass(frozen=True)
class PaperTrade:
    """Row in `paper_trades` table. 40+ columns by design."""

    trade_id: str
    strategy: StrategyType
    entry_ts: datetime
    expiry: date

    spot_at_entry: float
    short_strike: float
    long_strike: float
    short_contract: str
    long_contract: str
    width: float
    contracts: int

    # Pricing
    credit_mid: float
    credit_bid: float
    max_loss_bid: float
    credit_bid_ratio: float           # credit_bid / width / 100  (e.g. 0.05 = 5%)

    # Greeks at entry
    short_delta: float | None
    long_delta: float | None
    short_iv: float | None
    long_iv: float | None
    short_gamma: float | None
    short_theta: float | None
    short_vega: float | None
    short_oi: int
    long_oi: int
    short_volume: int

    # Market context at entry
    gamma_regime: str
    gamma_flip: float | None
    call_wall: float | None
    put_wall: float | None
    magnet_strike: float | None
    pin_score: float | None
    total_gex_b: float | None
    zero_dte_gex_share: float | None

    # Vols
    vix_1d: float | None
    cboe_skew: float | None
    rr25: float | None

    # Patterns
    active_patterns_l1: str            # comma-separated pattern IDs
    active_patterns_l2: str            # LLM-detected
    patterns_score_bonus: int

    # Score
    score: int
    score_breakdown_json: str
    source: str                        # "live" | "replay"
    notes: str = ""


@dataclass
class TradeTick:
    """One row per polling cycle while a position is open."""

    trade_id: str
    ts: datetime
    spot: float
    short_bid: float
    short_ask: float
    long_bid: float
    long_ask: float
    exit_cost_bid: float
    exit_cost_mid: float
    pnl_bid: float
    pnl_mid: float
    pct_of_max_profit: float
    status: PositionStatus


@dataclass(frozen=True)
class TradeOutcome:
    """Final row when a position is closed."""

    trade_id: str
    exit_ts: datetime
    exit_status: PositionStatus
    exit_spot: float
    exit_cost_bid: float
    exit_cost_mid: float
    pnl_bid: float
    pnl_mid: float
    held_minutes: int
    max_excursion_pct: float          # peak unrealized profit while open
    min_excursion_pct: float          # max drawdown while open
    exit_reason: str
