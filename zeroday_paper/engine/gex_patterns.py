"""Layer 1: rule-based GEX pattern library (15 patterns).

Ported from zeroday-trading's `scoring/gex_patterns.py` (originally adapted from
github.com/iAmGiG/gex-llm-patterns). Each pattern is a pure function over a
conditions dict. Missing fields default conservatively so replay (without
FlashAlpha HIRO/stability) still produces honest results.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, time
from enum import StrEnum
from zoneinfo import ZoneInfo

from zeroday_paper.config import settings
from zeroday_paper.engine.models import MarketState

ET = ZoneInfo(settings.engine.market_timezone)


class PatternConfidence(StrEnum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


@dataclass(frozen=True)
class PatternMatch:
    pattern_id: str
    name: str
    matched: bool
    confidence: PatternConfidence
    direction: str             # "BULLISH" / "BEARISH" / "NEUTRAL"
    score_bonus: int
    description: str


def conditions_from_state(state: MarketState, prior_state: MarketState | None = None) -> dict:
    """Translate a MarketState into the dict the pattern functions expect."""
    s = state.signals
    spot = state.spot
    gex_b = s.total_gex or 0.0

    if prior_state is not None:
        prev_gex = prior_state.signals.total_gex or gex_b
    else:
        prev_gex = gex_b

    now_et = state.asof.astimezone(ET)
    hours_to_close = max(0.0, (datetime.combine(now_et.date(), time(16, 0), ET) - now_et).total_seconds() / 3600.0)

    stability_pct = 100.0 if s.gamma_regime == "positive_gamma" else 10.0
    if s.pin_score is not None:
        stability_pct = max(stability_pct, s.pin_score)

    return {
        "gex_value": gex_b,
        "prev_gex_value": prev_gex,
        "gex_sign": "positive" if gex_b >= 0 else "negative",
        "stability_pct": stability_pct,
        "hiro_direction": "neutral",                     # no HIRO without FlashAlpha
        "prev_hiro_direction": "neutral",
        "spx_level": spot,
        "zero_gamma_level": s.gamma_flip or spot,
        "vol_trigger_level": s.put_wall or (spot - 30),
        "call_wall_level": s.call_wall or (spot + 50),
        "put_wall_level": s.put_wall or (spot - 50),
        "vix_change_pct": 0.0,
        "vix_direction": "flat",
        "iv_rank": 50,
        "volume_to_oi_ratio": 1.0,
        "bollinger_band_width": 10.0,
        "atr_14": 20.0,
        "hours_to_close": hours_to_close,
        "max_pain_strike": s.max_pain or spot,
    }


class GEXPatternLibrary:
    """Runs all 15 patterns; returns matches with confidence + direction."""

    def match_all(self, conditions: dict) -> list[PatternMatch]:
        return [m(conditions) for m in self._all]

    def best_match(self, conditions: dict) -> PatternMatch | None:
        matches = [m for m in self.match_all(conditions) if m.matched]
        if not matches:
            return None
        order = {PatternConfidence.HIGH: 0, PatternConfidence.MEDIUM: 1, PatternConfidence.LOW: 2}
        return sorted(matches, key=lambda m: order[m.confidence])[0]

    @property
    def _all(self):
        return [
            self._p01, self._p02, self._p03, self._p04, self._p05,
            self._p06, self._p07, self._p08, self._p09, self._p10,
            self._p11, self._p12, self._p13, self._p14, self._p15,
        ]

    def _p01(self, c: dict) -> PatternMatch:
        matched = c["gex_value"] < -1.0 and c["stability_pct"] < 20
        return PatternMatch("P01", "Negative Gamma Squeeze", matched,
                            PatternConfidence.HIGH if matched else PatternConfidence.LOW,
                            "BEARISH", 1 if matched else 0,
                            "High negative GEX + low stability = directional pressure down")

    def _p02(self, c: dict) -> PatternMatch:
        matched = c["gex_value"] > 0.5 and c["stability_pct"] > 30
        return PatternMatch("P02", "Positive Gamma Pin", matched,
                            PatternConfidence.MEDIUM, "NEUTRAL", 1 if matched else 0,
                            "Positive GEX + high stability = range-bound, iron condor day")

    def _p03(self, c: dict) -> PatternMatch:
        matched = abs(c["spx_level"] - c["zero_gamma_level"]) < 15
        return PatternMatch("P03", "Zero Gamma Flip Zone", matched,
                            PatternConfidence.HIGH if matched else PatternConfidence.LOW,
                            "NEUTRAL", 0,
                            "SPX near zero gamma — dealer behavior about to flip, avoid spreads")

    def _p04(self, c: dict) -> PatternMatch:
        matched = (c["gex_sign"] == "negative" and c["hiro_direction"] == "bullish") or \
                  (c["gex_sign"] == "positive" and c["hiro_direction"] == "bearish")
        return PatternMatch("P04", "HIRO-GEX Divergence", matched,
                            PatternConfidence.MEDIUM, "NEUTRAL", 0,
                            "HIRO conflicts with GEX sign — reduce size or skip")

    def _p05(self, c: dict) -> PatternMatch:
        matched = c["stability_pct"] < 10 and c["vix_change_pct"] > 5
        return PatternMatch("P05", "Stability Collapse", matched,
                            PatternConfidence.HIGH, "BEARISH", 1 if matched else 0,
                            "Stability < 10 + VIX spike = regime breakdown")

    def _p06(self, c: dict) -> PatternMatch:
        matched = c["spx_level"] < c["vol_trigger_level"] and c["gex_sign"] == "negative"
        return PatternMatch("P06", "Vol Trigger Break", matched,
                            PatternConfidence.HIGH if matched else PatternConfidence.LOW,
                            "BEARISH", 1 if matched else 0,
                            "SPX below vol trigger in negative GEX = vol amplification")

    def _p07(self, c: dict) -> PatternMatch:
        cw = c["call_wall_level"]
        matched = cw > c["spx_level"] and (cw - c["spx_level"]) < 20 and c["hiro_direction"] == "bearish"
        return PatternMatch("P07", "Call Wall Rejection", matched,
                            PatternConfidence.HIGH if matched else PatternConfidence.LOW,
                            "BEARISH", 1 if matched else 0,
                            "SPX near call wall + bearish HIRO = rejection likely")

    def _p08(self, c: dict) -> PatternMatch:
        pw = c["put_wall_level"]
        matched = pw < c["spx_level"] and (c["spx_level"] - pw) < 20 and c["hiro_direction"] == "bullish"
        return PatternMatch("P08", "Put Wall Support Bounce", matched,
                            PatternConfidence.HIGH if matched else PatternConfidence.LOW,
                            "BULLISH", 1 if matched else 0,
                            "SPX near put wall + bullish HIRO = support bounce")

    def _p09(self, c: dict) -> PatternMatch:
        matched = c["hours_to_close"] < 2 and c["stability_pct"] < 25
        return PatternMatch("P09", "Charm Acceleration", matched,
                            PatternConfidence.MEDIUM, "BEARISH", 0,
                            "Late session + low stability = charm flow accelerating")

    def _p10(self, c: dict) -> PatternMatch:
        matched = c["iv_rank"] < 30 and c["vix_direction"] == "falling"
        return PatternMatch("P10", "Vanna Tailwind", matched,
                            PatternConfidence.MEDIUM, "BULLISH", 1 if matched else 0,
                            "Low IV rank + falling VIX = vanna flow supporting upside")

    def _p11(self, c: dict) -> PatternMatch:
        matched = c["gex_value"] > 2.0 and c["volume_to_oi_ratio"] > 2.0
        return PatternMatch("P11", "Dealer Long Hedge Unwind", matched,
                            PatternConfidence.MEDIUM, "NEUTRAL", 0,
                            "Very high positive GEX + high volume = dealers unwinding")

    def _p12(self, c: dict) -> PatternMatch:
        cliff = (c["gex_value"] - c["prev_gex_value"]) < -1.0
        matched = cliff and c["gex_value"] < 0
        return PatternMatch("P12", "Gamma Cliff", matched,
                            PatternConfidence.HIGH if matched else PatternConfidence.LOW,
                            "BEARISH", 1 if matched else 0,
                            "Rapid GEX drop into negative = strong directional signal")

    def _p13(self, c: dict) -> PatternMatch:
        matched = c["hiro_direction"] != c["prev_hiro_direction"] and c["prev_hiro_direction"] != "neutral"
        return PatternMatch("P13", "HIRO Reversal", matched,
                            PatternConfidence.LOW, "NEUTRAL", 0,
                            "HIRO direction just flipped — wait for confirmation")

    def _p14(self, c: dict) -> PatternMatch:
        matched = c["bollinger_band_width"] < (c["atr_14"] * 0.5) and c["stability_pct"] < 15
        return PatternMatch("P14", "Squeeze Compression", matched,
                            PatternConfidence.HIGH if matched else PatternConfidence.LOW,
                            "NEUTRAL", 1 if matched else 0,
                            "Bollinger squeeze + ultra-low stability = explosive move imminent")

    def _p15(self, c: dict) -> PatternMatch:
        matched = abs(c["spx_level"] - c["max_pain_strike"]) < 30 and c["hours_to_close"] > 3
        return PatternMatch("P15", "Expiry Magnet", matched,
                            PatternConfidence.MEDIUM, "NEUTRAL", 1 if matched else 0,
                            "SPX near max pain with time remaining = gravitational pull")


def classify_layer1(state: MarketState, prior: MarketState | None = None) -> list[PatternMatch]:
    return [m for m in GEXPatternLibrary().match_all(conditions_from_state(state, prior)) if m.matched]
