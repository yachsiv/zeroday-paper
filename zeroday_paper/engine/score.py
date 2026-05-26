"""Pure scoring function.

Combines:
    - Regime gates (skip if too risky)
    - GEX level proximity bonuses
    - Volatility bonuses
    - L1 + L2 pattern bonuses

Maximum theoretical score: ~30. Threshold for paper entry: settings.engine.score_threshold (15).
"""

from __future__ import annotations

from datetime import time
from zoneinfo import ZoneInfo

from zeroday_paper.config import settings
from zeroday_paper.engine.gex_levels import compute_proximity, compute_rr25
from zeroday_paper.engine.models import (
    MarketState,
    PatternHit,
    ScoreResult,
    StrategyType,
)

ET = ZoneInfo(settings.engine.market_timezone)


def regime_ok(state: MarketState) -> tuple[bool, str | None]:
    """Top-level skip gates. Returns (ok, reason_if_blocked)."""
    g = settings.regime_gates
    if g.skip_negative_gex and state.signals.gamma_regime == "negative_gamma":
        return False, "negative_gamma_skip"
    if state.vols.vix_1d is not None and state.vols.vix_1d > g.vix_1d_skip_threshold:
        return False, f"vix1d_{state.vols.vix_1d}_above_{g.vix_1d_skip_threshold}"
    if state.vols.cboe_skew is not None and state.vols.cboe_skew > g.cboe_skew_skip_threshold:
        return False, f"skew_{state.vols.cboe_skew}_above_{g.cboe_skew_skip_threshold}"

    now_et = state.asof.astimezone(ET).time()
    if now_et < settings.engine.session_start:
        return False, "before_open"
    if now_et > settings.engine.session_end:
        return False, "after_session_end"
    if now_et > settings.engine.new_entry_cutoff:
        return False, "after_entry_cutoff_13_30"

    open_plus_warmup = time(
        settings.engine.session_start.hour,
        settings.engine.session_start.minute + settings.engine.warmup_minutes_after_open,
    )
    if now_et < open_plus_warmup:
        return False, "warmup_period"

    return True, None


def score_state(
    state: MarketState,
    strategy: StrategyType,
    *,
    l1_hits: list[PatternHit] | None = None,
    l2_hits: list[PatternHit] | None = None,
) -> ScoreResult:
    """Pure scorer. Higher is better."""
    breakdown: dict[str, int] = {}
    notes: list[str] = []

    ok, reason = regime_ok(state)
    if not ok:
        breakdown["regime_blocked"] = 0
        notes.append(f"regime_block:{reason}")
        return ScoreResult(total=0, breakdown=breakdown, regime_ok=False, notes=notes)

    # Base ----------------------------------------------------------------
    base = 10
    breakdown["base"] = base

    # Regime alignment ----------------------------------------------------
    regime_bonus = 0
    if state.signals.gamma_regime == "positive_gamma":
        regime_bonus = 4
        notes.append("positive_gamma:+4")
    breakdown["regime_alignment"] = regime_bonus

    # GEX proximity -------------------------------------------------------
    prox = compute_proximity(state)
    prox_bonus = 0
    if prox.above_flip is True:
        prox_bonus += 1
        notes.append("above_gamma_flip:+1")
    if strategy == StrategyType.BULL_PUT and prox.to_put_wall is not None and prox.to_put_wall < 25:
        prox_bonus += 2
        notes.append("near_put_wall_for_bull_put:+2")
    if strategy == StrategyType.BEAR_CALL and prox.to_call_wall is not None and prox.to_call_wall < 25 and prox.to_call_wall > -10:
        prox_bonus += 2
        notes.append("near_call_wall_for_bear_call:+2")
    if state.signals.pin_score is not None and state.signals.pin_score >= 70:
        prox_bonus += 2
        notes.append(f"pin_score_{state.signals.pin_score}:+2")
    breakdown["gex_proximity"] = prox_bonus

    # Volatility ----------------------------------------------------------
    vol_bonus = 0
    if state.vols.vix_1d is not None and state.vols.vix_1d < 14:
        vol_bonus += 2
        notes.append("vix1d_calm:+2")
    rr25 = compute_rr25(state)
    if rr25 is not None and rr25 < 5:
        vol_bonus += 1
        notes.append("skew_normal:+1")
    breakdown["volatility"] = vol_bonus

    # Patterns ------------------------------------------------------------
    pat_bonus = 0
    if l1_hits:
        pat_bonus += sum(h.score_bonus for h in l1_hits)
    if l2_hits:
        pat_bonus += sum(h.score_bonus for h in l2_hits)
    breakdown["patterns"] = pat_bonus

    total = base + regime_bonus + prox_bonus + vol_bonus + pat_bonus
    return ScoreResult(total=total, breakdown=breakdown, regime_ok=True, notes=notes)
