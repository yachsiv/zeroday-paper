"""Scoring + regime gates."""

from __future__ import annotations

from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import pytest

from zeroday_paper.config import settings
from zeroday_paper.engine.models import PatternHit, StrategyType
from zeroday_paper.engine.score import regime_ok, score_state

ET = ZoneInfo("America/New_York")


def _et_utc(h, m):
    return datetime(2025, 5, 28, h, m, tzinfo=ET).astimezone(UTC)


# ------------------------------------------------------------------ regime_ok gates


def test_regime_ok_passes_at_normal_hour(make_state):
    state = make_state(asof=_et_utc(11, 0))
    ok, reason = regime_ok(state)
    assert ok is True
    assert reason is None


def test_regime_ok_blocks_negative_gamma(make_state, make_signals):
    state = make_state(
        asof=_et_utc(11, 0),
        signals=make_signals(gamma_regime="negative_gamma"),
    )
    ok, reason = regime_ok(state)
    assert ok is False
    assert "negative_gamma" in reason


def test_regime_ok_blocks_high_vix1d(make_state, make_vols):
    state = make_state(asof=_et_utc(11, 0), vols=make_vols(vix_1d=30.0))
    ok, reason = regime_ok(state)
    assert ok is False
    assert "vix1d" in reason


def test_regime_ok_blocks_high_skew(make_state, make_vols):
    state = make_state(asof=_et_utc(11, 0), vols=make_vols(cboe_skew=200.0))
    ok, reason = regime_ok(state)
    assert ok is False
    assert "skew" in reason


def test_regime_ok_blocks_before_session_open(make_state):
    state = make_state(asof=_et_utc(8, 0))
    ok, reason = regime_ok(state)
    assert ok is False
    assert reason in {"before_open"}


def test_regime_ok_blocks_after_session_end(make_state):
    state = make_state(asof=_et_utc(16, 30))
    ok, reason = regime_ok(state)
    assert ok is False
    # session_end=15:45, so 16:30 is after_session_end
    assert reason in {"after_session_end", "after_entry_cutoff_13_30"}


def test_regime_ok_blocks_after_entry_cutoff(make_state):
    # session_start=9:30, cutoff=13:30 → 13:45 should block
    state = make_state(asof=_et_utc(13, 45))
    ok, reason = regime_ok(state)
    assert ok is False
    assert reason == "after_entry_cutoff_13_30"


def test_regime_ok_blocks_during_warmup(make_state):
    # warmup_minutes_after_open = 5; session_start = 9:30; so 9:34 is warmup
    state = make_state(asof=_et_utc(9, 32))
    ok, reason = regime_ok(state)
    assert ok is False
    assert reason == "warmup_period"


# ----------------------------------------------------------------- score_state math


def test_score_state_blocked_returns_zero(make_state, make_signals):
    state = make_state(
        asof=_et_utc(11, 0),
        signals=make_signals(gamma_regime="negative_gamma"),
    )
    res = score_state(state, StrategyType.BULL_PUT)
    assert res.total == 0
    assert res.regime_ok is False
    assert "regime_block" in res.notes[0]


def test_score_state_baseline(make_state):
    state = make_state(asof=_et_utc(11, 0))
    res = score_state(state, StrategyType.BULL_PUT)
    # base 10 + regime_alignment 4 (positive_gamma) + ... should be >= 14
    assert res.total >= 14
    assert res.regime_ok is True
    assert res.breakdown["base"] == 10
    assert res.breakdown["regime_alignment"] == 4


def test_score_state_bull_put_near_put_wall_bonus(make_state, make_signals):
    state = make_state(
        asof=_et_utc(11, 0), spot=5760.0,
        signals=make_signals(spot=5760.0, put_wall=5750.0, gamma_flip=5750.0),
    )
    res = score_state(state, StrategyType.BULL_PUT)
    assert any("near_put_wall_for_bull_put" in n for n in res.notes)


def test_score_state_bear_call_near_call_wall_bonus(make_state, make_signals):
    # to_call_wall must be in (-10, 25): spot just below wall
    state = make_state(
        asof=_et_utc(11, 0), spot=5845.0,
        signals=make_signals(spot=5845.0, call_wall=5850.0, gamma_flip=5800.0),
    )
    res = score_state(state, StrategyType.BEAR_CALL)
    assert any("near_call_wall_for_bear_call" in n for n in res.notes)


def test_score_state_pin_score_bonus(make_state, make_signals):
    state = make_state(
        asof=_et_utc(11, 0),
        signals=make_signals(pin_score=80.0),
    )
    res = score_state(state, StrategyType.BULL_PUT)
    assert any("pin_score" in n for n in res.notes)


def test_score_state_vix_calm_bonus(make_state, make_vols):
    state = make_state(asof=_et_utc(11, 0), vols=make_vols(vix_1d=10.0))
    res = score_state(state, StrategyType.BULL_PUT)
    assert any("vix1d_calm" in n for n in res.notes)


def test_score_state_pattern_bonus_aggregated(make_state):
    state = make_state(asof=_et_utc(11, 0))
    l1 = [PatternHit("P01", "x", "BEARISH", "HIGH", "L1_RULES", 1)]
    l2 = [PatternHit("P02", "y", "NEUTRAL", "MEDIUM", "L2_LLM", 2)]
    res = score_state(state, StrategyType.BULL_PUT, l1_hits=l1, l2_hits=l2)
    assert res.breakdown["patterns"] == 3


def test_score_state_neutral_regime_no_alignment_bonus(make_state, make_signals):
    state = make_state(
        asof=_et_utc(11, 0),
        signals=make_signals(gamma_regime="neutral"),
    )
    res = score_state(state, StrategyType.BULL_PUT)
    assert res.breakdown["regime_alignment"] == 0


def test_score_state_iron_condor_no_wall_bonus(make_state, make_signals):
    state = make_state(
        asof=_et_utc(11, 0), spot=5760.0,
        signals=make_signals(spot=5760.0, put_wall=5750.0),
    )
    res = score_state(state, StrategyType.IRON_CONDOR)
    assert all("near_put_wall_for_bull_put" not in n for n in res.notes)
    assert all("near_call_wall_for_bear_call" not in n for n in res.notes)


# ------------------------------------------------------------- threshold regression


def test_score_threshold_is_13_for_paper_warmup_window():
    """Regression: paper.toml `score_threshold` was lowered 15 → 13 on
    2026-05-26 to start writing paper trades faster while FlashAlpha + Anthropic
    fixes bake in. This test fails if it's accidentally raised back before we
    have ≥30 trades to evaluate quality."""
    assert settings.engine.score_threshold == 13, (
        "score_threshold must remain 13 for the first 5 paper trading days. "
        "See TODO in config/paper.toml before raising."
    )


def test_bull_put_clears_threshold_13_with_positive_gamma(make_state, make_signals):
    """A vanilla positive-gamma state (no FlashAlpha-only bonuses, no VIX-calm
    bonus, no L1/L2 pattern hits) must clear `score_threshold = 13`.

    Without this guarantee the scanner can't write a paper trade — which is
    exactly what happened all day 2026-05-26 when FlashAlpha was DNS-broken
    and Anthropic was 404'ing.

    Math: base 10 + positive_gamma 4 = 14 ≥ 13.
    """
    state = make_state(
        asof=_et_utc(11, 0),
        signals=make_signals(
            gamma_regime="positive_gamma",
            # Explicitly null-out the levels FlashAlpha provides so we prove the
            # baseline (no GEX-proximity bonuses) still clears the lowered bar.
            gamma_flip=None,
            call_wall=None,
            put_wall=None,
            magnet_strike=None,
            pin_score=None,
        ),
    )
    res = score_state(state, StrategyType.BULL_PUT)
    assert res.regime_ok is True
    assert res.total >= settings.engine.score_threshold, (
        f"BULL_PUT score {res.total} did not clear threshold "
        f"{settings.engine.score_threshold}; breakdown={res.breakdown}"
    )
