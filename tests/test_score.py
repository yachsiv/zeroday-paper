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
    # session_start=9:30, cutoff=14:30 → 14:45 should block.
    # The reason label "after_entry_cutoff_13_30" is a stale code constant in
    # score.py (kept to avoid touching scoring code); it fires for any
    # post-cutoff time regardless of the configured value.
    state = make_state(asof=_et_utc(14, 45))
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


def test_bull_put_clears_threshold_13_with_live_walls_far_from_spot(
    make_state, make_signals, make_vols,
):
    """Reproduces the 2026-05-27 live regime: walls are >100pt away from spot
    so the old hard <25pt wall bonus never fires; VIX1D=17 so the old <14
    bonus never fires either. Score must still clear threshold via the new
    graded proximity bonus + pin_score + base/regime baseline.

    Realistic numbers: spot=7519, put_wall=7395 (124pt below), call_wall=8200
    (681pt above), pin_score=85, vix_1d=17 — i.e. exactly today.
    """
    state = make_state(
        spot=7519.0,
        asof=_et_utc(11, 0),
        signals=make_signals(
            spot=7519.0,
            gamma_regime="positive_gamma",
            gamma_flip=7400.0,    # above-flip bonus +1
            call_wall=8200.0,     # 681pt above → 0pt bonus from call wall
            put_wall=7395.0,      # 124pt below → +0 (>100pt) ... regression
            magnet_strike=7520.0,
            pin_score=85.0,       # ≥70 → +2
            total_gex=3.5,
        ),
        vols=make_vols(vix_1d=17.0, cboe_skew=137.0),  # vix in [14,18] → +1
    )
    res = score_state(state, StrategyType.BULL_PUT)
    assert res.regime_ok is True
    # base 10 + regime 4 + above_flip 1 + pin 2 + vix1d_lt_18 1 = 18 ≥ 13
    assert res.total >= settings.engine.score_threshold, (
        f"realistic BULL_PUT score {res.total} did not clear threshold "
        f"{settings.engine.score_threshold}; breakdown={res.breakdown}; notes={res.notes}"
    )
    # Score should not double-count walls when they are >100pt away.
    assert all("near_put_wall" not in n for n in res.notes)


def test_proximity_bonus_grades_by_distance(make_state, make_signals):
    """Within 25pt → +3; within 50 → +2; within 100 → +1; beyond → +0."""
    # 10pt to put_wall → +3 (within 25)
    state_close = make_state(spot=5760.0, signals=make_signals(spot=5760.0, put_wall=5750.0))
    res_close = score_state(state_close, StrategyType.BULL_PUT)
    assert any("near_put_wall_for_bull_put" in n and ":+3" in n for n in res_close.notes), (
        f"expected +3 bonus at 10pt distance, got notes={res_close.notes}"
    )

    # 40pt → +2 (within 50)
    state_mid = make_state(spot=5790.0, signals=make_signals(spot=5790.0, put_wall=5750.0))
    res_mid = score_state(state_mid, StrategyType.BULL_PUT)
    assert any("near_put_wall_for_bull_put" in n and ":+2" in n for n in res_mid.notes), (
        f"expected +2 bonus at 40pt distance, got notes={res_mid.notes}"
    )

    # 80pt → +1 (within 100)
    state_far = make_state(spot=5830.0, signals=make_signals(spot=5830.0, put_wall=5750.0))
    res_far = score_state(state_far, StrategyType.BULL_PUT)
    assert any("near_put_wall_for_bull_put" in n and ":+1" in n for n in res_far.notes), (
        f"expected +1 bonus at 80pt distance, got notes={res_far.notes}"
    )

    # 150pt → +0 (beyond 100)
    state_beyond = make_state(spot=5900.0, signals=make_signals(spot=5900.0, put_wall=5750.0))
    res_beyond = score_state(state_beyond, StrategyType.BULL_PUT)
    assert all("near_put_wall_for_bull_put" not in n for n in res_beyond.notes)


def test_vix_bonus_grades_below_18_and_below_14(make_state, make_vols):
    # vix=15 should yield +1 (lt_18 band)
    state = make_state(asof=_et_utc(11, 0), vols=make_vols(vix_1d=15.0))
    res = score_state(state, StrategyType.BULL_PUT)
    assert any("vix1d_calm_15.0_lt_18" in n for n in res.notes), res.notes
    assert ":+1" in next(n for n in res.notes if "vix1d_calm" in n)

    # vix=10 should yield +2 (lt_14 band)
    state_calm = make_state(asof=_et_utc(11, 0), vols=make_vols(vix_1d=10.0))
    res_calm = score_state(state_calm, StrategyType.BULL_PUT)
    assert any("vix1d_calm_10.0_lt_14" in n for n in res_calm.notes), res_calm.notes
    assert ":+2" in next(n for n in res_calm.notes if "vix1d_calm" in n)

    # vix=22 should yield no bonus (and not block regime since gate=25)
    state_high = make_state(asof=_et_utc(11, 0), vols=make_vols(vix_1d=22.0))
    res_high = score_state(state_high, StrategyType.BULL_PUT)
    assert all("vix1d_calm" not in n for n in res_high.notes)
