"""Layer 1 rule-based GEX patterns (15 patterns).

Each pattern must produce matched=True AND matched=False outcomes given
tailored conditions dicts.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from zeroday_paper.engine.gex_patterns import (
    GEXPatternLibrary,
    PatternConfidence,
    PatternMatch,
    classify_layer1,
    conditions_from_state,
)


def _base_conditions(**overrides) -> dict:
    base = {
        "gex_value": 1.0,
        "prev_gex_value": 1.0,
        "gex_sign": "positive",
        "stability_pct": 50.0,
        "hiro_direction": "neutral",
        "prev_hiro_direction": "neutral",
        "spx_level": 5800.0,
        "zero_gamma_level": 5790.0,
        "vol_trigger_level": 5770.0,
        "call_wall_level": 5850.0,
        "put_wall_level": 5750.0,
        "vix_change_pct": 0.0,
        "vix_direction": "flat",
        "iv_rank": 50,
        "volume_to_oi_ratio": 1.0,
        "bollinger_band_width": 10.0,
        "atr_14": 20.0,
        "hours_to_close": 4.0,
        "max_pain_strike": 5800.0,
    }
    base.update(overrides)
    return base


# ----------------------------------------------- match_all is total (15 every time)


def test_match_all_returns_exactly_15():
    lib = GEXPatternLibrary()
    assert len(lib.match_all(_base_conditions())) == 15


def test_match_all_with_extreme_values_still_15():
    lib = GEXPatternLibrary()
    assert len(lib.match_all(_base_conditions(gex_value=-10.0, stability_pct=5.0))) == 15


# ---------------------------------------------------------------- per-pattern table


PATTERN_CASES = [
    # (id, matched-conds, unmatched-conds)
    ("P01",
     {"gex_value": -2.0, "stability_pct": 10.0},
     {"gex_value": 1.0, "stability_pct": 50.0}),
    ("P02",
     {"gex_value": 1.0, "stability_pct": 50.0},
     {"gex_value": -0.5, "stability_pct": 5.0}),
    ("P03",
     {"spx_level": 5800.0, "zero_gamma_level": 5800.0},
     {"spx_level": 5800.0, "zero_gamma_level": 5700.0}),
    ("P04",
     {"gex_sign": "negative", "hiro_direction": "bullish"},
     {"gex_sign": "negative", "hiro_direction": "neutral"}),
    ("P05",
     {"stability_pct": 5.0, "vix_change_pct": 10.0},
     {"stability_pct": 50.0, "vix_change_pct": 0.0}),
    ("P06",
     {"spx_level": 5700.0, "vol_trigger_level": 5800.0, "gex_sign": "negative"},
     {"spx_level": 5900.0, "vol_trigger_level": 5800.0, "gex_sign": "positive"}),
    ("P07",
     {"spx_level": 5840.0, "call_wall_level": 5850.0, "hiro_direction": "bearish"},
     {"spx_level": 5800.0, "call_wall_level": 5900.0, "hiro_direction": "neutral"}),
    ("P08",
     {"spx_level": 5760.0, "put_wall_level": 5750.0, "hiro_direction": "bullish"},
     {"spx_level": 5800.0, "put_wall_level": 5700.0, "hiro_direction": "neutral"}),
    ("P09",
     {"hours_to_close": 1.0, "stability_pct": 20.0},
     {"hours_to_close": 5.0, "stability_pct": 50.0}),
    ("P10",
     {"iv_rank": 20, "vix_direction": "falling"},
     {"iv_rank": 60, "vix_direction": "flat"}),
    ("P11",
     {"gex_value": 3.0, "volume_to_oi_ratio": 3.0},
     {"gex_value": 0.5, "volume_to_oi_ratio": 1.0}),
    ("P12",
     {"gex_value": -2.0, "prev_gex_value": 0.5},
     {"gex_value": 1.0, "prev_gex_value": 1.0}),
    ("P13",
     {"hiro_direction": "bullish", "prev_hiro_direction": "bearish"},
     {"hiro_direction": "neutral", "prev_hiro_direction": "neutral"}),
    ("P14",
     {"bollinger_band_width": 5.0, "atr_14": 40.0, "stability_pct": 5.0},
     {"bollinger_band_width": 50.0, "atr_14": 10.0, "stability_pct": 50.0}),
    ("P15",
     {"spx_level": 5810.0, "max_pain_strike": 5800.0, "hours_to_close": 4.0},
     {"spx_level": 5800.0, "max_pain_strike": 5900.0, "hours_to_close": 1.0}),
]


@pytest.mark.parametrize("pid,matched_overrides,unmatched_overrides", PATTERN_CASES)
def test_pattern_match_and_unmatch(pid, matched_overrides, unmatched_overrides):
    lib = GEXPatternLibrary()
    matches_on = {m.pattern_id: m for m in lib.match_all(_base_conditions(**matched_overrides))}
    matches_off = {m.pattern_id: m for m in lib.match_all(_base_conditions(**unmatched_overrides))}
    assert matches_on[pid].matched is True, f"{pid} should match with {matched_overrides}"
    assert matches_off[pid].matched is False, f"{pid} should NOT match with {unmatched_overrides}"


# ------------------------------------------------------------- classify_layer1 filter


def test_classify_layer1_filters_to_matched_only(make_state, make_signals):
    # Construct a state with clear positive_gamma and high stability → P02 will match
    state = make_state(signals=make_signals(
        gamma_regime="positive_gamma", total_gex=1.0, pin_score=80.0,
        gamma_flip=5790.0, magnet_strike=5800.0, max_pain=5800.0,
    ))
    matches = classify_layer1(state)
    assert all(m.matched for m in matches)
    assert len(matches) <= 15
    ids = {m.pattern_id for m in matches}
    assert "P02" in ids or "P15" in ids  # at least one expected to match


def test_classify_layer1_with_prior_state(make_state, make_signals):
    s1 = make_state(signals=make_signals(total_gex=1.0))
    s2 = make_state(signals=make_signals(total_gex=-2.0, gamma_regime="negative_gamma"))
    matches = classify_layer1(s2, s1)
    ids = {m.pattern_id for m in matches}
    # GEX dropped by 3.0 → P12 (Gamma Cliff) should fire
    assert "P12" in ids


# ----------------------------------------------------------- conditions_from_state


def test_conditions_from_state_has_all_keys(make_state):
    state = make_state()
    c = conditions_from_state(state)
    for k in [
        "gex_value", "prev_gex_value", "gex_sign", "stability_pct",
        "hiro_direction", "prev_hiro_direction", "spx_level",
        "zero_gamma_level", "vol_trigger_level", "call_wall_level",
        "put_wall_level", "vix_change_pct", "vix_direction", "iv_rank",
        "volume_to_oi_ratio", "bollinger_band_width", "atr_14",
        "hours_to_close", "max_pain_strike",
    ]:
        assert k in c


def test_conditions_from_state_handles_missing_gex(make_state, make_signals):
    state = make_state(signals=make_signals(total_gex=None))
    c = conditions_from_state(state)
    assert c["gex_value"] == 0.0
    assert c["gex_sign"] == "positive"  # 0 >= 0


def test_conditions_from_state_negative_gex_sign(make_state, make_signals):
    state = make_state(signals=make_signals(total_gex=-1.0))
    c = conditions_from_state(state)
    assert c["gex_sign"] == "negative"


def test_conditions_from_state_uses_prior_gex(make_state, make_signals):
    s1 = make_state(signals=make_signals(total_gex=5.0))
    s2 = make_state(signals=make_signals(total_gex=1.0))
    c = conditions_from_state(s2, s1)
    assert c["prev_gex_value"] == 5.0


# --------------------------------------------------------------- best_match ordering


def test_best_match_prefers_high_confidence():
    lib = GEXPatternLibrary()
    # P01 matches as HIGH; P05 also matches as HIGH; both BEARISH
    conds = _base_conditions(
        gex_value=-2.0, gex_sign="negative",
        stability_pct=5.0, vix_change_pct=10.0,
    )
    best = lib.best_match(conds)
    assert best is not None
    assert best.confidence == PatternConfidence.HIGH


def test_best_match_returns_none_when_no_match():
    lib = GEXPatternLibrary()
    conds = _base_conditions()  # base does not match much; let's clear specific
    # ensure P03 does not match: spx far from zero gamma
    conds = _base_conditions(zero_gamma_level=5000.0)
    best = lib.best_match(conds)
    # base may match P02 actually. Let's use unmatched-only
    conds_no_match = _base_conditions(
        gex_value=0.0, prev_gex_value=0.0,
        gex_sign="positive", stability_pct=25.0,   # avoid P02 (needs >0.5 GEX and >30 stab)
        spx_level=5800.0, zero_gamma_level=5500.0,  # avoid P03
        vol_trigger_level=5700.0, call_wall_level=5900.0, put_wall_level=5600.0,
        hiro_direction="neutral", prev_hiro_direction="neutral",
        iv_rank=80, vix_direction="rising", vix_change_pct=0.0,
        volume_to_oi_ratio=0.5, bollinger_band_width=30, atr_14=10,
        hours_to_close=5.0, max_pain_strike=5400.0,
    )
    assert lib.best_match(conds_no_match) is None
