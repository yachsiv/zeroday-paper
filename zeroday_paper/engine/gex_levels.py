"""GEX level proximity helpers.

Pure functions over `MarketState`. The scorer uses them to award/deduct points
based on where spot sits relative to gamma flip, call wall, put wall, magnet.
"""

from __future__ import annotations

from dataclasses import dataclass

from zeroday_paper.engine.models import MarketState


@dataclass(frozen=True)
class LevelProximity:
    """Distance (in points) from spot to each key level. None if level missing."""

    to_gamma_flip: float | None
    to_call_wall: float | None
    to_put_wall: float | None
    to_magnet: float | None
    above_flip: bool | None
    below_call_wall: bool | None
    above_put_wall: bool | None


def compute_proximity(state: MarketState) -> LevelProximity:
    spot = state.spot
    s = state.signals

    def dist(level: float | None) -> float | None:
        return None if level is None else round(spot - level, 2)

    above_flip = None if s.gamma_flip is None else spot >= s.gamma_flip
    below_cw = None if s.call_wall is None else spot <= s.call_wall
    above_pw = None if s.put_wall is None else spot >= s.put_wall

    return LevelProximity(
        to_gamma_flip=dist(s.gamma_flip),
        to_call_wall=dist(s.call_wall),
        to_put_wall=dist(s.put_wall),
        to_magnet=dist(s.magnet_strike),
        above_flip=above_flip,
        below_call_wall=below_cw,
        above_put_wall=above_pw,
    )


def compute_rr25(state: MarketState) -> float | None:
    """25-delta risk reversal: put_iv(25Δ) - call_iv(25Δ).

    Positive value = put skew (downside fear). Rapid increase = early dealer
    de-risking signal. Used as a thesis-invalidation tripwire.
    """
    target = 0.25
    put_iv = _iv_at_delta(state.chain.puts, -target, lambda q: q.delta)
    call_iv = _iv_at_delta(state.chain.calls, target, lambda q: q.delta)
    if put_iv is None or call_iv is None:
        return None
    return round((put_iv - call_iv) * 100.0, 2)   # in IV points


def _iv_at_delta(quotes, target, getter):
    """Linear interpolate IV at the requested delta."""
    pairs = [(getter(q), q.iv) for q in quotes if getter(q) is not None and q.iv is not None]
    if len(pairs) < 2:
        return None
    pairs.sort(key=lambda p: abs(p[0] - target))
    return pairs[0][1]
