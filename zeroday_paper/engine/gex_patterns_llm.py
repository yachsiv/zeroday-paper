"""Layer 2: Anthropic Claude pattern classifier.

Async + budgeted. Only called when Layer 1 has uncertainty (no high-confidence
match) and we are inside cost cap for the scan.

Output schema matches Layer 1's PatternMatch for clean unioning.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass

import structlog

from zeroday_paper.config import settings
from zeroday_paper.engine.gex_patterns import PatternConfidence, PatternMatch
from zeroday_paper.engine.models import MarketState
from zeroday_paper.secrets import anthropic_api_key

logger = structlog.get_logger(__name__)


SYSTEM_PROMPT = """You are a SPX 0DTE options market structure classifier.

You are given a real-time snapshot of SPX with GEX (gamma exposure) levels and
volatility. Decide which of these 15 patterns are matched. Output STRICT JSON only.

PATTERNS:
P01 Negative Gamma Squeeze     — high negative GEX + low stability
P02 Positive Gamma Pin         — high positive GEX + low realized vol
P03 Zero Gamma Flip Zone       — spot near gamma flip
P04 HIRO-GEX Divergence        — flow direction conflicts with GEX sign
P05 Stability Collapse         — fast vol expansion in negative gamma
P06 Vol Trigger Break          — spot below vol trigger in negative GEX
P07 Call Wall Rejection        — spot pressing into call wall, dealer sells
P08 Put Wall Support Bounce    — spot pressing into put wall, dealer buys
P09 Charm Acceleration         — late session pin-then-drift
P10 Vanna Tailwind             — falling vol, dealer hedging supports rally
P11 Dealer Long Hedge Unwind   — extreme positive GEX with volume spike
P12 Gamma Cliff                — rapid GEX collapse
P13 HIRO Reversal              — flow direction flipped
P14 Squeeze Compression        — ultra-low realized vol pre-breakout
P15 Expiry Magnet              — max-pain gravity with time remaining

For each MATCHED pattern, return:
  - id: "P01"..."P15"
  - name: human-readable
  - direction: "BULLISH" | "BEARISH" | "NEUTRAL"
  - confidence: 0.0..1.0 (be conservative)
  - rationale: <=20 words

Return JSON of shape:
  {"matches": [ {id, name, direction, confidence, rationale}, ... ] }

Hard rules:
  - If <2 patterns clearly match, return empty matches.
  - If confidence < 0.55 for a pattern, omit it.
  - Do NOT invent patterns outside P01..P15.
"""


@dataclass(frozen=True)
class LLMBudget:
    max_calls_per_scan: int
    timeout_s: float
    model: str
    min_confidence: float


def state_to_summary(state: MarketState) -> dict:
    s = state.signals
    return {
        "asof": state.asof.isoformat(),
        "spot": state.spot,
        "gamma_regime": s.gamma_regime,
        "total_gex_billions": s.total_gex,
        "gamma_flip": s.gamma_flip,
        "call_wall": s.call_wall,
        "put_wall": s.put_wall,
        "magnet_strike": s.magnet_strike,
        "max_pain": s.max_pain,
        "pin_score": s.pin_score,
        "vix_1d": state.vols.vix_1d,
        "cboe_skew": state.vols.cboe_skew,
        "source": s.source,
    }


async def classify_layer2(
    state: MarketState,
    *,
    budget: LLMBudget | None = None,
) -> list[PatternMatch]:
    if not settings.patterns.layer_2_llm_enabled:
        return []

    budget = budget or LLMBudget(
        max_calls_per_scan=settings.patterns.layer_2_max_calls_per_scan,
        timeout_s=settings.patterns.layer_2_timeout_seconds,
        model=settings.patterns.layer_2_model,
        min_confidence=settings.patterns.layer_2_min_confidence,
    )

    try:
        from anthropic import AsyncAnthropic
    except ImportError:
        logger.warning("anthropic.sdk.missing — install `anthropic`")
        return []

    try:
        api_key = anthropic_api_key()
    except Exception as exc:
        logger.warning("layer2.no_key", error=str(exc))
        return []

    client = AsyncAnthropic(api_key=api_key)
    user_msg = json.dumps(state_to_summary(state), indent=2)

    try:
        resp = await asyncio.wait_for(
            client.messages.create(
                model=budget.model,
                max_tokens=512,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
            ),
            timeout=budget.timeout_s,
        )
    except asyncio.TimeoutError:
        logger.warning("layer2.timeout")
        return []
    except Exception as exc:
        logger.warning("layer2.api_error", error=str(exc))
        return []

    try:
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        parsed = _extract_json(text)
        matches_raw = parsed.get("matches", [])
    except Exception as exc:
        logger.warning("layer2.parse_error", error=str(exc))
        return []

    out: list[PatternMatch] = []
    for m in matches_raw:
        try:
            conf = float(m.get("confidence", 0.0))
            if conf < budget.min_confidence:
                continue
            out.append(PatternMatch(
                pattern_id=str(m["id"]),
                name=str(m.get("name", "")),
                matched=True,
                confidence=_bucket_confidence(conf),
                direction=str(m.get("direction", "NEUTRAL")).upper(),
                # Lowered 2026-05-27 from 0.75 → 0.65 to actually award the bonus
                # on real Anthropic responses (which rarely emit ≥0.75 for SPX
                # 0DTE pattern matches given the inherent ambiguity).
                score_bonus=1 if conf >= 0.65 else 0,
                description=str(m.get("rationale", "")),
            ))
        except (KeyError, ValueError, TypeError):
            continue
    return out


def _bucket_confidence(c: float) -> PatternConfidence:
    if c >= 0.8:
        return PatternConfidence.HIGH
    if c >= 0.65:
        return PatternConfidence.MEDIUM
    return PatternConfidence.LOW


def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no json object in response")
    return json.loads(text[start:end + 1])
