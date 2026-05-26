"""Layer 2 LLM classifier with patched AsyncAnthropic."""

from __future__ import annotations

import asyncio
import json
import sys
import types
from dataclasses import dataclass

import pytest

from zeroday_paper.engine import gex_patterns_llm as llm
from zeroday_paper.engine.gex_patterns_llm import (
    LLMBudget,
    _bucket_confidence,
    _extract_json,
    classify_layer2,
    state_to_summary,
)


# ----------------------------------------------------------------- pure helpers


def test_bucket_confidence_thresholds():
    from zeroday_paper.engine.gex_patterns import PatternConfidence
    assert _bucket_confidence(0.9) == PatternConfidence.HIGH
    assert _bucket_confidence(0.8) == PatternConfidence.HIGH
    assert _bucket_confidence(0.7) == PatternConfidence.MEDIUM
    assert _bucket_confidence(0.65) == PatternConfidence.MEDIUM
    assert _bucket_confidence(0.5) == PatternConfidence.LOW


def test_extract_json_plain():
    text = '{"matches": []}'
    assert _extract_json(text) == {"matches": []}


def test_extract_json_with_markdown_fence():
    text = "```json\n{\"matches\": [{\"id\": \"P01\"}]}\n```"
    out = _extract_json(text)
    assert out["matches"][0]["id"] == "P01"


def test_extract_json_with_only_backticks():
    text = "```\n{\"x\":1}\n```"
    out = _extract_json(text)
    assert out == {"x": 1}


def test_extract_json_with_surrounding_text():
    text = 'Here is the result:\n{"matches": [{"id":"P02"}]}\nEnd.'
    out = _extract_json(text)
    assert out["matches"][0]["id"] == "P02"


def test_extract_json_no_braces_raises():
    with pytest.raises(ValueError):
        _extract_json("just a sentence")


def test_state_to_summary(make_state):
    s = state_to_summary(make_state())
    assert "asof" in s
    assert "spot" in s
    assert s["spot"] == 5800.0


# ----------------------------------------------------------- early returns


@pytest.mark.asyncio
async def test_classify_layer2_disabled_returns_empty(make_state, monkeypatch):
    fake_settings = type("S", (), {"patterns": type("P", (), {
        "layer_2_llm_enabled": False,
        "layer_2_max_calls_per_scan": 3,
        "layer_2_timeout_seconds": 8,
        "layer_2_model": "claude-3-5",
        "layer_2_min_confidence": 0.65,
    })()})()
    monkeypatch.setattr(llm, "settings", fake_settings)
    out = await classify_layer2(make_state())
    assert out == []


@pytest.mark.asyncio
async def test_classify_layer2_anthropic_import_missing(make_state, monkeypatch):
    fake_settings = type("S", (), {"patterns": type("P", (), {
        "layer_2_llm_enabled": True,
        "layer_2_max_calls_per_scan": 3,
        "layer_2_timeout_seconds": 8,
        "layer_2_model": "claude-3-5",
        "layer_2_min_confidence": 0.65,
    })()})()
    monkeypatch.setattr(llm, "settings", fake_settings)
    # Hide anthropic from the import system
    monkeypatch.setitem(sys.modules, "anthropic", None)
    out = await classify_layer2(make_state())
    assert out == []


@pytest.mark.asyncio
async def test_classify_layer2_no_api_key_returns_empty(make_state, monkeypatch):
    fake_settings = type("S", (), {"patterns": type("P", (), {
        "layer_2_llm_enabled": True,
        "layer_2_max_calls_per_scan": 3,
        "layer_2_timeout_seconds": 8,
        "layer_2_model": "claude-3-5",
        "layer_2_min_confidence": 0.65,
    })()})()
    monkeypatch.setattr(llm, "settings", fake_settings)
    monkeypatch.setattr(llm, "anthropic_api_key", lambda: (_ for _ in ()).throw(RuntimeError("no key")))
    out = await classify_layer2(make_state())
    assert out == []


# -------------------------------------------------------- Anthropic mocked path


def _install_anthropic_mock(monkeypatch, response_text: str):
    """Install a fake `anthropic` module with AsyncAnthropic returning canned text."""
    @dataclass
    class _Block:
        type: str
        text: str

    @dataclass
    class _Resp:
        content: list

    class _Messages:
        def __init__(self, text):
            self.text = text

        async def create(self, **_kwargs):
            return _Resp(content=[_Block(type="text", text=self.text)])

    class _AsyncAnthropic:
        def __init__(self, api_key=None):
            self.messages = _Messages(response_text)

    fake_mod = types.ModuleType("anthropic")
    fake_mod.AsyncAnthropic = _AsyncAnthropic
    monkeypatch.setitem(sys.modules, "anthropic", fake_mod)


def _enable_llm_settings(monkeypatch):
    fake_settings = type("S", (), {"patterns": type("P", (), {
        "layer_2_llm_enabled": True,
        "layer_2_max_calls_per_scan": 3,
        "layer_2_timeout_seconds": 8,
        "layer_2_model": "claude-3-5",
        "layer_2_min_confidence": 0.65,
    })()})()
    monkeypatch.setattr(llm, "settings", fake_settings)
    monkeypatch.setattr(llm, "anthropic_api_key", lambda: "test-key")


@pytest.mark.asyncio
async def test_classify_layer2_returns_parsed_matches(make_state, monkeypatch):
    _enable_llm_settings(monkeypatch)
    canned = json.dumps({
        "matches": [
            {"id": "P01", "name": "Negative Gamma", "direction": "BEARISH", "confidence": 0.85,
             "rationale": "test"},
            {"id": "P02", "name": "Pin", "direction": "NEUTRAL", "confidence": 0.70,
             "rationale": "med"},
        ]
    })
    _install_anthropic_mock(monkeypatch, canned)

    out = await classify_layer2(make_state())
    assert len(out) == 2
    assert out[0].pattern_id == "P01"
    assert out[0].score_bonus == 1  # 0.85 >= 0.75
    assert out[1].pattern_id == "P02"
    assert out[1].score_bonus == 0  # 0.70 < 0.75


@pytest.mark.asyncio
async def test_classify_layer2_filters_below_min_confidence(make_state, monkeypatch):
    _enable_llm_settings(monkeypatch)
    canned = json.dumps({
        "matches": [
            {"id": "P01", "name": "x", "direction": "BEARISH", "confidence": 0.4,
             "rationale": "low"},
            {"id": "P02", "name": "y", "direction": "NEUTRAL", "confidence": 0.9,
             "rationale": "high"},
        ]
    })
    _install_anthropic_mock(monkeypatch, canned)
    out = await classify_layer2(make_state())
    ids = {m.pattern_id for m in out}
    assert ids == {"P02"}


@pytest.mark.asyncio
async def test_classify_layer2_with_markdown_fence(make_state, monkeypatch):
    _enable_llm_settings(monkeypatch)
    canned = "```json\n" + json.dumps({
        "matches": [{"id": "P01", "name": "x", "direction": "BEARISH", "confidence": 0.9,
                     "rationale": "r"}]
    }) + "\n```"
    _install_anthropic_mock(monkeypatch, canned)
    out = await classify_layer2(make_state())
    assert len(out) == 1


@pytest.mark.asyncio
async def test_classify_layer2_missing_id_skipped(make_state, monkeypatch):
    _enable_llm_settings(monkeypatch)
    canned = json.dumps({
        "matches": [
            {"name": "x", "direction": "BEARISH", "confidence": 0.9},  # missing id
            {"id": "P02", "name": "y", "direction": "NEUTRAL", "confidence": 0.9, "rationale": "r"},
        ]
    })
    _install_anthropic_mock(monkeypatch, canned)
    out = await classify_layer2(make_state())
    assert len(out) == 1
    assert out[0].pattern_id == "P02"


@pytest.mark.asyncio
async def test_classify_layer2_unparseable_response(make_state, monkeypatch):
    _enable_llm_settings(monkeypatch)
    _install_anthropic_mock(monkeypatch, "garbage no json here")
    out = await classify_layer2(make_state())
    assert out == []


@pytest.mark.asyncio
async def test_classify_layer2_timeout(make_state, monkeypatch):
    _enable_llm_settings(monkeypatch)

    @dataclass
    class _Block:
        type: str = "text"
        text: str = ""

    class _Messages:
        async def create(self, **_kwargs):
            raise asyncio.TimeoutError()

    class _AsyncAnthropic:
        def __init__(self, api_key=None):
            self.messages = _Messages()

    fake_mod = types.ModuleType("anthropic")
    fake_mod.AsyncAnthropic = _AsyncAnthropic
    monkeypatch.setitem(sys.modules, "anthropic", fake_mod)
    out = await classify_layer2(make_state())
    assert out == []


@pytest.mark.asyncio
async def test_classify_layer2_api_error(make_state, monkeypatch):
    _enable_llm_settings(monkeypatch)

    class _Messages:
        async def create(self, **_kwargs):
            raise RuntimeError("API outage")

    class _AsyncAnthropic:
        def __init__(self, api_key=None):
            self.messages = _Messages()

    fake_mod = types.ModuleType("anthropic")
    fake_mod.AsyncAnthropic = _AsyncAnthropic
    monkeypatch.setitem(sys.modules, "anthropic", fake_mod)
    out = await classify_layer2(make_state())
    assert out == []


@pytest.mark.asyncio
async def test_classify_layer2_with_explicit_budget(make_state, monkeypatch):
    fake_settings = type("S", (), {"patterns": type("P", (), {
        "layer_2_llm_enabled": True,
        "layer_2_max_calls_per_scan": 3,
        "layer_2_timeout_seconds": 8,
        "layer_2_model": "claude-3-5",
        "layer_2_min_confidence": 0.65,
    })()})()
    monkeypatch.setattr(llm, "settings", fake_settings)
    monkeypatch.setattr(llm, "anthropic_api_key", lambda: "test-key")
    canned = json.dumps({"matches": []})
    _install_anthropic_mock(monkeypatch, canned)
    budget = LLMBudget(max_calls_per_scan=1, timeout_s=1.0, model="m", min_confidence=0.9)
    out = await classify_layer2(make_state(), budget=budget)
    assert out == []
