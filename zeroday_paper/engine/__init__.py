"""Paper-trading engine.

models.py       — typed dataclasses for the whole pipeline.
pricing.py      — bid-priced entry credit + exit cost.
journal.py      — DuckDB schema + idempotent writes.
state.py        — position state machine.
gex_levels.py   — proximity to key GEX levels.
gex_patterns.py — Layer 1 rule-based 15-pattern classifier.
gex_patterns_llm.py — Layer 2 Anthropic Claude classifier.
strike_select.py — 4-stage strike picker.
score.py        — pure scoring function.
scanner.py      — 2-min live loop.
monitor.py      — exit-trigger loop over open positions.
replay.py       — 365-day historical replay with chunked resume.
"""
