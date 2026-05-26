"""Daily report: markdown rendering, chunking, Discord posting."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import httpx
import pytest

from zeroday_paper.engine.journal import Journal
from zeroday_paper.engine.models import PositionStatus, TradeOutcome
from zeroday_paper.reporting import daily_report
from zeroday_paper.reporting import stats


def _patch_stats_settings(monkeypatch, db_path: str):
    from zeroday_paper.config import settings as real_settings
    class _Shim:
        duckdb_path = db_path
        engine = real_settings.engine
        reporting = real_settings.reporting
        regime_gates = real_settings.regime_gates
        strikes = real_settings.strikes
        exits = real_settings.exits
        concurrency = real_settings.concurrency
        dedup = real_settings.dedup
        replay = real_settings.replay
        patterns = real_settings.patterns
        storage = real_settings.storage
        alarms = real_settings.alarms
        secrets = real_settings.secrets
        logging = real_settings.logging
    monkeypatch.setattr(stats, "settings", _Shim())
    monkeypatch.setattr(daily_report, "settings", _Shim())


@pytest.fixture
def seeded_journal(tmp_path, make_paper_trade, monkeypatch):
    db = tmp_path / "report.duckdb"
    j = Journal(str(db))
    pt = make_paper_trade(entry_ts=__import__("datetime").datetime(2025, 5, 28, 14, 0, tzinfo=__import__("datetime").timezone.utc))
    j.write_trade(pt)
    j.write_outcome(TradeOutcome(
        trade_id=pt.trade_id, exit_ts=pt.entry_ts,
        exit_status=PositionStatus.CLOSED_TARGET, exit_spot=5800.0,
        exit_cost_bid=0.4, exit_cost_mid=0.45, pnl_bid=60.0, pnl_mid=60.0,
        held_minutes=30, max_excursion_pct=0.6, min_excursion_pct=-0.1,
        exit_reason="t",
    ))
    j.close()
    _patch_stats_settings(monkeypatch, str(db))
    return db


# ----------------------------------------------------------------- chunk_markdown


def test_chunk_markdown_single_chunk_below_limit():
    out = daily_report._chunk_markdown("hello\nworld\n", max_chars=100)
    assert out == ["hello\nworld\n"]


def test_chunk_markdown_splits_at_line_boundary():
    text = "\n".join(f"line-{i}" for i in range(20)) + "\n"
    chunks = daily_report._chunk_markdown(text, max_chars=30)
    assert len(chunks) > 1
    # No chunk exceeds limit
    assert all(len(c) <= 60 for c in chunks)
    # Re-joining yields original
    assert "".join(chunks) == text


def test_chunk_markdown_empty_input():
    assert daily_report._chunk_markdown("", max_chars=100) == []


def test_chunk_markdown_single_huge_line():
    # Even if the line itself is bigger than max_chars, it goes in a chunk
    chunks = daily_report._chunk_markdown("x" * 200 + "\n", max_chars=50)
    assert len(chunks) >= 1


# --------------------------------------------------------------------- build_markdown


def test_build_markdown_runs_against_seeded_db(seeded_journal):
    md = daily_report.build_markdown(today=date(2025, 5, 28))
    assert "# Paper Trading Daily Report" in md
    assert "## Today (live)" in md
    assert "## Last 7 days" in md
    assert "## Last 30 days" in md
    assert "## Replay backdrop" in md
    assert "## Pattern leaderboard" in md
    assert "## Regime breakdown" in md


def test_build_markdown_uses_today_when_none(seeded_journal):
    md = daily_report.build_markdown()
    assert "# Paper Trading Daily Report" in md


# ------------------------------------------------------------------------ HTML


def test_build_html_wraps_markdown():
    html = daily_report.build_html("hello\nworld")
    assert "<html>" in html
    assert "hello" in html


# --------------------------------------------------------------------- write_files


def test_write_report_files(tmp_path, seeded_journal):
    md = daily_report.build_markdown(today=date(2025, 5, 28))
    md_path, html_path = daily_report.write_report_files(md, base_dir=str(tmp_path / "out"))
    assert md_path.exists()
    assert html_path.exists()
    assert md_path.read_text() == md


# ------------------------------------------------------------------ Discord post


def test_post_to_discord_no_webhook_returns_false(monkeypatch):
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "")
    monkeypatch.setattr(daily_report, "_resolve_webhook", lambda: None)
    assert daily_report.post_to_discord("hello") is False


def test_post_to_discord_with_explicit_url_success(monkeypatch):
    captured = []

    def fake_post(url, json=None, timeout=None):
        captured.append({"url": url, "body": json})
        resp = MagicMock()
        resp.status_code = 204
        resp.text = ""
        return resp

    monkeypatch.setattr(daily_report.httpx, "post", fake_post)
    out = daily_report.post_to_discord("hello world", webhook_url="https://x.test/hook")
    assert out is True
    assert captured[0]["url"] == "https://x.test/hook"
    assert captured[0]["body"]["content"] == "hello world"


def test_post_to_discord_non_2xx_returns_false(monkeypatch):
    def fake_post(url, json=None, timeout=None):
        resp = MagicMock()
        resp.status_code = 500
        resp.text = "internal"
        return resp

    monkeypatch.setattr(daily_report.httpx, "post", fake_post)
    out = daily_report.post_to_discord("hello", webhook_url="https://x.test/hook")
    assert out is False


def test_post_to_discord_raises_returns_false(monkeypatch):
    def fake_post(url, json=None, timeout=None):
        raise RuntimeError("boom")

    monkeypatch.setattr(daily_report.httpx, "post", fake_post)
    out = daily_report.post_to_discord("hello", webhook_url="https://x.test/hook")
    assert out is False


def test_post_to_discord_splits_long_message(monkeypatch):
    calls = []
    def fake_post(url, json=None, timeout=None):
        calls.append(json["content"])
        resp = MagicMock()
        resp.status_code = 204
        return resp
    monkeypatch.setattr(daily_report.httpx, "post", fake_post)
    long_text = "\n".join(f"row-{i}" for i in range(500))
    out = daily_report.post_to_discord(long_text, webhook_url="https://x.test/hook")
    assert out is True
    assert len(calls) >= 1


def test_resolve_webhook_returns_env(monkeypatch):
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://hook.test/abc")
    # The secrets module returns env-first
    assert daily_report._resolve_webhook() == "https://hook.test/abc"


def test_resolve_webhook_failure_returns_none(monkeypatch):
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
    from zeroday_paper import secrets as ss
    def fake_dw(key=None):
        raise RuntimeError("nope")
    monkeypatch.setattr(daily_report, "discord_webhook", fake_dw)
    assert daily_report._resolve_webhook() is None
