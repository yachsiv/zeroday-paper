"""Daily report: Markdown + HTML + Discord post.

No noisy per-trade alerts. One report per day, posted to a single channel.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import httpx
import structlog

from zeroday_paper.config import settings
from zeroday_paper.reporting.stats import (
    PatternRow,
    WindowStats,
    pattern_breakdown,
    regime_breakdown,
    today_window_stats,
    window_stats,
)
from zeroday_paper.secrets import discord_webhook

logger = structlog.get_logger(__name__)


def build_markdown(*, today: date | None = None) -> str:
    today = today or datetime.now(UTC).date()
    week_start = today - timedelta(days=6)
    month_start = today - timedelta(days=29)

    live_today = today_window_stats(today)
    week_total = window_stats(start=week_start, end=today, label="7d")
    month_total = window_stats(start=month_start, end=today, label="30d")
    replay_total = window_stats(
        start=today - timedelta(days=365), end=today - timedelta(days=1),
        source="replay", label="replay-365d",
    )
    pat_rows = pattern_breakdown(start=month_start, end=today)
    regimes = regime_breakdown(start=month_start, end=today)

    lines: list[str] = []
    lines.append(f"# Paper Trading Daily Report — {today.isoformat()}")
    lines.append("")
    lines.append("_Score threshold: {}, poll cadence: {}s, source mix: live + replay._".format(
        settings.engine.score_threshold, settings.engine.poll_interval_seconds,
    ))
    lines.append("")
    lines.append("## Today (live)")
    lines.append(_format_window(live_today))
    lines.append("")
    lines.append("## Last 7 days (all sources)")
    lines.append(_format_window(week_total))
    lines.append("")
    lines.append("## Last 30 days (all sources)")
    lines.append(_format_window(month_total))
    lines.append("")
    lines.append("## Replay backdrop (365 days)")
    lines.append(_format_window(replay_total))
    lines.append("")
    lines.append("## Pattern leaderboard (30d)")
    lines.append(_format_patterns(pat_rows))
    lines.append("")
    lines.append("## Regime breakdown (30d)")
    for regime, m in regimes.items():
        lines.append(f"- **{regime}** — trades: {m['trades']}, win rate: {m['win_rate']*100:.1f}%, total P&L (bid): ${m['total_pnl']:,.2f}")

    return "\n".join(lines)


def _format_window(w: WindowStats) -> str:
    return (
        f"- Trades: **{w.trades}** (closed: {w.closed})\n"
        f"- Wins / Losses: {w.wins} / {w.losses}\n"
        f"- Win rate: **{w.win_rate*100:.1f}%**\n"
        f"- Total P&L (bid): **${w.total_pnl_bid:,.2f}**\n"
        f"- Avg / Median: ${w.avg_pnl_bid:,.2f} / ${w.median_pnl_bid:,.2f}\n"
        f"- Expectancy / trade: **${w.expectancy:,.2f}**\n"
        f"- Best / Worst: ${w.best_trade:,.2f} / ${w.worst_trade:,.2f}"
    )


def _format_patterns(rows: list[PatternRow]) -> str:
    if not rows:
        return "_No patterns logged in window._"
    out = ["| Pattern | Trades | Win rate | Avg P&L (bid) |", "|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r.pattern_id} | {r.trades} | {r.win_rate*100:.1f}% | ${r.avg_pnl_bid:,.2f} |")
    return "\n".join(out)


def build_html(markdown_text: str) -> str:
    body = markdown_text.replace("\n", "<br>\n")
    return (
        "<!doctype html><html><head>"
        "<meta charset='utf-8'><title>zeroday-paper report</title>"
        "<style>body{font-family:system-ui;max-width:900px;margin:24px auto;padding:0 16px;line-height:1.5;}"
        "table{border-collapse:collapse;}th,td{border:1px solid #ddd;padding:6px 10px;}"
        "code{background:#f4f4f4;padding:2px 4px;border-radius:4px;}"
        "h1,h2{border-bottom:1px solid #eee;padding-bottom:4px;}</style>"
        f"</head><body><pre style='white-space:pre-wrap;'>{markdown_text}</pre></body></html>"
    )


def post_to_discord(markdown_text: str, *, webhook_url: str | None = None) -> bool:
    url = webhook_url or _resolve_webhook()
    if not url:
        logger.warning("report.no_webhook")
        return False

    chunks = _chunk_markdown(markdown_text, max_chars=1900)
    ok = True
    for chunk in chunks:
        try:
            r = httpx.post(url, json={"content": chunk}, timeout=10.0)
            if r.status_code not in (200, 204):
                logger.warning("report.discord_status", status=r.status_code, body=r.text[:200])
                ok = False
        except Exception as exc:
            logger.warning("report.discord_error", error=str(exc))
            ok = False
    return ok


def _resolve_webhook() -> str | None:
    try:
        return discord_webhook(settings.reporting.discord_webhook_secret_key)
    except Exception as exc:
        logger.warning("report.webhook_lookup_failed", error=str(exc))
        return None


def _chunk_markdown(text: str, max_chars: int) -> list[str]:
    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for line in text.splitlines(keepends=True):
        if cur_len + len(line) > max_chars and cur:
            chunks.append("".join(cur))
            cur, cur_len = [], 0
        cur.append(line)
        cur_len += len(line)
    if cur:
        chunks.append("".join(cur))
    return chunks


def write_report_files(markdown_text: str, *, base_dir: str = "./reports") -> tuple[Path, Path]:
    today = datetime.now(UTC).date().isoformat()
    out_dir = Path(base_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / f"paper-report-{today}.md"
    html_path = out_dir / f"paper-report-{today}.html"
    md_path.write_text(markdown_text)
    html_path.write_text(build_html(markdown_text))
    return md_path, html_path
