"""Build today's report + write files + post to Discord.

Usage:
    uv run zp-report
    uv run zp-report --no-discord
"""

from __future__ import annotations

import argparse

from zeroday_paper.logging_setup import configure_logging, get_logger
from zeroday_paper.reporting.daily_report import (
    build_markdown,
    post_to_discord,
    write_report_files,
)


def main() -> None:
    configure_logging()
    log = get_logger(__name__)

    parser = argparse.ArgumentParser(description="zeroday-paper daily report")
    parser.add_argument("--no-discord", action="store_true", help="Skip Discord post (write files only)")
    parser.add_argument("--reports-dir", default="./reports")
    args = parser.parse_args()

    md = build_markdown()
    md_path, html_path = write_report_files(md, base_dir=args.reports_dir)
    log.info("report.written", markdown=str(md_path), html=str(html_path))

    if not args.no_discord:
        ok = post_to_discord(md)
        log.info("report.discord_posted" if ok else "report.discord_skipped", ok=ok)


if __name__ == "__main__":
    main()
