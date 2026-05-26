"""Build today's pre-market brief + post to Discord.

Usage:
    uv run zp-morning
    uv run zp-morning --no-discord    # render only, skip Discord post
    uv run zp-morning --print         # render only, print to stdout (no Discord)
"""

from __future__ import annotations

import argparse
import asyncio

from zeroday_paper.config import settings
from zeroday_paper.logging_setup import configure_logging, get_logger
from zeroday_paper.reporting.morning_brief import (
    build_bundle,
    post_to_discord,
    render_brief,
)


async def _run(no_discord: bool, print_to_stdout: bool) -> int:
    log = get_logger(__name__)

    if not settings.morning_brief.enabled:
        log.warning("morning.disabled_via_config")
        return 0

    bundle = await build_bundle()
    md = render_brief(bundle)

    if print_to_stdout:
        print(md)

    log.info(
        "morning.brief_built",
        chars=len(md),
        failures=list(bundle.failures.keys()),
        is_0dte=bundle.meta.is_0dte_day,
        is_holiday=bundle.meta.is_holiday,
    )

    if no_discord:
        return 0

    ok = post_to_discord(md)
    log.info("morning.discord_posted" if ok else "morning.discord_skipped", ok=ok)
    return 0 if ok else 1


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="zeroday-paper pre-market brief")
    parser.add_argument("--no-discord", action="store_true",
                        help="Build the brief but do not post to Discord")
    parser.add_argument("--print", dest="print_to_stdout", action="store_true",
                        help="Also print the rendered Markdown to stdout")
    args = parser.parse_args()
    rc = asyncio.run(_run(args.no_discord, args.print_to_stdout))
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
