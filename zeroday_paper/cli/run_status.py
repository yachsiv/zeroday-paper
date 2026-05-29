"""On-demand status report.

Reads the DuckDB journal (read-only) + optionally runs a single cheap
``diagnostic_snapshot`` cycle, renders a markdown status post, and ships it
to the ``webhook_status`` Discord channel. Designed to be triggered from a
terminal by ``./scripts/post-status.sh`` (Fargate one-shot) or directly by
``uv run zp-status`` for local dev.

Usage:
    uv run zp-status
    uv run zp-status --print        # render + print, no Discord
    uv run zp-status --no-diag      # skip the optional cycle snapshot
    uv run zp-status --no-discord   # build + log, but don't post
"""

from __future__ import annotations

import argparse
import asyncio
import os
import time
from typing import Any

from zeroday_paper.logging_setup import configure_logging, get_logger
from zeroday_paper.reporting.status_brief import (
    build_bundle,
    post_to_discord,
    render_brief,
)


async def _maybe_diag() -> dict[str, Any] | None:
    """Run one cheap diagnostic snapshot. Returns None on any failure.

    The snapshot hits Polygon + CBOE which can fail or be slow in the off
    hours. We intentionally swallow exceptions here because the status post
    must always render — losing the "last cycle" section is acceptable.
    """
    log = get_logger(__name__)
    try:
        # Imported inside the function so module-level test runs can stub
        # ``run_status._maybe_diag`` without paying for the import chain.
        from zeroday_paper.engine.scanner import diagnostic_snapshot
        return await diagnostic_snapshot()
    except Exception as exc:
        log.warning("status.diag_skipped", error=str(exc))
        return None


def _resolve_revision() -> str | None:
    """Best-effort: AWS metadata if set, else None."""
    return (
        os.getenv("ECS_TASK_DEFINITION_REVISION")
        or os.getenv("ZP_REVISION")
        or None
    )


def _resolve_task_id() -> str | None:
    """Pull the task ARN's last segment from ECS metadata env when on Fargate.

    The ECS agent injects ``ECS_CONTAINER_METADATA_URI_V4`` for every task; we
    don't HTTP-fetch it from the CLI to keep things synchronous. If a caller
    set ``ZP_TASK_ID`` (the trigger script can do this with
    ``--cli-input-json`` if it wants), we honor that too.
    """
    return os.getenv("ZP_TASK_ID") or os.getenv("ECS_TASK_ID") or None


async def _run(*, no_discord: bool, print_to_stdout: bool, no_diag: bool) -> int:
    log = get_logger(__name__)
    log.info("status.boot")
    started = time.monotonic()

    diag = None if no_diag else await _maybe_diag()

    bundle = build_bundle(
        diag_snapshot=diag,
        revision=_resolve_revision(),
        task_id=_resolve_task_id(),
        runtime_seconds=time.monotonic() - started,
    )
    md = render_brief(bundle)

    if print_to_stdout:
        print(md)

    log.info(
        "status.built",
        chars=len(md),
        market_state=bundle.market_state,
        entries=bundle.today.entries,
        open=bundle.today.open_count,
        errors=list(bundle.errors),
        diag_used=diag is not None,
    )

    if no_discord:
        return 0

    ok = post_to_discord(md)
    log.info("status.discord_posted" if ok else "status.discord_skipped", ok=ok)
    return 0 if ok else 1


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="zeroday-paper on-demand status post")
    parser.add_argument("--no-discord", action="store_true",
                        help="Build the status but do not post to Discord")
    parser.add_argument("--print", dest="print_to_stdout", action="store_true",
                        help="Also print the rendered Markdown to stdout")
    parser.add_argument("--no-diag", action="store_true",
                        help="Skip the optional diagnostic_snapshot cycle (faster, no Polygon hit)")
    args = parser.parse_args()
    rc = asyncio.run(_run(
        no_discord=args.no_discord,
        print_to_stdout=args.print_to_stdout,
        no_diag=args.no_diag,
    ))
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
