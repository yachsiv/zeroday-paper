#!/usr/bin/env bash
# Dispatch MODE → CLI command.
#
#   MODE=live    → zp-live   (2-min loop, exits at session end)
#   MODE=replay  → zp-replay (one-shot historical fill)
#   MODE=report  → zp-report (build + post daily report)
#
# Any extra args are forwarded.
set -euo pipefail

MODE="${MODE:-live}"
echo "[entrypoint] MODE=$MODE args=$*"

case "$MODE" in
  live)
    exec zp-live "$@"
    ;;
  replay)
    exec zp-replay "$@"
    ;;
  report)
    exec zp-report "$@"
    ;;
  *)
    echo "[entrypoint] unknown MODE: $MODE" >&2
    exit 64
    ;;
esac
