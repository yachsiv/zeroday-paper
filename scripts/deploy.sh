#!/usr/bin/env bash
# Deploy zeroday-paper to AWS.
#
# Prerequisites:
#   - aws CLI authenticated (us-east-1)
#   - docker daemon running
#   - cdk CLI installed (npm i -g aws-cdk)
#   - uv installed (https://docs.astral.sh/uv/)
#
# What it does:
#   1. Installs python deps via uv (incl. cdk extras)
#   2. `cdk bootstrap` (idempotent if already bootstrapped)
#   3. `cdk deploy ZerodayPaperStack` (builds + pushes Docker image, creates resources)
#
set -euo pipefail

cd "$(dirname "$0")/.."

echo "==> sync deps"
uv sync --extra cdk

echo "==> cdk bootstrap (idempotent)"
uv run cdk bootstrap "aws://146254095578/us-east-1" || true

echo "==> cdk synth"
uv run cdk synth ZerodayPaperStack --quiet

echo "==> cdk deploy"
uv run cdk deploy ZerodayPaperStack --require-approval never --outputs-file cdk-outputs.json

echo "==> deploy complete"
cat cdk-outputs.json | python3 -m json.tool
