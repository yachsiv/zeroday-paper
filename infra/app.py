#!/usr/bin/env python3
"""CDK app entry — single stack."""

from __future__ import annotations

import os

import aws_cdk as cdk

from infra.stack import ZerodayPaperStack


app = cdk.App()

account = os.getenv("CDK_DEFAULT_ACCOUNT") or "146254095578"
region = os.getenv("CDK_DEFAULT_REGION") or "us-east-1"

ZerodayPaperStack(
    app,
    "ZerodayPaperStack",
    env=cdk.Environment(account=account, region=region),
    description="SPX 0DTE paper trading engine — replay + live + reporting.",
    tags={
        "Project": "zeroday-paper",
        "Owner": "yachsiv",
        "CostCenter": "paper-trading",
    },
)

app.synth()
