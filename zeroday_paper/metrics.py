"""Fire-and-forget CloudWatch metrics.

Emits one metric data point per call. Failures are logged at warning level and
swallowed — metric emission must never break the scanner.

Used by:
  - scanner.run_one_cycle  → "cycle.complete" (heartbeat for ScannerSilentAlarm)
  - scanner._scan_for_entries → "trade.written" (rate of new paper entries)
  - scanner._monitor_open_positions → "position.exit" (rate of closes)
"""

from __future__ import annotations

import os
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

NAMESPACE = "zeroday/paper"

_client: Any | None = None
_disabled: bool = False


def _get_client() -> Any | None:
    global _client, _disabled
    if _disabled:
        return None
    if _client is not None:
        return _client
    try:
        import boto3
    except ImportError:
        _disabled = True
        return None
    region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"
    try:
        _client = boto3.client("cloudwatch", region_name=region)
    except Exception as exc:
        logger.debug("metrics.client_init_failed", error=str(exc))
        _disabled = True
        return None
    return _client


def emit(metric_name: str, value: float = 1.0, unit: str = "Count", **dims: str) -> None:
    """Emit a single metric. Safe to call from any code path."""
    client = _get_client()
    if client is None:
        return
    dimensions = [{"Name": k, "Value": str(v)} for k, v in dims.items()]
    payload: dict[str, Any] = {
        "MetricName": metric_name,
        "Value": float(value),
        "Unit": unit,
    }
    if dimensions:
        payload["Dimensions"] = dimensions
    try:
        client.put_metric_data(Namespace=NAMESPACE, MetricData=[payload])
    except Exception as exc:
        logger.debug("metrics.put_failed", metric=metric_name, error=str(exc))
