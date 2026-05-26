"""Secrets Manager loader.

Reads existing zeroday/* secrets directly (single source of truth with main zeroday).
Caches values in-memory for the task lifetime — no repeated GetSecretValue calls.

In local dev: falls back to environment variables.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache

import structlog

from zeroday_paper.config import settings

logger = structlog.get_logger(__name__)


class SecretsError(RuntimeError):
    pass


@lru_cache(maxsize=8)
def _fetch_secret(secret_id: str, region: str = "us-east-1") -> dict[str, str]:
    """Fetch + cache a single Secrets Manager entry as a dict."""
    env_key = secret_id.replace("/", "_").upper()
    env_value = os.getenv(env_key)
    if env_value:
        try:
            return json.loads(env_value)
        except json.JSONDecodeError:
            return {"value": env_value}

    try:
        import boto3
    except ImportError as e:
        raise SecretsError(f"boto3 required to fetch secret {secret_id}") from e

    client = boto3.client("secretsmanager", region_name=region)
    try:
        resp = client.get_secret_value(SecretId=secret_id)
    except Exception as e:
        raise SecretsError(f"Failed to fetch {secret_id}: {e}") from e

    secret_str = resp.get("SecretString")
    if not secret_str:
        raise SecretsError(f"Secret {secret_id} has no SecretString")

    try:
        return json.loads(secret_str)
    except json.JSONDecodeError:
        return {"value": secret_str}


def polygon_api_key() -> str:
    """Polygon Options Advanced API key."""
    local = os.getenv("POLYGON_API_KEY")
    if local:
        return local
    data = _fetch_secret(settings.secrets.polygon_secret_id)
    return data.get("api_key") or data.get("POLYGON_API_KEY") or data["value"]


def flashalpha_api_key() -> str:
    """FlashAlpha SDK API key."""
    local = os.getenv("FLASHALPHA_API_KEY")
    if local:
        return local
    data = _fetch_secret(settings.secrets.flashalpha_secret_id)
    return data.get("api_key") or data.get("FLASHALPHA_API_KEY") or data["value"]


def anthropic_api_key() -> str:
    """Anthropic Messages API key."""
    local = os.getenv("ANTHROPIC_API_KEY")
    if local:
        return local
    data = _fetch_secret(settings.secrets.anthropic_secret_id)
    return data.get("api_key") or data.get("ANTHROPIC_API_KEY") or data["value"]


def discord_webhook(key: str | None = None) -> str:
    """Discord webhook URL by key name (defaults to reporting.discord_webhook_secret_key).

    The zeroday/discord secret has multiple webhook URLs; we pick one by name.
    """
    local = os.getenv("DISCORD_WEBHOOK_URL")
    if local:
        return local
    webhook_key = key or settings.reporting.discord_webhook_secret_key
    data = _fetch_secret(settings.secrets.discord_secret_id)
    if webhook_key not in data:
        raise SecretsError(
            f"Discord secret has no key '{webhook_key}'. "
            f"Available keys: {list(data.keys())}"
        )
    return data[webhook_key]
