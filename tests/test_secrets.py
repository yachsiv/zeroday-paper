"""Secrets loader: env-first, fallback to boto3, JSON-vs-raw parsing."""

from __future__ import annotations

import json

import pytest

from zeroday_paper import secrets as ss


@pytest.fixture(autouse=True)
def clear_cache():
    """Reset lru_cache between tests so env changes are picked up."""
    ss._fetch_secret.cache_clear()
    yield
    ss._fetch_secret.cache_clear()


def test_polygon_api_key_env_var(monkeypatch):
    monkeypatch.setenv("POLYGON_API_KEY", "env-key")
    assert ss.polygon_api_key() == "env-key"


def test_flashalpha_api_key_env_var(monkeypatch):
    monkeypatch.setenv("FLASHALPHA_API_KEY", "fa-env")
    assert ss.flashalpha_api_key() == "fa-env"


def test_anthropic_api_key_env_var(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anth-env")
    assert ss.anthropic_api_key() == "anth-env"


def test_discord_webhook_env_var(monkeypatch):
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://hook.env")
    assert ss.discord_webhook() == "https://hook.env"


def test_fetch_secret_via_env_with_json(monkeypatch):
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)
    monkeypatch.setenv("ZERODAY_POLYGON", json.dumps({"api_key": "from-json"}))
    d = ss._fetch_secret("zeroday/polygon")
    assert d == {"api_key": "from-json"}


def test_fetch_secret_via_env_raw(monkeypatch):
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)
    monkeypatch.setenv("ZERODAY_POLYGON", "plain-string")
    d = ss._fetch_secret("zeroday/polygon")
    assert d == {"value": "plain-string"}


def test_polygon_api_key_uses_secret_value_fallback(monkeypatch):
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)
    monkeypatch.setenv("ZERODAY_POLYGON", json.dumps({"POLYGON_API_KEY": "uppercase-key"}))
    assert ss.polygon_api_key() == "uppercase-key"


def test_polygon_api_key_uses_value_fallback(monkeypatch):
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)
    monkeypatch.setenv("ZERODAY_POLYGON", "the-raw-key")
    assert ss.polygon_api_key() == "the-raw-key"


def test_discord_webhook_specific_key_missing_raises(monkeypatch):
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
    monkeypatch.setenv("ZERODAY_DISCORD", json.dumps({"webhook_other": "u1"}))
    with pytest.raises(ss.SecretsError):
        ss.discord_webhook("webhook_missing")


def test_discord_webhook_specific_key_present(monkeypatch):
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
    monkeypatch.setenv("ZERODAY_DISCORD",
                       json.dumps({"webhook_shadow": "https://shadow", "webhook_monitor": "https://mon"}))
    assert ss.discord_webhook("webhook_shadow") == "https://shadow"
    assert ss.discord_webhook("webhook_monitor") == "https://mon"


def test_fetch_secret_uses_boto3_path_when_no_env(monkeypatch):
    monkeypatch.delenv("ZERODAY_POLYGON", raising=False)

    captured = {}

    class _StubClient:
        def get_secret_value(self, SecretId):
            captured["secret_id"] = SecretId
            return {"SecretString": json.dumps({"api_key": "from-boto"})}

    class _StubBoto3:
        def client(self, name, region_name=None):
            captured["client_kind"] = name
            captured["region"] = region_name
            return _StubClient()

    monkeypatch.setitem(__import__("sys").modules, "boto3", _StubBoto3())
    out = ss._fetch_secret("zeroday/polygon")
    assert out == {"api_key": "from-boto"}
    assert captured["secret_id"] == "zeroday/polygon"
    assert captured["client_kind"] == "secretsmanager"


def test_fetch_secret_boto3_no_secret_string_raises(monkeypatch):
    monkeypatch.delenv("ZERODAY_POLYGON", raising=False)

    class _StubClient:
        def get_secret_value(self, SecretId):
            return {}  # no SecretString

    class _StubBoto3:
        def client(self, name, region_name=None):
            return _StubClient()

    monkeypatch.setitem(__import__("sys").modules, "boto3", _StubBoto3())
    with pytest.raises(ss.SecretsError):
        ss._fetch_secret("zeroday/polygon")


def test_fetch_secret_boto3_fetch_failure_raises(monkeypatch):
    monkeypatch.delenv("ZERODAY_POLYGON", raising=False)

    class _StubClient:
        def get_secret_value(self, SecretId):
            raise RuntimeError("network down")

    class _StubBoto3:
        def client(self, name, region_name=None):
            return _StubClient()

    monkeypatch.setitem(__import__("sys").modules, "boto3", _StubBoto3())
    with pytest.raises(ss.SecretsError):
        ss._fetch_secret("zeroday/polygon")


def test_fetch_secret_boto3_raw_string_returned(monkeypatch):
    monkeypatch.delenv("ZERODAY_POLYGON", raising=False)

    class _StubClient:
        def get_secret_value(self, SecretId):
            return {"SecretString": "plain"}

    class _StubBoto3:
        def client(self, name, region_name=None):
            return _StubClient()

    monkeypatch.setitem(__import__("sys").modules, "boto3", _StubBoto3())
    out = ss._fetch_secret("zeroday/polygon")
    assert out == {"value": "plain"}
