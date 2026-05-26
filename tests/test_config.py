"""Config: TOML loading + path resolution + Settings shape."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from zeroday_paper import config as cfg


def test_resolve_default_config_path_env_override(monkeypatch, tmp_path):
    fake = tmp_path / "alt.toml"
    fake.write_text("")
    monkeypatch.setenv("ZP_CONFIG_PATH", str(fake))
    assert cfg._resolve_default_config_path() == fake


def test_resolve_default_config_path_cwd_priority(monkeypatch, tmp_path):
    monkeypatch.delenv("ZP_CONFIG_PATH", raising=False)
    cwd_cfg = tmp_path / "config" / "paper.toml"
    cwd_cfg.parent.mkdir(parents=True)
    cwd_cfg.write_text("")
    monkeypatch.chdir(tmp_path)
    # /app/config/paper.toml does not exist locally
    resolved = cfg._resolve_default_config_path()
    assert resolved == cwd_cfg


def test_resolve_default_config_path_falls_back_to_project_root(monkeypatch, tmp_path):
    monkeypatch.delenv("ZP_CONFIG_PATH", raising=False)
    monkeypatch.chdir(tmp_path)  # CWD without config/paper.toml
    resolved = cfg._resolve_default_config_path()
    # Falls back to project root, which is where the real config lives
    assert resolved.name == "paper.toml"


def test_parse_hhmm():
    t = cfg._parse_hhmm("09:30")
    assert t.hour == 9
    assert t.minute == 30


def test_load_settings_default():
    s = cfg.load_settings()
    assert s.engine.poll_interval_seconds == 120
    assert s.engine.score_threshold == 15
    assert s.engine.market_timezone == "America/New_York"
    assert s.exits.stop_loss_multiple == 2.0


def test_load_settings_custom_path(tmp_path):
    custom = tmp_path / "custom.toml"
    # Just copy from the real config file then modify
    real_cfg = Path(__file__).resolve().parent.parent / "config" / "paper.toml"
    custom.write_text(real_cfg.read_text().replace(
        "poll_interval_seconds = 120", "poll_interval_seconds = 30"
    ))
    s = cfg.load_settings(custom)
    assert s.engine.poll_interval_seconds == 30


def test_get_settings_returns_singleton():
    a = cfg.get_settings()
    b = cfg.get_settings()
    assert a is b


def test_settings_duckdb_path_local(monkeypatch):
    monkeypatch.delenv("AWS_EXECUTION_ENV", raising=False)
    monkeypatch.delenv("ZP_DUCKDB_PATH", raising=False)
    s = cfg.get_settings()
    p = s.duckdb_path
    # On a dev machine without /data, should fall to local path
    if not os.path.exists("/data"):
        assert p == s.storage.duckdb_path_local


def test_settings_duckdb_path_env_override(monkeypatch):
    monkeypatch.delenv("AWS_EXECUTION_ENV", raising=False)
    monkeypatch.setenv("ZP_DUCKDB_PATH", "/tmp/custom.duckdb")
    s = cfg.get_settings()
    if not os.path.exists("/data"):
        assert s.duckdb_path == "/tmp/custom.duckdb"


def test_settings_duckdb_path_aws_path(monkeypatch):
    monkeypatch.setenv("AWS_EXECUTION_ENV", "AWS_ECS_FARGATE")
    s = cfg.get_settings()
    assert s.duckdb_path == s.storage.duckdb_path


def test_settings_dataclasses_all_present():
    s = cfg.get_settings()
    assert s.engine.session_start.hour == 9
    assert s.engine.session_end.hour == 15
    assert s.regime_gates.skip_negative_gex is True
    assert s.strikes.default_spread_width == 25
    assert s.patterns.layer_2_llm_enabled is True
    assert s.storage.schema_version == 1
    assert s.reporting.daily_report_hour_et == 16
    assert s.alarms.scanner_silent_threshold_minutes == 30
    assert s.secrets.polygon_secret_id.startswith("zeroday/")
    assert s.logging.level in ("INFO", "DEBUG", "WARNING", "ERROR")
