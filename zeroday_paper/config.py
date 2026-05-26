"""Runtime config loader.

Single source of truth for all tunables. Reads `config/paper.toml` and provides
a typed `Settings` object.

Usage:
    from zeroday_paper.config import settings
    settings.engine.poll_interval_seconds  # 120
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from datetime import time
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve_default_config_path() -> Path:
    """Locate config/paper.toml across packaged + dev + container layouts.

    Search order:
      1. $ZP_CONFIG_PATH
      2. /app/config/paper.toml (Docker WORKDIR)
      3. CWD / config / paper.toml
      4. PROJECT_ROOT / config / paper.toml (works in `uv run` dev mode)
    """
    env_override = os.getenv("ZP_CONFIG_PATH")
    if env_override:
        return Path(env_override)

    candidates = [
        Path("/app/config/paper.toml"),
        Path.cwd() / "config" / "paper.toml",
        PROJECT_ROOT / "config" / "paper.toml",
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[-1]


DEFAULT_CONFIG_PATH = _resolve_default_config_path()


def _parse_hhmm(value: str) -> time:
    h, m = value.split(":")
    return time(int(h), int(m))


@dataclass(frozen=True)
class EngineConfig:
    poll_interval_seconds: int
    score_threshold: int
    session_start: time
    session_end: time
    new_entry_cutoff: time
    market_timezone: str
    warmup_minutes_after_open: int


@dataclass(frozen=True)
class RegimeGatesConfig:
    skip_negative_gex: bool
    vix_1d_skip_threshold: float
    cboe_skew_skip_threshold: float
    rr25_skew_steepening_threshold_pp: float


@dataclass(frozen=True)
class StrikesConfig:
    default_short_delta_target: float
    default_short_delta_tolerance: float
    default_spread_width: int
    min_short_oi: int
    min_long_oi: int
    max_bid_ask_spread_dollars: float


@dataclass(frozen=True)
class ExitsConfig:
    profit_target_pre_11am: float
    profit_target_11_to_1pm: float
    profit_target_1_to_2pm: float
    profit_target_after_2pm: float
    stop_loss_multiple: float
    defensive_mode_start: time
    hard_close: time


@dataclass(frozen=True)
class ConcurrencyConfig:
    max_concurrent_total: int
    max_bull_put_per_day: int
    max_bear_call_per_day: int
    max_iron_condor_per_day: int
    min_minutes_between_same_strategy: int


@dataclass(frozen=True)
class DedupConfig:
    same_spread_window_minutes: int


@dataclass(frozen=True)
class ReplayConfig:
    days_back: int
    chunk_size_days: int
    rate_limit_polygon_req_per_min: int


@dataclass(frozen=True)
class PatternsConfig:
    layer_2_llm_enabled: bool
    layer_2_min_confidence: float
    layer_2_max_calls_per_scan: int
    layer_2_model: str
    layer_2_timeout_seconds: int


@dataclass(frozen=True)
class StorageConfig:
    duckdb_path: str
    duckdb_path_local: str
    backup_to_s3: bool
    backup_bucket: str
    backup_interval_minutes: int
    schema_version: int


@dataclass(frozen=True)
class ReportingConfig:
    daily_report_hour_et: int
    daily_report_minute_et: int
    discord_webhook_secret_key: str
    include_replay_summary_first_n_days: int


@dataclass(frozen=True)
class AlarmsConfig:
    scanner_silent_threshold_minutes: int
    alarm_discord_webhook_secret_key: str


@dataclass(frozen=True)
class SecretsConfig:
    polygon_secret_id: str
    flashalpha_secret_id: str
    anthropic_secret_id: str
    discord_secret_id: str


@dataclass(frozen=True)
class LoggingConfig:
    level: str
    format: str
    no_alerts_during_day: bool


@dataclass(frozen=True)
class Settings:
    engine: EngineConfig
    regime_gates: RegimeGatesConfig
    strikes: StrikesConfig
    exits: ExitsConfig
    concurrency: ConcurrencyConfig
    dedup: DedupConfig
    replay: ReplayConfig
    patterns: PatternsConfig
    storage: StorageConfig
    reporting: ReportingConfig
    alarms: AlarmsConfig
    secrets: SecretsConfig
    logging: LoggingConfig

    @property
    def duckdb_path(self) -> str:
        """Pick the right DuckDB path for the current environment."""
        on_aws = os.getenv("AWS_EXECUTION_ENV") is not None or os.path.exists("/data")
        if on_aws:
            return self.storage.duckdb_path
        return os.getenv("ZP_DUCKDB_PATH") or self.storage.duckdb_path_local


def load_settings(path: Path | None = None) -> Settings:
    cfg_path = path or DEFAULT_CONFIG_PATH
    raw: dict[str, Any] = tomllib.loads(cfg_path.read_text())

    return Settings(
        engine=EngineConfig(
            poll_interval_seconds=raw["engine"]["poll_interval_seconds"],
            score_threshold=raw["engine"]["score_threshold"],
            session_start=_parse_hhmm(raw["engine"]["session_start"]),
            session_end=_parse_hhmm(raw["engine"]["session_end"]),
            new_entry_cutoff=_parse_hhmm(raw["engine"]["new_entry_cutoff"]),
            market_timezone=raw["engine"]["market_timezone"],
            warmup_minutes_after_open=raw["engine"]["warmup_minutes_after_open"],
        ),
        regime_gates=RegimeGatesConfig(**raw["regime_gates"]),
        strikes=StrikesConfig(**raw["strikes"]),
        exits=ExitsConfig(
            profit_target_pre_11am=raw["exits"]["profit_target_pre_11am"],
            profit_target_11_to_1pm=raw["exits"]["profit_target_11_to_1pm"],
            profit_target_1_to_2pm=raw["exits"]["profit_target_1_to_2pm"],
            profit_target_after_2pm=raw["exits"]["profit_target_after_2pm"],
            stop_loss_multiple=raw["exits"]["stop_loss_multiple"],
            defensive_mode_start=_parse_hhmm(raw["exits"]["defensive_mode_start"]),
            hard_close=_parse_hhmm(raw["exits"]["hard_close"]),
        ),
        concurrency=ConcurrencyConfig(**raw["concurrency"]),
        dedup=DedupConfig(**raw["dedup"]),
        replay=ReplayConfig(**raw["replay"]),
        patterns=PatternsConfig(**raw["patterns"]),
        storage=StorageConfig(**raw["storage"]),
        reporting=ReportingConfig(**raw["reporting"]),
        alarms=AlarmsConfig(**raw["alarms"]),
        secrets=SecretsConfig(**raw["secrets"]),
        logging=LoggingConfig(**raw["logging"]),
    )


# Module-level singleton (lazy)
_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = load_settings()
    return _settings


settings = get_settings()
