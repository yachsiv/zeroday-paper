# zeroday-paper

Standalone SPX 0DTE **paper trading** engine. Designed to run for 30+ days unattended on AWS,
collecting maximum-density paper trade data with bid-price realism and 15-pattern GEX labels.

Companion to the main **zeroday-trading** repo. No code dependency on it — but mirrors the
critical clients (Polygon, FlashAlpha, CBOE) so this repo can be deployed independently.

## Design principles

- **Single purpose.** Collect maximum paper trades at maximum accuracy. Never places real orders.
- **Zero noise.** No Discord trade alerts. One nightly Discord report + one silent-scanner alarm.
- **Local-first DuckDB.** Single file storage. EFS-mounted on AWS for persistence across task restarts.
- **Bid pricing only.** Every credit / exit cost uses `short.bid – long.ask` (entry) and `short.ask – long.bid` (exit). Mid stored only for comparison.
- **Idempotent.** `trade_id = sha256(date + entry_minute + short_strike + long_strike + strategy)`. Restart = no duplicates.
- **Replay first, live second.** 365-day historical replay produces 350–500 trades on day 1. Live polling adds 3–5/day.

## Architecture

```
EventBridge (cron)
  ├─ 09:20 ET start  → ECS Fargate task: paper-live (2-min scan loop)
  ├─ 16:00 ET stop   → stop live task
  └─ 16:30 ET report → ECS Fargate task: paper-report → Discord webhook

ECS Fargate task: paper-replay (one-shot, ~6 h)
  └─ Iterates last 365 trading days, replays full pipeline at 2-min cadence

All tasks share:
  - EFS-mounted /data/paper.duckdb       (persistent storage)
  - Secrets Manager: zeroday/*           (Polygon, FlashAlpha, Anthropic, Discord)
  - CloudWatch Logs                      (30-day retention)
  - S3: zeroday-paper-backup             (nightly DuckDB sync + report archive)
```

## Quick start (local dev)

```bash
uv sync                                                       # installs deps
cp .env.example .env                                          # fill in keys
uv run python -m zeroday_paper.cli.run_replay --days 30       # smoke test on 30 days
uv run python -m zeroday_paper.cli.run_live                   # live scanner (market hours)
uv run python -m zeroday_paper.cli.run_report                 # generate today's report
```

## Deployment (AWS)

```bash
cd infra
uv sync --extra cdk
uv run cdk synth
uv run cdk deploy ZerodayPaperStack --require-approval never
```

## Configuration

All tunables in `config/paper.toml`. Change cadence, score threshold, profit targets, concurrency
caps without touching code.

## Data flow per trade

1. **Scan** — fetch chain + GEX + VIX1D every 2 min during market hours.
2. **Score** — pure function over `MarketState` → `ScoreResult`.
3. **Classify** — Layer 1 (rules, 15 patterns) + Layer 2 (Anthropic Claude, async).
4. **Strike select** — 4-stage: geometric → quality gates → ATR width → ranking.
5. **Entry decision** — score ≥ 15, regime gate, time gate, dedup, concurrency caps.
6. **Journal write** — DuckDB row with 40+ fields including `credit_bid`, `active_patterns`, `vix_1d`, `rr25`.
7. **Monitor** — every 2 min: profit target, stop loss, hard close, thesis invalidation.
8. **Exit** — bid-priced close cost. P&L = `(credit_bid – exit_cost_bid) × 100 × contracts`.

## Reports

- **Daily** (Mon–Fri 16:30 ET): Discord webhook → channel; full HTML on S3.
- **Weekly** (Sunday): pattern leaderboard, regime breakdown, anomaly review.
- **End of month**: graduation decision — port winning parameters to main zeroday for live trading.

## License

Proprietary. Not for redistribution.
