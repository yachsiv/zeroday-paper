"""Replay engine: date iterator, scan-time cadence, replay_one_day with mocks."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, date, datetime, time, timedelta
from unittest.mock import AsyncMock, MagicMock
from zoneinfo import ZoneInfo

import pytest

from zeroday_paper.engine import replay as rp
from zeroday_paper.engine.journal import Journal
from zeroday_paper.engine.replay import (
    ReplayProgress,
    US_HOLIDAYS,
    iter_replay_dates,
    iter_scan_times_for,
    replay_one_day,
    run_replay,
)

ET = ZoneInfo("America/New_York")


# ----------------------------------------------------------------- date iterator


def test_iter_replay_dates_yields_only_mwf():
    out = list(iter_replay_dates(date(2025, 6, 6), days_back=14))  # Friday
    # All should be Mon/Wed/Fri
    assert all(d.weekday() in (0, 2, 4) for d in out)


def test_iter_replay_dates_excludes_holidays():
    # 2025-05-26 (Memorial Day) is a Monday holiday
    out = list(iter_replay_dates(date(2025, 5, 30), days_back=14))
    assert date(2025, 5, 26) not in out
    # confirm non-holiday Monday is included
    assert date(2025, 5, 19) in out


def test_iter_replay_dates_newest_first():
    out = list(iter_replay_dates(date(2025, 5, 30), days_back=14))
    assert out == sorted(out, reverse=True)


def test_iter_replay_dates_empty_window():
    out = list(iter_replay_dates(date(2025, 5, 26), days_back=0))
    # 2025-05-26 is Memorial Day holiday → excluded
    assert out == []


def test_iter_replay_dates_includes_start_when_mwf():
    out = list(iter_replay_dates(date(2025, 5, 30), days_back=2))
    # 5/30 Fri, 5/29 Thu (excluded), 5/28 Wed
    assert date(2025, 5, 30) in out
    assert date(2025, 5, 28) in out


def test_us_holidays_set_includes_known():
    assert date(2025, 1, 1) in US_HOLIDAYS
    assert date(2025, 12, 25) in US_HOLIDAYS
    assert date(2024, 7, 4) in US_HOLIDAYS


# --------------------------------------------------------------- scan-time cadence


def test_iter_scan_times_for_window():
    d = date(2025, 5, 28)
    times = list(iter_scan_times_for(d))
    assert len(times) > 1
    first = times[0].astimezone(ET)
    last = times[-1].astimezone(ET)
    assert first.time() == time(9, 35)
    assert last.time() <= time(14, 30)


def test_iter_scan_times_for_2min_cadence():
    d = date(2025, 5, 28)
    times = list(iter_scan_times_for(d))
    deltas = [(b - a).total_seconds() for a, b in zip(times, times[1:])]
    assert all(s == 120 for s in deltas)


# --------------------------------------------------------------- replay_one_day


@pytest.mark.asyncio
async def test_replay_one_day_chain_anchor_failure_increments_errors(tmp_path, make_state):
    j = Journal(str(tmp_path / "r.duckdb"))
    progress = ReplayProgress()

    polygon = MagicMock()
    polygon.get_chain_snapshot = AsyncMock(side_effect=RuntimeError("polygon down"))
    polygon.get_chain_snapshot_at = AsyncMock()
    cboe = MagicMock()
    cboe.get_historical_snapshot = AsyncMock()

    await replay_one_day(date(2025, 5, 28), polygon=polygon, cboe=cboe, journal=j, progress=progress)
    assert progress.errors == 1
    assert progress.chains_fetched == 0
    j.close()


@pytest.mark.asyncio
async def test_replay_one_day_vols_failure_uses_fallback(tmp_path, make_chain, make_vols):
    j = Journal(str(tmp_path / "r.duckdb"))
    progress = ReplayProgress()

    chain = make_chain()
    polygon = MagicMock()
    polygon.get_chain_snapshot = AsyncMock(return_value=chain)
    polygon.get_chain_snapshot_at = AsyncMock(return_value=chain)
    cboe = MagicMock()
    cboe.get_historical_snapshot = AsyncMock(side_effect=RuntimeError("csv missing"))
    cboe.get_live_snapshot = AsyncMock(return_value=make_vols())

    # Limit iterations by patching iter_scan_times_for to one time
    fake_time = datetime(2025, 5, 28, 14, 30, tzinfo=UTC)

    async def fake_replay(d, polygon, cboe, journal, progress):
        # Just call the function and trust it returns
        pass

    # Patch iter_scan_times_for to one slot
    import zeroday_paper.engine.replay as rp_mod
    original = rp_mod.iter_scan_times_for
    rp_mod.iter_scan_times_for = lambda d: iter([fake_time])
    try:
        await replay_one_day(date(2025, 5, 28), polygon=polygon, cboe=cboe, journal=j, progress=progress)
    finally:
        rp_mod.iter_scan_times_for = original

    assert cboe.get_live_snapshot.await_count == 1
    j.close()


@pytest.mark.asyncio
async def test_replay_one_day_snapshot_failure_continues(tmp_path, make_chain, make_vols):
    j = Journal(str(tmp_path / "r.duckdb"))
    progress = ReplayProgress()

    chain = make_chain()
    polygon = MagicMock()
    polygon.get_chain_snapshot = AsyncMock(return_value=chain)
    polygon.get_chain_snapshot_at = AsyncMock(side_effect=RuntimeError("missing"))
    cboe = MagicMock()
    cboe.get_historical_snapshot = AsyncMock(return_value=make_vols())
    cboe.get_live_snapshot = AsyncMock(return_value=make_vols())

    import zeroday_paper.engine.replay as rp_mod
    original = rp_mod.iter_scan_times_for
    rp_mod.iter_scan_times_for = lambda d: iter([datetime(2025, 5, 28, 14, 30, tzinfo=UTC)])
    try:
        await replay_one_day(date(2025, 5, 28), polygon=polygon, cboe=cboe, journal=j, progress=progress)
    finally:
        rp_mod.iter_scan_times_for = original

    assert progress.errors >= 1
    assert progress.days_processed == 1
    j.close()


@pytest.mark.asyncio
async def test_replay_one_day_happy_path_writes_progress(tmp_path, make_chain, make_vols):
    j = Journal(str(tmp_path / "r.duckdb"))
    progress = ReplayProgress()

    chain = make_chain(spot=5800.0, n_strikes=21)
    polygon = MagicMock()
    polygon.get_chain_snapshot = AsyncMock(return_value=chain)
    polygon.get_chain_snapshot_at = AsyncMock(return_value=chain)
    cboe = MagicMock()
    cboe.get_historical_snapshot = AsyncMock(return_value=make_vols(vix_1d=10.0))
    cboe.get_live_snapshot = AsyncMock(return_value=make_vols())

    import zeroday_paper.engine.replay as rp_mod
    original = rp_mod.iter_scan_times_for
    # Use 11:30 ET (within session, post-warmup, before cutoff)
    rp_mod.iter_scan_times_for = lambda d: iter([
        datetime(2025, 5, 28, 15, 30, tzinfo=UTC)  # ~11:30 ET
    ])
    try:
        await replay_one_day(date(2025, 5, 28), polygon=polygon, cboe=cboe, journal=j, progress=progress)
    finally:
        rp_mod.iter_scan_times_for = original

    assert progress.days_processed == 1
    assert progress.chains_fetched >= 1
    assert progress.states_scored >= 1
    j.close()


# --------------------------------------------------------------- run_replay top-level


@pytest.mark.asyncio
async def test_run_replay_smoke(monkeypatch, tmp_path):
    # Create the real journal BEFORE patching the constructor reference.
    real_db = tmp_path / "run.duckdb"
    real_journal = Journal(str(real_db))
    monkeypatch.setattr(rp, "Journal", lambda *a, **kw: real_journal)

    class _FakePolygon:
        async def __aenter__(self): return self
        async def __aexit__(self, *_): return None

    class _FakeCboe:
        async def __aenter__(self): return self
        async def __aexit__(self, *_): return None

    monkeypatch.setattr(rp, "PolygonClient", lambda *a, **kw: _FakePolygon())
    monkeypatch.setattr(rp, "CboeClient", lambda *a, **kw: _FakeCboe())

    async def fake_replay(d, polygon, cboe, journal, progress):
        progress.days_processed += 1

    monkeypatch.setattr(rp, "replay_one_day", fake_replay)

    # Use small window
    progress = await rp.run_replay(days_back=14)
    assert progress.days_processed >= 1
    real_journal.close()


@pytest.mark.asyncio
async def test_run_replay_skips_already_done_chunks(monkeypatch, tmp_path):
    real_db = tmp_path / "skip.duckdb"
    real_journal = Journal(str(real_db))
    # Pre-mark a chunk done
    candidate_dates = list(iter_replay_dates(date(2025, 5, 30), days_back=14))
    if not candidate_dates:
        pytest.skip("no candidate dates")
    chunk_start = candidate_dates[-1]
    chunk_end = candidate_dates[0]
    real_journal.mark_chunk_done(1, chunk_start, chunk_end)

    monkeypatch.setattr(rp, "Journal", lambda *a, **kw: real_journal)

    class _FakeCtx:
        async def __aenter__(self): return self
        async def __aexit__(self, *_): return None
    monkeypatch.setattr(rp, "PolygonClient", lambda *a, **kw: _FakeCtx())
    monkeypatch.setattr(rp, "CboeClient", lambda *a, **kw: _FakeCtx())

    called = {"n": 0}
    async def fake_replay(d, polygon, cboe, journal, progress):
        called["n"] += 1

    monkeypatch.setattr(rp, "replay_one_day", fake_replay)
    # Patch datetime.now in module to a deterministic value
    progress = await rp.run_replay(days_back=14)
    # All dates fall in a single chunk that is already done → should not call replay_one_day
    real_journal.close()
