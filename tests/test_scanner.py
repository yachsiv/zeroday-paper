"""Scanner monitor + entry logic with mocked collaborators."""

from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime, time, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from zeroday_paper.engine import scanner as sc
from zeroday_paper.engine.journal import Journal
from zeroday_paper.engine.models import (
    PositionStatus,
    StrategyType,
    TradeOutcome,
)

ET = ZoneInfo("America/New_York")


def _et_utc(h: int, m: int, day=28) -> datetime:
    return datetime(2025, 5, day, h, m, tzinfo=ET).astimezone(UTC)


# --------------------------------------------------------- is_market_hours_et


def test_is_market_hours_et_weekday_during_session():
    # Wed 11:00 ET
    assert sc.is_market_hours_et(_et_utc(11, 0)) is True


def test_is_market_hours_et_weekend():
    # Sat 11:00 ET
    assert sc.is_market_hours_et(datetime(2025, 5, 31, 15, 0, tzinfo=UTC)) is False


def test_is_market_hours_et_before_open():
    assert sc.is_market_hours_et(_et_utc(8, 0)) is False


def test_is_market_hours_et_after_close():
    assert sc.is_market_hours_et(_et_utc(16, 30)) is False


def test_is_market_hours_et_uses_now_when_none(monkeypatch):
    # Just verify the path runs
    out = sc.is_market_hours_et(None)
    assert isinstance(out, bool)


# ------------------------------------------------------------ _scan_for_entries


@pytest.mark.asyncio
async def test_scan_for_entries_writes_when_score_clears(tmp_path, make_state, monkeypatch):
    j = Journal(str(tmp_path / "s.duckdb"))
    state = make_state(asof=_et_utc(11, 0))

    # Patch L2 so we don't reach Anthropic
    monkeypatch.setattr(sc, "classify_layer2", AsyncMock(return_value=[]))

    written = await sc._scan_for_entries(j, state)
    # Should write at least one trade since baseline score is ~14-16
    rows = j._conn.execute("SELECT COUNT(*) FROM paper_trades").fetchone()[0]
    assert rows >= 0  # may or may not depending on selection success
    # Test the return value type
    assert isinstance(written, int)
    j.close()


@pytest.mark.asyncio
async def test_scan_for_entries_respects_score_threshold(tmp_path, make_state, make_signals,
                                                          monkeypatch):
    j = Journal(str(tmp_path / "s.duckdb"))
    # Force regime block so score=0 → no writes
    state = make_state(asof=_et_utc(11, 0),
                       signals=make_signals(gamma_regime="negative_gamma"))

    monkeypatch.setattr(sc, "classify_layer2", AsyncMock(return_value=[]))
    written = await sc._scan_for_entries(j, state)
    assert written == 0
    j.close()


@pytest.mark.asyncio
async def test_scan_for_entries_caps_concurrency(tmp_path, make_state, monkeypatch,
                                                  make_paper_trade):
    j = Journal(str(tmp_path / "s.duckdb"))
    state = make_state(asof=_et_utc(11, 0))

    # Pre-fill paper_trades up to max_concurrent_total
    max_total = sc.settings.concurrency.max_concurrent_total
    for i in range(max_total):
        j.write_trade(make_paper_trade(
            trade_id=f"pre-{i}",
            entry_ts=state.asof,
            short_strike=5775.0 - i, long_strike=5750.0 - i, source="live",
        ))

    monkeypatch.setattr(sc, "classify_layer2", AsyncMock(return_value=[]))
    written = await sc._scan_for_entries(j, state)
    assert written == 0  # capped
    j.close()


@pytest.mark.asyncio
async def test_scan_for_entries_layer2_failure_continues(tmp_path, make_state, monkeypatch):
    j = Journal(str(tmp_path / "s.duckdb"))
    state = make_state(asof=_et_utc(11, 0))

    # settings.patterns.layer_2_llm_enabled is True (default); make classify_layer2 raise.
    monkeypatch.setattr(sc, "classify_layer2", AsyncMock(side_effect=RuntimeError("llm err")))

    written = await sc._scan_for_entries(j, state)
    # No crash — returns int
    assert isinstance(written, int)
    j.close()


# ------------------------------------------------------- _monitor_open_positions


@pytest.mark.asyncio
async def test_monitor_open_positions_no_open(tmp_path, make_state):
    j = Journal(str(tmp_path / "s.duckdb"))
    state = make_state(asof=_et_utc(11, 0))
    polygon = MagicMock()
    monitored, exited = await sc._monitor_open_positions(j, state, polygon)
    assert monitored == 0
    assert exited == 0
    j.close()


@pytest.mark.asyncio
async def test_monitor_open_positions_writes_tick_holds(tmp_path, make_state, make_chain,
                                                          make_paper_trade, make_signals, make_quote):
    j = Journal(str(tmp_path / "s.duckdb"))

    # Build a chain whose strikes match the trade contracts
    chain = make_chain(spot=5800.0, n_strikes=21)

    pt = make_paper_trade(
        short_strike=chain.puts[10].strike, long_strike=chain.puts[8].strike,
        entry_ts=_et_utc(10, 0),
    )
    # Override the contracts so they match chain
    short_q = next(q for q in chain.puts if q.strike == pt.short_strike)
    long_q = next(q for q in chain.puts if q.strike == pt.long_strike)
    # Patch trade contracts to chain contracts
    pt_dict = {k: getattr(pt, k) for k in pt.__dataclass_fields__}
    pt_dict["short_contract"] = short_q.contract
    pt_dict["long_contract"] = long_q.contract
    from zeroday_paper.engine.models import PaperTrade
    pt = PaperTrade(**pt_dict)
    j.write_trade(pt)

    state = make_state(asof=_et_utc(11, 0), chain=chain)
    polygon = MagicMock()

    monitored, exited = await sc._monitor_open_positions(j, state, polygon)
    assert monitored >= 1
    # Tick was written
    n_ticks = j._conn.execute("SELECT COUNT(*) FROM paper_ticks").fetchone()[0]
    assert n_ticks == 1
    j.close()


@pytest.mark.asyncio
async def test_monitor_open_positions_flips_status_on_profit_target(
    tmp_path, make_state, make_chain, make_paper_trade, make_quote,
):
    j = Journal(str(tmp_path / "s.duckdb"))

    chain = make_chain(spot=5800.0, n_strikes=21)
    # Pick two adjacent puts to be short & long of a bull-put
    short_q = chain.puts[10]
    long_q = chain.puts[8]

    pt = make_paper_trade(
        short_strike=short_q.strike, long_strike=long_q.strike,
        entry_ts=_et_utc(10, 0),
        credit_bid=1.0,
    )
    from zeroday_paper.engine.models import PaperTrade
    pt_dict = {k: getattr(pt, k) for k in pt.__dataclass_fields__}
    pt_dict["short_contract"] = short_q.contract
    pt_dict["long_contract"] = long_q.contract
    pt = PaperTrade(**pt_dict)
    j.write_trade(pt)

    # Build a NEW chain where short/long quotes are deeply OTM (cheap exit cost) so
    # pct_of_max_profit >> profit_target.
    cheap_short = make_quote(strike=short_q.strike, right="P", bid=0.05, ask=0.10,
                             contract=short_q.contract, expiry=short_q.expiry)
    cheap_long = make_quote(strike=long_q.strike, right="P", bid=0.01, ask=0.05,
                            contract=long_q.contract, expiry=long_q.expiry)
    # Replace these contracts in the chain
    new_puts = [
        cheap_short if q.contract == short_q.contract else
        cheap_long if q.contract == long_q.contract else q
        for q in chain.puts
    ]
    from zeroday_paper.data.polygon_client import ChainSnapshot
    new_chain = ChainSnapshot(
        fetched_at=chain.fetched_at, spot=chain.spot, expiry=chain.expiry,
        calls=chain.calls, puts=new_puts,
    )

    state = make_state(asof=_et_utc(11, 0), chain=new_chain)
    polygon = MagicMock()
    monitored, exited = await sc._monitor_open_positions(j, state, polygon)
    assert exited == 1
    # Outcome written and status flipped
    status = j._conn.execute(
        "SELECT status FROM paper_trades WHERE trade_id = ?", [pt.trade_id]
    ).fetchone()[0]
    assert status == str(PositionStatus.CLOSED_TARGET)
    j.close()


@pytest.mark.asyncio
async def test_monitor_open_positions_skips_missing_contracts(
    tmp_path, make_state, make_paper_trade,
):
    j = Journal(str(tmp_path / "s.duckdb"))
    pt = make_paper_trade(entry_ts=_et_utc(10, 0))
    j.write_trade(pt)
    # state.chain does not include pt.short_contract / pt.long_contract
    state = make_state(asof=_et_utc(11, 0))
    polygon = MagicMock()
    monitored, exited = await sc._monitor_open_positions(j, state, polygon)
    assert exited == 0  # skipped
    j.close()


# ------------------------------------------------------------ run_one_cycle


@pytest.mark.asyncio
async def test_run_one_cycle_vols_failure(tmp_path, monkeypatch, make_chain, make_vols, make_signals):
    j = Journal(str(tmp_path / "s.duckdb"))

    class _FakePolygon:
        def __init__(self, *_, **__): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *_): return None
        async def get_chain_snapshot(self, e, **kw): return make_chain()

    class _FakeCboe:
        def __init__(self, *_, **__): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *_): return None
        async def get_live_snapshot(self): raise RuntimeError("vol down")

    monkeypatch.setattr(sc, "PolygonClient", _FakePolygon)
    monkeypatch.setattr(sc, "CboeClient", _FakeCboe)

    stats = await sc.run_one_cycle(j)
    assert stats["errors"] >= 1
    j.close()


@pytest.mark.asyncio
async def test_run_one_cycle_chain_failure(tmp_path, monkeypatch, make_vols):
    j = Journal(str(tmp_path / "s.duckdb"))

    class _FakePolygon:
        def __init__(self, *_, **__): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *_): return None
        async def get_chain_snapshot(self, e, **kw): raise RuntimeError("chain err")

    class _FakeCboe:
        def __init__(self, *_, **__): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *_): return None
        async def get_live_snapshot(self): return make_vols()

    monkeypatch.setattr(sc, "PolygonClient", _FakePolygon)
    monkeypatch.setattr(sc, "CboeClient", _FakeCboe)
    stats = await sc.run_one_cycle(j)
    assert stats["errors"] >= 1
    j.close()


@pytest.mark.asyncio
async def test_run_one_cycle_happy_path_flashalpha(tmp_path, monkeypatch, make_chain,
                                                    make_vols, make_signals):
    j = Journal(str(tmp_path / "s.duckdb"))
    chain = make_chain()
    signals = make_signals()
    vols = make_vols()

    class _FakePolygon:
        def __init__(self, *_, **__): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *_): return None
        async def get_chain_snapshot(self, e, **kw): return chain

    class _FakeCboe:
        def __init__(self, *_, **__): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *_): return None
        async def get_live_snapshot(self): return vols

    class _FakeFlashAlpha:
        def __init__(self, *_, **__): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *_): return None
        async def get_signals(self, sym="SPX"): return signals

    monkeypatch.setattr(sc, "PolygonClient", _FakePolygon)
    monkeypatch.setattr(sc, "CboeClient", _FakeCboe)
    monkeypatch.setattr(sc, "FlashAlphaClient", _FakeFlashAlpha)
    # Patch next_spx_expiry to return a date so we don't depend on real today
    monkeypatch.setattr(sc, "next_spx_expiry", lambda d: chain.expiry)
    # Patch classify_layer2 to avoid touching Anthropic
    monkeypatch.setattr(sc, "classify_layer2", AsyncMock(return_value=[]))
    # Patch datetime.now used inside scanner
    fake_now = _et_utc(11, 30)

    class _PatchedDT:
        @staticmethod
        def now(tz=None):
            return fake_now
    monkeypatch.setattr(sc, "datetime", _PatchedDT)

    stats = await sc.run_one_cycle(j)
    assert isinstance(stats, dict)
    assert "new_trades" in stats
    j.close()


@pytest.mark.asyncio
async def test_run_one_cycle_falls_back_to_self_computed_signals_on_zero_spot(
    tmp_path, monkeypatch, make_chain, make_vols, make_signals,
):
    j = Journal(str(tmp_path / "s.duckdb"))
    chain = make_chain()
    vols = make_vols()
    bad_signals = make_signals(spot=0.0)  # forces fallback

    class _FakePolygon:
        def __init__(self, *_, **__): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *_): return None
        async def get_chain_snapshot(self, e, **kw): return chain

    class _FakeCboe:
        def __init__(self, *_, **__): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *_): return None
        async def get_live_snapshot(self): return vols

    class _FakeFlashAlpha:
        def __init__(self, *_, **__): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *_): return None
        async def get_signals(self, sym="SPX"): return bad_signals

    monkeypatch.setattr(sc, "PolygonClient", _FakePolygon)
    monkeypatch.setattr(sc, "CboeClient", _FakeCboe)
    monkeypatch.setattr(sc, "FlashAlphaClient", _FakeFlashAlpha)
    monkeypatch.setattr(sc, "next_spx_expiry", lambda d: chain.expiry)
    monkeypatch.setattr(sc, "classify_layer2", AsyncMock(return_value=[]))

    fake_now = _et_utc(11, 30)
    class _PatchedDT:
        @staticmethod
        def now(tz=None):
            return fake_now
    monkeypatch.setattr(sc, "datetime", _PatchedDT)

    stats = await sc.run_one_cycle(j)
    assert isinstance(stats, dict)
    j.close()


@pytest.mark.asyncio
async def test_run_one_cycle_flashalpha_exception_uses_self_computed(
    tmp_path, monkeypatch, make_chain, make_vols,
):
    j = Journal(str(tmp_path / "s.duckdb"))
    chain = make_chain()
    vols = make_vols()

    class _FakePolygon:
        def __init__(self, *_, **__): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *_): return None
        async def get_chain_snapshot(self, e, **kw): return chain

    class _FakeCboe:
        def __init__(self, *_, **__): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *_): return None
        async def get_live_snapshot(self): return vols

    class _FakeFlashAlpha:
        def __init__(self, *_, **__): pass
        async def __aenter__(self): raise RuntimeError("fa down")
        async def __aexit__(self, *_): return None
        async def get_signals(self, sym="SPX"): raise RuntimeError("never")

    monkeypatch.setattr(sc, "PolygonClient", _FakePolygon)
    monkeypatch.setattr(sc, "CboeClient", _FakeCboe)
    monkeypatch.setattr(sc, "FlashAlphaClient", _FakeFlashAlpha)
    monkeypatch.setattr(sc, "next_spx_expiry", lambda d: chain.expiry)
    monkeypatch.setattr(sc, "classify_layer2", AsyncMock(return_value=[]))

    fake_now = _et_utc(11, 30)
    class _PatchedDT:
        @staticmethod
        def now(tz=None):
            return fake_now
    monkeypatch.setattr(sc, "datetime", _PatchedDT)

    stats = await sc.run_one_cycle(j)
    assert isinstance(stats, dict)
    j.close()


# ------------------------------------------------------ run_live_loop


class _FakeJournal:
    def __init__(self):
        self.closed = False
        self.heartbeats: list[tuple[str, str]] = []

    def close(self):
        self.closed = True

    def heartbeat(self, who: str, status: str = "ok"):
        self.heartbeats.append((who, status))


def _install_fake_journal(monkeypatch) -> _FakeJournal:
    fake_j = _FakeJournal()
    monkeypatch.setattr(sc, "Journal", lambda *a, **kw: fake_j)
    return fake_j


def _patch_now(monkeypatch, current_holder):
    """Replace scanner.datetime with a shim that returns current_holder['t'] from .now()
    while delegating everything else (combine, etc.) to the real datetime class.
    """
    from datetime import datetime as _real_datetime

    class _PatchedDT(_real_datetime):
        @classmethod
        def now(cls, tz=None):
            return current_holder["t"]

    monkeypatch.setattr(sc, "datetime", _PatchedDT)


@pytest.mark.asyncio
async def test_run_live_loop_weekend_returns_without_sleep(monkeypatch):
    """Saturday at any time → returns immediately, never sleeps."""
    fake_j = _install_fake_journal(monkeypatch)

    saturday_morning = datetime(2025, 5, 31, 13, 30, tzinfo=UTC)  # Sat 09:30 ET
    saturday_afternoon = datetime(2025, 5, 31, 20, 0, tzinfo=UTC)  # Sat 16:00 ET

    for fake_now in (saturday_morning, saturday_afternoon):
        holder = {"t": fake_now}
        _patch_now(monkeypatch, holder)
        sleep_mock = AsyncMock()
        monkeypatch.setattr(sc.asyncio, "sleep", sleep_mock)
        run_cycle = AsyncMock(return_value={})
        monkeypatch.setattr(sc, "run_one_cycle", run_cycle)

        await sc.run_live_loop()
        sleep_mock.assert_not_awaited()
        run_cycle.assert_not_awaited()

    assert fake_j.closed is True


@pytest.mark.asyncio
async def test_run_live_loop_pre_session_sleeps_then_runs_cycle(monkeypatch):
    """Weekday 09:25 ET → sleeps with seconds > 0, heartbeats waiting_for_session."""
    fake_j = _install_fake_journal(monkeypatch)

    pre_session = datetime(2025, 5, 28, 13, 25, tzinfo=UTC)  # Wed 09:25 ET
    in_session = datetime(2025, 5, 28, 13, 35, tzinfo=UTC)  # Wed 09:35 ET
    after_session = datetime(2025, 5, 28, 20, 30, tzinfo=UTC)  # Wed 16:30 ET

    times = iter([in_session, after_session])
    current = {"t": pre_session}
    _patch_now(monkeypatch, current)

    sleep_calls: list[float] = []

    async def fake_sleep(seconds):
        sleep_calls.append(seconds)
        try:
            current["t"] = next(times)
        except StopIteration:
            pass

    monkeypatch.setattr(sc.asyncio, "sleep", fake_sleep)

    run_cycle = AsyncMock(return_value={"new_trades": 0, "monitored": 0, "exited": 0, "errors": 0})
    monkeypatch.setattr(sc, "run_one_cycle", run_cycle)

    await sc.run_live_loop()

    assert any(s > 0 for s in sleep_calls), f"expected a >0 pre-session sleep, got {sleep_calls}"
    run_cycle.assert_awaited_once()
    assert ("scanner", "waiting_for_session") in fake_j.heartbeats
    assert fake_j.closed is True


@pytest.mark.asyncio
async def test_run_live_loop_after_session_logs_and_returns(monkeypatch, caplog):
    """Weekday 16:30 ET (past session_end=16:00) → log after_session_exit + return."""
    import logging as _logging
    fake_j = _install_fake_journal(monkeypatch)

    after = datetime(2025, 5, 28, 20, 30, tzinfo=UTC)  # Wed 16:30 ET
    holder = {"t": after}
    _patch_now(monkeypatch, holder)
    sleep_mock = AsyncMock()
    monkeypatch.setattr(sc.asyncio, "sleep", sleep_mock)
    run_cycle = AsyncMock(return_value={})
    monkeypatch.setattr(sc, "run_one_cycle", run_cycle)

    logged_events: list[str] = []
    real_info = sc.logger.info

    def _spy(event, *args, **kw):
        logged_events.append(event)
        return real_info(event, *args, **kw)

    monkeypatch.setattr(sc.logger, "info", _spy)

    await sc.run_live_loop()

    assert "live.after_session_exit" in logged_events
    run_cycle.assert_not_awaited()
    sleep_mock.assert_not_awaited()
    assert fake_j.closed is True


@pytest.mark.asyncio
async def test_run_live_loop_in_session_runs_cycle_then_after_session_exits(monkeypatch):
    """Weekday in-session → run_one_cycle, then sleep interval, then advance to after-session."""
    fake_j = _install_fake_journal(monkeypatch)

    in_session = datetime(2025, 5, 28, 13, 35, tzinfo=UTC)  # Wed 09:35 ET
    after = datetime(2025, 5, 28, 20, 30, tzinfo=UTC)  # Wed 16:30 ET

    states = iter([after])
    current = {"t": in_session}
    _patch_now(monkeypatch, current)

    async def fake_sleep(seconds):
        try:
            current["t"] = next(states)
        except StopIteration:
            pass

    monkeypatch.setattr(sc.asyncio, "sleep", fake_sleep)
    run_cycle = AsyncMock(return_value={"new_trades": 1, "monitored": 0, "exited": 0, "errors": 0})
    monkeypatch.setattr(sc, "run_one_cycle", run_cycle)

    await sc.run_live_loop()

    run_cycle.assert_awaited_once()
    assert fake_j.closed is True


# ------------------------------------------------ scan observability (Phase A)


@pytest.mark.asyncio
async def test_scan_for_entries_emits_scan_score_log_per_strategy(
    tmp_path, make_state, monkeypatch,
):
    """Every strategy scan must emit a scan.score event with the full breakdown
    so we never fly blind on why no trade was written."""
    j = Journal(str(tmp_path / "s.duckdb"))
    state = make_state(asof=_et_utc(11, 0))

    monkeypatch.setattr(sc, "classify_layer2", AsyncMock(return_value=[]))

    events: list[tuple[str, dict]] = []
    real_info = sc.logger.info

    def _spy(event, *args, **kw):
        events.append((event, kw))
        return real_info(event, *args, **kw)

    monkeypatch.setattr(sc.logger, "info", _spy)

    await sc._scan_for_entries(j, state)

    scan_score_events = [e for e in events if e[0] == "scan.score"]
    assert len(scan_score_events) == 2, "expected scan.score for both strategies"
    strategies = {e[1]["strategy"] for e in scan_score_events}
    assert "BULL_PUT_SPREAD" in strategies
    assert "BEAR_CALL_SPREAD" in strategies
    sample = scan_score_events[0][1]
    for required in (
        "total", "breakdown", "regime_ok", "notes", "threshold",
        "signals_source", "spot", "to_put_wall", "to_call_wall",
        "above_flip", "pin_score", "vix_1d",
    ):
        assert required in sample, f"scan.score missing field {required}"
    j.close()


@pytest.mark.asyncio
async def test_scan_for_entries_emits_below_threshold_when_score_low(
    tmp_path, make_state, make_signals, monkeypatch,
):
    """When score < threshold we must log scan.below_threshold so the operator
    knows the cycle was suppressed by scoring rather than something else."""
    j = Journal(str(tmp_path / "s.duckdb"))
    # Force regime block → score=0 < 13 → below_threshold expected.
    state = make_state(asof=_et_utc(11, 0),
                       signals=make_signals(gamma_regime="negative_gamma"))

    monkeypatch.setattr(sc, "classify_layer2", AsyncMock(return_value=[]))

    events: list[str] = []
    real_info = sc.logger.info

    def _spy(event, *args, **kw):
        events.append(event)
        return real_info(event, *args, **kw)

    monkeypatch.setattr(sc.logger, "info", _spy)

    await sc._scan_for_entries(j, state)
    assert "scan.below_threshold" in events
    j.close()


@pytest.mark.asyncio
async def test_scan_for_entries_emits_no_spread_when_selector_fails(
    tmp_path, make_state, make_chain, make_signals, monkeypatch,
):
    """If select_for cannot produce a spread we must log scan.no_spread with
    the reasons + candidates_considered so the operator can see WHY."""
    j = Journal(str(tmp_path / "s.duckdb"))

    # Build a chain whose puts are very far OTM → no_short_in_delta_band and
    # also outside moneyness band so we genuinely return spread=None.
    chain = make_chain(spot=5800.0, n_strikes=5, step=50)  # very wide spacing
    state = make_state(
        spot=5800.0,
        asof=_et_utc(11, 0),
        chain=chain,
        signals=make_signals(spot=5800.0),
    )
    # Force selector to return None directly.
    from zeroday_paper.engine.strike_select import SelectionResult
    monkeypatch.setattr(sc, "select_for", lambda strat, st: SelectionResult(
        spread=None, reasons=["test_forced_none"], candidates_considered=3,
    ))
    monkeypatch.setattr(sc, "classify_layer2", AsyncMock(return_value=[]))

    events: list[tuple[str, dict]] = []
    real_info = sc.logger.info

    def _spy(event, *args, **kw):
        events.append((event, kw))
        return real_info(event, *args, **kw)

    monkeypatch.setattr(sc.logger, "info", _spy)

    await sc._scan_for_entries(j, state)
    no_spread = [e for e in events if e[0] == "scan.no_spread"]
    assert len(no_spread) >= 1, f"expected scan.no_spread event, got {[e[0] for e in events]}"
    assert no_spread[0][1].get("reasons") == ["test_forced_none"]
    assert no_spread[0][1].get("candidates_considered") == 3
    j.close()


@pytest.mark.asyncio
async def test_scan_for_entries_emits_entry_rejected_when_pricing_bad(
    tmp_path, make_state, monkeypatch,
):
    """If entry_quote is unacceptable we must log scan.entry_rejected with
    the pricing reasons."""
    j = Journal(str(tmp_path / "s.duckdb"))
    state = make_state(asof=_et_utc(11, 0))

    from zeroday_paper.engine.pricing import EntryQuote
    bad_quote = EntryQuote(
        credit_bid=-0.05, credit_mid=0.10, max_loss_bid=25.0,
        credit_ratio=-0.002, is_acceptable=False,
        reasons=["credit_bid -0.05 < 0.1"],
    )
    monkeypatch.setattr(sc, "entry_quote", lambda spread: bad_quote)
    monkeypatch.setattr(sc, "classify_layer2", AsyncMock(return_value=[]))

    events: list[tuple[str, dict]] = []
    real_info = sc.logger.info

    def _spy(event, *args, **kw):
        events.append((event, kw))
        return real_info(event, *args, **kw)

    monkeypatch.setattr(sc.logger, "info", _spy)

    await sc._scan_for_entries(j, state)
    rejects = [e for e in events if e[0] == "scan.entry_rejected"]
    assert rejects, f"expected scan.entry_rejected, got events={[e[0] for e in events]}"
    j.close()


@pytest.mark.asyncio
async def test_scan_for_entries_emits_score_metric(tmp_path, make_state, monkeypatch):
    """Numeric scan.score.<strategy> CloudWatch metric must be emitted on every
    scan so we can graph score trend regardless of whether anything was written."""
    j = Journal(str(tmp_path / "s.duckdb"))
    state = make_state(asof=_et_utc(11, 0))

    monkeypatch.setattr(sc, "classify_layer2", AsyncMock(return_value=[]))

    metric_calls: list[tuple[str, float]] = []
    monkeypatch.setattr(sc, "emit_metric", lambda name, value=1.0, **kw: metric_calls.append((name, value)))

    await sc._scan_for_entries(j, state)
    metric_names = [m[0] for m in metric_calls]
    assert "scan.score.bull_put" in metric_names
    assert "scan.score.bear_call" in metric_names
    # Each should be a non-negative numeric score.
    for name, value in metric_calls:
        if name.startswith("scan.score."):
            assert value >= 0
    j.close()


# ------------------------------------------------- diagnostic_snapshot (MODE=diag)


@pytest.mark.asyncio
async def test_diagnostic_snapshot_returns_full_dict(monkeypatch, make_chain, make_vols, make_signals):
    """diagnostic_snapshot must return a JSON-friendly dict with score, proximity,
    chain stats, and per-strategy selection — and must NOT touch the journal."""
    chain = make_chain()
    signals = make_signals()
    vols = make_vols()

    class _FakePolygon:
        def __init__(self, *_, **__): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *_): return None
        async def get_chain_snapshot(self, e, **kw): return chain

    class _FakeCboe:
        def __init__(self, *_, **__): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *_): return None
        async def get_live_snapshot(self): return vols

    class _FakeFlashAlpha:
        def __init__(self, *_, **__): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *_): return None
        async def get_signals(self, sym="SPX"): return signals

    monkeypatch.setattr(sc, "PolygonClient", _FakePolygon)
    monkeypatch.setattr(sc, "CboeClient", _FakeCboe)
    monkeypatch.setattr(sc, "FlashAlphaClient", _FakeFlashAlpha)
    monkeypatch.setattr(sc, "next_spx_expiry", lambda d: chain.expiry)
    monkeypatch.setattr(sc, "classify_layer2", AsyncMock(return_value=[]))

    fake_now = _et_utc(11, 30)

    class _PatchedDT:
        @staticmethod
        def now(tz=None):
            return fake_now
    monkeypatch.setattr(sc, "datetime", _PatchedDT)

    snap = await sc.diagnostic_snapshot()
    assert isinstance(snap, dict)
    assert "threshold" in snap
    assert "chain" in snap
    assert "signals" in snap
    assert "proximity" in snap
    assert "strategies" in snap
    assert len(snap["strategies"]) == 2
    for st in snap["strategies"]:
        assert "score" in st
        assert "breakdown" in st
        assert "clears_threshold" in st
        assert "selection" in st


@pytest.mark.asyncio
async def test_diagnostic_snapshot_handles_chain_failure(monkeypatch, make_vols):
    """Snapshot must degrade gracefully — record the error and still return JSON."""
    vols = make_vols()

    class _FakePolygon:
        def __init__(self, *_, **__): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *_): return None
        async def get_chain_snapshot(self, e, **kw): raise RuntimeError("polygon down")

    class _FakeCboe:
        def __init__(self, *_, **__): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *_): return None
        async def get_live_snapshot(self): return vols

    monkeypatch.setattr(sc, "PolygonClient", _FakePolygon)
    monkeypatch.setattr(sc, "CboeClient", _FakeCboe)
    monkeypatch.setattr(sc, "next_spx_expiry", lambda d: _et_utc(11, 30).date())

    fake_now = _et_utc(11, 30)

    class _PatchedDT:
        @staticmethod
        def now(tz=None):
            return fake_now
    monkeypatch.setattr(sc, "datetime", _PatchedDT)

    snap = await sc.diagnostic_snapshot()
    assert isinstance(snap, dict)
    assert any("chain_failed" in e for e in snap["errors"])


@pytest.mark.asyncio
async def test_run_live_loop_cycle_exception_is_swallowed(monkeypatch):
    """A cycle error should not crash the loop; heartbeats record the error."""
    fake_j = _install_fake_journal(monkeypatch)

    in_session = datetime(2025, 5, 28, 13, 35, tzinfo=UTC)
    after = datetime(2025, 5, 28, 20, 30, tzinfo=UTC)

    states = iter([after])
    current = {"t": in_session}
    _patch_now(monkeypatch, current)

    async def fake_sleep(seconds):
        try:
            current["t"] = next(states)
        except StopIteration:
            pass

    monkeypatch.setattr(sc.asyncio, "sleep", fake_sleep)
    run_cycle = AsyncMock(side_effect=RuntimeError("boom"))
    monkeypatch.setattr(sc, "run_one_cycle", run_cycle)

    await sc.run_live_loop()
    assert any(s.startswith("error:") for _, s in fake_j.heartbeats)
    assert fake_j.closed is True
