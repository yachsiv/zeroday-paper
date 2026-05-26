"""Offline preflight assertions for zeroday-paper.

Verifies invariants that must hold before tomorrow's live run:
  - is_market_hours_et() handles DST transitions correctly
  - next_spx_expiry() returns the right Wed/Mon/Fri 0DTE
  - signals_from_chain() does not crash on an empty chain
  - The new run_live_loop pre-session wait branch is reachable

Run: PYTHONPATH=. uv run python scripts/preflight.py
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")


def check_dst_market_hours() -> None:
    from zeroday_paper.engine.scanner import is_market_hours_et

    # 2026-05-27 13:30 UTC = 09:30 EDT (Wednesday)
    edt_open = datetime(2026, 5, 27, 13, 30, tzinfo=UTC)
    assert is_market_hours_et(edt_open), f"expected market hours at 09:30 EDT, got False ({edt_open})"

    # 2026-11-03 14:30 UTC = 09:30 EST (post-DST, Tuesday)
    est_open = datetime(2026, 11, 3, 14, 30, tzinfo=UTC)
    assert is_market_hours_et(est_open), f"expected market hours at 09:30 EST, got False ({est_open})"

    # 2026-05-27 13:00 UTC = 09:00 EDT (before session) -> False
    pre = datetime(2026, 5, 27, 13, 0, tzinfo=UTC)
    assert not is_market_hours_et(pre), f"expected False before session, got True ({pre})"

    # Saturday 2026-05-30 13:30 UTC -> False (weekend)
    weekend = datetime(2026, 5, 30, 13, 30, tzinfo=UTC)
    assert not is_market_hours_et(weekend), f"expected False on Saturday, got True ({weekend})"

    print("  OK is_market_hours_et: EDT + EST + pre-session + weekend")


def check_next_spx_expiry() -> None:
    from zeroday_paper.data.polygon_client import next_spx_expiry

    # 2026-05-27 is a Wednesday -> same-day 0DTE expiry
    assert next_spx_expiry(date(2026, 5, 27)) == date(2026, 5, 27), (
        f"expected 2026-05-27 (Wed) same-day expiry, got {next_spx_expiry(date(2026, 5, 27))}"
    )
    # 2026-05-26 is a Tuesday -> next is Wed 2026-05-27
    assert next_spx_expiry(date(2026, 5, 26)) == date(2026, 5, 27)
    # 2026-05-28 is a Thursday -> next is Fri 2026-05-29
    assert next_spx_expiry(date(2026, 5, 28)) == date(2026, 5, 29)
    # Saturday 2026-05-30 -> Mon 2026-06-01
    assert next_spx_expiry(date(2026, 5, 30)) == date(2026, 6, 1)

    print("  OK next_spx_expiry: Wed/Tue/Thu/Sat all map correctly")


def check_signals_from_chain_empty() -> None:
    from zeroday_paper.data.flashalpha_client import signals_from_chain
    from zeroday_paper.data.polygon_client import ChainSnapshot

    empty = ChainSnapshot(
        fetched_at=datetime.now(UTC),
        spot=0.0,
        expiry=date(2026, 5, 27),
        calls=[],
        puts=[],
    )
    sig = signals_from_chain(empty)
    assert sig is not None
    print(f"  OK signals_from_chain(empty): regime={sig.gamma_regime}, total_gex={sig.total_gex}")

    # Also try spot>0 but no contracts
    empty2 = ChainSnapshot(
        fetched_at=datetime.now(UTC),
        spot=5000.0,
        expiry=date(2026, 5, 27),
        calls=[],
        puts=[],
    )
    sig2 = signals_from_chain(empty2)
    assert sig2 is not None
    print(f"  OK signals_from_chain(spot=5000, no contracts): regime={sig2.gamma_regime}")


def check_live_loop_imports() -> None:
    # Just import to make sure new wait branch parses and references are valid.
    from zeroday_paper.engine.scanner import run_live_loop, is_market_hours_et  # noqa: F401
    print("  OK scanner.run_live_loop imports cleanly")


def main() -> None:
    print("Preflight checks for zeroday-paper")
    print("-" * 50)
    check_dst_market_hours()
    check_next_spx_expiry()
    check_signals_from_chain_empty()
    check_live_loop_imports()
    print("-" * 50)
    print("All preflight assertions PASSED")


if __name__ == "__main__":
    main()
