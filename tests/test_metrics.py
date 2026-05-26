"""CloudWatch metrics emitter — fully mocked, no real AWS calls.

The autouse `_silence_metrics` fixture in conftest replaces `metrics.emit` and
`metrics._get_client` so accidental emissions never reach boto3. Tests here
re-bind those attributes back to the real source via importlib.reload to
exercise the underlying code.
"""

from __future__ import annotations

import importlib
import sys
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def metrics_module():
    """Reload metrics so we exercise the real emit/_get_client implementations."""
    from zeroday_paper import metrics as m
    importlib.reload(m)
    m._client = None  # reset module state
    m._disabled = False
    yield m
    # restore reset state for next reload-using test
    importlib.reload(m)


def test_namespace_matches_alarm_expectation(metrics_module):
    assert metrics_module.NAMESPACE == "zeroday/paper"


def test_emit_uses_namespace_metric_name_value_unit(metrics_module, monkeypatch):
    boto_client = MagicMock()
    monkeypatch.setattr(metrics_module, "_get_client", lambda: boto_client)

    metrics_module.emit("cycle.complete", 1.0)

    boto_client.put_metric_data.assert_called_once()
    kwargs = boto_client.put_metric_data.call_args.kwargs
    assert kwargs["Namespace"] == "zeroday/paper"
    md = kwargs["MetricData"][0]
    assert md["MetricName"] == "cycle.complete"
    assert md["Value"] == 1.0
    assert md["Unit"] == "Count"
    assert "Dimensions" not in md


def test_emit_includes_dimensions(metrics_module, monkeypatch):
    boto_client = MagicMock()
    monkeypatch.setattr(metrics_module, "_get_client", lambda: boto_client)
    metrics_module.emit("trade.written", 2.0, strategy="bull_put")
    md = boto_client.put_metric_data.call_args.kwargs["MetricData"][0]
    assert {"Name": "strategy", "Value": "bull_put"} in md["Dimensions"]


def test_emit_swallows_client_exceptions(metrics_module, monkeypatch):
    boto_client = MagicMock()
    boto_client.put_metric_data.side_effect = RuntimeError("aws down")
    monkeypatch.setattr(metrics_module, "_get_client", lambda: boto_client)
    metrics_module.emit("position.exit", 1.0)  # must not raise


def test_emit_skips_when_client_is_none(metrics_module, monkeypatch):
    monkeypatch.setattr(metrics_module, "_get_client", lambda: None)
    metrics_module.emit("anything", 5.0)  # silent no-op


def test_get_client_handles_missing_boto3(metrics_module):
    real_boto3 = sys.modules.pop("boto3", None)
    sys.modules["boto3"] = None  # type: ignore[assignment]
    try:
        assert metrics_module._get_client() is None
        assert metrics_module._disabled is True
        # second call hits the disabled short-circuit
        assert metrics_module._get_client() is None
    finally:
        if real_boto3 is not None:
            sys.modules["boto3"] = real_boto3
        else:
            sys.modules.pop("boto3", None)


def test_get_client_handles_init_failure(metrics_module):
    class _FakeBoto:
        def client(self, *a, **kw):
            raise RuntimeError("no creds")

    sys.modules["boto3"] = _FakeBoto()  # type: ignore[assignment]
    try:
        assert metrics_module._get_client() is None
        assert metrics_module._disabled is True
    finally:
        sys.modules.pop("boto3", None)


def test_get_client_caches(metrics_module):
    created = []

    class _FakeBoto:
        def client(self, name, region_name=None):
            created.append((name, region_name))
            return MagicMock()

    sys.modules["boto3"] = _FakeBoto()  # type: ignore[assignment]
    try:
        c1 = metrics_module._get_client()
        c2 = metrics_module._get_client()
        assert c1 is c2
        assert len(created) == 1
        assert created[0][0] == "cloudwatch"
    finally:
        sys.modules.pop("boto3", None)


@pytest.mark.parametrize(
    "metric_name",
    ["cycle.complete", "trade.written", "position.exit", "cycle.errors"],
)
def test_known_metric_names_present_in_scanner(metric_name):
    """These names are the contract with the ScannerSilentAlarm dashboard."""
    from zeroday_paper.engine import scanner
    with open(scanner.__file__) as f:
        src = f.read()
    assert metric_name in src, f"scanner.py must still emit {metric_name}"


def test_run_one_cycle_emits_cycle_complete_metric(
    monkeypatch, tmp_path, make_chain, make_vols, make_signals,
):
    """End-to-end: a happy cycle should call emit('cycle.complete', 1.0)."""
    from zeroday_paper.engine import scanner as sc
    from zeroday_paper.engine.journal import Journal

    j = Journal(str(tmp_path / "m.duckdb"))
    chain = make_chain()
    signals = make_signals()
    vols = make_vols()

    calls: list[tuple[str, float]] = []

    def _spy(name, value=1.0, *a, **kw):
        calls.append((name, value))

    monkeypatch.setattr(sc, "emit_metric", _spy)

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

    from unittest.mock import AsyncMock
    monkeypatch.setattr(sc, "classify_layer2", AsyncMock(return_value=[]))

    import asyncio
    asyncio.run(sc.run_one_cycle(j))
    assert any(name == "cycle.complete" for name, _ in calls), calls
    j.close()
