"""The receipts' GPU peak must include intra-step transients.

Defect pinned here: ``gpuwm/prepared_single_domain_forecast.py`` (and
its domain-tree twin) sampled ``cudaMemGetInfo`` from a ``record_memory``
closure that fired only in the step-boundary callbacks -- the history
handler and the per-period progress callback.  ``execute_experiment``
trims the CuPy pool per STEP op and again at period commit BEFORE the
progress callback runs, so every sample landed after the intra-step
transient working set had been released, and the receipt's
``gpu_peak_used_bytes_observed`` under-reported the true high-water
mark (19.41 GiB reported against 22.34 GiB observed on the four-domain
tree shape, -13%).  A receipt that understates VRAM invites launching a
shape that demotes or OOMs.

These tests exercise the replacement accounting path CPU-side with
injected probes; no GPU is touched.
"""

from __future__ import annotations

import ast
from pathlib import Path
import threading
import time

import pytest

from gpuwm.core.gpu_mem_watch import (
    DEFAULT_INTERVAL_SECONDS,
    GpuPeakMemoryWatcher,
    MemoryProbe,
)


def _probe(name, read, scope="test scope"):
    return MemoryProbe(name=name, scope=scope, read=read)


class _SpikingSource:
    """A byte counter that spikes between two 'step boundary' reads."""

    BASELINE = 100
    SPIKE = 1000

    def __init__(self):
        self.value = self.BASELINE
        self.spike_seen = threading.Event()

    def read(self):
        value = self.value
        if value >= self.SPIKE:
            self.spike_seen.set()
        return value


def test_intra_boundary_spike_reaches_the_reported_peak():
    """The old boundary-only scheme misses the spike; the watcher must not."""
    source = _SpikingSource()
    watcher = GpuPeakMemoryWatcher(
        [_probe("device", source.read)], interval_seconds=0.001)
    boundary_samples = []
    watcher.start()
    try:
        # Step boundary N: transient not yet allocated.
        boundary_samples.append(source.read())
        # Intra-step transient appears...
        source.value = source.SPIKE
        assert source.spike_seen.wait(timeout=10.0), \
            "background sampler never observed the transient"
        # ...and is released before the next boundary callback fires.
        source.value = source.BASELINE
        boundary_samples.append(source.read())
    finally:
        watcher.stop()
    # Premise of the defect: boundary-only sampling cannot see the spike.
    assert max(boundary_samples) == source.BASELINE
    # The fix: the receipt-facing peak includes it.
    assert watcher.peak_bytes("device") == source.SPIKE


def test_explicit_samples_fold_into_the_peak_without_a_thread():
    values = iter([7, 42, 3])
    watcher = GpuPeakMemoryWatcher([_probe("device", lambda: next(values))])
    watcher.sample()
    watcher.sample()
    watcher.sample()
    assert watcher.peak_bytes("device") == 42
    watcher.stop()  # never started: stop must still be safe


def test_peak_defaults_to_zero_and_summary_says_no_samples():
    watcher = GpuPeakMemoryWatcher([_probe("device", lambda: 5)])
    assert watcher.peak_bytes("device") == 0
    assert watcher.summary()["probes"]["device"]["samples"] == 0


def test_unknown_probe_name_is_refused():
    watcher = GpuPeakMemoryWatcher([_probe("device", lambda: 5)])
    with pytest.raises(KeyError):
        watcher.peak_bytes("nonexistent")


def test_explicit_sample_propagates_probe_errors():
    """Boundary samples keep the old fail-loud behavior."""

    def broken():
        raise RuntimeError("probe exploded")

    watcher = GpuPeakMemoryWatcher(
        [_probe("good", lambda: 11), _probe("bad", broken)])
    with pytest.raises(RuntimeError, match="probe exploded"):
        watcher.sample()
    # The healthy probe was still folded before the error propagated.
    assert watcher.peak_bytes("good") == 11


def test_background_probe_error_is_recorded_not_raised():
    """A mid-run sampling hiccup must not fail the forecast, only the
    receipt's claim to completeness."""
    calls = {"n": 0}
    seen_error = threading.Event()

    def flaky():
        calls["n"] += 1
        if calls["n"] > 1:
            seen_error.set()
            raise RuntimeError("device fell over")
        return 500

    good_seen = threading.Event()

    def good():
        good_seen.set()
        return 200

    watcher = GpuPeakMemoryWatcher(
        [_probe("flaky", flaky), _probe("good", good)],
        interval_seconds=0.001)
    watcher.start()
    try:
        assert seen_error.wait(timeout=10.0)
        good_seen.clear()
        # The healthy probe keeps sampling after its sibling broke.
        assert good_seen.wait(timeout=10.0)
    finally:
        watcher.stop()
    assert watcher.peak_bytes("flaky") == 500
    assert watcher.peak_bytes("good") == 200
    summary = watcher.summary()
    assert "device fell over" in summary["probes"]["flaky"]["error"]
    assert summary["probes"]["good"]["error"] is None


def test_summary_labels_every_probe_scope_honestly():
    watcher = GpuPeakMemoryWatcher([
        _probe("device", lambda: 10, scope="device-wide-ish"),
        _probe("pool", lambda: 4, scope="in-process pool"),
    ])
    watcher.sample()
    summary = watcher.summary()
    assert summary["mechanism"] == (
        "background-polling-thread+boundary-samples")
    assert summary["interval_seconds"] == DEFAULT_INTERVAL_SECONDS
    assert summary["probes"]["device"]["scope"] == "device-wide-ish"
    assert summary["probes"]["pool"]["scope"] == "in-process pool"
    assert summary["probes"]["device"]["peak_bytes"] == 10
    assert summary["probes"]["pool"]["peak_bytes"] == 4
    assert summary["probes"]["device"]["samples"] == 1


def test_stop_is_idempotent_and_sampling_still_allowed_after_stop():
    values = {"v": 1}
    watcher = GpuPeakMemoryWatcher(
        [_probe("device", lambda: values["v"])], interval_seconds=0.001)
    watcher.start()
    watcher.stop()
    watcher.stop()
    values["v"] = 99
    watcher.sample()  # the end-of-run sample lands after stop()
    assert watcher.peak_bytes("device") == 99


def test_double_start_is_refused():
    watcher = GpuPeakMemoryWatcher(
        [_probe("device", lambda: 1)], interval_seconds=0.001)
    watcher.start()
    try:
        with pytest.raises(RuntimeError):
            watcher.start()
    finally:
        watcher.stop()


def test_constructor_refuses_degenerate_configurations():
    with pytest.raises(ValueError):
        GpuPeakMemoryWatcher([])
    with pytest.raises(ValueError):
        GpuPeakMemoryWatcher(
            [_probe("device", lambda: 1)], interval_seconds=0.0)
    with pytest.raises(ValueError):
        GpuPeakMemoryWatcher(
            [_probe("dup", lambda: 1), _probe("dup", lambda: 2)])


def test_default_interval_resolves_intra_step_transients():
    """Root periods run for seconds on nested shapes; the default poll
    cadence must be well inside that."""
    assert 0.0 < DEFAULT_INTERVAL_SECONDS <= 0.1


def test_default_cupy_probes_cover_device_and_pool_views(monkeypatch):
    import sys
    from unittest import mock

    from gpuwm.core import gpu_mem_watch

    fake_cupy = mock.MagicMock()
    fake_cupy.cuda.runtime.memGetInfo.return_value = (30, 100)
    pool = fake_cupy.get_default_memory_pool.return_value
    pool.total_bytes.return_value = 60
    pool.used_bytes.return_value = 45
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)

    probes = {probe.name: probe for probe in
              gpu_mem_watch.default_cupy_probes()}
    assert set(probes) == {
        "cuda_device_used", "cupy_pool_total", "cupy_pool_used"}
    assert probes["cuda_device_used"].read() == 70
    assert probes["cupy_pool_total"].read() == 60
    assert probes["cupy_pool_used"].read() == 45
    # Honest labels: the memGetInfo view is NOT the whole card on WDDM
    # (gpuwm.core.preflight.device_wide_used_bytes documents why), and
    # the pool views are in-process only.
    assert "WDDM" in probes["cuda_device_used"].scope
    assert "in-process" in probes["cupy_pool_total"].scope
    assert "in-process" in probes["cupy_pool_used"].scope


# ---------------------------------------------------------------------------
# Wiring pins: the runners must actually use the watcher.
# ---------------------------------------------------------------------------

def _runner_sources():
    import gpuwm.prepared_domain_tree_forecast as tree_runner
    import gpuwm.prepared_single_domain_forecast as single_runner

    return {
        "single": (Path(single_runner.__file__), "run_prepared_forecast"),
        "tree": (Path(tree_runner.__file__), "run_prepared_tree"),
    }


def _function_node(path: Path, name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name)


@pytest.mark.parametrize("runner_key", ["single", "tree"])
def test_runner_wires_watcher_started_and_stopped_in_finally(runner_key):
    path, func_name = _runner_sources()[runner_key]
    func = _function_node(path, func_name)
    constructed = [
        node for node in ast.walk(func)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "GpuPeakMemoryWatcher"]
    assert constructed, f"{func_name} does not construct GpuPeakMemoryWatcher"
    started = [
        node for node in ast.walk(func)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "start"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "memory_watch"]
    assert started, f"{func_name} never starts the memory watcher"
    stopped_in_finally = [
        node for node in ast.walk(func)
        if isinstance(node, ast.Try)
        and any(isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "stop"
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id == "memory_watch"
                for statement in node.finalbody
                for call in ast.walk(statement))]
    assert stopped_in_finally, (
        f"{func_name} must stop the memory watcher in a finally block")


@pytest.mark.parametrize("runner_key", ["single", "tree"])
def test_runner_boundary_only_sampler_is_gone(runner_key):
    path, _func_name = _runner_sources()[runner_key]
    tree = ast.parse(path.read_text(encoding="utf-8"))
    leftovers = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "record_memory"]
    assert not leftovers, (
        "the boundary-only record_memory sampler must not survive; it is "
        "the under-reporting mechanism this change removes")


@pytest.mark.parametrize("runner_key", ["single", "tree"])
def test_runner_receipt_memory_section_labels_the_measurement(runner_key):
    path, func_name = _runner_sources()[runner_key]
    func = _function_node(path, func_name)
    memory_dicts = [
        node for node in ast.walk(func)
        if isinstance(node, ast.Dict)
        and any(isinstance(key, ast.Constant)
                and key.value == "gpu_peak_used_bytes_observed"
                for key in node.keys)
        and any(isinstance(key, ast.Constant)
                and key.value == "gpu_peak_sampling"
                for key in node.keys)
        and any(isinstance(key, ast.Constant)
                and key.value == "cupy_pool_peak_total_bytes_observed"
                for key in node.keys)]
    assert memory_dicts, (
        f"{func_name}'s receipt memory section must keep the existing keys "
        "and add the gpu_peak_sampling provenance block")
