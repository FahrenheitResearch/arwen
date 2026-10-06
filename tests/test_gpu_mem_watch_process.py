"""The receipt says WHOSE bytes the card carried.

Defect pinned here: every GPU-memory view the receipts carried was
either in-process (the CuPy pool) or device-wide (``cudaMemGetInfo``,
NVML ``memory.used``), so a run could not be told apart from a card
that filled up underneath it.  A 5 h pair of nested runs on a 32 GB
card was read as "one leg climbs to 29.8 GB and runs 2.3x slower" when the
leg's own process sat at 8.2-8.6 GB for the whole leg and a second
process held 15-17.8 GB of the same card, running kernels on it, for
98% of the leg's wall.  The per-process NVML views and the watcher's
time-above-zero accounting are what let the next receipt say so.

CPU-side throughout: the nvidia-smi reader is injected.
"""

from __future__ import annotations

import ast
from pathlib import Path
import threading
import time

import pytest

from gpuwm.core.gpu_mem_watch import (
    GpuPeakMemoryWatcher,
    MemoryProbe,
    NVIDIA_SMI_INTERVAL_SECONDS,
    NVIDIA_SMI_TIMEOUT_SECONDS,
    ProcessMemoryUnavailable,
    TransientProbeError,
    nvidia_smi_process_probes,
    parse_nvidia_smi_views,
    process_memory_receipt,
)

_MIB = 1024 ** 2
_CARD = "GPU-aaaa"
_OTHER_CARD = "GPU-bbbb"


def _smi(apps: str, gpus: str):
    """An nvidia-smi stand-in answering the two queries the probes make."""
    calls = []

    def run(arguments):
        calls.append(arguments[0])
        if arguments[0].startswith("--query-compute-apps"):
            return apps
        assert arguments[0].startswith("--query-gpu")
        return gpus

    run.calls = calls
    return run


def test_split_attributes_bytes_to_this_process_and_the_others():
    reading = parse_nvidia_smi_views(
        f"2560995, 8194, {_CARD}\n2563817, 15214, {_CARD}\n",
        f"{_CARD}, 23409\n", pid=2560995)
    assert reading.this_process_bytes == 8194 * _MIB
    assert reading.other_processes_bytes == 15214 * _MIB
    assert reading.device_used_bytes == 23409 * _MIB
    assert reading.other_pids == (2563817,)
    assert reading.shared


def test_a_card_to_ourselves_reads_as_unshared():
    reading = parse_nvidia_smi_views(
        f"4242, 8574, {_CARD}\n", f"{_CARD}, 8574\n", pid=4242)
    assert reading.other_processes_bytes == 0
    assert reading.other_pids == ()
    assert not reading.shared


def test_other_cards_processes_are_not_counted_as_sharing():
    reading = parse_nvidia_smi_views(
        f"4242, 8000, {_CARD}\n999, 30000, {_OTHER_CARD}\n",
        f"{_OTHER_CARD}, 30000\n{_CARD}, 8000\n", pid=4242)
    assert reading.this_process_bytes == 8000 * _MIB
    assert reading.other_processes_bytes == 0
    assert reading.device_used_bytes == 8000 * _MIB


def test_before_our_context_exists_the_first_card_still_reports_residents():
    reading = parse_nvidia_smi_views(
        f"999, 17654, {_CARD}\n", f"{_CARD}, 17654\n", pid=4242)
    assert reading.this_process_bytes == 0
    assert reading.other_processes_bytes == 17654 * _MIB
    assert reading.shared


def test_wddm_not_available_is_a_named_refusal_not_a_zero():
    with pytest.raises(ProcessMemoryUnavailable, match="N/A"):
        parse_nvidia_smi_views(
            f"4242, [N/A], {_CARD}\n", f"{_CARD}, 8574\n", pid=4242)
    with pytest.raises(ProcessMemoryUnavailable):
        parse_nvidia_smi_views("", "", pid=4242)


def test_probes_share_one_nvidia_smi_pass_and_are_never_strict():
    run = _smi(f"1, 8194, {_CARD}\n2, 15214, {_CARD}\n", f"{_CARD}, 23409\n")
    probes = {p.name: p for p in nvidia_smi_process_probes(pid=1, run=run)}
    assert set(probes) == {"nvml_this_process_used",
                           "nvml_other_processes_used", "nvml_device_used"}
    assert all(not p.strict for p in probes.values())
    assert all(p.background_only for p in probes.values())
    assert all(p.interval_seconds == NVIDIA_SMI_INTERVAL_SECONDS
               for p in probes.values())
    assert probes["nvml_this_process_used"].read() == 8194 * _MIB
    assert probes["nvml_other_processes_used"].read() == 15214 * _MIB
    assert probes["nvml_device_used"].read() == 23409 * _MIB
    # Three reads, one snapshot: two nvidia-smi calls, not six.
    assert run.calls == ["--query-compute-apps=pid,used_memory,gpu_uuid",
                         "--query-gpu=uuid,memory.used"]
    assert "OTHER" in probes["nvml_other_processes_used"].scope
    assert "THIS" in probes["nvml_this_process_used"].scope


def test_a_non_strict_probe_error_is_recorded_by_the_boundary_sample():
    """A host without nvidia-smi must not lose its forecast to a receipt
    row; the strict pool views still fail loud."""

    def broken():
        raise ProcessMemoryUnavailable("no nvidia-smi on PATH")

    watcher = GpuPeakMemoryWatcher([
        MemoryProbe(name="pool", scope="s", read=lambda: 11),
        MemoryProbe(name="nvml", scope="s", read=broken, strict=False),
    ])
    watcher.sample()  # does not raise
    summary = watcher.summary()["probes"]
    assert summary["pool"]["peak_bytes"] == 11
    assert summary["nvml"]["samples"] == 0
    assert "no nvidia-smi" in summary["nvml"]["error"]
    assert watcher.peak_bytes_observed("nvml") is None
    assert watcher.peak_bytes_observed("pool") == 11


def test_slow_probes_keep_their_own_cadence_on_both_paths():
    reads = {"n": 0}

    def counted():
        reads["n"] += 1
        return 5

    watcher = GpuPeakMemoryWatcher(
        [MemoryProbe(name="fast", scope="s", read=lambda: 1),
         MemoryProbe(name="slow", scope="s", read=counted,
                     interval_seconds=60.0, strict=False)],
        interval_seconds=0.001)
    for _ in range(5):
        watcher.sample()
    assert reads["n"] == 1, "a 60 s probe was read on every boundary sample"
    watcher.start()
    time.sleep(0.05)
    watcher.stop()
    assert reads["n"] == 1, "a 60 s probe was read on every 1 ms tick"
    assert watcher.summary()["probes"]["slow"]["interval_seconds"] == 60.0
    assert watcher.summary()["probes"]["fast"]["interval_seconds"] == 0.001


def test_nonzero_seconds_is_the_time_the_last_reading_was_above_zero():
    values = iter([0, 0, 900, 900, 0])
    watcher = GpuPeakMemoryWatcher(
        [MemoryProbe(name="others", scope="s", read=lambda: next(values))])
    watcher.sample()
    time.sleep(0.02)
    watcher.sample()             # previous 0: adds nothing
    assert watcher.nonzero_seconds("others") == 0.0
    time.sleep(0.02)
    watcher.sample()             # previous 0: adds nothing
    assert watcher.nonzero_seconds("others") == 0.0
    time.sleep(0.03)
    watcher.sample()             # previous 900: adds the 30 ms gap
    shared = watcher.nonzero_seconds("others")
    assert shared >= 0.02
    time.sleep(0.03)
    watcher.sample()             # previous 900: adds another gap
    assert watcher.nonzero_seconds("others") >= shared + 0.02
    assert watcher.summary()["probes"]["others"]["last_bytes"] == 0
    assert watcher.peak_bytes("others") == 900


def test_receipt_rows_say_none_when_nvml_could_not_be_read():
    def broken():
        raise ProcessMemoryUnavailable("N/A")

    watcher = GpuPeakMemoryWatcher([
        MemoryProbe(name="nvml_this_process_used", scope="s", read=broken,
                    strict=False),
        MemoryProbe(name="nvml_other_processes_used", scope="s",
                    read=broken, strict=False),
        MemoryProbe(name="nvml_device_used", scope="s", read=broken,
                    strict=False),
    ])
    watcher.sample()
    assert process_memory_receipt(watcher) == {
        "this_process_peak_bytes_nvml": None,
        "other_processes_peak_bytes_nvml": None,
        "device_wide_peak_bytes_nvml": None,
        "card_shared_seconds": None,
        "card_shared": None,
    }


def test_receipt_rows_carry_the_split_and_the_shared_time():
    run = _smi(f"1, 8194, {_CARD}\n2, 15214, {_CARD}\n", f"{_CARD}, 23409\n")
    watcher = GpuPeakMemoryWatcher(
        nvidia_smi_process_probes(pid=1, run=run, interval_seconds=0.001),
        interval_seconds=0.001)
    watcher.start()
    try:
        # The receipt rounds shared time to 0.1 s.
        assert _wait_for(lambda: watcher.nonzero_seconds(
            "nvml_other_processes_used") >= 0.15)
    finally:
        watcher.stop()
    rows = process_memory_receipt(watcher)
    assert rows["this_process_peak_bytes_nvml"] == 8194 * _MIB
    assert rows["other_processes_peak_bytes_nvml"] == 15214 * _MIB
    assert rows["device_wide_peak_bytes_nvml"] == 23409 * _MIB
    assert rows["card_shared"] is True
    assert rows["card_shared_seconds"] > 0.0


# ---------------------------------------------------------------------------
# Who waits for nvidia-smi: only the NVML thread.
#
# Defect pinned here (2026-10-05, shared 8-card box): the boundary
# ``sample()`` the runners call on the forecast's MAIN thread ran the due
# NVML probes inline -- two nvidia-smi subprocesses every 2 s.  One call
# took 0.5 s to several seconds on the busy box, and a 750 m city run that
# stepped in 0.02 s per step took 4.5-8.8 s per outer step (240 s per
# forecast hour against about 24).
# ---------------------------------------------------------------------------

def _wait_for(predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.005)
    return True


class _SlowSmi:
    """An nvidia-smi stand-in that blocks until released, and records
    which thread called it."""

    def __init__(self, apps, gpus):
        self.apps, self.gpus = apps, gpus
        self.release = threading.Event()
        self.entered = threading.Event()
        self.threads = []

    def __call__(self, arguments):
        self.threads.append(threading.current_thread().name)
        self.entered.set()
        assert self.release.wait(timeout=30.0)
        if arguments[0].startswith("--query-compute-apps"):
            return self.apps
        return self.gpus


def test_boundary_sample_never_runs_nvidia_smi():
    smi = _SlowSmi(f"1, 10, {_CARD}\n", f"{_CARD}, 10\n")
    smi.release.set()
    watcher = GpuPeakMemoryWatcher(
        (MemoryProbe(name="pool", scope="s", read=lambda: 3),)
        + nvidia_smi_process_probes(pid=1, run=smi))
    for _ in range(20):
        watcher.sample()
    assert smi.threads == [], "a boundary sample ran nvidia-smi"
    assert watcher.peak_bytes("pool") == 3
    assert watcher.peak_bytes_observed("nvml_this_process_used") is None


def test_a_blocked_nvidia_smi_costs_the_main_thread_nothing():
    """The measured defect, CPU-side: nvidia-smi hangs, the step-boundary
    samples keep their microsecond cost, and the fast peak thread keeps
    folding the in-process views."""
    smi = _SlowSmi(f"1, 8194, {_CARD}\n2, 15214, {_CARD}\n",
                   f"{_CARD}, 23409\n")
    fast_reads = {"n": 0}

    def pool():
        fast_reads["n"] += 1
        return 7

    watcher = GpuPeakMemoryWatcher(
        (MemoryProbe(name="pool", scope="s", read=pool),)
        + nvidia_smi_process_probes(pid=1, run=smi, interval_seconds=0.001),
        interval_seconds=0.001)
    watcher.start()
    try:
        assert smi.entered.wait(timeout=10.0)
        started = time.perf_counter()
        for _ in range(200):
            watcher.sample()
        boundary_wall = time.perf_counter() - started
        before = fast_reads["n"]
        assert _wait_for(lambda: fast_reads["n"] > before + 50), (
            "the fast peak thread stalled behind nvidia-smi")
        smi.release.set()
        assert _wait_for(
            lambda: watcher.observed("nvml_other_processes_used"))
    finally:
        smi.release.set()
        watcher.stop()
    # 200 boundary samples while nvidia-smi was blocked: no subprocess
    # wait in them (the defect cost 0.5 s and more per call).
    assert boundary_wall < 0.25, boundary_wall
    assert set(smi.threads) == {"gpu-mem-watch-nvml"}
    assert watcher.peak_bytes("nvml_other_processes_used") == 15214 * _MIB


def test_stop_does_not_wait_for_an_nvidia_smi_in_flight():
    smi = _SlowSmi(f"1, 10, {_CARD}\n", f"{_CARD}, 10\n")
    watcher = GpuPeakMemoryWatcher(
        (MemoryProbe(name="pool", scope="s", read=lambda: 3),)
        + nvidia_smi_process_probes(pid=1, run=smi))
    watcher.start()
    try:
        assert smi.entered.wait(timeout=10.0)
        started = time.perf_counter()
        watcher.stop()
        assert time.perf_counter() - started < 1.0
    finally:
        smi.release.set()
    watcher.sample()  # the end-of-run boundary sample, still subprocess-free
    watcher._background_thread.join(timeout=10.0)
    summary = watcher.summary()
    assert summary["background_pass_in_flight_at_stop"] is True
    assert summary["background_readings_dropped_at_stop"] >= 1
    # Dropped, not folded after the run: the receipt says "not measured".
    assert watcher.peak_bytes_observed("nvml_this_process_used") is None
    assert summary["probes"]["nvml_this_process_used"]["read_on"] == (
        "background-thread-only")
    assert summary["probes"]["pool"]["read_on"] == (
        "boundary-samples+background-thread")


def test_a_timed_out_nvidia_smi_is_transient_not_a_retirement():
    calls = {"n": 0}

    def run(arguments):
        calls["n"] += 1
        if calls["n"] == 1:
            # One timed-out pass; its sibling probes must not re-run it.
            raise TransientProbeError("nvidia-smi ran past 10 s")
        if arguments[0].startswith("--query-compute-apps"):
            return f"1, 8194, {_CARD}\n"
        return f"{_CARD}, 8194\n"

    watcher = GpuPeakMemoryWatcher(
        nvidia_smi_process_probes(pid=1, run=run, interval_seconds=0.001),
        interval_seconds=0.001)
    watcher.start()
    try:
        assert _wait_for(lambda: all(
            watcher.observed(name) for name in (
                "nvml_this_process_used", "nvml_other_processes_used",
                "nvml_device_used")))
    finally:
        watcher.stop()
    probes = watcher.summary()["probes"]
    for name in ("nvml_this_process_used", "nvml_other_processes_used",
                 "nvml_device_used"):
        assert probes[name]["error"] is None
        assert probes[name]["transient_errors"] >= 1
        assert "ran past" in probes[name]["last_transient_error"]
    assert watcher.peak_bytes("nvml_this_process_used") == 8194 * _MIB


def test_the_nvidia_smi_call_carries_a_timeout_that_reads_as_transient(
        monkeypatch):
    import subprocess

    from gpuwm.core import gpu_mem_watch

    seen = {}

    def fake_run(command, **kwargs):
        seen.update(kwargs)
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(gpu_mem_watch.subprocess, "run", fake_run)
    with pytest.raises(TransientProbeError, match="ran past"):
        gpu_mem_watch._run_nvidia_smi_text(["--query-gpu=uuid"])
    assert seen["timeout"] == NVIDIA_SMI_TIMEOUT_SECONDS


def test_a_failed_pass_is_not_rerun_by_its_sibling_probes():
    calls = []

    def run(arguments):
        calls.append(arguments[0])
        raise ProcessMemoryUnavailable("no nvidia-smi on PATH")

    watcher = GpuPeakMemoryWatcher(
        nvidia_smi_process_probes(pid=1, run=run, interval_seconds=60.0),
        interval_seconds=0.001)
    watcher.start()
    try:
        assert _wait_for(lambda: all(
            row["error"] for row in watcher.summary()["probes"].values()))
    finally:
        watcher.stop()
    assert len(calls) == 1


def test_a_background_only_probe_cannot_be_strict():
    with pytest.raises(ValueError, match="background_only"):
        GpuPeakMemoryWatcher([MemoryProbe(
            name="nvml", scope="s", read=lambda: 1, background_only=True)])


# ---------------------------------------------------------------------------
# Wiring pins: both prepared runners carry the split into their receipts.
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
def test_runner_watches_the_per_process_views(runner_key):
    path, func_name = _runner_sources()[runner_key]
    func = _function_node(path, func_name)
    constructed = [
        node for node in ast.walk(func)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "GpuPeakMemoryWatcher"]
    assert len(constructed) == 1
    names = {
        node.func.id for node in ast.walk(constructed[0])
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert {"default_cupy_probes", "nvidia_smi_process_probes"} <= names, (
        f"{func_name} watches the pool and the card but not WHOSE bytes")


@pytest.mark.parametrize("runner_key", ["single", "tree"])
def test_runner_receipt_memory_section_carries_the_process_split(runner_key):
    path, func_name = _runner_sources()[runner_key]
    func = _function_node(path, func_name)
    memory_dicts = [
        node for node in ast.walk(func)
        if isinstance(node, ast.Dict)
        and any(isinstance(key, ast.Constant)
                and key.value == "gpu_peak_used_bytes_observed"
                for key in node.keys)]
    assert memory_dicts
    spread = [
        value for node in memory_dicts
        for key, value in zip(node.keys, node.values)
        if key is None
        and isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id == "process_memory_receipt"]
    assert spread, (
        f"{func_name}'s memory section does not spread "
        "process_memory_receipt(memory_watch)")
