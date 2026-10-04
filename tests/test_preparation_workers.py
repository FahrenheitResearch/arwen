"""CPU quotas and explicit preparation widths through inherited caps."""
from __future__ import annotations

import ctypes
import json

import numpy as np
import pytest

from gpuwm.ingest import preparation_workers as workers


@pytest.mark.parametrize("quota,period,expected", [
    ("18400000", "100000", 184), ("18460000", "100000", 184),
    ("50000", "100000", 1), ("max", "100000", None),
    ("-1", "100000", None), ("100", "0", None),
])
def test_cpu_quota_never_rounds_above_capacity(quota, period, expected):
    assert workers._quota(quota, period) == expected


def test_nested_v2_quota_limits_an_unlimited_child(tmp_path):
    child = tmp_path / "jobs" / "prep"
    child.mkdir(parents=True)
    (tmp_path / "cpu.max").write_text("max 100000")
    (child.parent / "cpu.max").write_text("3250000 100000")
    (child / "cpu.max").write_text("max 100000")
    assert workers.cgroup_cpu_count(tmp_path, "0::/jobs/prep\n") == 32
    (tmp_path / "cpu.max").write_text("800000 100000")
    assert workers.cgroup_cpu_count(tmp_path, "0::/jobs/prep\n") == 8


def test_v1_combined_controller_and_namespaced_root(tmp_path):
    child = tmp_path / "cpu,cpuacct" / "jobs" / "prep"
    child.mkdir(parents=True)
    (child.parent / "cpu.cfs_quota_us").write_text("400000")
    (child.parent / "cpu.cfs_period_us").write_text("100000")
    (child / "cpu.cfs_quota_us").write_text("-1")
    (child / "cpu.cfs_period_us").write_text("100000")
    assert workers.cgroup_cpu_count(tmp_path, "2:cpu,cpuacct:/jobs/prep\n") == 4
    (tmp_path / "cpu.max").write_text("200000 100000")
    assert workers.cgroup_cpu_count(tmp_path, "0::/invisible/host/path\n") == 2


def test_explicit_width_overrides_inherited_caps_and_preserves_evidence(monkeypatch):
    monkeypatch.setattr(workers, "memory_worker_limit", lambda: None)
    monkeypatch.setattr(workers, "cpu_budget", lambda: {
        "affinity_cpus": 192, "cgroup_cpus": 184, "available_cpus": 184})
    original = {name: "1" for name in workers.THREAD_DEFAULTS}
    environment = workers.worker_environment(182, environment=original)
    assert all(environment[name] == "182" for name in workers.THREAD_DEFAULTS)
    assert environment["GPUWM_MAPPED_ENGINE_THREADS"] == "182"
    assert json.loads(environment[workers.INHERITED_LIMITS_ENV]) == original
    limited = workers.worker_environment(256, environment=original)
    assert limited["RAYON_NUM_THREADS"] == "184"
    assert original["RAYON_NUM_THREADS"] == "1"


def test_quota_clamp_is_visible_in_log_and_receipt(monkeypatch, capsys):
    monkeypatch.setattr(workers, "memory_worker_limit", lambda: None)
    monkeypatch.setattr(workers, "cpu_budget", lambda: {
        "affinity_cpus": 192, "cgroup_cpus": 8, "available_cpus": 8})
    for name in workers.THREAD_DEFAULTS:
        monkeypatch.setenv(name, "1")
    monkeypatch.setenv("GPUWM_MAPPED_ENGINE_THREADS", "1")
    monkeypatch.setenv(workers.PREPARATION_THREADS_ENV, "1")
    monkeypatch.delenv(workers.INHERITED_LIMITS_ENV, raising=False)
    receipt = workers.configure_preparation_workers(182)
    assert receipt["requested_workers"] == 182
    assert receipt["effective_workers"] == 8
    assert receipt["inherited_library_limits"]["RAYON_NUM_THREADS"] == "1"
    assert "requested 182 workers" in capsys.readouterr().err


def test_memory_budget_prices_worker_scratch(monkeypatch):
    monkeypatch.setattr(workers, "host_available_bytes", lambda: 16 * workers.WORKER_SCRATCH_BYTES)
    monkeypatch.setattr(workers, "cpu_budget", lambda: {
        "affinity_cpus": 192, "cgroup_cpus": 184, "available_cpus": 184})
    assert workers.memory_worker_limit() == 11
    assert workers.effective_workers(182) == 11


def test_headroom_keeps_inactive_cache_and_the_smallest_ancestor(tmp_path, monkeypatch):
    monkeypatch.setattr(workers.sys, "platform", "linux")
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemAvailable: 10000 kB\n")
    root = tmp_path / "cgroup"
    child = root / "jobs" / "prep"
    child.mkdir(parents=True)
    (child / "memory.max").write_text("8000000")
    (child / "memory.current").write_text("5000000")
    (child / "memory.stat").write_text("inactive_file 2000000\n")
    (root / "memory.max").write_text("4000000")
    (root / "memory.current").write_text("1000000")
    assert workers.host_available_bytes(root, "0::/jobs/prep\n", meminfo) == 3000000


def test_standalone_preparation_does_not_import_forecast_packages(monkeypatch):
    import builtins
    original = builtins.__import__
    def without_forecast(name, *args, **kwargs):
        if name == "tilestream" or name.startswith("tilestream."):
            raise ModuleNotFoundError("standalone preparation has no tilestream")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", without_forecast)
    from gpuwm.ingest.prepared_writer import batch_budget
    assert workers.effective_workers() >= 1
    assert batch_budget()[0] >= 1


def test_cli_launch_overrides_caps_before_numpy_import(tmp_path, monkeypatch):
    import sys
    from gpuwm import source_cli
    monkeypatch.setattr(workers, "cpu_budget", lambda: {
        "affinity_cpus": 24, "cgroup_cpus": 2, "available_cpus": 2})
    monkeypatch.setattr(workers, "memory_worker_limit", lambda: None)
    for name in workers.THREAD_DEFAULTS:
        monkeypatch.setenv(name, "1")
    monkeypatch.delenv(workers.INHERITED_LIMITS_ENV, raising=False)
    target = tmp_path / "child.json"
    script = """
import json, os, sys
from pathlib import Path
import numpy
from threadpoolctl import threadpool_info
pools = [{key: row[key] for key in ('internal_api', 'num_threads')} for row in threadpool_info()]
names = ('RAYON_NUM_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'GPUWM_MAPPED_ENGINE_THREADS', 'GPUWM_PREPROCESS_THREADS')
Path(sys.argv[1]).write_text(json.dumps({'limits': {name: os.environ.get(name) for name in names}, 'loaded_pools': pools, 'inherited': json.loads(os.environ['GPUWM_PREP_INHERITED_LIMITS'])}))
"""
    result = source_cli._run_adapter_command(
        [sys.executable, "-c", script, str(target), "--preprocess-workers", "182"])
    assert result.returncode == 0
    receipt = json.loads(target.read_text())
    assert set(receipt["limits"].values()) == {"2"}
    assert set(receipt["inherited"].values()) == {"1"}
    if receipt["loaded_pools"]:
        assert all(row["num_threads"] == 2 for row in receipt["loaded_pools"])


@pytest.mark.parametrize("requested", [8, 182])
def test_direct_module_resizes_already_loaded_numeric_pools(tmp_path, requested):
    import os
    import subprocess
    import sys
    target = tmp_path / "direct.json"
    environment = dict(os.environ)
    environment.update({name: "1" for name in workers.THREAD_DEFAULTS})
    environment.pop(workers.INHERITED_LIMITS_ENV, None)
    script = """
import json, sys
from pathlib import Path
import numpy
from gpuwm.ingest.preparation_workers import configure_preparation_workers
receipt = configure_preparation_workers(int(sys.argv[2]))
Path(sys.argv[1]).write_text(json.dumps(receipt))
"""
    subprocess.run([sys.executable, "-c", script, str(target), str(requested)],
                   env=environment, check=True, capture_output=True, text=True)
    receipt = json.loads(target.read_text())
    assert receipt["numeric_pool_control"] == "applied"
    assert all(row["num_threads"] == 1 for row in receipt["loaded_numeric_pools_before"])
    assert all(row["num_threads"] == receipt["effective_workers"]
               for row in receipt["loaded_numeric_pools"])


def test_missing_numeric_controller_reports_uninspected_pools(monkeypatch, capsys):
    import sys
    monkeypatch.setitem(sys.modules, "threadpoolctl", None)
    receipt = workers.configure_preparation_workers(8)
    assert receipt["numeric_pool_control"] == "unavailable"
    assert receipt["loaded_numeric_pools"] == "not inspected: threadpoolctl unavailable"
    assert "cannot be inspected or resized" in capsys.readouterr().err


def test_native_rayon_width_and_outputs_ignore_inherited_global_limit(monkeypatch):
    from gpuwm.ingest.cpu_backend import CpuPreprocessBackend
    monkeypatch.setenv("RAYON_NUM_THREADS", "1")
    try:
        backend = CpuPreprocessBackend()
    except (FileNotFoundError, OSError) as error:
        pytest.skip(str(error))
    query = getattr(backend._library, "gpuwm_preprocess_cpu_parallelism", None)
    if query is None:
        pytest.skip("native library predates explicit Rayon pool receipt")
    query.argtypes = [ctypes.c_size_t]
    query.restype = ctypes.c_size_t
    count = workers.effective_workers()
    assert query(0) == count
    latitude = np.linspace(30, 35, 13)
    longitude = np.linspace(-101, -95, 15)
    y, x = np.meshgrid(np.linspace(30.2, 34.8, 65), np.linspace(-100.8, -95.2, 73), indexing="ij")
    field = np.random.default_rng(12).normal(size=(7, 13, 15)).astype(np.float32)
    plan = backend.regular_plan(latitude, longitude, y, x)
    reference = plan.apply(field, workers=1).tobytes()
    for width in (1, 8, 32, count):
        assert query(width) == min(width, count)
        assert plan.apply(field, workers=width).tobytes() == reference
