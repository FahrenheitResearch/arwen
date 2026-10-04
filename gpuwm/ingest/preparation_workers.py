"""Preparation worker policy and runtime evidence.

Explicit preparation workers override inherited library defaults. Affinity
and the smallest CPU cgroup quota remain hard limits.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

THREAD_DEFAULTS = (
    "RAYON_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
)
INHERITED_LIMITS_ENV = "GPUWM_PREP_INHERITED_LIMITS"
PREPARATION_THREADS_ENV = "GPUWM_PREPROCESS_THREADS"
# Native column scratch is small; this also reserves allocator arenas and
# Python column slabs. Saved 8/16/32/64-worker process-tree measurements in
# test_cpu_preparation_host_ram.py grew by at most 28 MiB per extra worker.
WORKER_SCRATCH_BYTES = 32 * 1024**2
WORKER_MEMORY_SHARE = 0.7


def diagnostic(text):
    """Console loss must not stop a detached producer or its publication."""
    try:
        print(text, file=sys.stderr, flush=True)
    except (OSError, ValueError):
        pass


def _quota(quota, period):
    try:
        quota, period = int(quota), int(period)
    except (ValueError, TypeError):
        return None
    return max(1, quota // period) if quota > 0 and period > 0 else None


def _ancestors(root, relative):
    if ".." in Path(relative).parts:
        return (root,)
    current = root / relative.lstrip("/")
    result = [current]
    while current != root and root in current.parents:
        current = current.parent
        result.append(current)
    return tuple(result)


def cgroup_cpu_count(root=Path("/sys/fs/cgroup"), membership=None):
    """Read v2 and v1 quotas through all visible ancestors."""
    root = Path(root)
    if membership is None:
        try:
            membership = Path("/proc/self/cgroup").read_text()
        except OSError:
            membership = ""
    groups = [(root, True)]
    for line in membership.splitlines():
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        _, controllers, relative = parts
        if not controllers:
            groups.extend((path, True) for path in _ancestors(root, relative))
        elif "cpu" in controllers.split(","):
            for controller in (controllers, "cpu", "cpu,cpuacct"):
                groups.extend((path, False) for path in _ancestors(root / controller, relative))
    limits = []
    for group, v2 in groups:
        try:
            parts = ((group / "cpu.max").read_text().split() if v2 else
                     [(group / "cpu.cfs_quota_us").read_text(),
                      (group / "cpu.cfs_period_us").read_text()])
            count = _quota(*parts) if len(parts) == 2 else None
            if count is not None:
                limits.append(count)
        except OSError:
            pass
    return min(limits) if limits else None


def cpu_budget():
    try:
        affinity = max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        affinity = max(1, int(os.cpu_count() or 1))
    quota = cgroup_cpu_count() if sys.platform.startswith("linux") else None
    return {"affinity_cpus": affinity, "cgroup_cpus": quota,
            "available_cpus": min(affinity, quota or affinity)}


def effective_workers(requested=None):
    available = cpu_budget()["available_cpus"]
    memory = memory_worker_limit()
    available = min(available, memory) if memory is not None else available
    return available if requested is None else min(max(1, int(requested)), available)


def memory_worker_limit():
    available = host_available_bytes()
    if available is None:
        return None
    return max(1, int(available * WORKER_MEMORY_SHARE) // WORKER_SCRATCH_BYTES)


def host_available_bytes(root=Path("/sys/fs/cgroup"), membership=None,
                         meminfo=Path("/proc/meminfo")):
    """Physical headroom, including reclaimable cache and ancestor limits.

    Kept in the preparation package: the standalone preparation wheel does
    not carry the forecast or tilestream packages.
    """
    if sys.platform == "win32":
        import ctypes
        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", ctypes.c_uint32), ("load", ctypes.c_uint32)] + [
                (name, ctypes.c_uint64) for name in
                ("total_phys", "avail_phys", "total_page", "avail_page",
                 "total_virtual", "avail_virtual", "avail_extended")]
        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        windll = getattr(ctypes, "windll", None)
        if windll is None or not windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None
        return int(status.avail_phys)
    try:
        available = next(int(line.split()[1]) * 1024
                         for line in Path(meminfo).read_text().splitlines()
                         if line.startswith("MemAvailable:"))
    except (OSError, ValueError, IndexError, StopIteration):
        return None
    root = Path(root)
    if membership is None:
        try:
            membership = Path("/proc/self/cgroup").read_text()
        except OSError:
            membership = ""
    groups = [(root, True)]
    for line in membership.splitlines():
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        _, controllers, relative = parts
        if not controllers:
            groups.extend((path, True) for path in _ancestors(root, relative))
        elif "memory" in controllers.split(","):
            for controller in (controllers, "memory"):
                groups.extend((path, False) for path in _ancestors(root / controller, relative))
    for group, v2 in groups:
        try:
            limit = int((group / ("memory.max" if v2 else "memory.limit_in_bytes")).read_text())
            current = int((group / ("memory.current" if v2 else "memory.usage_in_bytes")).read_text())
        except (OSError, ValueError):
            continue
        if limit < 0 or limit >= 1 << 62:
            continue
        inactive = 0
        try:
            key = "inactive_file" if v2 else "total_inactive_file"
            inactive = next(int(line.split()[1])
                            for line in (group / "memory.stat").read_text().splitlines()
                            if line.split()[0] == key)
        except (OSError, ValueError, IndexError, StopIteration):
            pass
        available = min(available, max(0, limit - max(0, current - inactive)))
    return available


def worker_receipt(requested=None, *, native_effective=None):
    budget = cpu_budget()
    effective = effective_workers(requested)
    if native_effective is not None:
        effective = min(effective, int(native_effective))
    inherited = {name: os.environ[name] for name in THREAD_DEFAULTS if name in os.environ}
    try:
        prior = json.loads(os.environ.get(INHERITED_LIMITS_ENV, "{}"))
        if isinstance(prior, dict):
            inherited = {name: str(prior.get(name, value)) for name, value in inherited.items()}
    except (ValueError, TypeError):
        pass
    return {"requested_workers": requested, "effective_workers": effective,
            **budget, "inherited_library_limits": {
                name: value for name, value in inherited.items()},
            "native_pool_effective_workers": native_effective,
            "memory_worker_limit": memory_worker_limit(),
            "worker_scratch_bytes": WORKER_SCRATCH_BYTES,
            "policy": "explicit-preparation-workers-override-library-defaults"}


def worker_environment(requested, *, environment=None):
    result = dict(os.environ if environment is None else environment)
    if requested is not None:
        result.setdefault(INHERITED_LIMITS_ENV, json.dumps({
            name: result[name] for name in THREAD_DEFAULTS if name in result}, sort_keys=True))
        count = str(effective_workers(requested))
        result.update({name: count for name in THREAD_DEFAULTS})
        result["GPUWM_MAPPED_ENGINE_THREADS"] = count
        result[PREPARATION_THREADS_ENV] = count
    return result


def configure_preparation_workers(requested):
    """Configure subprocess and lazy native pools before preparation starts.

    NumPy elementwise operations do not use BLAS threads. The Rust host
    kernels own their pool and pass explicit widths independently of these
    defaults; the environment also reaches static-field and writer pools.
    """
    receipt = worker_receipt(requested)
    try:
        from threadpoolctl import threadpool_info, threadpool_limits
    except ImportError:
        receipt["loaded_numeric_pools_before"] = "not inspected: threadpoolctl unavailable"
        receipt["loaded_numeric_pools"] = "not inspected: threadpoolctl unavailable"
        receipt["numeric_pool_control"] = "unavailable"
        if requested is not None:
            diagnostic("preparation worker warning: threadpoolctl is unavailable; "
                       "already loaded numeric pools cannot be inspected or resized. "
                       "Install the declared runtime dependencies. Native preparation "
                       "kernels still use their explicit Rust workers.")
    else:
        def numeric_pools():
            return [{key: pool.get(key) for key in ("internal_api", "num_threads", "version")}
                    for pool in threadpool_info()]
        receipt["loaded_numeric_pools_before"] = numeric_pools()
        if requested is not None:
            # The CLI sets defaults before spawning its child. Direct module
            # callers may already have imported NumPy, so also set the loaded
            # libraries through their runtime API. This call persists for the
            # preparation process; it is deliberately not a context manager.
            threadpool_limits(limits=receipt["effective_workers"])
        receipt["loaded_numeric_pools"] = numeric_pools()
        receipt["numeric_pool_control"] = "applied" if requested is not None else "not requested"
    if requested is not None:
        environment = worker_environment(requested)
        for name in (*THREAD_DEFAULTS, "GPUWM_MAPPED_ENGINE_THREADS", PREPARATION_THREADS_ENV):
            os.environ[name] = environment[name]
        os.environ[INHERITED_LIMITS_ENV] = environment[INHERITED_LIMITS_ENV]
    effective = receipt["effective_workers"]
    if requested is not None and effective < requested:
        diagnostic(f"preparation worker warning: requested {requested} workers, but affinity, "
                   f"CPU quota and worker memory permit {effective}; using {effective}.")
    if requested is not None and any(
            value.isdecimal() and int(value) < effective
            for value in receipt["inherited_library_limits"].values()):
        diagnostic("preparation worker warning: inherited numeric/Rayon limits were below "
                   "the requested preparation width. Subprocess defaults are overridden; "
                   "the proof records loaded numeric pools before and after runtime control. "
                   "Native preparation kernels own explicit Rust workers.")
    for pool in receipt["loaded_numeric_pools"] if isinstance(receipt["loaded_numeric_pools"], list) else ():
        if requested is not None and pool["num_threads"] != effective:
            diagnostic(f"preparation worker warning: loaded {pool['internal_api']} pool "
                       f"reports {pool['num_threads']} threads after requesting {effective}; "
                       "the proof records the actual width.")
    diagnostic(f"preparation workers: requested={requested or 'auto'}, effective={effective}, "
               f"affinity={receipt['affinity_cpus']}, cgroup={receipt['cgroup_cpus'] or 'unlimited'}; "
               "Rust kernels use explicit worker widths")
    return receipt
