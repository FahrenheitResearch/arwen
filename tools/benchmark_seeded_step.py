"""Repeatable, output-hashed benchmark for one GPU-resident model step.

The default workload is deliberately small and self-contained: it builds a
seeded, balanced state without downloading input data, warms the complete
NVRTC/CuPy path, times a fixed number of full RK/acoustic steps, and hashes
the externally visible FP32 state after timing.  A second, independently
seeded lane attributes device and host submission time to coarse solver
sections.  The two lanes must finish with the same hash when their step
counts match.

Use ``--profiler-capture`` with Nsight Systems'
``--capture-range=cudaProfilerApi`` to exclude setup, compilation, warmup,
and output hashing from the trace.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import functools
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Iterator

import numpy as np


SCHEMA = "gpuwm-seeded-step-benchmark-v1"
DEFAULT_SEED = 20_260_731
DEFAULT_NX = 64
DEFAULT_NY = 64
DEFAULT_NZ = 32
DEFAULT_WARMUP_STEPS = 2
DEFAULT_TIMED_STEPS = 5

_OUTPUT_FIELDS = (
    "u", "v", "w", "thp", "php", "mup", "p", "al", "alt",
    "qv", "qc", "qr", "qi", "qs", "qg", "qh",
    "nc", "nr", "ni", "ns", "ng", "nh",
    "qndrop", "qnr", "qni", "qns", "qng", "qnh", "qnn",
    "qvolg", "qvolh", "effc", "effr", "effi", "effs",
    "h_diabatic",
)

# These calls cover the device-producing parts of dycore.step without
# overlapping one another.  Work expressed directly as slice assignment in
# step() (time-t copies and tendency zeroing) is reported as "unattributed".
_FUNCTION_SECTIONS = (
    ("update_diagnostics", "diagnostics"),
    ("prepare_fixed_tendencies", "fixed_tendencies"),
    ("fixed_scalar_tendencies", "fixed_tendencies"),
    ("stage_fluxes", "stage_fluxes"),
    ("prepare_moist_cq", "stage_setup"),
    ("_add_slow_tendencies", "slow_tendencies"),
    ("add_h_diabatic_tendency", "held_tendencies"),
    ("add_diffusion_tendencies", "held_tendencies"),
    ("add_fixed_dry_tendencies", "held_tendencies"),
    ("apply_w_damping", "boundary_and_damping"),
    ("apply_state_lateral_boundaries", "boundary_and_damping"),
    ("apply_open_radiative_bc", "boundary_and_damping"),
    ("apply_open_zero_gradient", "boundary_and_damping"),
    ("apply_state_boundary_values", "boundary_and_damping"),
    ("set_w_surface", "boundary_and_damping"),
    ("prepare_acoustic_coefficients", "acoustic_setup"),
    ("_sumflux_launch", "scalar_flux_bookkeeping"),
    ("advance_scalars_stage", "scalar_transport"),
    ("update_up_heli_max", "diagnostics"),
    ("apply_microphysics", "microphysics"),
)

_FACTORY_SECTIONS = (
    ("_prepare_small_step_init_launch", "rk_bookkeeping"),
    ("_prepare_small_step_finish_launch", "rk_bookkeeping"),
    ("_prepare_emdiv_filter_launch", "acoustic_substeps"),
    ("prepare_acoustic_substep_launch", "acoustic_substeps"),
    ("_prepare_sumflux_launch", "scalar_flux_bookkeeping"),
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--nx", type=int, default=DEFAULT_NX)
    parser.add_argument("--ny", type=int, default=DEFAULT_NY)
    parser.add_argument("--nz", type=int, default=DEFAULT_NZ)
    parser.add_argument("--warmup-steps", type=int,
                        default=DEFAULT_WARMUP_STEPS)
    parser.add_argument("--steps", type=int, default=DEFAULT_TIMED_STEPS)
    parser.add_argument(
        "--profile-steps", type=int,
        help="section-attribution steps (default: --steps)",
    )
    parser.add_argument(
        "--microphysics", type=int, choices=(0, 1, 6, 10, 18), default=0,
        help="optional moist microphysics selector; 0 is the dry core",
    )
    parser.add_argument(
        "--profiler-capture", action="store_true",
        help="bracket only the section lane with cudaProfilerStart/Stop",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--estimate-only", action="store_true",
        help="print conservative run sizing without importing CuPy",
    )
    args = parser.parse_args()
    for name in ("nx", "ny", "nz", "steps", "warmup_steps"):
        minimum = 0 if name == "warmup_steps" else 1
        if getattr(args, name) < minimum:
            parser.error(f"--{name.replace('_', '-')} must be >= {minimum}")
    if args.profile_steps is None:
        args.profile_steps = args.steps
    if args.profile_steps < 1:
        parser.error("--profile-steps must be >= 1")
    return args


def estimate_run(args: argparse.Namespace) -> dict[str, int | float]:
    """Return a conservative, setup-free size estimate for this invocation."""
    cells = int(args.nx) * int(args.ny) * int(args.nz)
    # DomainState, dycore scratch, and a moist scheme carry different live
    # inventories.  128 FP32 cell equivalents is deliberately conservative
    # for the dry lane; moist microphysics doubles that allowance.
    fp32_cell_equivalents = 256 if args.microphysics else 128
    device_bytes = cells * fp32_cell_equivalents * 4
    # Staggered faces, vertical profiles, allocator rounding, and CUDA module
    # state are covered by a fixed allowance.  This is a launch-safety
    # estimate, not the measured resident-byte result emitted after a run.
    device_bytes += 256 * 1024**2
    traced_steps = args.profile_steps if args.profiler_capture else 0
    trace_bytes = traced_steps * 24 * 1024**2
    return {
        "cells": cells,
        "estimated_device_bytes": int(device_bytes),
        "estimated_device_mib": round(device_bytes / 1024**2, 3),
        "estimated_trace_bytes": int(trace_bytes),
        "estimated_trace_mib": round(trace_bytes / 1024**2, 3),
        "warmup_steps_per_lane": int(args.warmup_steps),
        "timed_steps": int(args.steps),
        "profile_steps": int(args.profile_steps),
        "total_executed_steps": int(
            2 * args.warmup_steps + args.steps + args.profile_steps),
    }


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "-C", str(Path(__file__).resolve().parents[1]),
         "rev-parse", "HEAD"],
        text=True,
    ).strip()


def _device_name(properties: dict[str, Any]) -> str:
    name = properties["name"]
    return name.decode("utf-8") if isinstance(name, bytes) else str(name)


def _build_state(args: argparse.Namespace):
    from dataclasses import replace

    from gpuwm.verify.npref import random_acoustic_state

    moist = args.microphysics != 0
    state, cfg = random_acoustic_state(
        seed=args.seed,
        nz=args.nz,
        ny=args.ny,
        nx=args.nx,
        moist=moist,
        mp_physics=args.microphysics,
    )
    cfg = replace(
        cfg,
        run_seconds=(
            args.warmup_steps + max(args.steps, args.profile_steps)
        ) * cfg.dt,
    )
    return state, cfg


def _unique_device_arrays(root: Any) -> Iterator[tuple[str, Any]]:
    """Walk state-owned containers and yield unique CuPy arrays."""
    import cupy as cp

    seen_objects: set[int] = set()
    seen_arrays: set[int] = set()

    def walk(value: Any, prefix: str, depth: int):
        identity = id(value)
        if identity in seen_objects or depth > 4:
            return
        seen_objects.add(identity)
        if isinstance(value, cp.ndarray):
            pointer = int(value.data.ptr)
            if pointer not in seen_arrays:
                seen_arrays.add(pointer)
                yield prefix, value
            return
        if isinstance(value, dict):
            for key, child in sorted(value.items(), key=lambda item: str(item[0])):
                yield from walk(child, f"{prefix}.{key}", depth + 1)
        elif isinstance(value, (tuple, list)):
            for index, child in enumerate(value):
                yield from walk(child, f"{prefix}.{index}", depth + 1)
        elif hasattr(value, "__dict__"):
            for name, child in sorted(vars(value).items()):
                yield from walk(child, f"{prefix}.{name}", depth + 1)

    yield from walk(root, "state", 0)


def _resident_bytes(state: Any) -> int:
    return sum(int(array.nbytes)
               for _name, array in _unique_device_arrays(state))


def _output_arrays(state: Any) -> Iterator[tuple[str, Any]]:
    import cupy as cp

    seen: set[int] = set()
    for name in _OUTPUT_FIELDS:
        value = getattr(state, name, None)
        if isinstance(value, cp.ndarray) and int(value.data.ptr) not in seen:
            seen.add(int(value.data.ptr))
            yield f"state.{name}", value

    physics = getattr(state, "physics", None)
    if physics is not None:
        for prefix, owner in (
                ("physics", vars(physics)),
                ("physics.fields", getattr(physics, "fields", {}))):
            for name, value in sorted(owner.items()):
                if (isinstance(value, cp.ndarray)
                        and value.dtype == cp.float32
                        and int(value.data.ptr) not in seen):
                    seen.add(int(value.data.ptr))
                    yield f"{prefix}.{name}", value

    for slot, value in sorted(getattr(state, "_scratch", {}).items()):
        if (slot.startswith(("mp_", "cu_"))
                and isinstance(value, cp.ndarray)
                and value.dtype == cp.float32
                and int(value.data.ptr) not in seen):
            seen.add(int(value.data.ptr))
            yield f"state.scratch.{slot}", value


def _hash_outputs(state: Any) -> tuple[str, dict[str, str]]:
    import cupy as cp

    combined = hashlib.sha256()
    fields: dict[str, str] = {}
    for name, array in _output_arrays(state):
        host = np.ascontiguousarray(cp.asnumpy(array))
        payload = (
            host.dtype.str.encode("ascii")
            + np.asarray(host.shape, dtype=np.int64).tobytes()
            + host.tobytes(order="C")
        )
        digest = hashlib.sha256(payload).hexdigest()
        fields[name] = digest
        combined.update(name.encode("utf-8"))
        combined.update(payload)
    if not fields:
        raise RuntimeError("the benchmark found no FP32 output arrays")
    return combined.hexdigest(), fields


class _AllocationCounter:
    """Context-managed CuPy memory hook with compact aggregate counters."""

    def __init__(self):
        import cupy as cp

        counter = self

        class Hook(cp.cuda.MemoryHook):
            name = "gpuwm-step-allocation-counter"

            def malloc_preprocess(self, **kwargs):
                counter.pool_requests += 1
                counter.pool_requested_bytes += int(kwargs.get("size", 0))

            def alloc_preprocess(self, **kwargs):
                counter.driver_allocations += 1
                counter.driver_allocated_bytes += int(kwargs.get("size", 0))

            def free_preprocess(self, **_kwargs):
                counter.pool_frees += 1

        self._hook = Hook()
        self.pool_requests = 0
        self.pool_requested_bytes = 0
        self.driver_allocations = 0
        self.driver_allocated_bytes = 0
        self.pool_frees = 0

    def __enter__(self):
        self._hook.__enter__()
        return self

    def __exit__(self, exc_type, exc, traceback):
        return self._hook.__exit__(exc_type, exc, traceback)

    def result(self, steps: int) -> dict[str, int | float]:
        return {
            "pool_requests": self.pool_requests,
            "pool_requests_per_step": self.pool_requests / steps,
            "pool_requested_bytes": self.pool_requested_bytes,
            "driver_allocations": self.driver_allocations,
            "driver_allocations_per_step": self.driver_allocations / steps,
            "driver_allocated_bytes": self.driver_allocated_bytes,
            "pool_frees": self.pool_frees,
            "pool_frees_per_step": self.pool_frees / steps,
        }


@dataclass
class _SectionCall:
    name: str
    start: Any
    end: Any
    host_seconds: float


class _SectionProfiler:
    """Patch coarse dycore call boundaries with events and NVTX ranges."""

    def __init__(self):
        self.calls: list[_SectionCall] = []

    def measure(self, name: str, function, *args, **kwargs):
        import cupy as cp

        start = cp.cuda.Event()
        end = cp.cuda.Event()
        start.record()
        host_started = time.perf_counter()
        if cp.cuda.nvtx.available:
            cp.cuda.nvtx.RangePush(name)
        try:
            return function(*args, **kwargs)
        finally:
            if cp.cuda.nvtx.available:
                cp.cuda.nvtx.RangePop()
            host_seconds = time.perf_counter() - host_started
            end.record()
            self.calls.append(_SectionCall(
                name=name, start=start, end=end, host_seconds=host_seconds))

    @contextmanager
    def install(self, module):
        patches: list[tuple[str, Any]] = []

        def patch_function(attribute: str, section: str) -> None:
            if not hasattr(module, attribute):
                return
            original = getattr(module, attribute)

            @functools.wraps(original)
            def wrapped(*args, **kwargs):
                return self.measure(section, original, *args, **kwargs)

            patches.append((attribute, original))
            setattr(module, attribute, wrapped)

        def patch_factory(attribute: str, section: str) -> None:
            if not hasattr(module, attribute):
                return
            original = getattr(module, attribute)

            @functools.wraps(original)
            def wrapped_factory(*args, **kwargs):
                launch = original(*args, **kwargs)

                @functools.wraps(launch)
                def wrapped_launch(*launch_args, **launch_kwargs):
                    return self.measure(
                        section, launch, *launch_args, **launch_kwargs)

                return wrapped_launch

            patches.append((attribute, original))
            setattr(module, attribute, wrapped_factory)

        for attribute, section in _FUNCTION_SECTIONS:
            patch_function(attribute, section)
        for attribute, section in _FACTORY_SECTIONS:
            patch_factory(attribute, section)
        try:
            yield
        finally:
            for attribute, original in reversed(patches):
                setattr(module, attribute, original)

    def result(self, steps: int, total_gpu_ms: float,
               total_host_seconds: float) -> dict[str, dict[str, float | int]]:
        import cupy as cp

        totals: dict[str, dict[str, float | int]] = {}
        for call in self.calls:
            row = totals.setdefault(
                call.name, {"calls": 0, "gpu_ms": 0.0, "host_seconds": 0.0})
            row["calls"] += 1
            row["gpu_ms"] += float(
                cp.cuda.get_elapsed_time(call.start, call.end))
            row["host_seconds"] += call.host_seconds

        attributed_gpu_ms = sum(float(row["gpu_ms"])
                                for row in totals.values())
        attributed_host_seconds = sum(float(row["host_seconds"])
                                      for row in totals.values())
        totals["unattributed"] = {
            "calls": steps,
            "gpu_ms": max(total_gpu_ms - attributed_gpu_ms, 0.0),
            "host_seconds": max(
                total_host_seconds - attributed_host_seconds, 0.0),
        }
        result = {}
        for name, row in sorted(totals.items()):
            result[name] = {
                "calls": int(row["calls"]),
                "calls_per_step": int(row["calls"]) / steps,
                "gpu_ms_per_step": float(row["gpu_ms"]) / steps,
                "host_submit_ms_per_step": (
                    1000.0 * float(row["host_seconds"]) / steps),
            }
        return result


def _warmup(state: Any, cfg: Any, steps: int) -> None:
    import cupy as cp
    from gpuwm.core.dycore import step

    for _ in range(steps):
        step(state, cfg)
    cp.cuda.runtime.deviceSynchronize()


def _timing_lane(args: argparse.Namespace) -> tuple[dict[str, Any], str, dict]:
    import cupy as cp
    from gpuwm.core.dycore import step

    state, cfg = _build_state(args)
    _warmup(state, cfg, args.warmup_steps)
    pool = cp.get_default_memory_pool()
    pool_before = {
        "used_bytes": int(pool.used_bytes()),
        "total_bytes": int(pool.total_bytes()),
    }
    start = cp.cuda.Event()
    end = cp.cuda.Event()
    start.record()
    wall_started = time.perf_counter()
    submit_started = time.perf_counter()
    for _ in range(args.steps):
        step(state, cfg)
    submit_seconds = time.perf_counter() - submit_started
    end.record()
    cp.cuda.runtime.deviceSynchronize()
    wall_seconds = time.perf_counter() - wall_started
    gpu_ms = float(cp.cuda.get_elapsed_time(start, end))
    pool_after = {
        "used_bytes": int(pool.used_bytes()),
        "total_bytes": int(pool.total_bytes()),
    }
    combined, fields = _hash_outputs(state)
    result = {
        "wall_ms_per_step": 1000.0 * wall_seconds / args.steps,
        "gpu_ms_per_step": gpu_ms / args.steps,
        "host_submit_ms_per_step": 1000.0 * submit_seconds / args.steps,
        "final_sync_ms_per_step": (
            1000.0 * max(wall_seconds - submit_seconds, 0.0) / args.steps),
        "resident_array_bytes": _resident_bytes(state),
        "pool_before": pool_before,
        "pool_after": pool_after,
    }
    return result, combined, fields


def _profile_lane(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    import cupy as cp
    from gpuwm.core import dycore

    state, cfg = _build_state(args)
    _warmup(state, cfg, args.warmup_steps)
    profiler = _SectionProfiler()
    allocation_counter = _AllocationCounter()
    start = cp.cuda.Event()
    end = cp.cuda.Event()
    if args.profiler_capture:
        cp.cuda.profiler.start()
    start.record()
    host_started = time.perf_counter()
    try:
        with allocation_counter, profiler.install(dycore):
            for _ in range(args.profile_steps):
                dycore.step(state, cfg)
    finally:
        host_seconds = time.perf_counter() - host_started
        end.record()
        cp.cuda.runtime.deviceSynchronize()
        if args.profiler_capture:
            cp.cuda.profiler.stop()
    gpu_ms = float(cp.cuda.get_elapsed_time(start, end))
    combined, _fields = _hash_outputs(state)
    return {
        "gpu_ms_per_step": gpu_ms / args.profile_steps,
        "host_submit_ms_per_step": (
            1000.0 * host_seconds / args.profile_steps),
        "sections": profiler.result(
            args.profile_steps, gpu_ms, host_seconds),
        "allocations": allocation_counter.result(args.profile_steps),
    }, combined


def run(args: argparse.Namespace) -> dict[str, Any]:
    import cupy as cp

    timing, combined, fields = _timing_lane(args)
    profile, profile_combined = _profile_lane(args)
    comparable = args.steps == args.profile_steps
    if comparable and combined != profile_combined:
        raise RuntimeError(
            "instrumented and uninstrumented lanes changed output bytes: "
            f"{combined} != {profile_combined}")
    properties = cp.cuda.runtime.getDeviceProperties(0)
    result = {
        "schema": SCHEMA,
        "source": {
            "commit": _git_head(),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "cupy": cp.__version__,
            "numpy": np.__version__,
            "cuda_runtime": int(cp.cuda.runtime.runtimeGetVersion()),
            "cuda_driver": int(cp.cuda.runtime.driverGetVersion()),
            "gpu": _device_name(properties),
            "compute_capability": cp.cuda.Device(0).compute_capability,
        },
        "workload": {
            "seed": args.seed,
            "nx": args.nx,
            "ny": args.ny,
            "nz": args.nz,
            "microphysics": args.microphysics,
            "warmup_steps_per_lane": args.warmup_steps,
            "timed_steps": args.steps,
            "profile_steps": args.profile_steps,
            "profiler_capture": bool(args.profiler_capture),
        },
        "sizing": estimate_run(args),
        "timing": timing,
        "profile": profile,
        "identity": {
            "combined_sha256": combined,
            "profile_combined_sha256": profile_combined,
            "lanes_comparable": comparable,
            "lanes_byte_identical": (
                comparable and combined == profile_combined),
            "fields": fields,
        },
    }
    return result


def main() -> int:
    args = _arguments()
    if args.estimate_only:
        print(json.dumps(
            {"schema": SCHEMA, "sizing": estimate_run(args)},
            indent=2, sort_keys=True))
        return 0
    result = run(args)
    payload = json.dumps(result, indent=2, sort_keys=True) + os.linesep
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
