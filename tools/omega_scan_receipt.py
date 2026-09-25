"""Validation and timing receipt for the per-column Omega kernel.

``gpuwm.core.dycore._omega_ref`` (WRF ``calc_ww_cp``) moved from a ufunc
divergence, a CuPy tree reduction and a CuPy batched ``cumsum`` to one
thread per column in the Fortran's own operation order.  The summation
order changed, so the two are not byte-identical; this tool measures how far
apart they are and what the change costs or saves, on the seeded state of
``tools/benchmark_seeded_step.py``, the way the repository records every
order change: readings beside the re-pin.

Three sub-commands, each writing one JSON receipt:

``validate``
    Two states from one seed, one stepped under the retired construction
    and one under the column kernel.  Reports the one-call Omega
    difference (max relative to max |Omega|, differing words, ULP
    histogram) on both the flat and the map-factor route, then the max
    relative difference of u, v, w, theta', phi', mu' and every moisture
    species after ``--steps`` steps.

``bench``
    Whole-step timing at each grid in ``--grids``: the two arms alternate
    (the order swapping every repetition), ``--reps`` repetitions of
    ``--steps`` steps each, CUDA-event timing, medians.  A repetition during
    which another process held the card is discarded and repeated, so a
    timing never carries a co-tenant.

``kernel``
    The routine in isolation on random data: the retired construction, the
    column kernel, and ``cp.cumsum`` alone, ``--reps`` calls each, medians
    in microseconds.

The retired construction is kept here, verbatim, as the comparison arm.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

import numpy as np

SCHEMA = "gpuwm-omega-column-scan-receipt-v1"
DEFAULT_SEED = 20_260_731          # tools/benchmark_seeded_step.py
DYNAMIC_FIELDS = ("u", "v", "w", "thp", "php", "mup")


def legacy_omega_ref(state, cfg, ru, rv):
    """The construction the column kernel replaced, verbatim.

    ``divv = (dnw * bracket) * msft`` through ufuncs, ``dmdt`` as CuPy's
    tree reduction over the vertical axis, and the recurrence as CuPy's
    batched scan of the pre-added operand ``(c1h*dnw)*dmdt + divv``.
    """
    import cupy as cp
    nz, ny, nx = state.p.shape
    rdx, rdy = 1.0 / cfg.dx, 1.0 / cfg.dy
    ww = state.scratch((nz + 1, ny, nx), "rk_ww")
    dnw = state.dnw[:, None, None]
    c1h = state.c1h[:, None, None]
    divv = dnw * (rdx * (ru[:, :, 1:] - ru[:, :, :-1])
                  + rdy * (rv[:, 1:, :] - rv[:, :-1, :]))
    if state.has_msf:
        divv *= state.msft[None]
    dmdt = divv.sum(axis=0)
    ww[0] = 0.0
    ww[1:nz] = -cp.cumsum((c1h * dnw)[:nz - 1] * dmdt[None] + divv[:nz - 1],
                          axis=0)
    ww[nz] = 0.0
    return ww


class Arms:
    """Swap the engine's ``_omega_ref`` between the two constructions."""

    def __init__(self):
        from gpuwm.core import dycore
        self.dycore = dycore
        self.new = dycore._omega_ref
        self.old = legacy_omega_ref

    def use(self, arm: str):
        self.dycore._omega_ref = self.new if arm == "new" else self.old


def other_gpu_processes() -> list[int]:
    """PIDs of compute processes on the card other than this one."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid",
             "--format=csv,noheader"], capture_output=True, text=True,
            timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    pids = []
    for line in out.splitlines():
        line = line.strip()
        if line.isdigit() and int(line) != os.getpid():
            pids.append(int(line))
    return pids


def build_state(seed, nz, ny, nx, microphysics, steps, msf_amp=0.0):
    from gpuwm.verify.npref import random_acoustic_state
    state, cfg = random_acoustic_state(
        seed=seed, nz=nz, ny=ny, nx=nx, moist=microphysics != 0,
        mp_physics=microphysics, msf_amp=msf_amp)
    cfg = replace(cfg, run_seconds=(steps + 4) * cfg.dt)
    return state, cfg


def ulp_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Whole ULPs between two float32 arrays (0 where equal)."""
    def ordered(x):
        i = x.view(np.int32).astype(np.int64)
        return np.where(i < 0, -(i & 0x7FFFFFFF), i)
    return np.abs(ordered(np.ascontiguousarray(a, dtype=np.float32))
                  - ordered(np.ascontiguousarray(b, dtype=np.float32)))


def field_difference(a, b) -> dict:
    import cupy as cp
    a = cp.asnumpy(a).astype(np.float32)
    b = cp.asnumpy(b).astype(np.float32)
    scale = float(np.abs(a).max())
    diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
    ulps = ulp_distance(a, b)
    words = int(a.size)
    return {
        "max_abs": float(diff.max()),
        "max_abs_of_field": scale,
        "max_relative_to_field_max": (float(diff.max() / scale)
                                      if scale > 0 else 0.0),
        "max_abs_in_ulps_of_field_max": (
            float(diff.max() / np.spacing(np.float32(scale)))
            if scale > 0 else 0.0),
        "differing_words": int((a.view(np.uint32) != b.view(np.uint32)).sum()),
        "words": words,
        "ulp_histogram": {"0": int((ulps == 0).sum()),
                          "1": int((ulps == 1).sum()),
                          "2": int((ulps == 2).sum()),
                          "3_or_more": int((ulps >= 3).sum())},
        "max_ulp": int(ulps.max()) if words else 0,
    }


#: Every moisture mass and number field a scheme can carry; presence on the
#: state decides which are compared.
MOISTURE_CANDIDATES = ("qv", "qc", "qr", "qi", "qs", "qg", "qh",
                       "nc", "nr", "ni", "ns", "ng", "nh")


def moisture_fields(state) -> list[str]:
    import cupy as cp
    names = []
    for name in MOISTURE_CANDIDATES:
        arr = getattr(state, name, None)
        if isinstance(arr, cp.ndarray):
            names.append(name)
    return names


def validate(args) -> dict:
    import cupy as cp
    arms = Arms()
    from gpuwm.core import dycore
    result = {"grid": {"nx": args.nx, "ny": args.ny, "nz": args.nz},
              "seed": args.seed, "microphysics": args.microphysics,
              "steps": args.steps, "one_call_omega": {}, "after_steps": {}}
    # One call, both routes: same state, both constructions.
    for route, msf_amp in (("flat", 0.0), ("map_factor", 0.05)):
        state, cfg = build_state(args.seed, args.nz, args.ny, args.nx,
                                 args.microphysics, 1, msf_amp)
        arms.use("old")
        _ru, _rv, ww = dycore.stage_fluxes(state, cfg)
        ww_old = ww.copy()
        arms.use("new")
        _ru, _rv, ww = dycore.stage_fluxes(state, cfg)
        cp.cuda.runtime.deviceSynchronize()
        result["one_call_omega"][route] = {
            "has_msf": bool(state.has_msf), **field_difference(ww_old, ww)}
        del state
    # Sixty steps, two states.
    states = {}
    for arm in ("old", "new"):
        state, cfg = build_state(args.seed, args.nz, args.ny, args.nx,
                                 args.microphysics, args.steps)
        arms.use(arm)
        for _ in range(args.steps):
            dycore.step(state, cfg)
        cp.cuda.runtime.deviceSynchronize()
        states[arm] = state
    arms.use("new")
    fields = list(DYNAMIC_FIELDS) + moisture_fields(states["new"])
    for name in fields:
        result["after_steps"][name] = field_difference(
            getattr(states["old"], name), getattr(states["new"], name))
    result["fields"] = fields
    return result


def time_steps(state, cfg, steps: int) -> float:
    """CUDA-event milliseconds for ``steps`` whole steps."""
    import cupy as cp
    from gpuwm.core import dycore
    start, end = cp.cuda.Event(), cp.cuda.Event()
    start.record()
    for _ in range(steps):
        dycore.step(state, cfg)
    end.record()
    end.synchronize()
    return float(cp.cuda.get_elapsed_time(start, end))


def bench(args) -> dict:
    import cupy as cp
    from gpuwm.core import dycore
    arms = Arms()
    grids = []
    for spec in args.grids.split(","):
        nx, ny, nz = (int(v) for v in spec.lower().split("x"))
        grids.append((nx, ny, nz))
    result = {"reps": args.reps, "steps_per_rep": args.steps,
              "microphysics": args.microphysics, "seed": args.seed,
              "grids": []}
    for nx, ny, nz in grids:
        state, cfg = build_state(args.seed, nz, ny, nx, args.microphysics,
                                 2 * args.reps * args.steps + 4)
        for arm in ("new", "old", "new"):            # compile both arms
            arms.use(arm)
            dycore.step(state, cfg)
        cp.cuda.runtime.deviceSynchronize()
        samples = {"old": [], "new": []}
        discarded = 0
        rep = 0
        while rep < args.reps:
            order = ("old", "new") if rep % 2 == 0 else ("new", "old")
            block = {}
            shared = False
            for arm in order:
                before = other_gpu_processes()
                arms.use(arm)
                ms = time_steps(state, cfg, args.steps)
                after = other_gpu_processes()
                if before or after:
                    shared = True
                block[arm] = ms / args.steps
            if shared:
                discarded += 1
                if discarded > 20:
                    raise RuntimeError("the card was shared through 20 "
                                       "repetitions; no clean timing")
                continue
            for arm in order:
                samples[arm].append(block[arm])
            rep += 1
        pool = cp.get_default_memory_pool()
        med_old = statistics.median(samples["old"])
        med_new = statistics.median(samples["new"])
        result["grids"].append({
            "grid": f"{nx}x{ny}x{nz}", "nx": nx, "ny": ny, "nz": nz,
            "cumsum_ms_per_step_median": med_old,
            "scan_ms_per_step_median": med_new,
            "speedup": med_old / med_new,
            "saved_ms_per_step": med_old - med_new,
            "samples_ms_per_step": samples,
            "discarded_shared_repetitions": discarded,
            "pool_used_bytes": int(pool.used_bytes()),
        })
        arms.use("new")
        del state
        pool.free_all_blocks()
    return result


def kernel(args) -> dict:
    import cupy as cp
    from types import SimpleNamespace
    from gpuwm.core import dycore
    nz, ny, nx = args.nz, args.ny, args.nx
    rng = np.random.default_rng(args.seed)
    ru = cp.asarray(rng.standard_normal((nz, ny, nx + 1)).astype(np.float32))
    rv = cp.asarray(rng.standard_normal((nz, ny + 1, nx)).astype(np.float32))
    dnw = cp.asarray((-np.full(nz, 1.0 / nz)).astype(np.float32))
    c1h = cp.asarray(np.ones(nz, np.float32))
    slots = {}

    def scratch(shape, slot, dtype=None):
        if slot not in slots:
            slots[slot] = cp.zeros(shape, dtype=np.float32)
        return slots[slot]

    state = SimpleNamespace(p=SimpleNamespace(shape=(nz, ny, nx)),
                            scratch=scratch, dnw=dnw, c1h=c1h,
                            has_msf=False, msft=cp.ones((ny, nx), np.float32))
    cfg = SimpleNamespace(dx=3000.0, dy=3000.0)
    src = cp.asarray(rng.standard_normal((nz - 1, ny, nx)).astype(np.float32))

    def timed(fn) -> float:
        for _ in range(3):
            fn()
        cp.cuda.runtime.deviceSynchronize()
        samples = []
        for _ in range(args.reps):
            start, end = cp.cuda.Event(), cp.cuda.Event()
            start.record()
            fn()
            end.record()
            end.synchronize()
            samples.append(1000.0 * float(cp.cuda.get_elapsed_time(start, end)))
        return statistics.median(samples)

    arms = Arms()
    while other_gpu_processes():
        time.sleep(30)
    return {
        "grid": f"{nx}x{ny}x{nz}", "reps": args.reps,
        "legacy_omega_ref_us": timed(
            lambda: legacy_omega_ref(state, cfg, ru, rv)),
        "column_scan_omega_ref_us": timed(
            lambda: arms.new(state, cfg, ru, rv)),
        "cp_cumsum_alone_us": timed(lambda: cp.cumsum(src, axis=0)),
        "moved_bytes_cumsum": int(2 * src.nbytes),
    }


def environment() -> dict:
    import cupy as cp
    from gpuwm import runtime_manifest
    properties = cp.cuda.runtime.getDeviceProperties(0)
    name = properties["name"]
    identity = runtime_manifest.provenance(Path(__file__).resolve().parents[1])
    return {
        "gpu": name.decode() if isinstance(name, bytes) else str(name),
        "driver": int(cp.cuda.runtime.driverGetVersion()),
        "cuda_runtime": int(cp.cuda.runtime.runtimeGetVersion()),
        "cupy": cp.__version__, "numpy": np.__version__,
        "python": sys.version.split()[0], "platform": platform.platform(),
        "git_commit": identity.get("git_commit"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("validate", "bench", "kernel"))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--ny", type=int, default=64)
    parser.add_argument("--nz", type=int, default=32)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--reps", type=int, default=8)
    parser.add_argument("--microphysics", type=int, default=10)
    parser.add_argument("--grids", default="250x200x49,320x256x49,480x384x49")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "bench" and args.steps == 60:
        args.steps = 20
    if args.command == "kernel" and args.reps == 8:
        args.reps = 50
    started = time.perf_counter()
    body = {"validate": validate, "bench": bench, "kernel": kernel}[args.command](args)
    result = {"schema": SCHEMA, "command": args.command,
              "environment": environment(), "result": body,
              "elapsed_seconds": round(time.perf_counter() - started, 1)}
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
