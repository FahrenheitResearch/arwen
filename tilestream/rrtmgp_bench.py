"""Price the RRTMGP reclamations: bytes saved, milliseconds spent, digest kept.

ONE TRIAL PER PROCESS.  A CuPy pool never shrinks, so two configurations
measured in one process share a high-water mark and the second one is free.
Every row here is a fresh subprocess and the driver reads JSON back.

THE PEAK MODEL, STATED BEFORE IT IS MEASURED
--------------------------------------------
Write ``W(c)`` for the workspace at column chunk ``c``, ``D_rad(c)`` for
everything else the process holds at the instant radiation is firing, and
``D_norm`` for what it holds at the peak of an ordinary step.  Radiation
fires once every ``radt/dt`` steps; ``D_norm`` therefore sets the occupancy
of 239 steps out of 240 and ``D_rad`` of one.

    persistent peak = max(D_rad(c), D_norm) + W(c)
    lazy peak       = max(D_rad(c) + W(c), D_norm)
    saving          = min(W(c), max(0, D_norm - D_rad(c)))

Read the third line before believing the second.  **At the shipped chunk the
saving is zero** -- ``D_rad`` exceeds ``D_norm`` by more than ``W``, the peak
is on the radiation step, and releasing bytes between firings moves a number
no capacity bisection can see.  Lazy release only starts paying once the
chunk has been cut far enough that an ordinary step is the busiest step, and
then it pays its full size.  The two reclamations multiply; neither is worth
much alone.  This module measures ``D_norm``, ``D_rad(c)`` and ``W(c)``
separately so that claim is arithmetic rather than assertion.

WHAT COUNTS AS THE PEAK
-----------------------
``cudaMemGetInfo``: total minus free, on the device, sampled at instants.
Not pool bookkeeping -- a private pool's bytes are invisible to the default
pool's counters, and an accounting that cannot see the thing being reclaimed
would report the reclamation as free.  The device is asked directly.

The sample DURING the firing is taken by ``peak_probe``, called from the
adapter's release hook before a single byte goes back, so the radiation peak
is observed while it is still standing rather than reconstructed afterwards.

DISCIPLINE
----------
Every trial prints its radiation fire count and its state digest.  A window
in which radiation never fired is not a radiation measurement, and three of
this project's false results came from exactly that window.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time


NZ = 49
PHYSICS_MOIST = dict(moist=True, mp_physics=10, ztop=20000.0)
PHYSICS_FULL = dict(PHYSICS_MOIST, km_opt=4, sf_sfclay_physics=91,
                    bl_pbl_physics=1, bldt=0.0, sf_surface_physics=2,
                    ra_sw_physics=4, ra_lw_physics=4, radt_minutes=12.0,
                    cu_physics=1, cudt_minutes=5.0)
RUNGS = {
    "full": PHYSICS_FULL,
    "full+MYNN": dict(PHYSICS_FULL, sf_sfclay_physics=5, bl_pbl_physics=5),
    "full+MYNN+Noah-MP": dict(PHYSICS_FULL, sf_sfclay_physics=5,
                              bl_pbl_physics=5, sf_surface_physics=4),
}

#: Production cadence: radt 12 min at dt 3 s.  Printed with every averaged
#: cost so a reader can redo the weighting for their own cadence.
STEPS_PER_FIRING = 240


def device_used_bytes() -> int:
    import cupy as cp

    free, total = cp.cuda.runtime.memGetInfo()
    return int(total - free)


def _sync():
    import cupy as cp

    cp.cuda.runtime.deviceSynchronize()


# --------------------------------------------------------------------------
# one trial
# --------------------------------------------------------------------------

def trial(*, rung: str, nx: int, ny: int, nz: int, mode: str,
          column_chunk: int | None, fire: bool, steps: int,
          hazard: str | None = None, warmup: int = 2,
          cycles: int = 3, poison: str | None = None) -> dict:
    """Measure one configuration: two peaks, two step costs, one digest.

    ``mode`` is ``"none"`` (no shared workspace -- the shipped tilestream
    path, which allocates its per-chunk cubes fresh inside every call),
    ``"persistent"`` (``SharedRRTMGPChunkWorkspace``, the vanilla model
    path), or ``"lazy"`` (this lane's release-after-firing workspace).

    THE TWO PEAKS ARE MEASURED SEPARATELY AND BOTH FROM A TRIMMED POOL.
    ``peak_ordinary`` is the device high-water over steps on which radiation
    does not fire; ``peak_radiation`` is the high-water over a step on which
    it does, sampled by the release hook while the workspace is still
    standing.  A CuPy pool never shrinks, so measuring them in sequence
    without ``free_all_blocks`` in between would report the first as the
    second.  Both are therefore preceded by a trim, which also makes the two
    numbers structural -- what this configuration NEEDS -- rather than a
    record of whatever the pool happened to be holding.

    ``fire`` still exists and still gates the radiation window, so a caller
    can produce a deliberately radiation-free window and see the fire count
    come back zero rather than trusting that it did.
    """
    import cupy as cp

    from tilestream import harness, physics_inventory as physinv
    from tilestream.rrtmgp_lazy import LazyRRTMGPChunkWorkspace, attach_lazy

    cfg = harness.make_config(nx, ny, nz, **RUNGS[rung])
    state, driver = physinv.default_builder(cfg)
    p_top = float(state.p_top)
    ncol = nx * ny
    mempool = cp.get_default_memory_pool()

    workspace = None
    if mode != "none":
        chunk = int(column_chunk or _default_chunk())
        chunk = max(1, min(chunk, ncol))
        if mode == "persistent":
            from gpuwm.core.model import SharedRRTMGPChunkWorkspace

            workspace = SharedRRTMGPChunkWorkspace(
                nz=nz, column_chunk=chunk, p_top=p_top)
        elif mode == "lazy":
            workspace = LazyRRTMGPChunkWorkspace(
                nz=nz, column_chunk=chunk, p_top=p_top, hazard=hazard)
        elif mode in ("tight", "tight+lazy"):
            from tilestream import rrtmgp_tight

            workspace = rrtmgp_tight.build(
                nz, chunk, p_top, tight=True, lazy=mode.endswith("lazy"),
                poison=poison)
        else:
            raise ValueError(f"unknown mode {mode!r}")
        attach_lazy(state, workspace)

    # Warm-up.  Step 1 fires every scheme and compiles every kernel, and a
    # radiation call is ~150 NVRTC modules; an unwarmed radiation step in a
    # timed window is a compile measurement wearing a physics label.  Two
    # forced firings, because the first one also populates the trace-gas and
    # interpolation-metadata caches.
    harness.run_steps(state, cfg, warmup)
    for _ in range(2):
        _force_radiation_due(state, cfg, driver)
        harness.run_steps(state, cfg, 1)
    _sync()

    rad_peak_samples: list[int] = []
    rad_pool_samples: list[int] = []
    if workspace is not None and hasattr(workspace, "on_call_end"):
        original = workspace.on_call_end

        def hooked():
            _sync()
            rad_peak_samples.append(device_used_bytes())
            rad_pool_samples.append(int(mempool.total_bytes()))
            return original()

        workspace.on_call_end = hooked

    before = dict(getattr(driver, "call_counts", {}) or {})

    # ---- ordinary steps, from a trimmed pool -----------------------------
    mempool.free_all_blocks()
    _sync()
    baseline = device_used_bytes()
    ordinary_ms: list[float] = []
    ordinary_samples: list[int] = []
    for _ in range(max(1, steps)):
        _sync()
        t = time.perf_counter()
        harness.run_steps(state, cfg, 1)
        _sync()
        ordinary_ms.append((time.perf_counter() - t) * 1e3)
        ordinary_samples.append(device_used_bytes())
    peak_ordinary = max(ordinary_samples)
    pool_ordinary = int(mempool.total_bytes())

    # ---- radiation steps, each from a trimmed pool -----------------------
    radiation_ms: list[float] = []
    radiation_samples: list[int] = []
    if fire:
        for _ in range(max(1, cycles)):
            mempool.free_all_blocks()
            _sync()
            _force_radiation_due(state, cfg, driver)
            _sync()
            t = time.perf_counter()
            harness.run_steps(state, cfg, 1)
            _sync()
            radiation_ms.append((time.perf_counter() - t) * 1e3)
            radiation_samples.append(device_used_bytes())
    # The in-call probe is the honest radiation peak for a lazy workspace;
    # for a persistent one the post-step sample already contains it.
    peak_radiation = max(radiation_samples + rad_peak_samples, default=0)
    pool_radiation = max(rad_pool_samples + [int(mempool.total_bytes())])

    counts = {k: int(v) - int(before.get(k, 0))
              for k, v in (getattr(driver, "call_counts", {}) or {}).items()
              if int(v) - int(before.get(k, 0))}
    fired = int(counts.get("radiation", 0))

    _sync()
    resident_between = (int(workspace.resident_bytes)
                        if hasattr(workspace, "resident_bytes") else None)
    w = int(workspace.nbytes) if workspace is not None else 0

    def med(values):
        return sorted(values)[len(values) // 2] if values else None

    rad_ms = med(radiation_ms)
    ord_ms = med(ordinary_ms)
    out = {
        "rung": rung, "nx": nx, "ny": ny, "nz": nz, "ncol": ncol,
        "mode": mode,
        "column_chunk": (int(workspace.column_chunk)
                         if workspace is not None else None),
        "hazard": hazard,
        "fire_requested": bool(fire),
        "radiation_firings": fired,
        "cadence_firings": counts,
        "workspace_bytes": w,
        "workspace_resident_between_firings": resident_between,
        "workspace_allocations": int(getattr(workspace, "allocations", 0)),
        "workspace_releases": int(getattr(workspace, "releases", 0)),
        "hazard_firings": int(getattr(workspace, "hazard_firings", 0)),
        "poison": poison,
        "poison_firings": int(getattr(workspace, "poison_firings", 0)),
        "baseline_device_bytes": baseline,
        "peak_ordinary_bytes": peak_ordinary,
        "peak_radiation_bytes": peak_radiation,
        "pool_ordinary_bytes": pool_ordinary,
        "pool_radiation_bytes": pool_radiation,
        "radiation_step_ms": rad_ms,
        "ordinary_step_ms": ord_ms,
        "radiation_step_ms_all": radiation_ms,
        "ordinary_step_ms_all": ordinary_ms,
        "radiation_over_ordinary": (rad_ms / ord_ms
                                    if rad_ms and ord_ms else None),
        "amortised_ms_per_step": (
            (rad_ms + (STEPS_PER_FIRING - 1) * ord_ms) / STEPS_PER_FIRING
            if rad_ms and ord_ms else None),
        "steps_per_firing": STEPS_PER_FIRING,
        "digest": harness.hash_state(state),
        "p_top": p_top,
    }
    return out


def _default_chunk() -> int:
    from gpuwm.config import DEFAULT_COLUMN_CHUNK

    return int(DEFAULT_COLUMN_CHUNK)


def _force_radiation_due(state, cfg, driver) -> float:
    from gpuwm.core.physics import _physics_interval_steps
    from tilestream import physics_inventory as physinv

    stepra = _physics_interval_steps(driver.radt_minutes, cfg.dt)
    elapsed = float(stepra) * float(cfg.dt)
    scalars = dict(physinv.carrier_scalars(state))
    scalars["elapsed_seconds"] = elapsed
    physinv.set_carrier_scalars(state, scalars)
    return elapsed


# --------------------------------------------------------------------------
# subprocess driver
# --------------------------------------------------------------------------

def run_trial_subprocess(**kwargs) -> dict:
    """Run one :func:`trial` in a fresh interpreter and return its JSON."""
    argv = [sys.executable, "-m", "tilestream.rrtmgp_bench", "trial"]
    for key, value in kwargs.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                argv.append(flag)
        elif value is not None:
            argv += [flag, str(value)]
    proc = subprocess.run(argv, capture_output=True, text=True)
    marker = "@@JSON@@"
    for line in proc.stdout.splitlines():
        if line.startswith(marker):
            return json.loads(line[len(marker):])
    return {"error": True, "returncode": proc.returncode,
            "stdout": proc.stdout[-4000:], "stderr": proc.stderr[-4000:]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("trial")
    t.add_argument("--rung", default="full", choices=sorted(RUNGS))
    t.add_argument("--nx", type=int, default=128)
    t.add_argument("--ny", type=int, default=128)
    t.add_argument("--nz", type=int, default=NZ)
    t.add_argument("--mode", default="persistent",
                   choices=("none", "persistent", "lazy", "tight",
                            "tight+lazy"))
    t.add_argument("--column-chunk", type=int, default=None)
    t.add_argument("--fire", action="store_true")
    t.add_argument("--steps", type=int, default=6)
    t.add_argument("--warmup", type=int, default=2)
    t.add_argument("--cycles", type=int, default=3)
    t.add_argument("--hazard", default=None)
    t.add_argument("--poison", default=None)
    args = parser.parse_args(argv)

    if args.cmd == "trial":
        out = trial(rung=args.rung, nx=args.nx, ny=args.ny, nz=args.nz,
                    mode=args.mode, column_chunk=args.column_chunk,
                    fire=args.fire, steps=args.steps, warmup=args.warmup,
                    cycles=args.cycles, hazard=args.hazard,
                    poison=args.poison)
        print("@@JSON@@" + json.dumps(out))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
