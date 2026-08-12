"""The measurement that decides what can be reclaimed, and from whom.

Run it::

    python -m tilestream.vram_probe attribute --rung full+MYNN+Noah-MP
    python -m tilestream.vram_probe buffers   --rung full+MYNN+Noah-MP -k 3
    python -m tilestream.vram_probe scaling   --rung full+MYNN+Noah-MP

Three questions, in the order they have to be answered.

1. ATTRIBUTE.  ``attribute`` arms a :class:`tilestream.vram.DeviceLedger`
   before the first CuPy call and walks one full+MYNN+Noah-MP buffer through
   its whole life -- vertical coordinate, ``init_at_rest``, physics
   initialisation, the first step (which fires radiation, cumulus and the
   surface/PBL stack because WRF's cadence predicates are all true at
   ``itimestep == 1``), then steady steps, then a FORCED radiation firing at
   the production cadence.  Every pooled allocation is charged to the source
   line that asked for it.  A category cannot be freed; a line can.

2. PER-PROCESS OR PER-BUFFER.  ``buffers`` builds K identical buffers one
   after another and reports each one's marginal cost.  Buffer 1 pays for
   everything a process pays for exactly once -- the RRTMGP k-distribution
   tables, the cloud-optics tables, the NVRTC module images, the CUDA
   context -- and buffers 2..K pay only for what genuinely scales with the
   number of buffers.  This is the number that decides whether ``nbuffers=2``
   is affordable on a 12 GB card, and nothing in the project had measured
   it.

3. SCALE.  ``scaling`` repeats the buffer measurement at several domain
   sizes and least-squares fits ``bytes = fixed + per_cell * cells``, then
   reports the largest domain that fits a stated budget.  The fit is
   reported with its residuals: a fixed term that is really a per-cell term
   in disguise shows up as a systematic residual, and quoting the intercept
   of a bad fit as "the fixed cost" is exactly the mistake this workstream
   was created to stop repeating.

WHY THE CLOCK IS PUSHED FORWARD FOR THE RADIATION MEASUREMENT
------------------------------------------------------------
At ``radt = 12 min`` and ``dt = 3 s`` radiation fires on step 1 and then not
again until step 241.  Running 240 steps to see the second firing costs
minutes per configuration and measures nothing new about ALLOCATION, whose
shapes are functions of ``(ncol, nz, ngpt)`` and not of the trajectory.  So
the probe sets ``state.elapsed_seconds = stepra * dt`` through
``physics_inventory.set_carrier_scalars`` -- the same setter the tiled driver
uses to keep every tile on the domain's clock -- which makes the next step's
``itimestep`` satisfy ``itimestep % stepra == 1``.  The probe PRINTS
``driver.call_counts`` before and after every window it measures, so a window
where radiation did not actually fire is visible rather than assumed.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from tilestream import vram


#: The physics selector sets, copied from tilestream/test_gate.py so the
#: probe measures the same rungs the bit-exact gate certifies.  Copied and
#: not imported: importing test_gate pulls in the whole gate matrix and its
#: reference cache, which allocates before the ledger can be armed.
PHYSICS_MOIST = dict(moist=True, mp_physics=10, ztop=20000.0)
PHYSICS_FULL = dict(PHYSICS_MOIST, km_opt=4, sf_sfclay_physics=91,
                    bl_pbl_physics=1, bldt=0.0, sf_surface_physics=2,
                    ra_sw_physics=4, ra_lw_physics=4, radt_minutes=12.0,
                    cu_physics=1, cudt_minutes=5.0)
RUNGS: dict[str, dict] = {
    "dry": dict(),
    "mp10": dict(PHYSICS_MOIST),
    "full": dict(PHYSICS_FULL),
    "full+MYNN": dict(PHYSICS_FULL, sf_sfclay_physics=5, bl_pbl_physics=5),
    "full+Noah-MP": dict(PHYSICS_FULL, sf_surface_physics=4),
    "full+MYNN+Noah-MP": dict(PHYSICS_FULL, sf_sfclay_physics=5,
                              bl_pbl_physics=5, sf_surface_physics=4),
}

NZ = 49


def make_cfg(rung: str, nx: int, ny: int, nz: int = NZ):
    from tilestream import harness

    return harness.make_config(nx, ny, nz, **RUNGS[rung])


def _counts(driver) -> dict:
    return dict(getattr(driver, "call_counts", {}) or {})


def _delta_counts(before: dict, after: dict) -> dict:
    return {k: after.get(k, 0) - before.get(k, 0)
            for k in sorted(set(before) | set(after))
            if after.get(k, 0) - before.get(k, 0)}


def force_radiation_due(state, cfg, driver) -> float:
    """Advance the carrier clock to the next radiation-due step.

    Returns the elapsed-seconds value it set.  Uses the driver's own
    ``stepra`` so a cadence change cannot make this silently miss.
    """
    from gpuwm.core.physics import _physics_interval_steps
    from tilestream import physics_inventory as physinv

    stepra = _physics_interval_steps(driver.radt_minutes, cfg.dt)
    elapsed = float(stepra) * float(cfg.dt)
    scalars = physinv.carrier_scalars(state)
    scalars = dict(scalars)
    scalars["elapsed_seconds"] = elapsed
    physinv.set_carrier_scalars(state, scalars)
    return elapsed


# --------------------------------------------------------------------------
# 1. attribute
# --------------------------------------------------------------------------

def attribute(rung: str, nx: int, ny: int, nz: int = NZ, *, top: int = 25,
              steps: int = 2) -> dict:
    """One buffer's whole life, every allocation charged to a source line."""
    import cupy as cp

    from tilestream import harness, physics_inventory as physinv

    ledger = vram.DeviceLedger()
    base = vram.device_snapshot()
    print(vram.format_snapshot(base, "before anything"))

    with ledger:
        cfg = make_cfg(rung, nx, ny, nz)
        with ledger.phase("build state + driver"):
            state, driver = physinv.default_builder(cfg)
        ledger.mark("state + driver built")
        built = ledger.live_by_site().copy()

        with ledger.phase("step 1 (fires everything)"):
            before = _counts(driver)
            harness.run_steps(state, cfg, 1)
            first_counts = _delta_counts(before, _counts(driver))
        ledger.mark("after step 1")
        after_first = ledger.live_by_site().copy()
        peak_first = ledger.peak_bytes

        with ledger.phase("steady steps"):
            before = _counts(driver)
            harness.run_steps(state, cfg, int(steps))
            steady_counts = _delta_counts(before, _counts(driver))
        ledger.mark(f"after {steps} steady steps")

        elapsed = force_radiation_due(state, cfg, driver)
        with ledger.phase("forced radiation step"):
            before = _counts(driver)
            peak_before = ledger.peak_bytes
            harness.run_steps(state, cfg, 1)
            rad_counts = _delta_counts(before, _counts(driver))
            peak_rad = ledger.peak_bytes
        ledger.mark("after forced radiation step")

        ledger.check()
        live = ledger.live_by_site().copy()
        by_phase = ledger.live_by_phase()
        alloc = dict(ledger.alloc_bytes)
        peaks = dict(ledger.peak_live_bytes)
        modules = ledger.live_by_module()
        owners = ledger.live_by_owner()
        chains = dict(ledger.chains)
        inventory = vram.resident_inventory(state)
        grouped = vram.group_inventory(inventory, depth=2)
        end = vram.device_snapshot()

    print()
    print("=" * 78)
    print(f"ATTRIBUTION  rung={rung}  {nx}x{ny}x{nz} = {nx * ny * nz:,} cells")
    print("=" * 78)
    print(f"  cadence firings, step 1        : {first_counts}")
    print(f"  cadence firings, {steps} steady steps: {steady_counts}")
    print(f"  cadence firings, forced step   : {rad_counts} "
          f"(clock pushed to {elapsed:.0f} s)")
    if not rad_counts.get("radiation"):
        print("  *** the forced step did NOT fire radiation; the radiation "
              "peak below is meaningless")
    print()
    print(vram.format_snapshot(end, "after everything"))
    print(f"  ledger live {ledger.live_bytes / 2**30:.3f} GiB, "
          f"ledger peak {ledger.peak_bytes / 2**30:.3f} GiB")
    print(f"  peak during step 1        {peak_first / 2**30:.3f} GiB")
    print(f"  peak during forced radiation "
          f"{peak_rad / 2**30:.3f} GiB (was {peak_before / 2**30:.3f})")
    print(f"  untracked frees {ledger.untracked_frees} "
          f"({ledger.untracked_free_bytes / 2**20:.1f} MiB)")
    print()
    print(vram.format_sites(live, title="RESIDENT AT THE END, by source line",
                            top=top, chains=chains))
    print()
    print(vram.format_sites(modules, title="RESIDENT AT THE END, by module",
                            top=top))
    print()
    print(vram.format_sites(owners,
                            title="RESIDENT AT THE END, by who asked",
                            top=top))
    print()
    print(vram.format_sites(inventory,
                            title="RESIDENT, walked from the state object",
                            top=top))
    print()
    print(vram.format_sites(grouped,
                            title="RESIDENT, rolled up two levels",
                            top=top))
    print(f"  walk total {sum(inventory.values()) / 2**20:.1f} MiB vs "
          f"ledger live {ledger.live_bytes / 2**20:.1f} MiB "
          f"(difference is held outside the state object)")
    print()
    print(vram.format_sites(peaks, title="PEAK SIMULTANEOUS LIVE, by line",
                            top=top))
    print()
    print(vram.format_sites(alloc, title="GREW THE POOL (cudaMalloc)",
                            top=top))
    print()
    print("LIVE BY PHASE")
    for phase, value in sorted(by_phase.items(), key=lambda kv: -kv[1]):
        print(f"  {value / 2**20:10.2f} MiB  {phase}")

    del state, driver
    vram.trim_pool()
    return {
        "rung": rung, "nx": nx, "ny": ny, "nz": nz,
        "cells": nx * ny * nz,
        "first_step_counts": first_counts,
        "steady_counts": steady_counts,
        "forced_counts": rad_counts,
        "live_bytes": ledger.live_bytes,
        "peak_bytes": ledger.peak_bytes,
        "peak_after_build": peak_first,
        "peak_radiation": peak_rad,
        "live_by_site": live,
        "live_by_module": modules,
        "live_by_owner": owners,
        "resident_inventory": inventory,
        "resident_grouped": grouped,
        "peak_live_by_site": peaks,
        "alloc_by_site": alloc,
        "built_by_site": built,
        "after_first_by_site": after_first,
        "snapshot_end": end,
        "snapshot_base": base,
    }


# --------------------------------------------------------------------------
# 2. per-process vs per-buffer
# --------------------------------------------------------------------------

def buffers(rung: str, nx: int, ny: int, nz: int = NZ, *, k: int = 3,
            warmup: int = 1, top: int = 20) -> dict:
    """Marginal device cost of the 1st, 2nd, ... Kth identical tile buffer.

    Each buffer is built exactly the way ``driver.make_physics_tile_state``
    builds one (same builder, same warmup), so the marginal number is the
    real cost of raising ``nbuffers``, not a proxy for it.
    """
    import cupy as cp

    from tilestream import driver as tdriver, harness

    ledger = vram.DeviceLedger()
    rows = []
    held = []
    with ledger:
        cfg = make_cfg(rung, nx, ny, nz)
        tile_cfg = cfg
        for index in range(1, int(k) + 1):
            before_live = ledger.live_bytes
            before_snap = vram.device_snapshot()
            label = f"buffer {index}"
            with ledger.phase(label):
                held.append(tdriver.make_physics_tile_state(
                    tile_cfg, warmup=warmup))
            cp.cuda.runtime.deviceSynchronize()
            after_snap = vram.device_snapshot()
            rows.append({
                "index": index,
                "ledger_delta": ledger.live_bytes - before_live,
                "pool_used_delta": (after_snap["pool_used"]
                                    - before_snap["pool_used"]),
                "pool_total_delta": (after_snap["pool_total"]
                                     - before_snap["pool_total"]),
                "device_delta": (after_snap["device_used"]
                                 - before_snap["device_used"]),
                "device_used": after_snap["device_used"],
            })
        ledger.check()
        by_phase = ledger.live_by_phase()
        per_site_phase: dict[str, dict[str, int]] = {}
        for size, site, phase in ledger._live.values():   # noqa: SLF001
            per_site_phase.setdefault(phase, {})
            per_site_phase[phase][site] = \
                per_site_phase[phase].get(site, 0) + size

    cells = nx * ny * nz
    print()
    print("=" * 78)
    print(f"PER-BUFFER COST  rung={rung}  {nx}x{ny}x{nz} = {cells:,} cells")
    print("=" * 78)
    print(f"  {'buffer':>7s} {'ledger':>12s} {'pool_used':>12s} "
          f"{'pool_total':>12s} {'device':>12s}   B/cell(ledger)")
    for row in rows:
        print(f"  {row['index']:>7d} "
              f"{row['ledger_delta'] / 2**20:>11.1f}M "
              f"{row['pool_used_delta'] / 2**20:>11.1f}M "
              f"{row['pool_total_delta'] / 2**20:>11.1f}M "
              f"{row['device_delta'] / 2**20:>11.1f}M "
              f"  {row['ledger_delta'] / cells:>9.1f}")
    if len(rows) >= 2:
        first = rows[0]["ledger_delta"]
        marginal = [r["ledger_delta"] for r in rows[1:]]
        mean_marginal = sum(marginal) / len(marginal)
        print()
        print(f"  buffer 1 costs {first / 2**20:.1f} MiB; buffers 2..{k} "
              f"cost {mean_marginal / 2**20:.1f} MiB each")
        print(f"  => ONE-TIME (per-process) part of buffer 1: "
              f"{(first - mean_marginal) / 2**20:.1f} MiB "
              f"({(first - mean_marginal) / 2**30:.3f} GiB)")
        print(f"  => PER-BUFFER part: {mean_marginal / 2**20:.1f} MiB "
              f"= {mean_marginal / cells:.1f} B/cell")
    print()
    print("WHAT THE FIRST BUFFER PAID FOR AND THE SECOND DID NOT")
    first_sites = per_site_phase.get("buffer 1", {})
    later_sites: dict[str, int] = {}
    for index in range(2, int(k) + 1):
        for site, value in per_site_phase.get(f"buffer {index}", {}).items():
            later_sites[site] = later_sites.get(site, 0) + value
    n_later = max(int(k) - 1, 1)
    one_time = {site: value - later_sites.get(site, 0) / n_later
                for site, value in first_sites.items()
                if value - later_sites.get(site, 0) / n_later > 1 << 20}
    print(vram.format_sites({s: int(v) for s, v in one_time.items()},
                            title="charged to buffer 1 only", top=top,
                            chains=dict(ledger.chains)))

    del held
    vram.trim_pool()
    return {"rung": rung, "nx": nx, "ny": ny, "nz": nz, "cells": cells,
            "rows": rows, "by_phase": by_phase,
            "one_time_sites": {s: int(v) for s, v in one_time.items()}}


# --------------------------------------------------------------------------
# 3. scaling and the ceiling it implies
# --------------------------------------------------------------------------

def scaling(rung: str, sizes: list[int], nz: int = NZ, *, warmup: int = 1,
            budget_gib: float = 11.0) -> dict:
    """Fit ``bytes = fixed + per_cell * cells`` across domain sizes.

    Reports residuals.  A model whose residuals are not small compared with
    the spread of the data is not a model, and its intercept is not "the
    fixed cost".
    """
    import cupy as cp

    from tilestream import driver as tdriver

    rows = []
    for n in sizes:
        vram.trim_pool()
        ledger = vram.DeviceLedger()
        with ledger:
            cfg = make_cfg(rung, n, n, nz)
            state = tdriver.make_physics_tile_state(cfg, warmup=warmup)
            cp.cuda.runtime.deviceSynchronize()
            ledger.check()
            snap = vram.device_snapshot()
            rows.append({"n": n, "cells": n * n * nz,
                         "live": ledger.live_bytes,
                         "peak": ledger.peak_bytes,
                         "pool_total": snap["pool_total"],
                         "device_used": snap["device_used"]})
            del state
        vram.trim_pool()
        print(f"  {n:>5d}^2  live {rows[-1]['live'] / 2**30:7.3f} GiB  "
              f"peak {rows[-1]['peak'] / 2**30:7.3f}  "
              f"device {rows[-1]['device_used'] / 2**30:7.3f}")

    cells = np.array([r["cells"] for r in rows], dtype=np.float64)
    for key in ("live", "peak", "device_used"):
        y = np.array([r[key] for r in rows], dtype=np.float64)
        design = np.stack([np.ones_like(cells), cells], axis=1)
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        fit = design @ coef
        resid = y - fit
        print()
        print(f"  {key}: fixed {coef[0] / 2**30:.3f} GiB + "
              f"{coef[1]:.1f} B/cell")
        print(f"    residuals (MiB): "
              + ", ".join(f"{r / 2**20:+.1f}" for r in resid))
        print(f"    max |residual| / spread = "
              f"{np.max(np.abs(resid)) / max(np.ptp(y), 1):.4f}")
        budget = budget_gib * 2**30
        if coef[1] > 0:
            fit_cells = (budget - coef[0]) / coef[1]
            side = np.sqrt(max(fit_cells, 0) / nz)
            print(f"    at a {budget_gib:.1f} GiB budget: "
                  f"{fit_cells / 1e6:.1f} Mcell = {side:.0f}^2 x {nz}")
    return {"rung": rung, "rows": rows}


# --------------------------------------------------------------------------
# 4. what it costs to be this process at all
# --------------------------------------------------------------------------

def process_nonpool_bytes() -> int:
    """Device bytes this process holds that the CuPy pool did not allocate.

    The CUDA context, every NVRTC module image, and any library-owned
    allocation.  It is the term a pool-only accounting omits entirely, and it
    is over 2 GiB at full physics -- on a 12 GB card, a sixth of the device
    before a single cell of weather exists.

    Two ways to get it, in order of trust:

    1. ask the driver for THIS pid's device total and subtract the pool.
       Exact, and unaffected by anything else on the card -- but many
       container runtimes report no pid rows at all, in which case
    2. fall back to ``device_used - pool_total``, which is only equal to
       this process's share IF THE CARD IS OTHERWISE IDLE.  The return value
       says which one it used so a caller can qualify the number rather than
       quote it blind.
    """
    import os
    import subprocess

    import cupy as cp

    pid = os.getpid()
    pool_total = int(cp.get_default_memory_pool().total_bytes())
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30, check=True).stdout
    except Exception:                                  # noqa: BLE001
        out = ""
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0] == str(pid):
            # MEASURED on the project's own WSL2 desktop: nvidia-smi finds
            # the pid and then reports its used_memory as the literal string
            # "[N/A]", because per-process accounting is unavailable through
            # WSL's GPU passthrough.  float() raised ValueError here, which
            # `trial` does not catch, so it exited 1, which `ceiling` turns
            # into "not an OOM" and aborts the whole sweep.  The branch
            # documented as the trustworthy one was the only one that could
            # crash, and it crashed exactly where the pid namespace is real.
            try:
                return int(float(parts[1]) * 2**20) - pool_total
            except ValueError:
                break
    return int(vram.device_snapshot()["nonpool"])


def capacity(rung: str, sizes: list[int], nz: int = NZ, *,
             budget_gib: float = 12.0, nbuffers: int = 2,
             rrtmgp_column_chunk: int | None = None) -> dict:
    """Measure the shared and per-buffer terms, then solve the 12 GB question.

    For each tile size it builds ``nbuffers`` buffers twice -- once with each
    buffer owning its workspaces, once with :mod:`tilestream.shared_workspace`
    -- and reports what a streamed run of that tile size would occupy on a
    card of ``budget_gib``.  The process's non-pool footprint is measured
    per configuration rather than assumed, because it is the term that a
    pool-only accounting silently omits and it is over 2 GiB.
    """
    import cupy as cp

    from tilestream import driver as tdriver, harness
    from tilestream import shared_workspace as sw

    rows = []
    for n in sizes:
        vram.trim_pool()
        cfg = make_cfg(rung, n, n, nz)
        record = {"n": n, "cells": n * n * nz}
        for mode in ("private", "shared"):
            vram.trim_pool()
            base = vram.device_snapshot()
            shared = (sw.build(cfg, rrtmgp_column_chunk=rrtmgp_column_chunk)
                      if mode == "shared" else None)
            after_shared = vram.device_snapshot()
            held = []
            marginal = []
            for _ in range(nbuffers):
                before = vram.device_snapshot()
                held.append(tdriver.make_physics_tile_state(
                    cfg, shared=shared))
                cp.cuda.runtime.deviceSynchronize()
                after = vram.device_snapshot()
                marginal.append(after["pool_used"] - before["pool_used"])
            end = vram.device_snapshot()
            record[mode] = {
                "shared_bytes": (after_shared["pool_used"]
                                 - base["pool_used"]),
                "buffer_bytes": marginal,
                "pool_used": end["pool_used"] - base["pool_used"],
                "pool_total": end["pool_total"] - base["pool_total"],
                "nonpool": process_nonpool_bytes(),
            }
            del held, shared
            vram.trim_pool()
        rows.append(record)
        priv, shr = record["private"], record["shared"]
        print(f"  {n:>4d}^2 x{nz}  private: shared {0:8.1f}M  "
              f"buffers {[f'{b / 2**20:.0f}M' for b in priv['buffer_bytes']]}"
              f"  total {priv['pool_used'] / 2**20:8.1f}M")
        print(f"            shared : shared "
              f"{shr['shared_bytes'] / 2**20:8.1f}M  "
              f"buffers {[f'{b / 2**20:.0f}M' for b in shr['buffer_bytes']]}"
              f"  total {shr['pool_used'] / 2**20:8.1f}M  "
              f"({priv['pool_used'] / max(shr['pool_used'], 1):.2f}x less)")
        print(f"            non-pool (context + modules) "
              f"{priv['nonpool'] / 2**30:.3f} GiB")
    return {"rung": rung, "nbuffers": nbuffers, "budget_gib": budget_gib,
            "rows": rows}


# --------------------------------------------------------------------------
# 5. the ceiling: how big a tile (or domain) actually fits
# --------------------------------------------------------------------------

#: Exit code a ``trial`` uses for "this configuration does not fit".  Chosen
#: so a crash (1) and an OOM (3) cannot be confused by the bisector -- a
#: bisector that reads every nonzero exit as "too big" will happily report a
#: ceiling produced entirely by a typo.
EXIT_DOES_NOT_FIT = 3


def trial(rung: str, n: int, nz: int = NZ, *, nbuffers: int = 1,
          share: bool = False, rrtmgp_column_chunk: int | None = None,
          mynn_column_chunk: int | None = None,
          pool_limit_bytes: int | None = None, steps: int = 2) -> int:
    """Build and STEP one configuration; return 0 if it fits, 3 if it does not.

    Building is not enough and never was: ``RESULTS.md`` records a 2400^2
    ``DomainState`` that allocated cleanly and then died inside the first
    ``dycore.step``.  So this runs ``steps`` steps, and the LAST of them is
    forced to be a radiation step -- the largest transient in the whole
    configuration, and one that a short run at the production 12-minute
    cadence would otherwise never reach.  The cadence counters are printed
    so a trial that quietly skipped radiation cannot be mistaken for one
    that survived it.
    """
    import cupy as cp

    from tilestream import driver as tdriver, harness
    from tilestream import shared_workspace as sw

    if pool_limit_bytes:
        cp.get_default_memory_pool().set_limit(size=int(pool_limit_bytes))
    if mynn_column_chunk:
        sw.set_mynn_column_chunk(int(mynn_column_chunk))

    cfg = make_cfg(rung, n, n, nz)
    try:
        shared = (sw.build(cfg, rrtmgp_column_chunk=rrtmgp_column_chunk)
                  if share else None)
        held = [tdriver.make_physics_tile_state(cfg, shared=shared)
                for _ in range(nbuffers)]
        state = held[0]
        driver_obj = state.physics
        before = _counts(driver_obj)
        force_radiation_due(state, cfg, driver_obj)
        harness.run_steps(state, cfg, int(steps))
        counts = _delta_counts(before, _counts(driver_obj))
    except cp.cuda.memory.OutOfMemoryError as exc:
        print(f"  {n}^2 x{nz} nbuffers={nbuffers} share={share}: "
              f"DOES NOT FIT -- {exc}")
        return EXIT_DOES_NOT_FIT
    snap = vram.device_snapshot()
    print(f"  {n}^2 x{nz} nbuffers={nbuffers} share={share} "
          f"rrtmgp_chunk={rrtmgp_column_chunk} mynn_chunk={mynn_column_chunk}"
          f": FITS  pool_used {snap['pool_used'] / 2**30:.3f} GiB  "
          f"pool_total {snap['pool_total'] / 2**30:.3f}  "
          f"shared {(0 if shared is None else shared.nbytes) / 2**30:.3f}  "
          f"nonpool {process_nonpool_bytes() / 2**30:.3f}  "
          f"cadence {counts}")
    if not counts.get("radiation"):
        print("  *** radiation did NOT fire in this trial; the peak is not "
              "the real peak")
        return 1
    del held, state, driver_obj, shared
    vram.trim_pool()
    return 0


def _trial_argv(args, n: int) -> list[str]:
    argv = ["trial", "--rung", args.rung, "--nx", str(n), "--ny", str(n),
            "--nz", str(args.nz), "--nbuffers", str(args.nbuffers),
            "--steps", str(args.steps)]
    if args.share:
        argv.append("--share")
    if args.rrtmgp_column_chunk:
        argv += ["--rrtmgp-column-chunk", str(args.rrtmgp_column_chunk)]
    if args.mynn_column_chunk:
        argv += ["--mynn-column-chunk", str(args.mynn_column_chunk)]
    if args.pool_limit_bytes:
        argv += ["--pool-limit-bytes", str(args.pool_limit_bytes)]
    return argv


def ceiling(args) -> dict:
    """Bisect the largest square extent that fits, one SUBPROCESS per trial.

    In-process retry after an ``OutOfMemoryError`` measures a fragmented
    pool, not a card: the failed attempt's partial allocations, the retained
    blocks it could not reuse and the modules it JIT-compiled all persist,
    and every later trial starts from a different, worse state.  A fresh
    process per trial is the only way the answer is a property of the
    configuration.
    """
    import subprocess
    import sys as _sys

    lo, hi = int(args.low), int(args.high)
    step = int(args.step)
    best = None
    log = []
    while lo <= hi:
        mid = ((lo + hi) // 2 // step) * step
        mid = max(mid, lo)
        proc = subprocess.run(
            [_sys.executable, "-u", "-m", "tilestream.vram_probe"]
            + _trial_argv(args, mid), capture_output=True, text=True)
        tail = proc.stdout.strip().splitlines()[-1:] or [proc.stderr[-300:]]
        print(f"  trial {mid:>5d}^2 -> exit {proc.returncode}: {tail[0]}")
        log.append({"n": mid, "exit": proc.returncode, "line": tail[0]})
        if proc.returncode == 0:
            best = mid
            lo = mid + step
        elif proc.returncode == EXIT_DOES_NOT_FIT:
            hi = mid - step
        else:
            raise RuntimeError(
                f"trial at {mid}^2 failed for a reason that is not an OOM "
                f"(exit {proc.returncode}); the bisection would turn that "
                f"into a ceiling:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
    print(f"  LARGEST THAT FITS: {best}^2 x {args.nz}"
          + ("" if best is None else
             f" = {best * best * args.nz / 1e6:.1f} Mcell"))
    return {"best": best, "log": log, "rung": args.rung,
            "nbuffers": args.nbuffers, "share": bool(args.share),
            "pool_limit_bytes": args.pool_limit_bytes}


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main(argv=None) -> int:
    import cupy as cp

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--rung", default="full+MYNN+Noah-MP",
                        choices=sorted(RUNGS))
    common.add_argument("--nx", type=int, default=128)
    common.add_argument("--ny", type=int, default=128)
    common.add_argument("--nz", type=int, default=NZ)
    common.add_argument("--json", default=None)
    common.add_argument("--top", type=int, default=25)

    p_att = sub.add_parser("attribute", parents=[common])
    p_att.add_argument("--steps", type=int, default=2)
    p_buf = sub.add_parser("buffers", parents=[common])
    p_buf.add_argument("-k", type=int, default=3)
    p_scale = sub.add_parser("scaling", parents=[common])
    p_scale.add_argument("--sizes", type=int, nargs="+",
                         default=[96, 128, 160, 192, 224])
    p_scale.add_argument("--budget-gib", type=float, default=11.0)
    p_cap = sub.add_parser("capacity", parents=[common])
    p_cap.add_argument("--sizes", type=int, nargs="+",
                       default=[96, 128, 192, 256])
    p_cap.add_argument("--budget-gib", type=float, default=12.0)
    p_cap.add_argument("--nbuffers", type=int, default=2)
    p_cap.add_argument("--rrtmgp-column-chunk", type=int, default=None)

    fit = argparse.ArgumentParser(add_help=False)
    fit.add_argument("--rung", default="full+MYNN+Noah-MP",
                     choices=sorted(RUNGS))
    fit.add_argument("--nz", type=int, default=NZ)
    fit.add_argument("--nbuffers", type=int, default=1)
    fit.add_argument("--share", action="store_true")
    fit.add_argument("--rrtmgp-column-chunk", type=int, default=None)
    fit.add_argument("--mynn-column-chunk", type=int, default=None)
    fit.add_argument("--pool-limit-bytes", type=int, default=0)
    fit.add_argument("--steps", type=int, default=2)
    fit.add_argument("--json", default=None)
    p_trial = sub.add_parser("trial", parents=[fit])
    p_trial.add_argument("--nx", type=int, required=True)
    p_trial.add_argument("--ny", type=int, default=0)
    p_ceil = sub.add_parser("ceiling", parents=[fit])
    p_ceil.add_argument("--low", type=int, default=64)
    p_ceil.add_argument("--high", type=int, default=1024)
    p_ceil.add_argument("--step", type=int, default=16)
    sub.add_parser("overhead")

    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    free, total = cp.cuda.runtime.memGetInfo()
    print(f"cupy {cp.__version__}  {name}  "
          f"{free / 2**30:.2f} GiB free of {total / 2**30:.2f}")

    if args.cmd == "overhead":
        print(vram.measure_ledger_overhead())
        return 0

    if args.cmd == "trial":
        return trial(args.rung, args.nx, args.nz, nbuffers=args.nbuffers,
                     share=args.share,
                     rrtmgp_column_chunk=args.rrtmgp_column_chunk,
                     mynn_column_chunk=args.mynn_column_chunk,
                     pool_limit_bytes=args.pool_limit_bytes,
                     steps=args.steps)

    if args.cmd == "ceiling":
        out = ceiling(args)
        if args.json:
            with open(args.json, "w", encoding="utf-8") as handle:
                json.dump(out, handle, indent=1, default=str)
        return 0

    if args.cmd == "attribute":
        out = attribute(args.rung, args.nx, args.ny, args.nz, top=args.top,
                        steps=args.steps)
    elif args.cmd == "buffers":
        out = buffers(args.rung, args.nx, args.ny, args.nz, k=args.k,
                      top=args.top)
    elif args.cmd == "capacity":
        out = capacity(args.rung, args.sizes, args.nz,
                       budget_gib=args.budget_gib, nbuffers=args.nbuffers,
                       rrtmgp_column_chunk=args.rrtmgp_column_chunk)
    else:
        out = scaling(args.rung, args.sizes, args.nz,
                      budget_gib=args.budget_gib)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(out, handle, indent=1, default=str)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
