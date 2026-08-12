"""What a full-physics domain ACTUALLY costs, against both models that price it.

THE QUESTION THIS SETTLES
-------------------------
:mod:`tilestream.ledger_ladder` shows that ArWen's two VRAM models disagree by
a factor that grows with the domain: preflight prices the full(real74) rung at
~887 B/cell of pool estimate (771 B/cell before its 15% allocator headroom)
and adds a 2.9 GiB non-pool intercept on top; autoplan prices the same rung at
541 B/cell plus a 3.6 GiB fixed term, inflated 6%.  Between those two numbers
there is a band of domain sizes on every card where autoplan says "resident is
fine" -- so ``[tiles] mode = "auto"`` does not stream -- and preflight then
refuses the run outright.

Arithmetic cannot say which of them is right.  A card can.  This probe builds
ONE resident domain at each requested size, steps it until radiation and
cumulus have both fired several times, and reads the pool and the device
against both predictions.  A size inside the disagreement band is the whole
point: if the domain allocates and steps, preflight is refusing a run that
works; if it OOMs, autoplan is planning a run that cannot.

WHY IT STEPS RATHER THAN ALLOCATING
-----------------------------------
"It allocated" is not "it runs": on these boxes the state allocates and the
FIRST step then OOMs on per-step scratch, and Kain-Fritsch's ``cumulus/w0avg``
and the radiation column packer are allocated LAZILY on first use.  A
zero-step allocation probe -- which is exactly what ``gpuwm check --alloc``
is -- therefore measures a footprint no running forecast ever has.  So this
one steps, and prints the physics driver's own call counts so a reader can
see that radiation and cumulus fired inside the measured window rather than
taking it on trust.

The cadence is set to the STEP cadence of ``configs/real74_d01.toml``
(radt 12 min / cudt 5 min at dt = 60 s -> every 12 and every 5 steps), not to
its wall-clock cadence, because this harness's dx = 500 m needs dt = 3 s and
the wall-clock cadence would then fire radiation once every 240 steps.

NO TIMINGS ARE QUOTED FROM THIS FILE.  It measures bytes.  Several of the
window sizes here are below the ~500-cell floor under which a timing measures
an idle GPU rather than the code.

WHAT IT MEASURED
----------------
RTX 5090 (shared with three other lanes, which is why the numbers below are
POOL numbers -- see :func:`resident`), harness ``full`` rung, nz = 49, 30
steps, radiation firing 3 times and cumulus 7 times inside every window, and
a state digest recorded per rung so a run that skipped work would show::

  n     pf alloc  pf held  ap device | pool used pk  pool held pk | u/alloc  h/held  h/ap
  256^2    3.343    3.343    5.491   |    1.603         3.117     |  0.479   0.932  0.568
  352^2    5.571    5.571    7.019   |    4.589         6.656     |  0.824   1.195  0.948

Three things fall out, and the third is the one that matters.

1.  Preflight's ``alloc_estimate`` holds as an upper bound on pool USED at
    both rungs, and tightens with size (0.479 -> 0.824).  That is the N0
    gate's ``alloc_measured_le_estimate`` leg, and it passes.
2.  Preflight's HELD projection does not.  On Linux the tier-2 retention term
    is zero, so ``held_projection == alloc_estimate``, and the measured pool
    held peak is already 1.195x it at 352^2.  Pool-held is device memory the
    process is holding, so this is a real over-run of a published projection,
    not a bookkeeping distinction.
3.  Autoplan is the OPTIMISTIC model, not preflight.  At 352^2 the CuPy pool
    alone holds 6.656 GiB against autoplan's 7.019 GiB prediction for the
    WHOLE DEVICE -- 0.948 of it -- leaving 0.36 GiB for a CUDA context that
    autoplan's own constant puts at 0.39 GiB, before the local-memory backing
    store is counted at all.  So in the band where the two models disagree
    (:mod:`tilestream.ledger_ladder`), it is ``auto``'s "resident is fine"
    that is wrong and preflight's refusal that is defensible.  Two rungs on
    one card is a direction, not a re-fit; the fix belongs on autoplan's side
    and wants the 29-point measurement redone before anything is changed.
"""

from __future__ import annotations

import argparse
import json
import sys

GIB = 1 << 30
MIB = 1 << 20

#: radt / cudt in MINUTES that reproduce real74_d01's STEP cadence at the
#: harness dt.  d01 runs dt = 60 s with radt = 12 min and cudt = 5 min, i.e.
#: radiation every 12 steps and cumulus every 5.
def _cadence_overrides(dt: float) -> dict:
    return {"radt_minutes": 12.0 * dt / 60.0,
            "cudt_minutes": 5.0 * dt / 60.0}


def _device_used() -> int:
    import cupy as cp

    free, total = cp.cuda.runtime.memGetInfo()
    return total - free


def _free() -> int:
    import cupy as cp

    return cp.cuda.runtime.memGetInfo()[0]


def _predictions(cfg, free_bytes: int) -> dict:
    """Both models' numbers for ``cfg`` on a card with ``free_bytes`` free."""
    import dataclasses
    from datetime import datetime

    from gpuwm.core import preflight as pf
    from gpuwm.experiment import experiment_from_run_config
    from tilestream import autoplan as A

    exp = experiment_from_run_config(cfg, datetime(1974, 4, 3, 12))
    est = pf.estimate_experiment(exp)
    fp = A.footprint_for(cfg)
    cells = int(cfg.nx) * int(cfg.ny) * int(cfg.nz)
    reserve = pf.ReservePolicy.n0_alloc(
        exp, estimate_bytes=est.alloc_estimate_bytes)
    machine = A.Machine(vram_bytes=free_bytes, host_bytes=1 << 40)
    return {
        "cells": cells,
        "pf_alloc": int(est.alloc_estimate_bytes),
        "pf_held": int(est.held_projection_bytes),
        "pf_footprint": int(est.footprint_projection_bytes),
        "pf_envelope": int(est.peak_envelope_bytes),
        "pf_nonpool": int(est.envelope_intercept_bytes),
        "pf_budget": int(reserve.budget_bytes(free_bytes)),
        "pf_refuses": bool(est.peak_envelope_bytes > free_bytes),
        "ap_resident": int(fp.resident_bytes(cells)),
        "ap_budget": int(machine.vram_budget_bytes),
        "ap_streams": bool(fp.resident_bytes(cells)
                           > machine.vram_budget_bytes),
    }


def resident(sizes, *, nz: int, steps: int, rung: str,
             report=print) -> list[dict]:
    """Build, step and measure one resident domain per size."""
    import cupy as cp

    from tilestream import autoplan as A
    from tilestream import harness as H
    from tilestream import physics_inventory as physinv

    out = []
    for n in sizes:
        cp.get_default_memory_pool().free_all_blocks()
        cp.cuda.runtime.deviceSynchronize()
        base_used = _device_used()
        free_now = _free()
        cfg = A._config_for_rung(n, n, nz, rung,
                                 **_cadence_overrides(3.0))
        pred = _predictions(cfg, free_now)
        report(f"\n--- resident {rung} {n}^2 x {nz} "
               f"({pred['cells'] / 1e6:.2f} Mcell) ---")
        report(f"    free before      {free_now / GIB:8.3f} GiB   "
               f"(other residency {base_used / GIB:.3f} GiB)")
        report(f"    preflight alloc  {pred['pf_alloc'] / GIB:8.3f} GiB   "
               f"held {pred['pf_held'] / GIB:.3f}   "
               f"footprint {pred['pf_footprint'] / GIB:.3f}   "
               f"envelope {pred['pf_envelope'] / GIB:.3f} GiB   "
               f"REFUSES={pred['pf_refuses']}")
        report(f"    autoplan resident{pred['ap_resident'] / GIB:8.3f} GiB   "
               f"budget   {pred['ap_budget'] / GIB:.3f} GiB   "
               f"STREAMS={pred['ap_streams']}")
        row = dict(n=n, nz=nz, rung=rung, steps=steps, **pred)
        pool = cp.get_default_memory_pool()
        # THE POOL, NOT THE DEVICE.  ``cudaMemGetInfo``'s total-minus-free is
        # the whole CARD, and every card in this project is shared: the first
        # run of this probe reported 24.575 GiB "measured" for a 256^2 domain
        # whose pool held 3.117, because three other lanes were on the same
        # 5090.  The CuPy pool is process-local and cannot be contaminated,
        # so the comparison that means anything is pool-used against
        # preflight's alloc estimate -- which is exactly the N0 gate's own
        # `alloc_measured_le_estimate` leg, priced against a POOL number by
        # construction.  Peaks are sampled every step, because the estimate
        # is a peak and an end-of-run reading is not.
        used_peak = held_peak = 0
        try:
            state, drv = physinv.default_builder(cfg, 4242)
            cp.cuda.runtime.deviceSynchronize()
            after_build = pool.used_bytes()
            for _ in range(int(steps)):
                H.run_steps(state, cfg, 1)
                used_peak = max(used_peak, pool.used_bytes())
                held_peak = max(held_peak, pool.total_bytes())
            cp.cuda.runtime.deviceSynchronize()
        except cp.cuda.memory.OutOfMemoryError as error:
            report(f"    OOM: {error}")
            row.update(ok=False, error=str(error)[:200])
            out.append(row)
            state = drv = None
            cp.get_default_memory_pool().free_all_blocks()
            continue
        contaminated = _device_used() - base_used
        calls = dict(drv.call_counts)
        digest = H.hash_state(state)
        row.update(
            ok=True,
            pool_used_peak=int(used_peak),
            pool_held_peak=int(held_peak),
            after_build_pool=int(after_build),
            device_total_minus_free=int(contaminated),
            call_counts=calls,
            digest=digest,
        )
        report(f"    POOL used peak   {used_peak / GIB:8.3f} GiB   "
               f"held peak {held_peak / GIB:.3f} GiB   "
               f"(after build, before stepping {after_build / GIB:.3f})")
        report(f"    pool-used-peak / preflight-alloc "
               f"{used_peak / pred['pf_alloc']:.3f}   "
               f"pool-held-peak / preflight-held "
               f"{held_peak / pred['pf_held']:.3f}")
        report(f"    pool-held-peak vs autoplan's WHOLE-DEVICE prediction "
               f"{held_peak / pred['ap_resident']:.3f} "
               f"({held_peak / GIB:.3f} vs {pred['ap_resident'] / GIB:.3f} "
               "GiB; autoplan's number has to cover the CUDA context and "
               "the local-memory backing store as well as the pool)")
        report(f"    device total-minus-free {contaminated / GIB:8.3f} GiB "
               "-- CONTAMINATED on a shared card, recorded not used")
        report(f"    physics fired in the measured window: {calls}")
        report(f"    digest {digest[:16]}")
        out.append(row)
        del state, drv
        cp.get_default_memory_pool().free_all_blocks()
    return out


def streamed(n: int, tile: int, *, nz: int, steps: int, rung: str,
             nbuffers: int = 2, report=print) -> dict:
    """The same domain, streamed, measured against the SAME preflight number.

    This is what the memory ledger would be comparing against if it had any
    opinion about streaming: a preflight estimate that prices a whole
    resident domain, and a live allocation that is ``nbuffers`` tile buffers.
    """
    import cupy as cp

    from gpuwm.core import streaming
    from tilestream import autoplan as A
    from tilestream import driver as _driver
    from tilestream import gather
    from tilestream import harness as H
    from tilestream import physics_inventory as physinv

    cp.get_default_memory_pool().free_all_blocks()
    cp.cuda.runtime.deviceSynchronize()
    free_now = _free()
    cfg = A._config_for_rung(n, n, nz, rung, **_cadence_overrides(3.0))
    pred = _predictions(cfg, free_now)
    halo = H.halo_radius(cfg)
    window = tile + 2 * halo
    report(f"\n--- streamed {rung} {n}^2 x {nz}, tile {tile}, halo {halo}, "
           f"window {window} ---")
    if window < 500:
        report("    NOTE: compute window below the 500-cell floor.  This "
               "measures BYTES, and no timing is quoted from it.")
    report(f"    preflight prices the RESIDENT domain at "
           f"{pred['pf_alloc'] / GIB:.3f} GiB pool / "
           f"{pred['pf_envelope'] / GIB:.3f} GiB envelope")

    options = streaming.StreamingOptions(
        mode="on", tile_nx=tile, tile_ny=tile, nbuffers=nbuffers,
        store="host")
    decision = streaming.decide(cfg, options)
    report(f"    decision: {decision.explain()}")

    # THE HONESTY THIS MEASUREMENT NEEDS.  A real out-of-core run fills its
    # pinned store from the ingest and never builds a resident domain at all
    # -- that is the whole premise, since no such domain fits.  This harness
    # DOES build one, because `attach` copies a prepared DomainState's
    # carriers into the store and `StreamedDomain.__call__` checks it is
    # still handed that same object.  So the device figure below is
    # unavoidably "resident domain + tile buffers", and the number that
    # answers the ledger question is the DELTA across `attach`: what the
    # streaming machinery itself costs on the card.
    pool = cp.get_default_memory_pool()
    domain, drv = physinv.default_builder(cfg, 4242)
    cp.cuda.runtime.deviceSynchronize()
    resident_only = pool.total_bytes()
    geo_inv = {k: gather.pinned_copy(v)
               for k, v in _driver.geography_inventory(domain).items()}
    scalars = physinv.carrier_scalars(domain)

    def build(state, run_cfg, dec):
        return streaming.attach(
            state, run_cfg, dec,
            tile_state_factory=lambda tile_cfg: physinv.default_builder(
                tile_cfg, 4242)[0],
            geography=geo_inv, scalars=scalars)

    stepper = streaming.make_stepper(domain, cfg, options, decision=decision,
                                     build=build)
    assert streaming.is_streaming(stepper)
    cp.cuda.runtime.deviceSynchronize()
    attached = pool.total_bytes()
    measured = attached
    for _ in range(int(steps)):
        stepper(domain, cfg, refl_10cm_due=False)
        measured = max(measured, pool.total_bytes())
    cp.cuda.runtime.deviceSynchronize()
    store_bytes = sum(int(a.nbytes) for a in stepper.store.values())
    calls = dict(physinv.carrier_scalars(domain).get("call_counts", {}))
    buffers_only = measured - resident_only
    report(f"    pool held: resident domain alone   "
           f"{resident_only / GIB:8.3f} GiB")
    report(f"    pool held: + tile buffers attached "
           f"{attached / GIB:8.3f} GiB")
    report(f"    pool held peak over {steps} streamed steps "
           f"{measured / GIB:8.3f} GiB")
    report(f"    => the STREAMING machinery costs "
           f"{buffers_only / GIB:.3f} GiB on this card")
    report(f"    pinned host store               "
           f"{store_bytes / GIB:8.3f} GiB")
    report(f"    preflight's ESTIMATE for this domain is "
           f"{pred['pf_alloc'] / max(1, buffers_only):.2f}x what the "
           "streaming machinery actually holds on the card")
    report(f"    physics fired in the streamed window: {calls}")
    return dict(n=n, tile=tile, halo=halo, window=window, steps=steps,
                pool_held_peak=int(measured),
                pool_held_resident_only=int(resident_only),
                pool_held_buffers_only=int(buffers_only),
                store_bytes=int(store_bytes),
                call_counts=calls, decision=decision.explain(), **pred)


def guard(report=print) -> dict:
    """The run-time ledger drift guard, run on a card, with its control.

    ``prepared_domain_tree_forecast`` allocates three shared workspaces and
    refuses the run if any of them differs by a byte from what
    ``estimate_experiment`` priced:

        if arena.nbytes != estimate.scratch_arena_bytes:
            raise RuntimeError("shared scratch allocation differs from
                                preflight")

    This runs that comparison verbatim on ``configs/real74_4dom.toml`` -- a
    real four-domain tree, the configuration the guard was written for --
    three ways:

    1.  RESIDENT, the control.  All three legs must pass, and the arena has
        to be really allocated on the device for that to mean anything: a
        comparison of two numbers neither of which touched a card proves the
        arithmetic and nothing else.
    2.  THE NEGATIVE CONTROL.  Price the tree, then build the arena from a
        DIFFERENT tree (d04 one cell wider).  The guard MUST fire.  Without
        this row every "the guard passed" above is equally consistent with a
        guard that cannot fail.
    3.  STREAMED, ``[tiles] mode = 'on'``.  Identical bytes on both
        sides, so the guard passes exactly as in (1) -- which is the finding:
        the guard cannot fire BECAUSE of streaming, and it offers a streamed
        run no protection either, because both sides of every comparison are
        computed by the same function from ``exp.domains``.
    """
    import dataclasses
    import tomllib
    from pathlib import Path

    import cupy as cp

    from gpuwm.core import streaming
    from gpuwm.core.model import (SharedRRTMGPChunkWorkspace,
                                  uses_modern_rrtmgp_workspace)
    from gpuwm.core.preflight import estimate_experiment
    from gpuwm.core.state import (build_shared_dycore_state_workspace,
                                  build_shared_scratch_arena)
    from gpuwm.experiment import build_experiment

    root = Path(__file__).resolve().parents[1]
    raw = tomllib.loads(
        (root / "configs" / "real74_4dom.toml").read_text(encoding="utf-8"))
    # [case_data] names ERA5 GRIB that is not on this machine and that the
    # memory question does not depend on: the estimate and the guard are
    # functions of geometry and physics selectors alone.
    raw.pop("case_data", None)
    exp = build_experiment(raw, source="real74_4dom (case_data stripped)")
    streamed_exp = dataclasses.replace(
        exp, tiles=streaming.StreamingOptions(mode="on"))

    def legs(priced, built, label):
        est = estimate_experiment(priced)
        arena = build_shared_scratch_arena(built.domains)
        rebuilt = build_shared_dycore_state_workspace(built.domains)
        rad = (SharedRRTMGPChunkWorkspace(
            nz=built.root.run.nz, column_chunk=built.column_chunk,
            p_top=built.vertical.p_top)
            if uses_modern_rrtmgp_workspace(built) else None)
        cp.cuda.runtime.deviceSynchronize()
        fired = []
        if arena.nbytes != est.scratch_arena_bytes:
            fired.append("shared scratch allocation differs from preflight")
        if rebuilt.nbytes != est.dycore_state_workspace_bytes:
            fired.append(
                "shared rebuilt-state allocation differs from preflight")
        if rad is not None and rad.nbytes != est.workspace_bytes:
            fired.append("shared radiation allocation differs from preflight")
        report(f"  {label}")
        report(f"    arena    built {arena.nbytes / GIB:8.4f} GiB   priced "
               f"{est.scratch_arena_bytes / GIB:8.4f} GiB")
        report(f"    rebuilt  built {rebuilt.nbytes / GIB:8.4f} GiB   priced "
               f"{est.dycore_state_workspace_bytes / GIB:8.4f} GiB")
        report(f"    rrtmgp   built "
               f"{(0 if rad is None else rad.nbytes) / GIB:8.4f} GiB   priced "
               f"{est.workspace_bytes / GIB:8.4f} GiB")
        report(f"    guard: {'FIRED -- ' + '; '.join(fired) if fired else 'passed'}")
        del arena, rebuilt, rad
        cp.get_default_memory_pool().free_all_blocks()
        return fired

    report("\n--- the run-time ledger drift guard, on a card ---")
    resident_fired = legs(exp, exp, "1. RESIDENT (the control)")

    # The negative control: price one tree, allocate for another.
    wider = dataclasses.replace(
        exp, domains=tuple(
            dc if dc.grid_id != 4 else dataclasses.replace(
                dc, run=dataclasses.replace(dc.run, nx=dc.run.nx + 32))
            for dc in exp.domains))
    control_fired = legs(exp, wider,
                         "2. NEGATIVE CONTROL (d04 32 cells wider than "
                         "the tree that was priced)")

    streamed_fired = legs(streamed_exp, streamed_exp,
                          "3. STREAMED, [tiles] mode = 'on'")

    ok = (not resident_fired) and control_fired and (not streamed_fired)
    report(f"\n  guard verdict: {'ARMED and mode-blind' if ok else 'BROKEN'}"
           f"  (resident passes={not resident_fired}, "
           f"negative control fires={bool(control_fired)}, "
           f"streamed passes={not streamed_fired})")
    return dict(resident_fired=resident_fired, control_fired=control_fired,
                streamed_fired=streamed_fired, ok=bool(ok))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tilestream.ledger_probe")
    ap.add_argument("--sizes", default="448,512")
    ap.add_argument("--nz", type=int, default=49)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--rung", default="full")
    ap.add_argument("--stream-n", type=int, default=0)
    ap.add_argument("--stream-tile", type=int, default=224)
    ap.add_argument("--stream-buffers", type=int, default=2)
    ap.add_argument("--json", default="")
    ap.add_argument("--guard", action="store_true",
                    help="run the ledger drift guard and its negative "
                         "control on a real four-domain tree")
    ap.add_argument("--need-free", type=float, default=0.0, metavar="GIB",
                    help="refuse to start unless this much VRAM is free.  "
                         "Every card in this project is shared.")
    args = ap.parse_args(argv)

    import cupy as cp

    free, total = cp.cuda.runtime.memGetInfo()
    props = cp.cuda.runtime.getDeviceProperties(0)
    name = props["name"]
    name = name.decode() if isinstance(name, bytes) else str(name)
    print(f"card {name}: {free / GIB:.3f} GiB free of {total / GIB:.3f} GiB "
          f"(cupy memGetInfo, the only figure trusted under WSL2)")
    print(f"steps per size: {args.steps}")
    if args.need_free and free < args.need_free * GIB:
        print(f"REFUSING: --need-free {args.need_free} GiB was asked for and "
              f"only {free / GIB:.3f} GiB is free.  This card is shared; a "
              "measurement that has to fight for the last gigabyte is a "
              "measurement of the fight.")
        return 2

    rows = []
    if args.guard:
        rows.append(guard())
    rows.extend(resident([int(s) for s in args.sizes.split(",") if s],
                         nz=args.nz, steps=args.steps, rung=args.rung))
    if args.stream_n:
        rows.append(streamed(args.stream_n, args.stream_tile, nz=args.nz,
                             steps=args.steps, rung=args.rung,
                             nbuffers=args.stream_buffers))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(rows, handle, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
