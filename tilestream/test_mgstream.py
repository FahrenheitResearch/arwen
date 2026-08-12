"""The gate for N-GPU streaming: bit-exact against one monolithic run.

Structure follows :mod:`tilestream.test_gate`.  ``compare`` is exact equality
of a digest over the whole persisted inventory -- not "close", identical --
because the claim is that sharing the tiles of one host-resident domain over
several GPUs changes nothing about the arithmetic.

WHAT EACH CASE SEPARATES.  Running two workers on two physical GPUs mixes
three things that fail differently, so they are gated in order and a later one
is only meaningful if the earlier ones passed:

``devices=(0,)``
    One worker.  Must reproduce :func:`tilestream.driver.run_tiled` exactly.
    If this fails, the port is wrong and nothing about GPUs was learned.
``devices=(0, 0)``
    TWO workers, ONE GPU.  Exercises the tile partition, the per-sweep
    barrier, the cross-worker event handshake and the shared arena, with the
    device kept out of it.  This is the case that catches a partition bug, and
    it runs on a single-GPU machine -- which is why it exists.
``devices=(0, 1)``
    The real thing.  Anything that fails here and passed above is genuinely
    about two devices.

``partition="stripe"`` puts neighbouring tiles on different GPUs, which is the
worst case for the read-at-time-t rule; ``"block"`` is the realistic one.
Both are gated.

THE NEGATIVE CONTROLS ARE NOT OPTIONAL.  A bit-exact multi-GPU result is
exactly the kind of good news this project has been wrong about before, and
the specific way it would be wrong is that the ordering being credited was
never load-bearing -- two workers that happen to run far enough apart give the
right answer for the wrong reason.  So each ordering rule is also run BROKEN,
and a control that fails to fire is reported as a failed control.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings

import numpy as np

from tilestream import driver, gather, harness, mgstream
from tilestream import spec as tspec

NZ = 49
SEED = harness.DEFAULT_SEED

_REF: dict = {}


def _digest(arrays) -> str:
    import hashlib
    h = hashlib.blake2b(digest_size=16)
    for name in sorted(arrays):
        a = arrays[name]
        a = a.get() if hasattr(a, "get") else np.asarray(a)
        h.update(name.encode())
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def monolithic(nx, ny, nsteps, *, nz=NZ, seed=SEED):
    """One un-tiled run on one GPU: the answer everything is compared to."""
    key = (nx, ny, nz, nsteps, seed)
    if key not in _REF:
        import cupy as cp
        cp.cuda.Device(0).use()
        cfg = harness.make_config(nx, ny, nz)
        state = harness.make_state(cfg, seed=seed)
        start = {n: np.array(a.get() if hasattr(a, "get") else a)
                 for n, a in harness.state_arrays(state).items()}
        harness.run_steps(state, cfg, nsteps)
        ref = {n: np.array(a.get() if hasattr(a, "get") else a)
               for n, a in harness.state_arrays(state).items()}
        del state
        cp.get_default_memory_pool().free_all_blocks()
        _REF[key] = (cfg, start, ref, _digest(ref))
    return _REF[key]


def _host_store(cfg, start):
    from tilestream.hoststore import HostDomainStore
    store = HostDomainStore(cfg)
    for name, arr in start.items():
        store.arrays[name][...] = arr
    return store


def run_case(nx, ny, tile_nx, tile_ny, halo, nsteps, *, devices=(0,),
             nbuffers=2, write_mode="shadow", partition="block",
             cross_worker_sync="events", sweep_barrier=True, nz=NZ,
             seed=SEED) -> dict:
    """One multi-GPU streamed configuration, digested and compared."""
    import cupy as cp

    cfg, start, ref, want = monolithic(nx, ny, nsteps, nz=nz, seed=seed)
    store = _host_store(cfg, start)
    store.assert_pinned()

    report: dict = {}
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mgstream.run_mgstream(
            store, cfg, tile_nx, tile_ny, halo=halo, nsteps=nsteps,
            devices=devices, nbuffers=nbuffers, write_mode=write_mode,
            partition=partition, cross_worker_sync=cross_worker_sync,
            sweep_barrier=sweep_barrier, report=report)
    elapsed = time.perf_counter() - t0

    final = driver._arrays_of(store)
    got = _digest(final)
    worst = 0.0
    where = None
    for name in sorted(ref):
        a = np.asarray(final[name])
        b = ref[name]
        if a.dtype.kind == "f":
            d = float(np.nanmax(np.abs(a.astype(np.float64)
                                       - b.astype(np.float64))))
        else:
            d = float(np.max(np.abs(a.astype(np.int64) - b.astype(np.int64))))
        if d > worst:
            worst, where = d, name
    nan = sum(int(np.isnan(np.asarray(final[n])).sum())
              for n in final if np.asarray(final[n]).dtype.kind == "f")

    store.free()
    del store, final
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()

    return {
        "devices": list(devices), "write_mode": write_mode,
        "partition": partition, "sync": cross_worker_sync,
        "sweep_barrier": sweep_barrier, "nbuffers": nbuffers,
        "domain": f"{nx}x{ny}x{nz}", "tile": f"{tile_nx}x{tile_ny}",
        "halo": halo, "steps": nsteps, "tiles": report["tiles"],
        "compute": report["compute"],
        "exact": got == want, "digest": got, "want": want,
        "max_abs_diff": worst, "worst_field": where, "nan_cells": nan,
        "seconds": elapsed,
        "host_bytes": report["host_bytes"], "host_gbs": report["host_gbs"],
        "per_worker": [(w["device"], w["ntiles"], w["gathered"] + w["scattered"])
                       for w in report["per_worker"]],
    }


# --------------------------------------------------------------------------
# positive cases
# --------------------------------------------------------------------------

def positive_cases(ngpu: int, *, nx=256, ny=256, tile=64, halo=16, nsteps=3,
                   nz=NZ) -> list[dict]:
    """Every configuration that must be bit-exact, cheapest discriminator first."""
    cases = [
        dict(devices=(0,), write_mode="shadow", partition="block"),
        dict(devices=(0, 0), write_mode="shadow", partition="block"),
        dict(devices=(0, 0), write_mode="shadow", partition="stripe"),
        dict(devices=(0, 0), write_mode="ring", partition="block"),
        dict(devices=(0, 0), write_mode="ring", partition="stripe"),
    ]
    if ngpu >= 2:
        cases += [
            dict(devices=(0, 1), write_mode="shadow", partition="block"),
            dict(devices=(0, 1), write_mode="shadow", partition="stripe"),
            dict(devices=(0, 1), write_mode="ring", partition="block"),
            dict(devices=(0, 1), write_mode="ring", partition="stripe"),
            dict(devices=(0, 1), write_mode="shadow", partition="block",
                 nbuffers=1),
            dict(devices=(1, 0), write_mode="shadow", partition="block"),
        ]
    if ngpu >= 4:
        cases += [
            dict(devices=(0, 1, 2, 3), write_mode="shadow", partition="block"),
            dict(devices=(0, 1, 2, 3), write_mode="shadow", partition="stripe"),
        ]
    out = []
    for kw in cases:
        rec = run_case(nx, ny, tile, tile, halo, nsteps, nz=nz, **kw)
        out.append(rec)
        tag = "EXACT" if rec["exact"] else f"DIFFERS max={rec['max_abs_diff']:.3e}"
        print(f"  {str(kw['devices']):12s} {kw['write_mode']:6s} "
              f"{kw['partition']:6s} nbuf={rec['nbuffers']}  {tag}"
              f"  {rec['seconds']:.2f}s")
    return out


# --------------------------------------------------------------------------
# negative controls
# --------------------------------------------------------------------------

def negative_controls(ngpu: int, *, nx=256, ny=256, tile=64, halo=16,
                      nz=NZ) -> list[dict]:
    """Each ordering rule, run broken.  Every one of these MUST differ.

    ``ring`` + ``cross_worker_sync="none"``
        Drops only the host-side "the event has been recorded" handshake.  The
        CUDA wait is still issued -- on an event that may not have been
        recorded yet, which CUDA treats as already complete.  This is the
        failure mode that cannot be seen by reading the code.
    ``shadow`` + ``sweep_barrier=False``
        A worker starts the next sweep while another is still writing the
        store the next sweep reads.  Needs at least two sweeps to have
        anything to corrupt.
    ``halo`` below the dependency radius
        The control that proves the comparison itself has teeth: the same
        machinery at halo 4 must be wrong, or the digest is not sensitive to
        the physics at all.
    """
    dev2 = (0, 1) if ngpu >= 2 else (0, 0)
    out = []
    trials = [
        ("ring, cross-worker handshake removed",
         dict(devices=dev2, write_mode="ring", partition="stripe",
              cross_worker_sync="none"), 3),
        ("ring, cross-worker handshake removed, block",
         dict(devices=dev2, write_mode="ring", partition="block",
              cross_worker_sync="none"), 3),
        ("shadow, per-sweep barrier removed",
         dict(devices=dev2, write_mode="shadow", partition="block",
              sweep_barrier=False), 4),
        ("halo below the dependency radius (comparison has teeth)",
         dict(devices=dev2, write_mode="shadow", partition="block"), 3),
    ]
    for label, kw, steps in trials:
        h = 4 if "halo below" in label else halo
        try:
            rec = run_case(nx, ny, tile, tile, h, steps, nz=nz, **kw)
        except Exception as exc:                            # noqa: BLE001
            rec = {"exact": False, "error": f"{type(exc).__name__}: {exc}",
                   "max_abs_diff": float("nan")}
        rec["control"] = label
        rec["fired"] = not rec["exact"]
        out.append(rec)
        state = "FIRED" if rec["fired"] else "DID NOT FIRE  <-- control failed"
        print(f"  {label:55s} {state}"
              + (f"  max={rec['max_abs_diff']:.3e}"
                 if rec.get("max_abs_diff") == rec.get("max_abs_diff") else ""))
    return out


def physics_case(rung, tile_nx, tile_ny, nsteps, *, devices=(0,), halo=None,
                 nx=None, ny=None, nz=NZ, nbuffers=2, write_mode="shadow",
                 partition="block", seed=SEED) -> dict:
    """One PHYSICS configuration, shared over ``devices``, against monolithic.

    The dry cases above exercise the transport.  They do NOT exercise the one
    piece of :mod:`tilestream.mgstream` that only physics reaches: the domain
    CLOCK.  ``dycore.step`` advances ``elapsed_seconds`` once per call and the
    physics driver turns that into the timestep index every cadence test reads,
    so a buffer that serves k tiles in a sweep would run k*dt ahead of the
    domain.  The single-GPU driver resets each buffer to the domain clock
    before every tile and advances the domain once per sweep; with several
    workers the buffers live in different threads on different devices, and
    the clock has to be reset and reconciled across all of them.  Get that
    wrong and tiles integrate different physics -- radiation due on one GPU
    and not the other -- with no NaN and a perfectly plausible field.

    That is why this runs a rung with a real cadence rather than a dry step.
    """
    import cupy as cp

    from tilestream import physics_inventory as physinv
    from tilestream import test_gate as tg

    nx = tg.PHYS_NX if nx is None else nx
    ny = tg.PHYS_NY if ny is None else ny
    cfg, start, start_scalars, ref_arrays, ref_scalars = tg.physics_reference(
        rung, nx, ny, nsteps, nz=nz, seed=seed)
    ref = physinv.field_digests(ref_arrays)
    if halo is None:
        halo = harness.halo_radius(cfg)

    store = {k: gather.pinned_copy(v) for k, v in start.items()}
    scalars = dict(start_scalars)
    report: dict = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mgstream.run_mgstream(
            store, cfg, tile_nx, tile_ny, halo=halo, nsteps=nsteps,
            devices=devices, nbuffers=nbuffers, write_mode=write_mode,
            partition=partition, report=report,
            inventory_fn=physinv.carrier_inventory, nz=int(cfg.nz),
            tile_state_factory=driver.make_physics_tile_state,
            scalars=scalars)
    cp.cuda.runtime.deviceSynchronize()

    got = physinv.field_digests(physinv.carrier_inventory(store))
    differing = sorted(k for k in ref if ref.get(k) != got.get(k))
    clock_ok = all(scalars.get(k) == ref_scalars.get(k)
                   for k in ("elapsed_seconds", "call_counts",
                             "microphysics_updates") if k in ref_scalars)
    del store
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    return {"rung": rung, "devices": list(devices), "write_mode": write_mode,
            "partition": partition, "carriers": len(ref),
            "bitexact": not differing, "differing": differing[:6],
            "clock_matches": clock_ok, "tiles": report["tiles"],
            "compute": report["compute"], "halo": int(halo)}


def physics_matrix(ngpu: int, *, rungs=None, tile=32, nsteps=3) -> list[dict]:
    """The physics rungs that most stress the cross-worker clock."""
    from tilestream import test_gate as tg
    if rungs is None:
        rungs = ["mp10 Morrison", "+YSU PBL", "full(real74) +KF",
                 "full fast cadence"]
    dev2 = (0, 1) if ngpu >= 2 else (0, 0)
    out = []
    for rung in rungs:
        for devs, part in ((dev2, "block"), (dev2, "stripe")):
            try:
                rec = physics_case(rung, tile, tile, nsteps, devices=devs,
                                   partition=part)
            except Exception as exc:                        # noqa: BLE001
                rec = {"rung": rung, "devices": list(devs), "partition": part,
                       "bitexact": False, "clock_matches": False,
                       "error": f"{type(exc).__name__}: {exc}"}
            out.append(rec)
            tag = "EXACT" if rec.get("bitexact") else "DIFFERS"
            print(f"  {rung:22s} {str(devs):8s} {part:6s} "
                  f"{rec.get('carriers', '?')} carriers  {tag}"
                  f"  clock={rec.get('clock_matches')}"
                  + (f"  {rec['error']}" if "error" in rec else "")
                  + (f"  first differing: {rec.get('differing')}"
                     if not rec.get("bitexact") and "error" not in rec else ""))
    return out


def repeatability(ngpu: int, *, reps=3, nx=256, ny=256, tile=64, halo=16,
                  nsteps=3, nz=NZ) -> dict:
    """The same 2-GPU configuration N times: the digest must not move.

    Two workers race by construction, so one passing run is one sample of a
    schedule.  If the digest is stable across repetitions AND the negative
    controls fire, the ordering is enforced rather than lucky.
    """
    dev2 = (0, 1) if ngpu >= 2 else (0, 0)
    seen = []
    for _ in range(reps):
        rec = run_case(nx, ny, tile, tile, halo, nsteps, devices=dev2,
                       write_mode="shadow", partition="stripe", nz=nz)
        seen.append(rec["digest"])
    ok = len(set(seen)) == 1 and seen[0] == rec["want"]
    print(f"  {reps} repetitions of {dev2} shadow/stripe: "
          f"{len(set(seen))} distinct digest(s), matches monolithic: {ok}")
    return {"reps": reps, "distinct": len(set(seen)), "exact": ok,
            "digests": seen}


def _main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=256)
    ap.add_argument("--ny", type=int, default=256)
    ap.add_argument("--tile", type=int, default=64)
    ap.add_argument("--halo", type=int, default=16)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--nz", type=int, default=NZ)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--skip-negative", action="store_true")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    import cupy as cp
    ngpu = cp.cuda.runtime.getDeviceCount()
    print(f"visible GPUs: {ngpu}")
    _, _, _, want = monolithic(args.nx, args.ny, args.steps, nz=args.nz)
    print(f"monolithic reference digest {want}\n")

    print("-- positive cases (all must be EXACT) --")
    pos = positive_cases(ngpu, nx=args.nx, ny=args.ny, tile=args.tile,
                         halo=args.halo, nsteps=args.steps, nz=args.nz)
    print("\n-- repeatability --")
    rep = repeatability(ngpu, reps=args.reps, nx=args.nx, ny=args.ny,
                        tile=args.tile, halo=args.halo, nsteps=args.steps,
                        nz=args.nz)
    neg = []
    if not args.skip_negative:
        print("\n-- negative controls (all must FIRE) --")
        neg = negative_controls(ngpu, nx=args.nx, ny=args.ny, tile=args.tile,
                                halo=args.halo, nz=args.nz)

    bad_pos = [r for r in pos if not r["exact"]]
    dead = [r for r in neg if not r["fired"]]
    print(f"\npositive: {len(pos) - len(bad_pos)}/{len(pos)} exact")
    print(f"negative: {len(neg) - len(dead)}/{len(neg)} fired")
    print(f"repeatable: {rep['exact']}")
    verdict = not bad_pos and not dead and rep["exact"]
    print(f"GATE: {'PASS' if verdict else 'FAIL'}")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"positive": pos, "negative": neg,
                       "repeatability": rep, "ngpu": ngpu,
                       "verdict": verdict}, fh, indent=2, default=str)
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(_main())
