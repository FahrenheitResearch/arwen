"""Multi-GPU streaming with PHYSICS ON: the composition nobody had run.

:mod:`tilestream.test_mgstream` proved the TRANSPORT -- N GPUs pulling tiles
out of one pinned host store reproduce a monolithic run bit for bit.  Every
arm of it is DRY: nine arrays, no ``PhysicsDriver``, no scalar clock, no
lazily-allocated carriers, no geography.  :mod:`tilestream.test_gate` proved
the opposite half -- the whole 139-to-229 carrier set streamed bit-exactly
through ONE GPU at fourteen physics rungs -- and every arm of THAT is single
device.  The product of the two had never been run, and that product is what
a user actually asks for.

WHAT THE COMPOSITION ADDS, and why each one could have been silently wrong
--------------------------------------------------------------------------
*The tile buffer.*  ``mgstream.run_mgstream`` hard-wired
``factory = driver.make_tile_state``, which attaches no driver, so
``dycore.step`` raised at every rung above dry and the multi-GPU path was
mechanically incapable of running physics.  That is a one-line seam
(``tile_state_factory=``, the same one ``run_tiled`` has had since
driver.py:1115) and it is the only edit this module needed.

*The clock.*  ``dycore.step`` advances ``elapsed_seconds`` per CALL and every
cadence test is a function of it, so the domain clock has to be reset onto a
buffer before each tile and advanced exactly once per sweep.  With ONE thread
that is a loop invariant.  With one thread per GPU it is a shared mutable
object written by worker 0 between two barriers and read by every worker, and
the failure mode is that two cards evaluate different due-flags for the same
sweep -- one integrates radiation, the other does not, and the digest is
wrong in a way that looks like weather.  ``driver._advance_clock`` cross-
checks every buffer of every worker against one reference, so a disagreement
raises instead of scattering.

*The ring arena, across devices.*  ``rings.RingArena`` mirrors the STORE's
memory class, so for the pinned host store this module runs on, the arena is
pinned host memory and both cards DMA into the same pages.  That is why one
arena can serve every device.  It also means the cross-device ordering is a
real ordering problem rather than an impossible-by-construction one, which is
what the negative control below exists to prove.

THE CADENCE TRAP, which this project has fallen into three times
----------------------------------------------------------------
``full(real74) +KF`` has ``radt=12 min`` and ``cudt=5 min``.  At ``dt=3 s``
that is one radiation call every 240 steps and one cumulus call every 100.
An 8-step window fires radiation ZERO times on BOTH sides, so a clock bug, a
cadence disagreement between two cards, and a transport that drops the
radiation tendencies are all invisible -- and the run passes.  Every arm here
therefore runs the ``full fast cadence`` rung (``radt=0.05 min``,
``cudt=0.1 min``, ``bldt=1.0 min``) and PRINTS the radiation and cumulus fire
counts of the monolithic reference AND of the arm being compared to it.  An
arm whose fire counts are zero is reported as no evidence, not as a pass.

THE NEGATIVE CONTROLS ARE THE POINT
-----------------------------------
A bit-exact two-GPU physics result is exactly the shape of good news this
project has been wrong about six times.  The specific way it would be wrong
here is that the ordering being credited was never load-bearing: two workers
that happen to run far enough apart give the right answer for the wrong
reason, and the ``cudaStreamWaitEvent``-on-an-unrecorded-event hazard
``mgstream`` documents is silent by construction.  So each arm is also run
BROKEN:

``shadow``  ``sweep_barrier=False``
    drops the one barrier the parallel path has.
``ring``    ``cross_worker_sync="none"``
    drops the host-side "this event has been RECORDED" handshake and keeps the
    CUDA wait, which is then a no-op on an unrecorded event.
both        ``carry_scalars=False``
    drops the domain clock, which at the fast cadence makes every tile
    evaluate a different due-flag.  This is the control that proves the
    comparison can see PHYSICS at all rather than only the dycore.

A control that does not fire is reported as a FAILED CONTROL, never as a
pass.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
import warnings

import numpy as np

from tilestream import driver, gather, harness, mgstream
from tilestream import spec as tspec
from tilestream import test_gate

NZ = 49
SEED = harness.DEFAULT_SEED
WARMUP = 1

#: The rung every arm runs.  See the module docstring: the long-cadence rungs
#: cannot see a clock bug in a window a gate can afford.
RUNG = "full fast cadence"

#: Correctness geometry.  256x192 with 4x4 tiles of 64x48 is the geometry
#: test_gate had to move its halo control to, because at 96x80 a 48x40 tile
#: with halo 16 already gathers 83% by 90% of the domain and the halo has
#: almost nothing left to get wrong.  16 tiles also divides evenly over 1, 2
#: and 4 workers, so no arm is comparing against a different tiling.
CX, CY = 256, 192
CTILE_NX, CTILE_NY = 64, 48


def _as_numpy(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


def _fires(before: dict, after: dict) -> dict:
    """Physics calls that happened BETWEEN two scalar-carrier snapshots.

    The whole reason this module exists in the form it does.  Reported for
    every arm on both sides of every comparison; a zero here means the window
    proved nothing about that scheme.
    """
    b = before.get("call_counts", {}) or {}
    a = after.get("call_counts", {}) or {}
    out = {k: int(a.get(k, 0)) - int(b.get(k, 0)) for k in sorted(a)}
    out["microphysics"] = (int(after.get("microphysics_updates", 0))
                           - int(before.get("microphysics_updates", 0)))
    return out


def _fire_line(fires: dict) -> str:
    return "  ".join(f"{k}={v}" for k, v in sorted(fires.items()) if v)


# --------------------------------------------------------------------------
# the reference: RESIDENT, one GPU, no tiling at all
# --------------------------------------------------------------------------

_REF: dict = {}


def reference(nx=CX, ny=CY, nsteps=8, *, rung=RUNG, nz=NZ, seed=SEED,
              warmup=WARMUP):
    """``(cfg, start, start_scalars, ref_digests, ref_scalars, fires, secs)``.

    ``warmup`` steps run BEFORE the snapshot because Kain-Fritsch allocates
    ``cumulus/w0avg`` on its first call (kf.py:335): a state that has never
    stepped has a SHORTER carrier manifest than one that has, and a store
    sized from it would be missing a field that later appears.
    """
    import cupy as cp

    from tilestream import physics_inventory as physinv

    key = (rung, nx, ny, nz, nsteps, seed, warmup)
    if key in _REF:
        return _REF[key]
    cp.cuda.Device(0).use()
    cfg = test_gate.physics_cfg(rung, nx, ny, nz)
    state, _drv = physinv.default_builder(cfg, seed)
    harness.run_steps(state, cfg, warmup)
    start = {k: _as_numpy(v).copy()
             for k, v in physinv.carrier_inventory(state).items()}
    start_scalars = physinv.carrier_scalars(state)
    t0 = time.perf_counter()
    harness.run_steps(state, cfg, nsteps)
    cp.cuda.runtime.deviceSynchronize()
    secs = time.perf_counter() - t0
    ref = {k: _as_numpy(v).copy()
           for k, v in physinv.carrier_inventory(state).items()}
    ref_scalars = physinv.carrier_scalars(state)
    del state, _drv
    cp.get_default_memory_pool().free_all_blocks()
    _REF[key] = (cfg, start, start_scalars, ref, ref_scalars,
                 _fires(start_scalars, ref_scalars), secs)
    return _REF[key]


# --------------------------------------------------------------------------
# one arm
# --------------------------------------------------------------------------

def arm(label, *, transport="mgstream", nx=CX, ny=CY, tile_nx=CTILE_NX,
        tile_ny=CTILE_NY, nsteps=8, rung=RUNG, nz=NZ, seed=SEED, halo=None,
        devices=(0,), nbuffers=2, write_mode="shadow", partition="block",
        cross_worker_sync="events", sweep_barrier=True, carry_scalars=True,
        expect=True) -> dict:
    """Run one configuration against the resident reference and digest it.

    ``transport="monolithic"`` re-runs the reference recipe itself, which is
    the only way to show that the comparison is not comparing a cached number
    to itself.  ``"tiled"`` is :func:`tilestream.driver.run_tiled` -- ONE GPU,
    the settled streamed path.  ``"mgstream"`` is the thing under test.
    """
    import cupy as cp

    from tilestream import physics_inventory as physinv

    cfg, start, start_scalars, ref_arrays, ref_scalars, ref_fires, _rs = \
        reference(nx, ny, nsteps, rung=rung, nz=nz, seed=seed)
    ref = physinv.field_digests(ref_arrays)
    if halo is None:
        halo = harness.halo_radius(cfg)

    specs = tspec.plan_tiles(nx, ny, tile_nx, tile_ny, halo, True)
    tspec.validate_plan(specs, ny, nx)
    scalars = dict(start_scalars) if carry_scalars else None
    report: dict = {}

    if transport == "monolithic":
        cp.cuda.Device(0).use()
        state, _d = physinv.default_builder(cfg, seed)
        # The warm-up is not cosmetic even here: two carriers are allocated
        # lazily on first use (Kain-Fritsch's cumulus/w0avg, kf.py:335), so a
        # state that has never stepped has a SHORTER manifest than the
        # snapshot and load_carriers refuses the key-set mismatch.
        harness.run_steps(state, cfg, WARMUP)
        physinv.load_carriers(state, (start, start_scalars))
        t0 = time.perf_counter()
        harness.run_steps(state, cfg, nsteps)
        cp.cuda.runtime.deviceSynchronize()
        elapsed = time.perf_counter() - t0
        final = {k: _as_numpy(v) for k, v in
                 physinv.carrier_inventory(state).items()}
        end_scalars = physinv.carrier_scalars(state)
        got = physinv.field_digests(final)
        magnitude = _localise(final, ref_arrays,
                              sorted(k for k in ref if ref.get(k) != got.get(k)))
        del state, _d, final
        cp.get_default_memory_pool().free_all_blocks()
        store_bytes = 0
        ntiles = 1
    else:
        store = {k: gather.pinned_copy(v) for k, v in start.items()}
        store_bytes = sum(int(a.nbytes) for a in store.values())
        kw = dict(
            halo=halo, nsteps=nsteps, nbuffers=nbuffers,
            write_mode=write_mode, report=report,
            inventory_fn=physinv.carrier_inventory, nz=int(cfg.nz),
            tile_state_factory=driver.make_physics_tile_state,
            scalars=scalars)
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            if transport == "tiled":
                cp.cuda.Device(0).use()
                driver.run_tiled(store, cfg, tile_nx, tile_ny, **kw)
            else:
                mgstream.run_mgstream(
                    store, cfg, tile_nx, tile_ny, devices=devices,
                    partition=partition,
                    cross_worker_sync=cross_worker_sync,
                    sweep_barrier=sweep_barrier, **kw)
        cp.cuda.runtime.deviceSynchronize()
        elapsed = time.perf_counter() - t0
        got = physinv.field_digests(store)
        end_scalars = scalars if scalars is not None else {}
        ntiles = report.get("tiles", len(specs))
        # The MAGNITUDE has to be taken while the store is still alive, and it
        # has to be a real difference: reporting the largest reference value
        # among the differing carriers reads like an error size and is not one.
        magnitude = _localise(store, ref_arrays,
                              sorted(k for k in ref if ref.get(k) != got.get(k)))
        del store
        cp.get_default_pinned_memory_pool().free_all_blocks()
        cp.get_default_memory_pool().free_all_blocks()

    differing = sorted(k for k in ref if ref.get(k) != got.get(k))
    fires = _fires(start_scalars, end_scalars) if end_scalars else {}
    rec = {
        "label": label, "transport": transport,
        "devices": list(devices) if transport == "mgstream" else [0],
        "write_mode": write_mode if transport != "monolithic" else "-",
        "partition": partition, "sync": cross_worker_sync,
        "sweep_barrier": sweep_barrier, "carry_scalars": carry_scalars,
        "nbuffers": nbuffers, "halo": int(halo), "steps": nsteps,
        "domain": f"{nx}x{ny}x{nz}", "tile": f"{tile_nx}x{tile_ny}",
        "tiles": ntiles, "carriers": len(ref),
        "bitexact": not differing, "ndiffer": len(differing),
        "differing": differing[:12],
        "scalars_ok": (not carry_scalars) or (end_scalars == ref_scalars),
        "fires": fires, "ref_fires": ref_fires,
        "seconds": elapsed, "expect": expect,
        "store_bytes": store_bytes,
        "compute": report.get("compute"),
        "host_gbs": report.get("host_gbs"),
        "per_worker": [(w["device"], w["ntiles"])
                       for w in report.get("per_worker", [])],
    }
    rec["ok"] = rec["bitexact"] == expect
    rec.update(magnitude)
    return rec


def _localise(store, ref_arrays, differing) -> dict:
    """``max|got - want|`` over the differing carriers, and which one.

    Triage only -- the DIGEST is the verdict, and a carrier that differs in
    one ulp is as much a failure as one that differs by 3000.  Reported
    because the size tells a reader which kind of wrong it is: ~1e-7 relative
    is floating-point reassociation and a different fix entirely from a
    dropped halo.
    """
    if not differing:
        return {}
    worst, max_abs, max_rel = differing[0], 0.0, 0.0
    for key in differing:
        got_a = np.asarray(_as_numpy(store[key]), dtype=np.float64)
        want_a = np.asarray(ref_arrays[key], dtype=np.float64)
        if not got_a.size:
            continue
        diff = np.abs(got_a - want_a)
        this_abs = float(np.nanmax(diff))
        scale = np.maximum(np.abs(want_a), np.abs(got_a))
        live = scale > 0.0
        if live.any():
            max_rel = max(max_rel, float(np.nanmax(diff[live] / scale[live])))
        if this_abs > max_abs:
            worst, max_abs = key, this_abs
    return {"worst_field": worst, "max_abs": max_abs, "max_rel": max_rel}


def _is_oom(exc: BaseException) -> bool:
    """Whether a raise was the box running out of VRAM, at any depth.

    ``mgstream`` wraps a worker's exception in ``MGStreamError``, and CuPy
    raises three different types for the same condition (``OutOfMemoryError``,
    ``CUDARuntimeError: cudaErrorMemoryAllocation``, ``CUDADriverError:
    CUDA_ERROR_OUT_OF_MEMORY``), so this walks the ``__cause__`` chain and
    matches on the text as well as the type.
    """
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        name = type(exc).__name__
        text = str(exc).lower()
        if name == "OutOfMemoryError" or "out of memory" in text:
            return True
        exc = exc.__cause__ or exc.__context__
    return False


def context_alive() -> bool:
    """Whether this process's CUDA context still works, on every device.

    An ``cudaErrorIllegalAddress`` anywhere destroys the context for the
    whole PROCESS, so every arm after the first one that hits it fails at
    some unrelated allocation.  Six identical-looking failures then read as
    six independent defects.  Checked after any raise, and the remaining
    arms are reported as NOT RUN rather than as failures.
    """
    import cupy as cp
    try:
        for d in range(cp.cuda.runtime.getDeviceCount()):
            cp.cuda.Device(d).use()
            cp.zeros(4, dtype=cp.float32).sum().item()
        cp.cuda.Device(0).use()
        return True
    except BaseException:                                   # noqa: BLE001
        return False


def free_vram(devices) -> dict[int, float]:
    """``{device: free GiB}``.  Under WSL2 nvidia-smi's used/free is
    unreliable, so this asks the runtime, which is what the allocator will
    ask too."""
    import cupy as cp
    out = {}
    for d in dict.fromkeys(devices):
        cp.cuda.Device(d).use()
        out[int(d)] = cp.cuda.runtime.memGetInfo()[0] / 2**30
    cp.cuda.Device(0).use()
    return out


def _print(rec: dict) -> None:
    mark = "PASS" if rec["ok"] else "FAIL"
    if rec["bitexact"]:
        detail = f"bit-exact over all {rec['carriers']} carriers"
    else:
        detail = (f"{rec['ndiffer']}/{rec['carriers']} carriers differ, "
                  f"maxabs={rec.get('max_abs', float('nan')):.3e} "
                  f"maxrel={rec.get('max_rel', float('nan')):.2e} "
                  f"worst={rec.get('worst_field')}")
    clock = "clock ok" if rec["scalars_ok"] else "CLOCK DIFFERS"
    print(f"  {mark}  {rec['label']}")
    print(f"        {detail}, {clock}, {rec['seconds']:.2f} s")
    print(f"        fires  this arm: {_fire_line(rec['fires']) or 'NONE'}")
    print(f"        fires  reference: {_fire_line(rec['ref_fires']) or 'NONE'}")
    if rec["per_worker"]:
        print(f"        per worker (device, tiles): {rec['per_worker']}")


# --------------------------------------------------------------------------
# the arms
# --------------------------------------------------------------------------

def correctness(ngpu: int, *, nsteps=8, nx=CX, ny=CY, tile_nx=CTILE_NX,
                tile_ny=CTILE_NY, nz=NZ, only: str | None = None) -> list[dict]:
    """Cheapest discriminator first; a later arm only means anything if the
    earlier ones passed.

    ``monolithic`` re-run
        the control on the control.  If a second resident run does not
        reproduce the reference, nothing below is a measurement of tiling.
    ``tiled``
        one GPU, ``run_tiled``.  Separates "streaming physics is broken" from
        "multi-GPU streaming physics is broken".
    ``mgstream devices=(0,)``
        one worker.  Must equal ``run_tiled``; a failure here is the port.
    ``mgstream devices=(0, 0)``
        TWO workers, ONE GPU.  Exercises the partition, the per-sweep
        barrier, the cross-worker handshake and the shared arena with the
        second device kept out of it.
    ``mgstream devices=(0, 1)``
        the real thing.
    """
    common = dict(nx=nx, ny=ny, tile_nx=tile_nx, tile_ny=tile_ny,
                  nsteps=nsteps, nz=nz)
    cases = [
        ("RESIDENT  monolithic re-run (control on the control)",
         dict(transport="monolithic")),
        ("STREAMED  run_tiled, 1 GPU, ring",
         dict(transport="tiled", write_mode="ring")),
        ("STREAMED  run_tiled, 1 GPU, shadow",
         dict(transport="tiled", write_mode="shadow")),
        ("MGSTREAM  1 worker  (0,)     shadow block",
         dict(devices=(0,), write_mode="shadow")),
        ("MGSTREAM  1 worker  (0,)     ring   block",
         dict(devices=(0,), write_mode="ring")),
        ("MGSTREAM  2 workers (0,0)    shadow block",
         dict(devices=(0, 0), write_mode="shadow")),
        ("MGSTREAM  2 workers (0,0)    shadow stripe",
         dict(devices=(0, 0), write_mode="shadow", partition="stripe")),
        ("MGSTREAM  2 workers (0,0)    ring   block",
         dict(devices=(0, 0), write_mode="ring")),
        ("MGSTREAM  2 workers (0,0)    ring   stripe",
         dict(devices=(0, 0), write_mode="ring", partition="stripe")),
    ]
    if ngpu >= 2:
        cases += [
            ("ARM A     2 GPUs   (0,1)    shadow block",
             dict(devices=(0, 1), write_mode="shadow")),
            ("ARM A     2 GPUs   (0,1)    shadow stripe",
             dict(devices=(0, 1), write_mode="shadow", partition="stripe")),
            ("ARM A     2 GPUs   (0,1)    shadow block nbuffers=1",
             dict(devices=(0, 1), write_mode="shadow", nbuffers=1)),
            ("ARM A     2 GPUs   (1,0)    shadow block (device order)",
             dict(devices=(1, 0), write_mode="shadow")),
            ("ARM B     2 GPUs   (0,1)    ring   block",
             dict(devices=(0, 1), write_mode="ring")),
            ("ARM B     2 GPUs   (0,1)    ring   stripe",
             dict(devices=(0, 1), write_mode="ring", partition="stripe")),
        ]
    # A 2-GPU arm needs room on BOTH cards.  These boxes are shared, and an
    # arm that dies of another agent's occupancy must be reported as NOT RUN
    # -- calling it a failure of this code would be a false result in the
    # other direction, which is just as bad.
    need = min_free_gib(nx, ny, nz, tile_nx, tile_ny)
    # ``only`` exists because these boxes are shared: a 15-arm pass needs BOTH
    # cards free for 40 minutes, and the arms that need two cards are six of
    # them.  Running just those in a window when two cards happen to be free
    # is the difference between evidence and six NOT RUNs.
    if only:
        cases = [(lab, kw) for lab, kw in cases if only in lab]
    out = []
    dead_context = False
    for label, kw in cases:
        if dead_context:
            print(f"  NOT RUN  {label}  (the CUDA context was destroyed by "
                  "an earlier arm)")
            out.append({"label": label, "ok": False, "bitexact": False,
                        "error": "not run: CUDA context destroyed earlier",
                        "expect": True, "not_run": True, "fires": {},
                        "ref_fires": {}, "carriers": 0, "ndiffer": -1,
                        "differing": [], "scalars_ok": False,
                        "seconds": 0.0, "per_worker": []})
            continue
        devices = kw.get("devices", (0,))
        free = free_vram(devices if kw.get("transport", "mgstream")
                         == "mgstream" else (0,))
        short = {d: g for d, g in free.items() if g < need}
        if short:
            print(f"  NOT RUN  {label}  (needs ~{need:.1f} GiB free per "
                  f"card, has {short})")
            out.append({"label": label, "ok": False, "bitexact": False,
                        "error": f"not run: only {short} GiB free",
                        "expect": True, "not_run": True, "fires": {},
                        "ref_fires": {}, "carriers": 0, "ndiffer": -1,
                        "differing": [], "scalars_ok": False,
                        "seconds": 0.0, "per_worker": []})
            continue
        try:
            rec = arm(label, **common, **kw)
        except Exception as exc:                            # noqa: BLE001
            oom = _is_oom(exc)
            if not oom:
                traceback.print_exc()
            dead_context = not context_alive()
            rec = {"label": label, "ok": False, "bitexact": False,
                   "error": f"{type(exc).__name__}: {exc}", "expect": True,
                   "fires": {}, "ref_fires": {}, "carriers": 0, "ndiffer": -1,
                   "differing": [], "scalars_ok": False, "seconds": 0.0,
                   "per_worker": [], "context_destroyed": dead_context,
                   "not_run": oom}
            # OUT OF MEMORY IS NOT A RESULT.  On a card shared with other
            # tenants it says the box was busy, not that this code is wrong,
            # and recording it as a failure would be a false result pointing
            # the other way.
            print(f"  {'NOT RUN' if oom else 'FAIL'}  {label}\n"
                  f"        {'ran out of VRAM: ' if oom else 'RAISED '}"
                  f"{rec['error'][:200]}"
                  + ("\n        AND THE CUDA CONTEXT IS GONE; every later "
                     "arm in this process is meaningless" if dead_context
                     else ""))
            out.append(rec)
            continue
        _print(rec)
        out.append(rec)
    return out


def min_free_gib(nx, ny, nz, tile_nx, tile_ny, *, halo=16, nbuffers=2,
                 bytes_per_column=11047.0) -> float:
    """VRAM one worker needs: its tile buffers plus room for physics scratch.

    ``bytes_per_column`` is MEASURED for the ``full fast cadence`` carrier
    set -- 11,047 B per column at both 96x80x49 and 160x128x49, i.e. flat in
    the horizontal, which is what makes the extrapolation legitimate.  The
    rest is the physics driver's own workspace, which is chunked by column
    and so does not scale with the tile.

    The 1.5 GiB constant is the PHYSICS DRIVER's own workspace and it dwarfs
    the buffers: MEASURED on a 4090 from a worker that died mid-arm, RRTMGP
    was 1,414,852,096 bytes in and asking for another 140,000,256 when the
    card ran out, against 0.16 GiB of tile buffer at this tiling.  It is
    chunked by column, so it does not scale with the tile the way the
    buffers do -- which is why it is a constant here and not a term.

    Threshold, not guarantee.  These cards are shared with other tenants and
    the free figure is stale the moment it is read; an arm that starts and
    then runs out is caught by the OutOfMemoryError branch and reported as
    not run too, one attempt further in.
    """
    cols = (tile_nx + 2 * halo) * (tile_ny + 2 * halo)
    return nbuffers * cols * bytes_per_column / 2**30 + 1.5


def negative_controls(ngpu: int, *, nsteps=8, nx=CX, ny=CY,
                      tile_nx=CTILE_NX, tile_ny=CTILE_NY, nz=NZ) -> list[dict]:
    """Every ordering rule, run broken.  Each MUST change the digest.

    Run on ``(0, 0)`` FIRST and on ``(0, 1)`` after.  The two-worker-one-card
    pass is the one that always runs -- it needs no second free card -- and it
    is what proves the ordering machinery is load-bearing at all.  The
    two-card pass proves the same rules still bind when the readers are on
    different devices, which is the claim this module exists for.  Reporting
    only the second would leave the whole set NOT RUN on a busy box and the
    positives standing on nothing.
    """
    common = dict(nx=nx, ny=ny, tile_nx=tile_nx, tile_ny=tile_ny,
                  nsteps=nsteps, nz=nz, expect=False)
    device_sets = [(0, 0)] + ([(0, 1)] if ngpu >= 2 else [])
    cases = []
    for dev2 in device_sets:
        cases += [
            (f"NEG shadow {dev2}: per-sweep barrier removed",
             dict(devices=dev2, write_mode="shadow", sweep_barrier=False)),
            (f"NEG ring {dev2} stripe: cross-worker event handshake removed",
             dict(devices=dev2, write_mode="ring", partition="stripe",
                  cross_worker_sync="none")),
            (f"NEG ring {dev2} block: cross-worker event handshake removed",
             dict(devices=dev2, write_mode="ring",
                  cross_worker_sync="none")),
            (f"NEG shadow {dev2}: domain clock NOT carried (fast cadence)",
             dict(devices=dev2, write_mode="shadow", carry_scalars=False)),
            (f"NEG ring {dev2}: domain clock NOT carried (fast cadence)",
             dict(devices=dev2, write_mode="ring", carry_scalars=False)),
            (f"NEG shadow {dev2}: halo 13, below the dependency radius",
             dict(devices=dev2, write_mode="shadow", halo=13)),
        ]
    need = min_free_gib(nx, ny, nz, tile_nx, tile_ny)
    out = []
    for label, kw in cases:
        free = free_vram(kw["devices"])
        short = {d: g for d, g in free.items() if g < need}
        if short:
            print(f"  NOT RUN  {label}  (needs ~{need:.1f} GiB per card, "
                  f"has {short})")
            out.append({"label": label, "ok": False, "fired": False,
                        "not_run": True,
                        "error": f"not run: only {short} GiB free"})
            continue
        try:
            rec = arm(label, **common, **kw)
        except Exception as exc:                            # noqa: BLE001
            # A control that RAISES has still detected the defect -- the run
            # did not silently produce a plausible answer -- so it counts as
            # fired, but the distinction is printed.
            rec = {"label": label, "ok": True, "bitexact": False,
                   "error": f"{type(exc).__name__}: {exc}", "expect": False,
                   "fires": {}, "ref_fires": {}, "carriers": 0,
                   "ndiffer": -1, "differing": [], "scalars_ok": False,
                   "seconds": 0.0, "per_worker": [], "raised": True}
            print(f"  FIRED (raised)  {label}\n"
                  f"        {rec['error'][:160]}")
            out.append(rec)
            continue
        rec["fired"] = not rec["bitexact"]
        state = ("FIRED" if rec["fired"]
                 else "DID NOT FIRE  <-- THE CONTROL FAILED")
        print(f"  {state}  {label}")
        if rec["fired"]:
            print(f"        {rec['ndiffer']}/{rec['carriers']} carriers "
                  f"differ, maxabs={rec.get('max_abs', float('nan')):.3e}, "
                  f"worst={rec.get('worst_field')}, {rec['seconds']:.2f} s")
        out.append(rec)
    return out


def repeatability(ngpu: int, *, reps=3, nsteps=8, nx=CX, ny=CY,
                  tile_nx=CTILE_NX, tile_ny=CTILE_NY, nz=NZ) -> dict:
    """The same 2-GPU physics configuration N times: the digest must not move.

    Two workers race by construction, so one passing run is one sample of one
    schedule.  Stability across repetitions plus firing negative controls is
    what separates "the ordering is enforced" from "the ordering was lucky".
    """
    dev2 = (0, 1) if ngpu >= 2 else (0, 0)
    seen = []
    for _ in range(reps):
        rec = arm("repeat", devices=dev2, write_mode="shadow",
                  partition="stripe", nx=nx, ny=ny, tile_nx=tile_nx,
                  tile_ny=tile_ny, nsteps=nsteps, nz=nz)
        seen.append((rec["bitexact"], rec["ndiffer"]))
    ok = len(set(seen)) == 1 and seen[0][0]
    print(f"  {reps} repetitions of {dev2} shadow/stripe, physics on: "
          f"{len(set(seen))} distinct outcome(s), all bit-exact: {ok}")
    return {"reps": reps, "distinct": len(set(seen)), "exact": ok}


# --------------------------------------------------------------------------
# timing: the ring is sequential and that is the finding, not a speedup
# --------------------------------------------------------------------------

def scaling(ngpu: int, *, nx=512, ny=512, tile=128, nsteps=8, nz=NZ,
            rung=RUNG, reps=2) -> list[dict]:
    """1 GPU vs 2 GPUs, shadow and ring, with physics on and the clock carried.

    Reported per DOMAIN cell so redundant halo compute cannot inflate it.  The
    ring numbers are evidence for the sequential limitation
    :mod:`tilestream.mgstream` argues for, NOT a speedup claim; the shadow
    numbers are the arm that can parallelise, and they cost a second host
    store.

    No digest comparison here -- correctness is settled above at a geometry
    chosen to discriminate, and re-deriving it at a size chosen for timing
    would only make the timing dishonest.  What IS checked is that the two
    device counts produce the SAME digest as each other, because a transport
    that silently skips work looks exactly like a speedup.
    """
    import cupy as cp

    from tilestream import physics_inventory as physinv

    # The start state is built RESIDENT, which is the one thing a real
    # out-of-core route must not do (REAL-DATA.md: the ingest writes the
    # pinned store field by field).  A bench may, because the size is chosen
    # to fit; it is built on whichever card has the most room, because these
    # boxes are shared and the build is the largest single allocation in the
    # whole run.
    build_dev, free = 0, -1
    for d in range(cp.cuda.runtime.getDeviceCount()):
        cp.cuda.Device(d).use()
        f, _t = cp.cuda.runtime.memGetInfo()
        if f > free:
            build_dev, free = d, f
    cp.cuda.Device(build_dev).use()
    print(f"  building the {nx}x{ny}x{nz} start state on device "
          f"{build_dev} ({free / 2**30:.1f} GiB free)")
    cfg = test_gate.physics_cfg(rung, nx, ny, nz)
    state, _d = physinv.default_builder(cfg, SEED)
    harness.run_steps(state, cfg, WARMUP)
    start = {k: _as_numpy(v).copy()
             for k, v in physinv.carrier_inventory(state).items()}
    start_scalars = physinv.carrier_scalars(state)
    # RESIDENT timing, taken here because the state is already built and the
    # comparison is worthless without it: "2 GPUs streamed is Xx 1 GPU
    # streamed" says nothing about whether streaming was worth doing.
    t0 = time.perf_counter()
    harness.run_steps(state, cfg, nsteps)
    cp.cuda.runtime.deviceSynchronize()
    resident_seconds = time.perf_counter() - t0
    resident_fires = _fires(start_scalars, physinv.carrier_scalars(state))
    del state, _d
    cp.get_default_memory_pool().free_all_blocks()
    print(f"  store is {sum(a.nbytes for a in start.values()) / 2**30:.2f} "
          f"GiB over {len(start)} carriers")
    print(f"  RESIDENT 1 GPU  {resident_seconds:7.2f} s  "
          f"{nz * ny * nx * nsteps / resident_seconds / 1e6:8.2f} Mcell/s"
          f"   fires: {_fire_line(resident_fires) or 'NONE'}")

    halo = harness.halo_radius(cfg)
    cases = [(1, "shadow"), (1, "ring")]
    if ngpu >= 2:
        cases += [(2, "shadow"), (2, "ring")]
    out = []
    for ndev, mode in cases:
        devices = tuple(range(ndev))
        best = None
        digest = None
        for _ in range(reps):
            store = {k: gather.pinned_copy(v) for k, v in start.items()}
            scalars = dict(start_scalars)
            report: dict = {}
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                mgstream.run_mgstream(
                    store, cfg, tile, tile, halo=halo, nsteps=nsteps,
                    devices=devices, nbuffers=2, write_mode=mode,
                    inventory_fn=physinv.carrier_inventory, nz=int(cfg.nz),
                    tile_state_factory=driver.make_physics_tile_state,
                    scalars=scalars, report=report)
            cp.cuda.runtime.deviceSynchronize()
            digest = physinv.field_digests(store)
            if best is None or report["seconds"] < best["seconds"]:
                best = dict(report)
                best["fires"] = _fires(start_scalars, scalars)
            del store
            cp.get_default_pinned_memory_pool().free_all_blocks()
            cp.get_default_memory_pool().free_all_blocks()
        rec = {
            "ndev": ndev, "mode": mode, "seconds": best["seconds"],
            "cells_per_s": mgstream.cells_per_second(best),
            "host_gbs": best["host_gbs"], "tiles": best["tiles"],
            "compute": best["compute"], "steps": nsteps,
            "fires": best["fires"],
            "second_store_bytes": best["second_store_bytes"],
            "ring_bytes": best.get("ring_bytes", 0),
            "digest": _hash(digest),
        }
        out.append(rec)
        print(f"  {ndev} GPU  {mode:6s}  {best['seconds']:7.2f} s  "
              f"{rec['cells_per_s'] / 1e6:8.2f} Mcell/s  "
              f"{best['host_gbs']:6.2f} GB/s host  "
              f"compute {best['compute']}  "
              f"2nd store {best['second_store_bytes'] / 2**30:.2f} GiB  "
              f"ring {best.get('ring_bytes', 0) / 2**30:.2f} GiB")
        print(f"          fires: {_fire_line(best['fires']) or 'NONE'}")
    for mode in ("shadow", "ring"):
        got = [r for r in out if r["mode"] == mode]
        if len(got) == 2:
            same = got[0]["digest"] == got[1]["digest"]
            speed = got[0]["seconds"] / got[1]["seconds"]
            verdict = ("digests MATCH" if same else
                       "digests DIFFER  <-- the timing is meaningless")
            print(f"  {mode}: 1 GPU -> 2 GPUs is {speed:.2f}x, {verdict}")
            for r in got:
                r["digest_agrees"] = same
                r["speedup"] = speed
    for r in out:
        r["resident_seconds"] = resident_seconds
        r["tiling_tax"] = r["seconds"] / resident_seconds
    shad = [r for r in out if r["mode"] == "shadow"]
    ring = [r for r in out if r["mode"] == "ring"]
    if shad and ring:
        same = shad[0]["digest"] == ring[0]["digest"]
        print(f"  shadow and ring agree with EACH OTHER on 1 GPU: {same}")
    return out


# --------------------------------------------------------------------------
# the same thing on a REAL projection, which is where geography joins in
# --------------------------------------------------------------------------

def geography_arms(ngpu: int, *, nsteps=8, nx=96, ny=80, tile_nx=48,
                   tile_ny=40, nz=NZ, rung=RUNG) -> list[dict]:
    """Two GPUs, physics on, real Lambert grid, real terrain, real lat/lon.

    Geography is INPUT, not state: it is gathered into a tile buffer when
    that buffer starts serving a different tile and NEVER scattered back.
    With one GPU that is one gather per buffer per tile change.  With two it
    is that PER DEVICE, out of one shared read-only host store, and each
    worker also has to impose the DOMAIN's ``has_msf``/``rotational`` flags
    on its own buffers -- state.py:799-803 derives them from ``.any()`` over
    whatever window ``set_map_coriolis`` was handed, and a per-tile window is
    not the domain.

    Two things are checked, not one: the carriers must be bit-exact AND the
    geography store must come back byte-identical.  A run that wrote
    geography back would still pass a carrier digest on a short window and
    then drift, and with two devices writing the same shared read-only store
    it is a race as well as a wrong answer.
    """
    import cupy as cp

    from tilestream import physics_inventory as physinv

    cp.cuda.Device(0).use()
    cfg, start, start_scalars, geo_start, ref, _ra, ref_scalars = \
        test_gate.geography_reference(rung, nx, ny, nsteps, nz=nz, seed=SEED)
    halo = harness.halo_radius(cfg)
    cases = [("1 worker  (0,)   shadow", (0,), "shadow", "block", True),
             ("1 worker  (0,)   ring  ", (0,), "ring", "block", True)]
    if ngpu >= 2:
        cases += [
            ("2 GPUs    (0,1)  shadow block", (0, 1), "shadow", "block", True),
            ("2 GPUs    (0,1)  shadow stripe", (0, 1), "shadow", "stripe",
             True),
            ("2 GPUs    (0,1)  ring   block", (0, 1), "ring", "block", True),
            # NEGATIVE: the buffers rebuild geography from their OWN config
            # instead of gathering the domain's, which displaces a tile by up
            # to 1022 km.  It is the control that proves the geography gather
            # is doing something on two devices, not only on one.
            ("2 GPUs    (0,1)  NEG geography REBUILT per tile", (0, 1),
             "shadow", "block", False),
        ]
    out = []
    for label, devices, mode, partition, gather_geo in cases:
        store = {k: gather.pinned_copy(v) for k, v in start.items()}
        geo_store = {k: gather.pinned_copy(v) for k, v in geo_start.items()}
        scalars = dict(start_scalars)
        build = (harness.neutral_geography if gather_geo
                 else harness.make_geography)
        kwargs = driver.geography_run_kwargs(cfg, None, geography=geo_store,
                                             geography_fn=build)
        kwargs["scalars"] = scalars
        if not gather_geo:
            kwargs.pop("geography")
        # ``geography_run_kwargs`` is written for ``run_tiled``, which takes a
        # ``shared`` tile-buffer arena; ``run_mgstream`` does not, and passing
        # the key through raised TypeError before the first arm ever ran --
        # this whole rung was dead on arrival and nothing said so, because the
        # crash is at call time rather than in a verdict.  Nobody here asks
        # for an arena, so the key is dropped when it is empty.  A non-empty
        # one is REFUSED rather than silently discarded: one arena cannot
        # serve buffers on two devices, and dropping it would run a different
        # experiment from the one the caller asked for.
        if kwargs.pop("shared", None) is not None:
            raise TypeError(
                "geography_arms was handed a shared tile arena, which "
                "run_mgstream cannot take: its buffers live on several "
                "devices and CuPy pools are per device.  Refusing rather "
                "than dropping it, so the arm cannot report a pass for a "
                "configuration it did not run.")
        report: dict = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mgstream.run_mgstream(store, cfg, tile_nx, tile_ny, halo=halo,
                                  nsteps=nsteps, devices=devices,
                                  write_mode=mode, partition=partition,
                                  report=report, **kwargs)
        cp.cuda.runtime.deviceSynchronize()
        got = physinv.field_digests(store)
        differing = sorted(k for k in ref if ref.get(k) != got.get(k))
        geo_now = physinv.field_digests(geo_store)
        geo_was = physinv.field_digests(geo_start)
        touched = sorted(k for k in geo_was if geo_was[k] != geo_now.get(k))
        rec = {
            "label": label, "devices": list(devices), "write_mode": mode,
            "partition": partition, "gathered_geography": gather_geo,
            "bitexact": not differing, "ndiffer": len(differing),
            "carriers": len(ref), "geo_readonly": not touched,
            "geo_touched": touched[:6],
            "geo_fields": report.get("geography_fields", 0),
            "geo_bytes": report.get("geography_bytes", 0),
            "scalars_ok": scalars == ref_scalars,
            "seconds": report["seconds"], "expect": gather_geo,
        }
        rec["ok"] = (rec["bitexact"] == gather_geo) and rec["geo_readonly"]
        out.append(rec)
        del store, geo_store
        cp.get_default_pinned_memory_pool().free_all_blocks()
        cp.get_default_memory_pool().free_all_blocks()
        detail = ("bit-exact over all %d carriers" % rec["carriers"]
                  if rec["bitexact"] else
                  "%d/%d carriers differ" % (rec["ndiffer"], rec["carriers"]))
        print(f"  {'PASS' if rec['ok'] else 'FAIL'}  {label}: {detail}, "
              f"geography read-only: {rec['geo_readonly']}, "
              f"clock {'ok' if rec['scalars_ok'] else 'WRONG'}, "
              f"{rec['seconds']:.1f} s")
    return out


# --------------------------------------------------------------------------
# ARM C: the domain tree
# --------------------------------------------------------------------------

def nest_probe(ngpu: int, *, nx=CX, ny=CY, tile_nx=CTILE_NX, tile_ny=CTILE_NY,
               nsteps=8, nz=NZ, rung=RUNG) -> dict:
    """Whether a nest can be forced from a parent that is being streamed.

    A full two-domain tree needs a ``DomainConfig`` pair, two
    ``LambertGrid``s, a ``Schedule``, two ``DomainClock``s and the executor.
    None of that changes the answer, because the entire parent-facing surface
    of the coupler is one line -- ``couple_nest_field(parent.state, kind,
    out=...)``, nest.py:244 -- and this probe runs THAT line, on a real
    streamed parent, before and after a real multi-GPU sweep.

    Three digests, and the verdict is which two agree:

    ``t0``      the coupled forcing fields at the start of the window.
    ``resident``  the same fields after ``nsteps`` of a RESIDENT run: what the
                child must be forced with.
    ``streamed``  the same fields read off the state the coupler would read,
                after the same ``nsteps`` streamed across ``ngpu`` cards.

    ``streamed == t0`` is the gap: the store advanced and the coupler's input
    did not.  ``streamed == resident`` after :func:`tilestream.nestbridge
    .refresh_nest_fields` is the fix.  Both are asserted, because a fix whose
    control does not fire is not a fix.
    """
    import cupy as cp

    from tilestream import nestbridge
    from tilestream import physics_inventory as physinv

    cp.cuda.Device(0).use()
    cfg = test_gate.physics_cfg(rung, nx, ny, nz)
    kinds = list(nestbridge.nest_carrier_keys(cfg))
    print(f"  forcing field set for this rung ({len(kinds)}): "
          f"{' '.join(kinds)}")

    state, _d = physinv.default_builder(cfg, SEED)
    harness.run_steps(state, cfg, WARMUP)
    start = {k: _as_numpy(v).copy()
             for k, v in physinv.carrier_inventory(state).items()}
    start_scalars = physinv.carrier_scalars(state)
    t0_digests = nestbridge.coupled_parent_digests(state, cfg)

    # The truth: what the child must be forced with after nsteps.
    harness.run_steps(state, cfg, nsteps)
    cp.cuda.runtime.deviceSynchronize()
    resident_digests = nestbridge.coupled_parent_digests(state, cfg)
    resident_fires = _fires(start_scalars, physinv.carrier_scalars(state))
    del state, _d
    cp.get_default_memory_pool().free_all_blocks()

    # The streamed parent.  Its state object is rebuilt from the SAME start,
    # which is exactly the situation a streamed route leaves it in: filled
    # once from the prepared state and never written again.
    state, _d = physinv.default_builder(cfg, SEED)
    harness.run_steps(state, cfg, WARMUP)
    physinv.load_carriers(state, (start, start_scalars))
    store = {k: gather.pinned_copy(v) for k, v in start.items()}
    scalars = dict(start_scalars)
    devices = (0, 1) if ngpu >= 2 else (0,)
    report: dict = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mgstream.run_mgstream(
            store, cfg, tile_nx, tile_ny,
            halo=harness.halo_radius(cfg), nsteps=nsteps, devices=devices,
            nbuffers=2, write_mode="shadow",
            inventory_fn=physinv.carrier_inventory, nz=int(cfg.nz),
            tile_state_factory=driver.make_physics_tile_state,
            scalars=scalars, report=report)
    cp.cuda.runtime.deviceSynchronize()

    stale_digests = nestbridge.coupled_parent_digests(state, cfg)
    moved = nestbridge.refresh_nest_fields(state, store, cfg)
    cp.cuda.runtime.deviceSynchronize()
    fixed_digests = nestbridge.coupled_parent_digests(state, cfg)

    # The feedback direction: a coupler write onto the parent state must
    # reach the store or it is discarded at the next sweep.
    marker = float(np.asarray(_as_numpy(state.thp)).ravel()[0]) + 7.5
    state.thp[0, 0, 0] = marker
    before_commit = float(np.asarray(store["state/thp"]).ravel()[0])
    nestbridge.commit_nest_fields(state, store, cfg)
    after_commit = float(np.asarray(store["state/thp"]).ravel()[0])

    store_moved = sum(int(a.nbytes) for a in store.values())
    del store, state, _d
    cp.get_default_pinned_memory_pool().free_all_blocks()
    cp.get_default_memory_pool().free_all_blocks()

    stale = stale_digests == t0_digests
    advanced = resident_digests != t0_digests
    fixed = fixed_digests == resident_digests
    fb_dropped = before_commit != marker
    fb_fixed = after_commit == marker

    print(f"  streamed across {devices}, {report['tiles']} tiles, "
          f"{nsteps} steps, fires {_fire_line(resident_fires) or 'NONE'}")
    print(f"  the window moved the forcing fields (resident != t0):   "
          f"{advanced}")
    print(f"  the coupler's input after a streamed sweep == t0:       "
          f"{stale}"
          + ("   <-- the child would be forced from the INITIAL CONDITION"
             if stale else ""))
    n_stale = sum(1 for k in resident_digests
                  if stale_digests.get(k) != resident_digests[k])
    print(f"  forcing fields that would be WRONG:                     "
          f"{n_stale}/{len(resident_digests)}")
    print(f"  after refresh_nest_fields ({moved / 2**20:.0f} MiB of a "
          f"{store_moved / 2**20:.0f} MiB store), input == resident: {fixed}")
    print(f"  a feedback write onto the parent state reaches the store "
          f"only via commit_nest_fields: dropped={fb_dropped} "
          f"committed={fb_fixed}")
    return {
        "devices": list(devices), "kinds": kinds, "steps": nsteps,
        "window_moved": advanced, "stale": stale, "n_stale": n_stale,
        "n_fields": len(resident_digests), "refresh_fixes": fixed,
        "refresh_bytes": moved, "store_bytes": store_moved,
        "feedback_dropped": fb_dropped, "feedback_commit_fixes": fb_fixed,
        # The probe passes when it has PROVEN the gap and proven the fix.
        # "stale" being False would mean the streamed sweep somehow wrote the
        # state, which would be a different and worse finding.
        "ok": bool(advanced and stale and fixed and fb_dropped and fb_fixed),
    }


def _hash(digests: dict) -> str:
    import hashlib
    h = hashlib.blake2b(digest_size=16)
    for k in sorted(digests):
        h.update(k.encode())
        h.update(str(digests[k]).encode())
    return h.hexdigest()


# --------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=CX)
    ap.add_argument("--ny", type=int, default=CY)
    ap.add_argument("--tile-nx", type=int, default=CTILE_NX)
    ap.add_argument("--tile-ny", type=int, default=CTILE_NY)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--nz", type=int, default=NZ)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--scaling", action="store_true")
    ap.add_argument("--scaling-nx", type=int, default=512)
    ap.add_argument("--scaling-ny", type=int, default=0)
    ap.add_argument("--scaling-tile", type=int, default=128)
    ap.add_argument("--scaling-steps", type=int, default=8)
    ap.add_argument("--skip-correctness", action="store_true")
    ap.add_argument("--skip-negative", action="store_true")
    ap.add_argument("--nest", action="store_true",
                    help="ARM C: force a nest from a streamed parent")
    ap.add_argument("--geography", action="store_true",
                    help="real Lambert grid, real terrain, two devices")
    ap.add_argument("--only", default=None,
                    help="run only correctness arms whose label contains this")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    import cupy as cp
    ngpu = cp.cuda.runtime.getDeviceCount()
    geom = dict(nx=args.nx, ny=args.ny, tile_nx=args.tile_nx,
                tile_ny=args.tile_ny, nsteps=args.steps, nz=args.nz)

    print("=" * 78)
    print(f"MULTI-GPU STREAMING WITH PHYSICS   rung {RUNG!r}")
    print("=" * 78)
    print(f"visible GPUs: {ngpu}   "
          f"domain {args.nx}x{args.ny}x{args.nz}  "
          f"tile {args.tile_nx}x{args.tile_ny}  steps {args.steps}")
    cfg, _s, ss, _r, rs, fires, secs = reference(
        args.nx, args.ny, args.steps, nz=args.nz)
    print(f"dt={cfg.dt} s  radt={cfg.radt_minutes} min  "
          f"cudt={cfg.cudt_minutes} min  halo={harness.halo_radius(cfg)}")
    print(f"RESIDENT reference: {secs:.2f} s, "
          f"fires in the compared window: {_fire_line(fires) or 'NONE'}")
    if not fires.get("radiation") or not fires.get("cumulus"):
        print("  *** WARNING: radiation or cumulus never fired in the "
              "reference window.")
        print("  *** Every comparison below is then blind to them and the "
              "arm proves nothing.")
    print()

    pos, neg, rep, scale, nest, geo = [], [], {}, [], {}, []
    if not args.skip_correctness:
        print("-- correctness (every arm must be bit-exact) " + "-" * 33)
        pos = correctness(ngpu, only=args.only, **geom)
        print()
        print("-- repeatability " + "-" * 61)
        rep = repeatability(ngpu, reps=args.reps, **geom)
        print()
    if not args.skip_negative:
        print("-- negative controls (every one must FIRE) " + "-" * 35)
        neg = negative_controls(ngpu, **geom)
        print()
    if args.geography:
        print("-- real projection, real terrain, two devices " + "-" * 32)
        geo = geography_arms(ngpu, nsteps=args.steps, nz=args.nz)
        print()
    if args.nest:
        print("-- ARM C: a nest forced from a streamed parent " + "-" * 31)
        nest = nest_probe(ngpu, **geom)
        print()
    if args.scaling:
        print("-- scaling, physics on " + "-" * 55)
        scale = scaling(ngpu, nx=args.scaling_nx,
                        ny=args.scaling_ny or args.scaling_nx,
                        tile=args.scaling_tile, nsteps=args.scaling_steps,
                        nz=args.nz, reps=args.reps)
        print()

    skipped = ([r for r in pos if r.get("not_run")]
               + [r for r in neg if r.get("not_run")])
    bad = ([r for r in pos if not r["ok"] and not r.get("not_run")]
           + [r for r in geo if not r["ok"]])
    dead = [r for r in neg if not r.get("ok") and not r.get("not_run")]
    ran = len(pos) + len(geo) - len([r for r in pos if r.get("not_run")])
    print("=" * 78)
    print(f"correctness: {ran - len(bad)}/{ran} as specified")
    print(f"negative:    {len(neg) - len(dead) - len([r for r in neg if r.get('not_run')])}"
          f"/{len(neg) - len([r for r in neg if r.get('not_run')])} fired")
    if skipped:
        # A silent omission reads as coverage.  Name every arm that did not
        # run and why, at the same volume as the ones that did.
        print(f"NOT RUN:     {len(skipped)} arm(s) never started or ran out "
              "of VRAM on a shared card:")
        for r in skipped:
            print(f"  - {r['label']}: {str(r.get('error'))[:110]}")
    if rep:
        print(f"repeatable:  {rep['exact']}")
    if nest:
        print(f"nest probe:  {nest['ok']}  "
              f"(gap proven: {nest['stale']}, fix proven: "
              f"{nest['refresh_fixes']})")
    verdict = (not bad and not dead and (not rep or rep["exact"])
               and (not nest or nest["ok"]) and not skipped)
    for r in bad:
        print(f"  * FAILED: {r['label']}"
              + (f"  {r.get('error', '')}" if r.get("error") else ""))
    for r in dead:
        print(f"  * CONTROL DID NOT FIRE: {r['label']}")
    print(f"GATE: {'PASS' if verdict else 'FAIL'}")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"ngpu": ngpu, "rung": RUNG, "geometry": geom,
                       "reference_fires": fires, "correctness": pos,
                       "negative": neg, "repeatability": rep,
                       "geography": geo,
                       "scaling": scale, "nest": nest, "verdict": verdict},
                      fh, indent=2, default=str)
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
