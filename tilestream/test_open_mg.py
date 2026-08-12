"""OPEN LATERAL BOUNDARIES on N GPUs: the multi-GPU arm of the same question.

:mod:`tilestream.mgstream` shares one pinned host domain's tiles out over
several GPUs.  It was written on its own lane and gated PERIODIC only, and it
reaches the same two places this lane found wrong:

* ``_spec.plan_tiles(dx, dy, tile_nx, tile_ny, halo, periodic)`` -- one
  boolean for two axes, the exact defect that corrupts the y-boundary tile
  rows of an ``open_x``-only domain;
* ``_harness.tile_config(cfg, cnx, cny)`` -- one tile cfg for every buffer,
  carrying ``open_x``/``open_y`` to all four window edges.

The module here is ``mgstream.py`` verbatim from the ``tilestream-mgstream``
branch with ONE change: ``periodic_x`` / ``periodic_y`` threaded into
``plan_tiles``.  ``periodic=`` still sets both, so that lane's own gate is
unaffected.

WHAT IS AND IS NOT COVERED.  ``run_mgstream`` hard-binds
``_driver.make_tile_state`` as its buffer factory (mgstream.py:268), which
builds a dry ``DomainState`` and attaches no ``PhysicsDriver``, so the
multi-GPU path cannot run a physics rung at all -- with or without open
boundaries.  This gate therefore runs DRY only, and says so rather than
implying physics coverage it does not have.  That is not a limitation of open
boundaries: the two defects are geometric, they are visible in nine dry
carriers after one step, and the single-GPU matrix covers the physics rungs.

Three device lists, and the split is what makes a failure interpretable:

``(0,)``      one worker, one GPU -- must equal ``driver.run_tiled``
``(0, 0)``    two workers, ONE GPU -- separates "the partition is wrong"
              from "two physical GPUs interfere"
``(0, 1)``    two workers, two GPUs -- the real thing

    python -m tilestream.test_open_mg              # every arm it can run
    python -m tilestream.test_open_mg --devices=0  # single-GPU rows only
"""

from __future__ import annotations

import sys
import time
import warnings

import numpy as np

from gpuwm.core import streaming
from tilestream import driver, gather, harness, mgstream
from tilestream import spec as tspec
from tilestream import test_open_bc as gate

NX, NY = 256, 192
TX, TY = 32, 32
STEPS = (1, 3, 8)


def host_store(arrays) -> dict:
    return {k: gather.pinned_copy(v) for k, v in arrays.items()}


def build_dry_domain(cfg, arm, *, seed=gate.SEED, warmup: int = 1):
    """The domain ``mgstream``'s hard-bound buffer factory can actually serve.

    ``driver.make_tile_state`` builds its base state from a CONSTANT 300 K
    profile (driver.py:253-256), which is ``harness.make_state``'s base and
    NOT ``harness.make_physics_state``'s WK82 sounding.  ``thb/pb/alb/phb``
    are setup arrays, never gathered, so a domain built on the sounding and
    tiles built on the constant profile integrate different base states --
    and the symptom is not subtle or localised: all nine dry carriers, max|d|
    5468 Pa in phi', verdict "uniform", identical on every arm including the
    periodic one.  That uniformity is the tell, and it is why this gate
    builds the domain the factory's way instead of assuming mgstream can take
    the join lane's states.
    """
    geo = harness.make_geography(
        cfg, terrain=False,
        periodic_faces_x=streaming._periodic_axes(cfg)[0],
        periodic_faces_y=streaming._periodic_axes(cfg)[1])
    state = harness.make_state(cfg, seed=seed, geography=geo)
    if warmup:
        harness.run_steps(state, cfg, int(warmup))
    return state, geo


def dry_reference(cfg, arm, nsteps, *, seed=gate.SEED, warmup: int = 1):
    """The resident answer on the same domain, as ``gather.inventory``."""
    import cupy as cp

    state, _geo = build_dry_domain(cfg, arm, seed=seed, warmup=warmup)
    harness.run_steps(state, cfg, int(nsteps))
    out = {k: cp.asnumpy(v).copy() for k, v in gather.inventory(state).items()}
    bad = sum(int(np.count_nonzero(~np.isfinite(v))) for v in out.values()
              if v.dtype.kind == "f")
    if bad:
        raise gate.UnstableReference(
            f"the MONOLITHIC reference has {bad} non-finite cells after "
            f"{nsteps} steps at dt={cfg.dt} s")
    del state
    cp.get_default_memory_pool().free_all_blocks()
    return out


def run_case(cfg, arm, nsteps, devices, *, halo, write_mode="shadow",
             partition="block", plan_axes=None) -> dict:
    """One mgstream configuration, against the resident answer."""
    import cupy as cp

    px, py = streaming._periodic_axes(cfg) if plan_axes is None else plan_axes

    ref = dry_reference(cfg, arm, nsteps)

    state, _geo = build_dry_domain(cfg, arm, warmup=1)
    geo_inv = {k: gather.pinned_copy(v)
               for k, v in driver.geography_inventory(state).items()}
    store = host_store({k: cp.asnumpy(v) for k, v in
                        gather.inventory(state).items()})
    del state
    cp.get_default_memory_pool().free_all_blocks()

    report: dict = {}
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mgstream.run_mgstream(
            store, cfg, TX, TY, halo=halo, nsteps=nsteps,
            devices=tuple(devices), periodic_x=px, periodic_y=py,
            write_mode=write_mode, partition=partition,
            geography=geo_inv, check_geography=False)
    cp.cuda.runtime.deviceSynchronize()
    report["seconds"] = time.perf_counter() - t0

    specs = tspec.plan_tiles(NX, NY, TX, TY, halo, periodic_x=px,
                             periodic_y=py)
    # AN EMPTY COMPARISON PASSES, and the first version of this took one.
    # ``physinv.carrier_inventory`` keys the persisted arrays ``state/<name>``
    # while ``gather.inventory`` keys them ``<name>``, so an intersection on
    # raw keys was EMPTY -- zero differing carriers, i.e. PASS, on every arm
    # including the ones that must fail.  The negative control is what
    # caught it.  The set equality below is asserted so that a vacuous
    # comparison raises instead of passing, whatever the inventories do next.
    ref_dry = dict(ref)
    if set(ref_dry) != set(store):
        raise AssertionError(
            f"the reference and the store do not describe the same "
            f"inventory: reference-only {sorted(set(ref_dry) - set(store))}, "
            f"store-only {sorted(set(store) - set(ref_dry))}")
    res = gate.localise(ref_dry, {k: np.asarray(v) for k, v in store.items()},
                        specs, halo, NX, NY)
    res.update(report)
    del store, geo_inv
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    return res


def run_case_retrying(*args, tries: int = 4, **kwargs) -> dict:
    """:func:`run_case`, retrying a worker that died of a NEIGHBOUR.

    ``run_mgstream`` re-raises a worker's exception as ``MGStreamError:
    worker w (device d) failed``, which loses the cupy type, so this cannot
    key on ``OutOfMemoryError`` the way :func:`test_open_bc.retry_on_oom``
    does.  Both boards here are shared with several other agents' processes
    and the observed failure was exactly that -- the periodic arm died on
    ``devices=(0, 1)`` at 18.6 GiB already in use on device 0 and passed on
    re-run with no change at all.  Every retry is PRINTED, and a case that
    fails ``tries`` times is reported as the failure it is.
    """
    import cupy as cp

    for attempt in range(int(tries)):
        try:
            return run_case(*args, **kwargs)
        except (cp.cuda.memory.OutOfMemoryError, mgstream.MGStreamError) as e:
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
            if attempt == tries - 1:
                raise
            print(f"          (retry {attempt + 1}/{tries} on a shared "
                  f"board: {type(e).__name__}: {str(e)[:60]})")
            time.sleep(45.0)


def main(argv=None) -> int:
    import cupy as cp

    argv = list(sys.argv[1:] if argv is None else argv)
    ngpu = cp.cuda.runtime.getDeviceCount()
    single_only = "--devices=0" in argv
    device_lists = [(0,), (0, 0)]
    if ngpu >= 2 and not single_only:
        device_lists.append((0, 1))

    print(f"cupy {cp.__version__}  {ngpu} visible GPU(s)")
    print("=" * 78)
    print("OPEN LATERAL BOUNDARIES, MULTI-GPU (mgstream) -- DRY RUNG ONLY")
    print(f"  {NX}x{NY}x{gate.NZ} at dx=12 km, tile {TX}x{TY}, N in {STEPS}")
    print("  run_mgstream binds driver.make_tile_state, which attaches no")
    print("  PhysicsDriver, so no physics rung is reachable on this path")
    print("=" * 78)

    failures: list[str] = []
    for arm in ("periodic", "open_x", "open_xy"):
        cfg = gate.open_cfg(NX, NY, "dry", arm)
        halo = harness.halo_radius(cfg)
        px, py = streaming._periodic_axes(cfg)
        print(f"-- ARM {arm}  (plan periodic x={px} y={py}, halo {halo})")
        for devices in device_lists:
            for n in STEPS:
                try:
                    res = run_case_retrying(cfg, arm, n, devices, halo=halo)
                except Exception as exc:                    # noqa: BLE001
                    failures.append(f"{arm}/{devices}/N={n}: "
                                    f"{type(exc).__name__}: {exc}")
                    print(gate._line(f"devices={devices} N={n}", False,
                                     f"RAISED {type(exc).__name__}: "
                                     f"{str(exc)[:70]}"))
                    cp.get_default_memory_pool().free_all_blocks()
                    continue
                ok = res["bitexact"] and res["nonfinite"] == 0
                if not ok:
                    failures.append(f"{arm}/devices={devices}/N={n} "
                                    "not bit-exact")
                print(gate._line(f"devices={devices} N={n}", ok,
                                 f"{res['ntotal']} carriers "
                                 f"ndiff={res['ndiff']} "
                                 f"{res['seconds']:.1f}s"))
                if not ok:
                    print(f"          verdict={res.get('verdict')} "
                          f"worst={res.get('worst_field')} "
                          f"max|d|={res.get('max_abs'):.6g} "
                          f"x {res.get('x_extent')} y {res.get('y_extent')}")
                    break
        print()

    # THE CONTROL.  One boolean for two axes is what this lane fixed; on the
    # multi-GPU path it must still be the thing that decides the answer, or
    # the per-axis plumbing here is decorative.
    print("-- CONTROL: plan clamps the periodic y axis (MUST differ)")
    cfg = gate.open_cfg(NX, NY, "dry", "open_x")
    halo = harness.halo_radius(cfg)
    for devices in device_lists:
        try:
            res = run_case_retrying(cfg, "open_x", 3, devices, halo=halo,
                                    plan_axes=(False, False))
            fired = not res["bitexact"]
        except Exception as exc:                            # noqa: BLE001
            print(gate._line(f"devices={devices}", False,
                             f"RAISED {type(exc).__name__}: {str(exc)[:60]}"))
            failures.append(f"control devices={devices}: {exc}")
            continue
        print(gate._line(f"devices={devices}", fired,
                         f"ndiff={res['ndiff']}/{res['ntotal']}"
                         + ("" if not fired
                            else f" max|d|={res['max_abs']:.6g}")))
        if not fired:
            failures.append(f"control did NOT fire on devices={devices}")

    print()
    print("=" * 78)
    if failures:
        print(f"{len(failures)} FAILURE(S)")
        for f in failures:
            print(f"  - {f}")
    else:
        print("mgstream reproduces the resident answer on every arm and "
              "every device list it can run")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
