"""Is streaming still nearly free when the PHYSICS is on?

Everything measured so far splits along an awkward line.  The streaming COST
(1.047x on the 5090, 2.82x the area) was measured on DRY DYNAMICS.  The
streaming CORRECTNESS (229 carriers bit-exact) was measured at
``full+MYNN+Noah-MP``.  Nobody has measured the streaming cost WITH the
physics on -- ``tilestream/linux_physics.py`` says so in its own docstring:
"Nothing here is timed."

That gap matters, because the whole scheme rests on one ratio:

    bytes of state per cell            32.26 B dry, ~275 B full physics
    ------------------------  vs  PCIe
    compute time per cell              ~3.7 ns dry,      ? ns full physics

Dry needs 8.7 GB/s to keep up and PCIe delivers ~26, so the transfers vanish
behind the arithmetic and the overhead is a rounding error.  Full physics
moves 8.5x more bytes per cell.  If it does not ALSO compute 8.5x more slowly
per cell, the transport stops hiding and the headline number is wrong for the
configuration anybody would actually run.

So: the same three-rung ladder the dry lane used, at a physics rung.

    monolithic          the resident baseline -- what the card does today
    tiled / VRAM store  the tiling tax alone: halo redundancy, more kernel
                        launches, smaller arrays.  No PCIe.
    tiled / host store   the above PLUS every byte crossing PCIe

The gap between rungs two and three is the price of living in host RAM, which
is the number this whole project is about.  Rung two is what separates "PCIe
is too slow" from "tiling itself costs something", and without it a bad result
cannot be attributed.

Measurement discipline, all of it learned the hard way on this project:

* the digest of the store is printed for every rung and MUST agree, because a
  transport that quietly skips work looks exactly like a speedup;
* warmup steps are discarded -- the first step of a physics rung pays for
  kernel compilation and the radiation packer;
* >= 3 timed reps, median reported, spread called out rather than averaged
  away;
* ``ns_per_cell`` is normalised by the domain's OWN cells, so the halo
  redundancy a tiled run pays shows up as a cost instead of hiding in a
  larger denominator.

    python -m tilestream.bench_physics --n 768 --tile 384
"""

from __future__ import annotations

import argparse
import statistics
import time
import warnings

import numpy as np


#: The rung definitions, inline rather than imported from ``test_gate``.
#: This benchmark has to run on machines whose checkout predates the physics
#: matrix, and a benchmark that cannot run where the hardware is is useless.
#: Keep in sync with ``test_gate.PHYSICS_RUNGS``; the gate remains the
#: authority on which of these are bit-exact.
#: ``ztop=20000.0`` is load-bearing, not decoration: the harness default is an
#: 8 km top, and RRTMGP then pads to 140 layers against its own limit of 128
#: and raises.  Copied verbatim from ``test_gate.PHYSICS_MOIST``.
_MOIST = dict(moist=True, mp_physics=10, ztop=20000.0)
_FULL = dict(_MOIST, km_opt=4, sf_sfclay_physics=91, bl_pbl_physics=1,
             bldt=0.0, sf_surface_physics=2, ra_sw_physics=4,
             ra_lw_physics=4, radt_minutes=12.0, cu_physics=1,
             cudt_minutes=5.0)

RUNGS: dict[str, dict] = {
    "dry": dict(),
    "mp10": dict(_MOIST),
    "full(real74)+KF": dict(_FULL),
    "full+MYNN": dict(_FULL, sf_sfclay_physics=5, bl_pbl_physics=5),
    "full+MYNN+Noah-MP": dict(_FULL, sf_sfclay_physics=5, bl_pbl_physics=5,
                              sf_surface_physics=4),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=768,
                    help="domain edge (square)")
    ap.add_argument("--tile", type=int, default=384,
                    help="interior tile edge; must divide --n")
    ap.add_argument("--nz", type=int, default=49)
    ap.add_argument("--rung", default="full+MYNN+Noah-MP")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--nbuffers", type=int, default=2)
    ap.add_argument("--skip-mono", action="store_true",
                    help="when the resident baseline does not fit")
    args = ap.parse_args()

    import cupy as cp

    from tilestream import driver, gather, harness
    from tilestream import physics_inventory as physinv
    from tilestream import spec as tspec
    if args.rung not in RUNGS:
        raise SystemExit(f"unknown rung {args.rung!r}; have {list(RUNGS)}")
    cfg = harness.make_config(args.n, args.n, args.nz, **RUNGS[args.rung])
    halo = harness.halo_radius(cfg)
    cells = args.n * args.n * args.nz
    free0, total = cp.cuda.runtime.memGetInfo()
    print(f"{cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}  "
          f"VRAM {free0/2**30:.2f} free / {total/2**30:.2f} GiB")
    print(f"rung={args.rung}  {args.n}^2 x {args.nz} = {cells/1e6:.1f} Mcell  "
          f"tile={args.tile}  halo={halo}")

    # ---------------------------------------------------------------- state
    t0 = time.perf_counter()
    state, drv = harness.make_physics_state(cfg, harness.DEFAULT_SEED)
    harness.run_steps(state, cfg, args.warmup)
    cp.cuda.runtime.deviceSynchronize()
    print(f"state built + {args.warmup} warmup steps in "
          f"{time.perf_counter()-t0:.1f}s; "
          f"VRAM {cp.cuda.runtime.memGetInfo()[0]/2**30:.2f} GiB free")

    inv = physinv.carrier_inventory(state)
    bytes_per_cell = sum(v.nbytes for v in inv.values()) / cells
    print(f"{len(inv)} carriers, {bytes_per_cell:.1f} B/cell of state")

    def timed(fn, label):
        """Median of --reps timings, with the spread kept visible."""
        samples = []
        for _ in range(args.reps):
            cp.cuda.runtime.deviceSynchronize()
            t = time.perf_counter()
            fn()
            cp.cuda.runtime.deviceSynchronize()
            samples.append((time.perf_counter() - t) / args.steps)
        med = statistics.median(samples)
        spread = max(samples) / min(samples)
        print(f"  {label:<22s} {med*1e3:9.2f} ms/step  "
              f"{med/cells*1e9:7.3f} ns/cell   spread={spread:.3f}"
              + ("  <-- NOISY" if spread > 1.10 else ""))
        return med

    results = {}

    # ------------------------------------------------------- 1. monolithic
    if not args.skip_mono:
        results["monolithic"] = timed(
            lambda: harness.run_steps(state, cfg, args.steps), "monolithic")

    # Snapshot AFTER the monolithic timing so all three rungs start from the
    # same state; the digests below are what proves they did the same work.
    start = {k: np.ascontiguousarray(cp.asnumpy(v)) for k, v in inv.items()}
    scalars = physinv.carrier_scalars(state)
    del state, drv
    cp.get_default_memory_pool().free_all_blocks()

    specs = tspec.plan_tiles(args.n, args.n, args.tile, args.tile, halo, True)
    tspec.validate_plan(specs, args.n, args.n)
    red = sum(s.cnx * s.cny for s in specs) / (args.n * args.n)
    print(f"{len(specs)} tiles, compute {specs[0].cny}x{specs[0].cnx}, "
          f"redundancy {red:.4f}x")

    def tiled(store_kind):
        if store_kind == "host":
            store = {k: gather.pinned_copy(v) for k, v in start.items()}
        else:
            store = {k: cp.asarray(v) for k, v in start.items()}
        sc = dict(scalars)
        report: dict = {}

        def run():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                driver.run_tiled(
                    store, cfg, args.tile, args.tile, halo=halo,
                    nsteps=args.steps, nbuffers=args.nbuffers, report=report,
                    inventory_fn=physinv.carrier_inventory, nz=int(cfg.nz),
                    tile_state_factory=driver.make_physics_tile_state,
                    scalars=sc)

        med = timed(run, f"tiled / {store_kind} store")
        dig = physinv.field_digests(physinv.carrier_inventory(store))
        h = hash(tuple(sorted(dig.items()))) & 0xFFFFFFFFFFFF
        moved = report.get("gathered_bytes", 0) + report.get("scattered_bytes", 0)
        print(f"      digest {h:012x}   moved {moved/1e9:.2f} GB total  "
              f"{moved/args.steps/args.reps/med/1e9 if med else 0:.1f} GB/s "
              f"(per timed step)")
        del store
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        return med

    results["tiled_vram"] = tiled("vram")
    results["tiled_host"] = tiled("host")

    # ------------------------------------------------------------- verdict
    print()
    base = results.get("monolithic")
    if base:
        print(f"tiling tax (VRAM store)   "
              f"{results['tiled_vram']/base:.3f}x")
        print(f"streaming cost (host)     "
              f"{results['tiled_host']/base:.3f}x   <- the number that matters")
    print(f"host store vs VRAM store  "
          f"{results['tiled_host']/results['tiled_vram']:.3f}x   "
          f"(PCIe alone)")


if __name__ == "__main__":
    main()
