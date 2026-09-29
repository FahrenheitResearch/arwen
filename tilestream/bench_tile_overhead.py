"""What one TILE costs a streamed step, apart from the columns it computes.

The pace model priced a streamed step as a per-column rate times the
domain's columns, floored by the bus.  That is right for the tilings the
rate was measured on (nine 400x300 tiles at 1.195x redundancy) and wrong
by two orders of magnitude for the tilings a nearly full card produces:
a 206x204x49 domain swept in 1,190 tiles of 6x6 (halo 18, 49.95x
redundancy) stepped at 237-547 s against a quoted 0.45-1.2 s, and a
profile of that run put 81% of its wall time in the per-tile health sync
inside ``tilestream.driver._sweep``.  Two costs were missing: the halo
work (the redundancy) and a fixed cost every tile pays for its own kernel
sequence and its sync, whatever its size.

This sweep separates them.  One full-physics domain is stepped resident
and then streamed from a pinned host store at a ladder of tile sizes, and
every sweep's wall time is fitted as::

    seconds_per_step = tile_seconds * ntiles + column_seconds * window_columns

where ``window_columns`` is every tile's compute window summed (the
domain's columns times the redundancy).  ``tile_seconds`` belongs with
this fit's own ``column_seconds`` and is not the figure
:mod:`gpuwm.core.pace` charges: the pace model charges a different column
term, and its per-tile figure is fitted net of that term
(:data:`gpuwm.core.pace.TILE_SECONDS_LOW`).

Timing discipline is ``bench_physics``'s: sweeps are timed between full
device synchronizations (``TiledRun`` records ``sweep_seconds``, and the
sweep observer passed below keeps that barrier in place), the
buffer construction is reported separately and never divided into a
step, and warmup sweeps -- the first carries kernel compilation and the
first radiation call -- are discarded.

    python -m tilestream.bench_tile_overhead --n 192 --tiles 96,48,32,24,16,12
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import warnings


#: The rung the reported case ran: Morrison, YSU + MM5 surface layer,
#: Noah, RTE+RRTMGP, no cumulus.  ``ztop=20000`` for the reason
#: ``bench_physics`` states (an 8 km top pads RRTMGP past its layer limit).
FULL_NO_CUMULUS = dict(moist=True, mp_physics=10, ztop=20000.0, km_opt=4,
                       sf_sfclay_physics=91, bl_pbl_physics=1, bldt=0.0,
                       sf_surface_physics=2, ra_sw_physics=4,
                       ra_lw_physics=4, radt_minutes=12.0, cu_physics=0)


def fit(rows):
    """Least squares for ``t = a * ntiles + b * window_columns``, both >= 0.

    Two unknowns and a handful of rows, solved in closed form so the
    instrument has no dependency the benchmark does not already import.
    """
    sxx = sum(r["ntiles"] ** 2 for r in rows)
    syy = sum(r["window_columns"] ** 2 for r in rows)
    sxy = sum(r["ntiles"] * r["window_columns"] for r in rows)
    sxt = sum(r["ntiles"] * r["seconds_per_step"] for r in rows)
    syt = sum(r["window_columns"] * r["seconds_per_step"] for r in rows)
    det = sxx * syy - sxy * sxy
    if det <= 0:
        return None
    a = (sxt * syy - syt * sxy) / det
    b = (syt * sxx - sxt * sxy) / det
    if a < 0 or b < 0:
        # One term explains nothing measurable: refit on the other alone.
        if a < 0:
            a, b = 0.0, syt / syy
        else:
            a, b = sxt / sxx, 0.0
    return {"tile_seconds": a, "column_seconds": b}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=192, help="domain edge (square)")
    ap.add_argument("--nz", type=int, default=49)
    ap.add_argument("--dx", type=float, default=3000.0)
    ap.add_argument("--dt", type=float, default=15.0)
    ap.add_argument("--tiles", default="96,48,32,24,16,12",
                    help="interior tile edges, comma separated; each must "
                         "divide --n")
    ap.add_argument("--nbuffers", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--json", default=None,
                    help="write the rows and the fit to this file")
    args = ap.parse_args()

    import cupy as cp

    from tilestream import driver, gather, harness
    from tilestream import physics_inventory as physinv
    from tilestream import spec as tspec

    cfg = harness.make_config(args.n, args.n, args.nz, dx=args.dx,
                              dy=args.dx, dt=args.dt, **FULL_NO_CUMULUS)
    halo = harness.halo_radius(cfg)
    columns = args.n * args.n
    props = cp.cuda.runtime.getDeviceProperties(0)
    card = props["name"].decode() if isinstance(props["name"], bytes) \
        else str(props["name"])
    print(f"{card}: {args.n}x{args.n}x{args.nz}, dx {args.dx:.0f} m, "
          f"dt {args.dt:.0f} s, halo {halo}", flush=True)

    started = time.perf_counter()
    state, drv = harness.make_physics_state(cfg, harness.DEFAULT_SEED)
    harness.run_steps(state, cfg, args.warmup)
    cp.cuda.runtime.deviceSynchronize()
    print(f"state built + {args.warmup} warmup steps in "
          f"{time.perf_counter() - started:.1f} s", flush=True)

    resident = []
    for _ in range(args.steps):
        cp.cuda.runtime.deviceSynchronize()
        t = time.perf_counter()
        harness.run_steps(state, cfg, 1)
        cp.cuda.runtime.deviceSynchronize()
        resident.append(time.perf_counter() - t)
    resident_step = statistics.median(resident)
    print(json.dumps({"road": "resident", "seconds_per_step": resident_step,
                      "samples": resident}), flush=True)

    inv = physinv.carrier_inventory(state)
    start = {k: cp.asnumpy(v) for k, v in inv.items()}
    scalars = physinv.carrier_scalars(state)
    del state, drv, inv
    cp.get_default_memory_pool().free_all_blocks()

    rows = []
    for tile in (int(t) for t in args.tiles.split(",") if t.strip()):
        if args.n % tile:
            raise SystemExit(f"tile {tile} does not divide --n {args.n}")
        specs = tspec.plan_tiles(args.n, args.n, tile, tile, halo, True)
        window_columns = sum(s.cnx * s.cny for s in specs)
        store = {k: gather.pinned_copy(v) for k, v in start.items()}
        report: dict = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            driver.run_tiled(
                store, cfg, tile, tile, halo=halo,
                nsteps=args.warmup + args.steps, nbuffers=args.nbuffers,
                report=report, inventory_fn=physinv.carrier_inventory,
                nz=int(cfg.nz),
                tile_state_factory=driver.make_physics_tile_state,
                scalars=dict(scalars),
                # A sweep observer keeps the per-sweep device barrier, so
                # every ``sweep_seconds`` entry is wall time and not issue
                # time (a chained run otherwise defers the seam).
                on_sweep=lambda *_: None)
        sweeps = list(report.get("sweep_seconds") or [])
        timed = sweeps[args.warmup:] if len(sweeps) > args.warmup else sweeps
        row = {"road": "streamed", "tile": tile, "ntiles": len(specs),
               "window": specs[0].cnx, "window_columns": window_columns,
               "redundancy": window_columns / columns,
               "seconds_per_step": statistics.median(timed),
               "samples": timed,
               "setup_seconds": report.get("setup_seconds"),
               "multiple_of_resident": statistics.median(timed) / resident_step}
        rows.append(row)
        print(json.dumps(row), flush=True)
        del store
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()

    result = fit(rows)
    summary = {"card": card, "n": args.n, "nz": args.nz, "halo": halo,
               "nbuffers": args.nbuffers, "resident_seconds_per_step":
               resident_step, "fit": result, "rows": rows}
    if result:
        for row in rows:
            row["fitted_seconds_per_step"] = (
                result["tile_seconds"] * row["ntiles"]
                + result["column_seconds"] * row["window_columns"])
        print(f"fit: {result['tile_seconds'] * 1e3:.2f} ms per tile per "
              f"step + {result['column_seconds'] * 1e9:.1f} ns per window "
              f"column per step", flush=True)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=1)


if __name__ == "__main__":
    main()
