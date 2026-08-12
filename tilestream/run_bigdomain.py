"""Run the big-domain forecast and drop one product file per output time.

Separate from :mod:`tilestream.bigdomain` for the same reason
:mod:`tilestream.run_bench` is separate from :mod:`tilestream.bench`: the
stages differ by tens of gigabytes of pinned residency, and a stage that
inherits another stage's allocations is measuring the order it ran in.

    python -m tilestream.run_bigdomain selftest
    python -m tilestream.run_bigdomain ceiling  --n 1408
    python -m tilestream.run_bigdomain forecast --n 1408 --tile 352 \
        --minutes 120 --every 20 --out ./bigdomain-out

``ceiling`` is the control the whole exercise depends on: it tries to build
the SAME configuration monolithically and reports how it fails.  A figure
captioned "vanilla cannot run this" needs the refusal in a log, not an
inherited number.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def _vram():
    import cupy as cp
    free, total = cp.cuda.runtime.memGetInfo()
    return free / 2 ** 30, total / 2 ** 30


def _host_gib():
    from tilestream import hoststore
    mem = hoststore.host_memory()
    return mem["available"] / 2 ** 30, mem["total"] / 2 ** 30


def _banner(title: str) -> None:
    free, total = _vram()
    avail, htotal = _host_gib()
    print(f"\n=== {title} ===")
    print(f"    VRAM {free:.2f}/{total:.2f} GiB free | "
          f"host {avail:.1f}/{htotal:.1f} GiB available")
    sys.stdout.flush()


def stage_selftest(args) -> None:
    from tilestream import bigdomain

    _banner("SELFTEST: the slab build is the domain a one-shot build makes")
    out = bigdomain.selftest_slab_build_matches_monolithic(
        nx=args.n, ny=args.n - 32, nz=args.nz, slab_rows=args.slab)
    print(f"    {out['carriers']} carriers, "
          f"{len(out['differ'])} differ, missing {out['missing']}")
    if out["differ"]:
        for name in out["differ"][:20]:
            print(f"      DIFFER {name}")
    print("    " + ("AGREE - the construction is column-local"
                    if out["agree"] else "DISAGREE - DO NOT TRUST THE STATE"))
    if not out["agree"]:
        raise SystemExit(1)


def stage_ceiling(args) -> None:
    """Try to build the domain resident, and report exactly how it refuses.

    Two numbers, because they fail at different places: the pinned host store
    the streamed run needs, and the device allocation a monolithic run needs
    before it can take its first step.  The second is the one that decides
    whether "vanilla cannot run this" is true on THIS card.
    """
    import cupy as cp

    from tilestream import bigdomain, harness, hoststore
    from tilestream import physics_inventory as physinv

    over = {"cu_physics": 0} if args.no_cumulus else {}
    _banner(f"CEILING: monolithic {args.n}x{args.n}x{args.nz}, "
            f"cu_physics={0 if args.no_cumulus else 1}")
    cfg = bigdomain.big_config(args.n, args.n, args.nz, dx=args.dx,
                               dt=args.dt, **over)
    cells = args.n * args.n * args.nz
    print(f"    {cells / 1e6:.1f} Mcell, dx={cfg.dx/1000:.0f} km, "
          f"{args.n * cfg.dx / 1000:.0f} km across")

    # The carrier footprint, measured on a small state rather than quoted.
    probe = bigdomain.big_config(96, 80, args.nz, dx=args.dx, dt=args.dt,
                                 **over)
    geo = harness.make_geography(probe)
    st, drv = harness.make_physics_state(probe, geography=geo)
    harness.run_steps(st, probe, 1)
    inv = physinv.carrier_inventory(st)
    per_cell = sum(v.nbytes for v in inv.values()) / (96 * 80 * args.nz)
    manifest = hoststore.manifest_from_arrays(inv, args.nz, 80, 96)
    del st, drv, inv
    cp.get_default_memory_pool().free_all_blocks()
    want = hoststore.domain_bytes(manifest, args.nz, args.n, args.n)
    print(f"    {len(manifest)} carriers at {per_cell:.1f} B/cell -> "
          f"{want / 2**30:.2f} GiB of carriers for this domain")

    free, total = _vram()
    print(f"    the card has {total:.2f} GiB total, {free:.2f} GiB free")
    verdict = {"n": args.n, "carrier_gib": want / 2 ** 30,
               "vram_total_gib": total, "vram_free_gib": free}
    try:
        t0 = time.perf_counter()
        # The geography has to be the REAL one.  ``geography=None`` delegates
        # to ``default_builder``, which builds a FLAT base state, and
        # ``load_base`` then refuses the 3-D profiles a ``terrain_opt=1``
        # config allocates -- a ValueError that would be reported here as
        # "vanilla refused" when it is nothing of the kind.
        state, driver = harness.make_physics_state(
            cfg, geography=harness.make_geography(cfg))
        harness.run_steps(state, cfg, 1)
        print(f"    MONOLITHIC RAN in {time.perf_counter()-t0:.1f}s "
              "-- this domain is NOT past the resident ceiling")
        verdict["monolithic"] = "ran"
        del state, driver
    except Exception as exc:
        print(f"    MONOLITHIC REFUSED: {type(exc).__name__}: "
              f"{str(exc)[:400]}")
        verdict["monolithic"] = f"{type(exc).__name__}: {str(exc)[:400]}"
    cp.get_default_memory_pool().free_all_blocks()
    print("RESULT " + json.dumps(verdict))


def stage_bitexact(args) -> None:
    """Is the configuration the figures show bit-exact through the stream?

    ``test_gate`` certifies the physics ladder at ``harness``'s own
    ``make_config`` defaults.  The forecast changes three things it does not
    cover -- ``dx``/``dt``, ``cu_physics``, and an initial condition built by
    :mod:`tilestream.bigdomain` rather than by the harness -- and a figure
    that says "streamed" should rest on a measurement of THAT configuration,
    not on a neighbouring one.  So this runs the forecast's own config at a
    size where a monolithic run still fits, from the forecast's own initial
    state, and compares the whole 213-carrier manifest after ``--steps``.

    A domain of ``--n`` split into four tiles is deliberate: with one tile
    the halo is the whole domain and the test cannot fail.
    """
    import hashlib
    import warnings

    import cupy as cp

    from tilestream import bigdomain, driver, gather, harness
    from tilestream import physics_inventory as physinv
    from tilestream import spec as tspec

    over = {"cu_physics": 0} if args.no_cumulus else {}
    nx, ny = args.n, args.n
    _banner(f"BIT-EXACT: {nx}x{ny}x{args.nz} monolithic vs streamed, "
            f"forecast configuration")
    cfg = bigdomain.big_config(nx, ny, args.nz, dx=args.dx, dt=args.dt,
                               **over)
    halo = harness.halo_radius(cfg)
    tile = args.tile
    specs = tspec.plan_tiles(nx, ny, tile, tile, halo, True)
    tspec.validate_plan(specs, ny, nx)
    print(f"    cu_physics={cfg.cu_physics}  dx={cfg.dx/1000:.0f} km  "
          f"dt={cfg.dt:.0f} s  {len(specs)} tiles of {specs[0].cny}x"
          f"{specs[0].cnx}  halo {halo}  N={args.steps}")

    geo = harness.make_geography(cfg)
    noise = bigdomain.SeededNoise.draw(args.nz, ny, nx, seed=args.seed)
    bubbles = bigdomain.bubble_centres(nx, ny, float(cfg.dx), count=4,
                                       edge_cells=16)

    def digest(arrays) -> str:
        per = physinv.field_digests(arrays)
        h = hashlib.sha256()
        for name in sorted(per):
            h.update(name.encode())
            h.update(bytes.fromhex(per[name]))
        return h.hexdigest()[:16]

    state, drv = bigdomain.build_slab_state(cfg, geo, noise, 0,
                                            bubbles=bubbles,
                                            dx=float(cfg.dx))
    harness.run_steps(state, cfg, 1)          # lazy carriers
    start = {k: np.ascontiguousarray(cp.asnumpy(v))
             for k, v in physinv.carrier_inventory(state).items()}
    scal = physinv.carrier_scalars(state)
    geo_start = {k: np.ascontiguousarray(cp.asnumpy(v))
                 for k, v in driver.geography_inventory(state).items()}
    harness.run_steps(state, cfg, args.steps)
    mono = digest(physinv.carrier_inventory(state))
    del state, drv
    cp.get_default_memory_pool().free_all_blocks()

    store = {k: gather.pinned_copy(v) for k, v in start.items()}
    geo_store = {k: gather.pinned_copy(v) for k, v in geo_start.items()}
    kwargs = driver.geography_run_kwargs(
        cfg, None, geography=geo_store,
        geography_fn=harness.neutral_geography)
    kwargs["scalars"] = dict(scal)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        driver.run_tiled(store, cfg, tile, tile, halo=halo,
                         nsteps=args.steps, nbuffers=2, write_mode="ring",
                         **kwargs)
    tiled = digest(physinv.carrier_inventory(store))
    ok = mono == tiled
    print(f"    monolithic {mono}   streamed {tiled}   "
          + ("BIT-EXACT" if ok else "*** DIFFER ***"))
    print("RESULT " + json.dumps({"n": nx, "tile": tile, "steps": args.steps,
                                  "carriers": len(start),
                                  "monolithic": mono, "streamed": tiled,
                                  "bit_exact": ok}))
    if not ok:
        raise SystemExit(1)


def stage_forecast(args) -> None:
    import cupy as cp
    import warnings

    from tilestream import bigdomain, driver, harness
    from tilestream import physics_inventory as physinv
    from tilestream import spec as tspec

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    _banner(f"FORECAST {args.n}x{args.n}x{args.nz} streamed, tile {args.tile}")
    over = {"cu_physics": 0} if args.no_cumulus else {}
    cfg = bigdomain.big_config(args.n, args.n, args.nz, dx=args.dx,
                               dt=args.dt, **over)
    halo = harness.halo_radius(cfg)
    specs = tspec.plan_tiles(args.n, args.n, args.tile, args.tile, halo, True)
    tspec.validate_plan(specs, args.n, args.n)
    red = sum(s.cnx * s.cny for s in specs) / (args.n * args.n)
    print(f"    cu_physics={cfg.cu_physics}  bl_pbl={cfg.bl_pbl_physics}  "
          f"mp={cfg.mp_physics}  sf_surface={cfg.sf_surface_physics}")
    print(f"    dx={cfg.dx/1000:.0f} km  dt={cfg.dt:.0f} s  "
          f"{args.n * cfg.dx / 1000:.0f} km across  "
          f"{args.n**2 * args.nz / 1e6:.1f} Mcell")
    print(f"    {len(specs)} tiles of {specs[0].cny}x{specs[0].cnx} "
          f"(halo {halo}), redundancy {red:.3f}x")
    sys.stdout.flush()

    t0 = time.perf_counter()
    geo = harness.make_geography(cfg)
    print(f"    Lambert grid: lat {geo.lat.min():.2f}..{geo.lat.max():.2f} "
          f"lon {geo.lon.min():.2f}..{geo.lon.max():.2f}, "
          f"terrain {geo.terrain.min():.0f}..{geo.terrain.max():.0f} m "
          f"({time.perf_counter()-t0:.1f}s)")

    t0 = time.perf_counter()
    noise = bigdomain.SeededNoise.draw(args.nz, args.n, args.n,
                                       seed=args.seed)
    bubbles = bigdomain.bubble_centres(args.n, args.n, float(cfg.dx),
                                       count=args.bubbles)
    print(f"    noise {noise.nbytes / 2**30:.2f} GiB, {len(bubbles)} warm "
          f"bubbles ({time.perf_counter()-t0:.1f}s)")
    sys.stdout.flush()

    t0 = time.perf_counter()
    manifest = bigdomain.carrier_manifest_for(cfg)
    print(f"    manifest: {len(manifest)} carriers "
          f"({time.perf_counter()-t0:.1f}s)")
    t0 = time.perf_counter()
    store, geo_store, scalars, missing = bigdomain.build_store_by_slabs(
        cfg, geo, slab_rows=args.slab, noise=noise, bubbles=bubbles,
        manifest=manifest,
        amplitudes=(bigdomain.GATE_AMPLITUDES
                    if args.perturbation == "gate" else None))
    nbytes = sum(v.nbytes for v in store.values())
    print(f"    store {nbytes / 2**30:.2f} GiB pinned "
          f"({nbytes / (args.n**2 * args.nz):.1f} B/cell), "
          f"geography {sum(v.nbytes for v in geo_store.values())/2**30:.2f} "
          f"GiB, built in {time.perf_counter()-t0:.1f}s")
    if missing:
        print(f"    lazily-allocated carriers left at zero: {missing}")
    del noise
    sys.stdout.flush()

    kwargs = driver.geography_run_kwargs(
        cfg, None, geography=geo_store,
        geography_fn=harness.neutral_geography)
    kwargs["scalars"] = scalars

    steps_per_dump = max(1, int(round(args.every * 60.0 / cfg.dt)))
    ndumps = max(1, int(round(args.minutes / args.every)))

    def dump(tag: str) -> None:
        t = time.perf_counter()
        snap = bigdomain.snapshot(store, geo_store, cfg,
                                  elapsed_s=scalars["elapsed_seconds"],
                                  slab_rows=args.refl_slab)
        path = out_dir / f"bigdom_{args.n}_{tag}.npz"
        np.savez_compressed(path, **snap)
        refl = snap.get("REFL_COMPOSITE")
        print(f"    [{tag}] t+{snap['elapsed_s']/60:.0f} min  "
              f"wmax {snap['WMAX'].max():+.1f} / wmin {snap['WMIN'].min():+.1f} m/s"
              f"  refl {np.nanmax(refl):.1f} dBZ (frac>20 dBZ "
              f"{float((refl > 20).mean()):.4f})"
              f"  T2 {snap['T2'].min():.1f}..{snap['T2'].max():.1f} K"
              f"  rain {float(snap['RAINNC'].max()):.1f} mm"
              f"  [{time.perf_counter()-t:.1f}s]")
        sys.stdout.flush()

    dump("f000")
    wall0 = time.perf_counter()
    for i in range(ndumps):
        t = time.perf_counter()
        # Free VRAM at the top of every sweep.  Two runs of this lane died in
        # RRTMGP's cloud-optics allocation on cards that were nominally
        # half-empty, and without this line there is no way to tell a tile
        # that is too big from a co-tenant that grew -- the traceback only
        # reports THIS process's pool, which was 6.7 GiB in one case and
        # 19 GiB in the other.
        free, total = _vram()
        print(f"    sweep {i + 1}/{ndumps}: VRAM {free:.2f}/{total:.2f} GiB "
              f"free before")
        sys.stdout.flush()
        report: dict = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            driver.run_tiled(store, cfg, args.tile, args.tile, halo=halo,
                             nsteps=steps_per_dump, nbuffers=args.nbuffers,
                             write_mode="ring", report=report, **kwargs)
        cp.cuda.runtime.deviceSynchronize()
        dt_wall = time.perf_counter() - t
        moved = report.get("gathered_bytes", 0) + report.get(
            "scattered_bytes", 0)
        print(f"    swept {steps_per_dump} steps in {dt_wall:.1f}s "
              f"({dt_wall/steps_per_dump*1e3:.0f} ms/step, "
              f"{dt_wall/steps_per_dump/(args.n**2*args.nz)*1e9:.2f} ns/cell)"
              f"  moved {moved/1e9:.1f} GB @ {moved/dt_wall/1e9:.1f} GB/s")
        sys.stdout.flush()
        dump(f"f{int(round((i + 1) * args.every)):03d}")
    print(f"    forecast complete: {ndumps * args.every:.0f} min of weather "
          f"in {(time.perf_counter()-wall0)/60:.1f} min of wall clock")
    digest = physinv.field_digests(store)
    print("RESULT " + json.dumps(
        {"n": args.n, "tile": args.tile, "dx": cfg.dx, "dt": cfg.dt,
         "minutes": args.minutes, "store_gib": nbytes / 2 ** 30,
         "carriers": len(digest)}))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage",
                    choices=["selftest", "ceiling", "bitexact", "forecast"])
    ap.add_argument("--n", type=int, default=1408)
    ap.add_argument("--nz", type=int, default=49)
    ap.add_argument("--tile", type=int, default=352)
    ap.add_argument("--slab", type=int, default=64,
                    help="rows per initialisation slab; must divide --n")
    ap.add_argument("--refl-slab", type=int, default=64,
                    help="rows per reflectivity slab; must divide --n")
    ap.add_argument("--nbuffers", type=int, default=2)
    ap.add_argument("--minutes", type=float, default=120.0)
    ap.add_argument("--every", type=float, default=20.0)
    ap.add_argument("--dx", type=float, default=None)
    ap.add_argument("--dt", type=float, default=None)
    ap.add_argument("--bubbles", type=int, default=60)
    ap.add_argument("--no-cumulus", action="store_true",
                    help="cu_physics=0.  At a convection-allowing dx this is "
                         "the meteorologically correct setting -- Kain-"
                         "Fritsch is a scheme for grids that CANNOT resolve "
                         "an updraught, and on a horizontally uniform "
                         "unstable sounding at 3 km it fires in nearly every "
                         "column and floods the domain (MEASURED: 95%% of a "
                         "192^2 test above 20 dBZ at t+30 min).  It is not "
                         "the certified rung, which is why it is a flag and "
                         "not the default.")
    ap.add_argument("--seed", type=int, default=20_260_808)
    ap.add_argument("--perturbation", choices=["forecast", "gate"],
                    default="forecast",
                    help="'gate' restores harness's seeding amplitudes -- "
                         "the control for bigdomain.NOISE_AMPLITUDES")
    ap.add_argument("--out", default="out")
    ap.add_argument("--steps", type=int, default=8)
    args = ap.parse_args()
    from tilestream import bigdomain
    if args.dx is None:
        args.dx = bigdomain.DX
    if args.dt is None:
        args.dt = bigdomain.DT
    {"selftest": stage_selftest, "ceiling": stage_ceiling,
     "bitexact": stage_bitexact,
     "forecast": stage_forecast}[args.stage](args)


if __name__ == "__main__":
    main()
