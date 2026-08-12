#!/usr/bin/env python3
"""Run one real HRRR-initialised outbreak case, monolithic or streamed.

    python -m tilestream.run_case_hrrr --case ks20190528 \
        --bridge  WORK/bridge --manifest-sha256 ... \
        --static  WORK/static_1200x900.npz \
        --outdir  WORK/run --mode streamed --tile 400 300

``--mode monolithic`` is the reference: one resident ``DomainState``, the same
frame boundary imposed after every step, and it is what a card large enough
for the domain would do.  ``--mode streamed`` runs the identical case through
``tilestream.driver.run_tiled`` from a pinned host store, which is the point:
the domain does not fit.

Both modes write the same history files, so a figure cannot tell you which
produced it and a digest can.
"""

from __future__ import annotations

import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import sys
import time
import warnings

import numpy as np


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _gib(x):
    return float(x) / 2 ** 30


def _tohost(a):
    import cupy as cp
    return cp.asnumpy(a) if isinstance(a, cp.ndarray) else np.asarray(a)


def _ashost(a):
    return np.ascontiguousarray(_tohost(a))


def patient(fn, *, log, what, attempts=45, wait=60.0):
    """Call ``fn()``, waiting out a neighbour that has taken the card.

    Every stage of a real-case ingest is a whole-domain allocation on the
    order of 15 GiB, and this estate runs several lanes per GPU: a stage can
    fail purely because somebody else's transient peak overlapped it.  That
    is a scheduling accident, and the right response is to release
    everything and ask again, not to lose the run.

    THE TRACEBACK IS DROPPED BEFORE THE COLLECT.  A caught
    ``OutOfMemoryError`` holds ``__traceback__``, the traceback holds the
    frames of the failed call, and those frames hold the half-built
    ``DomainState`` -- so collecting while the exception is still bound
    frees nothing and the pool grows on every retry.  MEASURED across three
    attempts on a 24 GiB card with no other tenant: 13.25 -> 13.68 ->
    14.31 GB allocated, 0.06 GiB free.
    """
    import gc
    import cupy as cp

    for attempt in range(1, int(attempts) + 1):
        failure = None
        try:
            return fn()
        except cp.cuda.memory.OutOfMemoryError as exc:
            failure = str(exc)
            exc.__traceback__ = None
        except cp.cuda.runtime.CUDARuntimeError as exc:
            failure = str(exc)
            exc.__traceback__ = None
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        free_gib = _gib(cp.cuda.runtime.memGetInfo()[0])
        log(f"   {what}: attempt {attempt}/{attempts} out of memory; "
            f"{free_gib:.2f} GiB free after release, waiting {wait:g}s "
            f"for the card  [{failure}]")
        time.sleep(wait)
    raise SystemExit(f"{what}: the card never came free in {attempts} "
                     "attempts")


def _write_frame_cache(cache, key, forced, frame_values, log):
    """Persist the frames after EVERY hour, not once at the end.

    The boundary ingest is the fragile phase on a shared card and it is
    twenty minutes of work; a crash in the forecast that threw it away would
    make every restart pay for it again.  Written to a temporary and
    renamed, so a crash during the write cannot leave a half cache that the
    next run trusts.
    """
    if cache is None:
        return
    done = len(frame_values[forced[0]])
    payload = {"key": np.array(key), "hours_done": np.array(done)}
    for n in forced:
        for k in range(done):
            for j, strip in enumerate(frame_values[n][k]):
                payload[f"{n}|{k}|{j}"] = np.ascontiguousarray(strip)
    tmp = Path(str(cache) + ".tmp.npz")
    Path(cache).parent.mkdir(parents=True, exist_ok=True)
    np.savez(tmp, **payload)
    os.replace(tmp, cache)
    log(f"   boundary frames cached to {cache} ({done} hour(s))")


def _peak_host_gib():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 ** 2


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", required=True)
    ap.add_argument("--dt", type=float, default=None,
                    help="model timestep in seconds (default: the case's "
                         "15 s).  SAFE TO CHANGE for the halo: "
                         "harness.halo_radius is 10 + 3*time_step_sound//2, "
                         "a stencil-REACH count that does not contain dt, so "
                         "16 stays correct.  20 s is what operational HRRR "
                         "uses at this 3 km spacing")
    ap.add_argument("--repair-surface", action="store_true", default=True,
                    help="hold NON-PROGNOSTIC land-surface carriers at their "
                         "analysis value in any cell that has gone "
                         "non-finite, at the sweep seam.  Contains a rare "
                         "Noah-MP column failure over high terrain; never "
                         "touches state/*")
    ap.add_argument("--no-repair-surface", dest="repair_surface",
                    action="store_false")
    ap.add_argument("--repair-cap", type=int, default=2000, metavar="N",
                    help="stop the run if one land-surface carrier needs "
                         "more than N cells repaired in a single sweep")
    ap.add_argument("--deep-soil", choices=("tmn", "soiltemp"), default="tmn",
                    help="which field is Noah-MP's deep-soil bottom "
                         "boundary: the elevation-corrected TMN WRF uses "
                         "(default), or the raw geogrid SOILTEMP the shipped "
                         "HRRR path passes.  'soiltemp' reproduces the "
                         "non-finite soil columns over the Rockies")
    ap.add_argument("--nan-guard", type=int, default=0, metavar="N",
                    help="check the first N tile states for non-finite "
                         "values BEFORE they are stepped, and report the "
                         "offending cells' position in the tile on any "
                         "failure.  A cell at the tile array's own edge is a "
                         "wrap-seam artefact; one in the middle of the "
                         "compute window is an instability")
    ap.add_argument("--store-check-every", type=int, default=0, metavar="N",
                    help="after every Nth sweep, scan the whole host store "
                         "for non-finite values and name the first carrier "
                         "and cell that has one")
    ap.add_argument("--max-hour", type=int, default=None,
                    help="last forecast hour of the cycle to use as boundary "
                         "material, and therefore the end of the forecast.  "
                         "MEASURED on this box: a whole-domain boundary hour "
                         "costs ~10 min of ingest at 1200x900x49 and the "
                         "forecast cannot start until every frame exists, so "
                         "this is the one knob that shortens the wait before "
                         "step 1")
    ap.add_argument("--bridge", type=Path, required=True)
    ap.add_argument("--manifest-sha256", default=None)
    ap.add_argument("--static", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--mode",
                    choices=("monolithic", "streamed", "vanilla-probe"),
                    default="streamed")
    ap.add_argument("--tile", type=int, nargs=2, default=(400, 300))
    ap.add_argument("--nbuffers", type=int, default=2)
    ap.add_argument("--nx", type=int, default=None)
    ap.add_argument("--ny", type=int, default=None)
    ap.add_argument("--hours", type=float, default=None,
                    help="forecast hours to integrate (default: the case's)")
    ap.add_argument("--history-minutes", type=float, default=30.0)
    ap.add_argument("--taper", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None,
                    help="override the step count, for smoke tests")
    ap.add_argument("--wrfout", action="store_true", default=True,
                    help="write real wrfout NetCDF history frames as well as "
                         "the npz ones, so `gpuwm render --engine rust` -- "
                         "the production renderer -- can read the run")
    ap.add_argument("--no-wrfout", dest="wrfout", action="store_false")
    ap.add_argument("--frame-cache", type=Path, default=None,
                    help="npz to read/write the boundary frames from; on a "
                         "shared card the boundary ingest is the fragile "
                         "phase and re-doing it after an unrelated crash "
                         "costs more than the forecast")
    args = ap.parse_args(argv)

    import cupy as cp

    from tilestream import case_hrrr as ch
    from tilestream import driver, gather, harness
    from tilestream import physics_inventory as physinv
    from tilestream import spec as tspec

    case = ch.CASES[args.case]
    if args.nx or args.ny or args.max_hour is not None:
        from dataclasses import replace
        hours = (case.hours if args.max_hour is None else
                 tuple(h for h in case.hours if h <= int(args.max_hour)))
        case = replace(case, nx=args.nx or case.nx, ny=args.ny or case.ny,
                       hours=hours)
    args.outdir.mkdir(parents=True, exist_ok=True)
    log_path = args.outdir / "run.log"
    log_file = log_path.open("a")

    def log(*a):
        msg = " ".join(str(x) for x in a)
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    dev = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    free0, total0 = cp.cuda.runtime.memGetInfo()
    log(f"== {case.name}  {case.headline}")
    log(f"   device {dev}  VRAM {_gib(free0):.2f} free / {_gib(total0):.2f} GiB")
    log(f"   domain {case.nx}x{case.ny}x{case.nz}  mode={args.mode}")

    target = ch.target_for(case)
    grid = target.grid()
    cfg = (ch.run_config(case) if args.dt is None
           else ch.run_config(case, dt=float(args.dt)))
    halo = harness.halo_radius(cfg)
    hard = halo + ch.HARD_MARGIN
    taper = ch.TAPER_CELLS if args.taper is None else int(args.taper)
    width = hard + taper
    log(f"   dx {cfg.dx:.3f} m  dt {cfg.dt:g} s  halo {halo}  "
        f"frame hard {hard} + taper {taper} = {width}")

    from gpuwm.core.grid import make_vertical_coord
    from gpuwm.static.build import GeogSelection

    coord = make_vertical_coord(cfg.nz, hybrid_opt=cfg.hybrid_opt,
                                etac=cfg.etac)
    # The land-use identity of the geogrid tree the STATICS were built from.
    # It is four constants, not data, so a box that holds the pre-built
    # static npz but not the 29 GB WPS_GEOG tree still runs; the geog root is
    # consulted when it is there so the two can never silently disagree.
    geog_env = os.environ.get("GPUWM_GEOG_ROOT")
    geog_root = Path(geog_env) if geog_env else None
    if geog_root is not None and geog_root.is_dir():
        landuse_attrs = GeogSelection.fallback(geog_root
                                               ).landuse_global_attrs()
    else:
        landuse_attrs = {"MMINLU": "MODIFIED_IGBP_MODIS_NOAH", "ISWATER": 17,
                         "ISLAKE": 21, "ISICE": 15, "ISURBAN": 13}
        log(f"   geog root {geog_root} absent; using the recorded land-use "
            f"identity {landuse_attrs}")

    t_static = time.perf_counter()
    static = {k: np.array(v) for k, v in
              np.load(args.static, allow_pickle=False).items()}

    # THE DEEP-SOIL BOUNDARY, and why it is substituted here.
    #
    # MEASURED, on this domain, from the run that died:
    # ``gpuwm.ingest.hrrr_physics.initialize_hrrr_physics`` hands
    # ``preprocess_land_surface_soil`` the RAW geogrid ``SOILTEMP`` as
    # Noah-MP's bottom boundary.  ``gpuwm/static/build.py`` already builds
    # the field WRF actually uses -- ``TMN = SOILTEMP - 0.0065 * HGT_M`` on
    # land, WRF v4.6.1 ``share/module_soil_pre.F:973`` -- and it is in the
    # same static bundle, unused by this path.  The difference is not small:
    #
    #   155 075 of 931 808 land columns have a bottom boundary >= 10 K too
    #   warm, 4 432 have it >= 20 K too warm, and the worst is 26.0 K at
    #   4 000 m.
    #
    # Over the Colorado/Utah Rockies that puts an 8 m boundary at ~292 K
    # under a soil column at ~270 K.  The first Noah-MP call on 16 of those
    # columns produced a NON-FINITE ``fields/tslb``, their surface fluxes
    # then made MYNN's mass-flux guard refuse ("MYNN mass-flux inputs must
    # be finite"), and the whole 52.9-million-cell forecast died at step 16.
    # Sixteen columns in the Rockies, 1 500 km from the storms this case is
    # about, took the run down.
    #
    # This substitution is not a workaround for a streaming problem -- the
    # streaming path never touches it, and a resident run at any size would
    # fail identically.  It is the ingest using the field it should already
    # have been using.
    if args.deep_soil == "tmn":
        if "TMN" not in static:
            raise SystemExit("--deep-soil tmn needs TMN in the static bundle")
        raw, corrected = static["SOILTEMP"], static["TMN"]
        land = static["LANDMASK"] > 0.5
        err = np.where(land, raw - corrected, 0.0)
        static["SOILTEMP"] = np.array(corrected)
        log(f"   deep-soil boundary: using the ELEVATION-CORRECTED TMN "
            f"(= SOILTEMP - 0.0065*HGT_M on land, WRF module_soil_pre.F:973) "
            f"instead of the raw geogrid SOILTEMP the HRRR path passes.  "
            f"{int((err >= 10).sum())} of {int(land.sum())} land columns "
            f"were >=10 K too warm, {int((err >= 20).sum())} were >=20 K, "
            f"worst {float(err.max()):.1f} K")
    else:
        log("   deep-soil boundary: RAW geogrid SOILTEMP, as the shipped "
            "HRRR path uses it.  Expect non-finite Noah-MP soil columns "
            "over terrain above ~2700 m")
    log(f"   statics loaded in {time.perf_counter() - t_static:.1f}s "
        f"({sum(v.nbytes for v in static.values()) / 1e9:.2f} GB host)")
    for key in ("HGT_M", "LANDMASK"):
        if static[key].shape != (case.ny, case.nx):
            raise SystemExit(f"static {key} is {static[key].shape}, "
                             f"expected {(case.ny, case.nx)}")

    t_snap = time.perf_counter()
    snaps = ch.snapshots_for(case, args.bridge, args.manifest_sha256)
    log(f"   HRRR series verified + mapped in "
        f"{time.perf_counter() - t_snap:.1f}s  hours={sorted(snaps)}"
        f"  window {snaps[case.hours[0]].nx}x{snaps[case.hours[0]].ny}")

    # ------------------------------------------------------------- time zero
    t_init = time.perf_counter()
    state, drv, result, met = patient(
        lambda: ch.build_case_state(
            case, snaps[case.hours[0]], grid, cfg, coord, static,
            landuse_attrs, progress=log),
        log=log, what="time-zero ingest")
    cp.cuda.runtime.deviceSynchronize()
    log(f"   time-zero state + physics in "
        f"{time.perf_counter() - t_init:.1f}s  "
        f"VRAM {_gib(cp.cuda.runtime.memGetInfo()[0]):.2f} GiB free")

    # SIMULATED REFLECTIVITY, and why it needs a wrapper.
    # ``dycore.step`` computes refl_10cm only when the caller says a history
    # step is due, and stashes it as a one-frame handoff on the driver that
    # must be consumed before the next one.  ``run_tiled`` calls ``step``
    # with no such flag and has nobody to consume the handoff, so a streamed
    # run carries no reflectivity at all -- the primary field a forecaster
    # reads.  The wrapper turns it on for EVERY step and clears the handoff
    # immediately; the array itself is state-owned ``refl_10cm`` scratch, so
    # it is already part of the streamed carrier set and rides the gather and
    # the scatter like any other field.  ``run_tiled`` binds ``step`` from
    # ``gpuwm.core.dycore`` inside its own body, so patching the module
    # attribute before the call is what reaches it.
    import gpuwm.core.dycore as _dycore
    _real_step = _dycore.step

    # WHERE A NON-FINITE VALUE CAME FROM, not merely that one exists.
    # The first real-data step of this design died inside MYNN's mass-flux
    # guard, which reports "inputs must be finite" and nothing else -- not
    # the tile, not the step, not whether the state ARRIVED bad or was made
    # bad.  Those three facts are the whole diagnosis, so they are collected
    # here: the tile state is checked BEFORE the step (did it arrive bad?)
    # and, on failure, the offending cells are located IN THE TILE's frame,
    # which is what distinguishes a wrap-seam artefact at the array edge
    # from an instability in the middle of the compute window.
    _watch = ("u", "v", "w", "thp", "php", "mup", "p", "al", "alt", "qv")
    _dbg = {"steps": 0, "checked": 0}

    def _nonfinite_report(st, label):
        import cupy as _cp
        lines = []
        for name in _watch:
            a = getattr(st, name, None)
            if a is None:
                continue
            bad = ~_cp.isfinite(a)
            n = int(bad.sum())
            if not n:
                continue
            idx = _cp.nonzero(bad)
            loc = tuple(int(x[0]) for x in idx)
            lines.append(f"{name}: {n} non-finite of {a.size}, first at "
                         f"{loc} of shape {tuple(int(s) for s in a.shape)}")
        return f"   {label}: " + ("; ".join(lines) if lines
                                  else "all watched arrays finite")

    def _step_with_refl(st, c, **kw):
        kw.setdefault("refl_10cm_due", True)
        if args.nan_guard and _dbg["checked"] < int(args.nan_guard):
            _dbg["checked"] += 1
            rep = _nonfinite_report(st, f"pre-step tile check "
                                        f"#{_dbg['checked']}")
            if "all watched" not in rep:
                log(rep)
        try:
            _real_step(st, c, **kw)
        except Exception as exc:                          # noqa: BLE001
            log(f"   STEP FAILED after {_dbg['steps']} completed tile steps: "
                f"{type(exc).__name__}: {exc}")
            log(_nonfinite_report(st, "tile state AT the failure"))
            raise
        _dbg["steps"] += 1
        d = getattr(st, "physics", None)
        if d is not None and getattr(d, "refl_10cm", None) is not None:
            d.refl_10cm = None

    _dycore.step = _step_with_refl

    if args.mode == "vanilla-probe":
        # THE CONTROL THE WHOLE PROJECT RESTS ON, measured rather than
        # asserted: can this card step this domain RESIDENTLY at all?  The
        # ingest's own intermediates are released first, so what is being
        # asked for is exactly the forecast's working set -- the state, the
        # physics driver and the dycore's per-step scratch -- and nothing
        # left over from setup.  A refusal here is the result, not a failure.
        del met, result
        cp.get_default_memory_pool().free_all_blocks()
        cp.cuda.runtime.deviceSynchronize()
        free_before = cp.cuda.runtime.memGetInfo()[0]
        log(f"   ingest intermediates released; state + driver resident, "
            f"VRAM {_gib(free_before):.2f} GiB free of {_gib(total0):.2f}")
        verdict = {"case": case.name, "nx": case.nx, "ny": case.ny,
                   "nz": case.nz, "device": dev,
                   "vram_total_gib": _gib(total0),
                   "vram_free_before_step_gib": _gib(free_before)}
        t0 = time.perf_counter()
        try:
            for k in range(int(args.steps or 3)):
                _step_with_refl(state, cfg)
            cp.cuda.runtime.deviceSynchronize()
        except Exception as exc:                      # noqa: BLE001
            verdict["verdict"] = "REFUSED"
            verdict["error"] = f"{type(exc).__name__}: {exc}"
            log(f"   VANILLA REFUSES this domain on this card: "
                f"{type(exc).__name__}: {exc}")
        else:
            n = int(args.steps or 3)
            verdict["verdict"] = "RAN"
            verdict["seconds_per_step"] = (time.perf_counter() - t0) / n
            verdict["peak_vram_gib"] = _gib(total0
                                            - cp.cuda.runtime.memGetInfo()[0])
            log(f"   vanilla RAN {n} resident steps at "
                f"{verdict['seconds_per_step']:.3f} s/step, peak VRAM "
                f"{verdict['peak_vram_gib']:.2f} GiB")
        (args.outdir / "vanilla_probe.json").write_text(
            json.dumps(verdict, indent=2) + "\n")
        return 0

    # The analysis-prescribable set, decided on the UN-stepped state so the
    # hour-zero boundary frame is the analysis itself and not the analysis
    # plus one step.
    inv0 = ch.carrier_inventory_with_refl(state)
    _dyn = ("u", "v", "w", "thp", "php", "mup")
    _moist = ("qv", "qc", "qr", "qi", "qs", "qg", "qh",
              "nc", "nr", "ni", "ns", "ng", "qni", "qns", "qnr", "qng")
    forced = tuple(n for n in sorted(inv0)
                   if n.split("/")[-1] in _dyn + _moist)
    log(f"   forced by the analysis: {forced}")
    frame_hour0 = ch.extract_frame({n: inv0[n] for n in forced}, width)

    # THE LAZY CARRIERS, and why they may NOT be found by stepping the domain.
    #
    # ``physics_inventory.carrier_manifest`` warns that a manifest taken at
    # t=0 is SHORTER than one taken after a step: the microphysics
    # precipitation accumulators, the effective-radii diagnostics and (with
    # Kain-Fritsch) ``cumulus/w0avg`` are allocated on first use.  A tile
    # buffer runs a one-step warmup for exactly that reason, so a store built
    # from an un-stepped domain state is missing fields the buffer holds and
    # the scatter fails.
    #
    # The obvious fix -- step the domain once -- is the one thing this lane
    # cannot do.  A monolithic step needs the dycore's own scratch on top of
    # the state, and at 1200x900x49 that is what does not fit: MEASURED, the
    # ingest alone leaves 2.49 GiB free on a 24 GiB card and the first
    # ``dycore.step`` dies in ``_prepare_atmosphere`` allocating 212 MB.
    # THAT REFUSAL IS THE PREMISE OF THE PROJECT, not a bug to work around.
    #
    # So the lazy set is discovered on a SMALL probe built by the same
    # factory ``run_tiled`` will use, stepped once, and every carrier the
    # probe has and the domain does not is allocated ZEROED at domain shape
    # straight into the pinned store.  Zero is the right value and not a
    # placeholder: these are accumulators and held tendencies, and a fresh
    # WRF state has them at zero.
    tile_factory = driver.geography_run_kwargs(
        cfg, None, geography={})["tile_state_factory"]
    from dataclasses import replace as _replace
    probe_cfg = _replace(cfg, nx=96, ny=80)
    t_probe = time.perf_counter()
    probe = tile_factory(probe_cfg)
    probe_inv = ch.carrier_inventory_with_refl(probe)
    inv = ch.carrier_inventory_with_refl(state)
    lazy = {}
    for key in sorted(probe_inv):
        if key in inv:
            continue
        a = probe_inv[key]
        lead = tuple(int(s) for s in a.shape[:-2])
        ny_ = case.ny + (int(a.shape[-2]) - probe_cfg.ny)
        nx_ = case.nx + (int(a.shape[-1]) - probe_cfg.nx)
        lazy[key] = (lead + (ny_, nx_), a.dtype)
    log(f"   probe tile state in {time.perf_counter() - t_probe:.1f}s; "
        f"{len(lazy)} lazily-allocated carriers to seed at domain shape: "
        + ", ".join(sorted(lazy)))
    extra = [k for k in inv if k not in probe_inv]
    if extra:
        raise SystemExit(
            "the domain state carries carriers a warmed tile buffer does "
            f"not, so the gather cannot fill them: {sorted(extra)}")
    del probe, probe_inv
    cp.get_default_memory_pool().free_all_blocks()

    cells = case.nx * case.ny * case.nz
    log(f"   {len(inv)} domain carriers, "
        f"{sum(v.nbytes for v in inv.values()) / cells:.1f} B/cell")

    geo = driver.geography_store(state, host=True)
    log(f"   geography: {len(geo)} arrays, "
        f"{sum(v.nbytes for v in geo.values()) / 1e9:.2f} GB pinned")

    # THE WRFOUT FRAME WRITER, captured HERE and not later.  The base state
    # (thb/pb/phb), MUB, HGT, ZNU/ZNW and P_TOP are INPUT: nothing in a
    # forecast writes them, they are not carriers, and the time-zero domain
    # state is the only place they exist at domain shape.  Capturing after
    # the state is released would mean a second full-domain ingest to get
    # them back.
    frame_writer = None
    if args.wrfout:
        from tilestream import case_wrfout
        frame_writer = case_wrfout.CaseFrameWriter.capture(
            state, cfg, grid, static, landuse_attrs=landuse_attrs,
            start_time=case.start_time, title="ArWen")
        log(f"   wrfout frame writer captured: "
            f"{frame_writer.bytes_held / 1e9:.2f} GB host held "
            f"(base state + grid metadata, input, never re-derived)")

    # ------------------------------------------------------- the pinned store
    # BUILT BEFORE THE BOUNDARY LOOP, and the order is the whole VRAM budget.
    # Each boundary hour is another full-domain ingest (MEASURED 15.8 GB of
    # device high-water at this size), so holding the time-zero state on the
    # device while ingesting the next hour asks a 24 GiB card for both.  Once
    # the store is in pinned host RAM the device is empty between hours.
    scalars = physinv.carrier_scalars(state)
    uh_ctx = dict(
        msfu=_ashost(geo["setup/msfu"]), msfv=_ashost(geo["setup/msfv"]),
        ht=_ashost(geo["setup/ht"]), phb=_ashost(geo["setup/phb"]),
        dn=_ashost(state.dn), dnw=_ashost(state.dnw),
        fnm=_ashost(state.fnm), fnp=_ashost(state.fnp),
        cf1=float(state.cf1), cf2=float(state.cf2), cf3=float(state.cf3),
        rdx=np.float32(1.0 / cfg.dx), rdy=np.float32(1.0 / cfg.dy))
    store = None
    if args.mode == "streamed":
        t_store = time.perf_counter()
        store = {k: gather.pinned_copy(_tohost(v)) for k, v in inv.items()}
        for key, (shape, dtype) in lazy.items():
            store[key] = gather.pinned_copy(np.zeros(shape, dtype=dtype))
        store = {k: store[k] for k in sorted(store)}
        log(f"   pinned host store: "
            f"{sum(v.nbytes for v in store.values()) / 2 ** 30:.2f} GiB over "
            f"{len(store)} carriers, in "
            f"{time.perf_counter() - t_store:.1f}s")
        # THE ANALYSIS ITSELF, checked once.  If the time-zero store is not
        # finite then nothing downstream can be, and every later diagnosis
        # would be chasing the wrong thing.
        bad0 = [k for k, v in store.items()
                if np.asarray(v).dtype.kind == "f"
                and not np.isfinite(np.asarray(v)).all()]
        log(f"   time-zero store finiteness: "
            + ("ALL FINITE" if not bad0 else f"NON-FINITE in {bad0[:8]}"))
        # EVERY device reference, named.  ``inv0`` is the one that bites:
        # it is a dict of the state's own device arrays, built pages earlier
        # to decide the forced set, and leaving it alive keeps the whole
        # 1200x900 state resident -- so the first boundary hour's ingest
        # asks the card for a second full domain and dies.  MEASURED: with
        # ``inv0`` held, 3.45 GiB free after this block; without it, the
        # card is empty.
        del state, drv, result, met, inv, inv0
        # gc FIRST.  ``DomainState`` is reachable through reference cycles
        # (the state holds its PhysicsDriver and the driver reaches back), so
        # ``del`` alone does not drop the last reference and
        # ``free_all_blocks`` -- which releases UNREFERENCED blocks only --
        # hands back nothing.  MEASURED: without this the card still reports
        # 3.45 of 23.52 GiB free after the store is built, and the first
        # boundary hour's ingest dies.  ``gpuwm.ingest.preprocess_backend
        # .release_backend_memory`` documents exactly this and is where the
        # ordering comes from.
        import gc
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        cp.cuda.runtime.deviceSynchronize()
        free_now = cp.cuda.runtime.memGetInfo()[0]
        log(f"   device released; VRAM {_gib(free_now):.2f} GiB free "
            f"of {_gib(total0):.2f}")
        if free_now < 0.60 * total0:
            log("   WARNING: the device did not come back empty -- a "
                "boundary-hour ingest needs a whole domain of headroom")

    # ------------------------------------------------------- boundary frames
    frame_values = {n: [] for n in forced}
    cache = args.frame_cache
    cache_key = f"{case.name}_{case.nx}x{case.ny}x{case.nz}_w{width}"
    if cache is not None and cache.is_file():
        with np.load(cache, allow_pickle=False) as z:
            if str(z["key"]) == cache_key:
                have = int(z["hours_done"])
                for n in forced:
                    frame_values[n] = [[z[f"{n}|{k}|{j}"] for j in range(4)]
                                       for k in range(have)]
                log(f"   {have} boundary frame(s) restored from {cache}")
            else:
                log(f"   boundary-frame cache {cache} is for "
                    f"{str(z['key'])!r}, not {cache_key!r}; rebuilding")
                frame_values = {n: [] for n in forced}
    t_frames = time.perf_counter()
    done = len(frame_values[forced[0]])
    for k, hour in list(enumerate(case.hours))[done:]:
        if k == 0:
            strips = frame_hour0
            for n in forced:
                frame_values[n].append(strips[n])
            log(f"   boundary frame f{hour:02d} taken from the analysis state")
            continue
        else:
            # RETRY, because the card is shared.  A boundary hour is a
            # whole-domain ingest (MEASURED ~15 GiB of device high-water at
            # 1200x900x49) and this estate runs several lanes per card, so a
            # neighbour's transient allocation can take the headroom away
            # mid-loop.  That is a scheduling accident, not a defect in the
            # ingest, and the right response is to release everything, wait
            # for the neighbour and ask again -- not to fail an hour of
            # boundary data and with it the whole run.
            r, _m = patient(
                lambda: ch.ingest_hour(
                    snaps[hour], grid, cfg, coord, static,
                    surface_fallback_radius=case.surface_fallback_radius),
                log=log, what=f"boundary hour f{hour:02d}")
            src = {n: v for n, v in
                   ch.carrier_inventory_with_refl(r.state).items()
                   if n in forced}
            missing = [n for n in forced if n not in src]
            if missing:
                raise SystemExit(f"boundary hour {hour} lacks {missing}")
        strips = ch.extract_frame(src, width)
        for n in forced:
            frame_values[n].append(strips[n])
        _write_frame_cache(cache, cache_key, forced, frame_values, log)
        if k:
            del r, _m, src, strips
            import gc as _gc
            _gc.collect()
            cp.get_default_memory_pool().free_all_blocks()
        log(f"   boundary frame f{hour:02d} done "
            f"({time.perf_counter() - t_frames:.0f}s cumulative)")
    frame_bytes = sum(a.nbytes for v in frame_values.values()
                      for h in v for a in h)
    log(f"   {len(case.hours)} boundary frames, "
        f"{frame_bytes / 1e9:.2f} GB host")


    # THE LAND-SURFACE SAFETY NET, and exactly what it is allowed to do.
    #
    # MEASURED: sixteen Colorado/Utah columns between 2717 m and 3612 m take
    # ``fields/tslb`` non-finite on the FIRST Noah-MP call.  Their soil
    # categories are ordinary (3 and 6), their land use is ordinary (1 and
    # 10), and the elevation-corrected deep-soil boundary does not change it
    # -- both were checked.  On the NEXT sweep those NaN soil temperatures
    # come back in as input, the surface fluxes go non-finite with them, and
    # MYNN's mass-flux guard refuses -- taking a 52 920 000-cell forecast
    # down over sixteen mountain columns 1 500 km from the storms the case
    # is about.
    #
    # So the LAND SURFACE ONLY is held at its analysis value in whatever
    # cells have gone non-finite, at the sweep seam, before the next sweep
    # reads them.  The rules this obeys, because a repair nobody can audit
    # is worse than a crash:
    #
    #   * it NEVER touches a prognostic (``state/*``).  If one of those goes
    #     non-finite the forecast is over and the run stops.
    #   * it is capped.  Above ``--repair-cap`` cells in one carrier this is
    #     no longer a rare-column defect and the run stops rather than
    #     quietly papering over a spreading instability.
    #   * every sweep's repaired-cell count is logged and carried into
    #     run.json, so a reader can see whether it stayed at sixteen.
    #
    # It is containment of a land-surface defect, not a fix, and nothing
    # about the streaming path causes or cures it: a resident run at a size
    # that fitted would fail in the same sixteen columns.
    surface_snapshot = {}
    if args.repair_surface and store is not None:
        for key, value in store.items():
            if key.startswith("state/"):
                continue
            arr = np.asarray(value)
            if arr.dtype.kind != "f" or arr.ndim > 3:
                continue
            if arr.ndim == 3 and arr.shape[0] > 8:
                continue                      # a 3-D atmospheric carrier
            surface_snapshot[key] = np.array(arr)
        log(f"   land-surface safety net armed: {len(surface_snapshot)} "
            f"non-prognostic carriers snapshotted at analysis time "
            f"({sum(a.nbytes for a in surface_snapshot.values()) / 1e9:.2f} "
            f"GB host)")

    reference = store if store is not None else inv
    frozen = ch.extract_frame({n: v for n, v in reference.items()
                               if n not in forced}, hard)
    forcing = ch.FrameForcing(
        width=width, hard=hard, taper=taper, hours=tuple(case.hours),
        names=forced, values=frame_values, frozen=frozen)
    log(f"   frozen carriers in the hard zone: {len(frozen)}")

    # ------------------------------------------------------------- the run
    hours = (len(case.hours) - 1) if args.hours is None else float(args.hours)
    nsteps = int(round(hours * 3600.0 / cfg.dt)) if args.steps is None \
        else int(args.steps)
    history_every = max(1, int(round(args.history_minutes * 60.0 / cfg.dt)))
    log(f"   integrating {nsteps} steps of {cfg.dt:g}s = "
        f"{nsteps * cfg.dt / 3600.0:.2f} h, history every "
        f"{history_every} steps")

    hist_dir = args.outdir / "history"
    hist_dir.mkdir(exist_ok=True)
    wrf_dir = args.outdir / "wrfout"
    written = []
    wrfout_written: list[str] = []

    # UPDRAFT HELICITY, computed on the host at history times only.
    # ``dycore``'s own UP_HELI_MAX lane refuses this configuration by design
    # (uh_diag._supported_boundary_geometry: WRF's cal_helicity has separate
    # non-periodic bound adjustments and the periodic branch is deliberately
    # not transcribed), and a tile IS periodic in its own array -- so the
    # running max cannot be carried through the tiles.  The same module's
    # host mirror of cal_helicity can be evaluated on the WHOLE domain at
    # the sweep seam, where the store is a complete non-periodic domain and
    # the refusal does not apply.  What that yields is INSTANTANEOUS 2-5 km
    # updraft helicity at each history time, not WRF's between-output
    # running maximum; a figure made from it must say so.
    from gpuwm.core.uh_diag import cal_helicity_uh_columns_np

    def updraft_helicity(arrays):
        try:
            u = _tohost(arrays["state/u"]).astype(np.float32)
            v = _tohost(arrays["state/v"]).astype(np.float32)
            w = _tohost(arrays["state/w"]).astype(np.float32)
            ph = _tohost(arrays["state/php"]).astype(np.float32)
        except KeyError:
            return None
        uh, _use = cal_helicity_uh_columns_np(
            u, v, w, ph, uh_ctx["phb"], uh_ctx["msfu"], uh_ctx["msfv"],
            uh_ctx["ht"], uh_ctx["dn"], uh_ctx["dnw"], uh_ctx["fnm"],
            uh_ctx["fnp"], uh_ctx["cf1"], uh_ctx["cf2"], uh_ctx["cf3"],
            uh_ctx["rdx"], uh_ctx["rdy"])
        return np.asarray(uh, dtype=np.float32)

    def write_history(step_index, arrays, elapsed):
        valid = case.start_time + timedelta(seconds=float(elapsed))
        out = hist_dir / f"h_{step_index:06d}.npz"
        payload = {}
        if ch.REFL_KEY in arrays:
            payload["refl_10cm"] = _tohost(arrays[ch.REFL_KEY]).max(axis=0)
        for key, dest in (("fields/u10", "u10"), ("fields/v10", "v10"),
                          ("fields/t2", "t2"), ("fields/q2", "q2"),
                          ("fields/psfc", "psfc"),
                          ("fields/pblh", "pblh"),
                          ("scratch/mp_rainnc", "rainnc"),
                          ("scratch/mp_snownc", "snownc"),
                          ("scratch/mp_graupelnc", "graupelnc")):
            if key in arrays:
                payload[dest] = _tohost(arrays[key])
        w = arrays.get("state/w")
        if w is not None:
            wh = _tohost(w)
            payload["wmax"] = wh.max(axis=0)
            payload["wmin"] = wh.min(axis=0)
        t_uh = time.perf_counter()
        uh = updraft_helicity(arrays)
        if uh is not None:
            payload["uh"] = uh
            payload["uh_seconds"] = np.float32(time.perf_counter() - t_uh)
        del t_uh
        np.savez_compressed(out, valid=str(valid),
                            elapsed=float(elapsed), **payload)
        written.append(out.name)
        log(f"      history {out.name} valid {valid:%Y-%m-%d %H:%M}Z "
            f"({len(payload)} fields)")
        if frame_writer is None:
            return
        # THE PRODUCT FILE.  Written at the sweep seam, from the same store
        # the npz frame was taken from, so the two describe one instant.
        from tilestream import case_wrfout as _cw
        from gpuwm.io.wrfout import wrfout_filename

        # ONE frame kind, at every history time.  A surface-only file is not
        # importable by the rust engine at all (MEASURED: "Open WRF ...
        # failed and the file is not a supported post-processed WRF
        # archive"), so there is no cheap animation tier to have -- see
        # case_wrfout's docstring for the field set and what each row buys.
        t_wrf = time.perf_counter()
        wrf_dir.mkdir(parents=True, exist_ok=True)
        path = wrf_dir / wrfout_filename(valid, 1)
        extra = {} if uh is None else {"UP_HELI_MAX": uh}
        frame_writer.write(path, arrays, valid, kind=_cw.KIND_FULL,
                           extra=extra)
        wrfout_written.append(str(path.relative_to(args.outdir)))
        log(f"      wrfout {path.name}  "
            f"{path.stat().st_size / 1e9:.2f} GB in "
            f"{time.perf_counter() - t_wrf:.1f}s"
            + ("" if uh is not None else "  (no UP_HELI_MAX: the store did "
                                         "not carry u/v/w/php)"))


    t_run = time.perf_counter()
    peak_vram = 0
    # HOW OFTEN EACH SCHEME ACTUALLY FIRED, printed rather than assumed.
    # Radiation runs on a 12-minute cadence and cumulus is OFF by
    # configuration at dx = 3 km, so "radiation 0, cumulus 0" over a window
    # would mean the window never crossed a cadence boundary and every
    # timing in it is a fiction.  The counters come off the tile state at
    # the sweep seam, which is where run_tiled restores and re-reads them.
    fired: dict = {}
    repair_log: dict = {}

    if args.mode == "monolithic":
        from gpuwm.core.dycore import step as dstep
        forcing.apply(_HostView(inv), 0.0, 0)
        for istep in range(nsteps):
            _step_with_refl(state, cfg)
            arrays = ch.carrier_inventory_with_refl(state)
            forcing.apply(_HostView(arrays), float(state.elapsed_seconds),
                          istep)
            if (istep + 1) % history_every == 0 or istep == nsteps - 1:
                write_history(istep + 1, arrays,
                              float(state.elapsed_seconds))
            peak_vram = max(peak_vram, total0 - cp.cuda.runtime.memGetInfo()[0])
    else:
        start = store
        specs = tspec.plan_tiles(case.nx, case.ny, args.tile[0], args.tile[1],
                                 halo, False)
        tspec.validate_plan(specs, case.ny, case.nx)
        red = sum(s.cnx * s.cny for s in specs) / (case.nx * case.ny)
        log(f"   {len(specs)} tiles, compute {specs[0].cny}x{specs[0].cnx}, "
            f"redundancy {red:.4f}x, nbuffers {args.nbuffers}")
        forcing.apply(start, 0.0, 0)
        # F+00: the analysis itself, through the same writer as every later
        # frame.  REFL_10CM is exactly zero here and that is not a defect --
        # no microphysics call has happened yet, and WRF's own t=0 history
        # frame carries the same zeros.
        write_history(0, start, 0.0)
        report: dict = {}
        state_holder = {"peak": 0}

        def store_nonfinite(store_, when):
            """Name the first carrier and cell of the DOMAIN that is bad.

            Reported in domain coordinates, and against the frame geometry,
            because "column 7 of 1200" and "column 600 of 1200" are two
            different defects: the first is inside the prescribed boundary
            zone and the second is the forecast.
            """
            found = []
            columns = None
            for name in sorted(store_):
                a = np.asarray(store_[name])
                if a.dtype.kind != "f":
                    continue
                bad = ~np.isfinite(a)
                n = int(bad.sum())
                if not n:
                    continue
                idx = np.nonzero(bad)
                loc = tuple(int(x[0]) for x in idx)
                j, i = loc[-2], loc[-1]
                depth = min(i, a.shape[-1] - 1 - i, j, a.shape[-2] - 1 - j)
                zone = ("HARD-zone" if depth < hard else
                        "taper" if depth < width else "FORECAST")
                found.append(f"{name}[{n}]@{loc}({zone},d={depth})")
                if columns is None:
                    columns = set(zip(idx[-2].tolist(), idx[-1].tolist()))
                else:
                    columns &= set(zip(idx[-2].tolist(), idx[-1].tolist()))
            if not found:
                return []
            log(f"   NON-FINITE {when}: {len(found)} carrier(s)")
            for chunk in found:
                log(f"      {chunk}")
            if columns:
                log(f"   the {len(columns)} (j,i) column(s) COMMON to every "
                    f"bad carrier: {sorted(columns)[:12]}")
            return found

        def on_sweep(istep, dst, clock):
            elapsed = float(clock.get("elapsed_seconds", (istep + 1) * cfg.dt))
            if surface_snapshot:
                repaired = 0
                worst = 0
                for key, clean in surface_snapshot.items():
                    arr = dst.get(key)
                    if arr is None:
                        continue
                    bad = ~np.isfinite(arr)
                    n = int(bad.sum())
                    if not n:
                        continue
                    repaired += n
                    worst = max(worst, n)
                    np.copyto(arr, clean, where=bad)
                    repair_log[key] = max(repair_log.get(key, 0), n)
                if repaired:
                    repair_log["_sweeps"] = repair_log.get("_sweeps", 0) + 1
                    repair_log["_worst_sweep"] = max(
                        repair_log.get("_worst_sweep", 0), repaired)
                    if (istep + 1) % 40 == 0 or istep < 3:
                        log(f"      land-surface net: {repaired} non-finite "
                            f"cell(s) held at analysis this sweep "
                            f"(worst carrier {worst})")
                    if worst > int(args.repair_cap):
                        raise SystemExit(
                            f"{worst} non-finite cells in one land-surface "
                            f"carrier at sweep {istep + 1} exceeds "
                            f"--repair-cap {args.repair_cap}; this is no "
                            "longer a rare-column defect and the run stops")
                # A PROGNOSTIC going non-finite is the end of the forecast and
                # is never repaired.
                for key, arr in dst.items():
                    if not key.startswith("state/"):
                        continue
                    a = np.asarray(arr)
                    if a.dtype.kind == "f" and not np.isfinite(a).all():
                        store_nonfinite({key: a}, f"sweep {istep + 1}")
                        raise SystemExit(
                            f"the PROGNOSTIC {key} went non-finite at sweep "
                            f"{istep + 1}; stopping rather than integrating "
                            "garbage")
            if args.store_check_every and (
                    (istep + 1) % int(args.store_check_every) == 0):
                bad = store_nonfinite(dst, f"after sweep {istep + 1}")
                fatal = [b for b in bad if b.startswith("state/")]
                if fatal:
                    raise SystemExit(
                        f"a PROGNOSTIC went non-finite at sweep {istep + 1}: "
                        f"{fatal}; stopping rather than integrating garbage")
            forcing.apply(dst, elapsed, istep)
            fired.update({k: v for k, v in clock.items()
                          if k != "elapsed_seconds"})
            used = total0 - cp.cuda.runtime.memGetInfo()[0]
            state_holder["peak"] = max(state_holder["peak"], used)
            if (istep + 1) % history_every == 0 or istep == nsteps - 1:
                write_history(istep + 1, dst, elapsed)
            if (istep + 1) % 40 == 0:
                el = time.perf_counter() - t_run
                log(f"      step {istep + 1}/{nsteps}  "
                    f"{el / (istep + 1):.3f}s/step  "
                    f"eta {(nsteps - istep - 1) * el / (istep + 1) / 60:.1f} min"
                    f"  VRAM {used / 2 ** 30:.2f} GiB"
                    f"  fired {fired.get('call_counts')}")

        geo_kwargs = driver.geography_run_kwargs(cfg, None, geography=geo)
        geo_kwargs["scalars"] = scalars
        geo_kwargs["inventory_fn"] = ch.carrier_inventory_with_refl
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            driver.run_tiled(start, cfg, args.tile[0], args.tile[1],
                             halo=halo, nsteps=nsteps,
                             nbuffers=args.nbuffers, periodic=False,
                             report=report, on_sweep=on_sweep,
                             check_geography=False, **geo_kwargs)
        peak_vram = state_holder["peak"]
        report.pop("specs", None)
        report.pop("tile_cfg", None)
        log("   report: " + json.dumps(
            {k: (v if isinstance(v, (int, float, str)) else str(v))
             for k, v in report.items()}, indent=None)[:2000])

    wall = time.perf_counter() - t_run
    log(f"== FINISHED  wall {wall / 60:.2f} min for {nsteps} steps "
        f"({wall / max(nsteps, 1):.3f} s/step)")
    log(f"   peak VRAM {peak_vram / 2 ** 30:.2f} GiB   "
        f"peak host RSS {_peak_host_gib():.2f} GiB")
    log(f"   scheme calls over the whole run: {fired.get('call_counts')}")
    log(f"   cumulus is 0 BY CONFIGURATION (cu_physics=0): a 3 km "
        f"convection-allowing forecast switches the cumulus scheme OFF, "
        f"because it would remove the instability the explicit updraughts "
        f"exist to release.  radiation fires every "
        f"{ch.PHYSICS['radt_minutes']:g} min.")
    log(f"   wrfout frames written: {len(wrfout_written)}")
    meta = dict(case=case.name, headline=case.headline,
                cycle=str(case.cycle), start=str(case.start_time),
                nx=case.nx, ny=case.ny, nz=case.nz, dx=float(cfg.dx),
                dt=float(cfg.dt), mode=args.mode, device=dev,
                tile=list(args.tile), halo=int(halo), hard=int(hard),
                taper=int(taper), nsteps=int(nsteps),
                wall_seconds=float(wall),
                peak_vram_gib=float(peak_vram / 2 ** 30),
                peak_host_rss_gib=float(_peak_host_gib()),
                history=written, wrfout=wrfout_written,
                scheme_calls=fired.get("call_counts"),
                surface_repairs=dict(repair_log),
                deep_soil=args.deep_soil,
                physics=dict(ch.PHYSICS),
                ref_lat=case.ref_lat, ref_lon=case.ref_lon,
                truelat1=38.5, truelat2=38.5, stand_lon=-97.5)
    (args.outdir / "run.json").write_text(json.dumps(meta, indent=2) + "\n")
    return 0


class _HostView(dict):
    """A ``store``-shaped view of device arrays, for the monolithic mode.

    ``FrameForcing.apply`` writes with numpy slice arithmetic, which CuPy
    arrays accept verbatim, so the same code drives both modes.  The host
    frame strips are uploaded per slice by CuPy's own assignment.
    """


if __name__ == "__main__":
    raise SystemExit(main())
