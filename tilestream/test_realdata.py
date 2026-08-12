"""THE REAL-DATA GATE: a slab-ingested store must equal a monolithic one, exactly.

Run it::

    python -m tilestream.test_realdata        # from the repository root

Every other gate in this prototype seeds its state from
``numpy.random.default_rng``.  This one reads a GRIB.  It initialises
``configs/real74_d01``'s own 250x200x49 CONUS Lambert domain from the April 3
1974 ERA5 bundle -- real terrain, real map factors, real Coriolis, real
analysed winds, temperature and moisture -- twice: once the way ``gpuwm``
does it today (one full-domain ``DomainState`` on the GPU) and once the way a
domain larger than VRAM must (one row slab at a time, straight into pinned
host RAM), and requires the two to be byte-identical.

WHY BYTE-IDENTICAL IS THE ONLY USEFUL BAR
-----------------------------------------
Every failure mode this lane has is SMOOTH.  A slab written one row off, a
slab that rebuilt its own projection, a v face that took the domain-edge
branch at a slab seam -- each of them produces a finite, plausible,
correctly-shaped field that differs from the truth by a fraction of a
percent.  ``allclose`` passes all three.  Only an exact comparison
distinguishes "the ingest is correct" from "the ingest is nearly correct",
and nearly correct is how this project has been wrong six times.

WHAT IS COMPARED, SECTION BY SECTION
------------------------------------
``--grid-only``
    :func:`tilestream.realdata.window_grid` against the parent grid's own
    slice, for ten derived fields in both axes.  Its negative control is
    :func:`tilestream.realdata.rebuilt_grid` -- a projection rebuilt from the
    window's extents, which is what a slab gets if nobody thinks about it --
    and the control reports how far the rebuilt window lands from the truth,
    in degrees and in kilometres.

``--ingest-only``
    The slab ingest against the monolithic ingest at the FULL real74 domain,
    for slab heights from the whole domain down to a single row, over both
    the 13 carriers and the 13 geography arrays.  Four negative controls, each
    naming a different way this could have been faked:

    * ``y_halo=0`` -- the slab does not carry the mass row below its first.
      ``real.py:1939``'s v-face pressure average then takes its own edge
      branch at every interior seam.  MEASURED: ``state/v`` alone, in exactly
      ``nslabs - 1`` rows, by up to 2.1 m/s.  It is the smallest wrong answer
      in this file and the one most likely to ship.
    * ``row_shift=1`` -- every slab written one row too high.
    * ``force_kind='mass'`` -- the staggering rule ignored, so the closing
      v face at row ``ny`` is never written by anybody.
    * ``grid_fn=rebuilt_grid`` -- every slab derives its own projection.

``--steps-only``
    A store filled from real data, streamed through
    :func:`tilestream.driver.run_tiled`, against a monolithic run from the
    same real data.  Bit-exact over the whole carrier set, at six geometries
    (both domains, one and two buffers, shadow and ring write modes, 8-row and
    1-row ingest slabs), with three negative controls.

    THE PERIODIC CAVEAT, stated here rather than in a footnote.  The tiled
    lane is bit-exact under PERIODIC lateral boundaries; ``real74_d01`` runs
    ``specified=True`` with ERA5 boundary frames, and specified boundaries in
    the tiled lane are a different workstream.  So this section closes the
    domain periodically (:func:`tilestream.realdata.periodic_closure`, applied
    identically to both sides) and pays a real price for it, which
    :data:`STEP_DT_NATIVE` documents with the measurement: wrapping a real
    analysis onto itself is a grid-scale shock, the first ``|w|`` maximum
    lands on the seam column, and the growth is ~2x per step AT EVERY dt from
    30 s down to 0.5 s -- so it is the wrapped configuration that is
    ill-posed, not the timestep.  It is not the terrain either: the flat-ocean
    window fails at step 2 just as fast.

    Two consequences, both visible in the output.  The row at the case's own
    dt=30 s gets N=1, which is every step it has.  The multi-step rows run at
    dt=0.1 s, the largest step in the sweep that stays finite for eight, where
    ``|w|`` grows 0.01 -> 0.18 m/s.  Every row reports ``nan_free``, because a
    bit-exact comparison of two NaN fields is not evidence of anything.

    THE DYNAMICS-ONLY REAL74 PATH IS ITSELF UNSTABLE IN THIS TREE, and that is
    a finding, not an excuse.  With the case's OWN ``specified=True`` boundary
    frames built from all three ERA5 times -- no periodic anything --
    ``verify.cases.real74_d01.config()`` at dt=30 s reaches ``|w|`` = 1.3 m/s
    after one step and 1.06e10 m/s after two, and the blowup is centred at
    ``j = 4``, inside the relax zone.  That is consistent with the defect
    ``tilestream-bcfix`` describes (the small-step uv kernels wrapped the
    boundary face column mass unconditionally, so on a specified domain the
    west boundary drew its mass from the east-most column); this file does not
    depend on it either way, because both of its sides run the same dycore.
"""

from __future__ import annotations

import dataclasses
import hashlib
import sys
import time
import traceback
import warnings

import numpy as np


# --------------------------------------------------------------------------
# geometry of the two comparison domains
# --------------------------------------------------------------------------

#: The real74 d01 domain, as the frozen case declares it.
FULL_NX, FULL_NY = 250, 200

#: The western-Atlantic window the multi-tile rows run on, and why it is that
#: window.  SEARCHED, not chosen: over every 64x64 window of the d01 grid at
#: stride 4, this one minimises the terrain discontinuity a periodic wrap
#: creates -- ``max|ht[:, -1] - ht[:, 0]|`` and its y counterpart come to
#: 0.29 m, against 247.88 m for the best 96x96 window and 1511 m of relief on
#: the full domain.  Centre 31.52N 72.23W, open ocean, ``|ht|max`` 0.3 m.
#:
#: Flat water does NOT make the periodic wrap stable -- it fails at step 2 at
#: dt=30 s exactly like the full domain, which is how we know the terrain is
#: not what drives it.  What it buys is a factor of two in the largest dt that
#: survives eight steps (0.1 s here, 0.05 s on the full domain) and a much
#: tamer trajectory while it does (``|w|`` 0.18 m/s at step 8, against
#: 208 m/s).  It is also 64x64, so 16x16 tiles with halo 16 give SIXTEEN
#: tiles and a compute window of 48x48 -- the most seams of anything here.
ATLANTIC_I0, ATLANTIC_J0 = 184, 0
ATLANTIC_NX, ATLANTIC_NY = 64, 64

#: Slab heights the ingest section runs.  ``FULL_NY`` is one slab -- the
#: monolithic shape, reached through the slab code path, which is the control
#: that says the slab machinery is not itself changing the answer.  ``1`` is
#: 200 slabs, the hardest case the loop has: every interior seam exercises the
#: y halo and every field is written 200 times.
SLAB_HEIGHTS = (FULL_NY, 64, 37, 8, 1)
QUICK_SLAB_HEIGHTS = (FULL_NY, 37, 1)


# --------------------------------------------------------------------------
# comparison machinery
# --------------------------------------------------------------------------

def _as_numpy(array) -> np.ndarray:
    import cupy as cp
    return (cp.asnumpy(array) if isinstance(array, cp.ndarray)
            else np.asarray(array))


def digest(name: str, array) -> str:
    """SHA-256 over name, dtype, shape and raw bytes -- the harness's own.

    Bytes, not values, so a NaN compares equal to the identical NaN and a
    shape change cannot masquerade as a value change.  Both matter here: a
    poisoned store row that nobody wrote is a NaN on both sides only if the
    same code left it, and a mis-staggered write changes a shape.
    """
    from tilestream import harness

    return hashlib.sha256(harness._digest_payload(name, array)).hexdigest()


def field_digests(arrays) -> dict[str, str]:
    return {name: digest(name, array) for name, array in arrays.items()}


def compare_arrays(got, want) -> dict:
    """Full record: which fields differ, by how much, and over how many rows.

    ``rows`` is the count of horizontal ROWS that contain any difference,
    which is the diagnostic that identifies a slab-seam bug: a y-halo failure
    touches exactly one row per interior seam, and nothing else in this file
    produces that signature.
    """
    gd, wd = field_digests(got), field_digests(want)
    differing = sorted(k for k in wd if wd[k] != gd.get(k))
    record: dict = {"bitexact": not differing, "fields": len(wd),
                    "differing": differing, "max_abs": 0.0, "max_rel": 0.0,
                    "worst": None, "rows": {}, "nonfinite": {}}
    for name in differing:
        a = _as_numpy(got[name]).astype(np.float64)
        b = _as_numpy(want[name]).astype(np.float64)
        if a.shape != b.shape:
            record["worst"] = record["worst"] or name
            record["rows"][name] = -1
            continue
        both = np.isfinite(a) & np.isfinite(b)
        diff = np.zeros_like(a)
        diff[both] = np.abs(a[both] - b[both])
        nonfinite = int((np.isfinite(a) != np.isfinite(b)).sum())
        if nonfinite:
            record["nonfinite"][name] = nonfinite
        mask = (diff > 0.0) | (np.isfinite(a) != np.isfinite(b))
        while mask.ndim > 2:
            mask = mask.any(axis=0)
        record["rows"][name] = int(mask.any(axis=-1).sum())
        this = float(diff.max()) if diff.size else 0.0
        if this > record["max_abs"]:
            record["max_abs"], record["worst"] = this, name
        scale = np.maximum(np.abs(a), np.abs(b))
        live = both & (scale > 0.0)
        if live.any():
            record["max_rel"] = max(record["max_rel"],
                                    float((diff[live] / scale[live]).max()))
    return record


_PASS, _FAIL = "  PASS", "  FAIL"
_RESULTS: list[tuple[bool, str]] = []


def report(ok: bool, headline: str, *detail: str) -> bool:
    print((_PASS if ok else _FAIL) + "  " + headline)
    for line in detail:
        print("        " + line)
    _RESULTS.append((ok, headline))
    return ok


def banner(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


# --------------------------------------------------------------------------
# shared setup
# --------------------------------------------------------------------------

_CACHE: dict = {}


def case_config(nx: int = FULL_NX, ny: int = FULL_NY, *, periodic: bool = False):
    """``real74_d01``'s frozen dynamics config at ``nx`` x ``ny``.

    ``periodic=True`` clears the four lateral-boundary flags exactly as
    ``harness.make_config`` does, and leaves ``spec_bdy_width``/``spec_zone``/
    ``relax_zone`` alone -- ``validate_run_config`` requires them to be legal
    whether or not ``specified`` reads them, so zeroing them is a refusal and
    not a simplification.
    """
    from gpuwm.config import validate_run_config
    from gpuwm.verify.cases.real74_d01 import config

    cfg = dataclasses.replace(config(1800.0), nx=int(nx), ny=int(ny))
    if periodic:
        cfg = dataclasses.replace(cfg, open_x=False, open_y=False,
                                  specified=False, nested=False)
    return validate_run_config(cfg)


def case_coord(cfg):
    """The case's own explicit 50-level eta table, rebuilt per call.

    ``finalize_vertical_coord`` MUTATES a ``VerticalCoord`` in place (it fills
    ``c1h..c4f`` from ``p_top``), and ``initialize_real`` calls it, so a coord
    handed to two ingests is not the same object the second time.  Rebuilding
    is cheap and removes the question.
    """
    from gpuwm.core.grid import make_vertical_coord
    from gpuwm.verify.cases.real74_d01 import ETA_LEVELS

    return make_vertical_coord(cfg.nz, hybrid_opt=cfg.hybrid_opt,
                               etac=cfg.etac, eta_levels=ETA_LEVELS)


def default_coord(cfg):
    """The DEFAULT vertical stretch -- what a tile buffer gets by accident.

    ``driver.make_tile_state`` and ``harness.geography_builder`` both call
    ``make_vertical_coord(nz, hybrid_opt, etac)`` with no ``eta_levels``.  For
    every domain this project has streamed so far that was right, because
    those domains had no eta table; ``real74_d01`` has one, and a buffer built
    without it rebuilds ``znu``/``znw``/``dnw``/``c1h..c4f`` from a different
    coordinate.  Nothing checks it, so this is a negative control rather than
    an error.
    """
    from gpuwm.core.grid import make_vertical_coord

    return lambda: make_vertical_coord(cfg.nz, hybrid_opt=cfg.hybrid_opt,
                                       etac=cfg.etac)


def bundle():
    from tilestream import realdata

    if "bundle" not in _CACHE:
        _CACHE["bundle"] = realdata.find_real74_bundle()
    return _CACHE["bundle"]


def full_geography():
    from tilestream import realdata

    if "geo" not in _CACHE:
        _CACHE["geo"] = realdata.real74_geography(bundle())
    return _CACHE["geo"]


def window_geography(i0: int, j0: int, nx: int, ny: int) -> dict:
    """``real74_geography`` restricted to a sub-rectangle, exactly.

    The grid goes through :func:`tilestream.realdata.window_grid` (so the
    window's projection is still the parent's arithmetic) and every geo_em
    field is sliced with its own staggering.  This is what lets the eight-step
    row run on 64x64 of real data without a second bundle: the window is a
    domain in its own right and the slab ingest windows it AGAIN, which
    ``ProjectedGrid.translated`` composes onto the original reference rather
    than accumulating.
    """
    from tilestream import realdata

    g = full_geography()
    mf = g["map_fields"]
    return dict(
        grid=realdata.window_grid(g["grid"], i0, j0, nx, ny),
        terrain=np.ascontiguousarray(g["terrain"][j0:j0 + ny, i0:i0 + nx]),
        source_orography=np.ascontiguousarray(
            g["source_orography"][j0:j0 + ny, i0:i0 + nx]),
        map_fields={
            "MAPFAC_M": mf["MAPFAC_M"][j0:j0 + ny, i0:i0 + nx],
            "MAPFAC_U": mf["MAPFAC_U"][j0:j0 + ny, i0:i0 + nx + 1],
            "MAPFAC_V": mf["MAPFAC_V"][j0:j0 + ny + 1, i0:i0 + nx],
            "F": mf["F"][j0:j0 + ny, i0:i0 + nx],
            "E": mf["E"][j0:j0 + ny, i0:i0 + nx],
            "SINALPHA": mf["SINALPHA"][j0:j0 + ny, i0:i0 + nx],
            "COSALPHA": mf["COSALPHA"][j0:j0 + ny, i0:i0 + nx],
        })


def monolithic_reference(cfg, geo) -> tuple[dict, dict, int]:
    """``(carriers, geography, device high-water)`` for one full-domain ingest.

    Host copies, so the device state is released before anything else runs --
    on the sizes this file uses that is politeness, and on the sizes this
    project targets it is the whole point.
    """
    import cupy as cp

    from tilestream import driver, gather, realdata

    result, peak = realdata.device_high_water(
        realdata.monolithic_ingest, bundle(), cfg, case_coord(cfg), geo=geo)
    carriers = {k: _as_numpy(v).copy()
                for k, v in gather.inventory(result.state).items()}
    geography = {k: _as_numpy(v).copy()
                 for k, v in driver.geography_inventory(result.state).items()}
    del result
    cp.get_default_memory_pool().free_all_blocks()
    return carriers, geography, peak


# --------------------------------------------------------------------------
# section 1: the window grid
# --------------------------------------------------------------------------

def _grid_fields(grid) -> dict[str, np.ndarray]:
    lat, lon = grid.latlon_mass()
    ulat, ulon = grid.latlon_u()
    vlat, vlon = grid.latlon_v()
    f, e = grid.coriolis_m()
    sina, cosa = grid.rotation_m()
    return dict(lat=lat, lon=lon, ulat=ulat, ulon=ulon, vlat=vlat, vlon=vlon,
                msft=grid.mapfac_m(), msfu=grid.mapfac_u(),
                msfv=grid.mapfac_v(), f=f, e=e, sina=sina, cosa=cosa)


def _slice_grid_fields(parent: dict, i0, j0, nx, ny) -> dict[str, np.ndarray]:
    y, x = slice(j0, j0 + ny), slice(i0, i0 + nx)
    yf, xf = slice(j0, j0 + ny + 1), slice(i0, i0 + nx + 1)
    return dict(lat=parent["lat"][y, x], lon=parent["lon"][y, x],
                ulat=parent["ulat"][y, xf], ulon=parent["ulon"][y, xf],
                vlat=parent["vlat"][yf, x], vlon=parent["vlon"][yf, x],
                msft=parent["msft"][y, x], msfu=parent["msfu"][y, xf],
                msfv=parent["msfv"][yf, x], f=parent["f"][y, x],
                e=parent["e"][y, x], sina=parent["sina"][y, x],
                cosa=parent["cosa"][y, x])


def _great_circle_km(lat1, lon1, lat2, lon2) -> float:
    r = 6370.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(lon2 - lon1)
    return float(r * np.arccos(np.clip(
        np.sin(p1) * np.sin(p2) + np.cos(p1) * np.cos(p2) * np.cos(dl),
        -1.0, 1.0)).max())


def grid_section() -> None:
    from tilestream import realdata

    banner("WINDOW GRID  (real74 d01 Lambert, 250x200 at dx=dy=12 km)")
    grid = full_geography()["grid"]
    parent = _grid_fields(grid)

    windows = ((0, 0, FULL_NX, FULL_NY), (0, 37, FULL_NX, 25),
               (11, 37, 40, 25), (ATLANTIC_I0, ATLANTIC_J0,
                                  ATLANTIC_NX, ATLANTIC_NY),
               (FULL_NX - 3, FULL_NY - 3, 3, 3))
    for i0, j0, nx, ny in windows:
        got = _grid_fields(realdata.window_grid(grid, i0, j0, nx, ny))
        want = _slice_grid_fields(parent, i0, j0, nx, ny)
        bad = sorted(k for k in want if not np.array_equal(got[k], want[k]))
        report(not bad,
               f"window_grid [{j0}, {j0 + ny}) x [{i0}, {i0 + nx}) is the "
               f"parent's own slice",
               f"{len(want)} derived fields compared byte for byte"
               + ("" if not bad else f"; DIFFER on {bad}"))

    # NEGATIVE: the projection rebuilt from the window's own extents.
    i0, j0, nx, ny = 11, 37, 40, 25
    rebuilt = _grid_fields(realdata.rebuilt_grid(grid, i0, j0, nx, ny))
    want = _slice_grid_fields(parent, i0, j0, nx, ny)
    same = sorted(k for k in want if np.array_equal(rebuilt[k], want[k]))
    dlat = float(np.abs(rebuilt["lat"] - want["lat"]).max())
    dlon = float(np.abs(rebuilt["lon"] - want["lon"]).max())
    km = _great_circle_km(rebuilt["lat"], rebuilt["lon"],
                          want["lat"], want["lon"])
    dmsf = float(np.abs(rebuilt["msft"] / want["msft"] - 1.0).max())
    dcor = float(np.abs(rebuilt["f"] / want["f"] - 1.0).max())
    report(not same,
           "NEGATIVE rebuilt_grid: a slab that derives its own projection",
           f"displaced by {dlat:.3f} deg lat / {dlon:.3f} deg lon = "
           f"{km:.0f} km; map factor off by {100 * dmsf:.2f}%, Coriolis by "
           f"{100 * dcor:.2f}%",
           f"fields still bit-identical: {same or 'none'}")

    # The EXACTLY CENTRED window is bit-exact even when rebuilt.  That is WHY
    # a one-window test certifies the bug: pick the middle and it passes.
    #
    # "Exactly centred" is the known-point equation, not ``(nx - nxw) // 2``.
    # ``projection.py:122-123`` defaults ``known_x`` to ``e_we / 2``, so a
    # rebuilt window agrees iff ``i0 + (nxw + 1) / 2 == (nx + 1) / 2`` -- which
    # has an integer solution only when ``nxw`` and ``nx`` share a parity.
    # 40x26 inside 250x200 does; the 40x25 window above does NOT, which is
    # itself worth knowing: half of all window sizes cannot be centred at all
    # and every one of them is wrong everywhere.
    cnx, cny = 40, 26
    ci0 = int(grid.known_x - (cnx + 1) / 2.0)
    cj0 = int(grid.known_y - (cny + 1) / 2.0)
    exact = (ci0 + (cnx + 1) / 2.0 == grid.known_x
             and cj0 + (cny + 1) / 2.0 == grid.known_y)
    centred = _grid_fields(realdata.rebuilt_grid(grid, ci0, cj0, cnx, cny))
    cwant = _slice_grid_fields(parent, ci0, cj0, cnx, cny)
    agree = sorted(k for k in cwant if np.array_equal(centred[k], cwant[k]))
    report(exact and len(agree) == len(cwant),
           "the EXACTLY CENTRED window agrees even when rebuilt -- which is "
           "why a one-window test certifies the bug",
           f"{len(agree)}/{len(cwant)} fields bit-identical for the "
           f"{cny}x{cnx} window at i0={ci0}, j0={cj0}")


# --------------------------------------------------------------------------
# section 2: the slab plan
# --------------------------------------------------------------------------

def plan_section() -> None:
    from tilestream import realdata

    banner("SLAB PLAN")
    ok = True
    detail = []
    for ny in (1, 7, 64, FULL_NY):
        for rows in (1, 3, 8, 64, FULL_NY, FULL_NY + 5):
            slabs = realdata.plan_row_slabs(ny, rows)
            owned = np.concatenate([np.arange(s.j0, s.j1) for s in slabs])
            if not np.array_equal(owned, np.arange(ny)):
                ok = False
                detail.append(f"ny={ny} rows={rows}: owned rows are not a "
                              f"partition of [0, {ny})")
            if sum(s.last for s in slabs) != 1:
                ok = False
                detail.append(f"ny={ny} rows={rows}: {sum(s.last for s in slabs)}"
                              " slabs claim the closing v face")
            for s in slabs:
                want_g0 = max(0, s.j0 - realdata.SLAB_Y_HALO)
                if s.g0 != want_g0 or s.g1 != s.j1:
                    ok = False
                    detail.append(f"ny={ny} rows={rows}: slab {s.index} is "
                                  f"built over [{s.g0}, {s.g1})")
    report(ok, "owned rows partition the domain and exactly one slab owns the "
               "closing v face, over 24 (ny, rows_per_slab) pairs", *detail)

    slabs = realdata.plan_row_slabs(FULL_NY, 37)
    report(all(realdata.window_rows("v", s)[1] - realdata.window_rows("v", s)[0]
               == s.owned_rows + (1 if s.last else 0) for s in slabs)
           and all(realdata.window_rows("mass", s)[1]
                   - realdata.window_rows("mass", s)[0] == s.owned_rows
                   for s in slabs),
           "window_rows gives every slab its mass rows and the last slab one "
           "extra v face",
           "  ".join(f"slab{s.index}:{realdata.window_rows('v', s)[:2]}"
                     for s in slabs))


# --------------------------------------------------------------------------
# section 3: the ingest
# --------------------------------------------------------------------------

def ingest_case(cfg, geo, rows_per_slab, ref_carriers, ref_geography, *,
                y_halo=None, row_shift=0, force_kind=None,
                grid_fn=None) -> dict:
    """One slab-ingest configuration, against the monolithic reference."""
    import cupy as cp

    from tilestream import driver, gather, hoststore, realdata

    if y_halo is None:
        y_halo = realdata.SLAB_Y_HALO
    store = hoststore.HostDomainStore(cfg)
    # POISON.  A row no slab writes must be a NaN, never a plausible zero;
    # ``force_kind='mass'`` leaves exactly one and this is what makes it show.
    for array in store.arrays.values():
        array.fill(np.nan)
    # Sized from the REFERENCE's own arrays, which are at domain shape, so
    # ``probe_ny`` is the domain's ny and the staggering classification is
    # the same one the write-back will make.  (The production caller sizes it
    # from a one-row probe slab instead -- see :func:`steps_case` -- because
    # it does not have a full-domain state to read shapes off, which is the
    # entire premise.)
    geography = realdata.allocate_domain_arrays(
        ref_geography, nz=int(cfg.nz), probe_ny=int(cfg.ny), ny=int(cfg.ny),
        nx=int(cfg.nx))
    rep: dict = {}
    t0 = time.perf_counter()
    try:
        realdata.stream_real_into_store(
            store, bundle(), cfg, case_coord(cfg),
            rows_per_slab=rows_per_slab, geo=geo, geography=geography,
            y_halo=y_halo, row_shift=row_shift, force_kind=force_kind,
            grid_fn=grid_fn, report=rep)
    except realdata.SlabIngestError as exc:
        store.free()
        cp.get_default_memory_pool().free_all_blocks()
        return {"refused": str(exc), "bitexact": False, "seconds":
                time.perf_counter() - t0}
    elapsed = time.perf_counter() - t0
    carriers = compare_arrays(store.arrays, ref_carriers)
    geo_cmp = compare_arrays(geography, ref_geography)
    out = {"carriers": carriers, "geography": geo_cmp,
           "bitexact": carriers["bitexact"] and geo_cmp["bitexact"],
           "seconds": elapsed, "refused": None}
    out.update(rep)
    store.free()
    del geography
    cp.get_default_memory_pool().free_all_blocks()
    return out


def _summarise(record: dict) -> str:
    car, geo = record["carriers"], record["geography"]
    parts = []
    for label, cmp in (("carriers", car), ("geography", geo)):
        if cmp["bitexact"]:
            parts.append(f"{cmp['fields']} {label} bit-exact")
        else:
            rows = ", ".join(f"{k}:{cmp['rows'][k]} rows"
                             for k in cmp["differing"][:4])
            parts.append(f"{len(cmp['differing'])}/{cmp['fields']} {label} "
                         f"differ ({rows}) maxabs={cmp['max_abs']:.4g} "
                         f"maxrel={cmp['max_rel']:.2e}")
    return "; ".join(parts)


def ingest_section(quick: bool) -> None:
    from tilestream import realdata

    cfg = case_config()
    geo = full_geography()
    banner(f"REAL INGEST  ({cfg.nz}x{cfg.ny}x{cfg.nx} real74 d01, "
           f"ERA5 {realdata.REAL74_START:%Y-%m-%d %H}Z, slab by slab)")

    ref_carriers, ref_geography, mono_peak = monolithic_reference(cfg, geo)
    cells = cfg.ny * cfg.nx
    print(f"  monolithic ingest: {len(ref_carriers)} carriers, "
          f"{len(ref_geography)} geography arrays, device high-water "
          f"{mono_peak / 2**30:.4f} GiB = {mono_peak / cells:.0f} B per mass "
          f"cell")

    heights = QUICK_SLAB_HEIGHTS if quick else SLAB_HEIGHTS
    for rows in heights:
        rec = ingest_case(cfg, geo, rows, ref_carriers, ref_geography)
        report(rec["bitexact"],
               f"rows_per_slab={rows} ({rec.get('slabs', '?')} slabs)",
               f"{_summarise(rec)}",
               f"device high-water {rec['device_high_water_bytes'] / 2**30:.4f}"
               f" GiB = {rec['device_high_water_bytes'] / cells:.0f} B/cell, "
               f"{rec['bytes_written'] / 2**20:.0f} MiB written, "
               f"{rec['seconds']:.1f} s")

    banner("REAL INGEST -- NEGATIVE CONTROLS  (each of these MUST fail)")
    rec = ingest_case(cfg, geo, 37, ref_carriers, ref_geography, y_halo=0)
    nslabs = len(realdata.plan_row_slabs(cfg.ny, 37))
    car = rec["carriers"]
    signature = (car["differing"] == ["v"]
                 and car["rows"].get("v") == nslabs - 1)
    report(not rec["bitexact"] and signature,
           "NEGATIVE y_halo=0: the slab drops the mass row below its first",
           _summarise(rec),
           f"expected exactly ['v'] in {nslabs - 1} rows (one per interior "
           f"seam); got {car['differing']} in "
           f"{[car['rows'].get(k) for k in car['differing']]} rows")

    rec = ingest_case(cfg, geo, 37, ref_carriers, ref_geography, row_shift=1)
    report(not rec["bitexact"] or rec["refused"],
           "NEGATIVE row_shift=1: every slab written one row too high",
           rec["refused"] or _summarise(rec))

    rec = ingest_case(cfg, geo, 37, ref_carriers, ref_geography,
                      force_kind="mass")
    car = rec.get("carriers")
    detail = (rec["refused"] if rec["refused"] else _summarise(rec))
    unwritten = (car["nonfinite"].get("v", 0) if car else 0)
    report(not rec["bitexact"],
           "NEGATIVE force_kind='mass': nobody owns the closing v face",
           detail,
           f"v cells left at the poison NaN: {unwritten}")

    rec = ingest_case(cfg, geo, 37, ref_carriers, ref_geography,
                      grid_fn=realdata.rebuilt_grid)
    report(not rec["bitexact"] or rec["refused"],
           "NEGATIVE grid_fn=rebuilt_grid: every slab derives its own "
           "projection",
           rec["refused"] or _summarise(rec),
           "the GEOGRAPHY stays bit-exact and that is correct, not a miss: "
           "this case reads its map factors, Coriolis and terrain from the "
           "bundle's geo_em and slices them by row, so a wrong grid moves "
           "only what the grid is used for -- the analysis interpolation "
           "targets.  A case that DERIVED its statics from the grid "
           "(runtime.prepare_real_case does, through build_static) would "
           "have both wrong, and only the analysis half shows here.")




# --------------------------------------------------------------------------
# section 4: streaming a real-data store
# --------------------------------------------------------------------------

#: The timestep each streamed row runs at, and WHY it is not the case's own.
#:
#: MEASURED, on the monolithic side, counting steps that stay finite out of 8:
#:
#: ==============  =====  =====  =====  ======  ======  ======
#: domain           dt=30  dt=6   dt=2   dt=0.5  dt=0.1  dt=0.05
#: ==============  =====  =====  =====  ======  ======  ======
#: real74 d01        1      -      2      3       6       8
#: Atlantic 64x64    1      2      2      4       8       -
#: ==============  =====  =====  =====  ======  ======  ======
#:
#: A periodic wrap of a real analysis is a genuine grid-scale shock: the first
#: |w| maximum lands on the seam column (i = 249 at dt=30, i = 0 at dt=2), and
#: the growth rate is ~2x PER STEP whatever dt is, which is what says it is a
#: numerical instability of the wrapped configuration and not a CFL violation.
#: It is NOT the terrain either: the Atlantic window's terrain is sea level to
#: within 0.3 m and it fails at step 2 just as fast at dt=30.
#:
#: So the row at the case's own dt=30 gets the one step it has, and the
#: multi-step rows run at the largest dt in the sweep that stays finite for
#: eight -- 0.1 s on the window, where |w| grows 0.01 -> 0.18 m/s and the
#: state stays recognisably the analysis.  Nothing about the halo changes with
#: dt: ``harness.halo_radius`` is a function of ``time_step_sound`` alone.
STEP_DT_NATIVE = 30.0
STEP_DT_STABLE = 0.1


def steps_case(label, i0, j0, nx, ny, tile, nsteps, *, dt=STEP_DT_STABLE,
               halo=None, rows_per_slab=8, gather_geography=True,
               nbuffers=2, write_mode="shadow", default_stretch=False) -> dict:
    """Ingest by slabs, stream ``nsteps``, compare with monolithic-then-step.

    The two sides differ in exactly one thing: whether the domain was ever
    resident.  Same GRIB (one decode, cached), same grid, same coordinate
    table, same periodic closure applied at the same point.
    """
    import cupy as cp

    from tilestream import driver, gather, harness, hoststore, realdata
    from tilestream import spec as tspec

    cfg = dataclasses.replace(case_config(nx, ny, periodic=True),
                              dt=float(dt))
    geo = (full_geography() if (i0, j0, nx, ny) == (0, 0, FULL_NX, FULL_NY)
           else window_geography(i0, j0, nx, ny))
    if halo is None:
        halo = harness.halo_radius(cfg)

    # -- monolithic: ingest the whole domain onto the GPU, close, step -----
    result = realdata.monolithic_ingest(bundle(), cfg, case_coord(cfg),
                                        geo=geo)
    state = result.state
    realdata.periodic_closure(gather.inventory(state), nz=cfg.nz, ny=ny, nx=nx)
    realdata.periodic_closure(driver.geography_inventory(state), nz=cfg.nz,
                              ny=ny, nx=nx)
    harness.run_steps(state, cfg, nsteps)
    ref = {k: _as_numpy(v).copy() for k, v in gather.inventory(state).items()}
    nan_free = all(bool(np.isfinite(v).all()) for v in ref.values())
    wmax = float(np.abs(np.nan_to_num(ref["w"])).max())
    del result, state
    cp.get_default_memory_pool().free_all_blocks()

    # -- streamed: slab ingest into pinned host RAM, then run_tiled --------
    store = hoststore.HostDomainStore(cfg)
    for array in store.arrays.values():
        array.fill(np.nan)
    # The geography store is sized from a ONE-ROW probe slab, not from a
    # domain state, because a domain state is the thing this lane may not
    # build.  ``allocate_domain_arrays`` reads each array's staggering off
    # the probe's shapes and re-expands y; the ingest comparison above is
    # what says the shapes it picks are the ones the reference has.
    probe = realdata.ingest_slab_state(
        realdata._decode(bundle(), realdata.REAL74_START), geo, cfg,
        case_coord(cfg),
        realdata.RowSlab(index=0, j0=0, j1=1, g0=0, g1=1, last=False),
        diagnose=False)
    geography = realdata.allocate_domain_arrays(
        driver.geography_inventory(probe.state), nz=int(cfg.nz), probe_ny=1,
        ny=ny, nx=nx)
    del probe
    cp.get_default_memory_pool().free_all_blocks()
    rep: dict = {}
    realdata.stream_real_into_store(store, bundle(), cfg, case_coord(cfg),
                                   rows_per_slab=rows_per_slab, geo=geo,
                                   geography=geography, report=rep)
    realdata.periodic_closure(store.arrays, nz=cfg.nz, ny=ny, nx=nx)
    realdata.periodic_closure(geography, nz=cfg.nz, ny=ny, nx=nx)

    specs = tspec.plan_tiles(nx, ny, tile, tile, halo, True)
    tspec.validate_plan(specs, ny, nx)
    out_report: dict = {}
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        driver.run_tiled(
            store.arrays, cfg, tile, tile, halo=halo, nsteps=nsteps,
            nbuffers=nbuffers, write_mode=write_mode,
            geography=(geography if gather_geography else None),
            tile_state_factory=lambda tc: realdata.make_real_tile_state(
                tc, (default_coord(cfg) if default_stretch
                     else (lambda: case_coord(cfg)))),
            report=out_report)
        cp.cuda.runtime.deviceSynchronize()
    elapsed = time.perf_counter() - t0
    record = dict(label=label, cmp=compare_arrays(store.arrays, ref),
                  nan_free=nan_free, wmax=wmax, tiles=len(specs), dt=float(dt),
                  compute=out_report.get("compute"), halo=int(halo),
                  nsteps=nsteps, seconds=elapsed, slabs=rep["slabs"],
                  ingest_peak=rep["device_high_water_bytes"],
                  gathered=out_report.get("gathered_bytes", 0),
                  write_mode=write_mode, nbuffers=nbuffers)
    store.free()
    del geography
    cp.get_default_memory_pool().free_all_blocks()
    return record


def _steps_detail(rec: dict) -> tuple[str, ...]:
    cmp = rec["cmp"]
    lines = [
        f"{rec['tiles']} tiles, compute {rec['compute']}, halo {rec['halo']}, "
        f"N={rec['nsteps']} at dt={rec['dt']} s, {rec['write_mode']} write, "
        f"{rec['nbuffers']} buffers",
        f"ingested in {rec['slabs']} slabs "
        f"({rec['ingest_peak'] / 2**30:.4f} GiB device high-water), "
        f"{rec['gathered'] / 2**20:.0f} MiB gathered, {rec['seconds']:.1f} s",
        f"monolithic run NaN-free: {rec['nan_free']}, |w|max "
        f"{rec['wmax']:.3f} m/s",
    ]
    if not cmp["bitexact"]:
        lines.append(f"{len(cmp['differing'])}/{cmp['fields']} differ: "
                     f"{cmp['differing'][:8]} maxabs={cmp['max_abs']:.4g} "
                     f"maxrel={cmp['max_rel']:.2e}")
    return tuple(lines)


def tile_buffer_section() -> None:
    """What a tile buffer must rebuild EXACTLY, checked rather than assumed.

    ``driver.geography_inventory`` gathers only the horizontally varying setup
    arrays and states plainly that the purely vertical ones -- ``c1h..c4f``,
    ``znu``, ``znw``, ``dnw``, ``rdnw``, ``dn``, ``rdn``, ``fnp``, ``fnm`` --
    are left to the tile to rebuild.  On a domain with an explicit eta table
    that is a real obligation on whoever writes the tile factory, and nothing
    in ``run_tiled`` enforces it: ``setup_window_mismatches`` and
    ``setup_scalar_mismatches`` exist and are never called from the run path.
    So they are called here, on a real buffer against a real parent.
    """
    import cupy as cp

    from tilestream import driver, harness, realdata
    from tilestream import spec as tspec

    banner("TILE BUFFER  (what a tile rebuilds, and what it must not)")
    nx, ny, tile = ATLANTIC_NX, ATLANTIC_NY, 16
    cfg = dataclasses.replace(
        case_config(nx, ny, periodic=True), dt=STEP_DT_STABLE)
    geo = window_geography(ATLANTIC_I0, ATLANTIC_J0, nx, ny)
    halo = harness.halo_radius(cfg)
    specs = tspec.plan_tiles(nx, ny, tile, tile, halo, True)
    parent = realdata.monolithic_ingest(bundle(), cfg, case_coord(cfg),
                                        geo=geo).state
    tile_cfg = harness.tile_config(cfg, specs[0].cnx, specs[0].cny)

    good = realdata.make_real_tile_state(tile_cfg, lambda: case_coord(cfg))
    bad = realdata.make_real_tile_state(tile_cfg, default_coord(cfg))
    dznu = float(cp.abs(cp.asarray(good.znu) - cp.asarray(bad.znu)).max())

    scalars = driver.setup_scalar_mismatches(good, parent)
    vertical = {k: v for k, v in
                driver.setup_window_mismatches(good, parent, specs[0]).items()
                if np.ndim(getattr(parent, k)) < 2}
    report(not scalars and not vertical,
           "a tile buffer built with the case's own eta table rebuilds every "
           "vertical setup array and scalar exactly",
           f"setup scalars differing: {scalars or 'none'}; vertical setup "
           f"arrays differing: {vertical or 'none'}",
           f"the default stretch would put znu up to {dznu:.4f} away")

    bad_scalars = driver.setup_scalar_mismatches(bad, parent)
    bad_vertical = {k: v for k, v in
                    driver.setup_window_mismatches(bad, parent,
                                                   specs[0]).items()
                    if np.ndim(getattr(parent, k)) < 2}
    report(bool(bad_scalars or bad_vertical),
           "NEGATIVE a tile buffer on the DEFAULT vertical stretch",
           f"setup scalars differing: {sorted(bad_scalars) or 'none'}; "
           f"vertical setup arrays differing: {sorted(bad_vertical) or 'none'}")
    del parent, good, bad
    cp.get_default_memory_pool().free_all_blocks()

    rec = steps_case("default stretch", i0=ATLANTIC_I0, j0=ATLANTIC_J0,
                     nx=nx, ny=ny, tile=tile, nsteps=8, default_stretch=True)
    report(not rec["cmp"]["bitexact"],
           "NEGATIVE the same buffer, streamed eight steps",
           *_steps_detail(rec))


def steps_section(quick: bool) -> None:
    from tilestream import harness

    banner("STREAMED STEPS FROM REAL DATA  (periodic closure -- read the "
           "module docstring and STEP_DT_NATIVE before quoting these)")

    rows = (
        ("real74 d01 250x200, the case's own dt, every step it has",
         dict(i0=0, j0=0, nx=FULL_NX, ny=FULL_NY, tile=100, nsteps=1,
              dt=STEP_DT_NATIVE, rows_per_slab=37)),
        ("real74 d01 250x200, four steps at the stable dt",
         dict(i0=0, j0=0, nx=FULL_NX, ny=FULL_NY, tile=100,
              nsteps=1 if quick else 4, dt=STEP_DT_STABLE, rows_per_slab=8)),
        ("western Atlantic 64x64, eight steps, 16 tiles, 8-row slabs",
         dict(i0=ATLANTIC_I0, j0=ATLANTIC_J0, nx=ATLANTIC_NX,
              ny=ATLANTIC_NY, tile=16, nsteps=1 if quick else 8,
              dt=STEP_DT_STABLE, rows_per_slab=8)),
    )
    if not quick:
        rows += (
            ("western Atlantic 64x64, ONE buffer (no prefetch overlap)",
             dict(i0=ATLANTIC_I0, j0=ATLANTIC_J0, nx=ATLANTIC_NX,
                  ny=ATLANTIC_NY, tile=16, nsteps=8, nbuffers=1)),
            ("western Atlantic 64x64, halo RINGS instead of a shadow store",
             dict(i0=ATLANTIC_I0, j0=ATLANTIC_J0, nx=ATLANTIC_NX,
                  ny=ATLANTIC_NY, tile=16, nsteps=8, write_mode="ring")),
            ("western Atlantic 64x64, one row per ingest slab (64 slabs)",
             dict(i0=ATLANTIC_I0, j0=ATLANTIC_J0, nx=ATLANTIC_NX,
                  ny=ATLANTIC_NY, tile=16, nsteps=8, rows_per_slab=1)),
        )
    for headline, kwargs in rows:
        rec = steps_case(headline, **kwargs)
        report(rec["cmp"]["bitexact"] and rec["nan_free"], headline,
               *_steps_detail(rec))

    tile_buffer_section()

    if quick:
        return

    banner("STREAMED STEPS -- NEGATIVE CONTROLS  (each of these MUST fail)")
    cfg = case_config(ATLANTIC_NX, ATLANTIC_NY, periodic=True)
    need = harness.halo_radius(cfg)
    atlantic = dict(i0=ATLANTIC_I0, j0=ATLANTIC_J0, nx=ATLANTIC_NX,
                    ny=ATLANTIC_NY, tile=16, nsteps=8)

    rec = steps_case("halo too short", halo=need - 9, **atlantic)
    report(not rec["cmp"]["bitexact"],
           f"NEGATIVE halo={need - 9} against the measured radius {need} "
           f"(at dx=12 km the gate's own smallest bit-exact halo is 8)",
           *_steps_detail(rec))

    rec = steps_case("geography rebuilt", gather_geography=False, **atlantic)
    report(not rec["cmp"]["bitexact"],
           "NEGATIVE geography not gathered: every tile keeps its buffer's "
           "own flat ground, identity map factors and zero Coriolis",
           *_steps_detail(rec))

    rec = steps_case("read-at-time-t", write_mode="inplace", **atlantic)
    report(not rec["cmp"]["bitexact"],
           "NEGATIVE write_mode='inplace': a tile's halo sees its neighbour "
           "at t+dt",
           *_steps_detail(rec))

# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    quick = "--quick" in argv
    only = ({"--grid-only", "--plan-only", "--ingest-only", "--steps-only"}
            & set(argv))
    run = {name: (not only or f"--{name}-only" in only)
           for name in ("grid", "plan", "ingest", "steps")}

    try:
        bundle()
    except FileNotFoundError as exc:
        print(f"SKIPPED: {exc}")
        return 0

    started = time.perf_counter()
    try:
        if run["grid"]:
            grid_section()
        if run["plan"]:
            plan_section()
        if run["ingest"]:
            ingest_section(quick)
        if run["steps"]:
            steps_section(quick)
    except Exception:                              # pragma: no cover
        traceback.print_exc()
        return 2

    failed = [name for ok, name in _RESULTS if not ok]
    banner("REAL-DATA GATE " + ("PASSED" if not failed else "FAILED"))
    print(f"{len(_RESULTS) - len(failed)}/{len(_RESULTS)} rows behaved as "
          f"specified, including every negative control, in "
          f"{time.perf_counter() - started:.1f} s")
    for name in failed:
        print("  FAILED: " + name)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
