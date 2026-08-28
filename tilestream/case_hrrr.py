"""A real outbreak, initialised from a real HRRR analysis, run out of core.

Everything in ``tilestream/`` before this module ran on a seeded synthetic
draw or, on the ingest lane, on the frozen April-1974 ERA5 bundle.  This one
runs a CONVECTION-ALLOWING FORECAST of a real severe-weather outbreak from the
real HRRR analysis of that day, on a domain no card in the estate can hold
resident, and it is meant to be looked at by a forecaster rather than by a
digest comparison.

WHAT IS REAL HERE, STATED FIRST BECAUSE IT IS THE POINT
-------------------------------------------------------
* the initial condition is the operational HRRR **analysis** (f00) of the
  case cycle, decoded by the packaged ``hrrr_grib2_bridge`` from the archived
  NOAA GRIB2 on ``noaa-hrrr-bdp-pds``, interpolated to the target Lambert grid
  by ``gpuwm.ingest.hrrr.interpolate_hrrr_to_lambert`` and turned into a model
  state by ``gpuwm.ingest.real.initialize_real`` -- the production ingest,
  called, not reimplemented;
* the terrain, land use, soil category, greenness, albedo and deep soil
  temperature are the real WPS_GEOG geogrid statics for the target grid,
  built by ``gpuwm.static.build.build_static``;
* the lateral boundaries are the SAME HRRR cycle's later forecast hours, each
  one decoded and initialised the same way, applied on the real hourly cadence
  and linearly interpolated between them -- never synthesised from the model's
  own state;
* the physics is the streaming gate's own ``full+MYNN+Noah-MP`` rung.

THE LATERAL BOUNDARY, AND WHY IT IS NOT WRF's
---------------------------------------------
This is the first real lateral boundary condition through the tiled path and
it needed a design decision, so the decision is written down rather than
buried.

WRF's specified boundary is a 1-cell ``spec_zone`` overwrite plus a 4-cell
Davies relaxation applied INSIDE the RK stages, on the DOMAIN's perimeter
(``gpuwm.ingest.lateral_bc.apply_state_lateral_boundaries``).  A tile is
stepped as a small domain of its own, so switching ``cfg.specified`` on for a
tile would apply that treatment to the TILE's four edges -- three of which are
usually interior seams of the real domain, where a boundary condition is
simply wrong.  The kernel walks a perimeter frame
(``gpuwm/core/kernels/lbc_state.cu``'s ``frame_point``) and has no per-side
switch, so "specified on the sides that are really the domain's, free on the
others" is not expressible today.  **That is the gap, and it is real: the
streaming path cannot run WRF's own specified/relaxation boundary unmodified.**

What it CAN run, exactly and with no dycore change, is a wide Davies frame
applied at the sweep seam:

    every tile is stepped exactly as the bit-exact gate steps it -- periodic
    in its own array, no boundary treatment at all -- and after every sweep
    the outer ``hard + taper`` cells of the DOMAIN are replaced by the
    time-interpolated analysis, hard inside ``hard`` and cosine-blended
    across ``taper``.

The correctness argument is the halo argument, run once more:

* a tile's own array edges wrap, so the outermost ``halo_radius(cfg)`` cells
  of a tile's array are contaminated by that wrap after one step.  For an
  INTERIOR tile those cells are the halo and are discarded, which is the
  whole reason the halo exists.  For an EDGE tile, whose window is clamped to
  the domain (``spec.plan_tiles(..., periodic=False)``), they are domain cells
  ``0 .. halo-1``;
* so as long as the hard zone is at least ``halo_radius(cfg) + 1`` cells deep,
  every contaminated cell is inside it and is overwritten before it can be
  read again.  A cell at depth ``hard`` reads a halo reaching to depth
  ``hard - halo >= 1``, i.e. only into cells the previous sweep prescribed
  correctly.  :data:`HARD_MARGIN` is the margin over that bound.

It is a real lateral boundary condition -- a wide-frame Davies scheme, the
family AROME/HIRLAM use -- driven by real analysis data on its real cadence.
It is NOT WRF's, it is not bit-comparable with a monolithic WRF-style run, and
a figure produced through it must not be described as one.  The outer
``hard + taper`` cells are boundary-zone artefact by construction and nothing
inside them is a forecast.

WHAT ELSE THE FRAME HOLDS STILL
-------------------------------
Only the analysis' own prognostic fields can be prescribed from an analysis.
Every OTHER streamed carrier in the hard zone -- the held tendencies, the
acoustic perturbations, ``h_diabatic``, the whole surface/soil/snow bundle --
is restored to its time-zero value there instead, because those cells are
being driven by wrapped garbage every step and are one long integration away
from a NaN that would then be prescribed into the domain.  They are column
local (see ``harness.halo_radius``), so freezing them inside the frame cannot
reach the interior.
"""

from __future__ import annotations

from dataclasses import dataclass, field as _field
from datetime import datetime, timedelta
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np


__all__ = [
    "CASES",
    "CaseSpec",
    "FrameForcing",
    "HARD_MARGIN",
    "PHYSICS",
    "build_case_state",
    "frame_slices",
    "frame_weight",
    "ingest_hour",
    "run_config",
    "snapshots_for",
    "target_for",
]


#: Cells of hard zone BEYOND ``halo_radius(cfg)``, which is the bound the
#: correctness argument actually needs.  Four is margin, not a measurement:
#: the radius is itself the conservative per-step figure (mass points grow 14
#: per step where the bound says 16), so this is margin on margin.  It costs
#: four columns of a 1200-column domain.
HARD_MARGIN = 4

#: Cells of cosine taper INSIDE the hard zone.  A hard frame alone reflects;
#: the taper is the Davies part.  24 cells is 72 km at dx = 3 km.
TAPER_CELLS = 24

#: The streaming gate's ``full+MYNN+Noah-MP`` rung
#: (``tilestream.test_gate.PHYSICS_RUNGS``), with ONE deliberate change:
#: ``cu_physics=0``.  The gated rung runs Kain-Fritsch because its 96x80
#: domain at dx = 500 m is not a cumulus question at all; a 3 km
#: convection-allowing forecast of a supercell outbreak is exactly the
#: configuration in which a cumulus parameterisation is switched OFF, because
#: it removes the instability the explicit updraughts are supposed to release.
#: Every other selector is the gated rung's, unchanged, and cumulus-off is
#: itself a gated setting (the ``+RRTMGP radiation`` rung).
PHYSICS: dict[str, Any] = dict(
    moist=True, mp_physics=10, km_opt=4,
    sf_sfclay_physics=5, bl_pbl_physics=5, bldt=0.0,
    sf_surface_physics=4,
    ra_sw_physics=4, ra_lw_physics=4, radt_minutes=12.0,
    cu_physics=0, cudt_minutes=0.0,
)


@dataclass(frozen=True)
class CaseSpec:
    """One real case: which cycle, which window, which piece of the map."""

    name: str
    #: The HRRR cycle the analysis and every boundary hour come from.
    cycle: datetime
    #: Forecast hours of that cycle to use.  ``0`` is the analysis and becomes
    #: the initial condition; the rest are boundary material.
    hours: tuple[int, ...]
    nx: int
    ny: int
    nz: int
    ref_lat: float
    ref_lon: float
    #: What actually happened, for the figure captions and the verification.
    headline: str
    def_dt: float = 15.0
    #: Bounded nearest-valid search radius for the masked surface
    #: interpolation.  The ingest REFUSES and names the exact value that
    #: works, so every number here is a transcription of that refusal, not
    #: a guess: 30 for the Kansas domain, 40 for the quad-state one (which
    #: reaches further into the Atlantic and the Gulf).
    surface_fallback_radius: int = 30

    @property
    def start_time(self) -> datetime:
        return self.cycle + timedelta(hours=int(self.hours[0]))

    @property
    def end_time(self) -> datetime:
        return self.cycle + timedelta(hours=int(self.hours[-1]))


CASES: dict[str, CaseSpec] = {
    # 2019-05-28.  A dryline across central Kansas fired discrete supercells
    # through the afternoon; the Lawrence-Linwood EF4 was on the ground
    # 2305-0000Z over Douglas and Leavenworth counties, and an EF2 hit Clay
    # County, Missouri at 0110Z.  The 15Z cycle puts model time zero eight
    # hours ahead of the violent tornado and about five ahead of initiation.
    "ks20190528": CaseSpec(
        name="ks20190528",
        cycle=datetime(2019, 5, 28, 15),
        hours=tuple(range(11)),
        nx=1200, ny=900, nz=49,
        ref_lat=38.7, ref_lon=-96.5,
        headline="2019-05-28 central/eastern Kansas dryline supercells "
                 "(Lawrence-Linwood EF4, 2340Z)",
    ),
    # 2021-12-10/11.  The quad-state QLCS/supercell event.  The 18Z cycle of
    # 12-10 puts model time zero about eight hours ahead of the Mayfield
    # tornado (0327Z on 12-11) and ahead of the northeast-Arkansas
    # initiation near 02Z.
    "quad20211210": CaseSpec(
        name="quad20211210",
        cycle=datetime(2021, 12, 10, 18),
        hours=tuple(range(11)),
        nx=1200, ny=900, nz=49,
        ref_lat=37.5, ref_lon=-90.0,
        headline="2021-12-10/11 quad-state derecho/QLCS "
                 "(Mayfield KY EF4, 0327Z)",
        surface_fallback_radius=40,
    ),
}


# --------------------------------------------------------------------------
# geometry and config
# --------------------------------------------------------------------------

def target_for(case: CaseSpec):
    """The :class:`gpuwm.ingest.hrrr_target.HrrrTargetDomain` of ``case``.

    ``dx`` is ``HRRR_WPS_EQUIVALENT_DX_M`` -- HRRR's own 3 km spacing
    corrected for the difference between HRRR's earth radius (6 371 229 m)
    and WPS's (``gpuwm.static.lambert.EARTH_RADIUS_M``) -- and the cone is
    HRRR's own (truelat 38.5/38.5, stand_lon -97.5).  The target is therefore
    the SAME projection as the source at the SAME spacing, so the horizontal
    interpolation is a translation rather than a change of map, which is the
    whole reason a HRRR-era case is easier than a reanalysis one.
    """
    from gpuwm.ingest.hrrr import HRRR_WPS_EQUIVALENT_DX_M
    from gpuwm.ingest.hrrr_target import HrrrTargetDomain

    return HrrrTargetDomain(
        name=case.name, map_proj="lambert",
        nx=int(case.nx), ny=int(case.ny), nz=int(case.nz),
        dx_m=HRRR_WPS_EQUIVALENT_DX_M, dy_m=HRRR_WPS_EQUIVALENT_DX_M,
        ref_lat=float(case.ref_lat), ref_lon=float(case.ref_lon),
        truelat1=38.5, truelat2=38.5, stand_lon=-97.5,
        time_step_seconds=int(case.def_dt),
        # 30, not the 8 of the shipped 192x160 target.  A 1200x900 CONUS
        # domain reaches the Gulf, the Atlantic and the Pacific, and its
        # masked-surface interpolation needs a land donor for every target
        # cell the GEOGRID calls land -- including one or two islands and
        # capes that HRRR's own 3 km land mask calls water.  The ingest
        # refused at radius 8 naming the exact remedy and proving that the
        # enlarged source window still fits inside the native grid; this is
        # that remedy, taken rather than worked around.
        surface_fallback_radius_cells=int(case.surface_fallback_radius))


def run_config(case: CaseSpec, **overrides):
    """The domain ``RunConfig``: real projection, real terrain, full physics.

    ``specified``/``nested`` stay FALSE.  A tile is stepped under this config
    with only ``nx``/``ny`` replaced (``harness.tile_config``), and switching
    them on would hand every tile a boundary treatment on its own four edges
    -- see the module docstring.  The domain's boundary is imposed at the
    sweep seam instead.
    """
    from tilestream import harness

    target = target_for(case)
    kwargs = dict(
        PHYSICS,
        dx=float(target.dx_m), dy=float(target.dy_m),
        dt=float(case.def_dt), ztop=20000.0,
        map_proj=1, terrain_opt=1, hybrid_opt=2, etac=0.2,
        time_step_sound=4,
        damp_opt=3, zdamp=5000.0, dampcoef=0.2, w_damping=1,
        diff_6th_opt=2, diff_6th_factor=0.10, diff_6th_slopeopt=1,
        epssm=0.5, smdiv=0.1, emdiv=0.01,
        h_sca_adv_order=5, moist_adv_opt=1, hypsometric_opt=2,
        base_temp=290.0, num_soil_layers=4,
        spec_bdy_width=5, spec_zone=1, relax_zone=4,
        run_seconds=float((len(case.hours) - 1) * 3600),
    )
    kwargs.update(overrides)
    return harness.make_config(case.nx, case.ny, case.nz,
                               periodic=False, specified=False, nested=False,
                               open_x=False, open_y=False, **kwargs)


# --------------------------------------------------------------------------
# ingest
# --------------------------------------------------------------------------

def snapshots_for(case: CaseSpec, bridge_root, manifest_sha256=None):
    """``{hour: HrrrNativeSnapshot}`` for the decoded native HRRR window.

    With ``manifest_sha256`` this is the evidence-loading API
    (``load_hrrr_native_series``), which re-hashes the sealed ``SHA256SUMS``
    against a digest recorded OUTSIDE the bridge directory.  Without it, it
    is ``load_hrrr_pipeline_ready_window``, whose contract is that the caller
    owns the decoder process and has consumed its verdict -- which is this
    lane's position: the decoder ran here, ``gate.txt`` reads ``status PASS``
    and the window/inventory/qice lines are in the run log.  The distinction
    is recorded rather than blurred: a bundle handed to somebody else must go
    through the sealed route.
    """
    from gpuwm.ingest.hrrr import (load_hrrr_native_series,
                                   load_hrrr_pipeline_ready_window)

    hours = tuple(int(h) for h in case.hours)
    if manifest_sha256:
        got = load_hrrr_native_series(
            Path(bridge_root), hours,
            expected_manifest_sha256=str(manifest_sha256))
        return dict(zip(hours, got))
    return {h: load_hrrr_pipeline_ready_window(Path(bridge_root), h)
            for h in hours}


def ingest_hour(snapshot, grid, cfg, coord, static, *, report=None,
                surface_fallback_radius: int = 30):
    """Analysis -> horizontal -> vertical -> ``DomainState``, for ONE hour.

    Exactly ``tools/hrrr_two_domain_forecast._initialize_state``, which is
    the tested single-domain HRRR path, with the map factors and Coriolis
    taken from the projected grid rather than from a ``geo_em`` file (this
    lane has no geo_em -- the statics are built from WPS_GEOG directly).
    """
    from gpuwm.ingest.hrrr import interpolate_hrrr_to_lambert
    from gpuwm.ingest.real import initialize_real

    met = interpolate_hrrr_to_lambert(
        snapshot, grid, target_landmask=static["LANDMASK"],
        surface_fallback_radius=int(surface_fallback_radius),
        soil_mapping_report=({} if report is None else report))
    result = initialize_real(met, cfg, coord, static["HGT_M"], grid=grid,
                             p_top=10000.0, sfcp_to_sfcp=True)
    f, e = grid.coriolis_m()
    sina, cosa = grid.rotation_m()
    result.state.set_map_coriolis(
        grid.mapfac_m(), grid.mapfac_u(), grid.mapfac_v(), f, e,
        sina=sina, cosa=cosa)
    return result, met


def build_case_state(case: CaseSpec, snapshot, grid, cfg, coord, static,
                     landuse_attrs, *, progress=print):
    """``(state, driver, result, met)`` at time zero, physics attached.

    The physics comes from ``gpuwm.ingest.hrrr_physics
    .initialize_hrrr_physics`` -- the native HRRR runner's OWN surface
    initialisation, which is not interchangeable with the ERA5/GFS one:
    HRRR's soil arrives on the nine RUC depths and goes through
    ``gpuwm.ingest.ruc_soil.preprocess_land_surface_soil``, its deep soil
    temperature is the geogrid ``SOILTEMP`` rather than the elevation-
    corrected ``TMN``, and its land/soil reconciliation is resolved from the
    RUC soil column.  Calling the ERA5 route on an HRRR met refuses outright
    (there is no ``RW_SOIL_TEMPERATURE`` in an HRRR snapshot), which is the
    fail-closed behaviour working.
    """
    from gpuwm.ingest.hrrr_physics import initialize_hrrr_physics

    start_time = case.start_time
    t0 = time.perf_counter()
    result, met = ingest_hour(
        snapshot, grid, cfg, coord, static,
        surface_fallback_radius=int(case.surface_fallback_radius))
    progress(f"    initialize_real f00 in {time.perf_counter() - t0:.1f}s")

    attrs = dict(landuse_attrs)
    attrs["CEN_LAT"] = float(case.ref_lat)
    t1 = time.perf_counter()
    driver = initialize_hrrr_physics(result, cfg, met, static, attrs, grid,
                                     start_time)
    progress(f"    initialize_hrrr_physics in "
             f"{time.perf_counter() - t1:.1f}s")
    return result.state, driver, result, met


# --------------------------------------------------------------------------
# the frame
# --------------------------------------------------------------------------

def frame_slices(shape, width: int):
    """The four rectangles of a ``width``-deep perimeter frame.

    Written against the LAST TWO axes only, so one function serves mass
    ``(.., ny, nx)``, u ``(.., ny, nx+1)``, v ``(.., ny+1, nx)`` and the 2-D
    surface fields without ever being told which it has.  The extra staggered
    face rides inside the frame, which is what we want: a staggered field's
    closing face is a boundary face and belongs to the boundary zone.
    """
    ny, nx = int(shape[-2]), int(shape[-1])
    if width * 2 >= min(ny, nx):
        raise ValueError(f"frame width {width} does not fit in {(ny, nx)}")
    return (
        (slice(0, width), slice(0, nx)),
        (slice(ny - width, ny), slice(0, nx)),
        (slice(width, ny - width), slice(0, width)),
        (slice(width, ny - width), slice(nx - width, nx)),
    )


def frame_weight(shape, hard: int, taper: int) -> np.ndarray:
    """``(ny, nx)`` blend weight: 1 in the hard zone, cosine to 0 at ``taper``.

    Weight 1 means "the analysis, exactly"; weight 0 means "the model, left
    alone".  The cosine is WRF's own relaxation shape in spirit rather than
    in coefficients -- this is a once-per-step domain-level nudge, not the
    per-RK-stage ``relax_bdytend_core`` -- and it is what stops the hard zone
    from behaving like a wall.
    """
    ny, nx = int(shape[-2]), int(shape[-1])
    jj = np.minimum(np.arange(ny), ny - 1 - np.arange(ny))[:, None]
    ii = np.minimum(np.arange(nx), nx - 1 - np.arange(nx))[None, :]
    d = np.minimum(jj, ii).astype(np.float64)
    w = np.zeros((ny, nx), dtype=np.float32)
    inner = hard + taper
    ramp = 0.5 * (1.0 + np.cos(np.pi * (d - hard) / max(taper, 1)))
    w[...] = np.where(d < hard, 1.0, np.where(d < inner, ramp, 0.0))
    return w


@dataclass
class FrameForcing:
    """The domain's lateral boundary, held as perimeter strips on the host.

    ``values[name][hour][k]`` is strip ``k`` of field ``name`` at forcing
    hour ``hour``; ``weights[shape][k]`` is that strip's blend weight, cached
    per trailing shape so u/v/mass each get their own.  ``frozen[name][k]``
    is the time-zero value of a carrier the analysis cannot prescribe.

    Nothing here is device memory and nothing is a full domain: at
    1200x900x49 with ``hard+taper = 44`` one hour of forced fields is
    0.38 GiB and the whole eleven-hour set is 4.2 GiB of ordinary host RAM.
    """

    width: int
    hard: int
    taper: int
    hours: tuple[int, ...]
    names: tuple[str, ...]
    values: dict[str, list[list[np.ndarray]]]
    frozen: dict[str, list[np.ndarray]]
    weights: dict[tuple, list[np.ndarray]] = _field(default_factory=dict)
    #: Steps between recomputations of the time-interpolated target.  The
    #: forcing is hourly and dt is seconds, so the target moves by ~0.3% of
    #: one interval over eight steps; the hard-zone COPY still happens every
    #: step, which is what the correctness argument needs.
    retarget_every: int = 8
    _cache: dict[str, list[np.ndarray]] = _field(default_factory=dict)
    _cache_step: int = -1

    def weight_for(self, shape) -> list[np.ndarray]:
        key = (int(shape[-2]), int(shape[-1]))
        got = self.weights.get(key)
        if got is None:
            w = frame_weight(shape, self.hard, self.taper)
            got = [np.ascontiguousarray(w[sj, si])
                   for sj, si in frame_slices(shape, self.width)]
            self.weights[key] = got
        return got

    def target(self, name: str, alpha: float, index: int) -> list[np.ndarray]:
        lo = self.values[name][index]
        hi = self.values[name][min(index + 1, len(self.values[name]) - 1)]
        out = self._cache.get(name)
        if out is None:
            out = [np.empty_like(a) for a in lo]
            self._cache[name] = out
        for dst, a, b in zip(out, lo, hi):
            np.subtract(b, a, out=dst)
            np.multiply(dst, np.float32(alpha), out=dst)
            np.add(dst, a, out=dst)
        return out

    def apply(self, store: Mapping[str, np.ndarray], elapsed: float,
              step: int) -> None:
        """Impose the frame on ``store`` for a domain clock of ``elapsed`` s."""
        span = 3600.0
        pos = max(0.0, elapsed) / span
        index = min(int(pos), len(self.hours) - 2) if len(self.hours) > 1 else 0
        alpha = float(min(max(pos - index, 0.0), 1.0))
        retarget = (step % self.retarget_every == 0)
        for name in self.names:
            dst = store.get(name)
            if dst is None:
                continue
            slices = frame_slices(dst.shape, self.width)
            weights = self.weight_for(dst.shape)
            if retarget or name not in self._cache:
                tgt = self.target(name, alpha, index)
            else:
                tgt = self._cache[name]
            device = _is_device(dst)
            for (sj, si), t, w in zip(slices, tgt, weights):
                # dst += w * (target - dst); w == 1 over the hard zone, so
                # this is a copy there and a blend across the taper.  On a
                # resident (monolithic) state the same arithmetic runs on the
                # device, with the host strip uploaded by the subtraction.
                if device:
                    import cupy as cp
                    view = dst[..., sj, si]
                    tmp = cp.asarray(t) - view
                    tmp *= cp.asarray(w)
                    dst[..., sj, si] = view + tmp
                else:
                    view = dst[..., sj, si]
                    tmp = t - view
                    tmp *= w
                    view += tmp
        for name, strips in self.frozen.items():
            dst = store.get(name)
            if dst is None:
                continue
            for (sj, si), src in zip(frame_slices(dst.shape, self.hard),
                                     strips):
                dst[..., sj, si] = src


#: Where simulated reflectivity lives once a step has computed it.
#:
#: ``refl_10cm`` is NOT a restart member, so ``physics_inventory
#: .carrier_manifest`` -- which is the restart manifest -- does not name it
#: and a streamed run therefore carries no reflectivity at all.  That is
#: correct for a checkpoint (the field is a pure diagnostic and is
#: recomputable) and wrong for this lane, whose entire output product is
#: simulated reflectivity.  :func:`carrier_inventory_with_refl` adds it, and
#: it needs nothing else: it is a plain ``(nz, ny, nx)`` mass field, so the
#: gather, the scatter and the ring geometry classify it like any other.
REFL_KEY = "scratch/refl_10cm"


def carrier_inventory_with_refl(obj, names=None) -> dict:
    """:func:`physics_inventory.carrier_inventory` plus ``refl_10cm``."""
    from tilestream import physics_inventory as physinv

    out = dict(physinv.carrier_inventory(obj, names))
    scratch = getattr(obj, "_scratch", None)
    if isinstance(scratch, dict) and scratch.get("refl_10cm") is not None:
        if names is None or REFL_KEY in names:
            out[REFL_KEY] = scratch["refl_10cm"]
    return {k: out[k] for k in sorted(out)}


def _is_device(array) -> bool:
    try:
        import cupy as cp
    except ImportError:                                   # pragma: no cover
        return False
    return isinstance(array, cp.ndarray)


def extract_frame(arrays: Mapping[str, Any], width: int
                  ) -> dict[str, list[np.ndarray]]:
    """Host copies of the ``width``-deep perimeter of every array given."""
    import cupy as cp

    out: dict[str, list[np.ndarray]] = {}
    for name, value in arrays.items():
        if value is None or not hasattr(value, "shape") or value.ndim < 2:
            continue
        strips = []
        for sj, si in frame_slices(value.shape, width):
            window = value[..., sj, si]
            host = (cp.asnumpy(window) if isinstance(window, cp.ndarray)
                    else np.asarray(window))
            strips.append(np.ascontiguousarray(host))
        out[name] = strips
    return out
