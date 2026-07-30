"""``gpuwm domain``: turn "my location + my GPU" into an experiment TOML.

The wizard emits a complete ``[experiment]``/``[projection]``/``[shared]``/
``[[domain]]`` TOML centered on ``--point``, with grid dimensions chosen so
the itemized VRAM estimate (:func:`gpuwm.core.preflight.estimate_experiment`)
times the observed 1.75 machine-peak envelope factor
(:data:`gpuwm.core.preflight.OBSERVED_PEAK_OVER_FOOTPRINT`) fits the
requested card's budget.  Nothing here is a new model of anything: the
physics/dynamics block is the product's default suite (the four-domain
reference configuration's selections with the microphysics slot on
Thompson mp8, the model-validated matched-run scheme), the projection
and nest-registration math is the existing :mod:`gpuwm.static.projection`
(lambert/mercator/polar, auto-selected from the point latitude), and the
memory arithmetic is the existing preflight estimator called in-process.

Honesty contract (terrain/static story): gpuwm builds static fields from a
locally staged NCAR WPS_GEOG tree (nine fixed dataset directories -- see
``GEOG_DATASETS``); nothing downloads it.  Forcing data for the
config-driven ``gpuwm check``/``run`` front door is decoded by the native
GRIB1 route, i.e. ERA5 today.  GFS/HRRR downloads (``gpuwm fetch``) feed
the ``rw-wps``/``gpuwm-wrf-init`` native initialization front door, which
consumes the same ``[experiment]``/``[[domain]]`` tables but not
``[case_data]``.  The wizard therefore emits ``[case_data]`` only for
``--source era5`` and prints the exact honest next step for every source
instead of pretending a pipeline exists.

Sizing conventions (all documented, none silent):

* VRAM budget = card capacity minus a flat reserve (3 GiB at 16, 4 GiB at
  24, 6 GiB at 32+) covering WDDM/driver/CUDA-context residency and the
  observed near-capacity instability ceiling of consumer cards.
* Fit criterion: ``footprint_projection_bytes * 1.75 <= budget`` -- the
  estimator's own observed peak envelope (the 1.75 was measured against
  the footprint projection, which contains the itemized alloc estimate
  plus the calibrated retention/overhead terms).  This is exactly the
  quantity ``gpuwm check`` warns about, so a wizard-emitted config
  passes ``gpuwm check`` with zero warnings, and it is deliberately
  stricter than the enforced ``estimate <= budget`` gate.
* Root time step: the certified real-data convention, 5 s per km of grid
  spacing (60 s at 12 km); children divide down the ratio chain exactly.
"""

from __future__ import annotations

import math
import os
import shutil
import tomllib
from datetime import datetime
from pathlib import Path

import numpy as np

from gpuwm.core.preflight import (GIB, OBSERVED_PEAK_OVER_FOOTPRINT,
                                  estimate_experiment,
                                  observed_peak_envelope_bytes)
from gpuwm.experiment import ExperimentConfig, build_experiment
from gpuwm.fetch import parse_cycle
from gpuwm.static.projection import (WRF_MAP_PROJ_CODES, _wrap180,
                                     projection_class)

#: Card tiers -> total VRAM (GiB).  ``--vram-gib`` accepts anything else.
CARD_VRAM_GIB = {"16gb": 16.0, "24gb": 24.0, "32gb": 32.0}

#: Nest ladders: dx chain in km (root fixed at 12 km) -> parent grid/time
#: ratios.  Ratios follow the certified 12->3->1 chain (4, 3); the 500 m
#: extension refines the 1 km nest by 2.
LADDER_RATIOS = {
    "12": (),
    "12-3": (4,),
    "12-3-1": (4, 3),
    "12-3-1-0.5": (4, 3, 2),
}
_LADDERS_DEEPEST_FIRST = ("12-3-1-0.5", "12-3-1", "12-3")

ROOT_DX_M = 12000.0
#: Certified real-data clock convention: 60 s at 12 km = 5 s per km.
ROOT_TIME_STEP_S = 60

#: Parent rows a child boundary must clear: spec_bdy_width + blend_width
#: (both 5, the emitted [experiment] values; gpuwm/experiment.py enforces).
_CLEARANCE_ROWS = 10

#: Child linear extent as a fraction of the parent's, by nest depth --
#: the certified four-domain layout's proportions, then quantized.
_CHILD_SPAN_FRACTION = (0.5, 0.36, 0.4)

#: diff_6th_factor by nest depth (certified ladder).
_DIFF6_FACTORS = (0.12, 0.10, 0.08, 0.06)


def vram_reserve_gib(vram_gib: float) -> float:
    """Flat VRAM reserve (GiB) by card capacity: WDDM/driver/CUDA context
    plus the near-capacity stability ceiling of consumer cards."""
    if vram_gib <= 16.0:
        return 3.0
    if vram_gib <= 24.0:
        return 4.0
    return 6.0


#: Projection auto-selection bands (absolute latitude of --point):
#: below MERCATOR_MAX_LAT the Lambert cone is ill-conditioned and the
#: wizard selects Mercator; above LAMBERT_MAX_LAT it selects polar
#: stereographic; between them, hemisphere-correct Lambert conformal.
#: --projection overrides the choice explicitly.  All three projections
#: are oracle-gated against the pinned WRF v4.6.1 module_llxy.F
#: (tests/test_projection_oracle.py).
MERCATOR_MAX_LAT, LAMBERT_MAX_LAT = 25.0, 60.0
#: Cells of clearance between the domain footprint and the projection
#: pole; a domain containing (or touching) the pole is refused -- the
#: lat-lon source interpolation and static-tile windowing are not
#: pole-capable (genuine limit, not a projection-math one).
_POLE_CLEARANCE_CELLS = 2.0

#: Degrees of forcing margin beyond the root domain for the fetch hint.
#: ERA5 needs only interpolation-halo coverage; HRRR files are CONUS-wide
#: (a larger request could leave HRRR's own coverage box and be refused
#: by fetch), so both keep the small margin.  GFS uses
#: :func:`gpuwm.fetch.gfs_suggested_fetch_margin_deg`: the GFS front
#: door's donor-coverage proof must find every model lake's nearest
#: source-water donor INSIDE the crop, so the wizard's suggested area
#: carries that documented margin instead of being rejected downstream.
_FETCH_MARGIN_DEG = 2.0


def _fetch_margin_deg(source: str) -> float:
    if source == "gfs":
        from gpuwm.fetch import gfs_suggested_fetch_margin_deg
        return gfs_suggested_fetch_margin_deg()
    return _FETCH_MARGIN_DEG

#: Forcing cadence per source: estimator LBC-interval sizing + fetch hint.
SOURCE_FORCING_INTERVAL_S = {"era5": 21600.0, "gfs": 10800.0,
                             "hrrr": 3600.0}
_SOURCE_CADENCE_H = {"era5": 6, "gfs": 3}

#: Grid-scale search bounds.  _MIN_SCALE puts the root at 60 x 48 mass
#: points, the smallest layout that still hosts the deepest ladder with
#: full Davies/blend clearance.
_MIN_SCALE, _MAX_SCALE = 0.55, 8.0

#: The nine WPS_GEOG dataset directories the static builder opens (the
#: ``default`` geog_data_res selector; gpuwm/static/build.py).
GEOG_DATASETS = (
    "topo_gmted2010_30s", "modis_landuse_20class_30s_with_lakes",
    "soiltype_top_30s", "soiltype_bot_30s", "greenfrac_fpar_modis",
    "lai_modis_10m", "albedo_modis", "maxsnowalb_modis", "soiltemp_1deg",
)

#: Certified 49-mass-level vertical coordinate (50 full eta levels,
#: p_top 100 hPa) -- the reference configuration's ladder.
_ETA_LEVELS = (
    1.0, 0.9978, 0.99519, 0.99212, 0.98849,
    0.98422, 0.97918, 0.97325, 0.96627, 0.95808,
    0.94846, 0.93719, 0.92402, 0.90866, 0.89079,
    0.87006, 0.84612, 0.81857, 0.78706, 0.75124,
    0.7108, 0.66556, 0.61547, 0.56067, 0.50519,
    0.45474, 0.40886, 0.36713, 0.32918, 0.29466,
    0.26328, 0.23473, 0.20877, 0.18516, 0.16369,
    0.14417, 0.12641, 0.11026, 0.09557, 0.08222,
    0.07007, 0.05902, 0.04898, 0.03984, 0.03153,
    0.02398, 0.0171, 0.01085, 0.00517, 0.0,
)

_PACKAGED_VTABLE = Path(__file__).parent / "data" / "vtables" / \
    "Vtable.ERA5_CDO"

#: The product's default physics/dynamics selections -- the reference
#: four-domain configuration's [shared] block with the microphysics slot
#: on Thompson (mp_physics 8: the model-validated matched-run scheme,
#: WRF's own tables packaged and hash-pinned): the MM5 surface layer
#: (91), Noah LSM (2), YSU PBL (1), RTE+RRTMGP radiation (4, the
#: ratified WRF-RRTMG 4/4 substitution), and the certified
#: diffusion/damping/acoustic settings.  Morrison (10) stays fully
#: selectable at its registry maturity label; its morr_rimed_ice knob is
#: Morrison-only and is deliberately absent here.
_SHARED_CERTIFIED = {
    "nz": 49, "ztop": 20000.0, "p_top": 10000.0,
    "eta_levels": _ETA_LEVELS,
    "hybrid_opt": 2, "etac": 0.2, "base_temp": 290.0,
    "time_step_sound": 4, "epssm": 0.5, "emdiv": 0.01,
    "hypsometric_opt": 2, "h_sca_adv_order": 5, "smdiv": 0.1,
    "top_lid": False, "moist": True, "moist_cq": True,
    "mp_physics": 8, "moist_adv_opt": 1,
    "km_opt": 4, "diff_6th_opt": 2, "diff_6th_slopeopt": 1,
    "w_damping": 1, "damp_opt": 3, "zdamp": 5000.0, "dampcoef": 0.2,
    "khdif": 0.0, "kvdif": 0.0, "spec_zone": 1, "relax_zone": 4,
    "terrain_opt": 1,
    "sf_sfclay_physics": 91, "sf_surface_physics": 2,
    "bl_pbl_physics": 1, "ra_physics": 4,
    "wrf_rrtmg_compatibility": "wrf-rrtmg-4-4-to-rte-rrtmgp-v1",
    "bldt": 0.0,
    # WRF nwp_diagnostics: wizard runs are convective forecasts whose
    # audience reads UH products, so the UP_HELI_MAX running-max
    # diagnostic ships ON (trajectory-inert; tests/test_uh_lifecycle.py).
    "nwp_diagnostics": 1,
}


class DomainFitError(ValueError):
    """The requested ladder cannot fit the requested card."""


def _parse_point(raw: str) -> tuple[float, float]:
    parts = raw.split(",")
    if len(parts) != 2:
        raise ValueError("--point must be lat,lon in decimal degrees")
    try:
        lat, lon = (float(part) for part in parts)
    except ValueError as error:
        raise ValueError(
            "--point must be lat,lon in decimal degrees") from error
    if not (math.isfinite(lat) and math.isfinite(lon)):
        raise ValueError("--point coordinates must be finite")
    if not -90.0 <= lat <= 90.0:
        raise ValueError(
            f"--point latitude {lat:g} must lie within [-90, 90]")
    if abs(lat) == 90.0:
        raise ValueError(
            f"--point latitude {lat:g} is the pole itself; a domain "
            "containing the pole is unsupported (lat-lon source "
            "interpolation and static-tile windowing are not "
            "pole-capable) -- move --point off the pole")
    if not -180.0 <= lon <= 180.0:
        raise ValueError(
            f"--point longitude {lon:g} must lie within [-180, 180]")
    return lat, lon


def _even(value: float) -> int:
    return max(2, 2 * round(value / 2.0))


def _dims_for_scale(scale: float, ratios: tuple[int, ...]
                    ) -> list[tuple[int, int]]:
    """Mass dimensions per domain at ``scale`` (root 110 x 88 at 1.0).

    Everything even so centered children register exactly; child mass
    dimensions are span * ratio, satisfying the loader's divisibility and
    clearance rules by construction (still re-validated by the loader).
    """
    dims = [(_even(110.0 * scale), _even(88.0 * scale))]
    for depth, ratio in enumerate(ratios):
        pnx, pny = dims[-1]
        spans = []
        for parent_extent in (pnx, pny):
            span = _even(_CHILD_SPAN_FRACTION[depth] * parent_extent)
            span = min(span, parent_extent - 2 * _CLEARANCE_ROWS)
            if span < 12:
                raise DomainFitError(
                    f"parent extent {parent_extent} cannot host a nest "
                    f"with {_CLEARANCE_ROWS}-row clearance at scale "
                    f"{scale:g}")
            spans.append(span)
        dims.append((spans[0] * ratio, spans[1] * ratio))
    return dims


def _radt_minutes(dx_m: float) -> float:
    return max(1.0, dx_m / 1000.0)


def _domain_tables(dims: list[tuple[int, int]],
                   ratios: tuple[int, ...]) -> list[dict]:
    """[[domain]] table dicts (centered children, certified cadences)."""
    tables = []
    dx = ROOT_DX_M
    for index, (nx, ny) in enumerate(dims):
        if index == 0:
            table = {
                "grid_id": 1, "parent_id": 0, "i_parent_start": 1,
                "j_parent_start": 1, "parent_grid_ratio": 1,
                "parent_time_step_ratio": 1, "nx": nx, "ny": ny,
                "time_step": ROOT_TIME_STEP_S, "dx": ROOT_DX_M,
                "specified": True, "nested": False,
                "history_interval_s": 3600.0,
                "radt": _radt_minutes(dx), "cu_physics": 1,
                "cudt_minutes": 5.0,
                "diff_6th_factor": _DIFF6_FACTORS[0],
            }
        else:
            ratio = ratios[index - 1]
            pnx, pny = dims[index - 1]
            dx = dx / ratio
            table = {
                "grid_id": index + 1, "parent_id": index,
                "i_parent_start": (pnx - nx // ratio) // 2 + 1,
                "j_parent_start": (pny - ny // ratio) // 2 + 1,
                "parent_grid_ratio": ratio,
                "parent_time_step_ratio": ratio, "nx": nx, "ny": ny,
                "specified": False, "nested": True,
                "history_interval_s": 900.0, "epssm": 0.1,
                "radt": _radt_minutes(dx), "cu_physics": 0,
                "diff_6th_factor": _DIFF6_FACTORS[index],
            }
        tables.append(table)
    return tables


def _toml_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%dT%H:%M:%S")
    if isinstance(value, (list, tuple)):
        if len(value) > 5:  # long numeric arrays: 5 per line
            lines = ["["]
            for start in range(0, len(value), 5):
                chunk = ", ".join(
                    _toml_value(v) for v in value[start:start + 5])
                lines.append(f"    {chunk},")
            lines.append("]")
            return "\n".join(lines)
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    return '"' + str(value).replace("\\", "/") + '"'


def _render_table(name: str, entries: dict, array_of_tables: bool = False,
                  comment: str | None = None) -> str:
    header = f"[[{name}]]" if array_of_tables else f"[{name}]"
    lines = ([f"# {comment}"] if comment else []) + [header]
    for key, value in entries.items():
        lines.append(f"{key} = {_toml_value(value)}")
    return "\n".join(lines) + "\n"


def _ladder_dx_km(ratios: tuple[int, ...]) -> list[float]:
    chain = [ROOT_DX_M / 1000.0]
    for ratio in ratios:
        chain.append(chain[-1] / ratio)
    return chain


def auto_projection(lat: float) -> str:
    """The auto-selected WPS projection for a point latitude."""
    if abs(lat) < MERCATOR_MAX_LAT:
        return "mercator"
    if abs(lat) <= LAMBERT_MAX_LAT:
        return "lambert"
    return "polar"


def _projection_entries(lat: float, lon: float,
                        choice: str = "auto") -> dict:
    """[projection] table for the point.

    ``auto`` selects by |lat| (:func:`auto_projection`).  Lambert:
    hemisphere-correct secant cone bracketing the point, truelats at
    |lat| +/- 10 deg clamped to [15, 70] and signed with the point's
    hemisphere.  Mercator: true at the point's latitude (stand_lon is
    recorded but does not enter Mercator math, module_llxy semantics).
    Polar stereographic: the point's hemisphere pole, scale true at the
    point's latitude, stand_lon at the point.
    """
    map_proj = auto_projection(lat) if choice == "auto" else choice
    if map_proj == "lambert":
        sign = -1.0 if lat < 0.0 else 1.0
        alat = abs(lat)
        return {
            "map_proj": "lambert", "ref_lat": lat, "ref_lon": lon,
            "truelat1": sign * round(max(15.0, alat - 10.0), 2),
            "truelat2": sign * round(min(70.0, alat + 10.0), 2),
            "stand_lon": lon,
        }
    if map_proj in ("mercator", "polar"):
        return {
            "map_proj": map_proj, "ref_lat": lat, "ref_lon": lon,
            "truelat1": round(lat, 2), "truelat2": round(lat, 2),
            "stand_lon": lon,
        }
    raise ValueError(
        f"--projection {map_proj!r} is not implemented (choices: auto, "
        "lambert, mercator, polar)")


def _root_grid(projection: dict, nx: int, ny: int):
    cls = projection_class(projection["map_proj"])
    return cls(
        ref_lat=projection["ref_lat"], ref_lon=projection["ref_lon"],
        truelat1=projection["truelat1"], truelat2=projection["truelat2"],
        stand_lon=projection["stand_lon"], dx=ROOT_DX_M, dy=ROOT_DX_M,
        e_we=nx + 1, e_sn=ny + 1)


def _pole_clearance_refusal(projection: dict, nx: int, ny: int) -> None:
    """Refuse a root footprint that contains (or nearly touches) the
    projection pole -- a genuine pipeline limit (lat-lon source
    interpolation and static windowing are not pole-capable), not a
    projection-math one.  Mercator never reaches a pole."""
    if projection["map_proj"] == "mercator":
        return
    grid = _root_grid(projection, nx, ny)
    pole_lat = 90.0 if projection["truelat1"] >= 0.0 else -90.0
    px, py = (float(v) for v in grid.latlon_to_ij(
        pole_lat, projection["stand_lon"]))
    margin = _POLE_CLEARANCE_CELLS
    if (0.5 - margin <= px <= grid.e_we - 0.5 + margin
            and 0.5 - margin <= py <= grid.e_sn - 0.5 + margin):
        raise ValueError(
            f"the fitted root domain ({nx} x {ny} mass points at "
            f"{ROOT_DX_M / 1000:g} km) contains or touches the "
            f"{'north' if pole_lat > 0 else 'south'} pole; lat-lon "
            "source interpolation and static-tile windowing are not "
            "pole-capable -- move --point away from the pole or choose "
            "a smaller layout (--vram-gib / a shallower --ladder)")


def _fetch_area(projection: dict, nx: int, ny: int,
                margin_deg: float = _FETCH_MARGIN_DEG
                ) -> tuple[float, float, float, float]:
    """Forcing bbox (S, W, N, E) = root corners + margin, worldwide.

    Corner longitudes are unwrapped onto the branch nearest the
    reference longitude, so a footprint straddling the antimeridian
    yields a continuous span; the emitted box then wraps back into the
    signed convention, producing W > E for a crossing box (the
    ``gpuwm fetch`` contract; NOMADS and CDS both consume it).  The
    latitude edges clamp to [-90, 90].  A footprint wider than 180
    degrees of longitude is refused (genuine limit: no source crop can
    serve it as one box).  The refusal is checked on the MARGINED span
    -- the margin is part of the emitted box, and a box whose margined
    width exceeds 180 degrees would be read back by
    :func:`gpuwm.fetch.parse_area` as the complementary
    antimeridian-crossing box (the wrong crop, silently)."""
    lat_c, lon_c = _root_grid(projection, nx, ny).latlon_c()
    center = float(projection["ref_lon"])
    lon_u = center + np.asarray(
        _wrap180(np.asarray(lon_c, dtype=float) - center))
    span = ((float(lon_u.max()) + margin_deg)
            - (float(lon_u.min()) - margin_deg))
    if span > 180.0:
        raise ValueError(
            f"the root domain's forcing footprint spans {span:.1f} "
            "degrees of longitude; boxes wider than 180 degrees cannot "
            "be served as a single source crop -- shrink the "
            "configuration or move --point equatorward")
    lat_s = max(-90.0, float(lat_c.min()) - margin_deg)
    lat_n = min(90.0, float(lat_c.max()) + margin_deg)
    lon_w = float(_wrap180(float(lon_u.min()) - margin_deg))
    lon_e = float(_wrap180(float(lon_u.max()) + margin_deg))
    if lon_e == -180.0:
        lon_e = 180.0
    return lat_s, lon_w, lat_n, lon_e


def _posix(path) -> str:
    return str(path).replace("\\", "/")


def _relative_or_absolute(path: Path, base: Path) -> str:
    try:
        return _posix(os.path.relpath(path, base))
    except ValueError:  # different drive on Windows
        return _posix(Path(path).resolve())


def render_config(*, name: str, start_time: datetime, hours: int,
                  projection: dict, dims: list[tuple[int, int]],
                  ratios: tuple[int, ...],
                  fetch_hints: dict, case_data: dict | None) -> str:
    """The emitted TOML text (the exact bytes the wizard validates)."""
    experiment = {
        "name": name, "start_time": start_time,
        "run_seconds": float(hours * 3600), "feedback": 0,
        "smooth_option": 0, "blend_width": 5, "spec_bdy_width": 5,
        # Single-domain emissions disable restart writing: the portable
        # prepared-forecast contract requires restart_interval_s = 0.
        "restart_interval_s": 0.0 if not ratios else 3600.0,
    }
    shared = dict(_SHARED_CERTIFIED)
    shared["map_proj"] = WRF_MAP_PROJ_CODES[projection["map_proj"]]
    parts = [
        "# Emitted by `gpuwm domain` -- point "
        f"{projection['ref_lat']:g},{projection['ref_lon']:g}, ladder "
        f"{'-'.join(f'{v:g}' for v in _ladder_dx_km(ratios))} km.\n"
        "# Physics/dynamics are the product's default suite (Thompson mp8 "
        "in the certified\n"
        "# real-data set; Morrison mp10 stays selectable); child dx/dt "
        "derive\n"
        "# exactly from the parent chain and are never hand-typed "
        "(gpuwm/experiment.py).\n",
        _render_table("experiment", experiment),
        _render_table("projection", projection),
        _render_table("shared", shared),
    ]
    for table in _domain_tables(dims, ratios):
        parts.append(_render_table("domain", table, array_of_tables=True))
    parts.append(_render_table(
        "fetch", fetch_hints,
        comment="Advisory data-acquisition hints (validated, not "
                "executed); keys mirror `gpuwm fetch` flags."))
    if case_data is not None:
        parts.append(_render_table(
            "case_data", case_data,
            comment="Declared inputs for the config-driven "
                    "check/static/ingest/run front door (ERA5 native-GRIB1 "
                    "route; era5_z_invariant source orography)."))
    return "\n".join(parts)


def experiment_from_text(text: str, *, source: str) -> ExperimentConfig:
    """Round-trip emitted TEXT through the real loaders (advisory [fetch]
    and [case_data] are split off exactly as the CLI loaders do)."""
    raw = tomllib.loads(text)
    fetch_table = raw.pop("fetch", None)
    if fetch_table is not None:
        from gpuwm.fetch import validate_fetch_hints
        validate_fetch_hints(fetch_table, source=source)
    raw.pop("case_data", None)
    return build_experiment(raw, source=source)


def fit_ladder(*, ladder: str, budget_bytes: int, hours: int,
               start_time: datetime, projection: dict, source: str,
               name: str) -> tuple[list[tuple[int, int]], ExperimentConfig]:
    """Largest centered layout whose estimate x 1.75 fits the budget.

    Bisects a continuous scale factor; every candidate is validated by the
    real experiment loader (clearance, cadence, ratio rules) and priced by
    the real estimator -- the wizard owns no memory arithmetic of its own.
    """
    ratios = LADDER_RATIOS[ladder]
    interval = SOURCE_FORCING_INTERVAL_S[source]

    def candidate(scale: float):
        dims = _dims_for_scale(scale, ratios)
        text = render_config(
            name=name, start_time=start_time, hours=hours,
            projection=projection, dims=dims, ratios=ratios,
            fetch_hints={"source": source}, case_data=None)
        exp = experiment_from_text(text, source=f"<candidate {ladder}>")
        estimate = estimate_experiment(
            exp, forcing_interval_seconds=interval)
        envelope = observed_peak_envelope_bytes(
            estimate.footprint_projection_bytes)
        return dims, exp, envelope

    dims, exp, envelope = candidate(_MIN_SCALE)
    if envelope > budget_bytes:
        raise DomainFitError(
            f"ladder {ladder} does not fit a {budget_bytes / GIB:.1f} GiB "
            f"budget even at the minimum layout ({dims[0][0]}x{dims[0][1]} "
            f"root): footprint projection x "
            f"{OBSERVED_PEAK_OVER_FOOTPRINT:.2f} = "
            f"{envelope / GIB:.2f} GiB; choose a shallower --ladder or a "
            "larger card")
    lo, hi = _MIN_SCALE, _MAX_SCALE
    best = (dims, exp)
    for _ in range(36):
        mid = 0.5 * (lo + hi)
        try:
            dims, exp, envelope = candidate(mid)
        except DomainFitError:
            hi = mid
            continue
        if envelope <= budget_bytes:
            best = (dims, exp)
            lo = mid
        else:
            hi = mid
    return best


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", newline="\n", encoding="utf-8") as stream:
        stream.write(text)
    os.replace(temporary, path)


def render_wps_namelist(projection: dict, dims: list[tuple[int, int]],
                        ratios: tuple[int, ...]) -> str:
    """Minimal namelist.wps matching the TOML bit-for-bit.

    The config-driven pipeline reads only geog_data_res/max_dom from it,
    but the native-WRF contract checker cross-checks every projection and
    layout key against the [projection]/[[domain]] tables, so the emitted
    pair must agree exactly.
    """
    tables = _domain_tables(dims, ratios)

    def csv(values):
        return ", ".join(str(v) for v in values) + ","

    return (
        "&share\n"
        " wrf_core = 'ARW',\n"
        f" max_dom = {len(tables)},\n"
        " io_form_geogrid = 2,\n"
        "/\n"
        "&geogrid\n"
        f" parent_id         = {csv([1] + [t['parent_id'] for t in tables[1:]])}\n"
        f" parent_grid_ratio = {csv(t['parent_grid_ratio'] for t in tables)}\n"
        f" i_parent_start    = {csv(t['i_parent_start'] for t in tables)}\n"
        f" j_parent_start    = {csv(t['j_parent_start'] for t in tables)}\n"
        f" e_we              = {csv(t['nx'] + 1 for t in tables)}\n"
        f" e_sn              = {csv(t['ny'] + 1 for t in tables)}\n"
        f" geog_data_res     = {csv(chr(39) + 'default' + chr(39) for _ in tables)}\n"
        f" dx = {ROOT_DX_M:g},\n"
        f" dy = {ROOT_DX_M:g},\n"
        f" map_proj = '{projection['map_proj']}',\n"
        f" ref_lat   = {projection['ref_lat']!r},\n"
        f" ref_lon   = {projection['ref_lon']!r},\n"
        f" truelat1  = {projection['truelat1']!r},\n"
        f" truelat2  = {projection['truelat2']!r},\n"
        f" stand_lon = {projection['stand_lon']!r},\n"
        "/\n")


def _default_name(lat: float, lon: float) -> str:
    ns = "n" if lat >= 0 else "s"
    ew = "e" if lon >= 0 else "w"
    return (f"area_{abs(lat):.2f}{ns}_{abs(lon):.2f}{ew}"
            .replace(".", "p"))


def _print_sizing_table(exp: ExperimentConfig, estimate,
                        budget_bytes: int, vram_gib: float) -> None:
    envelope = observed_peak_envelope_bytes(
        estimate.footprint_projection_bytes)
    print("sizing (itemized preflight estimator, in-process):")
    print("  domain    dx        mass grid      dt         resident")
    for dc, dom in zip(exp.domains, estimate.domains):
        dx_km = exp.dx_exact(dc.grid_id) / 1000
        dt = exp.dt_exact(dc.grid_id)
        dt_text = (f"{int(dt)} s" if dt.denominator == 1
                   else f"{dt.numerator}/{dt.denominator} s")
        print(f"  d{dc.grid_id:02d}     {float(dx_km):6.3f} km  "
              f"{dc.run.nx:4d} x {dc.run.ny:<4d}   {dt_text:>8}   "
              f"{dom.resident_bytes / GIB:6.2f} GiB")
    print(f"  itemized alloc estimate "
          f"{estimate.alloc_estimate_bytes / GIB:.2f} GiB; footprint "
          f"projection {estimate.footprint_projection_bytes / GIB:.2f} "
          f"GiB  x {OBSERVED_PEAK_OVER_FOOTPRINT:.2f} observed peak "
          f"envelope = {envelope / GIB:.2f} GiB")
    print(f"  budget {budget_bytes / GIB:.2f} GiB "
          f"({vram_gib:g} GiB card - {vram_reserve_gib(vram_gib):g} GiB "
          f"reserve); headroom {(budget_bytes - envelope) / GIB:.2f} GiB")


def _missing_case_inputs(out: Path, case_data: dict) -> list[str]:
    from gpuwm.case_data import expand_path_variables
    base = out.parent
    missing = []

    def resolve(raw: str) -> Path:
        expanded = Path(expand_path_variables(raw, "wizard", str(out)))
        return expanded if expanded.is_absolute() else base / expanded

    for raw in case_data["forcing"]:
        if not resolve(raw).is_file():
            missing.append(f"forcing {raw}")
    for key in ("vtable", "wps_namelist"):
        if not resolve(case_data[key]).is_file():
            missing.append(f"{key} {case_data[key]}")
    root = resolve(case_data["geog_root"])
    if not root.is_dir():
        missing.append(f"geog_root {case_data['geog_root']}")
    else:
        for dataset in GEOG_DATASETS:
            if not (root / dataset).is_dir():
                missing.append(f"geog_root dataset {dataset}")
    return missing


def _print_geog_help() -> None:
    print("  static geography: gpuwm reads a locally staged NCAR WPS_GEOG "
          "tree (no downloader exists).  The geog_root directory must "
          "contain these dataset directories from the standard "
          "high-resolution mandatory-fields download:")
    print("    " + ", ".join(GEOG_DATASETS))


def domain_main(args) -> int:
    lat, lon = _parse_point(args.point)
    if args.card is not None and args.vram_gib is not None:
        raise ValueError("--card and --vram-gib are mutually exclusive")
    vram_gib = (CARD_VRAM_GIB[args.card] if args.card is not None
                else args.vram_gib if args.vram_gib is not None
                else CARD_VRAM_GIB["24gb"])
    if not math.isfinite(vram_gib) \
            or vram_gib <= vram_reserve_gib(vram_gib):
        raise ValueError(
            f"--vram-gib {vram_gib:g} leaves no budget after the "
            f"{vram_reserve_gib(vram_gib):g} GiB reserve")
    budget_gib = vram_gib - vram_reserve_gib(vram_gib)
    budget = int(budget_gib * GIB)
    if args.hours < 1:
        raise ValueError("--hours must be at least 1")
    start_time = parse_cycle(args.cycle, args.source)
    name = args.name or _default_name(lat, lon)
    projection = _projection_entries(
        lat, lon, getattr(args, 'projection', 'auto'))
    out: Path = args.out

    ladders = ([args.ladder] if args.ladder != "auto"
               else list(_LADDERS_DEEPEST_FIRST))
    chosen = None
    for ladder in ladders:
        try:
            dims, _ = fit_ladder(
                ladder=ladder, budget_bytes=budget, hours=args.hours,
                start_time=start_time, projection=projection,
                source=args.source, name=name)
        except DomainFitError as error:
            if args.ladder != "auto":
                raise
            print(f"ladder {ladder}: {error}")
            continue
        chosen = (ladder, dims)
        break
    if chosen is None:
        raise DomainFitError(
            "no ladder fits the requested card; even the shallowest "
            f"ladder's smallest layout exceeds the {budget_gib:.1f} GiB "
            "budget")
    ladder, dims = chosen
    ratios = LADDER_RATIOS[ladder]
    # Genuine-limit refusal first (its message names the real problem;
    # a pole-containing footprint would otherwise also trip the
    # 180-degree fetch-span refusal below with a less useful message).
    _pole_clearance_refusal(projection, *dims[0])

    # Fetch hints from the fitted root footprint.  The default data
    # directory lives beside the emitted TOML so the declared forcing
    # paths stay short and the config directory stays relocatable.
    area = _fetch_area(projection, *dims[0],
                       margin_deg=_fetch_margin_deg(args.source))
    cadence = _SOURCE_CADENCE_H.get(args.source)
    data_dir = (Path(args.data_dir) if args.data_dir
                else out.parent / "data" / name)
    fetch_hints = {
        "source": args.source, "cycle": args.cycle,
        "hours": (args.hours if cadence is None else
                  max(cadence, math.ceil(args.hours / cadence) * cadence)),
        "area": ",".join(f"{v:.2f}" for v in area),
        "out": _relative_or_absolute(data_dir, Path.cwd()),
    }
    if cadence is not None:
        fetch_hints["cadence"] = cadence

    # [case_data] only where the config-driven front door can honestly
    # consume the fetched data (the native GRIB1 = ERA5 route).
    case_data = None
    vtable_path = None
    if args.source == "era5":
        if args.vtable is not None:
            vtable_path = Path(args.vtable)
            vtable_text = _posix(vtable_path.resolve())
        else:
            vtable_path = out.parent / _PACKAGED_VTABLE.name
            vtable_text = _PACKAGED_VTABLE.name
        forcing = ([_posix(Path(p).resolve()) for p in args.forcing]
                   if args.forcing
                   else [_relative_or_absolute(
                       data_dir / "era5-combined.grib", out.parent)])
        case_data = {
            "forcing": forcing,
            "vtable": vtable_text,
            "forcing_interval_s": SOURCE_FORCING_INTERVAL_S["era5"],
            "wps_namelist": f"{out.stem}.namelist.wps",
            "geog_root": (_posix(Path(args.geog_root).resolve())
                          if args.geog_root
                          else "${GPUWM_CASE_DATA_ROOT}/WPS_GEOG"),
            "sfcp_to_sfcp": True,
            "output_domain": 1,
            "output_title": f"gpuwm {name}",
        }

    text = render_config(
        name=name, start_time=start_time, hours=args.hours,
        projection=projection, dims=dims, ratios=ratios,
        fetch_hints=fetch_hints, case_data=case_data)
    # Round-trip the exact bytes through the real loader before writing.
    exp = experiment_from_text(text, source=str(out))
    estimate = estimate_experiment(
        exp,
        forcing_interval_seconds=SOURCE_FORCING_INTERVAL_S[args.source])
    envelope = observed_peak_envelope_bytes(
        estimate.footprint_projection_bytes)
    if envelope > budget:
        raise DomainFitError(
            "internal fit regression: emitted config's envelope "
            f"{envelope / GIB:.2f} GiB exceeds the budget "
            f"{budget_gib:.2f} GiB")

    _write_atomic(out, text)
    wps_path = out.parent / f"{out.stem}.namelist.wps"
    _write_atomic(wps_path, render_wps_namelist(projection, dims, ratios))
    written = [out, wps_path]
    if args.source == "era5" and args.vtable is None:
        if vtable_path.exists():
            if vtable_path.read_bytes() != _PACKAGED_VTABLE.read_bytes():
                raise ValueError(
                    f"{vtable_path} exists and differs from the packaged "
                    "Vtable.ERA5_CDO; refusing to overwrite")
        else:
            shutil.copyfile(_PACKAGED_VTABLE, vtable_path)
            written.append(vtable_path)

    print(f"gpuwm domain: {name!r} at ({lat:g}, {lon:g}), ladder {ladder} "
          f"({'-'.join(f'{v:g}' for v in _ladder_dx_km(ratios))} km), "
          f"card {vram_gib:g} GiB")
    _print_sizing_table(exp, estimate, budget, vram_gib)
    for path in written:
        print(f"wrote {path}")
    print("next: gpuwm fetch "
          f"--source {args.source} --cycle {args.cycle} "
          f"--hours {fetch_hints['hours']} --area {fetch_hints['area']} "
          f"--out {fetch_hints['out']}")

    # ---- final step: gpuwm check, honestly ---------------------------
    if case_data is None:
        print(
            f"note: fetched {args.source.upper()} GRIB2 feeds the "
            "rw-wps/gpuwm-wrf-init native initialization front door, not "
            "the [case_data] run path (the config-driven route decodes "
            "native GRIB1 = ERA5 today); this TOML's "
            "[experiment]/[[domain]]/[projection] tables are what that "
            "front door consumes.  `gpuwm check` validates the geometry "
            "and memory preflight for this file; the native front door "
            "validates its own inputs.")
        from gpuwm.cli import main as cli_main
        rc = cli_main(["check", str(out), "--budget-gib",
                       f"{budget_gib:g}"])
        if rc != 0:
            print(f"gpuwm check FAILED (rc {rc}) on the emitted config; "
                  "the wizard does not certify this file", flush=True)
            return rc
        print("gpuwm check: PASS (rc 0)")
        return 0
    missing = _missing_case_inputs(out, case_data)
    if missing:
        print("gpuwm check: deferred -- declared inputs not on disk yet:")
        for item in missing:
            print(f"  missing {item}")
        print(f"  after fetching, run: gpuwm check {_posix(out)} "
              f"--budget-gib {budget_gib:g}")
        _print_geog_help()
        return 0
    from gpuwm.cli import main as cli_main
    rc = cli_main(["check", str(out), "--budget-gib", f"{budget_gib:g}"])
    if rc != 0:
        print(f"gpuwm check FAILED (rc {rc}) on the emitted config; the "
              "wizard does not certify this file", flush=True)
        return rc
    print("gpuwm check: PASS (rc 0)")
    return 0


def register_cli(subparsers) -> None:
    parser = subparsers.add_parser(
        "domain",
        help="wizard: emit an experiment TOML for a point + GPU budget, "
             "sized by the in-process VRAM estimator")
    parser.add_argument("--point", required=True, metavar="LAT,LON",
                        help="domain center in decimal degrees, any land "
                             "point on earth; the projection is "
                             "auto-selected from |lat| (<25 Mercator, "
                             "25-60 Lambert conformal, >60 polar "
                             "stereographic) unless --projection is set")
    parser.add_argument("--projection", default="auto",
                        choices=("auto", "lambert", "mercator", "polar"),
                        help="map projection override (default: auto by "
                             "point latitude; all three are oracle-gated "
                             "against WRF v4.6.1 module_llxy)")
    parser.add_argument("--name", default=None,
                        help="experiment name (default derived from the "
                             "point)")
    parser.add_argument("--card", choices=sorted(CARD_VRAM_GIB),
                        default=None,
                        help="GPU tier; sets the VRAM budget (default "
                             "24gb when --vram-gib is absent)")
    parser.add_argument("--vram-gib", type=float, default=None,
                        metavar="N",
                        help="total VRAM in GiB (alternative to --card)")
    parser.add_argument("--ladder", default="auto",
                        choices=(*LADDER_RATIOS, "auto"),
                        help="nest dx chain in km; auto picks the deepest "
                             "ladder that fits the card")
    parser.add_argument("--hours", type=int, default=6, metavar="N",
                        help="forecast length (run_seconds = N*3600)")
    parser.add_argument("--source", default="era5",
                        choices=("gfs", "hrrr", "era5"),
                        help="forcing source for the [fetch] hints and "
                             "(era5) the [case_data] declarations")
    parser.add_argument("--cycle", required=True,
                        metavar="YYYY-MM-DDTHH",
                        help="start time (UTC) = the forcing cycle")
    parser.add_argument("--out", type=Path, required=True, metavar="TOML",
                        help="emitted experiment TOML path")
    parser.add_argument("--data-dir", default=None, metavar="DIR",
                        help="where fetched forcing lives/will live "
                             "(default data/<name>)")
    parser.add_argument("--forcing", nargs="+", default=None,
                        metavar="GRIB",
                        help="era5: explicit forcing GRIB path(s) already "
                             "on disk (default <data-dir>/"
                             "era5-combined.grib)")
    parser.add_argument("--vtable", type=Path, default=None,
                        help="era5: Vtable override (default: the "
                             "packaged Vtable.ERA5_CDO, copied beside "
                             "the TOML)")
    parser.add_argument("--geog-root", type=Path, default=None,
                        metavar="DIR",
                        help="staged WPS_GEOG tree (default "
                             "${GPUWM_CASE_DATA_ROOT}/WPS_GEOG)")
    parser.set_defaults(func=domain_main)
    return parser


__all__ = [
    "CARD_VRAM_GIB", "DomainFitError", "GEOG_DATASETS", "LADDER_RATIOS",
    "ROOT_DX_M", "ROOT_TIME_STEP_S", "domain_main", "experiment_from_text",
    "fit_ladder", "register_cli", "render_config", "render_wps_namelist",
    "vram_reserve_gib",
]
