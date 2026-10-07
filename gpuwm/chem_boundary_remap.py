"""Carry a 3-D composition source onto the model grid: decode, convert, remap.

A BOUNDARY SOURCE (a ``gpuwm/data/chem/sources/*.json`` row of kind
``boundary`` or ``oxidant`` whose ``vertical.kind`` is
``hybrid_sigma_pressure``) supplies species on its own grid and levels at its
own cadence.  This module turns the source's decoded frames into model-grid
``(nz, ny, nx)`` float32 fields for every active species row whose
``boundary`` list names that source, at one valid time and onto one vertical
target.  It is the chem counterpart of the WIF aerosol climatology ingest
(:mod:`gpuwm.ingest.wif_climatology`) and uses the same Rust operators, in the
same order WRF-Chem's boundary preprocessors apply them:

1. **Decode** (Rust, ``grib2_stack``): the row's GRIB2 selectors pick every
   record of every variable a species row needs, stacked by model level.
2. **Convert** on the source grid (Rust, ``gpuwm_weighted_combination_f32``):
   a row's boundary entry names the source fields, their weights and the
   conversion.  The moist mass mixing ratio becomes a dry one with the same
   frame's specific humidity (``q / (1 - q_v)``), then ppmv or ug/kg.  Bin
   mapping (several source fields into one row) is the same weighted sum.
3. **Time** (Rust, the same operator): two frames bracketing the valid time
   are blended linearly (a 3-hourly source onto the met boundary interval),
   surface pressure included; ``daily_mean`` rows average the valid day's
   frames.
4. **Source pressure** (Rust, ``gpuwm_hybrid_full_pressure_f32``): full-level
   pressure from the half-level A/B coefficients the GRIB records carry and
   the blended surface pressure (ECMWF convention, mean of the half levels).
5. **Horizontal** (Rust, ``gpuwm_regular_interp_f32``, bilinear): onto the
   model's mass points.
6. **Vertical** (Rust, ``gpuwm_wrf_vert_interp_f32``): real.exe's
   ``vert_interp`` linear in log pressure onto the model's TOTAL (moist)
   hydrostatic half-level pressure of the same forcing frame (WRF's
   ``p + pb``, the target WRF-Chem's boundary preprocessor mozbc uses), so
   source and target are both total pressure and the chem column lands at
   the height the source put it; the lowest source level's value sits in
   the surface slot at the source's surface pressure, extrapolation below it
   is constant, and the window is linear at every level (as the WIF
   climatology call; a quadratic window can undershoot a tracer below
   zero).

NOTHING HERE NAMES A SOURCE OR A SPECIES.  Variables, selectors, the
vertical coordinate, the cadence and every conversion are row data; the only
constant is WRF's dry-air molar mass.  Python moves arrays between the Rust
operators and does no arithmetic on field values: a missing operator is a
refusal naming the rebuild, never a NumPy fallback (the 2.5.0 data-path law).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence

import numpy as np

__all__ = [
    "MWDRY_G_MOL", "BoundaryRemapError", "TargetGrid", "SourceFrames",
    "boundary_rows", "grib2_selection", "decode_source_files",
    "boundary_fields_for_grid",
]

#: Molecular weight of dry air, g/mol: WRF's ``mwdry``
#: (share/module_model_constants.F:34), the value WRF-Chem's own PM sums and
#: the mass ledger convert ppmv with.
MWDRY_G_MOL = 28.966

#: real.exe's zap_close_levels default in Pa (Registry.EM_COMMON), the value
#: the WIF climatology ingest also uses.
ZAP_CLOSE_LEVELS_PA = 500.0
#: real.exe's force_sfc_in_vinterp default.
FORCE_SFC_IN_VINTERP = 1

#: conversion -> (scale, divide by dry air).  ``ppmv`` scales need the row's
#: molar mass and are built in :func:`_conversion`.
_CONVERSIONS = ("identity", "kg_kg_moist_to_ug_kg_dry",
                "kg_kg_moist_to_ppmv_dry")


class BoundaryRemapError(ValueError):
    """A boundary source cannot be carried onto the grid as asked."""


@dataclass(frozen=True)
class TargetGrid:
    """Where the fields go: mass-point geodesy and the vertical target."""

    latitude: np.ndarray       # (ny, nx) degrees
    longitude: np.ndarray      # (ny, nx) degrees
    pressure: np.ndarray       # (nz, ny, nx) Pa, total half-level, bottom-up

    def __post_init__(self):
        lat = np.asarray(self.latitude)
        lon = np.asarray(self.longitude)
        p = np.asarray(self.pressure)
        if lat.ndim != 2 or lat.shape != lon.shape:
            raise BoundaryRemapError(
                "target latitude/longitude must be equal-shaped (ny, nx)")
        if p.ndim != 3 or p.shape[1:] != lat.shape:
            raise BoundaryRemapError(
                f"target pressure {p.shape} must be (nz, ny, nx) over "
                f"the {lat.shape} mass points")


@dataclass(frozen=True)
class SourceFrames:
    """The decoded frames of one source: ``{valid_time: {variable: field}}``.

    A field is a :class:`gpuwm.grib2_stack.StackedField` (levels ascending,
    ``(nlevel, ny, nx)`` float32, 1-D latitude/longitude axes, Section 4
    coordinate values) or any object with the same attributes.
    """

    source: str
    frames: Mapping[datetime, Mapping[str, object]]
    files: tuple[str, ...] = ()

    @property
    def times(self) -> tuple[datetime, ...]:
        return tuple(sorted(self.frames))


def _utc(moment: datetime) -> datetime:
    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def _backend(backend=None):
    if backend is not None:
        return backend
    from gpuwm.ingest.cpu_backend import CpuPreprocessBackend
    return CpuPreprocessBackend()


def _require(backend, method: str):
    call = getattr(backend, method, None)
    if call is None:
        raise BoundaryRemapError(
            f"the CPU preprocessing bridge has no {method!r}; rebuild "
            "tools/grib1_bridge. The boundary remap runs on the Rust "
            "operators only, never on a NumPy stand-in")
    return call


def boundary_rows(table, source_name: str, available=None) -> tuple:
    """Active rows whose FIRST available boundary entry is ``source_name``.

    The first entry of a row's ``boundary`` list whose source is available
    wins (DESIGN 1.2); a row whose earlier entry names another available
    source is not this source's to fill.  ``available`` is the run's enabled
    sources (``cfg.chem_sources``); by default every source the table
    references.
    """
    available = set(table.sources if available is None else available)
    rows = []
    for row in table.rows:
        entries = [b for b in row.boundary if b["source"] in available]
        if entries and entries[0]["source"] == source_name:
            rows.append(row)
    return tuple(rows)


def _entry(row, source_name: str):
    for entry in row.boundary:
        if entry["source"] == source_name:
            return entry
    raise BoundaryRemapError(
        f"species {row.name!r} has no boundary entry for {source_name!r}")


def _needed_variables(source_row, rows) -> tuple[str, ...]:
    vertical = source_row.vertical or {}
    keys: list[str] = []
    for row in rows:
        for key in _entry(row, source_row.name)["fields"]:
            if key not in keys:
                keys.append(key)
    for extra in (vertical.get("humidity_variable"),
                  vertical.get("surface_pressure_variable"),
                  vertical.get("pressure_variable"),
                  vertical.get("temperature_variable")):
        if extra and extra not in keys:
            keys.append(extra)
    unknown = [k for k in keys if k not in source_row.variables]
    if unknown:
        raise BoundaryRemapError(
            f"source {source_row.name!r} declares no variable(s) {unknown}; "
            f"it has {sorted(source_row.variables)}")
    return tuple(keys)


def grib2_selection(source_row, variables: Iterable[str]) -> list[dict]:
    """The ``grib2_stack`` select list for ``variables`` of ``source_row``."""
    select = []
    for key in variables:
        spec = source_row.variables[key]
        sel = dict(spec["selector"])
        select.append({
            "key": key,
            "discipline": int(sel["discipline"]),
            "category": int(sel["category"]),
            "parameter": int(sel["parameter"]),
            "constituent_type": sel.get("constituent_type"),
            "aerosol_type": sel.get("aerosol_type"),
            "level_type": int(sel["level_type"]),
            "levels": None,
        })
    return select


def decode_source_files(source_row, files, variables: Iterable[str],
                        work_dir: str | Path, *, stacker=None) -> SourceFrames:
    """Decode every record of ``variables`` in ``files`` into frames.

    ``files`` maps each request group of the source row (``model_level``,
    ``single_level``, ...) to the file that answers it, or is a sequence of
    ``(group, path)`` pairs; each file is asked only for the variables of its
    own group, so a selector can never be pointed at a file that cannot hold
    it (the stacker refuses a selector that matches nothing).  ``stacker``
    defaults to a decode-once wrapper of :func:`gpuwm.grib2_stack.stack_grib2`;
    it takes ``(path, select, out_dir)`` and returns stacked fields with
    ``key`` and ``valid_time``.  A (variable, time) seen twice is refused.
    """
    if source_row.format != "grib2":
        raise BoundaryRemapError(
            f"source {source_row.name!r} is {source_row.format}; the boundary "
            "remap decodes GRIB2 sources")
    if stacker is None:
        stacker = _stack_once
    variables = tuple(variables)
    pairs = tuple(files.items()) if isinstance(files, Mapping) else tuple(files)
    frames: dict[datetime, dict[str, object]] = {}
    work_dir = Path(work_dir)
    for index, (group, path) in enumerate(pairs):
        wanted = [key for key in variables
                  if source_row.variables[key]["group"] == group]
        if not wanted:
            continue
        select = grib2_selection(source_row, wanted)
        # Short on purpose: a content-addressed file stem is 64 hex digits,
        # and Windows refuses paths past 260 characters.
        out = work_dir / f"s{index:02d}-{Path(path).stem[:16]}"
        for fld in stacker(Path(path), select, out):
            when = _utc(fld.valid_time)
            slot = frames.setdefault(when, {})
            if fld.key in slot:
                raise BoundaryRemapError(
                    f"{source_row.name}: {fld.key} at {when.isoformat()} is "
                    "in two input files; one frame per time")
            slot[fld.key] = fld
    return SourceFrames(
        source=source_row.name,
        frames=MappingProxyType({t: MappingProxyType(v)
                                 for t, v in sorted(frames.items())}),
        files=tuple(str(p) for _group, p in pairs))


def _stack_once(path: Path, select: list[dict], out: Path, *, expected_authority=None):
    """Keep one selected decoder through admission, decode and axis reads."""
    from gpuwm.ingest.cpu_backend import (
        cpu_bridge_scope, resolve_cpu_bridge, _selected_cpu_worker_cap)
    with cpu_bridge_scope(resolve_cpu_bridge(), worker_cap=_selected_cpu_worker_cap.get()):
        return _stack_once_selected(path, select, out, expected_authority=expected_authority)


def _stack_once_selected(path: Path, select: list[dict], out: Path, *, expected_authority=None):
    """Decode once per output folder: a finished folder is read back.

    ``grib2_stack`` refuses to overwrite, and a run initializes many forcing
    frames from the same files, so a folder that already holds a verified
    ``stack.json`` (every stack's sha256 is checked on read) is reused.  A
    new decode lands in a private temporary folder and is renamed into
    place, so two processes decoding the same file never see half a folder.
    """
    import os
    import tempfile
    from gpuwm.grib2_stack import (
        read_stack, stack_grib2, stack_cache_digest, stack_cache_identity)
    out = Path(out)
    authority = stack_cache_identity(path, select)
    if expected_authority is not None and authority != expected_authority:
        raise BoundaryRemapError("GRIB2 source authority changed before decode")
    out = out.with_name(f"{out.name}-{stack_cache_digest(authority)[:16]}")

    def read_current():
        import json
        try:
            stored = json.loads((out / "decode-authority.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise BoundaryRemapError(f"GRIB2 decode cache has no valid authority: {out}") from exc
        if stored != authority:
            raise BoundaryRemapError(f"GRIB2 decode cache authority mismatch: {out}")
        return read_stack(out)

    if (out / "stack.json").is_file():
        return read_current()
    out.parent.mkdir(parents=True, exist_ok=True)
    # Separate namespaces also cover two concurrent calls in one process.
    # The native output must not exist before decoding, so use a child of
    # the unique staging directory, then retain atomic directory publication.
    with tempfile.TemporaryDirectory(prefix=f".{out.name}.stage-", dir=out.parent) as staging:
        temporary = Path(staging) / "decoded"
        stack_grib2(path, authority["specification"]["select"], temporary)
        if stack_cache_identity(path, select) != authority:
            raise BoundaryRemapError("GRIB2 input or selected decoder changed during decode")
        import json
        (temporary / "decode-authority.json").write_text(
            json.dumps(authority, sort_keys=True, allow_nan=False), encoding="utf-8")
        try:
            os.replace(temporary, out)
        except OSError:
            if not (out / "stack.json").is_file():
                raise
    return read_current()


def _conversion(row, entry) -> tuple[float, bool]:
    kind = entry["conversion"]
    if kind == "identity":
        return 1.0, False
    if kind == "kg_kg_moist_to_ug_kg_dry":
        return 1.0e9, True
    if kind == "kg_kg_moist_to_ppmv_dry":
        if row.molar_mass_g_mol is None:
            raise BoundaryRemapError(
                f"{row.name}: a ppmv conversion needs the row's molar mass")
        return MWDRY_G_MOL / float(row.molar_mass_g_mol) * 1.0e6, True
    raise BoundaryRemapError(
        f"{row.name}: boundary conversion {kind!r} is not one a 3-D mixing-"
        f"ratio source supplies (this remap converts {list(_CONVERSIONS)})")


def _bracket(times: Sequence[datetime], when: datetime, name: str):
    if not times:
        raise BoundaryRemapError(f"source {name!r} has no decoded frame")
    if when < times[0] or when > times[-1]:
        raise BoundaryRemapError(
            f"source {name!r} frames span {times[0].isoformat()} to "
            f"{times[-1].isoformat()}; {when.isoformat()} is outside them, "
            "and a boundary value is never extrapolated in time")
    for t0, t1 in zip(times, times[1:]):
        if t0 <= when <= t1:
            if when == t0:
                return t0, t0, 0.0
            if when == t1:
                return t1, t1, 0.0
            span = (t1 - t0).total_seconds()
            return t0, t1, (when - t0).total_seconds() / span
    return times[0], times[0], 0.0


def _daily_frames(times: Sequence[datetime], when: datetime, cadence_s: float,
                  name: str) -> tuple[datetime, ...]:
    day = when.replace(hour=0, minute=0, second=0, microsecond=0)
    expected = int(round(86400.0 / cadence_s))
    chosen = tuple(t for t in times if day <= t < day + timedelta(days=1))
    if len(chosen) != expected:
        raise BoundaryRemapError(
            f"source {name!r}: a daily mean for {day.date()} needs its "
            f"{expected} frames at {cadence_s:g} s cadence; {len(chosen)} "
            "are decoded")
    return chosen


def _field(frame, key: str, when: datetime, source: str):
    try:
        return frame[key]
    except KeyError:
        raise BoundaryRemapError(
            f"source {source!r} frame {when.isoformat()} has no {key!r}") from None


def _array(fld) -> np.ndarray:
    return np.asarray(fld.values if hasattr(fld, "values") else fld.array,
                      dtype=np.float32)


def _grid_of(fields: Iterable[object]):
    lat = lon = coords = None
    for fld in fields:
        f_lat = np.asarray(fld.latitude, dtype=np.float64)
        f_lon = np.asarray(fld.longitude, dtype=np.float64)
        if lat is None:
            lat, lon = f_lat, f_lon
        elif not (np.array_equal(lat, f_lat) and np.array_equal(lon, f_lon)):
            raise BoundaryRemapError(
                "the source fields of one frame are on different grids")
        c = tuple(getattr(fld, "coordinate_values", ()) or ())
        if c:
            if coords is None:
                coords = c
            elif c != coords:
                raise BoundaryRemapError(
                    "the source fields carry different hybrid coefficients")
    return lat, lon, coords


def _horizontal_plan(backend, source_row, fields, lat_axis, lon_axis, target):
    """One bilinear plan from the source grid to the target mass points.

    A regular latitude/longitude source interpolates on its axes.  A Lambert
    conformal source (GRIB2 template 3.30, ``StackedField.projection``)
    projects every target point onto the stored grid with WPS's own Lambert
    transform (gpuwm.static.lambert, on WPS's 6370 km sphere, so the spacing
    is scaled by the row's ``grid.earth_radius_m``, as the HRRR route does)
    and interpolates at those fractional indices.  A target point outside the
    source grid is refused: the field there would be invented.
    """
    projection = getattr(fields[0], "projection", None) if fields else None
    target_lat = np.asarray(target.latitude, dtype=np.float64)
    target_lon = np.asarray(target.longitude, dtype=np.float64)
    if not projection:
        return backend.regular_plan(lat_axis, lon_axis, target_lat, target_lon)
    if any(getattr(f, "projection", None) != projection for f in fields):
        raise BoundaryRemapError(
            f"source {source_row.name!r}: the fields of one frame are on "
            "different projected grids")
    radius = (source_row.grid or {}).get("earth_radius_m")
    if not radius:
        raise BoundaryRemapError(
            f"source {source_row.name!r} is on a Lambert grid and its row names "
            "no grid.earth_radius_m; the projection would place it on the "
            "wrong sphere")
    from gpuwm.static.lambert import EARTH_RADIUS_M, LambertGrid

    def wrap(value):
        return ((float(value) + 180.0) % 360.0) - 180.0

    nx, ny = int(projection["nx"]), int(projection["ny"])
    spacing = float(projection["dlon"]) * EARTH_RADIUS_M / float(radius)
    grid = LambertGrid(ref_lat=float(projection["lat1"]), ref_lon=wrap(projection["lon1"]),
                       truelat1=float(projection["latin1"]),
                       truelat2=float(projection["latin2"]),
                       stand_lon=wrap(projection["lov"]), dx=spacing, dy=spacing,
                       e_we=nx + 1, e_sn=ny + 1, known_x=1.0, known_y=1.0)
    x, y = grid.latlon_to_ij(target_lat, target_lon)
    x = np.asarray(x, dtype=np.float64) - 1.0
    y = np.asarray(y, dtype=np.float64) - 1.0
    if x.min() < 0 or y.min() < 0 or x.max() > nx - 1 or y.max() > ny - 1:
        raise BoundaryRemapError(
            f"source {source_row.name!r}: the domain reaches outside its "
            f"{nx} x {ny} grid; values there would be invented")
    return backend.indexed_plan((ny, nx), y, x)


def boundary_fields_for_grid(table, source_row, frames: SourceFrames,
                             target: TargetGrid, valid_time: datetime, *,
                             backend=None, available=None
                             ) -> tuple[dict[str, np.ndarray], dict]:
    """``{species: (nz, ny, nx) float32}`` of every row this source fills.

    Returns the fields and a receipt naming the frames, weights and
    operators.  Raises :class:`BoundaryRemapError` for anything that cannot
    be done as the rows say.
    """
    backend = _backend(backend)
    combine = _require(backend, "weighted_combination")
    full_pressure = _require(backend, "hybrid_full_pressure")
    vertical = source_row.vertical or {}
    kind = vertical.get("kind")
    if kind not in ("hybrid_sigma_pressure", "pressure_field"):
        raise BoundaryRemapError(
            f"source {source_row.name!r} vertical kind "
            f"{kind!r}; this remap carries hybrid sigma-pressure sources and "
            "sources whose levels carry their own pressure field")
    pressure_key = vertical.get("pressure_variable")
    temperature_key = vertical.get("temperature_variable")
    if kind == "pressure_field" and not pressure_key:
        raise BoundaryRemapError(
            f"source {source_row.name!r}: a pressure_field source must name its "
            "pressure_variable, or its levels could not be placed in the vertical")
    rows = boundary_rows(table, source_row.name, available)
    if not rows:
        return {}, {"source": source_row.name, "rows": []}
    variables = _needed_variables(source_row, rows)
    humidity_key = vertical.get("humidity_variable")
    ps_key = vertical["surface_pressure_variable"]
    when = _utc(valid_time)
    times = frames.times
    interp = (source_row.time or {}).get("interpolation", "linear")
    if interp == "daily_mean":
        members = _daily_frames(times, when, float(source_row.time["cadence_s"]),
                                source_row.name)
        weights = [1.0 / len(members)] * len(members)
    elif interp == "linear":
        t0, t1, a = _bracket(times, when, source_row.name)
        members = (t0,) if t0 == t1 else (t0, t1)
        weights = [1.0] if t0 == t1 else [1.0 - a, a]
    else:
        raise BoundaryRemapError(
            f"source {source_row.name!r} time interpolation {interp!r} is not "
            "one this remap performs (linear, daily_mean)")

    # 2. Convert every row on the source grid, frame by frame.
    per_frame: dict[str, list[np.ndarray]] = {row.name: [] for row in rows}
    ps_frames: list[np.ndarray] = []
    p_frames: list[np.ndarray] = []
    grid_fields = []
    for moment in members:
        frame = frames.frames[moment]
        q = (_array(_field(frame, humidity_key, moment, source_row.name))
             if humidity_key else None)
        for row in rows:
            entry = _entry(row, source_row.name)
            src = [_field(frame, key, moment, source_row.name)
                   for key in entry["fields"]]
            grid_fields.extend(src)
            if entry["conversion"] == "kg_m3_to_ug_kg_dry":
                # A mass density (HRRR-Smoke's MASSDEN): weighted on the
                # source levels, then divided by each level's own density
                # P/(R T) in Rust (chem_conversions), before any remap.
                if not (pressure_key and temperature_key):
                    raise BoundaryRemapError(
                        f"{row.name}: a mass-density conversion needs the "
                        "source's pressure_variable and temperature_variable")
                from gpuwm.chem_conversions import convert_source
                params = [source_row.variables[k].get("conversion_parameters")
                          for k in entry["fields"]]
                if any(value != params[0] for value in params):
                    raise BoundaryRemapError(
                        f"{row.name}: boundary terms use different density "
                        "conventions; one conversion cannot serve them")
                combined = combine([_array(f) for f in src],
                                   [float(w) for w in entry["weights"]],
                                   scale=1.0, humidity=None)
                per_frame[row.name].append(convert_source(
                    "kg_m3_to_ug_kg_dry", combined,
                    {"PRES": _array(_field(frame, pressure_key, moment, source_row.name)),
                     "TT": _array(_field(frame, temperature_key, moment, source_row.name))},
                    parameters=params[0]))
                continue
            scale, dry = _conversion(row, entry)
            if dry and q is None:
                raise BoundaryRemapError(
                    f"{row.name}: a dry-air conversion needs the source's "
                    "humidity_variable, which the row does not declare")
            per_frame[row.name].append(combine(
                [_array(f) for f in src], [float(w) for w in entry["weights"]],
                scale=scale, humidity=q if dry else None))
        if pressure_key:
            p_frames.append(_array(_field(frame, pressure_key, moment, source_row.name)))
        ps = _field(frame, ps_key, moment, source_row.name)
        grid_fields.append(ps)
        ps_frames.append(_array(ps).reshape(_array(ps).shape[-2:]))

    lat_axis, lon_axis, coords = _grid_of(grid_fields)
    if kind == "hybrid_sigma_pressure" and not coords:
        raise BoundaryRemapError(
            f"source {source_row.name!r}: no record carried its hybrid A/B "
            "coefficients (GRIB2 Section 4 coordinate values)")
    coords = coords or ()
    nhalf = len(coords) // 2
    a_half = np.asarray(coords[:nhalf], dtype=np.float64)
    b_half = np.asarray(coords[nhalf:], dtype=np.float64)
    levels = None
    for key in variables:
        if key == ps_key:
            continue
        fld = frames.frames[members[0]][key]
        lv = np.asarray(fld.levels, dtype=np.int32)
        if levels is None:
            levels = lv
        elif not np.array_equal(levels, lv):
            raise BoundaryRemapError(
                f"source {source_row.name!r}: {key} has levels {lv.tolist()} "
                f"where the other variables have {levels.tolist()}")

    # 3. Time: one blend of the frames with the row's weights.
    def blend(stacks):
        if len(stacks) == 1:
            return stacks[0]
        return combine(stacks, weights, scale=1.0, humidity=None)

    species_src = {name: blend(stacks) for name, stacks in per_frame.items()}
    ps_src = blend(ps_frames)

    # 4. Source full-level pressure: from the blended surface pressure and
    # the A/B coefficients, or the source's own pressure on its levels.
    if kind == "pressure_field":
        p_src = blend(p_frames)
    else:
        p_src = full_pressure(a_half, b_half, levels, ps_src)
    # The vertical step reads the stacks top first (ascending pressure), the
    # order a hybrid sigma-pressure source's model levels arrive in; a
    # source numbered from the ground up (HRRR's hybrid levels) is turned
    # over, an index permutation and no arithmetic.
    if float(np.mean(p_src[0])) > float(np.mean(p_src[-1])):
        p_src = np.ascontiguousarray(p_src[::-1])
        species_src = {name: np.ascontiguousarray(stack[::-1])
                       for name, stack in species_src.items()}
        levels = levels[::-1]

    # 5. Horizontal, one plan for every stack.
    plan = _horizontal_plan(backend, source_row, grid_fields, lat_axis, lon_axis, target)
    p_h = plan.apply(p_src, method="bilinear")
    ps_h = plan.apply(ps_src, method="bilinear")
    nz = int(np.asarray(target.pressure).shape[0])
    out: dict[str, np.ndarray] = {}
    for name, stack in species_src.items():
        field_h = plan.apply(stack, method="bilinear")
        # 6. Vertical: lowest source level (largest pressure, the last level
        # in the top-first order above) in the surface slot at the source's
        # surface pressure; linear in log p at every level; constant below.
        bottom = int(p_src.shape[0]) - 1
        out[name] = backend.wrf_vertical_interpolate(
            field_h, field_h[bottom], p_h, ps_h,
            np.asarray(target.pressure, dtype=np.float32),
            interp_in_logp=True, extrap="constant",
            force_sfc_in_vinterp=FORCE_SFC_IN_VINTERP,
            zap_close_levels=ZAP_CLOSE_LEVELS_PA, vboundb=nz)
    receipt = {
        "schema": "gpuwm.chem.boundary-remap.v1",
        "source": source_row.name,
        "valid_time": when.isoformat(),
        "time_interpolation": interp,
        "frames": [t.isoformat() for t in members],
        "weights": list(weights),
        "rows": [row.name for row in rows],
        "levels": [int(levels.min()), int(levels.max()), int(levels.size)],
        "operators": {
            "convert_and_time": "gpuwm_weighted_combination_f32" + (
                ", gpuwm_kg_m3_to_ug_kg_dry_f32 on the source levels"
                if any(_entry(r, source_row.name)["conversion"] == "kg_m3_to_ug_kg_dry"
                       for r in rows) else ""),
            "source_pressure": ("the source's own pressure field on its levels"
                                if kind == "pressure_field" else
                                "gpuwm_hybrid_full_pressure_f32 (mean of half levels)"),
            "horizontal": ("Lambert conformal target-to-source indices "
                           "(gpuwm.static.lambert), bilinear"
                           if getattr(grid_fields[0], "projection", None) else
                           "gpuwm_regular_interp_f32 bilinear"),
            "vertical": ("gpuwm_wrf_vert_interp_f32, linear in log p onto the total "
                         "hydrostatic half-level pressure, lowest level at the "
                         "source surface pressure, constant below"),
        },
        "mwdry_g_mol": MWDRY_G_MOL,
    }
    return out, receipt
