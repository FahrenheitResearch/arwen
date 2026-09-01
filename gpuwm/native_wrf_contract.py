"""Shared fail-closed geometry contract for native stock-WRF adapters."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Mapping
import uuid

import numpy as np

from gpuwm.static.lambert import (
    _parse_wps_namelist,
    grids_from_projection_config,
    grids_from_wps_namelist,
)
from gpuwm.vertical_contract import validate_explicit_eta_grid


CERTIFIED_ETA_LEVELS = (
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

NATIVE_STATIC_REQUIRED = frozenset({
    "HGT_M", "LANDUSEF", "LANDMASK", "LU_INDEX", "SOILCTOP", "SCT_DOM",
    "SOILCBOT", "SCB_DOM", "GREENFRAC", "LAI12M", "ALBEDO12M", "SNOALB",
    "SOILTEMP", "TMN",
})

# The portable native-static contract is deliberately constrained to the
# 21-category MODIS/Noah land-use table and the 16-category Noah soil table.
# These values are the WRF table identities implied by the exact category
# dimensions validated below; adapters using another catalog must publish a
# different explicit contract instead of being interpreted with these codes.
NATIVE_LANDUSE_IDENTITY = {
    "MMINLU": "MODIFIED_IGBP_MODIS_NOAH",
    "ISWATER": 17,
    "ISLAKE": 21,
    "ISICE": 15,
}

_NATIVE_STATIC_MASS_2D = frozenset({
    "HGT_M", "LANDMASK", "LU_INDEX", "SCT_DOM", "SCB_DOM", "SNOALB",
    "SOILTEMP", "TMN",
})
_NATIVE_STATIC_MONTHLY = frozenset({"GREENFRAC", "LAI12M", "ALBEDO12M"})
_NATIVE_STATIC_CATEGORY_COUNT = {
    "LANDUSEF": 21,
    "SOILCTOP": 16,
    "SOILCBOT": 16,
}
_NATIVE_STATIC_GEOMETRY = frozenset({
    "MAPFAC_M", "MAPFAC_U", "MAPFAC_V", "F", "E", "SINALPHA",
    "COSALPHA",
})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def native_geometry_contract(grid, cfg) -> dict[str, object]:
    expected = (int(cfg.nx), int(cfg.ny), float(cfg.dx), float(cfg.dy))
    observed = (
        int(grid.e_we) - 1, int(grid.e_sn) - 1,
        float(grid.dx), float(grid.dy))
    if observed != expected:
        raise ValueError(
            "native grid geometry differs from the domain configuration: "
            f"grid={observed}, config={expected}")
    latitude, longitude = grid.latlon_mass()
    return {
        "mass_shape": [cfg.ny, cfg.nx],
        "nz": cfg.nz,
        "dx_m": cfg.dx,
        "dy_m": cfg.dy,
        "map_proj": getattr(grid, "map_proj", "lambert"),
        "ref_lat": grid.ref_lat,
        "ref_lon": grid.ref_lon,
        "truelat1": grid.truelat1,
        "truelat2": grid.truelat2,
        "stand_lon": grid.stand_lon,
        "center_lat": grid.cen_lat,
        "center_lon": grid.cen_lon,
        "lat_range": [float(latitude.min()), float(latitude.max())],
        "lon_range": [float(longitude.min()), float(longitude.max())],
    }


def write_native_static_cache(
        path: Path, fields: Mapping[str, object]) -> dict[str, object]:
    """Atomically write a finite numeric native-static NPZ."""

    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite native static cache {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".tmp-{uuid.uuid4().hex[:8]}.npz")
    arrays = {}
    for name, value in sorted(fields.items()):
        if not isinstance(name, str) or not name:
            raise ValueError("native static field names must be non-empty strings")
        array = np.asarray(value, dtype=np.float64)
        if array.dtype.hasobject or not np.isfinite(array).all():
            raise ValueError(f"native static field {name!r} is not finite numeric")
        arrays[name] = array
    if not arrays:
        raise ValueError("native static cache cannot be empty")
    try:
        np.savez(temporary, **arrays)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "fields": list(arrays),
    }


def write_native_geometry_receipt(
        path: Path, grid, cfg, static_path: Path) -> dict[str, object]:
    """Atomically write the standard geometry/static binding receipt."""

    path = Path(path)
    static_path = Path(static_path)
    if path.exists():
        raise FileExistsError(
            f"refusing to overwrite native geometry receipt {path}")
    if not static_path.is_file():
        raise FileNotFoundError(f"native static cache is missing: {static_path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    receipt = {
        "schema": "gpuwm-native-static-direct-v1",
        "status": "PASS",
        "geometry": native_geometry_contract(grid, cfg),
        "cache": {
            "path": static_path.name,
            "bytes": static_path.stat().st_size,
            "sha256": _sha256(static_path),
        },
    }
    temporary = path.with_name(f".tmp-{uuid.uuid4().hex[:8]}")
    try:
        temporary.write_text(
            json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False)
            + "\n", encoding="utf-8")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return receipt


def canonical_noah_surface(soil) -> dict[str, np.ndarray]:
    """Return the source-neutral Noah surface inventory used by WRF export."""

    return {
        "TSK": soil.tsk,
        "TSLB": soil.soil_temperature,
        "SMOIS": soil.soil_moisture,
        "SH2O": soil.liquid_moisture,
        "TMN": soil.deep_soil_temperature,
        "SEAICE": soil.xice,
        "XLAND": soil.xland,
        "LANDMASK": soil.landmask,
        "SNOW": soil.snow_water,
        "SNOWH": soil.snow_depth,
    }


def native_static_export_fields(
        fields: Mapping[str, object], grid) -> dict[str, object]:
    """Add geometry-derived map/coriolis fields to WPS_GEOG statics.

    Any caller-supplied copy must match the regenerated grid value exactly;
    editable static metadata cannot override the namelist-derived geometry.
    """

    result = dict(fields)
    generated = {
        "MAPFAC_M": grid.mapfac_m(),
        "MAPFAC_U": grid.mapfac_u(),
        "MAPFAC_V": grid.mapfac_v(),
    }
    generated["F"], generated["E"] = grid.coriolis_m()
    generated["SINALPHA"], generated["COSALPHA"] = grid.rotation_m()
    for name, value in generated.items():
        if name in result:
            stored = np.asarray(result[name], dtype=np.float64)
            regen = np.asarray(value, dtype=np.float64)
            # Bitwise equality held while every producer and consumer ran
            # on one machine.  Replaying a prepared cache on another OS
            # re-derives these trig fields through a different libm, whose
            # correctly-rounded-to-a-few-ulp answers differ in the last
            # bits (measured: 4 ulp worst case, MAPFAC_M, Windows-written
            # cache replayed on glibc).  16 ulps tells those apart from a
            # genuinely different geometry, which differs by orders of
            # magnitude more; and the regenerated value still wins below,
            # so nothing downstream ever sees the stored copy.
            tol = 16.0 * np.spacing(np.maximum(np.abs(stored),
                                               np.abs(regen)))
            if stored.shape != regen.shape or not bool(
                    np.all(np.abs(stored - regen) <= tol)):
                d = np.abs(stored - regen)
                raise ValueError(
                    f"native static {name} differs from regenerated grid"
                    f" geometry beyond libm rounding: max_abs"
                    f" {float(d.max())!r} at {int(np.count_nonzero(d))}"
                    f" of {d.size} points")
        result[name] = value
    return result


def validate_native_static_fields(
        fields: Mapping[str, object], grid, ny: int, nx: int,
) -> dict[str, np.ndarray]:
    """Validate the portable static cache shared by direct source adapters."""

    missing = sorted(NATIVE_STATIC_REQUIRED - set(fields))
    if missing:
        raise KeyError(f"native static fields are missing {missing}")
    retained = NATIVE_STATIC_REQUIRED | (_NATIVE_STATIC_GEOMETRY & set(fields))
    result = {
        name: np.asarray(fields[name], dtype=np.float64)
        for name in sorted(retained)
    }
    expected_shapes = {
        **{name: (ny, nx) for name in _NATIVE_STATIC_MASS_2D},
        **{name: (12, ny, nx) for name in _NATIVE_STATIC_MONTHLY},
        **{
            name: (categories, ny, nx)
            for name, categories in _NATIVE_STATIC_CATEGORY_COUNT.items()
        },
        "MAPFAC_M": (ny, nx),
        "MAPFAC_U": (ny, nx + 1),
        "MAPFAC_V": (ny + 1, nx),
        "F": (ny, nx),
        "E": (ny, nx),
        "SINALPHA": (ny, nx),
        "COSALPHA": (ny, nx),
    }
    for name, value in result.items():
        expected = expected_shapes[name]
        if value.shape != expected:
            raise ValueError(
                f"static field {name} has shape {value.shape}, "
                f"expected {expected}")
        if not np.isfinite(value).all():
            raise ValueError(f"static field {name} contains non-finite values")
    if not np.isin(result["LANDMASK"], (0.0, 1.0)).all():
        raise ValueError("static field LANDMASK must be exactly binary")
    for name, upper in (("LU_INDEX", 21), ("SCT_DOM", 16), ("SCB_DOM", 16)):
        value = result[name]
        if (np.any(value != np.floor(value)) or value.min() < 1.0
                or value.max() > float(upper)):
            raise ValueError(
                f"static field {name} is outside the exact 1..{upper} "
                "category contract")
    for name in _NATIVE_STATIC_CATEGORY_COUNT:
        value = result[name]
        if value.min() < 0.0 or value.max() > 1.0:
            raise ValueError(f"static field {name} fractions are outside 0..1")
        if not np.allclose(
                value.sum(axis=0), 1.0, rtol=0.0, atol=2.0e-6):
            raise ValueError(
                f"static field {name} category fractions do not sum to one")
    if result["GREENFRAC"].min() < 0.0 \
            or result["GREENFRAC"].max() > 1.0:
        raise ValueError("static field GREENFRAC is outside 0..1")
    if result["LAI12M"].min() < 0.0:
        raise ValueError("static field LAI12M contains negative values")
    if result["ALBEDO12M"].min() < 0.0 \
            or result["ALBEDO12M"].max() > 100.0:
        raise ValueError("static field ALBEDO12M is outside 0..100 percent")
    land = result["LANDMASK"] == 1.0
    if np.any(land) and not np.any(result["HGT_M"][land] != 0.0):
        raise ValueError(
            "native static HGT_M is identically zero over every land cell; "
            "mandatory WPS GEOG terrain is missing, outside its staged "
            "footprint, or degenerate")
    return native_static_export_fields(result, grid)


def load_native_static_cache(
        path: Path, grid, ny: int, nx: int,
) -> dict[str, np.ndarray]:
    """Load a pickle-free portable native static cache and validate it."""

    with np.load(Path(path), allow_pickle=False) as source:
        return validate_native_static_fields(source, grid, ny, nx)


def verify_native_static_receipt(
    receipt_path: Path, static_input: Path, grid, cfg,
) -> dict[str, object]:
    """Require a geometry and SHA-bound native static cache receipt."""

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (receipt.get("schema") != "gpuwm-native-static-direct-v1"
            or receipt.get("status") != "PASS"):
        raise ValueError("unrecognized or non-PASS native static receipt")
    if receipt.get("geometry") != native_geometry_contract(grid, cfg):
        raise ValueError("native static receipt geometry differs from target")
    cache = receipt.get("cache")
    expected_cache = {
        "path": static_input.name,
        "bytes": static_input.stat().st_size,
        "sha256": _sha256(static_input),
    }
    if cache != expected_cache:
        raise ValueError("native static receipt does not bind the supplied cache")
    return receipt


def validate_native_lambert_contracts(
    exp, wps_namelist: Path, *, source_name: str,
    source_top_pressure_pa: float | None = None,
):
    """Return every domain grid after exact geometry/eta validation.

    The returned tuple is parent-before-child in the same order as
    ``exp.domains``.  Geometry is compared against the standard WPS namelist
    one domain at a time; a missing, extra, reordered, or numerically drifting
    domain is a hard error.  All domains currently share the experiment's one
    explicit vertical coordinate, matching :class:`ExperimentConfig`'s
    fail-closed vertical-refinement policy.
    """

    label = source_name.upper()
    if not exp.domains:
        raise ValueError(f"{label} direct adapter requires at least one domain")
    if exp.projection is None:
        raise ValueError(
            f"{label} direct adapter requires a [projection] table")
    wps_values = _parse_wps_namelist(wps_namelist)
    wps_map_proj = str(wps_values.get("map_proj", [""])[0]).lower()
    if wps_map_proj not in ("lambert", "mercator", "polar"):
        raise ValueError(
            f"{label} WPS namelist must explicitly declare map_proj as "
            "'lambert', 'mercator', or 'polar'")
    if wps_map_proj != exp.projection.map_proj:
        raise ValueError(
            f"{label} WPS namelist declares map_proj={wps_map_proj!r} but "
            f"the experiment [projection] table declares "
            f"{exp.projection.map_proj!r}")
    wps_grids = grids_from_wps_namelist(wps_namelist)
    if not wps_grids:
        raise ValueError(f"{label} WPS namelist does not declare d01")
    expected_grids = grids_from_projection_config(exp)
    if len(wps_grids) != len(exp.domains):
        raise ValueError(
            f"{label} WPS/experiment domain-count mismatch: WPS declares "
            f"{len(wps_grids)}, experiment declares {len(exp.domains)}")
    if len(expected_grids) != len(exp.domains):
        raise ValueError(
            f"{label} experiment geometry resolved to {len(expected_grids)} "
            f"grids for {len(exp.domains)} domains")

    compared = (
        "ref_lat", "ref_lon", "truelat1", "truelat2", "stand_lon",
        "dx", "dy", "e_we", "e_sn", "known_x", "known_y",
    )
    drift = {}
    for index, (domain, observed, expected) in enumerate(zip(
            exp.domains, wps_grids, expected_grids), start=1):
        domain_drift = {
            name: {
                "wps": getattr(observed, name),
                "experiment": getattr(expected, name),
            }
            for name in compared
            if getattr(observed, name) != getattr(expected, name)
        }
        if domain.grid_id != index:
            domain_drift["grid_id"] = {
                "wps": index,
                "experiment": domain.grid_id,
            }
        if domain_drift:
            drift[f"d{index:02d}"] = domain_drift
    if drift:
        raise ValueError(f"WPS/experiment domain geometry mismatch: {drift}")

    vertical = exp.vertical
    if vertical.hybrid_opt != 2:
        raise ValueError(
            f"{label} direct export currently requires WRF hybrid_opt=2")
    # The vertical ladder refusals below speak in the mapped door's
    # voice (UX finding R2, walk C step 5): an import-namelist config
    # declares a level count and no explicit ladder -- the shape every
    # stock WRF namelist has, because real.exe generates the ladder
    # itself -- and this route used to answer it with the bare
    # ``explicit eta_levels has shape (0,)`` and no remedy at all.
    # Both refusals carry their own doors now, exactly like
    # ``gpuwm.mapped_direct._validate_target_contract``.
    from gpuwm.ingest.source_coverage import VerticalLadderRefusal

    if not tuple(vertical.eta_levels or ()):
        nz = int(exp.root.run.nz)
        raise VerticalLadderRefusal(
            f"{label} direct adapter vertical ladder is missing: the "
            f"experiment config declares nz={nz} mass levels (WRF "
            f"e_vert={nz + 1}) and no explicit eta_levels ladder.  This "
            "route interpolates every forcing time onto an explicit "
            "full-level eta ladder; a level count alone does not define "
            "one, and WRF's automatic level generator (real.exe) is not "
            "implemented",
            remedy=(
                f"remedy: two doors reconcile this.  Keep your {nz} "
                f"levels: add an explicit eta_levels ladder of {nz + 1} "
                "interfaces -- `eta_levels = [1.0, ..., 0.0]`, strictly "
                "decreasing -- to the [shared] block of the experiment "
                "config; prep adopts your ladder at your level count.  "
                "Or use the packaged reference ladder: `gpuwm domain` "
                "authors a config whose [shared] block carries the "
                "certified ladder; copy its nz/p_top/eta_levels lines "
                "into your imported config."))
    for domain in exp.domains:
        try:
            validate_explicit_eta_grid(
                vertical.eta_levels,
                nz=domain.run.nz,
                p_top=vertical.p_top,
                source_top_pressure_pa=source_top_pressure_pa,
                context=f"{label} direct adapter d{domain.grid_id:02d}",
            )
        except ValueError as error:
            raise VerticalLadderRefusal(
                str(error),
                remedy=(
                    "remedy: fix the [shared] eta_levels ladder in the "
                    "experiment config: nz + 1 entries (WRF e_vert), "
                    "running 1.0 (surface) to 0.0 (top), strictly "
                    "decreasing, with p_top in pascals inside the "
                    "source atmosphere.")) from error
    return tuple(expected_grids)


def validate_native_lambert_contract(
    exp, wps_namelist: Path, *, source_name: str,
    source_top_pressure_pa: float | None = None,
):
    """Backward-compatible single-domain validation wrapper."""

    label = source_name.upper()
    if len(exp.domains) != 1:
        raise ValueError(
            f"{label} single-domain adapter received {len(exp.domains)} "
            "domains; use validate_native_lambert_contracts for a hierarchy")
    return validate_native_lambert_contracts(
        exp, wps_namelist, source_name=source_name,
        source_top_pressure_pa=source_top_pressure_pa,
    )[0]
