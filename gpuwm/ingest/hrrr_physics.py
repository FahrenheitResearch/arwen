"""Shared prepared-real physics initialization for GPU forecast runners.

The historical public entry point remains :func:`initialize_hrrr_physics`,
but the actual physics setup consumes the source-neutral Noah surface contract
stored by ``gpuwm-prepared-real-cache-v1``.  GFS and ERA5 cache restores use
that same path instead of re-decoding source-specific soil fields.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType, SimpleNamespace

import numpy as np

from gpuwm.config import soil_layer_count
from gpuwm.core.noah import noah_initial_snow_albedo
from gpuwm.ingest.hrrr_surface import surface_fields_to_device


_CANONICAL_SURFACE_FIELDS = frozenset({
    "TSK", "TSLB", "SMOIS", "SH2O", "TMN", "SEAICE", "XLAND",
    "LANDMASK", "SNOW", "SNOWH",
})


def _validate_prepared_surface(surface, cfg) -> Mapping[str, object]:
    try:
        fields = surface.fields
    except AttributeError as exc:
        raise TypeError(
            "prepared physics surface must expose a fields mapping") from exc
    if not isinstance(fields, Mapping):
        raise TypeError("prepared physics surface fields must be a mapping")
    if set(fields) != _CANONICAL_SURFACE_FIELDS:
        raise ValueError(
            "prepared physics surface inventory differs from the canonical "
            f"Noah contract: expected {sorted(_CANONICAL_SURFACE_FIELDS)}, "
            f"got {sorted(fields)}")
    horizontal_shape = (int(cfg.ny), int(cfg.nx))
    # The prepared cache stores whatever geometry the run that wrote it used,
    # so the shape gate is the SELECTED scheme's layer count, not a literal.
    # A four-layer cache restored under a nine-layer scheme is refused here
    # by shape rather than broadcast into place downstream.
    soil_shape = (soil_layer_count(cfg), *horizontal_shape)
    host_fields = {}
    for name in sorted(_CANONICAL_SURFACE_FIELDS):
        value = fields[name]
        if hasattr(value, "get"):
            value = value.get()
        array = np.asarray(value)
        host_fields[name] = array
        expected = soil_shape if name in {"TSLB", "SMOIS", "SH2O"} \
            else horizontal_shape
        if array.shape != expected:
            raise ValueError(
                f"prepared surface {name} has shape {array.shape}, "
                f"expected {expected}")
        if (array.dtype.hasobject
                or not np.issubdtype(array.dtype, np.number)
                or not np.isfinite(array).all()):
            raise ValueError(
                f"prepared surface {name} must be finite numeric data")
    if not np.array_equal(
            host_fields["XLAND"],
            np.where(host_fields["LANDMASK"] >= 0.5, 1.0, 2.0)):
        raise ValueError(
            "prepared surface XLAND differs from its canonical LANDMASK")
    return fields


def _validate_prepared_near_surface(result, met, cfg) -> Mapping[str, np.ndarray]:
    """Validate exact, physical near-surface arrays before GPU allocation."""

    try:
        met_fields = met.fields
    except AttributeError as exc:
        raise TypeError(
            "prepared physics meteorology must expose a fields mapping") from exc
    if not isinstance(met_fields, Mapping):
        raise TypeError("prepared physics meteorology fields must be a mapping")
    ny, nx = int(cfg.ny), int(cfg.nx)
    specs = {
        "surface_pressure": (result.surface_pressure, (ny, nx), 1_000.0, 120_000.0),
        "surface_qv": (result.surface_qv, (ny, nx), 0.0, 0.2),
        "T2": (met_fields["T2"], (ny, nx), 100.0, 400.0),
        "SKINTEMP": (met_fields["SKINTEMP"], (ny, nx), 100.0, 400.0),
        "U10": (met_fields["U10"], (ny, nx + 1), -250.0, 250.0),
        "V10": (met_fields["V10"], (ny + 1, nx), -250.0, 250.0),
        "LANDSEA": (met_fields["LANDSEA"], (ny, nx), 0.0, 1.0),
    }
    validated = {}
    for name, (value, expected_shape, lower, upper) in specs.items():
        if hasattr(value, "get"):
            value = value.get()
        array = np.asarray(value)
        if array.shape != expected_shape:
            raise ValueError(
                f"prepared near-surface {name} has shape {array.shape}, "
                f"expected {expected_shape}")
        if (array.dtype.hasobject
                or not np.issubdtype(array.dtype, np.number)
                or not np.isfinite(array).all()):
            raise ValueError(
                f"prepared near-surface {name} must be finite numeric data")
        if array.min() < lower or array.max() > upper:
            # Name the numbers.  This refusal used to say only which field
            # and which range, and a field report ("surface_qv is outside
            # the physical range 0.0..0.2") cost a full re-run of a
            # two-stage preparation to learn that the excursion was two
            # cells at -1.8e-05 rather than, say, a unit error.  The
            # observed extremes and the offending-cell count are already
            # in hand here and decide which of those it is on sight.
            outside = int(np.count_nonzero(
                (array < lower) | (array > upper)))
            raise ValueError(
                f"prepared near-surface {name} is outside the physical "
                f"range {lower}..{upper}: {outside} of {array.size} "
                f"cell(s) are out, observed range "
                f"{float(array.min()):.6g}..{float(array.max()):.6g}")
        validated[name] = array
    if not np.isin(validated["LANDSEA"], (0.0, 1.0)).all():
        raise ValueError("prepared near-surface LANDSEA must be exactly binary")
    return MappingProxyType(validated)


def initialize_prepared_physics(
        result, cfg, met, surface, static, landuse_attrs, grid, valid_time, *,
        center_lat=None):
    """Attach physics to one restored source-neutral prepared initial state.

    ``surface`` is the canonical Noah inventory persisted with direct GFS and
    ERA5 caches.  Every field is validated before device allocation.  The
    actual land-use, diagnostics, physics-driver, and near-surface setup is the
    same implementation used by the native HRRR runner.
    """

    import cupy as cp

    from gpuwm.core.diagnostics import update_diagnostics
    from gpuwm.core.landuse import initialize_landuse
    from gpuwm.core.physics import initialize_physics
    from gpuwm.static.build import monthly_interp_to_date

    fields = _validate_prepared_surface(surface, cfg)
    near_surface_host = _validate_prepared_near_surface(result, met, cfg)
    required_attrs = {"MMINLU", "ISWATER", "ISLAKE", "ISICE"}
    if not isinstance(landuse_attrs, Mapping) \
            or set(landuse_attrs) != required_attrs:
        raise ValueError(
            "prepared physics land-use identity must contain exactly "
            f"{sorted(required_attrs)}")
    state = result.state
    update_diagnostics(state, cfg.hypsometric_opt)
    landuse = initialize_landuse(
        static["LU_INDEX"], soil_type=static["SCT_DOM"],
        landmask=static["LANDMASK"], snow=fields["SNOW"],
        xice=fields["SEAICE"], valid_time=valid_time,
        cen_lat=float(
            getattr(grid, "cen_lat", grid.ref_lat)
            if center_lat is None else center_lat),
        mminlu=str(landuse_attrs["MMINLU"]),
        iswater=int(landuse_attrs["ISWATER"]),
        islake=int(landuse_attrs["ISLAKE"]),
        isice=int(landuse_attrs["ISICE"]), fractional_seaice=True,
        # real.exe's landmask/soil-category reconciliation decides a
        # disagreeing column from its soil temperature, then its SST.
        soil_temperature=fields["TSLB"], sst=fields.get("SST"))
    vegfra = 100.0 * monthly_interp_to_date(static["GREENFRAC"], valid_time)
    lai = monthly_interp_to_date(static["LAI12M"], valid_time)
    lat, lon = grid.latlon_mass()
    driver = initialize_physics(
        state, cfg, landuse=landuse, tsk=fields["TSK"],
        soil_temperature=fields["TSLB"],
        soil_moisture=fields["SMOIS"],
        liquid_moisture=fields["SH2O"],
        ivgtyp=static["LU_INDEX"], isltyp=static["SCT_DOM"],
        vegfra=vegfra, tmn=fields["TMN"], xice=fields["SEAICE"],
        snow=fields["SNOW"], snow_depth=fields["SNOWH"],
        sst=fields.get("SST", fields["TSK"]),
        radiation_start_time=valid_time, radiation_latitude=lat,
        radiation_longitude=lon)
    driver.fields["snoalb"][...] = cp.asarray(
        noah_initial_snow_albedo(
            static["SNOALB"], static["LU_INDEX"], driver.noah_params,
            rdmaxalb=cfg.rdmaxalb),
        dtype=cp.float32)
    driver.fields["lai"][...] = cp.asarray(lai, dtype=cp.float32)
    driver.fields["shdmin"][...] = cp.asarray(
        100.0 * static["GREENFRAC"].min(axis=0), dtype=cp.float32)
    driver.fields["shdmax"][...] = cp.asarray(
        100.0 * static["GREENFRAC"].max(axis=0), dtype=cp.float32)
    driver.fields["psfc"][...] = cp.asarray(
        near_surface_host["surface_pressure"], dtype=cp.float32)
    near_surface = surface_fields_to_device(
        SimpleNamespace(fields=near_surface_host), cp)
    driver.fields["t2"][...] = near_surface["T2"]
    driver.fields["q2"][...] = cp.asarray(
        near_surface_host["surface_qv"], dtype=cp.float32)
    driver.fields["th2"][...] = (
        driver.fields["t2"]
        * (cp.float32(100000.0) / driver.fields["psfc"])
        ** cp.float32(287.0 / 1004.0))
    driver.fields["u10"][...] = 0.5 * (
        near_surface["U10"][:, :-1] + near_surface["U10"][:, 1:])
    driver.fields["v10"][...] = 0.5 * (
        near_surface["V10"][:-1] + near_surface["V10"][1:])
    return driver


def initialize_hrrr_physics(
        result, cfg, met, static, attrs, grid, valid_time):
    """Initialize physics, accepting either host or device ingestion fields."""

    from types import MappingProxyType, SimpleNamespace

    from gpuwm.ingest.ruc_soil import preprocess_land_surface_soil
    from gpuwm.native_wrf_contract import canonical_noah_surface

    soil = preprocess_land_surface_soil(
        met.fields, sf_surface_physics=int(cfg.sf_surface_physics),
        soil_type=static["SCT_DOM"],
        deep_soil_temperature=static["SOILTEMP"])
    surface = SimpleNamespace(fields=MappingProxyType(
        canonical_noah_surface(soil)))
    landuse_attrs = {
        name: attrs[name] for name in ("MMINLU", "ISWATER", "ISLAKE", "ISICE")
    }
    return initialize_prepared_physics(
        result, cfg, met, surface, static, landuse_attrs, grid, valid_time,
        center_lat=attrs["CEN_LAT"])


__all__ = ["initialize_hrrr_physics", "initialize_prepared_physics"]
