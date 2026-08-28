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

#: The native soil pair a met snapshot carries when nothing has solved
#: the surface yet.  Named here because the prepared-cache writer DROPS
#: it precisely when a solved surface rides along
#: (:func:`gpuwm.ingest.prepared_cache._prepared_met_names`), so these
#: two names are the whole difference between the two roads below.
_NATIVE_SOIL_FIELDS = ("SOILT", "SOILW")


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
        center_lat=None, constant_glw_wm2=None):
    """Attach physics to one restored source-neutral prepared initial state.

    ``surface`` is the canonical Noah inventory persisted with direct GFS and
    ERA5 caches.  Every field is validated before device allocation.  The
    actual land-use, diagnostics, physics-driver, and near-surface setup is the
    same implementation used by the native HRRR runner.

    ``constant_glw_wm2`` is the DECLARED constant downward longwave, for a
    suite that runs a land-surface scheme with ``ra_lw_physics = 0``.  The
    caller reads it off the experiment's acknowledgements
    (:func:`gpuwm.runtime.declared_constant_glw`); ``None`` -- the normal
    answer -- lets the attached longwave scheme own the field, and makes
    :func:`~gpuwm.core.physics.initialize_physics` refuse an undeclared
    suite that has no longwave scheme at all.
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
        glw=constant_glw_wm2,
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


def resolve_prepared_noah_surface(met, cfg, static, *, surface=None):
    """The canonical Noah surface this state initializes its land model from.

    Two roads arrive here holding the same forecast, and they hold
    DIFFERENT halves of the soil statement.

    The FRESH road has just decoded the source: its met snapshot carries
    the native ``SOILT``/``SOILW`` node column and nothing has solved a
    surface yet, so the surface is derived here
    (``preprocess_land_surface_soil`` -> ``canonical_noah_surface``).

    The RESTORE road holds a prepared cache, and the solved surface is
    exactly what that cache persists -- ``surface/TSLB``, ``SMOIS``,
    ``SH2O`` and the rest of the canonical Noah inventory.  Because the
    surface is stored, the writer deliberately DROPS the native pair from
    the met contract beside it
    (:func:`gpuwm.ingest.prepared_cache._prepared_met_names`: the pair is
    required only when ``surface is None``).  So a restore has the
    answer and not the ingredients.

    Re-deriving on the restore road is therefore not a slow path, it is a
    crash: the soil router, finding no native pair and no mapped or GFS
    markers, falls through to its ERA5-named arm and dies on ``missing
    soil input field(s): ['ST000007', ...]`` -- four frames deep, on a
    cache that was complete.  MEASURED 2026-08-18 on the Linux shakeout's
    prepared-cache restore arm, where the solved soil was sitting in
    ``surface/`` the whole time.

    ``surface`` is therefore consumed when the caller has one and derived
    only when it does not.  A state holding NEITHER is refused here by
    name rather than downstream by KeyError.
    """

    from types import MappingProxyType, SimpleNamespace

    if surface is not None:
        return surface

    from gpuwm.ingest.ruc_soil import preprocess_land_surface_soil
    from gpuwm.native_wrf_contract import canonical_noah_surface

    try:
        fields = met.fields
    except AttributeError as exc:
        raise TypeError(
            "prepared physics meteorology must expose a fields mapping"
        ) from exc
    absent = [name for name in _NATIVE_SOIL_FIELDS if name not in fields]
    if absent:
        raise ValueError(
            "this prepared state carries no solved Noah surface and no "
            f"native soil to derive one from ({', '.join(absent)} absent "
            "from its met fields), so the land model has no soil state to "
            "start from and the forecast would run on fabricated ground.\n"
            "  A prepared cache written by this release stores the solved "
            "surface (surface/TSLB, SMOIS, SH2O and the rest of the "
            "canonical Noah inventory) and drops the native SOILT/SOILW "
            "pair, so a restore must pass restore_prepared_cache(...)"
            ".surface through to this call rather than re-deriving.\n"
            "  A cache carrying neither is incomplete and must be prepared "
            "again; nothing here can reconstruct soil that was never "
            "written.")
    soil = preprocess_land_surface_soil(
        fields, sf_surface_physics=int(cfg.sf_surface_physics),
        # The RESOLVED count, the same number :func:`_soil_shape` above
        # allocates from.  Left off, RUC's soil ingest took its own
        # nine-level default, so a six-level config prepared a nine-level
        # surface and met a six-level allocation -- a shape error four
        # frames down instead of a forecast, which is the difference
        # between "six levels is unverified" and "six levels is
        # unreachable on this route".
        num_soil_layers=soil_layer_count(cfg),
        soil_type=static["SCT_DOM"],
        deep_soil_temperature=static["SOILTEMP"])
    return SimpleNamespace(fields=MappingProxyType(
        canonical_noah_surface(soil)))


def initialize_hrrr_physics(
        result, cfg, met, static, attrs, grid, valid_time, *,
        constant_glw_wm2=None, surface=None):
    """Initialize physics, accepting either host or device ingestion fields.

    ``surface`` is the solved Noah surface when the caller already holds
    one -- what :func:`gpuwm.ingest.prepared_cache.restore_prepared_cache`
    hands back.  Omitted, the surface is derived from the met snapshot's
    native soil, which is what a freshly decoded state carries.  See
    :func:`resolve_prepared_noah_surface` for why the two roads differ.
    """

    surface = resolve_prepared_noah_surface(met, cfg, static, surface=surface)
    landuse_attrs = {
        name: attrs[name] for name in ("MMINLU", "ISWATER", "ISLAKE", "ISICE")
    }
    return initialize_prepared_physics(
        result, cfg, met, surface, static, landuse_attrs, grid, valid_time,
        center_lat=attrs["CEN_LAT"], constant_glw_wm2=constant_glw_wm2)


__all__ = ["initialize_hrrr_physics", "initialize_prepared_physics",
           "resolve_prepared_noah_surface"]
