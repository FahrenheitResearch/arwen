"""WRF ``module_soil_pre.F`` preprocessing for Noah's four soil layers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from gpuwm.ingest.quantization import clamp_bound_kissing
from gpuwm.ingest.soil_contract import (
    MAPPED_SOIL_MOISTURE,
    MAPPED_SOIL_TEMPERATURE,
    conservative_overlap_weights,
    soil_layer_bounds,
    validate_soil_layer_contract,
)


ERA5_LAYER_BOTTOMS_M = np.array([0.07, 0.28, 1.00, 2.89], dtype=np.float64)
# WRF vertical nodes for layer-form soil input are the INTEGER-centimetre
# layer midpoints: module_optional_input.F:char2int2 computes
# (top+bottom)/2 in whole cm -- (0+7)/2=3, (7+28)/2=17, (28+100)/2=64,
# (100+289)/2=194 -- and init_soil_2_real stacks them between TSK at 0 m
# and TMN at 3 m (module_soil_pre.F:1591-1595).
ERA5_LAYER_MIDPOINTS_M = np.array([0.03, 0.17, 0.64, 1.94], dtype=np.float64)
NOAH_LAYER_THICKNESS_M = np.array([0.10, 0.30, 0.60, 1.00], dtype=np.float64)
NOAH_LAYER_MIDPOINTS_M = np.array([0.05, 0.25, 0.70, 1.50], dtype=np.float64)
HRRR_SOIL_NODE_DEPTHS_M = np.array(
    [0.0, 0.01, 0.04, 0.10, 0.30, 0.60, 1.0, 1.6, 3.0],
    dtype=np.float64,
)


def _host(value) -> np.ndarray:
    if hasattr(value, "get"):
        value = value.get()
    return np.asarray(value, dtype=np.float64)


@dataclass(frozen=True)
class NoahSoilState:
    """Setup-time, float64 Noah surface/soil initial conditions."""

    soil_temperature: np.ndarray  # (4,ny,nx), WRF TSLB
    soil_moisture: np.ndarray     # (4,ny,nx), WRF SMOIS
    liquid_moisture: np.ndarray   # (4,ny,nx), WRF SH2O
    deep_soil_temperature: np.ndarray  # (ny,nx), WRF TMN
    tsk: np.ndarray               # (ny,nx)
    landmask: np.ndarray          # WPS 1 land / 0 water
    xland: np.ndarray             # WRF 1 land / 2 water
    xice: np.ndarray              # ERA5 sea-ice fraction (zero when absent)
    snow_water: np.ndarray        # kg m-2
    snow_depth: np.ndarray        # m


_TEMP_NAMES = ("ST000007", "ST007028", "ST028100", "ST100289")
_MOIST_NAMES = ("SM000007", "SM007028", "SM028100", "SM100289")
_GFS_TEMP_NAMES = (
    "GFS_ST000010", "GFS_ST010040", "GFS_ST040100", "GFS_ST100200")
_GFS_MOIST_NAMES = (
    "GFS_SM000010", "GFS_SM010040", "GFS_SM040100", "GFS_SM100200")


def _require_same_shape(fields: Mapping[str, object], names) -> tuple[int, int]:
    missing = [name for name in names if name not in fields]
    if missing:
        raise KeyError(f"missing soil input field(s): {missing}")
    shapes = {_host(fields[name]).shape for name in names}
    if len(shapes) != 1:
        raise ValueError(f"soil input shapes differ: {sorted(shapes)}")
    shape = next(iter(shapes))
    if len(shape) != 2:
        raise ValueError("soil input fields must be 2-D")
    return shape


def _interp_nodes(nodes, zsource, ztarget):
    out = []
    for z in ztarget:
        lower = int(np.searchsorted(zsource, z) - 1)
        weight = (z - zsource[lower]) / (zsource[lower + 1] - zsource[lower])
        out.append(nodes[lower] + weight * (nodes[lower + 1] - nodes[lower]))
    return np.stack(out)


def _remap_declared_soil(
    temperature: np.ndarray,
    moisture: np.ndarray,
    contract: Mapping[str, object],
    *,
    tsk: np.ndarray,
    deep: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply one already validated source-independent soil remap."""

    source = soil_layer_bounds(contract, "source_layers")
    target = soil_layer_bounds(contract, "target_layers")
    remap = contract["remap"]
    if not isinstance(remap, Mapping):  # guarded by validation; defense in depth
        raise TypeError("soil remap must be an object")
    if remap["kind"] == "linear_point_samples":
        # WRF places layer-form soil values at the INTEGER-centimetre layer
        # midpoints (module_optional_input.F:char2int2, (top+bottom)/2 in
        # whole cm), bracketed by TSK at 0 m and TMN at 3 m
        # (module_soil_pre.F:1591-1595).
        source_depths = np.asarray(
            [0.0,
             *(((int(round(top * 100.0)) + int(round(bottom * 100.0))) // 2)
               / 100.0 for top, bottom in source),
             3.0],
            dtype=np.float64,
        )
        target_depths = np.asarray(
            [(top + bottom) / 2.0 for top, bottom in target],
            dtype=np.float64,
        )
        temperature_nodes = np.concatenate(
            (tsk[None, ...], temperature, deep[None, ...]), axis=0,
        )
        moisture_nodes = np.concatenate(
            (moisture[:1], moisture, moisture[-1:]), axis=0,
        )
        return (
            _interp_nodes(temperature_nodes, source_depths, target_depths),
            _interp_nodes(moisture_nodes, source_depths, target_depths),
        )
    if remap["kind"] == "conservative_layer_means":
        # The checked-in GFS contract has the exact Noah bounds. Avoid a
        # matrix multiply in that common case so the former copy path remains
        # bit-for-bit identical.
        if source == target:
            return temperature.copy(), moisture.copy()
        weights = conservative_overlap_weights(source, target)
        return (
            np.tensordot(weights, temperature, axes=(1, 0)),
            np.tensordot(weights, moisture, axes=(1, 0)),
        )
    raise ValueError(f"unsupported soil remap kind {remap['kind']!r}")


def _soil_temperature_elevation_delta(terrain, source_orography, terrestrial):
    """WRF ``adjust_soil_temp_new`` lapse increment (module_soil_pre.F:993-1073).

    Land cells receive ``-0.0065 * (ter - toposoil)`` on TSK and every soil
    temperature input level.  WRF's sanity guards skip cells whose soil
    elevation is below -1000 m, above 10000 m, or more than 3000 m away
    from the model terrain.  Returns the additive delta field (zero where
    no adjustment applies).
    """
    ter = _host(terrain)
    toposoil = _host(source_orography)
    if ter.shape != toposoil.shape or ter.shape != terrestrial.shape:
        raise ValueError(
            "terrain/source_orography/landmask shapes differ for the "
            "soil-temperature elevation adjustment")
    difference = ter - toposoil
    usable = (terrestrial
              & (toposoil >= -1000.0) & (toposoil <= 10000.0)
              & (np.abs(difference) <= 3000.0))
    return np.where(usable, -0.0065 * difference, 0.0)


def preprocess_noah_soil(fields: Mapping[str, object], *, soil_type,
                         deep_soil_temperature=None, lake_mask=None,
                         lake_skin_temperature=None,
                         soil_layer_contract=None,
                         landmask=None,
                         terrain=None,
                         source_orography=None) -> NoahSoilState:
    """Map ERA5 layers, GFS Noah layers, or native HRRR depth nodes to Noah.

    This is ``module_soil_pre.F:init_soil_2_real`` with layer input:
    temperature is linearly interpolated through TSK at 0 m and the deep
    temperature at 3 m; moisture repeats its shallow/deep layer at 0/3 m.
    ``soil_type`` is WRF ISLTYP and drives the subsequent Noah LSMINIT
    frozen-water partition of SMOIS into SH2O.
    Open-water columns are filled with SST (or valid SKINTEMP fallback) and
    unit moisture exactly like the Fortran.  SEAICE/XICE is optional: values
    at or above Noah's 0.5 threshold are made land-like in XLAND so Noah's
    sea-ice branch is reachable.  WRF's LSMSCHEME sea-ice postprocessing is
    mirrored at init: binary XICE, TMN=271.4 K, a four-level 3 m ice-column
    TSLB profile, and SH2O=0.  Snow reconciliation is independent of surface
    type, as in ``module_initialize_real.F``.  Absence is exactly the
    historical all-zero XICE path.

    ``lake_mask`` and ``lake_skin_temperature`` are an all-or-none setup
    override for raw GEOG lakes.  Such cells are water even when coarse ERA5
    LANDSEA calls them land, and their SKINTEMP is the separately selected
    nearest finite source-water value.  Non-lake source land/ocean behavior
    remains byte-identical.

    ``landmask`` selects the terrestrial/water decision surface.  WRF's
    ``process_soil_real`` drives every land/water branch with the geogrid
    ``grid%landmask``, never the met-source LANDSEA; passing the static-build
    LANDMASK here reproduces that and keeps the soil classification
    consistent with the physics driver's mask.  ``None`` retains the
    historical met-source LANDSEA decision for adapters that have not yet
    declared their static mask.

    ``terrain`` and ``source_orography`` (all-or-none) enable WRF's
    ``adjust_soil_temp_new`` elevation lapse: land skin and soil temperature
    inputs are shifted by ``-0.0065 * (terrain - source_orography)`` before
    the vertical mapping, exactly as ``process_soil_real`` adjusts TSK and
    ``st_input`` when the met source declares its own orography.  The 3 m
    deep temperature is NOT adjusted here: WRF's Noah branch subtracts
    ``0.0065 * terrain`` from the sea-level annual mean instead, which the
    static build already bakes into TMN.
    """
    mapped_markers = (MAPPED_SOIL_TEMPERATURE, MAPPED_SOIL_MOISTURE)
    mapped_marker_count = sum(name in fields for name in mapped_markers)
    if soil_layer_contract is None:
        if mapped_marker_count:
            raise ValueError(
                "mapped soil arrays require an explicit soil_layer_contract"
            )
        mapped_layers = False
        declared_contract = None
    else:
        declared_contract = validate_soil_layer_contract(soil_layer_contract)
        if mapped_marker_count != len(mapped_markers):
            raise KeyError(
                "declarative mapped soil input requires temperature and "
                "moisture arrays together"
            )
        mapped_layers = True

    hrrr_markers = ("SOILT", "SOILW")
    hrrr_marker_count = sum(name in fields for name in hrrr_markers)
    if hrrr_marker_count not in (0, len(hrrr_markers)):
        raise KeyError(
            "HRRR soil input requires SOILT and SOILW together")
    hrrr_nodes = hrrr_marker_count == len(hrrr_markers)
    gfs_markers = (*_GFS_TEMP_NAMES, *_GFS_MOIST_NAMES)
    gfs_marker_count = sum(name in fields for name in gfs_markers)
    if gfs_marker_count not in (0, len(gfs_markers)):
        raise KeyError(
            "GFS soil input requires all four temperature and moisture layers")
    gfs_layers = gfs_marker_count == len(gfs_markers)
    legacy_present = hrrr_nodes or gfs_layers or any(
        name in fields for name in (*_TEMP_NAMES, *_MOIST_NAMES)
    )
    if sum((mapped_layers, hrrr_nodes, gfs_layers)) > 1 \
            or (mapped_layers and legacy_present):
        raise ValueError("declarative, HRRR, GFS, and ERA5 soil modes cannot be mixed")
    if mapped_layers:
        shape = _require_same_shape(fields, ("LANDSEA", "SKINTEMP"))
        declared_temperature = _host(fields[MAPPED_SOIL_TEMPERATURE])
        declared_moisture = _host(fields[MAPPED_SOIL_MOISTURE])
        source_count = len(declared_contract["source_layers"])
        expected_shape = (source_count,) + shape
        if declared_temperature.shape != expected_shape \
                or declared_moisture.shape != expected_shape:
            raise ValueError(
                "declarative mapped soil arrays must have shape "
                f"{expected_shape}"
            )
    elif hrrr_nodes:
        shape = _require_same_shape(fields, ("LANDSEA", "SKINTEMP"))
        soil_temperature_nodes = _host(fields["SOILT"])
        soil_moisture_nodes = _host(fields["SOILW"])
        expected_node_shape = (HRRR_SOIL_NODE_DEPTHS_M.size,) + shape
        if (soil_temperature_nodes.shape != expected_node_shape
                or soil_moisture_nodes.shape != expected_node_shape):
            raise ValueError(
                "HRRR SOILT/SOILW must have shape "
                f"{expected_node_shape}")
        if (not np.isfinite(soil_temperature_nodes).all()
                or np.any((soil_temperature_nodes < 170.0)
                          | (soil_temperature_nodes > 400.0))):
            raise ValueError("HRRR SOILT nodes are outside 170..400 K")
        # Saturated soil is stored AT 1.0 and decodes a hair above it;
        # that is the decode rounding, not a broken node.
        soil_moisture_nodes, _ = clamp_bound_kissing(
            soil_moisture_nodes, minimum=0.0, maximum=1.0)
        if (not np.isfinite(soil_moisture_nodes).all()
                or np.any((soil_moisture_nodes < 0.0)
                          | (soil_moisture_nodes > 1.0))):
            raise ValueError("HRRR SOILW nodes are outside 0..1")
    elif gfs_layers:
        shape = _require_same_shape(
            fields, ("LANDSEA", "SKINTEMP", *gfs_markers))
    else:
        shape = _require_same_shape(
            fields, ("LANDSEA", "SKINTEMP", *_TEMP_NAMES, *_MOIST_NAMES))
    if landmask is not None:
        decision = _host(landmask)
        if decision.shape != shape:
            raise ValueError("landmask shape differs from soil fields")
        if (not np.isfinite(decision).all()
                or np.any((decision != 0.0) & (decision != 1.0))):
            raise ValueError("landmask must contain only boolean/0/1 values")
        terrestrial = decision >= 0.5
    else:
        terrestrial = _host(fields["LANDSEA"]) >= 0.5
    skin = _host(fields["SKINTEMP"]).copy()
    sst = _host(fields.get("SST", skin))
    if any(value.shape != shape for value in (terrestrial, skin, sst)):
        raise ValueError("surface and soil input shapes differ")
    if (lake_mask is None) != (lake_skin_temperature is None):
        raise ValueError(
            "lake_mask and lake_skin_temperature must be provided together")
    if lake_mask is not None:
        raw_lakes = _host(lake_mask)
        lake_temperature = _host(lake_skin_temperature)
        if raw_lakes.shape != shape or lake_temperature.shape != shape:
            raise ValueError("lake surface override shape differs from soil fields")
        if (not np.isfinite(raw_lakes).all()
                or np.any((raw_lakes != 0.0) & (raw_lakes != 1.0))):
            raise ValueError("lake_mask must contain only boolean/0/1 values")
        lakes = raw_lakes.astype(bool)
        if (not np.isfinite(lake_temperature[lakes]).all()
                or np.any((lake_temperature[lakes] < 170.0)
                          | (lake_temperature[lakes] > 400.0))):
            raise ValueError(
                "lake_skin_temperature is non-finite or outside 170..400 K")
        terrestrial = terrestrial.copy()
        terrestrial[lakes] = False
        skin[lakes] = lake_temperature[lakes]
    if (terrain is None) != (source_orography is None):
        raise ValueError(
            "terrain and source_orography must be provided together for the "
            "soil-temperature elevation adjustment")
    if terrain is not None:
        elevation_delta = _soil_temperature_elevation_delta(
            terrain, source_orography, terrestrial)
        skin = skin + elevation_delta
    else:
        elevation_delta = None
    raw_xice = fields.get("XICE", fields.get("SEAICE"))
    if raw_xice is None:
        xice = np.zeros(shape, dtype=np.float64)
    else:
        xice = _host(raw_xice)
        if xice.shape != shape or not np.isfinite(xice).all():
            raise ValueError("XICE must be a finite sea-ice fraction in [0, 1]")
        xice = xice.copy()
        # share/module_soil_pre.F:95-100 repairs GRIB flag values before
        # applying the physical fraction checks.
        xice[xice > 200.0] = 0.0
        if np.any((xice < 0.0) | (xice > 1.0)):
            raise ValueError("XICE must be a finite sea-ice fraction in [0, 1]")
    # Land cannot simultaneously be sea ice.  At the Noah threshold, sea ice
    # must be XLAND=1; otherwise the driver's earlier open-water return masks
    # the xice branch.
    xice[terrestrial] = 0.0
    sea_ice = (~terrestrial) & (xice >= 0.5)
    # fractional_seaice=0: adjust_for_seaice_post snaps ice to one and
    # removes sub-threshold fractions (module_soil_pre.F:258-262,301-302).
    xice[sea_ice] = 1.0
    xice[(~terrestrial) & (~sea_ice)] = 0.0
    effective_land = terrestrial | sea_ice
    landmask = effective_land.astype(np.float64)
    water_temperature = np.where(np.isfinite(sst) & (sst >= 170.0)
                                 & (sst <= 400.0), sst, skin)
    tsk = np.where(terrestrial | sea_ice, skin, water_temperature)
    if not np.isfinite(tsk).all() or np.any((tsk < 170.0) | (tsk > 400.0)):
        raise ValueError("TSK contains non-finite or nonphysical values")

    if mapped_layers:
        temperatures = []
        moistures = []
    elif hrrr_nodes:
        temperatures = []
        moistures = []
    elif gfs_layers:
        temperatures = [_host(fields[name]) for name in _GFS_TEMP_NAMES]
        moistures = [_host(fields[name]) for name in _GFS_MOIST_NAMES]
    else:
        temperatures = [_host(fields[name]) for name in _TEMP_NAMES]
        moistures = [_host(fields[name]) for name in _MOIST_NAMES]
    if elevation_delta is not None:
        # adjust_soil_temp_new applies the same lapse increment to every
        # soil temperature input level (module_soil_pre.F:1059-1067),
        # whether layer-form, level-form, or declaratively mapped.
        temperatures = [value + elevation_delta for value in temperatures]
        if hrrr_nodes:
            soil_temperature_nodes = soil_temperature_nodes + elevation_delta
        if mapped_layers:
            declared_temperature = declared_temperature + elevation_delta
    if deep_soil_temperature is None:
        if "TMN" not in fields:
            raise KeyError("missing required 3 m deep-soil temperature field: TMN")
        deep_input = _host(fields["TMN"])
    else:
        deep_input = _host(deep_soil_temperature)
    if deep_input.shape != shape:
        raise ValueError("deep_soil_temperature shape differs from soil fields")
    deep = deep_input
    land = terrestrial
    valid_deep = np.isfinite(deep) & (deep >= 170.0) & (deep <= 400.0)
    # module_initialize_real.F repairs an unreasonable land TMN from TSK and
    # sets water TMN to the selected SST/TSK before module_soil_pre consumes it.
    deep = np.where(land & valid_deep, deep, tsk)
    if mapped_layers:
        # Source-land gaps were rejected before horizontal interpolation.
        # At this post-interpolation stage, only target-ocean values may be
        # absent; the declared repair below replaces them. Target-land gaps
        # and nonphysical values remain fatal.
        # Same admission on the declarative route: a mapped saturated
        # cell reaches here one rounding step above 1.0 for exactly the
        # reasons the GFS bridge now clamps for.
        declared_moisture, _ = clamp_bound_kissing(
            declared_moisture, minimum=0.0, maximum=1.0)
        land_temperature = declared_temperature[:, terrestrial]
        land_moisture = declared_moisture[:, terrestrial]
        if not np.isfinite(land_temperature).all() \
                or np.any((land_temperature < 170.0) | (land_temperature > 400.0)):
            raise ValueError(
                "declarative mapped soil temperature is missing or outside "
                "170..400 K on land"
            )
        if not np.isfinite(land_moisture).all() \
                or np.any((land_moisture < 0.0) | (land_moisture > 1.0)):
            raise ValueError(
                "declarative mapped soil moisture is missing or outside 0..1 on land"
            )
        soil_t, soil_m = _remap_declared_soil(
            declared_temperature,
            declared_moisture,
            declared_contract,
            tsk=tsk,
            deep=deep,
        )
    elif hrrr_nodes:
        # HRRR supplies true depth nodes, including both 0 and 3 m.  Noah's
        # four midpoint values therefore come directly from WRF's sorted
        # linear node interpolation; the synthetic endpoint extension used
        # for ERA5 layers cannot affect these interior target depths.
        soil_t = _interp_nodes(
            soil_temperature_nodes, HRRR_SOIL_NODE_DEPTHS_M,
            NOAH_LAYER_MIDPOINTS_M)
        soil_m = _interp_nodes(
            soil_moisture_nodes, HRRR_SOIL_NODE_DEPTHS_M,
            NOAH_LAYER_MIDPOINTS_M)
    elif gfs_layers:
        # GFS supplies the exact four Noah slabs (0-10, 10-40, 40-100,
        # 100-200 cm).  They are layer values, not depth nodes, so copying is
        # the scientifically correct mapping and avoids ERA5 interpolation.
        soil_t = np.stack(temperatures)
        soil_m = np.stack(moistures)
    else:
        # init_soil_2_real (module_soil_pre.F:1591-1608): layer values sit at
        # WRF's integer-cm layer midpoints, bracketed by TSK at 0 m and TMN
        # at 3 m; moisture repeats its shallow/deep layer at the endpoints.
        zsource = np.concatenate(([0.0], ERA5_LAYER_MIDPOINTS_M, [3.0]))
        temp_nodes = np.stack([tsk, *temperatures, deep])
        moist_nodes = np.stack([moistures[0], *moistures, moistures[-1]])
        soil_t = _interp_nodes(temp_nodes, zsource, NOAH_LAYER_MIDPOINTS_M)
        soil_m = _interp_nodes(moist_nodes, zsource, NOAH_LAYER_MIDPOINTS_M)
    non_terrestrial = ~terrestrial
    soil_t[:, non_terrestrial] = tsk[non_terrestrial]
    soil_m[:, non_terrestrial] = 1.0
    # LSMSCHEME adjust_for_seaice_post builds four equispaced layers through
    # a 3 m ice column (module_soil_pre.F:289-300).
    tmn = np.array(deep, dtype=np.float64, copy=True)
    tmn[sea_ice] = 271.4
    ice_midpoints = (np.arange(4, dtype=np.float64) + 0.5) * (3.0 / 4.0)
    for layer, midpoint in enumerate(ice_midpoints):
        soil_t[layer, sea_ice] = (
            (3.0 - midpoint) * tsk[sea_ice] + midpoint * tmn[sea_ice]
        ) / 3.0
    if (not np.isfinite(soil_t).all() or np.any((soil_t < 170.0) | (soil_t > 400.0))):
        raise ValueError("soil temperature is outside 170..400 K")
    # Sixteen-point source stencils overshoot the saturated ceiling
    # where source cells sit at exactly 1.0 next to dry land (GFS
    # glacier/ice at high latitude; module_soil_pre.F:298/:392 itself
    # assigns 1.0 over ice and water) -- observed up to ~1.086 on the
    # Fairbanks smoke domain.  Stock real.exe carries such columns with
    # no upper clamp at all (the > 1.005 guard at
    # module_initialize_real.F:3383 is commented out in the pinned
    # source), so cap bounded Lagrange overshoot at the physical
    # ceiling instead of refusing a domain stock WRF accepts.  Values
    # already inside [0, 1] are untouched (every previously passing
    # case is byte-identical).  The 0.25 band is far beyond what a
    # sixteen-point stencil can produce from a 0..1-ranged field yet
    # still catches genuinely broken inputs (fill values, unit errors).
    if not np.isfinite(soil_m).all() \
            or np.any((soil_m < -0.25) | (soil_m > 1.25)):
        bad = soil_m[~np.isfinite(soil_m) | (soil_m < -0.25)
                     | (soil_m > 1.25)]
        raise ValueError(
            "soil moisture is outside 0..1 beyond the interpolation-"
            f"overshoot band: {bad.size} value(s), range "
            f"[{np.nanmin(soil_m):.6g}, {np.nanmax(soil_m):.6g}]")
    soil_m = np.clip(soil_m, 0.0, 1.0)
    if not np.isfinite(tmn).all() or np.any((tmn < 170.0) | (tmn > 400.0)):
        raise ValueError("deep soil temperature is outside 170..400 K")

    from gpuwm.core.noah import load_tables, pack_params, sh2o_init

    soil_type = _host(soil_type)
    if soil_type.shape != shape:
        raise ValueError("soil_type shape differs from soil fields")
    liquid_m = sh2o_init(
        soil_m, soil_t, soil_type, pack_params(load_tables()))
    liquid_m[:, sea_ice] = 0.0

    # dyn_em/module_initialize_real.F:517-543 reconciles the independently
    # optional SNOW (kg m-2 SWE) and SNOWH (m physical depth) fields.  Its
    # fixed 5:1 liquid-to-snow depth ratio is a 200 kg m-3 initial density.
    # SNOW_EC is ERA5 metres water equivalent and therefore counts as SNOW.
    snow_present = "SNOW" in fields or "SNOW_EC" in fields
    snowh_present = "SNOWH" in fields
    if "SNOW" in fields:
        snow = _host(fields["SNOW"])
    elif "SNOW_EC" in fields:
        snow = 1000.0 * _host(fields["SNOW_EC"])
    else:
        snow = np.zeros(shape, dtype=np.float64)
    snowh = (_host(fields["SNOWH"]) if snowh_present
             else np.zeros(shape, dtype=np.float64))
    for name, value in (("snow water", snow), ("snow depth", snowh)):
        if (value.shape != shape or not np.isfinite(value).all()
                or np.any(value < 0.0)):
            raise ValueError(
                f"{name} must be a finite non-negative 2-D field")
    if not snow_present and snowh_present:
        snow = snowh * (1000.0 / 5.0)
    elif snow_present and not snowh_present:
        snowh = snow / 1000.0 * 5.0
    return NoahSoilState(
        soil_temperature=soil_t,
        soil_moisture=soil_m,
        liquid_moisture=liquid_m,
        deep_soil_temperature=tmn,
        tsk=tsk,
        landmask=landmask,
        xland=np.where(effective_land, 1.0, 2.0),
        xice=xice,
        snow_water=snow,
        snow_depth=snowh,
    )


__all__ = ["ERA5_LAYER_BOTTOMS_M", "HRRR_SOIL_NODE_DEPTHS_M",
           "NOAH_LAYER_MIDPOINTS_M",
           "NOAH_LAYER_THICKNESS_M", "NoahSoilState", "preprocess_noah_soil"]
