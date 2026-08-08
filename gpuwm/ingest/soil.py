"""WRF ``module_soil_pre.F`` preprocessing for Noah's four soil layers."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
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


def _nonphysical_tsk_message(tsk: np.ndarray, land: np.ndarray) -> str:
    """Name the cells, the split, and the one fill value that causes this.

    ``module_initialize_real.F:3278-3296`` prints the offending cell and
    then SUBSTITUTES TMN, or SST, before its ``grid%tsk unreasonable``
    abort.  gpuwm refuses instead: a deep-soil temperature standing in for
    a skin temperature is a silent 3 m-depth initial condition at the
    surface, and the substitution hides the input gap that produced it.
    The divergence is deliberate; this message carries the diagnosis the
    WRF print carries, for every offending cell at once.
    """
    bad = ~np.isfinite(tsk) | (tsk < 170.0) | (tsk > 400.0)
    total = int(bad.sum())
    on_land = int((bad & land).sum())
    on_water = total - on_land
    finite = tsk[bad & np.isfinite(tsk)]
    detail = ""
    if finite.size:
        values = np.unique(finite)
        shown = ", ".join(f"{value:g}" for value in values[:4])
        detail = (f"; values {shown}"
                  + ("..." if values.size > 4 else ""))
        if values.size == 1 and values[0] == 0.0:
            detail += (
                ".  0 K is METGRID.TBL fill_missing for SKINTEMP, which "
                "means the masked interpolation found no usable source "
                "cell on that surface -- check that the forcing's "
                "land-sea mask actually resolves the land this domain "
                "resolves")
    return (
        f"TSK contains non-finite or nonphysical values: {total} cell(s) "
        f"outside 170..400 K ({on_land} on land/sea-ice, {on_water} on "
        f"open water) of {tsk.size}{detail}")


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
    #: Ingest-repair receipt from :func:`_floor_land_moisture_at_smcdry`:
    #: per-SMOIS-level floored-cell counts and pre-floor minima.  An EMPTY
    #: mapping whenever nothing was floored, so healthy preparations carry
    #: zero receipt noise and the presence of any key is itself the signal.
    moisture_floor: Mapping[str, object] = field(default_factory=dict)


_TEMP_NAMES = ("ST000007", "ST007028", "ST028100", "ST100289")
_MOIST_NAMES = ("SM000007", "SM007028", "SM028100", "SM100289")
_GFS_TEMP_NAMES = (
    "GFS_ST000010", "GFS_ST010040", "GFS_ST040100", "GFS_ST100200")
_GFS_MOIST_NAMES = (
    "GFS_SM000010", "GFS_SM010040", "GFS_SM040100", "GFS_SM100200")
#: The native-HRRR lane carries ONE stacked 3-D node column instead of a
#: per-layer name; its water cells are already filled with SKINTEMP by the
#: mapper (``gpuwm/ingest/hrrr.py``).
_HRRR_TEMP_NAME = "SOILT"

#: EVERY spelling of a source soil-temperature column, in lookup order, for
#: WRF-real's landmask/soil-category reconciliation
#: (:func:`gpuwm.core.landuse.reconciled_soil_category`).
#:
#: ONE table instead of an inline chain per call site, because the inline
#: chain has now been short by one spelling twice, and each time the symptom
#: was a hard ``mismatch_landmask_ivgtyp`` refusal on ordinary shoreline or
#: inland-water columns rather than anything that named a missing field:
#:
#: * 2026-08-06, native HRRR: ``SOILT`` was absent from the chain and a
#:   nested 1 km preparation aborted on 38 shoreline columns;
#: * 2026-08-08, nested GFS: ``GFS_ST000010`` was absent and a 3 km child
#:   aborted on 73 inland-water columns (reservoirs and rivers), which made
#:   nested-GFS preparation impossible for essentially any child holding
#:   inland water.
#:
#: A lane that adds a soil source adds its top-layer name HERE, once, and
#: every reconciler call site is current.  Order is by inventory rather than
#: preference: :func:`preprocess_noah_soil` refuses mixed soil modes, so at
#: most one of these names is ever present in a single field mapping.
SOIL_TEMPERATURE_RECONCILER_NAMES = (
    MAPPED_SOIL_TEMPERATURE,   # declarative mapped (rw-wps) sources
    _TEMP_NAMES[0],            # classic per-layer Vtable spelling
    _GFS_TEMP_NAMES[0],        # the GFS per-layer spelling
    _HRRR_TEMP_NAME,           # the native-HRRR stacked node column
)

#: The reconciler's SST evidence, in real.exe's own precedence.
#:
#: ``module_initialize_real.F:2844-2866``: where SST has no valid support
#: real.exe keeps exactly that column's SKINTEMP, so the mismatch pass at
#: ``:3608-3650`` reads a skin temperature there, not a hole.  The sibling
#: routes already do this -- ``gpuwm/ingest/hrrr_physics.py`` falls back to
#: ``TSK`` and ``preprocess_noah_soil`` below to ``SKINTEMP`` -- and this
#: table is those two spellings of the one fallback, met-source inventory
#: first, so a single call serves either inventory.
SST_RECONCILER_NAMES = ("SST", "SKINTEMP", "TSK")


def _first_present(fields: Mapping[str, object], names):
    for name in names:
        value = fields.get(name)
        if value is not None:
            return value
    return None


def reconciler_soil_temperature(fields: Mapping[str, object]):
    """This mapping's soil-temperature evidence, or ``None``.

    ``None`` means the mapping genuinely carries no soil column under ANY
    known spelling, which is the state WRF's third arm exists for; it must
    stay reachable, because a reconciliation with no evidence is a refusal
    and never a guess.
    """

    return _first_present(fields, SOIL_TEMPERATURE_RECONCILER_NAMES)


def reconciler_sst(fields: Mapping[str, object]):
    """This mapping's sea-surface evidence, or ``None``."""

    return _first_present(fields, SST_RECONCILER_NAMES)


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


#: How far below zero a bounded-stencil overshoot may carry a snow
#: field before it stops being an interpolation artifact, as a fraction
#: of that field's own positive maximum.  Mirrors the soil-moisture
#: overshoot band a few dozen lines below, and for the same reason: the
#: source fields are non-negative, the horizontal operators that carry
#: them to the model grid are not monotone, and a snow line is the
#: sharpest gradient either field has.
_SNOW_OVERSHOOT_FRACTION = 0.25


def _admitted_snow_field(name: str, value: np.ndarray, shape) -> np.ndarray:
    """Admit one snow field, repairing bounded overshoot at zero.

    Snow water and snow depth are physically non-negative, so a negative
    mapped value is never data: it is the horizontal interpolation
    operator overshooting across the snow line.  Refusing it refused the
    whole preparation -- a real nested HRRR domain over the mountainous
    west died on ONE cell of 88 844 at -4.9 cm of snow depth, beside a
    44.5 m maximum, with a message that named neither the field's
    numbers nor which of its three conditions had failed.

    So the physically impossible value is repaired to the only defined
    one and an overshoot far beyond what a bounded stencil can produce
    -- a fill value, a unit error, a broken decode -- still refuses,
    with the numbers in the sentence.  Fields already non-negative are
    untouched, so every previously passing case is byte-identical.
    """
    if value.shape != tuple(shape):
        raise ValueError(
            f"{name} must be a 2-D field shaped {tuple(shape)}, got "
            f"{value.shape}")
    if not np.isfinite(value).all():
        raise ValueError(
            f"{name} carries {int(np.count_nonzero(~np.isfinite(value)))} "
            f"non-finite value(s) of {value.size}")
    smallest = float(np.min(value))
    if smallest >= 0.0:
        return value
    floor = -_SNOW_OVERSHOOT_FRACTION * max(float(np.max(value)), 0.0)
    if smallest < floor:
        raise ValueError(
            f"{name} is negative beyond the interpolation-overshoot band: "
            f"{int(np.count_nonzero(value < 0.0))} value(s) of "
            f"{value.size}, most negative {smallest:.6g}, field maximum "
            f"{float(np.max(value)):.6g}")
    return np.maximum(value, 0.0)


#: DIVERGENCE, deliberate (no-inherited-bugs): sub-physical LAND soil
#: moisture is floored at the soil type's SMCDRY instead of WRF's
#: constant 0.005.  The full ledger entry is on
#: :func:`_floor_land_moisture_at_smcdry`.
#: WRF's own soil-type-blind floor for land soil moisture
#: (``dyn_em/module_initialize_real.F:3376``).  Used verbatim for land
#: cells whose soil category is water and therefore has no SMCDRY.
_WRF_ZERO_SOIL_MOISTURE = 0.005

_MOISTURE_FLOOR_WRF_REFERENCE = {
    "wrf_version": "v4.6.1",
    "wrf_citation": (
        "dyn_em/module_initialize_real.F:3363-3395 "
        "(account_for_zero_soil_moisture SELECT CASE :3363; "
        "CASE (LSMSCHEME, NOAHMPSCHEME) :3365; flag_soil_layers arm "
        ":3367-3395: condition :3371-3372, per-cell print :3373, "
        "whole-column reset to 0.005 :3376, total-count print :3393-3394)"),
    "wrf_behavior": (
        "real.exe: land cells (landmask>0.5, 170<TSLB(1)<400) whose TOP "
        "layer has SMOIS(1)<0.005 print 'bad soil moisture at i,j', reset "
        "the whole column to the constant 0.005, and print the total count"),
    "gpuwm_behavior": (
        "each layer of each land cell below the soil category's SMCDRY "
        "(SOILPARM.TBL DRYSMC; module_sf_noahlsm.F:2453) is floored to "
        "that SMCDRY; water, sea-ice, and healthy land values are "
        "byte-untouched"),
}


def _floor_land_moisture_at_smcdry(soil_m, pre_clip, terrestrial,
                                   soil_type, params):
    """Floor sub-air-dry LAND soil moisture at the category's SMCDRY.

    An ERA5 swvl value a hair below zero (GRIB packing, horizontal
    interpolation undershoot) is admitted by the overshoot band above and
    clipped to EXACTLY 0.0.  Noah's thermal conductivity then divides by
    SMC three times (kernels/noah.cu TDFCND: ``xunfroz = sh2o/smc`` 0/0,
    the ``ake`` divide, and ``powf(smcmax/smc, bexp)``), so ONE such land
    cell is NaN conductivity -> NaN ground heat flux -> NaN HFX and the
    run dies at step 0 blamed on the PBL scheme.  The threshold is a hard
    zero: 1e-12 survives the arithmetic, 0.0 does not -- but both are
    sub-physical, so both are floored.

    DIVERGENCE from WRF, deliberate (no-inherited-bugs framework):

    * **What WRF does** (v4.6.1 ``dyn_em/module_initialize_real.F``,
      ``account_for_zero_soil_moisture`` :3363, ``CASE (LSMSCHEME,
      NOAHMPSCHEME)`` :3365, layer-form arm :3367-3395): a land cell
      (``landmask>0.5``, ``170<TSLB(1)<400``) whose TOP layer has
      ``SMOIS(1) < 0.005`` gets its WHOLE column reset to the constant
      0.005 m3/m3, with a per-cell print and a total count.  The
      per-soil-type residual adjustment beside it (:3401, ``lqmi``) is
      commented out, so stock real.exe's floor is soil-type-blind.
    * **What gpuwm does**: floors each layer of each land cell below the
      soil category's SMCDRY (air-dry; ``SOILPARM.TBL`` DRYSMC, consumed
      as ``SMCDRY = DRYSMC(SOILTYP)`` in ``module_sf_noahlsm.F:2453``)
      to that SMCDRY, and records the per-level counts and pre-floor
      minima as a receipt on the returned state.
    * **Why**: WRF's 0.005 sits BELOW every land category's SMCDRY (sand
      is 0.010), inside the range where Noah's own direct evaporation
      ``SRATIO = (SMC-SMCDRY)/(SMCMAX-SMCDRY)``
      (``module_sf_noahlsm.F:1214``) goes negative, and WRF's trigger
      reads only the top layer, so a dry deep layer under a wet top
      layer passes stock real.exe unrepaired and still reaches TDFCND's
      divides.  SMCDRY is the smallest soil moisture Noah's physics
      treats as physical, and flooring there is what WRF's own
      preprocessing effectively achieves for the fatal exact-zero case.

    WRF's ``170<TSLB<400`` precondition is already guaranteed here: the
    caller refused any soil temperature outside 170..400 K before this
    function runs.  Cells whose category has no positive SMCDRY (the
    SOILPARM WATER row) or is outside the table are left untouched, so
    ``sh2o_init``'s own category refusal fires exactly as before.

    ``pre_clip`` is the moisture BEFORE the 0..1 clip, so the receipt's
    minima report what the source actually delivered (-1e-9, not the
    0.0 the clip made of it).  Returns ``(soil_m, receipt)``; when no
    cell needs flooring the input array is returned UNTOUCHED (same
    object, byte-identical) with an empty receipt.
    """
    from gpuwm.core.noah import SOIL_COLS

    categories = np.asarray(soil_type)
    in_table = (np.isfinite(categories)
                & (categories == np.floor(categories))
                & (categories >= 1) & (categories <= params.slcats))
    rows = np.where(in_table, categories, 1).astype(np.int64) - 1
    smcdry = params.soil[rows, SOIL_COLS.index("smcdry")]
    # A land cell whose SOIL CATEGORY is water (SOILPARM's WATER row,
    # DRYSMC 0.0) is not exotic: geogrid's landmask and its dominant soil
    # category are independent fields, and they disagree along coastlines
    # and around inland water in every domain.  Such a cell has no air-dry
    # value to floor at, and skipping it -- which this function used to do
    # -- leaves SMOIS at exactly 0.0 for Noah's TDFCND to divide by, which
    # is the very NaN this floor exists to prevent.  The comment that
    # justified skipping said ``sh2o_init``'s category refusal would catch
    # it; it does not.  That refusal fires only OUTSIDE the table
    # (core/noah.py: ``category < 1 or category > slcats``), and WATER is
    # inside it, so these cells sailed through to a step-0 non-finite HFX
    # blamed on the PBL scheme.
    #
    # WRF has no such hole, because its floor is soil-type-BLIND:
    # account_for_zero_soil_moisture repairs any land cell whose top layer
    # is below 0.005 without consulting the category at all
    # (module_initialize_real.F:3371-3376).  So the defined answer here is
    # WRF's own constant, applied exactly where gpuwm's category-aware
    # floor has nothing to say.  No new threshold is invented: real land
    # keeps SMCDRY, and category-less land gets the number stock real.exe
    # would have given the whole column.
    floor = np.where(in_table & (smcdry > 0.0),
                     smcdry, _WRF_ZERO_SOIL_MOISTURE)
    usable = np.asarray(terrestrial, dtype=bool)
    needs = usable[None, ...] & (soil_m < np.where(usable, floor, 0.0))
    if not needs.any():
        return soil_m, {}
    categorical = in_table & (smcdry > 0.0)
    result = np.array(soil_m, copy=True)
    per_level = {}
    for level in range(soil_m.shape[0]):
        mask = needs[level]
        count = int(np.count_nonzero(mask))
        if count == 0:
            continue
        no_category = mask & ~categorical
        per_level[f"SMOIS_L{level + 1}"] = {
            "floored_cells": count,
            "min_pre_floor": float(np.min(pre_clip[level][mask])),
            "smcdry_applied_min": float(np.min(floor[mask])),
            "smcdry_applied_max": float(np.max(floor[mask])),
            # Cells with no air-dry value of their own, floored at WRF's
            # constant instead.  A land/soil-category disagreement count.
            "wrf_constant_cells": int(np.count_nonzero(no_category)),
        }
        result[level][mask] = floor[mask]
    constant_cells = int(np.count_nonzero(needs & ~categorical[None, ...]))
    # The policy string stays exactly what it was whenever only the
    # category floor fired, so existing receipts keep reading unchanged;
    # the fallback clause appears only when it actually applied.
    policy = "land-below-smcdry-floored-to-smcdry"
    if constant_cells:
        policy += ("; land without a soil category floored to WRF's "
                   f"{_WRF_ZERO_SOIL_MOISTURE}")
    receipt = {
        "policy": policy,
        "wrf_reference": dict(_MOISTURE_FLOOR_WRF_REFERENCE),
        "fields": per_level,
        "total_floored_cells": int(np.count_nonzero(needs)),
        "wrf_constant_cells": constant_cells,
        "min_pre_floor": float(np.min(np.asarray(pre_clip)[needs])),
    }
    detail = (f"; {constant_cells} of them on land whose soil category is "
              f"water (no air-dry value) floored to WRF's "
              f"{_WRF_ZERO_SOIL_MOISTURE}" if constant_cells else "")
    print(
        "soil moisture floor: "
        f"{receipt['total_floored_cells']} sub-air-dry land value(s) "
        f"floored to the soil type's SMCDRY across "
        f"{sorted(per_level)} (min pre-floor "
        f"{receipt['min_pre_floor']:.6g}){detail}; WRF real.exe resets such "
        "columns to 0.005 (module_initialize_real.F:3376)",
        file=sys.stderr)
    return result, receipt


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
        raise ValueError(_nonphysical_tsk_message(tsk, terrestrial | sea_ice))

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
    pre_clip_moisture = soil_m
    soil_m = np.clip(soil_m, 0.0, 1.0)
    if not np.isfinite(tmn).all() or np.any((tmn < 170.0) | (tmn > 400.0)):
        raise ValueError("deep soil temperature is outside 170..400 K")

    from gpuwm.core.noah import load_tables, pack_params, sh2o_init

    soil_type = _host(soil_type)
    if soil_type.shape != shape:
        raise ValueError("soil_type shape differs from soil fields")
    noah_params = pack_params(load_tables())
    # The clip above makes EXACTLY 0.0 of any admitted sub-zero land value
    # and Noah's thermal conductivity divides by SMC; floor sub-air-dry
    # land layers at the category SMCDRY BEFORE sh2o_init so SMOIS and the
    # derived SH2O carry the same protection.
    soil_m, moisture_floor = _floor_land_moisture_at_smcdry(
        soil_m, pre_clip_moisture, terrestrial, soil_type, noah_params)
    liquid_m = sh2o_init(soil_m, soil_t, soil_type, noah_params)
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
    snow, snowh = (
        _admitted_snow_field("snow water", snow, shape),
        _admitted_snow_field("snow depth", snowh, shape))
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
        moisture_floor=moisture_floor,
    )


__all__ = ["ERA5_LAYER_BOTTOMS_M", "HRRR_SOIL_NODE_DEPTHS_M",
           "NOAH_LAYER_MIDPOINTS_M",
           "NOAH_LAYER_THICKNESS_M", "NoahSoilState",
           "SOIL_TEMPERATURE_RECONCILER_NAMES", "SST_RECONCILER_NAMES",
           "preprocess_noah_soil", "reconciler_soil_temperature",
           "reconciler_sst"]
