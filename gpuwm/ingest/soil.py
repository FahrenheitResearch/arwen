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
    linear_sample_plan,
    soil_layer_bounds,
    soil_node_depths,
    soil_source_sample_count,
    validate_soil_layer_contract,
)
from gpuwm.ingest.water_temperature import (
    require_assembled_water_temperature)


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
    #: Ingest-repair receipt for the WRF-faithful TMN = TSK substitution
    #: on LAND (:func:`preprocess_noah_soil`).  Same discipline as
    #: ``moisture_floor``: an EMPTY mapping whenever no land cell needed
    #: repairing, so the presence of any key is itself the signal.  The
    #: water half of that substitution is the ordinary path -- every run
    #: takes it on every water cell -- and is never counted here.
    deep_soil_repair: Mapping[str, object] = field(default_factory=dict)
    #: Receipt from :mod:`gpuwm.ingest.soil_downscale`: the forcing mesh
    #: this soil state came off, how many grid cells one source cell spans,
    #: whether the sub-source-cell reconstitution ran, and what it moved.
    #: UNLIKE the two repair receipts above this one is NOT conditional --
    #: the soil-state source resolution belongs on every run's provenance,
    #: because "how coarse was the soil you started from" is a question a
    #: reader of the output must be able to answer without the config.  It
    #: is empty only for a route that declared no source mesh at all.
    soil_texture_downscale: Mapping[str, object] = field(default_factory=dict)
    #: Ingest-repair receipt for real.exe's TSLB reasonableness rebuild
    #: (:func:`unreasonable_land_soil_columns`): how many land columns were
    #: rebuilt TSK-to-TMN, their pre-repair range and their bounding box.
    #: EMPTY whenever no land column needed it, the ``moisture_floor``
    #: discipline.
    soil_temperature_repair: Mapping[str, object] = field(default_factory=dict)


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


def door_reconciled_soil_category(static, fields: Mapping[str, object],
                                  landuse_attrs, *, route: str | None = None):
    """ISLTYP as the physics driver will integrate it, for a door's soil ingest.

    ONE assembly of :func:`gpuwm.core.landuse.reconciled_soil_category`'s
    arguments from what every front door already holds -- the static
    fields, the initial met fields and the selected land-use table -- so
    the ERA5 config door, the GFS door, the mapped door and the nested
    child all reconcile the same way.  The GFS and mapped doors handed the
    RAW ``SCT_DOM`` to ``preprocess_land_surface_soil`` after the GFS+RUC
    route refusal was retired, so a shoreline column carrying the water
    soil category under a land ``LU_INDEX`` reached RUC and evaluated
    ``0./0.`` into MAVAIL on the first surface call -- the exact death the
    retired refusal existed to avoid, now fixed where real.exe fixes it
    (``module_initialize_real.F:3108-3131``: silty clay loam, the land-use
    category kept), at initialization (ENG-009).

    ``landuse_attrs`` is ``None`` only for a prebuilt static cache with no
    geography tree beside it: there is no ISWATER/ISLAKE/ISICE to reconcile
    against, so the raw category is returned and, when ``route`` names the
    caller, the fact is printed rather than assumed away.
    """
    soil_type = static["SCT_DOM"]
    if landuse_attrs is None:
        if route:
            import sys

            print(
                f"{route}: the prebuilt static cache carries no land-use "
                "metadata (ISWATER/ISLAKE/ISICE), so the soil category is "
                "not reconciled against LU_INDEX the way real.exe does "
                "(module_initialize_real.F:3108-3131); a land column carrying "
                "the water soil category would reach the land-surface scheme "
                "as written.  Pass --geog-root, which is read only for the "
                "land-use index, to reconcile it.", file=sys.stderr)
        return soil_type
    from gpuwm.core.landuse import reconciled_soil_category

    return reconciled_soil_category(
        static["LU_INDEX"], soil_type=soil_type,
        xice=fields.get("XICE", 0.0),
        iswater=int(landuse_attrs["ISWATER"]),
        islake=int(landuse_attrs["ISLAKE"]),
        isice=int(landuse_attrs["ISICE"]),
        soil_temperature=reconciler_soil_temperature(fields),
        sst=reconciler_sst(fields))


#: Grids whose island soil columns were already announced, so a domain
#: prepared at several forcing times says it once.
_ANNOUNCED_ISLAND_SOIL: set = set()


def island_soil_columns(fields: Mapping[str, object], *, no_source_land,
                        soil_type, landmask=None, lake_mask=None):
    """The soil column of land the source holds no land for.

    An island in a source area of open sea: every soil search from its
    cells ends without a source land cell
    (:attr:`gpuwm.ingest.horiz.HorizontalSnapshot.soil_no_source_land`),
    and WPS writes METGRID.TBL fill_missing there, a 285 K column saturated
    at 1.0, which stock real.exe carries into the forecast.  The source's
    state where the island lies is the sea's, so the column takes what the
    source does hold there and what the soil itself determines:

    * soil temperature, every layer: the skin temperature mapped onto the
      cell (the source's skin on the other surface there), the value the
      column is anchored to at 0 m and the one real.exe puts in an
      unreasonable land deep temperature (TMN = TSK);
    * soil moisture, every layer: the field capacity of the cell's own soil
      category (``SOILPARM.TBL`` STAS ``REFSMC``), the water a drained soil
      holds.  A land cell carrying the water category is silty clay loam
      there, as real.exe makes it.

    The cells are those of ``no_source_land`` that are land by ``landmask``
    (``LANDSEA`` without one) and not an overridden lake.  Every soil
    temperature and moisture field ``fields`` carries takes the column, in
    whichever layer form it arrives, so each land-surface scheme builds its
    own levels from it.  Returns ``(fields, receipt)``: a copy with the
    columns in place and their counts and ranges, or ``fields`` and
    ``None`` when there is no such cell.
    """
    from gpuwm.core.landuse import _LAND_SOIL_FOR_WATER
    from gpuwm.core.noah import load_tables
    from gpuwm.ingest.horiz import (_SOIL_MOISTURE_FIELDS,
                                    _SOIL_TEMPERATURE_FIELDS)

    if no_source_land is None:
        return fields, None
    decision = landmask if landmask is not None else fields.get("LANDSEA")
    if decision is None:
        return fields, None
    cells = np.asarray(no_source_land, dtype=bool) & (_host(decision) >= 0.5)
    if lake_mask is not None:
        cells &= ~_host(lake_mask).astype(bool)
    if not np.any(cells):
        return fields, None
    skin = _host(fields["SKINTEMP"])
    categories = _host(soil_type)
    if categories.shape != cells.shape or skin.shape != cells.shape:
        raise ValueError(
            "island soil columns: soil_type, SKINTEMP and the "
            "no-source-land mask differ in shape")
    tables = load_tables()
    capacity_by_category = np.asarray(tables.refsmc, dtype=np.float64)
    category = np.rint(categories[cells]).astype(np.int64)
    if np.any((category < 1) | (category > capacity_by_category.size)):
        raise ValueError(
            "island soil columns: a soil category is outside SOILPARM.TBL "
            f"{tables.sltype} 1..{capacity_by_category.size}")
    capacity = capacity_by_category[category - 1]
    # The water category holds no soil water at all (REFSMC 0); a land cell
    # carrying it is silty clay loam (module_initialize_real.F:3108-3131).
    capacity = np.where(capacity > 0.0, capacity,
                        capacity_by_category[_LAND_SOIL_FOR_WATER - 1])
    temperature = skin[cells]
    patched = dict(fields)
    touched = []
    for name, value in fields.items():
        if name in _SOIL_TEMPERATURE_FIELDS:
            column = temperature
        elif name in _SOIL_MOISTURE_FIELDS:
            column = capacity
        else:
            continue
        array = np.array(_host(value), dtype=np.float32, copy=True)
        if array.shape == cells.shape:
            array[cells] = column
        elif array.ndim == 3 and array.shape[1:] == cells.shape:
            array[:, cells] = column
        else:
            raise ValueError(
                f"island soil columns: {name} has shape {array.shape} on a "
                f"{cells.shape} grid")
        patched[name] = array
        touched.append(name)
    count = int(np.count_nonzero(cells))
    receipt = {
        "cells": count,
        "soil_temperature": "skin temperature mapped onto the cell",
        "soil_moisture": f"field capacity, SOILPARM.TBL {tables.sltype} REFSMC",
        "fields": sorted(touched),
        "soil_temperature_range_k": [float(temperature.min()),
                                     float(temperature.max())],
        "soil_moisture_range": [float(capacity.min()), float(capacity.max())],
    }
    key = (cells.shape, count)
    if touched and key not in _ANNOUNCED_ISLAND_SOIL:
        _ANNOUNCED_ISLAND_SOIL.add(key)
        print(
            f"island soil: {count} land cell(s) of this "
            f"{cells.shape[0]}x{cells.shape[1]} grid lie where the source "
            "holds no land, so their soil column is their skin temperature "
            f"({receipt['soil_temperature_range_k'][0]:.1f}.."
            f"{receipt['soil_temperature_range_k'][1]:.1f} K) and their "
            "soil's field capacity "
            f"({receipt['soil_moisture_range'][0]:.3f}.."
            f"{receipt['soil_moisture_range'][1]:.3f}); WPS writes "
            "METGRID.TBL fill_missing there, a 285 K column saturated at 1.0",
            file=sys.stderr)
    return patched, receipt


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

    target = soil_layer_bounds(contract, "target_layers")
    remap = contract["remap"]
    if not isinstance(remap, Mapping):  # guarded by validation; defense in depth
        raise TypeError("soil remap must be an object")
    if remap["kind"] == "linear_node_samples":
        # Depth-node sources (RUC-family: HRRR, RAP) sample the column at
        # true depths INCLUDING 0 m and 3 m, so Noah's four midpoints come
        # directly from WRF's sorted linear node interpolation -- the same
        # arithmetic the native HRRR route runs on its fixed node table,
        # executed here on the depths the contract declares.
        node_depths = np.asarray(soil_node_depths(contract), dtype=np.float64)
        target_depths = np.asarray(
            [(top + bottom) / 2.0 for top, bottom in target],
            dtype=np.float64,
        )
        return (
            _interp_nodes(temperature, node_depths, target_depths),
            _interp_nodes(moisture, node_depths, target_depths),
        )
    source = soil_layer_bounds(contract, "source_layers")
    if remap["kind"] == "linear_point_samples":
        # Where the source's values sit is the contract's declaration, read
        # through the ONE plan the validator checked ordering and coverage
        # on (soil_contract.linear_sample_plan).  Its
        # wrf_integer_cm_layer_midpoint row is WRF's own layer-form
        # convention (module_optional_input.F:char2int2, (top+bottom)/2 in
        # whole cm); the bracketing anchors are TSK at 0 m and TMN at 3 m
        # (module_soil_pre.F:1591-1595).  The plan also says whether each
        # anchor is a node at all: a source that states its own value at
        # the anchor's depth supplies that boundary itself, and stacking
        # the anchor beside it would be two values at one depth.
        plan = linear_sample_plan(contract)
        source_depths = np.asarray(plan.depths, dtype=np.float64)
        target_depths = np.asarray(
            [(top + bottom) / 2.0 for top, bottom in target],
            dtype=np.float64,
        )
        temperature_parts = [temperature]
        moisture_parts = [moisture]
        if plan.top_anchor:
            temperature_parts.insert(0, tsk[None, ...])
            moisture_parts.insert(0, moisture[:1])
        if plan.bottom_anchor:
            temperature_parts.append(deep[None, ...])
            moisture_parts.append(moisture[-1:])
        temperature_nodes = np.concatenate(temperature_parts, axis=0)
        moisture_nodes = np.concatenate(moisture_parts, axis=0)
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


def soil_source_orography(declared, fields):
    """The terrain the SOURCE model's soil fields were defined on.

    ONE resolution for every route, for the same reason the per-source
    evidence spellings live in this module: the root, the nested child, and
    the ERA5 direct door each answered this question separately and drifted.
    A met source declares its own orography two ways -- an explicit
    artifact the case points at, or an invariant record inside the forcing
    itself, which the horizontal stage remaps onto the mass grid as
    ``SOURCE_OROGRAPHY`` (ERA5's SOILGEO geopotential, HRRR's SOILHGT).
    Every route resolved only the first, so a case whose orography rode
    inside its GRIB silently lost WRF's ``adjust_soil_temp_new`` elevation
    lapse -- and the ERA5 direct door, which passes ``terrain``
    unconditionally, died on :func:`preprocess_noah_soil`'s all-or-none
    guard instead.

    ``None`` is returned only when the route genuinely has neither, which
    stays the historical no-adjustment path for sources that declare no
    orography at all.  Callers that have already refused that case treat
    ``None`` as an internal inconsistency.

    This deliberately does NOT mirror ``initialize_real``'s refusal of a
    declared artifact beside an embedded record: the routes refuse that
    conflict upstream, where the declaration is still attributable to the
    config line that made it.
    """
    if declared is not None:
        return declared
    return fields.get("SOURCE_OROGRAPHY")


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


def _snow_undershoot_fraction() -> float:
    """How far below zero interpolation can carry a non-negative snow field.

    As a fraction of the field's own positive maximum: the overlapping
    parabola's whole negative weight, 9/32
    (:data:`gpuwm.ingest.horiz.WPS_PARABOLIC_NEGATIVE_WEIGHT`), widened by
    its FP32 evaluation slack.  A 2x2 patch holding a trace of snow inside
    snow at the maximum already reaches about 17/64 of it; the band that
    used to sit here, one quarter, was inside that reach and refused it.  The masked mapping
    carries snow with two weighted means and cannot go below zero, and the
    native HRRR decoder puts its own parabola's undershoot at zero against
    the source it mapped (gpuwm/ingest/hrrr.py), so a negative reaches
    this band only from an input that carries one, such as a met_em file
    made with a parabolic snow table.
    """
    from gpuwm.ingest.horiz import (_WPS_PARABOLIC_ENVELOPE_SLACK,
                                    WPS_PARABOLIC_NEGATIVE_WEIGHT)

    return WPS_PARABOLIC_NEGATIVE_WEIGHT * (1.0 + _WPS_PARABOLIC_ENVELOPE_SLACK)


#: The other side of the same band: ceilings past which a snow field is
#: a unit error rather than a snowpack.  There is no bounded-stencil
#: story up here -- the horizontal operators cannot overshoot ABOVE the
#: source maximum by orders of magnitude, so a value this large is what
#: the source said, in whatever unit the route believed it was said in.
#: The two ceilings are one column stated twice, related by the same
#: 5:1 liquid-to-depth ratio (200 kg m-3) the SNOW/SNOWH reconciliation
#: below uses, so a field derived from its partner cannot land outside
#: its own bound.  100 m is a bit over twice the deepest real field this
#: preprocessor has been handed (the 44.5 m maximum recorded below) and
#: well past the deepest snow ever measured; 20 000 kg m-2 is that
#: column's water equivalent, itself twice the ECMWF land scheme's cap
#: on permanent snow (10 m of water equivalent).  A ceiling catches a
#: factor of 1000 -- metres of water equivalent read as kg m-2, or the
#: reverse -- and it cannot catch a factor of two.  It is a plausibility
#: bound, not a substitute for the field arriving under the name that
#: states its unit.
_SNOW_WATER_CEILING_KG_M2 = 2.0e4
_SNOW_DEPTH_CEILING_M = 1.0e2

#: A route that declares no source mesh is announced once per process, not
#: once per forcing time and per domain, on the same reasoning as
#: ``horiz.py``'s ``_REPORTED_FRACTIONAL_RECOVERY``: the fact is a property
#: of the ROUTE, so repeating it per call is noise around a fact the reader
#: already has.
_REPORTED_MISSING_SOIL_MESH: set = set()


def _admitted_snow_field(name: str, value: np.ndarray, shape,
                         ceiling: float, unit: str) -> np.ndarray:
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

    The band is two-sided.  ``ceiling`` is the plausibility bound above
    which the field is not a snowpack at all (see
    ``_SNOW_WATER_CEILING_KG_M2``), and it is checked whatever the
    minimum is -- the early return below is for the overshoot repair,
    not for admission.
    """
    if value.shape != tuple(shape):
        raise ValueError(
            f"{name} must be a 2-D field shaped {tuple(shape)}, got "
            f"{value.shape}")
    if not np.isfinite(value).all():
        raise ValueError(
            f"{name} carries {int(np.count_nonzero(~np.isfinite(value)))} "
            f"non-finite value(s) of {value.size}")
    largest = float(np.max(value))
    if largest > ceiling:
        raise ValueError(
            f"{name} is above the plausibility ceiling of {ceiling:.6g} "
            f"{unit}: {int(np.count_nonzero(value > ceiling))} value(s) "
            f"of {value.size}, field maximum {largest:.6g} {unit}")
    smallest = float(np.min(value))
    if smallest >= 0.0:
        return value
    fraction = _snow_undershoot_fraction()
    floor = -fraction * max(largest, 0.0)
    if smallest < floor:
        raise ValueError(
            f"{name} is negative beyond the interpolation-overshoot band, "
            f"{fraction:.6g} of the field maximum, as far below zero as "
            f"interpolating a non-negative field can reach: "
            f"{int(np.count_nonzero(value < floor))} value(s) of "
            f"{value.size}, most negative {smallest:.6g}, field maximum "
            f"{largest:.6g}; a fill value or a broken decode, not snow")
    return np.maximum(value, 0.0)


def clamp_soil_moisture_overshoot(moisture, *, land=None, subject: str):
    """Put interpolation overshoot outside 0..1 on the range; refuse the rest.

    Volumetric soil moisture is a fraction of the soil volume, so a value
    outside 0..1 is never data.  One thing makes such a value from real
    soil moisture: an interpolation operator that is not a weighted mean
    of its donors.  WPS's ``sixteen_pt`` overlapping parabolas swing past
    the donors on a sharp soil-moisture edge (a reservoir, a river bottom,
    an irrigated field beside dry range) by up to 9/32 of the step, so
    from a 0..1 field they reach no further than
    :func:`gpuwm.ingest.horiz.parabolic_reach` (about -0.297..1.297), and
    every other operator in metgrid's chains is a weighted mean.  gpuwm's
    own masked mapping makes none (it answers from a weighted mean
    wherever ``sixteen_pt`` would leave the range), so what reaches here
    arrived with the route's input, as a met_em file carries WPS's own.
    Each such value goes on the range, counted, however sharp the edge
    that made it.

    A land value beyond that reach was not made by interpolating soil
    moisture at all: it is a fill value (metgrid's -1e30) or the field in
    another unit (percent, kg m-2).  Put on the range it would initialize
    that land saturated or bone dry from a number that never was soil
    moisture, so it is refused, with its count and extremes.

    ``land`` (the target land mask, broadcast over leading layers) picks
    the values judged and counted, since water columns are set to 1.0
    later whatever they held; without it every value is.  Values already
    inside 0..1 are untouched, and a non-finite value is left for the
    caller's own refusal.  Returns ``(moisture, counted)``.
    """
    from gpuwm.ingest.horiz import parabolic_reach

    moisture = np.asarray(moisture, dtype=np.float64)
    finite = np.isfinite(moisture)
    outside = finite & ((moisture < 0.0) | (moisture > 1.0))
    if not outside.any():
        return moisture, 0
    counted = outside if land is None else (
        outside & np.broadcast_to(np.asarray(land, dtype=bool),
                                  moisture.shape))
    where = "value(s)" if land is None else "land value(s)"
    lowest, highest = parabolic_reach(0.0, 1.0)
    beyond = counted & ((moisture < lowest) | (moisture > highest))
    if beyond.any():
        raise ValueError(
            f"{subject}: {int(np.count_nonzero(beyond))} {where} outside "
            f"{lowest:.4f}..{highest:.4f}, as far as interpolating a 0..1 "
            "field can reach (smallest "
            f"{float(np.min(moisture[beyond])):.6g}, largest "
            f"{float(np.max(moisture[beyond])):.6g}); a fill value or soil "
            "moisture in another unit, which put on 0..1 would initialize "
            "that land saturated or bone dry")
    moved = int(np.count_nonzero(counted))
    if moved:
        exceedance = float(np.max(np.where(
            counted, np.maximum(-moisture, moisture - 1.0), 0.0)))
        print(
            f"{subject}: {moved} {where} outside 0..1 (largest "
            f"exceedance {exceedance:.4f}) put on the range; interpolation "
            "overshoot on a sharp soil-moisture edge, inside the "
            f"{lowest:.4f}..{highest:.4f} a 0..1 field can be carried to",
            file=sys.stderr)
    return np.where(outside, np.clip(moisture, 0.0, 1.0), moisture), moved


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


_DEEP_SOIL_REPAIR_WRF_REFERENCE = {
    "wrf_version": "v4.6.1",
    "wrf_citation": (
        "dyn_em/module_initialize_real.F (land TMN outside a reasonable "
        "range is replaced by TSK before module_soil_pre consumes it)"),
    "wrf_behavior": (
        "real.exe: a land cell whose deep soil temperature is missing or "
        "unreasonable takes that cell's skin temperature"),
    "gpuwm_behavior": (
        "identical substitution, COUNTED: land cells only, with the "
        "pre-repair range, because a whole-domain deep-soil decode "
        "failure and one bad cell used to produce the same silence"),
}


#: real.exe's soil-temperature reasonableness band.
SOIL_TEMPERATURE_BAND_K = (170.0, 400.0)

_SOIL_TEMPERATURE_REPAIR_WRF_REFERENCE = {
    "wrf_version": "v4.7.1",
    "wrf_citation": (
        "dyn_em/module_initialize_real.F:3521-3596 ('Is the grid%tslb "
        "reasonable?'; first-time-level arm :3536-3595: land cell with "
        "TSLB(1) outside 170..400 K :3539-3540, TSK and TMN both inside "
        "the band :3560-3561, fake_soil_temp rebuild for LSMSCHEME, "
        "NOAHMPSCHEME and RUCLSMSCHEME :3568-3573, moisture reset to 0.3 "
        "only for schemes outside that list :3541-3558)"),
    "wrf_behavior": (
        "real.exe: a land column whose first-layer TSLB is outside "
        "170..400 K has every layer rebuilt as (tsk*(3-zs) + tmn*(0-zs))/3 "
        "and keeps its soil moisture under Noah, Noah-MP and RUC"),
    "gpuwm_behavior": (
        "the same rebuild on the land surface's own depths, linear from TSK "
        "at 0 m to TMN at 3 m: (tsk*(3-zs) + tmn*zs)/3.  WRF's tmn*(0-zs) "
        "sign takes the column below 170 K past about 0.6 m and to about "
        "0 K at 1.5 m, so it is not reproduced.  A column is rebuilt when "
        "ANY of its source samples is outside the band, not only the one "
        "the top layer is read from, because a bad deeper sample reaches "
        "the deeper layers; a column with a MISSING sample is not rebuilt "
        "and is refused, as real.exe's comparisons never select a NaN"),
}


def unreasonable_land_soil_columns(temperature, land):
    """``(ny, nx)`` land columns whose soil temperature real.exe rebuilds.

    ``temperature`` is the source soil temperature samples on the target
    grid, ``(samples, ny, nx)``, after the horizontal mapping and the
    elevation lapse.  A land column is unreasonable when every sample is
    finite and any lies outside :data:`SOIL_TEMPERATURE_BAND_K`.  A column
    with a missing sample is not selected, so the refusal that names a
    missing land sample still fires on it.

    Named breakage: HRRRv2 analyses (January 2017, western snowpack) carry
    land soil temperatures of 60 to 168 K beside a skin near 270 K, and
    every preparation whose domain reached them was refused.
    """
    low, high = SOIL_TEMPERATURE_BAND_K
    values = np.asarray(temperature, dtype=np.float64)
    finite = np.isfinite(values)
    outside = finite & ((values < low) | (values > high))
    return (outside.any(axis=0) & finite.all(axis=0)
            & np.asarray(land, dtype=bool))


def tsk_tmn_soil_profile(depths_m, tsk, deep):
    """real.exe's fake_soil_temp column, linear from TSK at 0 m to TMN at 3 m.

    ``(depths, ny, nx)`` float64.  WRF writes the TMN term as
    ``tmn*(0-zs)``; the sign is corrected here (see
    ``_SOIL_TEMPERATURE_REPAIR_WRF_REFERENCE``).
    """
    z = np.asarray(depths_m, dtype=np.float64).reshape((-1, 1, 1))
    tsk = np.asarray(tsk, dtype=np.float64)[None]
    deep = np.asarray(deep, dtype=np.float64)[None]
    return (tsk * (3.0 - z) + deep * z) / 3.0


def soil_temperature_repair_receipt(temperature, columns, land):
    """Counted receipt for the rebuilt columns; EMPTY when there are none.

    The bounding box is the smallest block of the grid, as 0-based
    inclusive ``rows`` (j) and ``columns`` (i), that holds every rebuilt
    column.
    """
    columns = np.asarray(columns, dtype=bool)
    count = int(np.count_nonzero(columns))
    if count == 0:
        return {}
    before = np.asarray(temperature, dtype=np.float64)[:, columns]
    low, high = SOIL_TEMPERATURE_BAND_K
    outside = (before < low) | (before > high)
    rows, cols = np.nonzero(columns)
    land_cells = int(np.count_nonzero(np.asarray(land, dtype=bool)))
    return {
        "policy": "land-soil-column-outside-170..400K-rebuilt-tsk-to-tmn",
        "wrf_reference": dict(_SOIL_TEMPERATURE_REPAIR_WRF_REFERENCE),
        "repaired_land_columns": count,
        "land_cells": land_cells,
        "source_samples": int(before.shape[0]),
        "samples_outside_band": int(np.count_nonzero(outside)),
        "pre_repair_min_k": float(before.min()),
        "pre_repair_max_k": float(before.max()),
        "grid_shape": [int(value) for value in columns.shape],
        "bounding_box": {
            "rows": [int(rows.min()), int(rows.max())],
            "columns": [int(cols.min()), int(cols.max())],
        },
    }


def soil_temperature_repair_proof(soil, grid):
    """The rebuilt-soil-column receipt with its box in degrees, or None.

    What a preparation proof records: the soil state's
    ``soil_temperature_repair`` with the bounding box's latitude and
    longitude span over ``grid``'s mass points added.  ``None`` when
    real.exe's TSLB reasonableness rebuild
    (:func:`unreasonable_land_soil_columns`) touched no land column, so a
    healthy preparation's proof and cache carry byte for byte what they
    always did.  One function for every route that records it, so the
    mapped and the native HRRR proofs spell it alike.
    """
    receipt = dict(getattr(soil, "soil_temperature_repair", None) or {})
    if not receipt:
        return None
    latitude, longitude = (np.asarray(value) for value in grid.latlon_mass())
    box = dict(receipt["bounding_box"])
    (j0, j1), (i0, i1) = box["rows"], box["columns"]
    block = (slice(j0, j1 + 1), slice(i0, i1 + 1))
    box["latitude"] = [float(np.min(latitude[block])),
                       float(np.max(latitude[block]))]
    box["longitude"] = [float(np.min(longitude[block])),
                        float(np.max(longitude[block]))]
    receipt["bounding_box"] = box
    return receipt


def _announce_soil_temperature_repair(receipt):
    """The one line that says which soil columns were rebuilt."""
    box = receipt["bounding_box"]
    ny, nx = receipt["grid_shape"]
    whole = (" -- that is EVERY land column in the domain, so no source "
             "soil temperature survives in it"
             if receipt["repaired_land_columns"] == receipt["land_cells"]
             else "")
    print(
        f"soil temperature rebuild: {receipt['repaired_land_columns']} of "
        f"{receipt['land_cells']} land column(s) carried a source soil "
        f"temperature outside 170..400 K "
        f"({receipt['pre_repair_min_k']:.6g}..{receipt['pre_repair_max_k']:.6g}"
        f" K, rows {box['rows'][0]}..{box['rows'][1]} and columns "
        f"{box['columns'][0]}..{box['columns'][1]} of the {ny}x{nx} grid) and "
        "were rebuilt linear in depth from the skin temperature at 0 m to "
        "the deep soil temperature at 3 m, following WRF real.exe's "
        "rebuild with its deep-temperature sign corrected (real.exe's "
        "tmn*(0-zs) takes the column toward 0 K); "
        f"their soil moisture is kept{whole}",
        file=sys.stderr)


def _count_land_deep_soil_repair(deep, land, valid_deep, tsk):
    """Receipt for the TMN = TSK substitution, LAND cells only.

    The substitution itself is WRF-faithful and stays exactly as it was.
    What was missing is the count.  Every run takes this branch on every
    water cell -- that is the ordinary path and it is not reportable --
    but on land it means the deep-soil source did not deliver, and a
    whole-domain failure was indistinguishable from one bad cell because
    neither was counted.  Deep soil temperature is the lower boundary
    condition of the soil column, so a domain-wide TMN = TSK removes the
    seasonal thermal reservoir for the whole forecast.

    Returns an EMPTY mapping when no land cell needed repairing, the same
    silence-when-healthy discipline :func:`_floor_land_moisture_at_smcdry`
    uses, so the presence of any key is itself the signal.
    """
    repaired = np.asarray(land, dtype=bool) & ~np.asarray(valid_deep, dtype=bool)
    count = int(np.count_nonzero(repaired))
    if count == 0:
        return {}
    land_cells = int(np.count_nonzero(np.asarray(land, dtype=bool)))
    before = np.asarray(deep, dtype=np.float64)[repaired]
    finite = before[np.isfinite(before)]
    receipt = {
        "policy": "land-tmn-outside-170..400K-replaced-by-tsk",
        "wrf_reference": dict(_DEEP_SOIL_REPAIR_WRF_REFERENCE),
        "repaired_land_cells": count,
        "land_cells": land_cells,
        "land_fraction_repaired": count / land_cells if land_cells else 0.0,
        "non_finite_cells": int(np.count_nonzero(~np.isfinite(before))),
        "min_pre_repair": float(np.min(finite)) if finite.size else None,
        "max_pre_repair": float(np.max(finite)) if finite.size else None,
        "tsk_applied_min": float(np.min(np.asarray(tsk)[repaired])),
        "tsk_applied_max": float(np.max(np.asarray(tsk)[repaired])),
    }
    whole_domain = (" -- that is EVERY land cell in the domain, so the "
                    "deep-soil source did not decode at all and this "
                    "forecast has no seasonal thermal reservoir under the "
                    "soil column" if count == land_cells else "")
    span = ("all non-finite" if finite.size == 0
            else f"[{receipt['min_pre_repair']:.6g}, "
                 f"{receipt['max_pre_repair']:.6g}] K")
    print(
        f"deep soil temperature repair: {count} of {land_cells} land "
        f"cell(s) had TMN outside 170..400 K ({span}) and took that "
        f"cell's TSK instead, as WRF real.exe does"
        f"{whole_domain}",
        file=sys.stderr)
    return receipt


def _floor_land_moisture_at_smcdry(soil_m, pre_clip, terrestrial,
                                   soil_type, params):
    """Floor sub-air-dry LAND soil moisture at the category's SMCDRY.

    An ERA5 swvl value a hair below zero (GRIB packing, horizontal
    interpolation undershoot) is put on the range by
    :func:`clamp_soil_moisture_overshoot` and so becomes EXACTLY 0.0.  Noah's thermal conductivity then divides by
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
                         source_orography=None,
                         water_temperature=None,
                         water_temperature_policy=None,
                         soil_mesh=None,
                         route=None, fractional_seaice: bool = False) -> NoahSoilState:
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

    ``water_temperature`` is the finished water-surface field the calling
    route assembled with
    :func:`gpuwm.ingest.water_temperature.assemble_for_route`, one provider
    per connected body of water.  It is not optional in the ordinary sense:
    a mapping that carries a raw ``SST`` beside its ``SKINTEMP`` and no
    assembled field is REFUSED here by ``route`` name, because the per-cell
    selector this replaced is what painted lakes as a quilt of two
    differently-mapped providers.  ``water_temperature_policy =
    "wrf_compat"`` is the one declaration that reopens that selector, for
    stock-WRF certification.  See
    :func:`gpuwm.ingest.water_temperature.require_assembled_water_temperature`
    for why a mapping WITHOUT an SST is the identity and passes.

    ``terrain`` and ``source_orography`` (all-or-none) enable WRF's
    ``adjust_soil_temp_new`` elevation lapse: land skin and soil temperature
    inputs are shifted by ``-0.0065 * (terrain - source_orography)`` before
    the vertical mapping, exactly as ``process_soil_real`` adjusts TSK and
    ``st_input`` when the met source declares its own orography.  The 3 m
    deep temperature is NOT adjusted here: WRF's Noah branch subtracts
    ``0.0065 * terrain`` from the sea-level annual mean instead, which the
    static build already bakes into TMN.

    ``soil_mesh`` is the route's
    :class:`gpuwm.ingest.soil_downscale.SoilMeshPlan`: the forcing mesh
    measured against this grid.  Supplied, the LAND soil state is
    reconstituted below the source spacing -- moisture through Noah's own
    ``SRATIO`` against the target grid's 30 arc-second soil texture, deep
    temperature through WRF's own linear-in-depth ``TMN`` anchoring -- and
    the receipt rides back on ``NoahSoilState.soil_texture_downscale``.
    This is the moisture and deep-temperature analogue of the elevation
    lapse above: both exist because the target grid knows things about the
    land surface that a 0.25 degree forcing mesh cannot.  See
    :mod:`gpuwm.ingest.soil_downscale` for what stock WRF does instead and
    why this diverges from it deliberately.

    ``None`` means the route declared no source mesh, which is announced
    rather than silently skipped: without it the source mesh stays visible
    in SMOIS for the whole forecast and in every field soil moisture drives.
    ``SoilMeshPlan(..., enabled=False)`` -- ``[ingest]
    soil_texture_downscale = false`` -- is the deliberate WRF-comparison
    path and is byte-identical to the historical behaviour.
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
        source_count = soil_source_sample_count(declared_contract)
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
        # A land column outside 170..400 K is rebuilt below, as real.exe
        # rebuilds it; what is left outside the band after that is refused
        # there.  A missing node is refused here.
        if not np.isfinite(soil_temperature_nodes).all():
            raise ValueError("HRRR SOILT nodes are non-finite")
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
    if not isinstance(fractional_seaice, (bool, np.bool_)):
        raise TypeError("fractional_seaice must be boolean")
    sea_ice = (~terrestrial) & (xice >= (0.02 if fractional_seaice else 0.5))
    # WRF adjust_for_seaice_post preserves fractions in its fractional arm;
    # the historical/default binary arm snaps retained ice to one.
    if not fractional_seaice:
        xice[sea_ice] = 1.0
    xice[(~terrestrial) & (~sea_ice)] = 0.0
    effective_land = terrestrial | sea_ice
    landmask = effective_land.astype(np.float64)
    if water_temperature is None:
        # The structural seam.  A route that hands over a raw SST beside its
        # SKINTEMP and no assembled field is refused BY NAME here rather
        # than quietly rerunning the per-cell fuse below; see
        # require_assembled_water_temperature for why an SST-less mapping is
        # the identity and is let through.
        require_assembled_water_temperature(
            route=route, fields=fields, water_temperature=None,
            policy=water_temperature_policy)
        # The historical per-cell fuse between two differently-mapped
        # fields.  Reached by an SST-less mapping (where it reduces to
        # SKINTEMP on every water cell) and by a declared ``wrf_compat``.
        water_temperature = np.where(np.isfinite(sst) & (sst >= 170.0)
                                     & (sst <= 400.0), sst, skin)
    else:
        water_temperature = _host(water_temperature)
        if water_temperature.shape != shape:
            raise ValueError(
                "water_temperature shape differs from soil fields")
        open_water = ~(terrestrial | sea_ice)
        candidate = water_temperature[open_water]
        if (not np.isfinite(candidate).all()
                or np.any((candidate < 170.0) | (candidate > 400.0))):
            raise ValueError(
                "assembled water_temperature is non-finite or outside "
                "170..400 K on open water")
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
    deep_repair = _count_land_deep_soil_repair(deep, land, valid_deep, tsk)
    deep = np.where(land & valid_deep, deep, tsk)
    # real.exe's TSLB reasonableness rebuild, on every soil source alike:
    # a land column carrying a source soil temperature outside 170..400 K
    # is held at TSK through the vertical mapping below and rebuilt
    # TSK-to-TMN on Noah's layers after it.
    if mapped_layers:
        samples = declared_temperature
    elif hrrr_nodes:
        samples = soil_temperature_nodes
    else:
        samples = np.stack(temperatures)
    rebuilt_columns = unreasonable_land_soil_columns(samples, terrestrial)
    temperature_repair = soil_temperature_repair_receipt(
        samples, rebuilt_columns, terrestrial)
    if temperature_repair:
        _announce_soil_temperature_repair(temperature_repair)
        samples = np.array(samples, copy=True)
        samples[:, rebuilt_columns] = tsk[rebuilt_columns]
        if mapped_layers:
            declared_temperature = samples
        elif hrrr_nodes:
            soil_temperature_nodes = samples
        else:
            temperatures = list(samples)
    if hrrr_nodes and np.any((soil_temperature_nodes < 170.0)
                             | (soil_temperature_nodes > 400.0)):
        raise ValueError("HRRR SOILT nodes are outside 170..400 K")
    if mapped_layers:
        # The horizontal mapping hands every target land cell a value: a
        # source land cell with none is not a donor, and a target land cell
        # with no source land near it takes the WPS search's nearest one
        # (gpuwm/ingest/horiz.py).  Only target-ocean values may be absent
        # here; the declared repair below replaces them.
        # A mapped saturated cell reaches here one rounding step above 1.0
        # for exactly the reasons the GFS bridge clamps for, and any
        # overshoot the route's own operator made goes on the range with
        # it (clamp_soil_moisture_overshoot).
        declared_moisture, _ = clamp_bound_kissing(
            declared_moisture, minimum=0.0, maximum=1.0)
        declared_moisture, _ = clamp_soil_moisture_overshoot(
            declared_moisture, land=terrestrial,
            subject="mapped soil moisture")
        land_temperature = declared_temperature[:, terrestrial]
        land_moisture = declared_moisture[:, terrestrial]
        if not np.isfinite(land_temperature).all() \
                or np.any((land_temperature < 170.0) | (land_temperature > 400.0)):
            raise ValueError(
                "declarative mapped soil temperature is missing or outside "
                "170..400 K on land"
            )
        if not np.isfinite(land_moisture).all():
            raise ValueError(
                "declarative mapped soil moisture carries no value on "
                f"{int(np.count_nonzero(~np.isfinite(land_moisture)))} land "
                "value(s): the horizontal mapping gives every land cell one, "
                "so these fields reached the initializer without it"
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
    if temperature_repair:
        soil_t = np.array(soil_t, copy=True)
        soil_t[:, rebuilt_columns] = tsk_tmn_soil_profile(
            NOAH_LAYER_MIDPOINTS_M, tsk, deep)[:, rebuilt_columns]
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
    # The deep-temperature half of the sub-source-cell reconstitution, after
    # the water and sea-ice columns are settled and before the physical
    # bound check -- it touches ONLY terrestrial cells, and every layer's
    # source-cell mean is unchanged, so a passing column cannot be pushed
    # out of 170..400 K by a fraction of a kelvin of anomaly.
    if soil_mesh is not None:
        from gpuwm.ingest.soil_downscale import (
            downscale_deep_soil_temperature)

        soil_t, deep_downscale_receipt = downscale_deep_soil_temperature(
            soil_t, deep_soil_temperature=tmn, terrestrial=terrestrial,
            plan=soil_mesh, layer_midpoints_m=NOAH_LAYER_MIDPOINTS_M)
    else:
        deep_downscale_receipt = {}
    if (not np.isfinite(soil_t).all() or np.any((soil_t < 170.0) | (soil_t > 400.0))):
        raise ValueError("soil temperature is outside 170..400 K")
    # Sixteen-point source stencils overshoot the saturated ceiling
    # where source cells sit at exactly 1.0 next to dry land (GFS
    # glacier/ice at high latitude; module_soil_pre.F:298/:392 itself
    # assigns 1.0 over ice and water) -- observed up to ~1.086 on a
    # high-latitude smoke domain -- and the dry floor beside a reservoir.
    # Stock real.exe carries such columns with no upper clamp at all
    # (the > 1.005 guard at module_initialize_real.F:3383 is commented
    # out in the pinned source).  gpuwm's masked mapping no longer makes
    # any; a route whose input carries them (a met_em file) has each one
    # inside the operator's reach from a 0..1 field put on the range.
    # The band that used to sit here (0.25) was itself inside that reach
    # (9/32 of the step) and refused genuine overshoot; the refusal now
    # sits at the reach, so it still catches a fill value or a unit error
    # on every route, the ones that never enter the masked mapping
    # included.  Values already inside [0, 1] are untouched.
    if not np.isfinite(soil_m).all():
        raise ValueError(
            "soil moisture carries no value on "
            f"{int(np.count_nonzero(~np.isfinite(soil_m)))} of "
            f"{soil_m.size} layer value(s) after the vertical mapping")
    pre_clip_moisture = soil_m
    soil_m, _ = clamp_soil_moisture_overshoot(
        soil_m, land=terrestrial, subject="soil moisture")
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
    # The moisture half of the sub-source-cell reconstitution, AFTER the
    # air-dry floor (so the receipt above still reports what the source
    # actually delivered) and BEFORE sh2o_init (so the frozen-water split
    # is derived from the moisture the forecast will integrate).  Its
    # output is bounded by [SMCDRY, SMCMAX] for this grid's own texture by
    # construction, so it cannot reintroduce what the floor just repaired.
    if soil_mesh is not None:
        from gpuwm.ingest.soil_downscale import downscale_soil_moisture

        soil_m, downscale_receipt = downscale_soil_moisture(
            soil_m, soil_type=soil_type, terrestrial=terrestrial,
            params=noah_params, plan=soil_mesh,
            layer_thickness_m=NOAH_LAYER_THICKNESS_M)
        downscale_receipt = dict(downscale_receipt)
        downscale_receipt["deep_soil_temperature"] = deep_downscale_receipt
    elif not _REPORTED_MISSING_SOIL_MESH:
        downscale_receipt = {}
        _REPORTED_MISSING_SOIL_MESH.add(True)
        print(
            "soil-state source mesh: this route declared none, so the "
            "soil state keeps whatever spacing the forcing arrived on and "
            "SMOIS will carry it for the whole forecast "
            "(gpuwm/ingest/soil_downscale.py)",
            file=sys.stderr)
    else:
        downscale_receipt = {}
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
        _admitted_snow_field("snow water", snow, shape,
                             _SNOW_WATER_CEILING_KG_M2, "kg m-2"),
        _admitted_snow_field("snow depth", snowh, shape,
                             _SNOW_DEPTH_CEILING_M, "m"))
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
        deep_soil_repair=deep_repair,
        soil_texture_downscale=downscale_receipt,
        soil_temperature_repair=temperature_repair,
    )


__all__ = ["ERA5_LAYER_BOTTOMS_M", "HRRR_SOIL_NODE_DEPTHS_M",
           "NOAH_LAYER_MIDPOINTS_M", "clamp_soil_moisture_overshoot",
           "NOAH_LAYER_THICKNESS_M", "NoahSoilState",
           "SOIL_TEMPERATURE_RECONCILER_NAMES", "SST_RECONCILER_NAMES",
           "island_soil_columns", "preprocess_noah_soil", "reconciler_soil_temperature",
           "reconciler_sst", "soil_source_orography"]
