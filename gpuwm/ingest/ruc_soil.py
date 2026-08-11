"""WRF ``module_soil_pre.F`` soil ingest for the RUC LSM's soil levels.

This is ``init_soil_depth_3`` (:1153-1194) and ``init_soil_3_real``
(:1817-2158) of WRF v4.6.1 ``share/module_soil_pre.F``, the pair
``process_soil_real`` (:817-830) selects when ``sf_surface_physics ==
RUCLSMSCHEME``.  It is a separate module from :mod:`gpuwm.ingest.soil`
because RUC's geometry is not a longer version of Noah's:

* Noah's ``zs`` are layer **midpoints** and its ``dzs`` are the layer
  thicknesses those midpoints sit inside.  RUC's ``zs`` are **level (node)
  depths**, and the shallowest one is ``0.0`` -- the ground surface itself,
  not the centre of a layer.  A RUC soil value is the value *at a depth*, not
  the mean *over a slab*.
* RUC's ``dzs`` is a derived control volume around each node, and WRF's
  derivation is not a partition of [0, 3] m -- see :func:`ruc_soil_depths`.

Everything here is ``float32``, in WRF's own expression order, because the
reference this module is measured against is WRF's own ``REAL`` arithmetic:
``gpuwm/data/ruc/oracle/soil_ingest.csv``, produced by
``tools/ruc_soil_ingest_wrf461_oracle`` driving the byte-unmodified routines.
Computing in float64 and rounding once is a third function, neither float32
nor correctly rounded, and it is how this project has produced real bugs
before.

Two WRF behaviours are deliberately NOT reproduced.  Both are refused with an
error instead, and both refusals are proved against oracle rows that show
what WRF does:

1. **An unsorted layer source.**  ``init_soil_3_real``'s sort (:1881-1899,
   :1913-1931) permutes ``st_input(1..n)``, but a layer source lives in
   ``st_input(2..n+1)`` -- WRF itself puts it there
   (``share/module_optional_input.F:1364-1370``) and then writes the skin
   temperature into slot 1 and the deep temperature into slot n+2.  The sort
   therefore permutes the wrong slots and drags uninitialised memory into the
   soil column.  The oracle's ``era5_layers_reversed_adj`` rows show the
   result: soil temperatures of order -1e29 K.  metgrid emits its levels
   shallow-to-deep so WRF never trips over this, but a source adapter that
   does not is silently corrupted rather than told.
2. **A zero-moisture land column under the moisture adjustment.**
   :2079 divides by ``0.9*smtotr``; with an all-zero source profile
   ``smtotr`` is zero and the quotient is 0/0.  The oracle's
   ``era5_layers_adj_nosst`` column 10 records what this toolchain then
   returns from ``MAX(0.02, NaN)`` -- 0.02 -- but that is a property of
   gfortran's ``maxss`` operand order, not of Fortran, and a soil column
   whose value depends on it is not a number this project will ship.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


#: ``init_soil_depth_3`` (:1161-1167) tabulates the level depths in metres and
#: supports exactly these two lengths.  The nine-level table is the one a
#: nine-layer RUC run uses, and it is bit-identical to the node depths a
#: RUC-native gridded source already carries.
RUC_LEVEL_DEPTHS_M: dict[int, tuple[float, ...]] = {
    6: (0.00, 0.05, 0.20, 0.40, 1.60, 3.00),
    9: (0.00, 0.01, 0.04, 0.10, 0.30, 0.60, 1.00, 1.60, 3.00),
}

#: :2110-2117 hardcodes these five factors, applied to levels 1-5 of a land
#: column whose moisture increases with depth over its top three levels.
_MOISTURE_DAYTIME_FACTORS = (0.75, 0.8, 0.85, 0.9, 0.95)

#: :1930-1934 forms the Noah source bucket with these weights against
#: ``sm_input(2..5)``.  They are hardcoded in WRF and they are NOT Noah's own
#: ``dzs`` (``init_soil_depth_2``: 0.1, 0.3, 0.6, 1.0): the total is the same
#: 2 m of column, the second and third weights are not.  Reproduced as WRF
#: has them, because this is the definition of the quantity the adjustment
#: conserves, not an independent statement about Noah's geometry.
_NOAH_BUCKET_WEIGHTS = (0.1, 0.2, 0.7, 1.0)

_LAYER_SOURCE = "layers"
_LEVEL_SOURCE = "levels"


@dataclass(frozen=True)
class RucSoilColumns:
    """``float32`` RUC soil state on the scheme's own levels."""

    soil_temperature: np.ndarray   # (num_soil_layers, ...), WRF TSLB
    soil_moisture: np.ndarray      # (num_soil_layers, ...), WRF SMOIS
    skin_temperature: np.ndarray   # (...), WRF TSK after the open-water fill
    level_depths: np.ndarray       # (num_soil_layers,), WRF ZS
    level_thicknesses: np.ndarray  # (num_soil_layers,), WRF DZS


def ruc_soil_depths(num_soil_layers: int = 9) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(zs, dzs)`` exactly as ``init_soil_depth_3`` builds them.

    ``zs`` is tabulated; ``dzs`` is derived (:1174-1183) through an interface
    array ``zs2``.  The derivation has an index shift that is reproduced here
    on purpose, because ``dzs`` is an *input* to the moisture adjustment and
    the reference is WRF's value, not a corrected one:

    ``zs2(1)`` is set to 0 and ``zs2(2)`` to the first half-interval, so
    ``dzs(1)`` is that half-interval.  The loop then overwrites ``zs2(2)``
    with the interface *below* level 2 and computes ``dzs(2) = zs2(2) -
    zs2(1)`` against the zero still in ``zs2(1)``, not against the
    half-interval it just discarded.  The result is that ``dzs(1)`` and
    ``dzs(2)`` both cover [0, 0.005] m and ``sum(dzs)`` is 3.005, not 3.0.
    A partition-of-the-column assumption about ``dzs`` is therefore wrong by
    5 mm at nine levels.
    """

    try:
        table = RUC_LEVEL_DEPTHS_M[int(num_soil_layers)]
    except KeyError:
        raise ValueError(
            f"num_soil_layers={num_soil_layers}: init_soil_depth_3 tabulates "
            f"only {sorted(RUC_LEVEL_DEPTHS_M)} level depths, and :1189-1192 "
            "makes 4 and 5 fatal.  A different count needs WRF's table "
            "extended first, not a remap invented here"
        ) from None

    count = len(table)
    zs = np.asarray(table, dtype=np.float32)
    zs2 = np.zeros(count, dtype=np.float32)
    dzs = np.zeros(count, dtype=np.float32)
    half = np.float32(0.5)

    zs2[0] = np.float32(0.0)
    zs2[1] = (zs[1] + zs[0]) * half
    dzs[0] = zs2[1] - zs2[0]
    for index in range(1, count - 1):
        zs2[index] = (zs[index + 1] + zs[index]) * half
        dzs[index] = zs2[index] - zs2[index - 1]
    zs2[count - 1] = zs[count - 1]
    dzs[count - 1] = zs2[count - 1] - zs2[count - 2]
    return zs, dzs


def _as_profile(value, name: str, count: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim < 1 or array.shape[0] != count:
        raise ValueError(
            f"{name} must have {count} leading source levels, got shape "
            f"{array.shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _bracket(target: np.float32, samples: np.ndarray) -> int:
    """Return WRF's chosen source interval for one target depth, or -1.

    :1958-1968 walks the source intervals in order and takes the FIRST that
    contains the target, with both comparisons inclusive -- so a target that
    coincides with an interior sample depth is taken from the interval ABOVE
    it, not the one below.  A target outside every interval leaves ``tslb`` /
    ``smois`` at whatever was in the INTENT(OUT) array, which is why the
    caller must be told rather than given a clamped value.
    """

    for index in range(samples.size - 1):
        if samples[index] <= target <= samples[index + 1]:
            return index
    return -1


def _interpolate(
    samples: np.ndarray, depths: np.ndarray, targets: np.ndarray, label: str,
) -> np.ndarray:
    """WRF's linear interpolation, in WRF's float32 expression order."""

    out = np.empty((targets.size,) + samples.shape[1:], dtype=np.float32)
    for want, target in enumerate(targets):
        have = _bracket(target, depths)
        if have < 0:
            raise ValueError(
                f"{label}: RUC level {want + 1} at {float(target)} m lies "
                f"outside the source depths "
                f"{[float(d) for d in depths]}; init_soil_3_real:1958-1968 "
                "leaves such a level UNSET in an INTENT(OUT) array rather "
                "than extrapolating, so there is no value to reproduce"
            )
        upper = depths[have]
        lower = depths[have + 1]
        out[want] = (
            samples[have] * (lower - target)
            + samples[have + 1] * (target - upper)
        ) / (lower - upper)
    return out


def remap_soil_to_ruc_levels(
    *,
    source_temperature,
    source_moisture,
    source_levels_cm,
    source_geometry: str,
    skin_temperature,
    deep_temperature,
    landmask,
    sea_surface_temperature=None,
    num_soil_layers: int = 9,
    moisture_adjustment: bool = False,
) -> RucSoilColumns:
    """Build RUC ``TSLB``/``SMOIS`` the way ``real.exe`` does.

    ``source_geometry`` is ``"layers"`` for a slab source that reaches WRF as
    metgrid's ``FLAG_SOIL_LAYERS=1`` (ERA5, GFS), and ``"levels"`` for a node
    source already on RUC's own depths (``FLAG_SOIL_LEVELS=1``; a RUC-native
    or HRRR grid).  The two are not interchangeable and WRF's two arms differ
    in more than bookkeeping:

    ``"layers"``
        ``source_levels_cm`` are the layer MIDPOINTS in whole centimetres,
        which is how WRF's reader forms them -- ``char2int2``
        (``share/module_optional_input.F:1949-1954``) and the
        ``flag_soil_layers`` block (:1339-1352) both compute
        ``(top + bottom) / 2`` in INTEGER arithmetic, so a 0-7 cm ERA5 layer
        is sampled at 3 cm, not 3.5 and not 7.  The profile is then anchored
        at 0 m by the skin temperature and at 3 m by the deep temperature
        (:1899-1907), and moisture is anchored at 0 m by a linear
        extrapolation off the top two layers and at 3 m by repeating the
        deepest (:1938-1948).
    ``"levels"``
        The nodes are the samples, with no anchors at all.  A target level
        outside them is an error, not an extrapolation.

    ``moisture_adjustment`` is WRF's ``flag_sm_adj`` (namelist default 0,
    ``Registry.EM_COMMON:2537``).  It applies only to a layer source
    (:2067) and rescales the top ``num_soil_layers - 1`` levels so their
    ``dzs``-weighted bucket matches the source's own four-layer bucket
    divided by 0.9, then floors each at 0.02, then -- if moisture rises
    with depth across the top three levels -- multiplies levels 1-5 by
    :data:`_MOISTURE_DAYTIME_FACTORS`.  The deepest level is never adjusted.
    """

    if source_geometry not in {_LAYER_SOURCE, _LEVEL_SOURCE}:
        raise ValueError(
            f"source_geometry={source_geometry!r} must be "
            f"{_LAYER_SOURCE!r} or {_LEVEL_SOURCE!r}"
        )

    zs, dzs = ruc_soil_depths(num_soil_layers)
    count = zs.size

    levels_cm = np.asarray(source_levels_cm)
    if levels_cm.ndim != 1 or levels_cm.size < 2:
        raise ValueError("source_levels_cm must be at least two depths")
    if levels_cm.dtype.kind not in "iu":
        raise ValueError(
            "source_levels_cm must be INTEGER centimetres: WRF's readers "
            "form them with integer division and init_soil_3_real:1994 "
            "divides the INTEGER by 100."
        )
    if np.any(np.diff(levels_cm) <= 0):
        raise ValueError(
            "source_levels_cm must be strictly increasing.  WRF sorts them "
            "(init_soil_3_real:1881-1899) but the accompanying swap moves "
            "st_input(1..n) while a layer source occupies st_input(2..n+1), "
            "so an unsorted source is not reordered -- it is corrupted.  See "
            "the oracle's era5_layers_reversed_adj rows, whose soil "
            "temperatures are of order -1e29 K."
        )
    if np.any(levels_cm < 0):
        raise ValueError("source_levels_cm must be non-negative depths")

    nlev = int(levels_cm.size)
    temperature = _as_profile(source_temperature, "source_temperature", nlev)
    moisture = _as_profile(source_moisture, "source_moisture", nlev)
    field_shape = temperature.shape[1:]
    if moisture.shape[1:] != field_shape:
        raise ValueError("source temperature and moisture shapes differ")

    tsk = np.array(skin_temperature, dtype=np.float32, copy=True)
    tmn = np.asarray(deep_temperature, dtype=np.float32)
    mask = np.asarray(landmask, dtype=np.float32)
    for name, value in (
        ("skin_temperature", tsk), ("deep_temperature", tmn),
        ("landmask", mask),
    ):
        if value.shape != field_shape:
            raise ValueError(
                f"{name} shape {value.shape} differs from the source field "
                f"shape {field_shape}"
            )
        if not np.isfinite(value).all():
            raise ValueError(f"{name} contains non-finite values")

    hundred = np.float32(100.0)
    if source_geometry == _LEVEL_SOURCE:
        depths = (levels_cm.astype(np.float32) / hundred).astype(np.float32)
        temperature_samples = temperature
        moisture_samples = moisture
        moisture_depths = depths
    else:
        depths = np.empty(nlev + 2, dtype=np.float32)
        depths[0] = np.float32(0.0)
        depths[1:nlev + 1] = levels_cm.astype(np.float32) / hundred
        # :1990 and :2033 spell the deep anchor as 300./100., not as 3.
        depths[nlev + 1] = np.float32(300.0) / hundred
        temperature_samples = np.concatenate(
            (tsk[None, ...], temperature, tmn[None, ...]), axis=0)
        # :1938-1943.  Note it divides by the difference of the TEMPERATURE
        # level depths; WRF reads st_levels_input here even though the array
        # being extrapolated is moisture.  They coincide for every metgrid
        # source, and this function enforces that by taking one level list.
        span = np.float32(int(levels_cm[1]) - int(levels_cm[0]))
        shallow = np.float32(int(levels_cm[0]))
        surface = (moisture[0] - moisture[1]) / span * shallow + moisture[0]
        moisture_samples = np.concatenate(
            (surface[None, ...], moisture, moisture[-1:]), axis=0)
        moisture_depths = depths

    soil_t = _interpolate(
        temperature_samples, depths, zs, "RUC soil temperature")
    soil_m = _interpolate(
        moisture_samples, moisture_depths, zs, "RUC soil moisture")

    if moisture_adjustment:
        if source_geometry != _LAYER_SOURCE:
            raise ValueError(
                "moisture_adjustment applies only to a layer source: "
                "init_soil_3_real:2067 guards the whole block with "
                "flag_soil_levels /= 1, so asking for it on a node source "
                "would silently do nothing"
            )
        if nlev != len(_NOAH_BUCKET_WEIGHTS):
            raise ValueError(
                f"moisture_adjustment needs exactly "
                f"{len(_NOAH_BUCKET_WEIGHTS)} source layers: "
                "init_soil_3_real:1930-1934 indexes sm_input(2..5) with "
                "hardcoded weights and has no general form"
            )
        soil_m = _apply_moisture_adjustment(soil_m, moisture, mask, dzs)

    land = mask >= np.float32(0.5)
    water = ~land
    if np.any(water):
        # :2131-2157.  With an SST field the whole water column, including
        # TSK, is reset to SST; without one it is reset to TSK.
        if sea_surface_temperature is None:
            fill = tsk
        else:
            sst = np.asarray(sea_surface_temperature, dtype=np.float32)
            if sst.shape != field_shape:
                raise ValueError(
                    "sea_surface_temperature shape differs from the source "
                    "field shape")
            if not np.isfinite(sst).all():
                raise ValueError(
                    "sea_surface_temperature contains non-finite values")
            # :2138 assigns tsk(i,j) = sst(i,j) on every water column with no
            # validity test at all -- and it is the ONE line by which the RUC
            # arm differs from Noah's, whose init_soil_2_real:1732-1754 writes
            # tslb/smois/sh2o over water and never touches tsk.  A gap in the
            # SST analysis therefore kills a RUC run and not a Noah one:
            # stock real.exe aborts a few hundred lines later at
            # dyn_em/module_initialize_real.F:3283 with "grid%tsk
            # unreasonable".  Measured on one production coarse-domain
            # metgrid frame of this project's own case: 1,494 of 14,397
            # water cells (10.4%) carry SST = 0, and real.exe at
            # sf_surface_physics=3 dies on the first of them while the same
            # input at sf_surface_physics=2 writes its wrfinput without
            # complaint.
            # Refused at the seam, where the caller can substitute a skin
            # temperature the way preprocess_noah_soil already does, rather
            # than propagated as a 0 K ocean.
            unusable = water & (
                (sst < np.float32(170.0)) | (sst > np.float32(400.0)))
            if np.any(unusable):
                raise ValueError(
                    f"{int(unusable.sum())} of {int(water.sum())} water "
                    "columns have a nonphysical sea_surface_temperature.  "
                    "init_soil_3_real:2131-2144 writes it into TSK unchecked "
                    "-- unlike Noah's init_soil_2_real, which never writes "
                    "TSK -- and stock real.exe then aborts at "
                    "module_initialize_real.F:3283.  Repair the SST field "
                    "against a valid skin temperature before this seam"
                )
            fill = sst
            tsk = np.where(water, sst, tsk).astype(np.float32)
        soil_t = np.where(water[None, ...], fill[None, ...], soil_t)
        soil_m = np.where(
            water[None, ...], np.float32(1.0), soil_m).astype(np.float32)

    return RucSoilColumns(
        soil_temperature=np.ascontiguousarray(soil_t, dtype=np.float32),
        soil_moisture=np.ascontiguousarray(soil_m, dtype=np.float32),
        skin_temperature=np.ascontiguousarray(tsk, dtype=np.float32),
        level_depths=zs,
        level_thicknesses=dzs,
    )


def _apply_moisture_adjustment(
    soil_m: np.ndarray,
    source_moisture: np.ndarray,
    landmask: np.ndarray,
    dzs: np.ndarray,
) -> np.ndarray:
    """``init_soil_3_real``:2062-2129, the Noah-to-RUC bucket rescale."""

    count = soil_m.shape[0]
    # :1930-1934, formed before the surface/deep anchors are written and so
    # from the source layers alone.
    bucket_source = source_moisture[0] * np.float32(_NOAH_BUCKET_WEIGHTS[0])
    for index in range(1, len(_NOAH_BUCKET_WEIGHTS)):
        bucket_source = bucket_source + (
            source_moisture[index] * np.float32(_NOAH_BUCKET_WEIGHTS[index])
        )

    # :2072-2077.  The RUC bucket omits the deepest level, and it is
    # accumulated in level order because float32 addition is not associative.
    bucket_target = np.zeros(soil_m.shape[1:], dtype=np.float32)
    for index in range(count - 1):
        bucket_target = bucket_target + soil_m[index] * dzs[index]

    land = landmask > np.float32(0.5)
    if np.any(land & (bucket_target == np.float32(0.0))):
        raise ValueError(
            "a land column has zero RUC soil-moisture bucket, so "
            "init_soil_3_real:2079 divides by zero.  WRF then evaluates "
            "MAX(0.02, NaN), whose result is gfortran's operand order "
            "rather than anything Fortran defines -- the oracle's "
            "era5_layers_adj_nosst column 10 records 0.02 on gfortran 13.3.0. "
            "Refused rather than reproduced"
        )

    adjusted = soil_m.copy()
    scale = bucket_source / (np.float32(0.9) * bucket_target)
    for index in range(count - 1):
        candidate = (adjusted[index] * bucket_source) / (
            np.float32(0.9) * bucket_target)
        adjusted[index] = np.where(
            land, np.maximum(np.float32(0.02), candidate), adjusted[index],
        ).astype(np.float32)
    del scale

    # :2100-2107.  Strict >, so an exactly flat profile does NOT arm it.
    daytime = land & (adjusted[1] > adjusted[0]) & (adjusted[2] > adjusted[1])
    for index in range(count - 1):
        if index < len(_MOISTURE_DAYTIME_FACTORS):
            factor = np.float32(_MOISTURE_DAYTIME_FACTORS[index])
        else:
            factor = np.float32(1.0)
        adjusted[index] = np.where(
            daytime, factor * adjusted[index], adjusted[index],
        ).astype(np.float32)
    return adjusted


# ---------------------------------------------------------------------------
# The runtime seam.
#
# Everything above is ``share/module_soil_pre.F`` and answers "what does
# real.exe compute".  Everything below answers the separate question this
# module existed for a week without answering: **how does a RUC run actually
# receive it.**  Until this section landed, ``gpuwm/ingest/ruc_soil.py`` had
# zero importers anywhere under ``gpuwm/`` -- the remap was finished, oracle
# matched, and unreachable, so every initializer called
# ``preprocess_noah_soil`` and a ``sf_surface_physics=3`` request received
# Noah's four layer MIDPOINTS.
# ---------------------------------------------------------------------------

#: WRF's ``RUCLSMSCHEME``.  Spelled here rather than imported so this module
#: keeps depending on nothing but numpy and :mod:`gpuwm.ingest.soil`.
RUCLSMSCHEME = 3

#: Every land-surface selector whose soil geometry is Noah's four layer
#: midpoints.  ``2`` is Noah and ``4`` is Noah-MP; both take
#: :func:`gpuwm.ingest.soil.preprocess_noah_soil` unchanged.
_NOAH_GEOMETRY_SCHEMES = (2, 4)


@dataclass(frozen=True)
class RucSoilState:
    """A RUC initial soil state in ``NoahSoilState``'s slot.

    The first ten fields are spelled exactly as :class:`
    gpuwm.ingest.soil.NoahSoilState`'s, because every consumer downstream of
    the initializers -- ``initialize_physics``,
    ``gpuwm.native_wrf_contract.canonical_noah_surface``,
    ``initialize_landuse`` -- reads them by name.  What differs is the leading
    axis: ``soil_temperature``/``soil_moisture``/``liquid_moisture`` are
    ``(num_soil_layers, ny, nx)`` on RUC's LEVEL depths, not ``(4, ny, nx)``
    on Noah's layer midpoints.  ``level_depths``/``level_thicknesses`` are
    the ZS/DZS that go with them and have no ``NoahSoilState`` counterpart.

    ``liquid_moisture`` is a copy of ``soil_moisture``.  That is not a
    partition and is not meant to be one: WRF derives a RUC run's SH2O in
    ``ruclsminit`` (``module_sf_ruclsm.F``), not in ``real.exe``, and
    :func:`gpuwm.core.ruc_runtime.ruc_cold_start` runs ``ruclsminit`` at
    driver construction and overwrites every value here.  Noah's
    ``sh2o_init`` freezing-curve partition is deliberately NOT applied --
    it is Noah's table lookup on Noah's geometry, and running it would put a
    Noah-derived number into a RUC field that is about to be recomputed.
    """

    soil_temperature: np.ndarray       # (num_soil_layers, ny, nx), WRF TSLB
    soil_moisture: np.ndarray          # (num_soil_layers, ny, nx), WRF SMOIS
    liquid_moisture: np.ndarray        # (num_soil_layers, ny, nx), WRF SH2O
    deep_soil_temperature: np.ndarray  # (ny, nx), WRF TMN
    tsk: np.ndarray                    # (ny, nx)
    landmask: np.ndarray               # WPS 1 land / 0 water
    xland: np.ndarray                  # WRF 1 land / 2 water
    xice: np.ndarray
    snow_water: np.ndarray             # kg m-2
    snow_depth: np.ndarray             # m
    level_depths: np.ndarray           # (num_soil_layers,), WRF ZS
    level_thicknesses: np.ndarray      # (num_soil_layers,), WRF DZS


def _midpoint_cm(name: str, prefix: str) -> int:
    """The INTEGER centimetre midpoint WRF's reader forms from a layer name.

    ``share/module_optional_input.F:1949-1954`` (``char2int2``) and the
    ``flag_soil_layers`` block at ``:1339-1352`` both compute
    ``(top + bottom) / 2`` in INTEGER arithmetic, so ``ST000007`` is sampled
    at 3 cm and not 3.5.  Parsed off the field name rather than tabulated,
    because the name IS the depth encoding and a hand table would be a second
    place for it to be wrong.
    """

    digits = name[len(prefix):]
    if len(digits) != 6 or not digits.isdigit():
        raise ValueError(f"{name!r} is not a <prefix><top3><bottom3> layer name")
    return (int(digits[:3]) + int(digits[3:])) // 2


def _source_soil_profiles(fields, soil_layer_contract):
    """Return ``(temperature, moisture, levels_cm, geometry)`` for the source.

    The four source modes are exactly the four
    :func:`gpuwm.ingest.soil.preprocess_noah_soil` dispatches on, and the
    marker names are imported from it rather than restated, so a source whose
    field names move cannot leave this function reading the old ones.
    """

    from gpuwm.ingest.soil import (_GFS_MOIST_NAMES, _GFS_TEMP_NAMES,
                                   _MOIST_NAMES, _TEMP_NAMES, _host)
    from gpuwm.ingest.soil_contract import (MAPPED_SOIL_MOISTURE,
                                            MAPPED_SOIL_TEMPERATURE)

    if soil_layer_contract is not None or MAPPED_SOIL_TEMPERATURE in fields:
        raise ValueError(
            "the declarative mapped-source path has no RUC target.  "
            "gpuwm/ingest/soil_contract.py:validate_soil_layer_contract "
            "compares target_layers against NOAH_LAYER_BOUNDS_M and refuses "
            "any other by design, because RUC's levels are node depths "
            "rather than layer bounds.  Admitting RUC there means adding its "
            "target and proving the remap against a WRF-real nine-level "
            "initialization; refused here rather than silently remapped "
            "through Noah's contract"
        )

    if "SOILT" in fields and "SOILW" in fields:
        # HRRR's native depth nodes ARE RUC's nine levels, bit for bit --
        # gpuwm.ingest.soil.HRRR_SOIL_NODE_DEPTHS_M is
        # RUC_LEVEL_DEPTHS_M[9] in metres.  So this arm is WRF's
        # ``flag_soil_levels`` case and the remap is the identity.
        levels_cm = np.array(
            [int(round(depth * 100.0)) for depth in RUC_LEVEL_DEPTHS_M[9]],
            dtype=np.int64)
        return (_host(fields["SOILT"]), _host(fields["SOILW"]),
                levels_cm, _LEVEL_SOURCE)

    if all(name in fields for name in _GFS_TEMP_NAMES):
        names, moist_names, prefix = (
            _GFS_TEMP_NAMES, _GFS_MOIST_NAMES, "GFS_ST")
    elif all(name in fields for name in _TEMP_NAMES):
        names, moist_names, prefix = (_TEMP_NAMES, _MOIST_NAMES, "ST")
    else:
        raise KeyError(
            "no recognised soil source: expected HRRR SOILT/SOILW nodes, the "
            f"four GFS layers {_GFS_TEMP_NAMES}, or the four ERA5 layers "
            f"{_TEMP_NAMES}")
    levels_cm = np.array(
        [_midpoint_cm(name, prefix) for name in names], dtype=np.int64)
    return (np.stack([_host(fields[name]) for name in names]),
            np.stack([_host(fields[name]) for name in moist_names]),
            levels_cm, _LAYER_SOURCE)


def preprocess_ruc_soil(
    fields,
    *,
    soil_type,
    deep_soil_temperature=None,
    lake_mask=None,
    lake_skin_temperature=None,
    soil_layer_contract=None,
    num_soil_layers: int = 9,
    landmask=None,
    terrain=None,
    source_orography=None,
    water_temperature=None,
    water_temperature_policy=None,
    route=None,
) -> RucSoilState:
    """Build a RUC initial soil state, in ``preprocess_noah_soil``'s signature.

    The surface bookkeeping -- the land/water/sea-ice partition, the lake
    override, the SST and TSK repair, the TMN repair, the 271.4 K sea-ice
    deep temperature and the SNOW/SNOWH reconciliation -- is
    ``dyn_em/module_initialize_real.F`` and ``share/module_soil_pre.F``'s
    surface half, which WRF runs identically for every ``sf_surface_physics``.
    So this function CALLS :func:`gpuwm.ingest.soil.preprocess_noah_soil` for
    it rather than restating it.  That is deliberate and it is what makes
    "Noah initialization is bitwise unchanged" a structural property instead
    of a measurement: the Noah routine is not edited, not copied and not
    parameterised, only called.  Its four-layer TSLB/SMOIS/SH2O are then
    DISCARDED, and the soil column is rebuilt from the source profiles by
    :func:`remap_soil_to_ruc_levels`, which is ``init_soil_3_real``.

    Three divergences from stock ``real.exe``, each with a reason:

    ``flag_sm_adj``
        Not applied.  ``gpuwm.config.RUC_OPTION_IDENTITY_EVIDENCE`` admits
        only ``flag_sm_adj = 0``, so ``moisture_adjustment`` is False here and
        the two must not drift apart.

    the open-water fill
        ``init_soil_3_real:2131-2144`` writes SST into TSK on every water
        column with no validity test, and stock ``real.exe`` then aborts a few
        hundred lines later at ``module_initialize_real.F:3283`` on any gap in
        the SST analysis -- measured at 10.4% of the water cells of one
        production coarse-domain frame.  ``preprocess_noah_soil`` has
        already resolved TSK to the valid SST wherever one exists and to the
        skin temperature where it does not, so this seam takes WRF's
        **no-SST** arm against that already-repaired TSK.  The result equals
        WRF's wherever WRF would not have aborted.

    sea ice
        Left to the runtime.  ``LSMRUC:862-895`` resets SOILMOIS, SMFR3D,
        SH2O, KEEPFR3DFLAG and ``min(271.4, TSO)`` on every sea-ice column on
        **every** step, not once at initialization, so writing an ice column
        here would be overwritten before it was read.  Noah's four-layer
        equispaced ice profile is therefore not carried over; TMN is, because
        ``preprocess_noah_soil`` sets it to 271.4 K on ice and RUC reads TMN
        as the bottom boundary at ``LSMRUC:1057``.

    ``landmask``, ``terrain`` and ``source_orography`` are the ERA5/mapped
    initializers' soil seam, with :func:`gpuwm.ingest.soil.
    preprocess_noah_soil`'s exact semantics: the static-build landmask is
    the terrestrial/water decision surface (WRF's ``process_soil_real``
    drives every branch with ``grid%landmask``, never the met-source
    LANDSEA), and terrain/source_orography (all-or-none) arm WRF's
    ``adjust_soil_temp_new`` elevation lapse.  That lapse is applied by
    ``process_soil_real`` to TSK and to EVERY ``st_input`` level BEFORE
    either ``init_soil_*_real`` arm runs, so it is scheme-independent:
    the Noah call below returns the adjusted TSK anchor, and the same
    float64 increment -- computed by the same helper on the same decision
    surface, lakes excluded -- is added to the RUC source temperature
    samples (layer or level form alike) before the float32 remap.
    Moisture and the 3 m deep anchor are not adjusted, exactly as on the
    Noah path.

    ``num_soil_layers`` reaches :func:`ruc_soil_depths`, which refuses
    anything but 6 or 9.  gpuwm's CUDA leaves index a nine-element
    ``__constant__`` table, so a six-level request is refused further in.
    """

    from gpuwm.ingest.soil import (_host, _soil_temperature_elevation_delta,
                                   preprocess_noah_soil)

    # Selected BEFORE the Noah call, so a source RUC has no arm for is
    # refused by name rather than by whatever preprocess_noah_soil says
    # about it first.
    temperature, moisture, levels_cm, geometry = _source_soil_profiles(
        fields, soil_layer_contract)

    surface = preprocess_noah_soil(
        fields, soil_type=soil_type,
        deep_soil_temperature=deep_soil_temperature,
        lake_mask=lake_mask, lake_skin_temperature=lake_skin_temperature,
        soil_layer_contract=soil_layer_contract,
        landmask=landmask, terrain=terrain,
        source_orography=source_orography,
        water_temperature=water_temperature,
        water_temperature_policy=water_temperature_policy,
        route=route)

    if terrain is not None:
        # The Noah call above validated the all-or-none pairing and
        # adjusted the TSK anchor; mirror the identical increment onto the
        # RUC source samples.  The decision surface is rebuilt exactly as
        # preprocess_noah_soil builds it (static landmask over met-source
        # LANDSEA; lake-override cells are water), because the increment
        # must be zero on every column WRF would not adjust.
        decision = (_host(landmask) if landmask is not None
                    else _host(fields["LANDSEA"])) >= 0.5
        if lake_mask is not None:
            decision = decision.copy()
            decision[_host(lake_mask).astype(bool)] = False
        temperature = temperature + _soil_temperature_elevation_delta(
            terrain, source_orography, decision)

    columns = remap_soil_to_ruc_levels(
        source_temperature=temperature,
        source_moisture=moisture,
        source_levels_cm=levels_cm,
        source_geometry=geometry,
        skin_temperature=surface.tsk,
        deep_temperature=surface.deep_soil_temperature,
        landmask=surface.landmask,
        sea_surface_temperature=None,
        num_soil_layers=num_soil_layers,
        moisture_adjustment=False,
    )

    return RucSoilState(
        soil_temperature=columns.soil_temperature,
        soil_moisture=columns.soil_moisture,
        liquid_moisture=np.array(columns.soil_moisture, copy=True),
        deep_soil_temperature=surface.deep_soil_temperature,
        tsk=surface.tsk,
        landmask=surface.landmask,
        xland=surface.xland,
        xice=surface.xice,
        snow_water=surface.snow_water,
        snow_depth=surface.snow_depth,
        level_depths=columns.level_depths,
        level_thicknesses=columns.level_thicknesses,
    )


def preprocess_land_surface_soil(
    fields,
    *,
    sf_surface_physics: int,
    soil_type,
    deep_soil_temperature=None,
    lake_mask=None,
    lake_skin_temperature=None,
    soil_layer_contract=None,
    num_soil_layers: int | None = None,
    landmask=None,
    terrain=None,
    source_orography=None,
    water_temperature=None,
    water_temperature_policy=None,
    route=None,
):
    """Route a soil source to the selected land surface's own geometry.

    This is the one seam every initializer calls, and it exists because the
    seven of them called :func:`gpuwm.ingest.soil.preprocess_noah_soil`
    unconditionally with no land-surface selector in sight -- so a
    ``sf_surface_physics=3`` request was initialised on Noah's four layer
    midpoints, silently, and the nine-level remap beside it was dead code.

    **Fail-closed on the selector, not on the shape.**  An unrecognised
    ``sf_surface_physics`` raises here rather than falling through to Noah.
    That direction matters: the failure this replaces was not a crash, it was
    a complete plausible forecast on the wrong soil discretization.  The
    shape check downstream (``_as_soil``) would have caught a four-layer array
    handed to a nine-layer allocation, but only as an unexplained broadcast
    error, and only for the schemes whose counts differ.
    """

    scheme = int(sf_surface_physics)
    if scheme == RUCLSMSCHEME:
        # The landmask/terrain/source-orography seam is forwarded whole:
        # preprocess_ruc_soil implements it with preprocess_noah_soil's
        # exact semantics (static-mask decision surface, scheme-independent
        # adjust_soil_temp_new lapse on TSK and every source temperature
        # level).  This replaced a fail-closed raise that existed while the
        # RUC side of the seam was unported; the genuine refusals -- an
        # unpaired terrain/source_orography, a non-boolean landmask, the
        # Noah-bounds mapped contract -- live in the callees and survive.
        return preprocess_ruc_soil(
            fields, soil_type=soil_type,
            deep_soil_temperature=deep_soil_temperature,
            lake_mask=lake_mask, lake_skin_temperature=lake_skin_temperature,
            soil_layer_contract=soil_layer_contract,
            landmask=landmask, terrain=terrain,
            source_orography=source_orography,
            water_temperature=water_temperature,
            water_temperature_policy=water_temperature_policy,
            route=route,
            **({} if num_soil_layers is None
               else {"num_soil_layers": int(num_soil_layers)}))
    if scheme in _NOAH_GEOMETRY_SCHEMES:
        from gpuwm.ingest.soil import preprocess_noah_soil

        # Called with exactly preprocess_noah_soil's own argument list and
        # nothing added, so a Noah or Noah-MP run's soil state is the same
        # object it was before this seam existed.
        return preprocess_noah_soil(
            fields, soil_type=soil_type,
            deep_soil_temperature=deep_soil_temperature,
            lake_mask=lake_mask, lake_skin_temperature=lake_skin_temperature,
            soil_layer_contract=soil_layer_contract,
            landmask=landmask, terrain=terrain,
            source_orography=source_orography,
            water_temperature=water_temperature,
            water_temperature_policy=water_temperature_policy,
            route=route)
    raise ValueError(
        f"sf_surface_physics={scheme} has no soil ingest.  gpuwm implements "
        f"the Noah geometry for {_NOAH_GEOMETRY_SCHEMES} "
        "(module_soil_pre.F:init_soil_2_real) and the RUC level geometry for "
        f"{RUCLSMSCHEME} (init_soil_3_real); routing anything else through "
        "either would initialise a forecast on a soil discretization the "
        "scheme does not use"
    )


__all__ = [
    "RUCLSMSCHEME",
    "RUC_LEVEL_DEPTHS_M",
    "RucSoilColumns",
    "RucSoilState",
    "preprocess_land_surface_soil",
    "preprocess_ruc_soil",
    "remap_soil_to_ruc_levels",
    "ruc_soil_depths",
]
