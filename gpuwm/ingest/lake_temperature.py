"""Explicit source lake state, independent of a land/sea majority mask.

A lake model may supply a hypothetical lake column at every source point,
including points where the resolved land fraction is one. Its mixed-layer
temperature is a water state; the grid-box skin temperature is not a
substitute for that state. The encoding adapters bind their named variables
to the three canonical fields below before this module is called.

Two providers read this state.  :func:`map_ice_free_lake_water` gives open
lake water: every contributing interpolation donor needs explicit zero
lake-ice depth, with the one-sided numerical-zero convention declared below.
A positive or unknown depth, or an inadmissible temperature, at any
positive-weight donor means it DECLINES that target cell (NaN in
:attr:`LakeWaterMapping.values`), never a temperature invented here.  The
assembly (:func:`gpuwm.ingest.water_temperature.assemble_water_temperature`)
then falls back, per cell, to the source the lake's component had before
this provider existed -- its coherent skin temperature -- and the receipt
counts and names every such cell (ENG-008: refusing the whole preparation
for one frozen lake was a default-on blocker).

:func:`map_lake_ice` gives the ice the same lake model carries: the
bilinear weight of the donors whose ice depth is positive is the cell's
lake-ice cover, and their ice-surface temperature is its skin.  WRF itself
carries lake ice as ``XICE`` on a water point whenever the lake model is off
(real.exe takes a GFS lake's ICEC the same way), so the ice enters the
preparation as the cell's ice fraction and the soil route's existing ice
branch does the rest.  Before this, a frozen lake started as open water near
273 K whenever the source's sea-ice analysis missed it, which on ERA5 is
most of the Great Lakes' ice and every small lake.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import numpy as np

LAKE_FIELD_UNITS = {
    "LAKE_WATER_TEMP": "K",
    "LAKE_ICE_TEMP": "K",
    "LAKE_ICE_DEPTH": "m",
}
LAKE_FIELDS = frozenset(LAKE_FIELD_UNITS)
LAKE_WATER_PROVIDER = "lake_model_ice_free_water"
# One-sided numerical-zero convention in the field's declared SI unit.
# CDS's constant ice-free fields can encode -1e-19 m (verified against
# ecCodes). Permit at most one binary64 unit roundoff in metres below zero;
# retain the raw source and count every contributing donor. No positive
# depth is admitted, and GRIB packingError is deliberately not a phase
# tolerance: constant zero fields can report packingError=0.5 m.
ICE_DEPTH_NEGATIVE_ZERO_M = float(np.finfo(np.float64).eps)

#: How many declined cells the receipt lists by (j, i); the count is always
#: complete.
MAX_LISTED_DECLINED_CELLS = 64


@dataclass(frozen=True)
class LakeWaterMapping:
    values: np.ndarray
    receipt: dict


def source_lake_fields(fields):
    """Return one complete lake-state contract, or None when absent."""
    present = LAKE_FIELDS.intersection(fields)
    if not present:
        return None
    missing = sorted(LAKE_FIELDS - present)
    if missing:
        raise ValueError(
            "explicit lake-state provider is incomplete; missing "
            f"{missing}. Retrieve lake water temperature, lake ice "
            "temperature and lake ice depth at the same forcing times/grid")
    result = {name: np.asarray(fields[name], dtype=np.float64)
              for name in LAKE_FIELDS}
    shape = result["LAKE_WATER_TEMP"].shape
    if len(shape) != 2 or any(value.shape != shape for value in result.values()):
        raise ValueError("explicit lake-state fields must share one 2-D source grid")
    return result


def _active_lake_targets(snapshot, target_lat, target_lon, target_lake):
    """``(fields, active, source_shape)``, or None with nothing to map."""
    fields = source_lake_fields(snapshot.fields)
    if fields is None:
        return None
    active = np.asarray(target_lake, dtype=bool)
    if active.shape != np.shape(target_lat) or active.shape != np.shape(target_lon):
        raise ValueError("target lake mask must match the interpolation coordinates")
    source_shape = (len(snapshot.latitude), len(snapshot.longitude))
    if fields["LAKE_WATER_TEMP"].shape != source_shape:
        raise ValueError("explicit lake-state fields do not match the source axes")
    if not np.any(active):
        return None
    return fields, active, source_shape


def _local_corners(snapshot, target_lat, target_lon, source_shape):
    """The four bilinear donors ``(j, i, weight)`` of every target cell."""
    from gpuwm.ingest.horiz import _regular_coordinates

    y, x = _regular_coordinates(
        snapshot.latitude, snapshot.longitude, target_lat, target_lon)
    j0 = np.minimum(np.floor(y).astype(np.intp), source_shape[0] - 2)
    i0 = np.minimum(np.floor(x).astype(np.intp), source_shape[1] - 2)
    fy, fx = y - j0, x - i0
    return ((j0, i0, (1 - fy) * (1 - fx)),
            (j0, i0 + 1, (1 - fy) * fx),
            (j0 + 1, i0, fy * (1 - fx)),
            (j0 + 1, i0 + 1, fy * fx))


def map_ice_free_lake_water(snapshot, target_lat, target_lon, target_lake):
    """Local bilinear lake water, with exact positive-weight donor checks.

    This samples the lake-model field itself, not another lake found by a
    search. There is no LANDSEA threshold, missing-donor renormalization,
    extrapolation, or temperature floor. Zero-weight adjacent cells are not
    donors, including at an exact source-grid point or boundary.
    """
    found = _active_lake_targets(snapshot, target_lat, target_lon, target_lake)
    if found is None:
        return None
    fields, active, source_shape = found

    from gpuwm.ingest.water_temperature import (
        MIN_WATER_TEMPERATURE_K, MAX_WATER_TEMPERATURE_K,
    )

    corners = _local_corners(snapshot, target_lat, target_lon, source_shape)
    values = np.zeros(active.shape, dtype=np.float64)
    unknown = np.zeros(active.shape, dtype=bool)
    frozen = np.zeros(active.shape, dtype=bool)
    invalid = np.zeros(active.shape, dtype=bool)
    water = fields["LAKE_WATER_TEMP"]
    depth = fields["LAKE_ICE_DEPTH"]
    negative_zero_donors = np.zeros(source_shape, dtype=bool)
    for jj, ii, weight in corners:
        contributes = active & (weight > 0.0)
        local_depth = depth[jj, ii]
        unknown |= contributes & (~np.isfinite(local_depth)
                                   | (local_depth < -ICE_DEPTH_NEGATIVE_ZERO_M))
        negative_zero = (contributes & (local_depth < 0)
                         & (local_depth >= -ICE_DEPTH_NEGATIVE_ZERO_M))
        negative_zero_donors[jj[negative_zero], ii[negative_zero]] = True
        frozen |= contributes & np.isfinite(local_depth) & (local_depth > 0)
        temperature = water[jj, ii]
        valid = (np.isfinite(temperature)
                 & (temperature >= MIN_WATER_TEMPERATURE_K)
                 & (temperature <= MAX_WATER_TEMPERATURE_K))
        invalid |= contributes & ~valid
        # A non-donor NaN must never enter the sum through 0*NaN.
        values += weight * np.where(contributes & valid, temperature, 0.0)
    context = f"valid_time={snapshot.valid_time.isoformat()} UTC"
    # NOT A REFUSAL.  A frozen or partially frozen donor, an unknown depth,
    # or an inadmissible temperature means the lake model has no ice-free
    # water to offer THIS cell: the provider declines it (NaN).  Where the
    # lake model has ice there, map_lake_ice gives the cell its ice; any
    # declined cell left open falls back per cell to the component's skin
    # temperature -- the source this route used before the provider
    # existed -- with the count and the cells in the receipt.  Refusing the
    # whole preparation here, after fetch and decode, for one frozen lake
    # was a default-on blocker on a route that ran in 2.6.5 (ENG-008).
    declined = active & (unknown | frozen | invalid)
    values = np.where(active & ~declined, values, np.nan)
    declined_where = np.argwhere(declined)

    def digest(value):
        return hashlib.sha256(np.asarray(value, dtype="<f8").tobytes()).hexdigest()

    count = int(negative_zero_donors.sum())
    return LakeWaterMapping(np.where(active, values, np.nan), {
        "schema": "arwen.lake-water-mapping.v1",
        "provider": LAKE_WATER_PROVIDER,
        "valid_time": snapshot.valid_time.isoformat() + "Z",
        "interpolation": "local_bilinear_all_positive_weight_donors",
        "source_shape": list(source_shape),
        "source_fields_sha256": {name: digest(value) for name, value in fields.items()},
        "source_latitude_sha256": digest(snapshot.latitude),
        "source_longitude_sha256": digest(snapshot.longitude),
        "target_lake_cells": int(active.sum()),
        "ice_phase": ("ice_free_where_provided" if declined.any()
                      else "ice_free"),
        "context": context,
        # The cells the provider declined, by cause, with their (j, i).
        "declined_cells": int(declined.sum()),
        "frozen_cells": int((active & frozen).sum()),
        "unknown_depth_cells": int((active & unknown).sum()),
        "invalid_temperature_cells": int((active & invalid).sum()),
        "declined_cell_indices": [
            [int(j), int(i)]
            for j, i in declined_where[:MAX_LISTED_DECLINED_CELLS]],
        "negative_zero_depth_tolerance_m": ICE_DEPTH_NEGATIVE_ZERO_M,
        "negative_zero_depth_donors": count,
        "maximum_negative_zero_depth_m": (
            float(-np.min(depth[negative_zero_donors])) if count else 0.0),
    })


@dataclass(frozen=True)
class LakeIceMapping:
    """The source lake model's ice on the target lake cells.

    ``fraction`` is the bilinear weight of the positive-weight donors whose
    ice depth is positive (0 where every donor is ice free; NaN off the
    lake mask and where a donor's depth or ice temperature is unusable).
    ``temperature`` is those frozen donors' weighted ice-surface
    temperature, NaN where ``fraction`` is not positive.
    """

    fraction: np.ndarray
    temperature: np.ndarray
    receipt: dict


#: The warmest admissible ice-surface temperature.  CDS and ARCO encode the
#: melting point as 273.15011 K; nothing warmer than the triple point is ice.
MAX_LAKE_ICE_TEMPERATURE_K = 273.16
#: The ice cover at which a lake cell is ice under WRF's default binary
#: sea-ice arm (``fractional_seaice = 0``, the 0.5 threshold of
#: gpuwm.ingest.soil.preprocess_noah_soil).  The receipt counts against it.
LAKE_ICE_COVER_THRESHOLD = 0.5
LAKE_ICE_PROVIDER = "lake_model_ice"


def map_lake_ice(snapshot, target_lat, target_lon, target_lake):
    """Lake-model ice cover and ice-surface temperature on the lake cells.

    The same donors :func:`map_ice_free_lake_water` uses: the four local
    bilinear corners with positive weight, no LANDSEA threshold, no search.
    A donor whose depth is unknown (non-finite or below the numerical-zero
    convention), or that is frozen with no admissible ice temperature, makes
    the cell unknown (NaN): the lake model is not used there and the cell
    keeps whatever the source's own sea-ice analysis gave it, counted and
    listed in the receipt.
    """
    found = _active_lake_targets(snapshot, target_lat, target_lon, target_lake)
    if found is None:
        return None
    fields, active, source_shape = found
    from gpuwm.ingest.water_temperature import MIN_WATER_TEMPERATURE_K

    corners = _local_corners(snapshot, target_lat, target_lon, source_shape)
    depth = fields["LAKE_ICE_DEPTH"]
    ice_temperature = fields["LAKE_ICE_TEMP"]
    fraction = np.zeros(active.shape, dtype=np.float64)
    weighted = np.zeros(active.shape, dtype=np.float64)
    unknown = np.zeros(active.shape, dtype=bool)
    for jj, ii, weight in corners:
        contributes = active & (weight > 0.0)
        local_depth = depth[jj, ii]
        unknown |= contributes & (~np.isfinite(local_depth)
                                   | (local_depth < -ICE_DEPTH_NEGATIVE_ZERO_M))
        frozen = contributes & np.isfinite(local_depth) & (local_depth > 0)
        local_t = ice_temperature[jj, ii]
        usable = (np.isfinite(local_t) & (local_t >= MIN_WATER_TEMPERATURE_K)
                  & (local_t <= MAX_LAKE_ICE_TEMPERATURE_K))
        unknown |= frozen & ~usable
        fraction += np.where(frozen, weight, 0.0)
        # A non-donor NaN must never enter the sum through 0*NaN.
        weighted += np.where(frozen & usable, weight * np.where(usable, local_t, 0.0), 0.0)
    known = active & ~unknown
    fraction = np.where(known, np.minimum(fraction, 1.0), np.nan)
    has_ice = known & (fraction > 0.0)
    temperature = np.where(
        has_ice, weighted / np.where(has_ice, fraction, 1.0), np.nan)
    covered = known & (fraction >= LAKE_ICE_COVER_THRESHOLD)
    unknown_where = np.argwhere(active & unknown)
    return LakeIceMapping(fraction, temperature, {
        "schema": "arwen.lake-ice-mapping.v1",
        "provider": LAKE_ICE_PROVIDER,
        "valid_time": snapshot.valid_time.isoformat() + "Z",
        "interpolation": "local_bilinear_frozen_donor_weight",
        "rule": ("ice cover = bilinear weight of the positive-weight donors "
                 "whose ice depth is > 0; ice skin = their weighted "
                 "ice-surface temperature"),
        "target_lake_cells": int(active.sum()),
        "cells_with_lake_ice": int(has_ice.sum()),
        "cells_at_or_above_cover_threshold": int(covered.sum()),
        "cover_threshold": LAKE_ICE_COVER_THRESHOLD,
        "unknown_cells": int((active & unknown).sum()),
        "unknown_cell_indices": [
            [int(j), int(i)]
            for j, i in unknown_where[:MAX_LISTED_DECLINED_CELLS]],
        "ice_temperature_K_min_median_max": (
            [float(np.min(temperature[has_ice])),
             float(np.median(temperature[has_ice])),
             float(np.max(temperature[has_ice]))] if has_ice.any() else None),
    })


__all__ = ["LAKE_FIELD_UNITS", "LAKE_FIELDS", "LAKE_WATER_PROVIDER",
           "ICE_DEPTH_NEGATIVE_ZERO_M", "MAX_LISTED_DECLINED_CELLS",
           "LakeWaterMapping", "LakeIceMapping",
           "MAX_LAKE_ICE_TEMPERATURE_K", "LAKE_ICE_COVER_THRESHOLD",
           "LAKE_ICE_PROVIDER",
           "source_lake_fields", "map_ice_free_lake_water", "map_lake_ice"]
