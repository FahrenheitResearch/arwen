"""Assemble the finished water-surface temperature before soil preprocessing.

Why this module exists
----------------------
The shipped chain chose a water temperature PER CELL, between two fields
that were mapped by two different algorithms::

    water_temperature = where(finite(SST) and 170 <= SST <= 400, SST, SKINTEMP)

SST reaches the target through METGRID.TBL's ``sixteen_pt+four_pt``, and both
of those operators require every point of their stencil to be usable.  An SST
analysis carries no values over land, so within two source cells of any
coastline the stencil is incomplete, both operators decline, and the target
takes ``fill_missing=0.``.  Zero is not a temperature, so the selector above
reads it as "no SST here" and switches that column to SKINTEMP -- which over
lakes is ERA5's FLake model state, several K warmer than the analysis and
quantized on the 0.25 degree source cell.

The consequence is a lake painted as a quilt of two source fields joined at
the SST-validity boundary.  Measured on the reproducing case (Lake Erie,
1985-05-31 12Z, 24 km parent with a ratio-8 3 km nest) by
``tools/water_temperature_probe.py``, over the nest's 4195 water cells and
the 7915 adjacencies between them, with the shipped operators:

===============================================  ===========================
quantity                                         value
===============================================  ===========================
water cells taking SST                           1955 (46.6 %)
water cells taking SKINTEMP                      2240 (53.4 %)
adjacent-step P99 across the water               7.81 K
adjacent-step P99 INSIDE the SST region          0.13 K
adjacent-step P99 INSIDE the SKINTEMP region     0.85 K
adjacent pairs straddling the boundary           397, median jump 5.64 K
===============================================  ===========================

Each adjacency is counted once.  Restricted to Erie itself (the largest
connected body, 2684 cells) 257 of the 269 steps above 1 K sit exactly on the
switch boundary, which is the attribution: neither field is rough on its own,
and the discontinuity is manufactured by switching between them.  So the
remedy is to stop switching: decide ONE provider for a whole connected body
of water and interpolate that provider coherently across it.

What this module guarantees
---------------------------
* Surface classes come from the target statics: land is ``LANDMASK >= 0.5``,
  lakes are the land-use table's own inland-water category (``ISLAKE``),
  ocean is the remaining water.
* A lake and the ocean can never share a connected component, so a lake can
  never be handed ocean water.
* Donors for a component are drawn from that component alone.  There is no
  global search, so a landlocked lake with no analysis of its own cannot
  import another basin's water.
* Every water cell is attributed in ``WATER_TEMP_SOURCE``.

Where the guarantee is enforced
-------------------------------
Not route by route.  :func:`require_assembled_water_temperature` is called by
the soil preprocessing routers, which are the single seam every forcing route
already crosses, and it REFUSES a mapping that still carries a raw SST beside
its SKINTEMP with no assembled field to consume.  A new forcing source
therefore cannot reintroduce the per-cell fuse by omission: it either routes
its own statics through :func:`assemble_for_route` or it is refused by name.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import sys
from types import MappingProxyType
from typing import Mapping

import numpy as np

#: The MODIS inland-water category.  Kept as documentation of the value the
#: MODIS land-use tables carry; NOTHING selects lakes with it.  The lake
#: category is read from the selected land-use table's own ``ISLAKE``
#: (:func:`lake_mask_from_landuse`), because USGS 24-category has no
#: inland-water class at all and WPS writes ``ISLAKE = -1`` there.
MODIS_LAKE_CATEGORY = 21

#: Physical admissibility, the same window the soil reconciler applies.
MIN_WATER_TEMPERATURE_K = 170.0
MAX_WATER_TEMPERATURE_K = 400.0

#: Declared policies.  Silence selects the first one.
WATER_TEMPERATURE_POLICIES = (
    "era5_class_coherent",
    "wrf_compat",
    "external_overlay",
)
DEFAULT_WATER_TEMPERATURE_POLICY = "era5_class_coherent"

#: WATER_TEMP_SOURCE provider ids.  Diagnostic only; nothing branches on them.
SOURCE_LAND = 0
SOURCE_ANALYSIS = 1        #: ERA5 SST, same-component donors
SOURCE_COMPONENT_SKIN = 2  #: coherent SKINTEMP for a whole component
SOURCE_PER_CELL = 3        #: the historical per-cell fuse (wrf_compat)
SOURCE_NAMES = {
    SOURCE_LAND: "land",
    SOURCE_ANALYSIS: "era5_sst_component",
    SOURCE_COMPONENT_SKIN: "era5_skintemp_component",
    SOURCE_PER_CELL: "per_cell_legacy",
}

#: A component takes the analysis when its own donors cover at least this
#: much of it.  Below the bar the whole component takes SKINTEMP instead,
#: because a sparse donor set cannot describe the body coherently.
MIN_COMPONENT_COVERAGE = 0.5


def validate_water_temperature_policy(policy):
    """Return a known policy name, or refuse by name."""
    if policy is None:
        return DEFAULT_WATER_TEMPERATURE_POLICY
    text = str(policy)
    if text not in WATER_TEMPERATURE_POLICIES:
        raise ValueError(
            f"water_temperature_policy {text!r} is not one of "
            + ", ".join(repr(name) for name in WATER_TEMPERATURE_POLICIES))
    return text


def resolve_water_temperature_policy(case_data):
    """The policy a case's declarations select.

    Declaration wins.  Absent a declaration, a declared overlay selects
    ``external_overlay``, because an overlay has already replaced both
    source fields over water and its precedence predates this policy.
    Silence with no overlay is the default, ``era5_class_coherent``.
    """
    declared = getattr(case_data, "water_temperature_policy", None)
    if declared is not None:
        return validate_water_temperature_policy(declared)
    if getattr(case_data, "water_temperature_overlay", None) is not None:
        return "external_overlay"
    return DEFAULT_WATER_TEMPERATURE_POLICY


# ---------------------------------------------------------------------------
# connected components, without adding a dependency
# ---------------------------------------------------------------------------
def _label_components(mask):
    """Label 8-connected components of a boolean mask (1-based, 0 = off).

    Eight-connectivity on purpose: a lake pinched to a diagonal on a coarse
    grid is one lake, not two.  Merging distinct BODIES is prevented by the
    caller, which labels each surface class separately.

    Two-pass union-find, so cost is near-linear rather than proportional to
    the longest lake in the domain.
    """
    mask = np.asarray(mask, dtype=bool)
    ny, nx = mask.shape
    labels = np.zeros((ny, nx), dtype=np.int32)
    parent = [0]

    def find(a):
        root = a
        while parent[root] != root:
            root = parent[root]
        while parent[a] != root:
            parent[a], a = root, parent[a]
        return root

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for j in range(ny):
        for i in range(nx):
            if not mask[j, i]:
                continue
            neighbours = []
            if j > 0:
                if labels[j - 1, i]:
                    neighbours.append(labels[j - 1, i])
                if i > 0 and labels[j - 1, i - 1]:
                    neighbours.append(labels[j - 1, i - 1])
                if i + 1 < nx and labels[j - 1, i + 1]:
                    neighbours.append(labels[j - 1, i + 1])
            if i > 0 and labels[j, i - 1]:
                neighbours.append(labels[j, i - 1])
            if not neighbours:
                parent.append(len(parent))
                labels[j, i] = len(parent) - 1
            else:
                smallest = min(neighbours)
                labels[j, i] = smallest
                for other in neighbours:
                    union(smallest, other)

    remap = {}
    out = np.zeros((ny, nx), dtype=np.int32)
    for j in range(ny):
        for i in range(nx):
            if labels[j, i]:
                root = find(labels[j, i])
                if root not in remap:
                    remap[root] = len(remap) + 1
                out[j, i] = remap[root]
    return out, len(remap)


#: Component labels are a pure function of two invariant statics, but the
#: assembly runs once per forcing time, so an N-time run paid for the same
#: union-find N times: measured 0.19 s at 550 squared and 0.61 s at 1000
#: squared, on the launch-to-first-plot path.  Keyed by the CONTENT of the
#: two masks, so a parent and its children never alias and a caller that
#: rebuilds its statics still hits.
#:
#: Four entries because the live set is one per domain and this tree's
#: deepest hierarchy is four; each holds one int32 label field, so the bound
#: also caps what a large domain can park here (16 MiB per entry at 2000
#: squared).  Concurrent child preparation can race the eviction, which
#: costs a recompute and never a wrong answer: the value is a pure function
#: of the key.
_LABEL_CACHE: dict[tuple, tuple] = {}
_LABEL_CACHE_LIMIT = 4


def _mask_key(mask):
    packed = np.packbits(np.ascontiguousarray(mask, dtype=bool))
    return (mask.shape, hashlib.blake2b(packed.tobytes(),
                                        digest_size=16).hexdigest())


def label_surface_components(target_land, target_lake):
    """Components of water, with lakes and ocean kept strictly apart.

    Returns ``(labels, classes)`` where ``classes[label]`` is ``"lake"`` or
    ``"ocean"``.  Labelling each class separately is what makes "never an
    ocean donor for a lake" a property of the data structure rather than a
    check someone has to remember to write.

    The result is memoized on the two masks' content.  ``labels`` is handed
    back read-only, because it is shared with every other caller holding the
    same statics.
    """
    land = np.asarray(target_land, dtype=bool)
    lake = np.asarray(target_lake, dtype=bool) & ~land
    key = (_mask_key(land), _mask_key(lake))
    cached = _LABEL_CACHE.get(key)
    if cached is not None:
        return cached
    ocean = (~land) & ~lake

    lake_labels, n_lake = _label_components(lake)
    ocean_labels, n_ocean = _label_components(ocean)
    labels = np.where(lake_labels > 0, lake_labels,
                      np.where(ocean_labels > 0, ocean_labels + n_lake, 0))
    classes = {}
    for k in range(1, n_lake + 1):
        classes[k] = "lake"
    for k in range(1, n_ocean + 1):
        classes[k + n_lake] = "ocean"
    labels = labels.astype(np.int32)
    labels.flags.writeable = False
    entry = (labels, MappingProxyType(classes))
    if len(_LABEL_CACHE) >= _LABEL_CACHE_LIMIT:
        _LABEL_CACHE.pop(next(iter(_LABEL_CACHE)))
    _LABEL_CACHE[key] = entry
    return entry


# ---------------------------------------------------------------------------
# coherent interpolation
# ---------------------------------------------------------------------------
def _target_longitude_in_source_frame(source_lon, target_lon):
    """Target longitudes expressed in the source's own 360-degree window.

    A Lambert grid reports longitude in -180..180 while an ERA5 crop may be
    stored in 0..360 or the other way round.  Comparing them directly puts
    every target a whole turn away from its donors, so shift by whole turns
    until the target sits in the source's window.
    """
    lon = np.asarray(source_lon, dtype=np.float64)
    target = np.asarray(target_lon, dtype=np.float64).copy()
    origin = lon[0]
    return origin + np.mod(target - origin + 180.0, 360.0) - 180.0


def _bilinear_corners(source_lat, source_lon, target_lat, target_lon):
    ny, nx = np.asarray(source_lat).size, np.asarray(source_lon).size
    lat = np.asarray(source_lat, dtype=np.float64)
    lon = np.asarray(source_lon, dtype=np.float64)
    y = (np.asarray(target_lat, dtype=np.float64) - lat[0]) / (lat[1] - lat[0])
    x = (_target_longitude_in_source_frame(lon, target_lon)
         - lon[0]) / (lon[1] - lon[0])
    y = np.clip(y, 0.0, ny - 1.0000001)
    x = np.clip(x, 0.0, nx - 1.0000001)
    j0 = np.floor(y).astype(np.intp)
    i0 = np.floor(x).astype(np.intp)
    fy, fx = y - j0, x - i0
    j1 = np.minimum(j0 + 1, ny - 1)
    i1 = np.minimum(i0 + 1, nx - 1)
    return ((j0, i0, (1.0 - fy) * (1.0 - fx)),
            (j0, i1, (1.0 - fy) * fx),
            (j1, i0, fy * (1.0 - fx)),
            (j1, i1, fy * fx))


def normalized_masked_bilinear(field, donors, corners, shape,
                               denominator_floor=1e-6):
    """Bilinear interpolation renormalized over the donors that exist.

    Interpolates ``field * donors`` and ``donors`` separately and divides.
    A target whose stencil holds one usable donor still gets that donor's
    value at full weight instead of being abandoned, and a target with no
    donor weight at all returns NaN rather than a fill that later reads as
    a temperature.
    """
    field = np.asarray(field, dtype=np.float64)
    donors = np.asarray(donors, dtype=bool)
    numerator = np.zeros(shape, dtype=np.float64)
    denominator = np.zeros(shape, dtype=np.float64)
    safe = np.where(donors, field, 0.0)
    for jj, ii, weight in corners:
        present = donors[jj, ii].astype(np.float64)
        numerator += weight * present * safe[jj, ii]
        denominator += weight * present
    out = np.full(shape, np.nan, dtype=np.float64)
    usable = denominator > denominator_floor
    out[usable] = numerator[usable] / denominator[usable]
    return out


def _fill_within_component(values, component, max_sweeps=1000):
    """Close residual holes from the component's OWN cells, never elsewhere."""
    out = np.array(values, dtype=np.float64, copy=True)
    out[~component] = np.nan
    for _ in range(max_sweeps):
        holes = component & np.isnan(out)
        if not np.any(holes):
            break
        have = component & np.isfinite(out)
        if not np.any(have):
            break
        accumulated = np.zeros(out.shape, dtype=np.float64)
        count = np.zeros(out.shape, dtype=np.float64)
        contribution = np.where(have, out, 0.0)
        for destination, source in ((np.s_[1:, :], np.s_[:-1, :]),
                                    (np.s_[:-1, :], np.s_[1:, :]),
                                    (np.s_[:, 1:], np.s_[:, :-1]),
                                    (np.s_[:, :-1], np.s_[:, 1:])):
            accumulated[destination] += contribution[source]
            count[destination] += have[source]
        ready = holes & (count > 0)
        if not np.any(ready):
            break
        out[ready] = accumulated[ready] / count[ready]
    return out


def _component_owner_of_source(labels, source_lat, source_lon,
                               target_lat, target_lon, source_shape):
    """Which target component each source cell belongs to (0 = none).

    A source cell is claimed by the component holding most of the target
    cells that fall nearest to it.  That is what confines a component's
    donor set to its own body: donors are selected by identity, not by
    distance, so no radius can leak another basin in.
    """
    ny, nx = source_shape
    lat = np.asarray(source_lat, dtype=np.float64)
    lon = np.asarray(source_lon, dtype=np.float64)
    y = np.rint((np.asarray(target_lat, dtype=np.float64) - lat[0])
                / (lat[1] - lat[0])).astype(np.intp)
    x = np.rint((_target_longitude_in_source_frame(lon, target_lon) - lon[0])
                / (lon[1] - lon[0])).astype(np.intp)
    inside = (y >= 0) & (y < ny) & (x >= 0) & (x < nx) & (labels > 0)
    flat = (y[inside] * nx + x[inside]).astype(np.int64)
    lab = labels[inside].astype(np.int64)
    if flat.size == 0:
        return np.zeros(source_shape, dtype=np.int32)
    n_labels = int(labels.max()) + 1
    key = flat * n_labels + lab
    unique, counts = np.unique(key, return_counts=True)
    cell = unique // n_labels
    which = unique % n_labels
    owner_flat = np.zeros(ny * nx, dtype=np.int32)
    best = np.zeros(ny * nx, dtype=np.int64)
    order = np.argsort(counts)
    for c, w, n in zip(cell[order], which[order], counts[order]):
        if n >= best[c]:
            best[c] = n
            owner_flat[c] = w
    return owner_flat.reshape(source_shape)


# ---------------------------------------------------------------------------
# the assembled field
# ---------------------------------------------------------------------------
def assemble_water_temperature(
        *, mapped_sst, mapped_skin, target_land, target_lake,
        source_sst=None, source_lat=None, source_lon=None,
        target_lat=None, target_lon=None,
        policy=DEFAULT_WATER_TEMPERATURE_POLICY):
    """Return ``(water_temperature, water_temperature_source, receipt)``.

    ``water_temperature`` is finished: every water cell carries a physical
    temperature chosen by ONE provider for its whole connected body.  Land
    cells carry the mapped skin temperature so the array is total, and the
    soil reconciler still decides what land does with it.
    """
    policy = validate_water_temperature_policy(policy)
    skin = np.asarray(mapped_skin, dtype=np.float64)
    land = np.asarray(target_land, dtype=bool)
    if skin.shape != land.shape:
        raise ValueError("mapped_skin and target_land shapes differ")
    shape = skin.shape
    water = ~land
    values = skin.copy()
    source = np.full(shape, SOURCE_LAND, dtype=np.int8)

    if policy in ("wrf_compat", "external_overlay"):
        # Byte-for-byte the historical selector, kept reachable so a
        # stock-WRF certification can still ask for it by name.  An absent
        # SST is the selector's own "no SST here", which is how the shipped
        # code read a source that carries none.
        #
        # A declared overlay takes this arm too, and takes it unchanged: the
        # overlay has already written one analysis into BOTH source fields
        # over water, so the per-cell choice has nothing left to switch
        # between, and every fingerprint an overlay run published before
        # this module existed still reproduces.
        sst = (np.full(shape, np.nan, dtype=np.float64) if mapped_sst is None
               else np.asarray(mapped_sst, dtype=np.float64))
        admissible = (np.isfinite(sst)
                      & (sst >= MIN_WATER_TEMPERATURE_K)
                      & (sst <= MAX_WATER_TEMPERATURE_K))
        values = np.where(admissible, sst, skin)
        source = np.where(water, np.int8(SOURCE_PER_CELL),
                          np.int8(SOURCE_LAND)).astype(np.int8)
        receipt = {
            "policy": policy,
            "water_cells": int(water.sum()),
            "components": 0,
            "per_provider": {
                SOURCE_NAMES[SOURCE_PER_CELL]: int(water.sum())},
            "components_on_analysis": 0,
            "components_on_skin": 0,
        }
        return values.astype(np.float64), source, receipt

    lake = np.asarray(target_lake, dtype=bool) & water
    labels, classes = label_surface_components(land, lake)

    per_provider = {name: 0 for name in SOURCE_NAMES.values()}
    on_analysis = 0
    on_skin = 0
    component_rows = []

    have_source = (source_sst is not None and source_lat is not None
                   and source_lon is not None and target_lat is not None
                   and target_lon is not None)
    if have_source:
        source_sst = np.asarray(source_sst, dtype=np.float64)
        corners = _bilinear_corners(source_lat, source_lon,
                                    target_lat, target_lon)
        owner = _component_owner_of_source(
            labels, source_lat, source_lon, target_lat, target_lon,
            source_sst.shape)
    else:
        corners = None
        owner = None

    for label in sorted(classes):
        selection = labels == label
        cells = int(selection.sum())
        if cells == 0:
            continue
        chosen = None
        coverage = 0.0
        donor_count = 0
        if have_source:
            donors = (owner == label) & np.isfinite(source_sst)
            donors &= ((source_sst >= MIN_WATER_TEMPERATURE_K)
                       & (source_sst <= MAX_WATER_TEMPERATURE_K))
            donor_count = int(donors.sum())
            if donor_count:
                estimate = normalized_masked_bilinear(
                    source_sst, donors, corners, shape)
                covered = selection & np.isfinite(estimate)
                coverage = covered.sum() / cells
                if coverage >= MIN_COMPONENT_COVERAGE:
                    filled = _fill_within_component(estimate, selection)
                    chosen = filled
        if chosen is None:
            # The whole body takes the coherent skin field, not a per-cell
            # mixture with whatever SST happened to reach part of it.
            values[selection] = skin[selection]
            source[selection] = SOURCE_COMPONENT_SKIN
            per_provider[SOURCE_NAMES[SOURCE_COMPONENT_SKIN]] += cells
            on_skin += 1
            provider_name = SOURCE_NAMES[SOURCE_COMPONENT_SKIN]
        else:
            usable = selection & np.isfinite(chosen)
            values[usable] = chosen[usable]
            source[usable] = SOURCE_ANALYSIS
            leftover = selection & ~usable
            if np.any(leftover):
                values[leftover] = skin[leftover]
                source[leftover] = SOURCE_COMPONENT_SKIN
                per_provider[SOURCE_NAMES[SOURCE_COMPONENT_SKIN]] += int(
                    leftover.sum())
            per_provider[SOURCE_NAMES[SOURCE_ANALYSIS]] += int(usable.sum())
            on_analysis += 1
            provider_name = SOURCE_NAMES[SOURCE_ANALYSIS]
        component_rows.append({
            "label": int(label), "class": classes[label], "cells": cells,
            "donors": donor_count, "coverage": float(coverage),
            "provider": provider_name})

    bad = water & ~(np.isfinite(values)
                    & (values >= MIN_WATER_TEMPERATURE_K)
                    & (values <= MAX_WATER_TEMPERATURE_K))
    if np.any(bad):
        raise ValueError(
            f"{int(bad.sum())} water cells have no admissible water "
            "temperature after class-coherent assembly")
    if np.any(water & (source == SOURCE_LAND)):
        raise ValueError("a water cell was left without a declared provider")

    receipt = {
        "policy": policy,
        "water_cells": int(water.sum()),
        "lake_cells": int(lake.sum()),
        "ocean_cells": int((water & ~lake).sum()),
        "components": len(component_rows),
        "components_on_analysis": on_analysis,
        "components_on_skin": on_skin,
        "per_provider": {k: v for k, v in per_provider.items() if v},
        "component_detail": component_rows,
    }
    return values, source, receipt


# ---------------------------------------------------------------------------
# the one seam every forcing route crosses
# ---------------------------------------------------------------------------
#: The mapped fields the historical per-cell selector chose BETWEEN.  A
#: mapping that carries both is a mapping the fuse can silently quilt.
_RAW_WATER_PAIR = ("SST", "SKINTEMP")


def lake_mask_from_landuse(lu_index, landuse_attrs, *, route):
    """``(mask, lake_category)`` for the target's inland water.

    The category is the selected land-use table's OWN ``ISLAKE``, never a
    constant: MODIS writes 21, and USGS 24-category has no inland-water
    class at all, for which WPS writes ``ISLAKE = -1``.

    ``lake_category`` is ``None`` in exactly that second case, and the mask
    is then all-False because the table genuinely cannot name a lake.  The
    caller must state that rather than let it pass as "this domain has no
    lakes" -- which is the failure mode this returns two values to prevent.
    A mapping with no ``ISLAKE`` at all is a refusal, because that is a
    caller that never resolved its land-use table.
    """
    if landuse_attrs is None or "ISLAKE" not in landuse_attrs:
        raise ValueError(
            f"{route}: the water-temperature assembly needs the selected "
            "land-use table's ISLAKE to tell an inland body from the ocean; "
            "the caller supplied no land-use attributes.  Resolve them from "
            "the GEOG index (GeogSelection.landuse_global_attrs) rather "
            "than assuming a category number.")
    category = int(landuse_attrs["ISLAKE"])
    if category < 1:
        return np.zeros(np.shape(lu_index), dtype=bool), None
    return np.asarray(lu_index) == category, category


#: ``eq=False`` on both containers below: they hold arrays, and a
#: generated ``__eq__`` would compare those elementwise and then die on
#: the truth value of the result.  Identity comparison is what callers
#: of these actually want.
@dataclass(frozen=True, eq=False)
class WaterTemperatureStatics:
    """One route's declaration of the surface it is assembling over.

    ``route`` names the caller in every refusal and rides the receipt, so a
    false or missing assembly can be attributed without reading the
    traceback.  ``lake_category`` is ``None`` when the land-use table names
    no lake class; the advisory says so out loud.
    """

    route: str
    land: np.ndarray
    lake: np.ndarray
    lake_category: int | None
    policy: str

    @classmethod
    def for_route(cls, *, route, policy, landmask, lu_index, landuse_attrs,
                  lake_override=False):
        """Build the statics from a route's own geogrid fields.

        ``landmask`` is the geogrid ``LANDMASK`` the soil router will decide
        land and water with, so the two agree by construction.

        ``lake_override=True`` when the caller ALSO hands the soil router a
        ``lake_mask``/``lake_skin_temperature`` pair.  That override makes
        every land-use lake cell water inside the router, so the assembly
        has to class it as water too; otherwise the router would read a land
        value out of the assembled field on exactly those cells.
        """
        land = np.asarray(landmask) >= 0.5
        lake, category = lake_mask_from_landuse(
            lu_index, landuse_attrs, route=route)
        if lake.shape != land.shape:
            raise ValueError(f"{route}: LU_INDEX and LANDMASK shapes differ")
        if lake_override:
            land = land & ~lake
        return cls(route=str(route), land=land, lake=lake,
                   lake_category=category,
                   policy=validate_water_temperature_policy(policy))


@dataclass(frozen=True, eq=False)
class WaterTemperatureAssembly:
    """What a route hands the soil router, and the receipt that names it."""

    values: np.ndarray
    provider: np.ndarray
    receipt: Mapping[str, object]
    statics: WaterTemperatureStatics


def assemble_for_route(statics, *, mapped_sst, mapped_skin, source_sst=None,
                       source_lat=None, source_lon=None,
                       target_lat=None, target_lon=None):
    """THE assembly entry point.  Every forcing route reaches it here.

    Closing this route by route is what produced the quilt in the first
    place, so there is one function, it takes the route's own statics, and
    :func:`require_assembled_water_temperature` refuses the routes that
    have not been through it.
    """
    if not isinstance(statics, WaterTemperatureStatics):
        raise TypeError("assemble_for_route needs WaterTemperatureStatics")
    values, provider, receipt = assemble_water_temperature(
        mapped_sst=mapped_sst, mapped_skin=mapped_skin,
        target_land=statics.land, target_lake=statics.lake,
        source_sst=source_sst, source_lat=source_lat, source_lon=source_lon,
        target_lat=target_lat, target_lon=target_lon,
        policy=statics.policy)
    receipt = dict(receipt)
    receipt["route"] = statics.route
    receipt["lake_class"] = (
        "declared" if statics.lake_category is not None
        else "absent_from_land_use_table")
    return WaterTemperatureAssembly(
        values=values, provider=provider, receipt=receipt, statics=statics)


def require_assembled_water_temperature(*, route, fields, water_temperature,
                                        policy):
    """Refuse a soil call that would silently rerun the per-cell fuse.

    The fuse is ``where(admissible(SST), SST, SKINTEMP)``.  It can only
    DIFFER from the assembled field where the mapping actually carries an
    SST beside its SKINTEMP -- absent an SST the selector reduces to
    ``SKINTEMP`` on every water cell, which is exactly what the assembly
    returns for a body with no donors, so those routes are the identity and
    are let through.  Where a mapped SST does exist, silence is the defect:
    the caller has two differently-mapped fields and no decision, so it is
    refused by name and told where the decision belongs.

    ``wrf_compat`` is the one declaration that reopens the fuse, because
    reproducing stock WRF byte for byte is the whole reason that policy has
    a name.
    """
    if water_temperature is not None:
        return
    if policy == "wrf_compat":
        return
    if not all(name in fields for name in _RAW_WATER_PAIR):
        return
    raise ValueError(
        f"{route or 'an unnamed forcing route'} handed soil preprocessing a "
        "raw SST beside its SKINTEMP with no assembled water temperature.  "
        "Choosing between two differently-mapped fields cell by cell is "
        "what painted lakes as a quilt of both.  Route this source's own "
        "statics through "
        "gpuwm.ingest.water_temperature.assemble_for_route and forward the "
        "result as water_temperature, or declare "
        "water_temperature_policy = \"wrf_compat\" to ask for the "
        "historical selector on purpose.")


def water_temperature_advisory(receipt):
    """One plain line naming the policy and what it did, or None."""
    if not receipt or receipt.get("policy") != "era5_class_coherent":
        return None
    if not receipt.get("water_cells"):
        return None
    counts = ", ".join(
        f"{name} {count}"
        for name, count in sorted(receipt.get("per_provider", {}).items()))
    route = receipt.get("route")
    where = f" on {route}" if route else ""
    # LOUD when the guarantee is narrower than the headline: a land-use
    # table with no lake class cannot keep a lake and the ocean in separate
    # provider decisions, and silence there would read as "this domain has
    # no lakes".
    caveat = ""
    if receipt.get("lake_class") == "absent_from_land_use_table":
        caveat = (" The selected land-use table names no lake category, so "
                  "inland water is classed with the ocean here and a lake "
                  "joined to the sea by a coarse coastline can share its "
                  "provider.")
    return (
        f"water temperature: policy era5_class_coherent{where} over "
        f"{receipt['water_cells']} water cells in {receipt['components']} "
        f"connected bodies ({counts}).{caveat} Declaring "
        "water_temperature_overlay with an observational analysis remains "
        "the higher-accuracy option.")


def announce_water_temperature(receipt, *, seen=None, scope=None):
    """Print the advisory once per ``(scope, message)``, or do nothing.

    ``scope`` is whatever identifies the domain to the caller (the mass-grid
    shape, on the mapping path).  Every forcing time of a domain assembles
    the same providers over the same cells, so the line belongs to the
    domain and not to the snapshot.
    """
    advisory = water_temperature_advisory(receipt)
    if advisory is None:
        return None
    if seen is not None:
        key = (scope, advisory)
        if key in seen:
            return advisory
        seen.add(key)
    print(advisory, file=sys.stderr)
    return advisory


__all__ = [
    "MODIS_LAKE_CATEGORY",
    "WATER_TEMPERATURE_POLICIES",
    "DEFAULT_WATER_TEMPERATURE_POLICY",
    "SOURCE_LAND", "SOURCE_ANALYSIS", "SOURCE_COMPONENT_SKIN",
    "SOURCE_PER_CELL", "SOURCE_NAMES",
    "WaterTemperatureAssembly",
    "WaterTemperatureStatics",
    "validate_water_temperature_policy",
    "resolve_water_temperature_policy",
    "lake_mask_from_landuse",
    "label_surface_components",
    "normalized_masked_bilinear",
    "assemble_water_temperature",
    "assemble_for_route",
    "require_assembled_water_temperature",
    "water_temperature_advisory",
    "announce_water_temperature",
]
