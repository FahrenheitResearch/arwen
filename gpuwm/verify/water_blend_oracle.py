"""The lake skin search and the water-temperature blends, as NumPy did them.

This module is a TEST ORACLE and nothing else.  The Rust preprocessing
library runs these steps at preparation time
(``tools/grib1_bridge/src/water_blend.rs``, entries
``gpuwm_lake_water_nearest_f64``, ``gpuwm_masked_bilinear_blend_f64``,
``gpuwm_component_fill_f64``, ``gpuwm_overlay_bilinear_sample_f64``,
``gpuwm_label_components_8``, in ``water_repair.rs``
``gpuwm_water_repair_f64`` and ``gpuwm_water_bodies_f64``, in
``water_owner.rs`` ``gpuwm_component_owner_f64`` and in
``surface_nearest.rs`` ``gpuwm_masked_nearest_f32``), and
``tests/test_water_blend_native.py``, ``tests/test_water_repair_native.py``
and ``tests/test_water_owner_native.py`` hold every output to the code
below byte for byte.  The functions are moved verbatim from
``gpuwm/ingest/horiz.py``, ``gpuwm/ingest/water_temperature.py``,
``gpuwm/ingest/water_overlay.py`` and
``gpuwm/ingest/preprocess_backend.py`` (the owner's count sort made
stable, as noted there), with the box repairs and the old
per-body loop of ``assemble_water_temperature`` (kept here whole, so the
tests compare the native assembly with it end to end); nothing at run
time imports this
module (a test enforces it), so it is never a silent NumPy fallback.
"""

from __future__ import annotations

import numpy as np


def _nearest_finite_source_water(skin, water, y: float, x: float) -> float:
    """Return the global Euclidean-nearest water value in source indices.

    The window begins at metgrid's established masked-search radius, then
    expands until the best point is provably closer than every point outside
    the searched rectangle.  Thus remote inland lakes are supported without
    allocating a target-by-global-source distance matrix.
    """
    ny, nx = skin.shape
    radius = 8.0
    while True:
        j0 = max(0, int(np.ceil(y - radius)))
        j1 = min(ny - 1, int(np.floor(y + radius)))
        i0 = max(0, int(np.ceil(x - radius)))
        i1 = min(nx - 1, int(np.floor(x + radius)))
        local = water[j0:j1 + 1, i0:i1 + 1]
        rows, cols = np.nonzero(local)
        if rows.size:
            rows = rows + j0
            cols = cols + i0
            distance_squared = (rows - y) ** 2 + (cols - x) ** 2
            nearest = int(np.argmin(distance_squared))
            best_squared = float(distance_squared[nearest])
            outside_distance = []
            if j0 > 0:
                outside_distance.append(y - (j0 - 1))
            if j1 < ny - 1:
                outside_distance.append((j1 + 1) - y)
            if i0 > 0:
                outside_distance.append(x - (i0 - 1))
            if i1 < nx - 1:
                outside_distance.append((i1 + 1) - x)
            # Strict comparison forces expansion on a possible distance tie;
            # the final np.argmin therefore preserves global row-major order.
            if (not outside_distance
                    or best_squared < min(outside_distance) ** 2):
                return float(skin[rows[nearest], cols[nearest]])
        if j0 == 0 and j1 == ny - 1 and i0 == 0 and i1 == nx - 1:
            raise RuntimeError("global source-water search lost validated support")
        radius *= 2.0


def lake_skin_search(skin, water, y, x, lakes):
    """The per-lake loop of ``interpolate_lake_skin_temperature``."""
    result = np.full(lakes.shape, np.nan, dtype=np.float64)
    for j, i in np.argwhere(lakes):
        result[j, i] = _nearest_finite_source_water(
            skin, water, float(y[j, i]), float(x[j, i]))
    return result


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


def masked_bilinear_sample(
        overlay: WaterTemperatureOverlay, target_lat, target_lon
        ) -> tuple[np.ndarray, np.ndarray]:
    """Sample the overlay at target points; invalid corners are excluded.

    Returns ``(values, covered)``: bilinear temperatures with weights
    renormalized over VALID corners only, and a bool coverage mask.  A
    point outside the overlay bounding box, or whose four corners are
    all invalid, is not covered (``values`` is NaN there).
    """

    latitude = overlay.latitude
    longitude = overlay.longitude
    target_lat = np.asarray(target_lat, dtype=np.float64)
    target_lon = np.asarray(target_lon, dtype=np.float64)
    # The horiz.py midpoint convention: unwrap targets into the frame
    # the overlay's finite axis occupies.
    lon_mid = 0.5 * (longitude[0] + longitude[-1])
    unwrapped = target_lon + 360.0 * np.round(
        (lon_mid - target_lon) / 360.0)
    inside = ((target_lat >= latitude[0]) & (target_lat <= latitude[-1])
              & (unwrapped >= longitude[0]) & (unwrapped <= longitude[-1]))
    y = np.interp(target_lat, latitude, np.arange(latitude.size))
    x = np.interp(unwrapped, longitude, np.arange(longitude.size))
    y0 = np.clip(np.floor(y).astype(np.intp), 0, latitude.size - 2)
    x0 = np.clip(np.floor(x).astype(np.intp), 0, longitude.size - 2)
    fy = np.clip(y - y0, 0.0, 1.0)
    fx = np.clip(x - x0, 0.0, 1.0)
    total = np.zeros(target_lat.shape, dtype=np.float64)
    accumulated = np.zeros(target_lat.shape, dtype=np.float64)
    for dj, di, weight in (
            (0, 0, (1.0 - fy) * (1.0 - fx)),
            (0, 1, (1.0 - fy) * fx),
            (1, 0, fy * (1.0 - fx)),
            (1, 1, fy * fx)):
        corner_valid = overlay.valid[y0 + dj, x0 + di]
        corner_value = overlay.temperature_k[y0 + dj, x0 + di]
        contribution = np.where(corner_valid, weight, 0.0)
        total += contribution
        accumulated += contribution * np.where(corner_valid,
                                               corner_value, 0.0)
    covered = inside & (total > 0.0)
    values = np.full(target_lat.shape, np.nan, dtype=np.float64)
    np.divide(accumulated, total, out=values, where=covered)
    return values, covered


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


# ---------------------------------------------------------------------------
# the box repairs and the per-body assembly (gpuwm_water_repair_f64,
# gpuwm_water_bodies_f64), moved verbatim from
# gpuwm/ingest/water_temperature.py
# ---------------------------------------------------------------------------
from gpuwm.ingest.water_temperature import (  # noqa: E402
    DEFAULT_WATER_TEMPERATURE_POLICY, MAX_LISTED_DECLINED_CELLS,
    MAX_WATER_TEMPERATURE_K, MIN_COMPONENT_COVERAGE, MIN_WATER_TEMPERATURE_K,
    SOURCE_ANALYSIS, SOURCE_COMPONENT_SKIN, SOURCE_LAKE_WATER, SOURCE_LAND,
    SOURCE_NAMES, SOURCE_NEAREST_WATER, SOURCE_PER_CELL,
    SOURCE_SURROUNDING_SKIN, _bilinear_corners,
    _target_longitude_in_source_frame, _water_temperature_refusal,
    label_surface_components, validate_water_temperature_policy)


# gpuwm_component_owner_f64, moved verbatim from
# gpuwm/ingest/water_temperature.py.  One change: the counts are sorted
# stably.  With NumPy's default unstable argsort the body that won a tie
# between two equally large claims on one source cell depended on the
# sort kernel of the machine; stable, the highest label wins, everywhere.
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
    order = np.argsort(counts, kind="stable")
    for c, w, n in zip(cell[order], which[order], counts[order]):
        if n >= best[c]:
            best[c] = n
            owner_flat[c] = w
    return owner_flat.reshape(source_shape)


def _admissible(values):
    """Finite and inside the physical window the soil reconciler applies."""
    return (np.isfinite(values)
            & (values >= MIN_WATER_TEMPERATURE_K)
            & (values <= MAX_WATER_TEMPERATURE_K))


def _label_boxes(labels, wanted):
    """``{label: box}`` for each wanted label, widened by one cell.

    One pass over the labelled cells, so a domain with thousands of small
    lakes never compares the whole label field once per lake.
    """
    rows, cols = np.nonzero(labels)
    which = labels[rows, cols].astype(np.int64)
    size = int(labels.max()) + 1
    ny, nx = labels.shape
    j0 = np.full(size, ny, dtype=np.int64)
    j1 = np.full(size, -1, dtype=np.int64)
    i0 = np.full(size, nx, dtype=np.int64)
    i1 = np.full(size, -1, dtype=np.int64)
    np.minimum.at(j0, which, rows)
    np.maximum.at(j1, which, rows)
    np.minimum.at(i0, which, cols)
    np.maximum.at(i1, which, cols)
    return {int(label): np.s_[max(0, j0[label] - 1):min(ny, j1[label] + 2),
                              max(0, i0[label] - 1):min(nx, i1[label] + 2)]
            for label in wanted}


def _propagate_into(values, holes, have):
    """Close ``holes`` ring by ring from their 8-neighbours in ``have``.

    Each sweep gives every hole touching a seed the mean of those seeds,
    and the ring it filled seeds the next sweep, so a body closes from its
    edge inward with no jump inside it.  ``values`` is written in place;
    the holes no seed could reach are returned.
    """
    open_ = np.array(holes, dtype=bool, copy=True)
    seeded = np.asarray(have, dtype=bool) & ~open_
    ny, nx = values.shape
    while np.any(open_):
        padded_value = np.pad(np.where(seeded, values, 0.0), 1)
        padded_seed = np.pad(seeded, 1).astype(np.float64)
        total = np.zeros(values.shape, dtype=np.float64)
        count = np.zeros(values.shape, dtype=np.float64)
        for dj in (-1, 0, 1):
            for di in (-1, 0, 1):
                if dj or di:
                    window = np.s_[1 + dj:1 + dj + ny, 1 + di:1 + di + nx]
                    total += padded_value[window]
                    count += padded_seed[window]
        ready = open_ & (count > 0)
        if not np.any(ready):
            break
        values[ready] = total[ready] / count[ready]
        open_ &= ~ready
        seeded |= ready
    return open_


def _edge_cells(mask):
    """Cells of ``mask`` with an 8-neighbour outside it or off the grid."""
    height, width = mask.shape
    padded = np.pad(mask, 1)
    interior = np.array(mask, dtype=bool, copy=True)
    for dj in (-1, 0, 1):
        for di in (-1, 0, 1):
            interior &= padded[1 + dj:1 + dj + height, 1 + di:1 + di + width]
    return mask & ~interior


def _donor_index(donors):
    """Row-major ``(rows, columns)`` of the donors that can ever be nearest.

    A donor whose eight neighbours are all donors is never the nearest one
    to a cell outside the set, because its neighbour toward that cell is
    strictly nearer.  Only the edge is kept, so an ocean costs a search its
    coastline and not its area.
    """
    return np.nonzero(_edge_cells(np.asarray(donors, dtype=bool)))


def _nearest_donor(body, box, index, shape):
    """``(row, column)`` of the donor nearest any cell of ``body``, or None.

    ``body`` is a mask over ``box`` of a grid of ``shape``, and ``index``
    is :func:`_donor_index` of the donors.  Euclidean distance in grid
    cells.  The search starts on the body's box widened by eight cells and
    doubles until the best donor is provably nearer than anything outside
    the window; donors are read from ``index`` by row band, so neither an
    empty stretch of the domain nor a distant ocean is scanned cell by
    cell.  Ties go to the first donor in row-major order.
    """
    donor_rows, donor_cols = index
    if donor_rows.size == 0:
        return None
    ny, nx = shape
    # Only the edge of a body can be nearest to something outside it.
    edge_rows, edge_cols = np.nonzero(_edge_cells(body))
    edge_rows = edge_rows + box[0].start
    edge_cols = edge_cols + box[1].start
    j0, j1 = int(edge_rows.min()), int(edge_rows.max())
    i0, i1 = int(edge_cols.min()), int(edge_cols.max())
    radius = 8
    while True:
        wj0, wj1 = max(0, j0 - radius), min(ny - 1, j1 + radius)
        wi0, wi1 = max(0, i0 - radius), min(nx - 1, i1 + radius)
        whole = wj0 == 0 and wi0 == 0 and wj1 == ny - 1 and wi1 == nx - 1
        lo = int(np.searchsorted(donor_rows, wj0, side="left"))
        hi = int(np.searchsorted(donor_rows, wj1, side="right"))
        rows, cols = donor_rows[lo:hi], donor_cols[lo:hi]
        inside = (cols >= wi0) & (cols <= wi1)
        rows, cols = rows[inside], cols[inside]
        if rows.size:
            best_squared = best_index = None
            # Chunked so a wide window never builds a donor-by-edge matrix
            # of more than about a million entries.
            step = max(1, 1_000_000 // edge_rows.size)
            for start in range(0, rows.size, step):
                squared = ((rows[start:start + step, None] - edge_rows) ** 2
                           + (cols[start:start + step, None] - edge_cols) ** 2
                           ).min(axis=1)
                local = int(np.argmin(squared))
                if best_squared is None or squared[local] < best_squared:
                    best_squared = int(squared[local])
                    best_index = start + local
            # A donor outside the window is at least radius + 1 cells from
            # every cell of the body along one axis.
            if whole or best_squared < (radius + 1) ** 2:
                return int(rows[best_index]), int(cols[best_index])
        elif whole:
            return None
        radius *= 2


def _fill_missing_water_temperature(values, source, water, labels):
    """Give a temperature to every water cell its provider left without one.

    A cell reaches here when the provider its body chose left it no
    admissible temperature.  On a regional crop of a coarse source that is
    the ordinary case for a lake: the source resolves no water anywhere
    near it, the masked skin interpolation had no water donor to read, and
    the chain left its fill value, zero.  Refusing the preparation for it
    refused a domain over the extent of the source crop, so each such cell
    is filled, in this order, and every fill is counted:

    1. from its own body's admissible water, ring by ring from the cells
       that have it;
    2. a body with none takes, as one value, the nearest admissible water
       on the domain: source water one of the providers above carried in.
       "Near" has the reach the masked source-water search already has,
       the whole of the data the preparation holds;
    3. with no admissible water on the domain, the body takes the skin
       temperature around it, ring by ring from its shore, and a body
       whose shore has none takes the nearest admissible skin.

    Returns ``(values, source, counts, filled)``.  A cell none of the three
    can reach stays inadmissible for the caller's refusal: nothing on the
    domain carries an admissible surface temperature, which is a defect of
    the source rather than of its crop.
    """
    bad = water & ~_admissible(values)
    counts = {"own_body": 0, "nearest_water": 0, "surrounding_skin": 0}
    if not np.any(bad):
        return values, source, counts, bad
    values = np.array(values, dtype=np.float64, copy=True)
    source = np.array(source, copy=True)
    # The donors of step 2 are the source water the providers placed; a
    # cell this function fills is never a donor for another body.
    donors = water & ~bad
    wanted = sorted(int(label) for label in np.unique(labels[bad]) if label)
    boxes = _label_boxes(labels, wanted)
    whole_bodies = []
    for label in wanted:
        box = boxes[label]
        body = labels[box] == label
        holes = body & bad[box]
        have = body & donors[box]
        if not np.any(have):
            whole_bodies.append(label)
            continue
        # A body is 8-connected, so its own water reaches every hole in it.
        _propagate_into(values[box], holes, have)
        source[box][holes] = SOURCE_NEAREST_WATER
        counts["own_body"] += int(holes.sum())
    skin_bodies = []
    index = _donor_index(donors) if whole_bodies else None
    for label in whole_bodies:
        box = boxes[label]
        body = labels[box] == label
        cell = _nearest_donor(body, box, index, labels.shape)
        if cell is None:
            skin_bodies.append(label)
            continue
        values[box][body] = values[cell]
        source[box][body] = SOURCE_NEAREST_WATER
        counts["nearest_water"] += int(body.sum())
    stranded = []
    for label in skin_bodies:
        box = boxes[label]
        body = labels[box] == label
        left = _propagate_into(
            values[box], body, _admissible(values[box]) & ~body)
        source[box][body & ~left] = SOURCE_SURROUNDING_SKIN
        counts["surrounding_skin"] += int((body & ~left).sum())
        if np.any(left):
            stranded.append((label, left))
    if stranded:
        # A shore with no admissible skin of its own: the nearest
        # admissible cell anywhere, or nothing when the domain has none.
        anywhere = _donor_index(_admissible(values))
        for label, left in stranded:
            box = boxes[label]
            cell = _nearest_donor(left, box, anywhere, labels.shape)
            if cell is None:
                continue
            values[box][left] = values[cell]
            source[box][left] = SOURCE_SURROUNDING_SKIN
            counts["surrounding_skin"] += int(left.sum())
    filled = bad & _admissible(values)
    return values, source, counts, filled


def assemble_water_temperature(
        *, mapped_sst, mapped_skin, target_land, target_lake,
        source_sst=None, source_lat=None, source_lon=None,
        target_lat=None, target_lon=None,
        policy=DEFAULT_WATER_TEMPERATURE_POLICY,
        diagnostic_context=None, diagnostic_latlon=None,
        mapped_lake_water=None):
    """Return ``(water_temperature, water_temperature_source, receipt)``.

    ``water_temperature`` is finished: every water cell carries a physical
    temperature attributed to its provider. Analysis is chosen for a whole
    connected body; optional lake-model water falls back to component skin
    at declined cells. Land cells carry mapped skin so the array is total,
    and the soil reconciler still decides what land does with it.  A water
    cell its provider left without an admissible temperature is filled by
    :func:`_fill_missing_water_temperature` and counted in the receipt's
    ``water_fill``.

    The optional diagnostic context and geographic latitude/longitude pair
    are used only to explain a refusal; they never enter interpolation.
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
    lake_water = (None if mapped_lake_water is None else
                  np.asarray(mapped_lake_water, dtype=np.float64))
    if lake_water is not None and lake_water.shape != shape:
        raise ValueError("mapped_lake_water and target_land shapes differ")
    labels, classes = label_surface_components(land, lake)

    per_provider = {name: 0 for name in SOURCE_NAMES.values()}
    on_analysis = 0
    on_skin = 0
    on_lake_water = 0
    lake_fallback = 0
    lake_fallback_cells: list = []
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
        provided = (selection & np.isfinite(lake_water)
                    if classes[label] == "lake" and lake_water is not None
                    else None)
        if chosen is None and provided is not None and np.any(provided):
            # The provider is an explicitly decoded lake-model water state.
            # Tiny inland lakes can have no majority-water source cell at
            # all, so it is chosen for the component wherever it answers.
            # Where it DECLINED a cell (a frozen or unknown-depth donor, an
            # inadmissible temperature -- see lake_temperature) that cell
            # falls back to the source this component had before the
            # provider existed, its coherent skin temperature, and is
            # counted and named in the receipt: refusing the preparation
            # for it was a default-on blocker on a route that ran in 2.6.5
            # (ENG-008).  Nothing here invents a temperature.
            values[provided] = lake_water[provided]
            source[provided] = SOURCE_LAKE_WATER
            per_provider[SOURCE_NAMES[SOURCE_LAKE_WATER]] += int(provided.sum())
            declined = selection & ~provided
            if np.any(declined):
                values[declined] = skin[declined]
                source[declined] = SOURCE_COMPONENT_SKIN
                per_provider[SOURCE_NAMES[SOURCE_COMPONENT_SKIN]] += int(
                    declined.sum())
            on_lake_water += 1
            provider_name = SOURCE_NAMES[SOURCE_LAKE_WATER]
        elif chosen is None:
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
        if chosen is None and provided is not None:
            # Count partial and wholly declined components alike. A frozen
            # lake commonly takes the all-skin branch above; it must still
            # be named by the preparation advisory. Limit the cell list for
            # the whole domain, rather than separately for every lake.
            declined = selection & ~provided
            lake_fallback += int(declined.sum())
            remaining = MAX_LISTED_DECLINED_CELLS - len(lake_fallback_cells)
            if remaining > 0:
                lake_fallback_cells.extend(
                    [int(j), int(i)]
                    for j, i in np.argwhere(declined)[:remaining])
        component_rows.append({
            "label": int(label), "class": classes[label], "cells": cells,
            "donors": donor_count, "coverage": float(coverage),
            "provider": provider_name})

    values, source, fill_counts, filled = _fill_missing_water_temperature(
        values, source, water, labels)
    bad = water & ~_admissible(values)
    if np.any(bad):
        raise _water_temperature_refusal(
            bad=bad, values=values, skin=skin, mapped_sst=mapped_sst,
            source=source, labels=labels, component_rows=component_rows,
            diagnostic_context=diagnostic_context,
            diagnostic_latlon=diagnostic_latlon)
    if np.any(water & (source == SOURCE_LAND)):
        raise ValueError("a water cell was left without a declared provider")
    if np.any(filled):
        # A filled cell changed provider, so the tally is read back off
        # the provider field rather than kept beside it.
        per_provider = {
            name: int(np.count_nonzero(water & (source == key)))
            for key, name in SOURCE_NAMES.items()}

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
    if lake_water is not None:
        receipt["components_on_lake_water"] = on_lake_water
        receipt["lake_fallback_cells"] = lake_fallback
        receipt["lake_fallback_cell_indices"] = lake_fallback_cells
    if np.any(filled):
        receipt["water_fill"] = {
            "cells": int(filled.sum()),
            **fill_counts,
            "cell_indices": [
                [int(j), int(i)]
                for j, i in np.argwhere(filled)[:MAX_LISTED_DECLINED_CELLS]],
        }
    return values, source, receipt


# ---------------------------------------------------------------------------
# the bounded surface-nearest search of the CPU backend
# (gpuwm_masked_nearest_f32), moved verbatim from
# gpuwm/ingest/preprocess_backend.py
# ---------------------------------------------------------------------------
from gpuwm.ingest.preprocess_backend import _host  # noqa: E402


def _masked_nearest_cpu(field, latitude, longitude, target_lat, target_lon,
                        source_landmask, target_landmask, *, surface="match",
                        fill_value=0.0, search_radius=8, strict=True):
    """NumPy FP32 mirror of the bounded WPS surface-nearest operator."""

    from gpuwm.ingest.horiz import _regular_coordinates

    if isinstance(search_radius, (bool, np.bool_)) \
            or not isinstance(search_radius, (int, np.integer)) \
            or int(search_radius) < 0:
        raise ValueError("search_radius must be a non-negative integer")
    y_raw, x_raw = _regular_coordinates(
        latitude, longitude, target_lat, target_lon)
    field = np.asarray(_host(field), dtype=np.float32)
    source_landmask = np.asarray(_host(source_landmask), dtype=np.bool_)
    target_landmask = np.asarray(_host(target_landmask), dtype=np.bool_)
    source_shape = (len(latitude), len(longitude))
    if field.ndim != 2 or field.shape != source_shape \
            or source_landmask.shape != source_shape:
        raise ValueError("field/source_landmask shape does not match source axes")
    if target_landmask.shape != y_raw.shape:
        raise ValueError(
            "target_landmask shape does not match target coordinates")
    if surface == "match":
        active = np.ones_like(target_landmask)
        desired_land = target_landmask
    elif surface == "land":
        active = target_landmask
        desired_land = np.ones_like(target_landmask)
    elif surface == "water":
        active = ~target_landmask
        desired_land = np.zeros_like(target_landmask)
    else:
        raise ValueError("surface must be 'match', 'land', or 'water'")

    y = np.asarray(y_raw, dtype=np.float32)
    x = np.asarray(x_raw, dtype=np.float32)
    center_y = np.rint(y).astype(np.int32)
    center_x = np.rint(x).astype(np.int32)
    best_distance = np.full(y.shape, np.inf, dtype=np.float32)
    best_value = np.full(y.shape, np.float32(fill_value), dtype=np.float32)
    ny, nx = source_shape
    radius = int(search_radius)
    for dj in range(-radius, radius + 1):
        jy = center_y + dj
        in_y = (jy >= 0) & (jy < ny)
        jy_safe = np.clip(jy, 0, ny - 1)
        for di in range(-radius, radius + 1):
            ix = center_x + di
            inside = in_y & (ix >= 0) & (ix < nx)
            ix_safe = np.clip(ix, 0, nx - 1)
            value = field[jy_safe, ix_safe]
            valid = (active & inside & np.isfinite(value)
                     & (source_landmask[jy_safe, ix_safe] == desired_land))
            distance = (y - jy_safe) ** np.float32(2.0) \
                + (x - ix_safe) ** np.float32(2.0)
            take = valid & (distance < best_distance)
            best_distance = np.where(take, distance, best_distance)
            best_value = np.where(take, value, best_value)
    if strict and np.any(active & ~np.isfinite(best_distance)):
        raise ValueError("no matching source surface within search_radius")
    return np.ascontiguousarray(best_value, dtype=np.float32)
