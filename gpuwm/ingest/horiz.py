"""GPU horizontal interpolation from regular ERA5 to a Lambert WRF C grid.

The overlapping-parabolic path is a direct vector transcription of WPS
v4.6.0 ``geogrid/src/interp_module.F`` (``sixteen_pt`` and ``oned``); WPS
metgrid links that same interpolation module.  The rotation convention is
local WRF v4.6.1 ``share/wrf_fddaobs_in.F:rotate_vector``::

    u_grid = u_earth*cos(alpha) + v_earth*sin(alpha)
    v_grid = v_earth*cos(alpha) - u_earth*sin(alpha)

CuPy is imported only when a GPU operation is requested, keeping the ERA5
decoder and package import usable on CPU-only installations.  Source setup
axes and target Lambert coordinates remain float64; source device fields,
weights, and results are FP32.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Mapping

import numpy as np

from gpuwm.ingest.soil_contract import (
    MAPPED_SOIL_MOISTURE,
    MAPPED_SOIL_TEMPERATURE,
)

from gpuwm.ingest.grib import Era5Snapshot
from gpuwm.static.lambert import LambertGrid
from gpuwm.static.projection import ProjectedGrid


ERA5_Z_INVARIANT_PROVIDER = "era5_z_invariant"
_WPS_GRAVITY_MS2 = 9.81


def _cupy():
    try:
        import cupy as cp
    except ImportError as exc:  # pragma: no cover - exercised on CPU installs
        raise RuntimeError("CuPy is required for GPU horizontal interpolation") from exc
    return cp


@dataclass(frozen=True)
class HorizontalSnapshot:
    """One ERA5 valid time horizontally interpolated to a projected C grid.

    ``levels_hpa`` keeps the source ordering.  Fields are CuPy FP32 arrays;
    mass scalars end in ``(ny,nx)``, ``UU/U10`` in ``(ny,nx+1)``, and
    ``VV/V10`` in ``(ny+1,nx)``.  The validated setup-time
    ``SOURCE_OROGRAPHY`` field remains host FP64; all other fields are CuPy
    FP32 arrays.  Pressure-level output uses metgrid names
    ``TT``, ``UU``, ``VV``, and ``GHT`` (with geopotential converted using
    WPS's 9.81 m s-2 convention).
    """

    valid_time: datetime
    levels_hpa: np.ndarray
    fields: Mapping[str, object]

    def __post_init__(self) -> None:
        levels = np.asarray(self.levels_hpa)
        if levels.dtype != np.float64 or levels.ndim != 1 or levels.size == 0:
            raise TypeError("levels_hpa must be a non-empty float64 1-D array")
        copied = levels.copy()
        copied.setflags(write=False)
        object.__setattr__(self, "levels_hpa", copied)
        object.__setattr__(self, "fields", MappingProxyType(dict(self.fields)))


def _regular_coordinates(latitude, longitude, target_lat, target_lon):
    """Validate axes and return zero-based target coordinates in float64."""
    latitude = np.asarray(latitude, dtype=np.float64)
    longitude = np.asarray(longitude, dtype=np.float64)
    target_lat = np.asarray(target_lat, dtype=np.float64)
    target_lon = np.asarray(target_lon, dtype=np.float64)
    if latitude.ndim != 1 or longitude.ndim != 1:
        raise ValueError("source latitude and longitude must be 1-D")
    if latitude.size < 2 or longitude.size < 2:
        raise ValueError("source axes must each contain at least two points")
    if target_lat.shape != target_lon.shape:
        raise ValueError("target latitude and longitude shapes differ")
    if not (np.isfinite(latitude).all() and np.isfinite(longitude).all()
            and np.isfinite(target_lat).all() and np.isfinite(target_lon).all()):
        raise ValueError("coordinates must be finite")
    dlat = np.diff(latitude)
    dlon = np.diff(longitude)
    if dlat[0] == 0.0 or dlon[0] == 0.0:
        raise ValueError("source axes must be strictly monotonic")
    if not np.allclose(dlat, dlat[0], rtol=1.0e-11, atol=1.0e-12):
        raise ValueError("source latitude axis must be uniform")
    if not np.allclose(dlon, dlon[0], rtol=1.0e-11, atol=1.0e-12):
        raise ValueError(
            "source longitude axis must be uniform (a single jump inside "
            "the axis usually means the source was rotated across the "
            "antimeridian seam; keep the crop's longitudes continuous, "
            "extending past +/-180 where the crop crosses it)")
    lon_mid = 0.5 * (longitude[0] + longitude[-1])
    unwrapped_lon = target_lon + 360.0 * np.round((lon_mid - target_lon) / 360.0)
    y = (target_lat - latitude[0]) / dlat[0]
    x = (unwrapped_lon - longitude[0]) / dlon[0]
    eps = 2.0e-10
    if (np.min(x) < -eps or np.max(x) > longitude.size - 1 + eps
            or np.min(y) < -eps or np.max(y) > latitude.size - 1 + eps):
        raise ValueError("target points fall outside the source grid")
    return np.clip(y, 0.0, latitude.size - 1.0), np.clip(
        x, 0.0, longitude.size - 1.0)


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


def interpolate_lake_skin_temperature(
        snapshot: Era5Snapshot, grid: ProjectedGrid, lake_mask) -> np.ndarray:
    """Select WPS-style source-water ``SKINTEMP`` for raw GEOG lakes.

    ERA5's coarser LANDSEA can classify a small model lake as land.  WPS
    nevertheless initializes that GEOG lake from the nearest finite source
    water point.  This setup-time CPU helper performs that search globally
    and returns float64 values at lake cells; non-lake entries are NaN and
    must not be consumed.
    """
    if not isinstance(snapshot, Era5Snapshot):
        raise TypeError("snapshot must be an Era5Snapshot")
    if not isinstance(grid, ProjectedGrid):
        raise TypeError("grid must be a ProjectedGrid")
    target_lat, target_lon = grid.latlon_mass()
    raw_mask = np.asarray(lake_mask)
    if raw_mask.shape != target_lat.shape:
        raise ValueError(
            f"lake_mask has shape {raw_mask.shape}; expected {target_lat.shape}")
    if raw_mask.dtype != np.bool_:
        if (not np.issubdtype(raw_mask.dtype, np.number)
                or not np.isfinite(raw_mask).all()
                or np.any((raw_mask != 0) & (raw_mask != 1))):
            raise ValueError("lake_mask must contain only boolean/0/1 values")
    lakes = raw_mask.astype(bool, copy=False)
    result = np.full(target_lat.shape, np.nan, dtype=np.float64)
    if not np.any(lakes):
        return result

    missing = [name for name in ("LANDSEA", "SKINTEMP")
               if name not in snapshot.fields]
    if missing:
        raise KeyError(f"missing lake skin source fields: {missing}")
    landsea = np.asarray(snapshot.fields["LANDSEA"], dtype=np.float64)
    skin = np.asarray(snapshot.fields["SKINTEMP"], dtype=np.float64)
    source_shape = (snapshot.latitude.size, snapshot.longitude.size)
    if landsea.shape != source_shape or skin.shape != source_shape:
        raise ValueError("LANDSEA and SKINTEMP must be 2-D source fields")
    water = np.isfinite(landsea) & (landsea < 0.5) & np.isfinite(skin)
    if not np.any(water):
        raise ValueError("no finite source-water SKINTEMP is available")

    y, x = _regular_coordinates(
        snapshot.latitude, snapshot.longitude, target_lat, target_lon)
    for j, i in np.argwhere(lakes):
        result[j, i] = _nearest_finite_source_water(
            skin, water, float(y[j, i]), float(x[j, i]))
    return result


def _wps_oned_gpu(x, a, b, c, d):
    """FP32 CuPy transcription of WPS ``interp_module.F:oned``."""
    cp = _cupy()
    zero = cp.float32(0.0)
    half = cp.float32(0.5)
    one = cp.float32(1.0)
    regular = ((one - x)
               * (b + x * (half * (c - a) + x * (half * (c + a) - b)))
               + x * (c + (one - x)
                      * (half * (b - d) + (one - x)
                         * (half * (b + d) - c))))
    out = cp.zeros_like(regular)
    out = cp.where(x == zero, b, out)
    out = cp.where(x == one, c, out)
    both = b * c != zero
    only_a = both & (a != zero) & (d == zero)
    only_d = both & (a == zero) & (d != zero)
    neither = both & (a == zero) & (d == zero)
    all_four = both & (a != zero) & (d != zero)
    out = cp.where(neither, b * (one - x) + c * x, out)
    out = cp.where(
        only_a, b + x * (half * (c - a) + x * (half * (c + a) - b)), out)
    out = cp.where(
        only_d,
        c + (one - x) * (half * (b - d) + (one - x)
                          * (half * (b + d) - c)),
        out,
    )
    return cp.where(all_four, regular, out)


class _RegularGpuPlan:
    """Reusable device index/weight plan for one target staggering."""

    def __init__(self, latitude, longitude, target_lat, target_lon):
        cp = _cupy()
        y, x = _regular_coordinates(latitude, longitude, target_lat, target_lon)
        self.source_shape = (len(latitude), len(longitude))
        self.target_shape = y.shape
        self.y = cp.asarray(y, dtype=cp.float32)
        self.x = cp.asarray(x, dtype=cp.float32)

    def apply(self, field, method="parabolic"):
        cp = _cupy()
        field = cp.asarray(field, dtype=cp.float32)
        if field.ndim < 2 or field.shape[-2:] != self.source_shape:
            raise ValueError("field trailing dimensions do not match source axes")
        lead = (slice(None),) * (field.ndim - 2)
        expand = (None,) * (field.ndim - 2)
        ny, nx = self.source_shape
        if method == "nearest":
            iy = cp.rint(self.y).astype(cp.int32)
            ix = cp.rint(self.x).astype(cp.int32)
            return field[lead + (iy, ix)]
        if method == "bilinear":
            one = cp.float32(1.0)
            iy = cp.minimum(cp.floor(self.y).astype(cp.int32), ny - 2)
            ix = cp.minimum(cp.floor(self.x).astype(cp.int32), nx - 2)
            fy = (self.y - iy)[expand]
            fx = (self.x - ix)[expand]
            lower = ((one - fx) * field[lead + (iy, ix)]
                     + fx * field[lead + (iy, ix + 1)])
            upper = ((one - fx) * field[lead + (iy + 1, ix)]
                     + fx * field[lead + (iy + 1, ix + 1)])
            return ((one - fy) * lower + fy * upper).astype(
                cp.float32, copy=False)
        if method != "parabolic":
            raise ValueError("method must be 'nearest', 'bilinear', or 'parabolic'")

        iy = cp.floor(self.y).astype(cp.int32)
        ix = cp.floor(self.x).astype(cp.int32)
        fy = (self.y - iy)[expand]
        fx = (self.x - ix)[expand]
        xindices = [cp.clip(ix + offset, 0, nx - 1)
                    for offset in (-1, 0, 1, 2)]
        yindices = [cp.clip(iy + offset, 0, ny - 1)
                    for offset in (-1, 0, 1, 2)]
        rows = []
        for jy in yindices:
            tiny = cp.float32(1.0e-20)
            values = [cp.where(field[lead + (jy, jx)] == cp.float32(0.0),
                               tiny, field[lead + (jy, jx)])
                      for jx in xindices]
            rows.append(_wps_oned_gpu(fx, *values))
        result = _wps_oned_gpu(fy, *rows)
        return cp.where(result == tiny, cp.float32(0.0), result).astype(
            cp.float32, copy=False)


def interpolate_regular_gpu(field, latitude, longitude, target_lat, target_lon,
                            method="parabolic"):
    """Interpolate a 2-D or ``(...,ny,nx)`` field on the GPU in FP32."""
    return _RegularGpuPlan(latitude, longitude, target_lat, target_lon).apply(
        field, method=method)


def _interpolate_regular_bilinear_cpu(field, latitude, longitude,
                                      target_lat, target_lon):
    """Float64 regular-grid bilinear interpolation for setup-time fields."""
    y, x = _regular_coordinates(latitude, longitude, target_lat, target_lon)
    field = np.asarray(field, dtype=np.float64)
    source_shape = (len(latitude), len(longitude))
    if field.ndim < 2 or field.shape[-2:] != source_shape:
        raise ValueError("field trailing dimensions do not match source axes")
    lead = (slice(None),) * (field.ndim - 2)
    expand = (None,) * (field.ndim - 2)
    ny, nx = source_shape
    iy = np.minimum(np.floor(y).astype(np.int64), ny - 2)
    ix = np.minimum(np.floor(x).astype(np.int64), nx - 2)
    fy = (y - iy)[expand]
    fx = (x - ix)[expand]
    lower = ((1.0 - fx) * field[lead + (iy, ix)]
             + fx * field[lead + (iy, ix + 1)])
    upper = ((1.0 - fx) * field[lead + (iy + 1, ix)]
             + fx * field[lead + (iy + 1, ix + 1)])
    return np.ascontiguousarray((1.0 - fy) * lower + fy * upper)


def _canonical_psfc_bilinear(field, latitude, longitude,
                             target_lat, target_lon):
    """Return backend-independent FP32 pressure from one final rounding.

    WRF's 500-Pa ``zap_close_levels`` predicate is intentionally strict and
    discontinuous.  A one-ULP difference in mapped surface pressure can
    therefore select a different vertical wind stencil even when all normal
    numeric parity bounds are satisfied.  Source values and fractional index
    coordinates are first normalized to the production FP32 contract; the
    bilinear polynomial is then evaluated in float64 and rounded once to
    FP32.  Both preprocessing backends consume these exact shared bytes.

    This setup-only 2-D operator is deliberately scoped to PSFC.  On the
    bound ERA5 real-data corpus it is byte-identical to the established CUDA
    result at every target point, while avoiding backend-dependent scalar
    contraction in the Rust implementation.
    """

    y, x = _regular_coordinates(
        latitude, longitude, target_lat, target_lon)
    y = np.asarray(y, dtype=np.float32)
    x = np.asarray(x, dtype=np.float32)
    field = np.asarray(field, dtype=np.float32)
    source_shape = (len(latitude), len(longitude))
    if field.ndim != 2 or field.shape != source_shape:
        raise ValueError("canonical PSFC field must match the 2-D source axes")
    ny, nx = source_shape
    iy = np.minimum(np.floor(y).astype(np.int32), ny - 2)
    ix = np.minimum(np.floor(x).astype(np.int32), nx - 2)
    fy = np.asarray(y - iy.astype(np.float32), dtype=np.float32)
    fx = np.asarray(x - ix.astype(np.float32), dtype=np.float32)

    source = field.astype(np.float64)
    fy64 = fy.astype(np.float64)
    fx64 = fx.astype(np.float64)
    lower = ((1.0 - fx64) * source[iy, ix]
             + fx64 * source[iy, ix + 1])
    upper = ((1.0 - fx64) * source[iy + 1, ix]
             + fx64 * source[iy + 1, ix + 1])
    return np.ascontiguousarray(
        (1.0 - fy64) * lower + fy64 * upper, dtype=np.float32)


def source_orography_from_catalog(catalog, grid: ProjectedGrid, *,
                                  provider=ERA5_Z_INVARIANT_PROVIDER,
                                  valid_time=None) -> np.ndarray:
    """Resolve source terrain from ERA5 invariant geopotential in a catalog.

    ``era5_z_invariant`` consumes the catalog's decoded ``SOILGEO`` field,
    verifies that every catalog copy is truly invariant, bilinearly remaps it
    to the Lambert mass grid, and converts geopotential to metres with WPS's
    9.81 m s-2 convention.  It is CPU-only setup work and therefore does not
    require CuPy or initialize a device.
    """
    if provider != ERA5_Z_INVARIANT_PROVIDER:
        raise ValueError(
            f"unknown source-orography provider {provider!r}; recognized: "
            f"[{ERA5_Z_INVARIANT_PROVIDER!r}]")
    if not isinstance(grid, ProjectedGrid):
        raise TypeError("grid must be a ProjectedGrid")
    snapshots = tuple(getattr(catalog, "snapshots", ()))
    candidates = tuple(snapshot for snapshot in snapshots
                       if "SOILGEO" in snapshot.fields)
    if not candidates:
        raise ValueError(
            "source-orography provider 'era5_z_invariant' requires catalog "
            "inventory field SOILGEO (ERA5 invariant geopotential)")
    missing_times = tuple(snapshot.valid_time for snapshot in snapshots
                          if "SOILGEO" not in snapshot.fields)
    if missing_times:
        raise ValueError(
            "source-orography provider 'era5_z_invariant' requires SOILGEO "
            f"at every catalog valid time; missing at {missing_times}")
    if valid_time is None:
        selected = candidates[0]
    else:
        matches = tuple(snapshot for snapshot in candidates
                        if snapshot.valid_time == valid_time)
        if len(matches) != 1:
            raise ValueError(
                "catalog has no unique SOILGEO field at requested valid_time "
                f"{valid_time!s}")
        selected = matches[0]

    reference = np.asarray(selected.fields["SOILGEO"], dtype=np.float64)
    if reference.ndim != 2 or reference.shape != (
            selected.latitude.size, selected.longitude.size):
        raise ValueError(
            "catalog SOILGEO must be a 2-D field matching its latitude/longitude axes")
    if not np.isfinite(reference).all():
        raise ValueError("catalog SOILGEO contains non-finite geopotential")
    for snapshot in candidates:
        value = np.asarray(snapshot.fields["SOILGEO"], dtype=np.float64)
        same_grid = (np.array_equal(snapshot.latitude, selected.latitude)
                     and np.array_equal(snapshot.longitude,
                                        selected.longitude))
        if (not same_grid or value.shape != reference.shape
                or not np.array_equal(value, reference)):
            raise ValueError(
                "catalog field SOILGEO is declared invariant but changes at "
                f"valid_time {snapshot.valid_time!s}")

    units = getattr(catalog, "units", {}).get("SOILGEO")
    if units is None:
        raise ValueError(
            "catalog SOILGEO units metadata is required and must be "
            "geopotential (m2 s-2)")
    if str(units).replace(" ", "") not in {
            "m2s-2", "m^2s^-2", "m**2s**-2"}:
        raise ValueError(
            f"catalog SOILGEO units must be geopotential (m2 s-2), got {units!r}")
    target_lat, target_lon = grid.latlon_mass()
    geopotential = _interpolate_regular_bilinear_cpu(
        reference, selected.latitude, selected.longitude,
        target_lat, target_lon)
    height = geopotential / _WPS_GRAVITY_MS2
    if not np.isfinite(height).all():
        raise ValueError("ERA5-Z-derived source orography contains non-finite values")
    return height


def masked_nearest_gpu(field, latitude, longitude, target_lat, target_lon,
                       source_landmask, target_landmask, *, surface="match",
                       fill_value=0.0, search_radius=8, strict=True):
    """Nearest finite 2-D value on a requested land/water surface, on GPU."""
    cp = _cupy()
    y_np, x_np = _regular_coordinates(latitude, longitude, target_lat, target_lon)
    field = cp.asarray(field, dtype=cp.float32)
    source_landmask = cp.asarray(source_landmask, dtype=cp.bool_)
    target_landmask = cp.asarray(target_landmask, dtype=cp.bool_)
    source_shape = (len(latitude), len(longitude))
    if field.ndim != 2 or field.shape != source_shape or source_landmask.shape != source_shape:
        raise ValueError("field/source_landmask shape does not match source axes")
    if target_landmask.shape != y_np.shape:
        raise ValueError("target_landmask shape does not match target coordinates")
    if surface == "match":
        active = cp.ones_like(target_landmask)
        desired_land = target_landmask
    elif surface == "land":
        active = target_landmask
        desired_land = cp.ones_like(target_landmask)
    elif surface == "water":
        active = ~target_landmask
        desired_land = cp.zeros_like(target_landmask)
    else:
        raise ValueError("surface must be 'match', 'land', or 'water'")

    y = cp.asarray(y_np, dtype=cp.float32)
    x = cp.asarray(x_np, dtype=cp.float32)
    center_y = cp.rint(y).astype(cp.int32)
    center_x = cp.rint(x).astype(cp.int32)
    best_distance = cp.full(y.shape, cp.inf, dtype=cp.float32)
    best_value = cp.full(y.shape, cp.float32(fill_value), dtype=cp.float32)
    ny, nx = source_shape
    for dj in range(-search_radius, search_radius + 1):
        jy = center_y + dj
        in_y = (jy >= 0) & (jy < ny)
        jy_safe = cp.clip(jy, 0, ny - 1)
        for di in range(-search_radius, search_radius + 1):
            ix = center_x + di
            inside = in_y & (ix >= 0) & (ix < nx)
            ix_safe = cp.clip(ix, 0, nx - 1)
            value = field[jy_safe, ix_safe]
            valid = (active & inside & cp.isfinite(value)
                     & (source_landmask[jy_safe, ix_safe] == desired_land))
            distance = (y - jy_safe) ** 2 + (x - ix_safe) ** 2
            take = valid & (distance < best_distance)
            best_distance = cp.where(take, distance, best_distance)
            best_value = cp.where(take, value, best_value)
    if strict and bool(cp.any(active & ~cp.isfinite(best_distance)).item()):
        raise ValueError("no matching source surface within search_radius")
    return best_value


def _wps_oned(x, a, b, c, d):
    """Vectorized float64 transcription of metgrid ``oned`` (interp_module.F).

    One-dimensional overlapping-parabolic interpolation with WPS's exact
    zero-value special cases: a zero ``b`` or ``c`` collapses the result to
    0 unless ``x`` is exactly 0 or 1, and a zero ``a`` or ``d`` selects the
    one-sided parabola or the linear form.
    """
    x = np.asarray(x, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)
    result = np.zeros(np.broadcast(x, a, b, c, d).shape, dtype=np.float64)
    result = np.where(x == 0.0, b, result)
    result = np.where(x == 1.0, c, result)
    parab_b = b + x * (0.5 * (c - a) + x * (0.5 * (c + a) - b))
    parab_c = c + (1.0 - x) * (0.5 * (b - d) + (1.0 - x) * (0.5 * (b + d) - c))
    linear = b * (1.0 - x) + c * x
    both = (1.0 - x) * parab_b + x * parab_c
    inner = np.where(
        (a == 0.0) & (d == 0.0), linear,
        np.where(a != 0.0,
                 np.where(d != 0.0, both, parab_b),
                 parab_c))
    return np.where(b * c != 0.0, inner, result)


_WPS_SEARCH_DEPTH = 1200
_WPS_FULL_CHAIN = (
    "sixteen_pt", "four_pt", "wt_average_4pt", "wt_average_16pt", "search")
_WPS_SNOW_CHAIN = ("four_pt", "average_4pt")
_WPS_SST_CHAIN = ("sixteen_pt", "four_pt")


def _wps_sixteen_pt(field, valid, yy, xx, todo):
    """Metgrid ``sixteen_pt`` on active cells; NaN marks fall-through.

    ``field``/``valid`` are (ny,nx) float64/bool source arrays; ``yy``/``xx``
    are zero-based fractional source coordinates of the target cells.  The
    16-point overlapping parabolic requires every stencil point unmasked
    (interp_module.F:1262-1272); edge stencils clamp indices exactly like
    the Fortran (``kk``/``ll`` clipping, :1227-1241).  WPS's REAL*4 quirk of
    substituting 1e-20 for exact zeros before ``oned`` and mapping an exact
    1e-20 result back to zero (:1255-1257,1299) is transcribed as-is.
    """
    ny, nx = field.shape
    out = np.full(yy.shape, np.nan, dtype=np.float64)
    if not np.any(todo):
        return out
    i = np.floor(xx + 1.0e-5).astype(np.int64)
    j = np.floor(yy + 1.0e-5).astype(np.int64)
    xf = xx - i
    yf = yy - j
    near = (np.abs(xf) <= 1.0e-4) & (np.abs(yf) <= 1.0e-4)
    # Coincident-point branch (interp_module.F:1301-1332): take the source
    # point when usable, otherwise fall through.
    sel = todo & near
    if np.any(sel):
        jj = np.clip(j[sel], 0, ny - 1)
        ii = np.clip(i[sel], 0, nx - 1)
        ok = valid[jj, ii]
        vals = np.where(ok, field[jj, ii], np.nan)
        out[sel] = vals
    sel = todo & ~near
    if np.any(sel):
        i_s = i[sel]
        j_s = j[sel]
        stl = np.empty((4, 4) + i_s.shape, dtype=np.float64)
        all_ok = np.ones(i_s.shape, dtype=bool)
        for k in range(4):          # x offset -1..2
            kk = np.clip(i_s + (k - 1), 0, nx - 1)
            for l in range(4):      # y offset -1..2
                ll = np.clip(j_s + (l - 1), 0, ny - 1)
                value = field[ll, kk]
                all_ok &= valid[ll, kk]
                stl[k, l] = np.where(value == 0.0, 1.0e-20, value)
        a = _wps_oned(xf[sel], stl[0, 0], stl[1, 0], stl[2, 0], stl[3, 0])
        b = _wps_oned(xf[sel], stl[0, 1], stl[1, 1], stl[2, 1], stl[3, 1])
        c = _wps_oned(xf[sel], stl[0, 2], stl[1, 2], stl[2, 2], stl[3, 2])
        d = _wps_oned(xf[sel], stl[0, 3], stl[1, 3], stl[2, 3], stl[3, 3])
        value = _wps_oned(yf[sel], a, b, c, d)
        value = np.where(value == 1.0e-20, 0.0, value)
        out[sel] = np.where(all_ok, value, np.nan)
    return out


def _wps_four_pt(field, valid, yy, xx, todo, *, average):
    """Metgrid ``four_pt`` bilinear or ``four_pt_average``; NaN falls through.

    ``four_pt`` requires all four corners usable (interp_module.F:1099-1148)
    and handles integer-coordinate degeneracy exactly (:1150-1169);
    ``average_4pt`` renormalizes over the usable corners (:691-732).
    """
    ny, nx = field.shape
    out = np.full(yy.shape, np.nan, dtype=np.float64)
    if not np.any(todo):
        return out
    fx = np.floor(xx).astype(np.int64)
    cx = np.ceil(xx).astype(np.int64)
    fy = np.floor(yy).astype(np.int64)
    cy = np.ceil(yy).astype(np.int64)
    fx = np.clip(fx, 0, nx - 1)
    cx = np.clip(cx, 0, nx - 1)
    fy = np.clip(fy, 0, ny - 1)
    cy = np.clip(cy, 0, ny - 1)
    v_ff = field[fy, fx]
    v_fc = field[cy, fx]
    v_cf = field[fy, cx]
    v_cc = field[cy, cx]
    ok_ff = valid[fy, fx]
    ok_fc = valid[cy, fx]
    ok_cf = valid[fy, cx]
    ok_cc = valid[cy, cx]
    if average:
        w_ff = np.where(ok_ff, 1.0, 0.0)
        w_fc = np.where(ok_fc, 1.0, 0.0)
        w_cf = np.where(ok_cf, 1.0, 0.0)
        w_cc = np.where(ok_cc, 1.0, 0.0)
        wsum = w_ff + w_fc + w_cf + w_cc
        with np.errstate(invalid="ignore", divide="ignore"):
            value = np.where(
                wsum > 0.0,
                (w_ff * v_ff + w_fc * v_fc + w_cf * v_cf + w_cc * v_cc)
                / np.where(wsum > 0.0, wsum, 1.0),
                np.nan)
        out[todo] = value[todo]
        return out
    all_ok = ok_ff & ok_fc & ok_cf & ok_cc
    x_int = fx == cx
    y_int = fy == cy
    lin_x = v_ff * (cx - xx) + v_cf * (xx - fx)
    lin_y = v_ff * (cy - yy) + v_fc * (yy - fy)
    bilinear = ((yy - fy) * (v_fc * (cx - xx) + v_cc * (xx - fx))
                + (cy - yy) * (v_ff * (cx - xx) + v_cf * (xx - fx)))
    value = np.where(
        x_int, np.where(y_int, v_ff, lin_y),
        np.where(y_int, lin_x, bilinear))
    out[todo] = np.where(all_ok, value, np.nan)[todo]
    return out


def _wps_wt_average(field, valid, yy, xx, todo, *, sixteen):
    """Metgrid ``wt_average_4pt``/``wt_average_16pt``; NaN falls through.

    Weights are ``max(0, 1-d)`` on the four corners (interp_module.F:776-779)
    or ``max(0, 2-d)`` on the 4x4 stencil (:1011-1024), zeroed on unusable
    points and renormalized; the 16-point form rejects stencils that would
    leave the array (:993-998) instead of clamping.
    """
    ny, nx = field.shape
    out = np.full(yy.shape, np.nan, dtype=np.float64)
    if not np.any(todo):
        return out
    if sixteen:
        fx = np.floor(xx).astype(np.int64)
        fy = np.floor(yy).astype(np.int64)
        inside = (fx >= 1) & (fx <= nx - 3) & (fy >= 1) & (fy <= ny - 3)
        num = np.zeros(yy.shape, dtype=np.float64)
        den = np.zeros(yy.shape, dtype=np.float64)
        fx_c = np.clip(fx, 1, max(nx - 3, 1))
        fy_c = np.clip(fy, 1, max(ny - 3, 1))
        for dx in (-1, 0, 1, 2):
            for dy in (-1, 0, 1, 2):
                ii = fx_c + dx
                jj = fy_c + dy
                w = np.maximum(
                    0.0, 2.0 - np.sqrt((xx - ii) ** 2 + (yy - jj) ** 2))
                w = np.where(valid[jj, ii], w, 0.0)
                num += w * field[jj, ii]
                den += w
        with np.errstate(invalid="ignore", divide="ignore"):
            value = np.where(den > 0.0, num / np.where(den > 0.0, den, 1.0),
                             np.nan)
        out[todo] = np.where(inside, value, np.nan)[todo]
        return out
    fx = np.clip(np.floor(xx).astype(np.int64), 0, nx - 1)
    cx = np.clip(np.ceil(xx).astype(np.int64), 0, nx - 1)
    fy = np.clip(np.floor(yy).astype(np.int64), 0, ny - 1)
    cy = np.clip(np.ceil(yy).astype(np.int64), 0, ny - 1)
    num = np.zeros(yy.shape, dtype=np.float64)
    den = np.zeros(yy.shape, dtype=np.float64)
    for ii, jj in ((fx, fy), (fx, cy), (cx, fy), (cx, cy)):
        w = np.maximum(0.0, 1.0 - np.sqrt((xx - ii) ** 2 + (yy - jj) ** 2))
        w = np.where(valid[jj, ii], w, 0.0)
        num += w * field[jj, ii]
        den += w
    with np.errstate(invalid="ignore", divide="ignore"):
        value = np.where(den > 0.0, num / np.where(den > 0.0, den, 1.0),
                         np.nan)
    out[todo] = value[todo]
    return out


def _wps_search_single(field, valid, yy, xx, visited, stamp):
    """One-target transcription of metgrid ``search_extrap``.

    Four-connected FIFO breadth-first search from ``NINT(xx), NINT(yy)``
    (interp_module.F:484-563): expansion stops once the first usable point
    is DEQUEUED (that iteration still enqueues its neighbours, exactly like
    the Fortran loop body), then only points remaining IN THE QUEUE compete
    on squared Euclidean distance with a strict ``<`` (:565-607), so the
    first-found point wins ties and never-enqueued points never win even if
    globally nearer.  Neighbour order is x-1, x+1, y-1, y+1, and the depth
    counter reproduces WRF's in-place ``qdata%depth`` mutation (:521-559),
    capped by ``interp_opts`` (default 1200, :267).  Distances are float64
    here versus WPS REAL; the substitution is bounded by the FP32 final
    cast and can only differ inside FP32 rounding of a distance tie.
    Returns NaN when no usable point is reachable.
    """
    ny, nx = field.shape
    # Fortran NINT for non-negative arguments.
    ix = int(np.floor(xx + 0.5))
    jy = int(np.floor(yy + 0.5))
    if ix < 0 or ix >= nx or jy < 0 or jy >= ny:
        return np.nan
    from collections import deque
    queue = deque()
    queue.append((ix, jy, 0))
    visited[jy, ix] = stamp
    found = None
    while queue and found is None:
        i, j, depth = queue.popleft()
        if valid[j, i]:
            found = (i, j)
        dd = depth
        for ni, nj in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
            if 0 <= ni < nx and 0 <= nj < ny and visited[nj, ni] != stamp:
                if dd < _WPS_SEARCH_DEPTH:
                    dd += 1
                    queue.append((ni, nj, dd))
                    visited[nj, ni] = stamp
    if found is None:
        return np.nan
    fi, fj = found
    best_d2 = (float(fi) - xx) ** 2 + (float(fj) - yy) ** 2
    best = field[fj, fi]
    while queue:
        i, j, _ = queue.popleft()
        if valid[j, i]:
            d2 = (float(i) - xx) ** 2 + (float(j) - yy) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best = field[j, i]
    return best


def _wps_search(field, valid, yy, xx, todo):
    """Metgrid ``search_extrap`` over the active cells; NaN falls through.

    Per-cell FIFO/queue-limited semantics (see :func:`_wps_search_single`);
    a global-nearest shortcut is NOT equivalent -- WPS only compares points
    already enqueued when the first usable point is dequeued.  The visited
    bitarray is generation-stamped so the scratch allocation is shared
    across target cells.
    """
    out = np.full(yy.shape, np.nan, dtype=np.float64)
    if not np.any(todo) or not np.any(valid):
        return out
    visited = np.zeros(field.shape, dtype=np.int64)
    stamp = 0
    indices = np.nonzero(todo)
    for flat, (jj, ii) in enumerate(zip(*indices)):
        stamp += 1
        out[jj, ii] = _wps_search_single(
            field, valid, float(yy[jj, ii]), float(xx[jj, ii]),
            visited, stamp)
    return out


def wps_masked_field_interpolate(field, latitude, longitude, target_lat,
                                 target_lon, *, source_valid, target_active,
                                 chain, fill_value):
    """WPS metgrid masked-field interpolation chain on the host in float64.

    Transcribes metgrid's ``interp_sequence`` fall-through semantics
    (interp_module.F:304-367): each operator either produces a value or
    defers to the next; targets never produced -- including every cell
    outside ``target_active``, exactly like metgrid's landmask-restricted
    processing -- receive ``fill_value`` (process_domain fill_missing).
    ``source_valid`` folds the field's interp_mask and missing-value
    exclusions into one usable-source predicate.

    Arithmetic is float64 where metgrid computes in REAL: a known
    non-bitwise substitution, bounded by the FP32 final cast -- it can
    only change a result where WPS's FP32 rounding sits exactly on a
    stencil-rejection or distance-comparison boundary.
    """
    field = np.asarray(field, dtype=np.float64)
    source_valid = np.asarray(source_valid, dtype=bool)
    if field.shape != source_valid.shape:
        raise ValueError("field and source_valid shapes differ")
    yy, xx = _regular_coordinates(latitude, longitude, target_lat, target_lon)
    target_active = np.asarray(target_active, dtype=bool)
    if target_active.shape != yy.shape:
        raise ValueError("target_active shape does not match target grid")
    # Missing source values must never enter a stencil product even at zero
    # weight (0*NaN pollutes); neutralize them outside the usable set.
    safe = np.where(source_valid & np.isfinite(field), field, 0.0)
    usable = source_valid & np.isfinite(field)
    result = np.full(yy.shape, np.float64(fill_value), dtype=np.float64)
    todo = target_active.copy()
    for op in chain:
        if not np.any(todo):
            break
        if op == "sixteen_pt":
            got = _wps_sixteen_pt(safe, usable, yy, xx, todo)
        elif op == "four_pt":
            got = _wps_four_pt(safe, usable, yy, xx, todo, average=False)
        elif op == "average_4pt":
            got = _wps_four_pt(safe, usable, yy, xx, todo, average=True)
        elif op == "wt_average_4pt":
            got = _wps_wt_average(safe, usable, yy, xx, todo, sixteen=False)
        elif op == "wt_average_16pt":
            got = _wps_wt_average(safe, usable, yy, xx, todo, sixteen=True)
        elif op == "search":
            got = _wps_search(safe, usable, yy, xx, todo)
        else:
            raise ValueError(f"unknown WPS interpolation operator {op!r}")
        produced = todo & np.isfinite(got)
        result[produced] = got[produced]
        todo &= ~produced
    return result


def rotate_earth_to_grid_gpu(u_earth, v_earth, sinalpha, cosalpha):
    """Rotate co-located earth-relative winds to grid-relative FP32 winds."""
    cp = _cupy()
    u = cp.asarray(u_earth, dtype=cp.float32)
    v = cp.asarray(v_earth, dtype=cp.float32)
    sina = cp.asarray(sinalpha, dtype=cp.float32)
    cosa = cp.asarray(cosalpha, dtype=cp.float32)
    if u.shape != v.shape:
        raise ValueError("u_earth and v_earth shapes differ")
    return u * cosa + v * sina, v * cosa - u * sina


def rotate_grid_to_earth_gpu(u_grid, v_grid, sinalpha, cosalpha):
    """Inverse of :func:`rotate_earth_to_grid_gpu`."""
    cp = _cupy()
    u = cp.asarray(u_grid, dtype=cp.float32)
    v = cp.asarray(v_grid, dtype=cp.float32)
    sina = cp.asarray(sinalpha, dtype=cp.float32)
    cosa = cp.asarray(cosalpha, dtype=cp.float32)
    if u.shape != v.shape:
        raise ValueError("u_grid and v_grid shapes differ")
    return u * cosa - v * sina, v * cosa + u * sina


def lambert_rotation(grid: ProjectedGrid, stagger="mass"):
    """Float64 ``(SINALPHA, COSALPHA)`` on one staggering.

    Delegates to the grid's own get_rotang transcription
    (:meth:`gpuwm.static.projection.ProjectedGrid.rotation_m`/``_u``/
    ``_v``), so every projection carries its own convention: Lambert
    ``cone * wrap(stand_lon - lon)``, polar stereographic
    ``wrap(stand_lon - lon)``, Mercator identically zero.  (Name kept
    for compatibility; it predates the worldwide projections.)
    """
    if stagger == "mass":
        return grid.rotation_m()
    if stagger == "u":
        return grid.rotation_u()
    if stagger == "v":
        return grid.rotation_v()
    raise ValueError("stagger must be 'mass', 'u', or 'v'")


def _era5_rh_to_water_gpu(relative_humidity, temperature):
    """WPS v4.6 ``ungrib/src/rrpr.F:fix_gfs_rh`` mixed-phase RH to liquid RH.

    ECMWF/ERA5 RH is saturation-blended over ice below freezing; real.exe
    assumes RH with respect to liquid.  Below 273.15 K WPS multiplies RH by
    ``r/ews`` where ``ews`` is Bolton 1980 liquid saturation vapor pressure
    (hPa), ``eis`` is Murphy and Koop 2005 ice saturation vapor pressure
    (hPa), and ``r`` blends linearly from liquid at 273.15 K to pure ice at
    and below 253.15 K (rrpr.F:1326-1363).  Float64 setup math, FP32 result.
    """
    cp = _cupy()
    rh = cp.asarray(relative_humidity, dtype=cp.float64)
    t = cp.asarray(temperature, dtype=cp.float64)
    if rh.shape != t.shape:
        raise ValueError("relative_humidity and temperature shapes differ")
    eis = 0.01 * cp.exp(9.550426 - 5723.265 / t + 3.53068 * cp.log(t)
                        - 0.00728332 * t)
    ews = 6.112 * cp.exp(17.67 * (t - 273.15) / ((t - 273.15) + 243.5))
    frac = (273.15 - t) / 20.0
    blended = frac * eis + (1.0 - frac) * ews
    r = cp.where(t > 253.15, blended, eis)
    converted = cp.where(t <= 273.15, rh * (r / ews), rh)
    return converted.astype(cp.float32)


_RENAMES = {
    "Z": "GHT", "T": "TT", "U": "UU", "V": "VV",
    "SEAICE": "XICE", "SOILGEO": "SOURCE_OROGRAPHY",
}
_PARABOLIC_SCALARS = {"Z", "T", "RH", "T2", "D2", "RH2", "PMSL"}
_MATCH_SURFACE_FIELDS = {"SKINTEMP"}
_WATER_FIELDS = {"SST", "SEAICE", "XICE"}
_LAND_FIELDS = {
    "ST000007", "ST007028", "ST028100", "ST100289",
    "SM000007", "SM007028", "SM028100", "SM100289", "SNOW_EC",
    "GFS_ST000010", "GFS_ST010040", "GFS_ST040100", "GFS_ST100200",
    "GFS_SM000010", "GFS_SM010040", "GFS_SM040100", "GFS_SM100200",
    MAPPED_SOIL_TEMPERATURE, MAPPED_SOIL_MOISTURE,
    "SNOW", "SNOWH",
}
_MASKED_SEARCH_RADIUS = 8
_SNOW_FAMILY = {"SNOW", "SNOWH", "SNOW_EC"}


def _as_host_bool(value):
    """Boolean host copy of a possibly-device array."""
    if hasattr(value, "get"):
        value = value.get()
    return np.asarray(value, dtype=bool)


def _as_host_float64(value):
    """Float64 host copy of a possibly-device array."""
    if hasattr(value, "get"):
        value = value.get()
    return np.asarray(value, dtype=np.float64)


def _wps_soil_fill(name: str) -> float:
    """METGRID.TBL fill_missing for masked soil families (SM 1.0, ST 285)."""
    if name == MAPPED_SOIL_MOISTURE or "SM" in name[:6] or name == "SOILW":
        return 1.0
    if name == MAPPED_SOIL_TEMPERATURE or "ST" in name[:6] or name == "SOILT":
        return 285.0
    raise ValueError(f"no METGRID.TBL fill is registered for {name!r}")


def interpolate_era5_to_lambert(snapshot: Era5Snapshot, grid: ProjectedGrid, *,
                                target_landmask=None,
                                source_orography_catalog=None,
                                relative_humidity_convention="era5_mixed",
                                backend="cuda", workers=None,
                                cpu_bridge=None,
                                ) -> HorizontalSnapshot:
    """Interpolate every field in ``snapshot`` to mass/U/V Lambert points.

    Atmospheric fields use WPS's 16-point overlapping-parabolic default;
    PSFC and unclassified continuous fields use bilinear interpolation.
    LANDSEA is nearest-neighbor. Masked surface/soil fields use the nearest
    finite source value on the matching surface, with zero fill where a field
    is not defined (SST on land, soil/snow on water). ``backend='cuda'``
    preserves the established device path; ``backend='cpu'`` selects the
    packaged deterministic parallel Rust path and returns host FP32 fields.
    """
    from gpuwm.ingest.preprocess_backend import resolve_preprocess_backend

    engine = resolve_preprocess_backend(
        backend, workers=workers, cpu_bridge=cpu_bridge)
    xp = engine.array_module
    if not isinstance(snapshot, Era5Snapshot):
        raise TypeError("snapshot must be an Era5Snapshot")
    if not isinstance(grid, ProjectedGrid):
        raise TypeError("grid must be a ProjectedGrid")
    if relative_humidity_convention not in {"era5_mixed", "water"}:
        raise ValueError(
            "relative_humidity_convention must be 'era5_mixed' or 'water'")

    mass_lat, mass_lon = grid.latlon_mass()
    u_lat, u_lon = grid.latlon_u()
    v_lat, v_lon = grid.latlon_v()
    mass_plan = engine.regular_plan(
        snapshot.latitude, snapshot.longitude, mass_lat, mass_lon)
    u_plan = engine.regular_plan(
        snapshot.latitude, snapshot.longitude, u_lat, u_lon)
    v_plan = engine.regular_plan(
        snapshot.latitude, snapshot.longitude, v_lat, v_lon)

    source_fields = snapshot.fields
    masked_names = _MATCH_SURFACE_FIELDS | _WATER_FIELDS | _LAND_FIELDS
    if masked_names.intersection(source_fields) and "LANDSEA" not in source_fields:
        raise ValueError("LANDSEA is required to interpolate masked fields")

    source_land = None
    if "LANDSEA" in source_fields:
        source_land = engine.float32(source_fields["LANDSEA"]) >= 0.5
    if target_landmask is None:
        if source_land is None:
            target_land = None
        else:
            target_land = mass_plan.apply(source_land, method="nearest") >= 0.5
    else:
        target_land = engine.bool_array(target_landmask)
        if target_land.shape != mass_lat.shape:
            raise ValueError("target_landmask shape does not match mass grid")

    out: dict[str, object] = {}

    def wind_pair(u_name, v_name, u_output, v_output):
        if u_name not in source_fields or v_name not in source_fields:
            missing = u_name if u_name not in source_fields else v_name
            raise ValueError(f"{missing} is required for vector wind interpolation")
        u_source = engine.float32(source_fields[u_name])
        v_source = engine.float32(source_fields[v_name])
        ue_u = u_plan.apply(u_source, method="parabolic")
        ve_u = u_plan.apply(v_source, method="parabolic")
        ue_v = v_plan.apply(u_source, method="parabolic")
        ve_v = v_plan.apply(v_source, method="parabolic")
        sina_u, cosa_u = lambert_rotation(grid, "u")
        sina_v, cosa_v = lambert_rotation(grid, "v")
        out[u_output] = engine.rotate_earth_to_grid(
            ue_u, ve_u, sina_u, cosa_u)[0]
        out[v_output] = engine.rotate_earth_to_grid(
            ue_v, ve_v, sina_v, cosa_v)[1]

    handled: set[str] = set()
    for name, raw in source_fields.items():
        if name in handled:
            continue
        if name == "U":
            wind_pair("U", "V", "UU", "VV")
            handled.update(("U", "V"))
            continue
        if name == "V":
            wind_pair("U", "V", "UU", "VV")
            handled.update(("U", "V"))
            continue
        if name == "U10":
            wind_pair("U10", "V10", "U10", "V10")
            handled.update(("U10", "V10"))
            continue
        if name == "V10":
            wind_pair("U10", "V10", "U10", "V10")
            handled.update(("U10", "V10"))
            continue
        if name == "LANDSEA":
            out[name] = target_land.astype(xp.float32)
            handled.add(name)
            continue
        if name == "SOILGEO":
            if source_orography_catalog is None:
                raise ValueError(
                    "SOILGEO interpolation requires the validated "
                    "era5_z_invariant source_orography_catalog")
            out[_RENAMES[name]] = source_orography_from_catalog(
                source_orography_catalog, grid,
                valid_time=snapshot.valid_time)
            handled.add(name)
            continue

        if name == "PSFC":
            # Share exact setup bytes across CPU and CUDA before WRF-real's
            # discontinuous 500-Pa vertical-stencil decision.
            out[name] = engine.float32(_canonical_psfc_bilinear(
                raw, snapshot.latitude, snapshot.longitude,
                mass_lat, mass_lon))
            handled.add(name)
            continue

        field = engine.float32(raw)
        output_name = _RENAMES.get(name, name)
        if name in masked_names:
            if name == "SST":
                # METGRID.TBL SST: sixteen_pt+four_pt, unmasked, bitmap
                # missing excluded, fill_missing=0 -- coastal stencils that
                # cross missing land collapse to zero exactly like the WPS
                # output fields.
                source_valid_host = np.ones(
                    (len(snapshot.latitude), len(snapshot.longitude)),
                    dtype=bool)
                target_active_host = np.ones(mass_lat.shape, dtype=bool)
                chain = _WPS_SST_CHAIN
                fill = 0.0
            else:
                if target_land is None or source_land is None:
                    raise ValueError(
                        "LANDSEA is required to interpolate masked fields")
                source_land_host = _as_host_bool(source_land)
                target_land_host = _as_host_bool(target_land)
                if name in _MATCH_SURFACE_FIELDS:
                    chain = _WPS_FULL_CHAIN
                    fill = 0.0
                    source_valid_host = None  # two-pass, handled below
                    target_active_host = None
                elif name in _WATER_FIELDS:
                    # SEAICE/XICE: masked=land, four_pt+average_4pt, fill 0.
                    source_valid_host = ~source_land_host
                    target_active_host = ~target_land_host
                    chain = _WPS_SNOW_CHAIN
                    fill = 0.0
                else:
                    source_valid_host = source_land_host
                    target_active_host = target_land_host
                    if name in _SNOW_FAMILY:
                        chain = _WPS_SNOW_CHAIN
                        fill = 0.0
                    else:
                        chain = _WPS_FULL_CHAIN
                        fill = _wps_soil_fill(name)

            def interpolate_masked_slab(slab):
                slab_host = _as_host_float64(slab)
                if name in _MATCH_SURFACE_FIELDS:
                    # METGRID.TBL masked=both: land targets from land-only
                    # sources, water targets from water-only sources, each
                    # through the full chain.
                    land_part = wps_masked_field_interpolate(
                        slab_host, snapshot.latitude, snapshot.longitude,
                        mass_lat, mass_lon,
                        source_valid=source_land_host,
                        target_active=target_land_host,
                        chain=_WPS_FULL_CHAIN, fill_value=fill)
                    water_part = wps_masked_field_interpolate(
                        slab_host, snapshot.latitude, snapshot.longitude,
                        mass_lat, mass_lon,
                        source_valid=~source_land_host,
                        target_active=~target_land_host,
                        chain=_WPS_FULL_CHAIN, fill_value=fill)
                    combined = np.where(
                        target_land_host, land_part, water_part)
                else:
                    combined = wps_masked_field_interpolate(
                        slab_host, snapshot.latitude, snapshot.longitude,
                        mass_lat, mass_lon,
                        source_valid=source_valid_host,
                        target_active=target_active_host,
                        chain=chain, fill_value=fill)
                return engine.float32(combined.astype(np.float32))

            try:
                if field.ndim == 2:
                    interpolated = interpolate_masked_slab(field)
                elif field.ndim == 3:
                    source_shape = (
                        len(snapshot.latitude), len(snapshot.longitude)
                    )
                    if field.shape[1:] != source_shape or field.shape[0] < 1:
                        raise ValueError(
                            "layered field shape does not match source axes"
                        )
                    interpolated = xp.stack(
                        tuple(
                            interpolate_masked_slab(field[layer])
                            for layer in range(field.shape[0])
                        ),
                        axis=0,
                    )
                else:
                    raise ValueError(
                        "masked field must be two-dimensional or a layered "
                        "three-dimensional array"
                    )
            except ValueError as error:
                raise ValueError(
                    f"masked interpolation failed for {name}: {error}") from error
            out[output_name] = interpolated
        else:
            if name == "RH":
                if relative_humidity_convention == "era5_mixed":
                    if "T" not in source_fields:
                        raise ValueError(
                            "T is required for ERA5 RH convention conversion")
                    field = engine.era5_rh_to_water(
                        field, source_fields["T"])
            method = "parabolic" if name in _PARABOLIC_SCALARS or field.ndim == 3 else "bilinear"
            out[output_name] = mass_plan.apply(field, method=method)
            if name == "Z":
                out[output_name] = out[output_name] / xp.float32(9.81)
        handled.add(name)

    return HorizontalSnapshot(
        valid_time=snapshot.valid_time,
        levels_hpa=snapshot.levels_hpa,
        fields=out,
    )


__all__ = [
    "HorizontalSnapshot",
    "interpolate_era5_to_lambert",
    "interpolate_lake_skin_temperature",
    "interpolate_regular_gpu",
    "lambert_rotation",
    "masked_nearest_gpu",
    "rotate_earth_to_grid_gpu",
    "rotate_grid_to_earth_gpu",
    "source_orography_from_catalog",
    "wps_masked_field_interpolate",
    "ERA5_Z_INVARIANT_PROVIDER",
]
