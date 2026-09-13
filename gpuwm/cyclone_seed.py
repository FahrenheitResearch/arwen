"""One canonical-field cyclone seeding path, independent of source names.

Arrays use canonical units: pressure in Pa, temperature in K, earth-relative
winds in m/s, latitude and longitude in degrees. A seed is a candidate center,
not a tropical-cyclone classification. A point is authoritative; an advisory
bounds a field search and is the final fallback when diagnostics are absent.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Mapping

import numpy as np

EARTH_RADIUS_M = 6371229.0


@dataclass(frozen=True)
class SeedInventory:
    fields: tuple[str, ...]
    pressure_levels_pa: tuple[float, ...] = ()


def source_inventory(source: str) -> SeedInventory:
    """Read only the source's own mapping, never a composed surface donor."""
    from gpuwm.source_adapters import get_source_adapter
    from gpuwm.source_authorities import packaged_authorities

    adapter = get_source_adapter(source)
    fields = set(adapter.seed_fields)
    levels = ()
    if adapter.packaged_profile is not None:
        mapping = json.loads(packaged_authorities(adapter.packaged_profile)["mapping"].read_text())
        fields.update(mapping.get("fields", {}))
        vertical = mapping.get("coordinates", {}).get("vertical", {})
        if vertical.get("kind") == "pressure" and vertical.get("units") == "Pa":
            levels = tuple(float(p) for p in vertical.get("levels", ()))
    return SeedInventory(tuple(sorted(fields)), levels)


@dataclass(frozen=True)
class SeedResult:
    point: tuple[float, float] | None
    method: str | None
    source: str
    messages: tuple[str, ...] = ()
    diagnostics: Mapping[str, float] | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _coordinates(fields: Mapping[str, object]) -> tuple[np.ndarray, np.ndarray]:
    lat = np.asarray(fields["latitude"], dtype=float)
    lon = np.asarray(fields["longitude"], dtype=float)
    if lat.ndim == lon.ndim == 1:
        lon, lat = np.meshgrid(lon, lat)
    if lat.ndim != 2 or lat.shape != lon.shape or not lat.size:
        raise ValueError("Seed latitude and longitude must describe the same nonempty two-dimensional grid")
    return lat, lon


def _distance_km(lat, lon, point):
    phi, phi0 = np.deg2rad(lat), np.deg2rad(point[0])
    dl = np.deg2rad((lon - point[1] + 180.) % 360. - 180.)
    h = np.sin((phi - phi0) / 2.) ** 2 + np.cos(phi) * np.cos(phi0) * np.sin(dl / 2.) ** 2
    return 2 * EARTH_RADIUS_M / 1000 * np.arcsin(np.sqrt(np.clip(h, 0., 1.)))


def _pressure_plane(fields, inventory, name, pressure_pa, shape):
    """Log-pressure interpolation of one field, with no vertical extrapolation."""
    if name not in inventory.fields or name not in fields:
        return None
    values = np.asarray(fields[name], dtype=float)
    if values.ndim != 3 or values.shape[1:] != shape:
        return None
    if "pressure_levels_pa" in fields:
        levels = np.asarray(fields["pressure_levels_pa"], dtype=float)
        if levels.shape != (values.shape[0],):
            return None
        if inventory.pressure_levels_pa and not np.all(np.isin(levels, inventory.pressure_levels_pa)):
            return None
        pressure = np.broadcast_to(levels[:, None, None], values.shape)
    elif "air_pressure" in inventory.fields and "air_pressure" in fields:
        pressure = np.asarray(fields["air_pressure"], dtype=float)
        if pressure.shape != values.shape:
            return None
    else:
        return None
    good = np.isfinite(pressure) & (pressure > 0.) & np.isfinite(values)
    order = np.argsort(np.where(good, pressure, np.inf), axis=0, kind="stable")
    pressure = np.take_along_axis(np.where(good, pressure, np.inf), order, axis=0)
    values = np.take_along_axis(values, order, axis=0)
    count = good.sum(axis=0)
    lower = (pressure <= pressure_pa).sum(axis=0) - 1
    i0 = np.clip(lower, 0, values.shape[0] - 1)[None, ...]
    i1 = np.clip(lower + 1, 0, values.shape[0] - 1)[None, ...]
    p0, p1 = np.take_along_axis(pressure, i0, 0)[0], np.take_along_axis(pressure, i1, 0)[0]
    v0, v1 = np.take_along_axis(values, i0, 0)[0], np.take_along_axis(values, i1, 0)[0]
    exact = (p0 == pressure_pa) & (lower >= 0) & (count > 0)
    bracket = ((lower >= 0) & (lower + 1 < count) & (p1 > p0)
               & (p0 < pressure_pa) & (pressure_pa < p1))
    out = np.full(shape, np.nan)
    out[exact] = v0[exact]
    if bracket.any():
        weight = np.log(pressure_pa / p0[bracket]) / np.log(p1[bracket] / p0[bracket])
        out[bracket] = v0[bracket] + weight * (v1[bracket] - v0[bracket])
    return out


def relative_vorticity(u, v, lat, lon):
    """Earth-relative curl on a curvilinear grid, including spherical metrics."""
    if min(lat.shape) < 3:
        return np.full(lat.shape, np.nan)
    phi, lam = np.deg2rad(lat), np.deg2rad(lon)
    # Unwrap both axes so a dateline crossing is a short grid edge.
    lam = np.unwrap(np.unwrap(lam, axis=1), axis=0)
    pj, pi = np.gradient(phi)
    lj, li = np.gradient(lam)
    xj, xi = EARTH_RADIUS_M * np.cos(phi) * lj, EARTH_RADIUS_M * np.cos(phi) * li
    yj, yi = EARTH_RADIUS_M * pj, EARTH_RADIUS_M * pi
    uj, ui = np.gradient(u)
    vj, vi = np.gradient(v)
    determinant = xi * yj - xj * yi
    with np.errstate(invalid="ignore", divide="ignore"):
        curl = ((vi * yj - vj * yi) - (uj * xi - ui * xj)) / determinant
        curl += u * np.tan(phi) / EARTH_RADIUS_M
    usable = np.isfinite(curl) & (np.abs(determinant) > 0.) & (np.abs(lat) < 89.)
    return np.where(usable, curl, np.nan)


def seed_cyclone(*, source: str, fields: Mapping[str, object] | None = None,
                 point: tuple[float, float] | None = None,
                 advisory: tuple[float, float] | None = None,
                 search_radius_km: float = 500.) -> SeedResult:
    from gpuwm.cyclone_sources import source_adapter, validate_center
    from gpuwm.source_coverage import points_outside

    adapter = source_adapter(source)
    source = adapter.source_id
    if point is not None:
        validate_center(source, point)
        return SeedResult(tuple(point), "point", source)
    if advisory is not None:
        validate_center(source, advisory)
    if not np.isfinite(search_radius_km) or search_radius_km <= 0.:
        raise ValueError("Seed search radius must be finite and positive")
    fields = {} if fields is None else fields
    messages = []
    if "latitude" not in fields or "longitude" not in fields:
        messages.append("No source-grid coordinates were supplied; field seeding was skipped")
    else:
        inventory = source_inventory(source)
        lat, lon = _coordinates(fields)
        mask = (np.isfinite(lat) & np.isfinite(lon) & (np.abs(lat) <= 85.)
                & ~points_outside(adapter.coverage_window, lat, lon))
        distance = _distance_km(lat, lon, advisory) if advisory is not None else None
        if distance is not None:
            mask &= distance <= search_radius_km
        candidates = []
        name = "mean_sea_level_pressure"
        pressure = (np.asarray(fields[name], dtype=float)
                    if name in inventory.fields and name in fields else None)
        if pressure is not None and pressure.shape == lat.shape:
            candidates.append(("mslp", -np.where(pressure > 0., pressure, np.nan)))
        else:
            messages.append("MSLP is not declared or supplied on this grid")
        u = _pressure_plane(fields, inventory, "eastward_wind", 85000., lat.shape)
        v = _pressure_plane(fields, inventory, "northward_wind", 85000., lat.shape)
        if u is not None and v is not None:
            cyclonic = relative_vorticity(u, v, lat, lon) * np.where(lat < 0., -1., 1.)
            candidates.append(("low_level_vorticity", np.where(cyclonic > 0., cyclonic, np.nan)))
        else:
            messages.append("Winds do not bracket 850 hPa; low-level vorticity is unavailable")
        warm = []
        for level in (30000., 50000.):
            temp = _pressure_plane(fields, inventory, "air_temperature", level, lat.shape)
            if temp is not None and np.any(mask & np.isfinite(temp)):
                warm.append(temp - np.median(temp[mask & np.isfinite(temp)]))
        if len(warm) == 2:
            anomaly = (warm[0] + warm[1]) / 2.
            candidates.append(("warm_core", np.where(anomaly > 0., anomaly, np.nan)))
        else:
            messages.append("Temperature does not bracket 300 and 500 hPa; warm-core seeding was skipped")
        for method, score in candidates:
            valid = mask & np.isfinite(score)
            if not valid.any() or np.ptp(score[valid]) == 0.:
                messages.append(f"{method} has no resolved extremum in the search area; trying the next method")
                continue
            best = valid & (score == np.max(score[valid]))
            index = (int(np.argmin(np.where(best, distance, np.inf))) if distance is not None
                     else int(np.flatnonzero(best)[0]))
            j, i = np.unravel_index(index, lat.shape)
            center = (float(lat[j, i]), float((lon[j, i] + 180.) % 360. - 180.))
            diagnostics = {"score": float(score[j, i])}
            if method == "mslp":
                diagnostics = {"mslp_pa": float(pressure[j, i])}
            return SeedResult(center, method, source, tuple(messages), diagnostics)
    if advisory is not None:
        messages.append("Using the supplied advisory position")
        return SeedResult(tuple(advisory), "advisory", source, tuple(messages))
    messages.append("No center was found; supply --point or --advisory-position")
    return SeedResult(None, None, source, tuple(messages))


def load_seed_fields(path: Path, *, source: str, cycle: str,
                     member: str | None = None) -> dict[str, np.ndarray]:
    """Load a caller-supplied canonical NPZ without object deserialization.

    The three identity scalars bind the fields to the selected source analysis,
    not just to a filename. Arrays must already be decoded into canonical units.
    """
    from gpuwm.cyclone_sources import source_adapter, resolve_cycle, selected_member

    expected_source = source_adapter(source).source_id
    expected_cycle = resolve_cycle(cycle, source=expected_source).strftime("%Y%m%d%H")
    expected_member = selected_member(expected_source, member) or ""
    with np.load(Path(path), allow_pickle=False) as archive:
        fields = {key: archive[key] for key in archive.files}
    for key, expected in (("source", expected_source), ("cycle", expected_cycle),
                          ("member", expected_member)):
        raw = np.asarray(fields.pop(key, None))
        if raw.shape != () or str(raw.item()) != expected:
            raise ValueError(f"Seed field {key} does not match {expected!r}; supply this source analysis's fields")
    return fields
