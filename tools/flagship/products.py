#!/usr/bin/env python3
"""Generate fail-loud flagship products from WRF output with wrf-rust."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import hashlib
import importlib.metadata as importlib_metadata
import json
import math
from pathlib import Path
import re
from typing import Iterable, Mapping, Sequence, TypeVar

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import netCDF4
import numpy as np
from wrf import WrfFile, getvar, interplevel


PINNED_WRF_RUST_VERSION = "0.2.35"
_DOMAIN_RE = re.compile(r"^wrfout_(d\d{2})(?:_|$)", re.IGNORECASE)
_FILENAME_TIME_RE = re.compile(
    r"wrfout_d\d{2}_(\d{4}-\d{2}-\d{2})[_T](\d{2})[:_](\d{2})[:_](\d{2})",
    re.IGNORECASE,
)
_TIME_FORMATS = (
    "%Y-%m-%d_%H:%M:%S",
    "%Y-%m-%d_%H_%M_%S",
    "%Y-%m-%dT%H:%M:%S",
)
_PLAN_CADENCE_SECONDS = {"d01": 3600, "d02": 900, "d03": 900, "d04": 900}
_OUTER_DOMAINS = frozenset({"d01", "d02"})
_INNER_DOMAINS = frozenset({"d03", "d04"})
_HYDROMETEORS = frozenset({"QRAIN", "QSNOW", "QGRAUP", "QICE", "QCLOUD"})
_DBZ_WARNING_THRESHOLD = 20.0
_COMPLETE_PRECIP_ATTRIBUTE = "FLAGSHIP_TOTAL_PRECIP_VARIABLE"


def _assert_wrf_rust_version() -> str:
    try:
        installed = importlib_metadata.version("wrf-rust")
    except importlib_metadata.PackageNotFoundError as exc:
        raise RuntimeError("required science core wrf-rust is not installed") from exc
    if installed != PINNED_WRF_RUST_VERSION:
        raise RuntimeError(
            f"wrf-rust version mismatch: required {PINNED_WRF_RUST_VERSION}, "
            f"found {installed}")
    return installed


# Importing this module is itself a startup check.  generate_products repeats
# the check so a changed environment cannot evade it in a long-lived process.
WRF_RUST_VERSION = _assert_wrf_rust_version()


@dataclass(frozen=True)
class Frame:
    """One valid-time record within a WRF output file."""

    domain: str
    path: Path
    time_index: int
    valid_time: str

    @property
    def slug(self) -> str:
        return re.sub(r"[^0-9A-Za-z]+", "", self.valid_time)


@dataclass(frozen=True)
class CoordinateGrid:
    """Validated native geolocation consumed by every rendered frame."""

    lon: np.ndarray
    lat: np.ndarray
    source: str


@dataclass(frozen=True)
class FrameScience:
    """Fully validated native and wrf-rust fields for one frame."""

    frame: Frame
    lon: np.ndarray
    lat: np.ndarray
    coordinate_source: str
    refl_low: np.ndarray
    refl_max: np.ndarray
    slp: np.ndarray
    uvmet10: np.ndarray
    wspd10: np.ndarray
    precip: np.ndarray
    precip_source: str
    dbz_max: np.ndarray | None
    dbz_disagreement_max: float | None
    dbz_warning: str | None
    height500: np.ndarray | None
    temp500: np.ndarray | None
    u500: np.ndarray | None
    v500: np.ndarray | None
    uhel: np.ndarray | None
    uhel_max: np.ndarray | None
    variables: tuple[str, ...]
    metadata: Mapping[str, object]


def _parse_time(value: str) -> datetime | None:
    value = value.strip().replace("\x00", "")
    for fmt in _TIME_FORMATS:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            pass
    return None


def _normal_time(value: str) -> str:
    parsed = _parse_time(value)
    if parsed is None:
        raise ValueError(f"invalid WRF Times entry {value!r}")
    return parsed.strftime("%Y-%m-%d_%H:%M:%S")


def _filename_time(path: Path) -> str | None:
    match = _FILENAME_TIME_RE.search(path.name)
    if match is None:
        return None
    return f"{match.group(1)}_{match.group(2)}:{match.group(3)}:{match.group(4)}"


def _decode_times(variable) -> list[str]:
    raw = np.ma.filled(variable[:], b"\x00")
    if raw.ndim == 1:
        raw = raw[None, :]
    values: list[str] = []
    for row in raw:
        flat = np.asarray(row).reshape(-1)
        if flat.dtype.kind == "S":
            text = b"".join(bytes(item) for item in flat).decode("ascii", errors="strict")
        else:
            text = "".join(str(item) for item in flat)
        values.append(_normal_time(text.rstrip("\x00 ")))
    return values


def discover_frames(run_dir: str | Path,
                    domains: Sequence[str] | None = None) -> dict[str, list[Frame]]:
    """Discover frames and enforce requested-domain and cadence inventories."""
    root = Path(run_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"run directory does not exist: {root}")
    if domains:
        normalized = [str(item).lower() for item in domains]
        if (any(re.fullmatch(r"d\d{2}", item) is None for item in normalized)
                or len(set(normalized)) != len(normalized)):
            raise ValueError("domains must be unique WRF IDs such as d01")
        wanted = set(normalized)
    else:
        wanted = None

    grouped: dict[str, list[Frame]] = {}
    candidates = sorted(
        path for path in root.rglob("wrfout_*")
        if path.is_file() and ".tmp" not in path.name
    )
    for path in candidates:
        match = _DOMAIN_RE.match(path.name)
        if match is None:
            continue
        domain = match.group(1).lower()
        if wanted is not None and domain not in wanted:
            continue
        try:
            with netCDF4.Dataset(path, "r") as ds:
                ntime = len(ds.dimensions["Time"]) if "Time" in ds.dimensions else 1
                if ntime <= 0:
                    raise ValueError(f"empty WRF frame (Time has length zero): {path}")
                if "Times" in ds.variables:
                    try:
                        times = _decode_times(ds.variables["Times"])
                    except (UnicodeError, ValueError) as exc:
                        raise ValueError(f"invalid Times inventory in {path}: {exc}") from exc
                    if len(times) != ntime:
                        raise ValueError(f"invalid Times inventory in {path}")
                else:
                    stamped = _filename_time(path)
                    if ntime != 1 or stamped is None:
                        raise ValueError(
                            f"{path} needs a usable Times variable for {ntime} records")
                    times = [_normal_time(stamped)]
        except (OSError, RuntimeError) as exc:
            raise ValueError(f"cannot read WRF frame {path}: {exc}") from exc
        grouped.setdefault(domain, []).extend(
            Frame(domain, path, index, value)
            for index, value in enumerate(times)
        )

    if not grouped:
        selected = "all domains" if wanted is None else ", ".join(sorted(wanted))
        raise ValueError(f"no WRF output frames found for {selected} under {root}")
    if wanted is not None:
        missing = sorted(wanted - set(grouped))
        if missing:
            raise ValueError(f"requested domains have no WRF frames: {', '.join(missing)}")

    for domain, frames in grouped.items():
        frames.sort(key=lambda item: (_parse_time(item.valid_time), str(item.path), item.time_index))
        parsed = [_parse_time(frame.valid_time) for frame in frames]
        if any(value is None for value in parsed):
            raise ValueError(f"unparsable {domain} valid time inventory")
        keys = [frame.valid_time for frame in frames]
        duplicates = sorted({key for key in keys if keys.count(key) > 1})
        if duplicates:
            raise ValueError(f"duplicate {domain} valid times: {duplicates}")
        cadence = _PLAN_CADENCE_SECONDS.get(domain)
        if cadence is not None:
            intervals = [
                int((right - left).total_seconds())
                for left, right in zip(parsed, parsed[1:], strict=False)
            ]
            if any(interval != cadence for interval in intervals):
                raise ValueError(
                    f"{domain} cadence mismatch: expected {cadence}s, found {intervals}")
    return dict(sorted(grouped.items()))


def _array(variable, time_index: int) -> np.ndarray:
    value = variable[time_index] if (
        variable.dimensions and variable.dimensions[0] == "Time") else variable[:]
    return np.asarray(np.ma.filled(value, np.nan), dtype=np.float64)


def _read(frame: Frame, name: str) -> np.ndarray | None:
    with netCDF4.Dataset(frame.path, "r") as ds:
        variable = ds.variables.get(name)
        return None if variable is None else _array(variable, frame.time_index)


def _inventory(frame: Frame) -> tuple[tuple[str, ...], dict[str, object], str | None]:
    with netCDF4.Dataset(frame.path, "r") as ds:
        names = tuple(sorted(ds.variables))
        metadata = {
            name: {
                "dimensions": list(ds.variables[name].dimensions),
                "dtype": str(ds.variables[name].dtype),
                "units": str(getattr(ds.variables[name], "units", "")),
            }
            for name in names
        }
        declared = getattr(ds, _COMPLETE_PRECIP_ATTRIBUTE, None)
    if declared is not None and (not isinstance(declared, str) or not declared.strip()):
        raise ValueError(
            f"{_COMPLETE_PRECIP_ATTRIBUTE} must name a complete precipitation variable")
    return names, metadata, declared.strip() if isinstance(declared, str) else None


def _require_finite(value: object, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        count = int(np.isfinite(array).sum())
        raise ValueError(f"{label} must be completely finite; finite={count}/{array.size}")
    return array


def _require_map(value: object, shape: tuple[int, int], label: str) -> np.ndarray:
    array = np.squeeze(_require_finite(value, label))
    if array.shape != shape:
        raise ValueError(f"{label} shape {array.shape} != {shape}")
    return array


def _native_reflectivity(frame: Frame) -> tuple[np.ndarray, np.ndarray]:
    value = _read(frame, "REFL_10CM")
    if value is None:
        raise ValueError(f"required REFL_10CM is missing from {frame.path}")
    value = np.squeeze(_require_finite(value, f"REFL_10CM in {frame.path}"))
    if value.ndim == 2:
        return value, value
    if value.ndim != 3:
        raise ValueError(f"REFL_10CM in {frame.path} must be 2-D or 3-D, got {value.shape}")
    return value[0], np.max(value, axis=0)


def _coordinates(frame: Frame, shape: tuple[int, int]) -> CoordinateGrid:
    for lat_name, lon_name in (("XLAT", "XLONG"), ("XLAT_M", "XLONG_M")):
        lat, lon = _read(frame, lat_name), _read(frame, lon_name)
        if lat is None and lon is None:
            continue
        if lat is None or lon is None:
            raise ValueError(f"{lat_name}/{lon_name} must both exist in {frame.path}")
        lat = _require_map(lat, shape, f"{lat_name} in {frame.path}")
        lon = _require_map(lon, shape, f"{lon_name} in {frame.path}")
        if np.any((lat < -90.0) | (lat > 90.0)):
            raise ValueError(f"{lat_name} outside [-90, 90] in {frame.path}")
        if np.any((lon < -360.0) | (lon > 360.0)):
            raise ValueError(f"{lon_name} outside [-360, 360] in {frame.path}")
        return CoordinateGrid(lon=lon, lat=lat, source=f"{lat_name}/{lon_name}")
    raise ValueError(f"valid finite XLAT/XLONG are required in {frame.path}")


def _precipitation(frame: Frame, variables: set[str], declared: str | None,
                   shape: tuple[int, int]) -> tuple[np.ndarray, str]:
    if {"RAINC", "RAINNC"} <= variables:
        rainc = _require_map(_read(frame, "RAINC"), shape, f"RAINC in {frame.path}")
        rainnc = _require_map(_read(frame, "RAINNC"), shape, f"RAINNC in {frame.path}")
        return rainc + rainnc, "RAINC+RAINNC"
    if declared is not None:
        if declared not in variables:
            raise ValueError(
                f"declared complete precipitation {declared!r} is missing from {frame.path}")
        return (_require_map(_read(frame, declared), shape,
                             f"declared complete precipitation {declared} in {frame.path}"),
                f"declared-complete:{declared}")
    present = sorted({"RAINC", "RAINNC"} & variables)
    raise ValueError(
        f"precipitation in {frame.path} requires RAINC and RAINNC or "
        f"{_COMPLETE_PRECIP_ATTRIBUTE}; present={present}")


def _wrf_array(wrf_file: WrfFile, frame: Frame, name: str, *,
               units: str | None = None) -> np.ndarray:
    kwargs = {"timeidx": frame.time_index}
    if units is not None:
        kwargs["units"] = units
    # Deliberately do not catch: wrf-rust failures are part of the public
    # fail-loud contract.
    return _require_finite(getvar(wrf_file, name, **kwargs),
                           f"wrf-rust {name} for {frame.path}[{frame.time_index}]")


def _prepare_frame(frame: Frame) -> FrameScience:
    names, metadata, declared_precip = _inventory(frame)
    variables = set(names)
    refl_low, refl_max = _native_reflectivity(frame)
    shape = tuple(int(item) for item in refl_max.shape)
    coordinates = _coordinates(frame, shape)
    if not isinstance(coordinates, CoordinateGrid) or coordinates.source not in {
            "XLAT/XLONG", "XLAT_M/XLONG_M"}:
        raise ValueError(
            f"coordinates for {frame.path} must carry a native XLAT/XLONG source")
    lon, lat = coordinates.lon, coordinates.lat
    precip, precip_source = _precipitation(
        frame, variables, declared_precip, shape)

    wrf_file = WrfFile(str(frame.path))
    slp = _require_map(
        _wrf_array(wrf_file, frame, "slp", units="hPa"), shape, "wrf-rust slp")
    uvmet10 = _wrf_array(wrf_file, frame, "uvmet10", units="m/s")
    if uvmet10.shape != (2, *shape):
        raise ValueError(f"wrf-rust uvmet10 shape {uvmet10.shape} != {(2, *shape)}")
    wspd10 = _require_map(
        _wrf_array(wrf_file, frame, "wspd10", units="m/s"), shape,
        "wrf-rust wspd10")

    dbz_max = None
    disagreement = None
    warning = None
    if variables & _HYDROMETEORS:
        dbz = _wrf_array(wrf_file, frame, "dbz")
        if dbz.ndim != 3 or dbz.shape[1:] != shape:
            raise ValueError(f"wrf-rust dbz shape {dbz.shape} is not (nz, {shape})")
        dbz_max = np.max(dbz, axis=0)
        disagreement = float(np.max(np.abs(refl_max - dbz_max)))
        if disagreement >= _DBZ_WARNING_THRESHOLD:
            warning = (
                "WARNING: model REFL_10CM and wrf-rust dbz column maxima "
                f"differ by up to {disagreement:.1f} dBZ")

    height500 = temp500 = u500 = v500 = None
    if frame.domain in _OUTER_DOMAINS:
        pressure = _wrf_array(wrf_file, frame, "pressure", units="hPa")
        height = _wrf_array(wrf_file, frame, "height", units="dam")
        temp = _wrf_array(wrf_file, frame, "temp", units="degC")
        uvmet = _wrf_array(wrf_file, frame, "uvmet", units="m/s")
        if pressure.ndim != 3 or pressure.shape[1:] != shape:
            raise ValueError(f"wrf-rust pressure shape {pressure.shape} is not (nz, {shape})")
        if height.shape != pressure.shape or temp.shape != pressure.shape:
            raise ValueError("wrf-rust pressure/height/temp 3-D shapes differ")
        if uvmet.shape != (2, *pressure.shape):
            raise ValueError(
                f"wrf-rust uvmet shape {uvmet.shape} != {(2, *pressure.shape)}")
        height500 = _require_map(interplevel(height, pressure, 500.0), shape,
                                 "wrf-rust height at 500 hPa")
        temp500 = _require_map(interplevel(temp, pressure, 500.0), shape,
                               "wrf-rust temperature at 500 hPa")
        u500 = _require_map(interplevel(uvmet[0], pressure, 500.0), shape,
                            "wrf-rust U wind at 500 hPa")
        v500 = _require_map(interplevel(uvmet[1], pressure, 500.0), shape,
                            "wrf-rust V wind at 500 hPa")

    uhel = uhel_max = None
    if frame.domain in _INNER_DOMAINS:
        uhel = _require_map(_wrf_array(wrf_file, frame, "uhel"), shape,
                            "wrf-rust uhel")
        uhel_max = _require_map(_wrf_array(wrf_file, frame, "uhel_max"), shape,
                                "wrf-rust uhel_max")

    return FrameScience(
        frame=frame, lon=lon, lat=lat, coordinate_source=coordinates.source,
        refl_low=refl_low, refl_max=refl_max,
        slp=slp, uvmet10=uvmet10, wspd10=wspd10, precip=precip,
        precip_source=precip_source, dbz_max=dbz_max,
        dbz_disagreement_max=disagreement, dbz_warning=warning,
        height500=height500, temp500=temp500, u500=u500, v500=v500,
        uhel=uhel, uhel_max=uhel_max, variables=names, metadata=metadata)


def _prepare_domains(frames_by_domain: Mapping[str, Sequence[Frame]]) -> dict[str, list[FrameScience]]:
    prepared = {
        domain: [_prepare_frame(frame) for frame in frames]
        for domain, frames in frames_by_domain.items()
    }
    for domain, entries in prepared.items():
        shapes = {entry.refl_max.shape for entry in entries}
        if len(shapes) != 1:
            raise ValueError(f"{domain} frame grid shapes differ: {sorted(shapes)}")
        sources = {entry.precip_source for entry in entries}
        if len(sources) != 1:
            raise ValueError(f"{domain} precipitation source changes across frames: {sources}")
        for before, after in zip(entries, entries[1:], strict=False):
            decrease = after.precip - before.precip
            if float(np.min(decrease)) < -1.0e-6:
                raise ValueError(
                    f"unexplained {domain} precipitation accumulator decrease between "
                    f"{before.frame.valid_time} and {after.frame.valid_time}: "
                    f"minimum delta={float(np.min(decrease)):.6g} mm")
    return prepared


def _validate_reference_coverage(
        candidate: Mapping[str, Sequence[Frame]],
        reference: Mapping[str, Sequence[Frame]]) -> None:
    if not reference:
        raise ValueError("reference_dir supplied but reference set is empty")
    if set(candidate) != set(reference):
        raise ValueError(
            f"reference domains {sorted(reference)} != candidate domains {sorted(candidate)}")
    for domain in candidate:
        candidate_times = [item.valid_time for item in candidate[domain]]
        reference_times = [item.valid_time for item in reference[domain]]
        if reference_times != candidate_times:
            raise ValueError(
                f"reference {domain} time inventory {reference_times} != "
                f"candidate inventory {candidate_times}")


def _finite_limits(arrays: Iterable[np.ndarray], *, include_zero: bool = False,
                   fixed: tuple[float, float] | None = None) -> tuple[float, float]:
    if fixed is not None:
        return fixed
    pieces = [np.asarray(value)[np.isfinite(value)] for value in arrays]
    if any(not piece.size for piece in pieces) or not pieces:
        raise ValueError("plotting limits require finite validated data")
    low = min(float(value.min()) for value in pieces)
    high = max(float(value.max()) for value in pieces)
    if include_zero:
        low = min(low, 0.0)
    if math.isclose(low, high):
        pad = max(abs(low) * 0.05, 1.0)
        low, high = low - pad, high + pad
    return low, high


def _decorate(ax, entry: FrameScience, field: str, units: str) -> None:
    frame = entry.frame
    ax.set_title(f"{frame.domain} | {frame.valid_time} | {field} ({units})", fontsize=9)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, color="0.65", linewidth=0.35, alpha=0.55)
    cosine = math.cos(math.radians(float(np.mean(entry.lat))))
    ax.set_aspect(1.0 / max(abs(cosine), 0.15), adjustable="box")


def _draw_map(ax, entry: FrameScience, value: np.ndarray, field: str, units: str,
              cmap: str, limits: tuple[float, float], *, contour40: bool = False):
    _decorate(ax, entry, field, units)
    mesh = ax.pcolormesh(entry.lon, entry.lat, value, shading="auto", cmap=cmap,
                         vmin=limits[0], vmax=limits[1])
    if contour40 and float(value.min()) <= 40.0 <= float(value.max()):
        ax.contour(entry.lon, entry.lat, value, levels=[40.0], colors="white",
                   linewidths=1.0)
    return mesh


def _save_single(path: Path, entry: FrameScience, value: np.ndarray, field: str,
                 units: str, cmap: str, limits: tuple[float, float], *,
                 contour40: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    mesh = _draw_map(ax, entry, value, field, units, cmap, limits,
                     contour40=contour40)
    fig.colorbar(mesh, ax=ax, label=f"{field} ({units})")
    fig.savefig(path, dpi=135)
    plt.close(fig)


def _save_mosaic(path: Path, entries: Sequence[FrameScience], field_name: str,
                 getter, units: str, cmap: str, limits: tuple[float, float], *,
                 contour40: bool = False) -> None:
    count = len(entries)
    columns = min(4, max(1, math.ceil(math.sqrt(count))))
    rows = math.ceil(count / columns)
    fig, axes = plt.subplots(rows, columns, squeeze=False,
                             figsize=(4.8 * columns, 3.8 * rows),
                             constrained_layout=True)
    mesh = None
    for ax, entry in zip(axes.flat, entries, strict=False):
        mesh = _draw_map(ax, entry, getter(entry), field_name, units, cmap, limits,
                         contour40=contour40)
    for ax in axes.flat[count:]:
        ax.set_visible(False)
    fig.colorbar(mesh, ax=list(axes.flat[:count]), label=f"{field_name} ({units})",
                 shrink=0.82)
    fig.savefig(path, dpi=125)
    plt.close(fig)


_T = TypeVar("_T")


def _selected(items: Sequence[_T]) -> list[_T]:
    if len(items) <= 3:
        return list(items)
    indices = sorted({0, len(items) // 2, len(items) - 1})
    return [items[index] for index in indices]


def _save_wind(path: Path, entries: Sequence[FrameScience]) -> None:
    chosen = _selected(entries)
    limits = _finite_limits((entry.wspd10 for entry in chosen), include_zero=True)
    fig, axes = plt.subplots(1, len(chosen), squeeze=False,
                             figsize=(5.3 * len(chosen), 4.7), constrained_layout=True)
    mesh = None
    for ax, entry in zip(axes.flat, chosen, strict=True):
        mesh = _draw_map(ax, entry, entry.wspd10, "Earth-relative 10 m wind",
                         "m s-1", "viridis", limits)
        stride = max(1, min(entry.wspd10.shape) // 18)
        ax.quiver(entry.lon[::stride, ::stride], entry.lat[::stride, ::stride],
                  entry.uvmet10[0, ::stride, ::stride],
                  entry.uvmet10[1, ::stride, ::stride], color="black",
                  pivot="middle", scale=None)
    fig.colorbar(mesh, ax=list(axes.flat), label="10 m wind speed (m s-1)")
    fig.savefig(path, dpi=135)
    plt.close(fig)


def _save_dbz_crosscheck(path: Path, entry: FrameScience) -> None:
    assert entry.dbz_max is not None
    difference = entry.refl_max - entry.dbz_max
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    for ax, value, title in zip(
            axes[:2], (entry.refl_max, entry.dbz_max),
            ("Model REFL_10CM column max", "wrf-rust dbz column max"), strict=True):
        mesh = _draw_map(ax, entry, value, title, "dBZ", "viridis", (0.0, 75.0),
                         contour40=True)
        fig.colorbar(mesh, ax=ax, shrink=0.82)
    diff_limit = max(1.0, float(np.max(np.abs(difference))))
    mesh = _draw_map(axes[2], entry, difference, "REFL_10CM minus wrf-rust dbz",
                     "dBZ", "coolwarm", (-diff_limit, diff_limit))
    fig.colorbar(mesh, ax=axes[2], shrink=0.82)
    note = entry.dbz_warning or (
        f"Cross-check within {_DBZ_WARNING_THRESHOLD:.0f} dBZ warning threshold; "
        f"max |difference|={entry.dbz_disagreement_max:.1f} dBZ")
    fig.suptitle(note, color="crimson" if entry.dbz_warning else "black", fontsize=10)
    fig.savefig(path, dpi=135)
    plt.close(fig)


def _save_500(path: Path, entry: FrameScience) -> None:
    assert all(item is not None for item in (
        entry.height500, entry.temp500, entry.u500, entry.v500))
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    limits = _finite_limits((entry.temp500,))
    mesh = _draw_map(ax, entry, entry.temp500, "500 hPa height / temperature / wind",
                     "degC", "coolwarm", limits)
    height = entry.height500
    if float(height.max()) > float(height.min()):
        levels = np.linspace(float(height.min()), float(height.max()), 10)
        contours = ax.contour(entry.lon, entry.lat, height, levels=levels,
                              colors="black", linewidths=0.75)
        ax.clabel(contours, inline=True, fontsize=7, fmt="%.0f dam")
    stride = max(1, min(entry.temp500.shape) // 18)
    ax.quiver(entry.lon[::stride, ::stride], entry.lat[::stride, ::stride],
              entry.u500[::stride, ::stride], entry.v500[::stride, ::stride],
              color="black", pivot="middle", scale=None)
    fig.colorbar(mesh, ax=ax, label="500 hPa temperature (degC)")
    fig.savefig(path, dpi=135)
    plt.close(fig)


def _save_hook(path: Path, entry: FrameScience) -> None:
    assert entry.uhel_max is not None
    flat_index = int(np.argmax(entry.uhel_max))
    center_y, center_x = np.unravel_index(flat_index, entry.uhel_max.shape)
    radius = min(30, max(2, min(entry.uhel_max.shape) // 2))
    ys = slice(max(0, center_y - radius), min(entry.uhel_max.shape[0], center_y + radius + 1))
    xs = slice(max(0, center_x - radius), min(entry.uhel_max.shape[1], center_x + radius + 1))
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    lon, lat = entry.lon[ys, xs], entry.lat[ys, xs]
    refl = entry.refl_low[ys, xs]
    mesh = ax.pcolormesh(lon, lat, refl, shading="auto", cmap="viridis",
                         vmin=0.0, vmax=75.0)
    stride = max(1, min(refl.shape) // 15)
    ax.quiver(lon[::stride, ::stride], lat[::stride, ::stride],
              entry.uvmet10[0, ys, xs][::stride, ::stride],
              entry.uvmet10[1, ys, xs][::stride, ::stride], color="white",
              pivot="middle", scale=None)
    ax.scatter(entry.lon[center_y, center_x], entry.lat[center_y, center_x],
               marker="*", s=120, color="red", edgecolor="black", zorder=5,
               label="wrf-rust uhel_max maximum")
    ax.legend(loc="upper right", fontsize=8)
    _decorate(ax, entry, "Low-level REFL_10CM hook / earth-relative wind", "dBZ")
    fig.colorbar(mesh, ax=ax, label="Low-level REFL_10CM (dBZ)")
    fig.savefig(path, dpi=145)
    plt.close(fig)


def _save_uh_track(path: Path, entries: Sequence[FrameScience]) -> None:
    track = np.max(np.stack([entry.uhel_max for entry in entries]), axis=0)
    final = entries[-1]
    _save_single(path, final, track, "Max-over-time wrf-rust uhel_max track",
                 "m2 s-2", "magma", _finite_limits((track,), include_zero=True))


def _save_precip_increment(path: Path, entries: Sequence[FrameScience]) -> None:
    increments = [after.precip - before.precip
                  for before, after in zip(entries, entries[1:], strict=False)]
    selected_entries = list(entries[1:])
    limits = _finite_limits(increments, include_zero=True)
    fig, axes = plt.subplots(1, len(increments), squeeze=False,
                             figsize=(5.0 * len(increments), 4.6),
                             constrained_layout=True)
    mesh = None
    for ax, entry, value in zip(axes.flat, selected_entries, increments, strict=True):
        mesh = _draw_map(ax, entry, value, "Trailing-cadence precipitation",
                         "mm", "viridis", limits)
    fig.colorbar(mesh, ax=list(axes.flat), label="Precipitation increment (mm)")
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _save_comparison(path: Path, candidate: FrameScience, reference: FrameScience,
                     field: str, getter, units: str, cmap: str,
                     limits: tuple[float, float] | None = None) -> None:
    candidate_value, reference_value = getter(candidate), getter(reference)
    if candidate_value.shape != reference_value.shape:
        raise ValueError(f"reference {field} shape differs at {candidate.frame.valid_time}")
    limits = _finite_limits((candidate_value, reference_value), fixed=limits)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), constrained_layout=True)
    meshes = []
    for ax, entry, value, label in zip(
            axes, (candidate, reference), (candidate_value, reference_value),
            ("candidate", "reference"), strict=True):
        meshes.append(_draw_map(ax, entry, value, f"{field} - {label}", units,
                                cmap, limits, contour40="REFL" in field))
    fig.colorbar(meshes[0], ax=axes, label=f"{field} ({units})", shrink=0.85)
    fig.savefig(path, dpi=135)
    plt.close(fig)


def _render_domain(domain: str, entries: Sequence[FrameScience], outdir: Path,
                   reference: Sequence[FrameScience] | None) -> list[Path]:
    products: list[Path] = []

    for entry in entries:
        path = outdir / f"{domain}_refl_{entry.frame.slug}.png"
        _save_single(path, entry, entry.refl_max, "Column-max model REFL_10CM", "dBZ",
                     "viridis", (0.0, 75.0), contour40=True)
        products.append(path)
        if entry.dbz_max is not None:
            path = outdir / f"{domain}_refl_dbz_crosscheck_{entry.frame.slug}.png"
            _save_dbz_crosscheck(path, entry)
            products.append(path)
    path = outdir / f"{domain}_refl_all_times.png"
    _save_mosaic(path, entries, "Column-max model REFL_10CM",
                 lambda item: item.refl_max, "dBZ", "viridis", (0.0, 75.0),
                 contour40=True)
    products.append(path)

    slp_limits = _finite_limits(entry.slp for entry in entries)
    for entry in entries:
        path = outdir / f"{domain}_mslp_{entry.frame.slug}.png"
        _save_single(path, entry, entry.slp, "wrf-rust mean sea-level pressure",
                     "hPa", "cividis", slp_limits)
        products.append(path)

    path = outdir / f"{domain}_wind10_selected.png"
    _save_wind(path, entries)
    products.append(path)

    for entry in entries:
        if entry.height500 is not None:
            path = outdir / f"{domain}_500hpa_{entry.frame.slug}.png"
            _save_500(path, entry)
            products.append(path)

    if entries[0].uhel_max is not None:
        for entry in entries:
            path = outdir / f"{domain}_uhel_{entry.frame.slug}.png"
            _save_single(path, entry, entry.uhel, "wrf-rust updraft helicity",
                         "m2 s-2", "magma",
                         _finite_limits((item.uhel for item in entries), include_zero=True))
            products.append(path)
            path = outdir / f"{domain}_hook_echo_{entry.frame.slug}.png"
            _save_hook(path, entry)
            products.append(path)
        path = outdir / f"{domain}_uhel_track_all_times.png"
        _save_uh_track(path, entries)
        products.append(path)

    path = outdir / f"{domain}_precip_accum_final.png"
    _save_single(path, entries[-1], entries[-1].precip, "Accumulated precipitation",
                 "mm", "viridis",
                 _finite_limits((item.precip for item in entries), include_zero=True))
    products.append(path)
    if len(entries) > 1:
        path = outdir / f"{domain}_precip_cadence_increments.png"
        _save_precip_increment(path, entries)
        products.append(path)

    if reference is not None:
        for candidate, baseline in zip(entries, reference, strict=True):
            for stem, field, getter, units, cmap, limits in (
                ("refl", "Column-max REFL_10CM", lambda item: item.refl_max,
                 "dBZ", "viridis", (0.0, 75.0)),
                ("mslp", "Mean sea-level pressure", lambda item: item.slp,
                 "hPa", "cividis", None),
                ("precip", "Accumulated precipitation", lambda item: item.precip,
                 "mm", "viridis", None),
            ):
                path = outdir / f"{domain}_{stem}_comparison_{candidate.frame.slug}.png"
                _save_comparison(path, candidate, baseline, field, getter, units, cmap,
                                 limits)
                products.append(path)
    return products


def _sha256_file(path: Path, *, block_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(block_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _stats(value: np.ndarray, units: str) -> dict[str, object]:
    return {
        "min": float(np.min(value)),
        "max": float(np.max(value)),
        "finite_count": int(value.size),
        "units": units,
    }


def _source_identity(frames_by_domain: Mapping[str, Sequence[Frame]], root: Path
                    ) -> list[dict[str, object]]:
    paths = sorted({frame.path.resolve() for frames in frames_by_domain.values()
                    for frame in frames}, key=str)
    identities = []
    root_resolved = root.resolve()
    for path in paths:
        if not path.is_relative_to(root_resolved):
            raise ValueError(f"input frame escaped run root: {path}")
        identities.append({
            "relative_path": path.relative_to(root_resolved).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        })
    return identities


def _frame_summary(entry: FrameScience, root: Path) -> dict[str, object]:
    diagnostics = ["slp", "uvmet10", "wspd10"]
    key_fields: dict[str, object] = {
        "REFL_10CM_COMPOSITE": _stats(entry.refl_max, "dBZ"),
        "SLP_WRF_RUST": _stats(entry.slp, "hPa"),
        "UVMET10_U": _stats(entry.uvmet10[0], "m s-1"),
        "UVMET10_V": _stats(entry.uvmet10[1], "m s-1"),
        "WSPD10_WRF_RUST": _stats(entry.wspd10, "m s-1"),
        "TOTAL_PRECIP": _stats(entry.precip, "mm"),
    }
    if entry.dbz_max is not None:
        diagnostics.append("dbz")
        key_fields["DBZ_WRF_RUST_COMPOSITE"] = _stats(entry.dbz_max, "dBZ")
    if entry.height500 is not None:
        diagnostics.extend(("pressure", "height", "temp", "uvmet", "interplevel"))
        key_fields["HEIGHT_500_WRF_RUST"] = _stats(entry.height500, "dam")
        key_fields["TEMP_500_WRF_RUST"] = _stats(entry.temp500, "degC")
    if entry.uhel is not None:
        diagnostics.extend(("uhel", "uhel_max"))
        key_fields["UHEL_WRF_RUST"] = _stats(entry.uhel, "m2 s-2")
        key_fields["UHEL_MAX_WRF_RUST"] = _stats(entry.uhel_max, "m2 s-2")
    resolved = entry.frame.path.resolve()
    return {
        "path": resolved.relative_to(root.resolve()).as_posix(),
        "time_index": entry.frame.time_index,
        "valid_time": entry.frame.valid_time,
        "coordinate_source": entry.coordinate_source,
        "precipitation_source": entry.precip_source,
        "wrf_rust_diagnostics": diagnostics,
        "dbz_crosscheck": ({
            "max_absolute_difference_dbz": entry.dbz_disagreement_max,
            "warning": entry.dbz_warning,
            "warning_threshold_dbz": _DBZ_WARNING_THRESHOLD,
        } if entry.dbz_max is not None else None),
        "key_fields": key_fields,
    }


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True,
                               allow_nan=False) + "\n", encoding="utf-8")


def generate_products(run_dir: str | Path, outdir: str | Path, *,
                      domains: Sequence[str] | None = None,
                      reference_dir: str | Path | None = None) -> dict[str, object]:
    """Validate all inputs, then generate the complete domain product inventory."""
    wrf_version = _assert_wrf_rust_version()
    normalized_domains = [item.lower() for item in domains] if domains else None
    frames_by_domain = discover_frames(run_dir, normalized_domains)
    prepared = _prepare_domains(frames_by_domain)
    source_inputs = _source_identity(frames_by_domain, Path(run_dir))

    reference_frames: dict[str, list[Frame]] | None = None
    reference_prepared: dict[str, list[FrameScience]] | None = None
    reference_inputs: list[dict[str, object]] | None = None
    if reference_dir is not None:
        reference_frames = discover_frames(reference_dir, normalized_domains)
        _validate_reference_coverage(frames_by_domain, reference_frames)
        reference_prepared = _prepare_domains(reference_frames)
        reference_inputs = _source_identity(reference_frames, Path(reference_dir))

    # No filesystem publication occurs above this line.
    output = Path(outdir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    product_paths: list[Path] = []
    for domain, entries in prepared.items():
        product_paths.extend(_render_domain(
            domain, entries, output,
            None if reference_prepared is None else reference_prepared[domain]))

    product_artifacts = [{
        "relative_path": path.relative_to(output).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "wrf_rust_version": wrf_version,
    } for path in product_paths]
    domain_summaries: dict[str, object] = {}
    root = Path(run_dir)
    for domain, entries in prepared.items():
        union = sorted({name for entry in entries for name in entry.variables})
        metadata: dict[str, object] = {}
        for entry in entries:
            metadata.update(entry.metadata)
        required_products = ["reflectivity", "slp", "uvmet10", "precipitation"]
        if domain in _OUTER_DOMAINS:
            required_products.append("500hpa")
        if domain in _INNER_DOMAINS:
            required_products.extend(("uhel_track", "hook_echo"))
        domain_summaries[domain] = {
            "frame_count": len(entries),
            "cadence_seconds": _PLAN_CADENCE_SECONDS.get(domain),
            "required_products": required_products,
            "frame_inventory": [_frame_summary(entry, root) for entry in entries],
            "field_inventory": union,
            "field_metadata": {name: metadata[name] for name in union},
        }

    summary: dict[str, object] = {
        "schema": 2,
        "science_core": {"distribution": "wrf-rust", "version": wrf_version},
        "input_identity": {
            "root": ".",
            "artifacts": source_inputs,
            "sha256": hashlib.sha256(json.dumps(
                source_inputs, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        },
        "reference_identity": (None if reference_inputs is None else {
            "root": ".", "artifacts": reference_inputs,
            "sha256": hashlib.sha256(json.dumps(
                reference_inputs, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        }),
        "diagnostics": {
            "run_dir_absolute": str(Path(run_dir).resolve()),
            "reference_dir_absolute": (str(Path(reference_dir).resolve())
                                       if reference_dir is not None else None),
        },
        "domains": domain_summaries,
        "product_count": len(product_paths),
        "products": [path.relative_to(output).as_posix() for path in product_paths],
        "product_artifacts": product_artifacts,
        "warnings": [entry.dbz_warning for entries in prepared.values()
                     for entry in entries if entry.dbz_warning is not None],
    }
    _write_json(output / "run-summary.json", summary)
    return summary


def _domains_arg(value: str) -> list[str]:
    domains = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not domains or any(re.fullmatch(r"d\d{2}", item) is None for item in domains):
        raise argparse.ArgumentTypeError(
            "domains must be comma-separated WRF IDs (for example d01,d02)")
    return domains


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--domains", type=_domains_arg,
                        default=["d01", "d02", "d03", "d04"])
    parser.add_argument("--reference-dir", type=Path)
    args = parser.parse_args(argv)
    generate_products(args.run_dir, args.outdir, domains=args.domains,
                      reference_dir=args.reference_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
