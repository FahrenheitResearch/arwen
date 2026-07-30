"""Metadata-driven NOAA 20CRv3 NetCDF-CF ensemble adapter primitives.

This module deliberately stops short of claiming a certified real-data path.
It owns the format-neutral parts that can be proved without a private 20CRv3
sample: CF coordinate discovery, cadence and coverage validation, canonical
unit/layout normalization, bounded member batching, and deterministic atomic
publication.  A concrete corpus must still pass the gates documented in
``docs/native-20crv3-source-adapter-spec.md`` before this NetCDF-CF route may
be marked runnable.  It is complementary to, and does not weaken or replace,
the exact member-bound GRIB2 route in ``gpuwm.twentycrv3_direct``.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Callable, Iterable, Mapping, Sequence
import uuid

import netCDF4
import numpy as np

from gpuwm.source_frame import (
    FieldDescriptor,
    GridDescriptor,
    SourceFrameHeader,
    TimeDescriptor,
    VerticalDescriptor,
    validate_source_frame,
)


DISCOVERY_SCHEMA = "gpuwm-20crv3-discovery-v1"
STREAM_SCHEMA = "gpuwm-20crv3-member-stream-v1"

_ROLE_ALIASES = {
    "member": {"member", "members", "ensemble", "ens", "realization", "number"},
    "time": {"time", "valid_time", "validtime"},
    "latitude": {"latitude", "lat", "y"},
    "longitude": {"longitude", "lon", "x"},
    "pressure": {"pressure", "plev", "level", "lev", "isobaric", "isobaricinhpa"},
}
_ROLE_STANDARD_NAMES = {
    "member": {"realization"},
    "time": {"time"},
    "latitude": {"latitude"},
    "longitude": {"longitude"},
    "pressure": {"air_pressure"},
}
_ROLE_AXES = {
    "member": {"E"},
    "time": {"T"},
    "latitude": {"Y"},
    "longitude": {"X"},
    "pressure": {"Z"},
}

_CANONICAL_UNITS = {
    "air_temperature": "K",
    "specific_humidity": "kg kg-1",
    "eastward_wind": "m s-1",
    "northward_wind": "m s-1",
    "geopotential_height": "m",
    "air_pressure": "Pa",
    "surface_pressure": "Pa",
    "terrain_height": "m",
    "skin_temperature": "K",
    "air_temperature_2m": "K",
    "specific_humidity_2m": "kg kg-1",
    "eastward_wind_10m": "m s-1",
    "northward_wind_10m": "m s-1",
    "land_fraction": "1",
    "soil_temperature": "K",
    "volumetric_soil_moisture": "m3 m-3",
    "snow_water_equivalent": "kg m-2",
    "snow_depth": "m",
    "sea_ice_fraction": "1",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True) + "\n").encode("utf-8")


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _normalized_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="strict").strip()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("member coordinate contains a non-finite value")
        if value.is_integer():
            return str(int(value))
    return str(value).strip()


def _coordinate_values(variable: netCDF4.Variable) -> np.ndarray:
    raw = variable[:]
    if np.ma.isMaskedArray(raw) and np.ma.getmaskarray(raw).any():
        raise ValueError(f"coordinate {variable.name!r} contains a fill value")
    array = np.asarray(raw)
    if array.dtype.kind in {"S", "U"} and array.ndim > 1:
        try:
            array = np.asarray(netCDF4.chartostring(array))
        except (ValueError, TypeError):
            pass
    return array.reshape(-1)


def _orientation(values: Sequence[float], *, label: str) -> str:
    numeric = np.asarray(values, dtype=np.float64)
    if numeric.size < 2:
        return "singleton"
    if not np.isfinite(numeric).all():
        raise ValueError(f"{label} coordinate contains a non-finite value")
    differences = np.diff(numeric)
    if np.all(differences > 0.0):
        return "ascending"
    if np.all(differences < 0.0):
        return "descending"
    if np.any(differences == 0.0):
        raise ValueError(f"{label} coordinate contains duplicate values")
    raise ValueError(f"{label} coordinate is not monotone")


def _unit_key(units: str) -> str:
    return re.sub(r"[\s_*^/]+", "", units.strip().lower())


def _pressure_pa(values: Sequence[float], units: str) -> np.ndarray:
    key = _unit_key(units)
    raw = np.asarray(values, dtype=np.float64)
    if key in {"pa", "pascal", "pascals"}:
        factor = 1.0
    elif key in {"hpa", "hectopascal", "hectopascals", "mb", "mbar"}:
        factor = 100.0
    else:
        raise ValueError(f"unsupported pressure-coordinate units {units!r}")
    result = raw * factor
    if not np.isfinite(result).all() or np.any(result <= 0.0):
        raise ValueError("pressure coordinate must contain finite positive values")
    if np.unique(result).size != result.size:
        raise ValueError("pressure coordinate contains duplicate values")
    _orientation(result, label="pressure")
    return result


def _datetime_iso(value: Any) -> str:
    # Formatting the decoded CF fields directly also represents legitimate
    # non-Gregorian dates (for example day 30 of a 360-day February).  Calendar
    # identity remains explicit in ``CoordinateMetadata.calendar``.
    microsecond = int(getattr(value, "microsecond", 0))
    suffix = f".{microsecond:06d}" if microsecond else ""
    return (
        f"{int(value.year):04d}-{int(value.month):02d}-{int(value.day):02d}T"
        f"{int(value.hour):02d}:{int(value.minute):02d}:{int(value.second):02d}"
        f"{suffix}Z"
    )


def _decode_times(variable: netCDF4.Variable) -> tuple[tuple[Any, ...], tuple[str, ...]]:
    units = str(getattr(variable, "units", "")).strip()
    calendar = str(getattr(variable, "calendar", "standard")).strip()
    if " since " not in units.lower():
        raise ValueError(
            f"time coordinate {variable.name!r} lacks CF '<unit> since <epoch>' units")
    raw = _coordinate_values(variable)
    if not np.issubdtype(raw.dtype, np.number):
        raise ValueError("time coordinate must be numeric CF time")
    try:
        decoded = tuple(netCDF4.num2date(
            raw, units=units, calendar=calendar,
            only_use_cftime_datetimes=True,
        ))
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"cannot decode time coordinate: {exc}") from exc
    return decoded, tuple(_datetime_iso(value) for value in decoded)


def _seconds_between(first: Any, second: Any) -> int:
    delta = second - first
    seconds = delta.total_seconds()
    rounded = int(round(seconds))
    if abs(seconds - rounded) > 1.0e-6:
        raise ValueError(f"time gap is not an integral number of seconds: {seconds}")
    return rounded


@dataclass(frozen=True)
class InputIdentity:
    path: str
    byte_count: int
    sha256: str


@dataclass(frozen=True)
class CoordinateMetadata:
    role: str
    name: str
    dimensions: tuple[str, ...]
    dtype: str
    units: str
    calendar: str | None
    count: int
    orientation: str
    first: str | float
    last: str | float
    values_sha256: str
    periodic: bool = False


@dataclass(frozen=True)
class VariableMetadata:
    name: str
    path: str
    dimensions: tuple[str, ...]
    dtype: str
    units: str
    standard_name: str
    fill_values: tuple[str, ...]


@dataclass(frozen=True)
class TwentyCrDiscovery:
    inputs: tuple[InputIdentity, ...]
    product_identity: Mapping[str, str]
    member_coordinate: CoordinateMetadata
    time_coordinate: CoordinateMetadata
    latitude_coordinate: CoordinateMetadata
    longitude_coordinate: CoordinateMetadata
    pressure_coordinate: CoordinateMetadata
    members: tuple[str, ...]
    valid_times: tuple[str, ...]
    latitude: tuple[float, ...]
    longitude: tuple[float, ...]
    pressure_pa: tuple[float, ...]
    cadence_seconds: int | None
    variables: Mapping[str, VariableMetadata]
    coordinate_paths: Mapping[str, str]
    schema: str = DISCOVERY_SCHEMA

    def to_manifest(self) -> dict[str, Any]:
        value = asdict(self)
        value["variables"] = {
            name: asdict(self.variables[name]) for name in sorted(self.variables)
        }
        value["coordinate_paths"] = dict(sorted(self.coordinate_paths.items()))
        return value


@dataclass(frozen=True)
class FieldBinding:
    canonical_name: str
    source_variable: str
    allow_shared_member: bool = False
    allow_static_time: bool = False


@dataclass(frozen=True)
class MemberBatchPlan:
    members: tuple[str, ...]
    worker_limit: int
    memory_budget_bytes: int
    estimated_member_bytes: int
    batch_size: int
    batches: tuple[tuple[str, ...], ...]


class EnsembleRunError(RuntimeError):
    """A member failed, so no global PASS pointer was published."""

    def __init__(self, failures: Mapping[str, str]):
        self.failures = dict(sorted(failures.items()))
        detail = "; ".join(f"{member}: {error}" for member, error in self.failures.items())
        super().__init__(f"20CRv3 member stream failed: {detail}")


def _values_digest(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


def _candidate_score(name: str, variable: netCDF4.Variable, role: str) -> int:
    normalized = name.strip().lower().replace("-", "_")
    score = 0
    if normalized in _ROLE_ALIASES[role]:
        score += 8
    standard_name = str(getattr(variable, "standard_name", "")).strip().lower()
    if standard_name in _ROLE_STANDARD_NAMES[role]:
        score += 16
    axis = str(getattr(variable, "axis", "")).strip().upper()
    if axis in _ROLE_AXES[role]:
        score += 4
    units = str(getattr(variable, "units", "")).strip().lower()
    if role == "time" and " since " in units:
        score += 12
    elif role == "latitude" and "degree" in units and "north" in units:
        score += 12
    elif role == "longitude" and "degree" in units and "east" in units:
        score += 12
    elif role == "pressure" and _unit_key(units) in {
        "pa", "pascal", "pascals", "hpa", "hectopascal", "hectopascals", "mb", "mbar"
    }:
        score += 3
    if variable.ndim == 1 and variable.dimensions == (name,):
        score += 2
    return score


def _select_coordinate(
    datasets: Sequence[tuple[Path, netCDF4.Dataset]], role: str,
    override: str | None,
) -> tuple[Path, netCDF4.Variable]:
    candidates: list[tuple[int, str, Path, netCDF4.Variable]] = []
    for path, dataset in datasets:
        for name, variable in dataset.variables.items():
            if override is not None and name != override:
                continue
            score = _candidate_score(name, variable, role)
            if override is not None:
                score = 10_000
            if score > 0:
                candidates.append((score, name, path, variable))
    if not candidates:
        suffix = f" named {override!r}" if override else ""
        raise ValueError(f"cannot discover {role} coordinate{suffix}")
    best_score = max(value[0] for value in candidates)
    best = [value for value in candidates if value[0] == best_score]
    names = {value[1] for value in best}
    if len(names) != 1:
        raise ValueError(
            f"ambiguous {role} coordinate candidates: {', '.join(sorted(names))}")
    chosen_name = next(iter(names))
    chosen = next(value for value in best if value[1] == chosen_name)
    variable = chosen[3]
    is_char_member = (
        role == "member" and np.dtype(variable.dtype).kind in {"S", "U", "O"}
        and variable.ndim == 2
    )
    if not is_char_member and (
            variable.ndim != 1 or len(variable.dimensions) != 1):
        raise ValueError(
            f"{role} coordinate {chosen_name!r} must be one-dimensional")
    return chosen[2], chosen[3]


def _validate_coordinate_occurrences(
    datasets: Sequence[tuple[Path, netCDF4.Dataset]], role: str,
    chosen: netCDF4.Variable,
) -> None:
    """Require repeated coordinate variables to carry identical semantics."""

    if role == "time":
        _, expected = _decode_times(chosen)
    elif role == "member":
        expected = tuple(_normalized_text(value) for value in _coordinate_values(chosen))
    elif role == "pressure":
        expected = tuple(map(float, _pressure_pa(
            _coordinate_values(chosen), str(getattr(chosen, "units", "")))))
    else:
        expected = tuple(map(float, _coordinate_values(chosen)))
    expected_dimensions = tuple(chosen.dimensions)
    for path, dataset in datasets:
        if chosen.name not in dataset.variables:
            continue
        variable = dataset.variables[chosen.name]
        if tuple(variable.dimensions) != expected_dimensions:
            raise ValueError(
                f"{role} coordinate {chosen.name!r} in {path} has dimensions "
                f"{tuple(variable.dimensions)!r}; expected {expected_dimensions!r}")
        if role == "time":
            _, observed = _decode_times(variable)
            if str(getattr(variable, "calendar", "standard")) != str(
                    getattr(chosen, "calendar", "standard")):
                raise ValueError(f"time calendar differs across 20CRv3 inputs at {path}")
        elif role == "member":
            observed = tuple(
                _normalized_text(value) for value in _coordinate_values(variable))
        elif role == "pressure":
            observed = tuple(map(float, _pressure_pa(
                _coordinate_values(variable), str(getattr(variable, "units", "")))))
        else:
            observed = tuple(map(float, _coordinate_values(variable)))
        if observed != expected:
            raise ValueError(
                f"{role} coordinate {chosen.name!r} differs across 20CRv3 inputs at {path}")


def _coordinate_metadata(
    role: str, variable: netCDF4.Variable, values: np.ndarray,
    *, orientation: str, periodic: bool = False,
) -> CoordinateMetadata:
    serialized = np.asarray(values)
    first: str | float
    last: str | float
    if serialized.dtype.kind in {"O", "S", "U"}:
        text = np.asarray([_normalized_text(value) for value in serialized], dtype="U")
        digest = hashlib.sha256("\0".join(text.tolist()).encode("utf-8")).hexdigest()
        first, last = text[0], text[-1]
    else:
        numeric = np.asarray(serialized, dtype=np.float64)
        digest = _values_digest(numeric.astype("<f8", copy=False))
        first, last = float(numeric[0]), float(numeric[-1])
    return CoordinateMetadata(
        role=role,
        name=variable.name,
        dimensions=tuple(variable.dimensions),
        dtype=str(variable.dtype),
        units=str(getattr(variable, "units", "")),
        calendar=(str(getattr(variable, "calendar", "standard"))
                  if role == "time" else None),
        count=int(serialized.size),
        orientation=orientation,
        first=first,
        last=last,
        values_sha256=digest,
        periodic=periodic,
    )


def _infer_periodic(longitude: np.ndarray) -> bool:
    values = np.asarray(longitude, dtype=np.float64)
    if values.size < 2:
        return False
    ordered = values if values[0] < values[-1] else values[::-1]
    differences = np.diff(ordered)
    if not np.allclose(differences, differences[0], rtol=0.0, atol=1.0e-8):
        return False
    return abs(float(differences[0]) * values.size - 360.0) <= 1.0e-6


def _infer_cadence(decoded_times: Sequence[Any], valid_times: Sequence[str]) -> int | None:
    if len(decoded_times) < 2:
        return None
    gaps = [_seconds_between(first, second)
            for first, second in zip(decoded_times, decoded_times[1:])]
    for index, gap in enumerate(gaps):
        if gap == 0:
            raise ValueError(f"duplicate valid time {valid_times[index + 1]}")
        if gap < 0:
            raise ValueError(
                f"reversed valid-time gap {valid_times[index]} -> "
                f"{valid_times[index + 1]} is {gap}s")
    expected = gaps[0]
    for index, gap in enumerate(gaps[1:], start=1):
        if gap != expected:
            raise ValueError(
                f"irregular valid-time gap {valid_times[index]} -> "
                f"{valid_times[index + 1]} is {gap}s; expected {expected}s")
    return expected


def discover_20crv3(
    paths: Sequence[str | Path], *,
    coordinate_names: Mapping[str, str] | None = None,
) -> TwentyCrDiscovery:
    """Discover a 20CR-like NetCDF catalog without assuming count/cadence.

    ``coordinate_names`` is an explicit disambiguation mechanism, not a way to
    bypass metadata validation.  Every selected coordinate is still checked.
    """

    if not paths:
        raise ValueError("at least one 20CRv3 input path is required")
    resolved = tuple(sorted(
        (Path(path).resolve() for path in paths), key=lambda value: str(value)))
    if len(set(resolved)) != len(resolved):
        raise ValueError("20CRv3 input paths contain duplicates")
    for path in resolved:
        if not path.is_file():
            raise FileNotFoundError(path)
    identities = tuple(InputIdentity(
        path=str(path), byte_count=path.stat().st_size, sha256=_sha256(path))
        for path in sorted(resolved, key=lambda value: str(value)))
    overrides = dict(coordinate_names or {})
    unknown_overrides = set(overrides) - set(_ROLE_ALIASES)
    if unknown_overrides:
        raise ValueError(
            "unknown coordinate-role overrides: " + ", ".join(sorted(unknown_overrides)))

    opened: list[tuple[Path, netCDF4.Dataset]] = []
    try:
        opened = [(path, netCDF4.Dataset(path, "r")) for path in resolved]
        selected = {
            role: _select_coordinate(opened, role, overrides.get(role))
            for role in ("member", "time", "latitude", "longitude", "pressure")
        }
        for role, (_, variable) in selected.items():
            _validate_coordinate_occurrences(opened, role, variable)

        member_raw = _coordinate_values(selected["member"][1])
        members = tuple(_normalized_text(value) for value in member_raw)
        if not members or any(not value for value in members):
            raise ValueError("member coordinate contains an empty label")
        if len(set(members)) != len(members):
            raise ValueError("member coordinate contains duplicate labels")

        decoded_times, valid_times = _decode_times(selected["time"][1])
        if not valid_times:
            raise ValueError("time coordinate is empty")
        cadence = _infer_cadence(decoded_times, valid_times)

        latitude_raw = np.asarray(_coordinate_values(selected["latitude"][1]),
                                  dtype=np.float64)
        longitude_raw = np.asarray(_coordinate_values(selected["longitude"][1]),
                                   dtype=np.float64)
        if latitude_raw.size < 2 or longitude_raw.size < 2:
            raise ValueError("latitude and longitude coordinates need at least two points")
        latitude_orientation = _orientation(latitude_raw, label="latitude")
        longitude_orientation = _orientation(longitude_raw, label="longitude")
        pressure_raw = _coordinate_values(selected["pressure"][1])
        pressure = _pressure_pa(
            pressure_raw, str(getattr(selected["pressure"][1], "units", "")))
        pressure_orientation = _orientation(pressure, label="pressure")

        coordinate_names_set = {variable.name for _, variable in selected.values()}
        variables: dict[str, VariableMetadata] = {}
        coordinate_paths = {
            role: str(path) for role, (path, _) in selected.items()
        }
        for path, dataset in opened:
            for name, variable in dataset.variables.items():
                if name in coordinate_names_set:
                    continue
                if name in variables:
                    raise ValueError(
                        f"data variable {name!r} occurs in multiple input files")
                fills: list[str] = []
                for attribute in ("_FillValue", "missing_value"):
                    if hasattr(variable, attribute):
                        raw_fill = np.asarray(getattr(variable, attribute)).reshape(-1)
                        fills.extend(_normalized_text(value) for value in raw_fill)
                variables[name] = VariableMetadata(
                    name=name,
                    path=str(path),
                    dimensions=tuple(variable.dimensions),
                    dtype=str(variable.dtype),
                    units=str(getattr(variable, "units", "")),
                    standard_name=str(getattr(variable, "standard_name", "")),
                    fill_values=tuple(fills),
                )

        product_identity: dict[str, str] = {}
        for _, dataset in opened:
            for attribute in (
                "title", "source", "institution", "dataset", "product",
                "experiment_id", "history",
            ):
                if hasattr(dataset, attribute):
                    value = str(getattr(dataset, attribute)).strip()
                    if value:
                        key = attribute
                        if key in product_identity and product_identity[key] != value:
                            key = f"{Path(dataset.filepath()).name}:{attribute}"
                        product_identity[key] = value
        if not product_identity:
            product_identity["identity"] = "unlabeled-netcdf-catalog"

        return TwentyCrDiscovery(
            inputs=identities,
            product_identity=dict(sorted(product_identity.items())),
            member_coordinate=_coordinate_metadata(
                "member", selected["member"][1], member_raw,
                orientation="catalog_order"),
            time_coordinate=_coordinate_metadata(
                "time", selected["time"][1], np.asarray(valid_times, dtype="U"),
                orientation="ascending"),
            latitude_coordinate=_coordinate_metadata(
                "latitude", selected["latitude"][1], latitude_raw,
                orientation=latitude_orientation),
            longitude_coordinate=_coordinate_metadata(
                "longitude", selected["longitude"][1], longitude_raw,
                orientation=longitude_orientation,
                periodic=_infer_periodic(longitude_raw)),
            pressure_coordinate=_coordinate_metadata(
                "pressure", selected["pressure"][1], pressure,
                orientation=pressure_orientation),
            members=members,
            valid_times=valid_times,
            latitude=tuple(map(float, latitude_raw)),
            longitude=tuple(map(float, longitude_raw)),
            pressure_pa=tuple(map(float, pressure)),
            cadence_seconds=cadence,
            variables=variables,
            coordinate_paths=coordinate_paths,
        )
    finally:
        for _, dataset in opened:
            dataset.close()


def validate_time_coverage(
    discovery: TwentyCrDiscovery, *, start: str, end: str,
    cadence_seconds: int | None = None,
) -> tuple[str, ...]:
    """Validate exact boundary coverage and return the selected time slice."""

    if cadence_seconds is not None and cadence_seconds <= 0:
        raise ValueError("requested cadence_seconds must be positive")
    times = discovery.valid_times
    try:
        first = times.index(start)
    except ValueError as exc:
        raise ValueError(f"requested initialization time {start} is absent") from exc
    try:
        last = times.index(end)
    except ValueError as exc:
        raise ValueError(f"requested forecast endpoint {end} is absent") from exc
    if last < first:
        raise ValueError("forecast endpoint precedes initialization time")
    selected = times[first:last + 1]
    if len(selected) > 1:
        actual = discovery.cadence_seconds
        if actual is None:
            raise ValueError("cannot infer cadence from selected time axis")
        if cadence_seconds is not None and actual != cadence_seconds:
            raise ValueError(
                f"source cadence is {actual}s; requested {cadence_seconds}s")
    return selected


def validate_member_selection(
    discovery: TwentyCrDiscovery, members: Iterable[str],
) -> tuple[str, ...]:
    selected = tuple(str(value) for value in members)
    if not selected:
        raise ValueError("at least one member must be selected")
    if len(set(selected)) != len(selected):
        raise ValueError("selected members contain duplicates")
    available = set(discovery.members)
    missing = [value for value in selected if value not in available]
    if missing:
        raise ValueError("unknown 20CRv3 members: " + ", ".join(missing))
    return selected


def validate_target_pressure_levels(
    discovery: TwentyCrDiscovery, target_pressure_pa: Sequence[float],
) -> tuple[float, ...]:
    """Validate an arbitrary target vertical grid against source-top coverage.

    The target count is intentionally unconstrained.  In particular, an
    ensemble member count and a model vertical count are unrelated concepts.
    """

    target = np.asarray(target_pressure_pa, dtype=np.float64)
    if target.ndim != 1 or target.size < 2:
        raise ValueError("target pressure grid must be a one-dimensional multi-level axis")
    if not np.isfinite(target).all() or np.any(target <= 0.0):
        raise ValueError("target pressure grid must contain finite positive values")
    if np.unique(target).size != target.size:
        raise ValueError("target pressure grid contains duplicate values")
    _orientation(target, label="target pressure")
    source_top = min(discovery.pressure_pa)
    target_top = min(target)
    if source_top > target_top:
        raise ValueError(
            f"source top {source_top:g} Pa does not cover requested model top "
            f"{target_top:g} Pa; upper-air extrapolation is forbidden")
    return tuple(map(float, target))


def _convert_units(canonical_name: str, source_units: str, values: np.ndarray) -> np.ndarray:
    try:
        target = _CANONICAL_UNITS[canonical_name]
    except KeyError as exc:
        raise ValueError(f"unknown canonical 20CRv3 field {canonical_name!r}") from exc
    source = _unit_key(source_units)
    factor = 1.0
    offset = 0.0
    if target == "K" and source in {"k", "kelvin"}:
        pass
    elif target == "K" and source in {"c", "degc", "degreecelsius", "degreescelsius"}:
        offset = 273.15
    elif target == "Pa" and source in {"pa", "pascal", "pascals"}:
        pass
    elif target == "Pa" and source in {"hpa", "hectopascal", "hectopascals", "mb", "mbar"}:
        factor = 100.0
    elif target == "kg kg-1" and source in {
        "1", "kgkg-1", "kgkg-1", "kgkg", "dimensionless"
    }:
        pass
    elif target == "kg kg-1" and source in {"gkg-1", "gkg"}:
        factor = 1.0e-3
    elif target == "m s-1" and source in {"ms-1", "msec-1", "meterpersecond", "metrespersecond"}:
        pass
    elif target == "m" and source in {"m", "meter", "meters", "metre", "metres"}:
        pass
    elif canonical_name == "geopotential_height" and source in {
        "m2s-2", "m2s-2", "m2sec-2"
    }:
        factor = 1.0 / 9.80665
    elif target == "1" and source in {"1", "fraction", "dimensionless"}:
        pass
    elif target == "1" and source in {"%", "percent", "percentage"}:
        factor = 0.01
    elif target == "m3 m-3" and source in {"m3m-3", "1", "fraction", "dimensionless"}:
        pass
    elif target == "kg m-2" and source in {"kgm-2", "mm", "kgm-2"}:
        pass
    else:
        raise ValueError(
            f"{canonical_name} has unsupported units {source_units!r}; expected {target}")
    converted = np.asarray(values, dtype=np.float64) * factor + offset
    if not np.isfinite(converted).all():
        raise ValueError(f"{canonical_name} contains a fill or non-finite value")
    return np.ascontiguousarray(converted, dtype=np.float32)


def validate_bindings(
    discovery: TwentyCrDiscovery, bindings: Sequence[FieldBinding], *,
    required_fields: Iterable[str],
) -> tuple[FieldBinding, ...]:
    bindings = tuple(bindings)
    canonical = [value.canonical_name for value in bindings]
    if len(set(canonical)) != len(canonical):
        raise ValueError("duplicate canonical 20CRv3 field binding")
    missing_variables = [value.source_variable for value in bindings
                         if value.source_variable not in discovery.variables]
    if missing_variables:
        raise ValueError(
            "missing 20CRv3 variables: " + ", ".join(sorted(missing_variables)))
    missing_fields = set(required_fields) - set(canonical)
    if missing_fields:
        raise ValueError(
            "missing canonical 20CRv3 fields: " + ", ".join(sorted(missing_fields)))
    for binding in bindings:
        if binding.canonical_name not in _CANONICAL_UNITS:
            raise ValueError(f"unknown canonical 20CRv3 field {binding.canonical_name!r}")
        metadata = discovery.variables[binding.source_variable]
        # Validate unit compatibility without materializing the variable.
        _convert_units(binding.canonical_name, metadata.units,
                       np.asarray([0.0], dtype=np.float32))
    return bindings


def _locate_index(values: Sequence[str], requested: str, label: str) -> int:
    try:
        return tuple(values).index(requested)
    except ValueError as exc:
        raise ValueError(f"unknown {label} {requested!r}") from exc


def read_member_frame(
    discovery: TwentyCrDiscovery, *, member: str, valid_time: str,
    bindings: Sequence[FieldBinding],
) -> dict[str, np.ndarray]:
    """Read one member/time and normalize dimensions to ``(z,y,x)``/``(y,x)``."""

    member_index = _locate_index(discovery.members, member, "member")
    time_index = _locate_index(discovery.valid_times, valid_time, "valid time")
    member_dim = discovery.member_coordinate.dimensions[0]
    time_dim = discovery.time_coordinate.dimensions[0]
    lat_dim = discovery.latitude_coordinate.dimensions[0]
    lon_dim = discovery.longitude_coordinate.dimensions[0]
    pressure_dim = discovery.pressure_coordinate.dimensions[0]
    lat_reverse = discovery.latitude_coordinate.orientation == "descending"
    lon_reverse = discovery.longitude_coordinate.orientation == "descending"
    pressure_reverse = discovery.pressure_coordinate.orientation == "ascending"

    by_path: dict[str, list[FieldBinding]] = {}
    for binding in bindings:
        by_path.setdefault(discovery.variables[binding.source_variable].path, []).append(binding)
    result: dict[str, np.ndarray] = {}
    for path, path_bindings in by_path.items():
        with netCDF4.Dataset(path, "r") as dataset:
            for binding in path_bindings:
                variable = dataset.variables[binding.source_variable]
                dimensions = list(variable.dimensions)
                selection: list[int | slice] = [slice(None)] * variable.ndim
                if member_dim in dimensions:
                    selection[dimensions.index(member_dim)] = member_index
                elif not binding.allow_shared_member:
                    raise ValueError(
                        f"{binding.source_variable} lacks member dimension {member_dim!r}")
                if time_dim in dimensions:
                    selection[dimensions.index(time_dim)] = time_index
                elif not binding.allow_static_time:
                    raise ValueError(
                        f"{binding.source_variable} lacks time dimension {time_dim!r}")
                raw = variable[tuple(selection)]
                if np.ma.isMaskedArray(raw):
                    if np.ma.getmaskarray(raw).any():
                        raise ValueError(
                            f"member {member} time {valid_time} field "
                            f"{binding.source_variable} contains a fill value")
                    raw = raw.data
                remaining = [dimension for dimension, selected in zip(dimensions, selection)
                             if isinstance(selected, slice)]
                vertical_dimensions = [
                    dimension for dimension in remaining
                    if dimension not in {lat_dim, lon_dim}
                ]
                if len(vertical_dimensions) > 1:
                    raise ValueError(
                        f"{binding.source_variable} has multiple unresolved vertical/extra "
                        f"dimensions: {tuple(vertical_dimensions)!r}")
                has_pressure = vertical_dimensions == [pressure_dim]
                expected = vertical_dimensions + [lat_dim, lon_dim]
                if set(remaining) != set(expected) or len(remaining) != len(expected):
                    raise ValueError(
                        f"{binding.source_variable} dimensions after member/time selection "
                        f"are {tuple(remaining)!r}; expected a permutation of {tuple(expected)!r}")
                permutation = tuple(remaining.index(dimension) for dimension in expected)
                normalized = np.transpose(np.asarray(raw), permutation)
                if has_pressure and pressure_reverse:
                    normalized = normalized[::-1, ...]
                if lat_reverse:
                    normalized = normalized[..., ::-1, :]
                if lon_reverse:
                    normalized = normalized[..., ::-1]
                result[binding.canonical_name] = _convert_units(
                    binding.canonical_name, str(getattr(variable, "units", "")),
                    normalized)
    return result


def _array_reference(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return "sha256:" + hashlib.sha256(array.view(np.uint8)).hexdigest()


def build_source_frame_header(
    discovery: TwentyCrDiscovery, *, member: str, valid_time: str,
    fields: Mapping[str, np.ndarray], bindings: Sequence[FieldBinding],
    earth_shape: str, initialization_policies: Mapping[str, str],
    soil_depth_m: Sequence[float] = (),
) -> SourceFrameHeader:
    """Bind one normalized member/time to gpuwm's canonical frame contract.

    ``earth_shape`` is mandatory because a generic NetCDF latitude/longitude
    file does not prove the upstream model's Earth convention.  Pressure is
    derived from the validated coordinate rather than copied into every file.
    """

    validate_member_selection(discovery, (member,))
    _locate_index(discovery.valid_times, valid_time, "valid time")
    if not earth_shape.strip():
        raise ValueError("earth_shape must be explicitly declared")
    ny = len(discovery.latitude)
    nx = len(discovery.longitude)
    pressure = np.asarray(discovery.pressure_pa, dtype=np.float64)
    if pressure[0] < pressure[-1]:
        pressure = pressure[::-1]
    soil = np.asarray(soil_depth_m, dtype=np.float64)
    if soil.size:
        if not np.isfinite(soil).all() or np.any(soil <= 0.0):
            raise ValueError("soil_depth_m must contain finite positive depths")
        if _orientation(soil, label="soil depth") != "ascending":
            raise ValueError("soil_depth_m must increase downward")

    source_names = {value.canonical_name: value.source_variable for value in bindings}
    descriptors: list[FieldDescriptor] = []
    time = TimeDescriptor(
        reference_time=valid_time,
        valid_time=valid_time,
        lead_seconds=0,
    )
    for canonical_name in sorted(fields):
        if canonical_name == "air_pressure":
            raise ValueError(
                "air_pressure is derived from the validated pressure coordinate; "
                "do not supply a second pressure field")
        if canonical_name not in _CANONICAL_UNITS:
            raise ValueError(f"unknown canonical 20CRv3 field {canonical_name!r}")
        raw = fields[canonical_name]
        if np.ma.isMaskedArray(raw) and np.ma.getmaskarray(raw).any():
            raise ValueError(f"canonical field {canonical_name} contains a fill value")
        array = np.ascontiguousarray(raw)
        if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
            raise ValueError(f"canonical field {canonical_name} is not finite numeric data")
        if array.ndim == 2:
            dimensions = ("latitude", "longitude")
            vertical = None
            expected = (ny, nx)
        elif array.ndim == 3 and canonical_name in {
            "soil_temperature", "volumetric_soil_moisture",
        }:
            if not soil.size:
                raise ValueError(f"{canonical_name} requires explicit soil_depth_m")
            dimensions = ("soil_depth", "latitude", "longitude")
            vertical = "soil_depth"
            expected = (soil.size, ny, nx)
        elif array.ndim == 3:
            dimensions = ("pressure", "latitude", "longitude")
            vertical = "pressure"
            expected = (pressure.size, ny, nx)
        else:
            raise ValueError(
                f"canonical field {canonical_name} has unsupported shape {array.shape}")
        if array.shape != expected:
            raise ValueError(
                f"canonical field {canonical_name} shape {array.shape} != {expected}")
        descriptors.append(FieldDescriptor(
            canonical_name=canonical_name,
            units=_CANONICAL_UNITS[canonical_name],
            dimensions=dimensions,
            grid_location="mass",
            vertical_coordinate=vertical,
            time=time,
            data_reference=_array_reference(array),
            dtype=array.dtype.str,
            shape=tuple(map(int, array.shape)),
            source_field=source_names.get(canonical_name, "derived_or_composed"),
        ))

    pressure_bytes = np.ascontiguousarray(pressure, dtype="<f8")
    descriptors.append(FieldDescriptor(
        canonical_name="air_pressure",
        units="Pa",
        dimensions=("pressure", "latitude", "longitude"),
        grid_location="mass",
        vertical_coordinate="pressure",
        time=time,
        data_reference=(
            "derived:pressure-coordinate:" +
            hashlib.sha256(pressure_bytes.view(np.uint8)).hexdigest()
        ),
        dtype="<f4",
        shape=(pressure.size, ny, nx),
        source_field=discovery.pressure_coordinate.name,
    ))
    verticals: dict[str, VerticalDescriptor] = {
        "pressure": VerticalDescriptor(
            coordinate="pressure", level_count=pressure.size,
            level_values=tuple(map(float, pressure)), units="Pa", positive="down"),
    }
    if soil.size:
        verticals["soil_depth"] = VerticalDescriptor(
            coordinate="soil_depth", level_count=soil.size,
            level_values=tuple(map(float, soil)), units="m", positive="down")
    header = SourceFrameHeader(
        source_id=f"20crv3/member/{member}",
        source_cycle=valid_time,
        grid=GridDescriptor(
            projection="regular_latitude_longitude",
            nx=nx,
            ny=ny,
            earth_shape=earth_shape,
            scan_order="+longitude,+latitude",
            wind_basis="earth_relative",
            parameters={
                "latitude_values_sha256": discovery.latitude_coordinate.values_sha256,
                "longitude_values_sha256": discovery.longitude_coordinate.values_sha256,
                "longitude_periodic": int(discovery.longitude_coordinate.periodic),
            },
        ),
        vertical_coordinates=verticals,
        fields=tuple(sorted(descriptors, key=lambda value: value.canonical_name)),
        initialization_policies=dict(initialization_policies),
    )
    validate_source_frame(header)
    return header


def estimate_member_bytes(
    discovery: TwentyCrDiscovery, bindings: Sequence[FieldBinding], *,
    valid_times: Sequence[str],
) -> int:
    """Conservative decoded FP32 payload estimate for one selected member."""

    if not valid_times:
        raise ValueError("valid_times is empty")
    for value in valid_times:
        _locate_index(discovery.valid_times, value, "valid time")
    member_dim = discovery.member_coordinate.dimensions[0]
    time_dim = discovery.time_coordinate.dimensions[0]
    total = 0
    for binding in bindings:
        metadata = discovery.variables[binding.source_variable]
        with netCDF4.Dataset(metadata.path, "r") as dataset:
            variable = dataset.variables[binding.source_variable]
            elements = 1
            for dimension, size in zip(variable.dimensions, variable.shape):
                if dimension in {member_dim, time_dim}:
                    continue
                elements *= int(size)
            time_multiplier = len(valid_times) if time_dim in variable.dimensions else 1
            total += elements * time_multiplier * np.dtype(np.float32).itemsize
    return total


def plan_member_batches(
    discovery: TwentyCrDiscovery, *, members: Iterable[str], worker_limit: int,
    memory_budget_bytes: int, estimated_member_bytes: int,
) -> MemberBatchPlan:
    selected = validate_member_selection(discovery, members)
    if worker_limit <= 0:
        raise ValueError("worker_limit must be positive")
    if memory_budget_bytes <= 0 or estimated_member_bytes <= 0:
        raise ValueError("memory budget and estimated member bytes must be positive")
    capacity = memory_budget_bytes // estimated_member_bytes
    if capacity < 1:
        raise ValueError(
            f"memory budget {memory_budget_bytes} B cannot admit one estimated "
            f"member ({estimated_member_bytes} B)")
    batch_size = min(worker_limit, capacity, len(selected))
    batches = tuple(tuple(selected[index:index + batch_size])
                    for index in range(0, len(selected), batch_size))
    return MemberBatchPlan(
        members=selected,
        worker_limit=worker_limit,
        memory_budget_bytes=memory_budget_bytes,
        estimated_member_bytes=estimated_member_bytes,
        batch_size=batch_size,
        batches=batches,
    )


def _safe_member(member: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", member):
        raise ValueError(f"member id {member!r} is not safe for an output path")
    return member


def _file_receipts(root: Path) -> list[dict[str, Any]]:
    receipts: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*"), key=lambda value: value.as_posix()):
        if path.is_symlink():
            raise ValueError(f"member output may not contain a symlink: {path}")
        if path.is_file():
            receipts.append({
                "path": path.relative_to(root).as_posix(),
                "byte_count": path.stat().st_size,
                "sha256": _sha256(path),
            })
    if not receipts:
        raise ValueError(f"member output directory {root} is empty")
    return receipts


def _verify_generation(
    generation: Path, manifest: Mapping[str, Any], manifest_sha: str,
) -> None:
    manifest_path = generation / "manifest.json"
    if not manifest_path.is_file() or _sha256(manifest_path) != manifest_sha:
        raise RuntimeError(
            f"existing generation {generation} does not match its content identity")
    for member in manifest["members"]:
        member_root = generation / "members" / _safe_member(member["member_id"])
        observed = _file_receipts(member_root)
        if observed != member["outputs"]:
            raise RuntimeError(
                f"existing generation {generation} has corrupt output for member "
                f"{member['member_id']}")


def stream_members_atomic(
    output_root: str | Path, *, run_name: str, plan: MemberBatchPlan,
    process_member: Callable[[str, Path], None],
    manifest_context: Mapping[str, Any],
) -> dict[str, Any]:
    """Process bounded member batches and atomically publish a PASS pointer.

    The callback must finish decoding and writing before it returns.  Its
    arrays are consequently out of scope before another batch is admitted.
    Only small hash receipts survive in the controller.  Worker count and
    completion order are intentionally omitted from scientific identity.
    """

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_name):
        raise ValueError("run_name must be a safe non-empty identifier")
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    staging_parent = root / ".staging"
    generations = root / "generations"
    staging_parent.mkdir(exist_ok=True)
    generations.mkdir(exist_ok=True)
    staging = staging_parent / f"{run_name}.{uuid.uuid4().hex}"
    staging.mkdir()
    failures: dict[str, str] = {}
    receipts: dict[str, list[dict[str, Any]]] = {}

    def one(member: str) -> tuple[str, list[dict[str, Any]]]:
        safe = _safe_member(member)
        member_root = staging / "members" / safe
        member_root.mkdir(parents=True, exist_ok=False)
        process_member(member, member_root)
        member_receipts = _file_receipts(member_root)
        gc.collect()
        return member, member_receipts

    try:
        for batch in plan.batches:
            with ThreadPoolExecutor(max_workers=min(plan.batch_size, len(batch))) as executor:
                futures = {executor.submit(one, member): member for member in batch}
                for future in as_completed(futures):
                    member = futures[future]
                    try:
                        member_id, member_receipts = future.result()
                        receipts[member_id] = member_receipts
                    except BaseException as exc:
                        failures[member] = f"{type(exc).__name__}: {exc}"
            if failures:
                raise EnsembleRunError(failures)

        manifest = {
            "schema": STREAM_SCHEMA,
            "verdict": "PASS",
            "run_name": run_name,
            "members": [
                {"member_id": member, "outputs": receipts[member]}
                for member in sorted(receipts)
            ],
            "context": manifest_context,
        }
        manifest_payload = _json_bytes(manifest)
        manifest_sha = hashlib.sha256(manifest_payload).hexdigest()
        (staging / "manifest.json").write_bytes(manifest_payload)
        generation = generations / manifest_sha
        if generation.exists():
            _verify_generation(generation, manifest, manifest_sha)
            shutil.rmtree(staging)
        else:
            os.replace(staging, generation)
        pointer = {
            "schema": "gpuwm-20crv3-current-pointer-v1",
            "generation": manifest_sha,
            "manifest_sha256": manifest_sha,
            "manifest": f"generations/{manifest_sha}/manifest.json",
        }
        _atomic_bytes(root / f"{run_name}.json", _json_bytes(pointer))
        return manifest
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


__all__ = [
    "DISCOVERY_SCHEMA",
    "STREAM_SCHEMA",
    "CoordinateMetadata",
    "EnsembleRunError",
    "FieldBinding",
    "InputIdentity",
    "MemberBatchPlan",
    "TwentyCrDiscovery",
    "VariableMetadata",
    "build_source_frame_header",
    "discover_20crv3",
    "estimate_member_bytes",
    "plan_member_batches",
    "read_member_frame",
    "stream_members_atomic",
    "validate_bindings",
    "validate_member_selection",
    "validate_target_pressure_levels",
    "validate_time_coverage",
]
