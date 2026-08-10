"""Fail-closed ``rw-wps.mapping.v1`` source-frame materialization.

This module is the gpuwm-side consumer of RW-WPS's sealed declarative
mapping contract.  It deliberately stops at a canonical, hash-bound source
frame: interpolation, WRF-real initialization, static geography, nesting and
NetCDF export remain owned by their existing gpuwm implementations.

The public ``mapped`` capability is implementation-runnable under this strict
contract. Stock-WRF certification remains keyed to exact retained mapping,
composition, source, and domain evidence; arbitrary mappings do not inherit
that evidence.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import tempfile
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence

import netCDF4
import numpy as np

from gpuwm.ingest.grib import Era5Snapshot, build_rust_bridge, inspect_grib1_envelopes
from gpuwm.ingest.quantization import admit_bounded
from gpuwm.ingest.soil_contract import (
    MAPPED_SOIL_MOISTURE,
    MAPPED_SOIL_TEMPERATURE,
)
from gpuwm.source_frame import (
    FieldDescriptor,
    GridDescriptor,
    SourceFrameHeader,
    TimeDescriptor,
    VerticalDescriptor,
    validate_source_frame,
)


MAPPING_SCHEMA = "rw-wps.mapping.v1"
FRAME_EVIDENCE_SCHEMA = "gpuwm-mapped-source-frames-v1"
INPUT_MANIFEST_SCHEMA = "gpuwm-mapped-source-inputs-v1"
INSPECTION_SCHEMA = "gpuwm-mapped-source-inspection-v1"
_MAX_RETAINED_AUTHORITY_BYTES = 16 * 1024 * 1024
_GRIB2_AUTHORITY_KEYS = (
    "center",
    "subcenter",
    "master_table_version",
    "local_table_version",
)

_FORMATS = {"grib1", "grib2", "netcdf"}
_AXES = {"time", "member", "vertical", "y", "x", "soil"}
_LOCATIONS = {"mass", "u_face", "v_face", "surface", "soil"}
_STAGGERING = {"none", "x", "y", "z"}
_VERTICAL_KINDS = {
    "pressure", "hybrid_sigma_pressure", "model_level", "height",
    "soil_depth", "embedded_levels",
}
_POLICY_FIELDS = {
    "cloud_water_mixing_ratio",
    "rain_water_mixing_ratio",
    "cloud_ice_mixing_ratio",
    "snow_mixing_ratio",
    "graupel_or_hail_mixing_ratio",
    "vertical_velocity",
    "snow_water_equivalent",
    "snow_depth",
    "sea_ice_fraction",
}
_DERIVATION_ARGUMENTS = {
    "copy": ({"source"}, set()),
    "wind_speed": ({"u", "v"}, set()),
    "specific_humidity_from_rh": (
        {"relative_humidity", "temperature", "pressure"}, set()
    ),
    "relative_humidity_from_dewpoint": ({"dewpoint", "temperature"}, set()),
    "geopotential_height": ({"geopotential"}, {"gravity_m_s2"}),
    "pressure_from_vertical_coordinate": (set(), set()),
    "specific_humidity_from_dewpoint": (
        {"dewpoint", "temperature", "pressure"}, set()
    ),
}
_CANONICAL_REQUIREMENTS = {
    "air_temperature": (("vertical", "y", "x"), "mass", "K"),
    "specific_humidity": (("vertical", "y", "x"), "mass", "kg kg-1"),
    "eastward_wind": (("vertical", "y", "x"), "mass", "m s-1"),
    "northward_wind": (("vertical", "y", "x"), "mass", "m s-1"),
    "geopotential_height": (("vertical", "y", "x"), "mass", "m"),
    "surface_pressure": (("y", "x"), "surface", "Pa"),
    "terrain_height": (("y", "x"), "surface", "m"),
    "skin_temperature": (("y", "x"), "surface", "K"),
    "air_temperature_2m": (("y", "x"), "surface", "K"),
    "specific_humidity_2m": (("y", "x"), "surface", "kg kg-1"),
    "eastward_wind_10m": (("y", "x"), "surface", "m s-1"),
    "northward_wind_10m": (("y", "x"), "surface", "m s-1"),
    "land_fraction": (("y", "x"), "surface", "1"),
    "soil_temperature": (("soil", "y", "x"), "soil", "K"),
    "volumetric_soil_moisture": (("soil", "y", "x"), "soil", "m3 m-3"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(array.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _reject_duplicate_object_pairs(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _load_json_document(path: Path, label: str) -> object:
    data = Path(path).read_bytes()
    return _load_json_bytes(data, label, path)


def _load_json_bytes(data: bytes, label: str, path: Path | None = None) -> object:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} is not UTF-8: {error}") from error
    try:
        return json.loads(text, object_pairs_hook=_reject_duplicate_object_pairs)
    except json.JSONDecodeError as error:
        location = "" if path is None else f" {path}"
        raise ValueError(f"invalid {label} JSON{location}: {error}") from error


@dataclass(frozen=True)
class _AuthoritySnapshot:
    path: Path
    sha256: str
    size: int
    mtime_ns: int
    device: int
    inode: int
    mode: int
    data: bytes | None = None

    @property
    def identity(self) -> tuple[object, ...]:
        return (
            self.path,
            self.sha256,
            self.size,
            self.mtime_ns,
            self.device,
            self.inode,
            self.mode,
        )


def _snapshot_authority(
    path: str | Path,
    *,
    retain_bytes: bool = False,
) -> _AuthoritySnapshot:
    path = Path(path).resolve()
    path_before = path.stat()
    digest = hashlib.sha256()
    retained = bytearray() if retain_bytes else None
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) \
        | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"authority is not a regular file: {path}")
        if retain_bytes and before.st_size > _MAX_RETAINED_AUTHORITY_BYTES:
            raise ValueError(
                f"retained authority exceeds {_MAX_RETAINED_AUTHORITY_BYTES} "
                f"bytes: {path}"
            )
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
            if retained is not None:
                retained.extend(block)
        after = os.fstat(stream.fileno())
    path_after = path.stat()

    def identity(status: os.stat_result) -> tuple[int, int, int, int]:
        return (
            int(status.st_dev),
            int(status.st_ino),
            int(status.st_size),
            int(status.st_mtime_ns),
        )

    # On Windows, ``fstat`` does not synthesize the execute permission bits
    # that ``Path.stat`` adds for ``.exe`` files.  Bind descriptor identity to
    # path identity, but compare permission modes within the same stat API.
    if (
        identity(before) != identity(after)
        or identity(path_before) != identity(path_after)
        or identity(after) != identity(path_after)
        or stat.S_IFMT(before.st_mode) != stat.S_IFMT(after.st_mode)
        or stat.S_IFMT(after.st_mode) != stat.S_IFMT(path_after.st_mode)
        or path_before.st_mode != path_after.st_mode
    ):
        raise ValueError(f"authority changed while snapshotting: {path}")
    data = None if retained is None else bytes(retained)
    if data is not None and len(data) != after.st_size:
        raise ValueError(f"authority changed while reading: {path}")
    return _AuthoritySnapshot(
        path=path,
        sha256=digest.hexdigest(),
        size=int(after.st_size),
        mtime_ns=int(after.st_mtime_ns),
        device=int(after.st_dev),
        inode=int(after.st_ino),
        mode=int(path_after.st_mode),
        data=data,
    )


def _require_authority_snapshot(snapshot: _AuthoritySnapshot) -> None:
    current = _snapshot_authority(snapshot.path)
    if current.identity != snapshot.identity:
        raise ValueError(f"authority changed after validation: {snapshot.path}")


def _object(
    value: object,
    label: str,
    *,
    allowed: Iterable[str],
    required: Iterable[str] = (),
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be an object")
    allowed_set = set(allowed)
    unknown = sorted(set(value) - allowed_set)
    if unknown:
        raise ValueError(f"{label} has unknown key(s): {unknown}")
    missing = sorted(set(required) - set(value))
    if missing:
        raise ValueError(f"{label} is missing required key(s): {missing}")
    return value


def _finite_json(value: object, label: str = "mapping") -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{label} contains a non-finite number")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _finite_json(item, f"{label}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _finite_json(item, f"{label}.{key}")
        return
    raise TypeError(f"{label} contains unsupported JSON value {type(value).__name__}")


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _integer(
    value: object, label: str, *, minimum: int = 0, maximum: int | None = None
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        bound = f"{minimum}..{maximum}" if maximum is not None else f">={minimum}"
        raise ValueError(f"{label} must be in {bound}")
    return value


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _axes(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty axis list")
    result = tuple(_string(item, label) for item in value)
    unknown = sorted(set(result) - _AXES)
    if unknown:
        raise ValueError(f"{label} has unsupported axes {unknown}")
    if len(set(result)) != len(result):
        raise ValueError(f"{label} contains a duplicate axis")
    return result


def _validate_selector(selector: object, expected_format: str, label: str) -> dict[str, object]:
    value = _object(
        selector,
        label,
        allowed={
            "format", "parameter", "table_version", "center", "subcenter",
            "master_table_version", "local_table_version", "level_type",
            "level_value", "second_level_type", "second_level_value",
            "discipline", "category", "member", "name", "standard_name",
        },
        required={"format"},
    )
    if value["format"] != expected_format:
        raise ValueError(
            f"{label}.format={value['format']!r} differs from mapping format "
            f"{expected_format!r}"
        )
    allowed_by_format = {
        "grib1": {
            "format", "parameter", "table_version", "center", "level_type",
            "level_value",
        },
        "grib2": {
            "format", "discipline", "category", "parameter", "level_type",
            "level_value", "second_level_type", "second_level_value", "member",
            *_GRIB2_AUTHORITY_KEYS,
        },
        "netcdf": {"format", "name", "standard_name"},
    }[expected_format]
    incompatible = sorted(set(value) - allowed_by_format)
    if incompatible:
        raise ValueError(
            f"{label} has keys incompatible with {expected_format}: {incompatible}"
        )
    required = {
        "grib1": {"parameter"},
        "grib2": {"discipline", "category", "parameter"},
        "netcdf": set(),
    }[expected_format]
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(f"{label} is missing selector key(s): {missing}")
    if expected_format == "netcdf" and not (
        isinstance(value.get("name"), str) and value.get("name")
        or isinstance(value.get("standard_name"), str) and value.get("standard_name")
    ):
        raise ValueError(f"{label} needs name and/or standard_name")
    if expected_format == "netcdf":
        for key in ("name", "standard_name"):
            if value.get(key) is not None:
                _string(value[key], f"{label}.{key}")
    else:
        integer_keys = (
            ("parameter", 0, 255), ("level_type", 0, 255),
            ("second_level_type", 0, 255),
            ("table_version", 0, 255),
            ("center", 0, 65535 if expected_format == "grib2" else 255),
            ("subcenter", 0, 65535),
            ("master_table_version", 0, 255),
            ("local_table_version", 0, 255),
            ("discipline", 0, 255), ("category", 0, 255),
            ("member", 0, 255),
        )
        for key, minimum, maximum in integer_keys:
            if value.get(key) is not None:
                _integer(
                    value[key], f"{label}.{key}",
                    minimum=minimum, maximum=maximum,
                )
            elif key in required:
                raise ValueError(f"{label}.{key} must not be null")
        if value.get("level_value") is not None:
            _number(value["level_value"], f"{label}.level_value")
        second_keys = {
            key for key in ("second_level_type", "second_level_value")
            if value.get(key) is not None
        }
        if second_keys and expected_format != "grib2":
            raise ValueError(f"{label} second fixed surfaces require GRIB2")
        if second_keys and second_keys != {"second_level_type", "second_level_value"}:
            raise ValueError(
                f"{label} second_level_type and second_level_value are an atomic pair"
            )
        if value.get("second_level_value") is not None:
            _number(value["second_level_value"], f"{label}.second_level_value")
        missing_identifiers = {
            key
            for key in (
                ("parameter", "level_type", "table_version", "center")
                if expected_format == "grib1"
                else ("discipline", "category", "parameter", "second_level_type")
            )
            if value.get(key) == 255
        }
        if missing_identifiers:
            raise ValueError(
                f"{label} uses missing/undefined identifier code 255 for "
                f"{sorted(missing_identifiers)}"
            )
        if expected_format == "grib2" and value.get("level_type") == 255 and any(
            value.get(key) is not None
            for key in ("level_value", "second_level_type", "second_level_value")
        ):
            raise ValueError(
                f"{label} uses level_type=255 with fixed-surface metadata"
            )
        if expected_format == "grib2":
            local_identifiers = {
                key: identifier
                for key in (
                    "discipline", "category", "parameter", "level_type",
                    "second_level_type",
                )
                if isinstance((identifier := value.get(key)), int)
                and 192 <= identifier <= 254
            }
            if local_identifiers:
                missing_authority = sorted(
                    set(_GRIB2_AUTHORITY_KEYS) - set(value)
                )
                rendered = ", ".join(
                    f"{key}={identifier}"
                    for key, identifier in sorted(local_identifiers.items())
                )
                if missing_authority:
                    raise ValueError(
                        f"{label} uses GRIB2 local-use identifier(s) {rendered}; "
                        "selector must bind complete Section 1 table authority "
                        f"(missing {missing_authority})"
                    )
                if value["local_table_version"] == 255:
                    raise ValueError(
                        f"{label} uses GRIB2 local-use identifier(s) {rendered} "
                        "with local_table_version=255 (no local table)"
                    )
    return value


def _optional_number_overlap(
    left: Mapping[str, object],
    right: Mapping[str, object],
    key: str,
) -> bool:
    if key not in left or key not in right:
        return True
    return math.isclose(
        float(left[key]),
        float(right[key]),
        rel_tol=2e-9,
        abs_tol=2e-9,
    )


def _optional_exact_overlap(
    left: Mapping[str, object],
    right: Mapping[str, object],
    key: str,
) -> bool:
    return key not in left or key not in right or left[key] == right[key]


def _grib_selectors_overlap(
    left: Mapping[str, object],
    right: Mapping[str, object],
    source_format: str,
) -> bool:
    """Return whether one runtime GRIB record can satisfy both selectors."""

    required = (
        ("parameter",)
        if source_format == "grib1"
        else ("discipline", "category", "parameter")
    )
    if any(left[key] != right[key] for key in required):
        return False
    optional_exact = (
        ("table_version", "center", "level_type")
        if source_format == "grib1"
        else ("level_type", "member", *_GRIB2_AUTHORITY_KEYS)
    )
    if any(
        not _optional_exact_overlap(left, right, key) for key in optional_exact
    ):
        return False
    if not _optional_number_overlap(left, right, "level_value"):
        return False
    if source_format == "grib1":
        return True
    left_second = "second_level_type" in left
    right_second = "second_level_type" in right
    if left_second != right_second:
        return False
    if not left_second:
        return True
    return left["second_level_type"] == right["second_level_type"] and math.isclose(
        float(left["second_level_value"]),
        float(right["second_level_value"]),
        rel_tol=2e-9,
        abs_tol=2e-9,
    )


def load_mapping(
    path: str | Path,
    *,
    _raw: object | None = None,
) -> dict[str, object]:
    """Read and independently validate the executable subset of the schema.

    RW-WPS performs the full user-facing validation.  The engine repeats the
    structural and target-contract checks that affect array materialization so
    a mapping cannot become less safe when the frontend and engine are invoked
    separately.
    """

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = _load_json_document(path, "mapping") if _raw is None else _raw
    _finite_json(raw)
    mapping = _object(
        raw,
        "mapping",
        allowed={"schema", "name", "format", "coordinates", "fields", "derivations", "target"},
        required={"schema", "name", "format", "coordinates", "fields", "target"},
    )
    if mapping["schema"] != MAPPING_SCHEMA:
        raise ValueError(
            f"unsupported mapping schema {mapping['schema']!r}; expected {MAPPING_SCHEMA!r}"
        )
    _string(mapping["name"], "mapping.name")
    source_format = mapping["format"]
    if source_format not in _FORMATS:
        raise ValueError(f"unsupported mapping format {source_format!r}")

    coordinates = _object(
        mapping["coordinates"],
        "mapping.coordinates",
        allowed={"horizontal", "vertical", "time", "member"},
        required={"horizontal", "vertical", "time"},
    )
    horizontal = _object(
        coordinates["horizontal"],
        "mapping.coordinates.horizontal",
        allowed={"kind", "latitude", "longitude"},
        required={"kind"},
    )
    if horizontal["kind"] == "embedded_grid":
        if source_format == "netcdf":
            raise ValueError("NetCDF mappings require latitude/longitude selectors")
    elif horizontal["kind"] == "variables":
        for name in ("latitude", "longitude"):
            if name not in horizontal:
                raise ValueError(f"horizontal coordinates are missing {name}")
            _validate_selector(horizontal[name], source_format, f"horizontal.{name}")
    else:
        raise ValueError(f"unsupported horizontal coordinate kind {horizontal['kind']!r}")

    vertical = _object(
        coordinates["vertical"],
        "mapping.coordinates.vertical",
        allowed={
            "kind", "selector", "units", "positive", "levels",
            "hybrid_a_field", "hybrid_b_field", "surface_pressure_field",
        },
        required={"kind", "units"},
    )
    if vertical["kind"] not in _VERTICAL_KINDS:
        raise ValueError(f"unsupported vertical kind {vertical['kind']!r}")
    _string(vertical["units"], "vertical.units")
    if vertical.get("positive") not in {None, "up", "down"}:
        raise ValueError("vertical.positive must be 'up' or 'down'")
    if "selector" in vertical and vertical["selector"] is not None:
        _validate_selector(vertical["selector"], source_format, "vertical.selector")
    if source_format == "netcdf" and vertical.get("selector") is None:
        raise ValueError("NetCDF mappings require a vertical coordinate selector")
    levels = vertical.get("levels", [])
    if not isinstance(levels, list):
        raise ValueError("vertical.levels must be a unique numeric list")
    numeric_levels = [
        _number(value, f"vertical.levels[{index}]")
        for index, value in enumerate(levels)
    ]
    if len(set(numeric_levels)) != len(numeric_levels):
        raise ValueError("vertical.levels must be a unique numeric list")
    if vertical["kind"] == "hybrid_sigma_pressure":
        missing = [
            name for name in (
                "hybrid_a_field", "hybrid_b_field", "surface_pressure_field"
            ) if not vertical.get(name)
        ]
        if missing:
            raise ValueError(f"hybrid vertical coordinate is incomplete: {missing}")
        for name in (
            "hybrid_a_field", "hybrid_b_field", "surface_pressure_field"
        ):
            _string(vertical[name], f"vertical.{name}")

    time = _object(
        coordinates["time"],
        "mapping.coordinates.time",
        allowed={"kind", "selector", "units", "calendar"},
        required={"kind"},
    )
    if time["kind"] == "dimension":
        if source_format != "netcdf":
            raise ValueError("dimension time coordinates are only supported for NetCDF")
        for name in ("selector", "units"):
            if name not in time:
                raise ValueError(f"time dimension is missing {name}")
        _validate_dimension_selector(time["selector"], "time.selector")
        _string(time["units"], "time.units")
        if time.get("calendar") is not None:
            _string(time["calendar"], "time.calendar")
    elif time["kind"] != "embedded_metadata":
        raise ValueError(f"unsupported time coordinate kind {time['kind']!r}")
    if source_format == "netcdf" and time["kind"] != "dimension":
        raise ValueError("NetCDF mappings require a dimension time coordinate")

    member = coordinates.get("member")
    if member is not None:
        member = _object(
            member,
            "mapping.coordinates.member",
            allowed={"kind", "selector"},
            required={"kind"},
        )
        if member["kind"] == "dimension":
            if source_format != "netcdf" or "selector" not in member:
                raise ValueError("member dimension requires NetCDF and a selector")
            _validate_dimension_selector(member["selector"], "member.selector")
        elif member["kind"] != "embedded_metadata":
            raise ValueError(f"unsupported member coordinate kind {member['kind']!r}")
        if source_format == "netcdf" and member["kind"] != "dimension":
            raise ValueError("NetCDF member coordinates must use a dimension")

    fields = mapping["fields"]
    if not isinstance(fields, dict) or not fields:
        raise ValueError("mapping.fields must be a non-empty object")
    for name, raw_field in fields.items():
        _string(name, "field name")
        field = _object(
            raw_field,
            f"fields.{name}",
            allowed={
                "selectors", "derivation", "units", "source_axes", "target_axes",
                "location", "staggering", "missing", "selector_stack_axis",
            },
            required={"units", "source_axes", "target_axes", "location", "missing"},
        )
        selectors = field.get("selectors", [])
        if not isinstance(selectors, list):
            raise TypeError(f"fields.{name}.selectors must be a list")
        direct = bool(selectors)
        derived = isinstance(field.get("derivation"), str) and bool(field["derivation"])
        if direct == derived:
            raise ValueError(
                f"fields.{name} must declare selectors or one derivation, exclusively"
            )
        for index, selector in enumerate(selectors):
            _validate_selector(selector, source_format, f"fields.{name}.selectors[{index}]")
        selector_stack_axis = field.get("selector_stack_axis")
        if selector_stack_axis is not None:
            if source_format != "netcdf" or not direct:
                raise ValueError(
                    f"fields.{name}.selector_stack_axis requires a direct NetCDF field"
                )
            selector_stack_axis = _string(
                selector_stack_axis, f"fields.{name}.selector_stack_axis"
            )
            if selector_stack_axis != "soil":
                raise ValueError(
                    f"fields.{name}.selector_stack_axis currently supports only 'soil'"
                )
            if len(selectors) < 2:
                raise ValueError(
                    f"fields.{name}.selector_stack_axis requires multiple selectors"
                )
        unit = _object(
            field["units"], f"fields.{name}.units",
            allowed={"source", "target", "scale", "offset"},
            required={"source", "target"},
        )
        _string(unit["source"], f"fields.{name}.units.source")
        _string(unit["target"], f"fields.{name}.units.target")
        _number(unit.get("scale", 1.0), f"fields.{name}.units.scale")
        _number(unit.get("offset", 0.0), f"fields.{name}.units.offset")
        source_axes = _axes(field["source_axes"], f"fields.{name}.source_axes")
        _axes(field["target_axes"], f"fields.{name}.target_axes")
        if selector_stack_axis is not None and selector_stack_axis not in source_axes:
            raise ValueError(
                f"fields.{name}.selector_stack_axis is absent from source_axes"
            )
        if field["location"] not in _LOCATIONS:
            raise ValueError(f"fields.{name} has unsupported location")
        if field.get("staggering", "none") not in _STAGGERING:
            raise ValueError(f"fields.{name} has unsupported staggering")
        missing = _object(
            field["missing"], f"fields.{name}.missing",
            allowed={"kind", "name", "value"}, required={"kind"},
        )
        if missing["kind"] == "attribute":
            if source_format != "netcdf" or not missing.get("name"):
                raise ValueError(
                    f"fields.{name} attribute missing policy requires NetCDF attribute name"
                )
        elif missing["kind"] == "value":
            _number(
                missing.get("value"),
                f"fields.{name} value missing policy",
            )
        elif missing["kind"] == "preserve_mask":
            if field["location"] != "soil":
                raise ValueError(
                    f"fields.{name} preserve_mask is currently restricted to "
                    "soil fields repaired by the land/water-aware initializer"
                )
        elif missing["kind"] != "reject":
            raise ValueError(f"fields.{name} has unsupported missing policy")

    if source_format in {"grib1", "grib2"}:
        direct_selectors: list[
            tuple[str, int, Mapping[str, object]]
        ] = []
        for field_name, field in fields.items():
            for index, selector in enumerate(field.get("selectors", [])):
                for previous_field, previous_index, previous in direct_selectors:
                    if _grib_selectors_overlap(
                        previous,
                        selector,
                        str(source_format),
                    ):
                        raise ValueError(
                            f"fields.{field_name}.selectors[{index}] overlaps "
                            f"fields.{previous_field}.selectors[{previous_index}]; "
                            "one GRIB record cannot directly provide two mapped "
                            "selector slots"
                        )
                direct_selectors.append((field_name, index, selector))
    else:
        netcdf_selectors: dict[str, tuple[str, int]] = {}
        for field_name, field in fields.items():
            for index, selector in enumerate(field.get("selectors", [])):
                identity = json.dumps(
                    selector,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                previous = netcdf_selectors.get(identity)
                if previous is not None:
                    raise ValueError(
                        f"fields.{field_name}.selectors[{index}] duplicates "
                        f"fields.{previous[0]}.selectors[{previous[1]}]; derive "
                        "aliases explicitly"
                    )
                netcdf_selectors[identity] = (field_name, index)

    derivations = mapping.get("derivations", [])
    if not isinstance(derivations, list):
        raise TypeError("mapping.derivations must be a list")
    derivation_names: set[str] = set()
    derivation_dependencies: dict[str, tuple[str, ...]] = {}
    for index, raw_derivation in enumerate(derivations):
        derivation = _object(
            raw_derivation,
            f"derivations[{index}]",
            allowed={
                "name", "operation", "source", "u", "v", "relative_humidity",
                "temperature", "pressure", "dewpoint", "geopotential", "gravity_m_s2",
            },
            required={"name", "operation"},
        )
        name = _string(derivation["name"], f"derivations[{index}].name")
        if name in derivation_names:
            raise ValueError(f"duplicate derivation {name!r}")
        derivation_names.add(name)
        operation = _string(
            derivation["operation"], f"derivations[{index}].operation"
        )
        try:
            required_arguments, optional_arguments = _DERIVATION_ARGUMENTS[operation]
        except KeyError as error:
            raise ValueError(
                f"derivations[{index}] has unsupported operation {operation!r}"
            ) from error
        supplied_arguments = set(derivation) - {"name", "operation"}
        missing_arguments = sorted(required_arguments - supplied_arguments)
        extra_arguments = sorted(
            supplied_arguments - required_arguments - optional_arguments
        )
        if missing_arguments or extra_arguments:
            raise ValueError(
                f"derivations[{index}] arguments disagree with {operation}: "
                f"missing={missing_arguments}, extra={extra_arguments}"
            )
        dependencies = []
        for argument in sorted(required_arguments):
            dependency = _string(
                derivation[argument], f"derivations[{index}].{argument}"
            )
            dependencies.append(dependency)
        if "gravity_m_s2" in derivation:
            gravity = derivation["gravity_m_s2"]
            try:
                gravity_value = _number(
                    gravity,
                    f"derivations[{index}].gravity_m_s2",
                )
            except ValueError as error:
                raise ValueError(
                    f"derivations[{index}].gravity_m_s2 must be finite and positive"
                ) from error
            if gravity_value <= 0.0:
                raise ValueError(
                    f"derivations[{index}].gravity_m_s2 must be finite and positive"
                )
        if operation == "pressure_from_vertical_coordinate" \
                and vertical["kind"] != "pressure":
            raise ValueError(
                "pressure_from_vertical_coordinate requires a pressure vertical coordinate"
            )
        derivation_dependencies[name] = tuple(dependencies)

    field_derivations: dict[str, str] = {}
    for field_name, field in fields.items():
        derivation_name = field.get("derivation")
        if derivation_name is None:
            continue
        if derivation_name not in derivation_names:
            raise ValueError(
                f"field {field_name!r} names unknown derivation {derivation_name!r}"
            )
        field_derivations[field_name] = str(derivation_name)
    for derivation_name, dependencies in derivation_dependencies.items():
        missing_dependencies = sorted(set(dependencies) - set(fields))
        if missing_dependencies:
            raise ValueError(
                f"derivation {derivation_name!r} depends on unmapped fields "
                f"{missing_dependencies}"
            )

    complete: set[str] = set()

    def visit(field_name: str, active: list[str]) -> None:
        if field_name in active:
            start = active.index(field_name)
            cycle = active[start:] + [field_name]
            raise ValueError("derived fields form a cycle: " + " -> ".join(cycle))
        if field_name in complete or field_name not in field_derivations:
            return
        active.append(field_name)
        derivation_name = field_derivations[field_name]
        for dependency in derivation_dependencies[derivation_name]:
            visit(dependency, active)
        active.pop()
        complete.add(field_name)

    for field_name in field_derivations:
        visit(field_name, [])

    target = _object(
        mapping["target"],
        "mapping.target",
        allowed={
            "name", "physics_suite", "max_dom", "require_lateral_boundaries",
            "target_vertical_levels", "soil_layer_count", "boundary_interval_seconds",
            "required_fields", "pressure_requirement", "policy_controlled_fields",
            "initialization_policies",
        },
        required={
            "name", "physics_suite", "max_dom", "require_lateral_boundaries",
            "required_fields",
        },
    )
    _string(target["name"], "target.name")
    _string(target["physics_suite"], "target.physics_suite")
    if not isinstance(target["max_dom"], int) or isinstance(target["max_dom"], bool) \
            or target["max_dom"] <= 0:
        raise ValueError("target.max_dom must be a positive integer")
    if not isinstance(target["require_lateral_boundaries"], bool):
        raise ValueError("target.require_lateral_boundaries must be boolean")
    for key in ("target_vertical_levels", "soil_layer_count"):
        if target.get(key) is None:
            raise ValueError(f"target.{key} is required")
        _integer(target[key], f"target.{key}", minimum=1)
    boundary_interval = target.get("boundary_interval_seconds")
    if target["require_lateral_boundaries"]:
        if boundary_interval is None:
            raise ValueError(
                "target.boundary_interval_seconds is required for lateral boundaries"
            )
        _integer(
            boundary_interval, "target.boundary_interval_seconds", minimum=1
        )
    elif boundary_interval is not None:
        _integer(
            boundary_interval, "target.boundary_interval_seconds", minimum=1
        )
    requirements = target["required_fields"]
    if not isinstance(requirements, list):
        raise TypeError("target.required_fields must be a list")
    requirement_names: set[str] = set()
    requirement_contracts: dict[str, tuple[tuple[str, ...], str, str]] = {}
    for index, raw_requirement in enumerate(requirements):
        requirement = _object(
            raw_requirement,
            f"target.required_fields[{index}]",
            allowed={"name", "axes", "location", "target_units"},
            required={"name", "axes", "location", "target_units"},
        )
        field_name = _string(requirement["name"], "requirement.name")
        if field_name in requirement_names:
            raise ValueError(f"target repeats required field {field_name!r}")
        requirement_names.add(field_name)
        if field_name not in fields:
            raise ValueError(f"target requires unmapped field {field_name!r}")
        requirement_axes = _axes(
            requirement["axes"], f"target.required_fields[{index}].axes"
        )
        if requirement["location"] not in _LOCATIONS:
            raise ValueError(f"target requirement {field_name!r} has unsupported location")
        requirement_units = _string(
            requirement["target_units"],
            f"target.required_fields[{index}].target_units",
        )
        requirement_contracts[field_name] = (
            requirement_axes, str(requirement["location"]), requirement_units
        )
        mapped = fields[field_name]
        if requirement_axes != tuple(mapped["target_axes"]):
            raise ValueError(f"target axes disagree for {field_name}")
        if requirement["location"] != mapped["location"]:
            raise ValueError(f"target location disagrees for {field_name}")
        if requirement["target_units"] != mapped["units"]["target"]:
            raise ValueError(f"target units disagree for {field_name}")
    for field_name, expected in _CANONICAL_REQUIREMENTS.items():
        if requirement_contracts.get(field_name) != expected:
            raise ValueError(
                f"target canonical requirement {field_name!r} must be "
                f"axes={expected[0]}, location={expected[1]!r}, units={expected[2]!r}"
            )
    pressure_requirement = target.get(
        "pressure_requirement", "air_pressure_or_hybrid_coordinate"
    )
    if pressure_requirement not in {
        "air_pressure", "hybrid_coordinate", "air_pressure_or_hybrid_coordinate",
    }:
        raise ValueError(f"unsupported target.pressure_requirement {pressure_requirement!r}")
    has_pressure = "air_pressure" in fields
    has_hybrid = vertical["kind"] == "hybrid_sigma_pressure"
    pressure_satisfied = {
        "air_pressure": has_pressure,
        "hybrid_coordinate": has_hybrid,
        "air_pressure_or_hybrid_coordinate": has_pressure or has_hybrid,
    }[pressure_requirement]
    if not pressure_satisfied:
        raise ValueError(
            f"target pressure requirement {pressure_requirement!r} is not satisfied"
        )
    controlled = target.get("policy_controlled_fields", [])
    policies = target.get("initialization_policies", {})
    if not isinstance(controlled, list) or not isinstance(policies, dict):
        raise TypeError("target policy fields/policies have invalid types")
    if len(set(controlled)) != len(controlled):
        raise ValueError("target policy_controlled_fields contains duplicates")
    omitted = _POLICY_FIELDS - set(controlled)
    if omitted:
        raise ValueError(f"target omits canonical policy-controlled fields {sorted(omitted)}")
    for name in controlled:
        _string(name, "target policy-controlled field")
        if name not in fields and not (
            isinstance(policies.get(name), str) and policies[name].strip()
        ):
            raise ValueError(f"absent policy-controlled field {name!r} has no policy")
    for name, policy in policies.items():
        _string(name, "target initialization policy field")
        _string(policy, f"target.initialization_policies.{name}")
    return mapping


def _validate_dimension_selector(value: object, label: str) -> dict[str, object]:
    selector = _object(
        value, label, allowed={"name", "standard_name"}
    )
    if not any(isinstance(selector.get(key), str) and selector[key]
               for key in ("name", "standard_name")):
        raise ValueError(f"{label} needs name and/or standard_name")
    return selector


@dataclass(frozen=True)
class CanonicalField:
    name: str
    units: str
    axes: tuple[str, ...]
    location: str
    staggering: str
    values: np.ndarray
    missing_count: int
    source_references: tuple[str, ...]

    def __post_init__(self) -> None:
        array = np.asarray(self.values, dtype=np.float64)
        if array.ndim != len(self.axes):
            raise ValueError(
                f"{self.name} rank {array.ndim} differs from axes {self.axes}"
            )
        if np.isinf(array).any():
            raise ValueError(f"{self.name} contains infinity")
        if int(np.isnan(array).sum()) != self.missing_count:
            raise ValueError(f"{self.name} missing count does not match its data")
        copied = array.copy()
        copied.setflags(write=False)
        object.__setattr__(self, "values", copied)


@dataclass(frozen=True)
class MappedSourceFrame:
    valid_time: datetime
    member: str | None
    source_cycle: datetime
    latitude: np.ndarray
    longitude: np.ndarray
    vertical_kind: str
    vertical_units: str
    vertical_values: np.ndarray
    fields: Mapping[str, CanonicalField]
    mapping_sha256: str
    input_sha256: Mapping[str, str]
    grid_fingerprint: str
    header: SourceFrameHeader

    def __post_init__(self) -> None:
        if self.valid_time.tzinfo is not None or self.source_cycle.tzinfo is not None:
            raise ValueError("mapped source times must be naive UTC")
        if self.valid_time < self.source_cycle:
            raise ValueError("mapped valid time precedes source cycle")
        latitude = np.asarray(self.latitude, dtype=np.float64)
        longitude = np.asarray(self.longitude, dtype=np.float64)
        vertical = np.asarray(self.vertical_values, dtype=np.float64)
        for name, axis in (("latitude", latitude), ("longitude", longitude),
                           ("vertical", vertical)):
            if axis.ndim != 1 or axis.size == 0 or not np.isfinite(axis).all():
                raise ValueError(f"mapped {name} coordinate must be finite non-empty 1-D")
        if latitude.size < 2 or longitude.size < 2:
            raise ValueError("mapped horizontal grid must be at least 2x2")
        for name, axis in (("latitude", latitude), ("longitude", longitude)):
            difference = np.diff(axis)
            if not (np.all(difference > 0.0) or np.all(difference < 0.0)):
                raise ValueError(f"mapped {name} coordinate is not strictly monotonic")
            if not np.allclose(difference, difference[0], rtol=1e-10, atol=1e-12):
                raise ValueError(
                    f"mapped {name} coordinate is not a regular axis; "
                    "curvilinear export is not enabled"
                )
        copied_fields = MappingProxyType(dict(self.fields))
        expected_horizontal = (latitude.size, longitude.size)
        for field in copied_fields.values():
            shape = dict(zip(field.axes, field.values.shape))
            if shape.get("y") != expected_horizontal[0] or shape.get("x") != expected_horizontal[1]:
                raise ValueError(f"{field.name} does not share the source horizontal grid")
            if "vertical" in shape and shape["vertical"] != vertical.size:
                raise ValueError(f"{field.name} does not share the vertical coordinate")
        for name, value in (("latitude", latitude), ("longitude", longitude),
                            ("vertical_values", vertical)):
            copied = value.copy()
            copied.setflags(write=False)
            object.__setattr__(self, name, copied)
        object.__setattr__(self, "fields", copied_fields)
        object.__setattr__(self, "input_sha256", MappingProxyType(dict(self.input_sha256)))
        validate_source_frame(self.header)


@dataclass(frozen=True)
class _DirectValue:
    name: str
    valid_time: datetime
    member: str | None
    source_cycle: datetime
    axes: tuple[str, ...]
    values: np.ndarray
    missing_count: int
    references: tuple[str, ...]


@dataclass(frozen=True)
class _DecodedCollection:
    latitude: np.ndarray
    longitude: np.ndarray
    vertical_values: np.ndarray
    direct: Mapping[tuple[datetime, str | None, str], _DirectValue]
    source_cycles: Mapping[tuple[datetime, str | None], datetime]
    grid_fingerprint: str


def _matching_nc_variables(
    dataset: netCDF4.Dataset, selector: Mapping[str, object]
) -> list[object]:
    candidates = []
    expected_name = selector.get("name")
    expected_standard = selector.get("standard_name")
    for name, variable in dataset.variables.items():
        if expected_name is not None and name != expected_name:
            continue
        if expected_standard is not None \
                and getattr(variable, "standard_name", None) != expected_standard:
            continue
        candidates.append(variable)
    return candidates


def _resolve_nc_variable(dataset: netCDF4.Dataset, selector: Mapping[str, object], label: str):
    candidates = _matching_nc_variables(dataset, selector)
    if len(candidates) != 1:
        raise ValueError(
            f"{label} selector resolved {len(candidates)} NetCDF variables; expected exactly one"
        )
    return candidates[0]


def _nc_coordinate_dimension(variable: object, label: str) -> str:
    dimensions = tuple(variable.dimensions)
    if len(dimensions) != 1:
        raise ValueError(f"{label} coordinate must use exactly one NetCDF dimension")
    return str(dimensions[0])


def _resolve_nc_dimension(dataset: netCDF4.Dataset, selector: Mapping[str, object], label: str):
    name = selector.get("name")
    standard = selector.get("standard_name")
    candidates = []
    for dimension_name in dataset.dimensions:
        variable = dataset.variables.get(dimension_name)
        if name is not None and dimension_name != name:
            continue
        if standard is not None and (
            variable is None or getattr(variable, "standard_name", None) != standard
        ):
            continue
        candidates.append((dimension_name, variable))
    if len(candidates) != 1 or candidates[0][1] is None:
        raise ValueError(
            f"{label} selector resolved {len(candidates)} coordinate dimensions; expected one"
        )
    return candidates[0]


def _utc_datetime(value: object, label: str) -> datetime:
    if isinstance(value, datetime):
        result = value
    else:
        try:
            result = datetime(
                int(value.year), int(value.month), int(value.day),
                int(value.hour), int(value.minute), int(value.second),
            )
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError(f"{label} is not a Gregorian UTC datetime") from error
    if result.tzinfo is not None:
        result = result.astimezone(timezone.utc).replace(tzinfo=None)
    return result


def _nc_coordinate_values(variable, *, expected_units: str | None, label: str) -> np.ndarray:
    if expected_units is not None and getattr(variable, "units", None) != expected_units:
        raise ValueError(
            f"{label} units {getattr(variable, 'units', None)!r} differ from "
            f"mapping {expected_units!r}"
        )
    value = np.ma.asarray(variable[:])
    if np.ma.getmaskarray(value).any():
        raise ValueError(f"{label} coordinate contains missing values")
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 1 or result.size == 0 or not np.isfinite(result).all():
        raise ValueError(f"{label} coordinate must be finite non-empty 1-D")
    return result


def _read_nc_values(variable, policy: Mapping[str, object], field_name: str) -> tuple[np.ndarray, int]:
    raw = np.ma.asarray(variable[:])
    mask = np.ma.getmaskarray(raw).copy()
    data = np.asarray(raw.filled(np.nan), dtype=np.float64)
    if policy["kind"] == "attribute":
        attribute = str(policy["name"])
        if attribute not in variable.ncattrs():
            raise ValueError(f"{field_name} lacks declared missing attribute {attribute!r}")
        marker = float(variable.getncattr(attribute))
        if not math.isfinite(marker):
            raise ValueError(f"{field_name} missing attribute is non-finite")
        mask |= data == marker
    mask |= ~np.isfinite(data)
    if policy["kind"] == "reject" and mask.any():
        raise ValueError(f"{field_name} contains missing/non-finite source values")
    if policy["kind"] == "value":
        data[mask] = float(policy["value"])
        mask[...] = False
    else:
        data[mask] = np.nan
    if np.isinf(data).any():
        raise ValueError(f"{field_name} contains infinity")
    return data, int(mask.sum())


def _unit_transform(values: np.ndarray, field: Mapping[str, object], name: str) -> np.ndarray:
    unit = field["units"]
    scale = float(unit.get("scale", 1.0))
    offset = float(unit.get("offset", 0.0))
    converted = np.asarray(values, dtype=np.float64) * scale + offset
    finite = np.isfinite(values)
    if not np.isfinite(converted[finite]).all():
        raise ValueError(f"unit conversion for {name} produced non-finite data")
    return converted


def _transpose_to_target(
    values: np.ndarray,
    source_axes: Sequence[str],
    target_axes: Sequence[str],
    name: str,
) -> np.ndarray:
    source = tuple(source_axes)
    target = tuple(target_axes)
    if set(source) != set(target):
        raise ValueError(
            f"{name} cannot map source axes {source} to target axes {target} without "
            "an explicit stacking dimension"
        )
    return np.transpose(values, tuple(source.index(axis) for axis in target))


def _decode_netcdf(mapping: Mapping[str, object], files: Sequence[Path]) -> _DecodedCollection:
    coordinates = mapping["coordinates"]
    horizontal = coordinates["horizontal"]
    vertical = coordinates["vertical"]
    time_contract = coordinates["time"]
    member_contract = coordinates.get("member")
    direct: dict[tuple[datetime, str | None, str], _DirectValue] = {}
    cycles: dict[tuple[datetime, str | None], datetime] = {}
    reference_latitude = reference_longitude = reference_vertical = None
    grid_fingerprint = None

    for source in files:
        with netCDF4.Dataset(source) as dataset:
            latitude_variable = _resolve_nc_variable(
                dataset, horizontal["latitude"], "latitude"
            )
            longitude_variable = _resolve_nc_variable(
                dataset, horizontal["longitude"], "longitude"
            )
            latitude = _nc_coordinate_values(
                latitude_variable, expected_units=None, label="latitude"
            )
            longitude = _nc_coordinate_values(
                longitude_variable, expected_units=None, label="longitude"
            )
            vertical_variable = _resolve_nc_variable(
                dataset, vertical["selector"], "vertical"
            )
            vertical_values = _nc_coordinate_values(
                vertical_variable,
                expected_units=str(vertical["units"]),
                label="vertical",
            )
            latitude_dimension = _nc_coordinate_dimension(
                latitude_variable, "latitude"
            )
            longitude_dimension = _nc_coordinate_dimension(
                longitude_variable, "longitude"
            )
            vertical_dimension = _nc_coordinate_dimension(
                vertical_variable, "vertical"
            )
            if len({latitude_dimension, longitude_dimension, vertical_dimension}) != 3:
                raise ValueError(
                    "NetCDF latitude, longitude, and vertical coordinates must use "
                    "distinct dimensions"
                )
            explicit_levels = np.asarray(vertical.get("levels", []), dtype=np.float64)
            if explicit_levels.size and not np.array_equal(explicit_levels, vertical_values):
                raise ValueError("NetCDF vertical coordinate differs from explicit mapping levels")

            time_dimension, time_variable = _resolve_nc_dimension(
                dataset, time_contract["selector"], "time"
            )
            declared_time_units = str(time_contract["units"])
            if getattr(time_variable, "units", None) != declared_time_units:
                raise ValueError("NetCDF time units differ from mapping")
            calendar = str(time_contract.get("calendar") or getattr(time_variable, "calendar", "standard"))
            if calendar not in {"standard", "gregorian", "proleptic_gregorian"}:
                raise ValueError(f"calendar {calendar!r} is not supported for WRF initialization")
            raw_times = netCDF4.num2date(
                np.asarray(time_variable[:]), declared_time_units, calendar=calendar,
                only_use_cftime_datetimes=False,
            )
            times = tuple(_utc_datetime(value, "time coordinate") for value in np.ravel(raw_times))
            if len(set(times)) != len(times):
                raise ValueError(f"NetCDF file {source} contains duplicate valid times")

            member_dimension = None
            member_value = None
            if member_contract is not None:
                if member_contract["kind"] != "dimension":
                    raise ValueError("NetCDF member coordinate must be a dimension")
                member_dimension, member_variable = _resolve_nc_dimension(
                    dataset, member_contract["selector"], "member"
                )
                member_values = np.asarray(member_variable[:]).reshape(-1)
                if member_values.size != 1:
                    raise ValueError(
                        "mapped WRF initialization requires exactly one NetCDF ensemble member"
                    )
                member_value = str(member_values[0])

            current_fingerprint = hashlib.sha256(
                np.ascontiguousarray(latitude).tobytes()
                + np.ascontiguousarray(longitude).tobytes()
                + np.ascontiguousarray(vertical_values).tobytes()
            ).hexdigest()
            if reference_latitude is None:
                reference_latitude = latitude
                reference_longitude = longitude
                reference_vertical = vertical_values
                grid_fingerprint = current_fingerprint
            elif not (
                np.array_equal(reference_latitude, latitude)
                and np.array_equal(reference_longitude, longitude)
                and np.array_equal(reference_vertical, vertical_values)
            ):
                raise ValueError("NetCDF source grid/vertical coordinates change between files")

            soil_dimension = None
            claimed_direct_variables: dict[str, str] = {}
            for field_name, field in mapping["fields"].items():
                if field.get("derivation") is not None:
                    continue
                selectors = field.get("selectors", [])
                stack_axis = field.get("selector_stack_axis")
                resolved = []
                for selector_index, selector in enumerate(selectors):
                    candidates = _matching_nc_variables(dataset, selector)
                    if len(candidates) > 1:
                        raise ValueError(
                            f"{field_name} selector resolves multiple NetCDF variables"
                        )
                    if not candidates:
                        if stack_axis is not None:
                            raise ValueError(
                                f"{field_name} stacked selector {selector_index} has no "
                                f"matching variable in {source}"
                            )
                        continue
                    variable = candidates[0]
                    if variable.name in {item.name for item in resolved}:
                        if stack_axis is not None:
                            raise ValueError(
                                f"{field_name} stacked selectors resolve duplicate "
                                f"variable {variable.name!r}"
                            )
                        continue
                    resolved.append(variable)
                if not resolved:
                    raise ValueError(f"{field_name} has no matching variable in {source}")
                if stack_axis is None and len(resolved) != 1:
                    raise ValueError(
                        f"{field_name} resolves multiple NetCDF variables; rw-wps.mapping.v1 "
                        "does not yet declare how alternatives become one field"
                    )
                if stack_axis is not None and len(resolved) != len(selectors):
                    raise ValueError(
                        f"{field_name} stacked selector inventory is incomplete"
                    )
                for variable in resolved:
                    previous = claimed_direct_variables.get(variable.name)
                    if previous is not None and previous != field_name:
                        raise ValueError(
                            f"NetCDF variable {variable.name!r} directly provides "
                            f"both {previous!r} and {field_name!r}; derive aliases "
                            "explicitly"
                        )
                    claimed_direct_variables[variable.name] = field_name
                declared_source_units = str(field["units"]["source"])
                source_axes = list(_axes(field["source_axes"], f"fields.{field_name}.source_axes"))
                variable_axes = list(source_axes)
                if stack_axis is not None:
                    variable_axes.remove(str(stack_axis))
                expected_dimensions = {
                    "vertical": vertical_dimension,
                    "y": latitude_dimension,
                    "x": longitude_dimension,
                }
                arrays = []
                reference_dimensions = None
                reference_shape = None
                for variable in resolved:
                    if getattr(variable, "units", None) != declared_source_units:
                        raise ValueError(
                            f"{field_name} source units "
                            f"{getattr(variable, 'units', None)!r} differ from mapping "
                            f"{declared_source_units!r}"
                        )
                    data, _ = _read_nc_values(
                        variable, field["missing"], field_name
                    )
                    if len(variable_axes) != data.ndim:
                        raise ValueError(
                            f"{field_name} source_axes rank differs from NetCDF "
                            f"variable {variable.name}"
                        )
                    for axis_role, dimension_name in expected_dimensions.items():
                        if axis_role not in variable_axes:
                            continue
                        axis = variable_axes.index(axis_role)
                        if variable.dimensions[axis] != dimension_name:
                            raise ValueError(
                                f"{field_name} {axis_role} axis does not use the declared "
                                f"coordinate dimension {dimension_name!r}"
                            )
                    if "time" in variable_axes:
                        time_axis = variable_axes.index("time")
                        if variable.dimensions[time_axis] != time_dimension:
                            raise ValueError(
                                f"{field_name} time axis does not use the declared dimension"
                            )
                    elif len(times) != 1:
                        raise ValueError(
                            f"static field {field_name} is ambiguous across a "
                            "multi-time NetCDF file"
                        )
                    if "member" in variable_axes:
                        if member_dimension is None:
                            raise ValueError(
                                f"{field_name} has a member axis without a member coordinate"
                            )
                        member_axis = variable_axes.index("member")
                        if variable.dimensions[member_axis] != member_dimension:
                            raise ValueError(
                                f"{field_name} member axis uses the wrong dimension"
                            )
                    dimensions = tuple(variable.dimensions)
                    if reference_dimensions is None:
                        reference_dimensions = dimensions
                        reference_shape = data.shape
                    elif dimensions != reference_dimensions or data.shape != reference_shape:
                        raise ValueError(
                            f"{field_name} stacked NetCDF variables have different shapes "
                            "or dimensions"
                        )
                    arrays.append(data)
                if stack_axis is not None:
                    expected_soil_count = mapping["target"].get("soil_layer_count")
                    if expected_soil_count is None:
                        raise ValueError(
                            f"{field_name} stacks soil without target.soil_layer_count"
                        )
                    if len(arrays) != int(expected_soil_count):
                        raise ValueError(
                            f"{field_name} has {len(arrays)} stacked soil selectors; "
                            f"target declares {expected_soil_count}"
                        )
                    current_soil_dimension = "@selector_stack:soil"
                    if soil_dimension is None:
                        soil_dimension = current_soil_dimension
                    elif soil_dimension != current_soil_dimension:
                        raise ValueError(
                            "mapped NetCDF soil fields do not share one soil dimension"
                        )
                    data = np.stack(arrays, axis=source_axes.index(str(stack_axis)))
                else:
                    data = arrays[0]
                    if "soil" in source_axes:
                        soil_axis = source_axes.index("soil")
                        current_soil_dimension = str(
                            resolved[0].dimensions[soil_axis]
                        )
                        expected_soil_count = mapping["target"].get(
                            "soil_layer_count"
                        )
                        if expected_soil_count is None:
                            raise ValueError(
                                f"{field_name} has a soil axis without "
                                "target.soil_layer_count"
                            )
                        observed_soil_count = len(
                            dataset.dimensions[current_soil_dimension]
                        )
                        if observed_soil_count != int(expected_soil_count):
                            raise ValueError(
                                f"{field_name} soil dimension has "
                                f"{observed_soil_count} layers; target declares "
                                f"{expected_soil_count}"
                            )
                        if soil_dimension is None:
                            soil_dimension = current_soil_dimension
                        elif soil_dimension != current_soil_dimension:
                            raise ValueError(
                                "mapped NetCDF soil fields do not share one soil dimension"
                            )

                for time_index, valid_time in enumerate(times):
                    selected = data
                    selected_axes = list(source_axes)
                    if "time" in selected_axes:
                        axis = selected_axes.index("time")
                        selected = np.take(selected, time_index, axis=axis)
                        selected_axes.pop(axis)
                    if "member" in selected_axes:
                        axis = selected_axes.index("member")
                        selected = np.take(selected, 0, axis=axis)
                        selected_axes.pop(axis)
                    converted = _unit_transform(selected, field, field_name)
                    target_axes = _axes(field["target_axes"], f"fields.{field_name}.target_axes")
                    converted = _transpose_to_target(
                        converted, selected_axes, target_axes, field_name
                    )
                    missing_count = int(np.isnan(converted).sum())
                    key = (valid_time, member_value, field_name)
                    value = _DirectValue(
                        name=field_name,
                        valid_time=valid_time,
                        member=member_value,
                        source_cycle=valid_time,
                        axes=target_axes,
                        values=converted,
                        missing_count=missing_count,
                        references=tuple(
                            f"{source.resolve()}:{variable.name}"
                            for variable in resolved
                        ),
                    )
                    if key in direct:
                        raise ValueError(f"duplicate mapped field {field_name} at {valid_time}")
                    direct[key] = value
                    cycles[(valid_time, member_value)] = valid_time

    if reference_latitude is None or grid_fingerprint is None:
        raise ValueError("no NetCDF source data were decoded")
    return _DecodedCollection(
        reference_latitude,
        reference_longitude,
        reference_vertical,
        MappingProxyType(direct),
        MappingProxyType(cycles),
        grid_fingerprint,
    )


@dataclass(frozen=True)
class _GribRecord:
    source: Path
    index: int
    reference_time: datetime
    valid_time: datetime
    member: str | None
    parameter: int
    level_type: int
    level_value: float
    table_version: int | None
    center: int | None
    subcenter: int | None
    master_table_version: int | None
    local_table_version: int | None
    discipline: int | None
    category: int | None
    second_level_type: int | None
    second_level_value: float | None
    process_identity: tuple[int, int] | None
    time_semantics: tuple[int, ...]
    values: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray
    grid_fingerprint: str


def _embedded_valid_time(
    reference: datetime, unit: int, amount: int, *, edition: int
) -> datetime:
    if edition == 1:
        factors = {
            0: timedelta(minutes=1), 1: timedelta(hours=1),
            2: timedelta(days=1), 10: timedelta(hours=3),
            11: timedelta(hours=6), 12: timedelta(hours=12),
            254: timedelta(seconds=1),
        }
    else:
        factors = {
            0: timedelta(minutes=1), 1: timedelta(hours=1),
            2: timedelta(days=1), 10: timedelta(hours=3),
            11: timedelta(hours=6), 12: timedelta(hours=12),
            13: timedelta(seconds=1),
        }
    if unit not in factors:
        raise ValueError(f"unsupported GRIB{edition} forecast time unit {unit}")
    return reference + amount * factors[unit]


def _grib1_records(source: Path, bridge: Path) -> tuple[_GribRecord, ...]:
    inspect_grib1_envelopes(source)
    with tempfile.TemporaryDirectory(prefix="gpuwm-mapped-grib1-") as temporary:
        dump = Path(temporary) / "dump"
        completed = subprocess.run(
            [os.fspath(bridge), os.fspath(source), os.fspath(dump)],
            text=True, capture_output=True, check=False,
        )
        if completed.returncode:
            detail = (completed.stderr or completed.stdout).strip()
            raise RuntimeError(f"GRIB1 bridge failed for {source}: {detail}")
        metadata = _load_json_document(dump / "metadata.json", "GRIB1 metadata")
        if not isinstance(metadata, dict):
            raise ValueError("GRIB1 bridge metadata must be a JSON object")
        if metadata.get("format_version") != 1 or metadata.get("edition") != 1:
            raise ValueError("GRIB1 bridge emitted an unsupported metadata schema")
        if metadata.get("dtype") != "<f8":
            raise ValueError("GRIB1 bridge emitted an unsupported dtype")
        values = np.fromfile(dump / "values.f64", dtype="<f8")
        latitude = np.asarray(metadata.get("latitude"), dtype=np.float64)
        longitude = np.asarray(metadata.get("longitude"), dtype=np.float64)
        shape = tuple(int(item) for item in metadata.get("shape", ()))
        if shape != (latitude.size, longitude.size):
            raise ValueError("GRIB1 bridge coordinate/grid shape mismatch")
        fingerprint = hashlib.sha256(
            np.ascontiguousarray(latitude).tobytes()
            + np.ascontiguousarray(longitude).tobytes()
        ).hexdigest()
        result = []
        for index, message in enumerate(metadata.get("messages", [])):
            count = int(message["count"])
            offset = int(message["offset_values"])
            message_shape = (int(message["ny"]), int(message["nx"]))
            if message_shape != shape:
                # Small control records are valid in concatenated CDO output,
                # but they can never satisfy a meteorological mapping field.
                continue
            if int(message["scan_mode"]) & 0x20:
                raise ValueError("GRIB1 j-consecutive scanning is unsupported")
            if count != shape[0] * shape[1] or offset < 0 or offset + count > values.size:
                raise ValueError("GRIB1 bridge message points outside its payload")
            array = np.asarray(values[offset:offset + count], dtype=np.float64).reshape(shape).copy()
            reference = datetime(
                int(message["year"]), int(message["month"]), int(message["day"]),
                int(message["hour"]), int(message["minute"]),
            )
            valid = _embedded_valid_time(
                reference, int(message["time_unit"]), int(message["p1"]), edition=1
            )
            result.append(_GribRecord(
                source=source.resolve(), index=index,
                reference_time=reference, valid_time=valid, member=None,
                parameter=int(message["parameter"]),
                level_type=int(message["level_type"]),
                level_value=float(message["level"]),
                table_version=int(message["table_version"]),
                center=int(message["center"]), subcenter=None,
                master_table_version=None, local_table_version=None,
                discipline=None, category=None,
                second_level_type=None, second_level_value=None,
                process_identity=None,
                time_semantics=(
                    int(message["time_range_indicator"]),
                    int(message["p1"]), int(message["p2"]),
                ),
                values=array,
                latitude=latitude, longitude=longitude,
                grid_fingerprint=fingerprint,
            ))
        return tuple(result)


def _parse_optional_int(value: str) -> int | None:
    return None if value == "-" else int(value)


def _build_grib2_tools() -> tuple[Path, Path]:
    crate = Path(__file__).resolve().parents[1] / "tools" / "grib1_bridge"
    command = [
        "cargo", "build", "--locked", "--offline", "--release",
        "--bin", "grib2_inventory", "--bin", "grib2_dump",
    ]
    completed = subprocess.run(
        command, cwd=crate, text=True, capture_output=True, check=False
    )
    if completed.returncode:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(f"failed to build vendored GRIB2 tools: {detail}")
    suffix = ".exe" if os.name == "nt" else ""
    inventory = crate / "target" / "release" / f"grib2_inventory{suffix}"
    dump = crate / "target" / "release" / f"grib2_dump{suffix}"
    if not inventory.is_file() or not dump.is_file():
        raise RuntimeError("cargo succeeded but the GRIB2 tools are missing")
    return inventory, dump


_GRIB2_INVENTORY_REQUIRED_COLUMNS = frozenset({
    "index", "discipline", "category", "parameter",
    *_GRIB2_AUTHORITY_KEYS,
    "reference_time", "forecast_unit", "forecast_time", "pdt",
    "level_type", "level_value", "second_level_type", "second_level_value",
    "member", "generating_process", "forecast_generating_process_id",
    "gdt", "nx", "ny", "lat1", "lon1", "dx", "dy", "latin1",
    "latin2", "lov", "scan_mode", "shape_of_earth", "resolution_flags",
    "drt", "bitmap",
})
_GRIB2_DUMP_REQUIRED_COLUMNS = frozenset({
    "index", "discipline", "category", "parameter",
    *_GRIB2_AUTHORITY_KEYS,
    "level_type", "level_value", "second_level_type", "second_level_value",
    "member", "pdt", "drt", "nx", "ny", "scan_mode", "bitmap",
    "count", "finite", "missing", "minimum", "maximum", "filename",
})
_GRIB2_INVENTORY_DUMP_PARITY_COLUMNS = (
    "discipline", "category", "parameter", *_GRIB2_AUTHORITY_KEYS,
    "level_type", "level_value", "second_level_type", "second_level_value",
    "member", "pdt", "drt", "nx", "ny", "scan_mode", "bitmap",
)


def _parse_grib2_tsv(
    lines: Sequence[str], *, required: frozenset[str], label: str
) -> list[dict[str, str]]:
    reader = csv.DictReader(lines, delimiter="\t")
    fields = reader.fieldnames or []
    if len(fields) != len(set(fields)):
        raise ValueError(f"{label} contains duplicate columns")
    missing = sorted(required - set(fields))
    if missing:
        raise ValueError(f"{label} is missing required columns {missing}")
    rows = list(reader)
    if not rows:
        raise ValueError(f"{label} contains no records")
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"{label} contains a malformed row")
    indices = [int(row["index"]) for row in rows]
    if len(indices) != len(set(indices)):
        raise ValueError(f"{label} contains duplicate field indices")
    for row in rows:
        for key, maximum in (
            ("center", 65535),
            ("subcenter", 65535),
            ("master_table_version", 255),
            ("local_table_version", 255),
        ):
            value = int(row[key])
            if not 0 <= value <= maximum:
                raise ValueError(
                    f"{label} field {row['index']} has invalid {key}={value}"
                )
    return rows


def _require_grib2_inventory_dump_parity(
    inventory: Mapping[str, str], dump: Mapping[str, str]
) -> None:
    mismatches = [
        key for key in _GRIB2_INVENTORY_DUMP_PARITY_COLUMNS
        if inventory[key] != dump[key]
    ]
    if mismatches:
        raise ValueError(
            f"GRIB2 dump metadata differs from inventory for field "
            f"{inventory['index']} in columns {mismatches}"
        )


def _grib2_inventory(source: Path, executable: Path) -> list[dict[str, str]]:
    completed = subprocess.run(
        [os.fspath(executable), os.fspath(source)],
        text=True, capture_output=True, check=False,
    )
    if completed.returncode:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(f"GRIB2 inventory failed for {source}: {detail}")
    rows = [line for line in completed.stdout.splitlines() if not line.startswith("#")]
    return _parse_grib2_tsv(
        rows,
        required=_GRIB2_INVENTORY_REQUIRED_COLUMNS,
        label="GRIB2 inventory",
    )


def _grib2_records(
    source: Path,
    inventory_executable: Path,
    dump_executable: Path,
    wanted_indices: set[int],
) -> tuple[_GribRecord, ...]:
    inventory = _grib2_inventory(source, inventory_executable)
    selected = [row for row in inventory if int(row["index"]) in wanted_indices]
    if not selected:
        return ()
    with tempfile.TemporaryDirectory(prefix="gpuwm-mapped-grib2-") as temporary:
        dump = Path(temporary) / "dump"
        command = [
            os.fspath(dump_executable), os.fspath(source), os.fspath(dump),
            *(str(int(row["index"])) for row in selected),
        ]
        completed = subprocess.run(
            command, text=True, capture_output=True, check=False
        )
        if completed.returncode:
            detail = (completed.stderr or completed.stdout).strip()
            raise RuntimeError(f"GRIB2 dump failed for {source}: {detail}")
        dump_rows = _parse_grib2_tsv(
            (dump / "metadata.tsv").read_text(encoding="utf-8").splitlines(),
            required=_GRIB2_DUMP_REQUIRED_COLUMNS,
            label="GRIB2 dump metadata",
        )
        dumped = {int(row["index"]): row for row in dump_rows}
        result = []
        for row in selected:
            index = int(row["index"])
            detail = dumped.get(index)
            if detail is None:
                raise ValueError(f"GRIB2 dump omitted selected field {index}")
            _require_grib2_inventory_dump_parity(row, detail)
            gdt = int(row["gdt"])
            scan = int(row["scan_mode"], 16)
            if gdt != 0 or scan != 0x40:
                raise ValueError(
                    "generic GRIB2 frame export currently requires regular "
                    "latitude/longitude GDT 0 with scan mode 0x40"
                )
            nx = int(row["nx"])
            ny = int(row["ny"])
            latitude = float(row["lat1"]) + np.arange(ny, dtype=np.float64) * float(row["dy"])
            longitude_raw = float(row["lon1"]) + np.arange(nx, dtype=np.float64) * float(row["dx"])
            longitude_wrapped = (longitude_raw + 180.0) % 360.0 - 180.0
            longitude_order = np.argsort(longitude_wrapped)
            longitude = longitude_wrapped[longitude_order]
            values = np.fromfile(dump / detail["filename"], dtype="<f8")
            if values.size != nx * ny:
                raise ValueError(f"GRIB2 field {index} decoded count differs from grid")
            values = values.reshape(ny, nx)[:, longitude_order].copy()
            second_type = int(row["second_level_type"])
            second_value = float(row["second_level_value"])
            reference = datetime.fromisoformat(row["reference_time"])
            valid = _embedded_valid_time(
                reference, int(row["forecast_unit"]), int(row["forecast_time"]), edition=2
            )
            fingerprint_payload = {
                key: row[key] for key in (
                    "gdt", "nx", "ny", "lat1", "lon1", "dx", "dy",
                    "latin1", "latin2", "lov", "scan_mode", "shape_of_earth",
                    "resolution_flags",
                )
            }
            fingerprint = hashlib.sha256(
                json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            result.append(_GribRecord(
                source=source.resolve(), index=index,
                reference_time=reference, valid_time=valid,
                member=None if row["member"] == "-" else row["member"],
                parameter=int(row["parameter"]),
                level_type=int(row["level_type"]),
                level_value=float(row["level_value"]),
                table_version=None, center=int(row["center"]),
                subcenter=int(row["subcenter"]),
                master_table_version=int(row["master_table_version"]),
                local_table_version=int(row["local_table_version"]),
                discipline=int(row["discipline"]), category=int(row["category"]),
                second_level_type=second_type, second_level_value=second_value,
                process_identity=(
                    int(row["generating_process"]),
                    int(row["forecast_generating_process_id"]),
                ),
                time_semantics=(int(row["pdt"]),),
                values=values, latitude=latitude, longitude=longitude,
                grid_fingerprint=fingerprint,
            ))
        return tuple(result)


def _selector_matches_record(
    selector: Mapping[str, object], record: _GribRecord, source_format: str
) -> bool:
    if source_format == "grib1":
        checks = (
            record.parameter == int(selector["parameter"]),
            selector.get("table_version") is None
            or record.table_version == int(selector["table_version"]),
            selector.get("center") is None or record.center == int(selector["center"]),
            selector.get("level_type") is None
            or record.level_type == int(selector["level_type"]),
            selector.get("level_value") is None
            or math.isclose(record.level_value, float(selector["level_value"]), abs_tol=1e-9),
        )
    else:
        checks = (
            record.discipline == int(selector["discipline"]),
            record.category == int(selector["category"]),
            record.parameter == int(selector["parameter"]),
            selector.get("center") is None
            or record.center == int(selector["center"]),
            selector.get("subcenter") is None
            or record.subcenter == int(selector["subcenter"]),
            selector.get("master_table_version") is None
            or record.master_table_version
            == int(selector["master_table_version"]),
            selector.get("local_table_version") is None
            or record.local_table_version
            == int(selector["local_table_version"]),
            selector.get("level_type") is None
            or record.level_type == int(selector["level_type"]),
            selector.get("level_value") is None
            or math.isclose(record.level_value, float(selector["level_value"]), abs_tol=1e-9),
            (
                record.second_level_type in {None, 255}
                and selector.get("second_level_type") is None
            ) or (
                selector.get("second_level_type") is not None
                and record.second_level_value is not None
                and record.second_level_type == int(selector["second_level_type"])
                and math.isclose(
                    float(record.second_level_value),
                    float(selector["second_level_value"]),
                    abs_tol=1e-9,
                )
            ),
            selector.get("member") is None
            or record.member == str(selector["member"]),
        )
    return all(checks)


def _grib2_wanted_indices(
    mapping: Mapping[str, object], rows: Sequence[Mapping[str, str]]
) -> set[int]:
    wanted = set()
    for row in rows:
        # Lightweight selector record; data/coordinates are irrelevant here.
        record = _GribRecord(
            source=Path("."), index=int(row["index"]),
            reference_time=datetime.min, valid_time=datetime.min,
            member=None if row["member"] == "-" else row["member"],
            parameter=int(row["parameter"]), level_type=int(row["level_type"]),
            level_value=float(row["level_value"]), table_version=None,
            center=int(row["center"]), subcenter=int(row["subcenter"]),
            master_table_version=int(row["master_table_version"]),
            local_table_version=int(row["local_table_version"]),
            discipline=int(row["discipline"]), category=int(row["category"]),
            second_level_type=int(row["second_level_type"]),
            second_level_value=float(row["second_level_value"]),
            process_identity=None, time_semantics=(int(row["pdt"]),),
            values=np.empty((0, 0)),
            latitude=np.empty(0), longitude=np.empty(0), grid_fingerprint="",
        )
        for field in mapping["fields"].values():
            if any(_selector_matches_record(selector, record, "grib2")
                   for selector in field.get("selectors", [])):
                wanted.add(record.index)
                break
    return wanted


def _assemble_grib(
    mapping: Mapping[str, object], records: Sequence[_GribRecord]
) -> _DecodedCollection:
    if not records:
        raise ValueError("no mapped GRIB records were decoded")
    source_format = str(mapping["format"])
    matched: dict[tuple[datetime, str | None, str], list[_GribRecord]] = {}
    for field_name, field in mapping["fields"].items():
        if field.get("derivation") is not None:
            continue
        selectors = field.get("selectors", [])
        for record in records:
            if any(_selector_matches_record(selector, record, source_format)
                   for selector in selectors):
                matched.setdefault((record.valid_time, record.member, field_name), []).append(record)

    if not matched:
        raise ValueError("no GRIB messages match the mapping selectors")
    selected_records = [record for group in matched.values() for record in group]
    grid_fingerprints = {record.grid_fingerprint for record in selected_records}
    if len(grid_fingerprints) != 1:
        raise ValueError("selected GRIB fields do not share one source grid")
    latitude = selected_records[0].latitude
    longitude = selected_records[0].longitude
    if any(not np.array_equal(record.latitude, latitude)
           or not np.array_equal(record.longitude, longitude)
           for record in selected_records):
        raise ValueError("selected GRIB coordinate axes differ")
    members = {record.member for record in selected_records}
    if len(members) != 1:
        raise ValueError("mapped WRF initialization requires exactly one GRIB member")
    processes_by_time: dict[
        tuple[datetime, str | None], set[tuple[int, int]]
    ] = {}
    for record in selected_records:
        if record.process_identity is not None:
            processes_by_time.setdefault(
                (record.valid_time, record.member), set()
            ).add(record.process_identity)
    mixed_processes = {
        key: identities for key, identities in processes_by_time.items()
        if len(identities) > 1
    }
    if mixed_processes:
        raise ValueError(
            "selected GRIB2 fields mix generating-process identities within "
            f"a valid time: {mixed_processes!r}"
        )
    unsupported_time_semantics = []
    for record in selected_records:
        if source_format == "grib1":
            time_range_indicator, _p1, p2 = record.time_semantics
            supported = time_range_indicator == 0 and p2 == 0
        else:
            supported = record.time_semantics[0] in {0, 1}
        if not supported:
            unsupported_time_semantics.append(
                (record.index, record.time_semantics)
            )
    if unsupported_time_semantics:
        raise ValueError(
            "selected GRIB fields use interval/derived time semantics that "
            "rw-wps.mapping.v1 cannot bind: "
            + repr(unsupported_time_semantics[:8])
        )

    explicit = tuple(float(value) for value in mapping["coordinates"]["vertical"].get("levels", []))
    if explicit:
        vertical_values = np.asarray(explicit, dtype=np.float64)
    else:
        level_sets = []
        for (_time, _member, field_name), group in matched.items():
            if "vertical" in mapping["fields"][field_name]["target_axes"]:
                level_sets.append(tuple(sorted({record.level_value for record in group})))
        if not level_sets or len(set(level_sets)) != 1:
            raise ValueError("GRIB atmospheric fields do not share one complete vertical inventory")
        vertical_values = np.asarray(level_sets[0], dtype=np.float64)

    direct: dict[tuple[datetime, str | None, str], _DirectValue] = {}
    cycles: dict[tuple[datetime, str | None], datetime] = {}
    for key, group in matched.items():
        valid_time, member, field_name = key
        field = mapping["fields"][field_name]
        source_axes = _axes(field["source_axes"], f"fields.{field_name}.source_axes")
        if "time" in source_axes or "member" in source_axes:
            raise ValueError(
                f"GRIB embedded time/member metadata must not also appear in {field_name}.source_axes"
            )
        stacking_axis = None
        stacking_values: Sequence[float] = ()
        if "vertical" in source_axes:
            stacking_axis = "vertical"
            stacking_values = vertical_values
        elif "soil" in source_axes:
            stacking_axis = "soil"
            expected_layers = mapping["target"].get("soil_layer_count")
            if expected_layers is None:
                raise ValueError("soil_layer_count is required to stack GRIB soil fields")
            selectors = field.get("selectors", [])
            if len(selectors) != int(expected_layers):
                raise ValueError(
                    f"{field_name} needs one ordered GRIB selector per soil layer; "
                    f"got {len(selectors)}, expected {expected_layers}"
                )
            if len(group) != int(expected_layers):
                raise ValueError(
                    f"{field_name} has {len(group)} GRIB soil layers; expected {expected_layers}"
                )
            stacking_values = tuple(range(len(group)))

        if stacking_axis is None:
            if len(group) != 1:
                raise ValueError(f"duplicate GRIB messages for scalar field {field_name} at {valid_time}")
            values = group[0].values
            materialized_axes = source_axes
        elif stacking_axis == "vertical":
            by_level: dict[float, _GribRecord] = {}
            for record in group:
                if record.level_value in by_level:
                    raise ValueError(
                        f"duplicate {field_name} GRIB level {record.level_value} at {valid_time}"
                    )
                by_level[record.level_value] = record
            missing = [level for level in stacking_values if level not in by_level]
            extra = [level for level in by_level if level not in set(stacking_values)]
            if missing or extra:
                raise ValueError(
                    f"{field_name} vertical coverage mismatch; missing={missing}, extra={extra}"
                )
            ordered = [by_level[level].values for level in stacking_values]
            axis = source_axes.index(stacking_axis)
            values = np.stack(ordered, axis=axis)
            materialized_axes = source_axes
        else:
            selectors = field["selectors"]
            by_selector: dict[int, _GribRecord] = {}
            for record in group:
                matched_selectors = [
                    index for index, selector in enumerate(selectors)
                    if _selector_matches_record(selector, record, source_format)
                ]
                if len(matched_selectors) != 1:
                    raise ValueError(
                        f"{field_name} GRIB soil record {record.index} matches "
                        f"{len(matched_selectors)} selectors; expected exactly one"
                    )
                selector_index = matched_selectors[0]
                if selector_index in by_selector:
                    raise ValueError(
                        f"duplicate {field_name} GRIB soil selector "
                        f"{selector_index} at {valid_time}"
                    )
                by_selector[selector_index] = record
            missing_selectors = sorted(set(range(len(selectors))) - set(by_selector))
            if missing_selectors:
                raise ValueError(
                    f"{field_name} is missing GRIB soil selectors {missing_selectors}"
                )
            ordered = [by_selector[index].values for index in range(len(selectors))]
            axis = source_axes.index(stacking_axis)
            values = np.stack(ordered, axis=axis)
            materialized_axes = source_axes

        missing_mask = ~np.isfinite(values)
        policy = field["missing"]
        if policy["kind"] == "reject" and missing_mask.any():
            raise ValueError(f"{field_name} contains missing/non-finite GRIB data")
        values = np.asarray(values, dtype=np.float64).copy()
        if policy["kind"] == "value":
            values[missing_mask] = float(policy["value"])
            missing_mask[...] = False
        else:
            values[missing_mask] = np.nan
        values = _unit_transform(values, field, field_name)
        target_axes = _axes(field["target_axes"], f"fields.{field_name}.target_axes")
        values = _transpose_to_target(values, materialized_axes, target_axes, field_name)
        references = tuple(
            f"{record.source}:{record.index}" for record in sorted(group, key=lambda item: item.index)
        )
        cycles_for_group = {record.reference_time for record in group}
        if len(cycles_for_group) != 1:
            raise ValueError(f"{field_name} GRIB records mix source cycles at {valid_time}")
        cycle = cycles_for_group.pop()
        cycle_key = (valid_time, member)
        if cycle_key in cycles and cycles[cycle_key] != cycle:
            raise ValueError(f"mapped GRIB fields mix source cycles at {valid_time}")
        cycles[cycle_key] = cycle
        direct[key] = _DirectValue(
            name=field_name, valid_time=valid_time, member=member,
            source_cycle=cycle, axes=target_axes, values=values,
            missing_count=int(np.isnan(values).sum()), references=references,
        )

    return _DecodedCollection(
        latitude=latitude,
        longitude=longitude,
        vertical_values=vertical_values,
        direct=MappingProxyType(direct),
        source_cycles=MappingProxyType(cycles),
        grid_fingerprint=next(iter(grid_fingerprints)),
    )


def _decode_grib(
    mapping: Mapping[str, object], files: Sequence[Path], *,
    grib1_bridge: Path | None,
    grib2_inventory: Path | None,
    grib2_dump: Path | None,
) -> _DecodedCollection:
    records: list[_GribRecord] = []
    if mapping["format"] == "grib1":
        bridge = Path(grib1_bridge) if grib1_bridge is not None else build_rust_bridge(release=True)
        if not bridge.is_file():
            raise FileNotFoundError(bridge)
        for source in files:
            records.extend(_grib1_records(source, bridge))
    else:
        if grib2_inventory is None or grib2_dump is None:
            inventory_executable, dump_executable = _build_grib2_tools()
        else:
            inventory_executable = Path(grib2_inventory)
            dump_executable = Path(grib2_dump)
        for executable in (inventory_executable, dump_executable):
            if not executable.is_file():
                raise FileNotFoundError(executable)
        for source in files:
            rows = _grib2_inventory(source, inventory_executable)
            wanted = _grib2_wanted_indices(mapping, rows)
            records.extend(_grib2_records(
                source, inventory_executable, dump_executable, wanted
            ))
    return _assemble_grib(mapping, records)


def _resolve_direct_decoder_paths(
    source_format: str,
    *,
    grib1_bridge: str | Path | None,
    grib2_inventory: str | Path | None,
    grib2_dump: str | Path | None,
) -> dict[str, Path]:
    """Resolve the exact decoder paths that a direct decode will execute."""

    if source_format == "netcdf":
        if any(
            value is not None
            for value in (grib1_bridge, grib2_inventory, grib2_dump)
        ):
            raise ValueError("NetCDF mapped input does not use GRIB decoders")
        return {}
    if source_format == "grib1":
        if grib2_inventory is not None or grib2_dump is not None:
            raise ValueError("GRIB1 mapped input does not use GRIB2 decoders")
        bridge = (
            build_rust_bridge(release=True)
            if grib1_bridge is None
            else Path(grib1_bridge)
        ).resolve()
        if not bridge.is_file():
            raise FileNotFoundError(bridge)
        return {"grib1_bridge": bridge}
    if source_format != "grib2":
        raise ValueError(f"unsupported mapped source format {source_format!r}")
    if grib1_bridge is not None:
        raise ValueError("GRIB2 mapped input does not use a GRIB1 decoder")
    if (grib2_inventory is None) != (grib2_dump is None):
        raise ValueError(
            "GRIB2 inventory and dump decoders must be supplied together"
        )
    if grib2_inventory is None:
        inventory, dump = _build_grib2_tools()
    else:
        inventory, dump = Path(grib2_inventory), Path(grib2_dump)
    result = {
        "grib2_inventory": inventory.resolve(),
        "grib2_dump": dump.resolve(),
    }
    for path in result.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    return result


def _derivation_table(mapping: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    return {str(item["name"]): item for item in mapping.get("derivations", [])}


def _specific_humidity_from_rh(
    relative_humidity: np.ndarray,
    temperature: np.ndarray,
    pressure: np.ndarray,
) -> np.ndarray:
    # Reuse gpuwm's WRF-real Bolton relation rather than maintaining another
    # humidity implementation in the generic adapter.
    from gpuwm.ingest.real import _saturation_mixing_ratio

    mixing_ratio = _saturation_mixing_ratio(
        temperature, pressure, relative_humidity
    )
    return mixing_ratio / (1.0 + mixing_ratio)


def _evaluate_derivation(
    operation: Mapping[str, object],
    available: Mapping[str, CanonicalField],
    collection: _DecodedCollection,
    field: Mapping[str, object],
    name: str,
) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...]]:
    kind = str(operation["operation"])

    def dependency(label: str) -> CanonicalField:
        dependency_name = str(operation[label])
        try:
            return available[dependency_name]
        except KeyError as error:
            raise KeyError(dependency_name) from error

    if kind == "copy":
        source = dependency("source")
        raw = source.values
        axes = source.axes
        references = source.source_references
    elif kind == "wind_speed":
        u = dependency("u")
        v = dependency("v")
        if u.axes != v.axes or u.values.shape != v.values.shape:
            raise ValueError(f"{name} wind derivation dependencies disagree")
        raw = np.hypot(u.values, v.values)
        axes = u.axes
        references = (*u.source_references, *v.source_references)
    elif kind == "geopotential_height":
        geopotential = dependency("geopotential")
        gravity = float(operation.get("gravity_m_s2", 9.80665))
        if not math.isfinite(gravity) or gravity <= 0.0:
            raise ValueError(f"{name} declares invalid gravity")
        raw = geopotential.values / gravity
        axes = geopotential.axes
        references = geopotential.source_references
    elif kind == "pressure_from_vertical_coordinate":
        axes = _axes(field["source_axes"], f"fields.{name}.source_axes")
        if axes != ("vertical", "y", "x"):
            raise ValueError(
                f"{name} pressure derivation currently requires source_axes "
                "['vertical','y','x']"
            )
        raw = np.broadcast_to(
            collection.vertical_values[:, None, None],
            (collection.vertical_values.size,
             collection.latitude.size, collection.longitude.size),
        )
        references = ("@coordinate.vertical",)
    elif kind == "relative_humidity_from_dewpoint":
        dewpoint = dependency("dewpoint")
        temperature = dependency("temperature")
        if dewpoint.axes != temperature.axes:
            raise ValueError(f"{name} dewpoint/temperature axes disagree")
        from gpuwm.ingest.real import _surface_relative_humidity

        raw = _surface_relative_humidity(dewpoint.values, temperature.values)
        axes = dewpoint.axes
        references = (*dewpoint.source_references, *temperature.source_references)
    elif kind in {"specific_humidity_from_rh", "specific_humidity_from_dewpoint"}:
        temperature = dependency("temperature")
        pressure = dependency("pressure")
        if kind == "specific_humidity_from_rh":
            humidity = dependency("relative_humidity")
            relative_humidity = humidity.values
            references = (
                *humidity.source_references, *temperature.source_references,
                *pressure.source_references,
            )
            axes = humidity.axes
        else:
            dewpoint = dependency("dewpoint")
            from gpuwm.ingest.real import _surface_relative_humidity

            relative_humidity = _surface_relative_humidity(
                dewpoint.values, temperature.values
            )
            references = (
                *dewpoint.source_references, *temperature.source_references,
                *pressure.source_references,
            )
            axes = dewpoint.axes
        if axes != temperature.axes or axes != pressure.axes:
            raise ValueError(f"{name} humidity derivation dependency axes disagree")
        raw = _specific_humidity_from_rh(
            relative_humidity, temperature.values, pressure.values
        )
    else:
        raise ValueError(f"unsupported derivation operation {kind!r}")

    source_axes = _axes(field["source_axes"], f"fields.{name}.source_axes")
    target_axes = _axes(field["target_axes"], f"fields.{name}.target_axes")
    if tuple(axes) != source_axes:
        raise ValueError(
            f"derived {name} produced axes {tuple(axes)}, expected {source_axes}"
        )
    converted = _unit_transform(raw, field, name)
    converted = _transpose_to_target(converted, source_axes, target_axes, name)
    return converted, target_axes, tuple(dict.fromkeys(references))


def _frame_header(
    mapping: Mapping[str, object],
    *,
    valid_time: datetime,
    source_cycle: datetime,
    latitude: np.ndarray,
    longitude: np.ndarray,
    vertical_values: np.ndarray,
    fields: Mapping[str, CanonicalField],
    source_id: str,
) -> SourceFrameHeader:
    vertical = mapping["coordinates"]["vertical"]
    vertical_kind = str(vertical["kind"])
    source_vertical_name = "atmosphere"
    descriptors: dict[str, VerticalDescriptor] = {
        source_vertical_name: VerticalDescriptor(
            coordinate={
                "hybrid_sigma_pressure": "hybrid",
                "embedded_levels": "model_level",
            }.get(vertical_kind, vertical_kind),
            level_count=int(vertical_values.size),
            level_values=tuple(float(value) for value in vertical_values),
            positive=str(vertical.get("positive", "down")),
            units=str(vertical["units"]),
        )
    }
    soil_fields = [field for field in fields.values() if "soil" in field.axes]
    if soil_fields:
        soil_count = soil_fields[0].values.shape[soil_fields[0].axes.index("soil")]
        descriptors["soil"] = VerticalDescriptor(
            coordinate="soil_depth", level_count=soil_count,
            units="index", positive="down",
        )
    lead = int((valid_time - source_cycle).total_seconds())
    time = TimeDescriptor(
        reference_time=source_cycle.replace(tzinfo=timezone.utc).isoformat(),
        valid_time=valid_time.replace(tzinfo=timezone.utc).isoformat(),
        lead_seconds=lead,
    )
    field_descriptors = []
    for field in fields.values():
        field_descriptors.append(FieldDescriptor(
            canonical_name=field.name,
            units=field.units,
            dimensions=field.axes,
            grid_location=field.location,
            vertical_coordinate=(
                source_vertical_name if "vertical" in field.axes
                else "soil" if "soil" in field.axes else None
            ),
            time=time,
            data_reference="sha256:" + _array_sha256(field.values),
            dtype=field.values.dtype.str,
            shape=field.values.shape,
            missing_value_policy=(
                "explicit_missing" if field.missing_count else "reject_nonfinite"
            ),
            source_field=";".join(field.source_references),
        ))
    return SourceFrameHeader(
        source_id=source_id,
        source_cycle=source_cycle.replace(tzinfo=timezone.utc).isoformat(),
        grid=GridDescriptor(
            projection="regular_latitude_longitude",
            nx=int(longitude.size), ny=int(latitude.size),
            earth_shape="source_metadata_bound",
            scan_order=(
                ("+x" if longitude[-1] > longitude[0] else "-x") + ","
                + ("+y" if latitude[-1] > latitude[0] else "-y")
            ),
            wind_basis="earth_relative",
            parameters={
                "latitude_first": float(latitude[0]),
                "latitude_last": float(latitude[-1]),
                "longitude_first": float(longitude[0]),
                "longitude_last": float(longitude[-1]),
            },
        ),
        vertical_coordinates=descriptors,
        fields=tuple(field_descriptors),
        initialization_policies=dict(
            mapping["target"].get("initialization_policies", {})
        ),
    )


def _materialize_frames(
    mapping: Mapping[str, object],
    collection: _DecodedCollection,
    *,
    mapping_sha256: str,
    input_sha256: Mapping[str, str],
) -> tuple[MappedSourceFrame, ...]:
    keys = sorted(collection.source_cycles, key=lambda item: (item[0], str(item[1])))
    if not keys:
        raise ValueError("mapped source has no valid times")
    members = {member for _time, member in keys}
    if len(members) != 1:
        raise ValueError("mapped WRF initialization requires exactly one member")
    derivations = _derivation_table(mapping)
    frames = []
    required_names = {
        str(item["name"]) for item in mapping["target"]["required_fields"]
    }
    finite_required = required_names - {
        "soil_temperature", "volumetric_soil_moisture",
    }
    for valid_time, member in keys:
        available: dict[str, CanonicalField] = {}
        for (time_value, member_value, field_name), direct in collection.direct.items():
            if time_value != valid_time or member_value != member:
                continue
            field = mapping["fields"][field_name]
            available[field_name] = CanonicalField(
                name=field_name,
                units=str(field["units"]["target"]),
                axes=direct.axes,
                location=str(field["location"]),
                staggering=str(field.get("staggering", "none")),
                values=direct.values,
                missing_count=direct.missing_count,
                source_references=direct.references,
            )
        pending = {
            name for name, field in mapping["fields"].items()
            if field.get("derivation") is not None
        }
        while pending:
            progress = False
            for name in sorted(tuple(pending)):
                field = mapping["fields"][name]
                derivation_name = str(field["derivation"])
                if derivation_name not in derivations:
                    raise ValueError(f"field {name} names unknown derivation {derivation_name!r}")
                operation = derivations[derivation_name]
                try:
                    values, axes, references = _evaluate_derivation(
                        operation, available, collection, field, name
                    )
                except KeyError:
                    continue
                available[name] = CanonicalField(
                    name=name, units=str(field["units"]["target"]), axes=axes,
                    location=str(field["location"]),
                    staggering=str(field.get("staggering", "none")),
                    values=values, missing_count=int(np.isnan(values).sum()),
                    source_references=references,
                )
                pending.remove(name)
                progress = True
            if not progress:
                raise ValueError(
                    "derived fields have missing dependencies or a cycle: "
                    + ", ".join(sorted(pending))
                )
        missing = sorted(required_names - set(available))
        if missing:
            raise ValueError(f"mapped frame at {valid_time} lacks required fields {missing}")
        for name in finite_required:
            if not np.isfinite(available[name].values).all():
                raise ValueError(f"required mapped field {name} is not finite at {valid_time}")
        soil_count = mapping["target"].get("soil_layer_count")
        if soil_count is not None:
            for name in ("soil_temperature", "volumetric_soil_moisture"):
                field = available[name]
                observed = field.values.shape[field.axes.index("soil")]
                if observed != int(soil_count):
                    raise ValueError(
                        f"{name} has {observed} layers, target declares {soil_count}"
                    )
        source_cycle = collection.source_cycles[(valid_time, member)]
        header = _frame_header(
            mapping, valid_time=valid_time, source_cycle=source_cycle,
            latitude=collection.latitude, longitude=collection.longitude,
            vertical_values=collection.vertical_values, fields=available,
            source_id=str(mapping["name"]),
        )
        frames.append(MappedSourceFrame(
            valid_time=valid_time, member=member, source_cycle=source_cycle,
            latitude=collection.latitude, longitude=collection.longitude,
            vertical_kind=str(mapping["coordinates"]["vertical"]["kind"]),
            vertical_units=str(mapping["coordinates"]["vertical"]["units"]),
            vertical_values=collection.vertical_values,
            fields=available, mapping_sha256=mapping_sha256,
            input_sha256=input_sha256,
            grid_fingerprint=collection.grid_fingerprint,
            header=header,
        ))

    times = tuple(frame.valid_time for frame in frames)
    if tuple(sorted(times)) != times or len(set(times)) != len(times):
        raise ValueError("mapped forcing times are not unique and increasing")
    target = mapping["target"]
    if target.get("require_lateral_boundaries"):
        if len(times) < 2:
            raise ValueError("mapped lateral-boundary forcing requires at least two times")
        deltas = {
            int((later - earlier).total_seconds())
            for earlier, later in zip(times, times[1:])
        }
        if len(deltas) != 1 or next(iter(deltas)) <= 0:
            raise ValueError("mapped forcing cadence must be positive and uniform")
        declared = target.get("boundary_interval_seconds")
        if declared is None or int(declared) != next(iter(deltas)):
            raise ValueError(
                f"mapped cadence {next(iter(deltas))} seconds differs from "
                f"target contract {declared!r}"
            )
    inventories = {tuple(frame.fields) for frame in frames}
    if len(inventories) != 1:
        raise ValueError("mapped field inventory changes between valid times")
    return tuple(frames)


def _verify_input_manifest(
    path: Path,
    expected_sha256: str,
    mapping_path: Path,
    files: Sequence[Path],
    *,
    _snapshots: Mapping[Path, _AuthoritySnapshot] | None = None,
    _recheck_snapshots: bool = True,
) -> dict[str, object]:
    path = path.resolve()
    supplied = {
        key.resolve(): value for key, value in (_snapshots or {}).items()
    }
    used: dict[Path, _AuthoritySnapshot] = {}

    def snapshot(actual: Path, *, retain_bytes: bool = False) -> _AuthoritySnapshot:
        actual = actual.resolve()
        value = supplied.get(actual)
        if value is None:
            value = _snapshot_authority(actual, retain_bytes=retain_bytes)
        if value.path != actual or (retain_bytes and value.data is None):
            raise ValueError(f"authority snapshot is incomplete for {actual}")
        used[actual] = value
        return value

    manifest_snapshot = snapshot(path, retain_bytes=True)
    observed = manifest_snapshot.sha256
    if observed != expected_sha256.lower():
        raise ValueError(
            f"mapped input-manifest SHA mismatch: expected {expected_sha256}, got {observed}"
        )
    raw = _load_json_bytes(
        manifest_snapshot.data,
        "mapped input manifest",
        path,
    )
    manifest = _object(
        raw, "input manifest",
        allowed={"schema", "mapping_sha256", "files"},
        required={"schema", "mapping_sha256", "files"},
    )
    if manifest["schema"] != INPUT_MANIFEST_SCHEMA:
        raise ValueError(f"unsupported mapped input manifest schema {manifest['schema']!r}")
    if manifest["mapping_sha256"] != snapshot(mapping_path).sha256:
        raise ValueError("input manifest mapping SHA does not match mapping bytes")
    rows = manifest["files"]
    if not isinstance(rows, list) or len(rows) != len(files):
        raise ValueError("input manifest file inventory differs from request")
    for index, (row, source) in enumerate(zip(rows, files)):
        spec = _object(
            row, f"input manifest files[{index}]",
            allowed={"path", "bytes", "sha256"},
            required={"path", "bytes", "sha256"},
        )
        declared = Path(str(spec["path"]))
        if not declared.is_absolute():
            declared = path.parent / declared
        if declared.resolve() != source.resolve():
            raise ValueError(f"input manifest path differs for file {index}")
        source_snapshot = snapshot(source)
        if (
            isinstance(spec["bytes"], bool)
            or not isinstance(spec["bytes"], int)
            or spec["bytes"] != source_snapshot.size
            or spec["sha256"] != source_snapshot.sha256
        ):
            raise ValueError(f"input manifest identity differs for {source}")
    if _recheck_snapshots:
        for value in used.values():
            _require_authority_snapshot(value)
    return manifest


def decode_mapped_source(
    mapping_path: str | Path,
    files: Sequence[str | Path],
    *,
    input_manifest: str | Path | None = None,
    input_manifest_sha256: str | None = None,
    grib1_bridge: str | Path | None = None,
    grib2_inventory: str | Path | None = None,
    grib2_dump: str | Path | None = None,
) -> tuple[MappedSourceFrame, ...]:
    """Decode arbitrary declared inputs into canonical source frames.

    All persistent authorities are hashed before decode and again afterward.
    The optional manifest is mandatory for the eventual public runner, but is
    optional here so small unit fixtures can exercise the materializer.
    """

    mapping_path = Path(mapping_path).resolve()
    sources = tuple(Path(path).resolve() for path in files)
    if not sources:
        raise ValueError("mapped source requires at least one input file")
    if len(set(sources)) != len(sources):
        raise ValueError("mapped source input list contains duplicates")
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(source)
    if (input_manifest is None) != (input_manifest_sha256 is None):
        raise ValueError("input_manifest and input_manifest_sha256 are an atomic pair")
    mapping_snapshot = _snapshot_authority(mapping_path, retain_bytes=True)
    mapping_raw = _load_json_bytes(
        mapping_snapshot.data,
        "mapping",
        mapping_path,
    )
    mapping = load_mapping(mapping_path, _raw=mapping_raw)
    mapping_digest = mapping_snapshot.sha256
    snapshots = {mapping_path: mapping_snapshot}
    for source in sources:
        snapshots[source] = _snapshot_authority(source)
    decoder_paths = _resolve_direct_decoder_paths(
        str(mapping["format"]),
        grib1_bridge=grib1_bridge,
        grib2_inventory=grib2_inventory,
        grib2_dump=grib2_dump,
    )
    for path in decoder_paths.values():
        snapshots[path] = _snapshot_authority(path)
    before = {
        str(source): snapshots[source].sha256 for source in sources
    }
    if input_manifest is not None:
        manifest_path = Path(input_manifest).resolve()
        snapshots[manifest_path] = _snapshot_authority(
            manifest_path,
            retain_bytes=True,
        )
        _verify_input_manifest(
            manifest_path, str(input_manifest_sha256),
            mapping_path, sources,
            _snapshots=snapshots,
            _recheck_snapshots=False,
        )
    _require_authority_snapshot(mapping_snapshot)
    if input_manifest is not None:
        _require_authority_snapshot(snapshots[manifest_path])
    if mapping["format"] == "netcdf":
        collection = _decode_netcdf(mapping, sources)
    else:
        collection = _decode_grib(
            mapping, sources,
            grib1_bridge=decoder_paths.get("grib1_bridge"),
            grib2_inventory=decoder_paths.get("grib2_inventory"),
            grib2_dump=decoder_paths.get("grib2_dump"),
        )
    frames = _materialize_frames(
        mapping, collection, mapping_sha256=mapping_digest,
        input_sha256=before,
    )
    for snapshot in snapshots.values():
        _require_authority_snapshot(snapshot)
    return frames


def inspect_mapped_source(
    mapping_path: str | Path,
    files: Sequence[str | Path],
    *,
    input_manifest: str | Path | None = None,
    input_manifest_sha256: str | None = None,
    grib1_bridge: str | Path | None = None,
    grib2_inventory: str | Path | None = None,
    grib2_dump: str | Path | None = None,
) -> dict[str, object]:
    """Decode real source bytes and report canonical materialization readiness.

    Inspection never upgrades an adapter's production status.  It is useful
    when a real file proves most selectors but lacks a required invariant or
    complementary product: direct arrays and their hashes are still recorded,
    while the exact materialization failure remains explicit.
    """

    mapping_path = Path(mapping_path).resolve()
    sources = tuple(Path(path).resolve() for path in files)
    if not sources:
        raise ValueError("mapped source inspection requires at least one input file")
    if len(set(sources)) != len(sources):
        raise ValueError("mapped source inspection input list contains duplicates")
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(source)
    if (input_manifest is None) != (input_manifest_sha256 is None):
        raise ValueError("input_manifest and input_manifest_sha256 are an atomic pair")

    mapping_snapshot = _snapshot_authority(mapping_path, retain_bytes=True)
    mapping_raw = _load_json_bytes(
        mapping_snapshot.data,
        "mapping",
        mapping_path,
    )
    mapping = load_mapping(mapping_path, _raw=mapping_raw)
    mapping_digest = mapping_snapshot.sha256
    snapshots = {mapping_path: mapping_snapshot}
    for source in sources:
        snapshots[source] = _snapshot_authority(source)
    decoder_paths = _resolve_direct_decoder_paths(
        str(mapping["format"]),
        grib1_bridge=grib1_bridge,
        grib2_inventory=grib2_inventory,
        grib2_dump=grib2_dump,
    )
    for path in decoder_paths.values():
        snapshots[path] = _snapshot_authority(path)
    before = {
        str(source): snapshots[source].sha256 for source in sources
    }
    if input_manifest is not None:
        manifest_path = Path(input_manifest).resolve()
        snapshots[manifest_path] = _snapshot_authority(
            manifest_path,
            retain_bytes=True,
        )
        _verify_input_manifest(
            manifest_path, str(input_manifest_sha256),
            mapping_path, sources,
            _snapshots=snapshots,
            _recheck_snapshots=False,
        )
    _require_authority_snapshot(mapping_snapshot)
    if input_manifest is not None:
        _require_authority_snapshot(snapshots[manifest_path])
    if mapping["format"] == "netcdf":
        collection = _decode_netcdf(mapping, sources)
    else:
        collection = _decode_grib(
            mapping, sources,
            grib1_bridge=decoder_paths.get("grib1_bridge"),
            grib2_inventory=decoder_paths.get("grib2_inventory"),
            grib2_dump=decoder_paths.get("grib2_dump"),
        )

    direct_field_names = {
        str(name) for name, field in mapping["fields"].items()
        if field.get("derivation") is None
    }
    frame_rows = []
    for valid_time, member in sorted(
        collection.source_cycles, key=lambda item: (item[0], str(item[1]))
    ):
        decoded = {
            field_name: value
            for (time_value, member_value, field_name), value
            in collection.direct.items()
            if time_value == valid_time and member_value == member
        }
        frame_rows.append({
            "valid_time": valid_time.isoformat(),
            "source_cycle": collection.source_cycles[(valid_time, member)].isoformat(),
            "member": member,
            "decoded_direct_fields": sorted(decoded),
            "unresolved_direct_fields": sorted(direct_field_names - set(decoded)),
            "fields": {
                name: {
                    "axes": list(value.axes),
                    "shape": list(value.values.shape),
                    "minimum": (
                        None if not np.isfinite(value.values).any()
                        else float(np.nanmin(value.values))
                    ),
                    "maximum": (
                        None if not np.isfinite(value.values).any()
                        else float(np.nanmax(value.values))
                    ),
                    "missing": value.missing_count,
                    "sha256": _array_sha256(value.values),
                    "source_references": list(value.references),
                }
                for name, value in sorted(decoded.items())
            },
        })

    materialization: dict[str, object]
    try:
        frames = _materialize_frames(
            mapping, collection, mapping_sha256=mapping_digest,
            input_sha256=before,
        )
    except (KeyError, TypeError, ValueError) as error:
        materialization = {
            "verdict": "INCOMPLETE",
            "error_type": type(error).__name__,
            "error": str(error),
        }
        status = "DECODED_INCOMPLETE_NOT_STOCK_WRF_CERTIFIED"
    else:
        materialization = {
            "verdict": "PASS",
            "frame_count": len(frames),
            "frame_header_sha256": [
                hashlib.sha256(
                    json.dumps(
                        frame.header.to_dict(), sort_keys=True,
                        separators=(",", ":"), allow_nan=False,
                    ).encode("utf-8")
                ).hexdigest()
                for frame in frames
            ],
        }
        status = "CANONICAL_FRAMES_MATERIALIZED_NOT_STOCK_WRF_CERTIFIED"

    for snapshot in snapshots.values():
        _require_authority_snapshot(snapshot)
    return {
        "schema": INSPECTION_SCHEMA,
        "status": status,
        "stock_wrf_certified": False,
        "mapping": {"path": str(mapping_path), "sha256": mapping_digest},
        "inputs": [
            {
                "path": str(source), "bytes": source.stat().st_size,
                "sha256": before[str(source)],
            }
            for source in sources
        ],
        "decoders": {
            role: {
                "path": str(path),
                "bytes": snapshots[path].size,
                "sha256": snapshots[path].sha256,
            }
            for role, path in decoder_paths.items()
        },
        "source_format": mapping["format"],
        "grid": {
            "ny": int(collection.latitude.size),
            "nx": int(collection.longitude.size),
            "vertical_count": int(collection.vertical_values.size),
            "fingerprint": collection.grid_fingerprint,
        },
        "frames": frame_rows,
        "materialization": materialization,
    }


def mapped_frames_to_regular_snapshots(
    frames: Sequence[MappedSourceFrame],
) -> tuple[Era5Snapshot, ...]:
    """Translate canonical frames to the existing regular-source join ABI.

    This is naming/packing only.  It does not reimplement interpolation or
    initialization. Soil arrays retain canonical names; the independently
    validated composition contract supplies their depth/remapping semantics.
    """

    frames = tuple(frames)
    if not frames:
        raise ValueError("at least one mapped frame is required")
    unsupported = {
        "cloud_water_mixing_ratio", "rain_water_mixing_ratio",
        "cloud_ice_mixing_ratio", "snow_mixing_ratio",
        "graupel_or_hail_mixing_ratio", "vertical_velocity",
    }
    result = []
    for frame in frames:
        present_unsupported = sorted(unsupported & set(frame.fields))
        if present_unsupported:
            raise ValueError(
                "the current regular-source join cannot inject mapped "
                f"prognostic fields {present_unsupported}"
            )
        canonical = frame.fields
        pressure = canonical["air_pressure"].values
        if not np.isfinite(pressure).all() or np.any(pressure <= 0.0):
            raise ValueError("mapped air pressure must be finite and positive")
        levels_hpa = np.median(pressure, axis=(1, 2)) / 100.0
        legacy_names = {
            "air_temperature": "T",
            "air_pressure": "PRES",
            "specific_humidity": "SPFH",
            "eastward_wind": "U",
            "northward_wind": "V",
            "geopotential_height": "GHT",
            "surface_pressure": "PSFC",
            "terrain_height": "SOURCE_OROGRAPHY",
            "skin_temperature": "SKINTEMP",
            "air_temperature_2m": "T2",
            "specific_humidity_2m": "Q2",
            "eastward_wind_10m": "U10",
            "northward_wind_10m": "V10",
            "land_fraction": "LANDSEA",
            "snow_water_equivalent": "SNOW_EC",
            "snow_depth": "SNOWH",
            "sea_ice_fraction": "SEAICE",
        }
        fields = {
            legacy: np.asarray(canonical[name].values, dtype=np.float64)
            for name, legacy in legacy_names.items() if name in canonical
        }
        soil_t = np.asarray(
            canonical["soil_temperature"].values, dtype=np.float64,
        )
        soil_m = np.asarray(
            canonical["volumetric_soil_moisture"].values, dtype=np.float64,
        )
        if soil_t.ndim != 3 or soil_m.ndim != 3 or soil_t.shape != soil_m.shape:
            raise ValueError(
                "mapped soil temperature/moisture must share soil/y/x shape"
            )
        source_land = np.asarray(
            canonical["land_fraction"].values, dtype=np.float64,
        )
        if source_land.shape != soil_t.shape[1:] \
                or not np.isfinite(source_land).all():
            raise ValueError(
                "mapped land fraction must be finite and share the soil grid"
            )
        # A fraction, and now checked as one.  Finiteness alone let a
        # mis-scaled unit transform deliver 2.0 here, which the >= 0.5
        # threshold below reads as land without complaint; a value that
        # merely kisses 0 or 1 is decode rounding and clamps.
        source_land, _ = admit_bounded(
            source_land, name="land fraction", minimum=0.0, maximum=1.0,
            subject="mapped")
        terrestrial = source_land >= 0.5
        if not np.isfinite(soil_t[:, terrestrial]).all():
            raise ValueError(
                "mapped soil temperature contains missing source-land values"
            )
        if not np.isfinite(soil_m[:, terrestrial]).all():
            raise ValueError(
                "mapped soil moisture contains missing source-land values"
            )
        fields[MAPPED_SOIL_TEMPERATURE] = soil_t
        fields[MAPPED_SOIL_MOISTURE] = soil_m
        result.append(Era5Snapshot(
            valid_time=frame.valid_time,
            levels_hpa=np.asarray(levels_hpa, dtype=np.float64),
            latitude=np.asarray(frame.latitude, dtype=np.float64),
            longitude=np.asarray(frame.longitude, dtype=np.float64),
            fields=fields,
        ))
    return tuple(result)


def mapped_frame_receipt(
    mapping_path: str | Path,
    files: Sequence[str | Path],
    frames: Sequence[MappedSourceFrame],
) -> dict[str, object]:
    """Return a compact immutable-evidence payload for tests and handoffs."""

    mapping_path = Path(mapping_path).resolve()
    sources = tuple(Path(path).resolve() for path in files)
    return {
        "schema": FRAME_EVIDENCE_SCHEMA,
        "status": "DECODED_DECODER_UNBOUND_NOT_STOCK_WRF_CERTIFIED",
        "mapping": {
            "path": str(mapping_path),
            "sha256": _sha256(mapping_path),
        },
        "inputs": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in sources
        ],
        "frames": [
            {
                "valid_time": frame.valid_time.isoformat(),
                "member": frame.member,
                "source_cycle": frame.source_cycle.isoformat(),
                "grid_fingerprint": frame.grid_fingerprint,
                "shape": [int(frame.latitude.size), int(frame.longitude.size)],
                "vertical_count": int(frame.vertical_values.size),
                "fields": {
                    name: {
                        "axes": list(field.axes),
                        "shape": list(field.values.shape),
                        "units": field.units,
                        "missing": field.missing_count,
                        "sha256": _array_sha256(field.values),
                    }
                    for name, field in frame.fields.items()
                },
            }
            for frame in frames
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Inspect mapped source bytes without enabling production export."""

    parser = argparse.ArgumentParser(
        prog="gpuwm-mapped-inspect",
        description=(
            "Decode an rw-wps.mapping.v1 source and print hash-bound "
            "canonical-frame readiness evidence. This does not certify or "
            "write WRF initialization files."
        ),
    )
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument(
        "--input", dest="inputs", type=Path, action="append", required=True
    )
    parser.add_argument("--input-manifest", type=Path)
    parser.add_argument("--input-manifest-sha256")
    parser.add_argument("--grib1-bridge", type=Path)
    parser.add_argument("--grib2-inventory", type=Path)
    parser.add_argument("--grib2-dump", type=Path)
    args = parser.parse_args(argv)
    # The evidence this prints is only as bindable as the tree that
    # produced it, so the tree is named before the evidence.
    import sys

    from gpuwm.provenance_gate import announce_for_main

    refusal = announce_for_main("gpuwm-mapped-inspect")
    if refusal is not None:
        print(f"gpuwm-mapped-inspect: {refusal}", file=sys.stderr)
        return 2
    report = inspect_mapped_source(
        args.mapping, args.inputs,
        input_manifest=args.input_manifest,
        input_manifest_sha256=args.input_manifest_sha256,
        grib1_bridge=args.grib1_bridge,
        grib2_inventory=args.grib2_inventory,
        grib2_dump=args.grib2_dump,
    )
    print(json.dumps(report, sort_keys=True, indent=2, allow_nan=False))
    return 0


__all__ = [
    "CanonicalField", "FRAME_EVIDENCE_SCHEMA", "INPUT_MANIFEST_SCHEMA",
    "INSPECTION_SCHEMA", "MAPPING_SCHEMA", "MappedSourceFrame",
    "decode_mapped_source", "inspect_mapped_source", "load_mapping",
    "mapped_frame_receipt", "mapped_frames_to_regular_snapshots",
]


if __name__ == "__main__":
    raise SystemExit(main())
