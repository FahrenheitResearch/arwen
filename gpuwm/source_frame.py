"""Canonical metadata contract between native decoders and WRF initialization.

The contract carries science semantics that raw GRIB messages do not make
uniform: grid and wind orientation, vertical coordinates, field staggering,
units, valid/interval times, and accumulation reset behavior.  Data arrays may
live in GPU memory or in a hash-bound cache; this module validates their
descriptors without importing NumPy or CuPy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Mapping, Sequence


SCHEMA = "gpuwm-canonical-source-frame-v1"


@dataclass(frozen=True)
class GridDescriptor:
    projection: str
    nx: int
    ny: int
    earth_shape: str
    scan_order: str
    wind_basis: str
    parameters: Mapping[str, float | int | str]


@dataclass(frozen=True)
class VerticalDescriptor:
    coordinate: str
    level_count: int
    level_values: tuple[float, ...] = ()
    a_coefficients: tuple[float, ...] = ()
    b_coefficients: tuple[float, ...] = ()
    positive: str = "down"
    units: str = "Pa"


@dataclass(frozen=True)
class TimeDescriptor:
    reference_time: str
    valid_time: str
    lead_seconds: int
    statistic: str = "instantaneous"
    interval_start: str | None = None
    interval_end: str | None = None
    accumulation_reset: str | None = None


@dataclass(frozen=True)
class FieldDescriptor:
    canonical_name: str
    units: str
    dimensions: tuple[str, ...]
    grid_location: str
    vertical_coordinate: str | None
    time: TimeDescriptor
    data_reference: str
    dtype: str
    shape: tuple[int, ...]
    missing_value_policy: str = "reject_nonfinite"
    source_field: str = ""


@dataclass(frozen=True)
class SourceFrameHeader:
    source_id: str
    source_cycle: str
    grid: GridDescriptor
    vertical_coordinates: Mapping[str, VerticalDescriptor]
    fields: tuple[FieldDescriptor, ...]
    initialization_policies: Mapping[str, str] = field(default_factory=dict)
    schema: str = SCHEMA

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


# Fields used to construct a dynamically meaningful limited-area state.  A
# source may satisfy pressure via an explicit 3-D field or via a declared
# hybrid vertical coordinate.  Optional state must still have an explicit
# initialization policy; it is never silently fabricated.
REQUIRED_3D_FIELDS = frozenset(
    {
        "air_temperature",
        "specific_humidity",
        "eastward_wind",
        "northward_wind",
        "geopotential_height",
    }
)
REQUIRED_SURFACE_FIELDS = frozenset(
    {
        "surface_pressure",
        "terrain_height",
        "skin_temperature",
        "air_temperature_2m",
        "specific_humidity_2m",
        "eastward_wind_10m",
        "northward_wind_10m",
        "land_fraction",
        "soil_temperature",
        "volumetric_soil_moisture",
    }
)
POLICY_CONTROLLED_FIELDS = frozenset(
    {
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
)

_INSTANTANEOUS = {"instantaneous"}
_INTERVAL_STATISTICS = {"accumulation", "average", "maximum", "minimum"}
_SUPPORTED_VERTICAL = {"pressure", "hybrid", "model_level", "soil_depth", "height"}
_SUPPORTED_WIND_BASIS = {"earth_relative", "grid_relative_with_rotation"}


def _parse_utc(text: str, label: str) -> datetime:
    normalized = text.replace("Z", "+00:00")
    try:
        value = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"{label} is not ISO-8601: {text!r}") from exc
    if value.tzinfo is None:
        raise ValueError(f"{label} must include a UTC offset")
    value = value.astimezone(timezone.utc)
    return value


def _validate_vertical(name: str, value: VerticalDescriptor) -> None:
    if value.coordinate not in _SUPPORTED_VERTICAL:
        raise ValueError(f"vertical coordinate {name!r} has unsupported type {value.coordinate!r}")
    if value.level_count <= 0:
        raise ValueError(f"vertical coordinate {name!r} has no levels")
    if value.level_values and len(value.level_values) != value.level_count:
        raise ValueError(f"vertical coordinate {name!r} level_values length mismatch")
    if value.coordinate == "hybrid":
        expected = value.level_count + 1
        if len(value.a_coefficients) != expected or len(value.b_coefficients) != expected:
            raise ValueError(
                f"hybrid coordinate {name!r} requires {expected} A and B interface coefficients"
            )


def _validate_time(value: TimeDescriptor, field_name: str) -> None:
    reference = _parse_utc(value.reference_time, f"{field_name}.reference_time")
    valid = _parse_utc(value.valid_time, f"{field_name}.valid_time")
    if value.lead_seconds < 0:
        raise ValueError(f"{field_name}.lead_seconds must be non-negative")
    if int((valid - reference).total_seconds()) != value.lead_seconds:
        raise ValueError(f"{field_name} valid/reference time does not match lead_seconds")
    if value.statistic in _INSTANTANEOUS:
        if value.interval_start is not None or value.interval_end is not None:
            raise ValueError(f"instantaneous field {field_name} must not declare an interval")
    elif value.statistic in _INTERVAL_STATISTICS:
        if value.interval_start is None or value.interval_end is None:
            raise ValueError(f"interval field {field_name} requires start and end times")
        start = _parse_utc(value.interval_start, f"{field_name}.interval_start")
        end = _parse_utc(value.interval_end, f"{field_name}.interval_end")
        if not start < end <= valid:
            raise ValueError(f"{field_name} interval must satisfy start < end <= valid_time")
        if value.statistic == "accumulation" and not value.accumulation_reset:
            raise ValueError(f"accumulated field {field_name} requires accumulation_reset semantics")
    else:
        raise ValueError(f"{field_name} has unsupported time statistic {value.statistic!r}")


def validate_source_frame(
    header: SourceFrameHeader,
    *,
    require_wrf_initial_state: bool = True,
) -> None:
    """Validate a source frame, raising ``ValueError`` on ambiguity.

    This is a metadata gate.  Array finite/range checks remain the adapter's
    responsibility when it materializes each ``data_reference``.
    """

    if header.schema != SCHEMA:
        raise ValueError(f"unsupported source-frame schema {header.schema!r}")
    if not header.source_id.strip():
        raise ValueError("source_id is empty")
    _parse_utc(header.source_cycle, "source_cycle")
    if header.grid.nx <= 1 or header.grid.ny <= 1:
        raise ValueError("source grid must be at least 2x2")
    if not header.grid.projection or not header.grid.earth_shape or not header.grid.scan_order:
        raise ValueError("projection, earth_shape, and scan_order are required")
    if header.grid.wind_basis not in _SUPPORTED_WIND_BASIS:
        raise ValueError(
            "wind_basis must be earth_relative or grid_relative_with_rotation"
        )

    for name, vertical in header.vertical_coordinates.items():
        _validate_vertical(name, vertical)

    names: set[str] = set()
    for descriptor in header.fields:
        name = descriptor.canonical_name
        if name in names:
            raise ValueError(f"duplicate canonical field {name!r}")
        names.add(name)
        if not descriptor.units or not descriptor.dimensions or not descriptor.data_reference:
            raise ValueError(f"field {name!r} is missing units/dimensions/data reference")
        if len(descriptor.shape) != len(descriptor.dimensions):
            raise ValueError(f"field {name!r} shape/dimensions rank mismatch")
        if any(size <= 0 for size in descriptor.shape):
            raise ValueError(f"field {name!r} has an empty dimension")
        if descriptor.vertical_coordinate not in {None, *header.vertical_coordinates}:
            raise ValueError(f"field {name!r} references an unknown vertical coordinate")
        _validate_time(descriptor.time, name)

    if not require_wrf_initial_state:
        return

    missing = set((REQUIRED_3D_FIELDS | REQUIRED_SURFACE_FIELDS) - names)
    if "air_pressure" not in names and not any(
        vertical.coordinate == "hybrid" for vertical in header.vertical_coordinates.values()
    ):
        missing.add("air_pressure_or_hybrid_coordinate")
    if missing:
        raise ValueError(
            "source frame is incomplete for WRF initialization: " + ", ".join(sorted(missing))
        )

    absent_controlled = POLICY_CONTROLLED_FIELDS - names
    missing_policies = absent_controlled - header.initialization_policies.keys()
    if missing_policies:
        raise ValueError(
            "missing explicit initialization policy for absent fields: "
            + ", ".join(sorted(missing_policies))
        )


def canonical_field_requirements() -> dict[str, Sequence[str]]:
    """Expose required/policy-controlled names for adapters and tooling."""

    return {
        "required_3d": tuple(sorted(REQUIRED_3D_FIELDS)),
        "required_surface_and_soil": tuple(sorted(REQUIRED_SURFACE_FIELDS)),
        "pressure_requirement": ("air_pressure", "hybrid_coordinate"),
        "policy_controlled_if_absent": tuple(sorted(POLICY_CONTROLLED_FIELDS)),
    }
