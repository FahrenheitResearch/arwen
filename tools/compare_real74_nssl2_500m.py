#!/usr/bin/env python3
"""Evidence-grade streaming comparison of matched real74 NSSL-2 runs.

The CPU-WRF and GPUWM directories are treated as immutable evidence.  A
``register`` pass freezes the comparison policy, evaluator source, and exact
CPU output identities before the GPU run starts.  ``compare`` requires that
registration, the GPU launch/completion manifests, and the exact 64-frame
calendar.  Scientific differences never alter the independent structural
PASS gates.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Iterable, Mapping, Sequence

import netCDF4
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from gpuwm.verify.n5s_metrics import fss_distance  # noqa: E402


POLICY_SCHEMA = "gpuwm.real74-nssl2-500m-comparison-policy/v1"
REGISTRATION_SCHEMA = "gpuwm.real74-nssl2-500m-comparison-registration/v1"
REPORT_SCHEMA = "gpuwm.real74-nssl2-500m-comparison-report/v1"
EVIDENCE_SCHEMA = "gpuwm.real74-nssl2-500m-comparison-evidence/v1"
GPU_LAUNCH_SCHEMA = "gpuwm.real74-nssl2-500m-launch/v1"
GPU_COMPLETION_SCHEMA = "gpuwm.real74-nssl2-500m-completion/v1"
START_TIME = datetime(1974, 4, 3, 12, 0, 0)
RUN_SECONDS = 12 * 60 * 60
G = 9.80665
DEFAULT_CHUNK_VALUES = 512 * 1024
# Existing Phase-3 projection oracle authority, documented and pinned in
# tests/test_lambert.py (observed residuals are float32-storage scale).
COORD_ABS_TOL_DEG = 1.0e-3
MAPFAC_REL_TOL = 1.0e-4
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_FRAME_RE = re.compile(
    r"^wrfout_(d0[1-4])_(\d{4}-\d{2}-\d{2})_(\d{2})"
    r"(?:_|:)(\d{2})(?:_|:)(\d{2})$")


class ComparisonError(ValueError):
    """Raised when evidence cannot support the requested comparison."""


@dataclass(frozen=True)
class DomainSpec:
    name: str
    grid_id: int
    parent_id: int
    i_parent_start: int
    j_parent_start: int
    parent_grid_ratio: int
    nx: int
    ny: int
    nz: int
    dx_m: float
    dt_s: float
    cadence_s: int


DOMAINS = (
    DomainSpec("d01", 1, 1, 1, 1, 1, 250, 200, 49, 12000.0, 60.0, 3600),
    DomainSpec("d02", 2, 1, 63, 51, 4, 500, 400, 49, 3000.0, 15.0, 3600),
    DomainSpec("d03", 3, 2, 167, 117, 3, 501, 501, 49, 1000.0, 5.0, 3600),
    DomainSpec("d04", 4, 3, 151, 151, 2, 400, 400, 49, 500.0, 2.5, 1800),
)
DOMAIN_BY_NAME = {domain.name: domain for domain in DOMAINS}


@dataclass(frozen=True)
class FieldSpec:
    canonical: str
    cpu_name: str
    gpu_name: str
    layout: str
    unit_family: str
    nonnegative: bool = False
    inventory_kind: str | None = None


# Live WRF v4.6.1 MP18 headers establish the native NSSL spellings.  The
# legacy/foreign names below are recorded only to reject an accidental
# Morrison/old-tool mapping; they are never CPU authority names.
LEGACY_FOREIGN_CPU_ALIASES = {
    "cloud_droplet_number": "QNCLOUD",
    "graupel_volume": "QVOLG",
    "hail_volume": "QVOLH",
}
FIELD_SPECS = (
    FieldSpec("u_wind", "U", "U", "u3", "speed"),
    FieldSpec("v_wind", "V", "V", "v3", "speed"),
    FieldSpec("vertical_velocity", "W", "W", "w3", "speed"),
    FieldSpec("theta_perturbation", "T", "T", "mass3", "temperature"),
    FieldSpec("pressure_perturbation", "P", "P", "mass3", "pressure"),
    FieldSpec("base_pressure", "PB", "PB", "mass3", "pressure"),
    FieldSpec("geopotential_perturbation", "PH", "PH", "w3", "geopotential"),
    FieldSpec("base_geopotential", "PHB", "PHB", "w3", "geopotential"),
    FieldSpec("column_mass_perturbation", "MU", "MU", "surface", "pressure"),
    FieldSpec("base_column_mass", "MUB", "MUB", "surface", "pressure"),
    FieldSpec("terrain_height", "HGT", "HGT", "surface", "length"),
    FieldSpec("surface_pressure", "PSFC", "PSFC", "surface", "pressure"),
    FieldSpec("two_meter_temperature", "T2", "T2", "surface", "temperature"),
    FieldSpec("water_vapor", "QVAPOR", "QVAPOR", "mass3", "mixing_ratio",
              True, "water_mass"),
    FieldSpec("cloud_water", "QCLOUD", "QCLOUD", "mass3", "mixing_ratio",
              True, "water_mass"),
    FieldSpec("rain_water", "QRAIN", "QRAIN", "mass3", "mixing_ratio",
              True, "water_mass"),
    FieldSpec("cloud_ice", "QICE", "QICE", "mass3", "mixing_ratio",
              True, "water_mass"),
    FieldSpec("snow", "QSNOW", "QSNOW", "mass3", "mixing_ratio",
              True, "water_mass"),
    FieldSpec("graupel", "QGRAUP", "QGRAUP", "mass3", "mixing_ratio",
              True, "water_mass"),
    FieldSpec("hail", "QHAIL", "QHAIL", "mass3", "mixing_ratio",
              True, "water_mass"),
    FieldSpec("cloud_droplet_number", "QNDROP", "QNDROP", "mass3",
              "number", True, "particle_number"),
    FieldSpec("rain_number", "QNRAIN", "QNRAIN", "mass3", "number",
              True, "particle_number"),
    FieldSpec("ice_number", "QNICE", "QNICE", "mass3", "number",
              True, "particle_number"),
    FieldSpec("snow_number", "QNSNOW", "QNSNOW", "mass3", "number",
              True, "particle_number"),
    FieldSpec("graupel_number", "QNGRAUPEL", "QNGRAUPEL", "mass3",
              "number", True, "particle_number"),
    FieldSpec("hail_number", "QNHAIL", "QNHAIL", "mass3", "number",
              True, "particle_number"),
    FieldSpec("ccn_number", "QNCCN", "QNCCN", "mass3", "number",
              True, "particle_number"),
    FieldSpec("graupel_volume", "QVGRAUPEL", "QVGRAUPEL", "mass3", "volume",
              True, "particle_volume"),
    FieldSpec("hail_volume", "QVHAIL", "QVHAIL", "mass3", "volume",
              True, "particle_volume"),
    FieldSpec("gridscale_precipitation_accumulation", "RAINNC", "RAINNC", "surface",
              "precipitation", True, "surface_total_water_mass"),
    FieldSpec("snow_accumulation", "SNOWNC", "SNOWNC", "surface",
              "precipitation", True, "surface_snow_diagnostic_mass"),
    FieldSpec("graupel_accumulation", "GRAUPELNC", "GRAUPELNC", "surface",
              "precipitation", True, "surface_graupel_diagnostic_mass"),
    FieldSpec("hail_accumulation", "HAILNC", "HAILNC", "surface",
              "precipitation", True, "surface_hail_diagnostic_mass"),
)
FIELD_BY_CANONICAL = {field.canonical: field for field in FIELD_SPECS}

_UNIT_KEYS = {
    "speed": {"ms-1", "m/s"},
    "temperature": {"k"},
    "pressure": {"pa"},
    "geopotential": {"m2s-2", "m2/s2"},
    "length": {"m"},
    "mixing_ratio": {"kgkg-1", "kg/kg"},
    "number": {"kg-1", "1/kg"},
    "volume": {"m3kg-1", "m3/kg"},
    "precipitation": {"mm"},
    "reflectivity": {"dbz"},
    "latitude": {"degreenorth", "degreesnorth"},
    "longitude": {"degreeeast", "degreeseast"},
    "map_factor": {"", "1"},
    "eta": {"", "1"},
}
_CANONICAL_UNITS = {
    "speed": "m s-1", "temperature": "K", "pressure": "Pa",
    "geopotential": "m2 s-2", "length": "m",
    "mixing_ratio": "kg kg-1", "number": "kg-1",
    "volume": "m3 kg-1", "precipitation": "mm",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path, *, block_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_bytes):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: Mapping[str, object]) -> None:
    atomic_text(
        path,
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def load_json(path: Path, label: str) -> dict[str, object]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ComparisonError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ComparisonError(f"{label} root must be a JSON object: {path}")
    return payload


def repository_identity(repo: Path = REPOSITORY_ROOT) -> dict[str, object]:
    def git(*arguments: str) -> str:
        try:
            return subprocess.run(
                ["git", *arguments], cwd=repo, check=True,
                capture_output=True, text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            raise ComparisonError(f"cannot bind evaluator repository: {exc}") from exc

    status = git("status", "--porcelain=v1")
    if status:
        raise ComparisonError(
            "comparison evaluator worktree must be clean before registration:\n"
            + status)
    return {
        "commit": git("rev-parse", "HEAD"),
        "tree": git("rev-parse", "HEAD^{tree}"),
        "git_status": "clean",
    }


def _require_outside_repository(path: Path) -> None:
    try:
        path.resolve().relative_to(REPOSITORY_ROOT.resolve())
    except ValueError:
        return
    raise ComparisonError(
        f"comparison evidence must be outside source worktree: {path.resolve()}")


def file_identity(path: Path, *, logical_name: str | None = None
                  ) -> dict[str, object]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ComparisonError(f"required evidence file is missing: {resolved}")
    result: dict[str, object] = {
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }
    if logical_name is not None:
        result["logical_name"] = logical_name
    else:
        result["path"] = str(resolved)
    return result


def expected_calendar() -> dict[str, tuple[datetime, ...]]:
    return {
        domain.name: tuple(
            START_TIME + timedelta(seconds=seconds)
            for seconds in range(0, RUN_SECONDS + 1, domain.cadence_s)
        )
        for domain in DOMAINS
    }


def _parse_frame_name(name: str) -> tuple[str, datetime]:
    match = _FRAME_RE.fullmatch(name)
    if match is None:
        raise ComparisonError(f"malformed wrfout filename: {name}")
    domain, date, hour, minute, second = match.groups()
    try:
        valid = datetime.fromisoformat(f"{date}T{hour}:{minute}:{second}")
    except ValueError as exc:
        raise ComparisonError(f"invalid wrfout timestamp in {name}") from exc
    return domain, valid


def discover_frames(root: Path) -> dict[tuple[str, datetime], Path]:
    root = root.resolve()
    if not root.is_dir():
        raise ComparisonError(f"wrfout root is not a directory: {root}")
    candidates = sorted(path for path in root.iterdir()
                        if path.name.startswith("wrfout_"))
    if not candidates:
        raise ComparisonError(f"no wrfout files found in {root}")
    frames: dict[tuple[str, datetime], Path] = {}
    for path in candidates:
        if not path.is_file():
            raise ComparisonError(f"wrfout candidate is not a file: {path}")
        key = _parse_frame_name(path.name)
        if key in frames:
            raise ComparisonError(
                f"duplicate {key[0]} frame at {key[1].isoformat()}: "
                f"{frames[key]} and {path}")
        frames[key] = path
    expected = {
        (domain, valid)
        for domain, times in expected_calendar().items() for valid in times
    }
    actual = set(frames)
    if actual != expected:
        missing = sorted(f"{d}/{t.isoformat()}" for d, t in expected - actual)
        extra = sorted(f"{d}/{t.isoformat()}" for d, t in actual - expected)
        raise ComparisonError(
            "wrfout calendar is not exact 13/13/13/25 (64 total): "
            f"missing={missing}, extra={extra}")
    return frames


def _unit_key(value: object) -> str:
    text = str(value).strip().lower().replace("**", "").replace("^", "")
    text = re.sub(r"\((-?\d+)\)", r"\1", text)
    text = text.replace("#", "").replace("_", "")
    return re.sub(r"\s+", "", text)


def _require_units(variable, family: str, label: str) -> str:
    raw = getattr(variable, "units", None)
    if raw is None:
        raise ComparisonError(f"{label} has no units attribute")
    key = _unit_key(raw)
    if key not in _UNIT_KEYS[family]:
        raise ComparisonError(
            f"{label} units {raw!r} do not match {family} "
            f"({_UNIT_KEYS[family]})")
    return str(raw)


def _layout_dimensions(layout: str) -> tuple[str, ...]:
    table = {
        "mass3": ("Time", "bottom_top", "south_north", "west_east"),
        "u3": ("Time", "bottom_top", "south_north", "west_east_stag"),
        "v3": ("Time", "bottom_top", "south_north_stag", "west_east"),
        "w3": ("Time", "bottom_top_stag", "south_north", "west_east"),
        "surface": ("Time", "south_north", "west_east"),
    }
    return table[layout]


def _layout_stagger(layout: str) -> str:
    return {"mass3": "", "u3": "X", "v3": "Y", "w3": "Z",
            "surface": ""}[layout]


def _require_variable_schema(variable, *, dimensions: Sequence[str],
                             unit_family: str, stagger: str,
                             label: str) -> dict[str, object]:
    actual_dimensions = tuple(variable.dimensions)
    if actual_dimensions != tuple(dimensions):
        raise ComparisonError(
            f"{label} dimensions {actual_dimensions} != {tuple(dimensions)}")
    actual_stagger = str(getattr(variable, "stagger", "")).strip().upper()
    if actual_stagger != stagger:
        raise ComparisonError(
            f"{label} stagger {actual_stagger!r} != {stagger!r}")
    units = _require_units(variable, unit_family, label)
    return {
        "dimensions": list(actual_dimensions), "stagger": actual_stagger,
        "units": units, "dtype": str(variable.dtype),
    }


def _decoded_time(variable) -> str:
    value = netCDF4.chartostring(np.asarray(variable[:]))
    flattened = np.asarray(value).reshape(-1)
    if flattened.size != 1:
        raise ComparisonError(
            f"Times must contain exactly one record, got {flattened.size}")
    return str(flattened[0])


def _expected_dimensions(domain: DomainSpec) -> dict[str, int]:
    return {
        "Time": 1, "DateStrLen": 19,
        "west_east": domain.nx, "south_north": domain.ny,
        "bottom_top": domain.nz, "west_east_stag": domain.nx + 1,
        "south_north_stag": domain.ny + 1,
        "bottom_top_stag": domain.nz + 1,
    }


def inspect_frame(dataset, *, side: str, domain: DomainSpec,
                  valid_time: datetime) -> dict[str, object]:
    for name, expected in _expected_dimensions(domain).items():
        if name not in dataset.dimensions:
            raise ComparisonError(f"{side}/{domain.name} missing dimension {name}")
        actual = len(dataset.dimensions[name])
        if actual != expected:
            raise ComparisonError(
                f"{side}/{domain.name} dimension {name}={actual}, expected {expected}")
    expected_time = valid_time.strftime("%Y-%m-%d_%H:%M:%S")
    if "Times" not in dataset.variables:
        raise ComparisonError(f"{side}/{domain.name} has no Times variable")
    if _decoded_time(dataset.variables["Times"]) != expected_time:
        raise ComparisonError(
            f"{side}/{domain.name} Times does not equal {expected_time}")
    for name in ("START_DATE", "SIMULATION_START_DATE"):
        if str(getattr(dataset, name, "")) != "1974-04-03_12:00:00":
            raise ComparisonError(
                f"{side}/{domain.name} global {name} is not the 12Z authority")
    if side == "gpu" and int(getattr(dataset, "GPUWM_WRITE_COMPLETE", 0)) != 1:
        raise ComparisonError(
            f"gpu/{domain.name}/{expected_time} lacks GPUWM_WRITE_COMPLETE=1")

    expected_attrs = {
        "GRID_ID": domain.grid_id,
        # WRF encodes the root as its own parent (1); GPUWM's internal
        # experiment graph uses parent 0.  Both are exact native conventions.
        "PARENT_ID": (
            0 if side == "gpu" and domain.grid_id == 1
            else domain.parent_id),
        "I_PARENT_START": domain.i_parent_start,
        "J_PARENT_START": domain.j_parent_start,
        "PARENT_GRID_RATIO": domain.parent_grid_ratio,
    }
    for name, expected in expected_attrs.items():
        if not hasattr(dataset, name) or int(getattr(dataset, name)) != expected:
            raise ComparisonError(
                f"{side}/{domain.name} global {name} must equal {expected}")
    for name, expected in (("DX", domain.dx_m), ("DY", domain.dx_m),
                           ("DT", domain.dt_s)):
        if not hasattr(dataset, name) or float(getattr(dataset, name)) != expected:
            raise ComparisonError(
                f"{side}/{domain.name} global {name} must equal {expected}")
    if side == "cpu":
        if not hasattr(dataset, "MP_PHYSICS"):
            raise ComparisonError(
                f"cpu/{domain.name} lacks MP_PHYSICS provenance")
        if int(getattr(dataset, "MP_PHYSICS")) != 18:
            raise ComparisonError(
                f"cpu/{domain.name} MP_PHYSICS is not native NSSL option 18")

    schemas: dict[str, object] = {}
    for field in FIELD_SPECS:
        name = field.cpu_name if side == "cpu" else field.gpu_name
        if name not in dataset.variables:
            legacy = LEGACY_FOREIGN_CPU_ALIASES.get(field.canonical)
            if side == "cpu" and legacy in dataset.variables:
                raise ComparisonError(
                    f"cpu/{domain.name} carries legacy/foreign {legacy} for "
                    f"{field.canonical}; native MP18 authority requires {name}")
            raise ComparisonError(
                f"{side}/{domain.name} is missing required {field.canonical} "
                f"variable {name}")
        schemas[field.canonical] = _require_variable_schema(
            dataset.variables[name], dimensions=_layout_dimensions(field.layout),
            unit_family=field.unit_family, stagger=_layout_stagger(field.layout),
            label=f"{side}/{domain.name}/{name}")

    auxiliaries = {
        "XLAT": (("Time", "south_north", "west_east"), "latitude", ""),
        "XLONG": (("Time", "south_north", "west_east"), "longitude", ""),
        "XLAT_U": (("Time", "south_north", "west_east_stag"), "latitude", "X"),
        "XLONG_U": (("Time", "south_north", "west_east_stag"), "longitude", "X"),
        "XLAT_V": (("Time", "south_north_stag", "west_east"), "latitude", "Y"),
        "XLONG_V": (("Time", "south_north_stag", "west_east"), "longitude", "Y"),
        "MAPFAC_M": (("Time", "south_north", "west_east"), "map_factor", ""),
        "MAPFAC_U": (("Time", "south_north", "west_east_stag"), "map_factor", "X"),
        "MAPFAC_V": (("Time", "south_north_stag", "west_east"), "map_factor", "Y"),
        "ZNU": (("Time", "bottom_top"), "eta", ""),
        "ZNW": (("Time", "bottom_top_stag"), "eta", "Z"),
    }
    for name, (dimensions, family, stagger) in auxiliaries.items():
        if name not in dataset.variables:
            raise ComparisonError(f"{side}/{domain.name} missing auxiliary {name}")
        schemas[name] = _require_variable_schema(
            dataset.variables[name], dimensions=dimensions,
            unit_family=family, stagger=stagger,
            label=f"{side}/{domain.name}/{name}")

    if "REFL_10CM" in dataset.variables:
        schemas["composite_reflectivity"] = _require_variable_schema(
            dataset.variables["REFL_10CM"],
            dimensions=_layout_dimensions("mass3"),
            unit_family="reflectivity", stagger="",
            label=f"{side}/{domain.name}/REFL_10CM")
    return schemas


def _read_array(variable, key, label: str) -> np.ndarray:
    value = np.ma.asarray(variable[key])
    if np.ma.is_masked(value) and np.any(np.ma.getmaskarray(value)):
        raise ComparisonError(f"{label} contains masked/fill values")
    array = np.asarray(value, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ComparisonError(f"{label} contains NaN or Inf")
    return array


def iter_chunks(payload_shape: Sequence[int], chunk_values: int
                ) -> Iterable[tuple[tuple[object, ...], int, int]]:
    if chunk_values < 1 or not payload_shape:
        raise ComparisonError("streaming chunk size and payload shape must be positive")
    tail = math.prod(payload_shape[1:]) if len(payload_shape) > 1 else 1
    slab = max(1, chunk_values // tail)
    for start in range(0, int(payload_shape[0]), slab):
        stop = min(int(payload_shape[0]), start + slab)
        yield ((0, slice(start, stop),
                *(slice(None) for _ in payload_shape[1:])), start, stop)


class StreamingPairStats:
    """Stable Chan-style paired moments, merged one NumPy slab at a time."""

    def __init__(self) -> None:
        self.count = 0
        self.mean_x = 0.0
        self.mean_y = 0.0
        self.m2_x = 0.0
        self.m2_y = 0.0
        self.co_moment = 0.0
        self.sum_difference = 0.0
        self.sum_absolute = 0.0
        self.sum_squared = 0.0
        self.max_absolute = 0.0

    def update(self, cpu: np.ndarray, gpu: np.ndarray) -> np.ndarray:
        x = np.asarray(cpu, dtype=np.float64).reshape(-1)
        y = np.asarray(gpu, dtype=np.float64).reshape(-1)
        if x.shape != y.shape or x.size == 0:
            raise ComparisonError("streaming operands must have one nonempty shape")
        difference = y - x
        absolute = np.abs(difference)
        n = int(x.size)
        mean_x = float(np.mean(x, dtype=np.float64))
        mean_y = float(np.mean(y, dtype=np.float64))
        dx = x - mean_x
        dy = y - mean_y
        m2_x = float(np.dot(dx, dx))
        m2_y = float(np.dot(dy, dy))
        co = float(np.dot(dx, dy))
        if self.count == 0:
            self.count = n
            self.mean_x, self.mean_y = mean_x, mean_y
            self.m2_x, self.m2_y, self.co_moment = m2_x, m2_y, co
        else:
            total = self.count + n
            delta_x = mean_x - self.mean_x
            delta_y = mean_y - self.mean_y
            correction = self.count * n / total
            self.m2_x += m2_x + delta_x * delta_x * correction
            self.m2_y += m2_y + delta_y * delta_y * correction
            self.co_moment += co + delta_x * delta_y * correction
            self.mean_x += delta_x * n / total
            self.mean_y += delta_y * n / total
            self.count = total
        self.sum_difference += float(np.sum(difference, dtype=np.float64))
        self.sum_absolute += float(np.sum(absolute, dtype=np.float64))
        self.sum_squared += float(np.dot(difference, difference))
        self.max_absolute = max(self.max_absolute, float(np.max(absolute)))
        return absolute

    def finish(self) -> dict[str, float | int | None]:
        if self.count < 1:
            raise ComparisonError("streaming metric has no samples")
        scale = math.sqrt(max(0.0, self.m2_x) * max(0.0, self.m2_y))
        if scale > 0.0:
            correlation: float | None = self.co_moment / scale
            correlation = min(1.0, max(-1.0, correlation))
        elif self.m2_x == 0.0 and self.m2_y == 0.0 \
                and self.mean_x == self.mean_y:
            correlation = 1.0
        else:
            correlation = None
        return {
            "count": self.count,
            "bias": self.sum_difference / self.count,
            "mae": self.sum_absolute / self.count,
            "rmse": math.sqrt(self.sum_squared / self.count),
            "max_abs": self.max_absolute,
            "correlation": correlation,
            "cpu_mean": self.mean_x,
            "gpu_mean": self.mean_y,
        }


def exact_nearest_rank_percentiles(values: np.memmap,
                                   percentiles: Sequence[float]
                                   ) -> dict[str, float]:
    count = int(values.size)
    if count < 1:
        raise ComparisonError("percentile sample is empty")
    indices = [max(0, min(count - 1, math.ceil(p * count / 100.0) - 1))
               for p in percentiles]
    values.partition(sorted(set(indices)))
    return {f"p{p:g}_abs": float(values[index])
            for p, index in zip(percentiles, indices)}


def _compare_geometry_auxiliary(cpu_ds, gpu_ds, name: str, *,
                                chunk_values: int, label: str
                                ) -> dict[str, object]:
    cpu_var = cpu_ds.variables[name]
    gpu_var = gpu_ds.variables[name]
    if cpu_var.shape != gpu_var.shape:
        raise ComparisonError(
            f"{label}/{name} geometry shapes differ: "
            f"{cpu_var.shape} vs {gpu_var.shape}")
    payload = cpu_var.shape[1:]
    maximum = 0.0
    for key, _start, _stop in iter_chunks(payload, chunk_values):
        cpu = _read_array(cpu_var, key, f"cpu/{label}/{name}")
        gpu = _read_array(gpu_var, key, f"gpu/{label}/{name}")
        if name.startswith("MAPFAC_"):
            if np.any(cpu <= 0.0):
                raise ComparisonError(f"cpu/{label}/{name} is nonpositive")
            maximum = max(
                maximum, float(np.max(np.abs(gpu / cpu - 1.0))))
        else:
            maximum = max(maximum, float(np.max(np.abs(gpu - cpu))))
    if name.startswith("XLAT") or name.startswith("XLONG"):
        threshold = COORD_ABS_TOL_DEG
        metric = "max_abs_degrees"
    elif name.startswith("MAPFAC_"):
        threshold = MAPFAC_REL_TOL
        metric = "max_abs_relative"
    else:
        threshold = 0.0
        metric = "max_abs"
    if maximum > threshold:
        if threshold == 0.0:
            raise ComparisonError(
                f"{label}/{name} is not exact across CPU/GPU geometry; "
                f"max_abs={maximum}")
        raise ComparisonError(
            f"{label}/{name} exceeds documented geometry tolerance: "
            f"{metric}={maximum}, threshold={threshold}")
    return {
        "field": name, "metric": metric, "value": maximum,
        "operator": "<=", "threshold": threshold,
        "authority": "tests/test_lambert.py Phase-3 WPS/geo_em oracle",
    }


def _load_inventory_weights(dataset, domain: DomainSpec, side: str
                            ) -> tuple[np.ndarray, np.ndarray]:
    mu = _read_array(dataset.variables["MU"], 0, f"{side}/{domain.name}/MU")
    mub = _read_array(dataset.variables["MUB"], 0, f"{side}/{domain.name}/MUB")
    mapfac = _read_array(
        dataset.variables["MAPFAC_M"], 0,
        f"{side}/{domain.name}/MAPFAC_M")
    znw = _read_array(dataset.variables["ZNW"], 0, f"{side}/{domain.name}/ZNW")
    if mu.shape != (domain.ny, domain.nx) or mub.shape != mu.shape \
            or mapfac.shape != mu.shape:
        raise ComparisonError(f"{side}/{domain.name} inventory geometry is invalid")
    if znw.shape != (domain.nz + 1,):
        raise ComparisonError(f"{side}/{domain.name} ZNW shape is invalid")
    if np.any(mapfac <= 0.0) or np.any(mu + mub <= 0.0):
        raise ComparisonError(
            f"{side}/{domain.name} dry-mass/map-factor weights are nonpositive")
    delta_eta = np.abs(np.diff(znw))
    if np.any(delta_eta <= 0.0) or not math.isclose(
            float(np.sum(delta_eta)), 1.0, rel_tol=0.0, abs_tol=2.0e-6):
        raise ComparisonError(
            f"{side}/{domain.name} ZNW does not define a complete eta column")
    # (MU+MUB)/g is column dry mass per physical square meter; DX*DY/MAPFAC^2
    # is physical cell area.  Multiplication by d-eta yields per-level dry
    # mass, the correct weight for WRF dry-air mixing ratios.
    horizontal_dry_mass = ((mu + mub) / G) * (
        domain.dx_m * domain.dx_m / (mapfac * mapfac))
    return horizontal_dry_mass, delta_eta


def _weighted_inventory(array: np.ndarray, field: FieldSpec, *, start: int,
                        horizontal_dry_mass: np.ndarray,
                        delta_eta: np.ndarray) -> float:
    if field.inventory_kind in {
            "water_mass", "particle_number", "particle_volume"}:
        if field.layout != "mass3":
            raise ComparisonError(
                f"internal inventory layout error for {field.canonical}")
        eta = delta_eta[start:start + array.shape[0], None, None]
        return float(np.sum(
            array * eta * horizontal_dry_mass[None, :, :], dtype=np.float64))
    return 0.0


def _surface_cell_area(dataset, domain: DomainSpec, side: str) -> np.ndarray:
    mapfac = _read_array(
        dataset.variables["MAPFAC_M"], 0,
        f"{side}/{domain.name}/MAPFAC_M")
    if np.any(mapfac <= 0.0):
        raise ComparisonError(f"{side}/{domain.name} MAPFAC_M is nonpositive")
    return domain.dx_m * domain.dx_m / (mapfac * mapfac)


def compare_field(cpu_var, gpu_var, field: FieldSpec, *, domain: DomainSpec,
                  lead_seconds: int, percentiles: Sequence[float],
                  negative_epsilon: float, chunk_values: int,
                  temporary_root: Path,
                  cpu_weights: tuple[np.ndarray, np.ndarray],
                  gpu_weights: tuple[np.ndarray, np.ndarray],
                  cpu_area: np.ndarray, gpu_area: np.ndarray,
                  ) -> dict[str, object]:
    if cpu_var.shape != gpu_var.shape:
        raise ComparisonError(
            f"{domain.name}/{field.canonical} shapes differ: "
            f"{cpu_var.shape} vs {gpu_var.shape}")
    payload_shape = cpu_var.shape[1:]
    count = math.prod(payload_shape)
    percentile_path = temporary_root / (
        f"{domain.name}-{lead_seconds:05d}-{field.canonical}.f64")
    absolute_values = np.memmap(
        percentile_path, dtype=np.float64, mode="w+", shape=(count,))
    stats = StreamingPairStats()
    cpu_minimum = math.inf
    cpu_maximum = -math.inf
    gpu_minimum = math.inf
    gpu_maximum = -math.inf
    cpu_inventory = 0.0
    gpu_inventory = 0.0
    offset = 0
    try:
        for key, start, _stop in iter_chunks(payload_shape, chunk_values):
            cpu = _read_array(
                cpu_var, key,
                f"cpu/{domain.name}/{lead_seconds}/{field.cpu_name}")
            gpu = _read_array(
                gpu_var, key,
                f"gpu/{domain.name}/{lead_seconds}/{field.gpu_name}")
            if cpu.shape != gpu.shape:
                raise ComparisonError(
                    f"{domain.name}/{field.canonical} slab shapes differ")
            cpu_minimum = min(cpu_minimum, float(np.min(cpu)))
            cpu_maximum = max(cpu_maximum, float(np.max(cpu)))
            gpu_minimum = min(gpu_minimum, float(np.min(gpu)))
            gpu_maximum = max(gpu_maximum, float(np.max(gpu)))
            if field.nonnegative and (cpu_minimum < -negative_epsilon
                                      or gpu_minimum < -negative_epsilon):
                raise ComparisonError(
                    f"{domain.name}/{lead_seconds}/{field.canonical} violates "
                    f"nonnegative health epsilon {negative_epsilon}: "
                    f"cpu_min={cpu_minimum}, gpu_min={gpu_minimum}")
            absolute = stats.update(cpu, gpu)
            absolute_values[offset:offset + absolute.size] = absolute
            offset += absolute.size
            if field.inventory_kind in {
                    "water_mass", "particle_number", "particle_volume"}:
                cpu_inventory += _weighted_inventory(
                    cpu, field, start=start,
                    horizontal_dry_mass=cpu_weights[0],
                    delta_eta=cpu_weights[1])
                gpu_inventory += _weighted_inventory(
                    gpu, field, start=start,
                    horizontal_dry_mass=gpu_weights[0],
                    delta_eta=gpu_weights[1])
            elif field.inventory_kind is not None \
                    and field.inventory_kind.startswith("surface_"):
                cpu_inventory += float(np.sum(
                    cpu * cpu_area[start:start + cpu.shape[0]],
                    dtype=np.float64))
                gpu_inventory += float(np.sum(
                    gpu * gpu_area[start:start + gpu.shape[0]],
                    dtype=np.float64))
        if offset != count:
            raise ComparisonError(
                f"{domain.name}/{field.canonical} streaming sample mismatch")
        absolute_values.flush()
        result = stats.finish()
        result.update(exact_nearest_rank_percentiles(
            absolute_values, percentiles))
    finally:
        del absolute_values
        percentile_path.unlink(missing_ok=True)
    result.update({
        "field": field.canonical,
        "cpu_variable": field.cpu_name,
        "gpu_variable": field.gpu_name,
        "unit_family": field.unit_family,
        "canonical_units": _CANONICAL_UNITS[field.unit_family],
        "cpu_units": str(getattr(cpu_var, "units")),
        "gpu_units": str(getattr(gpu_var, "units")),
        "domain": domain.name,
        "lead_seconds": lead_seconds,
        "cpu_min": cpu_minimum,
        "cpu_max": cpu_maximum,
        "gpu_min": gpu_minimum,
        "gpu_max": gpu_maximum,
    })
    if field.inventory_kind is not None:
        difference = gpu_inventory - cpu_inventory
        result["inventory"] = {
            "kind": field.inventory_kind,
            "units": _inventory_units(field.inventory_kind),
            "cpu": cpu_inventory,
            "gpu": gpu_inventory,
            "difference": difference,
            "relative_difference": (
                difference / abs(cpu_inventory)
                if cpu_inventory != 0.0 else None),
        }
    return result


def _composite_reflectivity(dataset, domain: DomainSpec, side: str,
                            chunk_values: int) -> np.ndarray:
    variable = dataset.variables["REFL_10CM"]
    composite = np.full((domain.ny, domain.nx), -np.inf, dtype=np.float64)
    for key, _start, _stop in iter_chunks(variable.shape[1:], chunk_values):
        slab = _read_array(variable, key, f"{side}/{domain.name}/REFL_10CM")
        composite = np.maximum(composite, np.max(slab, axis=0))
    if not np.isfinite(composite).all():
        raise ComparisonError(f"{side}/{domain.name} reflectivity composite is invalid")
    return composite


def _interior(array: np.ndarray, width: int) -> np.ndarray:
    if width < 0:
        raise ComparisonError("interior exclusion must be nonnegative")
    if width == 0:
        return array
    if min(array.shape[-2:]) <= 2 * width:
        raise ComparisonError("surface field is too small for policy interior mask")
    return array[..., width:-width, width:-width]


def categorical_scores(cpu: np.ndarray, gpu: np.ndarray,
                       threshold: float) -> dict[str, float | int | None]:
    if cpu.shape != gpu.shape or cpu.ndim != 2:
        raise ComparisonError("categorical operands must share one 2-D grid")
    cpu_event = cpu >= threshold
    gpu_event = gpu >= threshold
    hits = int(np.count_nonzero(cpu_event & gpu_event))
    misses = int(np.count_nonzero(cpu_event & ~gpu_event))
    false_alarms = int(np.count_nonzero(~cpu_event & gpu_event))
    correct_negatives = int(np.count_nonzero(~cpu_event & ~gpu_event))
    total = hits + misses + false_alarms + correct_negatives

    def ratio(numerator: float, denominator: float) -> float | None:
        return numerator / denominator if denominator != 0.0 else None

    random_hits = ((hits + misses) * (hits + false_alarms) / total
                   if total else 0.0)
    hss_numerator = 2.0 * (
        hits * correct_negatives - misses * false_alarms)
    hss_denominator = ((hits + misses) * (misses + correct_negatives)
                       + (hits + false_alarms)
                       * (false_alarms + correct_negatives))
    return {
        "hits": hits, "misses": misses, "false_alarms": false_alarms,
        "correct_negatives": correct_negatives,
        "cpu_event_fraction": ratio(hits + misses, total),
        "gpu_event_fraction": ratio(hits + false_alarms, total),
        "probability_of_detection": ratio(hits, hits + misses),
        "false_alarm_ratio": ratio(false_alarms, hits + false_alarms),
        "critical_success_index": ratio(
            hits, hits + misses + false_alarms),
        "frequency_bias": ratio(hits + false_alarms, hits + misses),
        "equitable_threat_score": ratio(
            hits - random_hits,
            hits + misses + false_alarms - random_hits),
        "heidke_skill_score": ratio(hss_numerator, hss_denominator),
    }


def _event_definition(policy: Mapping[str, object], name: str
                      ) -> dict[str, object]:
    events = policy["surface_events"]
    if not isinstance(events, dict) or name not in events \
            or not isinstance(events[name], dict):
        raise ComparisonError(f"policy lacks surface event {name}")
    return dict(events[name])


def surface_event_metrics(cpu: np.ndarray, gpu: np.ndarray, *,
                          event_name: str, definition: Mapping[str, object],
                          domain: DomainSpec, lead_seconds: int,
                          interior_width: int,
                          degenerate_floor: float,
                          policy_status: str = "candidate-unratified",
                          ) -> list[dict[str, object]]:
    cpu = _interior(cpu, interior_width)
    gpu = _interior(gpu, interior_width)
    radius_km = float(definition["neighborhood_radius_km"])
    half_width = max(0, int(math.floor(radius_km * 1000.0 / domain.dx_m + 0.5)))
    minima = definition.get("minimum_fss", {})
    if not isinstance(minima, dict):
        raise ComparisonError(f"{event_name} minimum_fss must be an object")
    records = []
    for raw_threshold in definition["thresholds"]:
        threshold = float(raw_threshold)
        scores = categorical_scores(cpu, gpu, threshold)
        fss = 1.0 - fss_distance(
            gpu, cpu, threshold=threshold, half_width_cells=half_width)
        cpu_coverage = float(np.mean(cpu >= threshold, dtype=np.float64))
        gpu_coverage = float(np.mean(gpu >= threshold, dtype=np.float64))
        minimum = minima.get(f"{threshold:g}")
        if cpu_coverage < degenerate_floor or gpu_coverage < degenerate_floor:
            evaluation = "held_degenerate"
        elif minimum is None:
            evaluation = "report_only"
        else:
            evaluation = _evaluation_label(
                fss >= float(minimum), policy_status)
        records.append({
            "event": event_name, "domain": domain.name,
            "lead_seconds": lead_seconds, "threshold": threshold,
            "threshold_units": str(definition["units"]),
            "interior_exclusion_cells": interior_width,
            "neighborhood_radius_km": radius_km,
            "neighborhood_half_width_cells": half_width,
            "fss": fss, "minimum_fss": minimum,
            "cpu_event_coverage": cpu_coverage,
            "gpu_event_coverage": gpu_coverage,
            "degenerate_event_floor": degenerate_floor,
            "evaluation": evaluation, **scores,
        })
    return records


def validate_policy(payload: Mapping[str, object]) -> dict[str, object]:
    policy = dict(payload)
    required = {
        "schema", "policy_id", "status", "authority",
        "fixed_before_gpu_run_required", "percentiles_absolute_error",
        "health", "surface_metric_convention", "surface_events",
        "continuous_metric_rules",
    }
    missing = required - set(policy)
    if missing:
        raise ComparisonError(f"comparison policy is missing {sorted(missing)}")
    if policy["schema"] != POLICY_SCHEMA:
        raise ComparisonError(f"comparison policy schema must be {POLICY_SCHEMA}")
    if not isinstance(policy["policy_id"], str) or not policy["policy_id"]:
        raise ComparisonError("comparison policy_id must be nonempty")
    status = policy["status"]
    if status not in {"candidate-unratified", "ratified"}:
        raise ComparisonError(
            "comparison policy status must be candidate-unratified or ratified")
    if policy["fixed_before_gpu_run_required"] is not True:
        raise ComparisonError("policy must require pre-run freezing")
    authority = policy["authority"]
    if not isinstance(authority, dict):
        raise ComparisonError("policy authority must be an object")
    ratified = authority.get("ratified_for_this_comparison")
    if ratified is not (status == "ratified"):
        raise ComparisonError("policy status and ratification declaration disagree")
    if status == "ratified":
        for key in ("approved_by", "approved_at_utc", "record"):
            if not isinstance(authority.get(key), str) or not authority[key]:
                raise ComparisonError(
                    f"ratified policy authority requires nonempty {key}")

    percentiles = policy["percentiles_absolute_error"]
    if (not isinstance(percentiles, list) or not percentiles
            or any(isinstance(value, bool) for value in percentiles)):
        raise ComparisonError("policy percentiles must be a nonempty list")
    converted = [float(value) for value in percentiles]
    if (converted != sorted(set(converted))
            or converted[0] <= 0.0 or converted[-1] > 100.0
            or any(not math.isfinite(value) for value in converted)):
        raise ComparisonError("policy percentiles must be unique increasing (0,100]")

    health = policy["health"]
    if not isinstance(health, dict):
        raise ComparisonError("policy health must be an object")
    epsilon = health.get("nonnegative_epsilon")
    if (isinstance(epsilon, bool) or not isinstance(epsilon, (int, float))
            or not math.isfinite(float(epsilon)) or float(epsilon) < 0.0):
        raise ComparisonError("policy nonnegative_epsilon must be finite and >= 0")

    surface = policy["surface_metric_convention"]
    if not isinstance(surface, dict):
        raise ComparisonError("surface_metric_convention must be an object")
    width = surface.get("interior_exclusion_cells")
    floor = surface.get("degenerate_event_floor")
    if isinstance(width, bool) or not isinstance(width, int) or width < 0:
        raise ComparisonError("surface interior exclusion must be an integer >= 0")
    if (isinstance(floor, bool) or not isinstance(floor, (int, float))
            or not 0.0 <= float(floor) < 1.0):
        raise ComparisonError("degenerate_event_floor must be in [0,1)")

    events = policy["surface_events"]
    required_events = {
        "gridscale_precipitation", "hail_accumulation",
        "composite_reflectivity",
    }
    if not isinstance(events, dict) or set(events) != required_events:
        raise ComparisonError(
            f"surface_events must contain exactly {sorted(required_events)}")
    expected_sources = {
        "gridscale_precipitation": "RAINNC",
        "hail_accumulation": "HAILNC",
        "composite_reflectivity": "REFL_10CM",
    }
    for name, source in expected_sources.items():
        definition = events[name]
        if not isinstance(definition, dict) or definition.get("source") != source:
            raise ComparisonError(f"surface event {name} must source {source}")
        thresholds = definition.get("thresholds")
        radius = definition.get("neighborhood_radius_km")
        units = definition.get("units")
        if (not isinstance(thresholds, list) or not thresholds
                or not isinstance(units, str) or not units):
            raise ComparisonError(f"surface event {name} has invalid thresholds/units")
        values = [float(value) for value in thresholds]
        if (values != sorted(set(values)) or values[0] < 0.0
                or any(not math.isfinite(value) for value in values)):
            raise ComparisonError(
                f"surface event {name} thresholds must be finite/increasing")
        if (isinstance(radius, bool) or not isinstance(radius, (int, float))
                or not math.isfinite(float(radius)) or float(radius) < 0.0):
            raise ComparisonError(f"surface event {name} radius is invalid")
        minima = definition.get("minimum_fss", {})
        if not isinstance(minima, dict):
            raise ComparisonError(f"surface event {name} minimum_fss is invalid")
        for threshold, minimum in minima.items():
            try:
                threshold_value = float(threshold)
            except (TypeError, ValueError) as exc:
                raise ComparisonError(
                    f"surface event {name} minimum_fss key is invalid") from exc
            if threshold_value not in values or isinstance(minimum, bool) \
                    or not isinstance(minimum, (int, float)) \
                    or not 0.0 <= float(minimum) <= 1.0:
                raise ComparisonError(
                    f"surface event {name} minimum_fss {threshold!r} is invalid")

    rules = policy["continuous_metric_rules"]
    if not isinstance(rules, list):
        raise ComparisonError("continuous_metric_rules must be a list")
    seen = set()
    metrics = {
        "bias", "mae", "rmse", "max_abs", "correlation",
        *{f"p{value:g}_abs" for value in converted},
    }
    for index, rule in enumerate(rules):
        if not isinstance(rule, dict):
            raise ComparisonError(f"continuous rule {index} must be an object")
        field = rule.get("field")
        metric = rule.get("metric")
        operator = rule.get("operator")
        domains = rule.get("domains", [domain.name for domain in DOMAINS])
        if field != "*" and field not in FIELD_BY_CANONICAL:
            raise ComparisonError(f"continuous rule {index} field is unknown")
        if metric not in metrics or operator not in {"<=", ">="}:
            raise ComparisonError(f"continuous rule {index} metric/operator is invalid")
        if (not isinstance(domains, list) or not domains
                or any(domain not in DOMAIN_BY_NAME for domain in domains)):
            raise ComparisonError(f"continuous rule {index} domains are invalid")
        value = rule.get("value")
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(float(value))):
            raise ComparisonError(f"continuous rule {index} value is invalid")
        if not isinstance(rule.get("units"), str) or not rule["units"]:
            raise ComparisonError(f"continuous rule {index} units are required")
        if field == "*" and metric != "correlation":
            raise ComparisonError(
                f"continuous rule {index} cannot wildcard dimensional fields")
        expected_units = (
            "1" if metric == "correlation" else
            _CANONICAL_UNITS[FIELD_BY_CANONICAL[str(field)].unit_family])
        if rule["units"] != expected_units:
            raise ComparisonError(
                f"continuous rule {index} units must be {expected_units!r}")
        key = (field, metric, tuple(sorted(domains)))
        if key in seen:
            raise ComparisonError(f"continuous rule {index} is duplicated")
        seen.add(key)
    return policy


def load_policy(path: Path) -> tuple[dict[str, object], dict[str, object]]:
    policy = validate_policy(load_json(path, "comparison policy"))
    identity = file_identity(path)
    identity.update({
        "policy_id": policy["policy_id"], "status": policy["status"],
        "payload_sha256": stable_hash(policy),
    })
    return policy, identity


def _logical_inventory(frames: Mapping[tuple[str, datetime], Path]
                       ) -> list[dict[str, object]]:
    return [
        file_identity(path, logical_name=path.name)
        for (_domain, _valid), path in sorted(frames.items())
    ]


def _inventory_map(inventory: Sequence[Mapping[str, object]]
                   ) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for row in inventory:
        name = row.get("logical_name")
        if not isinstance(name, str) or name in result:
            raise ComparisonError("evidence inventory has invalid logical names")
        sha = row.get("sha256")
        size = row.get("bytes")
        if (_HEX64.fullmatch(str(sha)) is None or isinstance(size, bool)
                or not isinstance(size, int) or size <= 0):
            raise ComparisonError(f"evidence inventory identity is invalid: {row}")
        result[name] = dict(row)
    return result


def _require_inventory_calendar(inventory: Sequence[Mapping[str, object]]) -> None:
    logical = _inventory_map(inventory)
    actual = {_parse_frame_name(name) for name in logical}
    expected = {
        (domain, valid)
        for domain, times in expected_calendar().items() for valid in times
    }
    if actual != expected or len(logical) != 64:
        raise ComparisonError(
            "bound output inventory does not encode the exact 64-frame calendar")


def _calendar_summary() -> dict[str, object]:
    calendar = expected_calendar()
    return {
        "start_time": START_TIME.isoformat(),
        "end_time": (START_TIME + timedelta(seconds=RUN_SECONDS)).isoformat(),
        "run_seconds": RUN_SECONDS,
        "counts": {domain: len(times) for domain, times in calendar.items()},
        "total_files_per_side": sum(len(times) for times in calendar.values()),
        "cadence_seconds": {
            domain.name: domain.cadence_s for domain in DOMAINS},
    }


def _preflight_cpu_structure(frames: Mapping[tuple[str, datetime], Path]
                             ) -> None:
    for (domain_name, valid), path in sorted(frames.items()):
        with netCDF4.Dataset(path) as dataset:
            inspect_frame(
                dataset, side="cpu", domain=DOMAIN_BY_NAME[domain_name],
                valid_time=valid)


def build_registration(cpu_root: Path, policy_path: Path, *,
                       source_identity: Mapping[str, object] | None = None,
                       created_at_utc: str | None = None) -> dict[str, object]:
    policy, policy_identity = load_policy(policy_path)
    frames = discover_frames(cpu_root)
    _preflight_cpu_structure(frames)
    source = dict(source_identity or repository_identity())
    binding = {
        "source": source,
        "evaluator": file_identity(Path(__file__)),
        "policy": policy_identity,
        "calendar": _calendar_summary(),
        "cpu_outputs": _logical_inventory(frames),
        "field_aliases": {
            field.canonical: {
                "cpu": field.cpu_name, "gpu": field.gpu_name,
                "layout": field.layout, "unit_family": field.unit_family,
            }
            for field in FIELD_SPECS
        },
        "rejected_legacy_foreign_cpu_aliases": LEGACY_FOREIGN_CPU_ALIASES,
    }
    return {
        "schema": REGISTRATION_SCHEMA,
        "created_at_utc": created_at_utc or utc_now(),
        "policy_status": policy["status"],
        "binding_sha256": stable_hash(binding),
        "binding": binding,
    }


def write_registration(cpu_root: Path, policy_path: Path, output: Path) -> None:
    _require_outside_repository(output)
    if output.exists():
        raise ComparisonError(
            f"refusing to replace frozen registration: {output.resolve()}")
    registration = build_registration(cpu_root, policy_path)
    atomic_json(output.resolve(), registration)


def validate_registration(registration: Mapping[str, object], *,
                          policy_identity: Mapping[str, object],
                          source_identity: Mapping[str, object],
                          cpu_inventory: Sequence[Mapping[str, object]],
                          ) -> dict[str, object]:
    if registration.get("schema") != REGISTRATION_SCHEMA:
        raise ComparisonError("comparison registration schema is invalid")
    binding = registration.get("binding")
    if not isinstance(binding, dict) \
            or registration.get("binding_sha256") != stable_hash(binding):
        raise ComparisonError("comparison registration binding hash is invalid")
    if binding.get("source") != dict(source_identity):
        raise ComparisonError("evaluator source differs from frozen registration")
    if binding.get("evaluator") != file_identity(Path(__file__)):
        raise ComparisonError("evaluator bytes differ from frozen registration")
    if binding.get("policy") != dict(policy_identity):
        raise ComparisonError("policy differs from frozen registration")
    if binding.get("calendar") != _calendar_summary():
        raise ComparisonError("registration calendar differs from evaluator authority")
    _require_inventory_calendar(binding.get("cpu_outputs", []))
    _require_inventory_calendar(cpu_inventory)
    if _inventory_map(binding.get("cpu_outputs", [])) != _inventory_map(cpu_inventory):
        raise ComparisonError("CPU wrfout identities differ from frozen registration")
    aliases = binding.get("field_aliases")
    expected_aliases = {
        field.canonical: {
            "cpu": field.cpu_name, "gpu": field.gpu_name,
            "layout": field.layout, "unit_family": field.unit_family,
        }
        for field in FIELD_SPECS
    }
    if aliases != expected_aliases:
        raise ComparisonError("registered native-field aliases have drifted")
    if (binding.get("rejected_legacy_foreign_cpu_aliases")
            != LEGACY_FOREIGN_CPU_ALIASES):
        raise ComparisonError("registered rejected legacy aliases have drifted")
    return dict(binding)


def _parse_utc(value: object, label: str) -> datetime:
    if not isinstance(value, str):
        raise ComparisonError(f"{label} must be an ISO UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ComparisonError(f"{label} is not a valid timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ComparisonError(f"{label} must carry UTC timezone")
    return parsed


def validate_gpu_manifests(gpu_root: Path, registration: Mapping[str, object],
                           source_identity: Mapping[str, object],
                           gpu_inventory: Sequence[Mapping[str, object]],
                           registration_path: Path,
                           policy_identity: Mapping[str, object],
                           ) -> dict[str, object]:
    launch_path = gpu_root / "metadata" / "launch-manifest.json"
    completion_path = gpu_root / "completion.json"
    launch = load_json(launch_path, "GPU launch manifest")
    completion = load_json(completion_path, "GPU completion manifest")
    if launch.get("schema") != GPU_LAUNCH_SCHEMA:
        raise ComparisonError("GPU launch manifest schema is invalid")
    launch_binding = launch.get("binding")
    if not isinstance(launch_binding, dict) \
            or launch.get("binding_sha256") != stable_hash(launch_binding):
        raise ComparisonError("GPU launch binding hash is invalid")
    if launch_binding.get("source") != dict(source_identity):
        raise ComparisonError(
            "GPU run source does not match preregistered comparator source")
    expected_comparison_binding = {
        "registration": file_identity(registration_path.resolve()),
        "registration_binding_sha256": registration["binding_sha256"],
        "registration_created_at_utc": registration["created_at_utc"],
        "policy": dict(policy_identity),
    }
    if (launch_binding.get("comparison_preregistration")
            != expected_comparison_binding):
        raise ComparisonError(
            "GPU launch does not directly bind this registration/policy")
    registered_at = _parse_utc(
        registration.get("created_at_utc"), "registration.created_at_utc")
    launched_at = _parse_utc(
        launch.get("created_at_utc"), "launch.created_at_utc")
    if registered_at > launched_at:
        raise ComparisonError(
            "comparison policy/registration was frozen after the GPU launch")
    requested = launch_binding.get("requested")
    expected_counts = {"d01": 13, "d02": 13, "d03": 13, "d04": 25}
    if (not isinstance(requested, dict)
            or requested.get("run_seconds") != RUN_SECONDS
            or requested.get("start_time") != START_TIME.isoformat()
            or requested.get("expected_output_counts") != expected_counts
            or requested.get("restart_interval_s") != 0):
        raise ComparisonError("GPU launch request is not the frozen 12 h calendar")
    expected_topology = [
        {
            "grid_id": domain.grid_id,
            "parent_id": 0 if domain.grid_id == 1 else domain.parent_id,
            "i_parent_start": domain.i_parent_start,
            "j_parent_start": domain.j_parent_start,
            "parent_grid_ratio": domain.parent_grid_ratio,
            "parent_time_step_ratio": domain.parent_grid_ratio,
            "mass_shape": [domain.ny, domain.nx],
            "dx_m": domain.dx_m, "dt_s": domain.dt_s,
            "history_interval_s": float(domain.cadence_s),
            "mp_physics": 18,
        }
        for domain in DOMAINS
    ]
    if launch_binding.get("topology") != expected_topology:
        raise ComparisonError("GPU launch topology/physics differs from authority")
    if completion.get("schema") != GPU_COMPLETION_SCHEMA:
        raise ComparisonError("GPU completion manifest schema is invalid")
    if (completion.get("binding_sha256") != launch.get("binding_sha256")
            or completion.get("output_counts") != expected_counts):
        raise ComparisonError("GPU completion does not bind the launch/calendar")
    declared: list[Mapping[str, object]] = []
    outputs = completion.get("outputs")
    if not isinstance(outputs, dict):
        raise ComparisonError("GPU completion outputs are missing")
    for domain in expected_counts:
        rows = outputs.get(domain)
        if not isinstance(rows, list):
            raise ComparisonError(f"GPU completion output list {domain} is missing")
        for row in rows:
            if not isinstance(row, dict):
                raise ComparisonError("GPU completion output identity is malformed")
            declared.append({
                "logical_name": Path(str(row.get("path", ""))).name,
                "bytes": row.get("bytes"), "sha256": row.get("sha256"),
            })
    if _inventory_map(declared) != _inventory_map(gpu_inventory):
        raise ComparisonError(
            "GPU wrfout identities differ from accepted completion.json")
    return {
        "launch_manifest": file_identity(launch_path),
        "completion_manifest": file_identity(completion_path),
        "launch_created_at_utc": launch["created_at_utc"],
        "launch_binding_sha256": launch["binding_sha256"],
    }


def _rule_for(policy: Mapping[str, object], *, field: str, metric: str,
              domain: str) -> Mapping[str, object] | None:
    matches = []
    for rule in policy["continuous_metric_rules"]:
        if (rule["metric"] == metric
                and rule["field"] in {field, "*"}
                and domain in rule.get(
                    "domains", [item.name for item in DOMAINS])):
            matches.append(rule)
    exact = [rule for rule in matches if rule["field"] == field]
    selected = exact or matches
    if len(selected) > 1:
        raise ComparisonError(
            f"policy has ambiguous rules for {domain}/{field}/{metric}")
    return selected[0] if selected else None


def _evaluation_label(passed: bool, policy_status: str) -> str:
    if policy_status == "ratified":
        return "pass" if passed else "fail"
    return "candidate_pass" if passed else "candidate_fail"


def apply_scientific_policy(result: Mapping[str, object],
                            policy: Mapping[str, object]
                            ) -> list[dict[str, object]]:
    metric_names = [
        "bias", "mae", "rmse", "max_abs", "correlation",
        *[f"p{float(value):g}_abs"
          for value in policy["percentiles_absolute_error"]],
    ]
    rows = []
    field_spec = FIELD_BY_CANONICAL[str(result["field"])]
    canonical_units = str(result.get(
        "canonical_units", _CANONICAL_UNITS[field_spec.unit_family]))
    for metric in metric_names:
        value = result.get(metric)
        rule = _rule_for(
            policy, field=str(result["field"]), metric=metric,
            domain=str(result["domain"]))
        if rule is None:
            evaluation = "report_only"
            operator = None
            threshold = None
        else:
            operator = str(rule["operator"])
            threshold = float(rule["value"])
            if value is None:
                passed = False
            elif operator == "<=":
                passed = float(value) <= threshold
            else:
                passed = float(value) >= threshold
            evaluation = _evaluation_label(passed, str(policy["status"]))
        rows.append({
            "category": "continuous", "domain": result["domain"],
            "lead_seconds": result["lead_seconds"],
            "field": result["field"], "metric": metric, "value": value,
            "value_units": (
                "1" if metric == "correlation" else canonical_units),
            "threshold_operator": operator, "threshold": threshold,
            "threshold_units": None if rule is None else rule["units"],
            "evaluation": evaluation,
        })
    return rows


def _simple_pair_summary(cpu: np.ndarray, gpu: np.ndarray
                         ) -> dict[str, float | int | None]:
    if cpu.shape != gpu.shape or cpu.size == 0:
        raise ComparisonError("diagnostic fields must share one nonempty shape")
    if not np.isfinite(cpu).all() or not np.isfinite(gpu).all():
        raise ComparisonError("diagnostic fields contain NaN or Inf")
    stats = StreamingPairStats()
    stats.update(cpu, gpu)
    return stats.finish()


def _frame_surface_events(cpu_ds, gpu_ds, *, policy: Mapping[str, object],
                          domain: DomainSpec, lead_seconds: int,
                          chunk_values: int) -> tuple[
                              list[dict[str, object]],
                              list[dict[str, object]]]:
    convention = policy["surface_metric_convention"]
    width = int(convention["interior_exclusion_cells"])
    floor = float(convention["degenerate_event_floor"])
    events: list[dict[str, object]] = []
    support: list[dict[str, object]] = []
    definitions = {
        "gridscale_precipitation": ("RAINNC", False),
        "hail_accumulation": ("HAILNC", False),
        "composite_reflectivity": ("REFL_10CM", True),
    }
    for event_name, (variable_name, composite) in definitions.items():
        cpu_supported = variable_name in cpu_ds.variables
        gpu_supported = variable_name in gpu_ds.variables
        support.append({
            "event": event_name, "domain": domain.name,
            "lead_seconds": lead_seconds, "cpu_supported": cpu_supported,
            "gpu_supported": gpu_supported,
            "scored": cpu_supported and gpu_supported,
        })
        if not (cpu_supported and gpu_supported):
            continue
        if composite:
            cpu = _composite_reflectivity(
                cpu_ds, domain, "cpu", chunk_values)
            gpu = _composite_reflectivity(
                gpu_ds, domain, "gpu", chunk_values)
        else:
            cpu = _read_array(
                cpu_ds.variables[variable_name], 0,
                f"cpu/{domain.name}/{variable_name}")
            gpu = _read_array(
                gpu_ds.variables[variable_name], 0,
                f"gpu/{domain.name}/{variable_name}")
        definition = _event_definition(policy, event_name)
        event_rows = surface_event_metrics(
            cpu, gpu, event_name=event_name, definition=definition,
            domain=domain, lead_seconds=lead_seconds,
            interior_width=width, degenerate_floor=floor,
            policy_status=str(policy["status"]))
        if composite:
            continuous = _simple_pair_summary(
                _interior(cpu, width), _interior(gpu, width))
            for row in event_rows:
                row["continuous_composite"] = continuous
        events.extend(event_rows)
    return events, support


def _aggregate_inventory(field_results: Sequence[Mapping[str, object]],
                         domain: str, lead_seconds: int
                         ) -> list[dict[str, object]]:
    groups: dict[str, dict[str, float]] = {}
    for result in field_results:
        inventory = result.get("inventory")
        if not isinstance(inventory, dict):
            continue
        kind = str(inventory["kind"])
        total = groups.setdefault(kind, {"cpu": 0.0, "gpu": 0.0})
        total["cpu"] += float(inventory["cpu"])
        total["gpu"] += float(inventory["gpu"])
    records = []
    for kind, total in sorted(groups.items()):
        difference = total["gpu"] - total["cpu"]
        records.append({
            "domain": domain, "lead_seconds": lead_seconds, "kind": kind,
            "units": _inventory_units(kind),
            "cpu": total["cpu"], "gpu": total["gpu"],
            "difference": difference,
            "relative_difference": (
                difference / abs(total["cpu"])
                if total["cpu"] != 0.0 else None),
        })
    if "water_mass" in groups and "surface_total_water_mass" in groups:
        cpu = (groups["water_mass"]["cpu"]
               + groups["surface_total_water_mass"]["cpu"])
        gpu = (groups["water_mass"]["gpu"]
               + groups["surface_total_water_mass"]["gpu"])
        difference = gpu - cpu
        records.append({
            "domain": domain, "lead_seconds": lead_seconds,
            "kind": "resolved_water_plus_gridscale_precipitation",
            "units": "kg",
            "cpu": cpu, "gpu": gpu, "difference": difference,
            "relative_difference": difference / abs(cpu) if cpu else None,
        })
    return records


def _inventory_units(kind: str) -> str:
    if kind == "particle_number":
        return "#"
    if kind == "particle_volume":
        return "m3"
    return "kg"


def conservation_changes(items: Sequence[Mapping[str, object]]
                         ) -> list[dict[str, object]]:
    """Report trajectory changes relative to each side's exact lead-zero state."""
    inventories = [item for item in items if "kind" in item]
    baselines: dict[tuple[str, str], Mapping[str, object]] = {}
    for item in inventories:
        key = (str(item["domain"]), str(item["kind"]))
        if int(item["lead_seconds"]) == 0:
            if key in baselines:
                raise ComparisonError(f"duplicate inventory baseline for {key}")
            baselines[key] = item
    records = []
    for item in inventories:
        key = (str(item["domain"]), str(item["kind"]))
        if key not in baselines:
            raise ComparisonError(f"inventory series has no lead-zero baseline: {key}")
        baseline = baselines[key]
        cpu_change = float(item["cpu"]) - float(baseline["cpu"])
        gpu_change = float(item["gpu"]) - float(baseline["gpu"])
        records.append({
            "domain": item["domain"], "lead_seconds": item["lead_seconds"],
            "kind": item["kind"], "units": _inventory_units(str(item["kind"])),
            "cpu_change_from_initial": cpu_change,
            "gpu_change_from_initial": gpu_change,
            "change_difference": gpu_change - cpu_change,
            "note": (
                "diagnostic only: lateral forcing and nesting mean an "
                "individual domain is not assumed closed"),
        })
    return records


def _projection_identity(dataset) -> dict[str, object]:
    values = {}
    for name in (
            "MAP_PROJ", "TRUELAT1", "TRUELAT2", "STAND_LON",
            "MOAD_CEN_LAT", "POLE_LAT", "POLE_LON"):
        if not hasattr(dataset, name):
            raise ComparisonError(f"wrfout lacks projection global {name}")
        value = getattr(dataset, name)
        if name == "MAP_PROJ":
            values[name] = int(value)
        else:
            # WPS/WRF projection fields are float32.  Compare globals at that
            # exact representation so f4-vs-f8 NetCDF attribute storage does
            # not fabricate a geometry difference.
            values[name] = float(np.float32(value))
    return values


def compare_frame_pair(cpu_path: Path, gpu_path: Path, *,
                       domain: DomainSpec, valid_time: datetime,
                       policy: Mapping[str, object], chunk_values: int,
                       temporary_root: Path,
                       ) -> tuple[dict[str, object], list[dict[str, object]],
                                  list[dict[str, object]],
                                  list[dict[str, object]]]:
    lead_seconds = int((valid_time - START_TIME).total_seconds())
    with netCDF4.Dataset(cpu_path) as cpu_ds, netCDF4.Dataset(gpu_path) as gpu_ds:
        cpu_schema = inspect_frame(
            cpu_ds, side="cpu", domain=domain, valid_time=valid_time)
        gpu_schema = inspect_frame(
            gpu_ds, side="gpu", domain=domain, valid_time=valid_time)
        if set(cpu_schema) != set(gpu_schema):
            only_cpu = sorted(set(cpu_schema) - set(gpu_schema))
            only_gpu = sorted(set(gpu_schema) - set(cpu_schema))
            # Reflectivity is optional, but scientific scoring requires both.
            if only_cpu != ["composite_reflectivity"] \
                    and only_gpu != ["composite_reflectivity"] \
                    and (only_cpu or only_gpu):
                raise ComparisonError(
                    f"{domain.name}/{valid_time} field schemas differ: "
                    f"cpu_only={only_cpu}, gpu_only={only_gpu}")
        geometry_residuals = []
        for name in (
                "XLAT", "XLONG", "XLAT_U", "XLONG_U", "XLAT_V", "XLONG_V",
                "MAPFAC_M", "MAPFAC_U", "MAPFAC_V", "ZNU", "ZNW"):
            geometry_residuals.append(_compare_geometry_auxiliary(
                cpu_ds, gpu_ds, name, chunk_values=chunk_values,
                label=f"{domain.name}/{lead_seconds}"))
        cpu_projection = _projection_identity(cpu_ds)
        gpu_projection = _projection_identity(gpu_ds)
        if cpu_projection != gpu_projection:
            raise ComparisonError(
                f"{domain.name}/{lead_seconds} projection globals differ: "
                f"cpu={cpu_projection}, gpu={gpu_projection}")

        cpu_weights = _load_inventory_weights(cpu_ds, domain, "cpu")
        gpu_weights = _load_inventory_weights(gpu_ds, domain, "gpu")
        cpu_area = _surface_cell_area(cpu_ds, domain, "cpu")
        gpu_area = _surface_cell_area(gpu_ds, domain, "gpu")
        field_results = []
        for field in FIELD_SPECS:
            field_results.append(compare_field(
                cpu_ds.variables[field.cpu_name],
                gpu_ds.variables[field.gpu_name], field,
                domain=domain, lead_seconds=lead_seconds,
                percentiles=[float(value) for value in
                             policy["percentiles_absolute_error"]],
                negative_epsilon=float(
                    policy["health"]["nonnegative_epsilon"]),
                chunk_values=chunk_values, temporary_root=temporary_root,
                cpu_weights=cpu_weights, gpu_weights=gpu_weights,
                cpu_area=cpu_area, gpu_area=gpu_area))
        event_results, support = _frame_surface_events(
            cpu_ds, gpu_ds, policy=policy, domain=domain,
            lead_seconds=lead_seconds, chunk_values=chunk_values)
        structure = {
            "domain": domain.name, "lead_seconds": lead_seconds,
            "valid_time": valid_time.isoformat(),
            "cpu_schema": cpu_schema, "gpu_schema": gpu_schema,
            "projection": cpu_projection,
            "geometry_residuals": geometry_residuals,
            "exact_geometry_authority": {
                "dimensions_topology_dx_dy_dt_znu_znw": "exact",
                "projection_globals": "exact after canonical WRF float32",
                "coordinate_max_abs_degrees": COORD_ABS_TOL_DEG,
                "map_factor_max_relative": MAPFAC_REL_TOL,
                "projection_oracle": "tests/test_lambert.py",
            },
        }
    inventories = _aggregate_inventory(
        field_results, domain.name, lead_seconds)
    return structure, field_results, event_results, support + inventories


def _scientific_summary(rows: Sequence[Mapping[str, object]],
                        events: Sequence[Mapping[str, object]],
                        policy_status: str) -> dict[str, object]:
    labels = [str(row["evaluation"]) for row in rows]
    event_labels = [str(row["evaluation"]) for row in events]
    counts = {label: labels.count(label) for label in sorted(set(labels))}
    counts.update({f"event_{label}": event_labels.count(label)
                   for label in sorted(set(event_labels))})
    if policy_status == "candidate-unratified":
        verdict = "UNRATIFIED_CANDIDATE_RESULTS_ONLY"
    elif "fail" in labels or "fail" in event_labels:
        verdict = "FAIL"
    elif "pass" not in labels and "pass" not in event_labels:
        verdict = "NO_RATIFIED_SCIENTIFIC_GATES_EVALUATED"
    else:
        verdict = "PASS_UNDER_EXPLICIT_RATIFIED_ROWS"
    return {
        "policy_status": policy_status, "verdict": verdict,
        "evaluation_counts": counts,
        "structural_pass_is_independent": True,
    }


def compare_runs(cpu_root: Path, gpu_root: Path, policy_path: Path,
                 registration_path: Path, *, chunk_values: int,
                 source_identity: Mapping[str, object] | None = None,
                 ) -> dict[str, object]:
    if chunk_values < 1:
        raise ComparisonError("chunk_values must be positive")
    policy, policy_identity = load_policy(policy_path)
    registration = load_json(registration_path, "comparison registration")
    source = dict(source_identity or repository_identity())
    cpu_frames = discover_frames(cpu_root)
    gpu_frames = discover_frames(gpu_root)
    cpu_inventory = _logical_inventory(cpu_frames)
    gpu_inventory = _logical_inventory(gpu_frames)
    registration_binding = validate_registration(
        registration, policy_identity=policy_identity,
        source_identity=source, cpu_inventory=cpu_inventory)
    gpu_provenance = validate_gpu_manifests(
        gpu_root.resolve(), registration, source, gpu_inventory,
        registration_path.resolve(), policy_identity)

    structures = []
    field_results = []
    event_results = []
    support_and_inventories = []
    with tempfile.TemporaryDirectory(prefix="gpuwm-nssl2-compare-") as temporary:
        temporary_root = Path(temporary)
        for domain_name, times in expected_calendar().items():
            domain = DOMAIN_BY_NAME[domain_name]
            for valid in times:
                structure, fields, events, other = compare_frame_pair(
                    cpu_frames[(domain_name, valid)],
                    gpu_frames[(domain_name, valid)], domain=domain,
                    valid_time=valid, policy=policy,
                    chunk_values=chunk_values, temporary_root=temporary_root)
                structures.append(structure)
                field_results.extend(fields)
                event_results.extend(events)
                support_and_inventories.extend(other)
    metric_rows = [row for result in field_results
                   for row in apply_scientific_policy(result, policy)]
    conservation = conservation_changes(support_and_inventories)
    report_core = {
        "schema": REPORT_SCHEMA,
        "generated_at_utc": utc_now(),
        "structural_status": "PASS",
        "structural_gates": [
            {"gate": "preregistration_binding", "passed": True},
            {"gate": "cpu_and_gpu_exact_64_file_calendars", "passed": True},
            {"gate": "gpu_launch_and_completion_binding", "passed": True},
            {"gate": "times_domain_geometry_projection", "passed": True},
            {"gate": "native_alias_units_dimensions_stagger", "passed": True},
            {"gate": "finite_nonnegative_health", "passed": True},
        ],
        "calendar": _calendar_summary(),
        "source": source,
        "evaluator": file_identity(Path(__file__)),
        "registration": file_identity(registration_path),
        "registration_binding_sha256": registration["binding_sha256"],
        "policy": policy,
        "policy_identity": policy_identity,
        "gpu_provenance": gpu_provenance,
        "field_aliases": registration_binding["field_aliases"],
        "rejected_legacy_foreign_cpu_aliases": (
            registration_binding["rejected_legacy_foreign_cpu_aliases"]),
        "streaming": {
            "chunk_values": chunk_values,
            "percentile_method": (
                "exact nearest-rank order statistics in one field-sized "
                "disk-backed float64 memmap; NetCDF is read in slabs"),
            "whole_wrfout_loaded": False,
        },
        "cpu_outputs": cpu_inventory,
        "gpu_outputs": gpu_inventory,
        "structure": structures,
        "field_results": field_results,
        "surface_event_results": event_results,
        "support_and_inventory_summaries": support_and_inventories,
        "conservation_change_summaries": conservation,
        "scientific_metric_rows": metric_rows,
        "scientific_summary": _scientific_summary(
            metric_rows, event_results, str(policy["status"])),
    }
    report_core["report_fingerprint"] = stable_hash(report_core)
    return report_core


_TSV_COLUMNS = (
    "category", "domain", "lead_seconds", "field_or_event",
    "event_threshold", "event_threshold_units", "metric", "value",
    "value_units", "threshold_operator", "policy_threshold",
    "policy_threshold_units", "evaluation",
)


def _tsv_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for metric in report["scientific_metric_rows"]:
        rows.append({
            "category": "continuous", "domain": metric["domain"],
            "lead_seconds": metric["lead_seconds"],
            "field_or_event": metric["field"], "event_threshold": "",
            "event_threshold_units": "", "metric": metric["metric"],
            "value": metric["value"],
            "value_units": metric.get("value_units", ""),
            "threshold_operator": metric["threshold_operator"] or "",
            "policy_threshold": (
                "" if metric["threshold"] is None else metric["threshold"]),
            "policy_threshold_units": metric.get("threshold_units") or "",
            "evaluation": metric["evaluation"],
        })
    event_metric_names = (
        "fss", "cpu_event_coverage", "gpu_event_coverage", "hits", "misses",
        "false_alarms", "correct_negatives", "probability_of_detection",
        "false_alarm_ratio", "critical_success_index", "frequency_bias",
        "equitable_threat_score", "heidke_skill_score",
    )
    for event in report["surface_event_results"]:
        for metric_name in event_metric_names:
            rows.append({
                "category": "surface_event", "domain": event["domain"],
                "lead_seconds": event["lead_seconds"],
                "field_or_event": event["event"],
                "event_threshold": event["threshold"],
                "event_threshold_units": event["threshold_units"],
                "metric": metric_name, "value": event.get(metric_name),
                "value_units": (
                    "count" if metric_name in {
                        "hits", "misses", "false_alarms", "correct_negatives"
                    } else "1"),
                "threshold_operator": (
                    ">=" if metric_name == "fss"
                    and event["minimum_fss"] is not None else ""),
                "policy_threshold": (
                    event["minimum_fss"] if metric_name == "fss"
                    and event["minimum_fss"] is not None else ""),
                "policy_threshold_units": (
                    "1" if metric_name == "fss"
                    and event["minimum_fss"] is not None else ""),
                "evaluation": (
                    event["evaluation"] if metric_name == "fss"
                    else "report_only"),
            })
    for item in report["support_and_inventory_summaries"]:
        if "kind" not in item:
            continue
        for metric_name in ("cpu", "gpu", "difference", "relative_difference"):
            rows.append({
                "category": "inventory", "domain": item["domain"],
                "lead_seconds": item["lead_seconds"],
                "field_or_event": item["kind"], "event_threshold": "",
                "event_threshold_units": "", "metric": metric_name,
                "value": item[metric_name], "threshold_operator": "",
                "value_units": (
                    "1" if metric_name == "relative_difference"
                    else _inventory_units(str(item["kind"]))),
                "policy_threshold": "", "policy_threshold_units": "",
                "evaluation": "report_only",
            })
    for item in report["conservation_change_summaries"]:
        for metric_name in (
                "cpu_change_from_initial", "gpu_change_from_initial",
                "change_difference"):
            rows.append({
                "category": "conservation_change",
                "domain": item["domain"],
                "lead_seconds": item["lead_seconds"],
                "field_or_event": item["kind"], "event_threshold": "",
                "event_threshold_units": "", "metric": metric_name,
                "value": item[metric_name], "threshold_operator": "",
                "value_units": _inventory_units(str(item["kind"])),
                "policy_threshold": "", "policy_threshold_units": "",
                "evaluation": "report_only",
            })
    return rows


def _render_tsv(report: Mapping[str, object]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream, fieldnames=_TSV_COLUMNS, dialect="excel-tab",
        lineterminator="\n", extrasaction="raise")
    writer.writeheader()
    for raw in _tsv_rows(report):
        row = {
            key: ("" if value is None else value) for key, value in raw.items()
        }
        writer.writerow(row)
    return stream.getvalue()


def write_evidence(output_dir: Path, report: Mapping[str, object]) -> dict[str, object]:
    output_dir = output_dir.resolve()
    _require_outside_repository(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "comparison-report.json"
    tsv_path = output_dir / "comparison-metrics.tsv"
    manifest_path = output_dir / "evidence-manifest.json"
    existing = [path for path in (report_path, tsv_path, manifest_path)
                if path.exists()]
    if existing:
        raise ComparisonError(
            f"refusing to overwrite comparison evidence: {existing}")
    atomic_json(report_path, report)
    atomic_text(tsv_path, _render_tsv(report))
    artifacts = [
        file_identity(report_path, logical_name=report_path.name),
        file_identity(tsv_path, logical_name=tsv_path.name),
    ]
    binding = {
        "report_fingerprint": report["report_fingerprint"],
        "registration_binding_sha256": report["registration_binding_sha256"],
        "policy_payload_sha256": report["policy_identity"]["payload_sha256"],
        "artifacts": artifacts,
    }
    manifest = {
        "schema": EVIDENCE_SCHEMA, "created_at_utc": utc_now(),
        "binding_sha256": stable_hash(binding), "binding": binding,
    }
    atomic_json(manifest_path, manifest)
    return {
        "report": file_identity(report_path),
        "metrics_tsv": file_identity(tsv_path),
        "manifest": file_identity(manifest_path),
        "evidence_binding_sha256": manifest["binding_sha256"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    register = subparsers.add_parser(
        "register", help="freeze policy/evaluator/CPU identities before GPU launch")
    register.add_argument("--cpu-root", type=Path, required=True)
    register.add_argument("--policy", type=Path, required=True)
    register.add_argument("--output", type=Path, required=True)

    compare = subparsers.add_parser(
        "compare", help="stream the bound CPU/GPU 64-frame comparison")
    compare.add_argument("--cpu-root", type=Path, required=True)
    compare.add_argument("--gpu-root", type=Path, required=True)
    compare.add_argument("--policy", type=Path, required=True)
    compare.add_argument("--registration", type=Path, required=True)
    compare.add_argument("--output-dir", type=Path, required=True)
    compare.add_argument(
        "--chunk-values", type=int, default=DEFAULT_CHUNK_VALUES,
        help="maximum decoded float64 values per NetCDF slab")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "register":
        write_registration(args.cpu_root, args.policy, args.output)
        print(json.dumps({
            "status": "REGISTERED", "registration": str(args.output.resolve()),
            "sha256": sha256_file(args.output.resolve()),
        }, sort_keys=True))
        return 0
    report = compare_runs(
        args.cpu_root, args.gpu_root, args.policy, args.registration,
        chunk_values=args.chunk_values)
    evidence = write_evidence(args.output_dir, report)
    print(json.dumps({
        "structural_status": report["structural_status"],
        "scientific_verdict": report["scientific_summary"]["verdict"],
        "evidence": evidence,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
