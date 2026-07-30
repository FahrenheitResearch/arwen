"""Pre-registered, shared metrics for the N5S matched-physics shadow.

The same functions score every CPU-WRF member pair and the gpuwm candidate
against unperturbed WRF.  Scoring is CPU-only and reads WRF-compatible
NetCDF history frames.  A complete, hash-checked registration is mandatory
on both sides of every comparison.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import itertools
import json
import math
from pathlib import Path
import re
import subprocess
from typing import Mapping, Sequence

import netCDF4
import numpy as np

from gpuwm.verify.n5s_common import (
    restored_input_sha256, sha256_file, stable_hash, write_json,
)


N5S_DOMAINS = ("d01", "d02", "d03", "d04")
N5S_STATE_FIELDS = ("U", "V", "W", "T", "PH", "MU", "QVAPOR")
N5S_DX_METERS = {
    "d01": 12000.0, "d02": 3000.0, "d03": 1000.0,
    "d04": 1000.0 / 3.0,
}
N5S_WRF_BUILD = {
    "wrf_version": "v4.6.1",
    "microphysics": "Morrison",
    "pbl": "YSU",
    "instrumented_build": "T8b",
}

_FRAME_RE = re.compile(
    r"^wrfout_(d0[1-4])_(\d{4}-\d{2}-\d{2})_(\d{2})[_:](\d{2})[_:](\d{2})(?:\..*)?$")
_REGISTRATION_KEYS = {
    "schema", "gate", "evaluator_commit", "start_time",
    "run_duration_seconds", "cadence_seconds", "cadence",
    "expected_samples", "parameters", "mask_parameter_hash",
}


def evaluator_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "0" * 40


def make_registration(*, run_minutes: int = 30, history_minutes: int = 5,
                      commit: str | None = None,
                      start_time: str = "1974-04-03T12:00:00",
                      parameters: Mapping[str, object] | None = None
                      ) -> dict[str, object]:
    """Create all metric/mask pins before any output is inspected."""
    run_seconds = int(run_minutes) * 60
    cadence_seconds = int(history_minutes) * 60
    if not 30 <= run_minutes <= 75:
        raise ValueError("N5S run_minutes must be in the registered 30..75 range")
    if history_minutes <= 0 or run_seconds % cadence_seconds:
        raise ValueError("history cadence must divide the N5S duration exactly")
    defaults: dict[str, object] = {
        "domains": list(N5S_DOMAINS),
        "state_fields": list(N5S_STATE_FIELDS),
        "domain_dx_m": dict(N5S_DX_METERS),
        "low_pass_filter": "separable edge-extended square boxcar",
        "low_pass_physical_width_m": 6000.0,
        "low_pass_interior_exclusion_cells": 5,
        "boundary_increment_definition": (
            "RMSE of (frame[t]-frame[t-cadence]) differences between runs "
            "over the outer spec_bdy_width cells"
        ),
        "spec_bdy_width_cells": 5,
        "reflectivity_threshold_dbz": 40.0,
        "reflectivity_composite": "vertical maximum of REFL_10CM",
        "fss_neighborhood_shape": "edge-extended square",
        "fss_neighborhood_half_width_m": 5000.0,
        "fss_zero_denominator": "FSS=1 when both fractions are identically zero",
        "object_connectivity": 8,
        "object_min_area_km2": 25.0,
        "object_timing_statistic": "first qualifying composite-reflectivity object",
        "zero_object_time_seconds": "run_duration_seconds+cadence_seconds",
        "first_scored_second": cadence_seconds,
        "missing_or_nonfinite_policy": "fail",
    }
    if parameters:
        unknown = set(parameters) - set(defaults)
        if unknown:
            raise ValueError(f"unknown N5S registration parameter(s): {sorted(unknown)}")
        tunable = {
            "low_pass_physical_width_m",
            "low_pass_interior_exclusion_cells",
            "spec_bdy_width_cells",
            "reflectivity_threshold_dbz",
            "fss_neighborhood_half_width_m",
            "object_connectivity",
            "object_min_area_km2",
        }
        changed_immutable = sorted(
            key for key, value in parameters.items()
            if key not in tunable and value != defaults[key])
        if changed_immutable:
            raise ValueError(
                "N5S implementation pins may not be overridden: "
                f"{changed_immutable}")
        defaults.update(dict(parameters))
    if (not math.isfinite(float(defaults["low_pass_physical_width_m"]))
            or float(defaults["low_pass_physical_width_m"]) <= 0.0
            or int(defaults["low_pass_interior_exclusion_cells"]) < 0
            or int(defaults["spec_bdy_width_cells"]) < 1
            or not math.isfinite(float(defaults["reflectivity_threshold_dbz"]))
            or not math.isfinite(float(
                defaults["fss_neighborhood_half_width_m"]))
            or float(defaults["fss_neighborhood_half_width_m"]) < 0.0
            or int(defaults["object_connectivity"]) not in (4, 8)
            or not math.isfinite(float(defaults["object_min_area_km2"]))
            or float(defaults["object_min_area_km2"]) <= 0.0):
        raise ValueError("N5S tunable metric parameters are invalid")
    frames = run_seconds // cadence_seconds
    expected = {
        "post_initial_frames_per_domain": frames,
        "boundary_intervals_per_domain": frames,
        "low_pass_samples_per_domain_field": frames,
        "d04_reflectivity_samples": frames,
        "storm_timing_samples_per_domain": 1,
    }
    return {
        "schema": 1,
        "gate": "N5S_matched_physics_wrf_shadow",
        "evaluator_commit": commit or evaluator_commit(),
        "start_time": start_time,
        "run_duration_seconds": run_seconds,
        "cadence_seconds": cadence_seconds,
        "cadence": (
            f"history_interval_seconds={cadence_seconds};"
            f"first_scored_second={cadence_seconds}"
        ),
        "expected_samples": expected,
        "parameters": defaults,
        "mask_parameter_hash": stable_hash(defaults),
    }


def validate_registration(registration: Mapping[str, object]
                          ) -> dict[str, object]:
    missing = _REGISTRATION_KEYS - set(registration)
    if missing:
        raise ValueError(f"N5S registration is missing pins: {sorted(missing)}")
    reg = dict(registration)
    if reg["schema"] != 1 or reg["gate"] != "N5S_matched_physics_wrf_shadow":
        raise ValueError("N5S registration schema/gate mismatch")
    commit = reg["evaluator_commit"]
    if (not isinstance(commit, str) or len(commit) != 40
            or any(ch not in "0123456789abcdef" for ch in commit.lower())):
        raise ValueError("N5S evaluator_commit must be a 40-digit Git SHA")
    duration = int(reg["run_duration_seconds"])
    cadence = int(reg["cadence_seconds"])
    if duration < 1800 or cadence <= 0 or duration % cadence:
        raise ValueError("N5S duration/cadence pins are invalid")
    parameters = reg["parameters"]
    if not isinstance(parameters, dict):
        raise ValueError("N5S parameters pin must be an object")
    if stable_hash(parameters) != reg["mask_parameter_hash"]:
        raise ValueError("N5S mask/parameter hash mismatch")
    rebuilt = make_registration(
        run_minutes=duration // 60, history_minutes=cadence // 60,
        commit=commit, start_time=str(reg["start_time"]),
        parameters=parameters)
    for key in ("cadence", "expected_samples"):
        if reg[key] != rebuilt[key]:
            raise ValueError(f"N5S registration {key} is inconsistent")
    return reg


def load_registration(path: str | Path) -> dict[str, object]:
    with Path(path).open("r", encoding="utf-8") as stream:
        return validate_registration(json.load(stream))


def require_matching_registrations(left: Mapping[str, object],
                                   right: Mapping[str, object]
                                   ) -> dict[str, object]:
    lhs = validate_registration(left)
    rhs = validate_registration(right)
    if lhs != rhs:
        raise ValueError("N5S registrations differ between scoring sides")
    return lhs


def _frame_seconds(path: Path, start: datetime, domain: str) -> int:
    match = _FRAME_RE.match(path.name)
    if match is None or match.group(1) != domain:
        raise ValueError(f"invalid N5S history-frame name: {path.name}")
    valid = datetime.strptime(
        f"{match.group(2)} {match.group(3)}:{match.group(4)}:{match.group(5)}",
        "%Y-%m-%d %H:%M:%S")
    seconds = int((valid - start).total_seconds())
    return seconds


def discover_frames(run_directory: str | Path, domain: str,
                    registration: Mapping[str, object]) -> dict[int, Path]:
    reg = validate_registration(registration)
    root = Path(run_directory)
    start = datetime.fromisoformat(str(reg["start_time"]))
    duration = int(reg["run_duration_seconds"])
    cadence = int(reg["cadence_seconds"])
    expected = set(range(0, duration + 1, cadence))
    found: dict[int, Path] = {}
    for path in root.glob(f"wrfout_{domain}_*"):
        seconds = _frame_seconds(path, start, domain)
        if seconds in found:
            raise ValueError(f"duplicate {domain} N5S frame at {seconds} seconds")
        found[seconds] = path
    if set(found) != expected:
        raise ValueError(
            f"{domain} frame times {sorted(found)} != registered {sorted(expected)}")
    return found


def _read_field(path: Path, field: str) -> np.ndarray:
    with netCDF4.Dataset(path) as dataset:
        if field not in dataset.variables:
            raise ValueError(f"N5S frame {path} is missing {field}")
        variable = dataset.variables[field]
        if variable.ndim < 3 or variable.shape[0] != 1:
            raise ValueError(f"N5S field {field} in {path} has invalid shape")
        value = np.ma.asarray(variable[0])
        if np.ma.isMaskedArray(value) and np.any(np.ma.getmaskarray(value)):
            raise ValueError(f"N5S field {field} in {path} carries masked data")
        result = np.asarray(value, dtype=np.float64)
    if result.size == 0 or not np.all(np.isfinite(result)):
        raise ValueError(f"N5S field {field} in {path} is empty/non-finite")
    return result


def _boxcar(array: np.ndarray, width: int) -> np.ndarray:
    """Odd-width, edge-extended separable boxcar over the last two axes."""
    if width < 1 or width % 2 != 1 or array.ndim < 2:
        raise ValueError("boxcar width must be a positive odd integer")
    if width == 1:
        return np.asarray(array, dtype=np.float64)
    radius = width // 2
    result = np.asarray(array, dtype=np.float64)
    for axis in (-1, -2):
        pads = [(0, 0)] * result.ndim
        pads[axis] = (radius, radius)
        padded = np.pad(result, pads, mode="edge")
        prefix_shape = list(padded.shape)
        prefix_shape[axis] = 1
        prefix = np.concatenate(
            [np.zeros(prefix_shape, dtype=np.float64),
             np.cumsum(padded, axis=axis, dtype=np.float64)], axis=axis)
        high = [slice(None)] * result.ndim
        low = [slice(None)] * result.ndim
        high[axis] = slice(width, None)
        low[axis] = slice(None, -width)
        result = (prefix[tuple(high)] - prefix[tuple(low)]) / float(width)
    return result


def _odd_width(physical_width_m: float, dx_m: float) -> int:
    cells = max(1, int(math.floor(physical_width_m / dx_m + 0.5)))
    return cells if cells % 2 else cells + 1


def rmse(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape or left.size == 0:
        raise ValueError("RMSE operands must have one non-empty common shape")
    difference = np.asarray(left, dtype=np.float64) - np.asarray(
        right, dtype=np.float64)
    if not np.all(np.isfinite(difference)):
        raise ValueError("RMSE operand difference is non-finite")
    return float(np.sqrt(np.mean(difference * difference, dtype=np.float64)))


def _aggregate_rmse(differences: Sequence[np.ndarray]) -> float:
    total = 0.0
    count = 0
    for difference in differences:
        value = np.asarray(difference, dtype=np.float64)
        if value.size == 0 or not np.all(np.isfinite(value)):
            raise ValueError("N5S metric sample is empty/non-finite")
        total += float(np.sum(value * value, dtype=np.float64))
        count += value.size
    if count == 0:
        raise ValueError("N5S metric has no samples")
    return float(math.sqrt(total / count))


def _interior(array: np.ndarray, width: int) -> np.ndarray:
    ny, nx = array.shape[-2:]
    if ny <= 2 * width or nx <= 2 * width:
        raise ValueError("N5S field is too small for the registered interior mask")
    return array[..., width:-width, width:-width]


def _boundary_values(array: np.ndarray, width: int) -> np.ndarray:
    ny, nx = array.shape[-2:]
    if ny <= 2 * width or nx <= 2 * width:
        raise ValueError("N5S field is too small for spec_bdy_width")
    j, i = np.indices((ny, nx))
    mask = np.minimum.reduce((i, nx - 1 - i, j, ny - 1 - j)) < width
    return array[..., mask]


def neighborhood_fraction(events: np.ndarray, half_width_cells: int
                          ) -> np.ndarray:
    if events.ndim != 2 or half_width_cells < 0:
        raise ValueError("FSS events must be 2-D and radius non-negative")
    return _boxcar(events.astype(np.float64), 2 * half_width_cells + 1)


def fss_distance(left: np.ndarray, right: np.ndarray, *, threshold: float,
                 half_width_cells: int) -> float:
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("FSS operands must be common-shape 2-D fields")
    lhs = neighborhood_fraction(left >= threshold, half_width_cells)
    rhs = neighborhood_fraction(right >= threshold, half_width_cells)
    numerator = float(np.sum((lhs - rhs) ** 2, dtype=np.float64))
    denominator = float(np.sum(lhs * lhs + rhs * rhs, dtype=np.float64))
    fss = 1.0 if denominator == 0.0 else 1.0 - numerator / denominator
    distance = 1.0 - fss
    if not math.isfinite(distance):
        raise ValueError("FSS distance is non-finite")
    return float(min(1.0, max(0.0, distance)))


def _composite(reflectivity: np.ndarray) -> np.ndarray:
    if reflectivity.ndim == 2:
        return reflectivity
    if reflectivity.ndim == 3:
        return np.max(reflectivity, axis=0)
    raise ValueError("REFL_10CM must be 2-D or 3-D after removing Time")


def has_qualifying_object(composite: np.ndarray, *, threshold: float,
                          min_cells: int, connectivity: int = 8) -> bool:
    """NumPy-only connected-component test with deterministic scan order."""
    if composite.ndim != 2 or min_cells < 1 or connectivity not in (4, 8):
        raise ValueError("invalid storm-object labeling parameters")
    active = np.asarray(composite >= threshold, dtype=bool)
    seen = np.zeros_like(active)
    ny, nx = active.shape
    offsets = [(-1, 0), (0, -1), (0, 1), (1, 0)]
    if connectivity == 8:
        offsets += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    for j0 in range(ny):
        for i0 in range(nx):
            if not active[j0, i0] or seen[j0, i0]:
                continue
            seen[j0, i0] = True
            stack = [(j0, i0)]
            count = 0
            while stack:
                j, i = stack.pop()
                count += 1
                if count >= min_cells:
                    return True
                for dj, di in offsets:
                    jj, ii = j + dj, i + di
                    if (0 <= jj < ny and 0 <= ii < nx and active[jj, ii]
                            and not seen[jj, ii]):
                        seen[jj, ii] = True
                        stack.append((jj, ii))
    return False


def first_object_time(frames: Mapping[int, np.ndarray], *, threshold: float,
                      min_cells: int, duration_seconds: int,
                      cadence_seconds: int, connectivity: int = 8) -> float:
    for seconds in sorted(frames):
        if seconds == 0:
            continue
        if has_qualifying_object(
                _composite(frames[seconds]), threshold=threshold,
                min_cells=min_cells, connectivity=connectivity):
            return float(seconds)
    return float(duration_seconds + cadence_seconds)


def score_run_pair(left_directory: str | Path, right_directory: str | Path,
                   left_registration: Mapping[str, object],
                   right_registration: Mapping[str, object], *,
                   _verified_run_directories: set[Path] | None = None
                   ) -> dict[str, float]:
    """Score one like-for-like run pair under identical preregistered pins."""
    reg = require_matching_registrations(left_registration, right_registration)
    left_directory = Path(left_directory).resolve()
    right_directory = Path(right_directory).resolve()
    for directory in (left_directory, right_directory):
        require_matching_registrations(reg, _read_sidecar(directory))
        if (_verified_run_directories is None
                or directory not in _verified_run_directories):
            verify_run_artifact(directory, reg)
            if _verified_run_directories is not None:
                _verified_run_directories.add(directory)
    params = reg["parameters"]
    duration = int(reg["run_duration_seconds"])
    cadence = int(reg["cadence_seconds"])
    score_times = range(cadence, duration + 1, cadence)
    boundary_width = int(params["spec_bdy_width_cells"])
    low_pass_exclusion = int(params["low_pass_interior_exclusion_cells"])
    scores: dict[str, float] = {}
    for domain in N5S_DOMAINS:
        left_frames = discover_frames(left_directory, domain, reg)
        right_frames = discover_frames(right_directory, domain, reg)
        dx = float(params["domain_dx_m"][domain])
        filter_width = _odd_width(
            float(params["low_pass_physical_width_m"]), dx)
        for field in N5S_STATE_FIELDS:
            low_pass_differences = []
            boundary_differences = []
            for seconds in score_times:
                lhs = _read_field(left_frames[seconds], field)
                rhs = _read_field(right_frames[seconds], field)
                if lhs.shape != rhs.shape:
                    raise ValueError(f"N5S {domain}/{field} shapes differ")
                filtered = _boxcar(lhs, filter_width) - _boxcar(
                    rhs, filter_width)
                low_pass_differences.append(
                    _interior(filtered, low_pass_exclusion))
                lhs_prev = _read_field(left_frames[seconds - cadence], field)
                rhs_prev = _read_field(right_frames[seconds - cadence], field)
                if lhs_prev.shape != lhs.shape or rhs_prev.shape != rhs.shape:
                    raise ValueError(f"N5S {domain}/{field} shape changes in time")
                increment_error = (lhs - lhs_prev) - (rhs - rhs_prev)
                boundary_differences.append(
                    _boundary_values(increment_error, boundary_width))
            scores[f"low_pass_state_rmse:{domain}:{field}"] = (
                _aggregate_rmse(low_pass_differences))
            scores[f"applied_boundary_increment_error:{domain}:{field}"] = (
                _aggregate_rmse(boundary_differences))

        reflectivity_left = {
            seconds: _read_field(left_frames[seconds], "REFL_10CM")
            for seconds in score_times}
        reflectivity_right = {
            seconds: _read_field(right_frames[seconds], "REFL_10CM")
            for seconds in score_times}
        threshold = float(params["reflectivity_threshold_dbz"])
        min_cells = max(1, int(math.ceil(
            float(params["object_min_area_km2"]) * 1.0e6 / (dx * dx))))
        left_timing = first_object_time(
            reflectivity_left, threshold=threshold, min_cells=min_cells,
            duration_seconds=duration, cadence_seconds=cadence,
            connectivity=int(params["object_connectivity"]))
        right_timing = first_object_time(
            reflectivity_right, threshold=threshold, min_cells=min_cells,
            duration_seconds=duration, cadence_seconds=cadence,
            connectivity=int(params["object_connectivity"]))
        scores[f"storm_object_timing_difference:{domain}:first_object"] = abs(
            left_timing - right_timing)
        if domain == "d04":
            half_width = max(0, int(math.floor(
                float(params["fss_neighborhood_half_width_m"]) / dx + 0.5)))
            distances = [
                fss_distance(
                    _composite(reflectivity_left[seconds]),
                    _composite(reflectivity_right[seconds]),
                    threshold=threshold, half_width_cells=half_width)
                for seconds in score_times
            ]
            scores[
                "d04_reflectivity_fss_distance:d04:REFL_10CM_40DBZ"
            ] = float(np.mean(distances, dtype=np.float64))
    if not scores or any(not math.isfinite(value) or value < 0.0
                         for value in scores.values()):
        raise ValueError("N5S scoring produced missing/invalid distances")
    return scores


def _read_sidecar(run_directory: Path) -> dict[str, object]:
    path = run_directory / "n5s-preregistration.json"
    if not path.is_file():
        raise ValueError(f"N5S run is missing preregistration sidecar: {path}")
    return load_registration(path)


def _require_success(run_directory: Path) -> int:
    candidates = (run_directory / "exit.status",
                  run_directory / "exit_status.txt")
    status_path = next((path for path in candidates if path.is_file()), None)
    if status_path is None:
        raise ValueError(f"N5S run has no exit-status record: {run_directory}")
    try:
        status = int(status_path.read_text(encoding="utf-8").strip())
    except ValueError as exc:
        raise ValueError(f"invalid N5S exit status in {status_path}") from exc
    if status != 0:
        raise ValueError(f"N5S run failed with exit status {status}: {run_directory}")
    return status


def _frame_inventory(run_directory: Path, registration: Mapping[str, object]
                     ) -> list[dict[str, object]]:
    _require_success(run_directory)
    inventory: list[dict[str, object]] = []
    for domain in N5S_DOMAINS:
        for seconds, path in sorted(
                discover_frames(run_directory, domain, registration).items()):
            inventory.append({
                "domain": domain, "seconds": seconds,
                "filename": path.name,
                "byte_length": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    return inventory


def verify_run_artifact(run_directory: str | Path,
                        registration: Mapping[str, object] | None = None
                        ) -> dict[str, object]:
    """Re-hash every frame and require exact agreement with its run artifact."""
    run_directory = Path(run_directory).resolve()
    record_path = run_directory / "n5s-run-artifact.json"
    if not record_path.is_file():
        raise ValueError(f"N5S run has no frame-inventory artifact: {record_path}")
    try:
        with record_path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid N5S run artifact: {record_path}") from exc
    if not isinstance(payload, dict) or payload.get("schema") != 2:
        raise ValueError(f"invalid N5S run artifact schema: {record_path}")
    embedded = payload.get("registration")
    if not isinstance(embedded, dict):
        raise ValueError(f"N5S run artifact lacks registration: {record_path}")
    reg = validate_registration(embedded)
    if registration is not None:
        reg = require_matching_registrations(reg, registration)
    require_matching_registrations(reg, _read_sidecar(run_directory))
    expected = _frame_inventory(run_directory, reg)
    if payload.get("frames") != expected:
        raise ValueError(
            f"N5S run frame inventory does not match artifact: {run_directory}")
    return payload


def _artifact(run_directory: Path, artifact_root: Path, *, run_id: str,
              registration: Mapping[str, object]) -> dict[str, str]:
    run_directory = run_directory.resolve()
    artifact_root = artifact_root.resolve()
    if not run_directory.is_relative_to(artifact_root):
        raise ValueError("N5S run artifact escapes its evidence root")
    _require_success(run_directory)
    record_path = run_directory / "n5s-run-artifact.json"
    if record_path.exists():
        payload = verify_run_artifact(run_directory, registration)
        if payload.get("id") != run_id:
            raise ValueError(f"N5S run artifact id mismatch: {record_path}")
    else:
        write_json(record_path, {
            "schema": 2, "id": run_id,
            "registration": dict(validate_registration(registration)),
            "frames": _frame_inventory(run_directory, registration),
        })
    return {
        "relative_path": record_path.relative_to(artifact_root).as_posix(),
        "sha256": sha256_file(record_path),
    }


def _require_ensemble_success_ledger(root: Path,
                                     run_directories: Sequence[Path]) -> None:
    """Validate the runner ledger before any history file is inspected."""
    ledger_path = root / "exit-status-ledger.json"
    if not ledger_path.is_file():
        raise ValueError(f"N5S ensemble has no exit-status ledger: {ledger_path}")
    try:
        with ledger_path.open("r", encoding="utf-8") as stream:
            ledger = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid N5S exit-status ledger: {ledger_path}") from exc
    if not isinstance(ledger, dict):
        raise ValueError(f"invalid N5S exit-status ledger schema: {ledger_path}")
    rows = ledger.get("runs")
    if ledger.get("schema") != 1 or not isinstance(rows, list):
        raise ValueError(f"invalid N5S exit-status ledger schema: {ledger_path}")
    expected_ids = {directory.name for directory in run_directories}
    by_id = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            raise ValueError("N5S exit-status ledger has an invalid run row")
        if row["id"] in by_id:
            raise ValueError(f"N5S exit-status ledger repeats {row['id']}")
        by_id[row["id"]] = row
    if set(by_id) != expected_ids:
        raise ValueError(
            f"N5S exit-status ledger runs {sorted(by_id)} != "
            f"expected {sorted(expected_ids)}")
    for directory in run_directories:
        status = _require_success(directory)
        ledger_status = by_id[directory.name].get("exit_status")
        if (isinstance(ledger_status, bool) or not isinstance(ledger_status, int)
                or ledger_status != status):
            raise ValueError(
                f"N5S exit-status ledger mismatch for {directory.name}")
    if ledger.get("all_succeeded") is not True:
        raise ValueError("N5S exit-status ledger reports a failed run")


def _artifact_durations(artifact: Mapping[str, object]) -> dict[str, int]:
    frames = artifact.get("frames")
    if not isinstance(frames, list):
        raise ValueError("N5S run artifact has no frame inventory")
    by_domain = {domain: [] for domain in N5S_DOMAINS}
    for frame in frames:
        if not isinstance(frame, dict) or frame.get("domain") not in by_domain:
            raise ValueError("N5S run artifact has an invalid frame record")
        by_domain[str(frame["domain"])].append(int(frame["seconds"]))
    if any(not seconds for seconds in by_domain.values()):
        raise ValueError("N5S run artifact omits a domain")
    return {domain: max(seconds) for domain, seconds in by_domain.items()}


def _resolve_ensemble_run_directory(root: Path,
                                    path: str | Path) -> Path:
    """Resolve a run path under ``root``, including a bare member ID.

    The CLI's explicit/default paths may already be relative to the process
    working directory (for example ``out/runs/unperturbed``), while a bare
    ``--member member-00`` is relative to the resolved ensemble root.  Test
    the former interpretation only when it is already contained; otherwise
    use root-relative semantics.  Resolving before ``relative_to`` also makes
    junction/symlink traversal and Windows case normalization part of the
    containment decision.
    """
    supplied = Path(path)
    if supplied.is_absolute():
        directory = supplied.resolve()
    else:
        cwd_relative = supplied.resolve()
        try:
            cwd_relative.relative_to(root)
        except ValueError:
            directory = (root / supplied).resolve()
        else:
            directory = cwd_relative
    try:
        directory.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            "N5S run directory escapes the ensemble root") from exc
    return directory


def build_ensemble_evidence(*, ensemble_root: str | Path,
                            unperturbed_directory: str | Path,
                            member_directories: Sequence[str | Path],
                            registration: Mapping[str, object],
                            restored_inputs: str | Path,
                            output: str | Path | None = None
                            ) -> dict[str, object]:
    root = Path(ensemble_root).resolve()
    unperturbed = _resolve_ensemble_run_directory(
        root, unperturbed_directory)
    members = [
        _resolve_ensemble_run_directory(root, path)
        for path in member_directories]
    reg = validate_registration(registration)
    if len(members) < 3:
        raise ValueError("N5S ensemble needs at least three perturbed members")
    run_directories = [unperturbed, *members]
    require_matching_registrations(reg, _read_sidecar(root))
    _require_ensemble_success_ledger(root, run_directories)
    for directory in run_directories:
        require_matching_registrations(reg, _read_sidecar(directory))
    member_records = []
    for index, directory in enumerate(members):
        perturb_path = directory / "perturbation.json"
        if not perturb_path.is_file():
            raise ValueError(f"N5S member lacks perturbation record: {directory}")
        with perturb_path.open("r", encoding="utf-8") as stream:
            perturbation = json.load(stream)
        if not perturbation:
            raise ValueError("N5S perturbation record may not be empty")
        artifact = _artifact(
            directory, root, run_id=directory.name, registration=reg)
        member_records.append({
            "id": directory.name or f"member-{index}", **artifact,
            "one_ulp_perturbation": perturbation,
        })
    unperturbed_artifact = _artifact(
        unperturbed, root, run_id="unperturbed", registration=reg)
    verified_runs = set(run_directories)
    pair_distances: dict[str, list[float]] = {}
    for left, right in itertools.combinations(members, 2):
        scores = score_run_pair(
            left, right, reg, reg,
            _verified_run_directories=verified_runs)
        if not pair_distances:
            pair_distances = {metric: [] for metric in scores}
        if set(scores) != set(pair_distances):
            raise ValueError("N5S CPU pair metric inventories differ")
        for metric, value in scores.items():
            pair_distances[metric].append(value)
    with (unperturbed / "n5s-run-artifact.json").open(
            "r", encoding="utf-8") as stream:
        unperturbed_payload = json.load(stream)
    payload: dict[str, object] = {
        "schema": 1,
        "registration": reg,
        "evaluator_commit": reg["evaluator_commit"],
        "mask_parameter_hash": reg["mask_parameter_hash"],
        "parameters": reg["parameters"],
        "expected_samples": reg["expected_samples"],
        "wrf_build": dict(N5S_WRF_BUILD),
        "unperturbed": unperturbed_artifact,
        "members": member_records,
        "restored_input_sha256": restored_input_sha256(restored_inputs),
        "domain_durations_seconds": _artifact_durations(unperturbed_payload),
        "cadence": reg["cadence"],
        "cpu_pair_distances": pair_distances,
    }
    target = Path(output) if output is not None else root / "n5s-ensemble.json"
    write_json(target, payload)
    return payload


def build_gpu_evidence(*, gpu_directory: str | Path,
                       unperturbed_directory: str | Path,
                       registration: Mapping[str, object],
                       restored_inputs: str | Path,
                       output: str | Path) -> dict[str, object]:
    gpu = Path(gpu_directory).resolve()
    unperturbed = Path(unperturbed_directory).resolve()
    target = Path(output).resolve()
    reg = validate_registration(registration)
    require_matching_registrations(reg, _read_sidecar(gpu))
    require_matching_registrations(reg, _read_sidecar(unperturbed))
    gpu_hash_file = gpu / "restored_input_sha256.txt"
    canonical_hash = restored_input_sha256(restored_inputs)
    if (not gpu_hash_file.is_file()
            or gpu_hash_file.read_text(encoding="utf-8").strip()
            != canonical_hash):
        raise ValueError("gpuwm restored-input digest sidecar is missing/mismatched")
    gpu_artifact = _artifact(
        gpu, target.parent, run_id=gpu.name, registration=reg)
    distances = score_run_pair(
        gpu, unperturbed, reg, reg,
        _verified_run_directories={gpu})
    with (gpu / "n5s-run-artifact.json").open(
            "r", encoding="utf-8") as stream:
        gpu_payload = json.load(stream)
    payload: dict[str, object] = {
        "schema": 1,
        "registration": reg,
        "evaluator_commit": reg["evaluator_commit"],
        "mask_parameter_hash": reg["mask_parameter_hash"],
        "parameters": reg["parameters"],
        "expected_samples": reg["expected_samples"],
        "gpu_candidate": gpu_artifact,
        "restored_input_sha256": canonical_hash,
        "domain_durations_seconds": _artifact_durations(gpu_payload),
        "cadence": reg["cadence"],
        "gpu_vs_unperturbed_distances": distances,
    }
    write_json(target, payload)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    register = commands.add_parser(
        "register", help="write metric pins before any N5S scoring")
    register.add_argument("--run-minutes", type=int, default=30)
    register.add_argument("--history-minutes", type=int, default=5)
    register.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "register":
        write_json(args.output, make_registration(
            run_minutes=args.run_minutes,
            history_minutes=args.history_minutes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "N5S_DOMAINS", "N5S_DX_METERS", "N5S_STATE_FIELDS", "N5S_WRF_BUILD",
    "build_ensemble_evidence", "build_gpu_evidence", "discover_frames",
    "first_object_time", "fss_distance", "has_qualifying_object",
    "load_registration", "make_registration", "neighborhood_fraction",
    "require_matching_registrations", "rmse", "score_run_pair",
    "validate_registration", "verify_run_artifact",
]
