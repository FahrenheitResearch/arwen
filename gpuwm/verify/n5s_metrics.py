"""Pre-registered pins and evidence emitters for the N5S matched-physics shadow.

The same registration scores every CPU-WRF member pair and the gpuwm candidate
against unperturbed WRF.  Scoring is CPU-only and reads WRF-compatible
NetCDF history frames.  A complete, hash-checked registration is mandatory
on both sides of every comparison.

The metric arithmetic itself lives in :mod:`gpuwm.verify.field_metrics`, which
carries no domain list, no grid spacing and no threshold.  What stays here is
what is specific to this campaign: the frozen pins, the four-domain ladder, the
registered metric key strings, and the run-artifact/evidence emitters.  The
delegation is scored by ``tests/test_field_metrics.py`` against a baseline
measured through this module before the arithmetic moved.

**Freeze note.**  These pins are a campaign artifact and stay frozen.  Every
number the registered gate record carries was measured under exactly the
domain list, spacing ladder, field set, thresholds and metric key strings
below; re-cutting them to a parameterized form would leave published numbers
attached to pins that no longer say what they said when the numbers were
taken.  So this module is not the place the general case grows: the generic
instruments -- :mod:`gpuwm.verify.field_metrics`,
:mod:`gpuwm.verify.chaos_envelope` and :mod:`gpuwm.verify.spectral` -- take
their geometry as arguments and score any campaign, and this one keeps the
one it measured.  Nothing here is imported by those modules; the delegation
runs one way only.
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

import numpy as np

from gpuwm.verify import field_metrics
from gpuwm.verify.n5s_common import (
    restored_input_sha256, sha256_file, stable_hash, write_json,
)


N5S_DOMAINS = ("d01", "d02", "d03", "d04")
N5S_STATE_FIELDS = ("U", "V", "W", "T", "PH", "MU", "QVAPOR")
N5S_DX_METERS = {
    "d01": 12000.0, "d02": 3000.0, "d03": 1000.0,
    "d04": 1000.0 / 3.0,
}
#: Domains carrying the registered reflectivity FSS row.  This names the
#: frozen gate-record selection; it is not a widening knob, and the generic
#: metric path does not consult it.  Frozen: the row this campaign scored is
#: the row it published, and a second domain here would be a different gate
#: record wearing this one's name.
N5S_FSS_DOMAINS = ("d04",)
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


def make_registration(*, start_time: str, run_minutes: int = 30,
                      history_minutes: int = 5, commit: str | None = None,
                      parameters: Mapping[str, object] | None = None
                      ) -> dict[str, object]:
    """Create all metric/mask pins before any output is inspected.

    ``start_time`` is required.  It anchors frame discovery, so a default
    would silently score one campaign's frame names on another campaign's
    output — and the frame ladder would simply come up empty, which reads as
    a staging error rather than as the wrong pin.
    """
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


#: The metric arithmetic is generic; these names keep the campaign's callers
#: and the registered spellings (``half_width_cells``, the duration/cadence
#: quiet sentinel) pointing at the one implementation.
rmse = field_metrics.rmse
has_qualifying_object = field_metrics.has_qualifying_object
_read_field = field_metrics.read_frame_field
_boxcar = field_metrics.boxcar
_odd_width = field_metrics.odd_width_cells
_aggregate_rmse = field_metrics.aggregate_rmse
_interior = field_metrics.interior
_boundary_values = field_metrics.boundary_values
_composite = field_metrics.composite


def neighborhood_fraction(events: np.ndarray, half_width_cells: int
                          ) -> np.ndarray:
    return field_metrics.neighborhood_fraction(events, half_width_cells)


def fss_distance(left: np.ndarray, right: np.ndarray, *, threshold: float,
                 half_width_cells: int) -> float:
    return field_metrics.fss_distance(
        left, right, threshold=threshold, half_width=half_width_cells)


def first_object_time(frames: Mapping[int, np.ndarray], *, threshold: float,
                      min_cells: int, duration_seconds: int,
                      cadence_seconds: int, connectivity: int = 8) -> float:
    """First qualifying object, with this gate's registered quiet sentinel."""
    return field_metrics.first_object_time(
        frames, threshold=threshold, min_cells=min_cells,
        quiet_time_seconds=duration_seconds + cadence_seconds,
        connectivity=connectivity)


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
            low_pass, boundary = field_metrics.state_and_boundary_rmse(
                left_frames=left_frames, right_frames=right_frames,
                field=field, score_times=score_times, cadence_seconds=cadence,
                filter_width_cells=filter_width,
                interior_exclusion_cells=low_pass_exclusion,
                boundary_width_cells=boundary_width,
                label=f"N5S {domain}/{field}")
            scores[f"low_pass_state_rmse:{domain}:{field}"] = low_pass
            scores[f"applied_boundary_increment_error:{domain}:{field}"] = (
                boundary)

        reflectivity_left = {
            seconds: _read_field(left_frames[seconds], "REFL_10CM")
            for seconds in score_times}
        reflectivity_right = {
            seconds: _read_field(right_frames[seconds], "REFL_10CM")
            for seconds in score_times}
        threshold = float(params["reflectivity_threshold_dbz"])
        min_cells = field_metrics.minimum_object_cells(
            float(params["object_min_area_km2"]), dx)
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
        # The registered gate record scores FSS on the innermost domain under
        # one frozen key, and that stays frozen: this scorer reproduces a
        # campaign, so its key strings are part of what was published.  Which
        # domains are scored is a caller's choice, not the metric's: the
        # generic path takes the spacing as an argument and will score any of
        # them.
        if domain in N5S_FSS_DOMAINS:
            scores[
                "d04_reflectivity_fss_distance:d04:REFL_10CM_40DBZ"
            ] = field_metrics.mean_fss_distance(
                reflectivity_left, reflectivity_right, threshold=threshold,
                radius_m=float(params["fss_neighborhood_half_width_m"]),
                dx_m=dx, times=score_times)
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
