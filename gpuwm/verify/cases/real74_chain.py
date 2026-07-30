"""Executable real74 N4/N5 chain rungs and controller-only N5S/N5B harnesses.

N4 consumes the immutable N3 d02 ratchet and publishes d03.  N5 consumes the
N4 d03 ratchet, checks every registered N5 row, and records the run-bound
evidence pins required by the gate ledger.  N5S and N5B are deliberately
split into manifest-driven controller entry points: CPU tests validate their
geometry/envelope logic, while the expensive WRF/GPU executions stay outside
pytest and outside this lane.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import re
from dataclasses import asdict, dataclass, replace
from datetime import timedelta
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from gpuwm.case_data import load_experiment_case
from gpuwm.verify import metrics as weather_metrics
from gpuwm.verify import nest_gates
from gpuwm.verify.cases import real74_d02 as shared


N4_METRICS = (
    "d01_bitwise_vs_phase4_13z",
    "d02_bitwise_vs_n3",
    "d03_mslp_pattern_correlation",
    "d03_t500_rmse_k",
    "d03_t850_rmse_k",
    "d03_boundary_zone_blowup",
    "d03_refl_10cm_structure",
    "d03_refl_10cm_fss",
    "d03_w_cfl_health",
)

N5_METRICS = (
    "d03_bitwise_vs_n4",
    "d04_mslp_pattern_correlation",
    "d04_t500_rmse_k",
    "d04_t850_rmse_k",
    "d04_boundary_zone_blowup",
    "d04_refl_10cm_structure",
    "d04_refl_10cm_fss",
    "d04_domain_mean_profiles",
    "d04_updraft_intensity_distribution",
    "d04_updraft_intensity_percentile_band",
    "N5S_matched_physics_wrf_shadow",
    "N5B_d04_boundary_location_invariance",
    "ancestor_inertness_every_output",
    "full_tree_restart_bit_identity",
    "tick_exact_sync_ledger",
    "memory_peak_le_estimate",
    "estimate_le_wddm_budget",
    "host_overhead_fraction",
)

# N5 consumes the union of the immutable N3 and N4 state-control inventories.
# Keep that consumer expectation explicit so a future role change cannot
# silently omit an ancestor from the post-run comparison.
N5_ANCESTOR_CONTROL_DOMAINS = ("d01", "d02", "d03")


def _gate(milestone: str, metric: str):
    return nest_gates.gate(milestone, metric)


def _nearest_rank(values: Iterable[float], percentile: float) -> float:
    values = sorted(float(value) for value in values)
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("nearest-rank sample is empty or non-finite")
    index = math.ceil(percentile * len(values) / 100.0) - 1
    return values[index]


def _actual_evidence_pins(*, mask: object, baseline_hash: str,
                          cadence: str, expected_samples: int
                          ) -> dict[str, object]:
    return {
        "evaluator_commit": shared._git_commit(),
        "mask_hash": shared.stable_hash(mask),
        "baseline_hash": baseline_hash,
        "cadence": cadence,
        "expected_samples": int(expected_samples),
    }


def _pin_n5_result(result: dict[str, object], *, mask: object,
                   baseline_hash: str, cadence: str,
                   expected_samples: int) -> None:
    pins = _actual_evidence_pins(
        mask=mask, baseline_hash=baseline_hash, cadence=cadence,
        expected_samples=expected_samples)
    evidence = dict(result.get("evidence", {}))
    evidence["manifest_pins"] = pins
    result["evidence"] = evidence
    # Mirror the NestGate evidence-field schema directly on each report row;
    # the nested copy keeps evidence artifacts self-contained.
    result.update(pins)


def _final_frame(run: shared.ProductionRun, exp, grid_id: int) -> Path:
    return shared.frame_path(
        run.output_dir, grid_id,
        exp.start_time + timedelta(seconds=shared.RUN_SECONDS))


def _d01_frame(run: shared.ProductionRun, exp) -> Path:
    return shared.frame_path(
        run.output_dir, 1, exp.start_time + timedelta(hours=1))


def w_cfl_evidence(path: str | Path, *, dx_m: float, dt_s: float
                   ) -> dict[str, object]:
    """Output-frame w/CFL evidence; the blocking verdict remains adjudicated."""
    w = shared._read_field(path, "W")
    u = shared._read_field(path, "U")
    v = shared._read_field(path, "V")
    ph = shared._read_field(path, "PH") + shared._read_field(path, "PHB")
    height = ph / 9.80665
    dz = np.diff(height, axis=0)
    # Output state has no missing-data convention.  Ordinary reductions
    # propagate any candidate NaN into the health verdict instead of finding
    # a reassuring maximum/minimum among only the surviving cells.
    dz_min = float(np.min(dz))
    w_max = float(np.max(np.abs(w)))
    wind_max = float(np.maximum(
        np.max(np.abs(u)), np.max(np.abs(v))))
    cfl = float(dt_s * max(wind_max / dx_m, w_max / dz_min))
    return {
        "finite": bool(all(math.isfinite(value)
                           for value in (dz_min, w_max, wind_max, cfl))
                       and dz_min > 0.0 and cfl >= 0.0),
        "minimum_layer_depth_m": dz_min,
        "w_max_ms": w_max, "horizontal_wind_max_ms": wind_max,
        "diagnostic_cfl": cfl,
    }


def domain_mean_profile_evidence(candidate: str | Path,
                                 reference: str | Path) -> dict[str, object]:
    """Domain-mean vertical profiles named by the N5 evidence list."""
    profiles: dict[str, object] = {}
    for name in ("T", "QVAPOR", "W"):
        actual = shared._read_field(candidate, name)
        oracle = shared._read_field(reference, name)
        if actual.shape != oracle.shape:
            raise ValueError(f"{name} profile shapes differ")
        if not np.isfinite(actual).all() or not np.isfinite(oracle).all():
            raise ValueError(f"{name} profile input is non-finite")
        actual_mean = np.mean(actual, axis=(-2, -1), dtype=np.float64)
        oracle_mean = np.mean(oracle, axis=(-2, -1), dtype=np.float64)
        profiles[name] = {
            "candidate": actual_mean.tolist(),
            "reference": oracle_mean.tolist(),
            "rmse": weather_metrics.rmse(actual_mean, oracle_mean),
            "pattern_correlation": weather_metrics.pattern_correlation(
                actual_mean, oracle_mean),
        }
    return {"profiles": profiles}


def _updraft_floor(record) -> float:
    match = re.search(r"values >= ([0-9.]+) m s-1", record.convention)
    if match is None:
        raise ValueError("updraft gate convention has no sampling floor")
    return float(match.group(1))


def updraft_percentile_evidence(candidate: str | Path,
                                reference: str | Path) -> tuple[bool, dict]:
    record = _gate("N5", "d04_updraft_intensity_percentile_band")
    floor = _updraft_floor(record)

    def sample(path):
        vertical_velocity = shared._read_field(path, "W")
        if not np.isfinite(vertical_velocity).all():
            raise ValueError("updraft sample is non-finite")
        column = np.max(vertical_velocity, axis=0)
        region = weather_metrics.interior_region(column.shape)
        values = column[region]
        if not np.isfinite(values).all():
            raise ValueError("updraft sample is non-finite")
        values = values[values >= floor]
        if not values.size:
            raise ValueError("updraft sample is empty")
        return values

    try:
        actual, oracle = sample(candidate), sample(reference)
        rows = []
        passed = True
        for percentile in nest_gates.UPDRAFT_INTENSITY_PERCENTILES:
            lhs = _nearest_rank(actual, percentile)
            rhs = _nearest_rank(oracle, percentile)
            ratio = lhs / rhs
            accepted = (math.isfinite(ratio)
                        and nest_gates.UPDRAFT_INTENSITY_RATIO_MIN <= ratio
                        <= nest_gates.UPDRAFT_INTENSITY_RATIO_MAX)
            passed &= accepted
            rows.append({
                "percentile": percentile, "candidate_ms": lhs,
                "reference_ms": rhs, "ratio": ratio,
                "minimum_ratio": nest_gates.UPDRAFT_INTENSITY_RATIO_MIN,
                "maximum_ratio": nest_gates.UPDRAFT_INTENSITY_RATIO_MAX,
                "passed": bool(accepted),
            })
        return bool(passed), {
            "sampling_floor_ms": floor,
            "candidate_samples": int(actual.size),
            "reference_samples": int(oracle.size), "percentiles": rows,
        }
    except (KeyError, OSError, ValueError) as exc:
        return False, {"error": str(exc), "sampling_floor_ms": floor}


def evaluate_n4(exp, case_data, run: shared.ProductionRun, *,
                phase4_root: str | Path, ratchet_root: str | Path,
                verdicts: Mapping[str, object]) -> dict[str, object]:
    """Evaluate every registered N4 record and no unregistered policy."""
    results: dict[str, dict[str, object]] = {}
    phase4 = shared.frame_path(
        phase4_root, 1, exp.start_time + timedelta(hours=1))
    d01_comparison = shared.compare_d01_phase4_frame(_d01_frame(run, exp), phase4)
    record = _gate("N4", "d01_bitwise_vs_phase4_13z")
    results[record.metric] = shared.gate_result(
        record, passed=bool(d01_comparison["passed"]),
        evidence=d01_comparison)

    n3_manifest = shared.load_ratchet(ratchet_root, "N3", domain="d02")
    shared.validate_ratchet_provenance(
        n3_manifest, shared.run_prefix_provenance(run, 2))
    comparison = shared.compare_ratchet_frames(
        run.paths_for_domain(2), ratchet_root, n3_manifest, "d02",
        candidate_state_hashes=shared.state_hashes_for_domain(
            run.execution, "d02"))
    record = _gate("N4", "d02_bitwise_vs_n3")
    results[record.metric] = shared.gate_result(
        record, passed=bool(comparison["passed"]), evidence=comparison)

    d03 = _final_frame(run, exp, 3)
    reference = shared.child_reference_path(case_data, "d03")
    results.update(shared.score_statistical_frame(
        "N4", "d03", d03, reference, dx_m=exp.domain(3).run.dx,
        run_summary=shared.run_summary_for_domain(run, "d03"),
        verdicts=verdicts,
        fss_reference=shared.matched_reference_path(case_data, "d03")))
    metric = "d03_w_cfl_health"
    record = _gate("N4", metric)
    passed, adjudication = shared.structural_verdict(verdicts, metric)
    health = w_cfl_evidence(
        d03, dx_m=exp.domain(3).run.dx, dt_s=exp.domain(3).run.dt)
    passed &= bool(health["finite"])
    health["adjudication"] = dict(adjudication)
    results[metric] = shared.gate_result(
        record, passed=passed, evidence=health)

    if tuple(results) != N4_METRICS:
        raise RuntimeError(f"N4 evaluator inventory drifted: {tuple(results)}")
    shared.gate_records("N4", results)
    return {
        "schema": 1, "rung": "N4", "generated_utc": shared._utc_now(),
        "evaluator_commit": shared._git_commit(),
        "config_source": str(shared.PRODUCTION_CONFIG),
        "domain_ids": [dc.grid_id for dc in exp.domains],
        "run_seconds": exp.run_seconds,
        "production_execution": {
            "output_dir": str(run.output_dir),
            "completed_seconds": run.completed_seconds,
            "execution": dict(run.execution), "timing": dict(run.timing),
            "memory": dict(run.memory),
        },
        "gates": list(results.values()),
        "passed": shared.report_passed(results.values()),
    }


def full_schedule_sync_ledger(
        config_path: str | Path, *, executed_exp=None,
        execution: Mapping[str, object] | None = None) -> dict[str, object]:
    """Pin the registered 12 h count and verify the executed rung clocks."""
    from gpuwm.core.clock import build_schedule, execute_schedule, resolve_clock

    exp, _data = load_experiment_case(config_path)
    schedule = build_schedule(exp, resolve_clock(exp))
    report = execute_schedule(schedule)
    clocks = {
        f"d{gid:02d}": {
            "ticks": int(clock.ticks), "step_count": int(clock.step_count),
            "expected_step_count": int(
                schedule.clock.run_ticks // clock.spec.step_ticks),
        } for gid, clock in sorted(report.clocks.items())
    }
    registered_d04_equivalent_steps = 25_920
    full_schedule_passed = (
        clocks.get("d04", {}).get("step_count")
        == registered_d04_equivalent_steps
        and all(row["step_count"] == row["expected_step_count"]
                and row["ticks"] == schedule.clock.run_ticks
                for row in clocks.values()))

    executed = {
        "measurement_present": False, "passed": False,
        "reason": "executed run clock ledger is missing",
    }
    if executed_exp is not None and execution is not None:
        actual_clocks = execution.get("clocks")
        if isinstance(actual_clocks, Mapping):
            executed_schedule = build_schedule(
                executed_exp, resolve_clock(executed_exp))
            expected = {
                f"d{spec.grid_id:02d}": {
                    "ticks": int(executed_schedule.clock.run_ticks),
                    "step_count": int(
                        executed_schedule.clock.run_ticks // spec.step_ticks),
                    "tick_den": int(executed_schedule.clock.tick_den),
                }
                for spec in executed_schedule.clock.domains
            }
            inventory_equal = set(actual_clocks) == set(expected)
            rows = {}
            executed_passed = inventory_equal
            for domain in sorted(set(actual_clocks) | set(expected)):
                actual = actual_clocks.get(domain)
                wanted = expected.get(domain)
                accepted = bool(
                    isinstance(actual, Mapping) and wanted is not None
                    and actual.get("ticks") == wanted["ticks"]
                    and actual.get("step_count") == wanted["step_count"]
                    and actual.get("tick_den") == wanted["tick_den"])
                executed_passed &= accepted
                rows[domain] = {
                    "actual": dict(actual) if isinstance(actual, Mapping) else None,
                    "expected": wanted, "passed": accepted,
                }
            executed = {
                "measurement_present": True,
                "passed": bool(executed_passed),
                "inventory_equal": inventory_equal,
                "run_ticks": int(executed_schedule.clock.run_ticks),
                "tick_den": int(executed_schedule.clock.tick_den),
                "clocks": rows,
            }
    return {
        "passed": bool(full_schedule_passed and executed["passed"]),
        "kind": "exact",
        "registered_d04_equivalent_steps": registered_d04_equivalent_steps,
        "full_schedule_passed": bool(full_schedule_passed),
        "run_ticks": schedule.clock.run_ticks,
        "tick_den": schedule.clock.tick_den, "periods": schedule.periods,
        "d04_equivalent_steps": clocks["d04"]["step_count"],
        "clocks": clocks, "steps": report.steps, "forces": report.forces,
        "feedback_calls": report.feedback_calls, "executed_run": executed,
    }


def ancestor_inertness_evidence(
        exp, run: shared.ProductionRun, *, phase4_root: str | Path,
        ratchet_root: str | Path) -> dict[str, object]:
    """Compare every live ancestor state hash with its no-younger control.

    N3 records a separate d01-only execution plus its terminal-child d02;
    N4 records its terminal-child d03.  This makes all three baselines true
    otherwise-identical controls with no younger child, while Phase-4 remains
    the separate d01 output-byte ratchet used by N3/N4.
    """
    del phase4_root  # Phase-4 is an output ratchet, not a live-state capture.
    n3 = shared.load_ratchet(ratchet_root, "N3")
    n4 = shared.load_ratchet(ratchet_root, "N4")
    declared_control_domains = (
        *n3.expected_state_domains, *n4.expected_state_domains)
    if declared_control_domains != N5_ANCESTOR_CONTROL_DOMAINS:
        raise ValueError(
            "N5 ancestor control domain inventory differs from its exact "
            f"consumer contract: expected {list(N5_ANCESTOR_CONTROL_DOMAINS)}, "
            f"got {list(declared_control_domains)}")
    shared.validate_ratchet_provenance(
        n3, shared.run_prefix_provenance(run, 2))
    shared.validate_ratchet_provenance(
        n4, shared.run_prefix_provenance(run, 3))
    control_samples = shared._normalize_state_hashes(
        (*n3.state_hashes, *n4.state_hashes))
    controls = {
        domain: {
            str(sample["frame"]): sample for sample in control_samples
            if sample["domain"] == domain}
        for domain in N5_ANCESTOR_CONTROL_DOMAINS
    }
    candidate_samples = run.execution.get("canonical_state_hashes", ())
    if not isinstance(candidate_samples, list):
        candidate_samples = ()
    candidate_samples = shared._normalize_state_hashes(candidate_samples)
    candidates = {
        domain: {
            str(sample["frame"]): sample for sample in candidate_samples
            if sample.get("domain") == domain}
        for domain in N5_ANCESTOR_CONTROL_DOMAINS
    }
    from gpuwm.core.clock import resolve_clock
    from gpuwm.io.wrfout import wrfout_filename
    tick_den = resolve_clock(exp).tick_den

    rows = []
    evaluator_commit = shared._git_commit()
    control_commits_match = (
        n3.evaluator_commit == evaluator_commit
        and n4.evaluator_commit == evaluator_commit)
    passed = control_commits_match
    for grid_id in (1, 2, 3):
        domain = f"d{grid_id:02d}"
        dc = exp.domain(grid_id)
        offsets = range(
            0, int(exp.run_seconds) + 1, int(dc.history_interval_s))
        expected_names = tuple(wrfout_filename(
            exp.start_time + timedelta(seconds=offset), grid_id)
            for offset in offsets)
        output_names = {path.name for path in run.paths_for_domain(grid_id)}
        candidate_names = set(candidates[domain])
        control_names = set(controls[domain])
        state_comparison = shared.compare_state_hash_samples(
            candidates[domain].values(), controls[domain].values(), domain)
        state_rows = {
            Fraction(
                int(item["instant_numerator"]),
                int(item["instant_denominator"])): item
            for item in state_comparison["samples"]}
        inventory_equal = (output_names == set(expected_names)
                           and candidate_names == set(expected_names)
                           and control_names == set(expected_names)
                           and state_comparison["tick_inventory_equal"])
        passed &= bool(inventory_equal)
        for offset, name in zip(offsets, expected_names, strict=True):
            actual_hash = candidates[domain].get(name, {
                "domain": domain, "frame": name, "missing": True})
            baseline_hash = controls[domain].get(name, {
                "domain": domain, "frame": name, "missing": True})
            expected_ticks = int(offset * tick_den)
            expected_instant = Fraction(offset, 1)
            hash_comparison = state_rows.get(expected_instant, {
                "passed": False, "inventory_equal": False,
                "hash_compared": False, "hash_equal": None,
            })
            equal = (
                not actual_hash.get("missing")
                and not baseline_hash.get("missing")
                and int(actual_hash["tick_den"]) == tick_den
                and int(actual_hash["ticks"]) == expected_ticks
                and shared._state_sample_instant(actual_hash) == expected_instant
                and shared._state_sample_instant(baseline_hash) == expected_instant
                and hash_comparison["passed"])
            passed &= bool(equal)
            rows.append({
                "domain": domain, "frame": name,
                "expected_ticks": expected_ticks,
                "candidate": actual_hash, "control": baseline_hash,
                "inventory_equal": hash_comparison["inventory_equal"],
                "hash_compared": hash_comparison["hash_compared"],
                "hash_equal": hash_comparison["hash_equal"],
                "passed": bool(equal),
            })
    return {
        "passed": bool(passed), "expected_samples": len(rows),
        "canonical_schema": shared.CANONICAL_STATE_SCHEMA,
        "control_manifests": {
            "N3": str(Path(ratchet_root) / "N3" / "manifest.json"),
            "N4": str(Path(ratchet_root) / "N4" / "manifest.json"),
        },
        "control_evaluator_commits": {
            "N3": n3.evaluator_commit, "N4": n4.evaluator_commit,
        },
        "candidate_evaluator_commit": evaluator_commit,
        "control_commits_match": control_commits_match,
        "samples": rows,
    }


def _controller_score_binding_sha256(report: Mapping[str, object]) -> str:
    """Hash the closed score/input inventory, excluding the claimed verdict.

    ``passed`` is deliberately absent: final consumers derive it from the
    bound comparison rows.  This makes changing a top-level boolean inert and
    makes any row, metric-inventory, or input-artifact edit detectable.
    """
    return shared.stable_hash({
        "schema": report.get("schema"),
        "metric": report.get("metric"),
        "evaluator_commit": report.get("evaluator_commit"),
        "mask_hash": report.get("mask_hash"),
        "baseline_hash": report.get("baseline_hash"),
        "cadence": report.get("cadence"),
        "expected_samples": report.get("expected_samples"),
        "score_inputs": report.get("score_inputs"),
        "comparisons": report.get("comparisons"),
    })


def _bind_controller_score_report(
        report: dict[str, object], *, score_inputs: Mapping[str, object]
        ) -> None:
    report["score_inputs"] = shared.json_safe(score_inputs)
    report["score_binding_sha256"] = _controller_score_binding_sha256(report)


def _validated_controller_score_report(
        metric: str, report: Mapping[str, object]) -> tuple[bool, dict[str, object]]:
    """Validate a controller add-on's closed inventory and derive its verdict."""
    if report.get("reason") == "controller add-on report not supplied":
        return False, dict(report)
    if report.get("schema") != 2 or report.get("metric") != metric:
        raise ValueError(f"external report does not identify secure {metric}")
    if report.get("evaluator_commit") != shared._git_commit():
        raise ValueError(f"external {metric} report is from another evaluator")
    for key in ("mask_hash", "baseline_hash", "score_binding_sha256"):
        if not _is_sha256(report.get(key)):
            raise ValueError(f"external {metric} report has no valid {key}")
    cadence = report.get("cadence")
    expected_samples = report.get("expected_samples")
    rows = report.get("comparisons")
    if (not isinstance(cadence, str) or not cadence
            or isinstance(expected_samples, bool)
            or not isinstance(expected_samples, int)
            or expected_samples <= 0
            or not isinstance(rows, list)
            or len(rows) != expected_samples):
        raise ValueError(f"external {metric} report lacks bound score rows")
    row_metrics = []
    for row in rows:
        if (not isinstance(row, dict)
                or not isinstance(row.get("metric"), str)
                or not row["metric"]
                or not isinstance(row.get("passed"), bool)):
            raise ValueError(f"external {metric} report has an invalid comparison")
        row_metrics.append(row["metric"])
    if len(set(row_metrics)) != len(row_metrics):
        raise ValueError(f"external {metric} report repeats a comparison metric")
    score_inputs = report.get("score_inputs")
    if not isinstance(score_inputs, dict):
        raise ValueError(f"external {metric} report has no score-input inventory")
    inventory = score_inputs.get("metric_inventory")
    artifact_hashes = score_inputs.get("input_artifact_sha256")
    if (not isinstance(inventory, list)
            or inventory != sorted(row_metrics)
            or not isinstance(artifact_hashes, dict)
            or not artifact_hashes
            or any(not isinstance(name, str) or not name
                   or not _is_sha256(value)
                   for name, value in artifact_hashes.items())):
        raise ValueError(f"external {metric} score/input inventory is not bound")
    if report["score_binding_sha256"] != _controller_score_binding_sha256(report):
        raise ValueError(f"external {metric} score binding is invalid")
    derived = all(bool(row["passed"]) for row in rows)
    evidence = dict(report)
    evidence["declared_passed"] = report.get("passed")
    evidence["passed"] = derived
    return derived, evidence


def _load_gate_report(path: str | Path | None, metric: str) -> dict[str, object]:
    if path is None:
        return {"metric": metric, "passed": False,
                "reason": "controller add-on report not supplied"}
    with Path(path).open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if payload.get("metric") != metric or not isinstance(payload.get("passed"), bool):
        raise ValueError(f"external report does not identify {metric}")
    if metric == "N5S_matched_physics_wrf_shadow":
        candidate_evidence = payload.get("candidate_evidence")
        wrf_run_directory = payload.get("wrf_run_directory")
        if (not isinstance(candidate_evidence, str) or not candidate_evidence
                or not isinstance(wrf_run_directory, str)
                or not wrf_run_directory):
            raise ValueError("N5S final report lacks re-score lineage")
        reconstructed = evaluate_n5s_shadow(
            candidate_evidence, wrf_run_directory)
    elif metric == "N5B_d04_boundary_location_invariance":
        frozen_manifest = payload.get("frozen_manifest")
        observation = payload.get("observation")
        anchor_sha256 = payload.get("freeze_anchor_sha256")
        if (not isinstance(frozen_manifest, str) or not frozen_manifest
                or not isinstance(observation, str) or not observation
                or not _is_sha256(anchor_sha256)):
            raise ValueError("N5B final report lacks re-score lineage")
        reconstructed = score_n5b_observation(
            frozen_manifest, observation,
            freeze_anchor_sha256=str(anchor_sha256))
    else:
        raise ValueError(f"unsupported external controller gate: {metric}")
    if (payload.get("score_binding_sha256")
            != reconstructed.get("score_binding_sha256")):
        raise ValueError(f"external {metric} report differs from artifact re-scoring")
    _validated_controller_score_report(metric, reconstructed)
    return reconstructed


def _consume_controller_gate_report(
        metric: str, report: Mapping[str, object]) -> dict[str, object]:
    """Materialize an N5 add-on row without collapsing its evidence report."""
    record = _gate("N5", metric)
    passed, evidence = _validated_controller_score_report(metric, report)
    return shared.gate_result(
        record, passed=passed, evidence=evidence)


def evaluate_n5(
        exp, case_data, run: shared.ProductionRun, *,
        phase4_root: str | Path, ratchet_root: str | Path,
        verdicts: Mapping[str, object], restart_evidence: Mapping[str, object],
        n5s_report: Mapping[str, object], n5b_report: Mapping[str, object],
        config_path: str | Path) -> dict[str, object]:
    """Evaluate all registered N5 blockers with concrete evidence pins."""
    results: dict[str, dict[str, object]] = {}
    n4_manifest = shared.load_ratchet(ratchet_root, "N4", domain="d03")
    shared.validate_ratchet_provenance(
        n4_manifest, shared.run_prefix_provenance(run, 3))
    comparison = shared.compare_ratchet_frames(
        run.paths_for_domain(3), ratchet_root, n4_manifest, "d03",
        candidate_state_hashes=shared.state_hashes_for_domain(
            run.execution, "d03"))
    record = _gate("N5", "d03_bitwise_vs_n4")
    results[record.metric] = shared.gate_result(
        record, passed=bool(comparison["passed"]), evidence=comparison)

    d04 = _final_frame(run, exp, 4)
    reference = shared.child_reference_path(case_data, "d04")
    reference_hash = shared.sha256_file(reference)
    results.update(shared.score_statistical_frame(
        "N5", "d04", d04, reference, dx_m=exp.domain(4).run.dx,
        run_summary=shared.run_summary_for_domain(run, "d04"),
        verdicts=verdicts,
        fss_reference=shared.matched_reference_path(case_data, "d04")))

    metric = "d04_domain_mean_profiles"
    record = _gate("N5", metric)
    passed, adjudication = shared.structural_verdict(verdicts, metric)
    try:
        profile = domain_mean_profile_evidence(d04, reference)
        profile["adjudication"] = dict(adjudication)
    except (KeyError, OSError, ValueError) as exc:
        passed, profile = False, {"error": str(exc),
                                  "adjudication": dict(adjudication)}
    results[metric] = shared.gate_result(record, passed=passed, evidence=profile)

    metric = "d04_updraft_intensity_distribution"
    record = _gate("N5", metric)
    passed, adjudication = shared.structural_verdict(verdicts, metric)
    percentile_passed, percentile_evidence = updraft_percentile_evidence(
        d04, reference)
    distribution_available = (
        "error" not in percentile_evidence
        and int(percentile_evidence.get("candidate_samples", 0)) > 0
        and int(percentile_evidence.get("reference_samples", 0)) > 0)
    results[metric] = shared.gate_result(
        record, passed=passed and distribution_available,
        evidence={"adjudication": dict(adjudication),
                  "distribution": percentile_evidence})
    metric = "d04_updraft_intensity_percentile_band"
    record = _gate("N5", metric)
    results[metric] = shared.gate_result(
        record, passed=percentile_passed, evidence=percentile_evidence)

    for metric, report in (
            ("N5S_matched_physics_wrf_shadow", n5s_report),
            ("N5B_d04_boundary_location_invariance", n5b_report)):
        # Keep the complete controller report in evidence.  In particular,
        # F28's documented/degenerate N5S state must reach the N5 rung row.
        results[metric] = _consume_controller_gate_report(metric, report)

    metric = "ancestor_inertness_every_output"
    record = _gate("N5", metric)
    ancestor = ancestor_inertness_evidence(
        exp, run, phase4_root=phase4_root, ratchet_root=ratchet_root)
    results[metric] = shared.gate_result(
        record, passed=bool(ancestor["passed"]), evidence=ancestor)

    metric = "full_tree_restart_bit_identity"
    record = _gate("N5", metric)
    results[metric] = shared.gate_result(
        record, passed=bool(restart_evidence.get("passed")),
        evidence=restart_evidence)
    metric = "tick_exact_sync_ledger"
    record = _gate("N5", metric)
    if record.kind != "exact":
        raise RuntimeError("tick_exact_sync_ledger is no longer an exact gate")
    sync = full_schedule_sync_ledger(
        config_path, executed_exp=exp, execution=run.execution)
    results[metric] = shared.gate_result(
        record, passed=bool(sync["passed"]), evidence=sync)

    from gpuwm.core.preflight import ReservePolicy
    measured = int(run.memory["pool_used_peak_bytes"])
    estimate = int(run.memory["alloc_estimate_bytes"])
    free = int(run.memory["free_before_bytes"])
    reserve = ReservePolicy.run_time()
    budget = reserve.budget_bytes(free)
    metric = "memory_peak_le_estimate"
    record = _gate("N5", metric)
    results[metric] = shared.gate_result(
        record, passed=measured <= estimate,
        evidence={"measured_peak_bytes": measured,
                  "estimate_bytes": estimate})
    metric = "estimate_le_wddm_budget"
    record = _gate("N5", metric)
    results[metric] = shared.gate_result(
        record, passed=estimate <= budget,
        evidence={"estimate_bytes": estimate, "measured_free_bytes": free,
                  "reserve_bytes": reserve.reserve_bytes,
                  "budget_bytes": budget})
    metric = "host_overhead_fraction"
    record = _gate("N5", metric)
    host_fraction = float(run.timing["host_overhead_fraction"])
    results[metric] = shared.gate_result(
        record, value=host_fraction, evidence=dict(run.timing))

    if tuple(results) != N5_METRICS:
        raise RuntimeError(f"N5 evaluator inventory drifted: {tuple(results)}")
    shared.gate_records("N5", results)

    for metric, result in results.items():
        if metric == "d03_bitwise_vs_n4":
            baseline_hash = comparison["baseline_inventory_sha256"]
            mask = {"scope": "whole NetCDF file", "domain": "d03",
                    "frames": "complete stored output inventory"}
        elif metric.startswith("d04_"):
            baseline_hash = reference_hash
            if metric == "d04_domain_mean_profiles":
                mask = {"scope": "full-domain vertical means",
                        "fields": ["T", "QVAPOR", "W"]}
            elif metric == "d04_boundary_zone_blowup":
                mask = {"scope": "boundary/interior partition",
                        "width_cells": int(exp.spec_bdy_width), "field": "W"}
            elif metric == "d04_refl_10cm_structure":
                mask = {"scope": "full-domain composite",
                        "field": "REFL_10CM"}
            else:
                mask = {"scope": "five-cell interior", "slice": "5:-5",
                        "domain": "d04"}
        elif metric.startswith("N5S_"):
            baseline_hash = str(n5s_report.get("baseline_hash", "missing"))
            mask = {"external_mask_hash": n5s_report.get("mask_hash", "missing"),
                    "gate": metric}
        elif metric.startswith("N5B_"):
            baseline_hash = str(n5b_report.get("baseline_hash", "missing"))
            mask = {"geometry": n5b_report.get("geometry", "missing"),
                    "scope": "central common core"}
        elif metric == "ancestor_inertness_every_output":
            baseline_hash = shared.stable_hash(
                [row["control"].get("sha256") for row in ancestor["samples"]])
            mask = {"scope": "complete canonical mutable state",
                    "domains": ["d01", "d02", "d03"],
                    "schema": ancestor["canonical_schema"]}
        elif metric == "full_tree_restart_bit_identity":
            baseline_hash = shared.stable_hash(restart_evidence)
            mask = {"scope": "whole-file output inventory",
                    "domains": ["d01", "d02", "d03", "d04"]}
        elif metric == "tick_exact_sync_ledger":
            baseline_hash = shared.stable_hash(result.get("evidence", {}))
            mask = {"scope": "integer schedule clocks and counters",
                    "domains": ["d01", "d02", "d03", "d04"]}
        elif metric in {"memory_peak_le_estimate", "estimate_le_wddm_budget"}:
            baseline_hash = shared.stable_hash(result.get("evidence", {}))
            mask = {"scope": "full-device allocation ledger"}
        else:
            baseline_hash = shared.stable_hash(result.get("evidence", {}))
            mask = {"scope": "main-thread production orchestration profile"}
        if metric == "ancestor_inertness_every_output":
            samples = ancestor["expected_samples"]
            cadence = "every scheduled ancestor output"
        elif metric == "tick_exact_sync_ledger":
            samples = sync["d04_equivalent_steps"]
            cadence = "every d04 step over the production 12 h schedule"
        elif metric == "full_tree_restart_bit_identity":
            comparisons = restart_evidence.get("comparisons", {})
            samples = sum(len(domain.get("frames", ()))
                          for domain in comparisons.values())
            cadence = "every straight/split output frame"
        elif metric.startswith("N5S_"):
            samples = int(n5s_report.get("expected_samples", 0))
            cadence = str(n5s_report.get("cadence", "missing"))
        elif metric.startswith("N5B_"):
            samples = int(n5b_report.get("expected_samples", 0))
            cadence = str(n5b_report.get("cadence", "missing"))
        elif metric == "d03_bitwise_vs_n4":
            samples = int(comparison["expected_frames"])
            cadence = "every scheduled stored d03 output frame"
        elif metric == "d04_boundary_zone_blowup":
            samples = int(shared.run_summary_for_domain(
                run, "d04")["dynamics_substeps"])
            cadence = "every d04 dynamics substep over the rung run"
        elif metric.startswith("d04_"):
            samples = 1
            cadence = "13:15 output frame"
        else:
            samples = 1
            cadence = "one 75-minute full-chain controller measurement"
        _pin_n5_result(
            result, mask={**mask, "metric": metric},
            baseline_hash=baseline_hash, cadence=cadence,
            expected_samples=samples)

    return {
        "schema": 1, "rung": "N5", "generated_utc": shared._utc_now(),
        "evaluator_commit": shared._git_commit(),
        "config_source": str(config_path),
        "domain_ids": [dc.grid_id for dc in exp.domains],
        "run_seconds": exp.run_seconds,
        "production_execution": {
            "output_dir": str(run.output_dir),
            "completed_seconds": run.completed_seconds,
            "execution": dict(run.execution), "timing": dict(run.timing),
            "memory": dict(run.memory),
        },
        "gates": list(results.values()),
        "passed": shared.report_passed(results.values()),
    }


# ---------------------------------------------------------------------------
# N5S matched-physics external WRF ensemble-envelope harness
# ---------------------------------------------------------------------------

N5S_CATEGORIES = (
    "low_pass_state_rmse", "applied_boundary_increment_error",
    "d04_reflectivity_fss_distance", "storm_object_timing_difference",
)
N5S_DOMAINS = ("d01", "d02", "d03", "d04")


def _n5s_min_members() -> int:
    convention = _gate("N5", "N5S_matched_physics_wrf_shadow").convention
    match = re.search(r"M >= ([0-9]+)", convention)
    if match is None:
        raise ValueError("N5S record has no CPU-member minimum")
    return int(match.group(1))


def _verified_artifact(root: str | Path, record: Mapping[str, object],
                       label: str) -> dict[str, str]:
    """Verify a manifest artifact exists under its declared directory."""
    root = Path(root).resolve()
    relative = record.get("relative_path")
    expected = record.get("sha256")
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{label} needs a relative_path")
    if not isinstance(expected, str) or not expected:
        raise ValueError(f"{label} needs a sha256")
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"{label} artifact escapes or is missing: {relative}")
    actual = shared.sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} artifact hash mismatch: {relative}")
    return {"relative_path": path.relative_to(root).as_posix(),
            "sha256": actual}


@dataclass(frozen=True)
class _VerifiedJSONArtifact:
    path: Path
    relative_path: str
    sha256: str
    document: Mapping[str, object]


def _declared_artifact_path(
        root: str | Path, record: Mapping[str, object],
        label: str) -> tuple[Path, str, str]:
    root = Path(root).resolve()
    relative = record.get("relative_path")
    expected = record.get("sha256")
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{label} needs a relative_path")
    if not isinstance(expected, str) or not expected:
        raise ValueError(f"{label} needs a sha256")
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"{label} artifact escapes or is missing: {relative}")
    return path, path.relative_to(root).as_posix(), expected


def _verified_json_artifact(
        root: str | Path, record: Mapping[str, object],
        label: str) -> _VerifiedJSONArtifact:
    """Hash and parse one immutable byte snapshot of a JSON dependency."""
    root = Path(root).resolve()
    path, normalized_relative, expected = _declared_artifact_path(
        root, record, label)
    encoded = path.read_bytes()
    actual = hashlib.sha256(encoded).hexdigest()
    if actual != expected:
        raise ValueError(f"{label} artifact hash mismatch: {normalized_relative}")
    try:
        document = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"{label} is not valid JSON: {normalized_relative}") from exc
    if not isinstance(document, dict):
        raise ValueError(
            f"{label} must contain a JSON object: {normalized_relative}")
    return _VerifiedJSONArtifact(
        path=path, relative_path=path.relative_to(root).as_posix(),
        sha256=actual, document=document)


def _is_sha256(value: object) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(ch in "0123456789abcdef" for ch in value))


def _n5s_artifact_durations(artifact: Mapping[str, object]) -> dict[str, int]:
    frames = artifact.get("frames")
    if not isinstance(frames, list):
        raise ValueError("N5S run artifact has no frame inventory")
    by_domain = {domain: [] for domain in N5S_DOMAINS}
    for frame in frames:
        if (not isinstance(frame, dict)
                or frame.get("domain") not in by_domain
                or isinstance(frame.get("seconds"), bool)
                or not isinstance(frame.get("seconds"), int)):
            raise ValueError("N5S run artifact has an invalid frame record")
        by_domain[str(frame["domain"])].append(int(frame["seconds"]))
    if any(not seconds for seconds in by_domain.values()):
        raise ValueError("N5S run artifact omits a domain")
    return {domain: max(seconds) for domain, seconds in by_domain.items()}


def _verified_n5s_documents(
        candidate_evidence: str | Path, wrf_run_directory: str | Path
        ) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    """Reconstruct both N5S score documents from registrations and frames."""
    from gpuwm.verify import n5s_metrics

    wrf_root = Path(wrf_run_directory).resolve()
    ensemble_path = (wrf_root / "n5s-ensemble.json").resolve()
    candidate_path = Path(candidate_evidence).resolve()
    if not ensemble_path.is_file() or not candidate_path.is_file():
        raise ValueError("N5S evidence summary is missing")
    ensemble_bytes = ensemble_path.read_bytes()
    candidate_bytes = candidate_path.read_bytes()
    try:
        ensemble = json.loads(ensemble_bytes)
        candidate = json.loads(candidate_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("N5S evidence summary is not valid JSON") from exc
    if not isinstance(ensemble, dict) or not isinstance(candidate, dict):
        raise ValueError("N5S evidence summaries must contain JSON objects")
    if ensemble.get("schema") != 1 or candidate.get("schema") != 1:
        raise ValueError("N5S evidence summary schema is invalid")
    ensemble_registration = ensemble.get("registration")
    candidate_registration = candidate.get("registration")
    if (not isinstance(ensemble_registration, dict)
            or not isinstance(candidate_registration, dict)):
        raise ValueError("N5S evidence summary lacks a full registration")
    registration = n5s_metrics.require_matching_registrations(
        ensemble_registration, candidate_registration)
    root_registration = n5s_metrics.load_registration(
        wrf_root / "n5s-preregistration.json")
    registration = n5s_metrics.require_matching_registrations(
        registration, root_registration)
    for label, payload in (("CPU-WRF", ensemble), ("gpuwm", candidate)):
        expected = {
            "evaluator_commit": registration["evaluator_commit"],
            "mask_parameter_hash": registration["mask_parameter_hash"],
            "parameters": registration["parameters"],
            "expected_samples": registration["expected_samples"],
            "cadence": registration["cadence"],
        }
        if any(shared.json_safe(payload.get(key)) != shared.json_safe(value)
               for key, value in expected.items()):
            raise ValueError(f"N5S {label} summary differs from its registration")

    dependencies: dict[str, str] = {
        "ensemble_manifest": hashlib.sha256(ensemble_bytes).hexdigest(),
        "candidate_evidence": hashlib.sha256(candidate_bytes).hexdigest(),
        "registration": shared.stable_hash(registration),
    }
    dependency_paths: dict[Path, str] = {
        ensemble_path: dependencies["ensemble_manifest"],
        candidate_path: dependencies["candidate_evidence"],
    }
    verified_directories: set[Path] = set()

    def verified_run(root: Path, record: object, label: str, *,
                     expected_id: str | None = None
                     ) -> tuple[Path, dict[str, str], dict[str, object]]:
        if not isinstance(record, dict):
            raise ValueError(f"{label} lacks a run artifact")
        snapshot = _verified_json_artifact(root, record, label)
        if snapshot.path.name != "n5s-run-artifact.json":
            raise ValueError(f"{label} does not name the canonical run artifact")
        artifact = n5s_metrics.verify_run_artifact(
            snapshot.path.parent, registration)
        if shared.json_safe(artifact) != shared.json_safe(snapshot.document):
            raise ValueError(f"{label} run artifact changed during verification")
        if expected_id is not None and artifact.get("id") != expected_id:
            raise ValueError(f"{label} run artifact id mismatch")
        verified_directories.add(snapshot.path.parent)
        normalized = {
            "relative_path": snapshot.relative_path,
            "sha256": snapshot.sha256,
        }
        dependencies[label] = snapshot.sha256
        dependency_paths[snapshot.path] = snapshot.sha256
        status_path = next(
            path for path in (
                snapshot.path.parent / "exit.status",
                snapshot.path.parent / "exit_status.txt")
            if path.is_file())
        status_key = f"{label}_exit_status"
        dependencies[status_key] = shared.sha256_file(status_path)
        dependency_paths[status_path] = dependencies[status_key]
        return snapshot.path.parent, normalized, artifact

    members = ensemble.get("members")
    if not isinstance(members, list):
        raise ValueError("N5S ensemble member inventory is invalid")
    member_directories = []
    normalized_members = []
    for index, member in enumerate(members):
        if not isinstance(member, dict) or not isinstance(member.get("id"), str):
            raise ValueError("N5S ensemble member inventory is invalid")
        directory, normalized, _artifact = verified_run(
            wrf_root, member, f"cpu_member_{index}",
            expected_id=str(member["id"]))
        perturbation_path = directory / "perturbation.json"
        try:
            perturbation = json.loads(perturbation_path.read_bytes())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"N5S member lacks perturbation record: {directory}") from exc
        if (not perturbation
                or shared.json_safe(perturbation)
                != shared.json_safe(member.get("one_ulp_perturbation"))):
            raise ValueError("N5S member perturbation differs from its record")
        dependencies[f"cpu_member_{index}_perturbation"] = shared.sha256_file(
            perturbation_path)
        dependency_paths[perturbation_path] = dependencies[
            f"cpu_member_{index}_perturbation"]
        member_directories.append(directory)
        normalized_members.append({**member, **normalized})
    unperturbed_directory, normalized_unperturbed, unperturbed_artifact = (
        verified_run(
            wrf_root, ensemble.get("unperturbed"), "cpu_unperturbed",
            expected_id="unperturbed"))
    candidate_root = candidate_path.parent
    gpu_directory, normalized_candidate, candidate_artifact = verified_run(
        candidate_root, candidate.get("gpu_candidate"), "gpu_candidate")
    if candidate_artifact.get("id") != gpu_directory.name:
        raise ValueError("N5S gpu candidate run artifact id mismatch")

    ledger_path = wrf_root / "exit-status-ledger.json"
    n5s_metrics._require_ensemble_success_ledger(
        wrf_root, [unperturbed_directory, *member_directories])
    dependencies["cpu_exit_status_ledger"] = shared.sha256_file(ledger_path)
    dependency_paths[ledger_path] = dependencies["cpu_exit_status_ledger"]
    restored_hash = ensemble.get("restored_input_sha256")
    gpu_hash_file = gpu_directory / "restored_input_sha256.txt"
    if (not _is_sha256(restored_hash) or not gpu_hash_file.is_file()
            or gpu_hash_file.read_text(encoding="utf-8").strip() != restored_hash
            or candidate.get("restored_input_sha256") != restored_hash):
        raise ValueError("N5S GPU/WRF restored-input hashes differ")
    dependencies["gpu_restored_input_digest"] = shared.sha256_file(gpu_hash_file)
    dependency_paths[gpu_hash_file] = dependencies["gpu_restored_input_digest"]

    recomputed_cpu: dict[str, list[float]] = {}
    for left, right in itertools.combinations(member_directories, 2):
        scores = n5s_metrics.score_run_pair(
            left, right, registration, registration,
            _verified_run_directories=verified_directories)
        if not recomputed_cpu:
            recomputed_cpu = {metric: [] for metric in scores}
        if set(scores) != set(recomputed_cpu):
            raise ValueError("N5S CPU pair metric inventories differ")
        for metric, value in scores.items():
            recomputed_cpu[metric].append(value)
    recomputed_gpu = n5s_metrics.score_run_pair(
        gpu_directory, unperturbed_directory, registration, registration,
        _verified_run_directories=verified_directories)
    if (shared.json_safe(ensemble.get("cpu_pair_distances"))
            != shared.json_safe(recomputed_cpu)):
        raise ValueError("N5S CPU distances differ from artifact re-scoring")
    if (shared.json_safe(candidate.get("gpu_vs_unperturbed_distances"))
            != shared.json_safe(recomputed_gpu)):
        raise ValueError("N5S GPU distances differ from artifact re-scoring")
    if (shared.json_safe(ensemble.get("domain_durations_seconds"))
            != shared.json_safe(_n5s_artifact_durations(unperturbed_artifact))
            or shared.json_safe(candidate.get("domain_durations_seconds"))
            != shared.json_safe(_n5s_artifact_durations(candidate_artifact))):
        raise ValueError("N5S summary duration differs from its run artifact")

    # Close the read/score window: a concurrent or accidental mutation of any
    # frame, run artifact, or summary invalidates the reconstructed score.
    for directory in sorted(verified_directories):
        n5s_metrics.verify_run_artifact(directory, registration)
    for path, expected_sha256 in dependency_paths.items():
        if not path.is_file() or shared.sha256_file(path) != expected_sha256:
            raise ValueError(f"N5S evidence dependency changed during verification: {path}")

    ensemble = dict(ensemble)
    ensemble["members"] = normalized_members
    ensemble["unperturbed"] = {
        **dict(ensemble["unperturbed"]), **normalized_unperturbed}
    ensemble["cpu_pair_distances"] = recomputed_cpu
    candidate = dict(candidate)
    candidate["gpu_candidate"] = {
        **dict(candidate["gpu_candidate"]), **normalized_candidate}
    candidate["gpu_vs_unperturbed_distances"] = recomputed_gpu
    provenance = {
        "candidate_evidence": str(candidate_path),
        "candidate_evidence_sha256": dependencies["candidate_evidence"],
        "ensemble_manifest": str(ensemble_path),
        "ensemble_manifest_sha256": dependencies["ensemble_manifest"],
        "registration": registration,
        "registration_sha256": dependencies["registration"],
        "input_artifact_sha256": dependencies,
    }
    return ensemble, candidate, provenance


def evaluate_n5s_shadow(candidate_evidence: str | Path,
                        wrf_run_directory: str | Path) -> dict[str, object]:
    """Compare GPU distances with E95 from an external CPU-WRF ensemble.

    The external directory supplies ``n5s-ensemble.json`` containing member
    hashes and every unordered CPU-pair distance.  The candidate JSON carries
    like-for-like GPU-vs-unperturbed distances.  Distances are never averaged:
    a single non-degenerate miss fails the registered compound gate.  F28
    converts rows whose E95 is exactly zero to documented evidence while
    retaining their measured distances; any row whose E95 is positive keeps
    the original binding comparator.
    """
    wrf_run_directory = Path(wrf_run_directory).resolve()
    ensemble, candidate, provenance = _verified_n5s_documents(
        candidate_evidence, wrf_run_directory)
    members = ensemble.get("members", [])
    minimum_members = _n5s_min_members()
    if len(members) < minimum_members:
        raise ValueError(
            f"N5S needs at least {minimum_members} CPU-WRF members")
    verified_members = []
    for index, member in enumerate(members):
        verified_members.append({
            **member,
            **_verified_artifact(
                wrf_run_directory, member, f"N5S CPU member {index}"),
        })
        if not member.get("one_ulp_perturbation"):
            raise ValueError("every N5S member must document its 1-ULP perturbation")
    if (len({member.get("id") for member in verified_members}) != len(members)
            or len({member["sha256"] for member in verified_members})
            != len(members)):
        raise ValueError("N5S CPU members need unique ids and artifacts")
    convention = _gate("N5", "N5S_matched_physics_wrf_shadow").convention
    build_match = re.search(
        r"WRF (v[0-9.]+) ([A-Za-z0-9]+)\+([A-Za-z0-9]+) "
        r"(T[0-9A-Za-z]+) build", convention)
    if build_match is None:
        raise ValueError("N5S record has no WRF build identity")
    expected_build = {
        "wrf_version": build_match.group(1),
        "microphysics": build_match.group(2),
        "pbl": build_match.group(3),
        "instrumented_build": build_match.group(4),
    }
    if ensemble.get("wrf_build") != expected_build:
        raise ValueError("N5S external run does not pin the registered WRF build")
    percentile_match = re.search(r"E([0-9.]+) = nearest-rank", convention)
    if percentile_match is None:
        raise ValueError("N5S record has no ensemble percentile")
    envelope_percentile = float(percentile_match.group(1))
    duration_match = re.search(r">= ([0-9.]+) min", convention)
    if duration_match is None:
        raise ValueError("N5S record has no minimum duration")
    minimum_duration_s = float(duration_match.group(1)) * 60.0
    unperturbed = ensemble.get("unperturbed")
    if not isinstance(unperturbed, dict):
        raise ValueError("N5S manifest needs an unperturbed WRF artifact")
    unperturbed = {
        **unperturbed,
        **_verified_artifact(
            wrf_run_directory, unperturbed, "N5S unperturbed WRF"),
    }
    restored_hash = ensemble.get("restored_input_sha256")
    if (not _is_sha256(restored_hash)
            or candidate.get("restored_input_sha256") != restored_hash):
        raise ValueError("N5S GPU/WRF restored-input hashes differ")
    for label, payload in (("CPU-WRF", ensemble), ("gpuwm", candidate)):
        durations = payload.get("domain_durations_seconds")
        if (not isinstance(durations, dict)
                or set(durations) != set(N5S_DOMAINS)
                or any(not math.isfinite(float(durations[domain]))
                       or float(durations[domain]) < minimum_duration_s
                       for domain in N5S_DOMAINS)):
            raise ValueError(
                f"N5S {label} needs the registered duration on all four domains")
    cpu_pairs = ensemble.get("cpu_pair_distances", {})
    gpu_distances = candidate.get("gpu_vs_unperturbed_distances", {})
    if set(cpu_pairs) != set(gpu_distances) or not cpu_pairs:
        raise ValueError("N5S CPU/GPU metric inventories differ or are empty")
    cadence = ensemble.get("cadence")
    if (not isinstance(cadence, str) or not cadence
            or candidate.get("cadence") != cadence):
        raise ValueError("N5S CPU/GPU sampling cadences differ or are missing")
    rows = []
    passed = True
    degenerate_rows = 0
    expected_pairs = len(members) * (len(members) - 1) // 2
    coverage = {category: set() for category in N5S_CATEGORIES}
    for metric in sorted(cpu_pairs):
        pair_values = cpu_pairs[metric]
        if len(pair_values) != expected_pairs:
            raise ValueError(f"{metric} does not carry every unordered CPU pair")
        if any(not math.isfinite(float(value)) or float(value) < 0.0
               for value in pair_values):
            raise ValueError(f"{metric} carries a negative/non-finite distance")
        parts = metric.split(":")
        if len(parts) < 3 or parts[0] not in coverage:
            raise ValueError(
                f"N5S metric must be category:domain:field/object, got {metric!r}")
        category, domain = parts[:2]
        if domain not in N5S_DOMAINS:
            raise ValueError(f"N5S metric has unknown domain: {metric!r}")
        coverage[category].add(domain)
        envelope = _nearest_rank(pair_values, envelope_percentile)
        distance = float(gpu_distances[metric])
        if not math.isfinite(distance) or distance < 0.0:
            raise ValueError(
                f"N5S metric {metric!r} carries a negative/non-finite distance")
        envelope_degenerate = envelope == 0.0
        accepted = envelope_degenerate or distance <= envelope
        passed &= accepted
        row = {"metric": metric, "gpu_distance": distance,
               "cpu_e95": envelope, "passed": bool(accepted)}
        if envelope_degenerate:
            degenerate_rows += 1
            row.update({
                "documented_evidence": True,
                "envelope_degenerate": True,
                "adjudication": "f28-degenerate-envelope",
            })
        rows.append(row)
    expected_coverage = {
        category: ({"d04"} if category == "d04_reflectivity_fss_distance"
                   else set(N5S_DOMAINS))
        for category in N5S_CATEGORIES}
    if coverage != expected_coverage:
        raise ValueError(
            f"N5S domain/category coverage {coverage} != {expected_coverage}")
    metric = "N5S_matched_physics_wrf_shadow"
    all_envelopes_degenerate = degenerate_rows == len(rows)
    report = {
        "schema": 2, "metric": metric, "passed": bool(passed),
        "degenerate_rows": degenerate_rows,
        "documented_evidence": bool(degenerate_rows),
        "all_envelopes_degenerate": all_envelopes_degenerate,
        "candidate_evidence": provenance["candidate_evidence"],
        "candidate_evidence_sha256": provenance[
            "candidate_evidence_sha256"],
        "wrf_run_directory": str(wrf_run_directory),
        "ensemble_manifest": provenance["ensemble_manifest"],
        "ensemble_manifest_sha256": provenance[
            "ensemble_manifest_sha256"],
        "registration": provenance["registration"],
        "registration_sha256": provenance["registration_sha256"],
        "gpu_candidate": candidate["gpu_candidate"],
        "unperturbed": unperturbed, "members": verified_members,
        "restored_input_sha256": restored_hash,
        "wrf_build": expected_build,
        "minimum_duration_seconds": minimum_duration_s,
        "envelope_percentile": envelope_percentile,
        "required_domains": list(N5S_DOMAINS),
        "domain_coverage": {
            category: sorted(domains) for category, domains in coverage.items()},
        "expected_cpu_pairs": expected_pairs,
        "comparisons": rows,
        "baseline_hash": shared.stable_hash(
            [unperturbed["sha256"],
             *[member["sha256"] for member in verified_members]]),
        "generated_utc": shared._utc_now(),
        "evaluator_commit": shared._git_commit(),
    }
    report.update(_actual_evidence_pins(
        mask={"categories": list(N5S_CATEGORIES),
              "domains": list(N5S_DOMAINS),
              "metric_inventory": sorted(cpu_pairs)},
        baseline_hash=report["baseline_hash"],
        cadence=cadence,
        expected_samples=len(rows)))
    _bind_controller_score_report(report, score_inputs={
        "metric_inventory": sorted(cpu_pairs),
        "registration_sha256": provenance["registration_sha256"],
        "restored_input_sha256": restored_hash,
        "input_artifact_sha256": provenance["input_artifact_sha256"],
    })
    return report


# ---------------------------------------------------------------------------
# N5B boundary-location geometry and pre-look envelope
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class N5BGeometry:
    production_shape: tuple[int, int]
    shrink_shape: tuple[int, int]
    production_core: tuple[tuple[int, int], tuple[int, int]]
    shrink_core: tuple[tuple[int, int], tuple[int, int]]

    @property
    def core_shape(self) -> tuple[int, int]:
        return tuple(stop - start for start, stop in self.production_core)


def n5b_geometry() -> N5BGeometry:
    convention = _gate("N5", "N5B_d04_boundary_location_invariance").convention
    shapes = re.search(
        r"production d04 ([0-9]+)x([0-9]+); B .* to ([0-9]+)x([0-9]+); "
        r"the verification core is the central ([0-9]+)x([0-9]+)",
        convention)
    if shapes is None:
        raise ValueError("N5B record does not carry the registered geometry")
    ay, ax, by, bx, cy, cx = map(int, shapes.groups())

    def core(shape, size):
        start = (shape - size) // 2
        return start, start + size

    return N5BGeometry(
        production_shape=(ay, ax), shrink_shape=(by, bx),
        production_core=(core(ay, cy), core(ax, cx)),
        shrink_core=(core(by, cy), core(bx, cx)))


def construct_n5b_shrink_case(
        config_path: str | Path = shared.PRODUCTION_CONFIG):
    """Construct the controller's parent-aligned 498-square d04 variant.

    F22 removes 51 child cells per side, an exact 17-parent-cell anchor shift
    at ratio three.  The 498-cell extent is parent-aligned and its central
    400-square core coincides with the production core.  This remains a
    controller add-on object built after validation of the production TOML;
    no alternate production TOML is committed.  The N5B harness owns the
    separately generated SHRINK child inputs and central-core coordinate
    manifest before execution.
    """
    exp, data = shared.construct_rung_case(4, config_path=config_path)
    geometry = n5b_geometry()
    d04 = exp.domain(4)
    ratio = d04.parent_grid_ratio
    # F22's 51-child-cell core offset is exactly 17 parent cells at ratio
    # three, so the two central cores coincide without an anchor residual.
    i_child_shift = (
        geometry.production_core[1][0] - geometry.shrink_core[1][0])
    j_child_shift = (
        geometry.production_core[0][0] - geometry.shrink_core[0][0])
    i_shift, i_remainder = divmod(i_child_shift, ratio)
    j_shift, j_remainder = divmod(j_child_shift, ratio)
    if i_remainder or j_remainder:
        raise AssertionError(
            "N5B core shift is not integral on the parent lattice: "
            f"child_shift=({j_child_shift}, {i_child_shift}), ratio={ratio}")
    if any(extent % ratio for extent in geometry.shrink_shape):
        raise AssertionError(
            "N5B SHRINK extent is not divisible by the parent ratio: "
            f"shape={geometry.shrink_shape}, ratio={ratio}")
    shrink = replace(
        d04,
        i_parent_start=d04.i_parent_start + i_shift,
        j_parent_start=d04.j_parent_start + j_shift,
        run=replace(
            d04.run, nx=geometry.shrink_shape[1],
            ny=geometry.shrink_shape[0]))
    domains = tuple(shrink if dc.grid_id == 4 else dc for dc in exp.domains)
    return replace(exp, name=f"{exp.name}-N5B-SHRINK", domains=domains), data, geometry


def _n5b_stated_discrepancies() -> dict[str, float]:
    from gpuwm.verify.cases import real74_n5b

    return real74_n5b.ratified_discrepancy_limits()


@dataclass(frozen=True)
class VerifiedN5BFreeze:
    """Fully reconstructed pre-look authority required by shifted metrics."""

    path: Path
    sha256: str
    freeze_timestamp: str
    anchor_path: Path
    anchor_sha256: str
    evaluator_manifest_sha256: str
    ensemble_manifest_sha256: str
    allowed_discrepancies: Mapping[str, float]
    geometry: Mapping[str, object]
    dependency_sha256: Mapping[str, str]


def _validated_n5b_ensemble(
        ensemble_path: Path, source: Mapping[str, object]) -> dict[str, object]:
    from gpuwm.verify.cases import real74_n5b

    if (source.get("schema") != 1
            or source.get("metric") != real74_n5b.METRIC
            or source.get("evaluator_commit") != shared._git_commit()):
        raise ValueError("N5B ensemble identity/evaluator commit is stale")
    members = source.get("members")
    if (not isinstance(members, list)
            or len(members) < real74_n5b.F_GATE_SAME_GEOMETRY_MEMBERS):
        raise ValueError(
            f"N5B needs at least {real74_n5b.F_GATE_SAME_GEOMETRY_MEMBERS} "
            "same-geometry members")
    registered_geometry = shared.json_safe(asdict(n5b_geometry()))
    if shared.json_safe(source.get("geometry")) != registered_geometry:
        raise ValueError("N5B ensemble geometry is missing or not registered")

    evaluator_record = source.get("evaluator_manifest")
    if not isinstance(evaluator_record, dict):
        raise ValueError("N5B ensemble must bind its evaluator manifest")
    evaluator_verified = _verified_json_artifact(
        ensemble_path.parent, evaluator_record, "N5B evaluator manifest")
    evaluator = dict(evaluator_verified.document)
    real74_n5b._validated_evaluator(
        evaluator, dx_m=real74_n5b.F_GATE_D04_DX_M)

    verified_members = []
    member_snapshots = []
    for index, member in enumerate(members):
        if not isinstance(member, dict):
            raise ValueError("N5B ensemble member inventory is malformed")
        member_path, relative_path, expected_sha256 = _declared_artifact_path(
            ensemble_path.parent, member,
            f"N5B same-geometry member {index}")
        snapshot = real74_n5b._snapshot_member_record(
            member_path, perturbed=True, expected_sha256=expected_sha256)
        record = snapshot.document
        real74_n5b._validate_perturbation(
            member.get("one_ulp_perturbation"), member_id=record.get("id"))
        comparisons = {
            "id": record.get("id"),
            "duration_seconds": float(record.get("duration_seconds")),
            "restored_input_sha256": record.get("restored_input_sha256"),
            "core_coordinate_sha256": record.get("core_coordinate_sha256"),
            "one_ulp_perturbation": record.get("one_ulp_perturbation"),
        }
        if any(shared.json_safe(member.get(name)) != shared.json_safe(value)
               for name, value in comparisons.items()):
            raise ValueError(
                f"N5B same-geometry member {index} metadata differs from record")
        verified_members.append({
            **member, "relative_path": relative_path,
            "sha256": snapshot.sha256})
        member_snapshots.append(snapshot)
    if (len({member.get("id") for member in verified_members}) != len(members)
            or len({member["sha256"] for member in verified_members})
            != len(members)):
        raise ValueError("N5B ensemble members need unique ids and artifacts")

    baseline_record = source.get("unperturbed")
    if not isinstance(baseline_record, dict):
        raise ValueError("N5B ensemble lacks its unperturbed record")
    baseline_path, _, baseline_expected_sha256 = _declared_artifact_path(
        ensemble_path.parent, baseline_record, "N5B unperturbed member")
    baseline_snapshot = real74_n5b._snapshot_member_record(
        baseline_path, perturbed=False,
        expected_sha256=baseline_expected_sha256)
    baseline = baseline_snapshot.document
    for name, value in {
            "id": baseline.get("id"),
            "duration_seconds": float(baseline.get("duration_seconds")),
            "restored_input_sha256": baseline.get("restored_input_sha256"),
            "core_coordinate_sha256": baseline.get("core_coordinate_sha256"),
            }.items():
        if shared.json_safe(baseline_record.get(name)) != shared.json_safe(value):
            raise ValueError("N5B unperturbed metadata differs from its record")
    all_records = [baseline, *(item.document for item in member_snapshots)]
    if (any(record.get("variant") != "production" for record in all_records)
            or len({record.get("restored_input_sha256")
                    for record in all_records}) != 1
            or len({record.get("core_coordinate_sha256")
                    for record in all_records}) != 1):
        raise ValueError("N5B ensemble records are not one production geometry")
    if (source.get("restored_input_sha256")
            != baseline.get("restored_input_sha256")
            or source.get("core_coordinate_sha256")
            != baseline.get("core_coordinate_sha256")
            or evaluator.get("core_coordinate_sha256")
            != baseline.get("core_coordinate_sha256")):
        raise ValueError("N5B ensemble provenance differs from its records")

    stated = _n5b_stated_discrepancies()
    frozen_discrepancies = source.get("same_geometry_discrepancies")
    if (not isinstance(frozen_discrepancies, dict)
            or set(frozen_discrepancies) != set(stated)):
        raise ValueError("N5B same-geometry discrepancy inventory is incomplete")
    expected_pairs = len(members) * (len(members) - 1) // 2
    discrepancies = {name: [] for name in stated}
    for left, right in itertools.combinations(member_snapshots, 2):
        pair = real74_n5b.metrics_to_discrepancies(
            real74_n5b._evaluate_same_geometry_snapshot_pair(
                left, right, evaluator=evaluator))
        for metric, value in pair.items():
            discrepancies[metric].append(float(value))
    p95 = {}
    allowed = {}
    for metric, limit in stated.items():
        stored_values = frozen_discrepancies[metric]
        if not isinstance(stored_values, list) or len(stored_values) != expected_pairs:
            raise ValueError(f"N5B {metric} lacks unordered pair distances")
        if any(isinstance(value, bool)
               or not isinstance(value, (int, float))
               or not math.isfinite(float(value)) or float(value) < 0.0
               for value in stored_values):
            raise ValueError(
                f"N5B {metric} carries a negative/non-finite discrepancy")
        if shared.json_safe(stored_values) != shared.json_safe(
                discrepancies[metric]):
            raise ValueError(
                f"N5B {metric} differs from member-artifact re-derivation")
        p95[metric] = _nearest_rank(
            discrepancies[metric], real74_n5b.F_GATE_ENVELOPE_PERCENTILE)
        allowed[metric] = max(limit, p95[metric])
    dependencies = {
        str(evaluator_verified.path): evaluator_verified.sha256,
        **baseline_snapshot.dependency_sha256,
    }
    for snapshot in member_snapshots:
        dependencies.update(snapshot.dependency_sha256)
    return {
        "verified_members": verified_members,
        "stated": stated, "p95": p95, "allowed": allowed,
        "evaluator_manifest_sha256": evaluator_verified.sha256,
        "discrepancies_sha256": shared.stable_hash(discrepancies),
        "dependency_sha256": dependencies,
    }


def _n5b_freeze_anchor_path(frozen_path: str | Path) -> Path:
    frozen_path = Path(frozen_path).resolve()
    return frozen_path.with_name(f"{frozen_path.name}.anchor.json")


def _write_new_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        shared.json_safe(payload), indent=2, sort_keys=True,
        allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(encoded)


def freeze_n5b_envelope(ensemble_manifest: str | Path,
                        output: str | Path) -> dict[str, object]:
    """Write the envelope and external digest receipt before shifted metrics."""
    from gpuwm.verify.cases import real74_n5b

    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"N5B frozen envelope already exists: {output}")
    anchor_path = _n5b_freeze_anchor_path(output)
    if anchor_path.exists():
        raise FileExistsError(
            f"N5B external freeze anchor already exists: {anchor_path}")
    ensemble_manifest = Path(ensemble_manifest).resolve()
    if not ensemble_manifest.is_relative_to(output.parent):
        raise ValueError("N5B ensemble manifest must be beneath freeze root")
    ensemble_bytes = ensemble_manifest.read_bytes()
    ensemble_hash = hashlib.sha256(ensemble_bytes).hexdigest()
    try:
        source = json.loads(ensemble_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("N5B ensemble manifest is not valid JSON") from exc
    if not isinstance(source, dict):
        raise ValueError("N5B ensemble manifest must contain a JSON object")
    reconstructed = _validated_n5b_ensemble(ensemble_manifest, source)
    freeze_timestamp = shared._utc_now()
    frozen = {
        "schema": 1, "metric": real74_n5b.METRIC,
        "freeze_timestamp": freeze_timestamp,
        "evaluator_commit": shared._git_commit(),
        "geometry": asdict(n5b_geometry()),
        "members": reconstructed["verified_members"],
        "stated_limits": reconstructed["stated"],
        "same_geometry_p95": reconstructed["p95"],
        "envelope_percentile": real74_n5b.F_GATE_ENVELOPE_PERCENTILE,
        "allowed_discrepancies": reconstructed["allowed"],
        "same_geometry_discrepancies_sha256": reconstructed[
            "discrepancies_sha256"],
        "ensemble_manifest": {
            "relative_path": ensemble_manifest.relative_to(
                output.parent).as_posix(),
            "sha256": ensemble_hash,
        },
        "ensemble_manifest_sha256": ensemble_hash,
        "evaluator_manifest_sha256": reconstructed[
            "evaluator_manifest_sha256"],
    }
    _write_new_json(output, frozen)
    frozen_hash = shared.sha256_file(output)
    anchor = {
        "schema": 1, "metric": real74_n5b.METRIC,
        "freeze_timestamp": freeze_timestamp,
        "evaluator_commit": shared._git_commit(),
        "frozen_manifest": {
            "relative_path": output.name,
            "sha256": frozen_hash,
        },
    }
    _write_new_json(anchor_path, anchor)
    return frozen


def verify_n5b_freeze(
        frozen_manifest: str | Path, *,
        expected_anchor_sha256: str) -> VerifiedN5BFreeze:
    """Reconstruct a freeze against its independently retained anchor digest."""
    from gpuwm.verify.cases import real74_n5b

    frozen_path = Path(frozen_manifest).resolve()
    frozen_bytes = frozen_path.read_bytes()
    frozen_hash = hashlib.sha256(frozen_bytes).hexdigest()
    try:
        frozen = json.loads(frozen_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("N5B freeze is not valid JSON") from exc
    if not isinstance(frozen, dict):
        raise ValueError("N5B freeze must contain a JSON object")
    expected_keys = {
        "schema", "metric", "freeze_timestamp", "evaluator_commit",
        "geometry", "members", "stated_limits", "same_geometry_p95",
        "envelope_percentile", "allowed_discrepancies",
        "same_geometry_discrepancies_sha256", "ensemble_manifest",
        "ensemble_manifest_sha256", "evaluator_manifest_sha256",
    }
    if set(frozen) != expected_keys:
        raise ValueError("N5B freeze inventory is not closed-world")
    if (frozen.get("schema") != 1 or frozen.get("metric") != real74_n5b.METRIC
            or frozen.get("evaluator_commit") != shared._git_commit()):
        raise ValueError("N5B freeze identity/evaluator commit is stale")
    if (not isinstance(frozen.get("freeze_timestamp"), str)
            or not frozen["freeze_timestamp"]):
        raise ValueError("N5B freeze timestamp is missing")
    anchor_path = _n5b_freeze_anchor_path(frozen_path)
    anchor_bytes = anchor_path.read_bytes()
    anchor_hash = hashlib.sha256(anchor_bytes).hexdigest()
    if (not _is_sha256(expected_anchor_sha256)
            or anchor_hash != expected_anchor_sha256):
        raise ValueError(
            "N5B freeze does not match the external pre-look anchor digest")
    try:
        anchor = json.loads(anchor_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("N5B external freeze anchor is not valid JSON") from exc
    expected_anchor = {
        "schema": 1, "metric": real74_n5b.METRIC,
        "freeze_timestamp": frozen["freeze_timestamp"],
        "evaluator_commit": shared._git_commit(),
        "frozen_manifest": {
            "relative_path": frozen_path.name,
            "sha256": frozen_hash,
        },
    }
    if shared.json_safe(anchor) != shared.json_safe(expected_anchor):
        raise ValueError(
            "N5B freeze timestamp/hash differs from external pre-look anchor")
    geometry = shared.json_safe(asdict(n5b_geometry()))
    if shared.json_safe(frozen.get("geometry")) != geometry:
        raise ValueError("N5B freeze geometry is not registered")
    ensemble_record = frozen.get("ensemble_manifest")
    if not isinstance(ensemble_record, dict):
        raise ValueError("N5B freeze does not bind its ensemble manifest")
    verified_ensemble = _verified_json_artifact(
        frozen_path.parent, ensemble_record, "N5B frozen ensemble manifest")
    ensemble_path = verified_ensemble.path
    if frozen.get("ensemble_manifest_sha256") != verified_ensemble.sha256:
        raise ValueError("N5B freeze ensemble-manifest hash mismatch")
    reconstructed = _validated_n5b_ensemble(
        ensemble_path, verified_ensemble.document)
    checks = {
        "members": reconstructed["verified_members"],
        "stated_limits": reconstructed["stated"],
        "same_geometry_p95": reconstructed["p95"],
        "allowed_discrepancies": reconstructed["allowed"],
        "same_geometry_discrepancies_sha256": reconstructed[
            "discrepancies_sha256"],
        "evaluator_manifest_sha256": reconstructed[
            "evaluator_manifest_sha256"],
        "envelope_percentile": real74_n5b.F_GATE_ENVELOPE_PERCENTILE,
    }
    for name, expected in checks.items():
        if shared.json_safe(frozen.get(name)) != shared.json_safe(expected):
            raise ValueError(f"N5B freeze {name} differs from reconstruction")
    dependencies = {
        str(frozen_path): frozen_hash,
        str(anchor_path): anchor_hash,
        str(verified_ensemble.path): verified_ensemble.sha256,
        **reconstructed["dependency_sha256"],
    }
    return VerifiedN5BFreeze(
        path=frozen_path, sha256=frozen_hash,
        freeze_timestamp=frozen["freeze_timestamp"],
        anchor_path=anchor_path, anchor_sha256=anchor_hash,
        evaluator_manifest_sha256=reconstructed[
            "evaluator_manifest_sha256"],
        ensemble_manifest_sha256=verified_ensemble.sha256,
        allowed_discrepancies=dict(reconstructed["allowed"]),
        geometry=asdict(n5b_geometry()), dependency_sha256=dependencies)


def _recheck_n5b_dependencies(verified: VerifiedN5BFreeze) -> None:
    for path, expected in verified.dependency_sha256.items():
        target = Path(path)
        if not target.is_file() or shared.sha256_file(target) != expected:
            raise ValueError(
                f"N5B freeze dependency changed before scoring: {target}")


def score_n5b_observation(
        frozen_manifest: str | Path, observation: str | Path, *,
        freeze_anchor_sha256: str) -> dict[str, object]:
    """Score the five registered predicates against a hash-pinned freeze."""
    from gpuwm.verify.cases import real74_n5b

    verified_freeze = verify_n5b_freeze(
        frozen_manifest, expected_anchor_sha256=freeze_anchor_sha256)
    frozen_path = verified_freeze.path
    observation_path = Path(observation).resolve()
    observation_bytes = observation_path.read_bytes()
    try:
        observed = json.loads(observation_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("N5B observation is not valid JSON") from exc
    if not isinstance(observed, dict):
        raise ValueError("N5B observation must contain a JSON object")
    if (observed.get("schema") != 1 or observed.get("metric") != real74_n5b.METRIC
            or observed.get("evaluator_commit") != shared._git_commit()):
        raise ValueError("N5B observation identity/evaluator commit is stale")
    frozen_hash = verified_freeze.sha256
    if observed.get("frozen_manifest_sha256") != frozen_hash:
        raise ValueError("N5B observation does not pin the pre-look freeze")
    if observed.get("freeze_anchor_sha256") != verified_freeze.anchor_sha256:
        raise ValueError("N5B observation does not pin the external freeze anchor")
    if (observed.get("evaluator_manifest_sha256")
            != verified_freeze.evaluator_manifest_sha256):
        raise ValueError("N5B observation does not pin the frozen evaluator")
    evaluator_record = observed.get("evaluator_manifest")
    if not isinstance(evaluator_record, dict):
        raise ValueError("N5B observation lacks its evaluator manifest")
    evaluator_verified = _verified_json_artifact(
        observation_path.parent, evaluator_record,
        "N5B observation evaluator manifest")
    if evaluator_verified.sha256 != verified_freeze.evaluator_manifest_sha256:
        raise ValueError("N5B observation evaluator differs from the freeze")
    evaluator = dict(evaluator_verified.document)
    real74_n5b._validated_evaluator(
        evaluator, dx_m=real74_n5b.F_GATE_D04_DX_M)
    artifact_records = observed.get("artifacts")
    if (not isinstance(artifact_records, dict)
            or set(artifact_records) != {"production", "shrink"}
            or not all(isinstance(value, dict)
                       for value in artifact_records.values())):
        raise ValueError("N5B observation needs production/shrink artifacts")
    restored_hashes = {
        record.get("restored_input_sha256")
        for record in artifact_records.values()}
    if len(restored_hashes) != 1 or not _is_sha256(next(iter(restored_hashes))):
        raise ValueError("N5B production/SHRINK restored-input hashes differ")
    core_coordinate_hashes = {
        record.get("core_coordinate_sha256")
        for record in artifact_records.values()}
    if (len(core_coordinate_hashes) != 1
            or not _is_sha256(next(iter(core_coordinate_hashes)))):
        raise ValueError("N5B production/SHRINK common-core coordinates differ")
    if evaluator.get("core_coordinate_sha256") != next(iter(
            core_coordinate_hashes)):
        raise ValueError("N5B observation evaluator coordinates differ")
    convention = _gate(
        "N5", "N5B_d04_boundary_location_invariance").convention
    duration_match = re.search(r"execute two sequential ([0-9.]+)-min runs",
                               convention)
    if duration_match is None:
        raise ValueError("N5B record has no run duration")
    expected_duration = float(duration_match.group(1)) * 60.0
    if any(not math.isfinite(float(record.get("duration_seconds", float("nan"))))
           or float(record.get("duration_seconds", float("nan")))
           != expected_duration for record in artifact_records.values()):
        raise ValueError("N5B production/SHRINK durations are not registered")
    artifacts = {
        name: {
            **_verified_artifact(
                observation_path.parent, record, f"N5B {name}"),
            "restored_input_sha256": record["restored_input_sha256"],
            "core_coordinate_sha256": record["core_coordinate_sha256"],
            "duration_seconds": float(record["duration_seconds"]),
        }
        for name, record in artifact_records.items()}
    raw = observed.get("metrics", {})
    required = {
        "refl_fss", "cold_pool_edge_km", "gust_front_arrival_mae_min",
        "unmatched_boundary_ci_count", "tke_ratio",
    }
    if set(raw) != required:
        raise ValueError("N5B observation metric inventory is incomplete")
    cadence = observed.get("cadence")
    if not isinstance(cadence, str) or not cadence:
        raise ValueError("N5B observation sampling cadence is missing")
    values = {key: float(value) for key, value in raw.items()}
    valid_source = {
        "refl_fss": math.isfinite(values["refl_fss"])
                    and 0.0 <= values["refl_fss"] <= 1.0,
        "cold_pool_edge_km": math.isfinite(values["cold_pool_edge_km"])
                             and values["cold_pool_edge_km"] >= 0.0,
        "gust_front_arrival_mae_min":
            math.isfinite(values["gust_front_arrival_mae_min"])
            and values["gust_front_arrival_mae_min"] >= 0.0,
        "unmatched_boundary_ci_count":
            math.isfinite(values["unmatched_boundary_ci_count"])
            and values["unmatched_boundary_ci_count"] >= 0.0
            and values["unmatched_boundary_ci_count"].is_integer(),
        "tke_ratio": math.isfinite(values["tke_ratio"])
                     and values["tke_ratio"] > 0.0,
    }
    discrepancies = {
        "one_minus_refl_fss": 1.0 - values["refl_fss"],
        "cold_pool_edge_km": values["cold_pool_edge_km"],
        "gust_front_arrival_mae_min": values["gust_front_arrival_mae_min"],
        "unmatched_boundary_ci_count": values["unmatched_boundary_ci_count"],
        "tke_lower_excursion": max(0.0, 1.0 - values["tke_ratio"]),
        "tke_upper_excursion": max(0.0, values["tke_ratio"] - 1.0),
    }
    source_for = {
        "one_minus_refl_fss": "refl_fss",
        "cold_pool_edge_km": "cold_pool_edge_km",
        "gust_front_arrival_mae_min": "gust_front_arrival_mae_min",
        "unmatched_boundary_ci_count": "unmatched_boundary_ci_count",
        "tke_lower_excursion": "tke_ratio",
        "tke_upper_excursion": "tke_ratio",
    }
    rows = []
    passed = True
    for metric, value in discrepancies.items():
        allowed = float(verified_freeze.allowed_discrepancies[metric])
        accepted = (valid_source[source_for[metric]]
                    and math.isfinite(value) and 0.0 <= value <= allowed)
        passed &= accepted
        rows.append({"metric": metric, "value": value,
                     "allowed": allowed, "passed": bool(accepted)})
    baseline_hash = shared.stable_hash({
        "production_sha256": artifacts["production"]["sha256"],
        "frozen_manifest_sha256": frozen_hash,
    })
    report = {
        "schema": 2, "metric": "N5B_d04_boundary_location_invariance",
        "passed": bool(passed), "frozen_manifest": str(frozen_path),
        "frozen_manifest_sha256": frozen_hash,
        "observation": str(observation_path), "comparisons": rows,
        "geometry": verified_freeze.geometry, "artifacts": artifacts,
        "baseline_hash": baseline_hash,
        "generated_utc": shared._utc_now(),
        "evaluator_commit": shared._git_commit(),
        "freeze_timestamp": verified_freeze.freeze_timestamp,
        "freeze_anchor_sha256": verified_freeze.anchor_sha256,
        "freeze_dependency_sha256": dict(verified_freeze.dependency_sha256),
    }
    report.update(_actual_evidence_pins(
        mask={"geometry": verified_freeze.geometry, "metric": "central-core",
              "metric_inventory": sorted(required)},
        baseline_hash=baseline_hash,
        cadence=cadence,
        expected_samples=len(rows)))
    input_artifact_sha256 = {
        "observation": hashlib.sha256(observation_bytes).hexdigest(),
        "production": artifacts["production"]["sha256"],
        "shrink": artifacts["shrink"]["sha256"],
        **{
            f"freeze_dependency:{index}": digest
            for index, (_path, digest) in enumerate(sorted(
                verified_freeze.dependency_sha256.items()))
        },
    }
    _bind_controller_score_report(report, score_inputs={
        "metric_inventory": sorted(row["metric"] for row in rows),
        "source_metric_inventory": sorted(required),
        "input_artifact_sha256": input_artifact_sha256,
    })
    _recheck_n5b_dependencies(verified_freeze)
    return report


def evaluate_d04_budget_records(path: str | Path) -> dict[str, object]:
    """Pre-N6 evaluator for the ledger's d04 budget/clipping record.

    The additive record is registered at N6 (not N5) in the authoritative
    table.  N5 can produce and validate this readiness artifact without
    silently reclassifying it as an N5 threshold.
    """
    record = _gate("N6", "d04_budget_records")
    with Path(path).open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    windows = payload.get("windows", [])
    if not windows:
        return {"metric": record.metric, "registered_milestone": "N6",
                "passed": False, "reason": "no budget windows"}
    dry = re.search(
        r"\|R_dry\| <= max\(([0-9.e+-]+)\*\|R_dry_oracle\|, "
        r"([0-9.e+-]+)\*dry_mass_throughput\)", record.convention)
    water = re.search(
        r"\|R_water\| <= max\(([0-9.e+-]+)\*\|R_water_oracle\|, "
        r"([0-9.e+-]+)\*water_throughput\)", record.convention)
    clipping = re.search(
        r"max\(([0-9.e+-]+)\*oracle_fraction, ([0-9.e+-]+)\)",
        record.convention)
    if dry is None or water is None or clipping is None:
        raise ValueError("d04 budget record does not expose its registered bounds")
    dry_oracle_factor, dry_throughput_factor = map(float, dry.groups())
    water_oracle_factor, water_throughput_factor = map(float, water.groups())
    clip_oracle_factor, clip_floor = map(float, clipping.groups())

    def finite_breakdown(value) -> bool:
        if not isinstance(value, dict) or not value:
            return False
        for item in value.values():
            if isinstance(item, dict):
                if not finite_breakdown(item):
                    return False
            else:
                try:
                    number = float(item)
                except (TypeError, ValueError):
                    return False
                if not math.isfinite(number):
                    return False
        return True

    rows, passed = [], True
    for window in windows:
        dry_storage = float(window["dry_mass_delta_storage"])
        dry_boundary = float(window["dry_mass_net_boundary_outflow"])
        dry_sources = float(window["dry_mass_net_sources"])
        dry_signed_residual = dry_storage + dry_boundary - dry_sources
        dry_residual_record = float(window["dry_mass_residual"])
        dry_residual = abs(dry_signed_residual)
        dry_oracle = abs(float(window["dry_mass_oracle_residual"]))
        dry_throughput = (abs(dry_storage) + abs(dry_boundary)
                          + abs(dry_sources))
        dry_throughput_record = float(window["dry_mass_throughput"])
        water_storage = float(window["total_water_delta_storage"])
        water_boundary = float(window["total_water_net_boundary_outflow"])
        water_sources = float(window["total_water_net_sources"])
        water_signed_residual = water_storage + water_boundary - water_sources
        water_residual_record = float(window["total_water_residual"])
        water_residual = abs(water_signed_residual)
        water_oracle = abs(float(window["total_water_oracle_residual"]))
        water_throughput = (abs(water_storage) + abs(water_boundary)
                            + abs(water_sources))
        water_throughput_record = float(window["total_water_throughput"])
        clipped = float(window["cumulative_clipped_condensate_fraction"])
        clip_oracle = float(window["oracle_clipped_condensate_fraction"])
        dry_limit = max(dry_oracle_factor * dry_oracle,
                        dry_throughput_factor * dry_throughput)
        water_limit = max(water_oracle_factor * water_oracle,
                          water_throughput_factor * water_throughput)
        clip_limit = max(clip_oracle_factor * clip_oracle, clip_floor)
        finite = all(math.isfinite(value) for value in (
            dry_storage, dry_boundary, dry_sources, dry_residual_record,
            dry_oracle, dry_throughput_record, water_storage, water_boundary,
            water_sources, water_residual_record, water_oracle,
            water_throughput_record, clipped, clip_oracle))
        formula_exact = (dry_residual_record == dry_signed_residual
                         and dry_throughput_record == dry_throughput
                         and water_residual_record == water_signed_residual
                         and water_throughput_record == water_throughput)
        valid_throughput = (dry_throughput >= 0.0 and water_throughput >= 0.0
                            and not (dry_throughput == 0.0
                                     and dry_residual != 0.0)
                            and not (water_throughput == 0.0
                                     and water_residual != 0.0))
        accepted = (finite and formula_exact and valid_throughput
                    and dry_residual <= dry_limit
                    and water_residual <= water_limit
                    and 0.0 <= clipped <= clip_limit
                    and clip_oracle >= 0.0
                    and finite_breakdown(window.get("per_species_breakdown"))
                    and finite_breakdown(
                        window.get("boundary_interior_breakdown")))
        passed &= accepted
        rows.append({**window,
                     "computed_dry_mass_residual": dry_signed_residual,
                     "computed_dry_mass_throughput": dry_throughput,
                     "computed_total_water_residual": water_signed_residual,
                     "computed_total_water_throughput": water_throughput,
                     "formula_exact": formula_exact,
                     "dry_mass_residual_limit": dry_limit,
                     "total_water_limit": water_limit,
                     "clipping_limit": clip_limit, "passed": bool(accepted)})
    return {"metric": record.metric, "registered_milestone": "N6",
            "passed": bool(passed), "windows": rows,
            "convention": record.convention}


def _validate_predecessor_manifests(
        exp, case_data, ratchet_root: str | Path,
        predecessors: Iterable[tuple[str, int]]) -> None:
    """Hash/config-check binding controls before any tree or GPU setup."""
    from gpuwm.ingest.preflight import build_input_catalog

    predecessors = tuple(predecessors)
    manifests = {
        rung: shared.load_ratchet(ratchet_root, rung)
        for rung, _domain_count in predecessors
    }
    catalog = build_input_catalog(case_data)
    for rung, domain_count in predecessors:
        manifest = manifests[rung]
        expected = shared.experiment_prefix_provenance(
            exp, catalog, domain_count)
        expected["evaluator_commit"] = shared._git_commit()
        shared.validate_ratchet_provenance(manifest, expected)


def run_n4(args) -> dict[str, object]:
    exp, data = shared.construct_rung_case(3, config_path=args.config)
    _validate_predecessor_manifests(
        exp, data, args.ratchet_root, (("N3", 2),))
    verdicts = shared.load_verdicts(args.verdicts)
    if args.existing_report is None:
        run = shared.execute_production_run(exp, data, args.outdir / "straight")
    else:
        with args.existing_report.open("r", encoding="utf-8") as stream:
            run = shared.production_run_from_report(json.load(stream))
    report = evaluate_n4(
        exp, data, run, phase4_root=args.phase4_root,
        ratchet_root=args.ratchet_root,
        verdicts=verdicts)
    report["config_source"] = str(args.config.resolve())
    report_path = shared.write_json(args.outdir / "N4-report.json", report)
    report["report_path"] = str(report_path)
    if report["passed"]:
        d03_inventory = shared.state_hash_inventory(
            exp, run.execution, "d03")
        if not d03_inventory["passed"]:
            raise RuntimeError("N4 d03 canonical-state inventory is incomplete")
        d03_hashes = shared.state_hashes_for_domain(run.execution, "d03")
        ratchet = shared.publish_ratchet(
            args.ratchet_root, "N4", "d03", run.paths_for_domain(3),
            provenance=shared.run_prefix_provenance(run, 3),
            expected_state_domains=shared.RATCHET_STATE_EVIDENCE_DOMAINS["N4"],
            state_hashes=d03_hashes,
            state_hash_schedules=(shared.state_hash_schedule(exp, "d03"),))
        report["ratchet_manifest"] = str(ratchet)
        report["ratchet_state_inventories"] = {"d03": d03_inventory}
        shared.write_json(report_path, report)
    return report


def run_n5(args) -> dict[str, object]:
    exp, data = shared.construct_rung_case(4, config_path=args.config)
    _validate_predecessor_manifests(
        exp, data, args.ratchet_root, (("N3", 2), ("N4", 3)))
    verdicts = shared.load_verdicts(args.verdicts)
    n5s = _load_gate_report(
        args.n5s_report, "N5S_matched_physics_wrf_shadow")
    n5b = _load_gate_report(
        args.n5b_report, "N5B_d04_boundary_location_invariance")
    if args.existing_report is None:
        run = shared.execute_production_run(exp, data, args.outdir / "straight")
        restart = shared.restart_split_stage(
            exp, data, straight=run, work_dir=args.outdir / "restart-split")
    else:
        with args.existing_report.open("r", encoding="utf-8") as stream:
            prior = json.load(stream)
        run = shared.production_run_from_report(prior)
        restart = shared.gate_evidence_from_report(
            prior, "full_tree_restart_bit_identity")
    report = evaluate_n5(
        exp, data, run, phase4_root=args.phase4_root,
        ratchet_root=args.ratchet_root,
        verdicts=verdicts,
        restart_evidence=restart, n5s_report=n5s, n5b_report=n5b,
        config_path=args.config)
    report["config_source"] = str(args.config.resolve())
    if args.d04_budget_report is not None:
        report["pre_n6_add_on"] = evaluate_d04_budget_records(
            args.d04_budget_report)
    report_path = shared.write_json(args.outdir / "N5-report.json", report)
    report["report_path"] = str(report_path)
    shared.write_json(report_path, report)
    return report


def _common_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=shared.PRODUCTION_CONFIG)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--ratchet-root", type=Path,
                        default=shared.REPOSITORY_ROOT / "out" / "rungs")
    parser.add_argument("--phase4-root", type=Path,
                        default=shared.default_phase4_root())
    parser.add_argument("--verdicts", type=Path,
                        help="JSON structural verdicts; missing verdicts fail")
    parser.add_argument(
        "--existing-report", type=Path,
        help="re-evaluate existing GPU evidence after controller review")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m gpuwm.verify.cases.real74_chain")
    commands = parser.add_subparsers(dest="command", required=True)
    n4 = commands.add_parser("N4", help="run/evaluate the three-domain rung")
    _common_run_arguments(n4)
    n5 = commands.add_parser("N5", help="run/evaluate the full-chain rung")
    _common_run_arguments(n5)
    n5.add_argument("--n5s-report", type=Path)
    n5.add_argument("--n5b-report", type=Path)
    n5.add_argument("--d04-budget-report", type=Path)

    n5s = commands.add_parser("N5S", help="score external CPU-WRF envelope")
    n5s.add_argument("--candidate-evidence", type=Path, required=True)
    n5s.add_argument("--wrf-run-directory", type=Path, required=True)
    n5s.add_argument("--output", type=Path, required=True)

    plan = commands.add_parser("N5B-plan", help="write SHRINK geometry manifest")
    plan.add_argument("--config", type=Path, default=shared.PRODUCTION_CONFIG)
    plan.add_argument("--output", type=Path, required=True)
    freeze = commands.add_parser("N5B-freeze", help="freeze pre-look P95 envelope")
    freeze.add_argument("--ensemble-manifest", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    score = commands.add_parser("N5B-score", help="score shifted observation")
    score.add_argument("--frozen-manifest", type=Path, required=True)
    score.add_argument("--freeze-anchor-sha256", required=True)
    score.add_argument("--observation", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "N4":
        report = run_n4(args)
    elif args.command == "N5":
        report = run_n5(args)
    elif args.command == "N5S":
        report = evaluate_n5s_shadow(
            args.candidate_evidence, args.wrf_run_directory)
        shared.write_json(args.output, report)
    elif args.command == "N5B-plan":
        exp, _data, geometry = construct_n5b_shrink_case(args.config)
        report = {
            "schema": 1, "metric": "N5B_d04_boundary_location_invariance",
            "production_config": str(args.config), "variant": exp.name,
            "d04": {"nx": exp.domain(4).run.nx,
                    "ny": exp.domain(4).run.ny,
                    "i_parent_start": exp.domain(4).i_parent_start,
                    "j_parent_start": exp.domain(4).j_parent_start},
            "geometry": asdict(geometry),
            "requires_controller_generated_shrink_inputs": True,
            "requires_common_core_coordinate_sha256": True,
            "gate_convention": _gate(
                "N5", "N5B_d04_boundary_location_invariance").convention,
        }
        shared.write_json(args.output, report)
    elif args.command == "N5B-freeze":
        report = freeze_n5b_envelope(args.ensemble_manifest, args.output)
        anchor_path = _n5b_freeze_anchor_path(args.output)
        report = {
            **report,
            "external_pre_look_anchor": {
                "path": str(anchor_path),
                "sha256": shared.sha256_file(anchor_path),
            },
        }
    else:
        report = score_n5b_observation(
            args.frozen_manifest, args.observation,
            freeze_anchor_sha256=args.freeze_anchor_sha256)
        shared.write_json(args.output, report)
    print(json.dumps(shared.json_safe(report), indent=2, sort_keys=True,
                     allow_nan=False))
    return 0 if bool(report.get("passed", True)) else 1


__all__ = [
    "N4_METRICS", "N5_ANCESTOR_CONTROL_DOMAINS", "N5_METRICS",
    "N5S_CATEGORIES", "N5S_DOMAINS",
    "N5BGeometry", "VerifiedN5BFreeze", "ancestor_inertness_evidence",
    "construct_n5b_shrink_case", "domain_mean_profile_evidence",
    "evaluate_d04_budget_records", "evaluate_n4", "evaluate_n5",
    "evaluate_n5s_shadow", "freeze_n5b_envelope", "full_schedule_sync_ledger",
    "n5b_geometry", "score_n5b_observation", "updraft_percentile_evidence",
    "verify_n5b_freeze",
    "w_cfl_evidence",
]


if __name__ == "__main__":  # pragma: no cover - controller entry point
    raise SystemExit(main())
