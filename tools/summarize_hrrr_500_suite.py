#!/usr/bin/env python3
"""Fail-closed summary for the native-HRRR 500 x 500 benchmark suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from tools.hrrr_build_native_static import benchmark_grid
from gpuwm.ingest.hrrr import hrrr_source_grid


def _read(path: Path) -> dict:
    value = json.loads(path.read_text())
    if value.get("status") != "PASS":
        raise ValueError(f"{path} is not a PASS receipt")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _healthy(report: dict, label: str) -> None:
    health = report["health"]
    if (not health["initial"]["ok"] or not health["final"]["ok"]
            or health["final_stability"]["nan"]):
        raise ValueError(f"{label} health receipt is not clean")
    if health["final_stability"]["cfl"] > health["limits"]["max_cfl"]:
        raise ValueError(f"{label} final CFL exceeds its sealed limit")
    if health["final_stability"]["w_max"] > health["limits"]["max_w_ms"]:
        raise ValueError(f"{label} final w exceeds its sealed limit")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--compute", type=Path, required=True)
    parser.add_argument("--io", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    static = _read(args.static)
    gate = _read(args.gate)
    compute = _read(args.compute)
    io = _read(args.io)

    expected_schema = "gpuwm-native-hrrr-500x500-benchmark-v1"
    for label, report in (("gate", gate), ("compute", compute), ("io", io)):
        if report.get("schema") != expected_schema:
            raise ValueError(f"{label} schema mismatch")
        if report.get("geometry") != {
                "nx": 500, "ny": 500, "nz": 49,
                "dx_m": 999.8071015811862,
                "dy_m": 999.8071015811862}:
            raise ValueError(f"{label} geometry drift")
        _healthy(report, label)
    if gate["run_seconds"] != 300.0 or gate["io_mode"] != "none":
        raise ValueError("short gate duration/mode drift")
    if (compute["run_seconds"] != 43200.0 or compute["io_mode"] != "none"
            or io["run_seconds"] != 43200.0 or io["io_mode"] != "hourly"):
        raise ValueError("full benchmark duration/mode drift")

    static_sha = static["cache"]["sha256"]
    static_hashes = {
        report["input"]["native_static_cache_sha256"]
        for report in (gate, compute, io)}
    if static_hashes != {static_sha}:
        raise ValueError("benchmark runs do not share the sealed static cache")
    bridge_hashes = {
        report["input"]["bridge_manifest_sha256"]
        for report in (gate, compute, io)}
    if bridge_hashes != {
            "ec9e5f6013ed297a012fb21291d933063284a089302f85b7b297c906f274884d"}:
        raise ValueError("benchmark runs do not share the exact bridge inventory")

    lat, lon = benchmark_grid().latlon_mass()
    source_x, source_y = hrrr_source_grid().latlon_to_ij(lat, lon)
    coverage_cells = {
        "west": float(source_x.min() - 781.0),
        "east": float(987.0 - source_x.max()),
        "south": float(source_y.min() - 315.0),
        "north": float(521.0 - source_y.max()),
    }
    if min(coverage_cells.values()) < 3.0:
        raise ValueError(
            "500x500 target lacks the three-cell native-source halo")

    compute_digest = compute["final_state_digest"]
    io_digest = io["final_state_digest"]
    if (compute_digest["sha256"] != io_digest["sha256"]
            or compute_digest["inventory_sha256"]
            != io_digest["inventory_sha256"]):
        raise ValueError("hourly gridded output changed the final trajectory")
    gridded = io["gridded_output"]
    if (gridded["cadence_seconds"] != 3600.0
            or gridded["frame_count"] != 13
            or len(gridded["files"]) != 13):
        raise ValueError("hourly gridded output inventory is incomplete")

    compute_seconds = compute["timing_seconds"]["forecast_execution"]
    io_execution = io["timing_seconds"]["forecast_execution_with_async_io"]
    io_inclusive = io["timing_seconds"]["forecast_and_io_inclusive"]
    summary = {
        "schema": "gpuwm-native-hrrr-500x500-suite-summary-v1",
        "status": "PASS",
        "identity": {
            "native_static_sha256": static_sha,
            "bridge_manifest_sha256": next(iter(bridge_hashes)),
            "final_trajectory_sha256": compute_digest["sha256"],
            "trajectory_inventory_sha256": compute_digest[
                "inventory_sha256"],
            "source_commits": sorted({
                report["source_identity"]["git_commit"]
                for report in (gate, compute, io)}),
        },
        "source_coverage": {
            "native_window_zero_based_inclusive": {
                "i": [781, 987], "j": [315, 521]},
            "target_source_coordinate_range": {
                "i": [float(source_x.min()), float(source_x.max())],
                "j": [float(source_y.min()), float(source_y.max())]},
            "margin_source_cells": coverage_cells,
            "margin_km": {
                side: value * 3.0 for side, value in coverage_cells.items()},
            "minimum_required_source_cells": 3.0,
            "status": "PASS",
        },
        "cold_static": static["timing_seconds"],
        "gate": {
            "downloaded_hrrr_to_first_gpu_step_seconds": gate[
                "downloaded_hrrr_to_first_gpu_step_seconds"],
            "forecast_execution_seconds": gate["timing_seconds"][
                "forecast_execution"],
            "simulated_seconds_per_wall_second": gate[
                "integration_simulated_seconds_per_wall_second"],
            "gpu_peak_used_bytes": gate["memory"][
                "gpu_peak_used_bytes_observed"],
            "final_stability": gate["health"]["final_stability"],
        },
        "full_compute": {
            "forecast_execution_seconds": compute_seconds,
            "simulated_seconds_per_wall_second": compute[
                "integration_simulated_seconds_per_wall_second"],
            "gpu_peak_used_bytes": compute["memory"][
                "gpu_peak_used_bytes_observed"],
            "final_stability": compute["health"]["final_stability"],
        },
        "full_hourly_io": {
            "forecast_execution_with_async_io_seconds": io_execution,
            "forecast_and_io_inclusive_seconds": io_inclusive,
            "integration_slowdown_ratio_vs_compute": (
                io_execution / compute_seconds),
            "inclusive_slowdown_ratio_vs_compute": (
                io_inclusive / compute_seconds),
            "gridded_output_frames": gridded["frame_count"],
            "gridded_output_bytes": gridded["total_bytes"],
            "gpu_peak_used_bytes": io["memory"][
                "gpu_peak_used_bytes_observed"],
            "final_stability": io["health"]["final_stability"],
        },
        "input_receipts": {
            str(path.resolve()): _sha256(path)
            for path in (args.static, args.gate, args.compute, args.io)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({
        "status": "PASS",
        "compute_simulated_seconds_per_wall_second": summary[
            "full_compute"]["simulated_seconds_per_wall_second"],
        "io_inclusive_slowdown_ratio": summary[
            "full_hourly_io"]["inclusive_slowdown_ratio_vs_compute"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
