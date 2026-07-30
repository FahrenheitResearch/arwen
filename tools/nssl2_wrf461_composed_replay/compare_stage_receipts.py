"""Attribute composed-column differences to the first production stage."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


REGISTRY_NAMES = (
    "qv", "qc", "qr", "qi", "qs", "qg", "qh", "qndrop", "qnr",
    "qni", "qns", "qng", "qnh", "qnn", "qvolg", "qvolh",
)
CPU_AN_INDEX = {
    "theta": 1,
    "qv": 2,
    "qc": 3,
    "qr": 4,
    "qi": 5,
    "qs": 6,
    "qg": 7,
    "qh": 8,
    "qnn": 9,
    "qndrop": 10,
    "qnr": 11,
    "qni": 12,
    "qns": 13,
    "qng": 14,
    "qnh": 15,
    "qvolg": 16,
    "qvolh": 17,
}
STAGES = (
    ("post_sediment", "post_sediment_concentration"),
    ("post_fused", "post_fused_concentration"),
    ("post_nucond", "post_nucond_concentration"),
)


def _tolerance(name: str) -> tuple[float, float]:
    if name == "theta":
        return 3.0e-6, 5.0e-5
    if name in {"qv", "qc", "qr", "qi", "qs", "qg", "qh"}:
        return 3.0e-5, 3.0e-9
    if name in {"qndrop", "qnr", "qni", "qns", "qng", "qnh", "qnn"}:
        return 3.0e-5, 5.0e-2 if name == "qni" else 64.0
    if name in {"qvolg", "qvolh"}:
        return 3.0e-5, 3.0e-12
    raise KeyError(name)


def _cpu_stages(path: Path) -> dict[str, list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["stage"], []).append(row)
    for stage_rows in grouped.values():
        stage_rows.sort(key=lambda row: int(row["k"]))
    return grouped


def _field_report(expected: np.ndarray, actual: np.ndarray, name: str) -> dict:
    expected = np.asarray(expected, dtype=np.float32).reshape(-1)
    actual = np.asarray(actual, dtype=np.float32).reshape(-1)
    if expected.shape != actual.shape:
        raise ValueError(f"{name} shape mismatch: {expected.shape} != {actual.shape}")
    finite = np.isfinite(expected) & np.isfinite(actual)
    absolute = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    rtol, atol = _tolerance(name)
    limit = atol + rtol * np.abs(expected.astype(np.float64))
    violations = (~finite) | (absolute > limit)
    exact = actual.view(np.uint32) == expected.view(np.uint32)
    worst = int(np.nanargmax(np.where(finite, absolute / limit, np.inf)))
    mismatch_levels = np.flatnonzero(~exact)
    return {
        "rtol": rtol,
        "atol": atol,
        "violations": int(np.count_nonzero(violations)),
        "exact_mismatches": int(np.count_nonzero(~exact)),
        "first_exact_mismatch_k": (
            int(mismatch_levels[0] + 1) if mismatch_levels.size else None
        ),
        "nonfinite": int(np.count_nonzero(~finite)),
        "max_abs_error": float(np.nanmax(absolute)),
        "max_error_over_limit": float(absolute[worst] / limit[worst]),
        "worst_k": worst + 1,
        "wrf_at_worst": float(expected[worst]),
        "gpu_at_worst": float(actual[worst]),
    }


def compare(cpu_trace: Path, gpu_receipt: Path, report_path: Path) -> dict:
    cpu = _cpu_stages(cpu_trace)
    report: dict[str, object] = {
        "schema": "gpuwm.nssl2.composed-stage-comparison/v1",
        "cpu_trace": str(cpu_trace.resolve()),
        "gpu_receipt": str(gpu_receipt.resolve()),
        "stages": {},
    }
    total_violations = 0
    with np.load(gpu_receipt, allow_pickle=False) as gpu:
        for cpu_name, gpu_name in STAGES:
            rows = cpu.get(cpu_name)
            if not rows:
                raise ValueError(f"CPU trace lacks stage {cpu_name!r}")
            state = np.asarray(gpu[gpu_name], dtype=np.float32)
            theta = np.asarray(gpu[f"{gpu_name}_theta"], dtype=np.float32)
            fields = {
                "theta": _field_report(
                    [float(row[f"an{CPU_AN_INDEX['theta']:02d}"]) for row in rows],
                    theta,
                    "theta",
                )
            }
            for field_index, name in enumerate(REGISTRY_NAMES):
                fields[name] = _field_report(
                    [float(row[f"an{CPU_AN_INDEX[name]:02d}"]) for row in rows],
                    state[field_index],
                    name,
                )
            stage_violations = sum(
                int(field["violations"]) for field in fields.values()
            )
            total_violations += stage_violations
            report["stages"][cpu_name] = {
                "status": "PASS" if stage_violations == 0 else "FAIL",
                "violations": stage_violations,
                "fields": fields,
            }
    report["total_violations"] = total_violations
    report["status"] = "PASS" if total_violations == 0 else "FAIL"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": report["status"],
        "total_violations": total_violations,
        "stages": {
            name: value["violations"]
            for name, value in report["stages"].items()
        },
    }, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cpu_trace", type=Path)
    parser.add_argument("gpu_receipt", type=Path)
    parser.add_argument("report_json", type=Path)
    args = parser.parse_args()
    report = compare(args.cpu_trace, args.gpu_receipt, args.report_json)
    raise SystemExit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
