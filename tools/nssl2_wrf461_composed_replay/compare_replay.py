"""Compare official-WRF and GPUWM composed-column replay outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


MASS_FIELDS = {"qv", "qc", "qr", "qi", "qs", "qg", "qh"}
NUMBER_FIELDS = {"qndrop", "qnr", "qni", "qns", "qng", "qnh", "qnn"}
VOLUME_FIELDS = {"qvolg", "qvolh"}
RADIUS_FIELDS = {"effc_m", "effi_m", "effs_m"}
PRECIPITATION_FIELDS = {
    "rainnc", "rainncv", "snownc", "snowncv", "graupelnc",
    "graupelncv", "hailnc", "hailncv", "sr",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"empty replay output: {path}")
    return rows


def _tolerance(name: str) -> tuple[float, float]:
    if name == "theta":
        return 3.0e-6, 5.0e-5
    if name in MASS_FIELDS:
        return 3.0e-5, 3.0e-9
    if name in NUMBER_FIELDS:
        # The old 256 #/kg ice-number allowance hid an incorrect NSSL GS
        # vertical-velocity centering path.  The matched real columns now
        # differ by at most 2.45e-4 #/kg, so keep a small absolute floor for
        # near-zero levels without masking process-scale errors.
        absolute = 5.0e-2 if name == "qni" else 64.0
        return 3.0e-5, absolute
    if name in VOLUME_FIELDS:
        return 3.0e-5, 3.0e-12
    if name == "refl_10cm":
        return 1.0e-4, 5.0e-2
    if name in RADIUS_FIELDS:
        return 3.0e-5, 3.0e-9
    if name in PRECIPITATION_FIELDS:
        return 5.0e-5, 1.0e-6
    raise KeyError(name)


def compare(
    wrf_path: Path, gpu_path: Path, report_path: Path
) -> dict[str, object]:
    wrf_rows = _rows(wrf_path)
    gpu_rows = _rows(gpu_path)
    if len(wrf_rows) != len(gpu_rows):
        raise ValueError(
            f"row-count mismatch: WRF={len(wrf_rows)} GPU={len(gpu_rows)}")
    if [row["k"] for row in wrf_rows] != [row["k"] for row in gpu_rows]:
        raise ValueError("vertical-level inventory mismatch")
    fields = [
        name for name in wrf_rows[0]
        if name not in {"engine", "k"}
    ]
    if fields != [
            name for name in gpu_rows[0] if name not in {"engine", "k"}]:
        raise ValueError("WRF/GPU replay schemas differ")

    results = {}
    total_violations = 0
    nonfinite = 0
    for name in fields:
        expected = np.asarray(
            [float(row[name]) for row in wrf_rows], dtype=np.float64)
        actual = np.asarray(
            [float(row[name]) for row in gpu_rows], dtype=np.float64)
        finite = np.isfinite(expected) & np.isfinite(actual)
        nonfinite_count = int(np.count_nonzero(~finite))
        nonfinite += nonfinite_count
        absolute = np.abs(actual - expected)
        rtol, atol = _tolerance(name)
        limit = atol + rtol * np.abs(expected)
        violations = (~finite) | (absolute > limit)
        violation_count = int(np.count_nonzero(violations))
        total_violations += violation_count
        worst = int(np.nanargmax(np.where(finite, absolute / limit, np.inf)))
        results[name] = {
            "rtol": rtol,
            "atol": atol,
            "violations": violation_count,
            "nonfinite": nonfinite_count,
            "max_abs_error": float(np.nanmax(absolute)),
            "max_error_over_limit": float(absolute[worst] / limit[worst]),
            "worst_k": int(wrf_rows[worst]["k"]),
            "wrf_at_worst": float(expected[worst]),
            "gpu_at_worst": float(actual[worst]),
            "bias_gpu_minus_wrf": float(np.nanmean(actual - expected)),
        }

    report = {
        "schema": "gpuwm.nssl2.composed-replay-comparison/v1",
        "status": "PASS" if total_violations == 0 and nonfinite == 0 else "FAIL",
        "wrf_csv": str(wrf_path.resolve()),
        "wrf_sha256": _sha256(wrf_path),
        "gpu_csv": str(gpu_path.resolve()),
        "gpu_sha256": _sha256(gpu_path),
        "levels": len(wrf_rows),
        "fields": len(fields),
        "total_comparisons": len(wrf_rows) * len(fields),
        "total_violations": total_violations,
        "nonfinite": nonfinite,
        "results": results,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        key: report[key]
        for key in (
            "status", "levels", "fields", "total_comparisons",
            "total_violations", "nonfinite",
        )
    }, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wrf_csv", type=Path)
    parser.add_argument("gpu_csv", type=Path)
    parser.add_argument("report_json", type=Path)
    args = parser.parse_args()
    report = compare(args.wrf_csv, args.gpu_csv, args.report_json)
    raise SystemExit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
