#!/usr/bin/env python3
"""Strict structural and numerical validation for the fused-GS oracle data."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path


CANONICAL_HEADER = [
    "schema_version",
    "case",
    "case_family",
    "repetition",
    "k",
    "dt_s",
    "dz_m",
    "rho_kg_m3",
    "pressure_pa",
    "exner",
    "w_lower_m_s",
    "w_upper_m_s",
    "w_center_m_s",
    "primary_ice_target_m3",
    "temperature_before_k",
    "theta_before_k",
    "qv_before",
    "qc_before",
    "qr_before",
    "qi_before",
    "qs_before",
    "qg_before",
    "qh_before",
    "qndrop_before",
    "qnr_before",
    "qni_before",
    "qns_before",
    "qng_before",
    "qnh_before",
    "qnn_before",
    "qvolg_before",
    "qvolh_before",
    "temperature_after_k",
    "theta_after_k",
    "qv_after",
    "qc_after",
    "qr_after",
    "qi_after",
    "qs_after",
    "qg_after",
    "qh_after",
    "qndrop_after",
    "qnr_after",
    "qni_after",
    "qns_after",
    "qng_after",
    "qnh_after",
    "qnn_after",
    "qvolg_after",
    "qvolh_after",
]

REQUIRED_CASES = {
    "zero_clear_dt0p1",
    "threshold_cleanup_dt1",
    "all_active_dt0p1",
    "all_active_dt1",
    "all_active_dt10",
    "all_active_dt60",
    "cloud_donor_compete_dt60",
    "rain_donor_compete_dt60",
    "ice_donor_compete_dt60",
    "snow_donor_compete_dt60",
    "frozen_vapor_signed_dt60",
    "frozen_vapor_limiter_dt300",
    "wetgrowth_shedding_hm_dt10",
    "wetgrowth_g2h_melt_dt60",
    "moment_bounds_dt0p1",
    "moment_bounds_dt60",
    "gate_243p15",
    "gate_265p15",
    "gate_268p15",
    "gate_270p15",
    "gate_271p15",
    "gate_273p15",
    "bigg_strict_temp_diameter",
    "bigg_default_snow_split",
    "rain_heat_cold_override",
    "rain_freezing_transfer_thresholds",
    "rain_freezing_moment_donors",
    "rain_heat_fwet_span",
    "combined_bigg_qiacr",
    "rain_freezing_long_dt_caps",
}

DIAGNOSTIC_HEADER = [
    "schema_version",
    "case",
    "case_family",
    "repetition",
    "k",
    "record",
    "field",
    "value",
]

REQUIRED_RECORDS = {
    "BIGG_ACTIVE",
    "FROZEN_DEP_PRE",
    "FROZEN_DEP_APPLY",
    "FROZEN_SUB_PRE",
    "FROZEN_SUB_APPLY",
    "RAIN_HEAT_PRE_COLD",
    "RAIN_HEAT_FACTOR",
    "RAIN_HEAT_POST",
    "LIMIT",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def validate(canonical_path: Path, diagnostics_path: Path) -> dict[str, object]:
    with canonical_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == CANONICAL_HEADER, "canonical 50-column schema mismatch")
        rows = list(reader)

    require(rows, "canonical fixture is empty")
    numeric_columns = CANONICAL_HEADER[4:]
    for line_number, row in enumerate(rows, start=2):
        require(row["schema_version"] == "gpuwm.nssl2.fused-gs.v1", f"line {line_number}: schema")
        for column in numeric_columns:
            try:
                value = float(row[column])
            except ValueError as error:
                raise SystemExit(f"line {line_number}: invalid {column}") from error
            require(math.isfinite(value), f"line {line_number}: non-finite {column}")

    cases = {row["case"] for row in rows}
    require(REQUIRED_CASES <= cases, f"missing required cases: {sorted(REQUIRED_CASES - cases)}")
    require(len(cases) == 30, f"expected 30 cases, found {len(cases)}")
    require(len(rows) == 30 * 2 * 4, f"expected 240 rows, found {len(rows)}")

    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["case"], int(row["repetition"]))].append(row)
    for case in cases:
        for repetition in (0, 1):
            levels = sorted(int(row["k"]) for row in grouped[(case, repetition)])
            require(levels == [1, 2, 3, 4], f"{case} repetition {repetition}: levels {levels}")
        left = sorted(grouped[(case, 0)], key=lambda row: int(row["k"]))
        right = sorted(grouped[(case, 1)], key=lambda row: int(row["k"]))
        for row_a, row_b in zip(left, right, strict=True):
            comparable_a = {key: value for key, value in row_a.items() if key != "repetition"}
            comparable_b = {key: value for key, value in row_b.items() if key != "repetition"}
            require(comparable_a == comparable_b, f"non-deterministic repetition: {case} k={row_a['k']}")

    dt_values = {round(float(row["dt_s"]), 6) for row in rows}
    require({0.1, 1.0, 10.0, 60.0, 300.0} <= dt_values, f"missing dt coverage: {dt_values}")
    require(any(float(row["primary_ice_target_m3"]) > 0.0 for row in rows), "no primary-ice target")
    require(any(float(row["qvolg_before"]) > 0.0 for row in rows), "no graupel volume")
    require(any(float(row["qvolh_before"]) > 0.0 for row in rows), "no hail volume")

    with diagnostics_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == DIAGNOSTIC_HEADER, "diagnostic schema mismatch")
        diagnostic_rows = list(reader)
    require(diagnostic_rows, "diagnostic fixture is empty")
    diagnostic_counts = Counter(row["record"] for row in diagnostic_rows)
    require(REQUIRED_RECORDS <= diagnostic_counts.keys(), "missing required diagnostic records")
    for line_number, row in enumerate(diagnostic_rows, start=2):
        require(
            row["schema_version"] == "gpuwm.nssl2.fused-gs.diagnostics.v1",
            f"diagnostic line {line_number}: schema",
        )
        value = float(row["value"])
        require(math.isfinite(value), f"diagnostic line {line_number}: non-finite value")

    factor_values = [
        float(row["value"])
        for row in diagnostic_rows
        if row["record"] == "RAIN_HEAT_FACTOR" and row["field"] == "shared_factor"
    ]
    require(any(value < 1.0 for value in factor_values), "rain heat limiter never active")
    require(any(value == 1.0 for value in factor_values), "rain heat limiter never inactive")

    bigg_diameters = [
        float(row["value"])
        for row in diagnostic_rows
        if row["record"] == "BIGG_ACTIVE" and row["field"] == "bigg_diameter_m"
    ]
    require(any(value < 0.0003 for value in bigg_diameters), "no sub-0.3-mm Bigg diameter")
    require(any(value > 0.008 for value in bigg_diameters), "no over-8-mm Bigg diameter")

    return {
        "schema": "gpuwm.nssl2.fused-gs.validation.v1",
        "canonical_sha256": sha256(canonical_path),
        "diagnostics_sha256": sha256(diagnostics_path),
        "rows": len(rows),
        "cases": len(cases),
        "case_families": len({row["case_family"] for row in rows}),
        "dt_s": sorted(dt_values),
        "diagnostic_rows": len(diagnostic_rows),
        "diagnostic_records": dict(sorted(diagnostic_counts.items())),
        "rain_heat_factor_min": min(factor_values),
        "rain_heat_factor_max": max(factor_values),
        "bigg_diameter_min_m": min(bigg_diameters),
        "bigg_diameter_max_m": max(bigg_diameters),
    }


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: validate_oracle.py CANONICAL.csv DIAGNOSTICS.csv")
    print(json.dumps(validate(Path(sys.argv[1]), Path(sys.argv[2])), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
