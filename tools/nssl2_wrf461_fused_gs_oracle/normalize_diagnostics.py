#!/usr/bin/env python3
"""Normalize non-acceptance WRF instrumentation into named long-form rows."""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path


SCHEMA = "gpuwm.nssl2.fused-gs.diagnostics.v1"

FIELDS = {
    "BIGG_ACTIVE": [
        "temperature_c",
        "qrain_kg_kg",
        "nrain_m3",
        "rain_diameter_m",
        "bigg_diameter_m",
        "qrfrz_total_rate_s-1",
        "qrfrz_snow_rate_s-1",
        "qrfrz_dense_rate_s-1",
        "crfrz_total_rate_m-3_s-1",
        "crfrz_snow_rate_m-3_s-1",
        "crfrz_dense_rate_m-3_s-1",
        "vrfrz_dense_rate_m3_m-3_s-1",
    ],
    "FROZEN_DEP_PRE": ["combined_rate_s-1", "shared_limit_s-1"],
    "FROZEN_DEP_APPLY": ["shared_factor"],
    "FROZEN_SUB_PRE": ["combined_rate_s-1", "shared_limit_s-1"],
    "FROZEN_SUB_APPLY": ["shared_factor"],
    "RAIN_HEAT_PRE_COLD": [
        "temperature_c",
        "qrztot_rate_s-1",
        "qrzmax_pre_cold_rate_s-1",
        "rain_dt_cap_rate_s-1",
        "fwet1",
        "rain_diameter_m",
        "rain_ventilation",
        "nrain_m3",
        "qrfrz_total_rate_s-1",
        "qrfrz_snow_rate_s-1",
        "qrfrz_dense_rate_s-1",
        "crfrz_total_rate_m-3_s-1",
        "crfrz_snow_rate_m-3_s-1",
        "crfrz_dense_rate_m-3_s-1",
        "qiacr_total_rate_s-1",
        "qiacr_snow_rate_s-1",
        "qiacr_dense_rate_s-1",
        "ciacr_total_rate_m-3_s-1",
        "ciacr_snow_rate_m-3_s-1",
        "ciacr_dense_rate_m-3_s-1",
        "viacr_dense_rate_m3_m-3_s-1",
        "qsacr_rate_s-1",
    ],
    "RAIN_HEAT_FACTOR": [
        "temperature_c",
        "qrztot_rate_s-1",
        "qrzmax_final_rate_s-1",
        "shared_factor",
    ],
    "RAIN_HEAT_POST": [
        "shared_factor",
        "qrfrz_total_rate_s-1",
        "qrfrz_snow_rate_s-1",
        "qrfrz_dense_rate_s-1",
        "crfrz_total_rate_m-3_s-1",
        "crfrz_snow_rate_m-3_s-1",
        "crfrz_dense_rate_m-3_s-1",
        "vrfrz_dense_rate_m3_m-3_s-1",
        "qiacr_total_rate_s-1",
        "qiacr_snow_rate_s-1",
        "qiacr_dense_rate_s-1",
        "ciacr_total_rate_m-3_s-1",
        "ciacr_snow_rate_m-3_s-1",
        "ciacr_dense_rate_m-3_s-1",
        "viacr_dense_rate_m3_m-3_s-1",
        "qsacr_rate_s-1",
    ],
    "LIMIT": [
        "qv_mass_cap_s-1",
        "qi_mass_cap_s-1",
        "qc_mass_cap_s-1",
        "qr_mass_cap_s-1",
        "qs_mass_cap_s-1",
        "qg_mass_cap_s-1",
        "qh_mass_cap_s-1",
        "ni_number_cap_m-3_s-1",
        "nc_number_cap_m-3_s-1",
        "nr_number_cap_m-3_s-1",
        "ns_number_cap_m-3_s-1",
        "nh_number_cap_m-3_s-1",
        "frozen_dep_shared_cap_s-1",
        "frozen_sub_shared_cap_s-1",
        "ice_deposition_post_s-1",
        "snow_deposition_post_s-1",
        "graupel_deposition_post_s-1",
        "hail_deposition_post_s-1",
        "ice_sublimation_post_s-1",
        "snow_sublimation_post_s-1",
        "graupel_sublimation_post_s-1",
        "hail_sublimation_post_s-1",
        "warm_autoconversion_s-1",
        "rain_cloud_accretion_s-1",
        "ice_cloud_collection_s-1",
        "snow_cloud_collection_s-1",
        "graupel_cloud_collection_s-1",
        "hail_cloud_collection_s-1",
        "homogeneous_cloud_freezing_s-1",
        "contact_cloud_freezing_s-1",
        "hm_secondary_ice_s-1",
        "rain_ice_collection_s-1",
        "rain_snow_collection_s-1",
        "rain_graupel_collection_s-1",
        "rain_hail_collection_s-1",
        "bigg_rain_freezing_s-1",
        "rain_evaporation_s-1",
        "ice_to_snow_nucleation_s-1",
        "riming_ice_to_snow_s-1",
        "graupel_ice_collection_s-1",
        "hail_ice_collection_s-1",
        "graupel_snow_collection_s-1",
        "hail_snow_collection_s-1",
        "snow_to_graupel_s-1",
        "snow_melt_s-1",
        "graupel_melt_s-1",
        "hail_melt_s-1",
        "graupel_shedding_s-1",
        "hail_shedding_s-1",
        "graupel_to_hail_s-1",
        "graupel_wetgrowth_conversion_s-1",
        "hail_wetgrowth_conversion_s-1",
        "qv_aggregate_tendency_s-1",
        "qc_aggregate_tendency_s-1",
        "qr_aggregate_tendency_s-1",
        "qi_aggregate_tendency_s-1",
        "qs_aggregate_tendency_s-1",
        "qg_aggregate_tendency_s-1",
        "qh_aggregate_tendency_s-1",
        "nc_aggregate_tendency_m-3_s-1",
        "nr_aggregate_tendency_m-3_s-1",
        "ni_aggregate_tendency_m-3_s-1",
        "ns_aggregate_tendency_m-3_s-1",
        "ng_aggregate_tendency_m-3_s-1",
        "nh_aggregate_tendency_m-3_s-1",
    ],
}


def fail(message: str) -> None:
    raise SystemExit(message)


def load_canonical_keys(path: Path) -> tuple[set[tuple[str, str, int, int]], set[tuple[str, str, int]]]:
    row_keys: set[tuple[str, str, int, int]] = set()
    case_keys: set[tuple[str, str, int]] = set()
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = (
                row["case"],
                row["case_family"],
                int(row["repetition"]),
                int(row["k"]),
            )
            if key in row_keys:
                fail(f"duplicate canonical key: {key}")
            row_keys.add(key)
            case_keys.add(key[:3])
    return row_keys, case_keys


def normalize(raw_path: Path, canonical_path: Path, output_path: Path) -> None:
    row_keys, expected_cases = load_canonical_keys(canonical_path)
    seen_cases: set[tuple[str, str, int]] = set()
    current: tuple[str, str, int] | None = None
    record_count = 0

    with raw_path.open(newline="", encoding="utf-8") as source, output_path.open(
        "w", newline="", encoding="utf-8"
    ) as destination:
        reader = csv.reader(source)
        writer = csv.writer(destination, lineterminator="\n")
        header = next(reader, None)
        if header != ["record", "case", "case_family", "repetition", "k", "values"]:
            fail(f"unexpected raw instrumentation header: {header}")
        writer.writerow(
            [
                "schema_version",
                "case",
                "case_family",
                "repetition",
                "k",
                "record",
                "field",
                "value",
            ]
        )

        for line_number, row in enumerate(reader, start=2):
            if not row:
                continue
            if row[0] == "CASE":
                if len(row) != 4:
                    fail(f"line {line_number}: malformed CASE record")
                current = (row[1], row[2], int(row[3]))
                if current not in expected_cases:
                    fail(f"line {line_number}: CASE absent from canonical fixture: {current}")
                if current in seen_cases:
                    fail(f"line {line_number}: duplicate CASE record: {current}")
                seen_cases.add(current)
                continue
            if current is None:
                fail(f"line {line_number}: diagnostic record precedes CASE")
            record = row[0]
            fields = FIELDS.get(record)
            if fields is None:
                fail(f"line {line_number}: unknown diagnostic record {record!r}")
            if len(row) != len(fields) + 2:
                fail(
                    f"line {line_number}: {record} has {len(row) - 2} values; "
                    f"expected {len(fields)}"
                )
            level = int(row[1])
            if (*current, level) not in row_keys:
                fail(f"line {line_number}: diagnostic key absent from canonical fixture")
            for field, value in zip(fields, row[2:], strict=True):
                parsed = float(value)
                if not math.isfinite(parsed):
                    fail(f"line {line_number}: non-finite {record}.{field}={value}")
                writer.writerow([SCHEMA, *current, level, record, field, value])
            record_count += 1

    if seen_cases != expected_cases:
        missing = sorted(expected_cases - seen_cases)
        fail(f"missing CASE records: {missing}")
    if record_count == 0:
        fail("instrumentation contained no diagnostic records")


def main() -> None:
    if len(sys.argv) != 4:
        fail("usage: normalize_diagnostics.py RAW.csv CANONICAL.csv OUTPUT.csv")
    normalize(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))


if __name__ == "__main__":
    main()
