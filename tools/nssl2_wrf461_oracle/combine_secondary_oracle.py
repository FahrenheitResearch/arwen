#!/usr/bin/env python3
"""Isolate the process-28 WRF tendency with a paired full/baseline run."""

from __future__ import annotations

import csv
import sys
from pathlib import Path


MUTABLE_PAIRS = (
    ("theta_before_k", "theta_after_k"),
    ("qv_before", "qv_after"),
    ("qc_before", "qc_after"),
    ("qnc_before_per_kg", "qnc_after_per_kg"),
    ("qr_before", "qr_after"),
    ("qnr_before_per_kg", "qnr_after_per_kg"),
    ("qi_before", "qi_after"),
    ("qni_before_per_kg", "qni_after_per_kg"),
    ("qs_before", "qs_after"),
    ("qns_before_per_kg", "qns_after_per_kg"),
    ("qg_before", "qg_after"),
    ("qng_before_per_kg", "qng_after_per_kg"),
    ("qvolg_before_m3_per_kg", "qvolg_after_m3_per_kg"),
    ("qh_before", "qh_after"),
    ("qnh_before_per_kg", "qnh_after_per_kg"),
    ("qvolh_before_m3_per_kg", "qvolh_after_m3_per_kg"),
)


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="ascii") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise RuntimeError(f"missing header: {path}")
        return reader.fieldnames, list(reader)


def main() -> None:
    if len(sys.argv) != 11:
        raise SystemExit(
            "usage: combine_secondary_oracle.py mapped|all BASELINE OUTPUT "
            "CONTACT HOMOGENEOUS HM ICE_TO_G SNOW_TO_G G_TO_H ALL")
    selection = sys.argv[1]
    if selection not in {"mapped", "all"}:
        raise SystemExit("first argument must be mapped or all")
    baseline_header, baseline_rows = read_csv(Path(sys.argv[2]))
    mode_names = ("contact", "homogeneous", "hm", "ice_to_g",
                  "snow_to_g", "g_to_h", "all")
    mode_rows: dict[str, list[dict[str, str]]] = {}
    for mode, path in zip(mode_names, sys.argv[4:], strict=True):
        header, rows = read_csv(Path(path))
        if header != baseline_header:
            raise RuntimeError(f"{mode} and baseline headers differ")
        if len(rows) != len(baseline_rows):
            raise RuntimeError(f"{mode} and baseline row counts differ")
        mode_rows[mode] = rows

    def mode_for_case(case: int) -> str:
        if selection == "all":
            return "all"
        if case <= 5:
            return "contact"
        if case <= 12:
            return "homogeneous"
        if case <= 21:
            return "hm"
        if case <= 25:
            return "ice_to_g"
        if case <= 28:
            return "snow_to_g"
        if case <= 34:
            return "g_to_h"
        return "all"

    output_rows: list[dict[str, str]] = []
    after_fields = {after for _, after in MUTABLE_PAIRS}
    for index, baseline in enumerate(baseline_rows):
        full = mode_rows[mode_for_case(int(baseline["case"]))][index]
        for field in baseline_header:
            if field not in after_fields and full[field] != baseline[field]:
                raise RuntimeError(f"input mismatch at row {index}, field {field}")
        isolated = dict(full)
        for before_field, after_field in MUTABLE_PAIRS:
            before = float(full[before_field])
            delta = float(full[after_field]) - float(baseline[after_field])
            isolated[after_field] = f"{before + delta:.16e}"
        output_rows.append(isolated)

    with Path(sys.argv[3]).open("w", newline="", encoding="ascii") as stream:
        writer = csv.DictWriter(stream, fieldnames=baseline_header, lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)


if __name__ == "__main__":
    main()
