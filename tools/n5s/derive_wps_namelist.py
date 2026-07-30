#!/usr/bin/env python3
"""Copy the bundle WPS namelist while changing only forcing-time controls."""

from __future__ import annotations

import argparse
from pathlib import Path
import re


def derive(source: str | Path) -> str:
    text = Path(source).read_text(encoding="utf-8")
    replacements = {
        "start_date": " '1974-04-03_12:00:00', '1974-04-03_12:00:00', '1974-04-03_12:00:00', '1974-04-03_12:00:00',",
        "end_date": " '1974-04-03_18:00:00', '1974-04-03_12:00:00', '1974-04-03_12:00:00', '1974-04-03_12:00:00',",
        "interval_seconds": " 21600,",
    }
    for key, value in replacements.items():
        pattern = re.compile(rf"(?m)^(\s*{key}\s*=).*$")
        text, count = pattern.subn(rf"\1{value}", text, count=1)
        if count != 1:
            raise ValueError(f"bundle namelist.wps has no unique {key}")

    # Keep met_em filenames portable and consistent with the bundle and the
    # downstream prep/run checks (1974-04-03_12_00_00, without colons).
    nocolons_line = " nocolons = .true.,"
    nocolons_pattern = re.compile(r"(?m)^\s*nocolons\s*=.*$")
    text, count = nocolons_pattern.subn(nocolons_line, text, count=1)
    if count == 0:
        section_pattern = re.compile(r"(?im)^(\s*&share\b[^\r\n]*)(\r?\n)")
        text, count = section_pattern.subn(
            rf"\1\2{nocolons_line}\2", text, count=1)
        if count != 1:
            raise ValueError("bundle namelist.wps has no unique &share section")

    # metgrid otherwise ignores the work-directory symlink and looks below
    # metgrid/METGRID.TBL, which fails with "Could not open file METGRID.TBL".
    table_line = " opt_metgrid_tbl_path = './',"
    table_pattern = re.compile(r"(?m)^\s*opt_metgrid_tbl_path\s*=.*$")
    text, count = table_pattern.subn(table_line, text, count=1)
    if count == 0:
        section_pattern = re.compile(r"(?im)^(\s*&metgrid\b[^\r\n]*)(\r?\n)")
        text, count = section_pattern.subn(
            rf"\1\2{table_line}\2", text, count=1)
        if count != 1:
            raise ValueError("bundle namelist.wps has no unique &metgrid section")
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(derive(args.source), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
