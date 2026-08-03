#!/usr/bin/env python3
"""Fold the instrumented ``t1d`` dumps into ``aero-exit-temperature.csv``.

The instrumented harness appends one 24-line block per scenario to
``T1D_ENTRY_AERO.csv`` and ``T1D_EXIT_AERO.csv`` in the order
``build_aero.sh`` runs them.  That order is not taken on trust: it is read out
of ``build_aero.sh``'s own scenario loop, and every block is then CHECKED
against the fixture it claims to belong to.

The check is exact and it is the whole point of the file.  The ENTRY block
must equal the scenario's ``before`` ``temp_k`` column BITWISE, because
``mp_gt_driver:1222`` computes ``t1d(k) = th(i,k,j)*pii(i,k,j)`` and
``run_column_aero.F90`` records exactly that product.  A block landing on the
wrong scenario cannot survive that comparison.  The EXIT column is then the
one quantity a pristine caller cannot obtain: the ``t1d``
``calc_refl10cm`` (:1459) and ``calc_effectRad`` (:1472) were handed.

usage:  make_exit_temperature_aero.py BUILD_DIR OUTPUT.csv
"""
from __future__ import annotations

import csv
import re
import struct
import sys
from pathlib import Path

HEADER = ("scenario,k,t1d_entry_hex,t1d_entry_k,t1d_exit_hex,t1d_exit_k,"
          "pii_hex,pii,round_trip_exit_hex,round_trip_matches")

NZ = 24


def _f32(hexed: str) -> float:
    return struct.unpack("<f", struct.pack("<I", int(hexed, 16)))[0]


def _bits(value: float) -> str:
    return f"{struct.unpack('<I', struct.pack('<f', value))[0]:08X}"


def _scenarios(build_script: Path) -> list[str]:
    text = build_script.read_text()
    match = re.search(r"^for scenario in (.*?); do$", text,
                      re.MULTILINE | re.DOTALL)
    if match is None:
        raise SystemExit(f"no scenario loop found in {build_script}")
    names = match.group(1).replace("\\\n", " ").split()
    if not names:
        raise SystemExit("empty scenario loop")
    return names


def _blocks(path: Path, count: int) -> list[list[list[str]]]:
    rows = [line.strip().split(",") for line in
            path.read_text().splitlines() if line.strip()]
    if len(rows) != count * NZ:
        raise SystemExit(
            f"{path} has {len(rows)} rows, expected {count * NZ} "
            f"({count} scenarios x {NZ} levels)")
    out = []
    for index in range(count):
        block = rows[index * NZ:(index + 1) * NZ]
        for level, row in enumerate(block, start=1):
            if int(row[0]) != level:
                raise SystemExit(f"{path}: level {row[0]} out of order")
        out.append(block)
    return out


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    build = Path(argv[1])
    output = Path(argv[2])
    here = Path(__file__).resolve().parent

    names = _scenarios(here / "build_aero.sh")
    entry = _blocks(build / "T1D_ENTRY_AERO.csv", len(names))
    exit_ = _blocks(build / "T1D_EXIT_AERO.csv", len(names))

    lines = [HEADER]
    mismatched = 0
    for name, entry_block, exit_block in zip(names, entry, exit_):
        fixture = build / "column-oracle-aero" / f"{name}-column.csv"
        with fixture.open(newline="", encoding="ascii") as stream:
            rows = list(csv.DictReader(stream))
        half = len(rows) // 2
        before, after = rows[:half], rows[half:]
        if len(before) != NZ:
            raise SystemExit(f"{fixture} has {len(before)} levels")

        for k in range(NZ):
            entry_hex, pii_hex = entry_block[k][1], entry_block[k][2]
            exit_hex = exit_block[k][1]
            if entry_hex != _bits(float(before[k]["temp_k"])):
                raise SystemExit(
                    f"{name} level {k + 1}: instrumented ENTRY t1d "
                    f"0x{entry_hex} does not match the fixture's before "
                    f"temp_k 0x{_bits(float(before[k]['temp_k']))}; the dump "
                    "is not aligned with the scenario loop")
            if pii_hex != _bits(float(before[k]["pii"])):
                raise SystemExit(f"{name} level {k + 1}: pii mismatch")
            round_trip = _bits(float(after[k]["temp_k"]))
            matches = "1" if round_trip == exit_hex else "0"
            mismatched += round_trip != exit_hex
            lines.append(
                f"{name},{k + 1},{entry_hex},{_f32(entry_hex):24.16E},"
                f"{exit_hex},{_f32(exit_hex):24.16E},{pii_hex},"
                f"{_f32(pii_hex):24.16E},{round_trip},{matches}")

    output.write_text("\n".join(lines) + "\n")
    total = len(names) * NZ
    print(f"{output}: {total} rows over {len(names)} scenarios")
    print(f"ENTRY t1d == fixture before temp_k: {total}/{total} bitwise")
    print(f"EXIT  t1d != fixture after  temp_k: {mismatched}/{total} rows "
          f"({100.0 * mismatched / total:.1f}%) -- the float32 round trip "
          "through mp_gt_driver:1358 that this receipt exists to replace")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
