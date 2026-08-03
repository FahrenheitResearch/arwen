#!/usr/bin/env python3
"""Verify the aerosol per-kernel test tables against regenerated Fortran output.

WHAT THIS IS FOR
----------------
``tests/test_thompson_aerosol_warm_gpu.py`` and
``tests/test_thompson_aerosol_cold_gpu.py`` embed literal Fortran reference
tables.  Those numbers used to come from programs that existed only in an
agent scratch directory, so a reader could not re-derive them.  The programs
are now committed --

    tools/thompson_wrf461_oracle/probe_warm_rates_aero.F90
    tools/thompson_wrf461_oracle/probe_cold_warm_loop_aero.F90

-- and built and run by ``build_aero_probes.sh``.  This script closes the
loop: it reads the freshly generated CSVs and asserts that every literal in
the two test files is reproduced, at the precision the literal was printed
with.  It never edits a test and never relaxes anything; it only reports.

USAGE
-----
    python3 check_probe_oracles_aero.py PROBE_OUTPUT_DIR [REPO_ROOT]

PROBE_OUTPUT_DIR is the directory ``build_aero_probes.sh`` wrote
``aero-warm-rates.csv``, ``aero-ncten-balance.csv`` and
``aero-cold-warm-loop.csv`` into.  REPO_ROOT defaults to the repository this
script lives in.

Exit status is 0 only if every embedded row was found and matched.
"""

from __future__ import annotations

import csv
import io
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Reading the embedded tables out of the test files, keeping the literal text
# so the comparison can be made at exactly the precision that was printed.
# ---------------------------------------------------------------------------

_NUMBER = re.compile(r"[-+]?(?:\d+\.\d+(?:[eE][-+]?\d+)?|\d+)")

#: How an embedded literal encodes its float64 -- see ``_canonical``.
EXACT = "exact"
ROUNDED = "rounded"


def _extract_block(source: str, name: str, opener: str, closer: str) -> str:
    start = source.index(f"{name} = {opener}")
    start = source.index(opener, start) + len(opener)
    end = source.index(closer, start)
    return source[start:end]


def _significant(literal: str) -> int | None:
    """Significant digits in a float literal, or None for an integer one."""
    if "." not in literal:
        return None
    mantissa = literal.split("e")[0].split("E")[0].lstrip("+-")
    digits = mantissa.replace(".", "").lstrip("0")
    return len(digits) if digits else 1


def _cold_rows(test_path: Path) -> list[list[str]]:
    """One list of literal strings per row of ``_WRF_COLD_WARM_LOOP``."""
    body = _extract_block(
        test_path.read_text(), "_WRF_COLD_WARM_LOOP", "(", "\n)\n")
    rows: list[list[str]] = []
    depth = 0
    buffer: list[str] = []
    for char in body:
        if char == "(":
            depth += 1
            buffer = []
        elif char == ")":
            depth -= 1
            rows.append(_NUMBER.findall("".join(buffer)))
            buffer = []
        elif depth:
            buffer.append(char)
    widths = {len(row) for row in rows}
    if widths != {20}:
        raise SystemExit(f"unexpected _WRF_COLD_WARM_LOOP row widths {widths}")
    return rows


def _csv_string_table(test_path: Path, name: str) -> tuple[list[str],
                                                           list[list[str]]]:
    """Header and literal cells of a triple-quoted embedded CSV block."""
    body = _extract_block(test_path.read_text(), name, '"""\\\n', '\n"""')
    stream = io.StringIO(body)
    header = stream.readline().strip().split(",")
    rows = [line.strip().split(",") for line in stream if line.strip()]
    for row in rows:
        if len(row) != len(header):
            raise SystemExit(f"{name}: ragged row {row}")
    return header, rows


def _load_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open() as handle:
        reader = csv.reader(handle)
        header = next(reader)
        return header, [row for row in reader if row]


# ---------------------------------------------------------------------------
# Comparison at the printed precision.
# ---------------------------------------------------------------------------

def _same(generated: str, embedded: str, mode: str) -> bool:
    """Does the full-precision Fortran cell agree with the embedded literal?"""
    return (_canonical(generated, embedded, mode)
            == _canonical(embedded, embedded, mode))


def _canonical(value: str, embedded: str, mode: str) -> str:
    """``value`` reduced to the precision the embedded literal carries.

    Two literal styles appear in the two test files and they must be compared
    differently, so the mode is stated explicitly per table rather than
    guessed per cell.

    ``EXACT``    ``_WARM_RATE_ORACLE`` and ``_NCTEN_BALANCE_ORACLE`` print
                 Python's shortest round-tripping repr, which carries the
                 float64 exactly; anything but bit equality is a mismatch.
    ``ROUNDED``  ``_WRF_COLD_WARM_LOOP`` prints fixed ``%.10e`` (inputs) and
                 ``%.12e`` (outputs) fields, which are roundings of the
                 float64.  The generated cell is rounded to the same number
                 of significant digits and then compared.
    """
    digits = _significant(embedded)
    if digits is None:
        return str(int(float(value)))
    if mode == EXACT or float(embedded) == 0.0:
        return repr(float(value))
    return "%.*e" % (digits - 1, float(value))


def _representative(rows: list[list[str]], column: int) -> str:
    """A literal from ``column`` that shows the format actually used.

    A zero cell carries no format information (``0.0000000000e+00`` and
    ``0.0`` are indistinguishable), so pick the first non-zero one.
    """
    for row in rows:
        if float(row[column]) != 0.0:
            return row[column]
    return rows[0][column]


def _index(rows: list[list[str]], columns: list[int],
           literals: list[str], mode: str) -> dict[tuple, list[str]]:
    table: dict[tuple, list[str]] = {}
    for row in rows:
        key = tuple(
            _canonical(row[column], literal, mode)
            for column, literal in zip(columns, literals))
        table.setdefault(key, row)
    return table


# ---------------------------------------------------------------------------
# The three checks.
# ---------------------------------------------------------------------------

_COLD_COLUMNS = (
    "p_pa", "temp_k", "qv", "qc", "nc_per_kg", "qr", "nr_per_kg",
    "nwfa_per_kg", "nifa_per_kg", "nu_c_entry", "nu_c_working",
    "nc_m3", "mvd_c", "mvd_r", "prr_wau", "pnr_wau", "pnc_wau",
    "pnc_rcw", "pna_rca", "pnd_rcd")
_COLD_KEY = 9          # the first nine columns are the state


def check_cold(probe_dir: Path, repo: Path) -> list[str]:
    problems: list[str] = []
    test_path = repo / "tests" / "test_thompson_aerosol_cold_gpu.py"
    embedded = _cold_rows(test_path)
    header, rows = _load_csv(probe_dir / "aero-cold-warm-loop.csv")
    column_of = {name: header.index(name) for name in _COLD_COLUMNS}

    key_literals = [_representative(embedded, i) for i in range(_COLD_KEY)]
    index = _index(rows, [column_of[_COLD_COLUMNS[i]]
                          for i in range(_COLD_KEY)], key_literals,
                  ROUNDED)

    matched = 0
    for number, literal_row in enumerate(embedded, start=1):
        key = tuple(
            _canonical(literal_row[i], key_literals[i], ROUNDED)
            for i in range(_COLD_KEY))
        found = index.get(key)
        if found is None:
            problems.append(
                f"cold row {number}: no generated row for state {key}")
            continue
        matched += 1
        for i, name in enumerate(_COLD_COLUMNS):
            generated = found[column_of[name]]
            if not _same(generated, literal_row[i], ROUNDED):
                problems.append(
                    f"cold row {number} field {name}: "
                    f"embedded {literal_row[i]} vs generated {generated}")

    disagree = sum(
        1 for row in rows
        if row[column_of["nu_c_entry"]] != row[column_of["nu_c_working"]])
    auto = sum(
        1 for row in rows
        if row[column_of["nu_c_entry"]] != row[column_of["nu_c_working"]]
        and float(row[column_of["prr_wau"]]) > 0.0)
    print(f"cold  : {len(rows)} generated rows, "
          f"{matched}/{len(embedded)} embedded rows matched")
    print(f"cold  : {disagree} rows where nu_c(:1832) != nu_c(:2170), "
          f"{auto} of those with prr_wau > 0")
    return problems


def _check_embedded_csv(name: str, test_path: Path, csv_path: Path,
                        key_names: tuple[str, ...],
                        label: str, mode: str = EXACT) -> list[str]:
    problems: list[str] = []
    embedded_header, embedded_rows = _csv_string_table(test_path, name)
    header, rows = _load_csv(csv_path)
    missing = [n for n in embedded_header if n not in header]
    if missing:
        raise SystemExit(f"{name}: generated CSV lacks columns {missing}")

    key_columns = [header.index(n) for n in key_names]
    key_literals = [
        _representative(embedded_rows, embedded_header.index(n))
        for n in key_names]
    index = _index(rows, key_columns, key_literals, mode)

    matched = 0
    for number, literal_row in enumerate(embedded_rows, start=1):
        key = tuple(
            _canonical(literal_row[embedded_header.index(n)], literal,
                       mode)
            for n, literal in zip(key_names, key_literals))
        found = index.get(key)
        if found is None:
            problems.append(f"{label} row {number}: no generated row for {key}")
            continue
        matched += 1
        for column, value in zip(embedded_header, literal_row):
            generated = found[header.index(column)]
            if not _same(generated, value, mode):
                problems.append(
                    f"{label} row {number} field {column}: "
                    f"embedded {value} vs generated {generated}")
    print(f"{label}: {len(rows)} generated rows, "
          f"{matched}/{len(embedded_rows)} embedded rows matched")
    return problems


def check_warm(probe_dir: Path, repo: Path) -> list[str]:
    test_path = repo / "tests" / "test_thompson_aerosol_warm_gpu.py"
    problems = _check_embedded_csv(
        "_WARM_RATE_ORACLE", test_path,
        probe_dir / "aero-warm-rates.csv",
        ("pres", "temp", "qv", "qc", "nc_per_kg", "qr", "nr_per_kg",
         "nwfa_per_kg", "nifa_per_kg"),
        "warm  ")
    problems += _check_embedded_csv(
        "_NCTEN_BALANCE_ORACLE", test_path,
        probe_dir / "aero-ncten-balance.csv",
        ("qc_entry", "nc_per_kg", "rho", "ncten_in", "qc_after"),
        "ncten ")
    return problems


def main(argv: list[str]) -> int:
    if not 2 <= len(argv) <= 3:
        print(__doc__)
        return 2
    probe_dir = Path(argv[1]).resolve()
    repo = (Path(argv[2]).resolve() if len(argv) == 3
            else Path(__file__).resolve().parents[2])

    problems = check_warm(probe_dir, repo) + check_cold(probe_dir, repo)
    if problems:
        print()
        print(f"{len(problems)} MISMATCHES:")
        for line in problems[:200]:
            print("  " + line)
        if len(problems) > 200:
            print(f"  ... and {len(problems) - 200} more")
        return 1
    print()
    print("every embedded literal reproduced by the committed Fortran probes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
