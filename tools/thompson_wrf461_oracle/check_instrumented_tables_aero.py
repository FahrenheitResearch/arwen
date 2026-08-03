#!/usr/bin/env python3
"""Verify the instrumented-WRF test tables against a fresh instrumented run.

Three test tables are taken from the MIDDLE of ``mp_thompson``, which no
entry/exit column fixture can reach:

* ``tests/test_thompson_aerosol_cold_gpu.py``  ``_WRF_COLD_REFERENCE``
  -- five 24-level fields for each of three scenarios, at the point where
  the cold network has finished (:3183).
* ``tests/test_thompson_aerosol_sed_gpu.py``   ``SED_AERO_NC_SED``
  -- WRF's cloud-fallout columns either side of :3824-3837, scenario
  ``aero-nc-sed``.
* ``tests/test_thompson_aerosol_sed_gpu.py``   ``CLEAN_CLASSIC``
  -- WRF's phase-cleanup columns either side of :3945-3967, scenario
  ``aero-reduces-to-classic``.

``build_aero_instrumented.sh`` regenerates all three.  For the cold table this
script rebuilds the five fields exactly as its derivation note describes --

    qi      = qi1d      + qiten*dt
    ni      = ni1d      + niten*dt
    ncten   = -(pnc_wau + pnc_rcw + pni_wfz + pnc_scw + pnc_gcw) * orho
    nwfaten = -(pna_rca + pna_sca + pna_gca + pni_iha)           * orho
    nifaten = -(pnd_rcd + pnd_scd + pnd_gcd + pni_inu)           * orho

with ``orho = 1./rho(k)`` in REAL(4) -- and compares at the precision the
literals were printed with.  The two sedimentation/cleanup tables are compared
BITWISE, because they measured bitwise.

USAGE
-----
    python3 check_instrumented_tables_aero.py INTERMEDIATES_DIR [REPO_ROOT]

Exit status is 0 only if every literal matched.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path


_FIELDS = ("qi", "ni", "ncten", "nwfaten", "nifaten")


def _embedded(test_path: Path) -> dict[str, dict[str, list[str]]]:
    """``_WRF_COLD_REFERENCE`` as literal strings, so precision survives."""
    source = test_path.read_text()
    start = source.index("_WRF_COLD_REFERENCE = {")
    end = source.index("\n}\n", start)
    body = source[start:end]
    table: dict[str, dict[str, list[str]]] = {}
    scenario = None
    field = None
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.endswith(": {") and stripped.startswith('"'):
            scenario = stripped.split('"')[1]
            table[scenario] = {}
        elif stripped.endswith(": (") and stripped.startswith('"'):
            field = stripped.split('"')[1]
            table[scenario][field] = []
        elif stripped.startswith(")"):
            field = None
        elif field is not None and stripped:
            for piece in stripped.rstrip(",").split(","):
                piece = piece.strip()
                if not piece:
                    continue
                try:
                    float(piece)
                except ValueError:
                    continue
                table[scenario][field].append(piece)
    return table


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def _significant(literal: str) -> int:
    mantissa = literal.split("e")[0].split("E")[0].lstrip("+-")
    digits = mantissa.replace(".", "").lstrip("0")
    return len(digits) if digits else 1


def _same(generated: float, literal: str) -> bool:
    if float(literal) == 0.0:
        return generated == 0.0
    digits = _significant(literal)
    return (float("%.*e" % (digits - 1, generated))
            == float("%.*e" % (digits - 1, float(literal))))


# ---------------------------------------------------------------------------
# The two mid-call tables in tests/test_thompson_aerosol_sed_gpu.py.  Both
# are float64 reprs of WRF REAL(4) values, so they are compared BITWISE.
# ---------------------------------------------------------------------------

#: (test constant, column-order constant, scenario, generated probe file)
_SED_TABLES = (
    ("SED_AERO_NC_SED", "SED_COLUMNS", "aero-nc-sed", "cloud-sed"),
    ("CLEAN_CLASSIC", "CLEAN_COLUMNS", "aero-reduces-to-classic",
     "phase-cleanup"),
    # The three WP-08 scratch scenarios, which used to have no committed
    # producer at all (gpuwm/data/thompson/PROVENANCE.md, "Still ungated by a
    # committed producer").  They are ordinary cases of run_column_aero.F90
    # now -- ids 120/121/122 -- so all five sed/cleanup tables regenerate
    # from one command.
    ("SED_NU_SWEEP", "SED_COLUMNS", "wp08-nusweep", "cloud-sed"),
    ("CLEAN_MELT", "CLEAN_COLUMNS", "wp08-melt", "phase-cleanup"),
    ("CLEAN_FREEZE", "CLEAN_COLUMNS", "wp08-freeze", "phase-cleanup"),
)


def _tuple_constant(source: str, name: str):
    start = source.index(f"{name} = (")
    end = source.index("\n)\n", start) + 3
    namespace: dict = {}
    exec(source[start:end], namespace)          # noqa: S102 - literal tuple
    return namespace[name]


def check_sed_tables(intermediates: Path, repo: Path) -> list[str]:
    problems: list[str] = []
    source = (repo / "tests" / "test_thompson_aerosol_sed_gpu.py").read_text()
    for table_name, column_name, scenario, probe in _SED_TABLES:
        expected = _tuple_constant(source, table_name)
        columns = _tuple_constant(source, column_name)
        path = intermediates / f"{scenario}-{probe}.csv"
        if not path.is_file():
            problems.append(f"{table_name}: {path} not generated")
            continue
        rows = [row for row in _rows(path)
                if int(row["ii"]) == 1 and int(row["jj"]) == 1]
        rows.sort(key=lambda row: int(row["k"]))
        if len(rows) != len(expected):
            problems.append(
                f"{table_name}: {len(expected)} embedded levels vs "
                f"{len(rows)} generated")
            continue
        matched = 0
        total = 0
        for level, (literal_row, row) in enumerate(zip(expected, rows)):
            for column, value in zip(columns, literal_row):
                total += 1
                generated = float(row[column])
                if repr(generated) == repr(float(value)):
                    matched += 1
                else:
                    problems.append(
                        f"{table_name} level {level} {column}: "
                        f"embedded {value!r} vs generated {generated!r}")
        print(f"{table_name:<24s} {scenario:<24s} "
              f"{matched}/{total} embedded values matched bitwise")
    return problems


def main(argv: list[str]) -> int:
    if not 2 <= len(argv) <= 3:
        print(__doc__)
        return 2
    intermediates = Path(argv[1]).resolve()
    repo = (Path(argv[2]).resolve() if len(argv) == 3
            else Path(__file__).resolve().parents[2])

    table = _embedded(repo / "tests" / "test_thompson_aerosol_cold_gpu.py")
    problems: list[str] = list(check_sed_tables(intermediates, repo))
    for scenario, expected in table.items():
        path = intermediates / f"{scenario}-cold-network.csv"
        if not path.is_file():
            problems.append(f"{scenario}: {path} not generated")
            continue
        rows = [row for row in _rows(path)
                if int(row["ii"]) == 1 and int(row["jj"]) == 1]
        rows.sort(key=lambda row: int(row["k"]))
        dt = float(rows[0]["dt"])
        got = {
            "qi": [float(r["qi_before"]) + float(r["qiten"]) * dt
                   for r in rows],
            "ni": [float(r["ni_before"]) + float(r["niten"]) * dt
                   for r in rows],
            "ncten": [float(r["ncten_cold"]) for r in rows],
            "nwfaten": [float(r["nwfaten_cold"]) for r in rows],
            "nifaten": [float(r["nifaten_cold"]) for r in rows],
        }
        matched = 0
        total = 0
        for field in _FIELDS:
            literals = expected[field]
            if len(literals) != len(got[field]):
                problems.append(
                    f"{scenario} {field}: {len(literals)} embedded levels vs "
                    f"{len(got[field])} generated")
                continue
            for level, (value, literal) in enumerate(
                    zip(got[field], literals)):
                total += 1
                if _same(value, literal):
                    matched += 1
                else:
                    problems.append(
                        f"{scenario} {field} level {level}: "
                        f"embedded {literal} vs generated {value!r}")
        print(f"{scenario:<24s} dt = {dt:>4.1f} s  "
              f"{matched}/{total} embedded values matched")

    if problems:
        print()
        print(f"{len(problems)} MISMATCHES:")
        for line in problems[:120]:
            print("  " + line)
        if len(problems) > 120:
            print(f"  ... and {len(problems) - 120} more")
        return 1
    print()
    print("every embedded literal reproduced by the instrumented WRF build")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
