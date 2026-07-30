"""Assert the gpuwm CPU reference against the widened WRF MYNN oracle.

Run under ``PYTHONPATH=<repo root>`` so ``gpuwm.core.mynn_surface`` imports.
Reports max_abs / max_rel / max_ulp per output column and fails closed on a
missing branch or a parity regression.

The gate is the measured FP32 ULP residue, column by column, not a relative
tolerance.  It used to be ``np.allclose(rtol=3e-6, atol=1e-8)``: this script
already computed ``max_ulp``, printed it, and then threw it away.  rtol=3e-6
admits ~25 FP32 ULP where the reference carries at most 10, and the atol arm
admitted a *one percent* error on QFX, whose columns run to
-1e-6 kg m-2 s-1.  A gate that loose cannot fail on a regression, which is
the only thing an oracle gate is for.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
import sys

import numpy as np

from gpuwm.core.fp32_ulp import fp32_ulp_distance
from gpuwm.core.mynn_surface import (
    mynn_sfclay_first_step_state,
    mynn_surface_layer_default,
)


INPUT_ALIASES = {
    "hfx": "hfx_input",
    "qfx": "qfx_input",
    "znt": "znt_input",
    "qsfc": "qsfc_input",
    "ust": "ust_input",
}
INPUT_NAMES = (
    "u1", "v1", "t1", "qv1", "p1", "rho1", "dz1",
    "u2", "v2", "dz2", "psfc", "tsk", "pblh", "mavail",
    "hfx", "qfx", "znt", "qsfc", "ust", "xland", "snowh",
)
COLUMN_OUTPUTS = (
    "regime", "zol", "rmol", "ust", "ustm", "mol", "psim", "psih",
    "chs", "chs2", "cqs2", "ch", "flhc", "flqc", "qgh", "qsfc",
    "hfx", "qfx", "lh", "u10", "v10", "th2", "t2", "q2",
    "gz1oz0", "wspd", "br", "ck", "cka", "cd", "cda", "wstar",
    "qstar", "cpm", "znt",
)
# SFCLAY_mynn keeps wstar/qstar as wrapper locals, so they are not reported.
WRAPPER_OUTPUTS = tuple(
    name for name in COLUMN_OUTPUTS if name not in ("wstar", "qstar")
)

EXPECTED_CASES = (
    "strong_stable_land",
    "clipped_stable_land",
    "damped_stable_land",
    "neutral_land",
    "free_convective_land",
    "land_qsfc_unset",
    "thin_land_level2_wind",
    "thin_land_log10_wind",
    "midres_water",
    "coarse_water",
)

#: module_sf_mynn.F:1027-1044.  ISFFLX<1 assigns these thirteen outputs the
#: literal constant 0 and leaves the other twenty-two on the ISFFLX=1 code
#: path, so the ISFFLX=0 stage reuses ``WIDE_ULP[(1, 1)]`` for the twenty-two
#: and 0 for these thirteen.  Both halves are checked below.
ISFFLX0_ZEROED = (
    "hfx", "qfx", "flhc", "flqc", "lh", "chs", "ch", "chs2", "cqs2",
    "ck", "cd", "cka", "cda",
)

#: The FP32 ULP distance of the CPU reference from this oracle, measured per
#: (stage, output, column) -- one integer per compared number, in
#: ``EXPECTED_CASES`` order.  An output a stage does not name is bitwise on
#: every column there and must stay bitwise: the lookup returns zeros.  There
#: is no margin to justify, because there is no margin: each entry is the
#: residue measured at that element and the gate is ``residue <= entry``.
#:
#: The numbers are the elementwise maximum over THREE NumPy builds -- Windows
#: NumPy 2.2.6 / CPython 3.13.7, WSL Ubuntu-24.04 NumPy 2.4.3 / CPython 3.12.3,
#: and Ubuntu-22.04 NumPy 2.5.1 / CPython 3.12.13 -- because NumPy's float32
#: ``arctan`` and ``**`` are not the same function on all three, and this
#: script is normally run under a Linux build while pytest runs under the
#: Windows one.  They disagree at 300 of 5150 CPU elements.  The derivation,
#: including which of the two primitives moves which column and why NumPy
#: 2.5.1 is the build closest to glibc, is in ``tests/test_mynn_surface.py``.
#:
#: These mirror ``tests/test_mynn_surface.py``, which asserts them equal --
#: two copies of one measurement drift otherwise.  A ratchet to lower, never
#: to raise.
WIDE_ULP = {
    (1, 1): {
        "zol": (1, 0, 0, 0, 1, 1, 0, 0, 0, 0),
        "rmol": (0, 0, 0, 0, 1, 1, 0, 0, 0, 0),
        "ust": (2, 0, 0, 0, 0, 1, 0, 0, 0, 0),
        "ustm": (2, 0, 0, 0, 0, 1, 0, 0, 0, 0),
        "psim": (1, 0, 0, 0, 1, 3, 0, 0, 1, 0),
        "psih": (1, 0, 0, 0, 0, 1, 2, 0, 1, 0),
        "chs": (1, 0, 0, 0, 0, 2, 0, 0, 0, 0),
        "chs2": (1, 0, 0, 0, 0, 1, 0, 0, 0, 0),
        "cqs2": (1, 0, 0, 0, 0, 1, 0, 0, 0, 0),
        "ch": (2, 0, 0, 0, 0, 1, 0, 0, 0, 0),
        "flhc": (2, 0, 0, 0, 0, 1, 0, 0, 0, 0),
        "flqc": (1, 0, 0, 0, 0, 2, 0, 0, 0, 0),
        "qgh": (0, 0, 0, 0, 0, 1, 0, 1, 3, 0),
        "qsfc": (0, 0, 0, 0, 0, 1, 0, 0, 1, 0),
        "hfx": (2, 0, 0, 0, 0, 1, 0, 0, 0, 0),
        "qfx": (1, 0, 0, 0, 0, 2, 0, 0, 10, 0),
        "lh": (1, 0, 0, 0, 0, 3, 0, 0, 6, 0),
        "u10": (1, 0, 1, 0, 2, 1, 0, 0, 0, 0),
        "v10": (1, 0, 1, 0, 2, 2, 0, 0, 0, 0),
        # land_qsfc_unset: 0 on NumPy 2.2.6/2.4.3, 1 on 2.5.1.        +2.5.1
        "ck": (0, 0, 1, 0, 2, 1, 0, 0, 0, 0),
        "cka": (1, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        "cd": (0, 0, 2, 0, 3, 0, 0, 0, 0, 0),
        "cda": (3, 0, 0, 0, 0, 2, 0, 0, 0, 0),
        "qstar": (0, 0, 0, 0, 0, 0, 0, 0, 5, 0),
    },
    (2, 1): {
        "zol": (0, 0, 2, 0, 1, 2, 0, 0, 0, 0),
        "rmol": (0, 0, 2, 0, 1, 3, 0, 0, 0, 0),
        "ust": (0, 0, 0, 0, 0, 1, 0, 0, 0, 0),
        "ustm": (0, 0, 0, 0, 0, 1, 0, 0, 0, 0),
        "mol": (0, 0, 2, 0, 0, 1, 0, 0, 0, 0),
        "psim": (0, 0, 2, 0, 1, 2, 0, 0, 1, 1),
        "psih": (0, 0, 2, 0, 0, 2, 0, 0, 1, 0),
        "chs": (0, 0, 1, 0, 0, 2, 0, 0, 0, 0),
        "chs2": (0, 0, 1, 0, 0, 1, 1, 0, 0, 0),
        "cqs2": (0, 0, 0, 0, 0, 1, 1, 0, 0, 0),
        "ch": (0, 0, 2, 0, 0, 2, 0, 0, 0, 0),
        "flhc": (0, 0, 2, 0, 0, 1, 0, 0, 0, 0),
        "flqc": (0, 0, 0, 0, 0, 1, 0, 0, 0, 0),
        "qgh": (0, 0, 0, 0, 0, 1, 0, 1, 3, 0),
        "qsfc": (0, 0, 0, 0, 0, 0, 0, 0, 1, 0),
        "hfx": (0, 0, 3, 0, 0, 1, 0, 0, 0, 0),
        "qfx": (0, 0, 0, 0, 0, 2, 0, 0, 8, 0),
        "lh": (0, 0, 0, 0, 0, 3, 0, 0, 10, 0),
        "u10": (0, 0, 0, 0, 1, 1, 0, 0, 2, 0),
        "v10": (0, 0, 0, 0, 1, 1, 0, 0, 2, 0),
        "ck": (0, 0, 0, 0, 2, 2, 0, 0, 0, 0),
        "cka": (0, 0, 0, 0, 1, 1, 0, 0, 0, 0),
        "cd": (0, 0, 0, 0, 2, 0, 0, 0, 0, 0),
        "cda": (0, 0, 0, 0, 1, 1, 0, 0, 0, 0),
        "qstar": (0, 0, 0, 0, 0, 1, 0, 0, 5, 0),
    },
}

WRAPPER_ULP = {
    1: {
        "zol": (5, 0, 1, 0, 1, 1, 0, 2, 0, 0),
        "rmol": (4, 0, 1, 0, 1, 0, 0, 2, 0, 0),
        "ust": (1, 0, 0, 0, 0, 1, 0, 0, 0, 0),
        "ustm": (1, 0, 0, 0, 0, 1, 0, 0, 0, 0),
        "mol": (2, 1, 0, 0, 0, 0, 1, 1, 0, 0),
        "psim": (2, 0, 1, 0, 2, 2, 1, 3, 0, 0),
        "psih": (3, 0, 1, 0, 2, 1, 1, 2, 0, 0),
        "chs": (4, 1, 0, 0, 0, 1, 1, 1, 0, 0),
        "chs2": (2, 0, 0, 0, 1, 1, 1, 1, 0, 0),
        "cqs2": (2, 0, 0, 0, 0, 1, 0, 2, 0, 0),
        "ch": (3, 1, 0, 0, 0, 1, 1, 1, 0, 0),
        "flhc": (2, 1, 0, 0, 0, 2, 1, 1, 0, 0),
        "flqc": (1, 0, 0, 0, 0, 3, 1, 1, 0, 0),
        "qgh": (0, 0, 0, 0, 0, 1, 0, 1, 3, 0),
        "qsfc": (0, 0, 0, 0, 0, 0, 0, 0, 1, 0),
        "hfx": (2, 1, 0, 0, 0, 2, 1, 1, 0, 0),
        "qfx": (1, 0, 0, 0, 0, 0, 1, 1, 7, 0),
        "lh": (1, 0, 0, 0, 0, 0, 1, 1, 8, 0),
        "u10": (1, 0, 0, 0, 0, 1, 0, 0, 0, 0),
        "v10": (0, 0, 0, 0, 0, 2, 0, 0, 1, 0),
        "ck": (1, 0, 0, 0, 0, 0, 0, 2, 0, 0),
        "cka": (4, 0, 0, 0, 0, 1, 2, 1, 0, 0),
        "cd": (2, 0, 0, 0, 0, 0, 0, 2, 0, 0),
        "cda": (5, 0, 0, 0, 0, 3, 2, 0, 0, 0),
    },
    2: {
        "zol": (0, 0, 0, 0, 0, 0, 0, 1, 0, 0),
        "rmol": (0, 0, 0, 0, 0, 0, 0, 1, 0, 0),
        "ust": (0, 0, 0, 0, 0, 0, 1, 1, 0, 0),
        "ustm": (0, 0, 0, 0, 0, 0, 1, 1, 0, 0),
        "psim": (0, 0, 0, 0, 1, 2, 1, 3, 0, 0),
        "psih": (1, 0, 0, 0, 0, 0, 0, 2, 0, 0),
        "chs": (0, 0, 0, 1, 0, 0, 1, 2, 0, 0),
        "chs2": (0, 0, 3, 0, 0, 0, 1, 1, 0, 0),
        "cqs2": (0, 0, 0, 0, 0, 0, 1, 1, 0, 0),
        "ch": (0, 0, 0, 1, 0, 0, 1, 4, 0, 0),
        "flhc": (0, 0, 0, 1, 0, 0, 1, 2, 0, 0),
        "flqc": (1, 0, 0, 0, 0, 0, 1, 0, 0, 0),
        "qgh": (0, 0, 0, 0, 0, 1, 0, 1, 3, 0),
        "qsfc": (0, 0, 0, 0, 0, 0, 0, 0, 1, 0),
        "hfx": (0, 0, 0, 0, 0, 0, 1, 3, 0, 0),
        "qfx": (1, 0, 0, 0, 0, 0, 1, 0, 6, 0),
        "lh": (1, 0, 0, 0, 0, 0, 1, 0, 8, 0),
        "u10": (0, 0, 0, 0, 1, 0, 0, 0, 0, 0),
        "v10": (0, 0, 0, 0, 1, 0, 0, 0, 1, 2),
        "ck": (2, 0, 0, 0, 0, 0, 0, 0, 2, 1),
        "cka": (1, 0, 0, 0, 0, 0, 0, 3, 0, 0),
        "cd": (0, 0, 0, 0, 2, 0, 0, 0, 2, 2),
        "cda": (0, 0, 0, 0, 0, 0, 0, 2, 0, 0),
    },
}


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="ascii") as stream:
        return list(csv.DictReader(stream))


def _stage_key(row: dict[str, str]) -> tuple[int, int]:
    return int(row["itimestep"]), int(row.get("isfflx", 1))


def _fields(rows: list[dict[str, str]]) -> dict[str, np.ndarray]:
    return {
        key: np.asarray([float(row[key]) for row in rows], dtype=np.float32)
        for key in rows[0]
        if key not in ("case", "itimestep", "isfflx")
    }


def _budget(table, name, zeroed=()) -> np.ndarray:
    row = () if name in zeroed else table.get(name, ())
    if not row:
        return np.zeros(len(EXPECTED_CASES), dtype=np.int64)
    if len(row) != len(EXPECTED_CASES):
        raise ValueError(f"{name}: table row is {len(row)} wide")
    return np.asarray(row, dtype=np.int64)


def _compare(label, names, actual, fields, table, report, failures,
             zeroed=()) -> None:
    for name in names:
        got = np.asarray(actual[name], dtype=np.float32)
        ref = fields[name]
        abs_err = np.abs(got - ref)
        denom = np.abs(ref)
        rel = np.where(denom > 0.0, abs_err / np.where(denom > 0.0, denom, 1.0),
                       0.0)
        residue = fp32_ulp_distance(got, ref)
        budget = _budget(table, name, zeroed)
        report.append((label, name, float(abs_err.max()), float(rel.max()),
                       int(residue.max()), int(budget.max())))
        if name == "regime":
            if not np.array_equal(got, ref):
                failures.append(f"{label}/regime is {got}, expected {ref}")
            continue
        for index in np.flatnonzero(residue > budget):
            failures.append(
                f"{label}/{name}[{EXPECTED_CASES[index]}]: "
                f"{int(residue[index])} ULP from the unmodified WRF oracle "
                f"exceeds the measured {int(budget[index])} "
                f"({got[index]!r} vs {ref[index]!r})"
            )


def _assert_branch_coverage(stages, failures) -> None:
    step1 = stages[(1, 1)]
    step2 = stages[(2, 1)]
    regimes = set(step1["regime"].tolist()) | set(step2["regime"].tolist())
    for wanted in (1.0, 2.0, 3.0, 4.0):
        if wanted not in regimes:
            failures.append(f"widened oracle never produces REGIME={wanted}")
    if step1["br"].max() <= 0.2:
        failures.append("widened oracle never exceeds BR=0.2")
    if not np.any(step2["mol_input"] != 0.0):
        failures.append("itimestep=2 rows carry no nonzero incoming MOL")
    if not np.any(step1["qsfc_input"][step1["xland"] < 1.5] <= 0.0):
        failures.append("no land column enters with QSFC<=0")
    za1 = 0.5 * step1["dz1"]
    za2 = step1["dz1"] + 0.5 * step1["dz2"]
    if not np.any((za1 <= 7.0) & (za2 > 7.0) & (za2 < 13.0)):
        failures.append("the ZA<=7 second-model-level 10 m branch is dead")
    if not np.any((za1 <= 7.0) & ~((za2 > 7.0) & (za2 < 13.0))):
        failures.append("the ZA<=7 neutral-log 10 m branch is dead")
    if not np.any((za1 > 7.0) & (za1 < 13.0)):
        failures.append("the 7<ZA<13 10 m branch is dead")
    if not np.any(za1 >= 13.0):
        failures.append("the coarse PSIX 10 m branch is dead")
    water = step1["xland"] > 1.5
    if not np.any(step1["znt"][water] != step1["znt_input"][water]):
        failures.append("no water column shows an updated ZNT")
    if not np.any(step2["znt_input"][water] != step1["znt_input"][water]):
        failures.append("the evolving water ZNT is not carried into step 2")
    fallback = (
        step1["th2"] - (step1["tsk"] + 2.0
                        * (step1["t1"] - step1["tsk"]) / za1)
    )
    if not np.any(np.abs(fallback) < 1.0):
        # THGB==TSK only where PSFC==100 kPa, which holds for every column.
        failures.append("the TH2 bracketing fallback is never taken")

    # ISFFLX<1 is a pure post-processing branch (:1027-1044): thirteen
    # outputs become the constant 0 and nothing else changes.  Both halves
    # are checked, because it is what licenses reusing the (1, 1) table.
    off = stages[(1, 0)]
    for name in ISFFLX0_ZEROED:
        if np.any(off[name] != 0.0):
            failures.append(f"isfflx=0 stage does not zero {name}")
        if not np.any(step1[name] != 0.0):
            failures.append(
                f"{name} is zero on the isfflx=1 stage too, so the isfflx=0"
                " zeroing check discriminates nothing"
            )
    for name in COLUMN_OUTPUTS:
        if name in ISFFLX0_ZEROED:
            continue
        if not np.array_equal(off[name], step1[name]):
            failures.append(
                f"isfflx=0 moved {name}, which :1027-1044 leaves alone"
            )


def validate(column_path: Path, wrapper_path: Path) -> None:
    failures: list[str] = []
    report: list[tuple[str, str, float, float, int, int]] = []

    column_rows = _read(column_path)
    stages: dict[tuple[int, int], dict[str, np.ndarray]] = {}
    for key in sorted({_stage_key(row) for row in column_rows}):
        rows = [row for row in column_rows if _stage_key(row) == key]
        names = tuple(row["case"] for row in rows)
        if names != EXPECTED_CASES:
            raise ValueError(f"stage {key} cases are {names}")
        stages[key] = _fields(rows)
    for key in ((1, 1), (2, 1), (1, 0)):
        if key not in stages:
            raise ValueError(f"missing SFCLAY1D_mynn stage {key}")

    for (itimestep, isfflx), fields in stages.items():
        if not all(math.isfinite(v) for a in fields.values() for v in a):
            raise ValueError(f"stage {itimestep}/{isfflx} has non-finite data")
        values = {
            name: fields[INPUT_ALIASES.get(name, name)] for name in INPUT_NAMES
        }
        actual = mynn_surface_layer_default(
            values, itimestep=itimestep, isfflx=isfflx,
            mol=fields["mol_input"], ustm=fields["ustm_input"],
        )
        _compare(f"sfclay1d/step{itimestep}/isfflx{isfflx}",
                 COLUMN_OUTPUTS, actual, fields, WIDE_ULP[(itimestep, 1)],
                 report, failures,
                 zeroed=ISFFLX0_ZEROED if isfflx == 0 else ())

    _assert_branch_coverage(stages, failures)

    wrapper_rows = _read(wrapper_path)
    for itimestep in (1, 2):
        rows = [r for r in wrapper_rows if int(r["itimestep"]) == itimestep]
        names = tuple(row["case"] for row in rows)
        if names != EXPECTED_CASES:
            raise ValueError(f"wrapper step {itimestep} cases are {names}")
        fields = _fields(rows)
        values = {
            name: fields[INPUT_ALIASES.get(name, name)] for name in INPUT_NAMES
        }
        mol = fields["mol_input"]
        if itimestep == 1:
            seed = mynn_sfclay_first_step_state(
                fields["u1"], fields["v1"], fields["qv1"]
            )
            # If the recorded entry state already equalled the seed, the
            # wrapper rows would not gate the :329-337 block at all.
            if np.allclose(seed["ust"], fields["ust_input"]):
                failures.append(
                    "wrapper step 1 entry UST already equals the seeded UST; "
                    "the itimestep==1 prologue is not gated"
                )
            if np.allclose(seed["qsfc"], fields["qsfc_input"]):
                failures.append(
                    "wrapper step 1 entry QSFC already equals the seeded "
                    "QSFC; the itimestep==1 prologue is not gated"
                )
            values["ust"] = seed["ust"]
            values["qsfc"] = seed["qsfc"]
            mol = seed["mol"]
        actual = mynn_surface_layer_default(
            values, itimestep=itimestep, isfflx=1,
            mol=mol, ustm=fields["ustm_input"],
        )
        _compare(f"sfclay_mynn/step{itimestep}", WRAPPER_OUTPUTS,
                 actual, fields, WRAPPER_ULP[itimestep], report, failures)

    width = max(len(f"{stage}/{name}") for stage, name, *_ in report)
    print(f"{'stage/field':{width}s} {'max_abs':>12s} {'max_rel':>12s} "
          f"{'max_ulp':>8s} {'gate':>6s}")
    for stage, name, abs_err, rel, ulp, budget in report:
        print(f"{stage + '/' + name:{width}s} {abs_err:12.4e} {rel:12.4e} "
              f"{ulp:8d} {budget:6d}")
    print(f"worst max_rel {max(r[3] for r in report):.4e} "
          f"(reported, not gated)")
    print(f"worst max_ulp {max(r[4] for r in report)} against a worst "
          f"measured {max(r[5] for r in report)}")
    if failures:
        raise SystemExit("FAIL\n  " + "\n  ".join(failures))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(
            "usage: validate_wide_oracle.py COLUMN.csv WRAPPER.csv"
        )
    validate(Path(sys.argv[1]), Path(sys.argv[2]))
    print(f"PASS {sys.argv[1]} {sys.argv[2]}")
