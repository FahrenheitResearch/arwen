#!/usr/bin/env python3
"""Parity check for the gpuwm ``sice`` port against unmodified WRF v4.6.1.

Loads the freshly generated ``sice`` CSV, replays every case through
``gpuwm.core.ruc.ruc_sea_ice_step`` and reports
``max_abs``/``max_rel``/``max_ulp`` per output.  The achieved bar is bitwise,
so the script fails on any nonzero ULP.
"""

from __future__ import annotations

import csv
from pathlib import Path
import sys

import numpy as np

from gpuwm.core.fp32_ulp import monotone_fp32_key
from gpuwm.core.ruc import RUC_SEA_ICE_COLUMN_INPUTS, ruc_sea_ice_step


CASES = (
    "cold_thick_ice",
    "near_melt_cap",
    "thin_ice_warm_base",
    "strong_forcing",
    "dew_condensing",
    "myj_condensing",
    "myj_evaporating",
    "rain_on_ice",
)
NCASE = len(CASES)
NZS = 9
ICE_CAP = np.float32(271.4)
PROFILE_OUTPUTS = ("tso", "soilmois", "soiliqw", "soilice", "smfrkeep", "keepfr")
COLUMN_OUTPUTS = (
    "dew", "soilt", "qvg", "qsg", "qcg", "eeta", "qfx", "hfx", "s",
    "evapl", "prcpl", "fltot",
)


#: Monotonic integer ordering of float32 bits; +0 and -0 collapse.  Shared
#: rather than re-derived: see gpuwm/core/fp32_ulp.py.
_ordinal = monotone_fp32_key


def _report(name: str, actual: np.ndarray, expected: np.ndarray) -> int:
    difference = np.abs(
        actual.astype(np.float64) - expected.astype(np.float64)
    )
    denominator = np.abs(expected.astype(np.float64))
    relative = np.zeros_like(difference)
    np.divide(difference, denominator, out=relative, where=denominator > 0.0)
    ulp = np.abs(_ordinal(actual) - _ordinal(expected))
    signed_zero = int(
        np.count_nonzero(
            (actual == 0.0)
            & (expected == 0.0)
            & (np.signbit(actual) != np.signbit(expected))
        )
    )
    print(
        f"  {name:<10s} max_abs={float(np.max(difference)):.6e} "
        f"max_rel={float(np.max(relative)):.6e} "
        f"max_ulp={int(np.max(ulp))} signed_zero_mismatch={signed_zero}"
    )
    return int(np.max(ulp)) + signed_zero


def main(path: str) -> None:
    with Path(path).open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != NCASE * NZS:
        raise SystemExit(f"expected {NCASE * NZS} rows, got {len(rows)}")
    cases = tuple(dict.fromkeys(row["case"] for row in rows))
    if cases != CASES:
        raise SystemExit(f"unexpected case inventory {cases!r}")

    numeric = tuple(key for key in rows[0] if key != "case")
    field = {
        name: np.asarray([float(row[name]) for row in rows], dtype=np.float32)
        .reshape(NCASE, NZS)
        .T
        for name in numeric
    }
    if not all(np.all(np.isfinite(value)) for value in field.values()):
        raise SystemExit("RUC sice oracle contains non-finite output")

    # Structural checks: every named regime must really be the regime it
    # claims, otherwise a bitwise pass would be proving nothing.
    myj = field["myj"][0] > 0.5
    if list(myj) != [False, False, False, False, False, True, True, False]:
        raise SystemExit("myj inventory does not match the case names")
    if not np.all(field["tso_after"] <= ICE_CAP):
        raise SystemExit("a returned ice temperature exceeds the 271.4 K cap")
    if field["soilt_after"][0, 1] != ICE_CAP:
        raise SystemExit("near_melt_cap did not reach the 271.4 K cap")
    if field["soilt_after"][0, 0] >= ICE_CAP:
        raise SystemExit("cold_thick_ice unexpectedly reached the cap")
    if not field["dew_after"][0, 4] > 0.0:
        raise SystemExit("dew_condensing did not enter the condensation branch")
    if not field["eeta"][0, 3] > 0.0:
        raise SystemExit("strong_forcing did not enter the evaporation branch")
    if field["qfx"][0, 4] == field["qfx"][0, 5]:
        raise SystemExit("the MYJ and RUC condensation branches agree")
    if not field["prcpl"][0, 7] > 0.0:
        raise SystemExit("rain_on_ice did not carry a precipitation rate")
    if not np.all(field["fltot"] == 0.0):
        raise SystemExit("fltot is not identically zero")

    actual = {name: np.empty((NZS, NCASE), dtype=np.float32)
              for name in PROFILE_OUTPUTS}
    actual.update({name: np.empty(NCASE, dtype=np.float32)
                   for name in COLUMN_OUTPUTS})
    for case in range(NCASE):
        values = {
            "capice": field["capice"][:, case : case + 1],
            "thdifice": field["thdifice"][:, case : case + 1],
            "tso": field["tso_before"][:, case : case + 1],
        }
        before = {"soilt": "soilt_before", "qvg": "qvg_before", "qsg": "qsg_before"}
        values.update({
            name: field[before.get(name, name)][0, case : case + 1]
            for name in RUC_SEA_ICE_COLUMN_INPUTS
        })
        result = ruc_sea_ice_step(
            values,
            delt=float(field["delt"][0, case]),
            conflx=float(field["conflx"][0, case]),
            myj=bool(myj[case]),
            cw=float(field["cw"][0, case]),
        )
        for name in PROFILE_OUTPUTS:
            actual[name][:, case] = getattr(result, name)[:, 0]
        for name in COLUMN_OUTPUTS:
            actual[name][case] = getattr(result, name)[0]

    print("RUC sice oracle parity (gpuwm.core.ruc.ruc_sea_ice_step):")
    failures = 0
    for name in PROFILE_OUTPUTS:
        expected = field["tso_after"] if name == "tso" else field[f"{name}_after"]
        failures += _report(name, actual[name], expected)
    for name in COLUMN_OUTPUTS:
        expected = (
            field[f"{name}_after"][0]
            if f"{name}_after" in field
            else field[name][0]
        )
        failures += _report(name, actual[name], expected)
    if failures:
        raise SystemExit("RUC sice oracle: FAIL (not bitwise)")
    print(f"RUC sice oracle: PASS (bitwise over {NCASE} regimes x {NZS} levels)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: validate_sice_oracle.py RUC_SICE.csv")
    main(sys.argv[1])
