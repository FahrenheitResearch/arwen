#!/usr/bin/env python3
"""Parity check for the gpuwm ``snowsoil`` port against unmodified WRF v4.6.1.

Reads a freshly generated ``snowsoil`` CSV, checks that the fixture is
structurally complete and that each named regime really entered the snow branch
it was designed to exercise, then replays every case through
``gpuwm.core.ruc.ruc_snow_soil_step`` and reports
``max_abs``/``max_rel``/``max_ulp`` per output.  The achieved bar is bitwise,
so the script fails on any nonzero ULP.

With ``--contract`` it validates the companion contract fixture instead: the
three regimes that ``run_snowsoil_contract.F90`` adds to kill the argument
mutants ``ruc-snowsoil.csv`` alone cannot detect (``qcatm``, ``snom``, ``sat``,
the entry ``qsg`` and the entry ``ilnb``).
"""

from __future__ import annotations

import csv
from pathlib import Path
import sys

import numpy as np

from gpuwm.core.fp32_ulp import monotone_fp32_key
from gpuwm.core.ruc import RUC_SNOW_SOIL_COLUMN_INPUTS, ruc_snow_soil_step


CASES = (
    "fresh_snow_cold_forest",
    "deep_aged_snow_grass",
    "melting_snow_rain",
    "thin_snow_frozen_crop",
)
CONTRACT_CASES = (
    "partial_cover_two_layer",
    "sublimating_patch",
    "blended_retained_ilnb",
)
NZS = 9
RIW = np.float32(np.float32(900.0) * np.float32(1.0e-3))
PROFILE_OUTPUTS = (
    "tso", "soilmois", "smfrkeep", "keepfr", "soilice", "soiliqw",
)
COLUMN_OUTPUTS = (
    "cst", "dew", "soilt", "soilt1", "tsnav", "qvg", "qsg", "qcg",
    "snwe", "snhei", "rhosn", "snweprint", "snheiprint", "rsm", "smelt",
    "snoh", "snflx", "snom", "edir1", "ec1", "ett1", "eeta", "qfx",
    "hfx", "s", "sublim", "prcpl", "fltot", "runoff1", "runoff2",
    "mavail", "infiltrp",
)
# snowsoil's inout scalars enter the routine under a *_before column name.
ENTRY_NAMES = {
    "snhei": "snhei_before", "snwe": "snwe_before",
    "rhosn": "rhosn_before", "soilt": "soilt_before",
    "soilt1": "soilt1_before", "qvg": "qvg_before", "qsg": "qsg_before",
    "qcg": "qcg_before", "cst": "cst_before", "snom": "snom_before",
}


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
        f"  {name:<11s} max_abs={float(np.max(difference)):.6e} "
        f"max_rel={float(np.max(relative)):.6e} "
        f"max_ulp={int(np.max(ulp))} signed_zero_mismatch={signed_zero}"
    )
    return int(np.max(ulp)) + signed_zero


def _shared_bounds(values: dict[str, np.ndarray], ncase: int) -> None:
    """Physical bounds and internal identities every snowsoil fixture obeys."""

    col = {name: field[0] for name, field in values.items()}
    for name in ("tso_before", "tso_after", "soilt_after", "soilt1_after"):
        field = values[name]
        if not np.all((field > 180.0) & (field < 330.0)):
            raise SystemExit(f"{name} left the plausible temperature range")
    if not np.all(values["soilice"] >= 0.0):
        raise SystemExit("snowsoil produced negative soil ice")
    if not np.all(values["soiliqw"] >= 0.0):
        raise SystemExit("snowsoil produced negative liquid soil water")
    ceiling = (col["dqm"] + col["qmin"])[:, None]
    if not np.all(values["soilmois_after"].T <= ceiling * (1.0 + 1e-5)):
        raise SystemExit("soil moisture exceeded porosity")
    # snowsoil rebuilds the freezing-curve partition after snowtemp but before
    # soilmoist, and soilmoist never writes soiliqw back.  The returned
    # soilice/soiliqw therefore partition the moisture state as it was on
    # ENTRY, not the moisture state that is returned alongside them.
    if not np.allclose(
        values["soilmois_before"],
        values["soiliqw"] + values["soilice"] * RIW,
        rtol=1e-5, atol=1e-6,
    ):
        raise SystemExit("liquid/ice partition does not sum to the entry moisture")
    if not np.all((col["mavail_after"] >= 0.0) & (col["mavail_after"] <= 1.0)):
        raise SystemExit("moisture availability left [0,1]")
    if not np.all((col["rhosn_after"] >= 58.8) & (col["rhosn_after"] <= 500.0)):
        raise SystemExit("snow density left the Koren 58.8-500 kg/m^3 band")
    if not np.all(col["snwe_after"] >= 0.0):
        raise SystemExit("snowsoil produced negative snow water equivalent")
    for name in ("smelt", "snoh", "rsm", "runoff1", "runoff2", "qcg_after"):
        if not np.all(col[name] >= 0.0):
            raise SystemExit(f"{name} went negative")
    if not np.all(np.isin(values["keepfr_after"], (0.0, 1.0))):
        raise SystemExit("keepfr left {0,1}")
    frozen = values["soilice"] > 0.0
    rising = ((values["tso_after"] > values["tso_before"])
              & (values["soilmois_after"] > values["soilmois_before"]))
    if not np.array_equal(values["keepfr_after"][frozen],
                          rising[frozen].astype(np.float32)):
        raise SystemExit("keepfr update rule was not reproduced")
    if not all(np.any(values["tso_after"][:, n] != values["tso_before"][:, n])
               for n in range(ncase)):
        raise SystemExit("snowsoil failed to evolve every temperature profile")
    if not all(
        np.any(values["soilmois_after"][:, n] != values["soilmois_before"][:, n])
        for n in range(ncase)
    ):
        raise SystemExit("snowsoil failed to evolve every moisture profile")


def _structure(values: dict[str, np.ndarray]) -> None:
    """Fail if the base fixture stops discriminating the four snow branches."""

    _shared_bounds(values, len(CASES))
    col = {name: field[0] for name, field in values.items()}
    if not np.allclose(
        col["snheiprint"], col["snweprint"] * 1.0e3 / col["rhosn_after"],
        rtol=1e-5, atol=1e-9,
    ):
        raise SystemExit("reported snow depth is inconsistent with its density")
    if not np.all(np.abs(col["fltot"]) <= 25.0):
        raise SystemExit(f"surface energy residual is too large: {col['fltot']!r}")

    if not np.array_equal(col["ilnb_after"], [1.0, 2.0, 2.0, 1.0]):
        raise SystemExit(f"unexpected snow layer counts {col['ilnb_after']!r}")
    snth = 0.01e3 / col["rhosn_before"]
    deltsn = 0.05e3 / col["rhosn_before"]
    snhei = col["snhei_before"]
    if not (snth[0] <= snhei[0] <= deltsn[0] + snth[0]):
        raise SystemExit("fresh-snow regime is not in the one-layer depth band")
    if not snhei[1] > deltsn[1] + snth[1]:
        raise SystemExit("aged-snow regime is not in the two-layer depth band")
    if not snhei[2] > deltsn[2] + snth[2]:
        raise SystemExit("melting regime is not in the two-layer depth band")
    if not snhei[3] < snth[3]:
        raise SystemExit("thin-snow regime is not below the blending threshold")

    for n in (0, 1, 3):
        if not np.all(values["soilice"][:, n] > 0.0):
            raise SystemExit(f"cold regime {CASES[n]!r} lost its frozen soil")
    if not np.all(values["soilice"][:, 2] == 0.0):
        raise SystemExit("melting regime should sit on fully thawed soil")
    if not np.all(values["keepfr_before"][:, 3] == 1.0):
        raise SystemExit("latched regime did not enter with keepfr = 1")
    latched = values["soilice"][:, 3]
    cap = values["smfrkeep_before"][:, 3]
    if not np.all(latched <= cap * (1.0 + 1e-5)):
        raise SystemExit("latched regime let soil ice exceed smfrkeep")
    if not np.any(np.isclose(latched, cap, rtol=1e-5, atol=1e-7)):
        raise SystemExit("latched regime never hit the smfrkeep ice cap")

    if not (col["smelt"][2] > 0.0 and col["snoh"][2] > 0.0
            and col["rsm"][2] > 0.0):
        raise SystemExit("melting regime produced no melt")
    if not col["snom_after"][2] > col["snom_before"][2]:
        raise SystemExit("melting regime did not accumulate snow melt")
    if not col["runoff1"][2] > 0.0:
        raise SystemExit("melting regime produced no surface runoff")
    if not abs(col["soilt_after"][2] - 273.15) < 1.0e-3:
        raise SystemExit("melting regime skin temperature was not held at 273.15")
    if not np.all(col["smelt"][[0, 1, 3]] == 0.0):
        raise SystemExit("a cold regime melted unexpectedly")
    if not (col["edir1"][0] > 0.0 and col["dew_after"][0] == 0.0):
        raise SystemExit("no regime took the evaporation branch")
    if not (col["dew_after"][2] > 0.0 and col["edir1"][2] == 0.0):
        raise SystemExit("no regime took the condensation branch")


def _structure_contract(values: dict[str, np.ndarray]) -> None:
    """Fail if the contract fixture stops killing the argument mutants."""

    _shared_bounds(values, len(CONTRACT_CASES))
    col = {name: field[0] for name, field in values.items()}
    # Every contract case runs under partial snow cover, which is the only way
    # soilmoist's r8 term - and with it qcatm - can reach an output at all.
    if not np.all((col["snowfrac"] > 0.0) & (col["snowfrac"] < 1.0)):
        raise SystemExit("contract fixture lost its partial snow cover")
    if not np.all(col["qcatm"] > 0.0):
        raise SystemExit("contract fixture lost its nonzero qcatm")
    if not np.all(col["snom_before"] > 0.0):
        raise SystemExit("contract fixture lost its nonzero entry snom")
    if not np.all(col["ilnb_before"] == 3.0):
        raise SystemExit("contract fixture lost its ilnb = 3 entry value")
    # Case 1: canopy water strictly inside the min(0.25, (cst/sat)**cn) clamp,
    # which is the only regime where sat and cn both move an output.
    fraction = (col["cst_before"] / col["sat"]) ** col["cn"]
    if not 0.0 < fraction[0] < 0.25:
        raise SystemExit("contract case 1 canopy fraction is not interior")
    if not col["cst_after"][0] < col["cst_before"][0]:
        raise SystemExit("contract case 1 did not draw down canopy water")
    if not col["ilnb_after"][0] == 2.0:
        raise SystemExit("contract case 1 is not the two-layer branch")
    # Case 2: the pack fully sublimates, so snowsoil's beta limiter fired and
    # the entry qsg reached edir1 through it.
    if not col["snwe_after"][1] == 0.0:
        raise SystemExit("contract case 2 did not sublimate its whole pack")
    if not col["edir1"][1] > 0.0:
        raise SystemExit("contract case 2 produced no direct evaporation")
    snth = 0.01e3 / col["rhosn_before"]
    if not col["snhei_before"][1] < snth[1]:
        raise SystemExit("contract case 2 is not a blended column")
    # Cases 2 and 3 are blended, so snowtemp never assigns ilnb and the
    # caller's value has to survive.
    if not np.all(col["ilnb_after"][1:] == 3.0):
        raise SystemExit("contract fixture no longer retains the entry ilnb")
    # Case 3 keeps its pack, so the retained ilnb > 1 selects the multi-layer
    # tsnav form on a blended column.
    if not col["snhei_after"][2] > 0.0:
        raise SystemExit("contract case 3 lost its pack")
    if not col["snhei_before"][2] < snth[2]:
        raise SystemExit("contract case 3 is not a blended column")
    if not np.all(col["smelt"] == 0.0):
        raise SystemExit("contract fixture melted unexpectedly")


def main(path: str, contract: bool = False) -> None:
    cases = CONTRACT_CASES if contract else CASES
    ncase = len(cases)
    with Path(path).open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != ncase * NZS:
        raise SystemExit(f"expected {ncase * NZS} rows, got {len(rows)}")
    found = tuple(dict.fromkeys(row["case"] for row in rows))
    if found != cases:
        raise SystemExit(f"unexpected case inventory {found!r}")

    numeric = tuple(key for key in rows[0] if key != "case")
    field = {
        name: np.asarray([float(row[name]) for row in rows], dtype=np.float32)
        .reshape(ncase, NZS)
        .T
        for name in numeric
    }
    if not all(np.all(np.isfinite(value)) for value in field.values()):
        raise SystemExit("RUC snowsoil oracle contains non-finite output")
    if contract:
        _structure_contract(field)
    else:
        _structure(field)

    actual = {name: np.empty((NZS, ncase), dtype=np.float32)
              for name in PROFILE_OUTPUTS}
    actual.update({name: np.empty(ncase, dtype=np.float32)
                   for name in COLUMN_OUTPUTS})
    layers = np.empty(ncase, dtype=np.int32)
    for case in range(ncase):
        values = {
            name: field[key][:, case : case + 1]
            for name, key in (
                ("soilmois", "soilmois_before"), ("tso", "tso_before"),
                ("smfrkeep", "smfrkeep_before"), ("keepfr", "keepfr_before"),
            )
        }
        values.update({
            name: field[ENTRY_NAMES.get(name, name)][0, case : case + 1]
            for name in RUC_SNOW_SOIL_COLUMN_INPUTS
        })
        result = ruc_snow_soil_step(
            values,
            field["iland"][0, case : case + 1].astype(np.int32),
            nroot=field["nroot"][0, case : case + 1].astype(np.int32),
            delt=float(field["delt"][0, case]),
            conflx=float(field["conflx"][0, case]),
            ilnb=field["ilnb_before"][0, case : case + 1].astype(np.int32),
            cw=float(field["cw"][0, case]),
        )
        for name in PROFILE_OUTPUTS:
            actual[name][:, case] = getattr(result, name)[:, 0]
        for name in COLUMN_OUTPUTS:
            actual[name][case] = getattr(result, name)[0]
        layers[case] = int(result.ilnb[0])

    label = "contract " if contract else ""
    print(
        f"RUC snowsoil {label}oracle parity "
        "(gpuwm.core.ruc.ruc_snow_soil_step):"
    )
    failures = 0
    for name in PROFILE_OUTPUTS:
        expected = field.get(f"{name}_after", field.get(name))
        failures += _report(name, actual[name], expected)
    for name in COLUMN_OUTPUTS:
        expected = field.get(f"{name}_after", field.get(name))[0]
        failures += _report(name, actual[name], expected)
    expected_layers = field["ilnb_after"][0].astype(np.int32)
    print(f"  {'ilnb':<11s} {list(layers)} expected {list(expected_layers)}")
    failures += int(np.count_nonzero(layers != expected_layers))
    if failures:
        raise SystemExit(f"RUC snowsoil {label}oracle: FAIL (not bitwise)")
    if contract:
        print(
            f"RUC snowsoil contract oracle: PASS (bitwise over {ncase} "
            "argument-contract regimes x 9 levels: partial cover with "
            "accumulated melt, full sublimation under beta < 1, blended "
            "column with a retained layer count)"
        )
    else:
        print(
            f"RUC snowsoil oracle: PASS (bitwise over {ncase} snow-covered "
            f"land regimes x {NZS} levels: 1-layer, 2-layer, melting, blended "
            "over latched frozen soil)"
        )


if __name__ == "__main__":
    argv = sys.argv[1:]
    is_contract = "--contract" in argv
    argv = [arg for arg in argv if arg != "--contract"]
    if len(argv) != 1:
        raise SystemExit(
            "usage: validate_snowsoil_oracle.py [--contract] RUC_SNOWSOIL.csv"
        )
    main(argv[0], contract=is_contract)
