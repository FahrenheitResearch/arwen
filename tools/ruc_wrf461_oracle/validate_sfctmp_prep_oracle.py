#!/usr/bin/env python3
"""Parity check for the gpuwm ``sfctmp`` snow-preparation port.

The fixture is read out of the running, UNMODIFIED WRF v4.6.1
``module_sf_ruclsm`` by gdb (see ``sfctmp_prep.gdb``), which brackets
``phys/module_sf_ruclsm.F:1400-1766`` with breakpoints at ``:1418`` and at
whichever of ``:1767``/``:2120`` the call reaches.

This script

  1. cross-checks every recorded input against the forcing
     ``run_sfctmp_prep.F90`` says it passed, if that CSV is supplied, so a
     misread gdb frame cannot pass unnoticed;
  2. asserts each named regime really is the regime it claims;
  3. replays every case through ``gpuwm.core.ruc.ruc_snow_preparation`` and
     reports ``max_abs``/``max_rel``/``max_ulp`` per output.  The achieved bar
     is bitwise, so it fails on any nonzero ULP;
  4. runs a MUTATION STUDY over the port.  For every argument the block
     reads, and for every distinct value that argument takes across the
     fixture, it builds the mutant that hard-codes the argument to that value
     -- i.e. the port that stops reading the argument -- and requires the
     mutant to FAIL to reproduce the fixture.  A surviving mutant means the
     fixture cannot detect that argument being dropped, so a bitwise match
     would be far weaker evidence than it looks.  ``isncovr_opt`` is the one
     argument excluded, with justification, because WRF pins it at compile
     time.
"""

from __future__ import annotations

import csv
from pathlib import Path
import sys

import numpy as np

from gpuwm.core.fp32_ulp import monotone_fp32_key
from gpuwm.core.ruc import (
    RUC_SNOW_PREP_COLUMN_INPUTS,
    RUC_SNOW_PREP_COLUMN_OUTPUTS,
    RUC_SNOW_PREP_PROFILE_OUTPUTS,
    ruc_snow_preparation,
)


CASES = (
    "warm_rain_canopy_drip",
    "bare_soil_rain",
    "deep_pack_densify",
    "shallow_snow_mosaic",
    "aged_pack_no_densify",
    "new_snow_mosaic_drip",
    "fresh_snow_keep_albedo",
    "graupel_dense_new_snow",
    "urban_snow_cap",
    "sea_ice_snow_deep",
    "sea_ice_snow_partial",
    "sea_ice_snow_mosaic",
    "sea_ice_bare",
    "snow_water_no_depth",
    "usgs_crop_rain_drip",
    "usgs_urban_snow_cap",
    "usgs_warm_pack_new_snow",
)
NCASE = len(CASES)
NZS = 9
# WRF's snow/ice land-use class fixes the land-use dataset.
DATASETS = {15: "MODIFIED_IGBP_MODIS_NOAH", 24: "USGS"}
# The two locals WRF only defines inside the snhei>0. branch.
CONDITIONAL_OUTPUTS = ("keep_snow_albedo", "snowfrac2")
SCALAR_ARGUMENTS = ("delt", "c1sn", "c2sn", "isice", "ivgtyp", "iland", "mminlu")
# Inputs whose CSV column carries the pre-block value.
BEFORE = {
    "snwe": "snwe_before",
    "snhei": "snhei_before",
    "snowfrac": "snowfrac_before",
    "rhosn": "rhosn_before",
    "rhosnfall": "rhosnfall_before",
    "cst": "cst_before",
    "alb": "alb_before",
    "emiss": "emiss_before",
    "znt": "znt_before",
}
# Outputs whose CSV column carries the post-block value.
AFTER = {
    "snwe": "snwe_after",
    "snhei": "snhei_after",
    "snowfrac": "snowfrac_after",
    "rhosn": "rhosn_after",
    "rhosnfall": "rhosnfall_after",
    "cst": "cst_after",
    "alb": "alb_after",
    "emiss": "emiss_after",
    "znt": "znt_after",
    "iland": "iland_after",
}
OUTPUTS = (
    RUC_SNOW_PREP_PROFILE_OUTPUTS + RUC_SNOW_PREP_COLUMN_OUTPUTS + ("iland",)
)


#: Monotonic integer ordering of float32 bits; +0 and -0 collapse.  Shared
#: rather than re-derived: see gpuwm/core/fp32_ulp.py.
_ordinal = monotone_fp32_key


def _report(name: str, actual: np.ndarray, expected: np.ndarray) -> int:
    difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
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
        f"  {name:<17s} max_abs={float(np.max(difference)):.6e} "
        f"max_rel={float(np.max(relative)):.6e} "
        f"max_ulp={int(np.max(ulp))} signed_zero_mismatch={signed_zero}"
    )
    return int(np.max(ulp)) + signed_zero


def _load(path: str) -> dict[str, np.ndarray]:
    with Path(path).open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != NCASE * NZS:
        raise SystemExit(f"{path}: expected {NCASE * NZS} rows, got {len(rows)}")
    indices = tuple(dict.fromkeys(int(row["case_index"]) for row in rows))
    if indices != tuple(range(1, NCASE + 1)):
        raise SystemExit(f"{path}: unexpected case inventory {indices!r}")
    field = {}
    for name in rows[0]:
        if name == "case":
            continue
        field[name] = (
            np.asarray([float(row[name]) for row in rows], dtype=np.float32)
            .reshape(NCASE, NZS)
            .T
        )
    return field


def _case_arguments(field: dict[str, np.ndarray], case: int) -> dict[str, object]:
    isice = int(field["isice"][0, case])
    if isice not in DATASETS:
        raise SystemExit(f"case {case + 1}: unsupported isice {isice}")
    arguments = {
        "ts1d": field["ts1d"][:, case : case + 1],
        "delt": float(field["delt"][0, case]),
        "c1sn": float(field["c1sn"][0, case]),
        "c2sn": float(field["c2sn"][0, case]),
        "isice": isice,
        "ivgtyp": np.asarray([int(field["ivgtyp"][0, case])], dtype=np.int32),
        "iland": np.asarray(
            [int(field["iland_before"][0, case])], dtype=np.int32
        ),
        "mminlu": DATASETS[isice],
    }
    for name in RUC_SNOW_PREP_COLUMN_INPUTS:
        arguments[name] = field[BEFORE.get(name, name)][0, case : case + 1]
    return arguments


def _run_case(arguments: dict[str, object]) -> dict[str, np.ndarray]:
    values = {name: arguments[name] for name in ("ts1d",)}
    values.update(
        {name: arguments[name] for name in RUC_SNOW_PREP_COLUMN_INPUTS}
    )
    result = ruc_snow_preparation(
        values,
        delt=arguments["delt"],
        ivgtyp=arguments["ivgtyp"],
        iland=arguments["iland"],
        isice=arguments["isice"],
        c1sn=arguments["c1sn"],
        c2sn=arguments["c2sn"],
        mminlu=arguments["mminlu"],
    )
    return {
        name: np.asarray(getattr(result, name), dtype=np.float32)
        for name in OUTPUTS
    }


def _replay(
    field: dict[str, np.ndarray],
    mutate: tuple[str, object] | None = None,
) -> dict[str, np.ndarray]:
    profile = {
        name: np.empty((NZS, NCASE), dtype=np.float32)
        for name in RUC_SNOW_PREP_PROFILE_OUTPUTS
    }
    column = {
        name: np.empty(NCASE, dtype=np.float32)
        for name in RUC_SNOW_PREP_COLUMN_OUTPUTS + ("iland",)
    }
    for case in range(NCASE):
        arguments = _case_arguments(field, case)
        if mutate is not None:
            arguments[mutate[0]] = mutate[1]
        result = _run_case(arguments)
        for name in profile:
            profile[name][:, case] = result[name][:, 0]
        for name in column:
            column[name][case] = result[name][0]
    return {**profile, **column}


def _identical(left: dict[str, np.ndarray], right: dict[str, np.ndarray]) -> bool:
    for name in left:
        if not np.array_equal(
            left[name].view(np.int32), right[name].view(np.int32)
        ):
            return False
    return True


def _structure(field: dict[str, np.ndarray]) -> None:
    seaice = list(field["seaice"][0] > 0.5)
    if seaice != [False] * 9 + [True] * 4 + [False] * 4:
        raise SystemExit("the sea-ice inventory does not match the case names")
    snow = list(field["snhei_after"][0] > 0.0)
    if snow != [
        False, False, True, True, True, True, True, True, True,
        True, True, True, False, False, False, True, True,
    ]:
        raise SystemExit("the snow-cover inventory does not match the case names")
    mosaic = list(field["snow_mosaic"][0])
    if mosaic != [1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0]:
        raise SystemExit(f"unexpected snow_mosaic inventory {mosaic!r}")
    if list(dict.fromkeys(field["isice"][0])) != [15, 24]:
        raise SystemExit("both land-use datasets must appear, MODI first")
    checks = (
        # The Koren compaction has to actually run somewhere, and the
        # goto-777 shortcut has to actually be taken somewhere.
        ("deep_pack_densify compacted the pack",
         field["rhosn_after"][0, 2] != field["rhosn_before"][0, 2]),
        ("aged_pack_no_densify skipped the compaction",
         field["rhosn_after"][0, 4] == field["rhosn_before"][0, 4]),
        ("usgs_warm_pack_new_snow clamped tsnav>0 through min(0.,tsnav)",
         field["tsnav"][0, 16] > 0.0
         and field["rhosn_after"][0, 16] != field["rhosn_before"][0, 16]),
        ("urban_snow_cap hit the URBAN=13 clamp",
         field["snowfrac_after"][0, 8] == np.float32(0.75)),
        ("usgs_urban_snow_cap hit the URBAN=1 clamp",
         field["snowfrac_after"][0, 15] == np.float32(0.75)),
        ("fresh_snow_keep_albedo latched keep_snow_albedo",
         field["keep_snow_albedo"][0, 6] == 1.0),
        ("fresh_snow_keep_albedo lifted albsn to 0.7",
         field["albsn"][0, 6] == np.float32(0.7)),
        ("usgs_warm_pack_new_snow lifted albsn to 0.7 as well",
         field["keep_snow_albedo"][0, 16] == 1.0
         and field["albsn"][0, 16] == np.float32(0.7)),
        ("graupel_dense_new_snow saturated rhosnfall at 500",
         field["rhosnfall_after"][0, 7] == np.float32(500.0)),
        ("graupel_dense_new_snow kept keep_snow_albedo clear",
         field["keep_snow_albedo"][0, 7] == 0.0),
        ("new_snow_mosaic_drip routed drip through the mosaic split",
         field["drip"][0, 5] > 0.0 and field["intwratio"][0, 5] > 0.0),
        ("warm_rain_canopy_drip spilled the canopy store",
         field["drip"][0, 0] > 0.0),
        ("bare_soil_rain took the vegfrac<=0.01 branch",
         field["vegfrac"][0, 1] == 0.0 and field["cst_after"][0, 1] == 0.0),
        ("shallow_snow_mosaic blended znt on the first branch",
         field["znt_after"][0, 3] != field["znt_before"][0, 3]),
        ("aged_pack_no_densify blended znt on the middle branch",
         field["znt_after"][0, 4] != field["znt_before"][0, 4]),
        ("deep_pack_densify reset znt to the snow class",
         field["znt_after"][0, 2] == field["zntsn"][0, 2]),
        ("sea_ice_snow_deep built the Zubov ice column",
         np.all(field["capice"][:, 9] > 0.0)),
        ("the land cases left the ice column at zero",
         np.all(field["capice"][:, :9] == 0.0)
         and np.all(field["capice"][:, 13:] == 0.0)),
        ("sea_ice_snow_deep took the alb_snow_free leg of albice",
         field["albice"][0, 9] == field["alb_snow_free"][0, 9]),
        ("sea_ice_snow_partial took the -0.05 leg of albice",
         field["albice"][0, 10]
         == np.float32(field["alb_snow_free"][0, 10] - np.float32(0.05))),
        ("sea_ice_snow_mosaic hit the albsn-0.1 albedo floor",
         field["alb_after"][0, 11]
         == np.float32(field["albsn"][0, 11] - np.float32(0.1))),
        ("warm_rain_canopy_drip passed alb through unchanged",
         field["alb_after"][0, 0] == field["alb_before"][0, 0]),
        ("bare_soil_rain passed emiss through unchanged",
         field["emiss_after"][0, 1] == field["emiss_before"][0, 1]
         and field["emiss_after"][0, 1] != field["emiss_snowfree"][0, 1]),
        ("the snow cases moved iland to the snow class",
         np.all(field["iland_after"][0, 2:12] == field["isice"][0, 2:12])
         and np.all(field["iland_after"][0, 15:] == field["isice"][0, 15:])),
        ("warm_rain_canopy_drip left iland alone",
         field["iland_after"][0, 0] == field["iland_before"][0, 0]),
        ("snow_water_no_depth kept snow water with zero depth",
         field["snwe_before"][0, 13] > 0.0
         and field["snhei_before"][0, 13] == 0.0
         and field["snwe_after"][0, 13] == field["snwe_before"][0, 13]),
        ("snow_water_no_depth left iland alone on a non-snow class",
         field["iland_after"][0, 13] == field["iland_before"][0, 13]
         and field["iland_after"][0, 13] != field["isice"][0, 13]),
    )
    for label, ok in checks:
        if not bool(ok):
            raise SystemExit(f"structural check failed: {label}")


def _cross_check(field: dict[str, np.ndarray], path: str) -> None:
    driver = _load(path)
    with Path(path).open(newline="", encoding="ascii") as stream:
        names = tuple(
            dict.fromkeys(row["case"] for row in csv.DictReader(stream))
        )
    if names != CASES:
        raise SystemExit(f"{path}: unexpected case names {names!r}")
    shared = [name for name in driver if name in field and name != "case_index"]
    if len(shared) < 30:
        raise SystemExit("the driver record shares too few columns to cross-check")
    for name in shared:
        if not np.array_equal(
            driver[name].view(np.int32), field[name].view(np.int32)
        ):
            raise SystemExit(
                f"gdb read a different {name} than run_sfctmp_prep.F90 passed"
            )
    print(
        f"input cross-check: {len(shared)} columns agree bitwise with "
        "run_sfctmp_prep.F90"
    )


def _mutation_study(field: dict[str, np.ndarray], baseline) -> None:
    """Require every argument's disappearance to break the fixture."""

    arguments = ("ts1d",) + RUC_SNOW_PREP_COLUMN_INPUTS + SCALAR_ARGUMENTS
    survivors = []
    total = 0
    for name in arguments:
        seen = []
        for case in range(NCASE):
            value = _case_arguments(field, case)[name]
            key = (
                np.asarray(value, dtype=np.float32).tobytes()
                if not isinstance(value, (str, int, float))
                else value
            )
            if key not in [entry[0] for entry in seen]:
                seen.append((key, value))
        killed = True
        for _, value in seen:
            total += 1
            try:
                mutant = _replay(field, mutate=(name, value))
            except (ValueError, TypeError):
                # The mutant cannot even run, so it cannot reproduce the
                # fixture.
                continue
            if _identical(mutant, baseline):
                survivors.append((name, value))
                killed = False
                break
        status = "killed" if killed else "SURVIVED"
        print(f"  {name:<17s} {len(seen):>2d} distinct value(s)  {status}")
    print(f"  ({total} mutants evaluated)")
    if survivors:
        detail = ", ".join(f"{name}={value!r}" for name, value in survivors)
        raise SystemExit(
            "the fixture cannot detect these arguments being dropped: " + detail
        )


def main(path: str, driver_path: str | None) -> None:
    field = _load(path)
    if not all(
        np.all(np.isfinite(value))
        for name, value in field.items()
        if name not in CONDITIONAL_OUTPUTS
    ):
        raise SystemExit("the RUC sfctmp prep oracle contains non-finite output")
    if driver_path is not None:
        _cross_check(field, driver_path)
    _structure(field)

    actual = _replay(field)
    print("RUC sfctmp snow-preparation parity "
          "(gpuwm.core.ruc.ruc_snow_preparation):")
    failures = 0
    defined = field["snhei_after"][0] > 0.0
    for name in RUC_SNOW_PREP_PROFILE_OUTPUTS:
        failures += _report(name, actual[name], field[name])
    for name in RUC_SNOW_PREP_COLUMN_OUTPUTS + ("iland",):
        expected = field[AFTER.get(name, name)][0]
        got = actual[name]
        if name in CONDITIONAL_OUTPUTS:
            expected = expected[defined]
            got = got[defined]
        failures += _report(name, got, expected)
    if failures:
        raise SystemExit("RUC sfctmp snow-preparation oracle: FAIL (not bitwise)")

    print(
        "mutation study (each mutant hard-codes one argument, so it must "
        "fail):"
    )
    _mutation_study(field, actual)
    print(
        f"RUC sfctmp snow-preparation oracle: PASS (bitwise over {NCASE} "
        f"regimes x {NZS} levels)"
    )


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        raise SystemExit(
            "usage: validate_sfctmp_prep_oracle.py RUC_SFCTMP_PREP.csv "
            "[RUC_SFCTMP_PREP_INPUTS.csv]"
        )
    main(sys.argv[1], sys.argv[2] if len(sys.argv) == 3 else None)
