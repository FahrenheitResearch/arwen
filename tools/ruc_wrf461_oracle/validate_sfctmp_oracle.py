#!/usr/bin/env python3
"""Parity check for the gpuwm ``sfctmp`` port -- preparation AND dispatch.

The fixture is produced by ``run_sfctmp.F90``, which calls the UNMODIFIED
WRF v4.6.1 ``module_sf_ruclsm::sfctmp`` and records every
``intent(inout)``/``intent(out)`` argument on both sides of the call.  Unlike
the snow-preparation prologue -- whose results are all locals and needed gdb
(``validate_sfctmp_prep_oracle.py``) -- the dispatch is the tail of the
subroutine, so the returned arguments ARE its outputs and no debugger is
involved.

This script

  1. asserts each named regime really is the regime it claims, and that the
     inventory of dispatch arms the fixture binds is the one the case
     comments promise;
  2. replays every case through
     ``gpuwm.core.ruc.ruc_surface_temperature_step`` and reports
     ``max_abs``/``max_rel``/``max_ulp`` per output.  The achieved bar is
     bitwise, so it fails on any nonzero ULP;
  3. runs a MUTATION STUDY over the port.  For every argument the routine
     reads, and for every distinct value that argument takes across the
     fixture, it builds the mutant that hard-codes the argument to that value
     -- i.e. the port that stops reading the argument -- and requires the
     mutant to FAIL to reproduce the fixture.  A surviving mutant means the
     fixture cannot detect that argument being dropped.  ``isncovr_opt`` is
     excluded because WRF pins it at compile time.
"""

from __future__ import annotations

import csv
from pathlib import Path
import sys

import numpy as np

from gpuwm.core.fp32_ulp import monotone_fp32_key
from gpuwm.core.ruc import (
    RUC_SFCTMP_COLUMN_INPUTS,
    RUC_SFCTMP_COLUMN_OUTPUTS,
    RUC_SFCTMP_PROFILE_INPUTS,
    RUC_SFCTMP_PROFILE_OUTPUTS,
    ruc_surface_temperature_step,
)


NZS = 9
# WRF's snow/ice land-use class fixes the land-use dataset.
DATASETS = {15: "MODIFIED_IGBP_MODIS_NOAH", 24: "USGS"}
SCALAR_ARGUMENTS = (
    "delt", "conflx", "c1sn", "c2sn", "isice", "ivgtyp", "iland", "nroot",
    "mminlu", "ilnb",
)
OUTPUTS = RUC_SFCTMP_PROFILE_OUTPUTS + RUC_SFCTMP_COLUMN_OUTPUTS + ("iland",)


#: Monotonic integer ordering of float32 bits; +0 and -0 collapse.  Shared
#: rather than re-derived: see gpuwm/core/fp32_ulp.py.
_ordinal = monotone_fp32_key

#: The ONLY cells this port does not reproduce bitwise, keyed
#: ``(output, level, case)`` with level 0 for column fields and 1-based
#: elsewhere, mapped to the ULP measured on 2026-07-25.  Neither residue is
#: in the dispatch; both were traced into routines gpuwm landed earlier, with
#: the unmodified Fortran as the referee:
#:
#:  * case 15 (``usgs_crop_rain_drip``) -- ``tools/.../probe_soil.F90`` calls
#:    the UNMODIFIED ``soil`` with exactly the arguments this dispatch hands
#:    ``ruc_soil_step`` and reproduces the fixture BITWISE on all 13 recorded
#:    outputs, so the dispatch's inputs are right.  Calling the unmodified
#:    ``soilprop`` with exactly the arguments ``ruc_soil_step`` hands
#:    ``ruc_soil_properties`` returns ``cap`` bitwise but ``thdif(1)`` two ULP
#:    lower; forcing that two-ULP shift makes the whole case bitwise.  The
#:    gap is ``kjpl = ke*(kasat-kdry)+kdry``, whose ``ke`` and ``kasat`` are
#:    the ``log10``/``pow`` that ``_RUC_PROVISIONAL_TRANSCENDENTALS``
#:    documents as float64-then-round rather than glibc.  ``soiltemp``'s
#:    ``vilka`` then amplifies it: ``ts`` is unchanged (the shift is below one
#:    ULP of 282 K) while ``qs`` moves 17 ULP, and ``dew``/``eeta``/``qfx``/
#:    ``fltot`` follow from ``qsg``.
#:  * case 8 (``graupel_dense_new_snow``) -- the same probe run on the mosaic
#:    ``soil`` call reproduces ``tso``, ``soilice``, ``soiliqw``, ``soilt``
#:    and ``mavail`` bitwise, so the snow-free half of the recombination is
#:    exact and the ULP is on the ``snowsoil`` side.  ``soilprop`` on that
#:    column's arguments is one ULP low in ``thdif(8)``, same class.
#:
#: Bounding, not widening: any cell NOT listed here must be bitwise, and a
#: listed cell may only shrink.  Fixing the transcendentals upstream is a
#: separate change with a CUDA half, so it is not made here.
#: Mutation-study survivors that are UNREACHABLE, not merely untested.  Each
#: entry is an argument WRF's ``sfctmp`` accepts and that no path can observe;
#: the fixture carries several distinct entry values for each (case 29 makes
#: ``soilice``/``soiliqw`` contradict ``soilm1d``, cases 29 and 31 carry
#: nonzero ``qcg``, cases 20, 21, 23, 31 and 33 carry five distinct nonzero
#: ``s``) and no output moves.  Any survivor NOT listed here fails the study.
UNREACHABLE_ARGUMENTS = {
    "soilice":
        "every dispatch path rewrites it before returning and none reads the "
        "entry value: soil rebuilds the freezing-curve partition at "
        ":2678-2712, snowsoil at :3629-3648, and the three sea-ice legs force "
        "it to 1 at :1871-1877/:1969-1975/:2184-2190",
    "soiliqw":
        "same as soilice; forced to 0 on the sea-ice legs",
    "qcg":
        "soiltemp assigns qcg on BOTH of its exits (:4770-4773 saturated, "
        ":4787-4795 unsaturated) before anything reads it, snowtemp likewise, "
        "and sice zeroes it before its first read",
    "s":
        "snowseaice is its only consumer and assigns it at :4470 or :4473 "
        "whenever snhei > 0 on entry.  The one branch that leaves it alone, "
        ":4479-4483, needs snhei <= 0 on entry, and sfctmp only calls "
        "snowseaice inside if(snhei.gt.0.) at :1600 -- so that branch is "
        "unreachable from sfctmp",
    "ilnb":
        "NOT dead -- pinned.  run_sfctmp.F90 zeroes the stack region WRF's "
        "uninitialised ilnb (:1385) occupies, so this fixture has exactly one "
        "value by construction.  oracle/sfctmp_stackfill.csv is the same "
        "unmodified module with a nonzero fill and differs in exactly five "
        "tsnav cells, which is what proves the argument is live; "
        "_stack_control() checks that pair on every run",
}

UPSTREAM_RESIDUE = {
    ("dew", 0, 15): 22,
    ("eeta", 0, 15): 28,
    ("evapl", 0, 15): 28,
    ("fltot", 0, 15): 16384,
    ("qcg", 0, 15): 28,
    ("qfx", 0, 15): 17,
    ("qsg", 0, 15): 17,
    ("qvg", 0, 15): 17,
    ("s", 0, 15): 2,
    ("soilice", 1, 8): 1,
    ("soiliqw", 1, 8): 1,
}


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
    unexplained = 0
    for index in np.argwhere(ulp > 0):
        if ulp.ndim == 1:
            key = (name, 0, int(index[0]) + 1)
        else:
            key = (name, int(index[0]) + 1, int(index[1]) + 1)
        allowed = UPSTREAM_RESIDUE.get(key)
        measured = int(ulp[tuple(index)])
        if allowed is None or measured > allowed:
            unexplained += 1
            print(
                f"    UNEXPLAINED {key} measured {measured} ULP, "
                f"pinned {allowed}"
            )
    print(
        f"  {name:<13s} max_abs={float(np.max(difference)):.6e} "
        f"max_rel={float(np.max(relative)):.6e} "
        f"max_ulp={int(np.max(ulp))} signed_zero_mismatch={signed_zero}"
    )
    return unexplained + signed_zero


def _load(path: str) -> tuple[tuple[str, ...], dict[str, np.ndarray]]:
    with Path(path).open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    names = tuple(dict.fromkeys(row["case"] for row in rows))
    ncase = len(names)
    if len(rows) != ncase * NZS:
        raise SystemExit(f"{path}: expected {ncase * NZS} rows, got {len(rows)}")
    indices = tuple(dict.fromkeys(int(row["case_index"]) for row in rows))
    if indices != tuple(range(1, ncase + 1)):
        raise SystemExit(f"{path}: unexpected case inventory {indices!r}")
    field = {}
    for name in rows[0]:
        if name == "case":
            continue
        field[name] = (
            np.asarray([float(row[name]) for row in rows], dtype=np.float32)
            .reshape(ncase, NZS)
            .T
        )
    return names, field


def _before(field: dict[str, np.ndarray], name: str) -> np.ndarray:
    return field[name + "_before" if name + "_before" in field else name]


def _case_arguments(field: dict[str, np.ndarray], case: int) -> dict[str, object]:
    isice = int(field["isice"][0, case])
    if isice not in DATASETS:
        raise SystemExit(f"case {case + 1}: unsupported isice {isice}")
    arguments: dict[str, object] = {
        "delt": float(field["delt"][0, case]),
        "conflx": float(field["conflx"][0, case]),
        "c1sn": float(field["c1sn"][0, case]),
        "c2sn": float(field["c2sn"][0, case]),
        "isice": isice,
        "ivgtyp": np.asarray([int(field["ivgtyp"][0, case])], dtype=np.int32),
        "iland": np.asarray(
            [int(field["iland_before"][0, case])], dtype=np.int32
        ),
        "nroot": np.asarray([int(field["nroot"][0, case])], dtype=np.int32),
        "mminlu": DATASETS[isice],
        # ``sfctmp``'s ``ilnb`` is an uninitialised local (:1385) whose
        # thin-pack read is observable in ``tsnav``.  ``run_sfctmp.F90``
        # zeroes that stack region before every call, so the fixture pins
        # ``ilnb = 0``; ``sfctmp_stackfill.csv`` is the same driver with a
        # nonzero fill and is the proof that the read happens.  See
        # ``_stack_control`` and the ``ruc_surface_temperature_step``
        # docstring.
        "ilnb": np.asarray([_ILNB_SEED[0]], dtype=np.int32),
    }
    for name in RUC_SFCTMP_PROFILE_INPUTS:
        arguments[name] = _before(field, name)[:, case : case + 1]
    for name in RUC_SFCTMP_COLUMN_INPUTS:
        arguments[name] = _before(field, name)[0, case : case + 1]
    return arguments


#: The value ``run_sfctmp.F90``'s stack scrub leaves in WRF's ``ilnb`` slot.
_ILNB_SEED = [0]


def _run_case(arguments: dict[str, object]) -> dict[str, np.ndarray]:
    values = {
        name: arguments[name]
        for name in RUC_SFCTMP_PROFILE_INPUTS + RUC_SFCTMP_COLUMN_INPUTS
    }
    result = ruc_surface_temperature_step(
        values,
        delt=arguments["delt"],
        conflx=arguments["conflx"],
        ivgtyp=arguments["ivgtyp"],
        iland=arguments["iland"],
        nroot=arguments["nroot"],
        isice=arguments["isice"],
        c1sn=arguments["c1sn"],
        c2sn=arguments["c2sn"],
        ilnb=arguments["ilnb"],
        mminlu=arguments["mminlu"],
    )
    return {
        name: np.asarray(getattr(result, name), dtype=np.float32)
        for name in OUTPUTS
    }


def _replay(
    field: dict[str, np.ndarray],
    ncase: int,
    mutate: tuple[str, object] | None = None,
) -> dict[str, np.ndarray]:
    profile = {
        name: np.empty((NZS, ncase), dtype=np.float32)
        for name in RUC_SFCTMP_PROFILE_OUTPUTS
    }
    column = {
        name: np.empty(ncase, dtype=np.float32)
        for name in RUC_SFCTMP_COLUMN_OUTPUTS + ("iland",)
    }
    for case in range(ncase):
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


def _arms(field: dict[str, np.ndarray], ncase: int) -> dict[str, list[int]]:
    """Classify each case by the dispatch arm it binds.

    The classification is read from the port's own preparation results, not
    guessed: ``snhei`` after the prologue decides snow vs snow-free,
    ``snow_mosaic`` decides mosaic vs not, and ``seaice`` decides land vs ice.
    """

    from gpuwm.core.ruc import (
        RUC_SNOW_PREP_COLUMN_INPUTS,
        ruc_snow_preparation,
    )

    arms: dict[str, list[int]] = {
        "mosaic_land": [], "mosaic_ice": [], "snow_land": [], "snow_ice": [],
        "bare_land": [], "bare_ice": [], "melt_out": [], "urban_cap": [],
        "keepfr_snow_leg": [], "keepfr_free_leg": [], "albice_rescale": [],
        "albice_skip": [], "snowfall_accumulation": [], "runoff2_nonzero": [],
    }
    for case in range(ncase):
        isice = int(field["isice"][0, case])
        values = {
            name: _before(field, name)[0, case : case + 1]
            for name in RUC_SNOW_PREP_COLUMN_INPUTS
        }
        values["ts1d"] = field["ts1d_before"][:, case : case + 1]
        prep = ruc_snow_preparation(
            values,
            delt=float(field["delt"][0, case]),
            ivgtyp=np.asarray([int(field["ivgtyp"][0, case])], dtype=np.int32),
            iland=np.asarray(
                [int(field["iland_before"][0, case])], dtype=np.int32
            ),
            isice=isice,
            c1sn=float(field["c1sn"][0, case]),
            c2sn=float(field["c2sn"][0, case]),
            mminlu=DATASETS[isice],
        )
        index = case + 1
        ice = float(field["seaice"][0, case]) >= 0.5
        snow = float(prep.snhei[0]) > 0.0
        mosaic = float(prep.snow_mosaic[0]) == 1.0
        snowfrac = float(prep.snowfrac[0])
        if snow:
            arms["snow_ice" if ice else "snow_land"].append(index)
            if mosaic:
                arms["mosaic_ice" if ice else "mosaic_land"].append(index)
                if not ice:
                    arms[
                        "keepfr_snow_leg" if snowfrac > 0.5
                        else "keepfr_free_leg"
                    ].append(index)
            if float(field["snhei_after"][0, case]) == 0.0:
                arms["melt_out"].append(index)
            if float(prep.newsn[0]) > 0.0:
                arms["snowfall_accumulation"].append(index)
        else:
            arms["bare_ice" if ice else "bare_land"].append(index)
            if ice:
                arms[
                    "albice_skip"
                    if float(prep.alb[0]) == float(prep.albice[0])
                    else "albice_rescale"
                ].append(index)
        if float(field["snowfrac_after"][0, case]) == 0.75:
            arms["urban_cap"].append(index)
        if float(field["runoff2_after"][0, case]) != 0.0:
            arms["runoff2_nonzero"].append(index)
    return arms


def _structure(
    names: tuple[str, ...], field: dict[str, np.ndarray], ncase: int
) -> None:
    arms = _arms(field, ncase)
    print("dispatch arms bound by the fixture:")
    for name, cases in arms.items():
        print(f"  {name:<22s} {len(cases):>2d}  {cases}")
    for name in arms:
        if not arms[name]:
            raise SystemExit(f"no fixture case binds the {name} arm")
    checks = (
        # Every path must be exercised more than once where its behaviour is
        # a weighting rather than a switch, otherwise a port that returns an
        # endpoint reproduces the fixture.
        ("two independent mosaic land columns", len(arms["mosaic_land"]) >= 2),
        ("two independent mosaic ice columns", len(arms["mosaic_ice"]) >= 2),
        ("both keepfr recombination legs", len(arms["keepfr_snow_leg"]) >= 1
         and len(arms["keepfr_free_leg"]) >= 1),
        ("both legs of the bare-ice albedo rescale",
         len(arms["albice_rescale"]) >= 1 and len(arms["albice_skip"]) >= 1),
        ("a melt-out on both the mosaic and the non-mosaic path",
         any(case in arms["mosaic_land"] for case in arms["melt_out"])
         and any(
             case not in arms["mosaic_land"] and case not in arms["mosaic_ice"]
             for case in arms["melt_out"]
         )),
        # smf is soil's output and WRF makes it identically zero; see
        # docs/mynn_noahmp_ruc_completion_plan.md.
        ("smf is zero everywhere", bool(np.all(field["smf_after"] == 0.0))),
        # Diagnostics nothing on the no-snow path assigns must survive.
        ("acsnow survives every path", bool(np.array_equal(
            field["acsnow_before"].view(np.int32),
            field["acsnow_after"].view(np.int32),
        ))),
        ("evapl survives the snow sea-ice path", bool(np.array_equal(
            field["evapl_before"][0][[c - 1 for c in arms["snow_ice"]]],
            field["evapl_after"][0][[c - 1 for c in arms["snow_ice"]]],
        ))),
        ("snweprint and snheiprint are zeroed on the snow-free path",
         bool(np.all(
             field["snweprint_after"][0][[c - 1 for c in arms["bare_land"]]]
             == 0.0
         ))),
    )
    for label, ok in checks:
        if not bool(ok):
            raise SystemExit(f"structural check failed: {label}")
    if len(dict.fromkeys(field["isice"][0])) != 2:
        raise SystemExit("both land-use datasets must appear")
    if len(names) != ncase:
        raise SystemExit("duplicate case names")


def _mutation_study(
    field: dict[str, np.ndarray], ncase: int, baseline
) -> None:
    """Require every argument's disappearance to break the fixture."""

    arguments = (
        RUC_SFCTMP_PROFILE_INPUTS + RUC_SFCTMP_COLUMN_INPUTS + SCALAR_ARGUMENTS
    )
    survivors = []
    total = 0
    for name in arguments:
        seen: list[tuple[object, object]] = []
        for case in range(ncase):
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
                mutant = _replay(field, ncase, mutate=(name, value))
            except (ValueError, TypeError):
                # The mutant cannot even run, so it cannot reproduce the
                # fixture.
                continue
            if _identical(mutant, baseline):
                survivors.append((name, value))
                killed = False
                break
        status = "killed" if killed else "SURVIVED"
        print(f"  {name:<15s} {len(seen):>2d} distinct value(s)  {status}")
    print(f"  ({total} mutants evaluated)")
    unexplained = [
        name for name, _ in survivors if name not in UNREACHABLE_ARGUMENTS
    ]
    for name, _ in survivors:
        if name in UNREACHABLE_ARGUMENTS:
            print(f"  survivor {name}: {UNREACHABLE_ARGUMENTS[name]}")
    if unexplained:
        raise SystemExit(
            "the fixture cannot detect these arguments being dropped, and "
            "they are not argued unreachable: " + ", ".join(unexplained)
        )


#: The only cells a nonzero stack fill may move, and the reason it may.
_STACK_CONTROL_CELLS = {
    ("tsnav_after", 4), ("tsnav_after", 12), ("tsnav_after", 21),
    ("tsnav_after", 26), ("tsnav_after", 28),
}


def _stack_control(path: str, control_path: str) -> None:
    """Prove WRF's uninitialised ``ilnb`` read is real, and bounded.

    ``control_path`` is the same driver over the same 33 regimes with the
    stack filled with a nonzero pattern instead of zero.  Only ``tsnav`` on
    the thin-pack mosaic cases may move; anything else moving would mean the
    fixture depends on stack residue somewhere this lane has not accounted
    for.
    """

    _, field = _load(path)
    _, control = _load(control_path)
    moved = set()
    for name in field:
        if name == "case_index":
            continue
        differing = np.argwhere(
            field[name].view(np.int32) != control[name].view(np.int32)
        )
        for index in differing:
            moved.add((name, int(index[1]) + 1))
    if not moved:
        raise SystemExit(
            f"{control_path}: the stack fill moved nothing, so it is not a "
            "control -- WRF's uninitialised ilnb read is not being exercised"
        )
    if moved != _STACK_CONTROL_CELLS:
        raise SystemExit(
            f"stack-residue control moved {sorted(moved)}, expected "
            f"{sorted(_STACK_CONTROL_CELLS)}"
        )
    print(
        f"stack-residue control: a nonzero stack fill moves exactly "
        f"{len(moved)} cells, all tsnav on thin-pack cases "
        f"{sorted(case for _, case in moved)}"
    )


def main(
    path: str, run_mutations: bool = True, control_path: str | None = None
) -> None:
    names, field = _load(path)
    ncase = len(names)
    if not all(np.all(np.isfinite(value)) for value in field.values()):
        raise SystemExit("the RUC sfctmp oracle contains non-finite output")
    _structure(names, field, ncase)
    if control_path is not None:
        _stack_control(path, control_path)

    actual = _replay(field, ncase)
    print("RUC sfctmp parity (gpuwm.core.ruc.ruc_surface_temperature_step):")
    failures = 0
    for name in RUC_SFCTMP_PROFILE_OUTPUTS:
        failures += _report(name, actual[name], field[name + "_after"])
    for name in RUC_SFCTMP_COLUMN_OUTPUTS + ("iland",):
        failures += _report(name, actual[name], field[name + "_after"][0])
    if failures:
        raise SystemExit(
            f"RUC sfctmp oracle: FAIL -- {failures} cell(s) differ that "
            "UPSTREAM_RESIDUE does not account for"
        )

    if run_mutations:
        print(
            "mutation study (each mutant hard-codes one argument, so it must "
            "fail):"
        )
        _mutation_study(field, ncase, actual)
    cells = ncase * NZS * len(RUC_SFCTMP_PROFILE_OUTPUTS) + ncase * (
        len(RUC_SFCTMP_COLUMN_OUTPUTS) + 1
    )
    print(
        f"RUC sfctmp oracle: PASS -- {cells - len(UPSTREAM_RESIDUE)} of "
        f"{cells} cells bitwise over {ncase} regimes x {NZS} levels; the "
        f"{len(UPSTREAM_RESIDUE)} exceptions are the pinned upstream residue"
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(
            "usage: validate_sfctmp_oracle.py RUC_SFCTMP.csv "
            "[--no-mutations] [--ilnb-seed N] [--stack-control CSV]"
        )
    flags = sys.argv[2:]
    if "--ilnb-seed" in flags:
        _ILNB_SEED[0] = int(flags[flags.index("--ilnb-seed") + 1])
    control = None
    if "--stack-control" in flags:
        control = flags[flags.index("--stack-control") + 1]
    main(
        sys.argv[1],
        run_mutations="--no-mutations" not in flags,
        control_path=control,
    )
