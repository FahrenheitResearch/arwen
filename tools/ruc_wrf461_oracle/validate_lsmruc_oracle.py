#!/usr/bin/env python3
"""Replay ``oracle/lsmruc.csv`` through ``gpuwm.core.ruc.ruc_land_surface_step``.

Fails closed on any nonzero ULP outside the pinned upstream residue, and on
any drift in the ``ilnb`` negative control.  Then it mutates arguments, in two
studies that answer two different questions.

**The control is the port's own output, not the fixture.**  This file used to
score a mutant as *killed* when its output was not bitwise-equal to
``oracle/lsmruc.csv``.  That is vacuous here: the unmutated port already
differs from the fixture in the 26 cells of :data:`UPSTREAM_RESIDUE`, so the
NULL mutant -- the port with nothing changed -- was itself "killed", and every
mutant scored as detected whether or not the fixture could see it.  Replaying
against the correct control turns 218 mutants / 0 survivors into 218 mutants /
30 survivors.  :func:`_null_mutant` is the assertion whose absence let that
ship, and it now runs first.

**Study 1, constant substitution.**  Every driver argument is hard-coded to
each distinct value the fixture gives it.  This asks "could the port replace
its read of this argument with this constant and still reproduce the fixture?"

**Study 2, the permutation control.**  Every argument the first study could not
kill is re-run with each case taking *another case's* value.  This asks the
different and sharper question "can the fixture see this argument AT ALL?"  A
constant survivor has exactly two possible causes, and only one of them is
unreachability:

* the port never reads the argument -- then every permutation survives too, and
  the argument belongs in :data:`UNOBSERVABLE_ARGUMENTS` with the WRF line that
  writes it before any read;
* the port does read it, and the surviving constant simply equals the only
  value the fixture ever lets that read see -- then some permutation kills it,
  and the substitution belongs in :data:`UNBOUND_CONSTANTS`, which is a
  *fixture gap*, not unreachability.

Both tables are exact: an unlisted survivor fails, and so does a listed entry
that no longer survives.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np

from gpuwm.core.ruc import (
    NUM_SOIL_LAYERS,
    RUC_DRIVER_COLUMN_FORCING,
    RUC_DRIVER_COLUMN_STATE,
    RUC_DRIVER_PROFILE_STATE,
    ruc_land_surface_step,
    ruc_soil_geometry,
)

NZS = NUM_SOIL_LAYERS
DATASETS = {0: "MODIFIED_IGBP_MODIS_NOAH", 1: "USGS"}

#: Column outputs the fixture records, in the CSV's order.
COLUMN_OUTPUTS = RUC_DRIVER_COLUMN_STATE
PROFILE_OUTPUTS = RUC_DRIVER_PROFILE_STATE
#: The CSV writes ``keepfr`` where the argument is ``keepfr3dflag``.
CSV_ALIAS = {"keepfr3dflag": "keepfr"}


def _constant_key(name: str, value: object) -> tuple[str, object]:
    """Table key for a substituted value, normalised through float32.

    The study's candidate values come from ``float(np.float32(...))``, so a
    readable literal such as ``1.0e-4`` has to be widened the same way before
    it can match one.
    """

    if isinstance(value, (bool, str, int)):
        return (name, value)
    return (name, float(np.float32(value)))

#: Cells that do not reproduce bitwise, with the ULP measured on the pinned
#: build.  Keyed ``(field, case_row, level)`` with ``case_row`` the 0-based
#: index into the fixture's 48 (run, step, column) groups and ``level`` the
#: 1-based soil level (0 for column fields).  A listed cell may only shrink;
#: an unlisted nonzero cell fails.
#:
#: NONE of these originates in the driver.  25 of the 26 are one function:
#: ``gpuwm.core.ruc._f32_tanh``, used by ``ruc_snow_preparation`` for WRF's
#: ``:1520-1521`` new-snow density.  ``_f32_tanh`` transcribes fdlibm's
#: ``tanhf``, but glibc 2.39's ``tanhf`` is NOT fdlibm's -- reconstructing the
#: fdlibm form from glibc's own ``expm1f`` gives 0.760541856 where glibc
#: ``tanhf`` returns 0.760541916 at ``x = 0.9974991083145142``.  Over the 80
#: ``tanh`` arguments this fixture reaches, ``_f32_tanh`` misses glibc on
#: four, by up to 3 ULP.  Substituting a measured glibc ``tanhf`` table for
#: ``_f32_tanh`` and replaying collapses this table from 26 cells / 425 ULP
#: to one cell / 1 ULP -- ``("grdflx", 21, 0)``, which is the separately
#: documented ``exp``/``pow``/``log10`` class in
#: ``_RUC_PROVISIONAL_TRANSCENDENTALS``.
#:
#: This is a defect in ``ruc_snow_preparation``, not in the driver, and it is
#: pinned rather than fixed here because fixing it needs a verified glibc
#: ``tanhf`` transcription, its CUDA twin in ``gpuwm/core/kernels/ruc.cu``,
#: and a re-verification of ``oracle/sfctmp_prep.csv`` and ``oracle/sfctmp.csv``
#: -- none of which is this lane.  ``oracle/lsmruc.csv`` is the first RUC
#: fixture to reach an argument where the two ``tanh`` implementations
#: disagree, which is why the snow lanes were bitwise without it.
UPSTREAM_RESIDUE: dict[tuple[str, int, int], int] = {
    # (2, 1, 'usgs_grass_allsnow') and (2, 2, ...): fresh snow at tabs = 270 K,
    # where 17*tanh((276.65-tabs)*0.15) lands on the 3 ULP tanh miss.
    ("rhosnf", 8, 0): 2,
    ("rhosnf", 20, 0): 2,
    ("rhosnf", 25, 0): 2,
    ("rhosnf", 37, 0): 2,
    ("snowfallac", 8, 0): 2,
    ("snowfallac", 20, 0): 1,
    ("snowfallac", 25, 0): 2,
    ("snowfallac", 37, 0): 1,
    ("snowh", 25, 0): 1,
    ("snowh", 37, 0): 1,
    ("snowc", 25, 0): 2,
    ("snowc", 30, 0): 1,
    ("snowc", 37, 0): 2,
    ("qvg", 25, 0): 1,
    ("qsg", 25, 0): 1,
    ("qsfc", 25, 0): 1,
    ("soilt1", 37, 0): 1,
    ("tsnav", 37, 0): 256,
    ("mavail", 39, 0): 1,
    ("grdflx", 37, 0): 425,
    ("sh2o", 25, 2): 1,
    ("sh2o", 37, 1): 27,
    ("sh2o", 37, 2): 23,
    ("tso", 37, 1): 1,
    ("tso", 37, 2): 1,
    # The one cell the glibc-tanhf substitution does NOT remove.
    ("grdflx", 21, 0): 1,
}


#: Driver arguments whose ENTRY value no supported call can observe, with the
#: ``phys/module_sf_ruclsm.F`` line that makes it so.  Every constant mutant of
#: these survives, and so does every permutation -- :func:`_permutation_control`
#: checks the second half, so these are measured claims, not assertions.
#:
#: "Every path" below means both arms of the ``:828`` water / land test, which
#: is total: the water arm ends at ``:855`` and falls to ``:1168``, the land arm
#: (sea ice included -- ``:866-895`` falls through, it does not exit) runs to
#: ``:1167``.  ``EM_CORE==1``'s ``goto 2999`` at ``:824`` is not in the pinned
#: object.
UNOBSERVABLE_ARGUMENTS: dict[str, str] = {
    "chklowq":
        "written before every read on every path -- :547 in the ktau==1 block, "
        ":843 on the water arm, and the total if/else at :1070/:1072 on the "
        "land arm.  Its one read, :1076, is inside "
        "if(wrf_at_debug_level(lsmruc_dbg_lvl)) and only prints",
    "chs":
        "read for VALUE at :681 and :682 only, both inside if(myj).  "
        "ruc_land_surface_step rejects myj=True, as ruc_soil_step and "
        "ruc_snow_soil_step already do, so no supported call reaches either "
        "line; test_lsmruc_rejects_the_configurations_its_leaves_do_not_support "
        "pins the gate.  It is also PRINTED at :593, inside "
        "if(wrf_at_debug_level(lsmruc_dbg_lvl)) -- the same class of read this "
        "table already discloses for chklowq at :1076, and equally invisible "
        "to a numerical output",
    "precipfr":
        "assigned unconditionally at :678, ABOVE the :828 water test, so every "
        "column overwrites it before the loop can branch (:525 also zeroes it "
        "at ktau==1).  LSMRUC never reads it -- it is declared intent(out) at "
        ":346 and exists to hand the time-step frozen precipitation to "
        "module_diagnostics",
    "qsfc":
        "written before every read on every path -- :521 at ktau==1, :842 on "
        "the water arm, :886 on the sea-ice leg and :1065 on the land arm, all "
        "of them qvg/(1+qvg).  No line of LSMRUC reads it",
    "smavail":
        "written before every read on every path -- :830 on the water arm, "
        ":1024 unconditionally on the land arm.  Its three reads (:1133, :1138, "
        ":1151) are all in the water-budget block, and wb is a scalar local, "
        "waterbudget and acwaterbudget are declared at :330-334 WITHOUT intent "
        "-- LSMRUC locals that never leave the routine -- and :1151 only prints",
    "smmax":
        "written before every read on every path -- :831 on the water arm, "
        ":1025 unconditionally on the land arm.  LSMRUC never reads it",
}

#: Constant substitutions that survive because they coincide with the only
#: value the fixture lets a LIVE read see.  These are NOT unreachable: the
#: permutation control kills every argument named here, which is the negative
#: control that separates them from :data:`UNOBSERVABLE_ARGUMENTS`.  Each entry
#: says where WRF reads the argument, why the fixture pins that read to one
#: value, and the case that would bind it.
#:
#: Seven of the eight are the same shape.  The ``:828`` water arm returns
#: without touching most of the driver's accumulators and diagnostics, so on a
#: water column they pass entry straight to exit -- and ``run_lsmruc.F90:352-363``
#: seeds every one of them to 0 on both of the fixture's water columns (cases 4
#: and 28), so the constant 0 is invisible there.
UNBOUND_CONSTANTS: dict[tuple[str, object], str] = {
    ("lh", 0.0):
        "live: the :973 SFCTMP call assigns lh on the land arm only, so a "
        "water column passes entry to exit.  Both fixture water columns enter "
        "with lh=0 at both steps -- run_lsmruc.F90:363 seeds 0 and the water "
        "arm hands it back -- so the constant 0 cannot move them.  Bound by a "
        "water column entering with lh != 0",
    ("qfx", 0.0):
        "live: :972 assigns qfx from SFCTMP's eeta on the land arm only, and "
        ":1095/:1116 read it after.  Same water pass-through as lh, same "
        "run_lsmruc.F90:362 zero seed.  Bound by a water column entering with "
        "qfx != 0",
    ("sfcexc", 0.0):
        "live: :1062 (sfcexc = tkms) is the only write and it is on the land "
        "arm, so a water column passes entry to exit.  run_lsmruc.F90:360 "
        "seeds 0.  Bound by a water column entering with sfcexc != 0",
    ("z0", 0.0):
        "live: :1061 (z0 = znt) is the only write and it is on the land arm, "
        "so a water column passes entry to exit.  run_lsmruc.F90:353 seeds 0, "
        "and because :1061 never runs there the water columns are still 0 at "
        "step 2.  Bound by a water column entering with z0 != 0",
    ("udrunoff", 0.0):
        "live: :1041 is udrunoff = udrunoff + runoff2*dt*1000, an accumulator "
        "that reads its own entry.  The fixture never binds it: :534 zeroes "
        "udrunoff at ktau==1, discarding run_lsmruc.F90:370's 1.0 seed, and "
        "runoff2 is EXACTLY 0 in all 48 columns (no column drains past "
        "zsmain(nzs)), so step 2 also enters with 0 and the accumulation is "
        "only ever 0+0.  Dropping the '+ udrunoff' would go unnoticed.  Bound "
        "by any column with runoff2 > 0.  NOTE: PROVENANCE.md's claim that "
        "step 2 pins udrunoff 'with a nonzero incoming value' is false, and is "
        "corrected there",
    ("rhosnf", -1000.0):
        "live: :695 reads it into rhosnfall, which SFCTMP overwrites only when "
        "newsn > 0, and :1113 writes it back -- so a column with no new snow "
        "passes entry to exit.  The fixture cannot show it: rainbl and frzfrac "
        "are the same at both steps, so every column whose entry rhosnf is a "
        "real density (13, 14, 17, 20, 37, 38, 39, 41, 42, 44, 47) gets new "
        "snow again and has it overwritten, and every column that does pass it "
        "through entered at the -1.e3 flag :526 wrote.  Bound by a third step, "
        "or by any column whose snowfall stops after a snowy step",
    ("znt", 1.0e-4):
        "live: SOILVEGIN reads the caller's znt at :6852 -- 'do not overwrite "
        "z0 over water with the table value' -- and :6911 hands znttoday back, "
        "so at a column with ivgtyp == iswater the entry value survives into "
        "znt, z0 (:1061) and the :955 SFCTMP call.  The fixture defeats it "
        "because RUCLSMINIT:7088 seeds znt = z0tbl(ivgtyp) for EVERY column, "
        "water included, and z0tbl is 1.e-4 at MODI-RUC 17 and USGS-RUC 16 -- "
        "so the one value the live read can see is 1.e-4.  In a real run the "
        "surface-layer scheme owns znt over water (that is what :6852 is for) "
        "and it is not 1.e-4.  Bound by a water-category column entering with "
        "znt != z0tbl(ivgtyp), which is any restart",
    ("iswater", 17):
        "live: it selects SOILVEGIN's :6840 arm, so it decides whether a "
        "column keeps the caller's znt (:6852) or takes z0tbl (:6848), and "
        "whether lai is laitbl or laitbl - deltalai*factor (:6842 vs :6851).  "
        "iswater := 16 IS killed, by MODI-RUC case 7/19 (ivgtyp=16), whose lai "
        "moves 0.75 -> 0.39 because run 1 has rdlai2d=.false.  17 survives "
        "because only run 2 changes, and run 2 has rdlai2d=.true., which "
        "freezes lai and leaves znt as the only channel -- and RUCLSMINIT:7088 "
        "makes the fixture's znt equal z0tbl(ivgtyp) at every column, which is "
        "exactly what the non-water arm assigns.  Bound by a USGS run with "
        "rdlai2d=.false., or by the same znt case as above",
}
UNBOUND_CONSTANTS = {
    _constant_key(name, value): reason
    for (name, value), reason in UNBOUND_CONSTANTS.items()
}


def _load(path: str) -> tuple[list[tuple[int, int, str]], dict[str, np.ndarray]]:
    with Path(path).open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    groups = tuple(
        dict.fromkeys(
            (int(row["run"]), int(row["step"]), row["case"]) for row in rows
        )
    )
    ncase = len(groups)
    if len(rows) != ncase * NZS:
        raise SystemExit(f"{path}: expected {ncase * NZS} rows, got {len(rows)}")
    field: dict[str, np.ndarray] = {}
    for name in rows[0]:
        if name == "case":
            continue
        raw = [row[name] for row in rows]
        if raw[0] in ("T", "F"):
            values = np.asarray([v == "T" for v in raw], dtype=bool)
        else:
            values = np.asarray([float(v) for v in raw], dtype=np.float32)
        field[name] = values.reshape(ncase, NZS).T
    return list(groups), field


def _entry(field: dict[str, np.ndarray], name: str) -> np.ndarray:
    key = CSV_ALIAS.get(name, name)
    return field[key + "_i"]


def _result(field: dict[str, np.ndarray], name: str) -> np.ndarray:
    return field[CSV_ALIAS.get(name, name)]


def _call_arguments(
    field: dict[str, np.ndarray], case: int
) -> dict[str, object]:
    lutype = int(field["lutype"][0, case])
    if lutype not in DATASETS:
        raise SystemExit(f"case {case}: unsupported lutype {lutype}")
    return {
        "dt": float(field["dt"][0, case]),
        "ktau": int(field["ktau"][0, case]),
        "myj": bool(field["myj"][0, case]),
        "em_core": 0,
        "frpcpn": bool(field["frpcpn"][0, case]),
        "rdlai2d": bool(field["rdlai2d"][0, case]),
        "mosaic_lu": int(field["mosaic_lu"][0, case]),
        "mosaic_soil": int(field["mosaic_soil"][0, case]),
        "iswater": int(field["iswater"][0, case]),
        "isice": int(field["isice"][0, case]),
        "xice_threshold": float(field["xice_threshold"][0, case]),
        "mminlu": DATASETS[lutype],
        # ``run_lsmruc.F90``'s stack scrub leaves 0 in SFCTMP's uninitialised
        # ``ilnb`` slot for the first column of each call; the chain from
        # there is the driver's own.  ``ilnb <= 1`` selects the same arm.
        "ilnb": _ILNB_SEED[0],
    }


#: The value ``run_lsmruc.F90``'s stack scrub leaves in WRF's ``ilnb`` slot.
_ILNB_SEED = [0]


def _entry_value(field: dict[str, np.ndarray], name: str, case: int) -> float:
    """The value the fixture hands ``name`` on entry to ``case``."""

    key = CSV_ALIAS.get(name, name)
    source = field[key + "_i"] if key + "_i" in field else field[key]
    return float(source[0, case])


def _replay(
    field: dict[str, np.ndarray],
    ncase: int,
    mutate: tuple[str, object] | None = None,
    permute: tuple[str, int] | None = None,
) -> dict[str, np.ndarray]:
    """Re-run every (run, step) call of the fixture through the port.

    Each call is replayed from its own recorded entry state, so a step-2 row
    is not contaminated by any residue in the step-1 replay.

    ``mutate`` hard-codes one argument to one constant for every case.
    ``permute`` instead gives case ``i`` the value case ``(i + shift) % ncase``
    carries, leaving every other argument alone; ``shift=0`` is the identity
    and must reproduce the unmutated replay.
    """

    zs = np.asarray(
        [float(field["zs"][k, 0]) for k in range(NZS)], dtype=np.float32
    )
    pinned, _ = ruc_soil_geometry()
    if not np.array_equal(zs, pinned):
        raise SystemExit("fixture zs is not the pinned RUC grid")

    column_out = {
        name: np.empty(ncase, dtype=np.float32) for name in COLUMN_OUTPUTS
    }
    profile_out = {
        name: np.empty((NZS, ncase), dtype=np.float32)
        for name in PROFILE_OUTPUTS
    }
    for case in range(ncase):
        values: dict[str, object] = {}
        for name in RUC_DRIVER_PROFILE_STATE:
            values[name] = _entry(field, name)[:, case : case + 1]
        for name in RUC_DRIVER_COLUMN_STATE:
            values[name] = _entry(field, name)[0, case : case + 1]
        for name in RUC_DRIVER_COLUMN_FORCING:
            values[name] = field[name][0, case : case + 1]
        arguments = _call_arguments(field, case)
        arguments["zs"] = zs
        arguments["ivgtyp"] = np.asarray(
            [int(field["ivgtyp"][0, case])], dtype=np.int32
        )
        arguments["isltyp"] = np.asarray(
            [int(field["isltyp"][0, case])], dtype=np.int32
        )
        if mutate is not None:
            if mutate[0] in values:
                values[mutate[0]] = np.asarray(
                    [mutate[1]], dtype=np.float32
                )
            else:
                arguments[mutate[0]] = mutate[1]
        if permute is not None:
            name, shift = permute
            donor = (case + shift) % ncase
            if name in values:
                values[name] = np.asarray(
                    [_entry_value(field, name, donor)], dtype=np.float32
                )
            else:
                arguments[name] = _call_arguments(field, donor)[name]
        result = ruc_land_surface_step(values, **arguments)
        for name in COLUMN_OUTPUTS:
            column_out[name][case] = np.asarray(getattr(result, name))[0]
        for name in PROFILE_OUTPUTS:
            profile_out[name][:, case] = np.asarray(getattr(result, name))[:, 0]
    return {**column_out, **profile_out}


def _ulp(reference: np.ndarray, produced: np.ndarray) -> np.ndarray:
    a = reference.astype(np.float32).view(np.int32).astype(np.int64)
    b = produced.astype(np.float32).view(np.int32).astype(np.int64)
    a = np.where(a < 0, np.int64(np.iinfo(np.int32).min) - a, a)
    b = np.where(b < 0, np.int64(np.iinfo(np.int32).min) - b, b)
    return np.abs(a - b)


def _compare(
    name: str, reference: np.ndarray, produced: np.ndarray
) -> tuple[int, int]:
    ulp = _ulp(reference, produced)
    unexplained = 0
    worst = int(np.max(ulp)) if ulp.size else 0
    for index in np.argwhere(ulp > 0):
        if ulp.ndim == 1:
            key = (name, int(index[0]), 0)
        else:
            key = (name, int(index[1]), int(index[0]) + 1)
        allowed = UPSTREAM_RESIDUE.get(key)
        measured = int(ulp[tuple(index)])
        if allowed is None or measured > allowed:
            unexplained += 1
            print(
                f"    UNEXPLAINED {key} measured {measured} ULP, "
                f"pinned {allowed}"
            )
    return unexplained, worst


def _stack_control(reference_path: str, control_path: str) -> int:
    """The fixture pair proves the uninitialised ``ilnb`` read is real."""

    with Path(reference_path).open(newline="", encoding="ascii") as stream:
        base = list(csv.DictReader(stream))
    with Path(control_path).open(newline="", encoding="ascii") as stream:
        fill = list(csv.DictReader(stream))
    if len(base) != len(fill):
        raise SystemExit("stack control has a different row count")
    moved: dict[str, int] = {}
    for left, right in zip(base, fill):
        for key in left:
            if left[key] != right[key]:
                moved[key] = moved.get(key, 0) + 1
    expected = {"tsnav", "tsnav_i"}
    if set(moved) != expected:
        print(f"  stack control moved {sorted(moved)}, expected {sorted(expected)}")
        return 1
    print(
        "  ilnb negative control: nonzero stack fill moves only "
        f"{sorted(moved)} ({moved})"
    )
    return 0


def _identical(
    left: dict[str, np.ndarray], right: dict[str, np.ndarray]
) -> bool:
    """Bitwise equality over every recorded output."""

    return all(
        np.array_equal(
            left[key].astype(np.float32).view(np.int32),
            right[key].astype(np.float32).view(np.int32),
        )
        for key in left
    )


def _fixture_state(field: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The exit state ``oracle/lsmruc.csv`` records, as a comparable dict."""

    state: dict[str, np.ndarray] = {}
    for name in COLUMN_OUTPUTS:
        state[name] = _result(field, name)[0]
    for name in PROFILE_OUTPUTS:
        state[name] = _result(field, name)
    return state


def _null_mutant(
    field: dict[str, np.ndarray],
    ncase: int,
    reference: dict[str, np.ndarray],
) -> int:
    """The unmutated port must SURVIVE.  Absent, this file shipped vacuous.

    Two checks.  The identity permutation routes every argument the studies
    below touch through the same mutation plumbing while changing nothing, so
    it exercises the machinery, not just the comparison.  Then the control
    itself is audited: if the port is not bitwise against the fixture -- and it
    is not, that is :data:`UPSTREAM_RESIDUE` -- then scoring against the
    fixture would kill this very mutant, and every study below would be free.
    """

    failures = 0
    names = sorted(
        set(UNOBSERVABLE_ARGUMENTS) | {name for name, _ in UNBOUND_CONSTANTS}
    )
    for name in names:
        if not _identical(reference, _replay(field, ncase, permute=(name, 0))):
            print(
                f"    NULL MUTANT KILLED: the identity permutation of {name} "
                "does not reproduce the unmutated port, so the mutation "
                "plumbing is not an identity and no survivor count below means "
                "anything"
            )
            failures += 1
    if failures == 0:
        print(
            f"  null mutant: the identity permutation of all {len(names)} "
            "studied arguments survives"
        )

    fixture = _fixture_state(field)
    moved = sum(
        int(np.count_nonzero(_ulp(fixture[key], reference[key])))
        for key in reference
    )
    if moved:
        print(
            f"  control audit: the port differs from the fixture in {moved} "
            "cells (the pinned upstream residue), so the FIXTURE IS NOT A "
            "VALID CONTROL -- against it the null mutant scores as killed and "
            "every mutant below would be 'detected' for free.  The control is "
            "the port's own output."
        )
    else:
        print(
            "  control audit: the port is bitwise against the fixture, so the "
            "two controls coincide; the port's own output is still used."
        )
    return failures


def _permutation_control(
    field: dict[str, np.ndarray],
    ncase: int,
    reference: dict[str, np.ndarray],
    names: list[str],
    shifts: tuple[int, ...] = (1, 12, 24),
) -> dict[str, tuple[int, ...]]:
    """Can the fixture see the argument AT ALL?

    Each case takes another case's value for one argument and its own for
    everything else, so a live read moves wherever the two cases differ.  The
    shifts are the fixture's own structure: 1 crosses the column-type boundary
    inside a call (twelve columns per call, and only one of them is water), 12
    crosses the timestep, 24 crosses the run.  A single shift is a lower bound
    on liveness -- surviving one shift proves nothing, surviving all three is
    the evidence :data:`UNOBSERVABLE_ARGUMENTS` rests on.
    """

    killed: dict[str, tuple[int, ...]] = {}
    for name in names:
        hits: list[int] = []
        for shift in shifts:
            try:
                permuted = _replay(field, ncase, permute=(name, shift))
            except Exception:
                # A permutation the port refuses is still a detection.
                hits.append(shift)
                continue
            if not _identical(reference, permuted):
                hits.append(shift)
        killed[name] = tuple(hits)
    return killed


def _mutation_study(
    field: dict[str, np.ndarray],
    ncase: int,
    reference: dict[str, np.ndarray],
) -> list[tuple[str, object]]:
    """Hard-code each argument to each value it takes; all mutants must fail."""

    scalars = (
        "dt",
        "ktau",
        "myj",
        "frpcpn",
        "rdlai2d",
        "iswater",
        "isice",
        "xice_threshold",
        "mminlu",
    )
    candidates: dict[str, list[object]] = {}
    for name in scalars:
        if name == "mminlu":
            observed = sorted(
                {
                    DATASETS[int(field["lutype"][0, case])]
                    for case in range(ncase)
                }
            )
        elif name in ("myj", "frpcpn", "rdlai2d"):
            observed = sorted({bool(field[name][0, case]) for case in range(ncase)})
        elif name in ("ktau", "iswater", "isice"):
            observed = sorted({int(field[name][0, case]) for case in range(ncase)})
        else:
            observed = sorted(
                {float(field[name][0, case]) for case in range(ncase)}
            )
        if len(observed) > 1:
            candidates[name] = observed
    for name in RUC_DRIVER_COLUMN_FORCING + RUC_DRIVER_COLUMN_STATE:
        key = CSV_ALIAS.get(name, name)
        source = field[key + "_i"] if key + "_i" in field else field[key]
        observed = sorted({float(source[0, case]) for case in range(ncase)})
        if len(observed) > 1:
            candidates[name] = observed[:4]

    total = 0
    survivors: list[tuple[str, object]] = []
    for name, observed in sorted(candidates.items()):
        for value in observed:
            total += 1
            try:
                produced = _replay(field, ncase, mutate=(name, value))
            except Exception:
                # A mutant the port refuses cannot reproduce the control.
                continue
            if _identical(reference, produced):
                survivors.append((name, value))
    print(
        f"  constant substitution: {total} mutants, {len(survivors)} survivors"
    )
    return survivors


def _classify(
    survivors: list[tuple[str, object]],
    field: dict[str, np.ndarray],
    ncase: int,
    reference: dict[str, np.ndarray],
) -> int:
    """Every survivor must be unreachable or a named, argued fixture gap."""

    failures = 0
    unlisted = [
        (name, value)
        for name, value in survivors
        if name not in UNOBSERVABLE_ARGUMENTS
        and _constant_key(name, value) not in UNBOUND_CONSTANTS
    ]
    for name, value in unlisted:
        print(
            f"    UNEXPLAINED SURVIVOR {name} := {value!r} -- the fixture "
            "cannot detect this substitution and nothing here says why"
        )
        failures += 1

    seen_names = {name for name, _ in survivors}
    stale_names = sorted(set(UNOBSERVABLE_ARGUMENTS) - seen_names)
    for name in stale_names:
        print(
            f"    NO LONGER A SURVIVOR {name} -- the fixture now binds it; "
            "drop it from UNOBSERVABLE_ARGUMENTS"
        )
        failures += 1
    seen_constants = {_constant_key(name, value) for name, value in survivors}
    stale_constants = sorted(
        key for key in UNBOUND_CONSTANTS if key not in seen_constants
    )
    for key in stale_constants:
        print(
            f"    NO LONGER A SURVIVOR {key[0]} := {key[1]!r} -- the fixture "
            "now binds it; drop it from UNBOUND_CONSTANTS"
        )
        failures += 1

    names = sorted(
        set(UNOBSERVABLE_ARGUMENTS) | {name for name, _ in UNBOUND_CONSTANTS}
    )
    killed = _permutation_control(field, ncase, reference, names)
    print(f"  permutation control over {len(names)} arguments:")
    for name in names:
        hits = killed[name]
        claim_unreachable = name in UNOBSERVABLE_ARGUMENTS
        if claim_unreachable and hits:
            print(
                f"    NOT UNREACHABLE {name}: the permutation control kills it "
                f"at shift {hits} -- the fixture DOES see this argument, so it "
                "is untested, not unreachable.  Move it to UNBOUND_CONSTANTS"
            )
            failures += 1
        elif not claim_unreachable and not hits:
            print(
                f"    NOT MERELY UNTESTED {name}: no permutation moves an "
                "output, so the fixture cannot see this argument at all and "
                "the UNBOUND_CONSTANTS entry overstates what it binds.  It "
                "needs an UNOBSERVABLE_ARGUMENTS argument, or the port has "
                "dropped a read it should have"
            )
            failures += 1
        else:
            state = (
                "unobservable at every shift"
                if claim_unreachable
                else f"observable, killed at shift {hits}"
            )
            print(f"    {name:<10s} {state}")

    if not failures:
        print(
            f"  {len(survivors)} survivors accounted for: "
            f"{sum(1 for n, _ in survivors if n in UNOBSERVABLE_ARGUMENTS)} on "
            f"{len(UNOBSERVABLE_ARGUMENTS)} unreachable arguments, "
            f"{len(UNBOUND_CONSTANTS)} argued fixture gaps"
        )
        for name in sorted(UNOBSERVABLE_ARGUMENTS):
            print(f"    unreachable  {name}: {UNOBSERVABLE_ARGUMENTS[name]}")
        for key in sorted(UNBOUND_CONSTANTS):
            print(
                f"    fixture gap  {key[0]} := {key[1]!r}: "
                f"{UNBOUND_CONSTANTS[key]}"
            )
    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("oracle")
    parser.add_argument("--stack-control")
    parser.add_argument("--skip-mutation", action="store_true")
    args = parser.parse_args()

    groups, field = _load(args.oracle)
    ncase = len(groups)
    print(f"{args.oracle}: {ncase} (run, step, column) groups x {NZS} levels")

    produced = _replay(field, ncase)
    failures = 0
    worst = 0
    for name in COLUMN_OUTPUTS:
        reference = _result(field, name)[0]
        bad, high = _compare(name, reference, produced[name])
        failures += bad
        worst = max(worst, high)
    for name in PROFILE_OUTPUTS:
        reference = _result(field, name)
        bad, high = _compare(name, reference, produced[name])
        failures += bad
        worst = max(worst, high)
    print(f"  max_ulp over all recorded outputs: {worst}")
    if failures:
        print(f"  {failures} unexplained cells")

    if args.stack_control:
        failures += _stack_control(args.oracle, args.stack_control)

    if not args.skip_mutation and failures == 0:
        # THE CONTROL IS ``produced`` -- the port's own output.  Scoring
        # against ``field`` instead is what made this study vacuous; see the
        # module docstring and _null_mutant.
        failures += _null_mutant(field, ncase, produced)
        if failures == 0:
            survivors = _mutation_study(field, ncase, produced)
            failures += _classify(survivors, field, ncase, produced)

    if failures:
        raise SystemExit(f"{args.oracle}: {failures} failures")
    print("  OK")


if __name__ == "__main__":
    sys.exit(main())
