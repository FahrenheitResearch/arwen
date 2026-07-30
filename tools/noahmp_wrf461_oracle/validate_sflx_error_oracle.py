#!/usr/bin/env python3
"""Structural and discrimination checks for the Noah-MP ERROR fixture.

No gpuwm comparison happens here -- ``tests/test_noahmp_sflx.py`` does that.
This proves the fixture is well formed, that its decimal and hex spellings of
every value agree, and that each case actually reached the branch it was built
to reach.  A fixture that is merely *consistent* proves nothing; one that is
consistent and discriminating is a gate.
"""

from __future__ import annotations

import argparse
import collections
import csv
import struct
from math import isfinite
from pathlib import Path

CASES = (
    "err_balanced",
    "err_gaining_inside",
    "err_losing_inside",
    "err_lake",
    "err_lake_dirty_wb",
    "err_second_substep",
    "err_flood_irrigation",
    "err_micro_irrigation",
    "err_both_irrigation",
    "err_tile_drain",
    "err_runoff_both",
    "err_snow_and_canopy",
    "err_mixed_layer_signs",
    "err_negative_pah",
    "err_sprinkler_heat",
    "err_no_soil_substep",
)
NSOIL = 4
ERRWAT_TOL = 0.1
ERRSW_TOL = 0.01
ERRENG_TOL = 0.01
#: run_sflx_compose.F90's ERRWAT poison, as the binary32 it becomes.
ERRWAT_POISON = struct.unpack('>f', struct.pack('>f', -9.0e30))[0]


def fail(message: str) -> None:
    raise SystemExit(f"Noah-MP ERROR oracle: {message}")


def _f32(bits: str) -> float:
    return struct.unpack(">f", bytes.fromhex(bits))[0]


def main(path: Path) -> None:
    rows = list(csv.DictReader(path.open(newline="")))
    if not rows:
        fail("empty fixture")

    table: dict[tuple[str, str], dict[tuple[str, int], object]] = \
        collections.defaultdict(dict)
    for row in rows:
        if row["leaf"] != "sflx_error":
            fail(f"unexpected leaf tag {row['leaf']!r}")
        key = (row["field"], int(row["index"]))
        bucket = table[(row["case"], row["stage"])]
        if key in bucket:
            fail(f"{row['case']}/{row['stage']}: duplicate {key}")
        if row["dtype"] == "int":
            bucket[key] = int(row["value"])
            continue
        if row["dtype"] != "real":
            fail(f"unknown dtype {row['dtype']!r}")
        value = _f32(row["bits"])
        # The decimal column is a convenience; the hex column is the fixture.
        # They must agree, or a reader that trusts the wrong one is silently
        # reading a different number.
        if struct.pack(">f", float(row["value"])) != bytes.fromhex(row["bits"]):
            fail(f"{row['case']}: {key} decimal and hex disagree")
        if not isfinite(value):
            fail(f"{row['case']}: non-finite {key}")
        bucket[key] = value

    order = tuple(dict.fromkeys(row["case"] for row in rows))
    if order != CASES:
        fail(f"unexpected case inventory {order!r}")

    reference = {stage: set(table[(CASES[0], stage)])
                 for stage in ("input", "output")}
    for case in CASES[1:]:
        for stage in ("input", "output"):
            if set(table[(case, stage)]) != reference[stage]:
                fail(f"{case}: {stage} inventory differs from {CASES[0]}")

    # ---- every case re-derives ERROR's own arithmetic ---------------------
    for case in CASES:
        i = table[(case, "input")]
        o = table[(case, "output")]

        errsw = i[("swdown", 0)] - (i[("fsa", 0)] + i[("fsr", 0)])
        if abs(errsw) > ERRSW_TOL:
            fail(f"{case}: reached the shortwave gate, so it cannot be a "
                 "fixture row at all")
        sink = (i[("fira", 0)] + i[("fsh", 0)] + i[("fcev", 0)]
                + i[("fgev", 0)] + i[("fctr", 0)] + i[("ssoil", 0)]
                + i[("firr", 0)] + i[("canhs", 0)])
        erreng = i[("sav", 0)] + i[("sag", 0)] - sink + i[("pah", 0)]
        if abs(erreng) > ERRENG_TOL:
            fail(f"{case}: reached the energy gate")

        if i[("ist", 0)] == 1:
            end_wb = (i[("canliq", 0)] + i[("canice", 0)] + i[("sneqv", 0)]
                      + i[("wa", 0)])
            for k in range(1, NSOIL + 1):
                end_wb += i[("smc", k)] * i[("dzsnso", k)] * 1000.0
            dt = i[("dt", 0)]
            want = {
                "acc_dwater": i[("acc_dwater", 0)] + (end_wb - i[("beg_wb", 0)]),
                "acc_prcp": i[("acc_prcp", 0)] + i[("prcp", 0)] * dt,
                "acc_ecan": i[("acc_ecan", 0)] + i[("ecan", 0)] * dt,
                "acc_etran": i[("acc_etran", 0)] + i[("etran", 0)] * dt,
                "acc_edir": i[("acc_edir", 0)] + i[("edir", 0)] * dt,
            }
            for name, expect in want.items():
                got = o[(name, 0)]
                if abs(got - expect) > 1e-3 * max(1.0, abs(expect)):
                    fail(f"{case}: {name} = {got}, expected about {expect}")
            if i[("calculate_soil", 0)] == 1:
                inner = (want["acc_prcp"] + i[("irfirate", 0)] * 1000.0
                         + i[("irmirate", 0)] * 1000.0
                         - want["acc_ecan"] - want["acc_etran"]
                         - want["acc_edir"] - i[("runsrf", 0)]
                         - i[("runsub", 0)] - i[("qtldrn", 0)])
                errwat = want["acc_dwater"] - inner
                if abs(errwat) > ERRWAT_TOL:
                    fail(f"{case}: reached the water gate, ERRWAT = {errwat}")
                if abs(o[("errwat", 0)] - errwat) > 1e-3:
                    fail(f"{case}: ERRWAT = {o[('errwat', 0)]}, expected "
                         f"about {errwat}")
        else:
            if o[("errwat", 0)] != 0.0:
                fail(f"{case}: IST /= 1 must report ERRWAT = 0")
            for name in ("acc_dwater", "acc_prcp", "acc_ecan", "acc_etran",
                         "acc_edir"):
                if o[(name, 0)] != i[(name, 0)]:
                    fail(f"{case}: IST /= 1 must leave {name} untouched")

    # ---- branch discrimination -------------------------------------------
    def out(case, name):
        return table[(case, "output")][(name, 0)]

    def inp(case, name):
        return table[(case, "input")][(name, 0)]

    if out("err_balanced", "errwat") != 0.0:
        fail("err_balanced: the reference column does not close")
    if not 0.0 < out("err_gaining_inside", "errwat") < ERRWAT_TOL:
        fail("err_gaining_inside: not a positive residual inside the gate")
    if not -ERRWAT_TOL < out("err_losing_inside", "errwat") < 0.0:
        fail("err_losing_inside: not a negative residual inside the gate")
    for case in ("err_lake", "err_lake_dirty_wb"):
        if inp(case, "ist") == 1:
            fail(f"{case}: the lake branch did not execute")
    if out("err_lake_dirty_wb", "errwat") != 0.0:
        fail("err_lake_dirty_wb: a lake column must report ERRWAT = 0 even "
             "with a water budget that would abort a soil column")
    if inp("err_second_substep", "acc_prcp") == 0.0:
        fail("err_second_substep: the accumulators did not enter live")
    if out("err_second_substep", "acc_prcp") <= inp("err_second_substep",
                                                    "acc_prcp"):
        fail("err_second_substep: the precipitation accumulator did not grow")
    if inp("err_flood_irrigation", "irfirate") <= 0.0 or \
            inp("err_flood_irrigation", "irmirate") != 0.0:
        fail("err_flood_irrigation: not a flood-only column")
    if inp("err_micro_irrigation", "irmirate") <= 0.0 or \
            inp("err_micro_irrigation", "irfirate") != 0.0:
        fail("err_micro_irrigation: not a micro-only column")
    if inp("err_both_irrigation", "irmirate") == \
            inp("err_both_irrigation", "irfirate"):
        fail("err_both_irrigation: equal rates cannot tell the two terms apart")
    if inp("err_tile_drain", "qtldrn") <= 0.0:
        fail("err_tile_drain: QTLDRN is not exercised")
    if inp("err_runoff_both", "runsrf") == inp("err_runoff_both", "runsub"):
        fail("err_runoff_both: equal runoffs cannot tell the two terms apart")
    if inp("err_snow_and_canopy", "sneqv") <= 0.0 or \
            inp("err_snow_and_canopy", "canliq") <= 0.0 or \
            inp("err_snow_and_canopy", "wa") <= 0.0:
        fail("err_snow_and_canopy: END_WB's non-soil terms are not exercised")
    layers = {table[("err_mixed_layer_signs", "input")][("smc", k)]
              for k in range(1, NSOIL + 1)}
    if len(layers) != NSOIL:
        fail("err_mixed_layer_signs: repeated SMC cannot order the layer sum")
    if inp("err_negative_pah", "pah") >= 0.0:
        fail("err_negative_pah: PAH is not negative")
    if inp("err_sprinkler_heat", "firr") <= 0.0 or \
            inp("err_sprinkler_heat", "canhs") >= 0.0:
        fail("err_sprinkler_heat: FIRR and CANHS are not both exercised")
    if inp("err_no_soil_substep", "calculate_soil") != 0:
        fail("err_no_soil_substep: calculate_soil is still true")
    if out("err_no_soil_substep", "acc_prcp") <= 0.0:
        fail("err_no_soil_substep: a non-soil substep must still accumulate")
    # The finding this case exists to record: ERRWAT is INTENT(OUT), and with
    # IST == 1 and calculate_soil false no statement in ERROR assigns it.  The
    # harness poisons it before the call, so a fixture that shows the poison is
    # showing that WRF wrote nothing -- and any other value would mean the
    # branch analysis is wrong.
    if out("err_no_soil_substep", "errwat") != ERRWAT_POISON:
        fail("err_no_soil_substep: ERRWAT was written on a path where no "
             "statement assigns it")
    for case in CASES:
        if case == "err_no_soil_substep":
            continue
        if out(case, "errwat") == ERRWAT_POISON:
            fail(f"{case}: ERRWAT left unassigned")

    values = sum(len(table[(c, s)]) for c in CASES for s in ("input", "output"))
    print(f"Noah-MP ERROR oracle: PASS ({len(CASES)} cases, "
          f"{values} recorded values)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("fixture", type=Path)
    main(ap.parse_args().fixture)
