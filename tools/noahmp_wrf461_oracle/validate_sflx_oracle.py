#!/usr/bin/env python3
"""Structural, conservation and branch checks for the unmodified-WRF
NOAHMP_SFLX oracle under the first admitted Noah-MP option identity.

No gpuwm comparison happens here: the Noah-MP column port does not exist yet.
This proves the fixture is well formed, closes WRF's own shortwave and energy
budgets, and that each intended regime actually took its intended branch.
"""

from __future__ import annotations

import csv
from math import isfinite
from pathlib import Path
import sys

CASES = (
    "veg_warm_day_dry",
    "veg_warm_night_rain",
    "snowpack_frozen_soil",
    "bare_thin_snow_melt",
)
NSOIL = 4
NSNOW = 3
# NOAHMP_SFLX assigns this sentinel to ALBEDO when SWDOWN is zero.
NIGHT_ALBEDO = -999.9


def fail(message: str) -> None:
    raise SystemExit(f"Noah-MP SFLX oracle: {message}")


def main(path: str) -> None:
    with Path(path).open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))

    order = tuple(dict.fromkeys(row["case"] for row in rows))
    if order != CASES:
        fail(f"unexpected case inventory {order!r}")

    state: dict[str, dict[str, dict[tuple[str, int], float]]] = {
        case: {"input": {}, "output": {}} for case in CASES
    }
    for row in rows:
        stage = row["stage"]
        if stage not in ("input", "output"):
            fail(f"unknown stage {stage!r}")
        key = (row["field"], int(row["index"]))
        bucket = state[row["case"]][stage]
        if key in bucket:
            fail(f"{row['case']}/{stage}: duplicate entry {key}")
        value = float(row["value"])
        if not isfinite(value):
            fail(f"{row['case']}/{stage}: non-finite {key}")
        bucket[key] = value

    reference = {stage: set(state[CASES[0]][stage]) for stage in ("input", "output")}
    for case in CASES[1:]:
        for stage in ("input", "output"):
            if set(state[case][stage]) != reference[stage]:
                fail(f"{case}: {stage} field inventory differs from {CASES[0]}")

    for case in CASES:
        i = state[case]["input"]
        o = state[case]["output"]

        # WRF's own shortwave budget check (module_sf_noahmplsm.F ERROR).
        swdown = i[("soldn", 0)]
        errsw = swdown - (o[("fsa", 0)] + o[("fsr", 0)])
        if abs(errsw) > 0.01:
            fail(f"{case}: shortwave budget open by {errsw} W/m2")
        if abs(o[("fsa", 0)] - (o[("sav", 0)] + o[("sag", 0)])) > 0.01:
            fail(f"{case}: FSA is not SAV + SAG")

        # WRF's own energy budget check, with FIRR = 0 (opt_irr = 0).
        erreng = (
            o[("sav", 0)] + o[("sag", 0)]
            - (o[("fira", 0)] + o[("fsh", 0)] + o[("fcev", 0)] + o[("fgev", 0)]
               + o[("fctr", 0)] + o[("ssoil", 0)] + o[("canhs", 0)])
            + o[("pah", 0)]
        )
        if abs(erreng) > 0.01:
            fail(f"{case}: energy budget open by {erreng} W/m2")

        # Precipitation partition: OPT_SNF = 1 must split the total exactly.
        prcp = (i[("prcpconv", 0)] + i[("prcpnonc", 0)] + i[("prcpshcv", 0)])
        if abs(o[("rain", 0)] + o[("snow", 0)] - prcp) > 1e-9:
            fail(f"{case}: RAIN + SNOW does not recover the forcing precipitation")
        if not 0.0 <= o[("fpice", 0)] <= 1.0:
            fail(f"{case}: FPICE outside [0, 1]")

        # Physical bounds on the state that leaves the column.
        for k in range(1, NSOIL + 1):
            smc = o[("smc", k)]
            sh2o = o[("sh2o", k)]
            if not 0.0 <= sh2o <= smc + 1e-6:
                fail(f"{case}: liquid soil water exceeds total at layer {k}")
            if not 0.0 < smc <= 1.0:
                fail(f"{case}: SMC({k}) = {smc} outside (0, 1]")
            if not 150.0 < o[("stc", k)] < 400.0:
                fail(f"{case}: soil temperature {o[('stc', k)]} K is unphysical")
            if o[("hcpct", k)] <= 0.0:
                fail(f"{case}: non-positive soil heat capacity at layer {k}")
        for name in ("tg", "tv", "trad", "tgb", "tgv", "t2mb"):
            if not 150.0 < o[(name, 0)] < 400.0:
                fail(f"{case}: {name} = {o[(name, 0)]} K is unphysical")
        # WRF only runs VEGE_FLUX where FVEG > 0; its 2 m diagnostics and the
        # canopy fluxes stay at their zero initialisation otherwise.
        if o[("fveg", 0)] > 0.0:
            if not 150.0 < o[("t2mv", 0)] < 400.0:
                fail(f"{case}: t2mv = {o[('t2mv', 0)]} K is unphysical")
        else:
            for name in ("t2mv", "q2v", "shc", "irc", "evc", "tr", "chleaf",
                         "chuc", "laisun", "laisha", "rb"):
                if o[(name, 0)] != 0.0:
                    fail(f"{case}: FVEG=0 must leave {name} at zero,"
                         f" got {o[(name, 0)]}")
        if not 0.0 <= o[("fsno", 0)] <= 1.0:
            fail(f"{case}: FSNO outside [0, 1]")
        if not 0.0 <= o[("fveg", 0)] <= 1.0:
            fail(f"{case}: FVEG outside [0, 1]")
        if not 0.0 < o[("emissi", 0)] <= 1.0:
            fail(f"{case}: EMISSI outside (0, 1]")
        if o[("sneqv", 0)] < 0.0 or o[("snowh", 0)] < 0.0:
            fail(f"{case}: negative snow mass or depth")
        if o[("qmelt", 0)] < 0.0 or o[("ponding", 0)] < 0.0:
            fail(f"{case}: negative melt or ponding")
        if o[("z0wrf", 0)] <= 0.0:
            fail(f"{case}: non-positive coupled roughness length")
        if swdown > 0.0 and not 0.0 < o[("albedo", 0)] < 1.0:
            fail(f"{case}: daytime albedo {o[('albedo', 0)]} outside (0, 1)")

        # DVEG = 4 keeps the carbon pools prognostically inert.
        for name in ("nee", "gpp", "npp"):
            if o[(name, 0)] != 0.0:
                fail(f"{case}: DVEG=4 must leave {name} at zero, got {o[(name, 0)]}")
        for name in ("lfmass", "rtmass", "stmass", "wood", "stblcp", "fastcp"):
            if o[(name, 0)] != i[(name, 0)]:
                fail(f"{case}: DVEG=4 must not evolve the carbon pool {name}")

    # ---- regime discrimination -------------------------------------------
    day = state["veg_warm_day_dry"]
    if day["output"][("isnow", 0)] != 0 or day["output"][("fsno", 0)] != 0.0:
        fail("veg_warm_day_dry: the snow-free branch did not execute")
    if day["output"][("sav", 0)] <= 0.0 or day["output"][("sag", 0)] <= 0.0:
        fail("veg_warm_day_dry: the sunlit two-stream branch did not execute")
    if day["output"][("ssoil", 0)] <= 0.0:
        fail("veg_warm_day_dry: expected a downward ground heat flux at midday")
    if day["output"][("fctr", 0)] <= 0.0 or day["output"][("psn", 0)] <= 0.0:
        fail("veg_warm_day_dry: the Ball-Berry transpiring canopy did not execute")

    night = state["veg_warm_night_rain"]
    if night["input"][("cosz", 0)] != 0.0 or night["input"][("soldn", 0)] != 0.0:
        fail("veg_warm_night_rain: case is not nocturnal")
    if night["output"][("fsa", 0)] != 0.0 or night["output"][("fsr", 0)] != 0.0:
        fail("veg_warm_night_rain: the zero-shortwave branch did not execute")
    if abs(night["output"][("albedo", 0)] - NIGHT_ALBEDO) > 1e-3:
        fail("veg_warm_night_rain: WRF's night albedo sentinel did not execute")
    if night["output"][("rain", 0)] <= 0.0 or night["output"][("snow", 0)] != 0.0:
        fail("veg_warm_night_rain: the all-liquid precipitation branch did not execute")
    if night["output"][("qintr", 0)] <= 0.0:
        fail("veg_warm_night_rain: canopy rain interception did not execute")
    if night["output"][("qdripr", 0)] <= 0.0 or night["output"][("qthror", 0)] <= 0.0:
        fail("veg_warm_night_rain: canopy drip/throughfall did not execute")
    partition = (night["output"][("qintr", 0)] + night["output"][("qdripr", 0)]
                 + night["output"][("qthror", 0)])
    if abs(partition - night["output"][("rain", 0)]) > 1e-9:
        fail("veg_warm_night_rain: rain interception partition does not close")
    if night["output"][("runsrf", 0)] <= 0.0:
        fail("veg_warm_night_rain: OPT_RUN=3 surface runoff did not execute")

    snowpack = state["snowpack_frozen_soil"]
    if snowpack["input"][("isnow", 0)] != -2:
        fail("snowpack_frozen_soil: SNOW_INIT did not build a two-layer pack")
    if snowpack["output"][("isnow", 0)] != -2:
        fail("snowpack_frozen_soil: the layered-snow branch did not survive the step")
    if snowpack["output"][("fsno", 0)] != 1.0:
        fail("snowpack_frozen_soil: expected complete snow cover")
    if snowpack["output"][("fpice", 0)] != 1.0 or snowpack["output"][("snow", 0)] <= 0.0:
        fail("snowpack_frozen_soil: the Jordan-91 all-snow branch did not execute")
    if snowpack["output"][("sneqv", 0)] <= snowpack["input"][("sneqv", 0)]:
        fail("snowpack_frozen_soil: the pack did not accumulate")
    for k in range(-NSNOW + 1, 1):
        if k > snowpack["output"][("isnow", 0)]:
            if snowpack["output"][("snice", k)] <= 0.0:
                fail(f"snowpack_frozen_soil: snow layer {k} carries no ice")
    if snowpack["output"][("sh2o", 1)] >= snowpack["output"][("smc", 1)]:
        fail("snowpack_frozen_soil: the frozen-soil (OPT_FRZ=1) branch did not execute")
    for band in (1, 2):
        if not 0.0 < snowpack["output"][("albsnd", band)] <= 1.0:
            fail("snowpack_frozen_soil: OPT_ALB=2 direct snow albedo did not execute")
        if not 0.0 < snowpack["output"][("albsni", band)] <= 1.0:
            fail("snowpack_frozen_soil: OPT_ALB=2 diffuse snow albedo did not execute")

    melt = state["bare_thin_snow_melt"]
    if melt["input"][("isnow", 0)] != 0 or melt["input"][("sneqv", 0)] <= 0.0:
        fail("bare_thin_snow_melt: case is not sub-layer snow")
    if melt["output"][("qmelt", 0)] <= 0.0:
        fail("bare_thin_snow_melt: the snow-without-layer melt branch did not execute")
    if melt["output"][("ponding", 0)] <= 0.0:
        fail("bare_thin_snow_melt: PHASECHANGE ponding did not execute")
    if melt["output"][("sneqv", 0)] >= melt["input"][("sneqv", 0)]:
        fail("bare_thin_snow_melt: the pack did not lose mass")
    lost = melt["input"][("sneqv", 0)] - melt["output"][("sneqv", 0)]
    if abs(lost - melt["output"][("ponding", 0)]) > 1e-3:
        fail("bare_thin_snow_melt: ponding does not account for the melted mass")
    if melt["output"][("fveg", 0)] != 0.0:
        fail("bare_thin_snow_melt: the ISBARREN zero-vegetation branch did not execute")
    if melt["output"][("fctr", 0)] != 0.0 or melt["output"][("etran", 0)] != 0.0:
        fail("bare_thin_snow_melt: a bare column must not transpire")
    if not 0.0 < melt["output"][("fsno", 0)] < 1.0:
        fail("bare_thin_snow_melt: expected partial snow cover")

    fields = sum(len(state[c][s]) for c in CASES for s in ("input", "output"))
    print(f"Noah-MP SFLX oracle: PASS ({len(CASES)} regimes, {fields} recorded values)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: validate_sflx_oracle.py NOAHMP_SFLX.csv")
    main(sys.argv[1])
