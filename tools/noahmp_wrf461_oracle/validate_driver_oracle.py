#!/usr/bin/env python3
"""Structural validator for the Noah-MP driver cold-start fixture.

Checks the fixture that ``run_driver.F90`` emits for
``module_sf_noahmpdrv::SNOW_INIT`` and ``module_sf_noahmpdrv::NOAHMP_INIT``.
It never compares against a port: it asserts that the fixture is
self-consistent, that every branch alive under the pinned option identity is
taken by some case, that each such branch is decidable **from the inputs
alone**, and that every argument declared inert really is inert everywhere.

A fixture that cannot discriminate is worse than no fixture, so the failure
mode of every check here is "your cases are too weak", never "widen the gate".

Run standalone:

    python3 validate_driver_oracle.py --fixture noahmp-driver.csv
"""

from __future__ import annotations

import argparse
import csv
import struct
import sys
from collections import defaultdict
from pathlib import Path

HEADER = ["leaf", "case", "stage", "field", "index", "dtype", "value", "bits"]


def f32(x: float) -> float:
    """Round a Python float through IEEE binary32, as gfortran's REAL does.

    Every literal in the two routines is a default REAL, so `SNODEP <= 0.05`
    compares against binary32 0.05 (= 0.05000000074505806), not against the
    decimal 0.05.  A validator that used Python's double literals would put
    the interval edges in the wrong place and reject a correct fixture.
    """
    return struct.unpack(">f", struct.pack(">f", x))[0]

# module_sf_noahmpdrv.F 1988-1991.
HLICE = f32(3.335e5)
GRAV = f32(9.81)
T0 = f32(273.15)
FREEZE_TEST = f32(273.149)  # 2094

# NOAHMP_INIT arguments that no live statement under the pinned identity
# writes, each with the WRF line numbers that prove it.
NI_INERT = {
    "xlat": "read only by gecros_init at 2258, inside IF(iopt_crop==2)",
    "tmn": "the only write, 2080, is commented out in the pinned source",
    "croptype": "read only at 2203/2238-2240, inside IF(iopt_crop==1/2)",
    "irnumsi": "written only at 2265, inside IF(iopt_irr>=1 .and. <=3)",
    "irnummi": "written only at 2270, inside IF(iopt_irr>=1 .and. <=3)",
    "irnumfi": "written only at 2274, inside IF(iopt_irr>=1 .and. <=3)",
    "irwatsi": "written only at 2266, inside IF(iopt_irr>=1 .and. <=3)",
    "irwatmi": "written only at 2271, inside IF(iopt_irr>=1 .and. <=3)",
    "irwatfi": "written only at 2275, inside IF(iopt_irr>=1 .and. <=3)",
    "ireloss": "written only at 2267, inside IF(iopt_irr>=1 .and. <=3)",
    "irsivol": "never written by NOAHMP_INIT under any option value",
    "irmivol": "written only at 2272, inside IF(iopt_irr>=1 .and. <=3)",
    "irfivol": "written only at 2276, inside IF(iopt_irr>=1 .and. <=3)",
    "irrsplh": "written only at 2268, inside IF(iopt_irr>=1 .and. <=3)",
}

# SNOW_INIT's depth ladder, module_sf_noahmpdrv.F 2381-2405.  Expressed as the
# predicate on SNODEP and the ISNOW it must produce, so the fixture's own
# isnowxy column is checked against the source rather than trusted.
E025, E05, E10, E25, E45 = (f32(0.025), f32(0.05), f32(0.10), f32(0.25),
                            f32(0.45))
SI_LADDER = (
    (lambda d: d < E025, 0),
    (lambda d: E025 <= d <= E05, -1),
    (lambda d: E05 < d <= E10, -2),
    (lambda d: E10 < d <= E25, -2),
    (lambda d: E25 < d <= E45, -3),
    (lambda d: d > E45, -3),
)


class Fail(SystemExit):
    def __init__(self, message: str) -> None:
        super().__init__(f"validate_driver_oracle: {message}")


def load(path: Path):
    table: dict[tuple[str, str, str], dict[tuple[str, int], object]] = (
        defaultdict(dict))
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        if header != HEADER:
            raise Fail(f"header is {header}, expected {HEADER}")
        for lineno, row in enumerate(reader, start=2):
            if len(row) != len(HEADER):
                raise Fail(f"line {lineno}: {len(row)} fields, expected 8")
            leaf, case, stage, field, index, dtype, value, bits = row
            key = (field, int(index))
            slot = table[(leaf, case, stage)]
            if key in slot:
                raise Fail(f"line {lineno}: duplicate {leaf}/{case}/{stage}"
                           f"/{field}[{index}]")
            if dtype == "int":
                if bits:
                    raise Fail(f"line {lineno}: int row carries a bit pattern")
                slot[key] = int(value)
            elif dtype == "real":
                if len(bits) != 8:
                    raise Fail(f"line {lineno}: bit pattern is {bits!r}")
                decoded = struct.unpack(">f", bytes.fromhex(bits))[0]
                reparsed = struct.unpack(
                    ">f", struct.pack(">f", float(value)))[0]
                same = (decoded != decoded and reparsed != reparsed) or (
                    struct.pack(">f", decoded) == struct.pack(">f", reparsed))
                if not same:
                    raise Fail(f"line {lineno}: decimal {value} and bits "
                               f"{bits} disagree")
                slot[key] = decoded
            else:
                raise Fail(f"line {lineno}: unknown dtype {dtype!r}")
    return table


def cases(table, leaf, stage="input"):
    return sorted(c for (lf, c, st) in table if lf == leaf and st == stage)


def check_stage_symmetry(table, leaf):
    for case in cases(table, leaf):
        a = table[(leaf, case, "input")]
        b = table[(leaf, case, "output")]
        if set(a) != set(b):
            missing = sorted(set(a) ^ set(b))
            raise Fail(f"{leaf}/{case}: input and output stages disagree on "
                       f"{missing[:6]}")


# ---------------------------------------------------------------------------
# SNOW_INIT
# ---------------------------------------------------------------------------

def check_snow_init(table):
    leaf = "snow_init"
    names = cases(table, leaf)
    if not names:
        raise Fail("no snow_init cases")
    check_stage_symmetry(table, leaf)

    taken = [0] * len(SI_LADDER)
    edges_seen = set()
    nsoil_seen = set()
    for case in names:
        inp = table[(leaf, case, "input")]
        out = table[(leaf, case, "output")]
        nsnow = inp[("nsnow", 0)]
        nsoil = inp[("nsoil", 0)]
        nsoil_seen.add(nsoil)
        if nsnow != 3:
            raise Fail(f"{case}: NSNOW is {nsnow}; the driver passes 3 at 2295")
        depth = inp[("snodep", 0)]
        swe = inp[("swe", 0)]
        isnow = out[("isnowxy", 0)]

        branch = [i for i, (pred, _) in enumerate(SI_LADDER) if pred(depth)]
        if len(branch) != 1:
            raise Fail(f"{case}: SNODEP={depth} matches {len(branch)} ladder "
                       "branches; the ladder must partition the reals")
        want = SI_LADDER[branch[0]][1]
        if isnow != want:
            raise Fail(f"{case}: SNODEP={depth} is ladder branch "
                       f"{branch[0]} which yields ISNOW={want}, fixture says "
                       f"{isnow}")
        taken[branch[0]] += 1
        for edge in (E025, E05, E10, E25, E45):
            if depth == edge:
                edges_seen.add(edge)

        # ISNOW is the number of snow layers; the layers above it must be the
        # untouched zero fill (2411-2413) and those at or below it must carry
        # the ground temperature (2415).
        for k in range(-nsnow + 1, 1):
            tsno = out[("tsnoxy", k)]
            snliq = out[("snliqxy", k)]
            if k <= isnow:
                if tsno != 0.0 or snliq != 0.0 or out[("snicexy", k)] != 0.0:
                    raise Fail(f"{case}: layer {k} is above ISNOW={isnow} but "
                               "is not the 2411-2413 zero fill")
            else:
                if tsno != inp[("tgxy", 0)]:
                    raise Fail(f"{case}: layer {k} temperature {tsno} is not "
                               f"TGXY {inp[('tgxy', 0)]} (2415)")
                if snliq != 0.0:
                    raise Fail(f"{case}: layer {k} SNLIQ is {snliq}, not 0 "
                               "(2416)")

        # INTENT(OUT) with no assignment: with ISNOW=0, ZSNSOXY's snow slots
        # are never written (2432-2435) and must equal their entry values.
        for k in range(-nsnow + 1, isnow + 1):
            if out[("zsnsoxy", k)] != inp[("zsnsoxy", k)]:
                raise Fail(f"{case}: ZSNSOXY[{k}] is above ISNOW={isnow} so "
                           "2432-2435 never writes it, yet it moved")

        # SNICE is the SWE/SNODEP density times the layer thickness; the
        # column must conserve SWE to the layer sum for a non-degenerate pack.
        if isnow < 0 and swe > 0.0 and depth > 0.0:
            total = sum(out[("snicexy", k)] for k in range(isnow + 1, 1))
            if not (0.98 * swe <= total <= 1.02 * swe):
                raise Fail(f"{case}: snow ice layers sum to {total}, SWE is "
                           f"{swe}; 2417 cannot have been applied")

    for i, count in enumerate(taken):
        if count == 0:
            raise Fail(f"snow_init ladder branch {i} "
                       f"({SI_LADDER[i][1]} layers) is never taken")
    if edges_seen != {E025, E05, E10, E25, E45}:
        raise Fail("snow_init does not sit exactly on every interval edge; "
                   f"missing {sorted({E025, E05, E10, E25, E45} - edges_seen)}")
    if len(nsoil_seen) < 2:
        raise Fail("snow_init exercises a single NSOIL; the soil-layer loops "
                   "at 2426-2435 are then unconstrained")
    return len(names)


# ---------------------------------------------------------------------------
# NOAHMP_INIT
# ---------------------------------------------------------------------------

def check_noahmp_init(table):
    leaf = "noahmp_init"
    names = [c for c in cases(table, leaf) if c != "table_identity"]
    if not names:
        raise Fail("no noahmp_init cases")
    for case in names:
        a = table[(leaf, case, "input")]
        b = table[(leaf, case, "output")]
        if set(a) != set(b):
            raise Fail(f"{leaf}/{case}: stages disagree on fields")

    ident = table[(leaf, "table_identity", "probe")]
    zero_veg = {ident[("isbarren_table", 0)], ident[("isice_table", 0)],
                ident[("iswater_table", 0)], ident[("isurban_table", 0)]}
    lcz = {ident[("lcz_1_table", 0)] + n for n in range(11)}

    seen = defaultdict(int)
    for case in names:
        inp = table[(leaf, case, "input")]
        out = table[(leaf, case, "output")]
        nsoil = inp[("nsoil", 0)]
        ivgtyp = inp[("ivgtyp", 0)]
        xice = inp[("xice", 0)]
        tsk = inp[("tsk", 0)]
        bexp = inp[("bexp_table", 0)]
        smcmax = inp[("smcmax_table", 0)]
        psisat = inp[("psisat_table", 0)]
        fndsnowh = inp[("fndsnowh", 0)] == 1
        snow_in = inp[("snow", 0)]
        snowh_in = inp[("snowh", 0)]

        seen["fndsnowh_true" if fndsnowh else "fndsnowh_false"] += 1

        # --- 2017-2025: SNOWH derivation, and 2037-2040: the 2000 mm cap.
        capped = snow_in > 2000.0
        if capped:
            seen["swe_cap"] += 1
        glacier = ivgtyp == ident[("isice_table", 0)] and xice <= 0.0
        if glacier:
            seen["glacier"] += 1
            if out[("snow", 0)] != max(snow_eff(snow_in, capped), 10.0):
                raise Fail(f"{case}: 2081 SNOW=MAX(SNOW,10) not reproduced")
            for k in range(1, nsoil + 1):
                if out[("smois", k)] != 1.0 or out[("sh2o", k)] != 0.0:
                    raise Fail(f"{case}: 2076-2077 glacier fill not applied")
                if out[("tslb", k)] > f32(263.15):
                    raise Fail(f"{case}: 2078 TSLB clamp not applied")
        else:
            seen["nonglacier"] += 1
            positive = bexp > 0.0 and smcmax > 0.0 and psisat > 0.0
            seen["params_positive" if positive else "params_degenerate"] += 1
            for k in range(1, nsoil + 1):
                smois_in = inp[("smois", k)]
                tslb = inp[("tslb", k)]
                clamped = min(smois_in, smcmax) if smois_in > smcmax else smois_in
                if smois_in > smcmax:
                    seen["smcmax_clamp"] += 1
                if out[("smois", k)] != clamped:
                    raise Fail(f"{case}: layer {k} 2090 SMCMAX clamp not "
                               "reproduced")
                if not positive:
                    if out[("sh2o", k)] != clamped:
                        raise Fail(f"{case}: layer {k} 2104-2106 fallback "
                                   "SH2O=SMOIS not reproduced")
                    continue
                if tslb < FREEZE_TEST:
                    seen["frozen_layer"] += 1
                    fk_raw = ((HLICE / (GRAV * (-psisat)))
                              * ((tslb - T0) / tslb)) ** (-1.0 / bexp) * smcmax
                    if fk_raw < f32(0.02):
                        seen["fk_floor"] += 1
                    if f32(0.02) <= fk_raw and clamped < fk_raw:
                        seen["fk_ceiling"] += 1
                    got = out[("sh2o", k)]
                    if got > clamped + 1e-6:
                        raise Fail(f"{case}: layer {k} SH2O {got} exceeds "
                                   f"SMOIS {clamped}; 2098 MIN not applied")
                    if got < f32(0.02) - 1e-6 and got != clamped:
                        raise Fail(f"{case}: layer {k} SH2O {got} is below "
                                   "the 2097 floor without being SMOIS-capped")
                else:
                    seen["unfrozen_layer"] += 1
                    if out[("sh2o", k)] != clamped:
                        raise Fail(f"{case}: layer {k} 2100 SH2O=SMOIS not "
                                   "reproduced")

        # --- 2117-2132: the snow/warm-skin 273.15 clamps.
        snow_out = out[("snow", 0)]
        warm_snow = snow_out > 0.0 and tsk > T0
        if warm_snow:
            seen["warm_snow_clamp"] += 1
            for field in ("tvxy", "tgxy", "tahxy", "t2mvxy", "t2mbxy"):
                if out[(field, 0)] != T0:
                    raise Fail(f"{case}: 2118-2132 clamp missing on {field}")
        else:
            for field in ("tvxy", "tgxy", "tahxy", "t2mvxy", "t2mbxy"):
                if out[(field, 0)] != tsk:
                    raise Fail(f"{case}: {field} is not TSK (2117-2131)")

        # --- 2164-2280: the vegetation cold start splits on land-use class.
        bare = ivgtyp in zero_veg or ivgtyp in lcz
        if bare:
            seen["veg_zeroed"] += 1
            for field in ("lai", "xsaixy", "lfmassxy", "stmassxy", "rtmassxy",
                          "woodxy", "stblcpxy", "fastcpxy"):
                if out[(field, 0)] != 0.0:
                    raise Fail(f"{case}: 2168-2175 did not zero {field}")
            if out[("cropcat", 0)] != 0:
                raise Fail(f"{case}: 2178 did not zero cropcat")
        else:
            seen["veg_grown"] += 1
            lai_out = out[("lai", 0)]
            if lai_out != max(inp[("lai", 0)], f32(0.05)):
                raise Fail(f"{case}: 2182 LAI floor not reproduced")
            if out[("xsaixy", 0)] != max(f32(f32(0.1) * lai_out), f32(0.05)):
                raise Fail(f"{case}: 2183 SAI form not reproduced")
            if out[("rtmassxy", 0)] != 500.0 or out[("woodxy", 0)] != 500.0:
                raise Fail(f"{case}: 2192-2193 constants not reproduced")
            if out[("stblcpxy", 0)] != 1000.0 or out[("fastcpxy", 0)] != 1000.0:
                raise Fail(f"{case}: 2194-2195 constants not reproduced")
            # iopt_crop=0 leaves this INTENT(OUT) dummy unwritten.
            if out[("cropcat", 0)] != inp[("cropcat", 0)]:
                raise Fail(f"{case}: cropcat moved on a vegetated column, but "
                           "with iopt_crop=0 no statement writes it")
            seen["cropcat_unwritten"] += 1

        # --- 2136-2153: the unconditional cold-start constants.
        for field, want in (("canwat", 0.0), ("canliqxy", 0.0),
                            ("canicexy", 0.0), ("eahxy", 2000.0),
                            ("chstarxy", 0.1), ("cmxy", 0.0), ("chxy", 0.0),
                            ("fwetxy", 0.0), ("sneqvoxy", 0.0),
                            ("alboldxy", f32(0.65)), ("qsnowxy", 0.0),
                            ("qrainxy", 0.0), ("wslakexy", 0.0),
                            ("qtdrain", 0.0), ("waxy", 4900.0),
                            ("grainxy", f32(1e-10)), ("gddxy", 0.0)):
            got = out[(field, 0)]
            ok = got == f32(want)
            if not ok:
                raise Fail(f"{case}: {field} cold start is {got}, expected "
                           f"{want}")
        if out[("wtxy", 0)] != out[("waxy", 0)]:
            raise Fail(f"{case}: 2147 WT=WA not reproduced")

        # --- inertness
        for field, reason in NI_INERT.items():
            for key in [k for k in inp if k[0] == field]:
                if out[key] != inp[key]:
                    raise Fail(f"{case}: {field}{list(key)[1:]} moved, but "
                               f"{reason}")

    required = ("glacier", "nonglacier", "frozen_layer", "unfrozen_layer",
                "smcmax_clamp", "params_positive", "params_degenerate",
                "fk_floor", "fk_ceiling", "swe_cap", "warm_snow_clamp",
                "veg_zeroed", "veg_grown", "cropcat_unwritten",
                "fndsnowh_true", "fndsnowh_false")
    for name in required:
        if seen[name] == 0:
            raise Fail(f"noahmp_init never takes branch {name!r}; the cases "
                       "are too weak to constrain a port")
    return len(names), dict(seen)


def snow_eff(snow_in: float, capped: bool) -> float:
    return 2000.0 if capped else snow_in


def check_discrimination(table):
    """No input slot may be constant across every case of its leaf."""
    for leaf in ("snow_init", "noahmp_init"):
        names = [c for c in cases(table, leaf) if c != "table_identity"]
        values = defaultdict(set)
        for case in names:
            for key, value in table[(leaf, case, "input")].items():
                values[key].add(struct.pack(">f", value)
                                if isinstance(value, float) else value)
        constant = sorted(k for k, v in values.items() if len(v) == 1)
        allowed = {"nsnow", "nsoil", "restart", "allowed_to_read", "iopt_run",
                   "iopt_crop", "iopt_irr", "iopt_irrm", "sf_urban_physics",
                   "fndsoilw", "dzs", "zsoil"}
        offenders = [k for k in constant if k[0] not in allowed]
        if offenders:
            raise Fail(f"{leaf}: input slots {offenders} never vary, so no "
                       "case can discriminate a port that ignores them")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True, type=Path)
    args = parser.parse_args(argv)

    table = load(args.fixture)
    n_si = check_snow_init(table)
    n_ni, seen = check_noahmp_init(table)
    check_discrimination(table)

    print(f"snow_init   : {n_si} cases, ladder fully covered on both edges")
    print(f"noahmp_init : {n_ni} cases")
    for name in sorted(seen):
        print(f"  {name:20s} {seen[name]}")
    print(f"inert and measured: {', '.join(sorted(NI_INERT))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
