#!/usr/bin/env python3
"""Mutation study for the Noah-MP driver cold-start port.

``max_ulp 0`` is not evidence that a port is right.  A sibling lane in this
project reached ``max_ulp 0`` on 29 columns and then found that 13 of 14
argument-drop mutants still reproduced its pinned CSV -- the fixture could not
tell whether the port read those arguments at all.

The cold start is almost entirely branch selection and constant assignment, so
its exposure is the *predicates*: a fixture that never sits on an interval edge
cannot tell ``<`` from ``<=``, and a fixture whose columns all take the same
land-use branch cannot tell which category test decided it.  Three families of
mutant are generated against ``gpuwm/core/noahmp_driver.py``:

*argument mutants*
    One per value either routine consumes.  The mutant overwrites the argument
    just after the type-coercion block, so the routine still runs but can no
    longer see what the caller passed.

*constant mutants*
    One per ``f32(<literal>)`` site, each perturbed by a relative 1e-3 -- large
    enough that FP32 cancellation cannot swallow it, and, for a threshold,
    large enough to move the interval edge past a case that sits exactly on it.

*structure mutants*
    Each predicate broken in the one way a careless transcription would break
    it: a comparison flipped, a category test dropped, an accumulation taken
    from the wrong slot, a MIN/MAX swapped, a clamp applied to the wrong
    variable.

Every mutant is run through ``tests/test_noahmp_driver.py``.  Survivors are
printed; each one has to be argued *unreachable*, not merely listed.

Usage::

    python3 mutation_study_driver.py [--quick] [--filter SUBSTRING]
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
SOURCE = REPO / "gpuwm" / "core" / "noahmp_driver.py"
TEST = REPO / "tests" / "test_noahmp_driver.py"

# The last statement of each routine's coercion block.  Everything above it is
# type coercion and shape validation; everything below is the transcription.
SI_ANCHOR = "    DZSNSO = _Col(dzsnso_data, -nsnow + 1)"
NI_ANCHOR = "    SH2O = _Col(sh2o, 1)"

SCALAR = "f32(0.137)"

ARG_MUTANTS: list[tuple[str, str, str]] = []


def read_source() -> str:
    """`SOURCE`, decoded from its exact bytes -- no newline translation.

    `Path.read_text()` opens in TEXT mode: universal newlines collapse
    whatever the file has to "\n", and the matching `Path.write_text()`
    translates every "\n" back to `os.linesep` on the way out.  On
    Windows that turns the `finally` at the end of `main()` -- which
    exists to RESTORE this file untouched -- into a whole-file CRLF
    rewrite.  Measured on the artifact: gpuwm/core/noahmp_water.py has 0
    CR bytes, and one `read_text()` / `write_text()` round trip through it
    leaves 352, one per line, for a study that changed nothing.  The four
    files these studies mutate are 2,678 lines of tracked core physics,
    all currently LF and none of them recorded debt, so a single Windows
    run of a mutation study would land four whole-file diffs.

    tools/build_registry.py made the same fix for the same reason at
    cdd08f005; `mutation_study_snow.py` uses the `newline=""` spelling.
    """
    return SOURCE.read_bytes().decode("utf-8")


def write_source(text: str) -> None:
    """`SOURCE`, written as the exact bytes given.  See `read_source`."""
    SOURCE.write_bytes(text.encode("utf-8"))


def _si(name: str, statement: str) -> None:
    ARG_MUTANTS.append((f"snow_init/{name}", SI_ANCHOR, statement))


def _ni(name: str, statement: str) -> None:
    ARG_MUTANTS.append((f"noahmp_init/{name}", NI_ANCHOR, statement))


# --- SNOW_INIT ------------------------------------------------------------
_si("swe", f"swe = {SCALAR}")
_si("tgxy", "tgxy = f32(261.37)")
_si("snodep", "snodep = f32(0.137)")
_si("zsoil", "zs[:] = np.float32(-0.137); ZSOIL = _Col(zs, 1)")
_si("zsnsoxy_entry",
    "zsnso[:] = np.float32(-0.137); ZSNSO = _Col(zsnso, -nsnow + 1)")
# tsnoxy/snicexy/snliqxy entries are unconditionally overwritten at 2411-2413,
# so their mutants are expected to survive; they are generated anyway so the
# claim is measured rather than asserted.
_si("tsnoxy_entry", "tsno[:] = np.float32(137.0); TSNO = _Col(tsno, -nsnow + 1)")
_si("snicexy_entry",
    "snice[:] = np.float32(13.7); SNICE = _Col(snice, -nsnow + 1)")
_si("snliqxy_entry",
    "snliq[:] = np.float32(1.37); SNLIQ = _Col(snliq, -nsnow + 1)")

# --- NOAHMP_INIT ----------------------------------------------------------
_ni("fndsnowh", "fndsnowh = True")
_ni("vegtyp", "vegtyp = identity.natural")
_ni("xice", "xice = f32(0.137)")
_ni("tsk", "tsk = f32(281.37)")
_ni("lai", "lai = f32(1.37)")
_ni("bexp", "bexp = f32(5.137)")
_ni("smcmax", "smcmax = f32(0.437)")
_ni("psisat", "psisat = f32(0.137)")
_ni("sla", "sla = f32(63.7)")
_ni("snow", "snow = f32(13.7)")
_ni("snowh", "snowh = f32(0.137)")
_ni("tslb", "tslb[:] = np.float32(271.37); TSLB = _Col(tslb, 1)")
_ni("smois", "smois[:] = np.float32(0.237); SMOIS = _Col(smois, 1)")
_ni("dzs", "dz[:] = np.float32(0.37); DZS = _Col(dz, 1)")
_ni("cropcat", "cropcat = 137")
_ni("identity.isice", "identity = dataclasses.replace(identity, isice=137)")
_ni("identity.isurban", "identity = dataclasses.replace(identity, isurban=138)")
_ni("identity.iswater", "identity = dataclasses.replace(identity, iswater=139)")
_ni("identity.isbarren",
    "identity = dataclasses.replace(identity, isbarren=140)")
_ni("identity.lcz", "identity = dataclasses.replace(identity, lcz=(141,))")
_ni("sf_urban_physics", "sf_urban_physics = 1")

# The mutants need `dataclasses` in scope; the port only imports `dataclass`.
_HELPER = "\nimport dataclasses  # mutation-study helper\n"


STRUCT_MUTANTS: list[tuple[str, str, str]] = [
    ("2381 no-snow gate becomes <=",
     "    if snodep < f32(0.025):",
     "    if snodep <= f32(0.025):"),
    ("2385 one-layer upper edge becomes <",
     "    elif f32(0.025) <= snodep <= f32(0.05):",
     "    elif f32(0.025) <= snodep < f32(0.05):"),
    ("2388 two-halves upper edge becomes <",
     "    elif f32(0.05) < snodep <= f32(0.10):",
     "    elif f32(0.05) < snodep < f32(0.10):"),
    ("2392 two-split upper edge becomes <",
     "    elif f32(0.10) < snodep <= f32(0.25):",
     "    elif f32(0.10) < snodep < f32(0.25):"),
    ("2396 three-even upper edge becomes <",
     "    elif f32(0.25) < snodep <= f32(0.45):",
     "    elif f32(0.25) < snodep < f32(0.45):"),
    ("2395 remainder taken from the wrong slot",
     "        DZSNO[0] = f32(snodep - DZSNO[-1])",
     "        DZSNO[0] = f32(snodep - DZSNO[0])"),
    ("2405 deep remainder drops the 0.05 term",
     "        DZSNO[0] = f32(f32(snodep - DZSNO[-1]) - DZSNO[-2])",
     "        DZSNO[0] = f32(snodep - DZSNO[-1])"),
    ("2415 snow temperature takes the layer index instead of TGXY",
     "        TSNO[iz] = tgxy",
     "        TSNO[iz] = f32(tgxy + float(iz))"),
    ("2417 snow ice ignores the pack density",
     "        SNICE[iz] = f32(float(DZSNO[iz]) * f32(swe / snodep))",
     "        SNICE[iz] = f32(float(DZSNO[iz]) * swe)"),
    ("2422 snow thickness kept positive",
     "        DZSNSO[iz] = f32(-float(DZSNO[iz]))",
     "        DZSNSO[iz] = f32(float(DZSNO[iz]))"),
    ("2434 depth accumulated from the wrong neighbour",
     "        ZSNSO[iz] = f32(float(ZSNSO[iz - 1]) + float(DZSNSO[iz]))",
     "        ZSNSO[iz] = f32(float(ZSNSO[iz]) + float(DZSNSO[iz]))"),
    ("2432 top interface seeded from zero instead of DZSNSO",
     "    ZSNSO[isnow + 1] = DZSNSO[isnow + 1]",
     "    ZSNSO[isnow + 1] = f32(0.0)"),
    ("2038 SNOWH cap scaling dropped",
     "        snowh = f32(f32(snowh * _SWE_CAP) / snow)",
     "        snowh = snowh"),
    ("2037 SWE cap gate becomes >=",
     "    if snow > _SWE_CAP:",
     "    if snow >= _SWE_CAP:"),
    ("2074 glacier gate ignores the sea-ice fraction",
     "    if vegtyp == identity.isice and xice <= f32(0.0):",
     "    if vegtyp == identity.isice:"),
    ("2078 glacier temperature clamp becomes a floor",
     "            TSLB[ns] = min(TSLB[ns], _GLACIER_TSLB_CAP)",
     "            TSLB[ns] = max(TSLB[ns], _GLACIER_TSLB_CAP)"),
    ("2081 glacier SWE floor becomes a cap",
     "        snow = max(snow, _GLACIER_SWE_FLOOR)",
     "        snow = min(snow, _GLACIER_SWE_FLOOR)"),
    ("2090 SMCMAX clamp dropped",
     "            if SMOIS[ns] > smcmax:\n                SMOIS[ns] = smcmax",
     "            pass"),
    ("2092 degenerate-parameter guard drops the PSISAT test",
     "        if bexp > f32(0.0) and smcmax > f32(0.0) and psisat > f32(0.0):",
     "        if bexp > f32(0.0) and smcmax > f32(0.0):"),
    ("2094 freeze test uses T0 instead of 273.149",
     "                if TSLB[ns] < _FREEZE_TEST:",
     "                if TSLB[ns] < T0:"),
    ("2097-2098 the FK floor and the SMOIS ceiling are swapped",
     "    fk = max(fk, _FK_FLOOR)\n    return np.float32(min(fk, float(smois)))",
     "    fk = min(fk, _FK_FLOOR)\n    return np.float32(max(fk, float(smois)))"),
    ("2098 SMOIS ceiling dropped",
     "    return np.float32(min(fk, float(smois)))",
     "    return np.float32(fk)"),
    ("2096 exponent loses its sign",
     "    fk = f32(float(powf(base, f32(f32(-1.0) / float(bexp)))) * float(smcmax))",
     "    fk = f32(float(powf(base, f32(f32(1.0) / float(bexp)))) * float(smcmax))"),
    ("2118 warm-skin clamp gate becomes >=",
     "    if snow > f32(0.0) and tsk > T0:",
     "    if snow >= f32(0.0) and tsk > T0:"),
    ("2118 warm-skin clamp ignores the snow test",
     "    if snow > f32(0.0) and tsk > T0:",
     "    if tsk > T0:"),
    ("2122 CANLIQ seeded before CANWAT is zeroed",
     "    canliq = canwat        # 2122 reads CANWAT *after* 2121 zeroed it",
     "    canliq = f32(0.137)"),
    ("2148 ZWT loses the aquifer term",
     "    zwt = f32(f32(f32(25.0) + f32(2.0))\n              - f32(f32(wa / f32(1000.0)) / f32(0.2)))",
     "    zwt = f32(f32(25.0) + f32(2.0))"),
    ("2164 barren category test dropped",
     "    bare = (vegtyp == identity.isbarren or vegtyp == identity.isice",
     "    bare = (vegtyp == identity.isice"),
    ("2166 water category test dropped",
     "            or vegtyp == identity.iswater)",
     "            or False)"),
    ("2165 urban routing dropped",
     "            or (sf_urban_physics == 0 and urbanpt)",
     "            or False"),
    ("2182 LAI floor becomes a cap",
     "        lai_out = max(lai, _LAI_FLOOR)",
     "        lai_out = min(lai, _LAI_FLOOR)"),
    ("2183 SAI takes the input LAI instead of the floored one",
     "        xsai = max(f32(_SAI_PER_LAI * lai_out), _LAI_FLOOR)",
     "        xsai = max(f32(_SAI_PER_LAI * lai), _LAI_FLOOR)"),
    ("2189 leaf mass takes SAI instead of LAI",
     "        lfmass = f32(lai_out * masslai)",
     "        lfmass = f32(xsai * masslai)"),
    ("2191 stem mass takes LAI instead of SAI",
     "        stmass = f32(xsai * _MASSSAI)",
     "        stmass = f32(lai_out * _MASSSAI)"),
    ("2288 first interface loses its sign",
     "    ZSOIL[1] = f32(-float(DZS[1]))",
     "    ZSOIL[1] = f32(float(DZS[1]))"),
    ("2290 interfaces accumulate upward",
     "        ZSOIL[ns] = f32(float(ZSOIL[ns - 1]) - float(DZS[ns]))",
     "        ZSOIL[ns] = f32(float(ZSOIL[ns - 1]) + float(DZS[ns]))"),
    ("2296 SNOW_INIT handed the pre-cap SWE",
     "    si = snow_init_column(nsnow, nsoil, zsoil, snow, tg, snowh,",
     "    si = snow_init_column(nsnow, nsoil, zsoil, f32(13.7), tg, snowh,"),
    ("2296 SNOW_INIT handed TV instead of TG",
     "    si = snow_init_column(nsnow, nsoil, zsoil, snow, tg, snowh,",
     "    si = snow_init_column(nsnow, nsoil, zsoil, snow, f32(261.37), snowh,"),
]

# Mutants expected to survive, with the reason.  A survivor NOT on this list is
# a hole in the fixture; an entry here that is killed is a stale claim.  Both
# are reported and only the first is a failure.
EXPECTED_SURVIVORS = {
    "arg/snow_init/tsnoxy_entry":
        "2411 zeroes TSNOXY(-NSNOW+1:0) unconditionally before 2415 refills "
        "the live slots, so no entry value survives the call.  Unlike ZSNSOXY, "
        "which 2432-2435 writes only from ISNOW+1 upward, TSNOXY has no "
        "unwritten slot at all.  The fixture measures this rather than "
        "assuming it: every case enters all three slots non-zero and the "
        "output stage shows them zeroed or set to TGXY.",
    "arg/snow_init/snicexy_entry":
        "2412 zeroes SNICEXY(-NSNOW+1:0) unconditionally; see tsnoxy_entry.",
    "arg/snow_init/snliqxy_entry":
        "2413 zeroes SNLIQXY(-NSNOW+1:0) unconditionally; see tsnoxy_entry.",
    "struct/2092 degenerate-parameter guard drops the PSISAT test":
        "Under opt_soil=1 all three of BEXP/SMCMAX/PSISAT come from the pinned "
        "SOILPARM.TBL indexed by one ISLTYP, and no STAS category has BEXP>0 "
        "and SMCMAX>0 with PSISAT<=0 -- category 14 (WATER) has all three at "
        "zero and every other category has all three positive.  The two forms "
        "therefore select the same branch on every soil type WRF can present. "
        "tests/test_noahmp_driver.py::"
        "test_degenerate_soil_guard_is_equivalent_to_the_bexp_test walks the "
        "table and asserts exactly that, so the claim is executable rather "
        "than narrative.  The port keeps the full conjunction because the "
        "guard reads runtime values.",
    "struct/2183 SAI takes the input LAI instead of the floored one":
        "The two forms are provably equal for every LAI.  When LAI >= 0.05, "
        "2182 leaves LAI unchanged and the expressions are literally the same. "
        "When LAI < 0.05, 2182 raises it to 0.05, and 0.1*0.05 = 0.005 is "
        "below the 0.05 floor 2183 applies -- as is 0.1*LAI for any smaller "
        "LAI -- so MAX picks 0.05 either way.  No fixture can separate them "
        "because nothing can.",
    "const/line474/1.0":
        "2187's MAX(SLA_TABLE(VEGTYP), 1.0) never selects the 1.0 under "
        "MODIFIED_IGBP_MODIS_NOAH: the only categories whose SLA is below 1.0 "
        "are 15/16/17, which are ISICE/ISBARREN/ISWATER and take the zeroing "
        "branch at 2164 long before 2187.  tests/test_noahmp_driver.py::"
        "test_sla_floor_is_unreachable_on_the_vegetated_branch asserts it over "
        "the pinned table.",
}


def build_arg_mutant(text: str, anchor: str, statement: str) -> str:
    if text.count(anchor) != 1:
        raise SystemExit(f"anchor not unique: {anchor!r}")
    return text.replace(anchor, anchor + "\n    " + statement, 1) + _HELPER


def build_struct_mutant(text: str, needle: str, replacement: str) -> str:
    if text.count(needle) != 1:
        raise SystemExit(f"structure mutant needle is not unique:\n{needle}")
    return text.replace(needle, replacement, 1)


_LITERAL = re.compile(r"f32\((-?\d+\.?\d*(?:[eE][-+]?\d+)?)\)")


def constant_sites(text: str):
    out = []
    for m in _LITERAL.finditer(text):
        value = float(m.group(1))
        if value == 0.0:
            continue
        line = text.count("\n", 0, m.start()) + 1
        out.append((line, m.start(), m.end(), value))
    return out


def build_const_mutant(text: str, site) -> str:
    _line, start, end, value = site
    return text[:start] + f"f32({value * 1.001!r})" + text[end:]


def run_tests(quick: bool) -> bool:
    cmd = [sys.executable, "-m", "pytest", str(TEST), "-q", "--no-header",
           "-p", "no:cacheprovider"]
    if quick:
        cmd.append("-x")
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    return proc.returncode == 0


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--filter", default="")
    args = ap.parse_args(argv)

    original = read_source()
    if not run_tests(args.quick):
        raise SystemExit("the unmutated port already fails; fix that first")

    survivors = []
    total = 0
    try:
        for label, anchor, statement in ARG_MUTANTS:
            name = f"arg/{label}"
            if args.filter and args.filter not in name:
                continue
            total += 1
            write_source(build_arg_mutant(original, anchor, statement))
            if run_tests(args.quick):
                survivors.append(name)
                print(f"SURVIVED  {name}")
            else:
                print(f"killed    {name}")

        for label, needle, replacement in STRUCT_MUTANTS:
            name = f"struct/{label}"
            if args.filter and args.filter not in name:
                continue
            total += 1
            write_source(build_struct_mutant(original, needle,
                                              replacement))
            if run_tests(args.quick):
                survivors.append(name)
                print(f"SURVIVED  {name}")
            else:
                print(f"killed    {name}")

        for site in constant_sites(original):
            name = f"const/line{site[0]}/{site[3]!r}"
            if args.filter and args.filter not in name:
                continue
            total += 1
            write_source(build_const_mutant(original, site))
            if run_tests(args.quick):
                survivors.append(name)
                print(f"SURVIVED  {name}")
            else:
                print(f"killed    {name}")
    finally:
        write_source(original)

    print(f"\n{total - len(survivors)} of {total} mutants killed")
    unexpected = [s for s in survivors if s not in EXPECTED_SURVIVORS]
    stale = [s for s in EXPECTED_SURVIVORS if s not in survivors]
    for s in survivors:
        if s in EXPECTED_SURVIVORS:
            print(f"expected survivor  {s}\n    {EXPECTED_SURVIVORS[s]}")
    if unexpected:
        print("UNEXPECTED survivors, each of which must be argued unreachable:")
        for s in unexpected:
            print(f"  {s}")
    if stale:
        print("stale EXPECTED_SURVIVORS entries (these are now killed):")
        for s in stale:
            print(f"  {s}")
    return 1 if unexpected else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
