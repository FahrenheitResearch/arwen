#!/usr/bin/env python3
"""Mutation study for the Noah-MP WATER port.

``max_ulp 0`` is not evidence that a port is right.  A sibling lane in this
project reached ``max_ulp 0`` on 29 columns and then found that 13 of 14
argument-drop mutants still reproduced its pinned CSV -- the fixture could not
tell whether the port read those arguments at all.

WATER is a composition, so its exposure is different from a leaf's: almost
every argument it takes is forwarded, and a forwarded argument is easy to
*lose* in the forwarding.  Three families of mutant are generated against
``gpuwm/core/noahmp_water.py``:

*argument mutants*
    One per argument :func:`gpuwm.core.noahmp_water.water` actually consumes,
    plus one per ``WaterParameters`` component it reads.  The mutant overwrites
    the argument at the top of the routine with a fixed, physically plausible
    value, so the routine still runs but can no longer see what the caller
    passed.

*constant mutants*
    One per ``_f(<literal>)`` site in the file, each perturbed by a relative
    1e-3 -- large enough that FP32 cancellation cannot swallow it, small enough
    that no branch flips for the wrong reason.

*structure mutants*
    WATER's own nine statements, each broken in the one way a careless
    transcription would break it: a dropped term, a flipped comparison, a
    reordered sum, a loop bound taken from the wrong variable.  These are the
    mutants a composition needs and a leaf does not, because a driver's whole
    job is the plumbing between calls.

Every mutant is run through ``tests/test_noahmp_water.py``.  Survivors are
printed; each one has to be argued *unreachable*, not merely listed.

Usage::

    python3 mutation_study_water.py [--quick] [--filter SUBSTRING]
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
SOURCE = REPO / "gpuwm" / "core" / "noahmp_water.py"
TEST = REPO / "tests" / "test_noahmp_water.py"

# The statement immediately after which an argument override can be injected
# without disturbing the transcription: everything above it is type coercion.
ANCHOR = "    etrani = np.zeros(nsoil, dtype=np.float32)                      # :6107"

SCALAR = "_f(0.137)"

ARG_MUTANTS: list[tuple[str, str]] = []


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


def _scalars(names, value=SCALAR):
    for n in names:
        ARG_MUTANTS.append((n, f"{n} = {value}"))


def _arrays(names, value="_f(0.137)"):
    for n in names:
        ARG_MUTANTS.append(
            (n, f"{n} = np.full(np.shape({n}), {value}, dtype=np.float32)"))


def _params(names, value=SCALAR):
    for n in names:
        ARG_MUTANTS.append(
            (f"parameters.{n}",
             f"parameters = _mut_p(parameters, '{n}', {value})"))


def _param_arrays(names):
    for n in names:
        ARG_MUTANTS.append(
            (f"parameters.{n}",
             f"parameters = _mut_p(parameters, '{n}', "
             f"np.full(np.shape(parameters.{n}), _f(0.137), dtype=np.float32))"))


def _column(field, statement):
    ARG_MUTANTS.append((f"col.{field}", statement))


# --- scalars WATER reads directly -----------------------------------------
_scalars(["dt"], "_f(900.0)")
_scalars(["qvap", "qdew", "qrain", "ponding", "fcev", "fctr", "elai", "esai",
          "fveg", "canliq", "canice", "qsnow", "snowhin",
          "acc_qinsur", "acc_qseva"])
_scalars(["bdfall"], "_f(137.0)")
_scalars(["tv"], "_f(271.3)")
_scalars(["sfctmp"], "_f(269.7)")
_scalars(["wslake"], "_f(4137.0)")
ARG_MUTANTS.append(("ist", "ist = 1"))
ARG_MUTANTS.append(("frozen_canopy", "frozen_canopy = False"))
ARG_MUTANTS.append(("frozen_ground", "frozen_ground = False"))

# --- arrays ----------------------------------------------------------------
_arrays(["btrani", "acc_etrani", "smc", "ficeold"])
ARG_MUTANTS.append(("zsoil",
                    "zsoil = np.asarray([-0.137, -0.437, -1.037, -2.037], "
                    "dtype=np.float32)"))
ARG_MUTANTS.append(("imelt", "imelt = [2, 2, 2]"))

# --- the column, which is INOUT and therefore easy to forward wrongly ------
_column("isnow", "col.isnow = 0")
_column("snowh", "col.snowh = _f(0.137)")
_column("sneqv", "col.sneqv = _f(13.7)")
_column("snice", "col.snice[:] = _f(0.137)")
_column("snliq", "col.snliq[:] = _f(0.137)")
_column("stc", "col.stc[:] = _f(271.37)")
_column("dzsnso", "col.dzsnso[:] = _f(0.137)")
_column("sh2o", "col.sh2o[:] = _f(0.137)")
_column("sice", "col.sice[:] = _f(0.0137)")
# ZSNSO is write-only: SNOWWATER rebuilds it and never reads the entry value
# (6522-6533).  The mutant is declared here so the survivor is *expected* and
# the study reports it rather than the study silently not probing it.
_column("zsnso", "col.zsnso[:] = _f(0.137)")

# --- parameters ------------------------------------------------------------
_params(["ch2op", "kdt", "frzx", "slope"])
_params(["ssi"], "_f(0.137)")
_params(["snow_ret_fac"], "_f(1.37e-4)")
_params(["nroot"], "4")
ARG_MUTANTS.append(("parameters.urban_flag",
                    "parameters = _mut_p(parameters, 'urban_flag', False)"))
_param_arrays(["smcmax", "smcwlt", "bexp", "dksat", "dwsat"])

_HELPER = '''

def _mut_p(p, name, value):
    import copy
    q = copy.copy(p)
    object.__setattr__(q, name, value)
    return q
'''

# ---------------------------------------------------------------------------
# Structure mutants: WATER's own nine statements, broken one at a time.
# (label, needle, replacement)
# ---------------------------------------------------------------------------

STRUCT_MUTANTS: list[tuple[str, str, str]] = [
    ("6127 gate becomes >=",
     "    if col.sneqv > _ZERO:                                           # :6127",
     "    if col.sneqv >= _ZERO:                                          # :6127"),
    ("6128 MIN becomes MAX",
     "        qsnsub = min(qvap, _f(col.sneqv / dt))                      # :6128",
     "        qsnsub = max(qvap, _f(col.sneqv / dt))                      # :6128"),
    ("6130 QSEVA drops the QSNSUB debit",
     "    qseva = _f(qvap - qsnsub)                                       # :6130",
     "    qseva = qvap                                                    # :6130"),
    ("6133 QSNFRO gate dropped, frost always taken",
     "    if col.sneqv > _ZERO:                                           # :6133\n        qsnfro = qdew                                               # :6134",
     "    if True:                                                        # :6133\n        qsnfro = qdew                                               # :6134"),
    ("6136 QSDEW drops the QSNFRO debit",
     "    qsdew = _f(qdew - qsnfro)                                       # :6136",
     "    qsdew = qdew                                                    # :6136"),
    ("6138 SNOWWATER given QSNSUB and QSNFRO the wrong way round",
     "        qsnfro, qsnsub, qrain, parameters.ssi, parameters.snow_ret_fac)",
     "        qsnsub, qsnfro, qrain, parameters.ssi, parameters.snow_ret_fac)"),
    ("6146 sign of the frozen-ground exchange flipped",
     "        sice[0] = _f(sice[0] + _f(_f(_f(qsdew - qseva) * dt)",
     "        sice[0] = _f(sice[0] + _f(_f(_f(qseva - qsdew) * dt)"),
    ("6146 divides by DZSNSO(1) without the 1000",
     "                                  / _f(dz[0] * _THOUSAND)))         # :6146",
     "                                  / dz[0]))                         # :6146"),
    ("6147-6148 the post-exchange zeroing dropped",
     "        qsdew = _ZERO                                               # :6147\n        qseva = _ZERO                                               # :6148",
     "        pass                                                        # :6147\n        pass                                                        # :6148"),
    ("6149 SICE deficit test becomes <=",
     "        if sice[0] < _ZERO:                                         # :6149",
     "        if sice[0] <= _f(-1.0e30):                                  # :6149"),
    ("6153 SMC(1) not rebuilt after the exchange",
     "        smc[0] = _f(sh2o[0] + sice[0])                              # :6153",
     "        pass                                                        # :6153"),
    ("6159 PONDING dropped from QINSUR",
     "    qinsur = _f(_f(_f(_f(ponding + ponding1) + ponding2) / dt) * _MILLI)  # :6159",
     "    qinsur = _f(_f(_f(ponding1 + ponding2) / dt) * _MILLI)  # :6159"),
    ("6161 ISNOW test inverted, QRAIN enters only under a pack",
     "    if col.isnow == 0:                                              # :6161",
     "    if col.isnow != 0:                                              # :6161"),
    ("6162 QSDEW dropped from the snow-free QINSUR",
     "        qinsur = _f(qinsur + _f(_f(_f(qsnbot + qsdew) + qrain) * _MILLI))  # :6162",
     "        qinsur = _f(qinsur + _f(_f(qsnbot + qrain) * _MILLI))  # :6162"),
    ("6164 QSNBOT dropped from the snow-covered QINSUR",
     "        qinsur = _f(qinsur + _f(_f(qsnbot + qsdew) * _MILLI))       # :6164",
     "        qinsur = _f(qinsur + _f(qsdew * _MILLI))                    # :6164"),
    ("6167 QSEVA left in mm/s",
     "    qseva = _f(qseva * _MILLI)                                      # :6167",
     "    qseva = qseva                                                   # :6167"),
    ("6169 ETRANI loop runs to NSOIL instead of NROOT",
     "    for iz in range(parameters.nroot):                              # :6169",
     "    for iz in range(nsoil):                                         # :6169"),
    ("6170 ETRANI left in mm/s",
     "        etrani[iz] = _f(_f(etran * btrani[iz]) * _MILLI)            # :6170",
     "        etrani[iz] = _f(etran * btrani[iz])                         # :6170"),
    ("6178 ACC_QINSUR overwritten instead of accumulated",
     "    acc_qinsur = _f(acc_qinsur + qinsur)                            # :6178",
     "    acc_qinsur = qinsur                                             # :6178"),
    ("6179 ACC_QSEVA overwritten instead of accumulated",
     "    acc_qseva = _f(acc_qseva + qseva)                               # :6179",
     "    acc_qseva = qseva                                               # :6179"),
    ("6180 ACC_ETRANI overwritten instead of accumulated",
     "    acc_etrani = acc_etrani + etrani                                # :6180",
     "    acc_etrani = etrani                                             # :6180"),
    ("6209 lake/soil test inverted",
     "    if ist == 2:                                                    # :6209  lake",
     "    if ist != 2:                                                    # :6209  lake"),
    ("6211 WSLMAX test becomes strict",
     "        if wslake >= WSLMAX:                                        # :6211",
     "        if wslake > WSLMAX:                                         # :6211"),
    ("6212 RUNSRF not subtracted from the lake store",
     "        wslake = _f(_f(wslake\n                       + _f(_f(_f(qinsur_avg - qseva_avg) * _THOUSAND)\n                            * dt_soil))\n                    - runsrf)                                       # :6212",
     "        wslake = _f(wslake\n                    + _f(_f(_f(qinsur_avg - qseva_avg) * _THOUSAND)\n                         * dt_soil))                                # :6212"),
    ("6214 SOILWATER handed the raw fluxes, not the accumulator averages",
     "            parameters, dt_soil, zsoil, col.dzsnso, qinsur_avg, qseva_avg,\n            etrani_avg, sice, sh2o, smc, runsub,",
     "            parameters, dt_soil, zsoil, col.dzsnso, qinsur, qseva,\n            etrani, sice, sh2o, smc, runsub,"),
    ("6235 QDRAIN dropped from RUNSUB",
     "        runsub = _f(runsub + qdrain)                                # :6235",
     "        runsub = runsub                                             # :6235"),
    ("6239 SMC not rebuilt from SH2O+SICE",
     "        for iz in range(nsoil):                                     # :6238\n            smc[iz] = _f(sh2o[iz] + sice[iz])                       # :6239",
     "        for iz in range(nsoil):                                     # :6238\n            pass                                                    # :6239"),
    ("6252 RUNSRF not scaled by DT_soil",
     "        runsrf = _f(runsrf * dt_soil)                               # :6252",
     "        runsrf = runsrf                                             # :6252"),
    ("6253 RUNSUB not scaled by DT_soil",
     "        runsub = _f(runsub * dt_soil)                               # :6253",
     "        runsub = runsub                                             # :6253"),
    ("6259 SNOFLOW*DT dropped from RUNSUB",
     "    runsub = _f(runsub + _f(snoflow * dt))                          # :6259",
     "    runsub = runsub                                                 # :6259"),
    ("6259 SNOFLOW added without the DT factor",
     "    runsub = _f(runsub + _f(snoflow * dt))                          # :6259",
     "    runsub = _f(runsub + snoflow)                                   # :6259"),
    ("6252-6254 scaling moved outside the soil branch",
     "        runsrf = _f(runsrf * dt_soil)                               # :6252\n        runsub = _f(runsub * dt_soil)                               # :6253\n        qtldrn = _f(qtldrn * dt_soil)                               # :6254",
     "        pass\n    runsrf = _f(runsrf * dt_soil)\n    runsub = _f(runsub * dt_soil)\n    qtldrn = _f(qtldrn * dt_soil)"),
    ("CANWATER called after the vapour split instead of before",
     "    (canliq, canice, tv, cmc, ecan, etran, fwet,\n     qsubc, qfroc, qfrzc, qmeltc, qevac, qdewc) = canwater(\n        parameters, dt, fcev, fctr, elai, esai, fveg, bdfall,\n        frozen_canopy, canliq, canice, tv)",
     "    (canliq, canice, tv, cmc, ecan, etran, fwet,\n     qsubc, qfroc, qfrzc, qmeltc, qevac, qdewc) = canwater(\n        parameters, dt, fcev, fctr, elai, esai, fveg, bdfall,\n        not frozen_canopy, canliq, canice, tv)"),
]

# Mutants that are expected to survive, with the reason.  A survivor NOT on
# this list is a hole in the fixture; an entry on this list that is killed is
# a stale claim.  Both are reported.
EXPECTED_SURVIVORS = {
    "struct/6127 gate becomes >=":
        "QVAP is MAX(FGEV/LATHEAG, 0.0) at 982, so it is never negative.  With "
        "SNEQV == 0 the mutated gate computes MIN(QVAP, 0.0/DT) = 0.0, which is "
        "exactly the value 6126 already assigned, so the two forms agree on "
        "every state NOAHMP_SFLX can produce.  The fixture is not weak here; "
        "the mutant is unreachable.  Note the 6133 gate is NOT equivalent under "
        "the same change -- QSNFRO would take QDEW and QSDEW would lose it -- "
        "and `struct/6133 QSNFRO gate dropped, frost always taken` kills that.",
    "arg/col.zsnso":
        "ZSNSO is write-only: SNOWWATER rebuilds every slot at 6522-6533 from "
        "ISNOW, DZSNSO and ZSOIL and reads no entry value.  The fixture's "
        "wat_inert_probe drives all seven slots to unrelated values and "
        "reproduces its baseline bit for bit, which is the same fact measured "
        "from the oracle side.",
}


def build_arg_mutant(text: str, statement: str) -> str:
    if ANCHOR not in text:
        raise SystemExit(f"anchor not found: {ANCHOR!r}")
    return text.replace(ANCHOR, ANCHOR + "\n    " + statement, 1) + _HELPER


def build_struct_mutant(text: str, needle: str, replacement: str) -> str:
    if needle not in text:
        raise SystemExit(f"structure mutant needle not found:\n{needle}")
    if text.count(needle) != 1:
        raise SystemExit(f"structure mutant needle is not unique:\n{needle}")
    return text.replace(needle, replacement, 1)


_LITERAL = re.compile(r"_f\((-?\d+\.?\d*(?:[eE][-+]?\d+)?)\)")


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
    return text[:start] + f"_f({value * 1.001!r})" + text[end:]


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
        for label, statement in ARG_MUTANTS:
            name = f"arg/{label}"
            if args.filter and args.filter not in name:
                continue
            total += 1
            write_source(build_arg_mutant(original, statement))
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
            write_source(build_struct_mutant(original, needle, replacement))
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
