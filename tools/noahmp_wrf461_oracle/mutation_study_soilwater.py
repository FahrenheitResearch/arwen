#!/usr/bin/env python3
"""Mutation study for the Noah-MP soil-water port.

``max_ulp 0`` is not evidence that a port is right.  A sibling lane in this
project reached ``max_ulp 0`` on 29 columns and then found that 13 of 14
argument-drop mutants still reproduced its pinned CSV -- the fixture could not
tell whether the port read those arguments at all.

Two families of mutant are generated against ``gpuwm/core/noahmp_soilwater.py``:

*argument mutants*
    One per argument each routine actually consumes, plus one per
    ``SoilParameters`` component.  The mutant overwrites the argument at the
    top of the routine with a fixed, physically plausible value, so the routine
    still runs but can no longer see what the caller passed.  A mutant that
    still reproduces the fixture means the fixture cannot detect that argument
    being dropped.

*constant mutants*
    One per ``_f(<literal>)`` site in the file, each perturbed by a relative
    1e-3 -- large enough that FP32 cancellation cannot swallow it, small enough
    that no branch flips for the wrong reason.  Site by site rather than name by
    name, so a constant used in six places is probed in all six.

Every mutant is run through ``tests/test_noahmp_soilwater.py``.  Survivors are
printed; each one has to be argued *unreachable*, not merely listed.

Usage::

    python3 mutation_study_soilwater.py [--quick] [--filter SUBSTRING]
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
SOURCE = REPO / "gpuwm" / "core" / "noahmp_soilwater.py"
TEST = REPO / "tests" / "test_noahmp_soilwater.py"

# Anchor line inside each routine, immediately after which an argument override
# can be injected without disturbing the transcription.
ANCHORS = {
    "canwater": "    ecan = _ZERO                                                    # :6318",
    "infil": "    sice = np.asarray(sice, dtype=np.float32)\n\n    if qinsur > _ZERO:",
    "srt": "    wdf = np.zeros(nsoil, dtype=np.float32)",
    "sstep": "    wplus = _ZERO                                                   # :7894",
    "soilwater": "    runsrf = _ZERO                                                  # :7318",
}

# Indentation to use for the injected statement in each routine.
INDENT = {"infil": "    "}

SCALAR = "_f(0.137)"
ARRAY = "[:] = _f(0.137)"

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


def _scalars(routine, names, value=SCALAR):
    for n in names:
        ARG_MUTANTS.append((routine, n, f"{n} = {value}"))


def _arrays(routine, names, value="_f(0.137)"):
    for n in names:
        ARG_MUTANTS.append(
            (routine, n,
             f"{n} = np.full(np.shape({n}), {value}, dtype=np.float32)"))


def _params(routine, names, value=SCALAR):
    for n in names:
        ARG_MUTANTS.append((routine, f"parameters.{n}", f"parameters = _mut_p(parameters, '{n}', {value})"))


def _param_arrays(routine, names):
    for n in names:
        ARG_MUTANTS.append(
            (routine, f"parameters.{n}",
             f"parameters = _mut_p(parameters, '{n}', "
             f"np.full(np.shape(parameters.{n}), _f(0.137), dtype=np.float32))"))


# CANWATER
_scalars("canwater", ["dt"], "_f(900.0)")
_scalars("canwater", ["fcev", "fctr", "elai", "esai", "fveg", "canliq",
                      "canice"])
_scalars("canwater", ["bdfall"], "_f(137.0)")
_scalars("canwater", ["tv"], "_f(271.3)")
ARG_MUTANTS.append(("canwater", "frozen_canopy", "frozen_canopy = False"))
_params("canwater", ["ch2op"])

# INFIL
_scalars("infil", ["dt"], "_f(900.0)")
_scalars("infil", ["sicemax", "qinsur", "pddum", "runsrf"])
_arrays("infil", ["sh2o", "sice"])
ARG_MUTANTS.append(("infil", "zsoil",
                    "zsoil = np.asarray([-0.137, -0.437, -1.037, -2.037], "
                    "dtype=np.float32)"))
_params("infil", ["kdt", "frzx"])
_param_arrays("infil", ["smcmax", "smcwlt", "bexp", "dksat", "dwsat"])

# SRT
_scalars("srt", ["pddum", "qseva"])
_arrays("srt", ["etrani", "smc", "fcr"])
ARG_MUTANTS.append(("srt", "zsoil",
                    "zsoil = np.asarray([-0.137, -0.437, -1.037, -2.037], "
                    "dtype=np.float32)"))
_params("srt", ["slope"])
_param_arrays("srt", ["smcmax", "bexp", "dksat", "dwsat"])

# SSTEP
_scalars("sstep", ["dt"], "_f(900.0)")
_arrays("sstep", ["sice", "sh2o", "ai", "bi", "ci", "rhstt"])
# `dz`, not `dzsnso`: the anchor sits after `dz = _soil(dzsnso, nsnow)`, so
# overwriting the argument itself would be a no-op and the mutant would survive
# for a reason that has nothing to do with the fixture.
ARG_MUTANTS.append(("sstep", "dzsnso (via dz)",
                    "dz = np.full(np.shape(dz), _f(0.137), dtype=np.float32)"))
_param_arrays("sstep", ["smcmax"])

# SOILWATER
_scalars("soilwater", ["dt"], "_f(900.0)")
_scalars("soilwater", ["qinsur", "qseva", "runsub"])
_arrays("soilwater", ["etrani", "sice", "sh2o", "smc"])
ARG_MUTANTS.append(("soilwater", "zsoil",
                    "zsoil = np.asarray([-0.137, -0.437, -1.037, -2.037], "
                    "dtype=np.float32)"))
ARG_MUTANTS.append(("soilwater", "dzsnso",
                    "dzsnso = np.full(np.shape(dzsnso), _f(0.137), "
                    "dtype=np.float32)"))
ARG_MUTANTS.append(("soilwater", "parameters.urban_flag",
                    "parameters = _mut_p(parameters, 'urban_flag', False)"))
_param_arrays("soilwater", ["smcmax"])

_HELPER = '''

def _mut_p(p, name, value):
    import copy
    q = copy.copy(p)
    object.__setattr__(q, name, value)
    return q
'''


def build_arg_mutant(text: str, routine: str, statement: str) -> str:
    anchor = ANCHORS[routine]
    if anchor not in text:
        raise SystemExit(f"anchor for {routine} not found: {anchor!r}")
    indent = INDENT.get(routine, "    ")
    return text.replace(anchor, anchor + "\n" + indent + statement, 1) + _HELPER


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
        for routine, label, statement in ARG_MUTANTS:
            name = f"arg/{routine}/{label}"
            if args.filter and args.filter not in name:
                continue
            total += 1
            write_source(build_arg_mutant(original, routine, statement))
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
    if survivors:
        print("survivors, each of which must be argued unreachable:")
        for s in survivors:
            print(f"  {s}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
