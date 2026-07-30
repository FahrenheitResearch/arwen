#!/usr/bin/env python3
"""Mutation study for the Noah-MP snow leaves.

Two families of mutant are generated against ``gpuwm/core/noahmp_snow.py``:

*argument mutants*
    One per argument each leaf actually consumes.  The mutant overwrites the
    argument at the top of the routine with a fixed, physically plausible
    value, so the routine still runs but can no longer see what the caller
    passed.  A mutant that still reproduces the fixture means the fixture
    cannot detect that argument being dropped.

*constant mutants*
    One per ``_f(<literal>)`` site in the file, each perturbed by a relative
    1e-3.  Site-by-site rather than name-by-name, so a constant used in six
    places is probed in all six.

Every mutant is run through ``tests/test_noahmp_snow.py``.  Survivors are
printed; each one has to be argued unreachable, not merely left untested.

Usage::

    python3 mutation_study_snow.py [--quick]
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
SOURCE = REPO / "gpuwm" / "core" / "noahmp_snow.py"
TEST = REPO / "tests" / "test_noahmp_snow.py"

# Anchor line inside each routine, immediately after which an argument
# override can be injected without disturbing the transcription.
ANCHORS = {
    "combo":     "    dz2, wliq2, wice2, t2 = _f(dz2), _f(wliq2), _f(wice2), _f(t2)",
    "snowfall":  "    newnode = 0",
    "compact":   "    burden = _ZERO",
    "combine":   "    isnow_old = col.isnow",
    "divide":    "    nsnow = col.nsnow",
    "snowh2o":   "    epore = _Col(np.zeros(col.nsnow, dtype=np.float32), -col.nsnow + 1)",
    "snowwater": "    snoflow = _ZERO",
}

# (leaf, argument label, override statement).  The override is what "ignoring
# the argument" means concretely: a plausible constant the routine would use if
# it never read the caller's value.
ARG_MUTANTS: list[tuple[str, str, str]] = []


def _col_fields(leaf: str, fields: str) -> None:
    for f in fields.split():
        if f == "isnow":
            ARG_MUTANTS.append((leaf, "col.isnow", "col.isnow = -2"))
        elif f in ("snowh", "sneqv"):
            ARG_MUTANTS.append((leaf, f"col.{f}", f"col.{f} = _f(0.123)"))
        else:
            ARG_MUTANTS.append((leaf, f"col.{f}", f"col.{f}[:] = _f(0.137)"))


# COMBO -- every one of its eight numeric arguments.
for a in ("dz", "wliq", "wice", "t", "dz2", "wliq2", "wice2", "t2"):
    val = "_f(271.3)" if a in ("t", "t2") else "_f(0.137)"
    ARG_MUTANTS.append(("combo", a, f"{a} = {val}"))

# SNOWFALL
for a, v in (("dt", "_f(900.0)"), ("qsnow", "_f(1.3e-4)"),
             ("snowhin", "_f(1.7e-6)"), ("sfctmp", "_f(271.3)")):
    ARG_MUTANTS.append(("snowfall", a, f"{a} = {v}"))
_col_fields("snowfall", "isnow snowh sneqv dzsnso stc snice snliq")

# COMPACT
ARG_MUTANTS.append(("compact", "dt", "dt = _f(900.0)"))
ARG_MUTANTS.append(("compact", "imelt", "IMELT.data[:] = 1"))
ARG_MUTANTS.append(("compact", "ficeold", "FICEOLD.data[:] = _f(0.83)"))
_col_fields("compact", "isnow stc snice snliq dzsnso zsnso")

# COMBINE
for a in ("ponding1", "ponding2"):
    ARG_MUTANTS.append(("combine", a, f"{a} = _f(0.0)"))
_col_fields("combine", "isnow snowh sneqv snice snliq stc dzsnso sice sh2o")

# DIVIDE
_col_fields("divide", "isnow stc snice snliq dzsnso")

# SNOWH2O
for a, v in (("dt", "_f(900.0)"), ("qsnfro", "_f(1.3e-5)"),
             ("qsnsub", "_f(1.9e-5)"), ("qrain", "_f(1.1e-4)"),
             ("ssi", "_f(0.07)"), ("snow_ret_fac", "_f(3.0e-4)"),
             ("ponding1", "_f(0.0)"), ("ponding2", "_f(0.0)")):
    ARG_MUTANTS.append(("snowh2o", a, f"{a} = {v}"))
_col_fields("snowh2o", "isnow dzsnso snowh sneqv snice snliq sh2o sice stc")

# SNOWWATER
for a, v in (("dt", "_f(900.0)"), ("sfctmp", "_f(271.3)"),
             ("snowhin", "_f(1.7e-6)"), ("qsnow", "_f(1.3e-4)"),
             ("qsnfro", "_f(1.3e-5)"), ("qsnsub", "_f(1.9e-5)"),
             ("qrain", "_f(1.1e-4)"), ("ssi", "_f(0.07)"),
             ("snow_ret_fac", "_f(3.0e-4)")):
    ARG_MUTANTS.append(("snowwater", a, f"{a} = {v}"))
ARG_MUTANTS.append(("snowwater", "zsoil", "ZSOIL.data[:] = _f(-0.25)"))
ARG_MUTANTS.append(("snowwater", "imelt", "imelt = np.ones_like(np.asarray(imelt))"))
ARG_MUTANTS.append(("snowwater", "ficeold",
                    "ficeold = np.full_like(np.asarray(ficeold, dtype=np.float32), _f(0.83))"))
_col_fields("snowwater", "isnow snowh sneqv snice snliq stc zsnso dzsnso sh2o sice")


def inject(src: str, leaf: str, stmt: str) -> str:
    anchor = ANCHORS[leaf]
    if anchor not in src:
        raise SystemExit(f"anchor for {leaf} not found: {anchor!r}")
    return src.replace(anchor, anchor + "\n    " + stmt, 1)


_LIT = re.compile(r"_f\((-?\d+\.?\d*(?:[eE][-+]?\d+)?)\)")


def constant_mutants(src: str) -> list[tuple[str, str]]:
    out = []
    for m in _LIT.finditer(src):
        raw = m.group(1)
        val = float(raw)
        new = val * 1.001 if val != 0.0 else 1.0e-4
        line = src[: m.start()].count("\n") + 1
        mutated = src[: m.start()] + f"_f({new!r})" + src[m.end():]
        out.append((f"constant {raw} at line {line}", mutated))
    return out


def run_test() -> bool:
    """True if the test suite passes (i.e. the mutant survived)."""
    r = subprocess.run(
        [sys.executable, "-m", "pytest", str(TEST), "-q", "--no-header", "-x"],
        cwd=REPO, capture_output=True, text=True,
    )
    return r.returncode == 0


def main(argv: list[str]) -> int:
    original = SOURCE.read_text(encoding="utf-8")
    quick = "--quick" in argv

    if run_test() is False:
        raise SystemExit("baseline test suite does not pass; fix that first")

    survivors: list[str] = []
    killed = 0
    try:
        mutants: list[tuple[str, str]] = [
            (f"{leaf}: ignore {label}", inject(original, leaf, stmt))
            for leaf, label, stmt in ARG_MUTANTS
        ]
        if not quick:
            mutants += constant_mutants(original)

        for name, mutated in mutants:
            if mutated == original:
                survivors.append(f"{name}  [NO-OP MUTANT -- generator bug]")
                continue
            SOURCE.write_text(mutated, encoding="utf-8", newline="")
            if run_test():
                survivors.append(name)
                print(f"SURVIVED  {name}")
            else:
                killed += 1
    finally:
        SOURCE.write_text(original, encoding="utf-8", newline="")

    total = killed + len(survivors)
    print(f"\n{killed}/{total} mutants killed, {len(survivors)} survived")
    for s in survivors:
        print(f"  survivor: {s}")
    return 1 if survivors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
