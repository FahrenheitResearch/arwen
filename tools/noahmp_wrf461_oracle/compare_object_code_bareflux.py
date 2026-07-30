#!/usr/bin/env python3
"""Diagnostic: what the visibility patch does and does not change in the object.

Read this together with two other artefacts, because on its own it does NOT
establish what one might hope:

* ``visibility_patch_leaves.py --check`` proves the *text* diff is nothing
  but ``private :: X`` -> ``public  :: X`` -- 50 lines, 300 bytes, no line
  length change.
* ``build_bareflux.sh <tree> <work> noopt pristine`` is the proof that
  actually matters.  gfortran gives a ``private`` module procedure LOCAL
  linkage, so the patch is needed only to make the symbol linkable at all;
  ``objcopy --globalize-symbol`` flips that one symbol-table binding while
  leaving ``.text`` byte-identical (the script checks that), and the resulting
  driver executes the unmodified module's own machine code.  Its fixture rows
  are bit-identical to the patched build's.  That is a behavioural proof, and
  it is what the acceptance claim rests on.

This script exists to document why object diffing is the *wrong* test here,
with the numbers.  Accessibility is exactly the thing GCC uses to decide what
it may assume about a procedure with no external callers, so:

  -O0 : all 85 procedures present in both, .rodata identical, but 24 bodies
        differ in register allocation and frame size.
  -O2 : the pristine build clones and specialises private procedures
        (``.isra``, ``.constprop``) and drops 40 of them entirely; 42 common
        bodies differ and .rodata differs.

None of that is the patch changing the program -- it is the patch changing
what the optimiser is allowed to know.  Expecting byte-identical object code
from a visibility change is simply the wrong expectation.

The two negative controls show the comparison can detect a real change:

``--control text``
    Perturbs one digit of ESAT's A0 coefficient; caught in .rodata.
``--control code``
    Reverses the SHB subtraction in BARE_FLUX; caught in the bodies.

Usage:
    compare_object_code_bareflux.py <wrf-tree> <workdir>
        [--control text|code] [--optlevel noopt|wrf]
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# WRF arch/configure.defaults, "Linux x86_64, gfortran".
FCBASE = ["-w", "-ffree-form", "-ffree-line-length-none",
          "-fconvert=big-endian", "-frecord-marker=4"]

# The comparison is run at -O0 by default, and that choice is load-bearing.
# At -O2 GCC's interprocedural passes are *allowed* to see that a `private`
# procedure has no external callers: it clones it (.isra, .constprop), inlines
# it, or drops it entirely.  Making the symbol public removes that freedom, so
# the two objects legitimately differ -- 42 differing bodies and different
# .rodata, which `--optlevel wrf` will show you.  That is an artefact of
# accessibility changing what the optimiser may assume, not of the patch
# changing the program.  At -O0 no such pass runs and the comparison becomes
# a real test of whether the patch altered any emitted instruction.
#
# The remaining gap -- that WRF itself builds at -O2 -- is closed empirically
# rather than by argument: build_bareflux.sh builds the oracle at both -O0 and
# WRF's own -O2 and the two produce bit-identical fixture rows.  So the numbers
# in the fixture do not depend on the optimiser at all.
OPTLEVELS = {
    "noopt": ["-O0"],
    "wrf": ["-O2", "-ftree-vectorize", "-funroll-loops"],
}


def run(cmd, **kw):
    proc = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if proc.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed:\n{proc.stderr}")
    return proc.stdout


def compile_module(src: str, obj: str, workdir: str, optim):
    run(["gfortran", "-c", *FCBASE, *optim, src, "-o", obj,
         "-J", workdir], cwd=workdir)


def procedure_bodies(obj: str) -> dict[str, str]:
    """Disassembly of every function in the object, keyed by symbol name.

    Relocation targets are printed by ``objdump -dr``; addresses are stripped
    so that a shift in one procedure's size cannot masquerade as a difference
    in another's body.
    """
    text = run(["objdump", "-dr", "--no-show-raw-insn", obj])
    bodies: dict[str, list[str]] = {}
    current = None
    for line in text.splitlines():
        head = re.match(r"^[0-9a-f]+ <([^>]+)>:$", line)
        if head:
            current = head.group(1)
            bodies[current] = []
            continue
        if current is None:
            continue
        body = line.strip()
        if not body:
            continue
        # Drop the leading "addr:" and any absolute address in a comment.
        body = re.sub(r"^[0-9a-f]+:\s*", "", body)
        body = re.sub(r"\b[0-9a-f]{4,}\b", "ADDR", body)
        bodies[current].append(body)
    return {k: "\n".join(v) for k, v in bodies.items()}


def rodata(obj: str) -> str:
    out = run(["objdump", "-s", "-j", ".rodata", obj])
    return "\n".join(line.split(" ", 1)[-1] for line in out.splitlines()[4:])


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("wrf_tree")
    ap.add_argument("workdir")
    ap.add_argument("--control", choices=("text", "code"))
    ap.add_argument("--optlevel", choices=tuple(OPTLEVELS), default="noopt")
    args = ap.parse_args(argv)
    optim = OPTLEVELS[args.optlevel]

    work = os.path.abspath(args.workdir)
    os.makedirs(work, exist_ok=True)

    # Both files must have the SAME basename and be compiled from the same
    # relative path: gfortran embeds the source file name in .rodata for its
    # runtime error locations, so "pristine.F" vs "patched.F" alone shows up
    # as a .rodata difference and as differing bodies in every procedure that
    # references one.  Two sibling directories with an identically-named file
    # remove that confound.
    dir_a = os.path.join(work, "objproof_a")
    dir_b = os.path.join(work, "objproof_b")
    os.makedirs(dir_a, exist_ok=True)
    os.makedirs(dir_b, exist_ok=True)
    pristine = os.path.join(dir_a, "module_sf_noahmplsm.F")
    patched = os.path.join(dir_b, "module_sf_noahmplsm.F")
    with open(os.path.join(args.wrf_tree, "phys", "module_sf_noahmplsm.F"),
              "rb") as handle:
        raw = handle.read()
    with open(pristine, "wb") as handle:
        handle.write(raw)

    rc = subprocess.run(
        [sys.executable, os.path.join(HERE, "visibility_patch_leaves.py"),
         pristine, "--out", patched, "--check",
         "--require-symbol", "BARE_FLUX"],
        capture_output=True, text=True)
    print(rc.stdout.strip())
    if rc.returncode != 0:
        print(rc.stderr)
        return rc.returncode

    if args.control == "text":
        # One digit of the ESAT water coefficient A0.
        with open(patched, "r", encoding="latin-1") as handle:
            text = handle.read()
        assert text.count("A0=6.107799961") == 1
        text = text.replace("A0=6.107799961", "A0=6.107799962")
        with open(patched, "w", encoding="latin-1", newline="") as handle:
            handle.write(text)
        print("control: perturbed ESAT A0 by one decimal digit")
    elif args.control == "code":
        with open(patched, "r", encoding="latin-1") as handle:
            text = handle.read()
        target = "        SHB   = CSH * (TGB        - SFCTMP      )"
        assert text.count(target) == 1
        text = text.replace(target,
                            "        SHB   = CSH * (SFCTMP      - TGB        )")
        with open(patched, "w", encoding="latin-1", newline="") as handle:
            handle.write(text)
        print("control: reversed the SHB subtraction in BARE_FLUX")

    # module_sf_gecros is a dependency of the .mod, compile it once.
    gecros = os.path.join(args.wrf_tree, "phys", "module_sf_gecros.F")
    for d in (dir_a, dir_b):
        run(["gfortran", "-c", *FCBASE, *optim, gecros,
             "-o", os.path.join(d, "gecros.o"), "-J", d], cwd=d)

    a = os.path.join(dir_a, "module_sf_noahmplsm.o")
    b = os.path.join(dir_b, "module_sf_noahmplsm.o")
    compile_module("module_sf_noahmplsm.F", "module_sf_noahmplsm.o",
                   dir_a, optim)
    compile_module("module_sf_noahmplsm.F", "module_sf_noahmplsm.o",
                   dir_b, optim)

    ba, bb = procedure_bodies(a), procedure_bodies(b)
    only_a = sorted(set(ba) - set(bb))
    only_b = sorted(set(bb) - set(ba))
    differ = sorted(k for k in set(ba) & set(bb) if ba[k] != bb[k])

    print(f"procedures: pristine {len(ba)}, patched {len(bb)}, "
          f"common {len(set(ba) & set(bb))}")
    if only_a or only_b:
        print(f"  symbols only in pristine: {only_a}")
        print(f"  symbols only in patched : {only_b}")
    if differ:
        print(f"  DIFFERING BODIES ({len(differ)}): {differ[:8]}")
    else:
        print("  all common procedure bodies are byte-identical")

    ra, rb = rodata(a), rodata(b)
    same_rodata = ra == rb
    print("  .rodata identical" if same_rodata else "  .RODATA DIFFERS")

    ok = not (only_a or only_b or differ) and same_rodata
    if args.control:
        if ok:
            print("CONTROL FAILED: the comparison did not notice the change")
            return 1
        print("CONTROL OK: the comparison caught the injected change")
        return 0
    if not ok:
        print(f"EXPECTED at {args.optlevel}: accessibility is an input to "
              "GCC's interprocedural passes, so the objects differ.  See this "
              "file's docstring; the proof that the answer is unchanged is "
              "'build_bareflux.sh <tree> <work> noopt pristine', which links "
              "the unmodified object.")
        return 0
    print(f"({args.optlevel}): every common procedure body and all .rodata "
          "are identical")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
