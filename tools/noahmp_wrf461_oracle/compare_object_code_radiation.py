#!/usr/bin/env python3
"""Prove the ``private ::`` -> ``public ::`` patch changes no generated code.

Compiles the pristine pinned module and the visibility-patched copy with
identical flags and identical source *basenames*, then compares, per module
procedure, the normalised disassembly of the procedure body plus every
read-only data section.

Normalisation, and why it is needed
-----------------------------------
Making a module procedure ``public`` changes nothing about the instructions
gfortran emits, but it does change how an intra-module ``call`` is *bound*:
a private callee is resolved to a local offset at assembly time, a public one
becomes a ``R_X86_64_PLT32`` relocation against the symbol.  So the raw ``.o``
bytes are NOT expected to match, and this script does not claim they do.
What it claims -- and checks -- is that after rewriting every ``call`` target
to the *symbol name* it resolves to (taken from the relocation when one
exists, from the direct target otherwise), the two disassemblies are
character-for-character identical, and every ``.rodata`` section hashes the
same.  An arithmetic change cannot hide inside that normalisation: it would
alter a non-``call`` instruction or a constant in ``.rodata``.

Two negative controls prove the comparison can fail:

  ``--negative-control constant``  perturbs one FP32 literal in TWOSTREAM
  ``--negative-control order``     re-associates one sum in SURRAD

Both must be reported as DIFFERENT for the run to be considered meaningful.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

FLAGS = [
    "-c",
    "-ffree-form",
    "-ffree-line-length-none",
    "-cpp",
    "-O0",
    "-ffp-contract=off",
    "-fno-fast-math",
    "-fno-range-check",
    "-g0",
]

_ADDR = re.compile(r"^\s*([0-9a-f]+):\t(.*)$")
_RELOC = re.compile(r"^\s*([0-9a-f]+): (R_\S+)\t(\S+)$")
_CALLTGT = re.compile(r"^(call\s+)[0-9a-f]+ <([^>]*)>$")


def compile_one(src: Path, workdir: Path, gecros: Path) -> Path:
    workdir.mkdir(parents=True, exist_ok=True)
    shutil.copy(gecros, workdir / gecros.name)
    subprocess.run(
        ["gfortran", *FLAGS, gecros.name, "-o", "gecros.o"],
        cwd=workdir, check=True, capture_output=True,
    )
    shutil.copy(src, workdir / "module_sf_noahmplsm.F")
    subprocess.run(
        ["gfortran", *FLAGS, "module_sf_noahmplsm.F", "-o", "noahmp.o"],
        cwd=workdir, check=True, capture_output=True,
    )
    return workdir / "noahmp.o"


def normalised_bodies(obj: Path) -> dict[str, str]:
    """{symbol: normalised instruction text} for every disassembled symbol."""
    out = subprocess.run(
        ["objdump", "-dr", "--no-show-raw-insn", str(obj)],
        check=True, capture_output=True, text=True,
    ).stdout
    bodies: dict[str, list[str]] = {}
    cur: list[str] | None = None
    lines = out.split("\n")
    for i, raw in enumerate(lines):
        head = re.match(r"^[0-9a-f]+ <(.+)>:$", raw)
        if head:
            cur = []
            bodies[head.group(1)] = cur
            continue
        m = _ADDR.match(raw)
        if not m or cur is None:
            continue
        insn = m.group(2).strip()
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        rel = _RELOC.match(nxt)
        ct = _CALLTGT.match(insn)
        if ct:
            target = rel.group(3).split("-0x")[0] if rel else ct.group(2)
            insn = f"call -> {target}"
        elif rel:
            insn = f"{insn}  [{rel.group(2)} {rel.group(3)}]"
        cur.append(insn)
    return {k: "\n".join(v) for k, v in bodies.items()}


def rodata_sections(obj: Path) -> dict[str, str]:
    out = subprocess.run(
        ["objdump", "-h", str(obj)], check=True, capture_output=True, text=True
    ).stdout
    names = [
        ln.split()[1]
        for ln in out.split("\n")
        if len(ln.split()) > 2 and ln.split()[1].startswith(".rodata")
    ]
    res = {}
    for n in names:
        r = subprocess.run(
            ["objcopy", "-O", "binary", "--only-section", n, str(obj), "/dev/stdout"],
            capture_output=True,
        )
        if r.returncode == 0:
            res[n] = hashlib.sha256(r.stdout).hexdigest()
    return res


def apply_negative_control(text: str, kind: str) -> str:
    if kind == "constant":
        needle = "     PHI1   = 0.5 - 0.633*CHIL - 0.330*CHIL*CHIL"
        repl = "     PHI1   = 0.5 - 0.633*CHIL - 0.3300001*CHIL*CHIL"
    elif kind == "order":
        needle = "    SAV     = SAV + CAD(IB) + CAI(IB)"
        repl = "    SAV     = SAV + (CAD(IB) + CAI(IB))"
    else:
        raise SystemExit(f"unknown negative control {kind}")
    if needle not in text:
        raise SystemExit(f"negative control anchor not found: {needle!r}")
    return text.replace(needle, repl, 1)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pristine", type=Path)
    ap.add_argument("patched", type=Path)
    ap.add_argument("gecros", type=Path)
    ap.add_argument("--negative-control", choices=["constant", "order"], default=None)
    ns = ap.parse_args(argv)

    tmp = Path(tempfile.mkdtemp(prefix="noahmp-objcmp-"))
    a_src, b_src = tmp / "a.F", tmp / "b.F"
    shutil.copy(ns.pristine, a_src)
    btxt = ns.patched.read_text(encoding="utf-8", errors="surrogateescape")
    if ns.negative_control:
        btxt = apply_negative_control(btxt, ns.negative_control)
    b_src.write_text(btxt, encoding="utf-8", errors="surrogateescape")

    a_obj = compile_one(a_src, tmp / "a", ns.gecros)
    b_obj = compile_one(b_src, tmp / "b", ns.gecros)

    ba, bb = normalised_bodies(a_obj), normalised_bodies(b_obj)
    same_syms = set(ba) == set(bb)
    differing = sorted(k for k in set(ba) & set(bb) if ba[k] != bb[k])
    ra, rb = rodata_sections(a_obj), rodata_sections(b_obj)

    print(f"module procedures compared  : {len(ba)}")
    print(f"identical symbol set        : {same_syms}")
    print(f"procedures with differences : {len(differing)}")
    for k in differing[:5]:
        print(f"    {k}")
    print(f"identical rodata sections   : {ra == rb}  ({sorted(ra)})")
    verdict = same_syms and not differing and ra == rb
    print(f"VERDICT: {'IDENTICAL' if verdict else 'DIFFERENT'}")
    if ns.negative_control:
        if verdict:
            print("NEGATIVE CONTROL FAILED -- comparison is blind", file=sys.stderr)
            return 2
        print("negative control OK (difference detected)")
        return 0
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
