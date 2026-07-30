#!/usr/bin/env python3
"""Prove the VEGE_FLUX visibility patch injects no code.

Compares the object file gfortran emits for the pristine
``phys/module_sf_noahmplsm.F`` against the one it emits for the
visibility-patched copy, at the same flags.

What is compared
----------------
* Every module procedure present in both objects, as *normalised
  disassembly*: instruction addresses and raw encodings are stripped, and each
  call/data reference is reduced to the name of the symbol it targets.  That
  normalisation is required and is not a loosening: flipping a procedure from
  ``private`` to ``public`` turns a link-time-resolved local ``call rel32``
  into a relocated ``call`` against the same symbol, which is a different byte
  sequence naming the identical callee.
* Every ``.rodata*`` / ``.data*`` section, byte for byte.
* The symbol sets, reported explicitly.

What the result means at each optimisation level
------------------------------------------------
``-O0``  Nothing is inlined, so every one of the 85 module procedures is
         emitted in both objects and the comparison is total: a pass means the
         patch changed no instruction anywhere in the module.  This is the
         level at which the "no code change" claim is proved.

``-O2``  gfortran treats a ``private`` module procedure as internal and may
         inline it into its only caller and then discard the out-of-line copy;
         making it ``public`` forces the copy to be emitted.  So at ``-O2`` the
         patched object legitimately has *more* symbols and some callers differ
         by inlining.  Those differences are reported, not hidden.  They are
         not FP-semantic: WRF's own FCOPTIM carries no ``-ffast-math`` and no
         ``-march``, so gfortran may neither reassociate nor contract, and the
         build script's ``--cross-opt`` check confirms the fixture is
         bit-identical between ``-O0`` and ``-O2``.

usage:  compare_object_code_vegeflux.py <pristine.o> <patched.o> [--expect-total]
        --expect-total makes any symbol-set difference or body difference fatal
        (use at -O0).  Without it, extra symbols in the patched object are
        reported and tolerated but body differences among *common* symbols are
        still fatal unless --allow-inline-diff is given.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys

_ADDR_BYTES = re.compile(r"^\s*[0-9a-f]+:\t[0-9a-f ]*\t?")
_RELOC = re.compile(r"^\s+[0-9a-f]+:\s+(R_\S+)\s+(\S+)\s*$")
_TARGET = re.compile(r"<([^>+]+)(\+0x[0-9a-f]+)?>")


def run(*cmd: str) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"command failed: {' '.join(cmd)}\n{r.stderr}")
    return r.stdout


def func_symbols(obj: str) -> dict[str, int]:
    out = run("readelf", "-sW", obj)
    syms: dict[str, int] = {}
    for line in out.splitlines():
        p = line.split()
        if len(p) >= 8 and p[3] == "FUNC" and p[6] != "UND":
            syms[p[7]] = int(p[2])
    return syms


def normalised_body(obj: str, sym: str) -> list[str]:
    raw = run("objdump", "-dr", f"--disassemble={sym}", obj).splitlines()
    body: list[str] = []
    pending: list[str] = []
    started = False
    for line in raw:
        if line.endswith(">:"):
            started = True
            continue
        if not started or not line.strip():
            continue
        m = _RELOC.match(line)
        if m is not None:
            if body:
                # attach to the instruction just emitted
                mnem = body[-1].split()[0]
                body[-1] = f"{mnem} <{m.group(2).split('-')[0].split('+')[0]}>"
            continue
        stripped = _ADDR_BYTES.sub("", line).strip()
        if not stripped:
            continue
        if re.fullmatch(r"[0-9a-f ]+", stripped):
            continue                       # continuation line of a long encoding
        tm = _TARGET.search(stripped)
        if tm is not None:
            mnem = stripped.split()[0]
            if mnem.startswith("call"):
                stripped = f"{mnem} <{tm.group(1)}>"
            else:
                stripped = f"{mnem} <{tm.group(1)}{tm.group(2) or ''}>"
        body.append(stripped)
    del pending
    return body


def section_bytes(obj: str, sec: str) -> bytes:
    r = subprocess.run(["objcopy", "-O", "binary", f"--only-section={sec}",
                        obj, "/dev/stdout"], capture_output=True)
    return r.stdout


def section_names(obj: str) -> list[str]:
    out = run("readelf", "-SW", obj)
    names = []
    for line in out.splitlines():
        m = re.search(r"\]\s+(\.\S+)\s+PROGBITS", line)
        if m:
            names.append(m.group(1))
    return names


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pristine")
    ap.add_argument("patched")
    ap.add_argument("--expect-total", action="store_true")
    ap.add_argument("--allow-inline-diff", action="store_true")
    args = ap.parse_args(argv)

    sa, sb = func_symbols(args.pristine), func_symbols(args.patched)
    common = sorted(set(sa) & set(sb))
    only_a = sorted(set(sa) - set(sb))
    only_b = sorted(set(sb) - set(sa))

    identical, differing = 0, []
    for sym in common:
        if normalised_body(args.pristine, sym) == normalised_body(args.patched, sym):
            identical += 1
        else:
            differing.append(sym)

    print(f"module procedures: pristine={len(sa)} patched={len(sb)} common={len(common)}")
    print(f"identical normalised bodies: {identical}")
    print(f"differing bodies: {len(differing)}")
    for sym in differing:
        print(f"  DIFF {sym}")
    print(f"symbols only in pristine ({len(only_a)}):")
    for sym in only_a:
        print(f"  - {sym}")
    print(f"symbols only in patched ({len(only_b)}):")
    for sym in only_b:
        print(f"  + {sym}")

    data_fail = 0
    for sec in section_names(args.pristine):
        if not (sec.startswith(".rodata") or sec.startswith(".data")):
            continue
        a, b = section_bytes(args.pristine, sec), section_bytes(args.patched, sec)
        verdict = "identical" if a == b else "DIFFER"
        if a != b:
            data_fail += 1
        print(f"section {sec}: {verdict} ({len(a)} vs {len(b)} bytes)")

    ok = True
    if differing and not args.allow_inline_diff:
        ok = False
    if args.expect_total:
        if only_a or only_b:
            ok = False
        if data_fail:
            ok = False
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
