#!/usr/bin/env python3
"""Compare two builds of `module_sf_noahmplsm.F` function body by function body.

This is the numerical half of the leaf-visibility safety argument.
`check_visibility_patch.py` proves the *text* diff is nothing but
`private ::` -> `public ::`.  This proves the *code* gfortran emits for every
routine body is unchanged by that rewrite -- across all ~9,300 lines, not just
the eight leaves the fixtures slice out.

Why not compare `.text` bytes directly
--------------------------------------
Two things move that are not physics:

* **Emission order.**  Lifting the accessibility statements re-orders the
  functions within `.text`, so the section byte stream differs even though
  every body is identical.
* **Call encoding.**  A `LOCAL` (private) callee can be reached by a direct
  PC-relative displacement fixed up at assembly time; the same routine as
  `GLOBAL` goes through an `R_X86_64_PLT32` relocation with a zero placeholder.
  The bytes differ; the call does not.

So the comparison is done on the disassembly with addresses stripped and both
call forms resolved back to the callee's symbol *name*, which is what
`objdump -dr` already prints.  Function bodies must then match exactly.

`.rodata` is compared as raw bytes, because that is where the FP32 literals
live and nothing legitimate re-orders it.  A changed constant shows up there.

Usage:
    compare_object_code.py BASELINE.o CANDIDATE.o
Exit status: 0 identical, 1 a body or constant moved, 2 usage error.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


_HEADER = re.compile(r"^[0-9a-f]+ <(?P<name>[^>]+)>:$")
# "  4a1:\tcall   4a6 <__module_sf_noahmplsm_MOD_esat>"  ->  keep mnemonic and
# operands, drop the leading address.
_INSN = re.compile(r"^\s*[0-9a-f]+:\t(?P<body>.*)$")
# A PC-relative operand annotated with a symbol: "1c2d <foo>", "1f2c0 <foo+0x7>".
_TARGET = re.compile(r"\b[0-9a-f]+ <(?P<sym>[^>+]+)(?P<off>\+0x[0-9a-f]+)?>")
_RELOC = re.compile(r"^\s*[0-9a-f]+:\s+R_\S+\s+(?P<sym>\S+?)(?:[-+]0x[0-9a-f]+)?$")


def disassemble(obj: Path) -> dict[str, list[str]]:
    """symbol name -> normalized instruction list.

    Two encodings of the same call must normalize to the same string:

        pristine (callee LOCAL)   call 1eb27 <..._MOD_albedo>
        patched  (callee GLOBAL)  call 1f2b5 <..._MOD_radiation+0x11c>
                                       R_X86_64_PLT32 ..._MOD_albedo-0x4

    so a relocation line overrides the target of the instruction above it, and
    branches that stay inside the current function are rewritten relative to
    the function symbol rather than to an absolute address.
    """
    out = subprocess.run(
        ["objdump", "-dr", "--no-show-raw-insn", str(obj)],
        check=True, capture_output=True, text=True).stdout

    bodies: dict[str, list[str]] = {}
    current: list[str] | None = None
    current_name = ""
    for line in out.splitlines():
        header = _HEADER.match(line)
        if header is not None:
            current_name = header.group("name")
            current = bodies.setdefault(current_name, [])
            continue
        if current is None:
            continue

        reloc = _RELOC.match(line)
        if reloc is not None and current:
            # Fold into the instruction above: that symbol is the real target.
            mnemonic = current[-1].split(None, 1)[0]
            current[-1] = f"{mnemonic} <{reloc.group('sym')}>"
            continue

        insn = _INSN.match(line)
        if insn is None:
            continue
        body = " ".join(insn.group("body").split())

        def _resolve(match: re.Match) -> str:
            symbol, offset = match.group("sym"), match.group("off") or ""
            if symbol == current_name:
                # Intra-function branch: keep it self-relative so the
                # function's own placement in .text does not participate.
                return f"<.{offset}>"
            return f"<{symbol}{offset}>"

        current.append(_TARGET.sub(_resolve, body))
    return bodies


def rodata(obj: Path) -> bytes:
    out = subprocess.run(
        ["objdump", "-s", "-j", ".rodata", str(obj)],
        check=False, capture_output=True, text=True).stdout
    # Keep only the hex payload columns, so the file name in the banner does
    # not participate.
    payload = []
    for line in out.splitlines():
        match = re.match(r"^\s[0-9a-f]+ ((?:[0-9a-f]{2,8} ){1,4})", line)
        if match is not None:
            payload.append(match.group(1).replace(" ", ""))
    return "".join(payload).encode("ascii")


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    baseline, candidate = Path(argv[1]), Path(argv[2])

    left, right = disassemble(baseline), disassemble(candidate)
    problems: list[str] = []

    only_left = sorted(set(left) - set(right))
    only_right = sorted(set(right) - set(left))
    if only_left:
        problems.append(f"{len(only_left)} symbols only in {baseline.name}: "
                        f"{only_left[:5]}")
    if only_right:
        problems.append(f"{len(only_right)} symbols only in {candidate.name}: "
                        f"{only_right[:5]}")

    moved = []
    for name in sorted(set(left) & set(right)):
        if left[name] != right[name]:
            first = next(
                (i for i, (a, b) in enumerate(zip(left[name], right[name]))
                 if a != b), min(len(left[name]), len(right[name])))
            moved.append(
                f"  {name}: {len(left[name])} vs {len(right[name])} insns, "
                f"first difference at {first}")
    if moved:
        problems.append(f"{len(moved)} function bodies differ:\n"
                        + "\n".join(moved[:20]))

    left_ro, right_ro = rodata(baseline), rodata(candidate)
    if left_ro != right_ro:
        problems.append(
            f".rodata differs ({len(left_ro)} vs {len(right_ro)} hex chars)")

    compared = len(set(left) & set(right))
    if problems:
        print(f"object code DIFFERS over {compared} common symbols",
              file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1

    print(f"object code identical: {compared} function bodies and "
          f"{len(left_ro)//2} .rodata bytes match")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
