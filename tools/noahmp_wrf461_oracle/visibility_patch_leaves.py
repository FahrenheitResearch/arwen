#!/usr/bin/env python3
"""Lift ``private ::`` to ``public ::`` in the pinned WRF v4.6.1 Noah-MP module.

Every per-leaf oracle in this directory has to call a module procedure that
``phys/module_sf_noahmplsm.F`` declares ``private``.  Copying the routine out
of the module would change what the compiler sees, so the accessibility
statements are flipped and nothing else -- and that claim is made runnable
rather than asserted.

One script, four lanes
----------------------
The BARE_FLUX, VEGE_FLUX and PHENOLOGY/PRECIP_HEAT lanes each wrote their own
copy of this, because none of them could see the others.  Run against the
pinned pristine source the three produced a **byte-identical** patched file
(sha256 ``bfdc0f3632cd30b87208b26a309c533b12d9bc2a39d1a36e9165ecf90d0a12c3``),
as does ``snow_visibility_patch.py``, so they are one script with three CLI
surfaces.  This is that script; it carries the union of the three, and no
lane's check was dropped:

``--check``
    Re-derives the patched text from the pristine text and asserts that the
    line count is unchanged, that every differing line is exactly a
    ``private :: X`` -> ``public  :: X`` rewrite of the same symbol *at the
    same byte length*, and that the changed-line count equals the number of
    symbols lifted.

``--emit-symbols``
    Prints the lifted symbols, so a reviewer can see the routine under test in
    the list.

``--require-symbol NAME``
    Fails unless NAME was lifted, so the routine under test cannot silently
    fall out of scope.

``--mutate NAME``
    Negative control.  Applies an *additional* source change on top of the
    visibility lift, so that ``--check`` and the object-code comparison must
    both fail.  Without this, "the checks pass" is not evidence they can fail.

``--self-test``
    Negative control for the checker itself, on a synthetic three-line source:
    a renamed symbol, an edited body, a dropped line and a width change must
    each be rejected.

Two spellings, both correct, and which one this is
-------------------------------------------------
``private`` is 7 characters and ``public`` is 6, so a rewrite either pads to
``public  ::`` (this script, and the four lanes above) or lets the line shrink
by one byte to ``public ::`` (``make_public_radiation.py`` and the tree's own
``patches/noahmp-lsm-leaf-visibility.patch``, which agree with each other byte
for byte at sha256 ``3cd3690d...``).  Both are pure accessibility rewrites and
neither can change numerics.  This one pads, which is the stronger property of
the two: no byte offset in the file moves, so a fixed-form continuation column
cannot shift even in principle.  Do not "harmonise" the two without
regenerating every fixture built from whichever file changes.

Provenance of the input file (checked by this script before anything else):
    tree   wrf-stock-v461-gate-20260721 (WSL; a pinned WRF v4.6.1
           checkout -- the commit and sha256 below are the identity)
    commit d66e442fccc04111067e29274c9f9eaccc3cef28
    sha256 bd592a5b7db29000e715250e3a7c779ffb5e0dcc356f6b5a7d9e1c9f69c55282

The object-code half of the safety argument lives in
``compare_object_code_bareflux.py`` / ``_vegeflux.py`` / ``_radiation.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import re

PRISTINE_SHA256 = "bd592a5b7db29000e715250e3a7c779ffb5e0dcc356f6b5a7d9e1c9f69c55282"

# The only accessibility form WRF uses in this module: optional leading blanks,
# the keyword, ``::``, one symbol, optional trailing blanks.
_PRIVATE = re.compile(r"^(?P<lead>\s*)private(?P<gap>\s*)::(?P<tail>\s*\S+\s*)$")

# Negative controls: each is applied *in addition to* the visibility lift.
# Every one of them must make --check fail (none is a private-statement
# rewrite) and must make the object-code comparison fail (each changes code).
MUTATIONS = {
    # flip the sign convention of the canopy-air interpolation weight
    "bta": ("        BTA  = CVH/COND", "        BTA  = CVH/COND*1.000001"),
    # perturb one physical constant
    "cpair": ("  REAL, PARAMETER :: CPAIR  = 1004.64",
              "  REAL, PARAMETER :: CPAIR  = 1004.65"),
    # flip a comparison in the ground-temperature reset
    "snowh": ("     IF (SNOWH > 0.05 .AND. TG > TFRZ) THEN",
              "     IF (SNOWH >= 0.05 .AND. TG > TFRZ) THEN"),
}


def read_bytes(path: str) -> bytes:
    with open(path, "rb") as handle:
        return handle.read()


def patch_text(text: str) -> tuple[str, list[str]]:
    """Return (patched_text, lifted_symbols_in_source_order)."""
    out: list[str] = []
    lifted: list[str] = []
    for line in text.split("\n"):
        match = _PRIVATE.match(line)
        if match is None:
            out.append(line)
            continue
        lifted.append(match.group("tail").strip())
        # 'private' -> 'public ' keeps the byte length of the line identical,
        # so no column in the file can shift.
        out.append(line.replace("private", "public ", 1))
    return "\n".join(out), lifted


def check(text: str, patched: str, lifted: list[str]) -> str | None:
    """Return None when the patch is provably accessibility-only, else why not."""
    before = text.split("\n")
    after = patched.split("\n")
    if len(before) != len(after):
        return "line count changed"
    changed = 0
    for lineno, (a, b) in enumerate(zip(before, after), start=1):
        if a == b:
            continue
        changed += 1
        if len(a) != len(b):
            return f"line {lineno} changed length"
        match = _PRIVATE.match(a)
        if match is None:
            return f"line {lineno} changed but was not an accessibility statement"
        if b != a.replace("private", "public ", 1):
            return f"line {lineno} rewrite is not private->public"
        if match.group("tail") != b.split("::", 1)[1]:
            return f"line {lineno} symbol changed"
    if changed != len(lifted):
        return "changed-line count does not match lifted-symbol count"
    return None


def self_test() -> int:
    """Prove the checker can fail.  Four tampered patches must be rejected."""
    src = "  private :: ALPHA\n  x = 1.0\n  private :: BETA\n"
    good, lifted = patch_text(src)
    if check(src, good, lifted) is not None:
        print("SELFTEST FAIL: honest patch rejected")
        return 1

    tampered = [
        ("renamed symbol", good.replace("BETA", "GAMMA")),
        ("body edited", good.replace("x = 1.0", "x = 2.0")),
        ("line dropped", "\n".join(good.split("\n")[1:])),
        ("width changed", good.replace("public  :: ALPHA", "public :: ALPHA")),
    ]
    for name, bad in tampered:
        if check(src, bad, lifted) is None:
            print(f"SELFTEST FAIL: tampered patch accepted ({name})")
            return 1
        print(f"  negative control rejected as required: {name}")
    print("SELFTEST OK: honest patch accepted, 4 tampered patches rejected")
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pristine", nargs="?")
    ap.add_argument("--out")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--emit-symbols", action="store_true")
    ap.add_argument("--require-symbol", action="append", default=[])
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--mutate", choices=sorted(MUTATIONS),
                    help="negative control: also apply this source mutation")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()

    if not args.pristine:
        ap.error("pristine path is required unless --self-test is given")

    raw = read_bytes(args.pristine)
    digest = hashlib.sha256(raw).hexdigest()
    if digest != PRISTINE_SHA256:
        print(f"FAIL: pristine sha256 {digest} != pinned {PRISTINE_SHA256}")
        return 2

    text = raw.decode("latin-1")
    patched, lifted = patch_text(text)

    if args.mutate:
        old, new = MUTATIONS[args.mutate]
        if old not in patched:
            print(f"FAIL: mutation anchor not found: {old!r}")
            return 10
        patched = patched.replace(old, new, 1)

    if args.emit_symbols:
        for symbol in lifted:
            print(symbol)

    for want in args.require_symbol:
        if want not in lifted:
            print(f"FAIL: symbol {want!r} was not lifted")
            return 3

    if args.out:
        with open(args.out, "wb") as handle:
            handle.write(patched.encode("latin-1"))

    if args.check:
        why = check(text, patched, lifted)
        if why is not None:
            print(f"FAIL: {why}")
            return 4
        print(f"OK: {len(lifted)} accessibility statements lifted, "
              f"{len(set(lifted))} distinct symbols, no other line differs")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
