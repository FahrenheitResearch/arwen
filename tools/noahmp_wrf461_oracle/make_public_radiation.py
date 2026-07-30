#!/usr/bin/env python3
"""Produce a visibility-only ``public ::`` copy of the pinned Noah-MP module.

The pinned WRF-4.6.1 module declares every internal routine ``private ::``,
so an external oracle driver cannot call the radiation leaves directly.  The
only edit this script performs is turning ``private ::`` declaration lines
into ``public ::`` declaration lines.  Nothing else in the file is touched.

Three things are asserted before the patched copy is written:

1.  the input file hashes to the pinned SHA-256 recorded below,
2.  the unified diff between pristine and patched consists *exclusively* of
    ``private ::`` -> ``public ::`` substitutions on otherwise identical
    lines (any other hunk aborts the run), and
3.  the number of substitutions equals the number of ``private ::`` lines the
    module declares.

``compare_object_code_radiation.py`` then proves gfortran emits the same
object code for both, so the patch cannot have changed any arithmetic.

Redundancy, measured
--------------------
Run against the pinned pristine source this script emits sha256
``3cd3690d6455cfb8549cb41979b7e101e7436464c478f7f7973ab226488ac206`` -- which
is byte for byte the ``PATCHED_SHA256`` that ``check_visibility_patch.py``
already pins and that ``patches/noahmp-lsm-leaf-visibility.patch`` already
produces.  This lane rewrote from scratch what the tree already had, and
landed on the identical file.  ``build_radiation.sh`` could therefore apply
the tree's patch and audit it with ``check_visibility_patch.py``, exactly as
``build_leaves.sh`` / ``build_fluxprep.sh`` / ``build_thermal.sh`` do; that
substitution is provably output-neutral but has not been made here because it
cannot be exercised without the WSL WRF tree and a full oracle rebuild, and an
untested edit to a fixture-regeneration path is worth less than the duplication
it removes.

Note that this is *not* the same output as ``visibility_patch_leaves.py``,
which pads to ``public  ::`` to hold every byte offset fixed and emits sha256
``bfdc0f36...``.  Both are pure accessibility rewrites; they are two spellings,
and the fixtures built from each were built from the file that script emits.

This script is owned by the *radiation* lane.  It writes to a caller-supplied
output path and never modifies the pinned tree.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import re
import sys
from pathlib import Path

PINNED_TREE_COMMIT = "d66e442fccc04111067e29274c9f9eaccc3cef28"
PINNED_SOURCE_SHA256 = (
    "bd592a5b7db29000e715250e3a7c779ffb5e0dcc356f6b5a7d9e1c9f69c55282"
)

_PRIVATE_DECL = re.compile(r"^(\s*)private(\s*)::(.*)$", re.IGNORECASE)


def patch_text(src: str) -> tuple[str, int]:
    out_lines = []
    n_sub = 0
    for line in src.split("\n"):
        m = _PRIVATE_DECL.match(line)
        if m:
            out_lines.append(f"{m.group(1)}public{m.group(2)}::{m.group(3)}")
            n_sub += 1
        else:
            out_lines.append(line)
    return "\n".join(out_lines), n_sub


def check_diff_is_visibility_only(src: str, dst: str) -> int:
    """Return the number of changed lines; abort if any change is not
    exactly ``private ::`` -> ``public ::`` on an otherwise identical line."""
    a = src.split("\n")
    b = dst.split("\n")
    if len(a) != len(b):
        raise SystemExit("visibility patch changed the line count -- refusing")
    changed = 0
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        if tag != "replace" or (i2 - i1) != (j2 - j1):
            raise SystemExit(f"visibility patch produced a {tag} hunk -- refusing")
        for x, y in zip(a[i1:i2], b[j1:j2]):
            mx = _PRIVATE_DECL.match(x)
            if mx is None:
                raise SystemExit(f"non-private line changed: {x!r}")
            want = f"{mx.group(1)}public{mx.group(2)}::{mx.group(3)}"
            if y != want:
                raise SystemExit(f"unexpected replacement: {x!r} -> {y!r}")
            changed += 1
    return changed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=Path, help="pinned phys/module_sf_noahmplsm.F")
    ap.add_argument("dest", type=Path, help="patched copy to write")
    ap.add_argument(
        "--allow-unpinned",
        action="store_true",
        help="skip the SHA-256 gate (diagnostics only; never for a fixture)",
    )
    ns = ap.parse_args(argv)

    raw = ns.source.read_bytes()
    got = hashlib.sha256(raw).hexdigest()
    if got != PINNED_SOURCE_SHA256:
        msg = f"source sha256 {got} != pinned {PINNED_SOURCE_SHA256}"
        if not ns.allow_unpinned:
            raise SystemExit(msg)
        print(f"WARNING: {msg}", file=sys.stderr)

    src = raw.decode("utf-8", errors="surrogateescape")
    dst, n_sub = patch_text(src)
    n_changed = check_diff_is_visibility_only(src, dst)
    if n_changed != n_sub:
        raise SystemExit(f"substitution accounting mismatch {n_changed} != {n_sub}")

    ns.dest.parent.mkdir(parents=True, exist_ok=True)
    ns.dest.write_bytes(dst.encode("utf-8", errors="surrogateescape"))
    print(f"pinned-source-sha256 {got}")
    print(f"pinned-tree-commit   {PINNED_TREE_COMMIT}")
    print(f"private->public      {n_sub}")
    print(f"patched-sha256       {hashlib.sha256(ns.dest.read_bytes()).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
