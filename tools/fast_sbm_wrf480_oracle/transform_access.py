"""Access transform for WRF v4.8.0 ``phys/module_mp_fast_sbm.F``.

``module module_mp_fast_sbm`` declares ``PRIVATE`` as its default
accessibility (line 3862) and makes only ``FAST_SBM`` and ``FAST_HUCMINIT``
public (line 3864).  The oracle runners need to call ``ONECOND1``,
``COAL_BOTT_NEW``, ``FALFLUXHUCM_Z``, ``Kernals_KS`` and ``BREAKINIT_KS`` and
to read the module's table arrays (``XL``, ``cwll``, ``ima``, ``chucm`` ...).

The transform deletes exactly that one line, so the default accessibility
becomes PUBLIC.  Nothing else changes: entities declared with an explicit
``private`` attribute (``xl_mg`` and friends, ``C2``..``C4``) stay private,
and no executable statement is touched.  The script refuses unless

* the input is LF text whose SHA-256 is the design pin,
* line 3862 is exactly `` PRIVATE`` (matched by content and cross-checked by
  number) inside ``MODULE module_mp_fast_sbm`` before its ``CONTAINS``,
* the resulting unified diff is exactly one deleted line and no added line.

Usage::

    python3 transform_access.py --in module_mp_fast_sbm.F \\
        --out module_mp_fast_sbm_access.F --diff transform_access.diff

The diff and a JSON summary (printed to stdout) are recorded in the fixture
manifest.  Standard library only, so it runs on a bare build node.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import sys
from pathlib import Path

SOURCE_SHA256 = "fd72158202a98d4ad618ebb608eeadffd50852a9875f8bfd9bb8ad63c15f1a2e"
DELETED_LINE_NUMBER = 3862
DELETED_LINE_CONTENT = " PRIVATE"
MODULE_LINE_NUMBER = 3843
MODULE_LINE_CONTENT = "  MODULE module_mp_fast_sbm"
PUBLIC_LINE_NUMBER = 3864
PUBLIC_LINE_CONTENT = " PUBLIC FAST_SBM,FAST_HUCMINIT"


class TransformRefusal(RuntimeError):
    """The input is not the pinned source or the edit is not the one line."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def access_transform(source: bytes) -> tuple[bytes, str, dict]:
    """Return (transformed bytes, unified diff text, summary dict)."""

    if b"\r" in source:
        raise TransformRefusal("input carries CR bytes; LF-normalize it first")
    digest = _sha256(source)
    if digest != SOURCE_SHA256:
        raise TransformRefusal(
            f"input SHA-256 {digest} is not the pinned {SOURCE_SHA256}")
    text = source.decode("ascii")
    lines = text.split("\n")
    idx = DELETED_LINE_NUMBER - 1
    if lines[idx] != DELETED_LINE_CONTENT:
        raise TransformRefusal(
            f"line {DELETED_LINE_NUMBER} is {lines[idx]!r}, "
            f"expected {DELETED_LINE_CONTENT!r}")
    if lines[MODULE_LINE_NUMBER - 1] != MODULE_LINE_CONTENT:
        raise TransformRefusal("module header is not where the pin says")
    if lines[PUBLIC_LINE_NUMBER - 1] != PUBLIC_LINE_CONTENT:
        raise TransformRefusal("PUBLIC statement is not where the pin says")
    # Content match: the only line equal to " PRIVATE" in the whole file.
    matches = [i + 1 for i, line in enumerate(lines)
               if line == DELETED_LINE_CONTENT]
    if matches != [DELETED_LINE_NUMBER]:
        raise TransformRefusal(
            f"content match found {matches}, expected [{DELETED_LINE_NUMBER}]")
    # It must sit between the module header and the module's CONTAINS.
    contains = next(i + 1 for i in range(MODULE_LINE_NUMBER, len(lines))
                    if lines[i].strip().upper() == "CONTAINS")
    if not MODULE_LINE_NUMBER < DELETED_LINE_NUMBER < contains:
        raise TransformRefusal("PRIVATE line is outside the module spec part")

    out_lines = lines[:idx] + lines[idx + 1:]
    out_text = "\n".join(out_lines)
    diff_lines = list(difflib.unified_diff(
        text.splitlines(keepends=True), out_text.splitlines(keepends=True),
        fromfile="a/phys/module_mp_fast_sbm.F",
        tofile="b/phys/module_mp_fast_sbm.F", n=0))
    diff_text = "".join(diff_lines)
    body = [d for d in diff_lines
            if not d.startswith(("---", "+++", "@@"))]
    removed = [d for d in body if d.startswith("-")]
    added = [d for d in body if d.startswith("+")]
    if len(removed) != 1 or added or removed[0] != "-" + DELETED_LINE_CONTENT + "\n":
        raise TransformRefusal(f"diff is not exactly one deleted line: {body}")
    hunks = [d for d in diff_lines if d.startswith("@@")]
    if hunks != [f"@@ -{DELETED_LINE_NUMBER} +{DELETED_LINE_NUMBER - 1},0 @@\n"]:
        raise TransformRefusal(f"unexpected hunk header {hunks}")
    out_bytes = out_text.encode("ascii")
    summary = {
        "transform": "access",
        "input_sha256": digest,
        "output_sha256": _sha256(out_bytes),
        "deleted_line_number": DELETED_LINE_NUMBER,
        "deleted_line_content": DELETED_LINE_CONTENT,
        "diff_removed_lines": 1,
        "diff_added_lines": 0,
        "hunk": hunks[0].strip(),
    }
    return out_bytes, diff_text, summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--in", dest="src", required=True, type=Path)
    parser.add_argument("--out", dest="dst", required=True, type=Path)
    parser.add_argument("--diff", dest="diff", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        out, diff, summary = access_transform(args.src.read_bytes())
    except TransformRefusal as exc:
        print(f"transform_access: REFUSED: {exc}", file=sys.stderr)
        return 2
    args.dst.write_bytes(out)
    args.diff.write_bytes(diff.encode("ascii"))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
