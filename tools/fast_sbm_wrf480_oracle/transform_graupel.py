"""Graupel transform for WRF v4.8.0 ``phys/module_mp_fast_sbm.F``.

WRF fixes the dense-ice class of the fast SBM at compile time:
``INTEGER,PARAMETER :: hail_opt = 1`` (line 3886 of the pinned source) gives
hail, ``hail_opt = 0`` graupel. There is no namelist switch, so the graupel
variant can only be built from edited source. This transform makes exactly
that edit and nothing else: it changes the one line ``hail_opt = 1`` to
``hail_opt = 0``, matched by content, and refuses unless

* the input is LF text with exactly one line equal to
  `` INTEGER,PARAMETER :: hail_opt = 1`` inside ``MODULE module_mp_fast_sbm``
  before its ``CONTAINS``,
* for the pristine source (``--variant pristine``) the input's SHA-256 is the
  design pin and the line is line 3886; for the access-transformed or
  probe-transformed source (``--variant access`` or ``--variant probe``) the
  input's SHA-256 is the one ``--expect-sha256`` names (the output hash the
  earlier transform recorded), so the graupel build is always "that source
  with one line changed",
* the resulting unified diff is exactly one removed and one added line at the
  same position, differing only in the ``1`` becoming ``0``.

Usage::

    python3 transform_graupel.py --in module_mp_fast_sbm.F --variant pristine \\
        --out module_mp_fast_sbm_graupel.F --diff transform_graupel.diff

The diff and a JSON summary (printed to stdout) are recorded in the fixture
manifest. Standard library only, so it runs on a bare build node.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from transform_access import SOURCE_SHA256, TransformRefusal  # noqa: E402

PRISTINE_LINE_NUMBER = 3886
OLD_LINE = " INTEGER,PARAMETER :: hail_opt = 1"
NEW_LINE = " INTEGER,PARAMETER :: hail_opt = 0"
MODULE_LINE_CONTENT = "  MODULE module_mp_fast_sbm"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def graupel_transform(source: bytes, variant: str,
                      expect_sha256: str | None = None) -> tuple[bytes, str, dict]:
    """Return (transformed bytes, unified diff text, summary dict)."""

    if b"\r" in source:
        raise TransformRefusal("input carries CR bytes; LF-normalize it first")
    digest = _sha256(source)
    if variant == "pristine":
        if digest != SOURCE_SHA256:
            raise TransformRefusal(f"input SHA-256 {digest} is not the pinned {SOURCE_SHA256}")
    elif variant in ("access", "probe"):
        if not expect_sha256 or digest != expect_sha256:
            raise TransformRefusal(
                f"input SHA-256 {digest} is not the recorded {variant} output {expect_sha256}")
    else:
        raise TransformRefusal(f"unknown variant {variant!r}")
    text = source.decode("ascii")
    lines = text.split("\n")
    hits = [i for i, line in enumerate(lines) if line == OLD_LINE]
    if len(hits) != 1:
        raise TransformRefusal(f"{OLD_LINE.strip()!r} found on {len(hits)} lines, expected 1")
    idx = hits[0]
    if variant == "pristine" and idx + 1 != PRISTINE_LINE_NUMBER:
        raise TransformRefusal(f"hail_opt is on line {idx + 1}, not {PRISTINE_LINE_NUMBER}")
    if any(line == NEW_LINE for line in lines):
        raise TransformRefusal("the input already carries hail_opt = 0")
    module = [i for i, line in enumerate(lines) if line == MODULE_LINE_CONTENT]
    if len(module) != 1:
        raise TransformRefusal("module_mp_fast_sbm header not found exactly once")
    contains = next(i for i in range(module[0], len(lines))
                    if lines[i].strip().upper() == "CONTAINS")
    if not module[0] < idx < contains:
        raise TransformRefusal("hail_opt is outside the module's specification part")

    out_lines = list(lines)
    out_lines[idx] = NEW_LINE
    out_text = "\n".join(out_lines)
    diff_lines = list(difflib.unified_diff(
        text.splitlines(keepends=True), out_text.splitlines(keepends=True),
        fromfile="a/phys/module_mp_fast_sbm.F", tofile="b/phys/module_mp_fast_sbm.F", n=0))
    body = [d for d in diff_lines if not d.startswith(("---", "+++", "@@"))]
    hunks = [d for d in diff_lines if d.startswith("@@")]
    if body != ["-" + OLD_LINE + "\n", "+" + NEW_LINE + "\n"]:
        raise TransformRefusal(f"diff is not exactly the one hail_opt line: {body}")
    n = idx + 1
    if hunks != [f"@@ -{n} +{n} @@\n"]:
        raise TransformRefusal(f"unexpected hunk header {hunks}")
    out_bytes = out_text.encode("ascii")
    summary = {
        "transform": "graupel",
        "variant": variant,
        "input_sha256": digest,
        "output_sha256": _sha256(out_bytes),
        "changed_line_number": n,
        "pristine_line_number": PRISTINE_LINE_NUMBER,
        "old_line": OLD_LINE,
        "new_line": NEW_LINE,
        "diff_removed_lines": 1,
        "diff_added_lines": 1,
        "hunk": hunks[0].strip(),
    }
    return out_bytes, "".join(diff_lines), summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--in", dest="src", required=True, type=Path)
    parser.add_argument("--variant", required=True, choices=("pristine", "access", "probe"))
    parser.add_argument("--expect-sha256", default=None)
    parser.add_argument("--out", dest="dst", required=True, type=Path)
    parser.add_argument("--diff", dest="diff", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        out, diff, summary = graupel_transform(args.src.read_bytes(), args.variant,
                                               args.expect_sha256)
    except TransformRefusal as exc:
        print(f"transform_graupel: REFUSED: {exc}", file=sys.stderr)
        return 2
    args.dst.write_bytes(out)
    args.diff.write_bytes(diff.encode("ascii"))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
