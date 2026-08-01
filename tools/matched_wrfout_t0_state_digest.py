#!/usr/bin/env python3
"""Score a staged candidate/reference wrfout pair at t=0 and emit a receipt.

The matched-run stream comparator answers "how far apart did the forecasts
drift".  This answers the prior question -- "did the initial states agree,
across the whole prognostic state and not just the eight carriers the
forecast comparison scores" -- and writes down the answer in a form a
reader can re-derive: a machine receipt with a per-array metric table, the
SHA-256 of every file it opened, the pre-registration it scored under, and
the commit of the tree that scored it.

Strictly CPU-only: no CuPy import, no device required.

    python tools/matched_wrfout_t0_state_digest.py \\
        --candidate-dir <dir> --reference-dir <dir> \\
        --out-json <receipt.json> [--out-md <table.md>]

Exit status
    0  every required carrier group scored, and the comparator passed
    1  a required carrier group scored nothing, or no frame pair was found,
       or the evaluator commit could not be resolved
    2  the pair scored but the comparator's verdict is FAIL

A FAIL verdict is data.  It is written to the receipt exactly as measured
and the exit status reports it; it is never a reason to move a ceiling,
which this tool could not do in any case -- the ceilings live in
``gpuwm/verify/nest_gates.py`` and were pinned long before this tool
existed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gpuwm.verify.t0_state_digest import (  # noqa: E402
    build_t0_receipt,
    canonical_json,
    render_markdown,
    resolve_evaluator_commit,
)

EXIT_OK = 0
EXIT_COVERAGE = 1
EXIT_VERDICT = 2


def _write_lf(path: Path, text: str) -> None:
    """Write with LF endings whatever platform is writing.

    The receipt is a pure function of its inputs, and "byte-identical on a
    re-run" is the property that makes it re-derivable and makes a
    dual-run byte comparison a corruption detector.  Default text-mode
    translation would break that across platforms -- the same inputs would
    produce a CRLF receipt on one machine and an LF receipt on another --
    so the ending is pinned rather than inherited.
    """
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidate-dir", type=Path, required=True,
                        help="directory of candidate wrfout frames")
    parser.add_argument("--reference-dir", type=Path, required=True,
                        help="directory of reference wrfout frames")
    parser.add_argument("--out-json", type=Path, required=True,
                        help="machine receipt to write")
    parser.add_argument("--out-md", type=Path, default=None,
                        help="reviewer-facing table of the same numbers")
    parser.add_argument("--valid-time", default=None,
                        help="score this valid time instead of the earliest "
                             "frame present on both sides")
    parser.add_argument("--evaluator-commit", default=None,
                        help="40-hex commit of the scoring tree; resolved "
                             "from git when omitted")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    evaluator_commit = args.evaluator_commit or resolve_evaluator_commit(
        REPO_ROOT)
    if evaluator_commit is None:
        print("refusing to write a receipt that cannot name its evaluator: "
              "`git rev-parse HEAD` did not return a commit id",
              file=sys.stderr)
        return EXIT_COVERAGE

    receipt = build_t0_receipt(args.candidate_dir, args.reference_dir,
                               evaluator_commit=evaluator_commit,
                               valid_time=args.valid_time)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    _write_lf(args.out_json, canonical_json(receipt) + "\n")
    if args.out_md is not None:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        _write_lf(args.out_md, render_markdown(receipt))

    domains = sorted(receipt["domains"])
    print(f"[t0-digest] receipt: {args.out_json}")
    print(f"[t0-digest] domains scored: {', '.join(domains) or 'none'}")
    for domain in domains:
        entry = receipt["domains"][domain]
        print(f"[t0-digest]   {domain} @ {entry['valid_time']}: "
              f"{entry['scored_carriers']} scored carrier(s)")
    print(f"[t0-digest] boundary group: {receipt['boundary']['status']}"
          + (f" ({receipt['boundary']['reason']})"
             if receipt["boundary"]["reason"] else ""))
    print(f"[t0-digest] covered groups: "
          f"{', '.join(receipt['covered_groups']) or 'none'}")
    print(f"[t0-digest] verdict: {receipt['verdict']}")

    if not domains:
        print("[t0-digest] no frame pair present on both sides", file=sys.stderr)
        return EXIT_COVERAGE
    unmet = receipt["uncovered_required_groups"]
    if unmet:
        print(f"[t0-digest] required carrier group(s) scored nothing: "
              f"{', '.join(unmet)}", file=sys.stderr)
        return EXIT_COVERAGE
    if receipt["verdict"] != "PASS":
        return EXIT_VERDICT
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
