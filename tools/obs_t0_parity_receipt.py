#!/usr/bin/env python3
"""Publish the t=0 gap between the two engines for one battery case.

The battery's arms start from one analysis.  This measures how far apart
their initial states actually are, over the full prognostic state, and
writes the campaign's per-case t=0 receipt.

The measurement is the tree's existing full-state digest.  What this adds is
how the answer is read: across two engines the digest's bit-parity ceilings
do not apply, so they are recorded verbatim, marked non-binding, and the gap
is published in the variables' own units instead of graded.  What *is*
gated is whether this is a t=0 receipt at all -- required carrier groups
must have scored, and every scored frame must be its own run's initial
frame.

Strictly CPU-only: no CuPy import, no device required.

    python tools/obs_t0_parity_receipt.py \\
        --candidate-dir <dir> --reference-dir <dir> \\
        --ic-route exporter-parity \\
        --out-json <receipt.json> [--out-md <table.md>]

Exit status
    0  the receipt is a t=0 receipt: coverage met, frames initial
    1  it is not: a required group scored nothing, no pair was found, a
       scored frame is not the initial frame, or the evaluator commit could
       not be resolved

There is deliberately no exit status for "the gap is large".  A cross-engine
t=0 gap is the finding, not the failure, and an exit code that graded it
would be asserting the answer this tool exists to report.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gpuwm.verify.obs.t0_parity import (  # noqa: E402
    IC_ROUTES,
    build_t0_parity_receipt,
    canonical_json,
    render_markdown,
    resolve_evaluator_commit,
)

EXIT_OK = 0
EXIT_COVERAGE = 1


def _write_lf(path: Path, text: str) -> None:
    """Write with LF endings whatever platform is writing."""
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidate-dir", type=Path, required=True,
                        help="directory of candidate-engine frames")
    parser.add_argument("--reference-dir", type=Path, required=True,
                        help="directory of reference-engine frames")
    parser.add_argument("--ic-route", required=True, choices=IC_ROUTES,
                        help="how the two engines came to share an initial "
                             "state")
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

    receipt = build_t0_parity_receipt(args.candidate_dir, args.reference_dir,
                                      evaluator_commit=evaluator_commit,
                                      ic_route=args.ic_route,
                                      valid_time=args.valid_time)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    _write_lf(args.out_json, canonical_json(receipt) + "\n")
    if args.out_md is not None:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        _write_lf(args.out_md, render_markdown(receipt))

    print(f"[t0-parity] receipt: {args.out_json}")
    print(f"[t0-parity] initial-condition route: {receipt['ic_route']}")
    for domain in sorted(receipt["gap"]):
        for name in sorted(receipt["gap"][domain]):
            entry = receipt["gap"][domain][name]
            value = entry["max_abs_diff"]
            print(f"[t0-parity]   {domain} {name}: "
                  f"{entry['scored_arrays']} array(s), max|d| = "
                  + ("-" if value is None else f"{value:.6g}")
                  + f" ({entry['max_abs_diff_variable'] or '-'})")
    print(f"[t0-parity] bit-parity gate (not binding across engines): "
          f"{receipt['bit_parity_gate']['verdict']}")
    print(f"[t0-parity] coverage: {receipt['coverage_verdict']}")
    for reason in receipt["coverage_reasons"]:
        print(f"[t0-parity]   {reason}", file=sys.stderr)
    print(f"[t0-parity] verdict: {receipt['verdict']}")

    return EXIT_OK if receipt["coverage_verdict"] == "PASS" else EXIT_COVERAGE


if __name__ == "__main__":
    raise SystemExit(main())
