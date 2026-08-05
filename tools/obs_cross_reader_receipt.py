#!/usr/bin/env python3
"""Measure the mandated science core against an independent NetCDF read.

The observation battery scores two models' history tapes through one
reader.  This writes down the receipt that premise needs: for one frame per
writer, the science core's ``t2``, ``uvmet10`` and column-maximum
``REFL_10CM`` beside the same quantities derived from a plain NetCDF read,
with the difference measured against a registered, mechanism-derived
tolerance.

Strictly CPU-only: no CuPy import, no device required.

    python tools/obs_cross_reader_receipt.py \\
        --frame <label>=<wrfout path> --frame <label>=<wrfout path> \\
        --out-json <receipt.json> [--out-md <table.md>]

The labels are the caller's -- they name the two writers in the receipt and
nothing selects behaviour from them.

Exit status
    0  every registered quantity scored on every side, within tolerance,
       at the pinned science-core version, on a paired case
    1  a registered quantity did not score, or the evaluator commit could
       not be resolved, or a frame is missing
    2  everything scored and the verdict is FAIL

A FAIL verdict is data.  It is written to the receipt exactly as measured
and the exit status reports it; it is never a reason to move a tolerance.
The tolerances live in ``gpuwm/verify/obs/cross_reader.py`` beside the
mechanism that sets them, and moving one moves the registration hash that
every receipt carries.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gpuwm.verify.obs.cross_reader import (  # noqa: E402
    build_cross_reader_receipt,
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
    re-run" is what makes it re-derivable.  Default text-mode translation
    would break that across platforms, so the ending is pinned.
    """
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def _parse_frame(spec: str) -> tuple[str, Path]:
    label, separator, raw = spec.partition("=")
    if not separator or not label.strip() or not raw.strip():
        raise argparse.ArgumentTypeError(
            f"--frame takes <label>=<path>; got {spec!r}")
    return label.strip(), Path(raw.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frame", action="append", required=True,
                        type=_parse_frame, metavar="LABEL=PATH",
                        help="one history-tape frame to score, labelled by "
                             "the writer that produced it; give this twice "
                             "to pair two writers")
    parser.add_argument("--out-json", type=Path, required=True,
                        help="machine receipt to write")
    parser.add_argument("--out-md", type=Path, default=None,
                        help="reviewer-facing table of the same numbers")
    parser.add_argument("--case-id", default=None,
                        help="the registered battery case these frames "
                             "belong to; omit when they do not belong to "
                             "one, and the receipt records that it "
                             "qualifies the reader rather than a case")
    parser.add_argument("--evaluator-commit", default=None,
                        help="40-hex commit of the scoring tree; resolved "
                             "from git when omitted")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    sides: dict[str, Path] = {}
    for label, path in args.frame:
        if label in sides:
            print(f"--frame label {label!r} given twice", file=sys.stderr)
            return EXIT_COVERAGE
        if not path.is_file():
            print(f"no such frame: {path}", file=sys.stderr)
            return EXIT_COVERAGE
        sides[label] = path

    evaluator_commit = args.evaluator_commit or resolve_evaluator_commit(
        REPO_ROOT)
    if evaluator_commit is None:
        print("refusing to write a receipt that cannot name its evaluator: "
              "`git rev-parse HEAD` did not return a commit id",
              file=sys.stderr)
        return EXIT_COVERAGE

    receipt = build_cross_reader_receipt(sides,
                                         evaluator_commit=evaluator_commit,
                                         case_id=args.case_id)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    _write_lf(args.out_json, canonical_json(receipt) + "\n")
    if args.out_md is not None:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        _write_lf(args.out_md, render_markdown(receipt))

    provenance = receipt["science_core_provenance"]
    pin = receipt["science_core_pin"]
    print(f"[cross-reader] receipt: {args.out_json}")
    print(f"[cross-reader] scope: {receipt['scope']}"
          + (f" ({receipt['battery_case_id']})"
             if receipt["battery_case_id"] else ""))
    print(f"[cross-reader] science core: {provenance['distribution']}"
          f"=={provenance['distribution_version']} "
          f"({pin['status']}, pin {pin['expected']})")
    if provenance.get("installed_editable"):
        print(f"[cross-reader]   editable install from "
              f"{provenance['installed_from']}")
    if pin.get("version_attribute_note"):
        print(f"[cross-reader]   note: {pin['version_attribute_note']}")
    for label in sorted(receipt["sides"]):
        side = receipt["sides"][label]
        title = side["identity"]["title"]
        print(f"[cross-reader] {label}: {title!r} @ "
              f"{side['identity']['valid_time']}")
        for name in sorted(side["quantities"]):
            entry = side["quantities"][name]
            if entry["status"] != "scored":
                print(f"[cross-reader]   {name}: {entry['status']} "
                      f"({entry['reason']})")
                continue
            print(f"[cross-reader]   {name}: max|d| = "
                  f"{entry['metrics']['max_abs_diff']:.6g} "
                  f"{entry['units']} (tol {entry['abs_tol']:g}) "
                  f"-> {entry['verdict']}")
    pairing = receipt["pairing"]
    print(f"[cross-reader] pairing: {pairing['status']}"
          + (f" ({pairing['reason']})" if pairing["reason"] else ""))
    print(f"[cross-reader] verdict: {receipt['verdict']}")

    if receipt["unavailable_quantities"]:
        print("[cross-reader] registered quantities that did not score: "
              + ", ".join(receipt["unavailable_quantities"]),
              file=sys.stderr)
        return EXIT_COVERAGE
    if receipt["verdict"] != "PASS":
        return EXIT_VERDICT
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
