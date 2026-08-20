"""CPU-only ``gpuwm spectral`` front door."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from gpuwm.verify import spectral
from gpuwm.verify import spectral_compare
from gpuwm.verify import spectral_plot
from gpuwm.verify import spectral_receipt


def _register(args: argparse.Namespace) -> int:
    registration = spectral_receipt.register_file(args.spec, args.output)
    print(f"registration {args.output}")
    print(f"registration_sha256 {registration['registration_sha256']}")
    print("spectral_compare_pins_sha256 "
          f"{registration['parameters']['spectral_compare_pins_sha256']}")
    print("legacy_spectral_v1_pins_sha256 "
          f"{registration['parameters']['legacy_spectral_v1_pins_sha256']}")
    return 0


def _score(args: argparse.Namespace) -> int:
    receipt = spectral_receipt.score_file(args.registration, args.output)
    print(f"receipt {args.output}")
    print(f"receipt_sha256 {receipt['receipt_sha256']}")
    print(f"comparisons {len(receipt['comparisons'])}")
    print(f"verdict {receipt['verdict']}")
    if args.plot_dir is not None:
        manifest = spectral_plot.plot_receipt(receipt, args.plot_dir)
        print(f"plots {len(manifest['files'])} manifest "
              f"{Path(args.plot_dir) / 'manifest.json'}")
    return 0 if receipt["verdict"] not in ("fail", "incomplete") else 1


def _run(args: argparse.Namespace) -> int:
    # Publication order is the contract: registration is durably written
    # before score_file opens the first model-output byte.
    registration = spectral_receipt.register_file(args.spec, args.registration)
    print(f"registration {args.registration}")
    print(f"registration_sha256 {registration['registration_sha256']}")
    receipt = spectral_receipt.score_file(args.registration, args.receipt)
    print(f"receipt {args.receipt}")
    print(f"receipt_sha256 {receipt['receipt_sha256']}")
    print(f"verdict {receipt['verdict']}")
    if args.plot_dir is not None:
        manifest = spectral_plot.plot_receipt(receipt, args.plot_dir)
        print(f"plots {len(manifest['files'])} manifest "
              f"{Path(args.plot_dir) / 'manifest.json'}")
    return 0 if receipt["verdict"] not in ("fail", "incomplete") else 1


def _check(args: argparse.Namespace) -> int:
    receipt = spectral_receipt.check_file(
        args.receipt, rehash_inputs=args.rehash_inputs)
    gates = receipt["gates"]
    print(f"receipt_sha256 {receipt['receipt_sha256']}")
    print(f"verdict {receipt['verdict']}")
    print(f"gates pass={gates['passed']} fail={gates['failed']} "
          f"incomplete={gates['incomplete']}")
    return 0 if receipt["verdict"] not in ("fail", "incomplete") else 1


def _plot(args: argparse.Namespace) -> int:
    manifest = spectral_plot.plot_file(args.receipt, args.output_dir)
    print(f"plots {len(manifest['files'])}")
    print(f"manifest {Path(args.output_dir) / 'manifest.json'}")
    print(f"manifest_sha256 {manifest['manifest_sha256']}")
    return 0


def _cross_box(args: argparse.Namespace) -> int:
    """Compare two boxes' receipts under the declared tolerance.

    The breakage this answers: two boxes scoring the SAME input bytes produce
    different receipt hashes -- measured, and correct, because the last bits
    of an FFT belong to the box.  Diffing the hashes therefore says
    "different" every time and a campaign cannot tell a port from a defect.
    This compares the numbers instead.
    """

    left = spectral_receipt.check_file(args.receipt, rehash_inputs=False)
    right = spectral_receipt.check_file(args.other, rehash_inputs=False)
    tolerance = (spectral_compare.CROSS_BOX_TOLERANCE
                 if args.tolerance is None else float(args.tolerance))
    # NOT ``registration_sha256``: that binds the RESOLVED absolute paths, so
    # two boxes registering the same campaign always disagree and this door
    # would refuse every real comparison.  Measured that way against the
    # artifact before this line was written.  The policy hash is the
    # campaign -- bands, fields, gates, crop, pins -- with the paths reduced
    # to basenames.
    if (left["registration_policy_sha256"]
            != right["registration_policy_sha256"]):
        raise ValueError(
            "these two receipts were scored under different campaign policy "
            f"({left['registration_policy_sha256'][:12]} and "
            f"{right['registration_policy_sha256'][:12]}), so a value "
            "difference would be the bands, fields, gates or crop differing "
            "rather than the boxes. Register both boxes from ONE source "
            "TOML -- differing absolute paths are fine, the policy hash "
            "ignores them -- and compare those receipts.")
    left_inputs = {item["path"].split("/")[-1].split("\\")[-1]: item["sha256"]
                   for item in left["inputs"]}
    right_inputs = {item["path"].split("/")[-1].split("\\")[-1]: item["sha256"]
                    for item in right["inputs"]}
    if left_inputs != right_inputs:
        raise ValueError(
            "these two receipts did not read the same input bytes (compared "
            "by file name and full-file SHA-256), so a value difference "
            "would be the data differing, not the boxes. Copy the same "
            "files to both boxes and rescore.")
    print(f"rule {spectral_compare.CROSS_BOX_RULE['rule']}")
    print(f"tolerance {tolerance:g}")
    print(f"receipt_sha256 {left['receipt_sha256']}")
    print(f"receipt_sha256 {right['receipt_sha256']}")
    print(f"receipt_sha256_equal "
          f"{left['receipt_sha256'] == right['receipt_sha256']}")
    total = 0
    for one, other in zip(left["comparisons"], right["comparisons"]):
        rows = spectral_compare.cross_box_differences(
            one["result"], other["result"], tolerance=tolerance)
        for row in rows:
            total += 1
            print(f"differs {one['pair']}:{one['field']}:{row['band']}:"
                  f"{row['component']}:{row['metric']} "
                  f"left={row['left']!r} right={row['right']!r} "
                  f"difference={row['difference']}")
    print(f"differences {total}")
    if total == 0:
        print("agree: every metric is inside the declared tolerance; the "
              "receipt hashes differ because a receipt hash is a this-box "
              "identity, not a portable one")
    return 0 if total == 0 else 1


def _pins(_args: argparse.Namespace) -> int:
    print(json.dumps({
        "spectral_compare": spectral_compare.implementation_registration(),
        "legacy_spectral_v1_pins_sha256": spectral.PINS_SHA256,
    }, indent=2, sort_keys=True))
    return 0


def register_cli(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "spectral",
        help="pre-register and score scale-resolved model/reference agreement")
    commands = parser.add_subparsers(dest="spectral_command", required=True)

    register = commands.add_parser(
        "register", help="pin bands, fields and gates before opening output")
    register.add_argument("spec", type=Path, metavar="SPEC.toml")
    register.add_argument("--output", type=Path, required=True,
                          metavar="REGISTRATION.json")
    register.set_defaults(func=_register)

    score = commands.add_parser(
        "score", help="score a pre-registration and emit a self-hashed receipt")
    score.add_argument("registration", type=Path, metavar="REGISTRATION.json")
    score.add_argument("--output", type=Path, required=True,
                       metavar="RECEIPT.json")
    score.add_argument("--plot-dir", type=Path, default=None, metavar="DIR")
    score.set_defaults(func=_score)

    run = commands.add_parser(
        "run", help="register first, then score, in one ordered invocation")
    run.add_argument("spec", type=Path, metavar="SPEC.toml")
    run.add_argument("--registration", type=Path, required=True,
                     metavar="REGISTRATION.json")
    run.add_argument("--receipt", type=Path, required=True,
                     metavar="RECEIPT.json")
    run.add_argument("--plot-dir", type=Path, default=None, metavar="DIR")
    run.set_defaults(func=_run)

    check = commands.add_parser(
        "check", help="validate receipt identity and optionally rehash inputs")
    check.add_argument("receipt", type=Path, metavar="RECEIPT.json")
    check.add_argument("--rehash-inputs", action="store_true")
    check.set_defaults(func=_check)

    plot = commands.add_parser(
        "plot", help="render receipt-bound evidence without reopening output")
    plot.add_argument("receipt", type=Path, metavar="RECEIPT.json")
    plot.add_argument("--output-dir", type=Path, required=True, metavar="DIR")
    plot.set_defaults(func=_plot)

    cross_box = commands.add_parser(
        "cross-box",
        help="compare two boxes' receipts by value under the declared "
             "tolerance, not by hash")
    cross_box.add_argument("receipt", type=Path, metavar="RECEIPT.json")
    cross_box.add_argument("other", type=Path, metavar="OTHER-RECEIPT.json")
    cross_box.add_argument(
        "--tolerance", type=float, default=None,
        help="override the declared tolerance "
             f"({spectral_compare.CROSS_BOX_TOLERANCE:g}); the default is "
             "measured, so a campaign widening it says why in its record")
    cross_box.set_defaults(func=_cross_box)

    pins = commands.add_parser(
        "pins", help="print the exact arithmetic pins and both pin hashes")
    pins.set_defaults(func=_pins)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    import sys

    parser = argparse.ArgumentParser(prog="gpuwm spectral")
    sub = parser.add_subparsers(dest="command", required=True)
    # Reuse the registered surface by creating an outer parser then selecting
    # its sole command.  This keeps python -m and gpuwm CLI behavior identical.
    register_cli(sub)
    arguments = parser.parse_args(
        ["spectral", *(sys.argv[1:] if argv is None else list(argv))])
    return int(arguments.func(arguments))


if __name__ == "__main__":  # pragma: no cover - exercised through CLI tests
    raise SystemExit(main())


__all__ = ["main", "register_cli"]
