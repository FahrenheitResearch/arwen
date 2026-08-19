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
