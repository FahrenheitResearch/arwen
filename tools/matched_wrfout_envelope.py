#!/usr/bin/env python3
"""Build and audit a chaos envelope over a set of matched wrfout runs.

Three commands, in the order they are meant to be used:

``register``
    Write the metric pins and their hash BEFORE any member output exists.
    The receipt carries this hash, so a pin edited after the fact is visible.
    The spectral pins are not among the arguments: they are pre-registered in
    ``gpuwm.verify.spectral`` and are folded into the registration, whose hash
    then covers them.  Both hashes are printed so an operator can record that
    they predate the run.

``build``
    Score every unordered member pair and the candidate against the
    unperturbed run, and emit the envelope receipt: per (metric, domain,
    lead) the pair count, the nearest-rank E-percentile, a degeneracy flag,
    the candidate distance and the verdict.

``check-identity``
    Refuse to let a receipt be cited beside a published run it does not
    belong to.  It compares the receipt's recorded member geometry and
    physics identity against the run's config, loaded through the production
    experiment loader, and against the spacings the public document states.
    Exits non-zero on any mismatch -- a receipt measured under some other
    campaign's pins cannot pass, which is the point.

CPU-only: no CuPy import on any path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gpuwm.verify import chaos_envelope  # noqa: E402


def _load(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _register(args: argparse.Namespace) -> int:
    domain_dx = dict(
        (name, float(value))
        for name, _, value in (item.partition("=") for item in args.domain))
    registration = chaos_envelope.make_registration(
        start_time=args.start_time,
        domain_dx_m=domain_dx,
        state_fields=args.state_field,
        leads_seconds=args.lead_seconds,
        cadence_seconds=args.cadence_seconds,
        reflectivity_field=args.reflectivity_field,
        reflectivity_threshold=args.reflectivity_threshold,
        low_pass_physical_width_m=args.low_pass_width_m,
        low_pass_interior_exclusion_cells=args.low_pass_exclusion_cells,
        boundary_width_cells=args.boundary_width_cells,
        fss_radius_m=args.fss_radius_m,
        object_min_area_km2=args.object_min_area_km2,
        object_connectivity=args.object_connectivity,
        evaluator_commit=args.evaluator_commit,
        envelope_percentile=args.envelope_percentile)
    chaos_envelope.write_json(args.output, registration)
    print(f"registration_sha256 {registration['registration_sha256']}")
    print("spectral_pins_sha256 "
          f"{registration['parameters']['spectral_pins_sha256']}")
    return 0


def _build(args: argparse.Namespace) -> int:
    receipt = chaos_envelope.build_envelope(
        registration=_load(args.registration),
        member_directories=args.member,
        candidate_directory=args.candidate,
        unperturbed_directory=args.unperturbed,
        member_identity=(chaos_envelope.config_identity(args.config)
                         if args.config else _load(args.identity)),
        output=args.output)
    print(f"rows {len(receipt['rows'])} "
          f"degenerate {receipt['degenerate_rows']} "
          f"outside {receipt['outside_rows']} "
          f"pairs {receipt['pair_count']}")
    return 0


def _check_identity(args: argparse.Namespace) -> int:
    issues = chaos_envelope.check_config_identity(
        _load(args.receipt), args.config, args.document)
    for issue in issues:
        print(f"identity mismatch: {issue}", file=sys.stderr)
    if issues:
        print(f"{len(issues)} mismatch(es); this receipt does not belong to "
              f"{Path(args.config).name}", file=sys.stderr)
        return 1
    print(f"receipt identity matches {Path(args.config).name}")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    register = commands.add_parser("register", help="pin metrics up front")
    register.add_argument("--start-time", required=True)
    register.add_argument("--domain", action="append", required=True,
                          metavar="NAME=DX_METRES")
    register.add_argument("--state-field", action="append", required=True)
    register.add_argument("--lead-seconds", action="append", type=int,
                          required=True)
    register.add_argument("--cadence-seconds", type=int, required=True)
    register.add_argument("--reflectivity-field", required=True)
    register.add_argument("--reflectivity-threshold", type=float, required=True)
    register.add_argument("--low-pass-width-m", type=float, required=True)
    register.add_argument("--low-pass-exclusion-cells", type=int, required=True)
    register.add_argument("--boundary-width-cells", type=int, required=True)
    register.add_argument("--fss-radius-m", type=float, required=True)
    register.add_argument("--object-min-area-km2", type=float, required=True)
    register.add_argument("--object-connectivity", type=int, default=8)
    register.add_argument("--envelope-percentile", type=float, default=95.0)
    register.add_argument("--evaluator-commit", required=True)
    register.add_argument("--output", type=Path, required=True)
    register.set_defaults(handler=_register)

    build = commands.add_parser("build", help="score pairs and reduce")
    build.add_argument("--registration", type=Path, required=True)
    build.add_argument("--member", action="append", required=True)
    build.add_argument("--candidate", required=True)
    build.add_argument("--unperturbed", required=True)
    identity = build.add_mutually_exclusive_group(required=True)
    identity.add_argument("--config", help="run config the members share")
    identity.add_argument("--identity", type=Path,
                          help="pre-extracted member identity document")
    build.add_argument("--output", type=Path, required=True)
    build.set_defaults(handler=_build)

    check = commands.add_parser(
        "check-identity", help="does this receipt belong to that run?")
    check.add_argument("--receipt", type=Path, required=True)
    check.add_argument("--config", required=True)
    check.add_argument("--document", type=Path, default=None)
    check.set_defaults(handler=_check_identity)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
