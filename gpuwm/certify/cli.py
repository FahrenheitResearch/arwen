"""``gpuwm certify`` and ``gpuwm dual-run``, both fail-closed.

Registered from :mod:`gpuwm.cli` before the ``--explain`` sweep, so both
commands carry the same layering convention every other subcommand does.
Neither command has a ``--force``, a ``--warn-only`` or a tolerance flag: the
band is data, the refusals are the band's and the capsule's, and a reader who
wants a different answer edits the band and says so in its provenance.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from gpuwm.certify.band import write_certification_json


def register_cli(subparsers: argparse._SubParsersAction) -> None:
    """Attach the two certification commands to the gpuwm parser."""
    certify = subparsers.add_parser(
        "certify",
        help="check one run capsule and its matched-comparison metrics "
             "against an acceptance band and a WRF reference manifest")
    certify.add_argument("--run-capsule", type=Path, required=True,
                         metavar="CAPSULE",
                         help="certification-capsule.json written by the run")
    certify.add_argument("--metrics-csv", type=Path, required=True,
                         metavar="CSV",
                         help="matched-comparison metrics CSV for the run")
    certify.add_argument("--band", type=Path, required=True, metavar="BAND",
                         help="acceptance band for this configuration; it is "
                              "addressed by the configuration's SHA-256, and "
                              "certify refuses a band keyed to another one")
    certify.add_argument("--wrf-reference-manifest", type=Path, required=True,
                         metavar="MANIFEST",
                         help="WRF reference manifest naming the executable, "
                              "build recipe, namelists and reference wrfout "
                              "bytes the comparison was made against")
    certify.add_argument("--out-verdict", type=Path, default=None,
                         metavar="VERDICT",
                         help="write the verdict document here (it is "
                              "printed either way)")
    certify.set_defaults(func=_certify_main)

    dual = subparsers.add_parser(
        "dual-run",
        help="compare two certification capsules field for field and name "
             "the first field they disagree on")
    dual.add_argument("--capsule-a", type=Path, required=True,
                      metavar="CAPSULE")
    dual.add_argument("--capsule-b", type=Path, required=True,
                      metavar="CAPSULE")
    dual.add_argument("--out-report", type=Path, default=None,
                      metavar="REPORT",
                      help="write the comparison document here")
    dual.set_defaults(func=_dual_run_main)


def _certify_main(args) -> int:
    from gpuwm.certify.verdict import certify, failing_conditions

    verdict = certify(
        capsule_path=args.run_capsule, metrics_csv=args.metrics_csv,
        band_path=args.band,
        wrf_reference_manifest=args.wrf_reference_manifest)
    if args.out_verdict is not None:
        temporary = args.out_verdict.with_suffix(
            args.out_verdict.suffix + ".partial")
        write_certification_json(temporary, verdict)
        temporary.replace(args.out_verdict)
    refused = failing_conditions(verdict)
    if not refused:
        print(f"certify: PASS ({len(verdict['comparisons'])} banded "
              f"comparisons, band provenance {verdict['band']['provenance']}, "
              f"band schema {verdict['band']['band_schema_version']})")
        return 0
    for item in refused:
        print(f"certify: REFUSED {item['condition']}: {item['detail']}",
              file=sys.stderr)
    return 1


def _dual_run_main(args) -> int:
    from gpuwm.certify.dualrun import (
        compare_capsule_files, input_bytes_divergence_note)

    comparison = compare_capsule_files(args.capsule_a, args.capsule_b)
    if args.out_report is not None:
        write_certification_json(args.out_report, comparison.report())
    if comparison.identical:
        # The count is not decoration.  "capsules are identical field for
        # field" is the same sentence whether the screen compared a
        # hundred quantities or one, and this command is the project's
        # only detector for silent VRAM corruption on a card with no ECC.
        # A reader has to be able to see that it did work.
        print("dual-run: capsules are identical field for field "
              f"({comparison.compared_count} compared quantities)")
        return 0
    print(f"dual-run: first divergent field "
          f"{comparison.first_divergent_field}", file=sys.stderr)
    for divergence in comparison.divergences:
        print(f"dual-run:   {divergence.describe()}", file=sys.stderr)
    # Still exit 1 -- the divergence is real and is not being excused.  What
    # this adds is what it usually means, because the one cause an operator
    # cannot see from a leaf name is that they prepared twice.
    note = input_bytes_divergence_note(comparison.divergences)
    if note is not None:
        print(f"dual-run: {note}", file=sys.stderr)
    return 1


__all__ = ["register_cli"]
