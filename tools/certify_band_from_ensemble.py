#!/usr/bin/env python3
"""Reduce an ensemble envelope receipt to the block an acceptance band cites.

The reducer's first job is refusal.  Every member declares the SHA-256 of
every initialization file it consumed, and a member whose set is not the
preserved one is refused -- not warned about, refused, before any number is
reduced.  A member built from a different base state carries a deterministic
initialization offset, and that offset would be absorbed into the measured
spread and would widen it.  A widened band is exactly what this program's
standing rule forbids, so provenance is enforced here rather than trusted.

What it emits is a band-input block: the interval statistic as the
registration words it, the member count, per-member config digests, the
pair-score artifact digest, and the per-row envelope with its degeneracy
flag.  The block is labelled internal scope and records digests only, never
a retrieval path, because member artifacts are not redistributable.

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


def reduce_from_files(*, receipt_path: str | Path, members_path: str | Path,
                      required_inputs_path: str | Path,
                      pair_scores_path: str | Path) -> dict[str, object]:
    receipt = _load(receipt_path)
    if receipt.get("schema") != chaos_envelope.ENVELOPE_SCHEMA:
        raise ValueError(
            f"{receipt_path} is not a {chaos_envelope.ENVELOPE_SCHEMA} receipt")
    members = _load(members_path)
    if isinstance(members, dict):
        members = members.get("members", [])
    admitted = chaos_envelope.admit_members(
        members, required_input_sha256=_load(required_inputs_path))
    return chaos_envelope.reduce_band(
        receipt, members=admitted,
        pair_score_sha256=chaos_envelope.sha256_file(pair_scores_path))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True,
                        help="envelope receipt from matched_wrfout_envelope")
    parser.add_argument("--members", type=Path, required=True,
                        help="member records carrying input_sha256 and config_digest")
    parser.add_argument("--required-inputs", type=Path, required=True,
                        help="the preserved initialization SHA-256 set")
    parser.add_argument("--pair-scores", type=Path, required=True,
                        help="pair-score artifact the band digest binds")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        band = reduce_from_files(
            receipt_path=args.receipt, members_path=args.members,
            required_inputs_path=args.required_inputs,
            pair_scores_path=args.pair_scores)
    except ValueError as exc:
        print(f"band reduction refused: {exc}", file=sys.stderr)
        return 1
    chaos_envelope.write_json(args.output, band)
    print(f"band provenance {band['provenance']} "
          f"members {band['member_count']} rows {len(band['rows'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
