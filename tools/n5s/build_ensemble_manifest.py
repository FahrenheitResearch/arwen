#!/usr/bin/env python3
"""Score CPU-WRF member pairs and emit evaluator-ready n5s-ensemble.json."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gpuwm.verify.n5s_metrics import build_ensemble_evidence, load_registration


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ensemble-root", type=Path, required=True)
    parser.add_argument("--restored-inputs", type=Path, required=True)
    parser.add_argument("--registration", type=Path)
    parser.add_argument("--unperturbed", type=Path)
    parser.add_argument("--member", type=Path, action="append", dest="members")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.ensemble_root
    registration_path = args.registration or root / "n5s-preregistration.json"
    unperturbed = args.unperturbed or root / "unperturbed"
    members = (args.members if args.members is not None
               else sorted(root.glob("member-*")))
    build_ensemble_evidence(
        ensemble_root=root, unperturbed_directory=unperturbed,
        member_directories=members,
        registration=load_registration(registration_path),
        restored_inputs=args.restored_inputs,
        output=args.output or root / "n5s-ensemble.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
