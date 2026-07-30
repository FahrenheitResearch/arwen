#!/usr/bin/env python3
"""Audit an ERA5 native-preprocessing proof and its artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gpuwm.preprocess_receipt_audit import audit_era5_preprocess_receipt


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proof", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--cpu-bridge", type=Path)
    parser.add_argument(
        "--receipt-only", action="store_true",
        help="inspect archived receipt/artifacts without matching this runtime",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = audit_era5_preprocess_receipt(
        args.proof, output_root=args.output_root,
        verify_runtime=not args.receipt_only,
        runtime_root=args.runtime_root, cpu_bridge=args.cpu_bridge)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        if args.output.exists():
            parser.error(f"refusing to overwrite {args.output}")
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
