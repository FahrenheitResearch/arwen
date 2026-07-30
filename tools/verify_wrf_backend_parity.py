#!/usr/bin/env python3
"""Write a recomputed semantic CPU/CUDA native-WRF parity report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gpuwm.wrf_backend_parity import compare_wrf_backend_directories


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    result = compare_wrf_backend_directories(
        args.reference, args.candidate)
    temporary = args.output.with_name(args.output.name + ".partial")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({
        "status": result["status"],
        "files": {
            name: {"status": value["status"],
                   "failed_fields": value.get("failed_fields", [])}
            for name, value in result.get("files", {}).items()
        },
    }, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
