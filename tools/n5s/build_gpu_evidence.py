#!/usr/bin/env python3
"""Score gpuwm against unperturbed WRF and emit N5S-gpu-evidence.json."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gpuwm.verify.n5s_metrics import build_gpu_evidence, load_registration


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-dir", type=Path, required=True)
    parser.add_argument("--unperturbed-dir", type=Path, required=True)
    parser.add_argument("--restored-inputs", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build_gpu_evidence(
        gpu_directory=args.gpu_dir,
        unperturbed_directory=args.unperturbed_dir,
        registration=load_registration(args.registration),
        restored_inputs=args.restored_inputs, output=args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
