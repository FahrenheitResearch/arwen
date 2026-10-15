#!/usr/bin/env python3
"""Grade WOOF's Dudhia shortwave against the WRF v4.6.1 column oracle.

    python compare_gpu.py BUILD_DIR [--layout columns|adapter|both]
                          [--write-fixture DIR] [--json OUT]

BUILD_DIR is build.sh's output directory.  Every output word (theta
tendency RTHRATEN, GSW and the driver's SWDOWN) is compared bitwise against
the -O0 reference of record, and one JSON document is printed.  Set
GPUWM_WRF_EXACT=1 for the strict build; leave it unset for default
arithmetic.  ``--write-fixture`` also writes the two ``.npz`` files the
pytest gate reads (tests/data/oracles/dudhia_sw/).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gpuwm.verify import dudhia_sw_oracle as do  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("build_dir", type=Path)
    ap.add_argument("--reference", default="dudhia-wrf-O0.bin")
    ap.add_argument("--layout", default="both",
                    choices=("columns", "adapter", "both"))
    ap.add_argument("--write-fixture", type=Path)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    inputs = do.read_inputs_bin(args.build_dir / "dudhia-inputs.bin")
    with np.load(args.build_dir / "dudhia-inputs.npz") as f:
        for k in do.INPUT_FIELDS + ("coszen", "albedo", "solcon",
                                    "swrad_scat", "icloud"):
            if f[k].tobytes() != inputs[k].tobytes():
                raise SystemExit(f"npz and bin inputs disagree on {k}")
        inputs["regime"] = f["regime"]
    nz, ncol = inputs["t"].shape
    ncase = inputs["solcon"].size
    wrf = do.read_outputs_bin(args.build_dir / args.reference, nz, ncol,
                              ncase)
    if args.write_fixture:
        args.write_fixture.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.write_fixture / "dudhia-inputs.npz",
                            **inputs)
        np.savez_compressed(args.write_fixture / "dudhia-wrf461.npz", **wrf)

    import cupy as cp
    dev = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    report = {
        "strict": os.environ.get("GPUWM_WRF_EXACT", "0") == "1",
        "reference": args.reference,
        "device": dev["name"].decode(),
        "nvrtc": ".".join(map(str, cp.cuda.nvrtc.getVersion())),
        "columns": ncol, "levels": nz,
        "cases": [{"solcon": float(s), "swrad_scat": float(w),
                   "icloud": int(i)} for s, w, i in
                  zip(inputs["solcon"], inputs["swrad_scat"],
                      inputs["icloud"])],
        "regimes": sorted(set(map(str, inputs["regime"]))),
    }
    layouts = (("columns", "adapter") if args.layout == "both"
               else (args.layout,))
    for layout in layouts:
        got = do.run_gpu(inputs, layout=layout)
        report[layout] = do.compare(got, wrf, inputs["regime"])
    text = json.dumps(report, indent=2)
    print(text)
    if args.json:
        args.json.write_text(text + "\n", encoding="utf-8")
    exact = all(row["not_bit_identical"] == 0
                for layout in layouts for row in report[layout].values())
    return 0 if exact else 1


if __name__ == "__main__":
    raise SystemExit(main())
