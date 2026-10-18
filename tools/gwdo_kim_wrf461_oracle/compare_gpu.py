#!/usr/bin/env python3
"""Grade the gwd_opt = 1 GPU kernel against the WRF v4.6.1 column oracle.

    python compare_gpu.py BUILD_DIR [--reference O0|O2vec]
                          [--write-fixture DIR] [--json OUT]

BUILD_DIR is build.sh's output directory.  For each vertical grid it runs
the production launcher (gpuwm.core.terrain_drag.launch_gwdo) over every
(dx, dt) case against the -O0 reference of record (or the stock-flags build
with ``--reference O2vec``) and prints one JSON document.  Set
GPUWM_WRF_EXACT=1 for the strict build; leave it unset for default
arithmetic.  ``--write-fixture`` also writes the ``.npz`` pairs the pytest
gate reads (tests/data/oracles/gwdo_kim/).  Exit status 0 only when every
word of every grid is bit-identical.
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

from gpuwm.verify import gwdo_kim_oracle as go  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("build_dir", type=Path)
    ap.add_argument("--reference", default="O0", choices=("O0", "O2vec"))
    ap.add_argument("--write-fixture", type=Path)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    import cupy as cp
    dev = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    report = {
        "strict": os.environ.get("GPUWM_WRF_EXACT", "0") == "1",
        "reference": args.reference,
        "device": dev["name"].decode(),
        "nvrtc": ".".join(map(str, cp.cuda.nvrtc.getVersion())),
        "grids": {},
    }
    exact = True
    for grid in go.GRIDS:
        inputs = go.read_inputs_bin(args.build_dir
                                    / f"gwdo-kim-inputs-{grid}.bin")
        with np.load(args.build_dir / f"gwdo-kim-inputs-{grid}.npz") as f:
            for k in inputs:
                if f[k].tobytes() != inputs[k].tobytes():
                    raise SystemExit(f"{grid}: npz and bin inputs disagree "
                                     f"on {k}")
            inputs["regime"] = f["regime"]
            inputs["edge_notes"] = f["edge_notes"]
        nz, ncol = inputs["u"].shape
        ncase = inputs["dx"].size
        wrf = go.read_outputs_bin(
            args.build_dir / f"gwdo-kim-wrf-{args.reference}-{grid}.bin",
            nz, ncol, ncase)
        if args.write_fixture:
            args.write_fixture.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.write_fixture
                                / f"gwdo-kim-inputs-{grid}.npz", **inputs)
            np.savez_compressed(args.write_fixture
                                / f"gwdo-kim-wrf461-{grid}.npz", **wrf)
        got = go.run_gpu(inputs)
        table = go.compare(got, wrf, inputs["regime"])
        exact &= all(row["not_bit_identical"] == 0 for row in table.values())
        report["grids"][grid] = {
            "columns": ncol, "levels": nz,
            "cases": [{"dx": float(a), "dt": float(b)}
                      for a, b in zip(inputs["dx"], inputs["dt"])],
            "regimes": sorted(set(map(str, inputs["regime"]))),
            "fields": table,
            "activity": go.activity(inputs, wrf),
        }
    report["bitwise_everywhere"] = bool(exact)
    text = json.dumps(report, indent=2)
    print(text)
    if args.json:
        args.json.write_text(text + "\n", encoding="utf-8")
    return 0 if exact else 1


if __name__ == "__main__":
    raise SystemExit(main())
