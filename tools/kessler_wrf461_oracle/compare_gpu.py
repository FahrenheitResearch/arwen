#!/usr/bin/env python3
"""Grade the Kessler GPU kernel against the WRF v4.6.1 column oracle.

    python compare_gpu.py BUILD_DIR [--write-fixture DIR] [--json OUT]

BUILD_DIR is build.sh's output directory.  The comparison runs the
production launcher (gpuwm.core.microphysics.launch_kessler) in both modes
(free running and per-call replay) against the -O0 reference of record, and
prints one JSON document.  Set GPUWM_WRF_EXACT=1 for the strict build; leave
it unset for default arithmetic.  ``--write-fixture`` also writes the two
``.npz`` files the pytest gate reads (tests/data/oracles/kessler/).
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

from gpuwm.verify import kessler_oracle as ko  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("build_dir", type=Path)
    ap.add_argument("--reference", default="kessler-wrf-O0.bin")
    ap.add_argument("--write-fixture", type=Path)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    inputs = ko.read_inputs_bin(args.build_dir / "kessler-inputs.bin")
    with np.load(args.build_dir / "kessler-inputs.npz") as f:
        for k in ko.INPUT_FIELDS + ("rainnc",):
            if f[k].tobytes() != inputs[k].tobytes():
                raise SystemExit(f"npz and bin inputs disagree on {k}")
        inputs["regime"] = f["regime"]
    nz, ncol = inputs["t"].shape
    total = int(inputs["nsteps"].sum())
    wrf = ko.read_outputs_bin(args.build_dir / args.reference, nz, ncol,
                              total)
    if args.write_fixture:
        args.write_fixture.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.write_fixture / "kessler-inputs.npz",
                            **inputs)
        np.savez_compressed(args.write_fixture / "kessler-wrf461.npz", **wrf)

    import cupy as cp
    dev = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    report = {
        "strict": os.environ.get("GPUWM_WRF_EXACT", "0") == "1",
        "reference": args.reference,
        "device": dev["name"].decode(),
        "nvrtc": ".".join(map(str, cp.cuda.nvrtc.getVersion())),
        "columns": ncol, "levels": nz,
        "cases": [{"dt": float(d), "steps": int(s)}
                  for d, s in zip(inputs["dt"], inputs["nsteps"])],
        "regimes": sorted(set(map(str, inputs["regime"]))),
    }
    for mode in ("free", "replay"):
        got = ko.run_gpu(inputs, mode=mode, wrf=wrf)
        report[mode] = ko.compare(got, wrf, inputs["regime"])
    text = json.dumps(report, indent=2)
    print(text)
    if args.json:
        args.json.write_text(text + "\n", encoding="utf-8")
    exact = all(row["not_bit_identical"] == 0
                for mode in ("free", "replay") for row in report[mode].values())
    return 0 if exact else 1


if __name__ == "__main__":
    raise SystemExit(main())
