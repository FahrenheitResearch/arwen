"""Grade WOOF against the SPP column oracle and write one JSON receipt.

    python tools/spp_wrf461_column_oracle/compare.py --output result.json \
        [--fixtures tests/data/oracles/spp/columns] [--cpu]

Run on the GPU card the mutex granted.  The arithmetic mode is whatever the
environment says: set GPUWM_WRF_EXACT=1 for the strict build, leave it unset
for default arithmetic.  ``--cpu`` also grades the NumPy MYNN reference.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import spp_column_fixtures as columns  # noqa: E402
import spp_column_grade as grade  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, default=columns.FIXTURE_DIR)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    import numpy as np
    meta = {"python": platform.python_version(), "numpy": np.__version__,
            "libc": list(platform.libc_ver()),
            "GPUWM_WRF_EXACT": os.environ.get("GPUWM_WRF_EXACT", ""),
            "fixtures": {p.name: columns.sha256(p) for p in sorted(args.fixtures.glob("*.npz"))}}
    import cupy as cp
    device = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
    meta.update(cupy=cp.__version__, cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),
                nvrtc=list(cp.cuda.nvrtc.getVersion()), device=device["name"].decode())
    result = {"metadata": meta, "mynn": {}, "gf": {}}
    backends = ("gpu", "cpu") if args.cpu else ("gpu",)
    for backend in backends:
        for mixlength in (1, 2):
            for replay in (True, False):
                key = f"{backend}-ml{mixlength}-{'replay' if replay else 'free'}"
                graded = grade.grade_mynn(args.fixtures / f"mynn-spp-ml{mixlength}.npz",
                                          mixlength, replay=replay, backend=backend)
                result["mynn"][key] = graded
                print(f"MYNN {key}: max {graded['max_ulp']} ULP, "
                      f"{graded['words_differ']}/{graded['words']} words differ", flush=True)
    for path in sorted(args.fixtures.glob("gf-spp-o*.npz")):
        graded = grade.grade_gf(path)
        result["gf"][path.stem] = graded
        print(f"GF {path.stem}: max {graded['max_ulp']} ULP, "
              f"{graded['words_differ']}/{graded['words']} words differ "
              f"({graded['converging_deep']} converging, {graded['clipped_columns']} clipped)",
              flush=True)
    result["gf_zero_pattern_identity_words_differ"] = grade.gf_zero_pattern_identity()
    args.output.write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
