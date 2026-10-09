#!/usr/bin/env python3
"""Pack one oracle case's WRF side into the regression fixture.

    pack_fixture.py CASE_DIR OUT.npz [--arm defined]

Stores what WRF v4.6.1 wrote (``outputs-<arm>.bin``: the RUCLSMINIT words
and every compared field after every step, with the single-count SFCEVP),
the sha256 of the ``inputs.bin`` both sides read, and the case options, so
``tests/test_ruc_gpu_column_oracle.py`` can rerun the WOOF GPU side on a box
with no Fortran and grade it word for word.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import layout  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("case_dir", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--arm", default="defined")
    args = ap.parse_args(argv)
    meta = json.loads((args.case_dir / "case.json").read_text())
    wrf = layout.read_outputs(args.case_dir / f"outputs-{args.arm}.bin",
                              meta["ncol"], meta["nzs"], meta["nsteps"])
    arrays = {f"init__{k}": v for k, v in wrf["init"].items()}
    for k, step in enumerate(wrf["steps"], start=1):
        for name, value in step.items():
            arrays[f"step{k}__{name}"] = value
    digest = hashlib.sha256((args.case_dir / "inputs.bin").read_bytes()).hexdigest()
    options = {key: meta[key] for key in ("ncol", "nzs", "nsteps", "dt",
                                          "fractional_seaice", "mosaic",
                                          "lakemodel", "rdlai2d")}
    receipt = {"inputs_sha256": digest, "options": options, "arm": args.arm,
               "regime": meta["regime"],
               "wrf": "WRF v4.6.1 phys/module_sf_ruclsm.F + "
                      "module_sf_sfcdiags_ruclsm.F, gfortran 13.3.0 / glibc "
                      "2.39, WRF GNU flags + -finit-local-zero"}
    arrays["receipt"] = np.frombuffer(json.dumps(receipt).encode(), np.uint8)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **arrays)
    print(args.out, args.out.stat().st_size, "bytes", digest)


if __name__ == "__main__":
    main()
