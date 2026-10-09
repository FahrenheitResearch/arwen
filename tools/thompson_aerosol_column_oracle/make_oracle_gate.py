#!/usr/bin/env python3
"""Cut the committed whole-oracle gate out of a finished oracle run.

The gate (``tests/data/mp28_column_oracle_wrf461.npz``, read by
``tests/test_thompson_aerosol_column_oracle_gpu.py``) carries every column
of the run (its raw inputs), WRF v4.6.1's own answers for all 23 compared
words at each time step (``summary-strict-dtN.wrf.npz``, WRF run on the
Exner function and layer depths the GPU adapter formed), the Exner function
itself, and which columns WOOF's strict build gave bit-identical on all 23
fields.  Every other column must be one where rain meets graupel: the
declared rain-graupel divergence (WRF reads those tables out of bounds),
which the cutter checks against a mandatory run of the measurement copy
(``make_racg_read_copy.sh``): there every column must be
bit-identical, or the cut is refused.

usage: make_oracle_gate.py RUN_DIR OUT.npz --wrfread RUN_DIR [--dts DT ...]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import oracle_io as io  # noqa: E402

f32 = np.float32

#: Column inputs gpu_run.py reads.
INPUTS = ("p", "th", "geop", "w", *io.SPECIES, "nwfa2d", "nifa2d")


def identical(run: Path, dt: int, mode: str = "strict") -> np.ndarray:
    """Per column: all 23 compared words bit-identical to WRF."""
    gpu = np.load(run / f"gpu-{mode}-dt{dt}.npz")
    wrf = np.load(run / f"summary-{mode}-dt{dt}.wrf.npz")
    ok = None
    for name in io.FIELDS:
        same = (np.asarray(gpu[name], f32).view(np.int32)
                == np.asarray(wrf[name], f32).view(np.int32))
        same = same if same.ndim == 1 else same.all(axis=1)
        ok = same if ok is None else ok & same
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run")
    ap.add_argument("out")
    ap.add_argument("--wrfread", required=True, help="the same columns run on the "
                    "measurement copy; every column must be identical there")
    ap.add_argument("--dts", nargs="+", type=int, default=[20, 5])
    a = ap.parse_args(argv)
    run = Path(a.run)
    cols = io.load(run / "columns.npz")
    payload = {k: np.asarray(cols[k]) for k in INPUTS}
    payload["labels"] = np.asarray(cols["labels"])
    payload["dts"] = np.asarray(a.dts, np.int32)
    payload["fields"] = np.asarray(io.FIELDS)
    checks = {}
    for dt in a.dts:
        wrf = np.load(run / f"summary-strict-dt{dt}.wrf.npz")
        gpu = np.load(run / f"gpu-strict-dt{dt}.npz")
        same = identical(run, dt)
        if a.wrfread:
            other = Path(a.wrfread)
            labels = [str(x) for x in io.load(other / "columns.npz")["labels"]]
            if labels != [str(x) for x in cols["labels"]]:
                raise SystemExit("--wrfread ran a different column set")
            other_cols = io.load(other / "columns.npz")
            for name in INPUTS:
                if not np.array_equal(np.asarray(cols[name], f32).view(np.uint32),
                                      np.asarray(other_cols[name], f32).view(np.uint32)):
                    raise SystemExit(f"--wrfread has different input words: {name}")
            other_gpu = np.load(other / f"gpu-strict-dt{dt}.npz")
            for name in ("in_pii", "in_dz", "in_hgt"):
                if not np.array_equal(np.asarray(gpu[name], f32).view(np.uint32),
                                      np.asarray(other_gpu[name], f32).view(np.uint32)):
                    raise SystemExit(f"--wrfread has different adapter inputs: {name}")
            emulated = identical(other, dt)
            if not emulated.all():
                bad = [labels[i] for i in np.flatnonzero(~emulated)]
                raise SystemExit(f"dt={dt}: with WRF's read reproduced these "
                                 f"columns still differ: {bad}")
        payload[f"identical_dt{dt}"] = same
        payload[f"in_pii_dt{dt}"] = np.asarray(gpu["in_pii"], f32)
        for name in io.FIELDS:
            payload[f"wrf_{name}_dt{dt}"] = np.asarray(wrf[name], f32)
            payload[f"woof_{name}_dt{dt}"] = np.asarray(gpu[name], f32)
        checks[str(dt)] = {
            "columns_checked": int(emulated.size),
            "columns_identical": int(emulated.sum()),
            "measurement_gpu_sha256": hashlib.sha256(
                (other / f"gpu-strict-dt{dt}.npz").read_bytes()).hexdigest(),
            "measurement_wrf_sha256": hashlib.sha256(
                (other / f"summary-strict-dt{dt}.wrf.npz").read_bytes()).hexdigest()}
        print(f"dt={dt}: {int(same.sum())} of {same.size} columns "
              "bit-identical on all 23 words")
    payload["measurement_copy_checks"] = np.asarray(json.dumps(checks, sort_keys=True))
    np.savez_compressed(a.out, **payload)
    print(f"{a.out}: {len(payload['labels'])} columns")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
