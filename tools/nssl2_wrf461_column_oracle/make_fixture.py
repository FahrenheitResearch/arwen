"""Pack a column-oracle run into the test fixture the GPU gates read.

  make_fixture.py RUN_PREFIX OUT.npz [--per-regime N] [--no-stages]

RUN_PREFIX is a run_oracle.sh output prefix (e.g. runs/r4/dt12.0_it2_exact).
The fixture holds, for the first N columns of every regime:

* ``in_*``: the microphysics-time inputs both engines started from
  (WOOF-prepared theta/rho/Exner/dz, pressure, staggered w, the 16 moments,
  the four precipitation accumulators);
* ``col_rho``, ``col_dz``: the raw column values the production step
  prepares those two inputs from;
* ``s{1,2,3}_*``: WRF's own driver slab at the three stage boundaries
  (the dump build of the reference) in concentration space -- theta, the 16
  moments, t0, t7 (omitted with --no-stages);
* ``out_*``: the reference's final outputs (compare_bitwise.OUT3/OUT2): WRF
  with the four documented defects corrected (wrf_bugfix.py), scalar libm,
  the build run_oracle.sh grades WOOF against.

Only WRF-side values and shared inputs are stored, so the fixture does not
change when WOOF does.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from compare_bitwise import OUT2, OUT3, read_fortran
from compare_stages import WOOF_FIELDS, read_dumps
from run_gpu_columns import FORTRAN_INPUT


def main():
    p = argparse.ArgumentParser()
    p.add_argument("prefix", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--per-regime", type=int, default=2)
    p.add_argument("--no-stages", action="store_true")
    a = p.parse_args()
    base = a.prefix.parent
    cols = np.load(base / "columns.npz")
    regime = cols["regime"]
    ncol, nz = cols["theta"].shape
    f = lambda s: a.prefix.with_name(a.prefix.name + s)
    meta = json.loads(f(".gpu.json").read_text())

    pick = []
    for r in dict.fromkeys(regime.tolist()):
        pick += [int(i) for i in np.nonzero(regime == r)[0][:a.per_regime]]
    pick = np.asarray(pick)

    fin = np.load(f(".fin.npz"))
    stages = None if a.no_stages else read_dumps(f(".wrf_stages.bin"), ncol)
    final = read_fortran(f(".wrf_fixed.bin"), ncol, nz)
    arrays = {"columns": pick, "regime": regime[pick],
              "dt_s": np.float32(meta["dt_s"]), "itimestep": np.int32(meta["itimestep"])}
    for name in FORTRAN_INPUT:
        arrays[f"in_{name}"] = fin[name][pick]
    arrays["in_w"] = fin["w"][pick]
    for name in ("rainnc", "snownc", "grplnc", "hailnc"):
        arrays[f"in_{name}"] = cols[name][pick]
    # The raw column rho and dz the production step starts from: WOOF
    # prepares rho = 1/alt and dz from the geopotential, which need not
    # round-trip to the prepared in_rho / in_dz bit for bit.
    for name in ("rho", "dz"):
        arrays[f"col_{name}"] = cols[name][pick]
    for stage in (() if stages is None else (1, 2, 3)):
        for name in ("theta", "t0", "t7") + WOOF_FIELDS:
            arrays[f"s{stage}_{name}"] = stages[stage][name][pick]
    for name in OUT3 + OUT2:
        arrays[f"out_{name}"] = final[name][pick]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **arrays)
    prov = {
        "schema": "woof.nssl2.column-oracle-fixture/v2",
        "source_run": str(a.prefix), "columns": pick.tolist(),
        "regimes": regime[pick].tolist(), "dt_s": meta["dt_s"],
        "itimestep": meta["itimestep"], "column_set_seed": int(cols["seed"]),
        "wrf_commit": "d66e442fccc04111067e29274c9f9eaccc3cef28",
        "wrf_module_sha256_lf": "1eb1b138b75ff3b0cfe33c23779f4ec9b72e57a5455a53ef11c9e55ae0f42722",
        "wrf_reference_build": "WRF serial em_real preprocessed module_mp_nssl_2mom.f90, "
                               "the four documented defects corrected by wrf_bugfix.py, "
                               "recompiled with WRF's GNU flags plus -fno-tree-vectorize",
        "stages": not a.no_stages,
        "fixture_sha256": hashlib.sha256(a.out.read_bytes()).hexdigest(),
    }
    a.out.with_suffix(".json").write_text(json.dumps(prov, indent=2) + "\n")
    print(json.dumps(prov))


if __name__ == "__main__":
    main()
