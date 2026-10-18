"""Which of the fork's processes a column-oracle batch exercises.

usage:
    python fork_rate_coverage.py RATES_DIR OUT.json BATCH.npz [BATCH.npz ...]

RATES_DIR is build_rates.sh's instrumented fork driver.  Each BATCH.npz is
a gpu_column_oracle.py batch dump (its ``in_*`` arrays are the exact words
both codes received, its dt and ITIMESTEP are read from the file name
``batch-<set>-dt<dt>-step<n>.npz``).  For every process rate of the fork's
mp_thompson the report counts the cells where the fork's own Fortran made
it nonzero, so a 0-ULP claim states which physics it covered.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "thompson_real_column_parity"))

import instrument_fork_rates as IF  # noqa: E402
from fork_stage_parity import read_streams, run_fork_binary  # noqa: E402


def main(argv):
    rates_dir = Path(argv[1]).resolve()
    out = Path(argv[2])
    total = {name: 0 for name in IF.RATES}
    report = {"batches": {}}
    for path in argv[3:]:
        z = np.load(path)
        m = re.search(r"dt([0-9.]+)-step(\d+)", Path(path).name)
        dt, step = float(m.group(1)), int(m.group(2))
        inp = {k[3:]: z[k] for k in z.files if k.startswith("in_")}
        ncol, nz = inp["p"].shape
        run_fork_binary(rates_dir / "run_columns_fork", rates_dir, inp, dt)
        cps = read_streams(rates_dir, ncol, nz)
        counts = {}
        for name in IF.RATES:
            cp = "cp2" if name in IF.LATE_RATES else "cp1"
            v = cps[cp][name]
            counts[name] = int((np.isfinite(v) & (v != 0)).sum())
            total[name] += counts[name]
        report["batches"][Path(path).name] = {
            "ncol": ncol, "nz": nz, "dt": dt, "itimestep": step,
            "cells_with_microphysics": int(np.isfinite(
                cps["cp1"]["t1d"]).sum()),
            "nonzero_cells_per_rate": counts}
    report["total_nonzero_cells_per_rate"] = total
    report["rates_never_active"] = [k for k, v in total.items() if v == 0]
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps({"never_active": report["rates_never_active"],
                      "min_active": min(total.values())}, indent=1))


if __name__ == "__main__":
    main(sys.argv)
