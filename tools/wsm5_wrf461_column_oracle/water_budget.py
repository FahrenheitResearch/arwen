"""Column water budget of the WRF WSM5 arms over the oracle runs (CPU only).

Total water W = sum_k (qv+qc+qr+qi+qs)*den*dz + rain (kg m-2; rain is the
accumulated surface water in mm, which includes snow).  WSM5 has no water
source or sink: W after the run minus W before is the scheme's water
error.  Reported per arm (scalar = WRF's own build, defined = WOOF's
arithmetic) and per run: the worst column, and for the columns the remap
fix changes, stock's loss against the defined arm's.

usage: python water_budget.py ORACLE_DIR [--out JSON]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import oracle_io as io  # type: ignore
else:
    from . import oracle_io as io

WATER = ("qv", "qc", "qr", "qi", "qs")


def column_water(level, surf, den, dz):
    q = sum(np.asarray(level[n], np.float64) for n in WATER)
    return (q * den * dz).sum(axis=0) + np.asarray(surf["rain"], np.float64)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("oracle_dir", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    d = args.oracle_dir
    arms_report = json.loads((d / "arms.json").read_text()) if (d / "arms.json").exists() else None
    report = {}
    for set_name, dt, nsteps in io.RUNS:
        tag = io.tag_of(set_name, dt, nsteps)
        names = io.column_names(d, set_name)
        _, _, lv0, sf0 = io.read_set(d / f"{set_name}.bin")
        den = lv0["den"].astype(np.float64)
        dz = lv0["dz"].astype(np.float64)
        w0 = column_water(lv0, {"rain": sf0["rain"]}, den, dz)
        row = {}
        for arm in ("scalar", "defined"):
            run = io.read_wrf(d / f"wrf-{arm}-{tag}.bin")
            lv, sf = run["steps"][-1]
            err = column_water({**lv, "qv": lv["qv"]}, sf, den, dz) - w0
            i = int(np.argmax(np.abs(err)))
            row[arm] = {"worst_column": names[i], "worst_error_kg_m2": float(err[i]),
                        "sum_abs_error_kg_m2": float(np.abs(err).sum()),
                        "per_column": {n: float(e) for n, e in zip(names, err)}}
        if arms_report is not None:
            changed = arms_report["runs"][tag]["remap_vs_stock"]["columns"]
            row["remap_changed_columns"] = {
                n: {"stock": row["scalar"]["per_column"][n],
                    "defined": row["defined"]["per_column"][n]} for n in changed}
        report[tag] = row
        print(f"{tag}: worst |error| stock {abs(row['scalar']['worst_error_kg_m2']):.3e} "
              f"({row['scalar']['worst_column']}), defined "
              f"{abs(row['defined']['worst_error_kg_m2']):.3e} "
              f"({row['defined']['worst_column']}); sum |error| stock "
              f"{row['scalar']['sum_abs_error_kg_m2']:.3e} defined "
              f"{row['defined']['sum_abs_error_kg_m2']:.3e}", flush=True)
    if args.out:
        args.out.write_text(json.dumps(report, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
