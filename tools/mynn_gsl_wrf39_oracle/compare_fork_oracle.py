"""Compare gpuwm's gsl_wrf39 MYNN surface layer with the fork oracle.

    python compare_fork_oracle.py DECK.txt ORACLE.csv[.gz] [--kernel] [--json OUT]

Runs the CPU reference (and, with ``--kernel``, the CUDA kernel) over the
deck's columns for the oracle's steps, carrying UST, USTM, MOL, QSFC, ZNT,
HFX and QFX between steps exactly as the Fortran harness does, and reports
per output the FP32 ULP distance from the oracle: maximum, median, and the
count of bitwise-equal words.  It also reports how far the WRF v4.6.1 form
(the default variant) sits from the same oracle on the same columns, which
is the size of the change the variant makes.  Imported by
tests/test_mynn_sfclay_gsl_wrf39.py.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_columns import columns, load_columns  # noqa: E402

OUTPUTS = ("regime", "zol", "rmol", "ust", "ustm", "mol", "psim", "psih",
           "chs", "chs2", "cqs2", "ch", "flhc", "flqc", "qgh", "qsfc", "hfx",
           "qfx", "lh", "u10", "v10", "th2", "t2", "q2", "gz1oz0", "wspd",
           "br", "ck", "cka", "cd", "cda", "wstar", "qstar", "cpm", "znt")


def load_oracle(path) -> dict[int, dict[str, np.ndarray]]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="ascii") as fh:
        rows = list(csv.DictReader(fh))
    steps: dict[int, dict[str, list]] = {}
    for row in rows:
        step = steps.setdefault(int(row["step"]), {})
        for key, value in row.items():
            if key in ("step", "column"):
                continue
            step.setdefault(key, []).append(float(value))
    return {s: {k: np.asarray(v, dtype=np.float64).astype(np.float32)
                for k, v in d.items()} for s, d in steps.items()}


def _carry(values, mol, ustm, out):
    values = dict(values)
    for name in ("hfx", "qfx", "znt", "qsfc", "ust"):
        values[name] = np.asarray(out[name], dtype=np.float32)
    return values, np.asarray(out["mol"], np.float32), np.asarray(
        out["ustm"], np.float32)


def run_reference(deck, nstep, variant):
    from gpuwm.core.mynn_surface import mynn_surface_layer_default
    _, cols = load_columns(deck)
    values = {k: v for k, v in cols.items() if k != "mol"}
    mol, ustm = cols["mol"].copy(), cols["ust"].copy()
    steps = {}
    for step in range(1, nstep + 1):
        out = mynn_surface_layer_default(
            values, dx=3000.0, itimestep=step, isfflx=1, isftcflx=0,
            mol=mol, ustm=ustm, variant=variant)
        steps[step] = {k: np.asarray(out[k], np.float32) for k in OUTPUTS}
        values, mol, ustm = _carry(values, mol, ustm, out)
    return steps


def run_kernel(deck, nstep, variant):
    import cupy as cp
    from gpuwm.core.mynn_sfclay import (
        MYNN_SURFACE_INPUTS, _allocate_result, launch_mynn_surface_layer)
    _, cols = load_columns(deck)
    n = cols["u1"].size
    shape = (1, n)
    dev = {k: cp.asarray(cols[k].reshape(shape)) for k in MYNN_SURFACE_INPUTS}
    mol = cp.asarray(cols["mol"].reshape(shape))
    ustm = cp.asarray(cols["ust"].reshape(shape))
    steps = {}
    for step in range(1, nstep + 1):
        result = _allocate_result(shape)
        launch_mynn_surface_layer(dev, mol, ustm, result, dx=3000.0,
                                  itimestep=step, isfflx=1, isftcflx=0,
                                  variant=variant)
        out = {k: cp.asnumpy(getattr(result, k)).reshape(n) for k in OUTPUTS}
        steps[step] = out
        for name in ("hfx", "qfx", "znt", "qsfc", "ust"):
            dev[name] = cp.ascontiguousarray(getattr(result, name))
        mol = cp.ascontiguousarray(result.mol)
        ustm = cp.ascontiguousarray(result.ustm)
    return steps


def ulp_table(got, want) -> dict:
    from gpuwm.core.fp32_ulp import fp32_ulp_distance
    table = {}
    for name in OUTPUTS:
        dist = np.concatenate([
            fp32_ulp_distance(got[s][name], want[s][name]).astype(np.int64)
            for s in sorted(want)])
        table[name] = {"max": int(dist.max()), "median": float(np.median(dist)),
                       "exact": int((dist == 0).sum()), "n": int(dist.size)}
    return table


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("deck")
    ap.add_argument("oracle")
    ap.add_argument("--kernel", action="store_true")
    ap.add_argument("--json")
    args = ap.parse_args(argv)
    oracle = load_oracle(args.oracle)
    nstep = max(oracle)
    report = {"columns": len(columns()[0]), "steps": nstep}
    report["cpu_gsl_wrf39_vs_oracle"] = ulp_table(
        run_reference(args.deck, nstep, "gsl_wrf39"), oracle)
    report["cpu_wrf_461_vs_oracle"] = ulp_table(
        run_reference(args.deck, nstep, "wrf_461"), oracle)
    if args.kernel:
        report["kernel_gsl_wrf39_vs_oracle"] = ulp_table(
            run_kernel(args.deck, nstep, "gsl_wrf39"), oracle)
        report["kernel_gsl_wrf39_vs_cpu"] = ulp_table(
            run_kernel(args.deck, nstep, "gsl_wrf39"),
            run_reference(args.deck, nstep, "gsl_wrf39"))
    for key, table in report.items():
        if isinstance(table, dict):
            worst = sorted(table.items(), key=lambda kv: -kv[1]["max"])[:8]
            print(key, " ".join(f"{k}:{v['max']}/{v['exact']}of{v['n']}"
                                for k, v in worst))
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
