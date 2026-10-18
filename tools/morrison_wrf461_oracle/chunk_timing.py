#!/usr/bin/env python3
"""Morrison's forecast-step column calls against one whole-domain call.

Tiles the regime columns of one set into an NY x NX domain, runs
``launch_morrison`` once as one call and once in ``work_chunk_columns``
column calls from the same input, checks every output word is identical,
and times both (CUDA events, median of N calls).  It also reports the WORK
bytes each way.

usage: chunk_timing.py REGIME_DIR SET [--ny 600 --nx 600 --repeat 10]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
for path in (HERE.parents[1], HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import regime_columns as rc  # noqa: E402


def main(argv=None):
    import cupy as cp
    from gpuwm.core import morrison
    from gpuwm.core.morrison_constants import NWORK, work_chunk_columns

    parser = argparse.ArgumentParser()
    parser.add_argument("outdir", type=Path)
    parser.add_argument("set")
    parser.add_argument("--ny", type=int, default=600)
    parser.add_argument("--nx", type=int, default=600)
    parser.add_argument("--repeat", type=int, default=10)
    args = parser.parse_args(argv)
    ncol0, nz, mode, dt, fields, surf = rc.read_input(
        args.outdir / f"{args.set}.in.bin")
    ny, nx = args.ny, args.nx
    ncol = ny * nx
    pick = np.arange(ncol) % ncol0

    def fresh():
        d = {k: cp.asarray(np.ascontiguousarray(fields[k][:, pick])
                           .reshape(nz, ny, nx)) for k in rc.LEVEL_IN}
        s = {k: cp.asarray(np.ascontiguousarray(surf[k][pick])
                           .reshape(ny, nx)) for k in rc.SURF}
        d["nc"] = cp.zeros((nz, ny, nx), cp.float32)
        return d, s

    def call(d, s, chunk, t1d):
        extra = {} if chunk is None else {"columns_per_call": chunk,
                                          "t1d_out": t1d}
        return morrison.launch_morrison(
            d["th"], d["qv"], d["qc"], d["qr"], d["qi"], d["qs"], d["qg"],
            d["nc"], d["nr"], d["ni"], d["ns"], d["ng"], d["rho"],
            d["pii"], d["p"], d["dz"], s["rainnc"], s["rainncv"],
            s["snownc"], s["snowncv"], s["graupelnc"], s["graupelncv"],
            s["sr"], float(dt), qrcuten=d["qrcuten"], qscuten=d["qscuten"],
            qicuten=d["qicuten"], morr_rimed_ice=mode, **extra)

    chunk = work_chunk_columns(ncol)
    results = {}
    outputs = {}
    for label, cpc in (("one_call", None), ("column_calls", chunk)):
        times = []
        for rep in range(args.repeat + 1):
            d, s = fresh()
            t1d = cp.empty_like(d["th"])
            cp.cuda.Device().synchronize()
            start, stop = cp.cuda.Event(), cp.cuda.Event()
            start.record()
            work = call(d, s, cpc, t1d)
            stop.record()
            stop.synchronize()
            if rep:
                times.append(cp.cuda.get_elapsed_time(start, stop))
        if cpc is None:
            t1d = work[12 * nz:13 * nz]
        outputs[label] = {k: cp.asnumpy(v) for k, v in
                          list(d.items()) + list(s.items())}
        outputs[label]["t1d"] = cp.asnumpy(t1d)
        work_bytes = (NWORK * nz * (ncol if cpc is None else cpc) * 4)
        results[label] = {"median_ms": float(np.median(times)),
                          "min_ms": float(np.min(times)),
                          "work_bytes": work_bytes,
                          "calls": 1 if cpc is None else -(-ncol // cpc)}
    same = all(np.array_equal(outputs["one_call"][k].view(np.uint32),
                              outputs["column_calls"][k].view(np.uint32))
               for k in outputs["one_call"])
    report = {"set": args.set, "ny": ny, "nx": nx, "nz": nz,
              "columns_per_call": chunk, "bitwise_identical": bool(same),
              **results}
    print(json.dumps(report, indent=2))
    return 0 if same else 1


if __name__ == "__main__":
    raise SystemExit(main())
