#!/usr/bin/env python3
"""Grade the production Kessler launcher on the refute column sets.

    python compare_refute.py ROOT SET [SET...] --json OUT

ROOT/SET holds kessler-inputs.bin/.npz (make_refute_columns.py) and
kessler-wrf-O0.bin (the lane's run_kessler_O0 on those inputs).  Modes:
free, replay, and free_layout (same columns laid out (nz, ny, nx) with
ny > 1, to exercise the kernel's 2-D column indexing).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from gpuwm.verify import kessler_oracle as ko


def run_gpu_layout(inputs, ny, nx):
    import cupy as cp
    from gpuwm.core.microphysics import launch_kessler
    nz, ncol = inputs["t"].shape
    assert ny * nx == ncol

    def dev(a):
        a = np.ascontiguousarray(a, np.float32)
        return cp.asarray(a.reshape(nz, ny, nx) if a.ndim == 2
                          else a.reshape(ny, nx))
    rho, pii = dev(inputs["rho"]), dev(inputs["pii"])
    z, dz8w = dev(inputs["z"]), dev(inputs["dz8w"])
    out = {k: [] for k in ko.OUTPUT_FIELDS}
    for dt, nsteps in zip(inputs["dt"], inputs["nsteps"]):
        st = {k: dev(inputs[k]) for k in ko.LEVEL_FIELDS}
        rainnc = dev(inputs["rainnc"])
        rainncv = cp.zeros((ny, nx), cp.float32)
        for _ in range(int(nsteps)):
            launch_kessler(st["t"], st["qv"], st["qc"], st["qr"], rho, pii,
                           z, dz8w, rainnc, rainncv, float(dt))
            for k in ko.LEVEL_FIELDS:
                out[k].append(cp.asnumpy(st[k]).reshape(nz, ncol))
            out["rainnc"].append(cp.asnumpy(rainnc).reshape(ncol))
            out["rainncv"].append(cp.asnumpy(rainncv).reshape(ncol))
    return {k: np.stack(v) for k, v in out.items()}


def wrf_census(wrf):
    c = {}
    for k, v in wrf.items():
        c[k] = {"nan": int(np.isnan(v).sum()), "inf": int(np.isinf(v).sum()),
                "neg_zero": int(((v == 0) & np.signbit(v)).sum()),
                "subnormal": int(((v != 0) & (np.abs(v) < np.finfo(
                    np.float32).tiny)).sum()),
                "negative": int((v < 0).sum())}
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("sets", nargs="+")
    ap.add_argument("--json", type=Path, required=True)
    args = ap.parse_args()
    import cupy as cp
    from gpuwm.core import kernels
    from gpuwm import wrf_exact
    dev = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    src = kernels.module_source("kessler")
    report = {"strict": wrf_exact.ENABLED,
              "strict_options": list(wrf_exact.STRICT_OPTIONS),
              "kessler_source_sha256": hashlib.sha256(
                  src.encode()).hexdigest(),
              "kessler_options": list(kernels.module_options("kessler")),
              "device": dev["name"].decode(),
              "nvrtc": ".".join(map(str, cp.cuda.nvrtc.getVersion())),
              "sets": {}}
    for s in args.sets:
        d = args.root / s
        inputs = ko.read_inputs_bin(d / "kessler-inputs.bin")
        with np.load(d / "kessler-inputs.npz") as f:
            for k in ko.INPUT_FIELDS + ("rainnc",):
                assert f[k].tobytes() == inputs[k].tobytes(), (s, k)
            regimes = f["regime"]
        nz, ncol = inputs["t"].shape
        total = int(inputs["nsteps"].sum())
        wrf = ko.read_outputs_bin(d / "kessler-wrf-O0.bin", nz, ncol, total)
        row = {"columns": ncol, "levels": nz, "calls": total,
               "values": int(sum(v.size for v in wrf.values())),
               "cases": [[float(a), int(b)] for a, b in
                         zip(inputs["dt"], inputs["nsteps"])],
               "regimes": sorted(set(map(str, regimes))),
               "wrf_output_census": wrf_census(wrf)}
        for mode in ("free", "replay"):
            got = ko.run_gpu(inputs, mode=mode, wrf=wrf)
            row[mode] = ko.compare(got, wrf, regimes)
        if ncol % 8 == 0:
            got = run_gpu_layout(inputs, 8, ncol // 8)
            row["free_layout_ny8"] = ko.compare(got, wrf, regimes)
        report["sets"][s] = row
        bad = {m: {k: v["not_bit_identical"] for k, v in row[m].items()
                   if v["not_bit_identical"]}
               for m in ("free", "replay", "free_layout_ny8") if m in row}
        print(s, "values", row["values"], "mismatches", bad, flush=True)
    receipt = wrf_exact.compile_receipt()
    sha = report["kessler_source_sha256"]
    report["kessler_compiles"] = [r for r in receipt["records"]
                                  if r["source_sha256"] == sha]
    args.json.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
