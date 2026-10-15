#!/usr/bin/env python3
"""Coverage facts for the refute sets, from the inputs and WRF's outputs
(CPU only, float32 arithmetic in the Fortran's order where it matters)."""
import json
import sys
from pathlib import Path

import numpy as np

from gpuwm.verify import kessler_oracle as ko

F = np.float32
SVP1, SVP2, SVP3, SVPT0 = F(0.6112), F(17.67), F(29.65), F(273.15)
root = Path(sys.argv[1])
out = {}
for s in sys.argv[2:]:
    d = root / s
    inp = ko.read_inputs_bin(d / "kessler-inputs.bin")
    with np.load(d / "kessler-inputs.npz") as f:
        reg = f["regime"]
    nz, ncol = inp["t"].shape
    total = int(inp["nsteps"].sum())
    wrf = ko.read_outputs_bin(d / "kessler-wrf-O0.bin", nz, ncol, total)
    # states entering every call (inputs for a case's first call, else
    # WRF's previous outputs)
    pre = {k: [] for k in ("t", "qv")}
    n = 0
    dts = []
    for dt, ns in zip(inp["dt"], inp["nsteps"]):
        for step in range(int(ns)):
            for k in pre:
                pre[k].append(inp[k] if step == 0 else wrf[k][n - 1])
            dts.append(dt)
            n += 1
    t = np.stack(pre["t"]); qv = np.stack(pre["qv"])
    pii = inp["pii"][None]
    temp = (pii * t).astype(F)
    with np.errstate(all="ignore"):
        pres = (F(1e5) * np.power(pii, F(1004.0) / F(287.0))).astype(F)
        es = (F(1000.0) * SVP1 * np.exp(SVP2 * (temp - SVPT0) / (temp - SVP3))
              ).astype(F)
    row = {"cells_x_calls": int(temp.size),
           "es_ge_p_cells": int((es >= pres).sum()),
           "pii_gt_1_cells": int((inp["pii"] > 1).sum()),
           "min_pressure_Pa": float(pres.min()),
           "temp_range_K": [float(temp.min()), float(temp.max())],
           "input_subnormal_words": int(sum(
               ((inp[k] != 0) & (np.abs(inp[k]) < np.finfo(F).tiny)).sum()
               for k in ("qv", "qc", "qr"))),
           "input_neg_zero_words": int(sum(
               ((inp[k] == 0) & np.signbit(inp[k])).sum()
               for k in ("qv", "qc", "qr"))),
           "input_negative_words": int(sum((inp[k] < 0).sum()
                                           for k in ("qv", "qc", "qr")))}
    # approximate first-call split count per case (float64)
    nf = {}
    for dt in inp["dt"]:
        qrr = np.maximum(0, inp["qr"].astype(float) * 1e-3 * inp["rho"])
        vt = 36.34 * qrr ** 0.1364 * np.sqrt(inp["rho"][0] / inp["rho"])
        cr = (vt * float(dt) / inp["dz8w"]).max(axis=0)
        nf[str(float(dt))] = int(np.max(np.maximum(1, np.rint(0.5 + cr / 0.75))))
    row["max_initial_nfall_by_dt"] = nf
    out[s] = row
    print(s, json.dumps(row))
Path(root / "coverage.json").write_text(json.dumps(out, indent=2) + "\n")
