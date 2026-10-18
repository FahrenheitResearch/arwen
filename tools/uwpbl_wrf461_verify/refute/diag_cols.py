"""Word-level detail for named column-steps: WRF word, GPU product word, ULP."""
import sys, json, numpy as np
from pathlib import Path
from gpuwm.verify.uwpbl_oracle import load, dims, step_arrays, MASS_OUTPUTS, FULL_OUTPUTS, SURFACE_OUTPUTS
from test_uwpbl_launcher_wrf471_parity import _launch
import os
OUTS = MASS_OUTPUTS + FULL_OUTPUTS + SURFACE_OUTPUTS
def ordered(a):
    i = a.view(np.int32).astype(np.int64); m = i & 0x7FFFFFFF
    return np.where(i < 0, -m, m)
spec = json.loads(sys.argv[1])   # {stem_path: [[col, step], ...]}
out = {"GPUWM_WRF_EXACT": os.environ.get("GPUWM_WRF_EXACT", ""), "cases": {}}
for stem, cs in spec.items():
    fx = load(stem); ncol, nk, ns = dims(fx)
    for step in sorted({s for _, s in cs}):
        want = step_arrays(fx, step); got = _launch(want, nk, ncol)
        for col, s in cs:
            if s != step: continue
            c = col - 1; rows = []
            for f in OUTS:
                w = np.asarray(want[f]).reshape(ncol, -1)[c]; g = np.asarray(got[f]).reshape(ncol, -1)[c]
                if w.dtype == np.int32:
                    bad = np.flatnonzero(w != g)
                    rows += [{"field": f, "level": int(k) + 1, "wrf": int(w[k]), "gpu": int(g[k])} for k in bad]
                else:
                    g = g.astype(np.float32)
                    bad = np.flatnonzero(w.view(np.uint32) != g.view(np.uint32))
                    rows += [{"field": f, "level": int(k) + 1, "wrf": float(w[k]), "gpu": float(g[k]),
                              "wrf_hex": hex(int(w.view(np.uint32)[k])), "gpu_hex": hex(int(g.view(np.uint32)[k])),
                              "ulp": int(abs(ordered(w[k:k+1]) - ordered(g[k:k+1]))[0])} for k in bad]
            nonfin_w = {f: int((~np.isfinite(np.asarray(want[f]).reshape(ncol, -1)[c].astype(np.float64))).sum()) for f in OUTS}
            nonfin_g = {f: int((~np.isfinite(np.asarray(got[f]).reshape(ncol, -1)[c].astype(np.float64))).sum()) for f in OUTS}
            out["cases"][f"{Path(stem).name} col {col} step {step}"] = {
                "mismatched_words": len(rows), "rows": rows[:60],
                "wrf_nonfinite": {k: v for k, v in nonfin_w.items() if v},
                "gpu_nonfinite": {k: v for k, v in nonfin_g.items() if v},
                "inputs": {"u1": float(want["u"][c, 0]), "v1": float(want["v"][c, 0]), "ust": float(want["ust"][c]),
                           "hfx": float(want["hfx"][c]), "qfx": float(want["qfx"][c])}}
json.dump(out, open(sys.argv[2], "w"), indent=1)
for k, v in out["cases"].items():
    print(k, v["mismatched_words"], "wrf_nonfinite", v["wrf_nonfinite"], "gpu_nonfinite", v["gpu_nonfinite"], v["inputs"])
    for r in v["rows"][:4]: print("   ", r)
