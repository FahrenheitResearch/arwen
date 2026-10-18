"""Calm-column check: on step 1 of a fixture, every word WRF wrote finite must equal the
product's word bitwise, and every product word must be finite."""
import sys, json, os, numpy as np
from gpuwm.verify.uwpbl_oracle import load, dims, step_arrays, MASS_OUTPUTS, FULL_OUTPUTS, SURFACE_OUTPUTS
from test_uwpbl_launcher_wrf471_parity import _launch
OUTS = MASS_OUTPUTS + FULL_OUTPUTS + SURFACE_OUTPUTS
fx = load(sys.argv[1]); ncol, nk, ns = dims(fx)
want = step_arrays(fx, 1); got = _launch(want, nk, ncol)
rep = {"GPUWM_WRF_EXACT": os.environ.get("GPUWM_WRF_EXACT", ""), "columns": {}}
for c in range(ncol):
    finite_bad = 0; wrf_nan = {}; gpu_nonfinite = 0; nan_words_gpu = {}
    for f in OUTS:
        w = np.asarray(want[f]).reshape(ncol, -1)[c]; g = np.asarray(got[f]).reshape(ncol, -1)[c]
        if w.dtype == np.int32:
            finite_bad += int((w != g).sum()); continue
        g = g.astype(np.float32)
        wf = np.isfinite(w)
        finite_bad += int((w.view(np.uint32)[wf] != g.view(np.uint32)[wf]).sum())
        if (~wf).any():
            wrf_nan[f] = [float(x) for x in g[~wf]]
        gpu_nonfinite += int((~np.isfinite(g)).sum())
    rep["columns"][c + 1] = {"u1": float(want["u"][c, 0]), "v1": float(want["v"][c, 0]), "ust": float(want["ust"][c]),
                             "mismatches_on_wrf_finite_words": finite_bad,
                             "product_value_where_wrf_nan": wrf_nan, "product_nonfinite_words": gpu_nonfinite}
json.dump(rep, open(sys.argv[2], "w"), indent=1)
for c, r in rep["columns"].items(): print(c, r)
