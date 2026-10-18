"""Host-libm WRF vs correctly-rounded-cos/acos WRF: every column-step whose
INPUT words are identical in both builds but whose OUTPUT words differ."""
import sys, json, numpy as np
from pathlib import Path
from gpuwm.verify.uwpbl_oracle import (load, dims, step_arrays, MASS_INPUTS, FULL_INPUTS,
    SURFACE_INPUTS, MASS_OUTPUTS, FULL_OUTPUTS, SURFACE_OUTPUTS)
OUTS = MASS_OUTPUTS + FULL_OUTPUTS + SURFACE_OUTPUTS
INS = MASS_INPUTS + FULL_INPUTS + SURFACE_INPUTS
def ordered(a):
    i = a.view(np.int32).astype(np.int64); m = i & 0x7FFFFFFF
    return np.where(i < 0, -m, m)
root = Path(sys.argv[1]); res = {}
for man in sorted((root / "host").glob("cases-*.manifest")):
    st = man.stem
    H = load(root / "host" / st); C = load(root / "cr" / st)
    ncol, nk, ns = dims(H); rows = []
    for s in range(1, ns + 1):
        a = step_arrays(H, s); b = step_arrays(C, s)
        same_in = np.ones(ncol, bool)
        for f in INS + ("kvm3d_in", "kvh3d_in", "tauresx2d_in", "tauresy2d_in"):
            same_in &= (a[f].reshape(ncol, -1).view(np.uint32) == b[f].reshape(ncol, -1).view(np.uint32)).all(axis=1)
        for c in np.flatnonzero(same_in):
            for f in OUTS:
                x = np.asarray(a[f]).reshape(ncol, -1)[c]; y = np.asarray(b[f]).reshape(ncol, -1)[c]
                if x.dtype == np.int32:
                    bad = x != y; ulp = np.abs(x.astype(np.int64) - y)
                else:
                    bad = x.view(np.uint32) != y.view(np.uint32); ulp = np.abs(ordered(x) - ordered(y))
                for k in np.flatnonzero(bad):
                    rows.append({"col": int(c) + 1, "step": s, "field": f, "level": int(k) + 1,
                                 "host": float(x[k]), "cr": float(y[k]), "ulp": int(ulp[k]),
                                 "host_hex": hex(int(x.view(np.uint32)[k])) if x.dtype != np.int32 else int(x[k]),
                                 "cr_hex": hex(int(y.view(np.uint32)[k])) if y.dtype != np.int32 else int(y[k])})
    if rows:
        cs = sorted({(r["col"], r["step"]) for r in rows})
        res[st] = {"column_steps_same_input_diff_output": cs, "words": len(rows),
                   "max_ulp": max(r["ulp"] for r in rows),
                   "max_abs": max(abs(r["host"] - r["cr"]) for r in rows),
                   "by_field": {f: sum(r["field"] == f for r in rows) for f in OUTS if any(r["field"] == f for r in rows)},
                   "rows": rows}
json.dump(res, open(sys.argv[2], "w"), indent=1)
for st, r in res.items():
    print(st, "colsteps", r["column_steps_same_input_diff_output"], "words", r["words"], "max_ulp", r["max_ulp"], "max_abs", r["max_abs"], r["by_field"])
