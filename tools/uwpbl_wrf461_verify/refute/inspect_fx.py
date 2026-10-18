"""Inspect refute fixtures: host vs CR equality, NaN census, qncA vs qncB output identity."""
import sys, json, numpy as np
from pathlib import Path
from gpuwm.verify.uwpbl_oracle import load, dims, step_arrays, MASS_OUTPUTS, FULL_OUTPUTS, SURFACE_OUTPUTS
OUTS = MASS_OUTPUTS + FULL_OUTPUTS + SURFACE_OUTPUTS
root = Path(sys.argv[1])
rep = {}
for man in sorted((root / "host").glob("cases-*.manifest")):
    stem = man.with_suffix("")
    h = (root / "host" / (stem.name + ".bin")).read_bytes()
    c = (root / "cr" / (stem.name + ".bin")).read_bytes()
    fx = load(stem)
    ncol, nk, ns = dims(fx)
    nan_cols = set(); nonfinite_in = set()
    for s in range(1, ns + 1):
        a = step_arrays(fx, s)
        for f in OUTS:
            x = np.asarray(a[f])
            if x.dtype != np.int32:
                bad = ~np.isfinite(x.reshape(ncol, -1)).all(axis=1)
                nan_cols |= {(int(i) + 1, s) for i in np.flatnonzero(bad)}
        for f in ("t", "u", "v", "qv", "th"):
            x = a[f].reshape(ncol, -1)
            nonfinite_in |= {(int(i) + 1, s) for i in np.flatnonzero(~np.isfinite(x).all(axis=1))}
        tmin = float(a["t"].min())
    rep[stem.name] = {"ncol": ncol, "nk": nk, "nsteps": ns, "host_eq_cr": h == c,
                      "nonfinite_output_colsteps": len(nan_cols),
                      "nonfinite_output_first": sorted(nan_cols)[:8],
                      "nonfinite_input_colsteps": len(nonfinite_in)}
# qnc pair
A = load(root / "host" / "cases-refute-qncA"); B = load(root / "host" / "cases-refute-qncB")
diff = []
for n in A.names():
    if n.endswith("/qnc"):
        continue
    if A[n].tobytes() != B[n].tobytes():
        diff.append(n)
rep["qnc_pair_differing_records_except_qnc"] = diff
qa = A["s1/qnc"]; rep["qncA_nonzero_words_s1"] = int((qa != 0).sum())
print(json.dumps(rep, indent=1))
