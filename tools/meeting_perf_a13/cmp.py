"""Field-by-field comparison of two history directories (wrfout* files matched by name)."""
import json, sys
from pathlib import Path
import numpy as np
import netCDF4
a, b = Path(sys.argv[1]), Path(sys.argv[2])
fa = {p.relative_to(a).as_posix(): p for p in a.rglob("wrfout*") if p.is_file() and p.suffix != ".json"}
fb = {p.relative_to(b).as_posix(): p for p in b.rglob("wrfout*") if p.is_file() and p.suffix != ".json"}
out = {"a": str(a), "b": str(b), "only_a": sorted(set(fa) - set(fb)), "only_b": sorted(set(fb) - set(fa)), "files": {}}
for name in sorted(set(fa) & set(fb)):
    da, db = netCDF4.Dataset(fa[name]), netCDF4.Dataset(fb[name])
    da.set_auto_mask(False); db.set_auto_mask(False)
    diffs = {}
    for v in sorted(set(da.variables) | set(db.variables)):
        if v not in da.variables or v not in db.variables:
            diffs[v] = "missing in one"; continue
        x, y = da.variables[v][:], db.variables[v][:]
        if x.shape != y.shape:
            diffs[v] = f"shape {x.shape} vs {y.shape}"; continue
        if x.tobytes() == y.tobytes():
            continue
        if x.dtype.kind in "fiu":
            d = np.abs(x.astype(np.float64) - y.astype(np.float64))
            diffs[v] = {"max_abs": float(np.nanmax(d)), "n_diff": int((d > 0).sum()), "n": int(d.size),
                        "max_at": [int(i) for i in np.unravel_index(np.nanargmax(d), d.shape)]}
        else:
            diffs[v] = "non-numeric differs"
    gattr = sorted(k for k in set(da.ncattrs()) | set(db.ncattrs())
                   if k not in da.ncattrs() or k not in db.ncattrs() or str(da.getncattr(k)) != str(db.getncattr(k)))
    out["files"][name] = {"differing_variables": diffs, "differing_global_attrs":
                          {k: [str(da.getncattr(k)) if k in da.ncattrs() else None,
                               str(db.getncattr(k)) if k in db.ncattrs() else None] for k in gattr}}
    da.close(); db.close()
json.dump(out, sys.stdout, indent=1)
