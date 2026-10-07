"""Price full HRRR [devices] 2x2 on the CPU: the scale-lane tip (every-column lake price) against this lane
(lake columns from the prepared statics), per card and per term, at a 30.86 GiB budget."""
import json, sys
from dataclasses import replace
from pathlib import Path
import numpy as np
from gpuwm.experiment import load_experiment
from gpuwm.core.devices import DeviceOptions
from gpuwm.core import devices_memory as dm
from gpuwm.prepared_single_domain_forecast import _devices_lake_mask
GIB = 2**30
case = Path(sys.argv[1])
exp = load_experiment(str(case / "experiment.toml"))
cfg = exp.root.run
static = np.load(case / "prepared" / "native-static.npz")
geo = {k: static[k] for k in static.files if k.upper() in ("LANDMASK", "XLAND", "LAKEMASK", "HGT_M", "LU_INDEX")}
print("static keys used", {k: v.shape for k, v in geo.items()}, file=sys.stderr)
from gpuwm.prepared_single_domain_forecast import _LANDUSE_IDENTITY
mask = _devices_lake_mask(cfg, geography=geo, landuse=dict(_LANDUSE_IDENTITY))
print("landuse", dict(_LANDUSE_IDENTITY), file=sys.stderr)
out = {"lake_mask_columns": None if mask is None else int(mask.sum()), "layouts": {}}
for grid in sys.argv[2:]:
    gy, gx = (int(v) for v in grid.split("x"))
    n = gy * gx
    opts = DeviceOptions(count=n, grid=(gy, gx), ids=tuple(range(n)), transport="auto")
    e = replace(exp, devices=opts)
    budgets = {i: int(30.86 * GIB) for i in range(n)}
    row = {}
    for label, lm in (("before", None), ("after", mask)):
        est = dm.estimate_devices(e, options=opts, forcing_intervals=2, source="rap-native",
                                  budgets=budgets, lake_mask=lm)
        row[label] = {"cards": [{k: (v / GIB if k.endswith("bytes") else v) for k, v in c.items()} for c in est["cards"]],
                      "ranks": [{k: (v / GIB if k.endswith("bytes") else v) for k, v in r.items()} for r in est["rank_shapes"]]}
    out["layouts"][grid] = row
json.dump(out, sys.stdout, indent=1, default=str)
