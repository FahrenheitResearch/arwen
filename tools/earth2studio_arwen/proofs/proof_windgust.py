"""hf-5 proof 1: NVIDIA WindgustAFNO on a WOOF run, through earth2studio_arwen.WoofHistory.

The data source is a hex (gpuwm-hex) 60 km global forecast's history, read
through gpuwm's Earth-2 exporter onto Earth2Studio's 0.25 degree lattice with
the channels WindgustAFNO's row in gpuwm's Earth-2 table asks for. One
earth2studio.run.diagnostic call over every 6-hourly valid time of the run
(nsteps=0: the diagnostic is applied to the WOOF state itself at each time).
"""
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import earth2studio.run as run
from earth2studio.data.base import DataSource
from earth2studio.io import NetCDF4Backend
from earth2studio.models.dx import WindgustAFNO
from earth2studio.models.px import Persistence

from earth2studio_arwen import WoofHistory

history, out = [Path(p) for p in sys.argv[1].split(",")], Path(sys.argv[2]).resolve()
out.mkdir(parents=True, exist_ok=True)
t0 = time.perf_counter()
src = WoofHistory(history, cache_dir=out / "earth2-export", consumers="windgust-afno")
assert isinstance(src, DataSource)
ds = src._ensure()
times = [t for t in ds["time"].values.astype("datetime64[h]")
         if int(t.astype("int64")) % 6 == 0]
t_export = time.perf_counter() - t0

dx = WindgustAFNO.load_model(WindgustAFNO.load_default_package())
ic = dx.input_coords()
px = Persistence(list(ic["variable"]), OrderedDict(lat=ic["lat"], lon=ic["lon"]))
io = NetCDF4Backend(str(out / "windgust_on_woof.nc"), backend_kwargs={"mode": "w"})
run.diagnostic(times, 0, px, dx, src, io, device="cuda")
io.close()
wall = time.perf_counter() - t0

import xarray as xr
g = xr.open_dataset(out / "windgust_on_woof.nc")["fg10m"]
receipt = {
    "proof": "earth2studio.run.diagnostic(WindgustAFNO) on earth2studio_arwen.WoofHistory",
    "history": [str(h) for h in history], "valid_times": [str(t) for t in times],
    "export_s": round(t_export, 1), "wall_s": round(wall, 1),
    "fg10m": {"shape": list(g.shape), "min": float(g.min()), "max": float(g.max()),
              "mean": float(g.mean()), "nan": int(np.isnan(g.values).sum())},
}
(out / "proof-windgust-receipt.json").write_text(json.dumps(receipt, indent=1))
print(json.dumps(receipt, indent=1))
