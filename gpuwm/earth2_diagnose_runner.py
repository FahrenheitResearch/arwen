"""Runner for ``gpuwm earth2-diagnose`` (gpuwm/earth2_diagnose.py); runs in the earth2studio interpreter.

Run Earth2Studio diagnostic models on a WOOF Earth2Studio-shaped export.

Table driven: each consumer is a row in earth2-models.v1.json "consumers"
(earth2studio_class). No per-model code: the class is looked up by name, its
input coordinates come from the model itself, the export is read through
earth2studio's own InferenceOutputSource, and map_coords puts the data on
the model's grid. Diagnostic outputs are written as CF NetCDF (time, lat,
lon); trackers (classes with a path_buffer) as CSV tracks.

Runs in the interpreter that has earth2studio (GPUWM_EARTH2_PYTHON).
"""
from __future__ import annotations

import argparse
import csv
import json
import time as _time
from collections import OrderedDict
from pathlib import Path

import numpy as np


def main() -> int:
    # torch and xarray live in the Earth2Studio environment this runner is launched in, not in gpuwm's own
    # dependencies, so they are imported here: importing the module (the wheel's import sweep) needs neither.
    import torch
    import xarray as xr

    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--table", required=True)
    ap.add_argument("--for", dest="consumers", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    from earth2studio.data import InferenceOutputSource
    from earth2studio.data.utils import fetch_data
    from earth2studio.models import dx
    from earth2studio.utils.coords import map_coords

    table = json.loads(Path(a.table).read_text())
    rows = table["consumers"]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    src = InferenceOutputSource(a.store)
    produced_by = str(xr.open_dataset(a.store, engine="zarr" if a.store.rstrip("/").endswith(".zarr") or Path(a.store).is_dir() else None).attrs.get("source", Path(a.store).name))
    valid = [np.datetime64(t, "ns") for t in src.da["time"].values]
    receipt = {"store": a.store, "valid_times": [str(t) for t in valid], "consumers": {}}
    dev = torch.device(a.device)

    for key in a.consumers.split(","):
        row = rows[key]
        cls = getattr(dx, row["earth2studio_class"], None)
        if cls is None:
            raise SystemExit(f"{key}: {row['earth2studio_class']} is not an earth2studio diagnostic "
                             "(earth2studio.models.dx); a prognostic consumer takes the export as its "
                             "conditioning_data_source instead")
        t0 = _time.time()
        if row.get("initialization") == "constructor":
            model = cls(**row.get("constructor_kwargs", {}))
        else:
            model = cls.load_model(cls.load_default_package()) if hasattr(cls, "load_default_package") else cls()
        model = model.to(dev)
        ic = model.input_coords()
        variables = list(ic["variable"])
        tracker = hasattr(model, "path_buffer")
        planes: dict[str, list[np.ndarray]] = {}
        times = []
        oc_last = None
        for t in valid:
            x, coords = fetch_data(src, time=np.array([t]), variable=np.array(variables),
                                   lead_time=np.array([np.timedelta64(0, "h")]), device=dev)
            # drop the size-1 time/lead_time dims the diagnostic does not take
            if "time" not in ic:
                x = x[0, 0]
                coords = OrderedDict((k, v) for k, v in coords.items() if k not in ("time", "lead_time"))
            if "batch" in ic:
                x = x.unsqueeze(0)
                coords = OrderedDict([("batch", np.empty(0))] + list(coords.items()))
            if "time" in ic and "batch" in ic:
                pass
            x, coords = map_coords(x, coords, ic)
            with torch.inference_mode():
                y, oc = model(x, coords)
            oc_last = oc
            times.append(t)
            if tracker:
                continue
            yv = y.detach().float().cpu().numpy()
            yv = yv.reshape((-1,) + yv.shape[-3:])[0]  # (variable, lat, lon)
            for i, v in enumerate(oc["variable"]):
                planes.setdefault(str(v), []).append(yv[i])
        info = {"class": row["earth2studio_class"], "inputs": variables,
                "seconds": round(_time.time() - t0, 2), "frames": len(times)}
        if tracker:
            paths = model.path_buffer.detach().float().cpu().numpy()
            names = [str(v) for v in oc_last["variable"]]
            fn = out / f"{key}-tracks.csv"
            with fn.open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["path", "step", "valid_time"] + names)
                if paths.size:
                    p = paths[0]
                    for pid in range(p.shape[0]):
                        for s in range(p.shape[1]):
                            if np.all(np.isnan(p[pid, s])):
                                continue
                            w.writerow([pid, s, str(times[s]) if s < len(times) else ""] + [f"{z:.4f}" for z in p[pid, s]])
            info["paths"] = int(paths.shape[1]) if paths.ndim >= 2 else 0
            info["file"] = str(fn)
        else:
            lat = np.asarray(oc_last["lat"])
            lon = np.asarray(oc_last["lon"])
            ds = xr.Dataset({v: (("time", "lat", "lon"), np.stack(pl).astype("float32")) for v, pl in planes.items()},
                            coords={"time": ("time", np.array(times), {"standard_name": "time", "axis": "T"}), "lat": ("lat", lat, {"units": "degrees_north", "standard_name": "latitude"}),
                                    "lon": ("lon", lon, {"units": "degrees_east", "standard_name": "longitude"})},
                            attrs={"source": f"{row['earth2studio_class']} (Earth2Studio) on {produced_by}",
                                   "Conventions": "CF-1.8"})
            fn = out / f"{key}.nc"
            ds.to_netcdf(fn)
            info["file"] = str(fn)
            info["ranges"] = {v: [float(np.nanmin(np.stack(pl))), float(np.nanmax(np.stack(pl)))] for v, pl in planes.items()}
        receipt["consumers"][key] = info
        print(json.dumps({key: info}), flush=True)
        del model
        torch.cuda.empty_cache()
    (out / "earth2-diagnose.json").write_text(json.dumps(receipt, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
