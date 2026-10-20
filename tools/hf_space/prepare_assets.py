# SPDX-License-Identifier: Apache-2.0
"""Precompute catalog assets from table rows using public native engine doors.

Run with the engine checkout on PYTHONPATH. This tool is never invoked by app.py.
The source table is local build configuration, not part of a Space runtime.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import time

from gpuwm import rustwx, soundings


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def record_deletions(root, receipt):
    with receipt.open("a") as stream:
        for path in sorted(Path(root).rglob("*")):
            if path.is_file() or path.is_symlink():
                stream.write(json.dumps({"path": str(path), "bytes": path.lstat().st_size, "symlink": path.is_symlink()}) + "\n")


def tensor_view(sample, frame, scratch):
    """Wrap a card tensor's exact surface values in the renderer's WRF schema.

    The shared input schema is table driven. No interpolation or forecast runs
    here. Native volumetric histories remain the public sounding door's input.
    """
    import netCDF4
    import numpy as np

    path = Path(scratch) / "surface-view.nc"
    fields = {}
    with netCDF4.Dataset(sample["tensor"]) as source:
        source.set_auto_mask(False)
        lat = source["lat"][:].astype("f4")
        lon = source["lon"][:].astype("f4")
        with netCDF4.Dataset(path, "w", format="NETCDF3_64BIT_OFFSET") as view:
            for name, size in (("Time", 1), ("south_north", len(lat)),
                               ("west_east", len(lon)), ("DateStrLen", 19), ("bottom_top", 1)):
                view.createDimension(name, size)
            dimensions = ("Time", "south_north", "west_east")
            for target, name in sample["surface_fields"].items():
                values = source[name][0, frame["tensor_lead_index"]].astype("f4")
                view.createVariable(target, "f4", dimensions)[:] = values[None]
                fields[name] = {"sha256_float32": hashlib.sha256(values.tobytes()).hexdigest(),
                                "finite": int(np.isfinite(values).sum()), "missing": int(np.isnan(values).sum())}
            x, y = np.meshgrid(lon, lat)
            for name, values in (("XLAT", y), ("XLONG", x),
                                 ("COSALPHA", np.ones_like(x)), ("SINALPHA", np.zeros_like(x))):
                view.createVariable(name, "f4", dimensions)[:] = values[None]
            stamp = frame["valid"].rstrip("Z").replace("T", "_")
            view.createVariable("Times", "S1", ("Time", "DateStrLen"))[:] = np.array(list(stamp), dtype="S1")[None]
            start = sample["provenance"]["initialization"].rstrip("Z").replace("T", "_")
            view.setncatts({"TITLE": "Exact surface view of card tensor sample", "MAP_PROJ": 6,
                           "MAP_PROJ_CHAR": "Cylindrical Equidistant", "START_DATE": start,
                           "SIMULATION_START_DATE": start, "CEN_LAT": float(lat.mean()),
                           "CEN_LON": float(lon.mean()), "STAND_LON": 0., "TRUELAT1": 0.,
                           "TRUELAT2": 0., "POLE_LAT": 90., "POLE_LON": 0.,
                           "DX": float(abs(lon[1]-lon[0])*111194.9),
                           "DY": float(abs(lat[1]-lat[0])*111194.9)})
    return path, fields


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.scratch.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.sources.read_text())
    renderer = rustwx.find_renderer()
    ok, detail = rustwx.probe_renderer(renderer)
    if not ok:
        raise RuntimeError(detail)
    started = time.perf_counter()
    catalog = {"schema": "woof-space.v1", "samples": [], "soundings": [], "run_markdown": config["run_markdown"]}
    receipt = {"renderer": str(renderer), "renderer_probe": detail,
               "soundings_binary": str(soundings.binary()), "samples": []}
    for sample in config["samples"]:
        rows = []
        public_receipt = {"sample": sample["label"], "provenance": sample["provenance"], "frames": []}
        if sample.get("tensor"):
            public_receipt["tensor"] = {"name": Path(sample["tensor"]).name,
                                         "sha256": digest(sample["tensor"]),
                                         "surface_fields": sample["surface_fields"]}
        with ThreadPoolExecutor(max_workers=4) as pool:
            hashes = list(pool.map(digest, [row["path"] for row in sample["frames"]]))
        for frame, source_hash in zip(sample["frames"], hashes):
            frame_start = time.perf_counter()
            if sample.get("pre_rendered"):
                reused = []
                for variable in config["variables"]:
                    picture = next(row for row in sample["pre_rendered"]
                                   if row["slug"] == variable["slug"] and row["valid"] == frame["valid"])
                    source = Path(picture["path"])
                    target = (args.output / "assets" / sample["id"] / sample["domain"] /
                              variable["slug"] / frame["valid"][:10] / source.name)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
                    rows.append({"variable": variable["label"], "lead": frame["lead"],
                                 "valid": frame["valid"], "image": str(target.relative_to(args.output))})
                    reused.append({"source_name": source.name, "sha256": digest(source)})
                public_receipt["frames"].append({"source_name": Path(frame["path"]).name,
                                                 "source_sha256": source_hash, "lead": frame["lead"],
                                                 "valid": frame["valid"], "pre_rendered": reused,
                                                 "map_seconds": round(time.perf_counter() - frame_start, 3)})
                continue
            # Store files are spent imports, isolated from the source run.
            with tempfile.TemporaryDirectory(prefix="map-", dir=args.scratch) as scratch:
                render_input = Path(frame["path"])
                fields = None
                if sample.get("tensor"):
                    render_input, fields = tensor_view(sample, frame, scratch)
                pictures, failures, skipped = rustwx.run_renderer(
                    renderer, render_input, store_root=Path(scratch) / "store",
                    out_dir=args.output / "assets" / sample["id"] / sample["domain"],
                    products=",".join(row["slug"] for row in config["variables"]), frames="all",
                    width=1280, height=820, source_label=sample["label"])
                if failures or skipped:
                    raise RuntimeError({"failures": failures, "skipped": skipped})
                for variable in config["variables"]:
                    matches = [p for p in pictures if variable["slug"] in p.parts or variable["slug"] in p.name]
                    if len(matches) != 1:
                        raise RuntimeError(f"Expected one {variable['slug']} image: {pictures}")
                    # Organize each native output as case/domain/product/day at build time.
                    target = (args.output / "assets" / sample["id"] / sample["domain"] /
                              variable["slug"] / frame["valid"][:10] / matches[0].name)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    matches[0].replace(target)
                    rows.append({"variable": variable["label"], "lead": frame["lead"],
                                 "valid": frame["valid"], "image": str(target.relative_to(args.output))})
                record_deletions(scratch, args.scratch / "deleted-files.jsonl")
            public_receipt["frames"].append({"source_name": Path(frame["path"]).name, "source_sha256": source_hash,
                                             "lead": frame["lead"], "valid": frame["valid"],
                                             "tensor_surface": fields,
                                             "map_seconds": round(time.perf_counter() - frame_start, 3)})
        catalog["samples"].append({"id": sample["id"], "label": sample["label"],
                                   "note": sample.get("display_note", ""), "maps": rows})
        if sample.get("points"):
            with tempfile.TemporaryDirectory(prefix="profiles-", dir=args.scratch) as scratch:
                run = Path(scratch) / "run"
                run.mkdir()
                for frame in sample["frames"]:
                    path = Path(frame["path"])
                    (run / path.name).symlink_to(path)
                manifest = soundings.render_run(run, points=sample["points"], modes=("det",), threads=4,
                                                out=args.output / "assets" / sample["id"],
                                                settings={"theme": "dark"}, interval_minutes=60)
                data = json.loads(manifest.read_text())
                for picture in data["pictures"]:
                    png = Path(picture["png"])
                    profile = json.loads(png.with_suffix(".json").read_text())
                    frame = next(row for row in sample["frames"] if row["valid"] == profile["valid"])
                    catalog["soundings"].append({"point": profile["point"]["label"], "lead": frame["lead"],
                                                 "sample_label": sample["label"], "valid": frame["valid"],
                                                 "image": str(png.relative_to(args.output)),
                                                 "json": str(png.with_suffix(".json").relative_to(args.output))})
                # Do not ship machine-specific manifest paths. PNG and profile JSON are portable.
                data["pictures"] = [{**row, "png": str(Path(row["png"]).relative_to(args.output))} for row in data["pictures"]]
                manifest.write_text(json.dumps(data, indent=2) + "\n")
                record_deletions(scratch, args.scratch / "deleted-files.jsonl")
        dest = args.output / "assets" / sample["id"] / "receipt.json"
        dest.write_text(json.dumps(public_receipt, indent=2) + "\n")
        receipt["samples"].append(public_receipt)
    catalog["soundings"].sort(key=lambda row: (row["point"], row["lead"]))
    (args.output / "catalog.json").write_text(json.dumps(catalog, indent=2) + "\n")
    receipt["seconds"] = round(time.perf_counter() - started, 3)
    receipt["maps"] = sum(len(row["maps"]) for row in catalog["samples"])
    receipt["soundings"] = len(catalog["soundings"])
    (args.scratch / "prepare-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({key: receipt[key] for key in ("maps", "soundings", "seconds")}))


if __name__ == "__main__":
    main()
