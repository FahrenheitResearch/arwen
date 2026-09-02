"""Hold a cells catalog's peak W against a direct numpy maximum on the wrfout.

Usage: python tools/cells_peak_w_check.py <series-root> [--frames N] [--out FILE.json]

<series-root> is the `<case>/<domain>/cells/<day>/` folder `gpuwm cells
analyze` wrote.  For the chosen frames (default: every frame; `--frames N`
takes the N frames with the most cells plus the frame holding the series'
strongest updraft) the check reads `W`, `PH` and `PHB` straight from the
wrfout with netCDF4 -- not through the door's own column reader -- decodes
titan's voxel indices with the stream layout written in the export receipt
(x fastest, then y, then z), takes the maximum of `W` over every w level of
the unique columns under those voxels, and compares it with the catalog's
`peak_w_mps`, `peak_w_ft_min` (m/s x 196.85), `peak_w_height_m_msl` and the
peak's column.  Exit 0 only when every cell agrees exactly.
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import netCDF4
import numpy as np

FT_PER_MIN_PER_MPS = 196.85
G = 9.81


def main(argv: list[str]) -> int:
    root = Path(argv[1])
    frames_wanted = None
    out_path = root / "peak-w-check.json"
    if "--frames" in argv:
        frames_wanted = int(argv[argv.index("--frames") + 1])
    if "--out" in argv:
        out_path = Path(argv[argv.index("--out") + 1])

    export = json.loads((root / "export-receipt.json").read_text("utf-8"))
    grid = export["grid"]
    nx, ny = int(grid["nx"]), int(grid["ny"])
    plane = nx * ny
    frame_by_ts = {int(f["timestamp_ms"]): f for f in export["frames"]}

    catalog = json.loads((root / "catalog.json").read_text("utf-8"))
    rows = catalog["rows"]
    row_by_key = {(int(r["timestamp_ms"]), int(r["object_id"])): r for r in rows}
    per_frame: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        per_frame[int(r["timestamp_ms"])].append(r)

    if frames_wanted is None:
        chosen = set(per_frame)
    else:
        by_count = sorted(per_frame, key=lambda ts: -len(per_frame[ts]))
        chosen = set(by_count[:frames_wanted])
        strongest = max(rows, key=lambda r: r["peak_w_mps"] if r.get("peak_w_mps") is not None else -1e9)
        chosen.add(int(strongest["timestamp_ms"]))

    checked = 0
    mismatches: list[dict] = []
    worst_abs = 0.0
    frames_done = []
    tick = time.perf_counter()
    with open(root / "titan" / "frames.jsonl", "r", encoding="utf-8") as handle:
        for line in handle:
            head = line[:80]
            if '"timestamp_ms"' not in head:
                continue
            ts = int(line.split('"timestamp_ms":', 1)[1].split(",", 1)[0].strip())
            if ts not in chosen:
                continue
            frame = json.loads(line)
            objects = frame.get("objects") or []
            if not objects:
                continue
            meta = frame_by_ts[ts]
            t_index = int(meta["time_index"])
            with netCDF4.Dataset(meta["path"], "r") as ds:
                w = np.asarray(ds.variables["W"][t_index], dtype=np.float32)          # (nz+1, ny, nx)
                z_w = ((np.asarray(ds.variables["PH"][t_index], dtype=np.float64)
                        + np.asarray(ds.variables["PHB"][t_index], dtype=np.float64)) / G).astype(np.float32)
                lat = np.asarray(ds.variables["XLAT"][t_index], dtype=np.float32)
                lon = np.asarray(ds.variables["XLONG"][t_index], dtype=np.float32)
            assert w.shape[1:] == (ny, nx), (w.shape, ny, nx)
            n_cells = 0
            for obj in objects:
                vox = np.asarray(obj["voxels"], dtype=np.int64)
                rem = vox % plane
                cols = np.unique(rem)                    # y * nx + x, x fastest
                ys, xs = cols // nx, cols % nx
                sub = w[:, ys, xs]
                flat = int(np.argmax(sub))
                k, c = divmod(flat, sub.shape[1])
                direct = float(sub[k, c])
                direct_ft = direct * FT_PER_MIN_PER_MPS
                height = float(z_w[k, ys[c], xs[c]])
                row = row_by_key[(ts, int(obj["object_id"]))]
                cat = float(row["peak_w_mps"])
                cat_ft = float(row["peak_w_ft_min"])
                cat_h = float(row["peak_w_height_m_msl"])
                cat_cols = int(row["footprint_columns"])
                ok = (cat == direct and cat_ft == direct_ft and cat_h == height
                      and cat_cols == int(cols.size)
                      and float(row["peak_w_lat"]) == float(lat[ys[c], xs[c]])
                      and float(row["peak_w_lon"]) == float(lon[ys[c], xs[c]])
                      and float(row["min_w_mps"]) == float(sub.min()))
                worst_abs = max(worst_abs, abs(cat - direct))
                if not ok:
                    mismatches.append({
                        "timestamp_ms": ts, "object_id": obj["object_id"],
                        "catalog": {"peak_w_mps": cat, "peak_w_ft_min": cat_ft,
                                    "peak_w_height_m_msl": cat_h, "footprint_columns": cat_cols},
                        "direct": {"peak_w_mps": direct, "peak_w_ft_min": direct_ft,
                                   "peak_w_height_m_msl": height, "footprint_columns": int(cols.size)}})
                checked += 1
                n_cells += 1
            frames_done.append({"timestamp_ms": ts, "valid": meta["valid"], "cells": n_cells,
                                "wrfout": meta["path"]})
            print(f"{meta['valid']}: {n_cells} cells checked")
    wall = time.perf_counter() - tick
    peak_row = max(rows, key=lambda r: r["peak_w_mps"] if r.get("peak_w_mps") is not None else -1e9)
    receipt = {
        "schema": "gpuwm-cells-peak-w-check/1",
        "written": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "root": str(root),
        "reader": f"netCDF4 {netCDF4.__version__} direct on W, PH, PHB, XLAT, XLONG",
        "voxel_decode": "column = index mod (nx*ny); x = column mod nx; y = column div nx",
        "grid": {"nx": nx, "ny": ny, "dx_m": grid["dx_m"]},
        "frames_checked": frames_done,
        "cells_checked": checked,
        "cells_in_catalog": len(rows),
        "mismatches": mismatches,
        "worst_abs_difference_mps": worst_abs,
        "ft_per_min_per_mps": FT_PER_MIN_PER_MPS,
        "series_strongest": {k: peak_row[k] for k in (
            "valid_time", "track_id", "object_id", "peak_w_mps", "peak_w_ft_min",
            "peak_w_height_m_msl", "peak_w_lat", "peak_w_lon", "projected_area_km2", "max_dbz")},
        "wall_seconds": wall,
    }
    out_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"checked {checked} cells over {len(frames_done)} frames: "
          f"{len(mismatches)} mismatches, worst |dW| {worst_abs:.3g} m/s, {wall:.1f} s -> {out_path}")
    return 0 if checked and not mismatches else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
