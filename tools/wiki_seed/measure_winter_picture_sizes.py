"""Measure missing picture rows with rw_wrfbatch, then use the table's collector.

CPU-only fixture seeded from rw-wrfbatch/tests/winter.rs, with varied columns
and both price brackets. This measures PNG bytes, not forecast skill. Run from
the repository root with --renderer PATH --scratch OWNED_EMPTY_DIRECTORY.
Raw inputs, stores and PNGs stay in scratch; retain the audit and size record.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import struct
import subprocess
from types import SimpleNamespace

import numpy as np

from gpuwm.io.wrfout import WrfoutWriter, wrf_global_attrs
from tools.wiki_seed.collect_picture_sizes import RECORD, TABLE, build_table, collect, write_document

PRODUCTS = (
    "10m_wind_gusts", "categorical_rain", "categorical_freezing_rain",
    "categorical_ice_pellets", "categorical_snow", "precipitation_type",
    "snowfall_window", "snow_10to1_window", "snow_kuchera_window",
)
AUDIT = RECORD.with_name("winter-products-audit.json")


def fixture(path, nx, ny, gid, hour):
    """The Rust winter fixture's required fields, with spatial detail."""
    nz = 4
    y, x = np.mgrid[:ny, :nx]
    detail = (1 + np.sin(x / 3) * np.cos(y / 4)).astype(np.float32)
    zone = (x * 4 // nx).clip(0, 3)
    def plane(value):
        return np.full((ny, nx), value, np.float32)

    def volume(values):
        return np.broadcast_to(np.asarray(values, np.float32)[:, None, None],
                               (len(values), ny, nx)).copy()
    pressure = np.array([90000., 80000., 65000., 50000.])
    temp = volume([270., 272., 268., 260.])
    temp[:, :, zone[0] < 1] += 5
    temp[1, :, zone[0] == 2] = 276
    fields = {
        "HGT": plane(1000), "PBLH": 300 + 300 * detail,
        "T2": np.where(zone == 0, 277, 270).astype(np.float32),
        "PSFC": plane(90000), "U10": 3 + 6 * detail, "V10": plane(1),
        "SNOWNC": hour * detail, "GRAUPELNC": plane(0),
        "SNOWFALLAC": 10 * hour * detail,
        "XLAT": (42 + y * 6 / max(1, ny - 1)).astype(np.float32),
        "XLONG": (-116 + x * 8 / max(1, nx - 1)).astype(np.float32),
        "T": (temp / (pressure[:, None, None] / 100000) ** (2 / 7) - 300).astype(np.float32),
        "P": volume([0] * nz), "PB": volume(pressure),
        "QVAPOR": volume([0.001] * nz),
        "QRAIN": np.broadcast_to(np.where(zone < 2, 0.0002, 0.000001), (nz, ny, nx)).astype(np.float32).copy(),
        "QSNOW": np.broadcast_to(np.where(zone == 3, 0.0002, 0.000001), (nz, ny, nx)).astype(np.float32).copy(),
        "QGRAUP": np.broadcast_to(np.where(zone == 2, 0.0002, 0), (nz, ny, nx)).astype(np.float32).copy(),
        "U": np.full((nz, ny, nx + 1), 20, np.float32),
        "V": np.zeros((nz, ny + 1, nx), np.float32),
        "PH": volume([0] * (nz + 1)),
        "PHB": volume(np.array([1000, 1300, 1700, 2300, 4000]) * 9.81),
    }
    grid = SimpleNamespace(truelat1=42., truelat2=48., stand_lon=-112.,
                           ref_lat=45., ref_lon=-112.)
    start = dt.datetime(2026, 1, 1)
    attrs = wrf_global_attrs(grid, start, grid_id=gid, parent_id=1,
                             i_parent_start=1, j_parent_start=1,
                             parent_grid_ratio=1, dt=6.)
    with WrfoutWriter(path, nx=nx, ny=ny, nz=nz, dx=16000, dy=16000,
                      global_attrs=attrs) as writer:
        writer.write_frame((start + dt.timedelta(hours=hour)).strftime("%Y-%m-%d_%H:%M:%S"), fields)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--renderer", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    args = parser.parse_args()
    if args.scratch.exists():
        raise ValueError("scratch must not exist: inputs and PNGs belong to this measurement")
    args.scratch.mkdir(parents=True)
    catalog_run = subprocess.run([str(args.renderer), "--list-products"], check=True,
                                  capture_output=True, text=True)
    record = json.loads(RECORD.read_text())
    catalog = dict(record["catalog"])
    for line in catalog_run.stdout.splitlines():
        row = line.split("\t")
        if len(row) >= 5 and row[0] == "WRFOUT" and row[1] in PRODUCTS:
            if row[3] != "drawable":
                raise ValueError(f"renderer cannot measure {row[1]}: {line}")
            catalog[row[1]] = [row[2], float(row[4])]
    if not set(PRODUCTS) <= catalog.keys():
        raise ValueError("renderer lacks the required winter product catalog")
    audit = {
        "checked_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "basis": "CPU fixture seeded from rw-wrfbatch/tests/winter.rs; default PNG size; both grid brackets",
        "renderer_sha256": hashlib.sha256(args.renderer.read_bytes()).hexdigest(),
        "renderer_abi": subprocess.check_output([str(args.renderer), "--abi"], text=True).strip(),
        "product_catalog": {name: catalog[name] for name in PRODUCTS},
        "commands": [], "frames": [], "pictures": [], "inputs": [],
    }
    for gid, (nx, ny) in enumerate(((36, 36), (156, 124)), 1):
        directory = args.scratch / f"d{gid:02}"
        directory.mkdir()
        for hour in (0, 6):
            # Snowfall history discovery uses WRF's ordinary timestamp name.
            path = directory / f"wrfout_d{gid:02}_2026-01-01_{hour:02}_00_00"
            fixture(path, nx, ny, gid, hour)
            audit["inputs"].append({"path": path.name, "bytes": path.stat().st_size,
                                     "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            audit["frames"].append({"attributes": {"GRID_ID": gid, "DX": 16000, "DY": 16000},
                                     "dimensions": {"west_east": nx, "south_north": ny}})
        output = directory / "png"
        command = [str(args.renderer), "--store-root", str(directory / "store"),
                   "--out-dir", str(output), "--products", ",".join(PRODUCTS), str(path)]
        result = subprocess.run(command, capture_output=True, text=True)
        (directory / "render.log").write_text(result.stdout + result.stderr)
        result.check_returncode()
        audit["commands"].append(command)
        for picture in sorted(output.rglob("*.png")):
            relative = picture.relative_to(output)
            product = next((name for name in PRODUCTS
                            if picture.name.endswith(f"_{name}.png")), None)
            if product is None:
                raise ValueError(f"unattributed picture: {relative}")
            payload = picture.read_bytes()
            audit["pictures"].append({"product": product, "bytes": len(payload),
                "path": f"png/d{gid:02}-fixture/{relative.as_posix()}",
                "sha256": hashlib.sha256(payload).hexdigest(),
                "pixels": list(struct.unpack(">II", payload[16:24]))})
        if {p["product"] for p in audit["pictures"] if f"d{gid:02}-fixture" in p["path"]} != set(PRODUCTS):
            raise ValueError(f"not every product rendered on grid {gid}: see {directory / 'render.log'}")
    audit["png_count"] = len(audit["pictures"])
    audit_path = args.scratch / "artifact-audit.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    addition = collect(args.scratch, catalog)
    record["catalog"] = addition["catalog"]
    record["runs"].update(addition["runs"])
    record["pictures"] = sorted(record["pictures"] + addition["pictures"])
    write_document(RECORD, record, "pictures")
    write_document(TABLE, build_table(record), "products")
    AUDIT.write_text(json.dumps(audit, indent=2) + "\n")
    print(f"Measured {audit['png_count']} PNGs; retained audit {AUDIT}")


if __name__ == "__main__":
    main()
