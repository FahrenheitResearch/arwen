"""Dual-write verification for the gridded observation products.

    python tools/obs_grid_dual_write.py --workdir DIR --receipt OUT.json

Writes ``gpuwm-obs.radar-grid.v1`` and ``gpuwm-obs.goes-grid.v1`` through
BOTH engines -- Drew's Rust classic writer (the default) and the netCDF4
workaround reached by ``GPUWM_OBS_GRID_WRITER=python`` -- at a full-size
grid, reads each back through the shipped reader, and compares the two:
variable inventory and definition order, dimension extents, global and
per-variable attributes (type as well as value), and every value bit.
Exits non-zero on any difference.

This is the front door for the claim the flip rests on, in the shape
``tools/wrfout_dual_write.py`` established: the full-size comparison can be
re-run on demand rather than quoted from a transcript.  The receipt records
the containers and the SIZES as well, because the classic container carries
no compression and that cost should be stated rather than discovered.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gpuwm.obs.goes_cwp import GriddedCwp                    # noqa: E402
from gpuwm.obs.goes_grid import read_goes_grid, write_goes_grid  # noqa: E402
from gpuwm.obs.grid_product import OBS_GRID_WRITER_ENV       # noqa: E402
from gpuwm.obs.radar_grid import read_radar_grid, write_radar_grid  # noqa: E402
from gpuwm.obs.superob import GriddedObservations, SuperobParams  # noqa: E402
from gpuwm.obs.target_grid import TargetGrid                 # noqa: E402
from gpuwm.static.lambert import LambertGrid                 # noqa: E402

RECEIPT_SCHEMA = "gpuwm-obs-grid-dual-write-receipt-v1"


def build_grid(nx: int, ny: int, nz: int, dx: float) -> TargetGrid:
    projection = LambertGrid(
        ref_lat=35.3331, ref_lon=-97.2778, truelat1=33.0, truelat2=37.0,
        stand_lon=-97.2778, dx=dx, dy=dx, e_we=nx + 1, e_sn=ny + 1)
    return TargetGrid.from_projection(
        projection, z_w=np.linspace(0.0, 15000.0, nz + 1), name="dual-write")


def build_radar(grid: TargetGrid, radars: int) -> GriddedObservations:
    """A full-size observation set with structure, not a constant field."""

    rng = np.random.default_rng(20260818)
    plane = (grid.nz, grid.ny, grid.nx)
    volume = (radars,) + plane
    z_obs = rng.uniform(-10.0, 65.0, plane).astype(np.float32)
    z_mask = (rng.random(plane) > 0.4).astype(np.int8)
    # Beam unit vectors: the reader checks they are unit length under the
    # mask, so they are built as directions rather than as noise.
    theta = rng.uniform(0.0, 2.0 * np.pi, volume)
    phi = rng.uniform(0.0, 0.35, volume)
    east = (np.cos(phi) * np.cos(theta)).astype(np.float32)
    north = (np.cos(phi) * np.sin(theta)).astype(np.float32)
    up = np.sin(phi).astype(np.float32)
    norm = np.sqrt(east.astype(np.float64) ** 2 + north.astype(np.float64) ** 2
                   + up.astype(np.float64) ** 2)
    east = (east / norm).astype(np.float32)
    north = (north / norm).astype(np.float32)
    up = (up / norm).astype(np.float32)
    return GriddedObservations(
        z_obs=z_obs, z_mask=z_mask,
        z_err=rng.uniform(1.0, 6.0, plane).astype(np.float32),
        z_max=z_obs + np.float32(1.5), z_mean=z_obs - np.float32(0.5),
        z_count=rng.integers(0, 40, plane).astype(np.int32),
        vr_obs=rng.uniform(-32.0, 32.0, volume).astype(np.float32),
        vr_mask=(rng.random(volume) > 0.5).astype(np.int8),
        vr_err=rng.uniform(0.5, 4.0, volume).astype(np.float32),
        vr_count=rng.integers(0, 30, volume).astype(np.int32),
        vr_rejected=rng.integers(0, 5, volume).astype(np.int32),
        vr_beam_east=east, vr_beam_north=north, vr_beam_up=up,
        radars=[{"id": f"K{index:03d}", "lat_deg": 35.0 + 0.1 * index,
                 "lon_deg": -97.0 - 0.1 * index, "alt_m": 300.0 + index,
                 "valid_time": "2026-08-18T21:00:00Z"}
                for index in range(radars)],
        counts=[{"radar": index, "gates": 100000} for index in range(radars)],
        provenance=[{"radar": index, "volume": f"synthetic-{index}"}
                    for index in range(radars)])


def build_goes(grid: TargetGrid) -> GriddedCwp:
    rng = np.random.default_rng(20260819)
    shape = (grid.ny, grid.nx)
    mask = (rng.random(shape) > 0.5).astype(np.int8)
    observed = mask.astype(bool)
    values = np.zeros(shape, dtype=np.float64)
    values[observed] = rng.uniform(0.0, 1200.0, int(observed.sum()))
    classes = np.full(shape, -1, dtype=np.int8)
    classes[observed] = rng.integers(1, 3, int(observed.sum())).astype(np.int8)
    errors = np.zeros(shape, dtype=np.float64)
    errors[observed] = rng.uniform(20.0, 200.0, int(observed.sum()))
    levels = np.full(shape, -1, dtype=np.int32)
    levels[observed] = rng.integers(0, grid.nz, int(observed.sum()))
    tops = np.full(shape, np.nan, dtype=np.float64)
    tops[observed] = rng.uniform(2000.0, 14000.0, int(observed.sum()))
    return GriddedCwp(
        cwp_obs=values, cwp_mask=mask, cwp_err=errors, cwp_class=classes,
        cwp_count=(mask * 6).astype(np.int32),
        cwp_pixels=(mask * 9).astype(np.int32),
        cloud_top_height_m=tops, obs_level=levels,
        counts={"observed": int(observed.sum())},
        provenance={"join": None, "error_model": {"note": "dual-write"}})


def _inventory(path: Path) -> dict:
    import netCDF4

    with netCDF4.Dataset(path) as dataset:
        return {
            "variables": list(dataset.variables),
            "dimensions": {name: len(dimension)
                           for name, dimension in dataset.dimensions.items()},
            "global_attributes": {
                name: [np.asarray(dataset.getncattr(name)).dtype.str,
                       np.asarray(dataset.getncattr(name)).tolist()]
                for name in dataset.ncattrs()},
            "variable_attributes": {
                name: {key: [np.asarray(variable.getncattr(key)).dtype.str,
                             np.asarray(variable.getncattr(key)).tolist()]
                       for key in variable.ncattrs()}
                for name, variable in dataset.variables.items()},
        }


def _values(path: Path) -> dict[str, bytes]:
    import netCDF4

    with netCDF4.Dataset(path) as dataset:
        dataset.set_auto_mask(False)
        return {name: np.asarray(variable[:]).tobytes()
                for name, variable in dataset.variables.items()}


def _container(path: Path) -> str:
    head = path.read_bytes()[:8]
    if head[:3] == b"CDF":
        return f"CDF-{head[3]}"
    if head == b"\x89HDF\r\n\x1a\n":
        return "HDF5"
    return repr(head)


def _write(kind: str, path: Path, engine: str, payload, grid) -> dict:
    os.environ[OBS_GRID_WRITER_ENV] = engine
    try:
        started = time.perf_counter()
        if kind == "radar":
            receipt = write_radar_grid(
                path, payload, grid, valid_time="2026-08-18T21:00:00Z",
                params=SuperobParams(), overwrite=True)
        else:
            receipt = write_goes_grid(
                path, payload, grid, valid_time="2026-08-18T21:00:00Z",
                overwrite=True)
        elapsed = time.perf_counter() - started
    finally:
        os.environ.pop(OBS_GRID_WRITER_ENV, None)
    return {"engine": engine, "path": str(path), "seconds": round(elapsed, 3),
            "bytes": receipt["bytes"], "sha256": receipt["sha256"],
            "container": _container(path)}


def compare(kind: str, workdir: Path, payload, grid) -> dict:
    rust = _write(kind, workdir / f"{kind}-rust.nc", "rust", payload, grid)
    python = _write(kind, workdir / f"{kind}-python.nc", "python", payload,
                    grid)

    differences: list[str] = []
    left, right = _inventory(Path(rust["path"])), _inventory(Path(python["path"]))
    for section in ("variables", "dimensions", "global_attributes",
                    "variable_attributes"):
        if left[section] != right[section]:
            differences.append(f"{section} differ")

    left_values, right_values = (_values(Path(rust["path"])),
                                 _values(Path(python["path"])))
    if set(left_values) != set(right_values):
        differences.append("variable sets differ")
    for name in sorted(set(left_values) & set(right_values)):
        if left_values[name] != right_values[name]:
            differences.append(f"values differ: {name}")

    # And each file is read back through the SHIPPED reader, so the
    # contract checks (grid binding, coordinate digests, beam unit length)
    # run against both engines' output rather than only the default's.
    reader = read_radar_grid if kind == "radar" else read_goes_grid
    documents = {}
    for side in (rust, python):
        started = time.perf_counter()
        document = reader(Path(side["path"]), expected_grid=grid)
        side["read_seconds"] = round(time.perf_counter() - started, 3)
        documents[side["engine"]] = document
    for name in sorted(documents["rust"]["variables"]):
        if (np.asarray(documents["rust"]["variables"][name]).tobytes()
                != np.asarray(documents["python"]["variables"][name]).tobytes()):
            differences.append(f"read-back values differ: {name}")

    return {
        "product": kind,
        "variables": len(left["variables"]),
        "rust": rust,
        "python": python,
        "size_ratio": round(rust["bytes"] / python["bytes"], 3),
        "differences": differences,
        "verdict": "PASS" if not differences else "FAIL",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="obs_grid_dual_write",
                                     description=__doc__)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--nx", type=int, default=200)
    parser.add_argument("--ny", type=int, default=200)
    parser.add_argument("--nz", type=int, default=30)
    parser.add_argument("--dx", type=float, default=2000.0)
    parser.add_argument("--radars", type=int, default=2)
    args = parser.parse_args(argv)

    args.workdir.mkdir(parents=True, exist_ok=True)
    grid = build_grid(args.nx, args.ny, args.nz, args.dx)
    results = [
        compare("radar", args.workdir, build_radar(grid, args.radars), grid),
        compare("goes", args.workdir, build_goes(grid), grid),
    ]
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "grid": {"nx": args.nx, "ny": args.ny, "nz": args.nz,
                 "dx_m": args.dx, "radars": args.radars},
        "products": results,
        "verdict": ("PASS" if all(item["verdict"] == "PASS" for item in results)
                    else "FAIL"),
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True),
                            encoding="utf-8", newline="\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
