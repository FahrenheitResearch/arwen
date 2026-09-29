"""Prepare portable statics witnesses with an explicitly selected fresh bridge.

Run on Windows with GPUWM_STATIC_BRIDGE set to the freshly built library.
The saved footprints also drive the Linux moving-nest GPU smoke run.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]

from gpuwm.native_wrf_contract import (native_static_export_fields,
    write_native_geometry_receipt, write_native_static_cache)
from gpuwm.static import rust_bridge
from gpuwm.static.build import build_static
from test_static_projection_portability import geog_30arcsecond, witness_grid, field_hashes, WITNESS_SPECS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--gpu-geog-root", type=Path)
    args = parser.parse_args()
    if not os.environ.get("GPUWM_STATIC_BRIDGE"):
        parser.error("GPUWM_STATIC_BRIDGE must name the freshly built library")
    bridge = rust_bridge.resolve_static_bridge()
    if rust_bridge.route("portable_projection_witness") is None:
        parser.error("a Python fallback cannot produce the portable Rust witness")
    record = {"platform": platform.platform(), "bridge": str(bridge),
              "bridge_sha256": hashlib.sha256(bridge.read_bytes()).hexdigest(),
              "grids": {}}
    print(json.dumps({k: v for k, v in record.items() if k != "grids"}), flush=True)
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    for spacing, spec in WITNESS_SPECS.items():
        geog = geog_30arcsecond(root / f"geog-{spacing}", spec["origin"])
        grid = witness_grid(spacing)
        fields = build_static(grid, geog)
        write_native_static_cache(root / f"statics-{spacing}.npz", fields)
        record["grids"][str(spacing)] = {
            "overlap_sha256": field_hashes(fields, slice(3, None), slice(3, None)),
            "grid_definition": grid.definition(), "terrain_spacing_degrees": 1 / 120,
        }
        print(f"prepared {spacing} m with {rust_bridge.resolve_static_bridge()}: "
              f"{record['grids'][str(spacing)]['overlap_sha256']}", flush=True)
    grid = witness_grid("off_center_2000")
    geog = args.gpu_geog_root or root / "geog-off_center_2000"
    fields = native_static_export_fields(build_static(grid, geog), grid)
    path = root / "gpu-windows-statics.npz"
    write_native_static_cache(path, fields)
    write_native_geometry_receipt(root / "gpu-windows-geometry.json", grid,
        SimpleNamespace(nx=240, ny=240, nz=24, dx=2000., dy=2000.), path)
    (root / "manifest.json").write_bytes((json.dumps(record, indent=2) + "\n").encode())


if __name__ == "__main__":
    main()
