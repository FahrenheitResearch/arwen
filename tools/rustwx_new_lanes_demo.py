#!/usr/bin/env python3
"""Drive the two new Rust renderer binaries against real inputs.

The demo leg for `rw_ensbatch` and `rw_obsgrid` (CLAUDE.md: "a capability
with no front door and no demo is not a feature").  Everything it feeds
them is written by the PROJECT'S OWN writers -- ``gpuwm.io.wrfout``'s
``WrfoutWriter`` for the ensemble members, ``gpuwm.obs.radar_grid``'s
``write_radar_grid`` for the observation grid -- so the readers are
exercised against the other lane's real writer rather than against a
fixture this file invented.

    python tools/rustwx_new_lanes_demo.py --bin-dir tools/rustwx/target/release \
        --work work/new-lanes --members 5

Prints one RENDERED line per panel and exits nonzero if any leg produced
no image.
"""

from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

_NZ, _NY, _NX = 6, 40, 48
_STAMP = "2026-07-28_20:00:00"


def _member_frame(seed: int) -> dict:
    """One member's state: a shared storm plus a member-specific wobble."""

    rng = np.random.default_rng(seed)
    lat = np.tile(np.linspace(34.4, 36.3, _NY)[:, None], (1, _NX))
    lon = np.tile(np.linspace(-98.4, -96.1, _NX)[None, :], (_NY, 1))
    yy, xx = np.mgrid[0:_NY, 0:_NX].astype(np.float32)
    # A common core, displaced a little per member: the spread field then
    # shows the displacement rather than noise, which is what makes the
    # demo's spread panel worth looking at.
    cy = _NY * 0.5 + rng.normal(0.0, 2.5)
    cx = _NX * 0.5 + rng.normal(0.0, 3.0)
    core = 62.0 * np.exp(-(((yy - cy) / 5.0) ** 2 + ((xx - cx) / 6.0) ** 2))
    refl2d = (core - 12.0).astype(np.float32)
    refl = np.stack([refl2d - level * 3.0 for level in range(_NZ)])
    return {
        "T": np.zeros((_NZ, _NY, _NX), np.float32),
        "MU": np.zeros((_NY, _NX), np.float32),
        "REFL_10CM": refl.astype(np.float32),
        "T2": (288.0 + 6.0 * np.exp(-(((yy - cy) / 9.0) ** 2))).astype(np.float32),
        "Q2": np.full((_NY, _NX), 0.010, np.float32),
        "PSFC": np.full((_NY, _NX), 97000.0, np.float32),
        "U10": (8.0 * np.sin(xx / 7.0)).astype(np.float32),
        "V10": (8.0 * np.cos(yy / 7.0)).astype(np.float32),
        "RAINC": np.zeros((_NY, _NX), np.float32),
        "RAINNC": np.clip(core * 0.6, 0.0, None).astype(np.float32),
        "XLAT": lat.astype(np.float32),
        "XLONG": lon.astype(np.float32),
        "HGT": np.full((_NY, _NX), 350.0, np.float32),
        "SINALPHA": np.zeros((_NY, _NX), np.float32),
        "COSALPHA": np.ones((_NY, _NX), np.float32),
    }


def build_ensemble(root: pathlib.Path, members: int) -> pathlib.Path:
    """`members` member run directories plus the manifest that rosters them."""

    from gpuwm.io.wrfout import WrfoutWriter, wrf_global_attrs

    if root.exists():
        shutil.rmtree(root)
    grid = SimpleNamespace(truelat1=33.0, truelat2=37.0, stand_lon=-97.2778,
                           ref_lat=35.3331, ref_lon=-97.2778)
    attrs = wrf_global_attrs(
        grid, datetime.datetime(2026, 7, 28, 20), grid_id=2, parent_id=1,
        i_parent_start=5, j_parent_start=5, parent_grid_ratio=3, dt=6.0)
    records = []
    for number in range(1, members + 1):
        member_dir = root / f"member_{number:03d}"
        member_dir.mkdir(parents=True)
        path = member_dir / "wrfout_d02_2026-07-28_20-00-00.nc"
        with WrfoutWriter(path, nx=_NX, ny=_NY, nz=_NZ, dx=2000.0, dy=2000.0,
                          global_attrs=attrs) as writer:
            writer.write_frame(_STAMP, _member_frame(seed=1000 + number))
        records.append({"member": number, "member_dir": member_dir.name,
                        "status": "DONE"})
    manifest = root / "ensemble-manifest.json"
    manifest.write_text(json.dumps({
        "schema": "gpuwm-ensemble-manifest.v1",
        "n_members": members,
        "members": records,
    }, indent=2), encoding="utf-8")
    return manifest


def build_obs_grid(path: pathlib.Path) -> pathlib.Path:
    """One real ``gpuwm-obs.radar-grid.v1`` file, from the real writer."""

    from gpuwm.obs.radar_grid import write_radar_grid
    from gpuwm.obs.sweeps import Moment, RadarSite, RadarVolume, Sweep
    from gpuwm.obs.superob import (SuperobParams, merge_contributions,
                                   superob_volume)
    from gpuwm.obs.target_grid import TargetGrid
    from gpuwm.static.lambert import LambertGrid

    projection = LambertGrid(
        ref_lat=35.3331, ref_lon=-97.2778, truelat1=33.0, truelat2=37.0,
        stand_lon=-97.2778, dx=2000.0, dy=2000.0, e_we=_NX + 1, e_sn=_NY + 1)
    grid = TargetGrid.from_projection(
        projection, z_w=np.linspace(0.0, 10000.0, 11), name="demo")

    gates = 60
    azimuths = np.arange(0.0, 360.0, 2.0, dtype=np.float32)
    reflectivity = np.tile(
        np.linspace(10.0, 58.0, gates, dtype=np.float32)[None, :],
        (azimuths.size, 1))
    velocity = np.tile(
        np.linspace(-24.0, 24.0, gates, dtype=np.float32)[None, :],
        (azimuths.size, 1))
    sweep = Sweep(
        sweep_index=0, elevation_number=1, elevation_angle_deg=0.5,
        nyquist_velocity_ms=32.0, start_status=3, end_status=2,
        cut_sector=0, complete=True, azimuth_deg=azimuths,
        elevation_deg=np.full(azimuths.size, 0.5, dtype=np.float32),
        moments={
            "REF": Moment("REF", "dBZ", gates, 2125.0, 250.0, reflectivity),
            "VEL": Moment("VEL", "m/s", gates, 2125.0, 250.0, velocity),
        })
    centre_j, centre_i = grid.ny // 2, grid.nx // 2
    volume = RadarVolume(
        site=RadarSite(id="KTLX", name="demo",
                       lat_deg=float(grid.lat[centre_j, centre_i]),
                       lon_deg=float(grid.lon[centre_j, centre_i]),
                       alt_m=380.0, source="demo"),
        valid_time="2026-07-28T20:03:16Z", station_id="KTLX",
        volume_file="KTLX20260728_200316_V06", volume_sha256="0" * 64,
        volume_bytes=1, pack_path=pathlib.Path("demo.rdrpack"),
        pack_sha256="1" * 64,
        params={"moments": ["REF", "VEL"], "max_range_km": 250.0},
        framing={"magic": "AR2V0006", "block_count": 1}, sweeps=(sweep,))

    params = SuperobParams()
    observations = merge_contributions(
        [superob_volume(volume, grid, params=params)], grid, params=params,
        z_reduce="max")
    path.parent.mkdir(parents=True, exist_ok=True)
    write_radar_grid(path, observations, grid,
                     valid_time="2026-07-28T20:03:16Z", params=params,
                     overwrite=True)
    return path


def drive(command: list[str], label: str) -> int:
    print(f"\n$ {' '.join(command)}")
    result = subprocess.run(command, capture_output=True, text=True,
                            errors="replace")
    rendered = 0
    for line in result.stdout.splitlines():
        if line.startswith(("RENDERED ", "SKIPPED ", "MEMBERS ", "COVERAGE ",
                            "TIES ", "OBSERVED ", "FINISHED ",
                            "PAINTBALL_LEGEND", "SITES")):
            print(f"  {line}")
        if line.startswith("RENDERED "):
            rendered += 1
    if result.returncode != 0:
        sys.stderr.write(result.stdout[-3000:])
        sys.stderr.write(result.stderr[-3000:])
        raise SystemExit(f"{label} exited {result.returncode}")
    return rendered


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin-dir", required=True)
    parser.add_argument("--work", required=True)
    parser.add_argument("--members", type=int, default=5)
    parser.add_argument("--width", type=int, default=1200)
    parser.add_argument("--height", type=int, default=900)
    args = parser.parse_args(argv)

    bin_dir = pathlib.Path(args.bin_dir)
    suffix = ".exe" if sys.platform.startswith("win") else ""
    ensbatch = bin_dir / f"rw_ensbatch{suffix}"
    obsgrid = bin_dir / f"rw_obsgrid{suffix}"
    for binary in (ensbatch, obsgrid):
        if not binary.is_file():
            raise SystemExit(
                f"{binary} is not built (cd tools/rustwx; cargo build "
                "--release --locked --offline)")

    work = pathlib.Path(args.work)
    work.mkdir(parents=True, exist_ok=True)

    manifest = build_ensemble(work / "ensemble", args.members)
    print(f"ensemble: {args.members} member wrfout(s) under {manifest.parent}")
    ens_out = work / "ens-png"
    total = drive([
        str(ensbatch),
        "--store-root", str(work / "ens-store"),
        "--out-dir", str(ens_out),
        "--manifest", str(manifest),
        "--field", "refl",
        "--products", "mean,spread,prob,pmm,paintball",
        "--threshold", "35",
        "--neighborhood-km", "20",
        "--width", str(args.width), "--height", str(args.height),
    ], "rw_ensbatch")

    obs_path = build_obs_grid(work / "obs" / "radar-grid.nc")
    print(f"observations: {obs_path} "
          f"({obs_path.stat().st_size / 1e6:.1f} MB)")
    obs_out = work / "obs-png"
    total += drive([
        str(obsgrid),
        "--obs", str(obs_path),
        "--out-dir", str(obs_out),
        "--products", "all",
        "--sites", "--rings-km", "50,100,150",
        "--width", str(args.width), "--height", str(args.height),
    ], "rw_obsgrid")

    pngs = sorted(list(ens_out.rglob("*.png")) + list(obs_out.rglob("*.png")))
    print(f"\nDEMO rendered={total} png_files={len(pngs)}")
    for png in pngs:
        print(f"  {png.stat().st_size:>9,} B  {png}")
    if not pngs:
        raise SystemExit("the demo produced no images")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
