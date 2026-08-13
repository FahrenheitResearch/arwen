"""Time the observation front door on one real radar volume.

The dealiasing lane carried counts and no clock, so its position on any
ranking was arithmetic rather than measurement.  This runs the real gridding
path over a real Level-II volume with :mod:`gpuwm.perf_timing` on, and prints
the per-stage receipt plus the arms needed to price the capability: the same
volume gridded with dealiasing off, so the dealias cost is a difference and
not an assertion.

    python tools/perf_obs_timing.py --volume <level2 file> --out receipt.json

``--pack`` skips the decode when a sweep pack already exists.  Nothing is
written inside the source tree: the decode lands wherever ``--work`` says,
and ``--work`` defaults to a temporary directory that is left in place so a
second run can reuse it.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from gpuwm import perf_timing
from gpuwm.obs import nexrad
from gpuwm.obs.dealias import DealiasParams
from gpuwm.obs.superob import SuperobParams, superob_volume
from gpuwm.obs.sweeps import read_sweep_pack
from gpuwm.obs.target_grid import TargetGrid
from gpuwm.static.lambert import LambertGrid


def _grid_around(lat: float, lon: float, *, nx: int, ny: int, nz: int,
                 dx_m: float, top_m: float) -> TargetGrid:
    """A plain Lambert box centred on the radar, at the asked-for spacing."""
    projection = LambertGrid(
        ref_lat=lat, ref_lon=lon, truelat1=lat, truelat2=lat, stand_lon=lon,
        dx=dx_m, dy=dx_m, e_we=nx + 1, e_sn=ny + 1)
    z_w = np.linspace(0.0, top_m, nz + 1)
    return TargetGrid.from_projection(projection, z_w=z_w, name="target",
                                      source="perf-timing-box")


def _decode(volume: Path, work: Path) -> Path:
    binary = nexrad.find_nexrad_bin()
    if binary is None:
        raise SystemExit(
            "the Rust NEXRAD decoder is not on this box; "
            + nexrad.nexrad_remedy())
    work.mkdir(parents=True, exist_ok=True)
    pack = work / (volume.name + ".sweeps")
    if pack.is_file():
        return pack
    started = time.perf_counter()
    nexrad.run_decode(binary, volume=volume, out=pack,
                      moments=("REF", "VEL"))
    perf_timing.record("obs.decode.rw_nexrad", time.perf_counter() - started,
                       bytes_read=volume.stat().st_size)
    return pack


def _arm(pack: Path, grid: TargetGrid, *, dealias: bool) -> dict:
    perf_timing.reset(on=True)
    params = SuperobParams(dealias=DealiasParams() if dealias else None)
    volume = read_sweep_pack(pack)
    started = time.perf_counter()
    contribution = superob_volume(volume, grid, params=params)
    wall = time.perf_counter() - started
    receipt = perf_timing.snapshot()
    receipt["arm"] = "dealias-on" if dealias else "dealias-off"
    receipt["superob_volume_seconds"] = round(wall, 6)
    receipt["sweeps"] = len(volume.sweeps)
    counts = contribution.counts
    receipt["velocity_cells"] = int((contribution.vr_count > 0).sum())
    receipt["velocity_gates_rejected_nyquist"] = int(
        getattr(counts, "velocity_gates_rejected_nyquist", 0))
    if contribution.dealias:
        receipt["dealias_totals"] = contribution.dealias.get("totals")
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--volume", type=Path,
                        help="a Level-II volume to decode first")
    source.add_argument("--pack", type=Path,
                        help="an already-decoded sweep pack")
    parser.add_argument("--work", type=Path, default=None,
                        help="where the decode lands (default: a temp dir)")
    parser.add_argument("--out", type=Path, default=None,
                        help="write the JSON receipt here")
    parser.add_argument("--nx", type=int, default=200)
    parser.add_argument("--ny", type=int, default=200)
    parser.add_argument("--nz", type=int, default=40)
    parser.add_argument("--dx-m", type=float, default=3000.0)
    parser.add_argument("--top-m", type=float, default=16000.0)
    parser.add_argument("--repeat", type=int, default=1,
                        help="run each arm this many times, report the last")
    args = parser.parse_args(argv)

    work = args.work or Path(tempfile.gettempdir()) / "gpuwm-perf-obs"
    pack = args.pack if args.pack is not None else _decode(args.volume, work)
    volume = read_sweep_pack(pack)
    site = volume.site
    grid = _grid_around(site.lat_deg, site.lon_deg, nx=args.nx, ny=args.ny,
                        nz=args.nz, dx_m=args.dx_m, top_m=args.top_m)

    arms = []
    for dealias in (False, True):
        for _ in range(max(1, args.repeat)):
            receipt = _arm(pack, grid, dealias=dealias)
        arms.append(receipt)

    off, on = arms
    report = {
        "schema": "gpuwm-perf.obs-timing.v1",
        "pack": str(pack),
        "site": site.id,
        "valid_time": str(volume.valid_time),
        "grid": {"nx": grid.nx, "ny": grid.ny, "nz": grid.nz,
                 "dx_m": grid.dx_m},
        "arms": arms,
        "dealias_seconds": round(on["superob_volume_seconds"]
                                 - off["superob_volume_seconds"], 6),
        "dealias_share_of_gridding": round(
            (on["superob_volume_seconds"] - off["superob_volume_seconds"])
            / on["superob_volume_seconds"], 4),
    }
    text = json.dumps(report, indent=2)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
