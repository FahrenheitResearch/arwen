"""Score stored forecast AOD550 against public AERONET V3 Level 2 files.

Only command orchestration lives here. Rust reads the observation CSV/NetCDF,
interpolates measured spectra and model times, matches stations and scores pairs.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import tempfile


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, nargs="+", required=True)
    parser.add_argument("--frames", type=Path, nargs="*", default=[])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-distance-km", type=float, default=10.)
    parser.add_argument("--max-time-gap-seconds", type=int, default=3600)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    from gpuwm.netcdf_bridge import find_netcdf_bin
    binary = find_netcdf_bin()
    if binary is None:
        parser.error("AERONET verification requires the bundled rw_netcdf reader")
    abi = subprocess.run([str(binary), "--abi"], check=True, capture_output=True, text=True, timeout=30)
    if "aeronet_aod_v1" not in abi.stdout:
        parser.error("rw_netcdf needs aeronet_aod_v1; rebuild tools/rustwx package rw-netcdf")
    manifest = dict(schema="gpuwm-aeronet-aod-v1",
                    observations=[str(p.resolve(strict=True)) for p in args.observations],
                    frames=[str(p.resolve(strict=True)) for p in args.frames],
                    max_distance_km=args.max_distance_km,
                    max_time_gap_seconds=args.max_time_gap_seconds)
    args.out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="aeronet-request-", dir=args.out) as work:
        request = Path(work) / "manifest.json"
        request.write_text(json.dumps(manifest, allow_nan=False), encoding="utf-8")
        subprocess.run([str(binary), "aeronet-aod", str(request), str(args.out), str(args.workers)], check=True)
    print((args.out / "summary.json").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
