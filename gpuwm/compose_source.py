"""Order independently exported segment frames through the shared Rust reader.

This module only launches the data processor. The manifest names source files,
their segment mesh/init provenance, coordinate names and required columns;
there is no producer-specific adapter or Python weather-array processing.
"""
from pathlib import Path
import argparse
import json
import subprocess
import sys


def register_cli(sub):
    parser = sub.add_parser(
        "compose-source", help="compose moving-segment exports for prep")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=0,
                        help="parallel frame readers (0: available cores)")
    parser.add_argument("--prepare", nargs=argparse.REMAINDER,
                        help="pass remaining options to gpuwm prep using the composed inputs and supplements")
    parser.set_defaults(func=main)


def main(args):
    from gpuwm.netcdf_bridge import find_netcdf_bin, netcdf_remedy
    binary = find_netcdf_bin()
    if binary is None:
        raise ValueError(netcdf_remedy())
    if args.jobs < 0:
        raise ValueError("--jobs must be nonnegative")
    capabilities = subprocess.run([str(binary), "--abi"], check=True,
                                  capture_output=True, text=True).stdout
    if "compose_source_v1" not in capabilities:
        raise ValueError("compose-source needs this release's rw_netcdf; "
                         + netcdf_remedy())
    result = subprocess.run([
        str(binary), "compose-source", str(args.manifest.resolve()),
        str(args.out.resolve()), str(args.jobs),
    ], check=False)
    if result.returncode or args.prepare is None:
        return result.returncode
    receipt = json.loads((args.out / "composition.json").read_text())
    command = [sys.executable, "-m", "gpuwm", "prep",
               "--input-list", str((args.out / "inputs.txt").resolve())]
    for binding in receipt["supplements"]:
        command.extend(("--supplement", binding))
    return subprocess.run([*command, *args.prepare], check=False).returncode
