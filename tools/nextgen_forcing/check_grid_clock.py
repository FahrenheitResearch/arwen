#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Exact shifted-weather regression; ncgen creates synthetic inputs only.

Python handles metadata/process orchestration. The production Rust executable
reads coordinates, transforms and remaps every field; no Python data processing.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--netcdf-format", choices=("classic", "netcdf4"), default="classic",
                        help="Exercise classic files or HDF5 coordinate dimension scales")
    args = parser.parse_args()
    root = args.scratch.resolve()
    root.mkdir(parents=True, exist_ok=True)
    table = Path(__file__).resolve().parents[2] / "gpuwm/data/hydro/forcing-sources.v1.json"
    profiles = json.loads(table.read_text())
    row = dict(profiles["profiles"]["decoded-canonical-grid"])
    row.update(time_variable="valid_end", time_offset_seconds=-3600)
    profiles["profiles"]["arbitrary-clock-source-731"] = row
    custom_table = root / "custom-table.json"
    custom_table.write_text(json.dumps(profiles, indent=2) + "\n")
    weights = root / "weights.json"
    weights.write_text(json.dumps(dict(catchment_ids=["catchment-1"], grid_shape=[1, 2],
                                      offsets=[0, 2], indices=[0, 1], weights=[0.25, 0.75])) + "\n")
    generated = []
    # Literal synthetic values, not transformations of user weather data.
    for filename, coordinate, stamp, temperature in (
            ("hour-0", "time", "0", "300,302"),
            ("hour-1", "time", "1", "310,312"),
            ("hour-2", "time", "2", "310,312"),
            ("end-1", "valid_end", "1", "300,302"),
            ("end-2", "valid_end", "2", "310,312")):
        cdl = root / (filename + ".cdl")
        cdl.write_text('''netcdf sample {
dimensions: time=1; y=1; x=2;
variables:
 double COORD(time); COORD:units="hours since 2015-12-01 00:00:00";
 COORD:calendar="proleptic_gregorian";
 double T2D(time,y,x); double Q2D(time,y,x); double U2D(time,y,x); double V2D(time,y,x);
 double PSFC(time,y,x); double LWDOWN(time,y,x); double SWDOWN(time,y,x); double RAINRATE(time,y,x);
data:
 COORD=STAMP; T2D=TEMP; Q2D=.005,.007; U2D=-1,3; V2D=2,4;
 PSFC=101000,100000; LWDOWN=350,360; SWDOWN=200,250; RAINRATE=.001,.003;
}
'''.replace("COORD", coordinate).replace("STAMP", stamp).replace("TEMP", temperature))
        argv = ["ncgen", "-k", "4" if args.netcdf_format == "netcdf4" else "1",
                "-o", str(root / (filename + ".nc")), str(cdl)]
        start = time.perf_counter()
        subprocess.run(argv, check=True, capture_output=True)
        generated.append(dict(argv=argv, seconds=time.perf_counter() - start))
    base = dict(source_profile=str(table), profile_row="decoded-canonical-grid",
                time_origin="2015-12-01T00:00:00Z", interval_seconds=3600,
                catchment_ids=["catchment-1"], area_weights=str(weights))
    results = []
    for label, names, changes, accepted in (
            ("ordered", ["hour-0", "hour-1"], {}, True),
            ("reversed", ["hour-1", "hour-0"], {}, False),
            ("repeated", ["hour-0", "hour-0"], {}, False),
            ("missing-hour", ["hour-0", "hour-2"], {}, False),
            ("wrong-epoch", ["hour-0", "hour-1"], {"time_origin": "2015-12-01T01:00:00Z"}, False),
            ("table-clock", ["end-1", "end-2"],
             {"source_profile": str(custom_table), "profile_row": "arbitrary-clock-source-731"}, True)):
        output = root / (label + "-packet")
        request = dict(base, inputs=[str(root / (name + ".nc")) for name in names],
                       output=str(output))
        request.update(changes)
        path = root / (label + "-request.json")
        path.write_text(json.dumps(request, indent=2) + "\n")
        argv = [str(args.executable.resolve()), str(path)]
        start = time.perf_counter()
        run = subprocess.run(argv, capture_output=True, text=True)
        result = dict(case=label, argv=argv, seconds=time.perf_counter() - start,
                      returncode=run.returncode, expected_acceptance=accepted, stderr=run.stderr)
        if run.returncode == 0:
            manifest = json.loads(run.stdout)
            result.update(data_sha256=manifest["data_sha256"], parallelism=manifest["parallelism"],
                          preparation_seconds=manifest["preparation_seconds"])
        result["pass"] = ((run.returncode == 0) if accepted else
                          (run.returncode != 0 and "timestamp differs" in run.stderr
                           and not (output / "manifest.json").exists()))
        results.append(result)
    same_packet = results[0].get("data_sha256") == results[-1].get("data_sha256")
    receipt = dict(scope="Synthetic NetCDF timestamp regression, not native or GPU proof",
                   netcdf_format=args.netcdf_format,
                   executable_sha256=hashlib.sha256(args.executable.read_bytes()).hexdigest(),
                   fixture_generation=generated, results=results,
                   table_only_clock_binding_exact=same_packet,
                   passed=all(row["pass"] for row in results) and same_packet)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
