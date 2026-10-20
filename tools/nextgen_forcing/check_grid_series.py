#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Exercise Rust multi-time NetCDF preparation on literal synthetic fixtures.

Python generates ncgen metadata and runs processes. Rust decodes, orders,
transforms and remaps fields; packet hashes compare transport, not physics.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", required=True, type=Path)
    parser.add_argument("--scratch", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    root = args.scratch.resolve()
    root.mkdir(parents=True, exist_ok=True)
    table = Path(__file__).resolve().parents[2] / "gpuwm/data/hydro/forcing-sources.v1.json"
    profiles = json.loads(table.read_text())
    row = json.loads(json.dumps(profiles["profiles"]["decoded-canonical-grid"]))
    row["fields"]["RAINRATE"]["temporal"] = "cumulative"
    profiles["profiles"]["arbitrary-cumulative-series-731"] = row
    custom = root / "table.json"
    custom.write_text(json.dumps(profiles, indent=2) + "\n")
    weights = root / "weights.json"
    weights.write_text(json.dumps(dict(catchment_ids=["catchment-1"], grid_shape=[2, 2],
                                      offsets=[0, 4], indices=[0, 1, 2, 3],
                                      weights=[0.125, 0.375, 0.125, 0.375])) + "\n")
    baseline = root / "baseline.f64"
    baseline.write_bytes(b"\x00" * 32)  # Four literal synthetic zero f64 cells.
    generated = []
    # Deliberately square time/spatial extents: a length-only check cannot find
    # a misplaced time axis. Literal permutations are independently declared.
    for name, axes, stamps, temperature, rain, count in (
            ("single-0", "time,y,x", "0", "300,302,304,306", "1,2,3,4", 4),
            ("single-1", "time,y,x", "1", "310,312,314,316", "3,5,7,9", 4),
            ("first", "time,y,x", "0,1", "300,302,304,306,310,312,314,316", "1,2,3,4,3,5,7,9", 8),
            ("middle", "y,time,x", "0,1", "300,302,310,312,304,306,314,316", "1,2,3,5,3,4,7,9", 8),
            ("last", "y,x,time", "0,1", "300,310,302,312,304,314,306,316", "1,3,2,5,3,7,4,9", 8),
            ("repeated", "time,y,x", "0,0", "300,302,304,306,310,312,314,316", "1,2,3,4,3,5,7,9", 8),
            ("gap", "time,y,x", "0,2", "300,302,304,306,310,312,314,316", "1,2,3,4,3,5,7,9", 8),
            ("reversed", "time,y,x", "1,0", "300,302,304,306,310,312,314,316", "1,2,3,4,3,5,7,9", 8),
            ("missing-axis", "y,x", "0,1", "300,302,304,306", "1,2,3,4", 4)):
        nt = len(stamps.split(","))
        declarations = ["double time(time); time:units=\"hours since 2015-12-01 00:00:00\";"]
        assignments = [f"time={stamps};", f"T2D={temperature};", f"RAINRATE={rain};"]
        fields = ("T2D", "Q2D", "U2D", "V2D", "PSFC", "LWDOWN", "SWDOWN", "RAINRATE")
        for field in fields:
            declarations.append(f"double {field}({axes});")
        for field, value in (("Q2D", ".005"), ("U2D", "-1"), ("V2D", "2"),
                             ("PSFC", "100000"), ("LWDOWN", "350"), ("SWDOWN", "200")):
            assignments.append(field + "=" + ",".join([value] * count) + ";")
        cdl = root / (name + ".cdl")
        cdl.write_text(f"netcdf sample {{\ndimensions: time={nt}; y=2; x=2;\nvariables:\n"
                       + "\n".join(declarations) + "\ndata:\n" + "\n".join(assignments) + "\n}\n")
        argv = ["ncgen", "-4", "-o", str(root / (name + ".nc")), str(cdl)]
        start = time.perf_counter()
        subprocess.run(argv, check=True, capture_output=True)
        generated.append(dict(argv=argv, seconds=time.perf_counter() - start))
    base = dict(source_profile=str(table), profile_row="decoded-canonical-grid",
                time_origin="2015-12-01T00:00:00Z", interval_seconds=3600,
                catchment_ids=["catchment-1"], area_weights=str(weights))
    cumulative = dict(source_profile=str(custom), profile_row="arbitrary-cumulative-series-731",
                      rain_baseline=str(baseline))
    results = []
    for label, names, changes, expected_error in (
            ("reference", ["single-0", "single-1"], {}, None),
            ("first", ["first"], {}, None),
            ("middle", ["middle"], {}, None),
            ("last", ["last"], {}, None),
            ("repeated", ["repeated"], {}, "timestamp differs"),
            ("gap", ["gap"], {}, "timestamp differs"),
            ("reversed", ["reversed"], {}, "timestamp differs"),
            ("missing-axis", ["missing-axis"], {}, "lacks time dimension"),
            ("cumulative-reference", ["single-0", "single-1"], cumulative, None),
            ("cumulative-first", ["first"], cumulative, None),
            ("cumulative-last", ["last"], cumulative, None),
            ("overlapping-files", ["first", "single-1"], {}, "timestamp differs")):
        output = root / (label + "-packet")
        request = dict(base, inputs=[str(root / (n + ".nc")) for n in names], output=str(output))
        request.update(changes)
        path = root / (label + "-request.json")
        path.write_text(json.dumps(request, indent=2) + "\n")
        argv = [str(args.executable.resolve()), str(path)]
        start = time.perf_counter()
        run = subprocess.run(argv, capture_output=True, text=True)
        result = dict(case=label, argv=argv, seconds=time.perf_counter() - start,
                      returncode=run.returncode, expected_error=expected_error, stderr=run.stderr)
        if run.returncode == 0:
            manifest = json.loads(run.stdout)
            result.update(shape=manifest["shape"], data_sha256=manifest["data_sha256"],
                          parallelism=manifest["parallelism"],
                          preparation_seconds=manifest["preparation_seconds"])
        result["pass"] = (run.returncode == 0 and result.get("shape") == [2, 8, 1]
                          if expected_error is None else
                          run.returncode != 0 and expected_error in run.stderr
                          and not (output / "manifest.json").exists())
        results.append(result)
    lookup = {row["case"]: row for row in results}
    exact = all(lookup[name].get("data_sha256") == lookup[reference].get("data_sha256")
                and lookup[name].get("data_sha256") is not None
                for reference, names in (("reference", ("first", "middle", "last")),
                                         ("cumulative-reference", ("cumulative-first", "cumulative-last")))
                for name in names)
    receipt = dict(scope="Synthetic multi-time NetCDF input/regrid transport; not native or GPU proof",
                   executable_sha256=hashlib.sha256(args.executable.read_bytes()).hexdigest(),
                   fixture_generation=generated, results=results,
                   axis_and_cumulative_packet_seals_exact=exact,
                   passed=all(row["pass"] for row in results) and exact)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
