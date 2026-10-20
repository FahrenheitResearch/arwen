#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Exercise the shipped AORC interval-end row through the real Rust loader.

ncgen supplies literal analytic grids; Python orchestrates metadata/processes
and compares opaque packet hashes. No native/event/physics evidence is implied.
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
    weights = root / "weights.json"
    weights.write_text(json.dumps(dict(catchment_ids=["catchment-1"], grid_shape=[1, 2],
                                      offsets=[0, 2], indices=[0, 1], weights=[0.25, 0.75])) + "\n")
    # Dyadic amounts/rates give an independently literal exact oracle, including
    # division by 3600 and CSR averaging. Deliberately distinct consecutive hours.
    cdl = '''netcdf sample {
dimensions: time=2; latitude=1; longitude=2;
variables:
 double time(time); time:units="hours since 2023-01-05 00:00:00";
 double T2D(time,latitude,longitude); double Q2D(time,latitude,longitude);
 double U2D(time,latitude,longitude); double V2D(time,latitude,longitude);
 double PSFC(time,latitude,longitude); double LWDOWN(time,latitude,longitude);
 double SWDOWN(time,latitude,longitude); double RAINRATE(time,latitude,longitude);
data:
 time=STAMPS; T2D=300,302,310,312; Q2D=.005,.007,.009,.011;
 U2D=-1,3,-2,4; V2D=2,4,3,5; PSFC=101000,100000,100000,99000;
 LWDOWN=350,360,370,380; SWDOWN=200,250,300,350; RAINRATE=RAIN;
}
'''
    names = dict(T2D="TMP_2maboveground", Q2D="SPFH_2maboveground",
                 U2D="UGRD_10maboveground", V2D="VGRD_10maboveground",
                 PSFC="PRES_surface", LWDOWN="DLWRF_surface",
                 SWDOWN="DSWRF_surface", RAINRATE="APCP_surface")
    generated = []
    for name, stamps, rain, aorc in (
            ("reference", "0,1", ".0009765625,.001953125,0,.00390625", False),
            ("hour-ending", "1,2", "3.515625,7.03125,0,14.0625", True),
            ("wrong-hour-starting", "0,1", "3.515625,7.03125,0,14.0625", True),
            ("missing-first-hour", "2,3", "3.515625,7.03125,0,14.0625", True)):
        text = cdl.replace("STAMPS", stamps).replace("RAIN;", rain + ";")
        if aorc:
            for canonical, native in names.items():
                text = text.replace(canonical, native)
        path = root / (name + ".cdl")
        path.write_text(text)
        argv = ["ncgen", "-4", "-o", str(root / (name + ".nc")), str(path)]
        start = time.perf_counter()
        subprocess.run(argv, check=True, capture_output=True)
        generated.append(dict(argv=argv, seconds=time.perf_counter() - start))
    results = []
    for name, row, accepted in (("reference", "decoded-canonical-grid", True),
                                ("hour-ending", "aorc-grid", True),
                                ("wrong-hour-starting", "aorc-grid", False),
                                ("missing-first-hour", "aorc-grid", False)):
        output = root / (name + "-packet")
        request = dict(source_profile=str(table), profile_row=row,
                       source_id="aorc" if row == "aorc-grid" else None,
                       inputs=[str(root / (name + ".nc"))], output=str(output),
                       time_origin="2023-01-05T00:00:00Z", interval_seconds=3600,
                       catchment_ids=["catchment-1"], area_weights=str(weights))
        path = root / (name + "-request.json")
        path.write_text(json.dumps(request, indent=2) + "\n")
        argv = [str(args.executable.resolve()), str(path)]
        start = time.perf_counter()
        run = subprocess.run(argv, capture_output=True, text=True)
        result = dict(case=name, argv=argv, seconds=time.perf_counter() - start,
                      returncode=run.returncode, expected_acceptance=accepted, stderr=run.stderr)
        if run.returncode == 0:
            manifest = json.loads(run.stdout)
            result.update(data_sha256=manifest["data_sha256"], shape=manifest["shape"],
                          parallelism=manifest["parallelism"],
                          preparation_seconds=manifest["preparation_seconds"])
        result["pass"] = (run.returncode == 0 and result.get("shape") == [2, 8, 1] if accepted else
                          run.returncode != 0 and "timestamp differs" in run.stderr
                          and not (output / "manifest.json").exists())
        results.append(result)
    exact = (results[0].get("data_sha256") is not None and
             results[0].get("data_sha256") == results[1].get("data_sha256"))
    receipt = dict(scope="Synthetic shipped-row AORC clock/amount regression; not native/event/GPU proof",
                   source_table_sha256=hashlib.sha256(table.read_bytes()).hexdigest(),
                   executable_sha256=hashlib.sha256(args.executable.read_bytes()).hexdigest(),
                   fixture_generation=generated, results=results,
                   interval_end_amount_packets_exact=exact,
                   passed=all(result["pass"] for result in results) and exact)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
