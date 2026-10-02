"""Write all word-comparison measurements; acceptance pins are reviewed separately."""
from __future__ import annotations
import argparse
import json
from gpuwm.verify.bigstep_prep_oracle import PREP_CASES,load_prep_fixture,prep_port_outputs,measure_prep

ap=argparse.ArgumentParser()
ap.add_argument("output")
args=ap.parse_args()
measurements={case:measure_prep(load_prep_fixture(case),prep_port_outputs(load_prep_fixture(case))) for case in PREP_CASES}
with open(args.output,"w") as f:
    json.dump(measurements,f,indent=2)
    f.write("\n")
for case,fields in measurements.items():
    print(case, {key:value["max_ulp"] for key,value in fields.items()},flush=True)
