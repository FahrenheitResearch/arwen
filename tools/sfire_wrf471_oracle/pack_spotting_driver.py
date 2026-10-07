"""Pack the full driver corpus with its explicit corrected-source receipt."""
from pathlib import Path
import argparse
import hashlib
import json

from tools.sfire_wrf471_oracle.pack_fixtures import pack


def run(build, destination):
    build,destination=Path(build),Path(destination)
    pack(build/"corrected",destination)
    receipt=json.loads((destination/"receipt.json").read_text())
    receipt["reference_kind"]="explicitly corrected original WRF firebrand driver"
    receipt["corrections"]=json.loads((build/"corrections.json").read_text())
    receipt["column_extraction"]=json.loads((build/"columns-extraction.json").read_text())
    receipt["original_checked_run_exit"]=int((build/"original/run.status").read_text())
    receipt["original_checked_failure"]="th_phy dimension 2 mismatch: full W-level destination has 8 levels, mass-level source has 7"
    receipt["original_binary_sha256"]=hashlib.sha256((build/"original/run").read_bytes()).hexdigest()
    receipt["corrected_binary_sha256"]=hashlib.sha256((build/"corrected/run").read_bytes()).hexdigest()
    (destination/"receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("build")
    p.add_argument("destination")
    a=p.parse_args()
    run(a.build,a.destination)
