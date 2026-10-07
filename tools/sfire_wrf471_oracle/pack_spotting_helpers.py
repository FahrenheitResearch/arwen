"""Pack original helpers and distinguish the corrected spherical mass case."""
from pathlib import Path
import argparse
import json

from tools.sfire_wrf471_oracle.pack_fixtures import pack


def run(build,destination):
    build,destination=Path(build),Path(destination)
    pack(build,destination)
    receipt=json.loads((destination/"receipt.json").read_text())
    receipt["reference_kind"]="byte-unmodified original WRF helpers plus labeled cubic-volume property control"
    receipt["extraction"]=json.loads((build/"extraction.json").read_text())
    receipt["cases"]["spotting/property_corrected"]["reference_kind"]="spherical initial mass corrected from D^2 to D^3"
    (destination/"receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("build")
    p.add_argument("destination")
    a=p.parse_args()
    run(a.build,a.destination)
