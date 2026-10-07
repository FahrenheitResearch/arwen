"""Preserve native streams and bind their labeled source corrections."""
from pathlib import Path
import argparse
import json
from tools.sfire_wrf471_oracle.pack_fixtures import pack


def pack_geometry(build, destination):
    build, destination = Path(build), Path(destination)
    pack(build, destination)
    path = destination / "receipt.json"
    receipt = json.loads(path.read_text())
    receipt["reference_kind"] = "original WRF initializer blocks with separately recorded original RUC depths and terrain-edge negative controls"
    receipt["corrections"] = [
        "RUC second soil thickness excludes the first midpoint interval",
        "terrain interpolation covers fine edge strips using original coarse boundary continuation"]
    receipt["extraction"] = json.loads((build / "extraction.json").read_text())
    receipt["original_extraction"] = json.loads((build / "extraction-original.json").read_text())
    path.write_text(json.dumps(receipt, indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("build")
    parser.add_argument("destination")
    args = parser.parse_args()
    pack_geometry(args.build, args.destination)
