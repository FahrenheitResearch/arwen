"""Pack native streams and retain the unmodified-routine provenance."""
from pathlib import Path
import argparse
import json

from tools.sfire_wrf471_oracle.pack_fixtures import pack


def package(build, destination):
    build, destination = Path(build), Path(destination)
    pack(build, destination)
    path = destination / "receipt.json"
    receipt = json.loads(path.read_text())
    receipt.update(reference_kind="byte-extracted unmodified WRF v4.7.1 rhs_ph",
                   extraction=json.loads((build / "rhs-extraction.json").read_text()),
                   unchanged_prep_extraction=json.loads((build / "prep-extraction.json").read_text()),
                   compiled_artifacts=(build / "artifacts.sha256").read_text(),
                   convention=("ph_old aliases ph, as at the atmospheric driver call; FNM and FNP "
                               "at kde are zero, as every WRF initializer leaves them; control 1 "
                               "holds the unassigned top U level at its zero allocation value and "
                               "control 2 sets it to a -17 sentinel"))
    path.write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("build")
    parser.add_argument("destination")
    args = parser.parse_args()
    package(args.build, args.destination)
