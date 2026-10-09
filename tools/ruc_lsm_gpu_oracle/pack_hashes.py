"""Keep full-field WRF regression hashes without retaining model arrays."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import layout
from columns import PROFILES


def digest(value):
    return hashlib.sha256(np.asarray(value, dtype="<f4").tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("case_dir", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    meta = json.loads((args.case_dir / "case.json").read_text())
    wrf = layout.read_outputs(args.case_dir / "outputs-defined.bin", meta["ncol"], meta["nzs"], meta["nsteps"])
    hashes = {f"init__{n}": digest(wrf["init"][n]) for n in layout.INIT_2D + layout.INIT_PROFILES}
    for k, step in enumerate(wrf["steps"], 1):
        for name in layout.OUT_2D + PROFILES:
            hashes[f"step{k}__{name}"] = digest(step[layout.SFCEVP_ONCE_ACC if name == "sfcevp" else name])
    receipt = {"options": {n: meta[n] for n in ("ncol", "nzs", "nsteps", "dt", "fractional_seaice", "mosaic", "lakemodel", "rdlai2d")},
               "round_xice": False, "sha256": hashes,
               "reference": "WRF v4.6.1, GNU O2 with defined local values; single-count SFCEVP",
               "inputs_sha256": hashlib.sha256((args.case_dir / "inputs.bin").read_bytes()).hexdigest()}
    args.out.write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
