"""Write momentum fixtures by executing the compiled, unchanged WRF routines."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gpuwm.verify.bigstep_momentum_oracle import (MOMENTUM_CASES, make_momentum_inputs,
                                                 pack_wrf_inputs, unpack_wrf_outputs)


def main(state: Path, build: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with np.load(state, allow_pickle=False) as archive:
        real = {key: archive[key].copy() for key in archive.files}
    receipt = json.loads((build / "momentum-build-receipt.json").read_text())
    receipt["state_sha256"] = hashlib.sha256(state.read_bytes()).hexdigest()
    receipt["halo_validation"] = "every output word outside each native output array is unchanged"
    receipt["cases"] = {}
    for case in MOMENTUM_CASES:
        inputs, metadata = make_momentum_inputs(real, case)
        stored = {"in_" + key: value for key, value in inputs.items()}
        stored["metadata"] = np.asarray(json.dumps(metadata))
        for routine, mode in (("horizontal_pressure_gradient", 1), ("coriolis", 2), ("curvature", 3), ("combined", 4), ("original_pressure", 5)):
            raw_in, layout = pack_wrf_inputs(inputs, metadata, mode)
            input_file = build / f"{case}-{routine}-input.bin"
            output_file = build / f"{case}-{routine}-output.bin"
            input_file.write_bytes(raw_in)
            subprocess.run([str((build / "momentum_oracle").resolve()), str(input_file.resolve()), str(output_file.resolve())], check=True)
            reference, halo = unpack_wrf_outputs(output_file.read_bytes(), layout)
            assert halo, f"WRF modified an output halo outside native grids: {case} {routine}"
            for field, value in reference.items():
                stored[f"{routine}_{field}"] = value
        name = f"momentum-{case}.npz"
        np.savez_compressed(destination / name, **stored)
        receipt["cases"][case] = {"fixture": name, "sha256": hashlib.sha256((destination / name).read_bytes()).hexdigest(), **metadata}
    (destination / "momentum-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="ascii")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("state", type=Path)
    parser.add_argument("build", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    main(args.state, args.build, args.destination)
