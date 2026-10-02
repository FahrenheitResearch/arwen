#!/usr/bin/env python3
"""Compile native W lid regression outputs from two isolated edge states."""
from pathlib import Path
import argparse
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gpuwm.verify.advect_oracle import (
    ADVECT_ORACLE_DIR, load_advect_cases, w_lid_control_cases,
    write_fortran_input, read_fortran_output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ADVECT_ORACLE_DIR)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    args = parser.parse_args()
    args.scratch.mkdir(parents=True, exist_ok=True)
    base = next(case for case in load_advect_cases(args.directory) if case.name == "real_interior")
    arrays = {}
    for case in w_lid_control_cases(base):
        input_path = args.scratch / f"{case.name}.in.bin"
        output_path = args.scratch / f"{case.name}.out.bin"
        write_fortran_input(case, "advect_w", input_path)
        subprocess.run([str(args.executable.resolve()), str(input_path), str(output_path)], check=True)
        arrays[case.name] = read_fortran_output(case, output_path)["tendency"]
    output = args.directory / "w-lid-controls-wrf.npz"
    np.savez_compressed(output, **arrays)
    print(f"w lid controls: {output.name}")


if __name__ == "__main__":
    main()
