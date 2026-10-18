"""Run unchanged WRF phy_prep expressions on the dumped WOOF state.

Invoke on a sprint box with WRF_SOURCE_ROOT set. Only dump artifacts are
read. This diagnoses the operation producing each input, independently
of the WRF wrapper and the full forecast.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from tools.wrf_exact_localize.compare_radiation_inputs import words


def reproduce(dump, work):
    work.mkdir(parents=True, exist_ok=False)
    state = dict(np.load(dump / "atmosphere.npz"))
    p = np.ascontiguousarray(state["state_p"], np.float32).ravel()
    theta = np.ascontiguousarray(state["theta"], np.float32).ravel()
    (work / "input.bin").write_bytes(np.array([p.size], "<i4").tobytes() + p.tobytes() + theta.tobytes())
    source = Path(os.environ["WRF_SOURCE_ROOT"]) / "share" / "module_model_constants.F"
    program = Path(__file__).with_name("phy_thermodynamics.F90")
    command = ["gfortran", "-O0", "-cpp", "-ffp-contract=off", "-ffree-form",
               "-ffree-line-length-none", str(source), str(program), "-o", "oracle"]
    subprocess.run(command, cwd=work, check=True)
    subprocess.run([str(work.resolve() / "oracle"), "input.bin", "output.bin"], cwd=work, check=True)
    output = np.fromfile(work / "output.bin", "<f4").reshape(2, -1)
    return {"command": command,
            "exner": words(output[0], state["exner"].ravel()),
            "temperature": words(output[1], state["temperature"].ravel())}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dump", type=Path)
    p.add_argument("work", type=Path)
    p.add_argument("receipt", type=Path)
    a = p.parse_args()
    a.receipt.write_text(json.dumps(reproduce(a.dump, a.work), indent=2) + "\n")
