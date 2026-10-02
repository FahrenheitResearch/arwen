"""Compile byte-unmodified WRF coupling routines and record the extraction."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

ROUTINES = ("calc_mu_uv", "calc_ww_cp", "calc_cq", "calc_php", "w_damp", "rk_rayleigh_damp")
PIN = "f52c197ed39d12e087d02c50f412d90d418f6186"
SOURCE_SHA256 = "bd177b6b5ba7949cf9e694d7ad654fd9ae2f07d39d85802f0716c5318889a815"
CONSTANTS_SHA256 = "5b80377fecdc18a5f0ad38d3b6c15cfc86ad5d76701adbbbb08a08698d0f7062"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("wrf_source", type=Path)
    p.add_argument("wrf_build", type=Path)
    p.add_argument("build", type=Path)
    a = p.parse_args()
    a.build.mkdir(parents=True, exist_ok=True)
    original = a.wrf_source / "dyn_em/module_big_step_utilities_em.F"
    constants = a.wrf_source / "share/module_model_constants.F"
    if sha(original) != SOURCE_SHA256 or sha(constants) != CONSTANTS_SHA256:
        raise ValueError("WRF source or constants differ from the v4.7.1 pin")
    raw = original.read_bytes()
    source = raw.decode("ascii")
    blocks = {}
    for name in ROUTINES:
        match = re.search(r"(?im)^ *SUBROUTINE " + name + r"\s*\(.*?^ *END SUBROUTINE\s+" + name + r"\s*$", source, re.DOTALL)
        if not match:
            raise ValueError(name)
        blocks[name] = match.group(0)
    # The real generated configuration type and Registry scalar index are
    # imported from WRF's compiled modules. Numerical constants are compiled
    # from the full pinned source below. Only logging is replaced.
    for mod in a.wrf_build.rglob("*.mod"):
        shutil.copyfile(mod, a.build / mod.name)
    output = a.build / "coupling_wrf.F90"
    output.write_text("module coupling_wrf\nuse module_model_constants\nuse module_configure, only: grid_config_rec_type\nuse module_state_description, only: PARAM_FIRST_SCALAR\nuse module_wrf_error\ncontains\n" + "\n".join(blocks.values()) + "\nend module coupling_wrf\n", encoding="ascii")
    script = Path(__file__).resolve().parent
    flags = ["-O0", "-cpp", "-Dwrfmodel", "-DEM_CORE=1", "-DNMM_CORE=0", "-DRWORDSIZE=4", "-DIWORDSIZE=4", "-DDWORDSIZE=8", "-DLWORDSIZE=4", "-ffree-form", "-ffree-line-length-none", "-ffp-contract=off", "-fcheck=bounds"]
    commands = []
    for sourcefile in (script / "coupling_services.F90", a.wrf_source / "share/module_model_constants.F", output, script / "coupling_run.F90"):
        command = ["gfortran", *flags, "-c", str(sourcefile.resolve())]
        subprocess.run(command, cwd=a.build, check=True)
        commands.append(command)
    command = ["gfortran", "-fcheck=bounds", "-o", "coupling_run", "coupling_services.o", "module_model_constants.o", "coupling_wrf.o", "coupling_run.o"]
    subprocess.run(command, cwd=a.build, check=True)
    commands.append(command)
    symbols = subprocess.check_output(["nm", "-u", "coupling_wrf.o"], cwd=a.build, text=True)
    if "_ZGV" in symbols:
        raise RuntimeError("vector libm must not select the reference arithmetic")
    (a.build / "coupling-build-receipt.json").write_text(json.dumps({
        "wrf_tag": "v4.7.1", "wrf_commit": PIN,
        "source_sha256": sha(original), "constants_sha256": sha(a.wrf_source / "share/module_model_constants.F"),
        "routine_sha256": {k: hashlib.sha256(v.encode("ascii")).hexdigest() for k, v in blocks.items()},
        "compiler": subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[0],
        "flags": flags, "commands": commands, "undefined_symbols": symbols,
        "executable_sha256": sha(a.build / "coupling_run"),
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
