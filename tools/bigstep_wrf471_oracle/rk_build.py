"""Build unchanged WRF RK routines behind a small C ABI harness."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

PIN = {
    "module_em.F": "11105cbf8255f30ca6a44cd7429a92cedce1fb91db6ce90fd7217002a72fb7fa",
    "module_model_constants.F": "5b80377fecdc18a5f0ad38d3b6c15cfc86ad5d76701adbbbb08a08698d0f7062",
}
COMMIT = "f52c197ed39d12e087d02c50f412d90d418f6186"


def build(source: Path, wrf_build: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    receipt = {"wrf_tag": "v4.7.1", "wrf_commit": COMMIT, "sources": {},
               "commands": [], "extracts": {}}
    for name, pin in PIN.items():
        data = (source / name).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != pin:
            raise ValueError(f"{name}: source hash differs from WRF v4.7.1 pin")
        receipt["sources"][name] = digest
    original = (source / "module_em.F").read_text()
    extracts = []
    for name in ("rk_addtend_dry", "rk_update_scalar"):
        match = re.search(rf"^SUBROUTINE {name}\b.*?^END SUBROUTINE {name}\s*$",
                          original, re.M | re.S)
        if match is None:
            raise ValueError(f"routine {name} not found")
        routine = match.group(0)
        extracts.append(routine)
        receipt["extracts"][name] = {
            "sha256": hashlib.sha256(routine.encode()).hexdigest(),
            "first_line": original[:match.start()].count("\n") + 1,
            "last_line": original[:match.end()].count("\n") + 1,
        }
    translation_unit = ("module rk_reference\n"
                        "use module_configure, only: grid_config_rec_type\n"
                        "use module_model_constants\ncontains\n"
                        + "\n".join(extracts) + "\nend module\n")
    (destination / "rk_reference.F90").write_text(translation_unit)
    # This is the real generated WRF configuration type, not a substitute.
    mod = wrf_build / "frame" / "module_configure.mod"
    shutil.copy2(mod, destination / mod.name)
    receipt["configure_module_sha256"] = hashlib.sha256(mod.read_bytes()).hexdigest()
    flags = ["-O0", "-fPIC", "-ffree-form", "-ffree-line-length-none",
             "-ffp-contract=off", "-fno-tree-vectorize", "-fcheck=bounds"]
    commands = [
        ["gfortran", "-c", *flags, "-cpp", str(source / "module_model_constants.F")],
        ["gfortran", "-c", *flags, "rk_reference.F90"],
        ["gfortran", "-c", *flags, str(Path(__file__).with_name("rk_harness.F90"))],
        ["gfortran", "-shared", "-o", "rk_oracle.so", "rk_harness.o",
         "rk_reference.o", "module_model_constants.o"],
    ]
    for cmd in commands:
        receipt["commands"].append([
            part.replace(str(source),"$WRF_SOURCE").replace(str(Path(__file__).parent),"$HARNESS_DIR")
            for part in cmd])
        subprocess.run(cmd, cwd=destination, check=True)
    receipt["compiler"] = subprocess.check_output(["gfortran", "--version"], text=True)
    receipt["undefined_symbols"] = subprocess.check_output(
        ["nm", "-u", str(destination / "rk_reference.o")], text=True)
    (destination / "rk-build-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return destination / "rk_oracle.so"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("wrf_build", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    build(args.source, args.wrf_build, args.destination)
