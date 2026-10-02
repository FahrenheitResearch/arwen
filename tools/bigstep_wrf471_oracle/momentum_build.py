"""Build byte-extracted WRF 4.7.1 momentum routines and record their hashes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

SOURCE_SHA256 = "bd177b6b5ba7949cf9e694d7ad654fd9ae2f07d39d85802f0716c5318889a815"
CONSTANTS_SHA256 = "5b80377fecdc18a5f0ad38d3b6c15cfc86ad5d76701adbbbb08a08698d0f7062"
ROUTINES = ("horizontal_pressure_gradient", "coriolis", "curvature")


def build(source: Path, constants: Path, output: Path, config_module: Path | None = None) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    raw = source.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SOURCE_SHA256
    assert hashlib.sha256(constants.read_bytes()).hexdigest() == CONSTANTS_SHA256
    extracted = []
    receipt = {"wrf_tag": "v4.7.1", "wrf_commit": "f52c197ed39d12e087d02c50f412d90d418f6186",
               "source_sha256": SOURCE_SHA256, "constants_sha256": CONSTANTS_SHA256,
               "extracted": {}, "compiler": subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[0]}
    for name in ROUTINES:
        expression = rb"(?im)^SUBROUTINE " + name.encode() + rb"\b[\s\S]*?^END SUBROUTINE " + name.encode() + rb"[^\r\n]*(?:\r?\n|$)"
        match = re.search(expression, raw)
        if match is None:
            raise ValueError(f"missing exact routine {name}")
        block = match.group(0)
        (output / f"{name}.original.F").write_bytes(block)
        extracted.append(block)
        receipt["extracted"][name] = {"sha256": hashlib.sha256(block).hexdigest(), "bytes": len(block),
                                       "source_line": raw[:match.start()].count(b"\n") + 1}
    referenced_config = set(re.findall(rb"(?i)config_flags%([a-z_][a-z_0-9]*)", b"\n".join(extracted)))
    config_fields = {b"specified", b"nested", b"polar", b"open_xs", b"open_xe", b"open_ys", b"open_ye", b"periodic_x", b"map_proj"}
    assert referenced_config == config_fields, referenced_config
    receipt["config_shim_fields"] = sorted(name.decode("ascii") for name in referenced_config)
    # Only the configuration record is reduced. Every field used by the
    # unchanged extracted routines is present; no numerical service is stubbed.
    preamble = b"""module module_big_step_utilities_em
use module_model_constants
implicit none
type grid_config_rec_type
logical :: specified=.false., nested=.false., polar=.false.
logical :: open_xs=.false., open_xe=.false., open_ys=.false., open_ye=.false.
logical :: periodic_x=.false.
integer :: map_proj=1
end type
contains
"""
    if config_module is not None:
        shutil.copyfile(config_module, output / "module_configure.mod")
        receipt["config_module_sha256"] = hashlib.sha256(config_module.read_bytes()).hexdigest()
        receipt["config_record"] = "actual generated WRF module_configure grid_config_rec_type"
        preamble = b"module module_big_step_utilities_em\nuse module_model_constants\nuse module_configure, only: grid_config_rec_type\nimplicit none\ncontains\n"
    else:
        receipt["config_record"] = "reduced semantic record; all referenced fields validated"
    (output / "momentum_reference.F90").write_bytes(preamble + b"\n".join(extracted) + b"end module\n")
    flags = ["-O0", "-ffp-contract=off", "-fno-tree-vectorize", "-ffree-form", "-ffree-line-length-none", "-fcheck=all"]
    defines = ["-Dwrfmodel", "-DEM_CORE=1", "-DNMM_CORE=0", "-DRWORDSIZE=4", "-DIWORDSIZE=4", "-DDWORDSIZE=8", "-DLWORDSIZE=4"]
    receipt["flags"] = flags
    receipt["defines"] = defines
    subprocess.run(["gfortran", *flags, "-cpp", *defines, "-c", str(constants.resolve()), "-o", "module_model_constants.o"], cwd=output, check=True)
    subprocess.run(["gfortran", *flags, "-c", "momentum_reference.F90"], cwd=output, check=True)
    driver = Path(__file__).with_name("momentum_driver.F90").resolve()
    subprocess.run(["gfortran", *flags, str(driver), "momentum_reference.o", "module_model_constants.o", "-o", "momentum_oracle"], cwd=output, check=True)
    symbols = subprocess.check_output(["nm", "momentum_reference.o"], cwd=output, text=True)
    for routine in ROUTINES:
        assert f"__module_big_step_utilities_em_MOD_{routine}" in symbols
    assert "_ZGV" not in symbols
    (output / "momentum-build-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="ascii")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("constants", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--config-module", type=Path)
    args = parser.parse_args()
    build(args.source, args.constants, args.output, args.config_module)
