"""Compile an exact source slice of WRF v4.7.1 sixth_order_diffusion.

The numerical routine is copied as bytes, with no edits. The configuration
service type supplies only the members this standalone routine reads. None of
its arithmetic, constants, dimensions, argument list, or loop bounds are stubbed.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import subprocess

PIN = "f52c197ed39d12e087d02c50f412d90d418f6186"
SOURCE_HASH = "bd177b6b5ba7949cf9e694d7ad654fd9ae2f07d39d85802f0716c5318889a815"
CONSTANTS_HASH = "5b80377fecdc18a5f0ad38d3b6c15cfc86ad5d76701adbbbb08a08698d0f7062"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(source, output):
    output.mkdir(parents=True, exist_ok=True)
    routine = source / "dyn_em/module_big_step_utilities_em.F"
    constants = source / "share/module_model_constants.F"
    assert digest(routine) == SOURCE_HASH, "WRF numerical source changed"
    assert digest(constants) == CONSTANTS_HASH, "WRF constants source changed"
    raw = routine.read_bytes()
    match = re.search(rb"(?im)^ *SUBROUTINE sixth_order_diffusion\(.*?^ *END SUBROUTINE sixth_order_diffusion[^\n]*\n", raw, re.S)
    assert match
    config = b"""module module_configure
  implicit none
  type grid_config_rec_type
    logical :: specified=.false., nested=.false.
    logical :: open_xs=.false., open_xe=.false., open_ys=.false., open_ye=.false.
    integer :: diff_6th_slopeopt=0
    real :: diff_6th_thresh=0.1
  end type
end module
module module_big_step_utilities_em
  use module_configure
  use module_model_constants
  implicit none
contains
"""
    (output / "sixth_order_slice.f90").write_bytes(config + match[0] + b"end module\n")
    here = Path(__file__).resolve().parent
    flags = ["-O0", "-ffp-contract=off", "-fno-tree-vectorize", "-fcheck=bounds", "-ffree-form", "-ffree-line-length-none"]
    command = ["gfortran", *flags, "-cpp", "-c", str(constants.resolve())]
    subprocess.run(command, cwd=output, check=True)
    command2 = ["gfortran", *flags, "-I", str(output.resolve()), "sixth_order_slice.f90", str(here / "diff6_driver.f90"), "module_model_constants.o", "-o", "diff6_driver"]
    subprocess.run(command2, cwd=output, check=True)
    metadata = {
        "wrf_version": "4.7.1", "wrf_commit": PIN,
        "source_sha256": SOURCE_HASH, "constants_sha256": CONSTANTS_HASH,
        "routine_slice_sha256": hashlib.sha256(match[0]).hexdigest(),
        "routine_source_lines": [raw[:match.start()].count(b"\n") + 1, raw[:match.end()].count(b"\n")],
        "compiler": subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[0],
        "commands": [["gfortran", *flags, "-cpp", "-c", "source/share/module_model_constants.F"],
                     ["gfortran", *flags, "-I", "build", "sixth_order_slice.f90", "diff6_driver.f90", "module_model_constants.o", "-o", "diff6_driver"]],
        "real_kind_bytes": 4,
        "slice_is_byte_unmodified": True,
    }
    (output / "diff6-build.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8", newline="\n")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.output), indent=2))
