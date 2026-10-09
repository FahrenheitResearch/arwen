"""Compile WRF v4.6.1 sixth_order_diffusion, byte-unmodified, two ways.

The routine slice is copied as bytes from the stock v4.6.1 source and its
hash is pinned.  It is the same slice hash as the v4.7.1 routine, so 4.6.1
and 4.7.1 share this routine word for word.  Only a configuration service
type is supplied: it declares the eight members the routine reads.

Two executables are built from the same slice and the same driver:

* ``strict``: -O0 -ffp-contract=off -fno-tree-vectorize -fcheck=bounds.
  Every REAL operation is a separate IEEE binary32 operation, and any
  out-of-bounds read aborts.
* ``stock``: the Fortran flags of WRF's own GNU configure entry
  (arch/configure.defaults, the flags W1's stock wrf.exe was built with),
  so the reference is the arithmetic stock WRF actually executes.

The comparison tool requires the two to agree word for word before any
GPU word is graded.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import subprocess

PIN = "d66e442fccc04111067e29274c9f9eaccc3cef28"   # tag v4.6.1
SOURCE_HASH = "8f0649b458fceabd5c31c87ec9f266840fb164e2629322904a3f6ee04c72331a"
CONSTANTS_HASH = "5b80377fecdc18a5f0ad38d3b6c15cfc86ad5d76701adbbbb08a08698d0f7062"
# Same routine bytes as v4.7.1 (tests/test_diff6_wrf471_parity.py pins this).
SLICE_HASH = "a4534919fdb15c91789f6d0dc504fe12f8eac6ee711f51053b3900999a055567"

STRICT = ["-O0", "-ffp-contract=off", "-fno-tree-vectorize", "-fcheck=bounds",
          "-ffree-form", "-ffree-line-length-none"]
STOCK = ["-O2", "-ftree-vectorize", "-funroll-loops", "-w", "-ffree-form",
         "-ffree-line-length-none", "-fconvert=big-endian", "-frecord-marker=4",
         "-fallow-argument-mismatch", "-fallow-invalid-boz"]

CONFIG = b"""module module_configure
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


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(source, output):
    output.mkdir(parents=True, exist_ok=True)
    routine = source / "dyn_em/module_big_step_utilities_em.F"
    constants = source / "share/module_model_constants.F"
    assert digest(routine) == SOURCE_HASH, "WRF 4.6.1 numerical source changed"
    assert digest(constants) == CONSTANTS_HASH, "WRF constants source changed"
    raw = routine.read_bytes()
    match = re.search(rb"(?im)^ *SUBROUTINE sixth_order_diffusion\(.*?^ *END SUBROUTINE sixth_order_diffusion[^\n]*\n", raw, re.S)
    assert match
    assert hashlib.sha256(match[0]).hexdigest() == SLICE_HASH, "routine slice changed"
    here = Path(__file__).resolve().parent
    commands = {}
    for flavour, flags in (("strict", STRICT), ("stock", STOCK)):
        work = output / flavour
        work.mkdir(exist_ok=True)
        (work / "sixth_order_slice.f90").write_bytes(CONFIG + match[0] + b"end module\n")
        c1 = ["gfortran", *flags, "-cpp", "-c", str(constants.resolve())]
        c2 = ["gfortran", *flags, "-I", str(work.resolve()), "sixth_order_slice.f90",
              str(here / "diff6_wrf461_driver.f90"), "module_model_constants.o", "-o", "diff6_driver"]
        subprocess.run(c1, cwd=work, check=True)
        subprocess.run(c2, cwd=work, check=True)
        dis = subprocess.run(["objdump", "-d", str(work / "diff6_driver")], capture_output=True, text=True).stdout
        commands[flavour] = {"compile": [c1[:-1] + ["share/module_model_constants.F"],
                                         c2[:1 + len(flags)] + ["-I", f"<build>/{flavour}", "sixth_order_slice.f90", "diff6_wrf461_driver.f90",
                                                                "module_model_constants.o", "-o", "diff6_driver"]],
                             "fma_instructions": len(re.findall(r"\bvf(?:n?m(?:add|sub))", dis)),
                             "executable_sha256": digest(work / "diff6_driver")}
    metadata = {
        "wrf_version": "4.6.1", "wrf_commit": PIN,
        "source_sha256": SOURCE_HASH, "constants_sha256": CONSTANTS_HASH,
        "routine_slice_sha256": SLICE_HASH,
        "routine_slice_equals_wrf471": True,
        "routine_source_lines": [raw[:match.start()].count(b"\n") + 1, raw[:match.end()].count(b"\n")],
        "compiler": subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[0],
        "builds": commands, "real_kind_bytes": 4, "slice_is_byte_unmodified": True,
        "driver_sha256": digest(here / "diff6_wrf461_driver.f90"),
    }
    (output / "diff6-wrf461-build.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8", newline="\n")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="WRF v4.6.1 source tree")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.output), indent=2))
