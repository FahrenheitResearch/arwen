"""Compile exact public scheme modules, with visibility-only declarations.

Run on S3: python tools/noahmp_snow_age_oracle/build.py SOURCE BUILD
SOURCE is the extracted src directory at a pinned Noah-MP commit.
"""
from pathlib import Path
import hashlib
import json
import re
import subprocess
import sys

source, build = map(Path, sys.argv[1:])
build.mkdir(parents=True, exist_ok=True)
here = Path(__file__).resolve().parent
flags = ["-ffree-form", "-ffree-line-length-none", "-cpp", "-O0",
         "-ffp-contract=off", "-fno-fast-math", "-fno-range-check",
         "-fcheck=all", "-fbacktrace", "-g"]
pins = {}
for name in ("module_sf_gecros.F", "module_sf_noahmplsm.F", "module_sf_noahmp_glacier.F"):
    raw = (source / name).read_bytes()
    pins[name] = hashlib.sha256(raw).hexdigest()
    text = raw.decode()
    # All numerical statements remain exactly the public release's.
    public = re.sub(r"(?im)^(\s*)private(\s*::)", r"\1public\2", text)
    (build / name).write_text(public)
    subprocess.run(["gfortran", *flags, "-c", name], cwd=build, check=True)
stub = here.parent / "noahmp_wrf461_oracle/stub_wrf.F90"
subprocess.run(["gfortran", *flags, "-c", str(stub)], cwd=build, check=True)
subprocess.run(["gfortran", *flags, str(here / "column.F90"),
                "module_sf_noahmplsm.o", "module_sf_gecros.o",
                "module_sf_noahmp_glacier.o", "stub_wrf.o",
                "-o", "column"], cwd=build, check=True)
receipt = {"sources_sha256": pins, "compiler": subprocess.check_output(
    ["gfortran", "--version"], text=True).splitlines()[0], "flags": flags,
    "scalar_kind": "default REAL, storage_size=32 checked in driver",
    "dependencies": ["module_sf_gecros", "noahmp_globals", "noahmp_glacier_globals",
                     "WRF fatal/message stubs only"],
    "visibility_only": True,
    "driver_sha256": hashlib.sha256((here / "column.F90").read_bytes()).hexdigest()}
(build / "build.json").write_text(json.dumps(receipt, indent=2))
print(json.dumps(receipt, indent=2))
