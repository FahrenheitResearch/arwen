"""Compile the unmodified release scheme and WRF-facing driver on S3."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

source, build = map(Path, sys.argv[1:3])
build.mkdir(parents=True, exist_ok=True)
here = Path(__file__).resolve().parent
flags = ["-cpp", "-ffree-form", "-ffree-line-length-none", "-O0",
         "-ffp-contract=off", "-fno-fast-math", "-fcheck=all", "-fbacktrace", "-g"]
if "--undefined-check" in sys.argv[3:]:
    flags += ["-finit-real=snan", "-ffpe-trap=invalid"]
files = {
    "module_model_constants.F": "share/module_model_constants.F",
    "module_sf_gecros.F": "phys/module_sf_gecros.F",
    "module_sf_noahmplsm.F": "phys/noahmp/src/module_sf_noahmplsm.F",
    "module_sf_noahmp_glacier.F": "phys/noahmp/src/module_sf_noahmp_glacier.F",
    "module_sf_noahmp_groundwater.F": "phys/noahmp/src/module_sf_noahmp_groundwater.F",
    "module_sf_noahmpdrv.F": "phys/noahmp/drivers/wrf/module_sf_noahmpdrv.F",
}
pins = {}
stub = here.parent / "noahmp_wrf461_oracle/stub_wrf.F90"
driver = here.parent / "noahmp_wrf461_oracle/column/run_column.F90"
subprocess.run(["gfortran", *flags, "-c", str(stub)], cwd=build, check=True)
for name, path in files.items():
    shutil.copyfile(source / path, build / name)
    pins[path] = hashlib.sha256((build / name).read_bytes()).hexdigest()
    extra = ["-DEM_CORE=0", "-fallow-argument-mismatch"] if "noahmpdrv" in name else []
    subprocess.run(["gfortran", *flags, *extra, "-c", name], cwd=build, check=True)
subprocess.run(["gfortran", *flags, "-c", str(driver)], cwd=build, check=True)
subprocess.run(["gfortran", *flags, "-o", "run_column", "run_column.o", "stub_wrf.o",
                *[Path(name).with_suffix(".o").name for name in files]], cwd=build, check=True)
for path in ("phys/noahmp/parameters/MPTABLE.TBL", "run/SOILPARM.TBL", "run/GENPARM.TBL"):
    shutil.copyfile(source / path, build / Path(path).name)
    pins[path] = hashlib.sha256((source / path).read_bytes()).hexdigest()
receipt = {"source_and_table_sha256": pins, "flags": flags,
           "driver_flags": ["-DEM_CORE=0", "-fallow-argument-mismatch"],
           "compiler": subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[0],
           "scheme_and_wrapper_unmodified": True, "real_kind": "binary32",
           "stub_sha256": hashlib.sha256(stub.read_bytes()).hexdigest(),
           "driver_sha256": hashlib.sha256(driver.read_bytes()).hexdigest()}
(build / "build.json").write_text(json.dumps(receipt, indent=2))
print(json.dumps(receipt, indent=2))
