"""Build the unchanged rk_tendency orchestrator with native WRF dependencies."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

from rk_build import PIN, COMMIT


def build(source, wrf_build, destination):
    destination.mkdir(parents=True,exist_ok=True)
    original=(source/"module_em.F").read_text()
    if hashlib.sha256((source/"module_em.F").read_bytes()).hexdigest()!=PIN["module_em.F"]:
        raise ValueError("module_em source differs from WRF v4.7.1 pin")
    original_header=original.split("CONTAINS",1)[0]
    header=original_header.replace("MODULE module_em","MODULE rk_full_reference")
    extracts=[]
    extract_receipt={}
    for name in ("rk_step_prep","rk_tendency"):
        match=re.search(rf"^SUBROUTINE {name}\b.*?^END SUBROUTINE {name}\s*$",original,re.M|re.S)
        extracts.append(match.group(0))
        extract_receipt[name]={"sha256":hashlib.sha256(match.group(0).encode()).hexdigest(),
                               "first_line":original[:match.start()].count("\n")+1,
                               "last_line":original[:match.end()].count("\n")+1}
    routine="\n".join(extracts)
    (destination/"rk_full_reference.F90").write_text(header+"\nCONTAINS\n"+routine+"\nEND MODULE\n")
    mods=destination/"modules"
    mods.mkdir(exist_ok=True)
    for path in wrf_build.rglob("*.mod"):
        shutil.copy2(path,mods/path.name)
    libs=[]
    for path in [wrf_build/"main/libwrflib.a",*sorted((wrf_build/"external").rglob("lib*.a"))]:
        target=destination/path.name
        shutil.copy2(path,target)
        libs.append(target)
    extras=[]
    for name in ("pack_utils.o","module_internal_header_util.o"):
        target=destination/name
        shutil.copy2(wrf_build/"frame"/name,target)
        extras.append(target)
    receipt={"wrf_commit":COMMIT,"extracts":extract_receipt,
             "libraries":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [*libs,*extras]},
             "dependency_build":"native WRF O2 build; rk_tendency itself O0, contraction off, bounds checked",
             "commands":[]}
    receipt["native_child_sources"]={
        f"dyn_em/{name}":hashlib.sha256((wrf_build/"dyn_em"/name).read_bytes()).hexdigest()
        for name in ("module_advect_em.F","module_big_step_utilities_em.F",
                     "module_ieva_em.F","module_damping_em.F")}
    receipt["native_fcoptim"]=next(
        line.strip() for line in (wrf_build/"configure.wrf").read_text().splitlines()
        if line.startswith("FCOPTIM"))
    flags=["-O0","-fPIC","-ffree-form","-ffree-line-length-none","-ffp-contract=off",
           "-fcheck=bounds","-I",str(mods)]
    cmds=[["gfortran","-c",*flags,"rk_full_reference.F90"],
          ["gfortran","-c",*flags,"-I",str(destination),str(Path(__file__).with_name("rk_tendency_harness.F90"))],
          ["gfortran","-no-pie","-fopenmp","-o","rk_tendency_harness",
           "rk_tendency_harness.o","rk_full_reference.o",*[str(p) for p in extras],
           "-Wl,--start-group",*[str(p) for p in libs],"-Wl,--end-group",
           "-lnetcdff","-lnetcdf","-l:libmpi_mpifh.so.40","-l:libmpi.so.40"]]
    for cmd in cmds:
        receipt["commands"].append([
            part.replace(str(destination),"$BUILD_DIR").replace(str(Path(__file__).parent),"$HARNESS_DIR")
            for part in cmd])
        subprocess.run(cmd,cwd=destination,check=True)
    receipt["symbols"]=subprocess.check_output(["nm","-u",str(destination/"rk_tendency_harness")],text=True)
    (destination/"rk-full-build-receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("source","wrf_build","destination"):parser.add_argument(name,type=Path)
    args=parser.parse_args()
    build(args.source,args.wrf_build,args.destination)
