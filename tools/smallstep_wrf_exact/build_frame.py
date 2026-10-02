"""Compile the unchanged WRF specified-frame routine for driver verification."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


def build(source_path: Path, output: Path):
    data=source_path.read_bytes()
    text=data.decode("utf-8")
    match=re.search(r"(?im)^   SUBROUTINE spec_bdyupdate\([\s\S]*?^   END SUBROUTINE spec_bdyupdate\s*$",text)
    if match is None:
        raise ValueError("The unchanged WRF specified-frame routine was not found")
    routine=match.group()
    configure="""module module_configure
type grid_config_rec_type
logical :: periodic_x=.false.
end type
end module
"""
    module="module module_exact_frame\nuse module_configure\ncontains\n"+routine+"\nend module\n"
    wrapper="""subroutine oracle_frame(field,field_tend,dt,nx,ny,nz,spec_zone,periodic_x) bind(C)
use iso_c_binding
use module_exact_frame
implicit none
integer(c_int) :: nx,ny,nz,spec_zone,periodic_x
real(c_float) :: field(nx,nz,ny),field_tend(nx,nz,ny),dt
type(grid_config_rec_type) :: cfg
cfg%periodic_x=periodic_x/=0
call spec_bdyupdate(field,field_tend,dt,'t',cfg,spec_zone, &
  1,nx+1,1,ny+1,1,nz+1,1,nx,1,ny,1,nz, &
  1,nx,1,ny,1,nz,1,nx,1,ny,1,nz)
end subroutine
"""
    output.mkdir(parents=True,exist_ok=True)
    source=output/"frame.F90"
    source.write_text(configure+module+wrapper,encoding="utf-8")
    command=["gfortran","-shared","-fPIC","-ffree-line-length-none","-O0",
             "-ffp-contract=off","-fcheck=bounds",str(source.resolve()),"-o","libframe.so"]
    subprocess.run(command,cwd=output,check=True)
    receipt={"source_sha256":hashlib.sha256(data).hexdigest(),
             "unchanged_routine_sha256":hashlib.sha256(routine.encode()).hexdigest(),
             "commands":[command],
             "compiler":subprocess.check_output(["gfortran","--version"],text=True).splitlines()[0],
             "library_sha256":hashlib.sha256((output/"libframe.so").read_bytes()).hexdigest()}
    (output/"receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(receipt["library_sha256"])


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    build(a.source,a.output)
