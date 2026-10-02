"""Compile the exact WRF diff_opt=1 numerical routines, with pinned bytes."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import subprocess

ROUTINES=("horizontal_diffusion","horizontal_diffusion_3dmp","vertical_diffusion","vertical_diffusion_3dmp","vertical_diffusion_u","vertical_diffusion_v")
SOURCE_HASH="bd177b6b5ba7949cf9e694d7ad654fd9ae2f07d39d85802f0716c5318889a815"
CONSTANTS_HASH="5b80377fecdc18a5f0ad38d3b6c15cfc86ad5d76701adbbbb08a08698d0f7062"


def build(source, output):
    output.mkdir(parents=True,exist_ok=True)
    raw=(source/"dyn_em/module_big_step_utilities_em.F").read_bytes()
    constants=source/"share/module_model_constants.F"
    assert hashlib.sha256(raw).hexdigest()==SOURCE_HASH
    assert hashlib.sha256(constants.read_bytes()).hexdigest()==CONSTANTS_HASH
    prefix=b"""module module_configure
  implicit none
  type grid_config_rec_type
    logical :: specified=.false., nested=.false.
    logical :: open_xs=.false., open_xe=.false., open_ys=.false., open_ye=.false.
    logical :: periodic_x=.false., polar=.false.
  end type
end module
module module_big_step_utilities_em
  use module_configure
  use module_model_constants
  implicit none
contains
"""
    slices={}
    for name in ROUTINES:
        m=re.search(rb"(?im)^ *SUBROUTINE "+name.encode()+rb" *\(.*?^ *END SUBROUTINE "+name.encode()+rb"[^\n]*\n",raw,re.S)
        assert m,name
        prefix+=m[0]
        slices[name]={"sha256":hashlib.sha256(m[0]).hexdigest(),"lines":[raw[:m.start()].count(b"\n")+1,raw[:m.end()].count(b"\n")]}
    (output/"constant-slice.f90").write_bytes(prefix+b"end module\n")
    flags=["-O0","-ffp-contract=off","-fno-tree-vectorize","-fcheck=bounds","-ffree-form","-ffree-line-length-none"]
    subprocess.run(["gfortran",*flags,"-cpp","-c",str(constants.resolve())],cwd=output,check=True)
    subprocess.run(["gfortran",*flags,"-I",str(output.resolve()),"constant-slice.f90",str(Path(__file__).with_name("constant_driver.f90")),"module_model_constants.o","-o","constant_driver"],cwd=output,check=True)
    metadata={"wrf_version":"4.7.1","wrf_commit":"f52c197ed39d12e087d02c50f412d90d418f6186","source_sha256":SOURCE_HASH,
              "constants_sha256":CONSTANTS_HASH,"slices":slices,"flags":flags,"real_kind_bytes":4,
              "compiler":subprocess.check_output(["gfortran","--version"],text=True).splitlines()[0]}
    (output/"constant-build.json").write_text(json.dumps(metadata,indent=2)+"\n", encoding="utf-8", newline="\n")
    print(json.dumps(metadata,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("source",type=Path);p.add_argument("output",type=Path)
    a=p.parse_args();build(a.source,a.output)
