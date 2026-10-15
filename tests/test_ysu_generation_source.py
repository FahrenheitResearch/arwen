"""Grade the live generation's expression against the pinned public source.

This check compiles the actual expression from each source, not a second
Python implementation of it. Run on a CPU box with YSU_WRF480_SOURCE set.
"""
from pathlib import Path
import os
import re
import subprocess

import numpy as np
import pytest


@pytest.mark.parametrize("version,plant", [("wrf_461", False), ("wrf_480", False), ("wrf_480", True)])
def test_default_topdown_term_matches_pinned_fortran(tmp_path, version, plant):
    source = os.environ.get("YSU_WRF480_SOURCE")
    if not source:
        pytest.skip("requires the pinned public WRF 4.8 MMM source")
    if version == "wrf_461":
        source = str(Path(source).parents[3] / "v4.6.1/phys/physics_mmm/bl_ysu.F90")
    expression = re.search(r"ent_eff\s*=\s*([^\n]+)",
                           Path(source).read_text()).group(1).split("!")[0]
    from gpuwm.config import RunConfig
    from gpuwm.ysu_generations import ysu_generation
    assert RunConfig.__dataclass_fields__["ysu_version"].default == "wrf_480"
    unit = ysu_generation(version)["kernel_module"]
    kernel = (Path(__file__).parents[1] / f"gpuwm/core/kernels/{unit}.cu").read_text()
    expression_c = re.search(r"real ent_eff = ([^;]+);", kernel).group(1)
    if plant:
        # 0.4 is 0x3ecccccd; this adjacent float flips only its lowest bit.
        assert "0.4f" in expression_c
        expression_c = expression_c.replace("0.4f", "0x1.999998p-2f")
    values = [0., .01, .125, np.nextafter(np.float32(.25), np.float32(0)),
              .25, np.nextafter(np.float32(.25), np.float32(1)), 1., 100.]
    rows = ", ".join(str(float(x)) for x in values)
    f = tmp_path / "term.f90"
    f.write_text("program term\nimplicit none\nreal :: tmp1, ent_eff\n"
                 "real :: values(8) = [" + rows + "]\n"
                 "integer :: i\ndo i=1,8\ntmp1=values(i)\nent_eff=" + expression +
                 "\nwrite(*,'(Z8.8)') transfer(ent_eff,0)\nend do\nend program\n")
    c = tmp_path / "term.cpp"
    c.write_text("#include <cstdio>\n#include <cstring>\n#include <cmath>\n"
                 "using real=float; float ysu_min(float a,float b){return a<b?a:b;}\n"
                 "int main(){float values[]={" + ",".join(str(float(x))+"f" for x in values) +
                 "}; for(float tmp1:values){real ent_eff=" + expression_c +
                 "; unsigned u; memcpy(&u,&ent_eff,4); printf(\"%08X\\n\",u);}}\n")
    subprocess.run(["gfortran", "-O0", "-ffree-line-length-none", str(f), "-o", str(tmp_path / "f")], check=True)
    subprocess.run(["g++", "-O0", "-ffp-contract=off", str(c), "-o", str(tmp_path / "c")], check=True)
    want = subprocess.check_output([str(tmp_path / "f")])
    got = subprocess.check_output([str(tmp_path / "c")])
    if plant:
        assert got != want, "source oracle missed the one-bit cap error"
    else:
        assert got == want, f"actual kernel term {got!r} != pinned Fortran {want!r}"
