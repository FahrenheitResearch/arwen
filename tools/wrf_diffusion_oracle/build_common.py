"""Compile byte-preserved WRF routine bodies with a small service ABI.

The configuration type only contains fields read by these routines. Constants
are compiled from WRF's source. No arithmetic or model routine is stubbed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess

WRF_COMMIT = "f52c197ed39d12e087d02c50f412d90d418f6186"
SOURCE_PINS = {
    "module_diffusion_em.F": "a7d4570c97e51c635e86a0dbd628c6846457ac5b93d5a7af798b118c7d8d2d54",
    "module_big_step_utilities_em.F": "bd177b6b5ba7949cf9e694d7ad654fd9ae2f07d39d85802f0716c5318889a815",
    "module_model_constants.F": "5b80377fecdc18a5f0ad38d3b6c15cfc86ad5d76701adbbbb08a08698d0f7062",
    "module_bc.F": "61b9235004b2a7799faabaa928276af8a7ef2e4672619c8ad120c857461301ad",
}
FLAGS = ["-O0", "-g", "-fPIC", "-cpp", "-ffree-form", "-ffree-line-length-none",
         "-ffp-contract=off", "-fno-tree-vectorize", "-fcheck=bounds"]

CONFIG_SOURCE = """module module_oracle_config
implicit none
type grid_config_rec_type
logical :: open_xs=.false., open_xe=.false., open_ys=.false., open_ye=.false.
logical :: symmetric_xs=.false., symmetric_xe=.false., symmetric_ys=.false., symmetric_ye=.false.
logical :: periodic_x=.false., periodic_y=.false., specified=.false., nested=.false., polar=.false.
logical :: mix_full_fields=.false.
integer :: use_theta_m=0
logical :: moist_mix2_off=.false., chem_mix2_off=.false., scalar_mix2_off=.false.
logical :: tke_mix2_off=.false., tracer_mix2_off=.false.
integer :: km_opt=4, diff_opt=2, sfs_opt=0, m_opt=0, bl_pbl_physics=0
integer :: isfflx=0, cu_physics=0, shcu_physics=0, spec_bdy_width=1
real :: c_s=.25, c_k=.15, tke_drag_coefficient=0., tke_heat_flux=0.
end type
! Species slots are explicit fixture layout, not physical constants.
integer, parameter :: param_first_scalar=2, p_qv=2, p_qc=3, p_qi=4
integer, parameter :: p_m11=1,p_m22=2,p_m33=3,p_m12=4,p_m13=5,p_m23=6
integer, parameter :: p_r12=1,p_r13=2,p_r23=3
integer, parameter :: p_qns=4,p_qnr=5,p_qng=6,p_qt=7,p_qnh=8,p_qvolg=9
end module
subroutine wrf_error_fatal(message)
character(*) :: message
print *, message
error stop 1
end subroutine
"""

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def extract_routines(source, names):
    """Return exact byte slices and their original line ranges."""
    raw = Path(source).read_bytes()
    pattern = rb"(?im)^\s*(?:real\s+)?(subroutine|function)\s+([a-z][a-z0-9_]*)\s*\("
    bodies, spans = [], {}
    for name in names:
        matches = [m for m in re.finditer(pattern, raw) if m[2].decode().lower() == name.lower()]
        if len(matches) != 1:
            raise ValueError(f"Expected one WRF body for {name}, found {len(matches)}")
        start = matches[0].start()
        kind = matches[0][1]
        endmatch = re.search(rb"(?im)^\s*end\s+" + kind + rb"(?:\s+" + name.encode() + rb")?\s*(?:!.*)?$", raw[matches[0].end():])
        if endmatch is None:
            raise ValueError(f"Missing end for {name}")
        end = matches[0].end() + endmatch.end()
        body = raw[start:end]
        bodies.append(body)
        spans[name] = {"first_line": raw[:start].count(b"\n")+1,
                       "last_line": raw[:end].count(b"\n")+1,
                       "sha256": hashlib.sha256(body).hexdigest()}
    return b"\n".join(bodies), spans

def build(source, constants, wrapper, output_dir, routines, *, module="module_diffusion_em", extra_sources=None, module_declarations=""):
    """Build a shared library; receipt records compiler, bodies and bytes."""
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    for path in [source, constants, *(extra_sources or {})]:
        pin=SOURCE_PINS.get(Path(path).name)
        if pin is None or sha256(path)!=pin:
            raise ValueError(f"WRF source changed: {Path(path).name}. The reference must use the pinned v4.7.1 bytes.")
    bodies, spans = extract_routines(source, routines)
    extra_receipts = {}
    for extra, names in (extra_sources or {}).items():
        extra_bodies, extra_spans = extract_routines(extra,names)
        bodies += b"\n" + extra_bodies
        extra_receipts[Path(extra).name] = {"sha256":sha256(extra),"routines":extra_spans}
    generated = out / "wrf_routines.F90"
    generated.write_bytes(f"module {module}\nuse module_oracle_config\nuse module_model_constants\n{module_declarations}\ncontains\n".encode()+bodies+f"\nend module {module}\n".encode())
    (out / "oracle_config.F90").write_text(CONFIG_SOURCE, encoding="utf-8", newline="\n")
    commands = []
    for src, obj in [(out/"oracle_config.F90", "oracle_config.o"), (Path(constants).resolve(), "constants.o"),
                     (generated, "routines.o"), (Path(wrapper).resolve(), "wrapper.o")]:
        command = ["gfortran", *FLAGS, "-c", str(src), "-o", obj]
        commands.append(command)
        subprocess.run(command, cwd=out, check=True)
    command = ["gfortran", "-shared", "oracle_config.o", "constants.o", "routines.o", "wrapper.o", "-o", "oracle.so"]
    commands.append(command)
    subprocess.run(command, cwd=out, check=True)
    compiler = subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[0]
    receipt = {"wrf_release": "4.7.1", "wrf_commit": WRF_COMMIT, "compiler": compiler,
               "commands": commands, "routines": spans,
               "extra_sources": extra_receipts,
               "source_sha256": sha256(source), "constants_sha256": sha256(constants),
               "wrapper_sha256": sha256(wrapper), "config_sha256": sha256(out/"oracle_config.F90"),
               "generated_sha256": sha256(generated), "library_sha256": sha256(out/"oracle.so")}
    (out/"build-receipt.json").write_text(json.dumps(receipt, indent=2)+"\n", encoding="utf-8", newline="\n")
    return out/"oracle.so"
