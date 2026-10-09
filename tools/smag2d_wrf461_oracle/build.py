"""Compile the WRF v4.6.1 km_opt=4 / diff_opt=2 column oracle library.

Every numerical routine is a byte-unmodified slice of pristine WRF v4.6.1
source (tag v4.6.1, commit d66e442f).  The slices are wrapped in one
module together with a configuration record holding only the fields those
routines read.  The model constants module is compiled whole and unchanged.
``pipeline.F90`` is the C ABI driver; it calls the WRF routines in the order
dyn_em/module_first_rk_step_part2.F calls them.

Build flags follow the strict reference: -O0, no FMA contraction, no
vectorisation, bounds checking on.  The receipt records the source hashes,
each extracted slice's line span and hash, the compiler and every command.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

WRF_RELEASE = "4.6.1"
WRF_COMMIT = "d66e442fccc04111067e29274c9f9eaccc3cef28"
SOURCE_PINS = {
    "dyn_em/module_diffusion_em.F": "a7d4570c97e51c635e86a0dbd628c6846457ac5b93d5a7af798b118c7d8d2d54",
    "dyn_em/module_big_step_utilities_em.F": "8f0649b458fceabd5c31c87ec9f266840fb164e2629322904a3f6ee04c72331a",
    "share/module_model_constants.F": "5b80377fecdc18a5f0ad38d3b6c15cfc86ad5d76701adbbbb08a08698d0f7062",
    "share/module_bc.F": "61b9235004b2a7799faabaa928276af8a7ef2e4672619c8ad120c857461301ad",
}
DIFFUSION_ROUTINES = (
    "compute_diff_metrics", "cal_deform_and_div", "calculate_km_kh", "calculate_N2",
    "cal_dampkm", "isotropic_km", "smag_km", "smag2d_km", "tke_km", "calc_l_scale",
    "phy_bc", "horizontal_diffusion_2", "horizontal_diffusion_u_2",
    "horizontal_diffusion_v_2", "horizontal_diffusion_w_2", "horizontal_diffusion_s",
    "cal_titau_11_22_33", "cal_titau_12_21", "cal_titau_13_31", "cal_titau_23_32",
    "pthl", "pu")
EXTRA_ROUTINES = {
    "dyn_em/module_big_step_utilities_em.F": ("phy_prep",),
    "share/module_bc.F": ("set_physical_bc2d", "set_physical_bc3d"),
}
FLAGS = ["-O0", "-g", "-fPIC", "-cpp", "-ffree-form", "-ffree-line-length-none",
         "-ffp-contract=off", "-fno-tree-vectorize", "-fcheck=bounds"]

# Only the configuration fields the extracted routines read.  Species slots
# follow WRF's Thompson (mp=8) moist array: 1 is WRF's unused slot, then
# qv qc qr qi qs qg.  Absent species carry WRF's absent index 1.
CONFIG_SOURCE = """module module_oracle_config
implicit none
type grid_config_rec_type
logical :: open_xs=.false., open_xe=.false., open_ys=.false., open_ye=.false.
logical :: symmetric_xs=.false., symmetric_xe=.false., symmetric_ys=.false., symmetric_ye=.false.
logical :: periodic_x=.false., periodic_y=.false., specified=.false., nested=.false., polar=.false.
logical :: mix_full_fields=.true.
integer :: use_theta_m=0
logical :: moist_mix2_off=.false., chem_mix2_off=.false., scalar_mix2_off=.false.
logical :: tke_mix2_off=.false., tracer_mix2_off=.false.
integer :: km_opt=4, diff_opt=2, sfs_opt=0, m_opt=0, bl_pbl_physics=0
integer :: isfflx=1, cu_physics=0, shcu_physics=0, spec_bdy_width=5
real :: c_s=.25, c_k=.15, tke_drag_coefficient=0., tke_heat_flux=0.
end type
integer, parameter :: param_first_scalar=2
integer, parameter :: p_qv=2, p_qc=3, p_qr=4, p_qi=5, p_qs=6, p_qg=7
integer, parameter :: p_m11=1,p_m22=2,p_m33=3,p_m12=4,p_m13=5,p_m23=6
integer, parameter :: p_r12=1,p_r13=2,p_r23=3
integer, parameter :: p_qns=1,p_qnr=1,p_qng=1,p_qt=1,p_qnh=1,p_qvolg=1
end module
subroutine wrf_error_fatal(message)
character(*) :: message
print *, message
error stop 1
end subroutine
subroutine wrf_debug(level, message)
integer :: level
character(*) :: message
end subroutine
"""


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extract_routines(source: Path, names):
    """Exact byte slices of the named WRF routines and their line spans."""
    raw = source.read_bytes()
    pattern = rb"(?im)^\s*(?:real\s+)?(subroutine|function)\s+([a-z][a-z0-9_]*)\s*\("
    bodies, spans = [], {}
    for name in names:
        found = [m for m in re.finditer(pattern, raw) if m[2].decode().lower() == name.lower()]
        if len(found) != 1:
            raise ValueError(f"expected one WRF body for {name}, found {len(found)}")
        start, kind = found[0].start(), found[0][1]
        end_match = re.search(rb"(?im)^\s*end\s+" + kind + rb"(?:\s+" + name.encode()
                              + rb")?\s*(?:!.*)?$", raw[found[0].end():])
        if end_match is None:
            raise ValueError(f"missing end for {name}")
        end = found[0].end() + end_match.end()
        body = raw[start:end]
        bodies.append(body)
        spans[name] = {"first_line": raw[:start].count(b"\n") + 1,
                       "last_line": raw[:end].count(b"\n") + 1,
                       "sha256": hashlib.sha256(body).hexdigest()}
    return b"\n".join(bodies), spans


def periodic_top_slope_control(bodies: bytes) -> bytes:
    """ATTRIBUTION CONTROL ONLY, never the reference.

    WRF v4.6.1 compute_diff_metrics computes interior zx/zy for k = 1..kte,
    but its periodic-boundary branches compute the seam face (ids/ide,
    jds/jde) only for k = 1..ktf = kde-1.  The model-top w-level slope at the
    periodic seam is therefore never written and stays 0 while every other
    column carries the true slope.  This control widens those eight loops to
    kte, so a comparison against it isolates that one WRF defect.
    """
    text = bodies.decode()
    start = text.index("SUBROUTINE compute_diff_metrics")
    end = text.index("END SUBROUTINE compute_diff_metrics", start)
    body = text[start:end]
    out, branch, changed = [], None, 0
    for line in body.splitlines(keepends=True):
        flat = line.replace(" ", "").lower()
        if flat.startswith("if(.not.config_flags%periodic_"):
            branch = "open"
        elif branch == "open" and flat.startswith("else"):
            branch = "periodic"
        elif branch == "periodic" and flat.startswith("endif") and line.startswith("    END"):
            branch = None
        if branch == "periodic" and flat.startswith("dok=1,ktf"):
            line = line.replace("ktf", "kte")
            changed += 1
        out.append(line)
    if changed != 8:
        raise ValueError(f"periodic-top-slope control expected 8 loops, changed {changed}")
    return (text[:start] + "".join(out) + text[end:]).encode()


def build(wrf: Path, output: Path, control: str | None = None) -> Path:
    wrf, out = wrf.resolve(), output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    for relative, pin in SOURCE_PINS.items():
        if sha256(wrf / relative) != pin:
            raise ValueError(f"{relative} is not the pinned WRF v{WRF_RELEASE} source")
    bodies, spans = extract_routines(wrf / "dyn_em/module_diffusion_em.F", DIFFUSION_ROUTINES)
    extra_spans = {}
    for relative, names in EXTRA_ROUTINES.items():
        more, more_spans = extract_routines(wrf / relative, names)
        bodies += b"\n" + more
        extra_spans[relative] = more_spans
    if control == "periodic-top-slope":
        bodies = periodic_top_slope_control(bodies)
    elif control is not None:
        raise ValueError(f"unknown control {control}")
    generated = out / "wrf_routines.F90"
    generated.write_bytes(
        b"module module_diffusion_em\nuse module_oracle_config\nuse module_model_constants\n"
        b"implicit none\nINTEGER, PARAMETER :: bdyzone = 4\ncontains\n" + bodies
        + b"\nend module module_diffusion_em\n")
    (out / "oracle_config.F90").write_text(CONFIG_SOURCE, encoding="utf-8", newline="\n")
    pipeline = Path(__file__).with_name("pipeline.F90")
    commands = []
    for src, obj in ((out / "oracle_config.F90", "oracle_config.o"),
                     (wrf / "share/module_model_constants.F", "constants.o"),
                     (generated, "routines.o"), (pipeline.resolve(), "pipeline.o")):
        command = ["gfortran", *FLAGS, "-c", str(src), "-o", obj]
        commands.append(command)
        subprocess.run(command, cwd=out, check=True)
    command = ["gfortran", "-shared", "oracle_config.o", "constants.o", "routines.o",
               "pipeline.o", "-o", "oracle.so"]
    commands.append(command)
    subprocess.run(command, cwd=out, check=True)
    for command in commands:
        for i, arg in enumerate(command):
            if arg.startswith("/"):
                command[i] = Path(arg).name
    receipt = {
        "schema": "smag2d-wrf461-oracle-build-v1", "wrf_release": WRF_RELEASE,
        "control": control,
        "wrf_commit": WRF_COMMIT, "source_sha256": SOURCE_PINS,
        "compiler": subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[0],
        "commands": commands, "routines": spans, "extra_routines": extra_spans,
        "pipeline_sha256": sha256(pipeline), "config_sha256": sha256(out / "oracle_config.F90"),
        "generated_sha256": sha256(generated), "library_sha256": sha256(out / "oracle.so")}
    (out / "build-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n",
                                            encoding="utf-8", newline="\n")
    return out / "oracle.so"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wrf_source", type=Path, help="pristine WRF v4.6.1 checkout")
    parser.add_argument("output", type=Path)
    parser.add_argument("--control", choices=("periodic-top-slope",),
                        help="attribution-only modified WRF; never the reference")
    args = parser.parse_args()
    print(build(args.wrf_source, args.output, args.control))
