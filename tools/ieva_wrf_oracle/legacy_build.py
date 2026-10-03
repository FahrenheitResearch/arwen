"""Build a pinned earlier WRF implicit-advection reference, without WRF MPI.

The Fortran arithmetic is extracted from the operational source, with only
the HYBRID_COORD=1 macros expanded and the declared A179 lower w-boundary
correction applied. Configuration and logging stand-ins contain no numerical
work. The original and corrected routines are compiled separately so the
fixture records the divergence, not just the corrected answer.

Usage: python legacy_build.py SOURCE_DIRECTORY BUILD_DIRECTORY
Missing source files are fetched from the immutable public commit below.
The WRF notice is reproduced in ../../licenses/LICENSE-WRF-public-domain.txt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from urllib.request import urlopen

from wrf_a179 import _LOWER

COMMIT = "40ee6058c2fc6624cbfbbe8cf1c20c59e6a45827"
BASE = (f"https://raw.githubusercontent.com/NOAA-EMC/HRRR/{COMMIT}/"
        "sorc/hrrr_wrfarw.fd/WRFV3.9/dyn_em/")
PIN = {
    "module_advect_em.F": "3c347937a5d78e60f2fe7ff85e208ef144e79c7a6b46cc4b22fcaa2c98a9b5c1",
    "module_big_step_utilities_em.F": "88b180f1c3cce95ba965d8183a6e51c62df0d58754d941dd17c0db6f22df7239",
    "module_em.F": "0a92ea1ca2f73d267ccedf04f37e99fbaded85831186774bba3703aea8491cc1",
    "solve_em.F": "84a246f0fcb4b9521a89c437ecccf2f08b1539b996e0e5151617d88e28b88585",
}

SHIM = """module module_configure
type grid_config_rec_type
integer :: rk_ord=3,zadvect_implicit=1
logical :: specified=.true.,nested=.false.,periodic_x=.false.,periodic_y=.false.
logical :: open_xs=.false.,open_xe=.false.,open_ys=.false.,open_ye=.false.
logical :: polar=.false.
end type
end module
module module_wrf_error
character(len=512) :: wrf_err_message
contains
subroutine wrf_debug(level,message)
integer,intent(in):: level
character(len=*),intent(in):: message
end subroutine
logical function wrf_at_debug_level(level)
integer,intent(in):: level
wrf_at_debug_level=.false.
end function
end module
module legacy_reference
use module_configure
use module_wrf_error
real, parameter :: g=9.81
contains
"""


def extract(source: str, name: str) -> tuple[str, dict]:
    match = re.search(rf"^\s*SUBROUTINE {name}\b.*?^\s*END SUBROUTINE {name}\s*$",
                      source, re.M | re.S | re.I)
    if match is None:
        raise ValueError(f"missing source routine {name}")
    body = match.group(0)
    return body, {"sha256": hashlib.sha256(body.encode()).hexdigest(),
                  "first_line": source[:match.start()].count("\n") + 1,
                  "last_line": source[:match.end()].count("\n") + 1}


def hybrid(body: str, split: bool = False) -> str:
    """Expand exactly the source's variadic mass macros on executable lines."""
    names = ("mut",) if split else ("mut", "muu", "muv")
    coord = "f" if split else ""
    result = []
    for line in body.splitlines(keepends=True):
        code, bang, comment = line.partition("!")
        for name in names:
            code = re.sub(rf"\b{name}\(i,j\)",
                          f"(c1{coord}(k)*{name}(i,j)+c2{coord}(k))", code)
        result.append(code + bang + comment)
    return "".join(result)


def lower_boundary(body: str) -> str:
    body = body.replace("                              cf1, cf2, cf3,                 &",
                        "                              cf1, cf2, cf3,                 &\n"
                        "                              c1h, c2h, muu, muv,            &")
    marker = "   REAL , DIMENSION( ims:ime , jms:jme ) , INTENT(IN   ) :: mut"
    if body.count(marker) != 1:
        raise ValueError("lower boundary mass declaration anchor changed")
    body = body.replace(marker, marker + "\n"
                        "   REAL, DIMENSION(ims:ime,jms:jme), INTENT(IN) :: muu,muv\n"
                        "   REAL, DIMENSION(kms:kme), INTENT(IN) :: c1h,c2h")
    for old, new in _LOWER:
        if body.count(old) != 1:
            raise ValueError("lower boundary arithmetic anchor changed")
        body = body.replace(old, new)
    return body


def driver() -> str:
    text = Path(__file__).with_name("ieva_oracle.F90").read_text()
    text = text.replace("  use module_ieva_em", "  use legacy_reference")
    text = text.replace("  use module_big_step_utilities_em, only: calc_mu_uv_1\n", "")
    first = text.index("  call CALC_MUT_NEW")
    last = text.index("  call advect_u_implicit", first)
    text = text[:first] + "  mut_new = mut\n  call write2('mut_new', mut_new, ny, nx)\n\n" + text[last:]
    text = text.replace("WW_SPLIT(wwE, wwI, u, v, ww, mut,",
                        "WW_SPLIT(wwE, wwI, ph, phb, u, v, ww, w_old, mut,")
    text = text.replace("WW_SPLIT(wwE_m, wwI_m, u_old, v_old, ww_m, muts,",
                        "WW_SPLIT(wwE_m, wwI_m, ph, phb, u, v, ww_m, w_old, muts,")
    text = text.replace("muu_old, muu, muu_new, cf", "mut, cf")
    text = text.replace("muv_old, muv, muv_new, cf", "mut, cf")
    text = text.replace("mut_old, mut, mut_new, cf", "mut, cf")
    text = text.replace("mu0s, muts, muts, cf", "muts, cf")
    # The legacy wind routines take the actual coupled stage face mass.
    for wind, face in (("u", "muu"), ("v", "muv")):
        start = text.index(f"  call advect_{wind}_implicit")
        end = text.index("  call write3", start)
        block = text[start:end].replace("fnm, fnp, dt, rdx", f"fnm, fnp, dt, {face}, rdx")
        text = text[:start] + block + text[end:]
    text = text.replace("ph, ph_old, rph_t, c1f, c2f, cf1, cf2, cf3, &",
                        "c1f, c2f, ph, ph_old, rph_t, cf1, cf2, cf3, &")
    return text


def build(source: Path, output: Path) -> None:
    source.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    originals = {}
    for name, pin in PIN.items():
        path = source / name
        if not path.exists():
            path.write_bytes(urlopen(BASE + name, timeout=60).read())
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != pin:
            raise ValueError(f"source hash differs for {name}")
        originals[name] = data.decode()
    extracts = {}
    bodies = []
    for name in ("TRIDIAG2D", "advect_ph_implicit", "advect_s_implicit",
                 "advect_u_implicit", "advect_v_implicit", "advect_w_implicit"):
        body, extracts[name] = extract(originals["module_advect_em.F"], name)
        bodies.append(hybrid(body))
    body, extracts["WW_SPLIT"] = extract(originals["module_big_step_utilities_em.F"], "WW_SPLIT")
    bodies.append(hybrid(body, split=True))
    flags = ["-O2", "-ffp-contract=off", "-ffree-form", "-ffree-line-length-none",
             "-fcheck=bounds", "-fno-fast-math"]
    receipt = {"commit": COMMIT, "sources": PIN, "extracts": extracts,
               "compiler": subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[0],
               "flags": flags, "transformations": ["HYBRID_COORD=1 source macro expansion",
               "A179 lower w boundary uncoupling in corrected arm only"],
               "configuration": "minimal consumed fields; logging calls are inert",
               "gravity": "WRF module_model_constants g=9.81, REAL",
               "call_sites": {"dynamics": "module_em.F:461-727, final RK stage",
                   "scalar": "module_em.F:1221-1352, final RK stage; solve_em.F:2401-2415 supplies current u_2/v_2",
                   "dt": "module_em.F:473 and 1217; solve_em.F:601, full dt on final RK stage"}}
    for corrected in (False, True):
        out = output / ("corrected" if corrected else "original")
        out.mkdir(exist_ok=True)
        selected = [lower_boundary(b) if corrected and "SUBROUTINE advect_w_implicit" in b else b
                    for b in bodies]
        (out / "legacy_reference.f90").write_text(SHIM + "\n".join(selected) + "\nend module\n")
        (out / "legacy_driver.F90").write_text(driver())
        subprocess.run(["gfortran", *flags, "-c", "legacy_reference.f90"], cwd=out, check=True)
        subprocess.run(["gfortran", *flags, *(["-DA179"] if corrected else []),
                        "legacy_driver.F90", "legacy_reference.o", "-o", "legacy_oracle"],
                       cwd=out, check=True)
        receipt["corrected" if corrected else "original"] = {
            "source_sha256": hashlib.sha256((out / "legacy_reference.f90").read_bytes()).hexdigest(),
            "driver_sha256": hashlib.sha256((out / "legacy_driver.F90").read_bytes()).hexdigest()}
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build(args.source.resolve(), args.output.resolve())
