"""Extract and compile the pinned fork's sun-angle and sea-ice albedo blocks.

Usage: python build.py SOURCE_DIRECTORY BUILD_DIRECTORY

The arithmetic and branch bodies are read directly from the public source
files after their SHA256 hashes are checked.  The wrapper supplies array
bounds and C bindings only.  Build on a CPU with gfortran; the compiler
flags preserve each float32 statement's rounding.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from urllib.request import urlopen

COMMIT = "40ee6058c2fc6624cbfbbe8cf1c20c59e6a45827"
TAG = "v4.1.21"
BASE = (f"https://raw.githubusercontent.com/NOAA-EMC/HRRR/{COMMIT}/"
        "sorc/hrrr_wrfarw.fd/WRFV3.9/phys/")
PIN = {
    "module_radiation_driver.F":
        "7464639e53f6f40b4810943ee9fa21a40b0f53f7525c08cc13b84ae4c7d82801",
    "module_surface_driver.F":
        "5aed6cc50973ada58661262ded97d5ce8b1bf5407ad423f3df2f34e64da5612a",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(source_dir: Path) -> dict[str, str]:
    source_dir.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name, want in PIN.items():
        path = source_dir / name
        if not path.is_file():
            with urlopen(BASE + name) as source:
                path.write_bytes(source.read())
        got = sha256(path)
        if got != want:
            raise ValueError(f"{name} SHA256 {got} differs from pinned {want}")
        hashes[name] = got
    return hashes


def lines(path: Path, first: int, last: int) -> str:
    return "\n".join(path.read_text(encoding="utf-8").splitlines()[
        first - 1:last]) + "\n"


def assemble(source_dir: Path) -> str:
    radiation = source_dir / "module_radiation_driver.F"
    surface = source_dir / "module_surface_driver.F"
    # Parameter declarations and DATA are preserved; comments need no
    # compilation and are omitted from this generated extraction.
    declarations = "\n".join(line for line in lines(
        radiation, 781, 789).splitlines() if not line.lstrip().startswith("!"))
    radiation_block = lines(radiation, 1038, 1063)
    # The complete driver pre/post bodies include emissivity and TSK.
    # Their outputs are carried too, so the extraction has no replacement
    # for any numerical line in the albedo branch.
    override = lines(surface, 3295, 3301)
    deblend = lines(surface, 3307, 3316)
    reblend = lines(surface, 3374, 3381)
    return f"""module solar_albedo_oracle
use iso_c_binding
implicit none
contains
subroutine solar(n,itimestep,alb_sol,albedo,albbck,xland,snow,xice,ivgtyp,albsol,albbcksol,coszen) bind(C)
integer(c_int), value :: n,itimestep,alb_sol
real(c_float), intent(in) :: albedo(n,1),albbck(n,1),xland(n,1),snow(n,1),xice(n,1),coszen(n,1)
integer(c_int), intent(in) :: ivgtyp(n,1)
real(c_float), intent(inout) :: albsol(n,1),albbcksol(n,1)
integer :: i,j,its,ite,jts,jte
{declarations}
its=1
ite=n
jts=1
jte=1
{radiation_block}
end subroutine
subroutine ice_pre(n,fractional_seaice,xice_threshold,seaice_albedo_default,xice,albsol,albbcksol,emiss,tsk,tsk_save) bind(C)
integer(c_int), value :: n,fractional_seaice
real(c_float), value :: xice_threshold,seaice_albedo_default
real(c_float), intent(in) :: xice(n,1),tsk_save(n,1)
real(c_float), intent(inout) :: albsol(n,1),albbcksol(n,1),emiss(n,1),tsk(n,1)
integer :: i,j,ij,j_start(1),j_end(1),i_start(1),i_end(1)
ij=1
j_start=1
j_end=1
i_start=1
i_end=n
{override}
if (fractional_seaice == 1) then
{deblend}
endif
end subroutine
subroutine ice_post(n,fractional_seaice,xice_threshold,xice,albsol,emiss) bind(C)
integer(c_int), value :: n,fractional_seaice
real(c_float), value :: xice_threshold
real(c_float), intent(in) :: xice(n,1)
real(c_float), intent(inout) :: albsol(n,1),emiss(n,1)
integer :: i,j,ij,j_start(1),j_end(1),i_start(1),i_end(1)
ij=1
j_start=1
j_end=1
i_start=1
i_end=n
if (fractional_seaice == 1) then
{reblend}
endif
end subroutine
end module
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("build_dir", type=Path)
    args = parser.parse_args()
    hashes = fetch(args.source_dir)
    args.build_dir.mkdir(parents=True, exist_ok=True)
    source = args.build_dir / "extract.f90"
    source.write_text(assemble(args.source_dir), encoding="utf-8")
    flags = ["-O0", "-ffp-contract=off", "-ffree-line-length-none",
             "-fcheck=bounds", "-shared", "-fPIC"]
    version = subprocess.run(["gfortran", "--version"], check=True,
                             capture_output=True, text=True).stdout.splitlines()[0]
    subprocess.run(["gfortran", *flags, "-J", str(args.build_dir),
                    "-o", str(args.build_dir / "libsolar_albedo.so"),
                    str(source)], check=True)
    receipt = {"commit": COMMIT, "tag": TAG, "sources": hashes,
               "gfortran": version, "flags": flags,
               "extract_sha256": sha256(source),
               "blocks": {"radiation_parameters": [781, 789],
                          "radiation_update": [1038, 1063],
                          "surface_override": [3295, 3301],
                          "surface_deblend": [3307, 3316],
                          "surface_reblend": [3374, 3381]}}
    (args.build_dir / "receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
