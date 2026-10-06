"""Build a prescribed-smoke oracle from the immutable public source text.

Usage: python -m tools.hrrr_radiation_driver_oracle.build_smoke SOURCE_DIR BUILD_DIR
Use --assemble-only to prepare the source receipt without a compiler.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from urllib.request import urlopen

from tools.hrrr_radiation_driver_oracle import build as base

ROOT_URL = (f"https://raw.githubusercontent.com/NOAA-EMC/HRRR/{base.COMMIT}/"
            "sorc/hrrr_wrfarw.fd/WRFV3.9/")
POST_URL = (f"https://raw.githubusercontent.com/NOAA-EMC/HRRR/{base.COMMIT}/"
            "sorc/hrrr_wrfpost.fd/")
SMOKE_SOURCE_SHA256 = "024c838f64e5c5de098a63374b2163760f680e1e7d9bf81d882fc051dea3b31f"
POST_SOURCE_SHA256 = {
    "MDLFLD.f": "86a9189c2e6d3dd642250eab007ce572b4c1525654e49250bd4bd91d864e937f",
    "params.F": "2e1cec8bea69c256d3a2b401e50fbcb6e30dd885f4691d531c371ca0e206c161",
}


def source_line(text, pattern):
    hits = [line for line in text.splitlines() if re.search(pattern, line, re.I)]
    if len(hits) != 1:
        raise ValueError(f"source statement {pattern!r}: found {len(hits)} matches")
    return hits[0].split("!", 1)[0].strip()


def smoke_unit(smoke_source, driver, post_source, post_params):
    constants = source_line(smoke_source, r"parameter\s*::\s*sc_me=")
    extinction = source_line(smoke_source, r"^\s*ext2=\s*sc_me\s*\+\s*ab_me")
    aod = source_line(smoke_source, r"^\s*aod3d\(i,k,j\)\s*=\s*1.e-6")
    addition = source_line(driver, r"^\s*taod5503d\(i,k,j\)\s*=.*MIN\(3.0,AOD3D_SMOKE")
    rd = source_line(post_params, r"^\s*real,\s*parameter\s*::\s*RD=")
    posted = source_line(post_source, r"^\s*GRID1\(I,J\)\s*=.*PMID.*SMOKE.*1.0e-9")
    return f"""module prescribed_smoke_source_oracle
implicit none
{constants}
{rd}
contains
subroutine fill_source_smoke_aod(ncol,nz,smoke,rho,dz,out)
integer,intent(in)::ncol,nz
real,intent(in)::smoke(ncol,nz),rho(ncol,nz),dz(ncol,nz)
real,intent(out)::out(ncol,nz)
real::chem(ncol,nz,1,1),rho_phy(ncol,nz,1),dz8w(ncol,nz,1),aod3d(ncol,nz,1),ext2
integer::i,j,k,p_smoke
p_smoke=1
chem(:,:,1,1)=smoke
rho_phy(:,:,1)=rho
dz8w(:,:,1)=dz
{extinction}
do j=1,1
do k=1,nz
do i=1,ncol
{aod}
enddo
enddo
enddo
out=aod3d(:,:,1)
end subroutine
subroutine add_source_smoke_aod(ncol,nz,taod,smoke)
integer,intent(in)::ncol,nz
real,intent(inout)::taod(ncol,nz)
real,intent(in)::smoke(ncol,nz)
real::taod5503d(ncol,nz,1),aod3d_smoke(ncol,nz,1)
integer::i,j,k
taod5503d(:,:,1)=taod
aod3d_smoke(:,:,1)=smoke
do j=1,1
do k=1,nz
do i=1,ncol
{addition}
enddo
enddo
enddo
taod=taod5503d(:,:,1)
end subroutine
subroutine source_post_inverse(ncol,nz,pin,tin,qin,pmout,qout)
integer,intent(in)::ncol,nz
real,intent(in)::pin(ncol,nz),tin(ncol,nz),qin(ncol,nz)
real,intent(out)::pmout(ncol,nz),qout(ncol,nz)
real::PMID(ncol,1,nz),T(ncol,1,nz),SMOKE(ncol,1,nz,1),GRID1(ncol,1)
integer::i,j,k,ll
PMID(:,1,:)=pin
T(:,1,:)=tin
SMOKE(:,1,:,1)=qin
do k=1,nz
ll=k
do j=1,1
do i=1,ncol
{posted}
pmout(i,k)=GRID1(i,j)
qout(i,k)=(GRID1(i,j)/(1.e-9))/((1./RD)*(PMID(i,j,ll)/T(i,j,ll)))
enddo
enddo
enddo
end subroutine
end module
"""


def program_text():
    original = (base.HERE / "oracle_aer3.f90").read_text(encoding="utf-8")
    text = original.replace("program oracle_aer3", "program oracle_aer3_smoke")
    text = text.replace("  use module_ra_aerosol", "  use module_ra_aerosol\n  use prescribed_smoke_source_oracle")
    text = text.replace("  real, allocatable :: tmp(:,:)",
        "  real, allocatable :: tmp(:,:), smoke(:,:), rho_dry(:,:), smoke_aod(:,:), pm_post(:,:), q_recovered(:,:)")
    text = text.replace("  allocate(tmp(ncol, nz))",
        "  allocate(tmp(ncol, nz),smoke(ncol,nz),rho_dry(ncol,nz),smoke_aod(ncol,nz),pm_post(ncol,nz),q_recovered(ncol,nz))")
    text = text.replace("  read(10) ht(:,1)",
        "  read(10) ht(:,1)\n  read(10) smoke\n  read(10) rho_dry")
    needle = "  do i = 1, ncol\n     do k = 1, nz\n        aod2"
    replacement = ("  call fill_source_smoke_aod(ncol,nz,smoke,rho_dry,dz(:,1:nz,1),smoke_aod)\n"
                   "  call add_source_smoke_aod(ncol,nz,taod3(:,1:nz,1),smoke_aod)\n" + needle)
    if needle not in text:
        raise ValueError("base aerosol oracle no longer has the expected column-sum point")
    text = text.replace(needle, replacement)
    text = text.replace("  write(11) aod2(:,1)",
        "  write(11) aod2(:,1)\n  write(11) smoke_aod\n"
        "  call source_post_inverse(ncol,nz,p(:,1:nz,1),t(:,1:nz,1),smoke,pm_post,q_recovered)\n"
        "  write(11) pm_post\n  write(11) q_recovered")
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("build_dir", type=Path)
    parser.add_argument("--assemble-only", action="store_true")
    args = parser.parse_args()
    sources = base.fetch(args.source_dir)
    path = args.source_dir / "module_add_emiss_burn.F"
    if not path.is_file():
        with urlopen(ROOT_URL + "smoke/module_add_emiss_burn.F") as response:
            path.write_bytes(response.read())
    sources[path.name] = base.sha256_of(path)
    if sources[path.name] != SMOKE_SOURCE_SHA256:
        raise ValueError("smoke source SHA256 differs from the immutable public source pin")
    smoke = path.read_text(encoding="utf-8", errors="replace")
    driver = (args.source_dir / "module_radiation_driver.F").read_text(encoding="utf-8", errors="replace")
    post = {}
    for name in ("MDLFLD.f", "params.F"):
        path = args.source_dir / name
        if not path.is_file():
            with urlopen(POST_URL + name) as response:
                path.write_bytes(response.read())
        sources[name] = base.sha256_of(path)
        if sources[name] != POST_SOURCE_SHA256[name]:
            raise ValueError(f"{name}: source SHA256 differs from the immutable public source pin")
        post[name] = path.read_text(encoding="utf-8", errors="replace")
    args.build_dir.mkdir(parents=True, exist_ok=True)
    unit = args.build_dir / "wrf_smoke_extract.F90"
    unit.write_text(base.assemble(args.source_dir) + smoke_unit(smoke, driver, post["MDLFLD.f"], post["params.F"]), encoding="utf-8")
    program = args.build_dir / "oracle_aer3_smoke.f90"
    program.write_text(program_text(), encoding="utf-8")
    flags = ["-O0", "-ffp-contract=off", "-ffree-form", "-ffree-line-length-none", "-cpp", "-DEM_CORE=1"]
    receipt = {"commit": base.COMMIT, "tag": base.TAG, "sources": sources,
               "flags": flags, "extract_sha256": base.sha256_of(unit),
               "program_sha256": base.sha256_of(program), "compiled": False}
    if not args.assemble_only:
        receipt["gfortran"] = subprocess.run(["gfortran", "--version"],
            check=True, capture_output=True, text=True).stdout.splitlines()[0]
        subprocess.run(["gfortran", *flags, "-J", str(args.build_dir), "-o",
            str(args.build_dir / "oracle_aer3_smoke"), str(unit), str(program)], check=True)
        receipt["compiled"] = True
    (args.build_dir / "smoke-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"compiled": receipt["compiled"], "receipt": str(args.build_dir / "smoke-receipt.json")}))


if __name__ == "__main__":
    main()
