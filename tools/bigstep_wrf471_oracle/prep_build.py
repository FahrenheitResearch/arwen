"""Extract unchanged WRF routines and generate only their binary I/O driver."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

SOURCE_SHA = "bd177b6b5ba7949cf9e694d7ad654fd9ae2f07d39d85802f0716c5318889a815"
FIELDS3 = "u v ww ph ph_old phb w p pb alt t".split()
FIELDS2 = "mut muu muv msfux msfuy msfvx msfvx_inv msfvy msftx msfty".split()
FIELDS1 = "c1h c2h c1f c2f fnm fnp rdnw dnw znw".split()
OUTPUTS = "ph_tend rho th_phy th_phy_m_t0 p_phy pi_phy u_phy v_phy p8w t_phy t8w z z_at_w dz8w p_hyd p_hyd_w".split()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("wrf_source", type=Path)
    ap.add_argument("out", type=Path)
    args = ap.parse_args()
    raw = args.wrf_source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA:
        raise SystemExit("WRF bigstep source differs from the pinned v4.7.1 bytes")
    source = raw.decode()
    args.out.mkdir(parents=True, exist_ok=True)
    extracts = []
    for name in ("rhs_ph", "phy_prep"):
        match = re.search(rf"(?im)^\s*SUBROUTINE {name}\s*\(.*?^\s*END SUBROUTINE {name}\s*$", source, re.S | re.M)
        if match is None:
            raise SystemExit(f"missing exact routine {name}")
        extracts.append(match.group())
    module = "module oracle_prep\nuse module_model_constants\nuse module_configure, only: grid_config_rec_type\nuse module_state_description\nimplicit none\ncontains\n" + "\n".join(extracts) + "\nend module\n"
    (args.out / "prep_exact.F90").write_text(module)
    decl = lambda names, rank: "\n".join(f"real, allocatable :: {name}({','.join(':' for _ in range(rank))})" for name in names)
    alloc = lambda names, shape: "\n".join(f"allocate({name}({shape}))" for name in names)
    read = lambda names: "\n".join(f"read(10) {name}" for name in names)
    text = f"""program run_prep
use oracle_prep
implicit none
type(grid_config_rec_type) :: config
integer :: nx,ny,nz,order,specified,phi_adv,i
integer :: ims,ime,jms,jme,kms,kme
character(64) :: arg
{decl(FIELDS3 + OUTPUTS, 3)}
{decl(FIELDS2, 2)}
{decl(FIELDS1, 1)}
real, allocatable :: moist(:,:,:,:)
real :: cfn,cfn1,rdx,rdy,p_top
call get_command_argument(1,arg); read(arg,*) nx
call get_command_argument(2,arg); read(arg,*) ny
call get_command_argument(3,arg); read(arg,*) nz
call get_command_argument(4,arg); read(arg,*) order
call get_command_argument(5,arg); read(arg,*) specified
call get_command_argument(6,arg); read(arg,*) phi_adv
ims=-3; ime=nx+4; jms=-3; jme=ny+4; kms=1; kme=nz+1
{alloc(FIELDS3 + OUTPUTS, 'ims:ime,kms:kme,jms:jme')}
{alloc(FIELDS2, 'ims:ime,jms:jme')}
{alloc(FIELDS1, 'kms:kme')}
allocate(moist(ims:ime,kms:kme,jms:jme,7))
open(10,file='inputs.bin',access='stream',form='unformatted',status='old')
{read(FIELDS3 + FIELDS2 + FIELDS1)}
read(10) moist
read(10) cfn,cfn1,rdx,rdy,p_top
close(10)
config%specified=specified==1; config%nested=.false.
config%open_xs=.false.; config%open_xe=.false.
config%open_ys=.false.; config%open_ye=.false.
config%h_sca_adv_order=order; config%phi_adv_z=phi_adv
config%use_theta_m=0
p_qv=2; p_qc=3; p_qr=4; p_qi=5; p_qs=6; p_qg=7
{'; '.join(f'{name}=-987654.0' for name in OUTPUTS)}
ph_tend=0.0
call rhs_ph(ph_tend,u,v,ww,ph,ph_old,phb,w,mut,muu,muv,c1f,c2f,fnm,fnp,rdnw,cfn,cfn1,rdx,rdy, &
msfux,msfuy,msfvx,msfvx_inv,msfvy,msftx,msfty,.true.,config, &
1,nx+1,1,ny+1,1,nz+1,ims,ime,jms,jme,kms,kme,1,nx+1,1,ny+1,1,nz+1)
call phy_prep(config,mut,muu,muv,c1h,c2h,c1f,c2f,u,v,p,pb,alt,ph,phb,t,moist,7, &
rho,th_phy,th_phy_m_t0,p_phy,pi_phy,u_phy,v_phy,p8w,t_phy,t8w,z,z_at_w,dz8w,p_hyd,p_hyd_w, &
dnw,fnm,fnp,znw,p_top,1,nx+1,1,ny+1,1,nz+1,ims,ime,jms,jme,kms,kme,1,nx+1,1,ny+1,1,nz+1)
open(11,file='outputs.bin',access='stream',form='unformatted',status='replace')
{'; '.join(f'write(11) {name}' for name in OUTPUTS)}
close(11)
end program
"""
    (args.out / "run_prep.F90").write_text(text)
    (args.out / "prep-extraction.json").write_text(json.dumps({
        "wrf_tag": "v4.7.1", "wrf_commit": "f52c197ed39d12e087d02c50f412d90d418f6186",
        "source_sha256": SOURCE_SHA, "routines": {name: hashlib.sha256(body.encode()).hexdigest()
                                                    for name, body in zip(("rhs_ph", "phy_prep"), extracts)},
        "outputs": OUTPUTS,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
