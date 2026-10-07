"""Extract the labeled native driver column-preparation statements."""
from pathlib import Path
import argparse
import hashlib
import json


def extract(source, destination):
    raw=Path(source).read_bytes()
    start=raw.index(b"            ! Half Levels",raw.index(b"SUBROUTINE firebrand_spotting_em_driver"))
    stop=raw.index(b"            DO kk = 1, ke",start)
    stop=raw.index(b"            ENDDO",stop)+len(b"            ENDDO")
    body=raw[start:stop]
    constants=raw[raw.index(b"    INTEGER, PARAMETER :: dp"):raw.index(b"    ! Generic variables for multiple use")]
    result=b"""module module_spotting_columns_oracle
use module_domain
use module_state_description
implicit none
"""+constants+b"""
contains
subroutine prepare_columns(grid,nx,ny,nz,ml,mb,u,v,w,p_hyd,th8w,rho,z_at_w,msft,p8w,th_phy)
type(domain),intent(in)::grid
integer,intent(in)::nx,ny,nz,ml,mb
real,intent(out)::u(ml:ml+nx-1,nz,mb:mb+ny-1),v(ml:ml+nx-1,nz,mb:mb+ny-1),w(ml:ml+nx-1,nz,mb:mb+ny-1)
real,intent(out)::p_hyd(ml:ml+nx-1,nz,mb:mb+ny-1),th8w(ml:ml+nx-1,nz,mb:mb+ny-1)
real,intent(out)::rho(ml:ml+nx-1,nz,mb:mb+ny-1),z_at_w(ml:ml+nx-1,nz,mb:mb+ny-1),p8w(ml:ml+nx-1,nz,mb:mb+ny-1)
real,intent(out)::msft(ml:ml+nx-1,mb:mb+ny-1)
real,intent(out)::th_phy(ml:ml+nx-1,nz,mb:mb+ny-1)
real::p_phy(ml:ml+nx-1,nz,mb:mb+ny-1)
real::qtot(ml:ml+nx-1,nz,mb:mb+ny-1),dz8w(ml:ml+nx-1,nz,mb:mb+ny-1),z(ml:ml+nx-1,nz,mb:mb+ny-1)
real::w1(ml:ml+nx-1,mb:mb+ny-1),znw(nz),rdx,rdy,dt
integer::ims,ime,jms,jme,ks,ke,kde,k_end,kk
ims=ml;ime=ml+nx-1;jms=mb;jme=mb+ny-1;ks=1;ke=nz;kde=nz
"""+body+b"\nend subroutine\nend module\n"
    Path(destination).write_bytes(result)
    receipt=dict(input_source_sha256=hashlib.sha256(raw).hexdigest(),body_sha256=hashlib.sha256(body).hexdigest(),
        wrapper_sha256=hashlib.sha256(result).hexdigest(),transformation="byte-extracted corrected full-driver preparation statements")
    Path(destination).with_suffix(".json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(receipt))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("source")
    p.add_argument("destination")
    a=p.parse_args()
    extract(a.source,a.destination)
