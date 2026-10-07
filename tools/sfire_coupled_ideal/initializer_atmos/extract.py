"""Pinned native atmospheric initializer blocks with a lightweight domain shell."""
from pathlib import Path
import argparse
import hashlib
import json
import re

PINS = {
    "module_initialize_fire.F": "c92d30ddb871bd2188fc7c37391b330eb6c9eca24a81abbfc34429fc5ab966b7",
    "module_init_utilities.F": "ea7bb9242a79fde1a818abc69a8e41afa5fdd8bc87a60aad6ffbaf813c61469e",
}


def main():
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args()
    root=Path(__file__).parent/'reference';source={}
    for name,digest in PINS.items():
        data=(root/name).read_bytes()
        if hashlib.sha256(data).hexdigest()!=digest:raise ValueError('native initializer source pin differs: '+name)
        source[name]=data.decode()
    s=source['module_initialize_fire.F'];u=source['module_init_utilities.F']
    begin=s.index('  DO j=jts,jte\n  DO i=its,ite\n    grid%phb')
    end=s.index(' END SUBROUTINE init_domain_rk',begin)
    body=s[begin:end]
    sounding=s[s.index('      subroutine get_sounding('):s.index('      end subroutine read_sounding')+len('      end subroutine read_sounding')]
    interp=u[u.index(' real function interp_0('):u.index(' END FUNCTION interp_0')+len(' END FUNCTION interp_0')]
    fields={}
    for m in re.finditer(r'grid%(\w+)\s*\(([^()]*)\)',body,re.I):fields[m[1].lower()]=m[2].count(',')+1
    for m in re.finditer(r'grid%(\w+)',body,re.I):fields.setdefault(m[1].lower(),0)
    declarations=[];allocation=[]
    for name,rank in fields.items():
        if rank:
            dimensions={1:'nz+1',2:'nx+1,ny+1',3:'nx+1,nz+1,ny+1'}[rank]
            declarations.append('real,allocatable::'+name+'('+','.join(':' for _ in range(rank))+')')
            allocation.append('allocate(grid%'+name+'('+dimensions+'));grid%'+name+'=0.')
        else:declarations.append('real::'+name+'=0.')
    # A separate correction variant computes each column's own temperature
    # immediately before its default skin assignment. Original bytes are kept.
    start=s.index('     thtmp   = grid%t_2(i,1,j)+t0',begin)
    stop=s.index('\n!     grid%tsk(I,J)',start)
    temp=s[start:stop]
    target='\n     grid%tsk(I,J)=grid%cf1*temp(1)+grid%cf2*temp(2)+grid%cf3*temp(3)'
    assert body.count(target)==1
    original=body
    corrected=body.replace(target,'\n     if(correct_skin)then\n'+temp+'     endif\n'+target)
    wrapper='''module sfire_ideal_atmos_oracle
use module_model_constants
use module_wrf_error
implicit none
integer,parameter::P_QV=1,PARAM_FIRST_SCALAR=1
type ideal_config
real::dx=50.,dy=50.,ztop=5000.,delt=0.,xr=0.,yr=0.,zr=0.,height=0.
integer::use_theta_m=1
logical::sfc_full_init=.false.
end type
type ideal_grid
DECLARATIONS
end type
contains
subroutine allocate_grid(grid,nx,ny,nz)
type(ideal_grid)::grid
integer::nx,ny,nz
ALLOCATIONS
end subroutine
logical function wrf_dm_on_monitor()
wrf_dm_on_monitor=.true.
end function
subroutine wrf_dm_bcast_real(a,n)
integer::n
real::a(n)
end subroutine
INTERP
SOUNDING
subroutine initialize_columns(grid,moist,config_flags,nx,ny,nz,correct_skin)
type(ideal_grid)::grid
type(ideal_config)::config_flags
integer::nx,ny,nz
logical::correct_skin
real::moist(nx+1,nz+1,ny+1,1)
integer,parameter::nl_max=1000
integer::i,j,k,nl_in,ids,ide,jds,jde,kds,kde,its,ite,jts,jte,kts,kte,nxc,nyc
logical::dry_sounding,sfc_init
real::zk(nl_max),p_in(nl_max),pd_in(nl_max),theta(nl_max),rho(nl_max),u(nl_max),v(nl_max),qv(nl_max)
real::p_surf,pd_surf,p_level,qvf,qvf1,qvf2,z_at_v,z_at_u,thtmp,ptmp,temp(3)
real::delt,x_rad,y_rad,z_rad,hght_pert,xrad,yrad,zrad,rad,pi
ids=1;ide=nx+1;jds=1;jde=ny+1;kds=1;kde=nz+1
its=1;ite=ide;jts=1;jte=jde;kts=1;kte=kde;nxc=nx/2;nyc=ny/2
pi=2.*asin(1.0);delt=config_flags%delt;x_rad=config_flags%xr;y_rad=config_flags%yr
z_rad=config_flags%zr;hght_pert=config_flags%height;sfc_init=config_flags%sfc_full_init
dry_sounding=.true.
call get_sounding(zk,p_in,pd_in,theta,rho,u,v,qv,dry_sounding,nl_max,nl_in)
BODY
end subroutine
end module
'''
    wrapper=wrapper.replace('DECLARATIONS','\n'.join(declarations)).replace('ALLOCATIONS','\n'.join(allocation)).replace('INTERP',interp).replace('SOUNDING',sounding)
    a.output.mkdir(parents=True,exist_ok=True)
    path=a.output/'native_atmos.F90';path.write_text(wrapper.replace('BODY',corrected),newline='\n')
    (a.output/'original_atmospheric_block.F').write_bytes(original.encode())
    receipt={'source_sha256':PINS,'native_block_lines':[s[:begin].count('\n')+1,s[:end].count('\n')],
             'native_block_sha256':hashlib.sha256(original.encode()).hexdigest(),
             'get_sounding_read_sounding_sha256':hashlib.sha256(sounding.encode()).hexdigest(),
             'interp_0_sha256':hashlib.sha256(interp.encode()).hexdigest(),
             'wrapper_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
             'scientific_original_transformations':[],
             'corrected_skin_transform':'Exact native three-level temperature expressions inserted before each TSK assignment only when corrected control selected',
             'shell':'lightweight allocated domain, single-process broadcast no-op, monitor true; original/corrected same wrapper'}
    (a.output/'extraction-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__=='__main__':main()
