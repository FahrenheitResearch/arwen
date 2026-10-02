! C ABI storage adapters only. Numerical bodies are byte-preserved WRF.
module diffopt1_wrapper
use iso_c_binding
use module_diffusion_em
implicit none
contains
subroutine oracle_deform(nx,ny,nz,bx,by,u,v,w,msfu,msfv,msft,rdz,rdzw,zx,zy, &
      dn,dnw,fnm,fnp,rdx,rdy,cf1,cf2,cf3,div,d11,d22,d33,d12,d13,d23) bind(C)
    integer(c_int),value::nx,ny,nz,bx,by
    real(c_float),intent(in)::u(-2:nx+3,1:nz+1,-2:ny+3),v(-2:nx+3,1:nz+1,-2:ny+3),w(-2:nx+3,1:nz+1,-2:ny+3)
    real(c_float),intent(in)::msfu(-2:nx+3,-2:ny+3),msfv(-2:nx+3,-2:ny+3),msft(-2:nx+3,-2:ny+3)
    real(c_float),intent(in)::rdz(-2:nx+3,1:nz+1,-2:ny+3),rdzw(-2:nx+3,1:nz+1,-2:ny+3)
    real(c_float),intent(in)::zx(-2:nx+3,1:nz+1,-2:ny+3),zy(-2:nx+3,1:nz+1,-2:ny+3)
    real(c_float),intent(in)::dn(nz+1),dnw(nz+1),fnm(nz+1),fnp(nz+1)
    real(c_float),value::rdx,rdy,cf1,cf2,cf3
    real(c_float),intent(inout)::div(-2:nx+3,1:nz+1,-2:ny+3),d11(-2:nx+3,1:nz+1,-2:ny+3),d22(-2:nx+3,1:nz+1,-2:ny+3)
    real(c_float),intent(inout)::d33(-2:nx+3,1:nz+1,-2:ny+3),d12(-2:nx+3,1:nz+1,-2:ny+3),d13(-2:nx+3,1:nz+1,-2:ny+3),d23(-2:nx+3,1:nz+1,-2:ny+3)
    real::ub(nz+1),vb(nz+1),nba(-2:nx+3,1:nz+1,-2:ny+3,9)
    type(grid_config_rec_type)::cfg
    cfg%open_xs=bx/=0;cfg%open_xe=bx/=0;cfg%open_ys=by/=0;cfg%open_ye=by/=0
    cfg%periodic_x=bx==0;cfg%periodic_y=by==0;cfg%diff_opt=1;cfg%mix_full_fields=.false.
    ub=0.; vb=0.; nba=0.
    call cal_deform_and_div(cfg,u,v,w,div,d11,d22,d33,d12,d13,d23,nba,9,ub,vb, &
      msfu,msfu,msfv,msfv,msft,msft,rdx,rdy,dn,dnw,rdz,rdzw,fnm,fnp,cf1,cf2,cf3,zx,zy, &
      1,nx+1,1,ny+1,1,nz+1,-2,nx+3,-2,ny+3,1,nz+1,1,nx+1,1,ny+1,1,nz+1)
  end subroutine
subroutine oracle_metrics(nx,ny,nz,bx,by,phb,rdz,rdzw) bind(C)
integer(c_int),value::nx,ny,nz,bx,by
real(c_float),intent(in)::phb(-2:nx+3,1:nz+1,-2:ny+3)
real(c_float),intent(inout)::rdz(-2:nx+3,1:nz+1,-2:ny+3),rdzw(-2:nx+3,1:nz+1,-2:ny+3)
real::ph(-2:nx+3,1:nz+1,-2:ny+3),z(-2:nx+3,1:nz+1,-2:ny+3), &
 zx(-2:nx+3,1:nz+1,-2:ny+3),zy(-2:nx+3,1:nz+1,-2:ny+3)
type(grid_config_rec_type)::cfg
cfg%open_xs=bx/=0;cfg%open_xe=bx/=0;cfg%open_ys=by/=0;cfg%open_ye=by/=0
cfg%periodic_x=bx==0;cfg%periodic_y=by==0
ph=0.;z=0.;zx=0.;zy=0.
call compute_diff_metrics(cfg,ph,phb,z,rdz,rdzw,zx,zy,.001,.001, &
 1,nx+1,1,ny+1,1,nz+1,-2,nx+3,-2,ny+3,1,nz+1,1,nx+1,1,ny+1,1,nz+1)
end subroutine

subroutine oracle_horizontal(nx,ny,nz,bx,by,stag,perturb,field,km,mu,c1,c2,base, &
  mt,mfu,mfv,rdx,rdy,tend) bind(C)
integer(c_int),value::nx,ny,nz,bx,by,stag,perturb
real(c_float),intent(in)::field(-2:nx+3,1:nz+1,-2:ny+3),km(-2:nx+3,1:nz+1,-2:ny+3), &
 base(-2:nx+3,1:nz+1,-2:ny+3),mu(-2:nx+3,-2:ny+3), &
 mt(-2:nx+3,-2:ny+3),mfu(-2:nx+3,-2:ny+3),mfv(-2:nx+3,-2:ny+3),c1(nz+1),c2(nz+1)
real(c_float),value::rdx,rdy
real(c_float),intent(inout)::tend(-2:nx+3,1:nz+1,-2:ny+3)
real::mvinv(-2:nx+3,-2:ny+3)
type(grid_config_rec_type)::cfg
character::name
cfg%open_xs=bx/=0;cfg%open_xe=bx/=0;cfg%open_ys=by/=0;cfg%open_ye=by/=0
cfg%periodic_x=bx==0;cfg%periodic_y=by==0;cfg%diff_opt=1
name='m';if(stag==1)name='u';if(stag==2)name='v';if(stag==3)name='w'
mvinv=1./mfv
if(perturb/=0)then
call horizontal_diffusion_3dmp(name,field,tend,mu,c1,c2,cfg,base,mfu,mfu,mfv,mvinv,mfv,mt,mt, &
  0.,km,rdx,rdy,1,nx+1,1,ny+1,1,nz+1,-2,nx+3,-2,ny+3,1,nz+1,1,nx+1,1,ny+1,1,nz+1)
else
call horizontal_diffusion(name,field,tend,mu,c1,c2,cfg,mfu,mfu,mfv,mvinv,mfv,mt,mt, &
  0.,km,rdx,rdy,1,nx+1,1,ny+1,1,nz+1,-2,nx+3,-2,ny+3,1,nz+1,1,nx+1,1,ny+1,1,nz+1)
endif
end subroutine

subroutine oracle_n2(nx,ny,nz,bx,by,theta,rdz,rdzw,dn,dnw,bn2) bind(C)
integer(c_int),value::nx,ny,nz,bx,by
real(c_float),intent(in)::theta(-2:nx+3,1:nz+1,-2:ny+3),rdz(-2:nx+3,1:nz+1,-2:ny+3), &
 rdzw(-2:nx+3,1:nz+1,-2:ny+3),dn(nz+1),dnw(nz+1)
real(c_float),intent(inout)::bn2(-2:nx+3,1:nz+1,-2:ny+3)
real::moist(-2:nx+3,1:nz+1,-2:ny+3,4),t(-2:nx+3,1:nz+1,-2:ny+3), &
 p(-2:nx+3,1:nz+1,-2:ny+3),p8w(-2:nx+3,1:nz+1,-2:ny+3),t8w(-2:nx+3,1:nz+1,-2:ny+3)
type(grid_config_rec_type)::cfg
cfg%open_xs=bx/=0;cfg%open_xe=bx/=0;cfg%open_ys=by/=0;cfg%open_ye=by/=0
cfg%periodic_x=bx==0;cfg%periodic_y=by==0;cfg%diff_opt=1
moist=0.;t=300.;p=100000.;p8w=100000.;t8w=300.
call calculate_N2(cfg,bn2,moist,theta,t,p,p8w,t8w,dnw,dn,rdz,rdzw, &
 4,1.875,-1.25,.375,.false.,1,nx+1,1,ny+1,1,nz+1,-2,nx+3,-2,ny+3,1,nz+1, &
 1,nx+1,1,ny+1,1,nz+1)
end subroutine

subroutine oracle_km4(nx,ny,nz,bx,by,d11,d22,d12,mt,rdzw,zx,zy,dx,dy,cs,km,kh) bind(C)
integer(c_int),value::nx,ny,nz,bx,by
real(c_float),intent(in)::d11(-2:nx+3,1:nz+1,-2:ny+3),d22(-2:nx+3,1:nz+1,-2:ny+3), &
 d12(-2:nx+3,1:nz+1,-2:ny+3),mt(-2:nx+3,-2:ny+3),rdzw(-2:nx+3,1:nz+1,-2:ny+3), &
 zx(-2:nx+3,1:nz+1,-2:ny+3),zy(-2:nx+3,1:nz+1,-2:ny+3)
real(c_float),value::dx,dy,cs
real(c_float),intent(inout)::km(-2:nx+3,1:nz+1,-2:ny+3),kh(-2:nx+3,1:nz+1,-2:ny+3)
real::kmv(-2:nx+3,1:nz+1,-2:ny+3),khv(-2:nx+3,1:nz+1,-2:ny+3)
type(grid_config_rec_type)::cfg
cfg%open_xs=bx/=0;cfg%open_xe=bx/=0;cfg%open_ys=by/=0;cfg%open_ye=by/=0
cfg%periodic_x=bx==0;cfg%periodic_y=by==0;cfg%diff_opt=1;cfg%c_s=cs
call smag2d_km(cfg,km,kmv,kh,khv,d11,d22,d12,rdzw,dx,dy,mt,mt,zx,zy, &
  1,nx+1,1,ny+1,1,nz+1,-2,nx+3,-2,ny+3,1,nz+1,1,nx+1,1,ny+1,1,nz+1)
end subroutine

subroutine oracle_km2(nx,ny,nz,bx,by,isotropic,isfflx,tke,p8w,t8w,theta,bn2,rdz,rdzw,mt, &
  dx,dy,dt,ck,upper,kmh,khh,kmv,khv) bind(C)
integer(c_int),value::nx,ny,nz,bx,by,isotropic,isfflx
real(c_float),intent(inout)::tke(-2:nx+3,1:nz+1,-2:ny+3)
real(c_float),intent(in)::p8w(-2:nx+3,1:nz+1,-2:ny+3),t8w(-2:nx+3,1:nz+1,-2:ny+3), &
 theta(-2:nx+3,1:nz+1,-2:ny+3),bn2(-2:nx+3,1:nz+1,-2:ny+3), &
 rdz(-2:nx+3,1:nz+1,-2:ny+3),rdzw(-2:nx+3,1:nz+1,-2:ny+3),mt(-2:nx+3,-2:ny+3)
real(c_float),value::dx,dy,dt,ck,upper
real(c_float),intent(inout)::kmh(-2:nx+3,1:nz+1,-2:ny+3),khh(-2:nx+3,1:nz+1,-2:ny+3), &
 kmv(-2:nx+3,1:nz+1,-2:ny+3),khv(-2:nx+3,1:nz+1,-2:ny+3)
real::hpbl(-2:nx+3,-2:ny+3),dlk(-2:nx+3,1:nz+1,-2:ny+3),meso(-2:nx+3,1:nz+1,-2:ny+3), &
 d11(-2:nx+3,1:nz+1,-2:ny+3),d22(-2:nx+3,1:nz+1,-2:ny+3),d12(-2:nx+3,1:nz+1,-2:ny+3), &
 zx(-2:nx+3,1:nz+1,-2:ny+3),zy(-2:nx+3,1:nz+1,-2:ny+3)
type(grid_config_rec_type)::cfg
cfg%open_xs=bx/=0;cfg%open_xe=bx/=0;cfg%open_ys=by/=0;cfg%open_ye=by/=0
cfg%periodic_x=bx==0;cfg%periodic_y=by==0;cfg%diff_opt=1;cfg%km_opt=2;cfg%c_k=ck;cfg%isfflx=isfflx
hpbl=0.;dlk=0.;meso=0.;d11=0.;d22=0.;d12=0.;zx=0.;zy=0.
call tke_km(cfg,kmh,kmv,khh,khv,bn2,tke,p8w,t8w,theta,rdz,rdzw,dx,dy,dt,isotropic, &
 upper,mt,mt,hpbl,dlk,meso,d11,d22,d12,zx,zy, &
 1,nx+1,1,ny+1,1,nz+1,-2,nx+3,-2,ny+3,1,nz+1,1,nx+1,1,ny+1,1,nz+1)
end subroutine
end module
