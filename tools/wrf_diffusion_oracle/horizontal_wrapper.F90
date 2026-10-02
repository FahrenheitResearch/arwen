! C ABI storage adapter; the operators below are WRF's unchanged subroutines.
module horizontal_wrapper
use iso_c_binding
use module_diffusion_em
implicit none
contains
subroutine oracle_horizontal(nx,ny,nz,bx,by,doing_tke,var,kmh,kmv,khh, &
  div,d11,d22,d12,d13,d23,tke,rho,rdz,rdzw,zx,zy,msfu,msfv,msft, &
  dn,dnw,fnm,fnp,rdx,rdy,cf1,cf2,cf3,tu,tv,tw,ts) bind(C)
integer(c_int),value::nx,ny,nz,bx,by,doing_tke
real(c_float),intent(in)::var(-2:nx+3,1:nz+1,-2:ny+3),kmh(-2:nx+3,1:nz+1,-2:ny+3), &
 kmv(-2:nx+3,1:nz+1,-2:ny+3),khh(-2:nx+3,1:nz+1,-2:ny+3),div(-2:nx+3,1:nz+1,-2:ny+3), &
 d11(-2:nx+3,1:nz+1,-2:ny+3),d22(-2:nx+3,1:nz+1,-2:ny+3),d12(-2:nx+3,1:nz+1,-2:ny+3), &
 d13(-2:nx+3,1:nz+1,-2:ny+3),d23(-2:nx+3,1:nz+1,-2:ny+3),tke(-2:nx+3,1:nz+1,-2:ny+3), &
 rho(-2:nx+3,1:nz+1,-2:ny+3),rdz(-2:nx+3,1:nz+1,-2:ny+3),rdzw(-2:nx+3,1:nz+1,-2:ny+3), &
 zx(-2:nx+3,1:nz+1,-2:ny+3),zy(-2:nx+3,1:nz+1,-2:ny+3)
real(c_float),intent(in)::msfu(-2:nx+3,-2:ny+3),msfv(-2:nx+3,-2:ny+3),msft(-2:nx+3,-2:ny+3)
real(c_float),intent(in)::dn(nz+1),dnw(nz+1),fnm(nz+1),fnp(nz+1)
real(c_float),value::rdx,rdy,cf1,cf2,cf3
real(c_float),intent(inout)::tu(-2:nx+3,1:nz+1,-2:ny+3),tv(-2:nx+3,1:nz+1,-2:ny+3), &
 tw(-2:nx+3,1:nz+1,-2:ny+3),ts(-2:nx+3,1:nz+1,-2:ny+3)
type(grid_config_rec_type)::cfg
real::nba(-2:nx+3,1:nz+1,-2:ny+3,9)
cfg%open_xs=bx/=0; cfg%open_xe=bx/=0
cfg%open_ys=by/=0; cfg%open_ye=by/=0
cfg%periodic_x=bx==0; cfg%periodic_y=by==0
cfg%sfs_opt=0; cfg%m_opt=0
nba=0.
call horizontal_diffusion_u_2(tu,cfg,d11,d12,div,nba,9,tke,msfu,msfu,kmh,rdx,rdy, &
 fnm,fnp,dnw,zx,zy,rdzw,rho,1,nx+1,1,ny+1,1,nz+1,-2,nx+3,-2,ny+3,1,nz+1,1,nx+1,1,ny+1,1,nz+1)
call horizontal_diffusion_v_2(tv,cfg,d12,d22,div,nba,9,tke,msfv,msfv,kmh,rdx,rdy, &
 fnm,fnp,dnw,zx,zy,rdzw,rho,1,nx+1,1,ny+1,1,nz+1,-2,nx+3,-2,ny+3,1,nz+1,1,nx+1,1,ny+1,1,nz+1)
call horizontal_diffusion_w_2(tw,cfg,d13,d23,div,nba,9,tke,msft,msft,kmv,rdx,rdy, &
 fnm,fnp,dn,zx,zy,rdz,rho,1,nx+1,1,ny+1,1,nz+1,-2,nx+3,-2,ny+3,1,nz+1,1,nx+1,1,ny+1,1,nz+1)
call horizontal_diffusion_s(ts,cfg,var,msft,msft,msfu,msfu,msfv,msfv,khh,rdx,rdy, &
 fnm,fnp,cf1,cf2,cf3,zx,zy,rdz,rdzw,dnw,dn,rho,doing_tke/=0, &
 1,nx+1,1,ny+1,1,nz+1,-2,nx+3,-2,ny+3,1,nz+1,1,nx+1,1,ny+1,1,nz+1)
end subroutine
end module
