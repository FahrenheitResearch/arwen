! Storage adapter only. The actual WRF outer driver calls its unchanged leaves.
module horizontal_driver_wrapper
use iso_c_binding
use module_diffusion_em
implicit none
contains
subroutine oracle_horizontal_driver(nx,ny,nz,bx,by,km,thp,theta,tke,moist,kmh,kmv,khh, &
 div,d11,d22,d12,d13,d23,rho,rdz,rdzw,zx,zy,msfu,msfv,msft,dn,dnw,fnm,fnp, &
 rdx,rdy,cf1,cf2,cf3,tu,tv,tw,tth,ttke,tmoist,tchem,tscalar,ttracer,nba) bind(C)
integer(c_int),value::nx,ny,nz,bx,by,km
real(c_float),intent(in)::thp(-2:nx+3,1:nz+1,-2:ny+3),theta(-2:nx+3,1:nz+1,-2:ny+3), &
 tke(-2:nx+3,1:nz+1,-2:ny+3),moist(-2:nx+3,1:nz+1,-2:ny+3,4), &
 kmh(-2:nx+3,1:nz+1,-2:ny+3),kmv(-2:nx+3,1:nz+1,-2:ny+3),khh(-2:nx+3,1:nz+1,-2:ny+3), &
 div(-2:nx+3,1:nz+1,-2:ny+3),d11(-2:nx+3,1:nz+1,-2:ny+3),d22(-2:nx+3,1:nz+1,-2:ny+3), &
 d12(-2:nx+3,1:nz+1,-2:ny+3),d13(-2:nx+3,1:nz+1,-2:ny+3),d23(-2:nx+3,1:nz+1,-2:ny+3), &
 rho(-2:nx+3,1:nz+1,-2:ny+3),rdz(-2:nx+3,1:nz+1,-2:ny+3),rdzw(-2:nx+3,1:nz+1,-2:ny+3), &
 zx(-2:nx+3,1:nz+1,-2:ny+3),zy(-2:nx+3,1:nz+1,-2:ny+3)
real(c_float),intent(in)::msfu(-2:nx+3,-2:ny+3),msfv(-2:nx+3,-2:ny+3),msft(-2:nx+3,-2:ny+3)
real(c_float),intent(in)::dn(nz+1),dnw(nz+1),fnm(nz+1),fnp(nz+1)
real(c_float),value::rdx,rdy,cf1,cf2,cf3
real(c_float),intent(inout)::tu(-2:nx+3,1:nz+1,-2:ny+3),tv(-2:nx+3,1:nz+1,-2:ny+3), &
 tw(-2:nx+3,1:nz+1,-2:ny+3),tth(-2:nx+3,1:nz+1,-2:ny+3),ttke(-2:nx+3,1:nz+1,-2:ny+3), &
 tmoist(-2:nx+3,1:nz+1,-2:ny+3,4),tchem(-2:nx+3,1:nz+1,-2:ny+3,1), &
 tscalar(-2:nx+3,1:nz+1,-2:ny+3,1),ttracer(-2:nx+3,1:nz+1,-2:ny+3,1), &
 nba(-2:nx+3,1:nz+1,-2:ny+3,9)
type(grid_config_rec_type)::cfg
real::unused(-2:nx+3,1:nz+1,-2:ny+3,1)
cfg%open_xs=bx/=0;cfg%open_xe=bx/=0;cfg%open_ys=by/=0;cfg%open_ye=by/=0
cfg%periodic_x=bx==0;cfg%periodic_y=by==0;cfg%km_opt=km
cfg%sfs_opt=0;cfg%m_opt=0;cfg%mix_full_fields=.true.
unused=0.
call horizontal_diffusion_2(tth,tu,tv,tw,ttke,tmoist,4,tchem,1,tscalar,1,ttracer,1, &
 thp,theta,tke,cfg,d11,d22,d12,d13,d23,nba,9,div,moist,unused,unused,unused, &
 msfu,msfu,msfv,msfv,msft,msft,kmh,kmv,khh,km,rdx,rdy,rdz,rdzw,fnm,fnp, &
 cf1,cf2,cf3,zx,zy,dn,dnw,rho,1,nx+1,1,ny+1,1,nz+1,-2,nx+3,-2,ny+3,1,nz+1, &
 1,nx+1,1,ny+1,1,nz+1)
end subroutine
end module
