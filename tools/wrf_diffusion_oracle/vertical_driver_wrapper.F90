! C ABI for the actual WRF vertical_diffusion_2 driver and surface switches.
module vertical_driver_oracle_wrapper
  use iso_c_binding
  use module_diffusion_em
  implicit none
contains
  subroutine oracle_vertical_driver(nx,ny,nz,bx,by,km,isfflx,cd0,heat,u,v,thp,theta, &
      tke,moist,d13,d23,d33,div,kmh,kmv,khv,rho,rdz,rdzw,fnm,fnp,dn,dnw,hfx,qfx,ust, &
      tu,tv,tw,tt,te,tmoist) bind(C)
    integer(c_int), value :: nx,ny,nz,bx,by,km,isfflx
    real(c_float), value :: cd0,heat
    real(c_float), intent(in) :: u(-2:nx+3,1:nz+1,-2:ny+3),v(-2:nx+3,1:nz+1,-2:ny+3), &
      theta(-2:nx+3,1:nz+1,-2:ny+3),tke(-2:nx+3,1:nz+1,-2:ny+3), &
      d13(-2:nx+3,1:nz+1,-2:ny+3),d23(-2:nx+3,1:nz+1,-2:ny+3), &
      d33(-2:nx+3,1:nz+1,-2:ny+3),div(-2:nx+3,1:nz+1,-2:ny+3), &
      kmh(-2:nx+3,1:nz+1,-2:ny+3),kmv(-2:nx+3,1:nz+1,-2:ny+3), &
      khv(-2:nx+3,1:nz+1,-2:ny+3),rho(-2:nx+3,1:nz+1,-2:ny+3), &
      rdz(-2:nx+3,1:nz+1,-2:ny+3),rdzw(-2:nx+3,1:nz+1,-2:ny+3)
    real(c_float), intent(inout) :: thp(-2:nx+3,1:nz+1,-2:ny+3), &
      moist(-2:nx+3,1:nz+1,-2:ny+3,4),hfx(-2:nx+3,-2:ny+3),qfx(-2:nx+3,-2:ny+3)
    real(c_float), intent(in) :: ust(-2:nx+3,-2:ny+3),fnm(nz+1),fnp(nz+1),dn(nz+1),dnw(nz+1)
    real(c_float), intent(out) :: tu(-2:nx+3,1:nz+1,-2:ny+3),tv(-2:nx+3,1:nz+1,-2:ny+3), &
      tw(-2:nx+3,1:nz+1,-2:ny+3),tt(-2:nx+3,1:nz+1,-2:ny+3), &
      te(-2:nx+3,1:nz+1,-2:ny+3),tmoist(-2:nx+3,1:nz+1,-2:ny+3,4)
    type(grid_config_rec_type) :: cfg
    real :: ub(nz+1),vb(nz+1),tb(nz+1),qb(nz+1), &
      nba(-2:nx+3,1:nz+1,-2:ny+3,9),dummy(-2:nx+3,1:nz+1,-2:ny+3,1), &
      chem_t(-2:nx+3,1:nz+1,-2:ny+3,1),scalar_t(-2:nx+3,1:nz+1,-2:ny+3,1), &
      tracer_t(-2:nx+3,1:nz+1,-2:ny+3,1)
    cfg%open_xs=bx/=0; cfg%open_xe=bx/=0
    cfg%open_ys=by/=0; cfg%open_ye=by/=0
    cfg%periodic_x=bx==0; cfg%periodic_y=by==0
    cfg%specified=.false.; cfg%nested=.false.
    cfg%sfs_opt=0; cfg%m_opt=0; cfg%km_opt=km
    cfg%isfflx=isfflx; cfg%use_theta_m=0; cfg%mix_full_fields=.true.
    cfg%tke_drag_coefficient=cd0; cfg%tke_heat_flux=heat
    ub=0.; vb=0.; tb=0.; qb=0.; nba=0.; dummy=0.
    chem_t=0.; scalar_t=0.; tracer_t=0.
    tu=0.; tv=0.; tw=0.; tt=0.; te=0.; tmoist=0.
    call vertical_diffusion_2(tu,tv,tw,tt,te,tmoist,4,chem_t,1,scalar_t,1,tracer_t,1, &
      u,v,thp,ub,vb,tb,qb,tke,theta,cfg,d13,d23,d33,nba,9,div,moist,dummy,dummy,dummy, &
      kmv,khv,kmh,km,fnm,fnp,dn,dnw,rdz,rdzw,hfx,qfx,ust,rho, &
      1,nx+1,1,ny+1,1,nz+1,-2,nx+3,-2,ny+3,1,nz+1,1,nx+1,1,ny+1,1,nz+1)
  end subroutine
end module
