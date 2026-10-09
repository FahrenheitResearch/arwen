! C ABI for the km_opt=2 column oracle.  All numerical work is done by the
! unmodified WRF v4.6.1 routines: calculate_km_kh (calculate_N2 + tke_km),
! with the namelist switches tke_km reads for its seed rule passed in.
module km2_oracle_wrapper
  use iso_c_binding
  use module_diffusion_em
  use deformation_wrappers, only: deformation_config
  implicit none
contains
  subroutine oracle_km2(nx,ny,nz,bx,by,isotropic,isfflx,theta,t,p,p8w,t8w,moist,tke, &
      msft,rdz,rdzw,zx,zy,dn,dnw,div,d11,d22,d33,d12,d13,d23,dx,dy,dt,cf1,cf2,cf3, &
      ck,upper,cd0,heat,kmh,kmv,khh,khv,bn2) bind(C)
    integer(c_int),value::nx,ny,nz,bx,by,isotropic,isfflx
    real(c_float),intent(in)::theta(-2:nx+3,1:nz+1,-2:ny+3),t(-2:nx+3,1:nz+1,-2:ny+3),p(-2:nx+3,1:nz+1,-2:ny+3)
    real(c_float),intent(in)::p8w(-2:nx+3,1:nz+1,-2:ny+3),t8w(-2:nx+3,1:nz+1,-2:ny+3)
    real(c_float),intent(inout)::moist(-2:nx+3,1:nz+1,-2:ny+3,4),tke(-2:nx+3,1:nz+1,-2:ny+3)
    real(c_float),intent(in)::msft(-2:nx+3,-2:ny+3)
    real(c_float),intent(in)::rdz(-2:nx+3,1:nz+1,-2:ny+3),rdzw(-2:nx+3,1:nz+1,-2:ny+3)
    real(c_float),intent(in)::zx(-2:nx+3,1:nz+1,-2:ny+3),zy(-2:nx+3,1:nz+1,-2:ny+3)
    real(c_float),intent(in)::dn(nz+1),dnw(nz+1)
    real(c_float),intent(in)::div(-2:nx+3,1:nz+1,-2:ny+3),d11(-2:nx+3,1:nz+1,-2:ny+3),d22(-2:nx+3,1:nz+1,-2:ny+3)
    real(c_float),intent(in)::d33(-2:nx+3,1:nz+1,-2:ny+3),d12(-2:nx+3,1:nz+1,-2:ny+3)
    real(c_float),intent(in)::d13(-2:nx+3,1:nz+1,-2:ny+3),d23(-2:nx+3,1:nz+1,-2:ny+3)
    real(c_float),value::dx,dy,dt,cf1,cf2,cf3,ck,upper,cd0,heat
    real(c_float),intent(inout)::kmh(-2:nx+3,1:nz+1,-2:ny+3),kmv(-2:nx+3,1:nz+1,-2:ny+3)
    real(c_float),intent(inout)::khh(-2:nx+3,1:nz+1,-2:ny+3),khv(-2:nx+3,1:nz+1,-2:ny+3)
    real(c_float),intent(inout)::bn2(-2:nx+3,1:nz+1,-2:ny+3)
    real::hpbl(-2:nx+3,-2:ny+3),dlk(-2:nx+3,1:nz+1,-2:ny+3),kmv_meso(-2:nx+3,1:nz+1,-2:ny+3)
    type(grid_config_rec_type)::cfg
    call deformation_config(cfg,bx,by,2,isotropic)
    cfg%c_k=ck
    cfg%isfflx=isfflx
    cfg%tke_drag_coefficient=cd0
    cfg%tke_heat_flux=heat
    hpbl=0.; dlk=0.; kmv_meso=0.
    call calculate_km_kh(cfg,dt,0.,0.,0,kmh,kmv,khh,khv,bn2,0.,0.,div, &
      d11,d22,d33,d12,d13,d23,tke,p8w,t8w,theta,t,p,moist,dn,dnw,dx,dy,rdz,rdzw,isotropic, &
      4,cf1,cf2,cf3,.false.,upper,msft,msft,zx,zy,hpbl,dlk,kmv_meso, &
      1,nx+1,1,ny+1,1,nz+1,-2,nx+3,-2,ny+3,1,nz+1,1,nx+1,1,ny+1,1,nz+1)
  end subroutine
end module
