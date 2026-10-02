! The called routines are unchanged extracts of WRF v4.7.1 module_em.F.
! Only this C ABI boundary assigns configuration and passes dimensions.
module rk_harness
  use iso_c_binding
  use module_configure, only: grid_config_rec_type
  use rk_reference, only: rk_addtend_dry, rk_update_scalar
  implicit none
contains
subroutine oracle_rk_dry(nx,ny,nz,step,a,b,c,maps,c1,c2,mut,mu) bind(C)
  integer(c_int),value :: nx,ny,nz,step
  real(c_float) :: a(-3:nx+4,1:nz+1,-3:ny+4,5)
  real(c_float) :: b(-3:nx+4,1:nz+1,-3:ny+4,5)
  real(c_float) :: c(-3:nx+4,1:nz+1,-3:ny+4,6)
  real(c_float) :: maps(-3:nx+4,-3:ny+4,7)
  real(c_float) :: c1(nz+1),c2(nz+1),mut(-3:nx+4,-3:ny+4)
  real(c_float) :: mu(-3:nx+4,-3:ny+4,2)
  call rk_addtend_dry(a(:,:,:,1),a(:,:,:,2),a(:,:,:,3),a(:,:,:,4),a(:,:,:,5), &
       b(:,:,:,1),b(:,:,:,2),b(:,:,:,3),b(:,:,:,4),b(:,:,:,5), &
       c(:,:,:,1),c(:,:,:,2),c(:,:,:,3),c(:,:,:,4),c(:,:,:,5), &
       mu(:,:,1),mu(:,:,2),step,c1,c2,c(:,:,:,6),mut, &
       maps(:,:,1),maps(:,:,2),maps(:,:,3),maps(:,:,4), &
       maps(:,:,5),maps(:,:,6),maps(:,:,7), &
       1,nx+1,1,ny+1,1,nz+1,-3,nx+4,-3,ny+4,1,nz+1, &
       1,nx+1,1,ny+1,1,nz+1,1,nx+1,1,ny+1,1,nz+1)
end subroutine

subroutine oracle_rk_scalar(nx,ny,nz,ns,step,specified,nested,periodic_x,spec_zone, &
                           dt,s1,s2,st,adv,decomp,maps,c1,c2,mu) bind(C)
  integer(c_int),value :: nx,ny,nz,ns,step,specified,nested,periodic_x,spec_zone
  real(c_float),value :: dt
  real(c_float) :: s1(-3:nx+4,1:nz+1,-3:ny+4,ns)
  real(c_float) :: s2(-3:nx+4,1:nz+1,-3:ny+4,ns)
  real(c_float) :: st(-3:nx+4,1:nz+1,-3:ny+4,ns)
  real(c_float) :: adv(-3:nx+4,1:nz+1,-3:ny+4)
  real(c_float) :: decomp(-3:nx+4,1:nz+1,-3:ny+4,4)
  real(c_float) :: maps(-3:nx+4,-3:ny+4,2)
  real(c_float) :: c1(nz+1),c2(nz+1),mu(-3:nx+4,-3:ny+4,3)
  type(grid_config_rec_type) :: config
  config%specified=specified/=0
  config%nested=nested/=0
  config%periodic_x=periodic_x/=0
  config%rk_ord=3
  call rk_update_scalar(1,ns,s1,s2,st,decomp(:,:,:,1),decomp(:,:,:,2),adv, &
       decomp(:,:,:,3),decomp(:,:,:,4),maps(:,:,1),maps(:,:,2),c1,c2, &
       mu(:,:,1),mu(:,:,2),mu(:,:,3),step,dt,spec_zone,config,.false., &
       1,nx+1,1,ny+1,1,nz+1,-3,nx+4,-3,ny+4,1,nz+1, &
       1,nx+1,1,ny+1,1,nz+1)
end subroutine
end module
