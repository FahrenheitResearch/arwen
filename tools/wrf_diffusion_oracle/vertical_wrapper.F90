! C ABI only. All diffusion calculations remain in the extracted WRF routines.
module vertical_oracle_wrapper
  use iso_c_binding
  use module_diffusion_em
  implicit none
contains
  subroutine oracle_vertical(nx,ny,nz,bx,by,doing_tke,defor13,defor23,defor33, &
      div,tke,xkmv,xkhv,var,rho,rdz,rdzw,fnm,fnp,dn,dnw,tu,tv,tw,ts) bind(C)
    integer(c_int), value :: nx,ny,nz,bx,by,doing_tke
    real(c_float), intent(in) :: defor13(-2:nx+3,1:nz+1,-2:ny+3), &
      defor23(-2:nx+3,1:nz+1,-2:ny+3),defor33(-2:nx+3,1:nz+1,-2:ny+3), &
      div(-2:nx+3,1:nz+1,-2:ny+3),tke(-2:nx+3,1:nz+1,-2:ny+3), &
      xkmv(-2:nx+3,1:nz+1,-2:ny+3),xkhv(-2:nx+3,1:nz+1,-2:ny+3), &
      var(-2:nx+3,1:nz+1,-2:ny+3),rho(-2:nx+3,1:nz+1,-2:ny+3), &
      rdz(-2:nx+3,1:nz+1,-2:ny+3),rdzw(-2:nx+3,1:nz+1,-2:ny+3)
    real(c_float), intent(in) :: fnm(nz+1),fnp(nz+1),dn(nz+1),dnw(nz+1)
    real(c_float), intent(inout) :: tu(-2:nx+3,1:nz+1,-2:ny+3), &
      tv(-2:nx+3,1:nz+1,-2:ny+3),tw(-2:nx+3,1:nz+1,-2:ny+3), &
      ts(-2:nx+3,1:nz+1,-2:ny+3)
    type(grid_config_rec_type) :: cfg
    real :: nba(-2:nx+3,1:nz+1,-2:ny+3,9)
    cfg%open_xs=bx/=0; cfg%open_xe=bx/=0
    cfg%open_ys=by/=0; cfg%open_ye=by/=0
    cfg%periodic_x=bx==0; cfg%periodic_y=by==0
    cfg%specified=.false.; cfg%nested=.false.
    cfg%sfs_opt=0; cfg%m_opt=0
    nba=0.
    call vertical_diffusion_u_2(tu,cfg,defor13,xkmv,nba,9,dnw,rdzw,fnm,fnp,rho, &
      1,nx+1,1,ny+1,1,nz+1,-2,nx+3,-2,ny+3,1,nz+1,1,nx+1,1,ny+1,1,nz+1)
    call vertical_diffusion_v_2(tv,cfg,defor23,xkmv,nba,9,dnw,rdzw,fnm,fnp,rho, &
      1,nx+1,1,ny+1,1,nz+1,-2,nx+3,-2,ny+3,1,nz+1,1,nx+1,1,ny+1,1,nz+1)
    call vertical_diffusion_w_2(tw,cfg,defor33,tke,nba,9,div,xkmv,dn,rdz,fnm,fnp,rho, &
      1,nx+1,1,ny+1,1,nz+1,-2,nx+3,-2,ny+3,1,nz+1,1,nx+1,1,ny+1,1,nz+1)
    call vertical_diffusion_s(ts,cfg,var,xkhv,dn,dnw,rdz,rdzw,fnm,fnp,rho,doing_tke/=0, &
      1,nx+1,1,ny+1,1,nz+1,-2,nx+3,-2,ny+3,1,nz+1,1,nx+1,1,ny+1,1,nz+1)
  end subroutine
end module
