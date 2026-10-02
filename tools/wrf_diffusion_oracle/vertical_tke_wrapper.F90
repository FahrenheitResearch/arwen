! C ABI around the unmodified WRF TKE source chain, with intermediate receipts.
module vertical_tke_oracle_wrapper
  use iso_c_binding
  use module_diffusion_em
  implicit none
contains
  subroutine oracle_tke_rhs(nx,ny,nz,bx,by,isfflx,isotropic,ck,dx,dy,dt,cd0,heat, &
      cf1,cf2,cf3,u,v,w,d11,d22,d33,d12,d13,d23,div,tke,bn2,theta,p,p8w,t8w,z, &
      rdz,rdzw,zx,zy,kmh,kmv,khv,qv,rho,msft,mu,ust,hfx,qfx, &
      dn,dnw,fnm,fnp,c1,c2,shear,buoyancy,dissipation,rhs) bind(C)
    integer(c_int), value :: nx,ny,nz,bx,by,isfflx,isotropic
    real(c_float), value :: ck,dx,dy,dt,cd0,heat,cf1,cf2,cf3
    real(c_float), intent(in) :: u(-2:nx+3,1:nz+1,-2:ny+3), &
      v(-2:nx+3,1:nz+1,-2:ny+3),w(-2:nx+3,1:nz+1,-2:ny+3), &
      d11(-2:nx+3,1:nz+1,-2:ny+3),d22(-2:nx+3,1:nz+1,-2:ny+3), &
      d33(-2:nx+3,1:nz+1,-2:ny+3),d12(-2:nx+3,1:nz+1,-2:ny+3), &
      d13(-2:nx+3,1:nz+1,-2:ny+3),d23(-2:nx+3,1:nz+1,-2:ny+3), &
      div(-2:nx+3,1:nz+1,-2:ny+3),tke(-2:nx+3,1:nz+1,-2:ny+3), &
      bn2(-2:nx+3,1:nz+1,-2:ny+3),theta(-2:nx+3,1:nz+1,-2:ny+3), &
      p(-2:nx+3,1:nz+1,-2:ny+3),p8w(-2:nx+3,1:nz+1,-2:ny+3), &
      t8w(-2:nx+3,1:nz+1,-2:ny+3),z(-2:nx+3,1:nz+1,-2:ny+3), &
      rdz(-2:nx+3,1:nz+1,-2:ny+3),rdzw(-2:nx+3,1:nz+1,-2:ny+3), &
      zx(-2:nx+3,1:nz+1,-2:ny+3),zy(-2:nx+3,1:nz+1,-2:ny+3), &
      kmh(-2:nx+3,1:nz+1,-2:ny+3),kmv(-2:nx+3,1:nz+1,-2:ny+3), &
      khv(-2:nx+3,1:nz+1,-2:ny+3),qv(-2:nx+3,1:nz+1,-2:ny+3), &
      rho(-2:nx+3,1:nz+1,-2:ny+3)
    real(c_float), intent(in) :: msft(-2:nx+3,-2:ny+3),mu(-2:nx+3,-2:ny+3), &
      ust(-2:nx+3,-2:ny+3),hfx(-2:nx+3,-2:ny+3),qfx(-2:nx+3,-2:ny+3)
    real(c_float), intent(in) :: dn(nz+1),dnw(nz+1),fnm(nz+1),fnp(nz+1),c1(nz+1),c2(nz+1)
    real(c_float), intent(out) :: shear(-2:nx+3,1:nz+1,-2:ny+3), &
      buoyancy(-2:nx+3,1:nz+1,-2:ny+3),dissipation(-2:nx+3,1:nz+1,-2:ny+3), &
      rhs(-2:nx+3,1:nz+1,-2:ny+3)
    type(grid_config_rec_type) :: cfg
    real :: l_diss(-2:nx+3,1:nz+1,-2:ny+3),nlflux(-2:nx+3,1:nz+1,-2:ny+3), &
      dlk(-2:nx+3,1:nz+1,-2:ny+3),hpbl(-2:nx+3,-2:ny+3)
    cfg%open_xs=bx/=0; cfg%open_xe=bx/=0
    cfg%open_ys=by/=0; cfg%open_ye=by/=0
    cfg%periodic_x=bx==0; cfg%periodic_y=by==0
    cfg%specified=.false.; cfg%nested=.false.
    cfg%sfs_opt=0; cfg%m_opt=0; cfg%km_opt=2
    cfg%c_k=ck; cfg%isfflx=isfflx
    cfg%tke_drag_coefficient=cd0; cfg%tke_heat_flux=heat
    l_diss=0.; nlflux=0.; dlk=0.; hpbl=0.
    shear=0.; buoyancy=0.; dissipation=0.; rhs=0.
    call tke_shear(shear,cfg,d11,d22,d33,d12,d13,d23,u,v,w,tke,ust,mu,c1,c2, &
      fnm,fnp,cf1,cf2,cf3,msft,msft,kmh,kmv,1./dx,1./dy,zx,zy,rdz,rdzw,dnw,dn, &
      1,nx+1,1,ny+1,1,nz+1,-2,nx+3,-2,ny+3,1,nz+1,1,nx+1,1,ny+1,1,nz+1)
    buoyancy=shear
    call tke_buoyancy(buoyancy,cfg,mu,c1,c2,tke,khv,bn2,theta,dt,hfx,qfx,qv,rho,nlflux, &
      1,nx+1,1,ny+1,1,nz+1,-2,nx+3,-2,ny+3,1,nz+1,1,nx+1,1,ny+1,1,nz+1)
    dissipation=buoyancy
    call tke_dissip(dissipation,cfg,mu,c1,c2,tke,bn2,theta,p8w,t8w,z,dx,dy,rdz,rdzw, &
      isotropic,msft,msft,hpbl,dlk,l_diss, &
      1,nx+1,1,ny+1,1,nz+1,-2,nx+3,-2,ny+3,1,nz+1,1,nx+1,1,ny+1,1,nz+1)
    call tke_rhs(rhs,bn2,cfg,d11,d22,d33,d12,d13,d23,u,v,w,div,tke,mu,c1,c2, &
      theta,p,p8w,t8w,z,fnm,fnp,cf1,cf2,cf3,msft,msft,kmh,kmv,khv, &
      1./dx,1./dy,dx,dy,dt,zx,zy,rdz,rdzw,dn,dnw,isotropic,hfx,qfx,qv,ust,rho, &
      l_diss,nlflux,hpbl,dlk, &
      1,nx+1,1,ny+1,1,nz+1,-2,nx+3,-2,ny+3,1,nz+1,1,nx+1,1,ny+1,1,nz+1)
  end subroutine
end module
