! C ABI adapters for the km_opt=1 column oracle (WRF v4.6.1).
! Storage adapters only: every number is computed by unmodified WRF bodies
! (calculate_km_kh -> isotropic_km, vertical_diffusion_2, horizontal_diffusion_2),
! called in module_first_rk_step_part2.F's order and with its argument lists.
module km1_wrapper
  use iso_c_binding
  use module_diffusion_em
  implicit none
  integer, parameter :: km1_nmoist = 7
contains
  subroutine km1_config(cfg,bx,by,pbl,isfflx,cd0,heat)
    type(grid_config_rec_type),intent(out)::cfg
    integer,intent(in)::bx,by,pbl,isfflx
    real,intent(in)::cd0,heat
    cfg%open_xs=bx/=0; cfg%open_xe=bx/=0
    cfg%open_ys=by/=0; cfg%open_ye=by/=0
    cfg%periodic_x=bx==0; cfg%periodic_y=by==0
    cfg%specified=.false.; cfg%nested=.false.; cfg%polar=.false.
    cfg%mix_full_fields=.true.; cfg%sfs_opt=0; cfg%m_opt=0; cfg%use_theta_m=0
    cfg%km_opt=1; cfg%diff_opt=2; cfg%bl_pbl_physics=pbl
    cfg%isfflx=isfflx; cfg%tke_drag_coefficient=cd0; cfg%tke_heat_flux=heat
    cfg%c_s=.25; cfg%c_k=.15
  end subroutine

  ! calculate_km_kh with km_opt=1: isotropic_km fills the four coefficients
  ! from the namelist constants (calculate_N2 also runs, as in WRF).
  subroutine oracle_km1_coefficients(nx,ny,nz,bx,by,khdif,kvdif,theta,t,p,p8w,t8w,moist,tke, &
      msft,rdz,rdzw,zx,zy,dn,dnw,div,d11,d22,d33,d12,d13,d23,dx,dy,dt,cf1,cf2,cf3, &
      kmh,kmv,khh,khv,bn2) bind(C)
    integer(c_int),value::nx,ny,nz,bx,by
    real(c_float),value::khdif,kvdif,dx,dy,dt,cf1,cf2,cf3
    real(c_float),intent(in)::theta(-2:nx+3,1:nz+1,-2:ny+3),t(-2:nx+3,1:nz+1,-2:ny+3),p(-2:nx+3,1:nz+1,-2:ny+3)
    real(c_float),intent(in)::p8w(-2:nx+3,1:nz+1,-2:ny+3),t8w(-2:nx+3,1:nz+1,-2:ny+3)
    real(c_float),intent(inout)::moist(-2:nx+3,1:nz+1,-2:ny+3,4),tke(-2:nx+3,1:nz+1,-2:ny+3)
    real(c_float),intent(in)::msft(-2:nx+3,-2:ny+3)
    real(c_float),intent(in)::rdz(-2:nx+3,1:nz+1,-2:ny+3),rdzw(-2:nx+3,1:nz+1,-2:ny+3)
    real(c_float),intent(in)::zx(-2:nx+3,1:nz+1,-2:ny+3),zy(-2:nx+3,1:nz+1,-2:ny+3)
    real(c_float),intent(in)::dn(nz+1),dnw(nz+1),div(-2:nx+3,1:nz+1,-2:ny+3)
    real(c_float),intent(in)::d11(-2:nx+3,1:nz+1,-2:ny+3),d22(-2:nx+3,1:nz+1,-2:ny+3),d33(-2:nx+3,1:nz+1,-2:ny+3)
    real(c_float),intent(in)::d12(-2:nx+3,1:nz+1,-2:ny+3),d13(-2:nx+3,1:nz+1,-2:ny+3),d23(-2:nx+3,1:nz+1,-2:ny+3)
    real(c_float),intent(inout)::kmh(-2:nx+3,1:nz+1,-2:ny+3),kmv(-2:nx+3,1:nz+1,-2:ny+3)
    real(c_float),intent(inout)::khh(-2:nx+3,1:nz+1,-2:ny+3),khv(-2:nx+3,1:nz+1,-2:ny+3),bn2(-2:nx+3,1:nz+1,-2:ny+3)
    real::hpbl(-2:nx+3,-2:ny+3),dlk(-2:nx+3,1:nz+1,-2:ny+3),kmv_meso(-2:nx+3,1:nz+1,-2:ny+3)
    type(grid_config_rec_type)::cfg
    call km1_config(cfg,bx,by,0,1,0.,0.)
    hpbl=0.; dlk=0.; kmv_meso=0.
    call calculate_km_kh(cfg,dt,0.,0.,0,kmh,kmv,khh,khv,bn2,khdif,kvdif,div, &
      d11,d22,d33,d12,d13,d23,tke,p8w,t8w,theta,t,p,moist,dn,dnw,dx,dy,rdz,rdzw,0, &
      4,cf1,cf2,cf3,.false.,.1,msft,msft,zx,zy,hpbl,dlk,kmv_meso, &
      1,nx+1,1,ny+1,1,nz+1,-2,nx+3,-2,ny+3,1,nz+1,1,nx+1,1,ny+1,1,nz+1)
  end subroutine

  ! The diff_opt=2 forward mixing package exactly as
  ! module_first_rk_step_part2.F:1011-1100 calls it: vertical_diffusion_2
  ! (PBL off only) and then horizontal_diffusion_2 into the same tendencies.
  subroutine oracle_km1_package(nx,ny,nz,bx,by,pbl,isfflx,cd0,heat,u,v,t2,thphy,tke,moist, &
      d11,d22,d33,d12,d13,d23,div,kmh,kmv,khh,khv,rho,rdz,rdzw,zx,zy,msfu,msfv,msft, &
      dn,dnw,fnm,fnp,rdx,rdy,cf1,cf2,cf3,hfx,qfx,ust,tu,tv,tw,tth,tmoist) bind(C)
    integer(c_int),value::nx,ny,nz,bx,by,pbl,isfflx
    real(c_float),value::cd0,heat,rdx,rdy,cf1,cf2,cf3
    real(c_float),intent(in)::u(-2:nx+3,1:nz+1,-2:ny+3),v(-2:nx+3,1:nz+1,-2:ny+3)
    real(c_float),intent(inout)::t2(-2:nx+3,1:nz+1,-2:ny+3)
    real(c_float),intent(in)::thphy(-2:nx+3,1:nz+1,-2:ny+3),tke(-2:nx+3,1:nz+1,-2:ny+3)
    real(c_float),intent(inout)::moist(-2:nx+3,1:nz+1,-2:ny+3,km1_nmoist)
    real(c_float),intent(in)::d11(-2:nx+3,1:nz+1,-2:ny+3),d22(-2:nx+3,1:nz+1,-2:ny+3),d33(-2:nx+3,1:nz+1,-2:ny+3)
    real(c_float),intent(in)::d12(-2:nx+3,1:nz+1,-2:ny+3),d13(-2:nx+3,1:nz+1,-2:ny+3),d23(-2:nx+3,1:nz+1,-2:ny+3)
    real(c_float),intent(in)::div(-2:nx+3,1:nz+1,-2:ny+3)
    real(c_float),intent(in)::kmh(-2:nx+3,1:nz+1,-2:ny+3),kmv(-2:nx+3,1:nz+1,-2:ny+3)
    real(c_float),intent(in)::khh(-2:nx+3,1:nz+1,-2:ny+3),khv(-2:nx+3,1:nz+1,-2:ny+3)
    real(c_float),intent(in)::rho(-2:nx+3,1:nz+1,-2:ny+3),rdz(-2:nx+3,1:nz+1,-2:ny+3),rdzw(-2:nx+3,1:nz+1,-2:ny+3)
    real(c_float),intent(in)::zx(-2:nx+3,1:nz+1,-2:ny+3),zy(-2:nx+3,1:nz+1,-2:ny+3)
    real(c_float),intent(in)::msfu(-2:nx+3,-2:ny+3),msfv(-2:nx+3,-2:ny+3),msft(-2:nx+3,-2:ny+3)
    real(c_float),intent(in)::dn(nz+1),dnw(nz+1),fnm(nz+1),fnp(nz+1)
    real(c_float),intent(inout)::hfx(-2:nx+3,-2:ny+3),qfx(-2:nx+3,-2:ny+3)
    real(c_float),intent(in)::ust(-2:nx+3,-2:ny+3)
    real(c_float),intent(inout)::tu(-2:nx+3,1:nz+1,-2:ny+3),tv(-2:nx+3,1:nz+1,-2:ny+3)
    real(c_float),intent(inout)::tw(-2:nx+3,1:nz+1,-2:ny+3),tth(-2:nx+3,1:nz+1,-2:ny+3)
    real(c_float),intent(inout)::tmoist(-2:nx+3,1:nz+1,-2:ny+3,km1_nmoist)
    real::tket(-2:nx+3,1:nz+1,-2:ny+3)
    real::tchem(-2:nx+3,1:nz+1,-2:ny+3,1),tscalar(-2:nx+3,1:nz+1,-2:ny+3,1),ttracer(-2:nx+3,1:nz+1,-2:ny+3,1)
    real::chem(-2:nx+3,1:nz+1,-2:ny+3,1),nba(-2:nx+3,1:nz+1,-2:ny+3,9)
    real::ub(nz+1),vb(nz+1),tb(nz+1),qb(nz+1)
    type(grid_config_rec_type)::cfg
    call km1_config(cfg,bx,by,pbl,isfflx,cd0,heat)
    tket=0.; tchem=0.; tscalar=0.; ttracer=0.; chem=0.; nba=0.
    ub=0.; vb=0.; tb=0.; qb=0.
    if (pbl==0) then
      call vertical_diffusion_2(tu,tv,tw,tth,tket,tmoist,km1_nmoist,tchem,1,tscalar,1,ttracer,1, &
        u,v,t2,ub,vb,tb,qb,tke,thphy,cfg,d13,d23,d33,nba,9,div,moist,chem,chem,chem, &
        kmv,khv,kmh,1,fnm,fnp,dn,dnw,rdz,rdzw,hfx,qfx,ust,rho, &
        1,nx+1,1,ny+1,1,nz+1,-2,nx+3,-2,ny+3,1,nz+1,1,nx+1,1,ny+1,1,nz+1)
    endif
    call horizontal_diffusion_2(tth,tu,tv,tw,tket,tmoist,km1_nmoist,tchem,1,tscalar,1,ttracer,1, &
      t2,thphy,tke,cfg,d11,d22,d12,d13,d23,nba,9,div,moist,chem,chem,chem, &
      msfu,msfu,msfv,msfv,msft,msft,kmh,kmv,khh,1,rdx,rdy,rdz,rdzw,fnm,fnp, &
      cf1,cf2,cf3,zx,zy,dn,dnw,rho,1,nx+1,1,ny+1,1,nz+1,-2,nx+3,-2,ny+3,1,nz+1, &
      1,nx+1,1,ny+1,1,nz+1)
  end subroutine
end module
