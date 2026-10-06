! C ABI adapters for the mix_full_fields probe: the pinned deformation
! wrappers, unchanged, plus oracle_deform_mix, which takes the flag and
! the two 1-D base-state wind profiles WRF's .false. branch subtracts
! (dyn_em/module_diffusion_em.F:842-860, :1017-1035).  All numerical
! work is performed by unmodified WRF routines.
module mix_full_fields_wrappers
  use iso_c_binding
  use module_diffusion_em
  implicit none
contains
  subroutine oracle_bc(nx,ny,nz,bx,by,stagger,field) bind(C)
    integer(c_int),value::nx,ny,nz,bx,by,stagger
    real(c_float),intent(inout)::field(-2:nx+3,1:nz+1,-2:ny+3)
    real::large(-3:nx+5,1:nz+1,-3:ny+5)
    type(grid_config_rec_type)::cfg
    character::stag
    call deformation_config(cfg,bx,by,4,0)
    large=0.; large(-2:nx+3,:,-2:ny+3)=field
    stag='t'
    if(stagger==1) stag='u'
    if(stagger==2) stag='v'
    if(stagger==3) stag='w'
    if(stagger==4) stag='d'
    if(stagger==5) stag='e'
    if(stagger==6) stag='f'
    call set_physical_bc3d(large,stag,cfg, &
      1,nx+1,1,ny+1,1,nz+1,-3,nx+5,-3,ny+5,1,nz+1, &
      1,nx+1,1,ny+1,1,nz+1,1,nx+1,1,ny+1,1,nz+1)
    field=large(-2:nx+3,:,-2:ny+3)
  end subroutine

  subroutine oracle_metrics(nx,ny,nz,bx,by,ph,phb,rdx,rdy,z,rdz,rdzw,zx,zy) bind(C)
    integer(c_int),value::nx,ny,nz,bx,by
    real(c_float),intent(in)::ph(-2:nx+3,1:nz+1,-2:ny+3),phb(-2:nx+3,1:nz+1,-2:ny+3)
    real(c_float),value::rdx,rdy
    real(c_float),intent(inout)::z(-2:nx+3,1:nz+1,-2:ny+3),rdz(-2:nx+3,1:nz+1,-2:ny+3),rdzw(-2:nx+3,1:nz+1,-2:ny+3)
    real(c_float),intent(inout)::zx(-2:nx+3,1:nz+1,-2:ny+3),zy(-2:nx+3,1:nz+1,-2:ny+3)
    type(grid_config_rec_type)::cfg
    call deformation_config(cfg,bx,by,4,0)
    ! The first call computes distributed-halo metric values using the same
    ! WRF arithmetic. The second applies the true physical-domain conventions.
    call compute_diff_metrics(cfg,ph,phb,z,rdz,rdzw,zx,zy,rdx,rdy, &
      -1,nx+3,-1,ny+3,1,nz+1,-2,nx+3,-2,ny+3,1,nz+1,0,nx+1,0,ny+1,1,nz+1)
    if(bx/=0) then
      zx(1,:,1:ny+1)=0.; zx(nx+1,:,1:ny+1)=0.
    endif
    if(by/=0) then
      zy(1:nx+1,:,1)=0.; zy(1:nx+1,:,ny+1)=0.
    endif
    call compute_diff_metrics(cfg,ph,phb,z,rdz,rdzw,zx,zy,rdx,rdy, &
      1,nx+1,1,ny+1,1,nz+1,-2,nx+3,-2,ny+3,1,nz+1,1,nx+1,1,ny+1,1,nz+1)
    call oracle_bc(nx,ny,nz,bx,by,0,z)
    call oracle_bc(nx,ny,nz,bx,by,0,rdz)
    call oracle_bc(nx,ny,nz,bx,by,0,rdzw)
    call oracle_bc(nx,ny,nz,bx,by,5,zx)
    call oracle_bc(nx,ny,nz,bx,by,6,zy)
  end subroutine

  subroutine oracle_phy(nx,ny,nz,bx,by,u,v,p,alt,ph,phb,thp,moist,mut, &
      c1h,c2h,c1f,c2f,dnw,fzm,fzp,znw,ptop,rho,theta,temp,p8w,t8w,z,zw) bind(C)
    integer(c_int),value::nx,ny,nz,bx,by
    real(c_float),intent(in)::u(-2:nx+3,1:nz+1,-2:ny+3),v(-2:nx+3,1:nz+1,-2:ny+3),p(-2:nx+3,1:nz+1,-2:ny+3),alt(-2:nx+3,1:nz+1,-2:ny+3)
    real(c_float),intent(in)::ph(-2:nx+3,1:nz+1,-2:ny+3),phb(-2:nx+3,1:nz+1,-2:ny+3),thp(-2:nx+3,1:nz+1,-2:ny+3)
    real(c_float),intent(in)::moist(-2:nx+3,1:nz+1,-2:ny+3,4),mut(-2:nx+3,-2:ny+3)
    real(c_float),intent(in)::c1h(nz+1),c2h(nz+1),c1f(nz+1),c2f(nz+1),dnw(nz+1),fzm(nz+1),fzp(nz+1),znw(nz+1)
    real(c_float),value::ptop
    real(c_float),intent(inout)::rho(-2:nx+3,1:nz+1,-2:ny+3),theta(-2:nx+3,1:nz+1,-2:ny+3),temp(-2:nx+3,1:nz+1,-2:ny+3)
    real(c_float),intent(inout)::p8w(-2:nx+3,1:nz+1,-2:ny+3),t8w(-2:nx+3,1:nz+1,-2:ny+3),z(-2:nx+3,1:nz+1,-2:ny+3),zw(-2:nx+3,1:nz+1,-2:ny+3)
    real::pb(-2:nx+3,1:nz+1,-2:ny+3),thm(-2:nx+3,1:nz+1,-2:ny+3),pp(-2:nx+3,1:nz+1,-2:ny+3),pii(-2:nx+3,1:nz+1,-2:ny+3)
    real::up(-2:nx+3,1:nz+1,-2:ny+3),vp(-2:nx+3,1:nz+1,-2:ny+3),dz(-2:nx+3,1:nz+1,-2:ny+3),hyd(-2:nx+3,1:nz+1,-2:ny+3),hydw(-2:nx+3,1:nz+1,-2:ny+3)
    type(grid_config_rec_type)::cfg
    call deformation_config(cfg,bx,by,4,0)
    pb=0.; thm=0.; pp=0.; pii=0.; up=0.; vp=0.; dz=0.; hyd=0.; hydw=0.
    call phy_prep(cfg,mut,mut,mut,c1h,c2h,c1f,c2f,u,v,p,pb,alt,ph,phb,thp,moist,4, &
      rho,theta,thm,pp,pii,up,vp,p8w,temp,t8w,z,zw,dz,hyd,hydw,dnw,fzm,fzp,znw,ptop, &
      1,nx+1,1,ny+1,1,nz+1,-2,nx+3,-2,ny+3,1,nz+1,1,nx+1,1,ny+1,1,nz+1)
    call oracle_bc(nx,ny,nz,bx,by,0,rho)
    call oracle_bc(nx,ny,nz,bx,by,0,theta)
    call oracle_bc(nx,ny,nz,bx,by,0,temp)
    call oracle_bc(nx,ny,nz,bx,by,3,p8w)
    call oracle_bc(nx,ny,nz,bx,by,3,t8w)
    call oracle_bc(nx,ny,nz,bx,by,0,z)
    call oracle_bc(nx,ny,nz,bx,by,3,zw)
  end subroutine
  subroutine deformation_config(cfg,bx,by,km,isotropic)
    type(grid_config_rec_type),intent(out)::cfg
    integer,intent(in)::bx,by,km,isotropic
    cfg%open_xs=bx/=0; cfg%open_xe=bx/=0
    cfg%open_ys=by/=0; cfg%open_ye=by/=0
    cfg%periodic_x=bx==0; cfg%periodic_y=by==0
    cfg%specified=.false.; cfg%nested=.false.; cfg%polar=.false.
    cfg%mix_full_fields=.true.; cfg%sfs_opt=0
    cfg%km_opt=km; cfg%diff_opt=2; cfg%bl_pbl_physics=0
    cfg%isfflx=0; cfg%tke_drag_coefficient=0.; cfg%tke_heat_flux=0.
    cfg%c_s=.25; cfg%c_k=.15
  end subroutine

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
    call deformation_config(cfg,bx,by,4,0)
    ub=0.; vb=0.; nba=0.
    call cal_deform_and_div(cfg,u,v,w,div,d11,d22,d33,d12,d13,d23,nba,9,ub,vb, &
      msfu,msfu,msfv,msfv,msft,msft,rdx,rdy,dn,dnw,rdz,rdzw,fnm,fnp,cf1,cf2,cf3,zx,zy, &
      1,nx+1,1,ny+1,1,nz+1,-2,nx+3,-2,ny+3,1,nz+1,1,nx+1,1,ny+1,1,nz+1)
  end subroutine

  subroutine oracle_km(nx,ny,nz,bx,by,km_opt,isotropic,theta,t,p,p8w,t8w,moist,tke, &
      msft,rdz,rdzw,zx,zy,dn,dnw,div,d11,d22,d33,d12,d13,d23,dx,dy,dt,cf1,cf2,cf3, &
      cs,ck,upper,seed_on,kmh,kmv,khh,khv,bn2) bind(C)
    integer(c_int),value::nx,ny,nz,bx,by,km_opt,isotropic,seed_on
    real(c_float),intent(in)::theta(-2:nx+3,1:nz+1,-2:ny+3),t(-2:nx+3,1:nz+1,-2:ny+3),p(-2:nx+3,1:nz+1,-2:ny+3)
    real(c_float),intent(in)::p8w(-2:nx+3,1:nz+1,-2:ny+3),t8w(-2:nx+3,1:nz+1,-2:ny+3)
    real(c_float),intent(inout)::moist(-2:nx+3,1:nz+1,-2:ny+3,4),tke(-2:nx+3,1:nz+1,-2:ny+3)
    real(c_float),intent(in)::msft(-2:nx+3,-2:ny+3)
    real(c_float),intent(in)::rdz(-2:nx+3,1:nz+1,-2:ny+3),rdzw(-2:nx+3,1:nz+1,-2:ny+3),zx(-2:nx+3,1:nz+1,-2:ny+3),zy(-2:nx+3,1:nz+1,-2:ny+3)
    real(c_float),intent(in)::dn(nz+1),dnw(nz+1),div(-2:nx+3,1:nz+1,-2:ny+3),d11(-2:nx+3,1:nz+1,-2:ny+3),d22(-2:nx+3,1:nz+1,-2:ny+3)
    real(c_float),intent(in)::d33(-2:nx+3,1:nz+1,-2:ny+3),d12(-2:nx+3,1:nz+1,-2:ny+3),d13(-2:nx+3,1:nz+1,-2:ny+3),d23(-2:nx+3,1:nz+1,-2:ny+3)
    real(c_float),value::dx,dy,dt,cf1,cf2,cf3,cs,ck,upper
    real(c_float),intent(inout)::kmh(-2:nx+3,1:nz+1,-2:ny+3),kmv(-2:nx+3,1:nz+1,-2:ny+3),khh(-2:nx+3,1:nz+1,-2:ny+3),khv(-2:nx+3,1:nz+1,-2:ny+3),bn2(-2:nx+3,1:nz+1,-2:ny+3)
    real::hpbl(-2:nx+3,-2:ny+3),dlk(-2:nx+3,1:nz+1,-2:ny+3),kmv_meso(-2:nx+3,1:nz+1,-2:ny+3)
    type(grid_config_rec_type)::cfg
    call deformation_config(cfg,bx,by,km_opt,isotropic)
    cfg%c_s=cs; cfg%c_k=ck
    if(seed_on==0) cfg%isfflx=1
    hpbl=0.; dlk=0.; kmv_meso=0.
    call calculate_km_kh(cfg,dt,0.,0.,0,kmh,kmv,khh,khv,bn2,0.,0.,div, &
      d11,d22,d33,d12,d13,d23,tke,p8w,t8w,theta,t,p,moist,dn,dnw,dx,dy,rdz,rdzw,isotropic, &
      4,cf1,cf2,cf3,.false.,upper,msft,msft,zx,zy,hpbl,dlk,kmv_meso, &
      1,nx+1,1,ny+1,1,nz+1,-2,nx+3,-2,ny+3,1,nz+1,1,nx+1,1,ny+1,1,nz+1)
  end subroutine
  subroutine oracle_deform_mix(nx,ny,nz,bx,by,mix,ub,vb,u,v,w,msfu,msfv,msft,rdz,rdzw,zx,zy, &
      dn,dnw,fnm,fnp,rdx,rdy,cf1,cf2,cf3,div,d11,d22,d33,d12,d13,d23) bind(C)
    integer(c_int),value::nx,ny,nz,bx,by,mix
    real(c_float),intent(in)::ub(nz+1),vb(nz+1)
    real(c_float),intent(in)::u(-2:nx+3,1:nz+1,-2:ny+3),v(-2:nx+3,1:nz+1,-2:ny+3),w(-2:nx+3,1:nz+1,-2:ny+3)
    real(c_float),intent(in)::msfu(-2:nx+3,-2:ny+3),msfv(-2:nx+3,-2:ny+3),msft(-2:nx+3,-2:ny+3)
    real(c_float),intent(in)::rdz(-2:nx+3,1:nz+1,-2:ny+3),rdzw(-2:nx+3,1:nz+1,-2:ny+3)
    real(c_float),intent(in)::zx(-2:nx+3,1:nz+1,-2:ny+3),zy(-2:nx+3,1:nz+1,-2:ny+3)
    real(c_float),intent(in)::dn(nz+1),dnw(nz+1),fnm(nz+1),fnp(nz+1)
    real(c_float),value::rdx,rdy,cf1,cf2,cf3
    real(c_float),intent(inout)::div(-2:nx+3,1:nz+1,-2:ny+3),d11(-2:nx+3,1:nz+1,-2:ny+3),d22(-2:nx+3,1:nz+1,-2:ny+3)
    real(c_float),intent(inout)::d33(-2:nx+3,1:nz+1,-2:ny+3),d12(-2:nx+3,1:nz+1,-2:ny+3),d13(-2:nx+3,1:nz+1,-2:ny+3),d23(-2:nx+3,1:nz+1,-2:ny+3)
    real::nba(-2:nx+3,1:nz+1,-2:ny+3,9)
    type(grid_config_rec_type)::cfg
    call deformation_config(cfg,bx,by,4,0)
    cfg%mix_full_fields=(mix/=0)
    nba=0.
    call cal_deform_and_div(cfg,u,v,w,div,d11,d22,d33,d12,d13,d23,nba,9,ub,vb, &
      msfu,msfu,msfv,msfv,msft,msft,rdx,rdy,dn,dnw,rdz,rdzw,fnm,fnp,cf1,cf2,cf3,zx,zy, &
      1,nx+1,1,ny+1,1,nz+1,-2,nx+3,-2,ny+3,1,nz+1,1,nx+1,1,ny+1,1,nz+1)
  end subroutine
end module
