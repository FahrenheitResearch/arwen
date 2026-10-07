program run_physics
  use module_fr_fire_phys
  use module_fr_fire_util
  use oracle_io
  implicit none
  integer, parameter :: nx=54, ny=3, mx=8, my=3, nc=5
  integer :: i,j,k,s
  integer :: cats(nx)
  type(fire_params) :: fp
  real,target :: vx(nx,ny),vy(nx,ny),zsf(nx,ny),dzdx(nx,ny),dzdy(nx,ny)
  real,target :: bbb(nx,ny),betafl(nx,ny),phiwc(nx,ny),r0(nx,ny),fgip(nx,ny)
  real,target :: chap(nx,ny),iboros(nx,ny),ft(nx,ny),fmc(nx,ny)
  real :: nf(nx,ny),burn(nx,ny),hfx(nx,ny),qfx(nx,ny),px(nx,ny),py(nx,ny)
  real :: rb(nx,ny),rw(nx,ny),rs(nx,ny)
  real :: rainc(mx,my),rainnc(mx,my),t2(mx,my),q2(mx,my),psfc(mx,my)
  real :: rain_old(mx,my),t2_old(mx,my),q2_old(mx,my),psfc_old(mx,my),rh(mx,my)
  real :: fmcgc(mx,nc,my),fmep(mx,2,my),equi(mx,nc,my),lag(mx,nc,my),dt
  character(len=1024) :: root
  character(len=40) :: name
  call oracle_root(root)
  fire_print_msg=0; fire_print_file=0; fire_fmc_read=0
  call init_fuel_cats(.true.)
  cats=[(i,i=1,14),(i,i=101,109),(i,i=121,124),(i,i=141,149), &
        (i,i=161,165),(i,i=181,189),(i,i=201,204)]
  fp%vx=>vx;fp%vy=>vy;fp%zsf=>zsf;fp%dzdxf=>dzdx;fp%dzdyf=>dzdy
  fp%bbb=>bbb;fp%betafl=>betafl;fp%phiwc=>phiwc;fp%r_0=>r0
  fp%fgip=>fgip;fp%ischap=>chap;fp%iboros=>iboros;fp%fuel_time=>ft;fp%fmc_g=>fmc
  do j=1,ny
    do i=1,nx
      nf(i,j)=real(cats(i)); fmc(i,j)=0.02+0.03*real(j)
      vx(i,j)=real(mod(i,5)-2)*0.1;vy(i,j)=real(mod(i,7)-3)*0.1
      zsf(i,j)=100.+real(i*2+j);dzdx(i,j)=real(mod(i,3)-1)*0.05
      dzdy(i,j)=real(mod(i,4)-2)*0.02
      px(i,j)=0.6;py(i,j)=0.8
      if(mod(i,2)==0)px(i,j)=-0.6
      if(mod(i,3)==0)py(i,j)=-0.8
      burn(i,j)=real(mod(i+j,11))*0.01
    enddo
  enddo
  call oracle_open('params')
  call oracle_put('nfuel_cat_in',nf);call oracle_put('fmc_g_in',fmc)
  call set_fire_params(1,nx,1,ny,1,nx,1,ny,1,nx,1,ny,5.,5.,1,nf,ft,fp)
  call oracle_put('fgip_out',fgip);call oracle_put('ischap_out',chap)
  call oracle_put('bbb_out',bbb);call oracle_put('betafl_out',betafl)
  call oracle_put('phiwc_out',phiwc);call oracle_put('r_0_out',r0)
  call oracle_put('fuel_time_out',ft);call oracle_put('iboros_out',iboros)
  call oracle_put('fmc_g_out',fmc);call oracle_close()
  do s=0,1
    write(name,'(a,i0)')'ros_advection',s
    call oracle_open(trim(name));fire_advection=s
    call oracle_put('vx_in',vx);call oracle_put('vy_in',vy)
    call oracle_put('dzdxf_in',dzdx);call oracle_put('dzdyf_in',dzdy)
    call oracle_put('propx_in',px);call oracle_put('propy_in',py)
    do j=1,ny
      do i=1,nx
        rb(i,j)=0.;rw(i,j)=0.;rs(i,j)=0.
        call fire_ros(rb(i,j),rw(i,j),rs(i,j),px(i,j),py(i,j),i,j,fp)
      enddo
    enddo
    call oracle_put('ros_base_out',rb);call oracle_put('ros_wind_out',rw)
    call oracle_put('ros_slope_out',rs);call oracle_close()
  enddo
  call oracle_open('flux')
  call oracle_put('dt',15.);call oracle_put('fuel_frac_burnt_in',burn)
  call heat_fluxes(15.,fp,1,nx,1,ny,1,nx,1,ny,1,nx,1,ny,fgip,burn,hfx,qfx)
  call oracle_put('grnhft_out',hfx);call oracle_put('grnqft_out',qfx);call oracle_close()
  vx=30.;vy=0.;dzdx=1.;dzdy=0.;px=1.;py=0.
  call oracle_open('original_ros_cap_defect')
  call oracle_put('vx_in',vx);call oracle_put('vy_in',vy)
  call oracle_put('dzdxf_in',dzdx);call oracle_put('dzdyf_in',dzdy)
  call oracle_put('propx_in',px);call oracle_put('propy_in',py)
  fire_advection=0
  do j=1,ny
    do i=1,nx
      rb(i,j)=0.;rw(i,j)=0.;rs(i,j)=0.
      call fire_ros(rb(i,j),rw(i,j),rs(i,j),px(i,j),py(i,j),i,j,fp)
    enddo
  enddo
  call oracle_put('ros_base_out',rb);call oracle_put('ros_wind_out',rw)
  call oracle_put('ros_slope_out',rs);call oracle_close()
  nf=1.;fmc=.4
  call oracle_open('original_moisture_extinction_defect')
  call oracle_put('nfuel_cat_in',nf);call oracle_put('fmc_g_in',fmc)
  call set_fire_params(1,nx,1,ny,1,nx,1,ny,1,nx,1,ny,5.,5.,1,nf,ft,fp)
  call oracle_put('r_0_out',r0);call oracle_put('iboros_out',iboros);call oracle_close()
  rainc=0.;rainnc=0.;rain_old=0.;t2_old=0.;q2_old=0.;psfc_old=0.
  do j=1,my
    do i=1,mx
      t2(i,j)=273.15+real(i*3+j*2)
      q2(i,j)=real(i-1)*0.002
      psfc(i,j)=80000.+real(i*1000+j*100)
      fmep(i,1,j)=0.01*real(mod(i,3)-1);fmep(i,2,j)=0.02*real(mod(i,4)-1)
      do k=1,nc
        fmcgc(i,k,j)=real(i+k)*0.025
      enddo
    enddo
  enddo
  do s=0,2
    dt=0.
    if(s==1)dt=30.
    if(s==2)dt=1800.
    do j=1,my
      do i=1,mx
        if(s>0)then
          rainnc(i,j)=rainnc(i,j)+dt*real(mod(i,4))*0.002
          t2(i,j)=t2(i,j)+0.25*real(mod(i,3)-1)
          q2(i,j)=max(q2(i,j)+0.0001*real(mod(i,3)-1),0.)
        endif
      enddo
    enddo
    write(name,'(a,i0)')'moisture_s',s
    call oracle_open(trim(name))
    call oracle_put('dt',dt);call oracle_put('fmep_decay_tlag',48.)
    call oracle_put('rainc_in',rainc);call oracle_put('rainnc_in',rainnc)
    call oracle_put('t2_in',t2);call oracle_put('q2_in',q2);call oracle_put('psfc_in',psfc)
    call oracle_put('rain_old_in',rain_old);call oracle_put('t2_old_in',t2_old)
    call oracle_put('q2_old_in',q2_old);call oracle_put('psfc_old_in',psfc_old)
    call oracle_put('fmc_gc_in',fmcgc);call oracle_put('fmep_in',fmep)
    call advance_moisture(s==0,1,mx,1,my,1,mx,1,my,nc,dt,48.,rainc,rainnc,t2,q2,psfc, &
                         rain_old,t2_old,q2_old,psfc_old,rh,fmcgc,fmep,equi,lag)
    call oracle_put('rain_old_out',rain_old);call oracle_put('t2_old_out',t2_old)
    call oracle_put('q2_old_out',q2_old);call oracle_put('psfc_old_out',psfc_old)
    call oracle_put('fmc_gc_out',fmcgc);call oracle_put('fmep_out',fmep)
    call oracle_put('rh_fire_out',rh);call oracle_put('fmc_equi_out',equi)
    call oracle_put('fmc_lag_out',lag);call oracle_close()
  enddo
end program run_physics
