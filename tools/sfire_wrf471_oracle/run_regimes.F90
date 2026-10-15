program run_regimes
  use module_fr_fire_core
  use module_fr_fire_util
  use module_fr_fire_phys
  use module_fr_fire_atm
  use module_fr_fire_model
  use module_domain
  use oracle_io
  implicit none
  integer,parameter :: nx=40,ny=34,lo=-3,hx=nx+4,hy=ny+4
  integer :: i,j,k,regime,step,ifun,ignitions_done,ignited_tile(1),a,b,c,d
  type(domain) :: grid
  type(fire_params) :: fp
  type(ignition_line_type) :: ignition(1)
  logical :: need_update
  real,target :: vx(lo:hx,lo:hy),vy(lo:hx,lo:hy),zs(lo:hx,lo:hy),dxs(lo:hx,lo:hy),dys(lo:hx,lo:hy)
  real,target :: bbb(lo:hx,lo:hy),beta(lo:hx,lo:hy),phiw(lo:hx,lo:hy),r0(lo:hx,lo:hy)
  real,target :: fgi(lo:hx,lo:hy),chap(lo:hx,lo:hy),ib(lo:hx,lo:hy),ft(lo:hx,lo:hy),fmc(lo:hx,lo:hy)
  real :: nf(lo:hx,lo:hy),lfn(lo:hx,lo:hy),hist(lo:hx,lo:hy),tign(lo:hx,lo:hy)
  real :: ros(lo:hx,lo:hy),l0(lo:hx,lo:hy),l1(lo:hx,lo:hy),l2(lo:hx,lo:hy),out(lo:hx,lo:hy)
  real :: s0(lo:hx,lo:hy),s1(lo:hx,lo:hy),s2(lo:hx,lo:hy),s3(lo:hx,lo:hy)
  real :: ff(lo:hx,lo:hy),area(lo:hx,lo:hy),burnt(lo:hx,lo:hy),hfx(lo:hx,lo:hy),qfx(lo:hx,lo:hy)
  real :: flame(lo:hx,lo:hy),front(lo:hx,lo:hy),coordx(lo:hx,lo:hy),coordy(lo:hx,lo:hy)
  integer,parameter :: ax=nx/2,ay=ny/2,nz=7,nc=5
  real :: rainc(lo:hx,lo:hy),rainnc(lo:hx,lo:hy),t2(lo:hx,lo:hy),q2(lo:hx,lo:hy),psfc(lo:hx,lo:hy)
  real :: rain_old(lo:hx,lo:hy),t2_old(lo:hx,lo:hy),q2_old(lo:hx,lo:hy),psfc_old(lo:hx,lo:hy),rh(lo:hx,lo:hy)
  real :: classes(lo:hx,nc,lo:hy),fmep(lo:hx,2,lo:hy),equi(lo:hx,nc,lo:hy),lag(lo:hx,nc,lo:hy)
  real :: hg(ax,ay),qg(ax,ay),af(ax,ay),zero(ax,ay),terrain(ax,ay),mu(ax,ay)
  real :: zw(ax,nz+1,ay),dz(ax,nz+1,ay),rho(ax,nz+1,ay),c1(nz+1),c2(nz+1)
  real :: th(ax,nz+1,ay),qv(ax,nz+1,ay)
  character(len=20) :: label
  real :: now,dt
  character(len=1024) :: root
  character(len=40) :: name
  call oracle_root(root)
  fire_print_msg=0;fire_print_file=0;fire_fmc_read=0
  fire_advection=0;fire_upwind_split=0;boundary_guard=-1
  fire_upwinding=9;fire_upwinding_reinit=4;fire_lsm_reinit=.true.
  fire_lsm_reinit_iter=1;fire_lsm_band_ngp=4;fire_viscosity_ngp=4
  fire_viscosity=.4;fire_viscosity_bg=.4;fire_viscosity_band=.5;fire_lfn_ext_up=1.
  fire_const_time=-1.;fire_const_grnhfx=0.;fire_const_grnqfx=0.
  call init_fuel_cats(.true.)
  fp%vx=>vx;fp%vy=>vy;fp%zsf=>zs;fp%dzdxf=>dxs;fp%dzdyf=>dys
  fp%bbb=>bbb;fp%betafl=>beta;fp%phiwc=>phiw;fp%r_0=>r0
  fp%fgip=>fgi;fp%ischap=>chap;fp%iboros=>ib;fp%fuel_time=>ft;fp%fmc_g=>fmc
  do regime=1,7
  fire_upwinding=9;fire_upwind_split=0
  if(regime==1.or.regime>=4)then
    label='grass_flat';vx=.5;vy=-.15;dxs=0.;dys=0.;nf=1.;fmc=.08
  elseif(regime==2)then
    label='timber_slope';vx=25.;vy=3.;dxs=.6;dys=.2;nf=8.;fmc=.06
  else
    label='wet_rain';vx=2.;vy=-.5;dxs=.05;dys=-.025;nf=1.;fmc=.2
  endif
  if(regime==4)then
    label='grass_central';fire_upwinding=0
  elseif(regime==5)then
    label='grass_sethian';fire_upwinding=4
  elseif(regime==6)then
    label='grass_split_negative';fire_upwind_split=1;vx=-2.;vy=.5
  elseif(regime==7)then
    label='grass_cap';vx=25.;vy=3.;dxs=.6;dys=.2
  endif
  zs=100.
  rainc=0.;rainnc=0.;t2=285.;q2=.0095;psfc=90000.
  rain_old=0.;t2_old=0.;q2_old=0.;psfc_old=0.;rh=0.
  classes=0.;fmep=0.;equi=0.;lag=0.
  if(regime==3)then
    call moist(.true.,0.)
    call weight_moist()
  endif
  zero=0.
  do j=1,ay
    do i=1,ax
      terrain(i,j)=100.;mu(i,j)=90000.+real(i*j)*37.
      do k=1,nz+1
        zw(i,k,j)=terrain(i,j)+18.*real((k-1)*(k-1))
        dz(i,k,j)=18.*real(2*k-1);rho(i,k,j)=1.2-real(k)*.03
      enddo
    enddo
  enddo
  do k=1,nz+1
    c1(k)=1.-real(k)*.08;c2(k)=real(k)*73.
  enddo
  bbb=0.;beta=0.;phiw=0.;r0=0.;fgi=0.;chap=0.;ib=0.;ft=0.
  lfn=0.;hist=0.;tign=0.;ff=0.;area=0.;burnt=0.;hfx=0.;qfx=0.;ros=0.
  l0=0.;l1=0.;l2=0.;out=0.;s0=0.;s1=0.;s2=0.;s3=0.;flame=0.;front=0.
  call set_ideal_coord(5.,5.,1,nx,1,ny,lo,hx,lo,hy,lo,hx,lo,hy,coordx,coordy)
  zs=100.+dxs*coordx+dys*coordy
  ignition(1)%start_x=80.;ignition(1)%start_y=85.
  ignition(1)%end_x=120.;ignition(1)%end_y=85.
  ignition(1)%start_time=0.;ignition(1)%end_time=4.;ignition(1)%radius=15.;ignition(1)%ros=2.
  ignition(1)%stop_time=0.;ignition(1)%wind_red=1.;ignition(1)%wrdist=0.;ignition(1)%wrupwind=0.
  now=0.;dt=1.
  if(regime==2)dt=.25
  call call_model(1,now,dt)
  call call_model(2,now,dt)
  call oracle_open('regimes/'//trim(label)//'/initial')
  call put_inputs()
  call put_state()
  call put_moist()
  call oracle_put('terrain',terrain);call oracle_put('mu',mu)
  call oracle_put('z_at_w',zw);call oracle_put('dz8w',dz(:,1:nz,:));call oracle_put('rho',rho(:,1:nz,:))
  call oracle_put('c1h',c1(1:nz));call oracle_put('c2h',c2(1:nz))
  call oracle_close()
  do step=1,20
    now=real(step-1)*dt
    if(regime==3)then
      if(step>1)rainnc=rainnc+.01
      call moist(.false.,merge(0.,dt,step==1))
      call weight_moist()
    endif
    do ifun=3,6
      if(ifun==6)then
        ! Fuel quadrature requires the completed node state on its one-cell
        ! halo. Supply the standalone caller's physical-boundary service.
        call continue_at_boundary(1,1,fire_lfn_ext_up,lo,hx,lo,hy,1,nx,1,ny, &
            1,nx,1,ny,1,nx,1,ny,a,b,c,d,lfn)
        call continue_at_boundary(1,1,0.,lo,hx,lo,hy,1,nx,1,ny, &
            1,nx,1,ny,1,nx,1,ny,a,b,c,d,tign)
      endif
      call call_model(ifun,now,dt)
    enddo
    call sum_2d_cells(lo,hx,lo,hy,1,nx,1,ny,hfx,1,ax,1,ay,1,ax,1,ay,hg)
    call sum_2d_cells(lo,hx,lo,hy,1,nx,1,ny,qfx,1,ax,1,ay,1,ax,1,ay,qg)
    call sum_2d_cells(lo,hx,lo,hy,1,nx,1,ny,ff,1,ax,1,ay,1,ax,1,ay,af)
    hg=hg*.25;qg=qg*.25;af=af*.25
    th=0.;qv=0.
    call fire_tendency(1,ax+1,1,nz+1,1,ay+1,1,ax,1,nz+1,1,ay, &
        2,ax-1,1,nz,2,ay-1,hg,qg,zero,zero,60.,90.,20.,1,32.,1000., &
        terrain,zw,dz,mu,c1,c2,rho,th,qv)
    write(name,'(a,i0)')'step',step
    call oracle_open('regimes/'//trim(label)//'/'//trim(name))
    call oracle_put('step',step);call oracle_put('dt',dt)
    call put_state();call put_moist()
    call oracle_put('fmc_g',fmc)
    call oracle_put('grnhfx',hg);call oracle_put('grnqfx',qg);call oracle_put('avg_fuel_frac',af)
    call oracle_put('rthfrten',th(:,1:nz,:));call oracle_put('rqvfrten',qv(:,1:nz,:))
    call oracle_put('rainc_in',rainc);call oracle_put('rainnc_in',rainnc)
    call oracle_put('t2_in',t2);call oracle_put('q2_in',q2);call oracle_put('psfc_in',psfc)
    call oracle_close()
  enddo
  enddo
contains
  subroutine moist(initialize,delta)
    logical,intent(in) :: initialize
    real,intent(in) :: delta
    call advance_moisture(initialize,lo,hx,lo,hy,lo,hx,lo,hy,nc,delta,999999., &
        rainc,rainnc,t2,q2,psfc,rain_old,t2_old,q2_old,psfc_old,rh,classes,fmep,equi,lag)
  end subroutine moist
  subroutine weight_moist()
    call fuel_moisture(0,nc,1,nx,1,ny,lo,hx,lo,hy,1,nx,1,ny,1,nx,1,ny, &
        1,nx,1,ny,lo,hx,lo,hy,1,nx,1,ny,1,1,nf,classes,fmc)
  end subroutine weight_moist
  subroutine put_moist()
    call oracle_put('rain_old',rain_old);call oracle_put('t2_old',t2_old)
    call oracle_put('q2_old',q2_old);call oracle_put('psfc_old',psfc_old)
    call oracle_put('fmc_gc',classes);call oracle_put('fmep',fmep)
    call oracle_put('rh_fire',rh);call oracle_put('fmc_equi',equi);call oracle_put('fmc_lag',lag)
  end subroutine put_moist

  subroutine call_model(pass,time,delta)
    integer,intent(in) :: pass
    real,intent(in) :: time,delta
    call fire_model(0,pass,.false.,need_update,regime==3,1,-1,1, &
         1,nx,1,ny,lo,hx,lo,hy,1,nx,1,ny,1,nx,1,ny,time,delta,5.,5., &
         ignition,ignitions_done,ignited_tile,coordx,coordy,1.,1., &
         lfn,hist,.false.,l0,l1,l2,s0,s1,s2,s3,flame,front, &
         out,tign,ff,area,burnt,hfx,qfx,ros,nf,ft,fp,grid, &
         1,nx,1,ny,1,2,lo,hx,lo,hy,1,2,1,nx,1,ny,1,2)
  end subroutine call_model
  subroutine put_inputs()
    call oracle_put('core_upwinding',fire_upwinding);call oracle_put('core_split',fire_upwind_split)
    call oracle_put('bounds',[1,nx,1,ny,lo,hx,lo,hy])
    call oracle_put('dx',5.);call oracle_put('dy',5.);call oracle_put('dt',dt)
    call oracle_put('nfuel_cat',nf);call oracle_put('zsf',zs)
    call oracle_put('dzdxf',dxs);call oracle_put('dzdyf',dys)
    call oracle_put('vx',vx);call oracle_put('vy',vy)
    call oracle_put('coord_xf',coordx);call oracle_put('coord_yf',coordy)
    call oracle_put('fgip',fgi);call oracle_put('ischap',chap)
    call oracle_put('betafl',beta);call oracle_put('bbb',bbb)
    call oracle_put('fuel_time',ft);call oracle_put('phiwc',phiw)
    call oracle_put('r_0',r0);call oracle_put('iboros',ib);call oracle_put('fmc_g',fmc)
    call oracle_put('ignition',[80.,85.,120.,85.,0.,4.,15.,2.])
  end subroutine put_inputs
  subroutine put_state()
    call oracle_put('lfn',lfn);call oracle_put('tign',tign)
    call oracle_put('fuel_frac',ff);call oracle_put('fire_area',area)
    call oracle_put('lfn_out',out);call oracle_put('ros',ros)
    call oracle_put('ros_front',front);call oracle_put('flame_length',flame)
    call oracle_put('burnt_area_dt',burnt);call oracle_put('fgrnhfx',hfx);call oracle_put('fgrnqfx',qfx)
    call oracle_put('lfn_0',l0);call oracle_put('lfn_1',l1);call oracle_put('lfn_2',l2)
    call oracle_put('lfn_s0',s0);call oracle_put('lfn_s1',s1);call oracle_put('lfn_s2',s2);call oracle_put('lfn_s3',s3)
  end subroutine put_state
end program run_regimes
