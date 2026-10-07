program run_model
  use module_fr_fire_core
  use module_fr_fire_util
  use module_fr_fire_phys
  use module_fr_fire_model
  use module_domain
  use oracle_io
  implicit none
  integer,parameter :: nx=40,ny=34,lo=-3,hx=nx+4,hy=ny+4
  integer :: i,j,step,ifun,ignitions_done,ignited_tile(1)
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
  vx=.5;vy=-.15;zs=100.;dxs=.01;dys=-.015;nf=1.;fmc=.08
  bbb=0.;beta=0.;phiw=0.;r0=0.;fgi=0.;chap=0.;ib=0.;ft=0.
  lfn=0.;hist=0.;tign=0.;ff=0.;area=0.;burnt=0.;hfx=0.;qfx=0.;ros=0.
  l0=0.;l1=0.;l2=0.;out=0.;s0=0.;s1=0.;s2=0.;s3=0.;flame=0.;front=0.
  call set_ideal_coord(5.,5.,1,nx,1,ny,lo,hx,lo,hy,lo,hx,lo,hy,coordx,coordy)
  ignition(1)%start_x=80.;ignition(1)%start_y=85.
  ignition(1)%end_x=120.;ignition(1)%end_y=85.
  ignition(1)%start_time=0.;ignition(1)%end_time=4.;ignition(1)%radius=15.;ignition(1)%ros=2.
  ignition(1)%stop_time=0.;ignition(1)%wind_red=1.;ignition(1)%wrdist=0.;ignition(1)%wrupwind=0.
  now=0.;dt=1.
  call call_model(1,now,dt)
  call call_model(2,now,dt)
  call oracle_open('model_initial')
  call put_inputs()
  call put_state()
  call oracle_close()
  do step=1,60
    now=real(step-1)
    do ifun=3,6
      call call_model(ifun,now,dt)
    enddo
    if(step==1.or.step==10.or.step==30.or.step==60)then
      write(name,'(a,i0)')'model_step',step
      call oracle_open(trim(name))
      call oracle_put('step',step)
      call oracle_put('time_start',now)
      call oracle_put('dt',dt)
      call put_state()
      call oracle_close()
    endif
  enddo
  ! Native inverted freeze control, retained as a documented divergence.
  do j=lo,hy
    do i=lo,hx
      lfn(i,j)=sqrt((real(i)-20.)**2+(real(j)-17.)**2)*5.-15.
    enddo
  enddo
  fire_const_time=20.
  do step=1,2
    if(step==1)then
      now=19.;name='model_freeze_before'
    else
      now=21.;name='model_freeze_after'
    endif
    out=lfn
    call oracle_open(trim(name))
    call oracle_put('lfn_in',lfn)
    call oracle_put('time_start',now)
    call oracle_put('const_time',fire_const_time)
    call call_model(4,now,dt)
    call oracle_put('lfn_out',out)
    call oracle_close()
  enddo
contains
  subroutine call_model(pass,time,delta)
    integer,intent(in) :: pass
    real,intent(in) :: time,delta
    call fire_model(0,pass,.false.,need_update,.false.,1,0,1, &
         1,nx,1,ny,lo,hx,lo,hy,1,nx,1,ny,1,nx,1,ny,time,delta,5.,5., &
         ignition,ignitions_done,ignited_tile,coordx,coordy,1.,1., &
         lfn,hist,.false.,l0,l1,l2,s0,s1,s2,s3,flame,front, &
         out,tign,ff,area,burnt,hfx,qfx,ros,nf,ft,fp,grid, &
         1,nx,1,ny,1,2,lo,hx,lo,hy,1,2,1,nx,1,ny,1,2)
  end subroutine call_model
  subroutine put_inputs()
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
end program run_model
