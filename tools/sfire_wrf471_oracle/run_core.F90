program run_core
  use module_fr_fire_core
  use module_fr_fire_util
  use module_fr_fire_phys
  use module_domain
  use oracle_io
  implicit none
  integer,parameter :: nx=40,ny=34,lo=-3,hx=nx+4,hy=ny+4
  integer :: i,j,m
  integer :: modes(7)=[1,2,3,5,6,7,8]
  type(domain) :: grid
  type(fire_params) :: fp
  real,target :: vx(lo:hx,lo:hy),vy(lo:hx,lo:hy),zs(lo:hx,lo:hy),dxs(lo:hx,lo:hy),dys(lo:hx,lo:hy)
  real,target :: bbb(lo:hx,lo:hy),beta(lo:hx,lo:hy),phiw(lo:hx,lo:hy),r0(lo:hx,lo:hy)
  real,target :: fgi(lo:hx,lo:hy),chap(lo:hx,lo:hy),ib(lo:hx,lo:hy),ft(lo:hx,lo:hy),fmc(lo:hx,lo:hy)
  real :: nf(lo:hx,lo:hy),initial(lo:hx,lo:hy),lfn(lo:hx,lo:hy),tign(lo:hx,lo:hy)
  real :: tend(lo:hx,lo:hy),ros(lo:hx,lo:hy),l0(lo:hx,lo:hy),l1(lo:hx,lo:hy),l2(lo:hx,lo:hy),out(lo:hx,lo:hy)
  real :: s0(lo:hx,lo:hy),s1(lo:hx,lo:hy),s2(lo:hx,lo:hy),s3(lo:hx,lo:hy)
  real :: ff(lo:hx,lo:hy),area(lo:hx,lo:hy),tbound
  character(len=1024) :: root
  character(len=40) :: name
  call oracle_root(root)
  fire_print_msg=0;fire_print_file=0;fire_fmc_read=0
  fire_advection=0;fire_upwind_split=0;boundary_guard=-1
  call init_fuel_cats(.true.)
  fp%vx=>vx;fp%vy=>vy;fp%zsf=>zs;fp%dzdxf=>dxs;fp%dzdyf=>dys
  fp%bbb=>bbb;fp%betafl=>beta;fp%phiwc=>phiw;fp%r_0=>r0
  fp%fgip=>fgi;fp%ischap=>chap;fp%iboros=>ib;fp%fuel_time=>ft;fp%fmc_g=>fmc
  vx=0.5;vy=-0.15;zs=100.;dxs=0.01;dys=-0.015;nf=1.;fmc=.08
  call set_fire_params(1,nx,1,ny,lo,hx,lo,hy,lo,hx,lo,hy,5.,5.,1,nf,ft,fp)
  do j=lo,hy
    do i=lo,hx
      initial(i,j)=sqrt((real(i)-20.)**2+(real(j)-17.)**2)*5.-35.
    enddo
  enddo
  do m=1,8
    fire_upwinding=9
    if(m<8)fire_upwinding=modes(m)
    write(name,'(a,i0)')'tend_mode',fire_upwinding
    call oracle_open(trim(name));call put_params()
    lfn=initial;tend=-987.;ros=-986.;tign=3.
    call oracle_put('lfn_in',lfn)
    call tend_ls(0,1,nx,1,ny,1,nx,1,ny,1,nx,1,ny,lo,hx,lo,hy,10.,.2,5.,5.,lfn,tbound,tend,ros,fp)
    call oracle_put('lfn_out',lfn);call oracle_put('tend_out',tend)
    call oracle_put('ros_out',ros);call oracle_put('tbound_out',tbound)
    call oracle_close()
  enddo
  fire_upwinding=9
  call oracle_open('rk3');call put_params()
  lfn=initial;tign=3.;l0=initial;l1=initial;l2=initial;out=-985.;ros=-986.
  call oracle_put('lfn_in',lfn);call oracle_put('tign_in',tign)
  call prop_ls_rk3(0,1,nx,1,ny,lo,hx,lo,hy,1,nx,1,ny,1,nx,1,ny, &
                  10.,.2,5.,5.,tbound,lfn,l0,l1,l2,out,tign,ros,fp,grid, &
                  1,nx,1,ny,1,2,lo,hx,lo,hy,1,2,1,nx,1,ny,1,2)
  call oracle_put('lfn_out',out);call oracle_put('lfn_0_out',l0)
  call oracle_put('lfn_1_out',l1);call oracle_put('lfn_2_out',l2)
  call oracle_put('lfn_modified_in',lfn);call oracle_put('tign_out',tign)
  call oracle_put('ros_out',ros);call oracle_put('tbound_out',tbound);call oracle_close()
  do m=1,4
    write(name,'(a,i0)')'reinit_mode',m
    call oracle_open(trim(name));call put_params()
    fire_upwinding_reinit=m
    lfn=initial;l2=initial*1.7;out=-985.;tign=3.;s0=initial;s1=initial;s2=initial;s3=initial
    call oracle_put('lfn_in',lfn);call oracle_put('lfn_2_in',l2)
    call reinit_ls_rk3(0,1,nx,1,ny,lo,hx,lo,hy,1,nx,1,ny,1,nx,1,ny, &
                      10.,.2,5.,5.,lfn,l2,s0,s1,s2,s3,out,tign,grid, &
                      1,nx,1,ny,1,2,lo,hx,lo,hy,1,2,1,nx,1,ny,1,2)
    call oracle_put('lfn_out',out);call oracle_put('lfn_s0_out',s0)
    call oracle_put('lfn_s1_out',s1);call oracle_put('lfn_s2_out',s2)
    call oracle_put('lfn_s3_out',s3);call oracle_close()
  enddo
  lfn=initial;tign=3.;ff=-989.;area=-988.
  call oracle_open('fuel_grid');call put_params()
  call oracle_put('lfn_in',lfn);call oracle_put('tign_in',tign)
  call fuel_left(lo,hx,lo,hy,1,nx,1,ny,lo,hx,lo,hy,lfn,tign,ft,10.,ff,area)
  call oracle_put('fuel_frac_out',ff);call oracle_put('fire_area_out',area);call oracle_close()
contains
  subroutine put_params()
    call oracle_put('bounds',[1,nx,1,ny,lo,hx,lo,hy])
    call oracle_put('dx',5.);call oracle_put('dy',5.);call oracle_put('ts',10.);call oracle_put('dt',.2)
    call oracle_put('vx_in',vx);call oracle_put('vy_in',vy)
    call oracle_put('dzdxf_in',dxs);call oracle_put('dzdyf_in',dys)
    call oracle_put('bbb_in',bbb);call oracle_put('betafl_in',beta)
    call oracle_put('phiwc_in',phiw);call oracle_put('r_0_in',r0)
    call oracle_put('ischap_in',chap);call oracle_put('fuel_time_in',ft)
  end subroutine put_params
end program run_core
