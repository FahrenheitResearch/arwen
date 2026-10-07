program run_ignition
  use module_fr_fire_core
  use module_fr_fire_util
  use oracle_io
  implicit none
  integer,parameter :: nx=24,ny=20
  type(ignition_line_type) :: line
  real :: lfn(0:nx+1,0:ny+1),old(0:nx+1,0:ny+1),tign(0:nx+1,0:ny+1)
  real :: fuel(0:nx+1,0:ny+1),area(0:nx+1,0:ny+1),x(0:nx+1,0:ny+1),y(0:nx+1,0:ny+1)
  real :: ros(0:nx+1,0:ny+1),iboros(0:nx+1,0:ny+1),flame(0:nx+1,0:ny+1),rosfl(0:nx+1,0:ny+1)
  real :: ax(20),ay(20),d(20),t(20)
  integer :: i,j,ignited
  character(len=1024) :: root
  call oracle_root(root)
  fire_print_msg=0;fire_print_file=0;boundary_guard=-1
  lfn=-999.;tign=-999.;fuel=-999.;area=-999.
  call init_no_fire(1,nx,1,ny,0,nx+1,0,ny+1,1,nx,1,ny,5.,7.,0.,fuel,area,lfn,tign)
  call oracle_open('no_fire')
  call oracle_put('fuel_frac_out',fuel);call oracle_put('fire_area_out',area)
  call oracle_put('lfn_out',lfn);call oracle_put('tign_out',tign);call oracle_close()
  line%ros=.4;line%stop_time=0.;line%wind_red=1.;line%wrdist=0.;line%wrupwind=0.
  line%start_x=30.;line%start_y=35.;line%end_x=80.;line%end_y=98.
  line%start_time=2.;line%end_time=20.;line%radius=10.
  call set_ideal_coord(5.,7.,1,nx,1,ny,0,nx+1,0,ny+1,0,nx+1,0,ny+1,x,y)
  do i=1,20
    ax(i)=real(i)*7.-13.;ay(i)=real(mod(i,7))*19.-3.
    call nearest(d(i),t(i),ax(i),ay(i),line%start_x,line%start_y,line%start_time, &
                line%end_x,line%end_y,line%end_time,1.,4.)
  enddo
  call oracle_open('nearest')
  call oracle_put('ax_in',ax);call oracle_put('ay_in',ay)
  call oracle_put('distance_out',d);call oracle_put('time_out',t)
  call put_line();call oracle_close()
  call oracle_open('ignite_line')
  call oracle_put('coord_x_in',x);call oracle_put('coord_y_in',y)
  call oracle_put('lfn_in',lfn);call oracle_put('tign_in',tign);call put_line()
  call ignite_fire(1,nx,1,ny,0,nx+1,0,ny+1,1,nx,1,ny,line,0.,12.,x,y,1.,1.,lfn,tign,ignited)
  call oracle_put('lfn_out',lfn);call oracle_put('tign_out',tign)
  call oracle_put('ignited_out',ignited);call oracle_close()
  old=lfn
  ros=-999.;iboros=-999.;area=-999.
  do j=1,ny
    do i=1,nx
      lfn(i,j)=old(i,j)-0.7*real(mod(i+j,5))
      ros(i,j)=real(mod(i*2+j,11))*0.1
      iboros(i,j)=1000.+real(i*j)*7.
      area(i,j)=real(mod(i+j,11))*0.1
    enddo
  enddo
  call oracle_open('ignition_time')
  call oracle_put('lfn_in',old);call oracle_put('lfn_after_in',lfn);call oracle_put('tign_in',tign)
  call tign_update(1,nx,1,ny,0,nx+1,0,ny+1,1,1,nx,ny,12.,1.5,old,lfn,tign)
  call oracle_put('tign_out',tign);call oracle_close()
  flame=-999.;rosfl=-999.
  call oracle_open('flame')
  call oracle_put('ros_in',ros);call oracle_put('iboros_in',iboros);call oracle_put('fire_area_in',area)
  call calc_flame_length(1,nx,1,ny,0,nx+1,0,ny+1,ros,iboros,flame,rosfl,area)
  call oracle_put('flame_length_out',flame);call oracle_put('ros_fl_out',rosfl);call oracle_close()
contains
  subroutine put_line()
    call oracle_put('line',[line%start_x,line%start_y,line%end_x,line%end_y, &
                    line%start_time,line%end_time,line%radius,line%ros])
  end subroutine put_line
end program run_ignition
