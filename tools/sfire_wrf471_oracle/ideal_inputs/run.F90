program run_input_reference
use input_reference
implicit none
character(len=512)::mode,source,prefix,text
real,allocatable::a(:,:)
real::ps,ts,qvs,h(1000),th(1000),qv(1000),u(1000),v(1000)
integer::ni,nj,i,j,unit,nl
call get_command_argument(1,mode)
call get_command_argument(2,source)
call get_command_argument(3,prefix)
if(trim(mode)=='array')then
 call get_command_argument(4,text);read(text,*)ni
 call get_command_argument(5,text);read(text,*)nj
 allocate(a(ni,nj));a=-999.
 call read_array_2d_real(trim(source),a,1,ni,1,nj,1,ni,1,nj)
 open(newunit=unit,file=trim(prefix)//'.f32',form='unformatted',access='stream',status='replace',convert='little_endian')
 do j=1,nj
  do i=1,ni
   write(unit)a(i,j)
  enddo
 enddo
 close(unit)
else if(trim(mode)=='sounding')then
 call read_sounding(ps,ts,qvs,h,th,qv,u,v,1000,nl,.false.)
 call output('psurf',[ps]);call output('theta_surface',[ts]);call output('qv_surface',[qvs])
 call output('height',h(:nl));call output('theta',th(:nl));call output('qv',qv(:nl))
 call output('u',u(:nl));call output('v',v(:nl))
 open(newunit=unit,file=trim(prefix)//'.levels',status='replace');write(unit,*)nl;close(unit)
else
 error stop 'unknown native reader mode'
endif
contains
subroutine output(name,data)
character(len=*),intent(in)::name
real,intent(in)::data(:)
integer::number
open(newunit=number,file=trim(prefix)//'.'//name//'.f32',form='unformatted',access='stream',status='replace',convert='little_endian')
write(number)data
close(number)
end subroutine
end program
