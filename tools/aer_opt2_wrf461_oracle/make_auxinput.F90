program make_auxinput
use netcdf
implicit none
integer::nc,x,y,tt,sl,ids(4),tid,i,k,rc
real::data(3,2,2)
character(len=19)::times(2)
character(len=8),parameter::names(4)=(/'AOD5502D','ANGEXP2D','AERSSA2D','AERASY2D'/)
character(len=1024)::path
call get_command_argument(1,path)
call check(nf90_create(trim(path),nf90_clobber,nc))
call check(nf90_def_dim(nc,'west_east',3,x))
call check(nf90_def_dim(nc,'south_north',2,y))
call check(nf90_def_dim(nc,'Time',2,tt))
call check(nf90_def_dim(nc,'DateStrLen',19,sl))
call check(nf90_def_var(nc,'Times',nf90_char,(/sl,tt/),tid))
do i=1,4
call check(nf90_def_var(nc,trim(names(i)),nf90_float,(/x,y,tt/),ids(i)))
enddo
call check(nf90_enddef(nc))
times=(/'2025-06-01_18:00:00','2025-06-01_18:10:00'/)
call check(nf90_put_var(nc,tid,times))
do i=1,4
do k=1,2
data(:,:,k)=0.1*i+0.01*k
enddo
call check(nf90_put_var(nc,ids(i),data))
enddo
call check(nf90_close(nc))
contains
subroutine check(code)
integer,intent(in)::code
if(code/=nf90_noerr)then
write(*,*)nf90_strerror(code)
stop 1
endif
end subroutine
end program
