program run_debug
use iso_fortran_env,only:int32
use debug_reference,only:write_array_m3
implicit none
integer(int32),parameter :: special(24)=[ &
 int(z'00000000',int32),int(z'80000000',int32),int(z'00000001',int32),int(z'80000001',int32), &
 int(z'00800000',int32),int(z'80800000',int32),int(z'006CE3EE',int32),int(z'7F7FFFFF',int32), &
 int(z'FF7FFFFF',int32),int(z'3F800000',int32),int(z'BF800000',int32),int(z'3DCCCCCD',int32), &
 int(z'4B800001',int32),int(z'CB800001',int32),int(z'7FC00000',int32),int(z'FFC00000',int32), &
 int(z'7F800000',int32),int(z'FF800000',int32),int(z'3F7FFFFF',int32),int(z'3F800001',int32), &
 int(z'3EAAAAAB',int32),int(z'BEAAAAAB',int32),int(z'00800001',int32),int(z'007FFFFF',int32)]
real :: a(-2:1,-1:0,5:7),edge(16777217:16777217,2147483646:2147483646,-16777217:-16777217)
real :: broad(0:4095,0:0,0:0)
integer :: i,j,k,n,unit
integer(int32) :: state
n=0
do k=-1,0
 do j=5,7
  do i=-2,1
   n=n+1
   a(i,k,j)=transfer(special(n),a(i,k,j))
  enddo
 enddo
enddo
call write_array_m3(-2,1,-1,0,5,7,-2,1,-1,0,5,7,a,'special',1)
open(newunit=unit,file='special.f32',access='stream',form='unformatted',status='replace')
write(unit)(((a(i,k,j),i=-2,1),j=5,7),k=-1,0)
close(unit)
edge=transfer(int(z'3F123456',int32),0.0)
call write_array_m3(16777217,16777217,2147483646,2147483646,-16777217,-16777217, &
 16777217,16777217,2147483646,2147483646,-16777217,-16777217,edge,'header_edges',31)
open(newunit=unit,file='header_edges.f32',access='stream',form='unformatted',status='replace')
write(unit)edge
close(unit)
state=int(z'4B1D8AF3',int32)
do i=0,4095
 state=ieor(state,ishft(state,13))
 state=ieor(state,ishft(state,-17))
 state=ieor(state,ishft(state,5))
 broad(i,0,0)=transfer(state,0.0)
enddo
call write_array_m3(0,4095,0,0,0,0,0,4095,0,0,0,0,broad,'broad',99999999)
open(newunit=unit,file='broad.f32',access='stream',form='unformatted',status='replace')
write(unit)broad
close(unit)
open(newunit=unit,file='cases.json',status='replace')
write(unit,'(a)')'[{"name":"special","shape":[2,3,4],"bounds":[-2,1,5,7,-1,0],"step":1},'
write(unit,'(a)')'{"name":"header_edges","shape":[1,1,1],"bounds":[16777217,16777217,-16777217,-16777217,'
write(unit,'(a)')'2147483646,2147483646],"step":31},'
write(unit,'(a)')'{"name":"broad","shape":[1,1,4096],"bounds":[0,4095,0,0,0,0],"step":99999999}]'
close(unit)
end program run_debug
