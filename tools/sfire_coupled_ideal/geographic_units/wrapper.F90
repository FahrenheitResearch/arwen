module native_geographic
implicit none
!NATIVE_CONSTANTS
integer,parameter::fire_print_msg=0
contains
subroutine geographic(lat_ctr,unit_fxlong,unit_fxlat)
real,intent(in)::lat_ctr
real,intent(out)::unit_fxlong,unit_fxlat
!NATIVE_UNITS
end subroutine
subroutine message(msg)
character(*),intent(in)::msg
end subroutine
!NATIVE_NEAREST
end module

program main
use native_geographic
implicit none
integer,parameter::n=32
real::latitude(n),units(n,2),coords(n,6),result(n,2,2),ex,ey
integer::i,kind,k
latitude=[-90.,-89.99999,-89.999,-80.,-66.5,-60.,-45.,-40.,-37.75,-30.,-10.,-1., &
 -0.,0.,1.,10.,30.,37.75,40.,45.,60.,66.5,80.,89.999,89.99999,90., &
 -89.5,-0.000001,0.000001,20.125,-52.625,72.875]
do i=1,n
 call geographic(latitude(i),units(i,1),units(i,2))
 coords(i,:)=[-121.005,latitude(i)+0.003,-121.,latitude(i),-120.992,latitude(i)+0.009]
 do kind=1,2
  ex=coords(i,3);ey=coords(i,4)
  if(kind==2)then
   ex=coords(i,5);ey=coords(i,6)
  endif
  call nearest(result(i,1,kind),result(i,2,kind),coords(i,1),coords(i,2),coords(i,3),coords(i,4), &
   0.,ex,ey,60.,units(i,1)*units(i,1),units(i,2)*units(i,2))
 enddo
enddo
open(newunit=k,file='latitude.bin',form='unformatted',access='stream',status='replace');write(k)latitude;close(k)
open(newunit=k,file='units.bin',form='unformatted',access='stream',status='replace');write(k)units;close(k)
open(newunit=k,file='coords.bin',form='unformatted',access='stream',status='replace');write(k)coords;close(k)
open(newunit=k,file='nearest.bin',form='unformatted',access='stream',status='replace');write(k)result;close(k)
end program
