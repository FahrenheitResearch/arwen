program lake_search_oracle
use interp_module
use map_utils
implicit none
integer :: pattern, depth, point, i, j, unit
integer :: list(2), opts(2)
integer, parameter :: depths(7) = [0, 1, 2, 3, 4, 5, 8]
real :: array(11,11,1), value
type(proj_info) :: projection
real :: lat, lon, gx, gy, sums(3,3), counts(3,3), means(3,3)
integer :: ii, jj
real, parameter :: missing = -9999.
real, parameter :: x(8) = [6., 6.4, 5.5, 1., 11., .49, 6., 6.]
real, parameter :: y(8) = [6., 6.4, 6.5, 1., 11., 6., .49, 11.51]
list = [SEARCH, 0]
open(newunit=unit,file='search-real.bin',access='stream',form='unformatted',status='replace')
do pattern=0,7
 array=missing
 select case(pattern)
 case(0)
  do j=1,11
   do i=1,11
    array(i,j,1)=100*j+i
   enddo
  enddo
 case(1)
  array(7,6,1)=17.
 case(2)
  array(6,7,1)=27.
 case(3)
  array(6,8,1)=37.
 case(4)
  array(4,6,1)=47.
 case(5)
  array(5,6,1)=15.
  array(7,6,1)=17.
  array(6,5,1)=25.
  array(6,7,1)=27.
 case(7)
  array(5,5,1)=11.
  array(7,7,1)=33.
  array(9,6,1)=22.
 end select
 do depth=1,size(depths)
  opts=[depths(depth),0]
  do point=1,size(x)
   value=interp_sequence(x(point),y(point),1,array,1,11,1,11,1,1,missing,list,opts,1)
   write(unit) value
  enddo
 enddo
enddo
close(unit)
call map_init(projection)
call map_set(PROJ_LC,projection,lat1=35.,lon1=-90.,knowni=2.,knownj=2.,dx=50000.,stdlon=-90.,truelat1=30.,truelat2=60.)
sums=0.
counts=0.
do j=1,21
 do i=1,21
  lat=33.+real(j-1)*.2
  lon=-92.+real(i-1)*.2
  call latlon_to_ij(projection,lat,lon,gx,gy)
  ii=nint(gx)
  jj=nint(gy)
  if(ii>=1.and.ii<=3.and.jj>=1.and.jj<=3)then
   sums(ii,jj)=sums(ii,jj)+real(31+20*(i-1)+100*(j-1)+mod((i-1)*(j-1),17))
   counts(ii,jj)=counts(ii,jj)+1.
  endif
 enddo
enddo
means=(sums/counts)*.02
open(newunit=unit,file='lake-gcell-real.bin',access='stream',form='unformatted',status='replace')
write(unit)means
close(unit)
end program lake_search_oracle
