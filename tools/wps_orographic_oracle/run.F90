program static_oracle
use map_utils
use interp_module
implicit none
type(proj_info) :: p
integer :: proj, i, j, unit, unit_inverse, k, ii, jj
real :: lat, lon, a(1:2,1:2,1:1), x(8), y(8), avg, four, gx, gy
real :: sums(3,3), counts(3,3), values(3,3), value, near_half(12)
integer :: list(2), opts(2)
open(newunit=unit,file='mesh-real.bin',access='stream',form='unformatted',status='replace')
open(newunit=unit_inverse,file='inverse-real.bin',access='stream',form='unformatted',status='replace')
do proj=1,3
 call map_init(p)
 if (proj==1) then
  call map_set(PROJ_LC,p,lat1=34.2,lon1=-118.2,knowni=75.5,knownj=75.5,dx=3000.,stdlon=-118.2,truelat1=30.,truelat2=60.)
 else if (proj==2) then
  call map_set(PROJ_MERC,p,lat1=12.5,lon1=140.,knowni=8.5,knownj=6.5,dx=3000.,truelat1=20.)
 else
  call map_set(PROJ_PS,p,lat1=70.,lon1=15.,knowni=8.5,knownj=6.5,dx=3000.,stdlon=0.,truelat1=60.)
 endif
 print *,proj,p%polei,p%polej,p%rebydx,p%rsw,p%cone,RAD_PER_DEG,DEG_PER_RAD
 do j=1,150
  do i=1,150
   call ij_to_latlon(p,real(i),real(j),lat,lon)
   write(unit) lat,lon
   if (mod(i-1,30)==0 .and. mod(j-1,30)==0) then
    call latlon_to_ij(p,lat,lon,gx,gy)
    write(unit_inverse) gx,gy
   endif
  enddo
 enddo
enddo
close(unit)
open(newunit=unit,file='mesh-edge-real.bin',access='stream',form='unformatted',status='replace')
do proj=1,4
 call map_init(p)
 if (proj==1) then
  call map_set(PROJ_LC,p,lat1=35.,lon1=-90.,knowni=8.5,knownj=8.5,dx=3000.,stdlon=-90.,truelat1=30.,truelat2=30.)
 else if (proj==2) then
  call map_set(PROJ_LC,p,lat1=-35.,lon1=18.,knowni=8.5,knownj=8.5,dx=3000.,stdlon=18.,truelat1=-30.,truelat2=-60.)
 else if (proj==3) then
  call map_set(PROJ_MERC,p,lat1=-15.,lon1=140.,knowni=8.5,knownj=8.5,dx=3000.,truelat1=-20.)
 else
  call map_set(PROJ_PS,p,lat1=-70.,lon1=15.,knowni=8.5,knownj=8.5,dx=3000.,stdlon=0.,truelat1=-60.)
 endif
 do j=1,16
  do i=1,16
   call ij_to_latlon(p,real(i),real(j),lat,lon)
   write(unit) lat,lon
   if ((i==1 .or. i==8 .or. i==16) .and. (j==1 .or. j==8 .or. j==16)) then
    call latlon_to_ij(p,lat,lon,gx,gy)
    write(unit_inverse) gx,gy
   endif
  enddo
 enddo
enddo
close(unit)
close(unit_inverse)
call map_init(p)
call map_set(PROJ_LC,p,lat1=35.,lon1=-90.,knowni=2.,knownj=2.,dx=100000.,stdlon=-90.,truelat1=30.,truelat2=60.)
sums=0.
counts=0.
do j=1,21
 do i=1,21
  lat=33.+real(j-1)*.2
  lon=-92.+real(i-1)*.2
  call latlon_to_ij(p,lat,lon,gx,gy)
  ii=nint(gx)
  jj=nint(gy)
  if (ii>=1 .and. ii<=3 .and. jj>=1 .and. jj<=3) then
   value=real(31+20*(i-1)+100*(j-1))
   sums(ii,jj)=sums(ii,jj)+value
   counts(ii,jj)=counts(ii,jj)+1.
  endif
 enddo
enddo
values=sums/counts*.02
open(newunit=unit,file='gcell-real.bin',access='stream',form='unformatted',status='replace')
write(unit) sums,counts,values
close(unit)
near_half=[nearest(-1.5,-1.),-1.5,nearest(-1.5,1.),nearest(-.5,-1.),-.5,nearest(-.5,1.), &
           nearest(.5,-1.),.5,nearest(.5,1.),nearest(1.5,-1.),1.5,nearest(1.5,1.)]
open(newunit=unit,file='nint-real.bin',access='stream',form='unformatted',status='replace')
do i=1,12
 write(unit) near_half(i),nint(near_half(i))
enddo
close(unit)
a(1,1,1)=31.
a(1,2,1)=245.
a(2,1,1)=80.
a(2,2,1)=301.
x=[1.25,1.,1.25,1.,0.75,1.99999988,1.25,1.75]
y=[1.75,1.75,1.,1.,1.25,1.00000012,1.25,1.25]
list=[0,0]
opts=[0,0]
open(newunit=unit,file='interp-real.bin',access='stream',form='unformatted',status='replace')
do k=1,8
 avg=four_pt_average(x(k),y(k),1,a,1,2,1,2,1,1,-99999.,list,opts,1)
 four=four_pt(x(k),y(k),1,a,1,2,1,2,1,1,-99999.,list,opts,1)
 if (avg/=-99999.) avg=avg*.02
 if (four/=-99999.) four=four*.02
 write(unit) avg,four
enddo
do i=1,2
 if (i==1) then
  a(1,1,1)=65535.
 else
  a=65535.
 endif
 do k=1,4
  avg=four_pt_average(x(k),y(k),1,a,1,2,1,2,1,1,65535.,list,opts,1)
  four=four_pt(x(k),y(k),1,a,1,2,1,2,1,1,65535.,list,opts,1)
  if (avg/=65535.) avg=avg*.02
  if (four/=65535.) four=four*.02
  write(unit) avg,four
 enddo
enddo
close(unit)
end program
