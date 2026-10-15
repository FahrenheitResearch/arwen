end module
program counts
 use native_counts
 implicit none
 integer::ci,si,fi,iCell,nCells,initial_count,terrain_count,soil_count
 integer::starts(8)=[-2,-1,0,1,huge(1)-1,huge(1),-huge(1)-1,-huge(1)]
 integer(kind=I8KIND)::sums(4)=[0_I8KIND,1234567890123_I8KIND,huge(1_I8KIND)-1,-huge(1_I8KIND)]
 integer(kind=I8KIND)::pixel(1)=[7_I8KIND],initial_sum,terrain_sum,soil_sum
 real(kind=RKIND)::scales(4)=[1.0,0.01,-2.0,-0.0],scalefactor,ter(1),soilcomp(1,1)
 nCells=1
 allocate(nhs(1),landmask(1),ter_integer(1),soilcomp_int(1,1))
 landmask=1
 do ci=1,size(starts)
 do si=1,size(sums)
 do fi=1,size(scales)
  initial_count=starts(ci);initial_sum=sums(si);scalefactor=scales(fi)
  nhs=initial_count;ter_integer=initial_sum;ter=0.0
  call terrain_interp_accumulation(1,pixel)
  terrain_count=nhs(1);terrain_sum=ter_integer(1)
! TERRAIN_MEAN
  nhs=initial_count;soilcomp_int=initial_sum;soilcomp=0.0
  call soilcomp_interp_accumulation(1,pixel)
  soil_count=nhs(1);soil_sum=soilcomp_int(1,1)
! SOIL_MEAN
  write(*,'(3(z8.8,1x),3(z16.16,1x),3(z8.8,1x))')initial_count,terrain_count,soil_count, &
   initial_sum,terrain_sum,soil_sum,transfer(scalefactor,0),transfer(ter(1),0),transfer(soilcomp(1,1),0)
 enddo
 enddo
 enddo
end program
