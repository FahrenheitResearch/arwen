! Own driver invokes unmodified public WPS scalar interpolation functions.
program bitmap_oracle
 use iso_fortran_env,only:int8
 use wps_ref
 implicit none
 character(len=1024)::arg,src,maskfile,coordsfile,out
 integer::nx,ny,nl,nt,l,t
 integer::list(1)=[0],opts(1)=[0]
 real,allocatable::data(:,:,:),coords(:,:)
 integer(int8),allocatable::valid(:,:,:)
 real::value,x,y
 real,parameter::missing=-1.e30
 call get_command_argument(1,src)
 call get_command_argument(2,maskfile)
 call get_command_argument(3,coordsfile)
 call get_command_argument(4,out)
 call get_command_argument(5,arg)
 read(arg,*) nx
 call get_command_argument(6,arg)
 read(arg,*) ny
 call get_command_argument(7,arg)
 read(arg,*) nl
 call get_command_argument(8,arg)
 read(arg,*) nt
 allocate(data(nx,ny,nl),valid(nx,ny,nl),coords(2,nt))
 open(10,file=trim(src),access='stream',form='unformatted',status='old')
 read(10)data
 close(10)
 open(10,file=trim(maskfile),access='stream',form='unformatted',status='old')
 read(10)valid
 close(10)
 open(10,file=trim(coordsfile),access='stream',form='unformatted',status='old')
 read(10)coords
 close(10)
 where(valid==0)data=missing
 open(11,file=trim(out),access='stream',form='unformatted',status='replace')
 do l=1,nl
  do t=1,nt
   x=coords(1,t)+1.0
   y=coords(2,t)+1.0
   value=nearest_neighbor(x,y,l,data,1,nx,1,ny,1,nl,missing,list,opts,1)
   if(value==missing)value=four_pt(x,y,l,data,1,nx,1,ny,1,nl,missing,list,opts,1)
   if(value==missing)value=four_pt_average(x,y,l,data,1,nx,1,ny,1,nl,missing,list,opts,1)
   if(value==missing)value=0.0
   write(11)value
  end do
 end do
 close(11)
end program
