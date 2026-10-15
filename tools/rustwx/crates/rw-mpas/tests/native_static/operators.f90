! Calls the complete pristine atm_advection module on a real MPAS grid.
program operators
 use netcdf
 use atm_advection
 implicit none
 integer::file,nc,ne,nv,me,u,padded
 integer,allocatable::rows(:,:)
 real::radius
 type(mpas_pool_type)::mesh
 character(1024)::filename,output,padding,geometry
 logical::on_sphere
 call get_command_argument(1,filename)
 call get_command_argument(2,output)
 call check(nf90_open(trim(filename),nf90_nowrite,file))
 call dimension('nCells',nc)
 call dimension('nEdges',ne)
 call dimension('nVertices',nv)
 call dimension('maxEdges',me)
 radius=6371229.0
 allocate(xCell(nc),yCell(nc),zCell(nc),xVertex(nv),yVertex(nv),zVertex(nv),angleEdge(ne),dcEdge(ne))
 allocate(gradient_x(me,nc),gradient_y(me,nc),deriv(15,2,ne),adv(25,nc))
 allocate(counts(nc),coc(me,nc),eoc(me,nc),voc(me,nc),voe(2,ne),coe(2,ne))
 call read_real('xCell',xCell);call read_real('yCell',yCell);call read_real('zCell',zCell)
 call read_real('xVertex',xVertex);call read_real('yVertex',yVertex);call read_real('zVertex',zVertex)
 call read_real('angleEdge',angleEdge);call read_real('dcEdge',dcEdge)
 xCell=xCell*radius;yCell=yCell*radius;zCell=zCell*radius
 xVertex=xVertex*radius;yVertex=yVertex*radius;zVertex=zVertex*radius
 dcEdge=dcEdge*radius
 call read_int1('nEdgesOnCell',counts)
 call read_int2('cellsOnCell',coc);call read_int2('edgesOnCell',eoc);call read_int2('verticesOnCell',voc)
 call read_int2('verticesOnEdge',voe);call read_int2('cellsOnEdge',coe)
 call check(nf90_close(file))
 ! Row capacity is independent of the active nEdgesOnCell stencil.
 call get_command_argument(3,padding)
 if(len_trim(padding)>0)then
  read(padding,*)padded
  if(padded<me)stop 3
  allocate(rows(padded,nc));rows=0;rows(1:me,:)=coc;call move_alloc(rows,coc)
  allocate(rows(padded,nc));rows=0;rows(1:me,:)=eoc;call move_alloc(rows,eoc)
  allocate(rows(padded,nc));rows=0;rows(1:me,:)=voc;call move_alloc(rows,voc)
  deallocate(gradient_x,gradient_y)
  allocate(gradient_x(padded,nc),gradient_y(padded,nc))
  me=padded
 endif
 call get_command_argument(4,geometry)
 on_sphere=trim(geometry)/="plane"
 call atm_initialize_advection_rk(mesh,nc,ne,me,on_sphere,radius)
 call atm_initialize_deformation_weights(mesh,nc,on_sphere,radius)
 open(newunit=u,file=trim(output),access='stream',form='unformatted',status='replace')
 write(u)gradient_x,gradient_y,deriv
 close(u)
contains
 subroutine check(status)
 integer::status
 if(status/=nf90_noerr)then
  write(*,*)nf90_strerror(status)
  stop 2
 endif
 end subroutine
 subroutine dimension(name,length)
 character(*)::name
 integer::length,id
 call check(nf90_inq_dimid(file,name,id))
 call check(nf90_inquire_dimension(file,id,len=length))
 end subroutine
 subroutine read_real(name,array)
 character(*)::name
 real::array(:)
 integer::id
 call check(nf90_inq_varid(file,name,id));call check(nf90_get_var(file,id,array))
 end subroutine
 subroutine read_int1(name,array)
 character(*)::name
 integer::array(:),id
 call check(nf90_inq_varid(file,name,id));call check(nf90_get_var(file,id,array))
 end subroutine
 subroutine read_int2(name,array)
 character(*)::name
 integer::array(:,:),id
 call check(nf90_inq_varid(file,name,id));call check(nf90_get_var(file,id,array))
 end subroutine
end program
