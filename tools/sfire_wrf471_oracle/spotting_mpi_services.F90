! Native MPI grid services with explicit patch ownership; no particle math.
module module_domain
implicit none
type domain
 integer::ips=1,ipe=4,jps=1,jpe=4
end type
contains
subroutine get_ijk_from_grid(grid,ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme,ips,ipe,jps,jpe,kps,kpe)
type(domain),intent(in)::grid
integer,intent(out)::ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme,ips,ipe,jps,jpe,kps,kpe
ids=1;ide=13;jds=1;jde=13;kds=1;kde=2
ims=grid%ips-4;ime=grid%ipe+4;jms=grid%jps-4;jme=grid%jpe+4;kms=1;kme=2
ips=grid%ips;ipe=grid%ipe;jps=grid%jps;jpe=grid%jpe;kps=1;kpe=2
end subroutine
end module
module module_configure
implicit none
type grid_config_rec_type
 integer::unused=0
end type
end module
module module_dm
implicit none
integer::ntasks_x=3,ntasks_y=3,mytask_x=0,mytask_y=0
end module
subroutine wrf_debug(level,message)
integer,intent(in)::level
character(len=*),intent(in)::message
end subroutine
subroutine wrf_error_fatal(message)
character(len=*),intent(in)::message
print *,message
error stop 'native MPI helper fatal condition'
end subroutine
