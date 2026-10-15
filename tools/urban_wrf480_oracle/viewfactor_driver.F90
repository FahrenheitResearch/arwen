! The unchanged release helper, including its own geometry refusal.
program viewfactor_probe
  use module_sf_urban, only: VF_TREE_ANALYTICAL
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  real :: w,h,ht,rt,tau,values(10)
  integer :: k
  character(len=64) :: arg
  call get_command_argument(1,arg); read(arg,*) w
  call get_command_argument(2,arg); read(arg,*) h
  call get_command_argument(3,arg); read(arg,*) ht
  call get_command_argument(4,arg); read(arg,*) rt
  call get_command_argument(5,arg); read(arg,*) tau
  values=-999.
  call VF_TREE_ANALYTICAL(w,h,ht,rt,tau, &
       values(1),values(2),values(3),values(4),values(5),values(6),values(7), &
       values(8),values(9),values(10))
  do k=1,10
    write(*,'(I0,",",Z8.8,",",L1)') k,transfer(values(k),0),ieee_is_finite(values(k))
  end do
end program viewfactor_probe
