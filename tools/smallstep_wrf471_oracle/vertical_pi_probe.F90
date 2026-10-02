program vertical_pi_probe
  use iso_fortran_env, only: int32
  use module_model_constants, only: g
  implicit none
  real :: pi
  pi = 4.*atan(1.)
  write(*,'(Z8.8,1X,Z8.8)') transfer(pi,0_int32), transfer(0.5*pi,0_int32)
end program vertical_pi_probe
