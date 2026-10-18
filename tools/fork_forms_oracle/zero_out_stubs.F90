! The three WRF framework modules phys/module_microphysics_zero_out.F USEs,
! reduced to what it reads (see zero_out_driver.F90).

module module_state_description
  implicit none
  integer, parameter :: PARAM_FIRST_SCALAR = 2
  integer, parameter :: P_QV = 2
end module module_state_description

module module_configure
  implicit none
  type grid_config_rec_type
    integer :: mp_zero_out = 0
    real :: mp_zero_out_thresh = 1.e-8
  end type grid_config_rec_type
end module module_configure

module module_wrf_error
  implicit none
contains
  subroutine wrf_debug(level, message)
    integer, intent(in) :: level
    character(len=*), intent(in) :: message
  end subroutine wrf_debug
end module module_wrf_error
