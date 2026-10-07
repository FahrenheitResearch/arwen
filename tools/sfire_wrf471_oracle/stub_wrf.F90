! Standalone single-rank WRF services. No fire mathematics is implemented here.
module module_domain
  implicit none
  type domain
    integer :: unused = 0
  end type domain
end module module_domain

module module_configure
  implicit none
  type grid_config_rec_type
    integer :: unused = 0
  end type grid_config_rec_type
end module module_configure

module module_state_description
  implicit none
  integer, parameter :: num_tracer = 1, p_fire_smoke = 1
end module module_state_description

subroutine wrf_abort()
  error stop 'WRF fatal error reached in SFIRE oracle'
end subroutine wrf_abort

subroutine wrf_debug(level, msg)
  integer, intent(in) :: level
  character(len=*), intent(in) :: msg
end subroutine wrf_debug

logical function wrf_dm_on_monitor()
  wrf_dm_on_monitor = .true.
end function wrf_dm_on_monitor

subroutine wrf_dm_bcast_real(values, n)
  real :: values
  integer, intent(in) :: n
end subroutine wrf_dm_bcast_real

subroutine wrf_dm_bcast_integer(values, n)
  integer :: values
  integer, intent(in) :: n
end subroutine wrf_dm_bcast_integer

subroutine wrf_dm_bcast_string(value, n)
  character(len=*) :: value
  integer, intent(in) :: n
end subroutine wrf_dm_bcast_string

subroutine wrf_get_nproc(n)
  integer, intent(out) :: n
  n = 1
end subroutine wrf_get_nproc

subroutine wrf_get_myproc(n)
  integer, intent(out) :: n
  n = 0
end subroutine wrf_get_myproc
