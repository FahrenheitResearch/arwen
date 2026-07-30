module module_model_constants
  implicit none
  real, parameter :: g = 9.81
  real, parameter :: cp = 1004.5
  real, parameter :: r_d = 287.0
  real, parameter :: r_v = 461.6
  real, parameter :: rovcp = r_d / cp
  real, parameter :: p1000mb = 100000.0
  real, parameter :: xlv = 2.5e6
  real, parameter :: xls = 2.85e6
  real, parameter :: xlf = 3.50e5
  real, parameter :: stbolt = 5.67051e-8
  real, parameter :: piconst = 3.141592653589793
  real, parameter :: rhowater = 1000.0
end module module_model_constants

module module_wrf_error
  implicit none
contains
  subroutine wrf_message(message)
    character(len=*), intent(in) :: message
  end subroutine wrf_message

  subroutine wrf_error_fatal(message)
    character(len=*), intent(in) :: message
    write(*, '(A)') trim(message)
    error stop 1
  end subroutine wrf_error_fatal

  logical function wrf_at_debug_level(level)
    integer, intent(in) :: level
    wrf_at_debug_level = .false.
  end function wrf_at_debug_level

end module module_wrf_error

logical function wrf_dm_on_monitor()
  implicit none
  wrf_dm_on_monitor = .true.
end function wrf_dm_on_monitor

! Deliberately external: WRF's communication shims have implicit interfaces
! here and are no-ops in this single-process oracle.
subroutine wrf_dm_bcast_real(values, count)
  real :: values
  integer, intent(in) :: count
end subroutine wrf_dm_bcast_real

subroutine wrf_dm_bcast_integer(values, count)
  integer :: values
  integer, intent(in) :: count
end subroutine wrf_dm_bcast_integer

subroutine wrf_dm_bcast_string(value, count)
  character(len=*) :: value
  integer, intent(in) :: count
end subroutine wrf_dm_bcast_string

subroutine wrf_debug(level, message)
  integer, intent(in) :: level
  character(len=*), intent(in) :: message
end subroutine wrf_debug
