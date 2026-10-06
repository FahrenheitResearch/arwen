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
