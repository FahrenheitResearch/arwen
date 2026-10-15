! Service-only stub for the standalone Purdue Lin oracle.
!
! module_mp_lin.F and module_mp_radar.F USE module_wrf_error.  This supplies
! the message buffer, the debug sink and the fatal-error routine those two
! byte-unmodified modules name.  It contains no physical constant and no
! arithmetic.  wrf_error_fatal stops the oracle with a nonzero status, so a
! column that would kill WRF (ggamma's argument guard) kills the build too.
module module_wrf_error
  implicit none
  character(len=256) :: wrf_err_message
end module module_wrf_error

subroutine wrf_debug(level, message)
  implicit none
  integer, intent(in) :: level
  character(len=*), intent(in) :: message
end subroutine wrf_debug

subroutine wrf_error_fatal(message)
  implicit none
  character(len=*), intent(in) :: message
  print *, 'wrf_error_fatal: ', trim(message)
  stop 9
end subroutine wrf_error_fatal
