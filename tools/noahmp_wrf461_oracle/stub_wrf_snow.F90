! Service-only stubs for the standalone Noah-MP snow-leaf oracle harness.
!
! MODULE_SF_NOAHMPLSM references exactly three external procedures --
! wrf_message, wrf_error_fatal and wrf_debug -- all of which are WRF logging
! and termination services, not physics.  Nothing in this file computes a
! physical quantity, and wrf_error_fatal aborts rather than returning, so a
! harness case that trips a WRF fatal path fails loudly instead of emitting a
! fabricated row.

subroutine wrf_message(msg)
  implicit none
  character(len=*), intent(in) :: msg
  write(*, '(A)') trim(msg)
end subroutine wrf_message

subroutine wrf_error_fatal(msg)
  implicit none
  character(len=*), intent(in) :: msg
  write(*, '(A)') 'noahmp snow oracle: WRF fatal: '//trim(msg)
  error stop 1
end subroutine wrf_error_fatal

subroutine wrf_debug(level, msg)
  implicit none
  integer, intent(in) :: level
  character(len=*), intent(in) :: msg
end subroutine wrf_debug
