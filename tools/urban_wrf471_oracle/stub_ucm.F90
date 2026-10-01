! Service-only stubs for the UCM column oracle.  Nothing here computes a
! physical quantity: module_model_constants.F and module_wrf_error.F are
! compiled from the pinned tree (build_ucm.sh), and module_wrf_error.o leaves
! exactly these two entry points undefined.

subroutine wrf_abort()
  implicit none
  error stop 1
end subroutine wrf_abort

subroutine wrf_debug(level, msg)
  implicit none
  integer, intent(in) :: level
  character(len=*), intent(in) :: msg
end subroutine wrf_debug
