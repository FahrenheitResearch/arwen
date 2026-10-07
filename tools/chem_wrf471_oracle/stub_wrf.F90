! Service-only stubs for the standalone chem WRF v4.7.1 oracle harness.
!
! NOTHING here computes a physical quantity.  These are the WRF framework
! services the extracted chem routines call for messages and fatal stops,
! which exist only inside a configured WRF build (frame/module_wrf_error.F
! and its C helpers).  A driver that needs a constant takes it from the
! pinned share/module_model_constants.F (through a sources-<lane>.list) or
! states it in its prelude with the WRF file:line it came from.

subroutine wrf_error_fatal(msg)
  implicit none
  character(len=*), intent(in) :: msg
  write(0, '(A)') 'wrf_error_fatal: ' // trim(msg)
  error stop 1
end subroutine wrf_error_fatal

subroutine wrf_message(msg)
  implicit none
  character(len=*), intent(in) :: msg
end subroutine wrf_message

subroutine wrf_debug(level, msg)
  implicit none
  integer, intent(in) :: level
  character(len=*), intent(in) :: msg
end subroutine wrf_debug

subroutine wrf_abort()
  implicit none
  error stop 1
end subroutine wrf_abort

! flow_dep_bdy_chem USE dependency; unsupported CAM branch fails in driver.
module module_cam_mam_initmixrats
  implicit none
  external :: bdy_chem_value_cam_mam
end module module_cam_mam_initmixrats
