! Service-only stubs for the UCM Noah-coupling oracle (ucm_noah_oracle.F90).
! module_model_constants.F, module_wrf_error.F, module_sf_urban.F and the
! BEP/BEM modules are compiled from the pinned sources; what is left here is
! what module_wrf_error.o and SOIL_VEG_GEN_PARM leave undefined, and none of
! it computes a physical quantity.  With one rank every broadcast is a no-op
! by construction.

subroutine wrf_abort()
  implicit none
  error stop 1
end subroutine wrf_abort

subroutine wrf_debug(level, msg)
  implicit none
  integer, intent(in) :: level
  character(len=*), intent(in) :: msg
end subroutine wrf_debug

logical function wrf_dm_on_monitor()
  implicit none
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
