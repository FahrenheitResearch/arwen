! Service-only stubs for the standalone urban WRF v4.7.1 oracle harness.
!
! NOTHING here computes a physical quantity.  module_model_constants and
! module_wrf_error are compiled from the pinned tree (build.sh), so every
! constant the reference is measured in is WRF's own.  What is left is what
! `nm -u` reports undefined outside libm/libgfortran once the urban, BEP,
! BEP_BEM, BEM, Noah and myjurb modules are built from the pinned sources:
!
!   module_wrf_error.o  : wrf_abort_, wrf_debug_
!   module_sf_noahdrv.o : wrf_dm_on_monitor_, wrf_dm_bcast_{real,integer,
!                         string}_ (SOIL_VEG_GEN_PARM's single-rank shims)
!
! cal_mon_day is NOT stubbed: build.sh extracts WRF's own CAL_MON_DAY from
! phys/module_ra_gfdleta.F byte for byte into a one-routine module, because
! the rest of that radiation module needs MODULE_CONFIGURE and the Ferrier
! tables and cal_mon_day is all the urban models call from it.

subroutine wrf_abort()
  implicit none
  error stop 1
end subroutine wrf_abort

subroutine wrf_debug(level, msg)
  implicit none
  integer, intent(in) :: level
  character(len=*), intent(in) :: msg
end subroutine wrf_debug

! One rank, so the monitor is this process and every broadcast is a no-op.
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
