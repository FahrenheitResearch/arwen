! Service-only stubs for the standalone Noah-MP WRF v4.6.1 oracle harness.
!
! Everything here supplies infrastructure that WRF's build normally provides:
! logging, fatal-error termination, an opaque domain handle, and link targets
! for physics packages this harness deliberately does not build.  No stub in
! this file computes a physical quantity.  The urban-canopy, BEP, BEP-BEM and
! GFDL calendar entry points abort instead of returning fabricated values, so
! reaching them is a hard failure rather than a silent substitution.

! WRF's logging shims.  module_sf_noahmplsm.F, module_sf_noahmp_glacier.F,
! module_sf_noahmp_groundwater.F and module_sf_noahmpdrv.F all call these with
! implicit interfaces, so they stay external subroutines.
subroutine wrf_message(msg)
  implicit none
  character(len=*), intent(in) :: msg
  write(*, '(A)') trim(msg)
end subroutine wrf_message

subroutine wrf_error_fatal(msg)
  implicit none
  character(len=*), intent(in) :: msg
  write(*, '(A)') trim(msg)
  error stop 1
end subroutine wrf_error_fatal

subroutine wrf_debug(level, msg)
  implicit none
  integer, intent(in) :: level
  character(len=*), intent(in) :: msg
end subroutine wrf_debug

! module_sf_noahmpdrv.F's GROUNDWATER_INIT declares TYPE(domain), TARGET :: grid
! unconditionally but never dereferences a component outside the EM_CORE +
! DM_PARALLEL halo block, which this harness compiles out.  An opaque handle is
! therefore sufficient and carries no state.
module module_domain
  implicit none
  type :: domain
    integer :: opaque_handle = 0
  end type domain
end module module_domain

! module_sf_noahmpdrv.F uses IRI_SCHEME (a namelist switch, not physics) from
! module_sf_urban, and calls urban/bep/bep_bem from noahmp_urban.  Declaring the
! procedures EXTERNAL keeps them USE-associable without this file having to
! restate -- and therefore risk mis-stating -- their real argument lists.
module module_sf_urban
  implicit none
  integer :: IRI_SCHEME = 0
  external :: urban
end module module_sf_urban

module module_sf_bep
  implicit none
  external :: bep
end module module_sf_bep

module module_sf_bep_bem
  implicit none
  external :: bep_bem
end module module_sf_bep_bem

module module_ra_gfdleta
  implicit none
  external :: cal_mon_day
end module module_ra_gfdleta

subroutine urban()
  implicit none
  write(*, '(A)') 'noahmp oracle: module_sf_urban::urban is not built into this harness'
  error stop 91
end subroutine urban

subroutine bep()
  implicit none
  write(*, '(A)') 'noahmp oracle: module_sf_bep::bep is not built into this harness'
  error stop 92
end subroutine bep

subroutine bep_bem()
  implicit none
  write(*, '(A)') 'noahmp oracle: module_sf_bep_bem::bep_bem is not built into this harness'
  error stop 93
end subroutine bep_bem

subroutine cal_mon_day()
  implicit none
  write(*, '(A)') 'noahmp oracle: module_ra_gfdleta::cal_mon_day is not built into this harness'
  error stop 94
end subroutine cal_mon_day
