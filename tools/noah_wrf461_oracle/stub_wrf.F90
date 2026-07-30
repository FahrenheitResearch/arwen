! Service-only stubs for the standalone Noah LSM WRF v4.6.1 oracle harness.
!
! The stub set here is deliberately as small as `nm -u` allows, and NOTHING in
! it computes a physical quantity.  In particular there is NO stub for
! module_model_constants and NO stub for module_wrf_error: build.sh compiles
! both from the pinned WRF tree, byte-unmodified, because CP, R_D, XLF, XLV,
! RHOWATER, STBOLT and KARMAN are the constants the whole reference is measured
! in and a stub would be free to get them wrong.  (The RUC harness in
! tools/ruc_wrf461_oracle/stub_wrf.F90 does stub them; that was a weaker
! choice, and it is why its module_model_constants writes r_d = 287.0 by hand
! where WRF derives cp = 7*r_d/2 from it.)
!
! `wrf_error_fatal` is NOT stubbed either -- frame/module_wrf_error.F defines
! it, so this file supplies only the four service entry points that file in
! turn leaves undefined (wrf_abort, wrf_debug) plus the single-process forms
! of WRF's MPI broadcast shims, which SOIL_VEG_GEN_PARM calls after the
! monitor rank has read the tables.  With one rank those broadcasts are
! no-ops by construction, not by approximation.
!
! What is left is exactly what the link step reports as undefined outside
! libm/libgfortran:
!
!   module_wrf_error.o  : wrf_abort_, wrf_debug_
!   module_sf_noahdrv.o : wrf_dm_on_monitor_, wrf_dm_bcast_{real,integer,
!                         string}_, urban_, bep_, bep_bem_, cal_mon_day_,
!                         and the module handles module_sf_urban::{iri_scheme,
!                         oasis}
!
! The urban, BEP, BEP-BEM and GFDL-calendar entry points ABORT rather than
! return a fabricated value, so a fixture row that reached one of them would
! be a hard failure instead of a silent substitution.  No fixture row reaches
! them: run_lsm.F90 calls lsm with sf_urban_physics = 0.

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

! Deliberately external with implicit interfaces, exactly as WRF declares
! them: SOIL_VEG_GEN_PARM calls wrf_dm_bcast_real with both array and scalar
! first arguments.
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

! module_sf_noahdrv.F uses IRI_SCHEME (a namelist switch, not physics) and
! OASIS from module_sf_urban, and calls urban/bep/bep_bem.  Declaring the
! procedures EXTERNAL keeps them USE-associable without this file having to
! restate -- and therefore risk mis-stating -- their real argument lists.
module module_sf_urban
  implicit none
  integer :: iri_scheme = 0
  real :: oasis = 1.0
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
  write(*, '(A)') 'noah oracle: module_sf_urban::urban is not built into this harness'
  error stop 91
end subroutine urban

subroutine bep()
  implicit none
  write(*, '(A)') 'noah oracle: module_sf_bep::bep is not built into this harness'
  error stop 92
end subroutine bep

subroutine bep_bem()
  implicit none
  write(*, '(A)') 'noah oracle: module_sf_bep_bem::bep_bem is not built into this harness'
  error stop 93
end subroutine bep_bem

subroutine cal_mon_day()
  implicit none
  write(*, '(A)') 'noah oracle: module_ra_gfdleta::cal_mon_day is not built into this harness'
  error stop 94
end subroutine cal_mon_day
