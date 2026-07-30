! Stubs for WRF infrastructure required to compile phys/module_ra_rrtmg_lw.F
! from WRF v4.6.1 EXACTLY as shipped (no edits to the physics module).
! Pattern follows tools/ruc_wrf461_oracle/stub_wrf.F90 and the Phase-A RRTMG
! scratch build (jobs/fea7a141/tmp/rrtmg_phaseA), which proved this set links.
!
! Constants below mirror share/module_model_constants.F of the same bundle.

module module_model_constants
  implicit none
  real, parameter :: g = 9.81
  real, parameter :: r_d = 287.
  real, parameter :: cp = 7.*r_d/2.        ! 1004.5, passed to rrtmg_lw_ini
  real, parameter :: r_v = 461.6
  real, parameter :: rcp = r_d/cp
  real, parameter :: p1000mb = 100000.
  real, parameter :: stbolt = 5.67051e-8
end module module_model_constants

module module_wrf_error
  implicit none
contains
  subroutine wrf_message(message)
    character(len=*), intent(in) :: message
    write(*,'(A)') trim(message)
  end subroutine wrf_message

  subroutine wrf_error_fatal(message)
    character(len=*), intent(in) :: message
    write(*,'(A)') trim(message)
    error stop 1
  end subroutine wrf_error_fatal

  logical function wrf_at_debug_level(level)
    integer, intent(in) :: level
    wrf_at_debug_level = .false.
  end function wrf_at_debug_level
end module module_wrf_error

module module_state_description
  implicit none
  ! Values from Registry/Registry.EM_COMMON (WRF v4.6.1 mp_physics ids).
  ! Only used by the RRTMG wrapper in Ferrier-scheme comparisons; the
  ! oracle never drives those schemes, so the branches stay inert.
  integer, parameter :: FER_MP_HIRES = 5
  integer, parameter :: FER_MP_HIRES_ADVECT = 15
  integer, parameter :: ETAMP_HWRF = 85
end module module_state_description

module module_ra_clwrf_support
  implicit none
contains
  ! Real routine reads CAMtr_volume_mixing_ratio; only reached when
  ! ghg_input == 1.  The oracle drives ghg_input = 0, so this must not fire.
  subroutine read_CAMgases(yr, julian, READtrFILE, model, co2vmr, n2ovmr, &
                           ch4vmr, cfc11vmr, cfc12vmr)
    integer, intent(in)          :: yr
    real, intent(in)             :: julian
    logical                      :: READtrFILE
    character(len=*), intent(in) :: model
    real(8), intent(out)         :: co2vmr, n2ovmr, ch4vmr, cfc11vmr, cfc12vmr
    co2vmr = 0d0; n2ovmr = 0d0; ch4vmr = 0d0; cfc11vmr = 0d0; cfc12vmr = 0d0
    write(*,'(A)') 'STUB read_CAMgases reached: ghg_input must be 0'
    error stop 2
  end subroutine read_CAMgases
end module module_ra_clwrf_support

logical function wrf_dm_on_monitor()
  implicit none
  wrf_dm_on_monitor = .true.
end function wrf_dm_on_monitor

! Deliberately external with implicit interfaces, as in WRF's own frame:
! single-process oracle, so all broadcasts are no-ops.
subroutine wrf_dm_bcast_bytes(values, count)
  integer :: values(*)
  integer, intent(in) :: count
end subroutine wrf_dm_bcast_bytes

subroutine wrf_dm_bcast_real(values, count)
  real :: values
  integer, intent(in) :: count
end subroutine wrf_dm_bcast_real

subroutine wrf_dm_bcast_integer(values, count)
  integer :: values
  integer, intent(in) :: count
end subroutine wrf_dm_bcast_integer

subroutine wrf_debug(level, message)
  integer, intent(in) :: level
  character(len=*), intent(in) :: message
end subroutine wrf_debug

! External (non-module) variants: several RRTMG core routines call these
! without USE module_wrf_error, so they link against plain externals.
subroutine wrf_error_fatal(message)
  implicit none
  character(len=*), intent(in) :: message
  write(*,'(A)') trim(message)
  error stop 1
end subroutine wrf_error_fatal

subroutine wrf_message(message)
  implicit none
  character(len=*), intent(in) :: message
  write(*,'(A)') trim(message)
end subroutine wrf_message
