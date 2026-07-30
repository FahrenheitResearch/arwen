! Stubs for WRF infrastructure required to compile phys/module_ra_rrtmg_lw.F
! and phys/module_ra_rrtmg_sw.F from WRF v4.6.1 EXACTLY as shipped.
! Pattern follows the ruc_wrf461_oracle stub for this bundle generation.
!
! Constants below mirror share/module_model_constants.F of the same bundle.

module module_model_constants
  implicit none
  real, parameter :: g = 9.81
  real, parameter :: r_d = 287.
  real, parameter :: cp = 7.*r_d/2.        ! 1004.5, passed to rrtmg_*_ini
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
  ! Values from Registry/Registry.EM_COMMON (WRF v4.6.1 mp_physics scheme ids).
  ! Only used by RRTMG wrappers in mp_physics comparisons for Ferrier schemes;
  ! campaign schemes are 6/8/10/18 so these branches stay inert.
  integer, parameter :: FER_MP_HIRES = 5
  integer, parameter :: FER_MP_HIRES_ADVECT = 15
  integer, parameter :: ETAMP_HWRF = 85
end module module_state_description

module module_ra_clwrf_support
  implicit none
contains
  ! Real routine reads CAMtr_volume_mixing_ratio; only reached when
  ! ghg_input == 1.  Oracle drives ghg_input = 0, so this must never fire.
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

module module_ra_rrtmg_sw_cmaq
  implicit none
contains
  ! Real routine lives in phys/module_ra_rrtmg_aero_optical_util_cmaq.F and is
  ! only called on the WRF-CMAQ twoway path (proceed_cmaq_sw); oracle never
  ! enables it.  Signature mirrors the shipped one so the call site compiles.
  subroutine get_aerosol_Optics_RRTMG_SW(ns, nmode, delta_z, INMASS_ws,     &
                                         INMASS_in, INMASS_ec, INMASS_ss,   &
                                         INMASS_h2o, INDGN, INSIG,          &
                                         tauaer, waer, gaer)
    integer, intent(in) :: ns, nmode
    real, intent(in)    :: delta_z
    real, intent(in)    :: INMASS_ws(nmode), INMASS_in(nmode)
    real, intent(in)    :: INMASS_ec(nmode), INMASS_ss(nmode)
    real, intent(in)    :: INMASS_h2o(nmode)
    real, intent(in)    :: INDGN(nmode), INSIG(nmode)
    real, intent(out)   :: tauaer, waer, gaer
    tauaer = 0.; waer = 0.; gaer = 0.
    write(*,'(A)') 'STUB get_aerosol_Optics_RRTMG_SW reached: CMAQ path must stay off'
    error stop 3
  end subroutine get_aerosol_Optics_RRTMG_SW
end module module_ra_rrtmg_sw_cmaq

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
