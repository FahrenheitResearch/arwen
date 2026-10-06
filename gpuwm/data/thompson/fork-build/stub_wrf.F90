! Minimal WRF environment for the HRRR v4.1.21 fork's (WRFV3.9 generation)
! module_mp_thompson.F, built serially (no DM_PARALLEL) on a node CPU.
!
! Differences from tools/thompson_wrf461_oracle/stub_wrf.F90, each named
! by what the fork's source calls:
!   * wrf_at_debug_level (module_mp_thompson.F:434-438, 1165-1166, 4742):
!     the fork sets its print flags from it; .false. keeps every debug
!     WRITE off, which changes no number.
!   * no module_model_constants: the fork's module carries its own
!     effective-radius bounds and never USEs it.
! nl_get_force_read_thompson = .false. and nl_get_write_thompson_tables =
! .true. make the first thompson_init compute every table and write the
! three caches (qr_acr_qg.dat, qr_acr_qs.dat, freezeH2O.dat).
module module_wrf_error
  implicit none
  character(len=512) :: wrf_err_message
contains
  logical function wrf_at_debug_level(level)
    integer, intent(in) :: level
    wrf_at_debug_level = .false.
  end function wrf_at_debug_level

  subroutine wrf_debug(level, message)
    integer, intent(in) :: level
    character(len=*), intent(in) :: message
  end subroutine wrf_debug

  subroutine wrf_message(message)
    character(len=*), intent(in) :: message
    print '(A)', trim(message)
  end subroutine wrf_message

  subroutine wrf_error_fatal(message)
    character(len=*), intent(in) :: message
    print '(A)', trim(message)
    error stop 1
  end subroutine wrf_error_fatal
end module module_wrf_error

module module_domain
  implicit none
end module module_domain

module module_dm
  implicit none
end module module_dm

subroutine nl_get_force_read_thompson(domain, value)
  implicit none
  integer, intent(in) :: domain
  logical, intent(out) :: value
  value = .false.
end subroutine nl_get_force_read_thompson

subroutine nl_get_write_thompson_tables(domain, value)
  implicit none
  integer, intent(in) :: domain
  logical, intent(out) :: value
  value = .true.
end subroutine nl_get_write_thompson_tables

logical function wrf_dm_on_monitor()
  implicit none
  wrf_dm_on_monitor = .true.
end function wrf_dm_on_monitor
