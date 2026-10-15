! Stubs for the WRF framework symbols phys/module_ra_rrtm.F (WRF v4.6.1)
! needs, so the scheme compiles EXACTLY as shipped with no edit to it.
!
! The module touches the framework in five places only:
!   * rrtm_lookuptable USEs module_wrf_error (wrf_error_fatal on an
!     unreadable RRTM_DATA) and calls the external wrf_dm_on_monitor and
!     wrf_dm_bcast_bytes (a single-process oracle: monitor is always true,
!     every broadcast is a no-op);
!   * RRTMLWRAD USEs module_ra_clwrf_support for read_CAMgases, reached only
!     with ghg_input == 1 -- the oracle drives ghg_input = 0, so the stub
!     stops loudly if it is ever reached;
!   * RRTMLWRAD calls the external wrf_debug, MM5ATM the external
!     wrf_message (the TBOUND > 340 K reset message).

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
end module module_wrf_error

module module_ra_clwrf_support
  implicit none
contains
  subroutine read_CAMgases(yr, julian, READtrFILE, model, co2vmr, n2ovmr, &
                           ch4vmr, cfc11vmr, cfc12vmr)
    integer, intent(in)          :: yr
    real, intent(in)             :: julian
    logical                      :: READtrFILE
    character(len=*), intent(in) :: model
    real(8), intent(out)         :: co2vmr, n2ovmr, ch4vmr, cfc11vmr, cfc12vmr
    co2vmr = 0d0; n2ovmr = 0d0; ch4vmr = 0d0; cfc11vmr = 0d0; cfc12vmr = 0d0
    write(*,'(A)') 'STUB read_CAMgases reached: the oracle runs ghg_input = 0'
    error stop 2
  end subroutine read_CAMgases
end module module_ra_clwrf_support

logical function wrf_dm_on_monitor()
  implicit none
  wrf_dm_on_monitor = .true.
end function wrf_dm_on_monitor

subroutine wrf_dm_bcast_bytes(values, count)
  integer :: values(*)
  integer, intent(in) :: count
end subroutine wrf_dm_bcast_bytes

subroutine wrf_debug(level, message)
  integer, intent(in) :: level
  character(len=*), intent(in) :: message
end subroutine wrf_debug

subroutine wrf_message(message)
  character(len=*), intent(in) :: message
  write(*,'(A)') trim(message)
end subroutine wrf_message
