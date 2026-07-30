! tools/noahmp_wrf461_oracle/wrf_stubs_radiation.F90
!
! Minimal link stubs for the two WRF framework entry points that
! module_sf_noahmplsm.F references.  Neither is reachable from any leaf in
! the radiation lane (they live in ERROR, VEGE_FLUX, SFCDIF1, FRH2O and the
! table readers), but the linker still needs them.  They abort loudly rather
! than returning, so a fixture can never be produced through an error path.

subroutine wrf_message(msg)
  implicit none
  character(len=*), intent(in) :: msg
  write(0,'(A)') 'wrf_message: '//trim(msg)
end subroutine wrf_message

subroutine wrf_error_fatal(msg)
  implicit none
  character(len=*), intent(in) :: msg
  write(0,'(A)') 'wrf_error_fatal: '//trim(msg)
  stop 9
end subroutine wrf_error_fatal
