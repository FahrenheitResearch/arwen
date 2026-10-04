! Diagnostic adapters only. No lake arithmetic is replaced.
module module_wrf_error
  implicit none
contains
  subroutine wrf_message(message)
    character(len=*), intent(in) :: message
  end subroutine
  subroutine wrf_error_fatal(message)
    character(len=*), intent(in) :: message
    write(*,'(A)') trim(message)
    error stop 1
  end subroutine
end module
subroutine wrf_debug(level,message)
  integer,intent(in) :: level
  character(len=*),intent(in) :: message
end subroutine
