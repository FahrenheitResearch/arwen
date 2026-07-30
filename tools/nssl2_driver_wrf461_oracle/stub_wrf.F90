subroutine wrf_error_fatal(message)
  implicit none
  character(len=*), intent(in) :: message
  print '(A)', trim(message)
  error stop 1
end subroutine wrf_error_fatal

logical function wrf_dm_on_monitor()
  implicit none
  ! Prevent the standalone module from appending namelist.output.
  wrf_dm_on_monitor = .false.
end function wrf_dm_on_monitor
