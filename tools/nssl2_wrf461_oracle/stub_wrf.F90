subroutine wrf_error_fatal(message)
  implicit none
  character(len=*), intent(in) :: message
  print '(A)', trim(message)
  error stop 1
end subroutine wrf_error_fatal

logical function wrf_dm_on_monitor()
  implicit none
  ! Avoid WRF's attempt to append to namelist.output.  The NSSL internal
  ! namelist read is permitted to return a nonzero iostat in this harness.
  wrf_dm_on_monitor = .false.
end function wrf_dm_on_monitor
