! Logging services only. No numerical constants or configuration substitutes.
module module_wrf_error
  character(len=512) :: wrf_err_message
contains
  subroutine wrf_debug(level, message)
    integer, intent(in) :: level
    character(len=*), intent(in) :: message
  end subroutine
  subroutine wrf_error_fatal(message)
    character(len=*), intent(in) :: message
    error stop 'WRF fatal error'
  end subroutine
end module
subroutine get_current_time_string(message)
  character(len=*), intent(out) :: message
  message = 'oracle'
end subroutine
subroutine get_current_grid_name(message)
  character(len=*), intent(out) :: message
  message = 'oracle'
end subroutine
