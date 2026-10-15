module module_wrf_error
character(len=4096) :: wrf_err_message
end module
subroutine wrf_debug(level,message)
integer,intent(in)::level
character(len=*),intent(in)::message
end subroutine
subroutine wrf_error_fatal(message)
character(len=*),intent(in)::message
write(*,*) message
stop 2
end subroutine
