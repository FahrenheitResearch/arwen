module module_wrf_error
 implicit none
 contains
 subroutine wrf_error_fatal(message)
 character(*),intent(in)::message
 print *,trim(message)
 error stop 1
 end subroutine
end module
module module_checkerror
 implicit none
 contains
 subroutine checkerror(routine,quantity,i,k,j,value)
 character(*),intent(in)::routine,quantity
 integer,intent(in)::i,k,j
 real,intent(in)::value
 if (.not.(abs(value)<huge(value))) error stop 'non-finite oracle input'
 end subroutine
end module
subroutine wrf_dm_bcast_bytes(value,n)
 implicit none
 integer,intent(in)::n
 real,intent(inout)::value(*)
 ! Serial column oracle. No distributed memory communication.
end subroutine
logical function wrf_dm_on_monitor()
 wrf_dm_on_monitor=.true.
end function
