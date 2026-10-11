! WRF framework stubs for the fast SBM oracle harness.
!
! module_mp_fast_sbm.F (WRF v4.8.0) USEs module_domain and module_dm inside
! FAST_HUCMINIT and BREAKINIT_KS and calls five framework routines. None of
! them carries physics; these stand-ins let the unmodified WRF source compile
! and run serially:
!   wrf_error_fatal    writes the message to stderr and to a refusal record
!                      (file named by FSBM_REFUSAL_FILE, default
!                      fsbm_refusal.txt in the working directory), then
!                      STOP 99, so a refusal is a process exit with a message
!                      the harness can record
!   wrf_message        writes the message to stderr
!   wrf_debug          no-op
!   wrf_dm_on_monitor  .true. (a single serial task is its own monitor)
!   wrf_dm_bcast_bytes no-op (BREAKINIT_KS calls it outside DM_PARALLEL)
module module_domain
end module module_domain

module module_dm
end module module_dm

subroutine wrf_error_fatal(msg)
  implicit none
  character(len=*), intent(in) :: msg
  character(len=1024) :: path
  integer :: u, st, ln
  write(0,'(a)') 'WRF_ERROR_FATAL: '//trim(msg)
  path = ' '
  call get_environment_variable('FSBM_REFUSAL_FILE', path, ln, st)
  if (st /= 0 .or. ln == 0) path = 'fsbm_refusal.txt'
  open(newunit=u, file=trim(path), status='replace', action='write', iostat=st)
  if (st == 0) then
    write(u,'(a)') trim(msg)
    close(u)
  end if
  stop 99
end subroutine wrf_error_fatal

subroutine wrf_message(msg)
  implicit none
  character(len=*), intent(in) :: msg
  write(0,'(a)') trim(msg)
end subroutine wrf_message

subroutine wrf_debug(level, msg)
  implicit none
  integer, intent(in) :: level
  character(len=*), intent(in) :: msg
end subroutine wrf_debug

logical function wrf_dm_on_monitor()
  implicit none
  wrf_dm_on_monitor = .true.
end function wrf_dm_on_monitor

subroutine wrf_dm_bcast_bytes(buf, nbytes)
  implicit none
  integer :: buf
  integer, intent(in) :: nbytes
end subroutine wrf_dm_bcast_bytes
