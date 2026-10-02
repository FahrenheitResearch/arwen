! Read-only integer counters for the native inline limiter branches.
module advect_limiter_spy
  use iso_fortran_env, only: int64
  implicit none
  integer(int64) :: pd_all=0, pd_physical=0, mono_in_all=0, mono_in_physical=0
  integer(int64) :: mono_out_all=0, mono_out_physical=0
contains
  subroutine hit_pd(physical)
    logical, intent(in) :: physical
    pd_all=pd_all+1
    if (physical) pd_physical=pd_physical+1
  end subroutine hit_pd
  subroutine hit_mono_in(physical)
    logical, intent(in) :: physical
    mono_in_all=mono_in_all+1
    if (physical) mono_in_physical=mono_in_physical+1
  end subroutine hit_mono_in
  subroutine hit_mono_out(physical)
    logical, intent(in) :: physical
    mono_out_all=mono_out_all+1
    if (physical) mono_out_physical=mono_out_physical+1
  end subroutine hit_mono_out
  subroutine write_limiter_counts(path)
    character(*), intent(in) :: path
    integer :: unit
    open(newunit=unit,file=path,status='replace',action='write')
    write(unit,'(A)') '{'
    write(unit,'(A,I0,A)') '  "pd_all": ',pd_all,','
    write(unit,'(A,I0,A)') '  "pd_physical": ',pd_physical,','
    write(unit,'(A,I0,A)') '  "mono_in_all": ',mono_in_all,','
    write(unit,'(A,I0,A)') '  "mono_in_physical": ',mono_in_physical,','
    write(unit,'(A,I0,A)') '  "mono_out_all": ',mono_out_all,','
    write(unit,'(A,I0,A)') '  "mono_out_physical": ',mono_out_physical
    write(unit,'(A)') '}'
    close(unit)
  end subroutine write_limiter_counts
end module advect_limiter_spy
