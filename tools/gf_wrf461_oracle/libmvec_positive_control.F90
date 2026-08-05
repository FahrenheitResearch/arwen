! Positive control for build.sh's `nm -u | grep _ZGV` guard.
!
! A guard that has never been seen to fire is not evidence.  On this project
! the answer has already gone both ways on the same toolchain: bl_ysu.F90
! pulled in NO glibc vector-math symbol even at -Ofast, while
! module_bl_shinhong.F pulled _ZGVbN4vv_powf in at WRF's own
! -O2 -ftree-vectorize.  Which way the three Grell-Freitas modules fall is
! exactly what the undefined-*.txt receipts record, so the guard passing on
! the -O0 objects proves nothing on its own.
!
! This subroutine is the same expf in a loop the vectoriser CAN take.  Built
! at the same -Ofast -ftree-vectorize, it must emit _ZGVbN4v_expf; build.sh
! fails if it does not, because that would mean the grep is looking for a
! symbol this toolchain never produces and the check on the reference objects
! is vacuous.
subroutine libmvec_positive_control(n, x, y)
  implicit none
  integer, intent(in) :: n
  real, intent(in) :: x(n)
  real, intent(out) :: y(n)
  integer :: i
  do i = 1, n
    y(i) = exp(x(i))
  end do
end subroutine libmvec_positive_control
