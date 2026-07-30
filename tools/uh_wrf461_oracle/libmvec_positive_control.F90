! Positive control for build.sh's `nm -u | grep _ZGV` guard.
!
! A guard that has never been seen to fire is not evidence.  bl_ysu.F90 turns
! out NOT to pull in a glibc vector-math symbol on gfortran 13.3.0, at -O0, at
! WRF's own -O2 -ftree-vectorize, or even at -Ofast -- its expf/powf calls all
! sit in loops carrying conditionals the vectoriser gives up on.  So the guard
! passing on bl_ysu.o proves nothing on its own.
!
! This subroutine is the same expf in a loop the vectoriser CAN take.  Built at
! the same -Ofast -ftree-vectorize, it must emit _ZGVbN4v_expf; build.sh fails
! if it does not, because that would mean the grep is looking for a symbol this
! toolchain never produces and the check on bl_ysu.o is vacuous.
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
