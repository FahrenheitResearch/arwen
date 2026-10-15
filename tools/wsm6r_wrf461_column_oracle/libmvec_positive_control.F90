! Positive control for build.sh's `nm -u | grep _ZGV` guard.
!
! A guard that has never been seen to fire is not evidence.  This is an expf
! in a loop the vectoriser CAN take; built at -Ofast -ftree-vectorize it must
! emit a glibc vector symbol (_ZGV...), and build.sh fails if it does not,
! because the -O0 check on module_mp_wsm6r.o would then be vacuous.
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
