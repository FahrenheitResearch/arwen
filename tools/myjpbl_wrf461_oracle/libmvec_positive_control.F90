! Positive control for build.sh's `nm -u | grep _ZGV` guard.
!
! A guard that has never been seen to fire is not evidence.  build.sh checks
! that the -O0 object of module_bl_myjpbl.F calls only the SCALAR expf/powf
! (no glibc vector-math _ZGV* entry point).  If this toolchain never emitted a
! _ZGV symbol at all, that check would pass vacuously.
!
! This subroutine is an expf in a loop the vectoriser CAN take.  Built at
! -Ofast -ftree-vectorize it must emit _ZGVbN4v_expf; build.sh fails if it
! does not.
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
