! Positive control for build.sh's `nm -u | grep _ZGV` guard.
!
! A guard that has never been seen to fire is not evidence.  The -O0 object
! this oracle links carries no glibc vector-math symbol -- but so would an
! object built on a toolchain that has no libmvec at all, and so would an
! object the grep simply failed to read.  The claim only means something if
! the same grep, on the same toolchain, CAN find _ZGV*.
!
! This subroutine is the same expf in a loop the vectoriser does take.  Built
! at -Ofast -ftree-vectorize it must emit _ZGVbN4v_expf; build.sh fails if it
! does not.
!
! For this module the guard is doubly checkable: WRF's own -O2
! -ftree-vectorize build of share/module_soil_pre.F DOES pull in
! _ZGVbN4v_expf (from adjust_soil_temp_new's exp), and build.sh records that
! object's symbols beside the -O0 reference's.  init_soil_3_real itself
! contains no transcendental, so the swap cannot move the numbers this oracle
! publishes -- but the difference between the two objects is on the record
! rather than asserted.
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
