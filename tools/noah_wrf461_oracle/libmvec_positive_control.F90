! Positive control for build.sh's `nm -u | grep _ZGV` guard.
!
! A guard that has never been seen to fire is not evidence.  Neither
! module_sf_noahlsm.F nor module_sf_noahdrv.F pulls in a glibc vector-math
! symbol on gfortran 13.3.0 -- not at -O0, not at WRF's own -O2
! -ftree-vectorize, not at -Ofast -- because every expf/powf/logf/atanf/log10f
! call in the Noah column sits inside a loop the vectoriser gives up on
! (soil-layer loops carrying conditionals, the FRH2O Newton loop with its
! goto-based exit, SNOPAC's branch tree).  So the guard passing on those
! objects proves nothing on its own.
!
! This subroutine is the same expf in a loop the vectoriser CAN take.  Built at
! -Ofast -ftree-vectorize it must emit _ZGVbN4v_expf; build.sh fails if it does
! not, because that would mean the grep is hunting a symbol this toolchain
! never produces and the check on the Noah objects is vacuous.
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
