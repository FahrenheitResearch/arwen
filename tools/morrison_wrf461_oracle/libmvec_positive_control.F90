! Positive control for build.sh's libmvec guard.
!
! EXP is one of the transcendentals in module_mp_morr_two_moment.F.  This
! uncomplicated loop is deliberately vectorisable at WRF's physics flags.
! gfortran 13.3 / glibc 2.39 must therefore leave a _ZGV* undefined symbol;
! if it does not, silence from nm on the -O0 reference object proves nothing.
subroutine morrison_libmvec_positive_control(x, y, n)
  implicit none
  integer, intent(in) :: n
  real, intent(in) :: x(n)
  real, intent(out) :: y(n)
  integer :: k

  do k = 1, n
    y(k) = exp(x(k))
  end do
end subroutine morrison_libmvec_positive_control
