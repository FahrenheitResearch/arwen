! gfortran 13.3 -O0 (the oracle's build) MAX/MIN NaN and signed-zero semantics.
! Measured output (J, 2026-10-06): MAX(-35.,NaN)=NaN, MAX(NaN,-35.)=NaN,
! MAX(1.,NaN)=NaN, MAX(NaN,1.)=1, MIN(1.,NaN)=NaN, MIN(NaN,1.)=1,
! MAX(x=0,-0)=-0, MAX(-0,x=0)=+0: kernels/morrison.cu FMAX/FMAXK follow it.
program maxprobe
  implicit none
  real :: x, y, nan, nz, r
  nan = transfer(int(z'7FC00000'), 1.0)
  nz = sign(0.0, -1.0)
  x = nan; y = 1.0
  call pr('MAX(-35.,NaN)', max(-35., x))
  call pr('MAX(NaN,-35.)', max(x, -35.))
  call pr('MAX(1.,NaN) var', max(y, x))
  call pr('MAX(NaN,1.) var', max(x, y))
  call pr('MIN(1.,NaN) var', min(y, x))
  call pr('MIN(NaN,1.) var', min(x, y))
  call pr('MAX(0.,-0.)', max(0., nz))
  call pr('MAX(-0.,0.)', max(nz, 0.))
  x = 0.0
  call pr('MAX(x=0,-0) var', max(x, nz))
  call pr('MAX(-0,x=0) var', max(nz, x))
contains
  subroutine pr(s, v)
    character(*) :: s
    real :: v
    write(*,'(A20,1X,Z8.8)') s, transfer(v, 0)
  end subroutine
end program
