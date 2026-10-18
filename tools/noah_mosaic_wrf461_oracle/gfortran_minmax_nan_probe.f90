! What gfortran -O0 returns for Fortran MAX/MIN when an argument is NaN.
! The Fortran standard leaves this processor dependent. The probe shows
! gfortran's answer also depends on how the call is written: the nested
! form SSTEP uses, MAX(MIN(STOT,SMCMAX),0.02), returns NaN for a NaN STOT,
! while the same MIN written alone returns SMCMAX. So no source-order rule
! (and neither fmaxf/fminf) reproduces WRF's NaN propagation.
! Build and run: gfortran -O0 gfortran_minmax_nan_probe.f90 -o probe && ./probe
subroutine s(stot, smcmax, r)
  real :: stot, smcmax, r(10)
  real :: pz, nz
  pz = 0.0; nz = -pz
  r(1) = MAX ( MIN (STOT,SMCMAX),0.02 )
  r(2) = MIN(stot, smcmax); r(3) = MIN(smcmax, stot)
  r(4) = MAX(stot, smcmax); r(5) = MAX(smcmax, stot)
  r(6) = MAX(pz,nz); r(7) = MAX(nz,pz); r(8) = MIN(pz,nz); r(9) = MIN(nz,pz)
  r(10) = MAX(0.0, stot)
end subroutine
program probe
  real :: n, r(10)
  n = transfer(int(z"7fc00000"), 1.0)
  call s(n, 0.464, r)
  write(*,"(a)") "MAX(MIN(NaN,SMCMAX),.02) MIN(NaN,SMCMAX) MIN(SMCMAX,NaN) MAX(NaN,SMCMAX)" // &
                 " MAX(SMCMAX,NaN) MAX(+0,-0) MAX(-0,+0) MIN(+0,-0) MIN(-0,+0) MAX(0.,NaN)"
  write(*,"(10(z8.8,1x))") r
end program
