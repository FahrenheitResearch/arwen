! External dump routines for the DIAGNOSTIC instrumented Milbrandt-Yau copy
! (instrument.py).  Never linked into the oracle.  Appends to my2dump.bin:
!   state record:  int32 kind=1, tag, ni, nk; 21 (ni,nk) float32 arrays
!                  T Q QC QR QI QN QG QH NC NR NY NN NG NH DE QSW QSI pres
!                  gamfact DZ iDZ
!   const record:  int32 kind=2, which, n; n float32
subroutine my2dump_open(u)
  implicit none
  integer, intent(out) :: u
  logical, save :: first = .true.
  if (first) then
     open(newunit=u, file='my2dump.bin', access='stream', form='unformatted', &
          status='replace', convert='little_endian')
     first = .false.
  else
     open(newunit=u, file='my2dump.bin', access='stream', form='unformatted', &
          status='old', position='append', convert='little_endian')
  end if
end subroutine my2dump_open

subroutine my2dump_state(tag, ni, nk, T, Q, QC, QR, QI, QN, QG, QH, NC, NR, NY, &
     NN, NG, NH, DE, QSW, QSI, pres, gamfact, DZ, iDZ)
  implicit none
  integer, intent(in) :: tag, ni, nk
  real, dimension(ni,nk), intent(in) :: T, Q, QC, QR, QI, QN, QG, QH, NC, NR, &
       NY, NN, NG, NH, DE, QSW, QSI, pres, gamfact, DZ, iDZ
  integer :: u
  call my2dump_open(u)
  write(u) 1, tag, ni, nk
  write(u) T, Q, QC, QR, QI, QN, QG, QH, NC, NR, NY, NN, NG, NH, DE, QSW, QSI, &
       pres, gamfact, DZ, iDZ
  close(u)
end subroutine my2dump_state

subroutine my2dump_const(which, n, vals)
  implicit none
  integer, intent(in) :: which, n
  real, intent(in) :: vals(n)
  integer :: u
  call my2dump_open(u)
  write(u) 2, which, n
  write(u) vals
  close(u)
end subroutine my2dump_const
