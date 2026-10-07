! Drive GSL's get_niwfa (mp_thompson.F90:1025-1068) on fixed aerosol columns
! and record inputs and outputs.  aerfld(ncol, nlev, 15) is kg/kg in the
! MERRA-2 bin layout the routine documents; only bins 1-11 and 15 are read.
! The outputs are REAL(kind_phys) = REAL(8); they are written as the REAL(4)
! the engine stores (nwfa/nifa are float32 state), rounded to nearest by the
! assignment, and ALSO as the two float32 halves of the float64 bit pattern so
! a port can be graded on the double before it is rounded.
program niwfa_driver
  use niwfa_harness, only: get_niwfa, kind_phys
  use oracle_io
  implicit none
  integer, parameter :: ncol = 6, nlev = 5
  real(kind_phys) :: aerfld(ncol, nlev, 15), nifa(ncol, nlev), nwfa(ncol, nlev)
  real(4) :: a4(ncol, nlev, 15), nifa4(ncol, nlev), nwfa4(ncol, nlev)
  integer(4) :: nifa_bits(2, ncol, nlev), nwfa_bits(2, ncol, nlev)
  character(len=1024) :: root
  integer :: i, k, b
  real(kind_phys) :: base

  call oracle_root(root)
  ! A deterministic spread of magnitudes: clean (1e-12 kg/kg), continental
  ! (1e-9), polluted (1e-8), dust storm (1e-6 .. 1e-4) and exact zeros, one
  ! column each, levels scaling the column by a different factor per bin so no
  ! two bins carry the same number.
  do i = 1, ncol
    select case (i)
    case (1); base = 1.0e-12_kind_phys
    case (2); base = 1.0e-9_kind_phys
    case (3); base = 1.0e-8_kind_phys
    case (4); base = 1.0e-6_kind_phys
    case (5); base = 1.0e-4_kind_phys
    case default; base = 0.0_kind_phys
    end select
    do k = 1, nlev
      do b = 1, 15
        aerfld(i, k, b) = base * (1.0_kind_phys + 0.37_kind_phys * b) &
                          / (1.0_kind_phys + 0.5_kind_phys * (k - 1))
      end do
    end do
  end do
  ! column 6: only sulfate and OC nonzero, the dust and sea-salt terms are +0
  aerfld(6, :, 11) = 3.0e-9_kind_phys
  aerfld(6, :, 15) = 2.0e-9_kind_phys
  ! aerfld is input in kg/kg as float64; the port receives it as float32
  ! species converted to kg/kg in float64, so the fixture records the exact
  ! float32 values the port will be handed and the driver uses those.
  a4 = real(aerfld, 4)
  aerfld = real(a4, kind_phys)

  call get_niwfa(aerfld, nifa, nwfa, ncol, nlev)

  nifa4 = real(nifa, 4)
  nwfa4 = real(nwfa, 4)
  nifa_bits = reshape(transfer(nifa, nifa_bits), shape(nifa_bits))
  nwfa_bits = reshape(transfer(nwfa, nwfa_bits), shape(nwfa_bits))

  call oracle_open('niwfa')
  call oracle_put('aerfld', a4)
  call oracle_put('nifa', nifa4)
  call oracle_put('nwfa', nwfa4)
  call oracle_put('nifa_f64_bits', nifa_bits)
  call oracle_put('nwfa_f64_bits', nwfa_bits)
  call oracle_close()
end program niwfa_driver
