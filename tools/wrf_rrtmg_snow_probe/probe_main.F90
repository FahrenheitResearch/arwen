! WRF v4.6.1 RRTMG option-4 snow-radiation coupling probe.
!
! Purpose: generate a bitwise FP32 fixture for the explicit-snow-radius
! coupling (inflg/iceflg = 5) that WRF's RRTMG driver applies before its
! radiative transfer: the metre->micron conversion with a 10-micron floor,
! and the 130-micron cap with the (130/re_s)^2 snow mass discount.
!
! The two coupled statement blocks are NOT transcribed by hand.  They are
! extracted verbatim (bytes unchanged) by extract_and_build.sh from the
! read-only authority
!   WRF_source_v4.6.1_group/phys/module_ra_rrtmg_sw.F
!     lines 10823-10825  (resnow floor + unit conversion; include file
!                         wrf_resnow_floor.inc)
!     lines 11055-11067  (snow mass discount + 130 micron cap; include file
!                         wrf_snow_discount.inc)
! (file SHA-256 447345d2658cd370e6bc97ff2ab582a5d12b84adffc58f72a938b353e017987e).
! The LW block, module_ra_rrtmg_lw.F:12515-12532, is arithmetically identical
! (it adds only a debug WRITE); its floor line is module_ra_rrtmg_lw.F:12242.
!
! WRF compiles these modules with default REAL = 4 bytes (RWORDSIZE=4), so
! every quantity below is FP32, matching the group build.
!
! One probe call runs a single layer (kts = kte = 1).  Because
! snow_mass_factor, gsnowp, and gicewp are scalars in the enclosing WRF
! subroutine scope, their post-loop values expose the per-layer results
! bit-exactly with no reconstruction arithmetic.  gicewp is initialised to
! zero so its post-block value isolates the "1% of snow overlaps the cloud
! ice category" increment -- WRF computes it, but never stores it back into
! cicewp (last written at module_ra_rrtmg_sw.F:11043 / _lw.F:12503), so the
! increment is dead.  It is recorded here to document that fact.
!
! Input:  probe_inputs.txt, one row per case:
!         re_snow_m_bits qs_bits pdel_bits cldfrac_bits   (Z8 hex, FP32 bits)
! Output: probe_outputs.csv with hex FP32 bit patterns:
!         re_snow_m,resnow_floored_um,resnow_final_um,snow_mass_factor,
!         gsnowp,csnowp,gicewp_dead_increment
program wrf_rrtmg_snow_probe
  implicit none
  integer, parameter :: nmax = 100000
  integer :: kts, kte, k, ncol, i, j, iceflgsw, ios, n
  real :: snow_mass_factor, gicewp, gsnowp, gravmks
  real :: qs1d(1:1)
  real :: pdel(1,1), resnow1d(1,1), csnowp(1,1), cldfrac(1,1)
  real :: re_snow(1,1,1)
  real :: resnow_floored
  integer :: bits(4)

  ! WRF passes gravmks = g from module_model_constants.F (g = 9.81).
  gravmks = 9.81
  kts = 1
  kte = 1
  ncol = 1
  i = 1
  j = 1
  iceflgsw = 5

  open(10, file='probe_inputs.txt', status='old', action='read')
  open(11, file='probe_outputs.csv', status='replace', action='write')
  write(11,'(a)') 're_snow_m,resnow_floored_um,resnow_final_um,' // &
       'snow_mass_factor,gsnowp,csnowp,gicewp_dead_increment'
  n = 0
  do
     read(10, '(4(Z8,1X))', iostat=ios) bits
     if (ios /= 0) exit
     n = n + 1
     if (n > nmax) stop 'too many probe rows'
     re_snow(1,1,1) = transfer(bits(1), 1.0)
     qs1d(1)        = transfer(bits(2), 1.0)
     pdel(1,1)      = transfer(bits(3), 1.0)
     cldfrac(1,1)   = transfer(bits(4), 1.0)
     gicewp = 0.0
     gsnowp = 0.0
     snow_mass_factor = -1.0

     ! --- verbatim WRF statements: floor + unit conversion ---
     include 'wrf_resnow_floor.inc'
     resnow_floored = resnow1d(1,1)

     ! --- verbatim WRF statements: snow mass discount + 130 um cap ---
     include 'wrf_snow_discount.inc'

     write(11,'(6(Z8.8,","),Z8.8)') &
          transfer(re_snow(1,1,1), bits(1)), &
          transfer(resnow_floored, bits(1)), &
          transfer(resnow1d(1,1), bits(1)), &
          transfer(snow_mass_factor, bits(1)), &
          transfer(gsnowp, bits(1)), &
          transfer(csnowp(1,1), bits(1)), &
          transfer(gicewp, bits(1))
  end do
  close(10)
  close(11)
  write(*,'(a,i0,a)') 'probe: ', n, ' rows written to probe_outputs.csv'
end program wrf_rrtmg_snow_probe
