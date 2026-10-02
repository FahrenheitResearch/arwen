module gwd_cases
  ! The columns both orographic-drag oracles (run_gwdo, run_gwdo_gsl) drive
  ! WRF's schemes with.  Nothing here is WRF code and nothing here computes an
  ! expected value: it builds inputs, which the drivers write out beside the
  ! words WRF's routines return, so the port reads its inputs from the
  ! fixture rather than regenerating them.
  !
  ! Layout is WRF's memory order, (ims:ime, kms:kme, jms:jme) with one j row:
  ! i = column, k = 1..nz mass levels (nz+1 for interfaces), j = 1.
  !
  ! The first columns are edge cases chosen to reach branches; the rest come
  ! from a fixed generator over stability, shear, wind direction (all eight
  ! of the schemes' wind sectors), terrain height and the orographic
  ! statistics.
  implicit none
  integer, parameter :: nz = 40, nzi = nz + 1, ncol = 64
  real, parameter :: g = 9.81, r_d = 287.0, r_v = 461.6
  real, parameter :: cp = 7.0 * r_d / 2.0, ep_1 = r_v / r_d - 1.0
  real, parameter :: p_top = 5000.0

  real :: u3d(ncol, nzi, 1), v3d(ncol, nzi, 1), t3d(ncol, nzi, 1)
  real :: qv3d(ncol, nzi, 1), p3d(ncol, nzi, 1), p3di(ncol, nzi, 1)
  real :: pi3d(ncol, nzi, 1), z(ncol, nzi, 1), dz(ncol, nzi, 1)
  real :: rub0(ncol, nzi, 1), rvb0(ncol, nzi, 1), rthb0(ncol, nzi, 1)
  real :: znu(nzi), znw(nzi)
  real :: var2d(ncol, 1), oc12d(ncol, 1), oa(ncol, 1, 4), ol(ncol, 1, 4)
  real :: varss(ncol, 1), ocss(ncol, 1), oass(ncol, 1, 4), olss(ncol, 1, 4)
  real :: sina(ncol, 1), cosa(ncol, 1), xland(ncol, 1), br(ncol, 1)
  real :: pblh(ncol, 1), ht(ncol)
  integer :: kpbl(ncol, 1)

contains

  real function lcg(state)
    integer, intent(inout) :: state
    integer(8) :: wide
    wide = modulo(1103515245_8 * int(state, 8) + 12345_8, 2147483647_8)
    state = int(wide)
    lcg = real(modulo(state, 1000003)) / 1000003.0
  end function lcg

  subroutine build_columns()
    integer :: i, k, m, seed, stab
    real :: zw(nzi), zc(nz), dzk(nz), theta, lapse_lo, lapse_hi, zinv, jump
    real :: speed, shear, dirdeg, dirrad, uu, vv, qv0, psfc, alpha
    real :: crit, s

    zw(1) = 0.0
    do k = 1, nz
      dzk(k) = 40.0 + 20.0 * real(k - 1)
      zw(k + 1) = zw(k) + dzk(k)
      zc(k) = 0.5 * (zw(k) + zw(k + 1))
    end do

    seed = 2718281
    do i = 1, ncol
      ! --- column controls ------------------------------------------------
      stab = modulo(i - 1, 4)
      select case (stab)
      case (0)            ! convective mixed layer under an inversion
        lapse_lo = -0.0008; zinv = 1400.0; jump = 3.0; lapse_hi = 0.0045
      case (1)            ! stable night: surface inversion
        lapse_lo = 0.012; zinv = 350.0; jump = 0.5; lapse_hi = 0.0040
      case (2)            ! near neutral
        lapse_lo = 0.0005; zinv = 900.0; jump = 1.5; lapse_hi = 0.0035
      case default        ! very stable deep layer (blocking)
        lapse_lo = 0.020; zinv = 2500.0; jump = 0.0; lapse_hi = 0.0080
      end select
      speed = 2.0 + 22.0 * lcg(seed)
      shear = 0.0025 * (lcg(seed) - 0.2)
      dirdeg = 360.0 * lcg(seed)
      ht(i) = 2800.0 * lcg(seed) ** 2
      crit = 0.0
      if (modulo(i, 7) == 3) crit = 3500.0 + 4000.0 * lcg(seed)  ! reversal aloft
      qv0 = 0.002 + 0.014 * lcg(seed)
      alpha = 0.35 * (lcg(seed) - 0.5)
      sina(i, 1) = sin(alpha)
      cosa(i, 1) = cos(alpha)
      xland(i, 1) = 1.0
      if (modulo(i, 11) == 5) xland(i, 1) = 2.0
      s = lcg(seed)
      br(i, 1) = -0.6 + 1.2 * s
      pblh(i, 1) = 60.0 + 2600.0 * lcg(seed)
      ! --- orographic statistics -------------------------------------------
      var2d(i, 1) = 1300.0 * lcg(seed) ** 2
      oc12d(i, 1) = 8.0 * lcg(seed) ** 2
      varss(i, 1) = 320.0 * lcg(seed) ** 2
      ocss(i, 1) = 8.0 * lcg(seed) ** 2
      do m = 1, 4
        oa(i, 1, m) = 2.0 * lcg(seed) - 1.0
        ol(i, 1, m) = lcg(seed)
        oass(i, 1, m) = 2.0 * lcg(seed) - 1.0
        olss(i, 1, m) = lcg(seed)
      end do

      ! --- edge cases on the first columns ---------------------------------
      select case (i)
      case (1)            ! calm: ulow clamps to exactly 1 -> no drag
        speed = 0.0; shear = 0.0
      case (2)            ! no sub-grid variance at all
        var2d(i, 1) = 0.0; varss(i, 1) = 0.0
      case (3)            ! wind reversed between the two lowest levels
        crit = 60.0
      case (4)            ! blocking: deep stable layer, weak wind, tall terrain
        speed = 3.0; shear = 0.0; var2d(i, 1) = 900.0; ht(i) = 1500.0
        ol(i, 1, :) = 0.8
      case (5)            ! water with large statistics: SS and TOFD skip it
        xland(i, 1) = 2.0; varss(i, 1) = 200.0
      case (6)            ! 2*varss above the PBL height: no small-scale wave
        varss(i, 1) = 300.0; pblh(i, 1) = 400.0
      case (7)            ! zero orographic lengths (olmin floor)
        ol(i, 1, :) = 0.0
      case (8)            ! stable surface layer, strong wind, small varss
        br(i, 1) = 0.3; varss(i, 1) = 12.0; speed = 18.0
      end select
      dirrad = dirdeg * 3.14159265 / 180.0

      ! --- profiles --------------------------------------------------------
      psfc = 101325.0 * exp(-ht(i) / 8200.0)
      do k = 1, nzi
        p3di(i, k, 1) = psfc * exp(-zw(k) / 8500.0)
      end do
      do k = 1, nz
        if (zc(k) <= zinv) then
          theta = 295.0 + lapse_lo * zc(k)
        else
          theta = 295.0 + lapse_lo * zinv + jump + lapse_hi * (zc(k) - zinv)
        end if
        if (zc(k) > 11000.0) theta = theta + 0.02 * (zc(k) - 11000.0)
        uu = (speed + shear * zc(k)) * sin(dirrad)
        vv = (speed + shear * zc(k)) * cos(dirrad)
        if (crit > 0.0 .and. zc(k) > crit) then
          uu = -0.6 * uu; vv = -0.6 * vv
        end if
        u3d(i, k, 1) = uu
        v3d(i, k, 1) = vv
        qv3d(i, k, 1) = qv0 * exp(-zc(k) / 2500.0)
        p3d(i, k, 1) = 0.5 * (p3di(i, k, 1) + p3di(i, k + 1, 1))
        pi3d(i, k, 1) = (p3d(i, k, 1) / 100000.0) ** (r_d / cp)
        t3d(i, k, 1) = theta * pi3d(i, k, 1)
        z(i, k, 1) = ht(i) + zc(k)
        dz(i, k, 1) = dzk(k)
        rub0(i, k, 1) = 1.0e-4 * cos(0.3 * real(k + i))
        rvb0(i, k, 1) = -2.0e-4 * sin(0.2 * real(k + 2 * i))
        rthb0(i, k, 1) = 3.0e-5 * cos(0.1 * real(k * i))
      end do
      ! The top interface/padding level of the 3-D arrays is never read as a
      ! mass level; give it defined values all the same.
      u3d(i, nzi, 1) = 0.0; v3d(i, nzi, 1) = 0.0; t3d(i, nzi, 1) = 0.0
      qv3d(i, nzi, 1) = 0.0; p3d(i, nzi, 1) = 0.0; pi3d(i, nzi, 1) = 0.0
      z(i, nzi, 1) = 0.0; dz(i, nzi, 1) = 0.0
      rub0(i, nzi, 1) = 0.0; rvb0(i, nzi, 1) = 0.0; rthb0(i, nzi, 1) = 0.0
      ! The PBL top level the PBL scheme would report (first level above).
      kpbl(i, 1) = nz
      do k = 1, nz
        if (zc(k) > pblh(i, 1)) then
          kpbl(i, 1) = k
          exit
        end if
      end do
    end do

    ! Eta at the mass levels and interfaces of a sea-level column (znu only
    ! fixes kpblmax = 1 + the last level with znu > 0.6 in the GSL scheme).
    do k = 1, nzi
      znw(k) = (101325.0 * exp(-zw(k) / 8500.0) - p_top) / (101325.0 - p_top)
    end do
    do k = 1, nz
      znu(k) = 0.5 * (znw(k) + znw(k + 1))
    end do
    znu(nzi) = 0.0
  end subroutine build_columns

end module gwd_cases
