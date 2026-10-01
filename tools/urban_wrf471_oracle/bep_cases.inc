! Shared BEP fixture columns for run_bep.F90 and run_bep_couple.F90
! (included in each program's CONTAINS section; host-associated arrays).
  subroutine define_cases()
    ! nlcd: 3 classes x 6 regimes + 3 non-urban/edge columns.
    ! lcz : 11 classes x 3 regimes + 3 edge columns.
    integer :: nclass, nreg, ic, ir, n
    if (use_wudapt_lcz == 0) then
      nclass = 3; nreg = 6
    else
      nclass = 11; nreg = 3
    end if
    ncol = nclass * nreg + 3
    allocate(utype_in(ncol), tsoil1(ncol))
    n = 0
    do ic = 1, nclass
      do ir = 1, nreg
        n = n + 1
        utype_in(n) = ic
      end do
    end do
    utype_in(n + 1) = 0          ! non-urban: FRC_URB2D = 0, BEP must skip
    utype_in(n + 2) = 1          ! tiny urban fraction
    utype_in(n + 3) = nclass     ! urban fraction exactly 1
  end subroutine define_cases

  subroutine build_columns()
    integer :: nreg, ir, kk, lay
    real :: zc, zbot, th0, lapse, ws, wd, pbot, tsfc_off
    real :: dzv(nz)
    ! Stretched levels: fine in the canopy, coarse aloft.
    do kk = 1, nz
      if (kk <= 6) then
        dzv(kk) = 8. + 2. * real(kk - 1)
      else
        dzv(kk) = 18. * 1.12**(kk - 6)
      end if
    end do
    if (use_wudapt_lcz == 0) then
      nreg = 6
    else
      nreg = 3
    end if
    do icol = 1, ncol
      if (icol <= ncol - 3) then
        ir = mod(icol - 1, 6) + 1   ! all six regimes in both table sets
      else
        ir = icol - (ncol - 3)
      end if
      utype_urb2d(icol, 1) = utype_in(icol)
      ! regime table: 1 day unstable, 2 night stable, 3 low sun windy,
      ! 4 gridded morphology, 5 near-calm noon, 6 dawn rotated street wind
      select case (ir)
      case (1)
        th0 = 297.; lapse = 0.004; ws = 5.; wd = 0.35
        cosz_urb2d(icol, 1) = 0.82; swdown(icol, 1) = 780.; glw(icol, 1) = 370.
        omg_urb2d(icol, 1) = 0.21; frc_urb2d(icol, 1) = 0.9; tsfc_off = 6.
      case (2)
        th0 = 291.; lapse = 0.008; ws = 1.3; wd = 2.4
        cosz_urb2d(icol, 1) = -0.35; swdown(icol, 1) = 0.; glw(icol, 1) = 305.
        omg_urb2d(icol, 1) = 2.7; frc_urb2d(icol, 1) = 0.5; tsfc_off = -3.
      case (3)
        th0 = 294.; lapse = 0.003; ws = 14.; wd = 4.1
        cosz_urb2d(icol, 1) = 0.09; swdown(icol, 1) = 60.; glw(icol, 1) = 330.
        omg_urb2d(icol, 1) = -1.35; frc_urb2d(icol, 1) = 0.99; tsfc_off = 1.
      case (4)
        th0 = 299.; lapse = 0.002; ws = 7.; wd = 5.5
        cosz_urb2d(icol, 1) = 0.55; swdown(icol, 1) = 520.; glw(icol, 1) = 360.
        omg_urb2d(icol, 1) = -0.6; frc_urb2d(icol, 1) = 0.7; tsfc_off = 4.
      case (5)
        th0 = 301.; lapse = 0.001; ws = 0.004; wd = 1.0
        cosz_urb2d(icol, 1) = 0.97; swdown(icol, 1) = 950.; glw(icol, 1) = 390.
        omg_urb2d(icol, 1) = 0.02; frc_urb2d(icol, 1) = 0.8; tsfc_off = 9.
      case default
        th0 = 289.; lapse = 0.006; ws = 3.; wd = 3.3
        cosz_urb2d(icol, 1) = 0.21; swdown(icol, 1) = 150.; glw(icol, 1) = 310.
        omg_urb2d(icol, 1) = -1.9; frc_urb2d(icol, 1) = 0.6; tsfc_off = -1.
      end select
      if (icol == ncol - 2) frc_urb2d(icol, 1) = 0.
      if (icol == ncol - 1) frc_urb2d(icol, 1) = 0.02
      if (icol == ncol) frc_urb2d(icol, 1) = 1.

      zbot = 0.
      pbot = 99000.
      do kk = 1, nz
        dz8w(icol, kk, 1) = dzv(kk)
        zc = zbot + 0.5 * dzv(kk)
        th_phy(icol, kk, 1) = th0 + lapse * zc + 0.3 * sin(0.7 * real(kk + icol))
        ! log-ish wind profile turning with height
        u_phy(icol, kk, 1) = ws * log(1. + zc / 2.) / log(1. + 400. / 2.) * cos(wd + 0.002 * zc)
        v_phy(icol, kk, 1) = ws * log(1. + zc / 2.) / log(1. + 400. / 2.) * sin(wd + 0.002 * zc)
        p_phy(icol, kk, 1) = pbot * exp(-zc / 8400.)
        rho(icol, kk, 1) = p_phy(icol, kk, 1) / (287. * th_phy(icol, kk, 1) * &
                           (p_phy(icol, kk, 1) / 1.e5)**0.2857)
        zbot = zbot + dzv(kk)
      end do
      dz8w(icol, nz + 1, 1) = 0.
      u_phy(icol, nz + 1, 1) = 0.; v_phy(icol, nz + 1, 1) = 0.
      th_phy(icol, nz + 1, 1) = 0.; rho(icol, nz + 1, 1) = 0.; p_phy(icol, nz + 1, 1) = 0.

      ! Morphology: default (all zero) except the gridded regime (4).
      lp_urb2d(icol, 1) = 0.; lb_urb2d(icol, 1) = 0.; hgt_urb2d(icol, 1) = 0.
      hi_urb2d(icol, :, 1) = 0.
      if (ir == 4 .and. icol <= ncol - 3) then
        hgt_urb2d(icol, 1) = 12. + 3. * real(utype_in(icol))
        lp_urb2d(icol, 1) = 0.35
        lb_urb2d(icol, 1) = 1.4
        hi_urb2d(icol, 1, 1) = 0.10
        hi_urb2d(icol, 2, 1) = 0.25
        hi_urb2d(icol, 3, 1) = 0.30
        hi_urb2d(icol, 4, 1) = 0.20
        hi_urb2d(icol, 5, 1) = 0.10
        hi_urb2d(icol, 6, 1) = 0.05
      end if

      ! urban_var_init BEP arm, then a recorded perturbation.
      tsoil1(icol) = th0 - 2.5 + tsfc_off * 0.5
      if (utype_urb2d(icol, 1) > 0) then
        trb_urb4d(icol, :, 1) = tblend_tbl(utype_urb2d(icol, 1))
        tw1_urb4d(icol, :, 1) = tblend_tbl(utype_urb2d(icol, 1))
        tw2_urb4d(icol, :, 1) = tblend_tbl(utype_urb2d(icol, 1))
      else
        trb_urb4d(icol, :, 1) = tsoil1(icol)
        tw1_urb4d(icol, :, 1) = tsoil1(icol)
        tw2_urb4d(icol, :, 1) = tsoil1(icol)
      end if
      tgb_urb4d(icol, :, 1) = tsoil1(icol)
      sfw1_urb3d(icol, :, 1) = 0.; sfw2_urb3d(icol, :, 1) = 0.
      sfr_urb3d(icol, :, 1) = 0.; sfg_urb3d(icol, :, 1) = 0.
      do lay = 1, urban_map_zwd
        trb_urb4d(icol, lay, 1) = trb_urb4d(icol, lay, 1) + tsfc_off * &
          (1. + 0.4 * sin(0.37 * real(lay) + real(icol)))
        tw1_urb4d(icol, lay, 1) = tw1_urb4d(icol, lay, 1) + 0.8 * tsfc_off * &
          (1. + 0.3 * cos(0.23 * real(lay) + real(icol)))
        tw2_urb4d(icol, lay, 1) = tw2_urb4d(icol, lay, 1) + 0.6 * tsfc_off * &
          (1. + 0.3 * sin(0.29 * real(lay) - real(icol)))
      end do
      do lay = 1, urban_map_gd
        tgb_urb4d(icol, lay, 1) = tgb_urb4d(icol, lay, 1) + tsfc_off * &
          exp(-0.25 * real((lay + 1) / 2 - 1))
      end do
    end do
  end subroutine build_columns
