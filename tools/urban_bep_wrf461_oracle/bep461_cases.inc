! Extended BEP column set for the WRF v4.6.1 oracle (run_bep461_*.F90 and
! run_bep461_couple_*.F90; included in each program's CONTAINS section,
! host-associated arrays).
!
! Every urban class of the table set crosses sixteen weather regimes; the
! "nlcd" set (3 classes) runs the crossing twice with a different urban
! fraction and morphology variant, so each table set has at least 64 urban
! columns.  Four edge columns close each set.
!
! Regimes (ir):
!   1 convective day            2 stable night            3 low sun, windy
!   4 gridded mid-rise          5 near-calm noon          6 dawn, rotated wind
!   7 high terrain, noon        8 high terrain, cold night
!   9 snow-season cold, inversion                        10 coastal gale, humid
!  11 hot desert afternoon      12 tall towers to 75 m    13 exactly zero wind, night
!  14 sun on the horizon        15 overcast, rain-cooled  16 strong night inversion, calm
! Morphology variants (morph):
!   0 class defaults            1 mid-rise histogram + hgt/lp/lb
!   2 towers: histogram reaching the 15th 5 m bin (75 m)
!   3 histogram only (hgt = lp = lb = 0)
!   4 hgt/lp/lb set, histogram all zero
!   5 lp/frc >= 1: param() falls back to the class widths
!   6 lp >= lb: param() falls back to the class widths
!   7 one negative histogram bin: icBEPHI_XY rejects it (go to 20)
! Edge columns: FRC_URB2D = 0 on UTYPE 0 (BEP must skip it), 1.0e-3 and 0.02
! on class 1, exactly 1.0 on the last class.
  subroutine define_cases()
    integer :: nclass, nreg, nrep, ic, ir, irep, n
    nreg = 16
    if (use_wudapt_lcz == 0) then
      nclass = 3; nrep = 2
    else
      nclass = 11; nrep = 1
    end if
    ncol = nclass * nreg * nrep + 4
    allocate(utype_in(ncol), tsoil1(ncol), reg_in(ncol), rep_in(ncol))
    n = 0
    do irep = 1, nrep
      do ic = 1, nclass
        do ir = 1, nreg
          n = n + 1
          utype_in(n) = ic
          reg_in(n) = ir
          rep_in(n) = irep
        end do
      end do
    end do
    utype_in(n + 1) = 0;      reg_in(n + 1) = 1; rep_in(n + 1) = 1
    utype_in(n + 2) = 1;      reg_in(n + 2) = 2; rep_in(n + 2) = 1
    utype_in(n + 3) = 1;      reg_in(n + 3) = 7; rep_in(n + 3) = 1
    utype_in(n + 4) = nclass; reg_in(n + 4) = 11; rep_in(n + 4) = 1
  end subroutine define_cases

  subroutine build_columns()
    integer :: ir, kk, lay, morph, irep, ilu
    real :: zc, zbot, th0, lapse, ws, wd, pbot, tsfc_off, frc, wjit
    real :: dzv(nz)
    ! Stretched levels: fine in the canopy, coarse aloft.
    do kk = 1, nz
      if (kk <= 6) then
        dzv(kk) = 8. + 2. * real(kk - 1)
      else
        dzv(kk) = 18. * 1.12**(kk - 6)
      end if
    end do
    do icol = 1, ncol
      ir = reg_in(icol)
      irep = rep_in(icol)
      utype_urb2d(icol, 1) = utype_in(icol)
      pbot = 99000.
      select case (ir)
      case (1)
        th0 = 297.; lapse = 0.004; ws = 5.; wd = 0.35
        cosz_urb2d(icol, 1) = 0.82; swdown(icol, 1) = 780.; glw(icol, 1) = 370.
        omg_urb2d(icol, 1) = 0.21; frc = 0.9; tsfc_off = 6.; morph = 0
      case (2)
        th0 = 291.; lapse = 0.008; ws = 1.3; wd = 2.4
        cosz_urb2d(icol, 1) = -0.35; swdown(icol, 1) = 0.; glw(icol, 1) = 305.
        omg_urb2d(icol, 1) = 2.7; frc = 0.5; tsfc_off = -3.; morph = 0
      case (3)
        th0 = 294.; lapse = 0.003; ws = 14.; wd = 4.1
        cosz_urb2d(icol, 1) = 0.09; swdown(icol, 1) = 60.; glw(icol, 1) = 330.
        omg_urb2d(icol, 1) = -1.35; frc = 0.99; tsfc_off = 1.; morph = 3
      case (4)
        th0 = 299.; lapse = 0.002; ws = 7.; wd = 5.5
        cosz_urb2d(icol, 1) = 0.55; swdown(icol, 1) = 520.; glw(icol, 1) = 360.
        omg_urb2d(icol, 1) = -0.6; frc = 0.7; tsfc_off = 4.; morph = 1
      case (5)
        th0 = 301.; lapse = 0.001; ws = 0.004; wd = 1.0
        cosz_urb2d(icol, 1) = 0.97; swdown(icol, 1) = 950.; glw(icol, 1) = 390.
        omg_urb2d(icol, 1) = 0.02; frc = 0.8; tsfc_off = 9.; morph = 0
      case (6)
        th0 = 289.; lapse = 0.006; ws = 3.; wd = 3.3
        cosz_urb2d(icol, 1) = 0.21; swdown(icol, 1) = 150.; glw(icol, 1) = 310.
        omg_urb2d(icol, 1) = -1.9; frc = 0.6; tsfc_off = -1.; morph = 4
      case (7)
        ! ~2 km plateau city at noon: thin air, strong sun, dry sky
        pbot = 78500.
        th0 = 306.; lapse = 0.005; ws = 6.5; wd = 1.7
        cosz_urb2d(icol, 1) = 0.93; swdown(icol, 1) = 1060.; glw(icol, 1) = 265.
        omg_urb2d(icol, 1) = 0.05; frc = 0.85; tsfc_off = 11.; morph = 1
      case (8)
        ! high basin city on a clear night
        pbot = 72000.
        th0 = 288.; lapse = 0.011; ws = 4.; wd = 5.0
        cosz_urb2d(icol, 1) = -0.62; swdown(icol, 1) = 0.; glw(icol, 1) = 205.
        omg_urb2d(icol, 1) = -2.9; frc = 0.65; tsfc_off = -7.; morph = 6
      case (9)
        ! snow-season afternoon: cold air, low sun, an elevated inversion
        pbot = 101300.
        th0 = 258.; lapse = 0.012; ws = 2.5; wd = 0.9
        cosz_urb2d(icol, 1) = 0.24; swdown(icol, 1) = 115.; glw(icol, 1) = 188.
        omg_urb2d(icol, 1) = 0.85; frc = 0.75; tsfc_off = -4.; morph = 0
      case (10)
        ! coastal city in an onshore gale, cloudy and humid
        pbot = 101600.
        th0 = 293.; lapse = 0.0015; ws = 22.; wd = 0.8
        cosz_urb2d(icol, 1) = 0.6; swdown(icol, 1) = 290.; glw(icol, 1) = 405.
        omg_urb2d(icol, 1) = -0.4; frc = 0.4; tsfc_off = 0.5; morph = 5
      case (11)
        ! hot desert afternoon, superadiabatic near the ground
        pbot = 96000.
        th0 = 316.; lapse = -0.002; ws = 4.; wd = 2.0
        cosz_urb2d(icol, 1) = 0.88; swdown(icol, 1) = 1010.; glw(icol, 1) = 380.
        omg_urb2d(icol, 1) = 0.55; frc = 0.95; tsfc_off = 14.; morph = 0
      case (12)
        ! downtown towers, windy day
        th0 = 296.; lapse = 0.003; ws = 9.; wd = 3.9
        cosz_urb2d(icol, 1) = 0.71; swdown(icol, 1) = 640.; glw(icol, 1) = 350.
        omg_urb2d(icol, 1) = 0.33; frc = 1.0; tsfc_off = 5.; morph = 2
      case (13)
        ! wind exactly zero on every level, night
        th0 = 287.; lapse = 0.009; ws = 0.; wd = 0.
        cosz_urb2d(icol, 1) = -0.15; swdown(icol, 1) = 0.; glw(icol, 1) = 300.
        omg_urb2d(icol, 1) = 2.1; frc = 0.55; tsfc_off = -2.; morph = 0
      case (14)
        ! sun on the horizon: tiny positive cosz, a little diffuse light
        th0 = 290.; lapse = 0.004; ws = 2.; wd = 1.2
        cosz_urb2d(icol, 1) = 1.e-4; swdown(icol, 1) = 1.5; glw(icol, 1) = 315.
        omg_urb2d(icol, 1) = -1.62; frc = 0.88; tsfc_off = 0.; morph = 7
      case (15)
        ! overcast, rain-cooled noon: diffuse light, warm sky
        th0 = 292.; lapse = 0.0045; ws = 5.5; wd = 4.6
        cosz_urb2d(icol, 1) = 0.74; swdown(icol, 1) = 95.; glw(icol, 1) = 420.
        omg_urb2d(icol, 1) = -0.12; frc = 0.72; tsfc_off = -0.6; morph = 3
      case default
        ! strong surface inversion, near calm, clear night
        th0 = 279.; lapse = 0.025; ws = 0.6; wd = 3.0
        cosz_urb2d(icol, 1) = -0.81; swdown(icol, 1) = 0.; glw(icol, 1) = 240.
        omg_urb2d(icol, 1) = 3.05; frc = 0.92; tsfc_off = -9.; morph = 4
      end select
      ! second crossing (nlcd only): other fraction, other morphology
      if (irep == 2) then
        frc = max(0.05, min(1.0, 1.15 - frc))
        morph = mod(morph + 3, 8)
      end if
      frc_urb2d(icol, 1) = frc
      if (icol == ncol - 3) frc_urb2d(icol, 1) = 0.
      if (icol == ncol - 2) frc_urb2d(icol, 1) = 1.e-3
      if (icol == ncol - 1) frc_urb2d(icol, 1) = 0.02
      if (icol == ncol) frc_urb2d(icol, 1) = 1.
      if (icol > ncol - 4) morph = 0

      wjit = 1. + 0.03 * sin(1.3 * real(icol))
      zbot = 0.
      do kk = 1, nz
        dz8w(icol, kk, 1) = dzv(kk)
        zc = zbot + 0.5 * dzv(kk)
        th_phy(icol, kk, 1) = th0 + lapse * zc + 0.3 * sin(0.7 * real(kk + icol))
        if (ir == 16) th_phy(icol, kk, 1) = th0 + 12. * (1. - exp(-zc / 120.)) &
                                          + 0.002 * zc
        u_phy(icol, kk, 1) = wjit * ws * log(1. + zc / 2.) / log(1. + 400. / 2.) &
                             * cos(wd + 0.002 * zc)
        v_phy(icol, kk, 1) = wjit * ws * log(1. + zc / 2.) / log(1. + 400. / 2.) &
                             * sin(wd + 0.002 * zc)
        if (ir == 13) then
          u_phy(icol, kk, 1) = 0.; v_phy(icol, kk, 1) = 0.
        end if
        p_phy(icol, kk, 1) = pbot * exp(-zc / 8400.)
        rho(icol, kk, 1) = p_phy(icol, kk, 1) / (287. * th_phy(icol, kk, 1) * &
                           (p_phy(icol, kk, 1) / 1.e5)**0.2857)
        zbot = zbot + dzv(kk)
      end do
      dz8w(icol, nz + 1, 1) = 0.
      u_phy(icol, nz + 1, 1) = 0.; v_phy(icol, nz + 1, 1) = 0.
      th_phy(icol, nz + 1, 1) = 0.; rho(icol, nz + 1, 1) = 0.; p_phy(icol, nz + 1, 1) = 0.

      lp_urb2d(icol, 1) = 0.; lb_urb2d(icol, 1) = 0.; hgt_urb2d(icol, 1) = 0.
      hi_urb2d(icol, :, 1) = 0.
      select case (morph)
      case (1)
        hgt_urb2d(icol, 1) = 12. + 3. * real(utype_in(icol))
        lp_urb2d(icol, 1) = 0.35; lb_urb2d(icol, 1) = 1.4
        hi_urb2d(icol, 1:6, 1) = [0.10, 0.25, 0.30, 0.20, 0.10, 0.05]
      case (2)
        hgt_urb2d(icol, 1) = 38. + 2. * real(utype_in(icol))
        lp_urb2d(icol, 1) = 0.48; lb_urb2d(icol, 1) = 2.6
        do ilu = 1, num_urban_hi
          hi_urb2d(icol, ilu, 1) = 0.02 + 0.01 * real(mod(ilu * 7 + icol, 5))
        end do
        hi_urb2d(icol, num_urban_hi, 1) = 0.09
      case (3)
        hi_urb2d(icol, 2:5, 1) = [0.40, 0.35, 0.15, 0.10]
      case (4)
        hgt_urb2d(icol, 1) = 9. + real(utype_in(icol))
        lp_urb2d(icol, 1) = 0.30; lb_urb2d(icol, 1) = 1.1
      case (5)
        hgt_urb2d(icol, 1) = 15.
        lp_urb2d(icol, 1) = frc_urb2d(icol, 1) * 1.05; lb_urb2d(icol, 1) = 1.9
        hi_urb2d(icol, 1:4, 1) = [0.20, 0.30, 0.30, 0.20]
      case (6)
        hgt_urb2d(icol, 1) = 11.
        lp_urb2d(icol, 1) = 0.42; lb_urb2d(icol, 1) = 0.40
        hi_urb2d(icol, 1:3, 1) = [0.50, 0.30, 0.20]
      case (7)
        hgt_urb2d(icol, 1) = 14.
        lp_urb2d(icol, 1) = 0.33; lb_urb2d(icol, 1) = 1.25
        hi_urb2d(icol, 1:4, 1) = [0.30, -0.05, 0.45, 0.30]
      case default
      end select

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
