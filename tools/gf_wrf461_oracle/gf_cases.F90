module gf_cases
  ! The fixture's case table and column builder, shared by both oracle
  ! harnesses so they cannot drift apart.
  !
  !   run_cu_gf.F90  drives GFDRV, the WRF entry point -- the pinned boundary.
  !   run_cup_gf.F90 drives cup_gf directly, replicating GFDRV's own column
  !                  preparation, so the per-stage internals (ierr, xmb,
  !                  kbcon, k22, jmin, edt, forcing) that never leave the
  !                  driver become visible to a failing port.
  !
  ! Both must see byte-identical inputs or the second is not a decomposition
  ! of the first, which is why the construction lives here and not in either
  ! program.
  implicit none
  public

  integer, parameter :: gf_nz = 40
  integer, parameter :: gf_ncase = 18
  integer, parameter :: gf_ndx = 6

  ! dx sweep chosen against module_cu_gf_deep.F:463-469.  With csum = 0 and
  ! imid = 0 the deep entrainment rate is 7.e-5, so radius = .2/7.e-5 =
  ! 2857.14 m and frh = min(1, 3.14*radius^2/dx^2) hits the frh_thresh = .9
  ! clamp for dx below about 5337 m.  1000/4000 sit inside the clamp (sig ==
  ! sig_thresh == .01 exactly, the value the deep-shutoff test at :663
  ! compares against); 6000 is just outside it; 9000/15000/27000 walk sig up
  ! toward 1.
  real, parameter :: gf_dxsweep(gf_ndx) = &
       (/ 1000.0, 4000.0, 6000.0, 9000.0, 15000.0, 27000.0 /)

  ! WRF module_model_constants, single-precision build.  These are what the
  ! WRF solver hands the cumulus driver.  GFDRV itself takes g/cp/xlv/r_v from
  ! module_gfs_physcons, which does NOT agree with them (con_g = 9.80665 vs
  ! 9.81, con_cp = 1004.6 vs 1004.).  That disagreement is real WRF behaviour
  ! and is inside the pinned boundary; nothing here harmonises it.
  real, parameter :: gf_g = 9.81
  real, parameter :: gf_rd = 287.0
  real, parameter :: gf_rv = 461.6
  real, parameter :: gf_cp = 7.0 * gf_rd / 2.0
  real, parameter :: gf_rovcp = gf_rd / gf_cp
  real, parameter :: gf_p1000mb = 100000.0
  real, parameter :: svp1 = 0.6112, svp2 = 17.67, svp3 = 29.65
  real, parameter :: svpt0 = 273.15, ep_2 = gf_rd / gf_rv

  ! per-case scalars
  real, dimension(gf_ncase) :: c_tsfc, c_lapse, c_rhsfc, c_rhmid, c_ztrop
  real, dimension(gf_ncase) :: c_hfx, c_qfx, c_xland, c_ht, c_ubase, c_ushear
  real, dimension(gf_ncase) :: c_vbase, c_wamp, c_thf, c_qvf, c_thbl, c_qvbl
  real, dimension(gf_ncase) :: c_thrad, c_zcap, c_dtcap, c_capw, c_rhtop
  integer, dimension(gf_ncase) :: c_kpbl

contains

  ! WRF's own saturation form; used ONLY to build inputs, never to check an
  ! answer.  The scheme computes its own qes through satvap (Goff-Gratch,
  ! module_cu_gf_deep.F:3646) and does not see this.
  real function gf_qsat(t, p)
    real, intent(in) :: t, p
    real :: es
    es = 1000.0 * svp1 * exp(svp2 * (t - svpt0) / (t - svp3))
    gf_qsat = ep_2 * es / max(p - es, 1.0)
  end function gf_qsat

  subroutine gf_build_case_table()
    integer :: n
    ! Defaults: a moist tropical maritime sounding with a deep troposphere.
    do n = 1, gf_ncase
      c_tsfc(n) = 300.0
      c_lapse(n) = 0.0065
      c_rhsfc(n) = 0.90
      c_rhmid(n) = 0.60
      c_rhtop(n) = 0.15
      c_ztrop(n) = 15000.0
      c_hfx(n) = 120.0
      c_qfx(n) = 1.2e-4
      c_xland(n) = 2.0
      c_ht(n) = 10.0
      c_ubase(n) = 5.0
      c_ushear(n) = 0.0015
      c_vbase(n) = 1.0
      c_wamp(n) = 0.02
      c_thf(n) = 2.0e-5
      c_qvf(n) = 1.0e-8
      c_thbl(n) = 4.0e-5
      c_qvbl(n) = 2.0e-8
      c_thrad(n) = -1.5e-5
      c_kpbl(n) = 8
      c_zcap(n) = 0.0
      c_dtcap(n) = 0.0
      c_capw(n) = 120.0
    end do

    ! ---- 1-12: the phase-1 deep-convection case set -------------------------
    ! 1  tropical maritime deep convection, the reference case
    ! 2  continental deep convection: drier, hotter, land
    c_tsfc(2) = 305.0; c_rhsfc(2) = 0.70; c_rhmid(2) = 0.35
    c_xland(2) = 1.0; c_ht(2) = 350.0; c_hfx(2) = 350.0; c_qfx(2) = 1.8e-4
    c_kpbl(2) = 14; c_ubase(2) = 8.0; c_ushear(2) = 0.0025
    ! 3  strongly capped: a 3 K inversion at 1.6 km kills the parcel
    c_tsfc(3) = 302.0; c_rhsfc(3) = 0.80; c_rhmid(3) = 0.25
    c_xland(3) = 1.0; c_zcap(3) = 1600.0; c_dtcap(3) = 3.0
    c_kpbl(3) = 10
    ! 4  shallow-cumulus regime: moist BL under a sharp 1.4 km trade
    !    inversion
    c_tsfc(4) = 298.0; c_rhsfc(4) = 0.94; c_rhmid(4) = 0.20
    c_zcap(4) = 1400.0; c_dtcap(4) = 6.0; c_hfx(4) = 140.0; c_qfx(4) = 2.2e-4
    c_kpbl(4) = 5; c_wamp(4) = -0.01
    c_thbl(4) = 1.2e-4; c_qvbl(4) = 6.0e-8
    ! 5  weak forcing, marginal against the cap
    c_tsfc(5) = 297.0; c_rhsfc(5) = 0.75; c_rhmid(5) = 0.45
    c_hfx(5) = 20.0; c_qfx(5) = 2.0e-5; c_thbl(5) = 5.0e-6; c_qvbl(5) = 2.0e-9
    c_wamp(5) = 0.002
    ! 6  cold dry stable: no convection anywhere, the all-ierr column
    c_tsfc(6) = 265.0; c_lapse(6) = 0.0040; c_rhsfc(6) = 0.40
    c_rhmid(6) = 0.20; c_xland(6) = 1.0; c_ht(6) = 900.0
    c_hfx(6) = -35.0; c_qfx(6) = 0.0; c_kpbl(6) = 3
    c_thbl(6) = -2.0e-5; c_qvbl(6) = 0.0; c_wamp(6) = 0.0
    ! 7  dry mid-level layer: drives the downdraft/edt path hard
    c_tsfc(7) = 301.0; c_rhsfc(7) = 0.88; c_rhmid(7) = 0.12
    c_ubase(7) = 12.0; c_ushear(7) = 0.0030; c_vbase(7) = -6.0
    ! 8  strong moisture convergence: the Krishnamurti closure limb
    c_wamp(8) = 0.12; c_qvf(8) = 8.0e-8; c_thf(8) = 6.0e-5
    ! 9  kpbl at the floor
    c_kpbl(9) = 2; c_hfx(9) = 15.0; c_qfx(9) = 3.0e-5
    ! 10 kpbl halfway up the column
    c_kpbl(10) = gf_nz / 2; c_hfx(10) = 500.0; c_qfx(10) = 4.0e-4
    c_tsfc(10) = 306.0; c_xland(10) = 1.0
    ! 11 zero surface fluxes exactly: ztexec/zqexec both vanish
    c_hfx(11) = 0.0; c_qfx(11) = 0.0; c_thbl(11) = 0.0; c_qvbl(11) = 0.0
    ! 12 nocturnal: negative hfx over land with residual CAPE aloft
    c_tsfc(12) = 291.0; c_rhsfc(12) = 0.85; c_rhmid(12) = 0.55
    c_xland(12) = 1.0; c_ht(12) = 200.0; c_hfx(12) = -45.0; c_qfx(12) = -1.0e-6
    c_kpbl(12) = 4; c_thbl(12) = -3.0e-5; c_qvbl(12) = -1.0e-8

    ! ---- 13-18: built against the shallow trigger ---------------------------
    ! CUP_gf_sh reached only 2 of the 12 cases above.  Its gate is not the
    ! deep gate: k22 is the level of maximum moist static energy below kbmax
    ! (module_cu_gf_sh.F:376), cap_max collapses to po_cup(kpbl) as soon as
    ! kpbl > 3 (:371), and ktop comes from get_inversion_layers' 800 hPa slot
    ! or from 200 hPa above kbcon.  These six hold the moisture maximum at the
    ! surface, keep kpbl low enough for the cap not to be pinned high, and put
    ! a real second-derivative feature where the shallow ktop search looks.
    ! 13 trade cumulus: very moist shallow BL, sharp 900 m inversion
    c_tsfc(13) = 299.0; c_rhsfc(13) = 0.96; c_rhmid(13) = 0.30
    c_zcap(13) = 900.0; c_dtcap(13) = 5.0; c_capw(13) = 90.0
    c_hfx(13) = 110.0; c_qfx(13) = 2.6e-4; c_kpbl(13) = 3
    c_thbl(13) = 1.5e-4; c_qvbl(13) = 9.0e-8; c_wamp(13) = -0.015
    ! 14 deeper shallow: the same regime with a 2.2 km inversion
    c_tsfc(14) = 300.0; c_rhsfc(14) = 0.93; c_rhmid(14) = 0.35
    c_zcap(14) = 2200.0; c_dtcap(14) = 5.0; c_capw(14) = 140.0
    c_hfx(14) = 160.0; c_qfx(14) = 2.4e-4; c_kpbl(14) = 6
    c_thbl(14) = 1.3e-4; c_qvbl(14) = 7.0e-8; c_wamp(14) = -0.008
    ! 15 continental shallow: land, drier aloft, strong sensible heating
    c_tsfc(15) = 303.0; c_rhsfc(15) = 0.82; c_rhmid(15) = 0.22
    c_xland(15) = 1.0; c_ht(15) = 420.0
    c_zcap(15) = 1800.0; c_dtcap(15) = 4.0; c_capw(15) = 110.0
    c_hfx(15) = 320.0; c_qfx(15) = 1.1e-4; c_kpbl(15) = 5
    c_thbl(15) = 2.2e-4; c_qvbl(15) = 4.0e-8
    ! 16 stratocumulus-to-cumulus: a hard 8 K cap at 1.1 km
    c_tsfc(16) = 295.0; c_rhsfc(16) = 0.97; c_rhmid(16) = 0.15
    c_zcap(16) = 1100.0; c_dtcap(16) = 8.0; c_capw(16) = 70.0
    c_hfx(16) = 45.0; c_qfx(16) = 1.6e-4; c_kpbl(16) = 3
    c_thbl(16) = 6.0e-5; c_qvbl(16) = 1.1e-7; c_wamp(16) = -0.02
    ! 17 weak shallow: a moist column with barely any BL forcing, so the
    !    dhdt closure is small but nonzero
    c_tsfc(17) = 298.0; c_rhsfc(17) = 0.95; c_rhmid(17) = 0.28
    c_zcap(17) = 1300.0; c_dtcap(17) = 5.0; c_capw(17) = 90.0
    c_hfx(17) = 12.0; c_qfx(17) = 2.0e-5; c_kpbl(17) = 3
    c_thbl(17) = 8.0e-6; c_qvbl(17) = 5.0e-9; c_wamp(17) = 0.0
    ! 18 shallow under strong shear: exercises the momentum arms with a
    !    shallow cloud
    c_tsfc(18) = 299.0; c_rhsfc(18) = 0.95; c_rhmid(18) = 0.26
    c_zcap(18) = 1200.0; c_dtcap(18) = 5.5; c_capw(18) = 85.0
    c_hfx(18) = 130.0; c_qfx(18) = 2.5e-4; c_kpbl(18) = 4
    c_thbl(18) = 1.4e-4; c_qvbl(18) = 8.0e-8
    c_ubase(18) = 16.0; c_ushear(18) = 0.0045; c_vbase(18) = -9.0
  end subroutine gf_build_case_table

  ! One column's worth of state for case `ic`.
  subroutine gf_column(ic, zc, dz, tt, qq, pp, ppw, ppi, rr, uu, vv, ww)
    integer, intent(in) :: ic
    real, intent(out) :: zc(gf_nz), dz(gf_nz), tt(gf_nz), qq(gf_nz)
    real, intent(out) :: pp(gf_nz), ppw(gf_nz + 1), ppi(gf_nz), rr(gf_nz)
    real, intent(out) :: uu(gf_nz), vv(gf_nz), ww(gf_nz)
    integer :: kk
    real :: zlo, tv, rh, zfrac, tcap

    ! Stretched vertical grid: 80 m at the surface to 782 m at 17.2 km.
    zlo = 0.0
    do kk = 1, gf_nz
      dz(kk) = 80.0 + 18.0 * real(kk - 1)
      zc(kk) = zlo + 0.5 * dz(kk)
      zlo = zlo + dz(kk)
    end do

    do kk = 1, gf_nz
      tt(kk) = c_tsfc(ic) - c_lapse(ic) * zc(kk)
      if (zc(kk) > c_ztrop(ic)) then
        tt(kk) = c_tsfc(ic) - c_lapse(ic) * c_ztrop(ic)
      end if
      ! Optional capping inversion.  A logistic step, not a Gaussian bump, so
      ! d2T/dz2 has a genuine local feature for get_inversion_layers
      ! (module_cu_gf_deep.F:4063) to lock onto rather than a smooth hump.
      if (c_dtcap(ic) > 0.0) then
        tcap = c_dtcap(ic) / (1.0 + exp(-(zc(kk) - c_zcap(ic)) / c_capw(ic)))
        tt(kk) = tt(kk) + tcap
      end if
    end do

    ! Hydrostatic pressure on the full levels, then mass levels between them.
    ppw(1) = 101300.0 * exp(-c_ht(ic) / 8500.0)
    do kk = 1, gf_nz
      tv = tt(kk) * 1.02
      ppw(kk + 1) = ppw(kk) * exp(-gf_g * dz(kk) / (gf_rd * tv))
      pp(kk) = 0.5 * (ppw(kk) + ppw(kk + 1))
    end do

    do kk = 1, gf_nz
      ppi(kk) = (pp(kk) / gf_p1000mb) ** gf_rovcp
      zfrac = min(1.0, zc(kk) / 6000.0)
      rh = c_rhsfc(ic) + (c_rhmid(ic) - c_rhsfc(ic)) * zfrac
      if (zc(kk) > 10000.0) rh = c_rhtop(ic)
      qq(kk) = max(1.0e-9, rh * gf_qsat(tt(kk), pp(kk)))
      rr(kk) = pp(kk) / (gf_rd * tt(kk) * (1.0 + 0.61 * qq(kk)))
      uu(kk) = c_ubase(ic) + c_ushear(ic) * zc(kk)
      vv(kk) = c_vbase(ic) - 0.0004 * zc(kk)
      ! A half-sine mass-weighted ascent profile through the troposphere.
      ww(kk) = c_wamp(ic) * sin(3.14159265 * min(1.0, zc(kk) / 12000.0))
    end do
  end subroutine gf_column

end module gf_cases
