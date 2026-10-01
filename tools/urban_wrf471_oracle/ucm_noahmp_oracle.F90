! Noah-MP coupling oracle for the single-layer UCM: WRF v4.7.1's
! noahmp_urban (phys/noahmp/drivers/wrf/module_sf_noahmpdrv.F,
! byte-unmodified) called with sf_urban_physics = 1, the way the surface
! driver calls it right after noahmplsm (module_surface_driver.F:3184).
!
! noahmp_urban takes the grid fields noahmplsm left (TSK HFX QFX LH GRDFLX
! ALBEDO EMISS QSFC UST CHS CHS2 CQS2) and blends the UCM into them, so no
! tap is needed: every input is a word this driver writes and every output a
! word noahmp_urban writes.  The "post-noahmplsm" fields are set here
! directly; noahmp_urban never recomputes them.
!
! The same driver then runs WRF's own option-1 surface-driver override
! block, module_surface_driver.F:3383-3404, which build_ucm_noahmp.sh
! extracts VERBATIM (by line range, pinned by the file's sha256) into
! ucm_sd_overrides.inc: the Noah-MP T2/Q2/TH2 blend and the U10/V10/PSIM/
! PSIH/GZ1OZ0/AKHS/AKMS overrides.
!
! Usage:  run_ucm_noahmp OUT.csv

program run_ucm_noahmp
  use module_sf_noahmpdrv, only: noahmp_urban
  use module_sf_urban, only: urban_param_init, frc_urb_tbl
  use noahmp_tables, only: t_isurban => isurban_table,                        &
      t_lcz_1 => lcz_1_table, t_lcz_2 => lcz_2_table, t_lcz_3 => lcz_3_table,  &
      t_lcz_4 => lcz_4_table, t_lcz_5 => lcz_5_table, t_lcz_6 => lcz_6_table,  &
      t_lcz_7 => lcz_7_table, t_lcz_8 => lcz_8_table, t_lcz_9 => lcz_9_table,  &
      t_lcz_10 => lcz_10_table, t_lcz_11 => lcz_11_table
  implicit none

  integer, parameter :: ncase = 12
  integer, parameter :: nsoil = 4
  integer, parameter :: nstep = 4
  real, parameter :: dt = 60.0
  real, parameter :: rcp = 287. / (7. * 287. / 2.)
  integer, parameter :: isurban = 13
  integer, parameter :: lcz_1_table = 51, lcz_2_table = 52, lcz_3_table = 53
  integer, parameter :: lcz_4_table = 54, lcz_5_table = 55, lcz_6_table = 56
  integer, parameter :: lcz_7_table = 57, lcz_8_table = 58, lcz_9_table = 59
  integer, parameter :: lcz_10_table = 60, lcz_11_table = 61

  real, dimension(ncase, 2, 1) :: t3d, qv3d, u_phy, v_phy, p8w3d, dz8w
  real, dimension(ncase, 1) :: cosz_urb2d, xlat_urb2d, swdown, swddif, swddir
  real, dimension(ncase, 1) :: glw, rainbl, znt, tsk, hfx, qfx, lh, grdflx
  real, dimension(ncase, 1) :: albedo, emiss, qsfc, ust, chs, chs2, cqs2
  real, dimension(ncase, 1) :: omg_urb2d, frc_urb2d
  integer, dimension(ncase, 1) :: ivgtyp, utype_urb2d
  real, dimension(ncase, 1) :: cmr_sfcdif, chr_sfcdif, cmc_sfcdif, chc_sfcdif
  real, dimension(ncase, 1) :: cmgr_sfcdif, chgr_sfcdif
  real, dimension(ncase, 1) :: tr_urb2d, tb_urb2d, tg_urb2d, tc_urb2d, qc_urb2d
  real, dimension(ncase, 1) :: uc_urb2d, xxxr_urb2d, xxxb_urb2d, xxxg_urb2d
  real, dimension(ncase, 1) :: xxxc_urb2d, sh_urb2d, lh_urb2d, g_urb2d
  real, dimension(ncase, 1) :: rn_urb2d, ts_urb2d, psim_urb2d, psih_urb2d
  real, dimension(ncase, 1) :: u10_urb2d, v10_urb2d, gz1oz0_urb2d, akms_urb2d
  real, dimension(ncase, 1) :: th2_urb2d, q2_urb2d, ust_urb2d
  real, dimension(ncase, 1) :: cmcr_urb2d, tgr_urb2d, drelr_urb2d, drelb_urb2d
  real, dimension(ncase, 1) :: drelg_urb2d, flxhumr_urb2d, flxhumb_urb2d
  real, dimension(ncase, 1) :: flxhumg_urb2d
  real, dimension(ncase, 1) :: lp_urb2d, lb_urb2d, hgt_urb2d, mh_urb2d
  real, dimension(ncase, 1) :: stdh_urb2d, lf_urb2d_s, z0_urb2d, vegfra
  real, dimension(ncase, 4, 1) :: lf_urb2d
  real, dimension(ncase, nsoil, 1) :: trl_urb3d, tbl_urb3d, tgl_urb3d
  real, dimension(ncase, nsoil, 1) :: tgrl_urb3d, smr_urb3d
  ! surface-driver override inputs/outputs
  real, dimension(ncase, 1) :: psfc, fvegxy, t2mvxy, t2mbxy, q2mvxy, q2mbxy
  real, dimension(ncase, 1) :: t2, th2, q2, u10, v10, psim, psih, gz1oz0
  real, dimension(ncase, 1) :: akhs, akms
  real :: dzr(nsoil), dzb(nsoil), dzg(nsoil), declin
  real :: pre(44, ncase), pre_f(24, ncase), pre_g(8, ncase)
  integer :: julday, julyr, st, i, j, u
  integer :: sf_urban_physics
  integer :: i_start(1), i_end(1), j_start(1), j_end(1), ij
  character(len=1024) :: out_path

  call get_command_argument(1, out_path)
  if (len_trim(out_path) == 0) then
    write(*, '(A)') 'usage: run_ucm_noahmp OUT.csv'
    error stop 2
  end if
  dzr = 0.
  dzb = 0.
  dzg = 0.
  call urban_param_init(dzr, dzb, dzg, nsoil, 1, 0, .false.)
  ! The category numbers read_mp_veg_parameters would set from MPTABLE.TBL's
  ! MODIFIED_IGBP_MODIS_NOAH block (ISURBAN 13, LCZ_1..11 = 51..61);
  ! noahmp_urban reads nothing else from NOAHMP_TABLES.
  t_isurban = isurban
  t_lcz_1 = lcz_1_table
  t_lcz_2 = lcz_2_table
  t_lcz_3 = lcz_3_table
  t_lcz_4 = lcz_4_table
  t_lcz_5 = lcz_5_table
  t_lcz_6 = lcz_6_table
  t_lcz_7 = lcz_7_table
  t_lcz_8 = lcz_8_table
  t_lcz_9 = lcz_9_table
  t_lcz_10 = lcz_10_table
  t_lcz_11 = lcz_11_table
  julday = 196
  julyr = 2025
  declin = 0.37
  sf_urban_physics = 1
  ij = 1
  i_start(1) = 1
  i_end(1) = ncase
  j_start(1) = 1
  j_end(1) = 1

  call build_fixture()

  open(newunit=u, file=trim(out_path), status='replace', action='write')
  call write_header()
  do st = 1, nstep
    call forcing(st)
    call record_pre(st)
    call noahmp_urban(1, nsoil, ivgtyp, st + 1,                             &
                      dt, cosz_urb2d, xlat_urb2d,                           &
                      t3d, qv3d, u_phy, v_phy, swdown,                      &
                      swddir, swddif,                                       &
                      glw, p8w3d, rainbl, dz8w, znt,                        &
                      tsk, hfx, qfx, lh, grdflx,                            &
                      albedo, emiss, qsfc,                                  &
                      1, ncase, 1, 1, 1, 2,                                 &
                      1, ncase, 1, 1, 1, 2,                                 &
                      1, ncase, 1, 1, 1, 1,                                 &
                      cmr_sfcdif, chr_sfcdif, cmc_sfcdif,                   &
                      chc_sfcdif, cmgr_sfcdif, chgr_sfcdif,                 &
                      tr_urb2d, tb_urb2d, tg_urb2d,                         &
                      tc_urb2d, qc_urb2d, uc_urb2d,                         &
                      xxxr_urb2d, xxxb_urb2d, xxxg_urb2d, xxxc_urb2d,       &
                      trl_urb3d, tbl_urb3d, tgl_urb3d,                      &
                      sh_urb2d, lh_urb2d, g_urb2d, rn_urb2d, ts_urb2d,      &
                      psim_urb2d, psih_urb2d, u10_urb2d, v10_urb2d,         &
                      gz1oz0_urb2d, akms_urb2d,                             &
                      th2_urb2d, q2_urb2d, ust_urb2d,                       &
                      declin, omg_urb2d,                                    &
                      nsoil, nsoil, nsoil,                                  &
                      dzr, dzb, dzg,                                        &
                      cmcr_urb2d, tgr_urb2d, tgrl_urb3d, smr_urb3d,         &
                      drelr_urb2d, drelb_urb2d, drelg_urb2d,                &
                      flxhumr_urb2d, flxhumb_urb2d, flxhumg_urb2d,          &
                      julday, julyr,                                        &
                      frc_urb2d, utype_urb2d,                               &
                      chs, chs2, cqs2,                                      &
                      1, 1, 1, 1,                                           &
                      1, 1, 1, 1,                                           &
                      1, 1, 1,                                              &
                      1,                                                    &
                      lp_urb2d = lp_urb2d, lb_urb2d = lb_urb2d,             &
                      hgt_urb2d = hgt_urb2d, mh_urb2d = mh_urb2d,           &
                      stdh_urb2d = stdh_urb2d, lf_urb2d = lf_urb2d,         &
                      lf_urb2d_s = lf_urb2d_s, z0_urb2d = z0_urb2d,         &
                      vegfra = vegfra, ust = ust)
    call overrides()
    call write_post()
  end do
  close(u)

contains

  ! module_surface_driver.F:3383-3404, verbatim (see the header).
  subroutine overrides()
#include "ucm_sd_overrides.inc"
  end subroutine overrides

  subroutine build_fixture()
    real :: t0
    do i = 1, ncase
      ivgtyp(i, 1) = isurban
      utype_urb2d(i, 1) = 1 + mod(i - 1, 3)
      frc_urb2d(i, 1) = frc_urb_tbl(utype_urb2d(i, 1))
      p8w3d(i, 1, 1) = 99000.0
      p8w3d(i, 2, 1) = 98100.0
      t3d(i, 1, 1) = 297.0
      t3d(i, 2, 1) = 296.0
      qv3d(i, 1, 1) = 0.012
      qv3d(i, 2, 1) = 0.012
      dz8w(i, 1, 1) = 70.0
      dz8w(i, 2, 1) = 70.0
      xlat_urb2d(i, 1) = 34.0
      cosz_urb2d(i, 1) = 0.8
      omg_urb2d(i, 1) = 0.3
      znt(i, 1) = 0.6
      emiss(i, 1) = 0.96
      vegfra(i, 1) = 30.0
      tsk(i, 1) = 299.0
      t0 = tsk(i, 1)
      tr_urb2d(i, 1) = t0
      tb_urb2d(i, 1) = t0
      tg_urb2d(i, 1) = t0
      tc_urb2d(i, 1) = t0
      ts_urb2d(i, 1) = t0
      tgr_urb2d(i, 1) = t0
      qc_urb2d(i, 1) = 0.01
      uc_urb2d(i, 1) = 0.
      xxxr_urb2d(i, 1) = 0.
      xxxb_urb2d(i, 1) = 0.
      xxxg_urb2d(i, 1) = 0.
      xxxc_urb2d(i, 1) = 0.
      trl_urb3d(i, :, 1) = (/ 298.0, 297.0, 296.0, 295.13 /)
      tbl_urb3d(i, :, 1) = trl_urb3d(i, :, 1)
      tgrl_urb3d(i, :, 1) = trl_urb3d(i, :, 1)
      tgl_urb3d(i, :, 1) = (/ 298.0, 296.0, 293.0, 290.0 /)
      smr_urb3d(i, :, 1) = (/ 0.2, 0.2, 0.2, 0.0 /)
      sh_urb2d(i, 1) = 0.
      lh_urb2d(i, 1) = 0.
      g_urb2d(i, 1) = 0.
      rn_urb2d(i, 1) = 0.
      cmcr_urb2d(i, 1) = 0.
      drelr_urb2d(i, 1) = 0.
      drelb_urb2d(i, 1) = 0.
      drelg_urb2d(i, 1) = 0.
      flxhumr_urb2d(i, 1) = 0.
      flxhumb_urb2d(i, 1) = 0.
      flxhumg_urb2d(i, 1) = 0.
      cmr_sfcdif(i, 1) = 0.
      chr_sfcdif(i, 1) = 0.
      cmc_sfcdif(i, 1) = 0.
      chc_sfcdif(i, 1) = 0.
      cmgr_sfcdif(i, 1) = 0.
      chgr_sfcdif(i, 1) = 0.
      lp_urb2d(i, 1) = 0.
      lb_urb2d(i, 1) = 0.
      hgt_urb2d(i, 1) = 0.
      mh_urb2d(i, 1) = 0.
      stdh_urb2d(i, 1) = 0.
      lf_urb2d(i, :, 1) = 0.
      lf_urb2d_s(i, 1) = 0.
      z0_urb2d(i, 1) = 0.
      psim_urb2d(i, 1) = 0.
      psih_urb2d(i, 1) = 0.
      u10_urb2d(i, 1) = 0.
      v10_urb2d(i, 1) = 0.
      gz1oz0_urb2d(i, 1) = 0.
      akms_urb2d(i, 1) = 0.
      th2_urb2d(i, 1) = 0.
      q2_urb2d(i, 1) = 0.
      ust_urb2d(i, 1) = 0.
    end do
    frc_urb2d(4, 1) = 1.0
    frc_urb2d(5, 1) = 0.05
    ivgtyp(12, 1) = 10              ! grass: noahmp_urban must not touch it
    utype_urb2d(12, 1) = 0
    frc_urb2d(12, 1) = 0.0
  end subroutine build_fixture

  ! "post-noahmplsm" fields for step s; columns 7-9 night, 10-11 rain,
  ! column 6 weak exchange (the 1.0E-02 floors fire).
  subroutine forcing(step)
    integer, intent(in) :: step
    real :: s
    s = real(step - 1)
    do i = 1, ncase
      swdown(i, 1) = 800.0 - 25.0 * s
      swddir(i, 1) = 0.7 * swdown(i, 1)
      swddif(i, 1) = 0.3 * swdown(i, 1)
      glw(i, 1) = 380.0
      rainbl(i, 1) = 0.0
      u_phy(i, 1, 1) = 2.5 + 0.3 * real(i)
      v_phy(i, 1, 1) = 1.0 - 0.2 * real(i)
      u_phy(i, 2, 1) = u_phy(i, 1, 1)
      v_phy(i, 2, 1) = v_phy(i, 1, 1)
      hfx(i, 1) = 120.0 + 3.0 * real(i) - 5.0 * s
      qfx(i, 1) = 1.1e-4 + 1.0e-6 * real(i)
      lh(i, 1) = 270.0 + 2.0 * real(i)
      grdflx(i, 1) = 45.0 - real(i)
      albedo(i, 1) = 0.17 + 0.002 * real(i)
      qsfc(i, 1) = 0.0135 + 1.0e-4 * real(i)
      ust(i, 1) = 0.30 + 0.01 * real(i)
      tsk(i, 1) = 300.5 + 0.1 * real(i) - 0.2 * s
      chs(i, 1) = 0.018
      chs2(i, 1) = 0.012
      cqs2(i, 1) = 0.014
      psfc(i, 1) = p8w3d(i, 1, 1)
      fvegxy(i, 1) = 0.35
      t2mvxy(i, 1) = 298.4 + 0.05 * real(i)
      t2mbxy(i, 1) = 299.1 + 0.05 * real(i)
      q2mvxy(i, 1) = 0.0128
      q2mbxy(i, 1) = 0.0125
      t2(i, 1) = t2mbxy(i, 1)
      th2(i, 1) = 0.
      q2(i, 1) = q2mbxy(i, 1)
      u10(i, 1) = 0.
      v10(i, 1) = 0.
      psim(i, 1) = 0.
      psih(i, 1) = 0.
      gz1oz0(i, 1) = 0.
      akhs(i, 1) = 0.
      akms(i, 1) = 0.
      if (i >= 7 .and. i <= 9) then
        swdown(i, 1) = 0.0
        swddir(i, 1) = 0.0
        swddif(i, 1) = 0.0
        omg_urb2d(i, 1) = 2.9
        hfx(i, 1) = -25.0
        tsk(i, 1) = 289.0
        t3d(i, 1, 1) = 290.0
      end if
      if (i == 8) then               ! calm: UA below 1 m/s clamps to 1
        u_phy(i, 1, 1) = 0.4
        v_phy(i, 1, 1) = -0.3
      end if
      if (i >= 10 .and. i <= 11) rainbl(i, 1) = 0.15 * real(step) * real(i - 9)
      if (i == 6) then
        chs(i, 1) = 0.004
        chs2(i, 1) = 0.003
        cqs2(i, 1) = 0.006
      end if
    end do
  end subroutine forcing

  subroutine write_header()
    character(len=4096) :: h
    h = 'step,case,ivgtyp,utype,frc,u1,v1,omg,t3d1,qv1,p8w1,p8w2,dz8w1,swdown,rainbl,psfc,' // &
        'fvegxy,t2mvxy,t2mbxy,q2mvxy,q2mbxy'
    h = trim(h) // ',tsk_in,hfx_in,qfx_in,lh_in,grdflx_in,albedo_in,qsfc_in,' // &
        'ust_in,chs_in,chs2_in,cqs2_in,glw,znt_in'
    h = trim(h) // ',tr_in,tb_in,tg_in,tc_in,qc_in,uc_in,xxxr_in,xxxb_in,' //  &
        'xxxg_in,xxxc_in,cmr_in,chr_in,cmc_in,chc_in,cmgr_in,chgr_in,' //      &
        'cmcr_in,tgr_in,drelr_in,drelb_in,drelg_in,flxhumr_in,flxhumb_in,' //  &
        'flxhumg_in,trl1_in,trl2_in,trl3_in,trl4_in,tbl1_in,tbl2_in,' //       &
        'tbl3_in,tbl4_in,tgl1_in,tgl2_in,tgl3_in,tgl4_in,tgrl1_in,' //          &
        'tgrl2_in,tgrl3_in,tgrl4_in,smr1_in,smr2_in,smr3_in,smr4_in'
    h = trim(h) // ',tsk,hfx,qfx,lh,grdflx,albedo,qsfc,ust,chs,chs2,cqs2,znt'
    h = trim(h) // ',tr,tb,tg,tc,qc,uc,xxxr,xxxb,xxxg,xxxc,cmr,chr,cmc,chc,' // &
        'cmgr,chgr,cmcr,tgr,drelr,drelb,drelg,flxhumr,flxhumb,flxhumg,' //      &
        'trl1,trl2,trl3,trl4,tbl1,tbl2,tbl3,tbl4,tgl1,tgl2,tgl3,tgl4,' //        &
        'tgrl1,tgrl2,tgrl3,tgrl4,smr1,smr2,smr3,smr4,ts_urb,sh_urb,lh_urb,' //   &
        'g_urb,rn_urb,psim_urb,psih_urb,gz1oz0_urb,u10_urb,v10_urb,th2_urb,' //  &
        'q2_urb,ust_urb,akms_urb'
    h = trim(h) // ',t2,th2,q2,u10,v10,psim,psih,gz1oz0,akhs,akms'
    write(u, '(A)') trim(h)
  end subroutine write_header

  subroutine record_pre(step)
    integer, intent(in) :: step
    integer :: k
    do i = 1, ncase
      pre(1:6, i) = (/ tr_urb2d(i, 1), tb_urb2d(i, 1), tg_urb2d(i, 1),           &
                       tc_urb2d(i, 1), qc_urb2d(i, 1), uc_urb2d(i, 1) /)
      pre(7:10, i) = (/ xxxr_urb2d(i, 1), xxxb_urb2d(i, 1), xxxg_urb2d(i, 1),     &
                        xxxc_urb2d(i, 1) /)
      pre(11:16, i) = (/ cmr_sfcdif(i, 1), chr_sfcdif(i, 1), cmc_sfcdif(i, 1),    &
                         chc_sfcdif(i, 1), cmgr_sfcdif(i, 1), chgr_sfcdif(i, 1) /)
      pre(17:24, i) = (/ cmcr_urb2d(i, 1), tgr_urb2d(i, 1), drelr_urb2d(i, 1),    &
                         drelb_urb2d(i, 1), drelg_urb2d(i, 1), flxhumr_urb2d(i, 1), &
                         flxhumb_urb2d(i, 1), flxhumg_urb2d(i, 1) /)
      do k = 1, nsoil
        pre(24 + k, i) = trl_urb3d(i, k, 1)
        pre(28 + k, i) = tbl_urb3d(i, k, 1)
        pre(32 + k, i) = tgl_urb3d(i, k, 1)
        pre(36 + k, i) = tgrl_urb3d(i, k, 1)
        pre(40 + k, i) = smr_urb3d(i, k, 1)
      end do
      pre_f(:, i) = (/ real(step), real(i), frc_urb2d(i, 1), u_phy(i, 1, 1),     &
                       v_phy(i, 1, 1), omg_urb2d(i, 1), t3d(i, 1, 1), qv3d(i, 1, 1), &
                       p8w3d(i, 1, 1), p8w3d(i, 2, 1), dz8w(i, 1, 1), swdown(i, 1), &
                       rainbl(i, 1), psfc(i, 1), fvegxy(i, 1), t2mvxy(i, 1),     &
                       t2mbxy(i, 1), q2mvxy(i, 1), q2mbxy(i, 1), tsk(i, 1),      &
                       hfx(i, 1), qfx(i, 1), lh(i, 1), grdflx(i, 1) /)
      pre_g(:, i) = (/ albedo(i, 1), qsfc(i, 1), ust(i, 1), chs(i, 1), chs2(i, 1), &
                       cqs2(i, 1), glw(i, 1), znt(i, 1) /)
    end do
  end subroutine record_pre

  subroutine write_post()
    integer :: k
    real :: post(44)
    do i = 1, ncase
      write(u, '(I0,",",I0,",",I0,",",I0)', advance='no') nint(pre_f(1, i)),     &
           nint(pre_f(2, i)), ivgtyp(i, 1), utype_urb2d(i, 1)
      write(u, '(22(",",ES16.8E3))', advance='no') pre_f(3:24, i)
      write(u, '(8(",",ES16.8E3))', advance='no') pre_g(:, i)
      write(u, '(44(",",ES16.8E3))', advance='no') pre(:, i)
      write(u, '(12(",",ES16.8E3))', advance='no') tsk(i, 1), hfx(i, 1),         &
           qfx(i, 1), lh(i, 1), grdflx(i, 1), albedo(i, 1), qsfc(i, 1),         &
           ust(i, 1), chs(i, 1), chs2(i, 1), cqs2(i, 1), znt(i, 1)
      post(1:6) = (/ tr_urb2d(i, 1), tb_urb2d(i, 1), tg_urb2d(i, 1),            &
                     tc_urb2d(i, 1), qc_urb2d(i, 1), uc_urb2d(i, 1) /)
      post(7:10) = (/ xxxr_urb2d(i, 1), xxxb_urb2d(i, 1), xxxg_urb2d(i, 1),      &
                      xxxc_urb2d(i, 1) /)
      post(11:16) = (/ cmr_sfcdif(i, 1), chr_sfcdif(i, 1), cmc_sfcdif(i, 1),     &
                       chc_sfcdif(i, 1), cmgr_sfcdif(i, 1), chgr_sfcdif(i, 1) /)
      post(17:24) = (/ cmcr_urb2d(i, 1), tgr_urb2d(i, 1), drelr_urb2d(i, 1),     &
                       drelb_urb2d(i, 1), drelg_urb2d(i, 1), flxhumr_urb2d(i, 1), &
                       flxhumb_urb2d(i, 1), flxhumg_urb2d(i, 1) /)
      do k = 1, nsoil
        post(24 + k) = trl_urb3d(i, k, 1)
        post(28 + k) = tbl_urb3d(i, k, 1)
        post(32 + k) = tgl_urb3d(i, k, 1)
        post(36 + k) = tgrl_urb3d(i, k, 1)
        post(40 + k) = smr_urb3d(i, k, 1)
      end do
      write(u, '(44(",",ES16.8E3))', advance='no') post
      write(u, '(14(",",ES16.8E3))', advance='no') ts_urb2d(i, 1), sh_urb2d(i, 1), &
           lh_urb2d(i, 1), g_urb2d(i, 1), rn_urb2d(i, 1), psim_urb2d(i, 1),       &
           psih_urb2d(i, 1), gz1oz0_urb2d(i, 1), u10_urb2d(i, 1), v10_urb2d(i, 1), &
           th2_urb2d(i, 1), q2_urb2d(i, 1), ust_urb2d(i, 1), akms_urb2d(i, 1)
      write(u, '(10(",",ES16.8E3))') t2(i, 1), th2(i, 1), q2(i, 1), u10(i, 1),   &
           v10(i, 1), psim(i, 1), psih(i, 1), gz1oz0(i, 1), akhs(i, 1), akms(i, 1)
    end do
  end subroutine write_post

end program run_ucm_noahmp
