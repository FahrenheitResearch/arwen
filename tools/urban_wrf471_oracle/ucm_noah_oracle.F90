! Noah coupling oracle for the single-layer UCM: WRF v4.7.1's Noah driver
! `lsm` (phys/module_sf_noahdrv.F, byte-unmodified) called with
! sf_urban_physics = 1 over a row of urban columns (and one grass column
! that must come out untouched), several steps with the urban state carried.
!
! What the fixture pins is module_sf_noahdrv.F 1317-1600: the UCM inputs
! lsm prepares, the CHS/CHS2/CQS2 floors, the ALBEDO/HFX/QFX/LH/GRDFLX/TSK/
! Q1->QSFC/UST blend, the urban state renewal and AKMS_URB2D.  gpuwm runs
! that block after the LSM kernel on the rural values the LSM hands over, so
! the fixture also carries those values AS WRF HELD THEM at the UCM's entry
! (t1 sheat eta_kinematic eta ssoil albedok q1 sfctmp q2k sfcprs zlvl soldn,
! rainbl and the chs chs2 cqs2 ust znt glw words), read through ONE inserted
! statement, `CALL ucm_oracle_tap(...)`, immediately before :1332.  The
! build runs the pristine and the tapped objects on the same inputs and
! fails unless every output they share is byte-identical, so the tap is
! proven not to perturb what it reads.
!
! Usage:  run_ucm_noah OUT.csv

module ucm_tap_store
  implicit none
  integer, parameter :: ntap = 19
  real :: tap(ntap, 64)
  logical :: tapped(64) = .false.
end module ucm_tap_store

subroutine ucm_oracle_tap(i, j, t1, sheat, eta_kinematic, eta, ssoil, albedok, &
                          q1, sfctmp, q2k, sfcprs, zlvl, soldn, rainbl, chs,    &
                          chs2, cqs2, ust, znt, glw)
  use ucm_tap_store
  implicit none
  integer, intent(in) :: i, j
  real, intent(in) :: t1, sheat, eta_kinematic, eta, ssoil, albedok, q1, sfctmp
  real, intent(in) :: q2k, sfcprs, zlvl, soldn, rainbl, chs, chs2, cqs2, ust
  real, intent(in) :: znt, glw
  tap(:, i) = (/ t1, sheat, eta_kinematic, eta, ssoil, albedok, q1, sfctmp,  &
                 q2k, sfcprs, zlvl, soldn, rainbl, chs, chs2, cqs2, ust, znt, glw /)
  tapped(i) = .true.
end subroutine ucm_oracle_tap

program run_ucm_noah
  use module_sf_noahdrv, only: lsm, soil_veg_gen_parm
  use module_sf_urban, only: urban_param_init, frc_urb_tbl
  use ucm_tap_store
  implicit none

  integer, parameter :: ncase = 16
  integer, parameter :: nsoil = 4
  integer, parameter :: nstep = 4
  real, parameter :: dt = 60.0
  real, parameter :: r_d_over_cp = 287.0 / (7.0 * 287.0 / 2.0)

  real, dimension(ncase, 2, 1) :: dz8w, qv3d, p8w3d, t3d, u_phy, v_phy
  real, dimension(ncase, 1) :: tsk, hfx, qfx, lh, grdflx, qgh, gsw, swdown
  real, dimension(ncase, 1) :: swddir, swddif, glw, smstav, smstot
  real, dimension(ncase, 1) :: sfcrunoff, udrunoff, vegfra, albedo, albbck
  real, dimension(ncase, 1) :: znt, z0, tmn, xland, xice, emiss, embck
  real, dimension(ncase, 1) :: snowc, qsfc, rainbl, snow, canwat
  real, dimension(ncase, 1) :: chs, chs2, cqs2, cpm, sr, chklowq, lai, qz0
  real, dimension(ncase, 1) :: snowh, snoalb, shdmin, shdmax, snotime
  real, dimension(ncase, 1) :: acsnom, acsnow, snopcx, potevp, rib, noahres
  real, dimension(ncase, 1) :: flx4_2d, fvb_2d, fbur_2d, fgsn_2d, ust
  real, dimension(ncase, 1) :: sda_hfx, sda_qfx, hfx_both, qfx_both, qnorm
  real, dimension(ncase, nsoil, 1) :: smois, tslb, sh2o, smcrel
  integer, dimension(ncase, 1) :: ivgtyp, isltyp, utype_urb2d
  real, dimension(nsoil) :: dzs
  ! urban
  real, dimension(ncase, 1) :: frc_urb2d, cosz_urb2d, omg_urb2d, xlat_urb2d
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
  real, dimension(ncase, 1) :: stdh_urb2d, lf_urb2d_s, z0_urb2d
  real, dimension(ncase, 4, 1) :: lf_urb2d
  real, dimension(ncase, nsoil, 1) :: trl_urb3d, tbl_urb3d, tgl_urb3d
  real, dimension(ncase, nsoil, 1) :: tgrl_urb3d, smr_urb3d
  real :: dzr(nsoil), dzb(nsoil), dzg(nsoil), declin
  real :: pre(44, ncase), pre_in(10, ncase)
  integer :: julday, julyr, st, i, u

  character(len=1024) :: out_path

  call get_command_argument(1, out_path)
  if (len_trim(out_path) == 0) then
    write(*, '(A)') 'usage: run_ucm_noah OUT.csv'
    error stop 2
  end if

  tap = 0.
  call soil_veg_gen_parm('MODIFIED_IGBP_MODIS_NOAH', 'STAS')
  dzr = 0.
  dzb = 0.
  dzg = 0.
  call urban_param_init(dzr, dzb, dzg, nsoil, 1, 0, .false.)
  dzs = (/ 0.10, 0.30, 0.60, 1.00 /)
  julday = 196
  julyr = 2025
  declin = 0.37

  call build_fixture()

  open(newunit=u, file=trim(out_path), status='replace', action='write')
  call write_header()
  do st = 1, nstep
    call forcing(st)
    tapped = .false.
    call write_pre(st)
    call lsm(dz8w, qv3d, p8w3d, t3d, tsk,                              &
             hfx, qfx, lh, grdflx, qgh, gsw, swdown, swddir, swddif,   &
             glw, smstav, smstot,                                      &
             sfcrunoff, udrunoff, ivgtyp, isltyp, 13, 15, vegfra,      &
             albedo, albbck, znt, z0, tmn, xland, xice, emiss, embck,  &
             snowc, qsfc, rainbl, 'MODIFIED_IGBP_MODIS_NOAH',          &
             nsoil, dt, dzs, st + 1,                                   &
             smois, tslb, snow, canwat,                                &
             chs, chs2, cqs2, cpm, r_d_over_cp, sr, chklowq, lai, qz0, &
             .false., .false.,                                         &
             sh2o, snowh,                                              &
             u_phy = u_phy, v_phy = v_phy,                             &
             snoalb = snoalb, shdmin = shdmin, shdmax = shdmax,        &
             snotime = snotime,                                        &
             acsnom = acsnom, acsnow = acsnow,                         &
             snopcx = snopcx, potevp = potevp, smcrel = smcrel,        &
             xice_threshold = 0.5,                                     &
             rdlai2d = .false., usemonalb = .false.,                   &
             rib = rib, noahres = noahres, opt_thcnd = 1,              &
             ua_phys = .false., flx4_2d = flx4_2d, fvb_2d = fvb_2d,    &
             fbur_2d = fbur_2d, fgsn_2d = fgsn_2d,                     &
             ids = 1, ide = ncase, jds = 1, jde = 1, kds = 1, kde = 2, &
             ims = 1, ime = ncase, jms = 1, jme = 1, kms = 1, kme = 2, &
             its = 1, ite = ncase, jts = 1, jte = 1, kts = 1, kte = 1, &
             sf_urban_physics = 1,                                     &
             cmr_sfcdif = cmr_sfcdif, chr_sfcdif = chr_sfcdif,         &
             cmc_sfcdif = cmc_sfcdif, chc_sfcdif = chc_sfcdif,         &
             cmgr_sfcdif = cmgr_sfcdif, chgr_sfcdif = chgr_sfcdif,     &
             tr_urb2d = tr_urb2d, tb_urb2d = tb_urb2d,                 &
             tg_urb2d = tg_urb2d, tc_urb2d = tc_urb2d,                 &
             qc_urb2d = qc_urb2d, uc_urb2d = uc_urb2d,                 &
             xxxr_urb2d = xxxr_urb2d, xxxb_urb2d = xxxb_urb2d,         &
             xxxg_urb2d = xxxg_urb2d, xxxc_urb2d = xxxc_urb2d,         &
             trl_urb3d = trl_urb3d, tbl_urb3d = tbl_urb3d,             &
             tgl_urb3d = tgl_urb3d,                                    &
             sh_urb2d = sh_urb2d, lh_urb2d = lh_urb2d,                 &
             g_urb2d = g_urb2d, rn_urb2d = rn_urb2d,                   &
             ts_urb2d = ts_urb2d,                                      &
             psim_urb2d = psim_urb2d, psih_urb2d = psih_urb2d,         &
             u10_urb2d = u10_urb2d, v10_urb2d = v10_urb2d,             &
             gz1oz0_urb2d = gz1oz0_urb2d, akms_urb2d = akms_urb2d,     &
             th2_urb2d = th2_urb2d, q2_urb2d = q2_urb2d,               &
             ust_urb2d = ust_urb2d,                                    &
             declin_urb = declin, cosz_urb2d = cosz_urb2d,             &
             omg_urb2d = omg_urb2d, xlat_urb2d = xlat_urb2d,           &
             num_roof_layers = nsoil, num_wall_layers = nsoil,         &
             num_road_layers = nsoil, dzr = dzr, dzb = dzb, dzg = dzg, &
             cmcr_urb2d = cmcr_urb2d, tgr_urb2d = tgr_urb2d,           &
             tgrl_urb3d = tgrl_urb3d, smr_urb3d = smr_urb3d,           &
             drelr_urb2d = drelr_urb2d, drelb_urb2d = drelb_urb2d,     &
             drelg_urb2d = drelg_urb2d,                                &
             flxhumr_urb2d = flxhumr_urb2d,                            &
             flxhumb_urb2d = flxhumb_urb2d,                            &
             flxhumg_urb2d = flxhumg_urb2d,                            &
             julian = julday, julyr = julyr,                           &
             frc_urb2d = frc_urb2d, utype_urb2d = utype_urb2d,         &
             num_urban_ndm = 1, urban_map_zrd = 1, urban_map_zwd = 1,  &
             urban_map_gd = 1, urban_map_zd = 1, urban_map_zdf = 1,    &
             urban_map_bd = 1, urban_map_wd = 1, urban_map_gbd = 1,    &
             urban_map_fbd = 1, urban_map_zgrd = 1, num_urban_hi = 1,  &
             lp_urb2d = lp_urb2d, lb_urb2d = lb_urb2d,                 &
             hgt_urb2d = hgt_urb2d, mh_urb2d = mh_urb2d,               &
             stdh_urb2d = stdh_urb2d, lf_urb2d = lf_urb2d,             &
             lf_urb2d_s = lf_urb2d_s, z0_urb2d = z0_urb2d,             &
             ust = ust,                                                &
             sda_hfx = sda_hfx, sda_qfx = sda_qfx,                     &
             hfx_both = hfx_both, qfx_both = qfx_both, qnorm = qnorm,  &
             fasdas = 0)
    call write_post()
  end do
  close(u)

contains

  subroutine build_fixture()
    real :: t0
    do i = 1, ncase
      ivgtyp(i, 1) = 13                 ! ISURBAN
      isltyp(i, 1) = 8
      utype_urb2d(i, 1) = 1 + mod(i - 1, 3)
      frc_urb2d(i, 1) = frc_urb_tbl(utype_urb2d(i, 1))
      p8w3d(i, 1, 1) = 98000.0
      p8w3d(i, 2, 1) = 97000.0
      t3d(i, 1, 1) = 296.0
      t3d(i, 2, 1) = 295.0
      qv3d(i, 1, 1) = 0.011
      qv3d(i, 2, 1) = 0.011
      dz8w(i, 1, 1) = 64.0
      dz8w(i, 2, 1) = 64.0
      qgh(i, 1) = 0.018
      glw(i, 1) = 370.0
      rainbl(i, 1) = 0.0
      sr(i, 1) = 0.0
      chs(i, 1) = 0.02
      chs2(i, 1) = 0.015
      cqs2(i, 1) = 0.015
      cpm(i, 1) = 1010.0
      qz0(i, 1) = 0.01
      rib(i, 1) = -0.1
      vegfra(i, 1) = 20.0
      shdmin(i, 1) = 5.0
      shdmax(i, 1) = 40.0
      tmn(i, 1) = 288.0
      xland(i, 1) = 1.0
      xice(i, 1) = 0.0
      snoalb(i, 1) = 0.7
      embck(i, 1) = 0.95
      emiss(i, 1) = 0.95
      albedo(i, 1) = 0.18
      albbck(i, 1) = 0.18
      znt(i, 1) = 0.8
      z0(i, 1) = 0.8
      tsk(i, 1) = 298.0
      hfx(i, 1) = 0.0
      qfx(i, 1) = 0.0
      lh(i, 1) = 0.0
      grdflx(i, 1) = 0.0
      qsfc(i, 1) = 0.012
      canwat(i, 1) = 0.1
      snow(i, 1) = 0.0
      snowc(i, 1) = 0.0
      snowh(i, 1) = 0.0
      snotime(i, 1) = 0.0
      lai(i, 1) = 1.0
      smstav(i, 1) = 0.
      smstot(i, 1) = 0.
      sfcrunoff(i, 1) = 0.
      udrunoff(i, 1) = 0.
      acsnom(i, 1) = 0.
      acsnow(i, 1) = 0.
      snopcx(i, 1) = 0.
      potevp(i, 1) = 0.
      ust(i, 1) = 0.35
      smois(i, :, 1) = (/ 0.28, 0.29, 0.30, 0.31 /)
      sh2o(i, :, 1) = smois(i, :, 1)
      tslb(i, :, 1) = (/ 297.0, 295.0, 292.0, 289.0 /)
      xlat_urb2d(i, 1) = 38.5
      cosz_urb2d(i, 1) = 0.8
      omg_urb2d(i, 1) = 0.2
      sda_hfx(i, 1) = 0.
      sda_qfx(i, 1) = 0.
      hfx_both(i, 1) = 0.
      qfx_both(i, 1) = 0.
      qnorm(i, 1) = 0.
    end do
    ! column specialisations: FRC edge cases, weak exchange (the 0.01
    ! floors fire), night, rain, a non-urban column
    frc_urb2d(4, 1) = 0.99          ! FRC >= 0.99: the pre-SFLX T1 is TSK itself
    frc_urb2d(5, 1) = 1.0
    frc_urb2d(6, 1) = 0.05
    chs(7, 1) = 0.004               ! all three below 1.0E-02
    chs2(7, 1) = 0.002
    cqs2(7, 1) = 0.003
    chs(8, 1) = 0.008
    ivgtyp(16, 1) = 10              ! grass: the UCM must not touch it
    utype_urb2d(16, 1) = 0
    frc_urb2d(16, 1) = 0.0
    ! urban_var_init (module_sf_urban.F:2840-2889) for every column
    do i = 1, ncase
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
      trl_urb3d(i, 1, 1) = tslb(i, 1, 1) + 0.
      trl_urb3d(i, 2, 1) = 0.5 * (tslb(i, 1, 1) + tslb(i, 2, 1))
      trl_urb3d(i, 3, 1) = tslb(i, 2, 1) + 0.
      trl_urb3d(i, 4, 1) = tslb(i, 2, 1) + (tslb(i, 3, 1) - tslb(i, 2, 1)) * 0.29
      tbl_urb3d(i, :, 1) = trl_urb3d(i, :, 1)
      tgrl_urb3d(i, :, 1) = trl_urb3d(i, :, 1)
      tgl_urb3d(i, :, 1) = tslb(i, :, 1)
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
  end subroutine build_fixture

  ! Per-step forcing.  Columns 9-12 are night, 13-15 rain; the rest day.
  subroutine forcing(step)
    integer, intent(in) :: step
    do i = 1, ncase
      swdown(i, 1) = 750.0 - 30.0 * real(step - 1)
      gsw(i, 1) = swdown(i, 1) * 0.82
      swddir(i, 1) = swdown(i, 1) * 0.8
      swddif(i, 1) = swdown(i, 1) * 0.2
      u_phy(i, 1, 1) = 3.0 + 0.25 * real(i)
      v_phy(i, 1, 1) = -1.5 + 0.1 * real(i)
      u_phy(i, 2, 1) = u_phy(i, 1, 1)
      v_phy(i, 2, 1) = v_phy(i, 1, 1)
      rainbl(i, 1) = 0.0
      if (i >= 9 .and. i <= 12) then
        swdown(i, 1) = 0.0
        gsw(i, 1) = 0.0
        swddir(i, 1) = 0.0
        swddif(i, 1) = 0.0
        omg_urb2d(i, 1) = 2.8
        cosz_urb2d(i, 1) = -0.3
        t3d(i, 1, 1) = 288.0
      end if
      if (i >= 13 .and. i <= 15) rainbl(i, 1) = 0.1 * real(i - 12) * real(step)
      if (i == 10) then                 ! calm night: UA below 1 m/s clamps to 1
        u_phy(i, 1, 1) = 0.3
        v_phy(i, 1, 1) = 0.2
      end if
    end do
  end subroutine forcing

  subroutine write_header()
    character(len=4096) :: h
    h = 'step,case,ivgtyp,utype,frc,u1,v1,omg,swdown,rainbl_grid'
    h = trim(h) // ',tap_t1,tap_sheat,tap_eta_kinematic,tap_eta,tap_ssoil,' // &
        'tap_albedok,tap_q1,tap_sfctmp,tap_q2k,tap_sfcprs,tap_zlvl,tap_soldn,' // &
        'tap_rainbl,tap_chs,tap_chs2,tap_cqs2,tap_ust,tap_znt,tap_glw,tapped'
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
    write(u, '(A)') trim(h)
  end subroutine write_header

  ! The pre-call half of each row lives in these buffers until write_post.
  subroutine write_pre(step)
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
      pre_in(:, i) = (/ real(step), real(i), real(ivgtyp(i, 1)), real(utype_urb2d(i, 1)), &
                        frc_urb2d(i, 1), u_phy(i, 1, 1), v_phy(i, 1, 1),        &
                        omg_urb2d(i, 1), swdown(i, 1), rainbl(i, 1) /)
    end do
  end subroutine write_pre

  subroutine write_post()
    integer :: k
    real :: post(44)
    do i = 1, ncase
      write(u, '(I0,",",I0,",",I0,",",I0)', advance='no') nint(pre_in(1, i)),   &
           nint(pre_in(2, i)), nint(pre_in(3, i)), nint(pre_in(4, i))
      write(u, '(6(",",ES16.8E3))', advance='no') pre_in(5:10, i)
      write(u, '(19(",",ES16.8E3))', advance='no') tap(:, i)
      write(u, '(",",I0)', advance='no') merge(1, 0, tapped(i))
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
      write(u, '(14(",",ES16.8E3))') ts_urb2d(i, 1), sh_urb2d(i, 1),             &
           lh_urb2d(i, 1), g_urb2d(i, 1), rn_urb2d(i, 1), psim_urb2d(i, 1),       &
           psih_urb2d(i, 1), gz1oz0_urb2d(i, 1), u10_urb2d(i, 1), v10_urb2d(i, 1), &
           th2_urb2d(i, 1), q2_urb2d(i, 1), ust_urb2d(i, 1), akms_urb2d(i, 1)
    end do
  end subroutine write_post

end program run_ucm_noah
