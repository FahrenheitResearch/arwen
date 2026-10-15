program run_noah_seaice_oracle
  ! Drive the WRF v4.6.1 Noah sea-ice driver `seaice_noah`
  ! (phys/module_sf_noah_seaice_drv.F) over a 1-D row of independent columns
  ! and dump every driver input and every driver output as CSV.
  !
  ! The surface driver calls seaice_noah after lsm / lsm_mosaic on every
  ! Noah run (phys/module_surface_driver.F:2917-2929).  gpuwm's port is
  ! noah_seaice_column in gpuwm/core/kernels/noah.cu, whose scope is exactly
  ! this subroutine's: the per-column prep, SFLX_SEAICE and its subtree, and
  ! the post-update.  Every value in the CSV is a word this program placed in
  ! an input array before the call or a word seaice_noah wrote during it;
  ! `sfcprs` is the one labelled echo (the driver forms it from p8w3d, gpuwm
  ! takes it as an input), recomputed here in the same float32 arithmetic.
  !
  ! The namelist arguments are WRF's Registry defaults (Registry.EM_COMMON):
  ! seaice_albedo_opt = 0, seaice_thickness_opt = 0 with default 3.0,
  ! seaice_snowdepth_opt = 0 with min 0.001 and max 1.E10.  Only
  ! seaice_albedo_default (argument 3) and frpcpn (argument 2) are varied,
  ! because those are the two gpuwm exposes.

  use module_sf_noah_seaice_drv, only: seaice_noah
  implicit none

  integer, parameter :: ncase = 24
  integer, parameter :: nsoil = 4
  real, parameter :: dt = 60.0

  real, dimension(ncase, 2, 1) :: t3d, qv3d, p8w3d, dz8w
  real, dimension(ncase, 1) :: sr, glw, swdown, rainbl, snoalb, qgh, xice
  real, dimension(ncase, 1) :: albsi, icedepth, snowsi
  real, dimension(ncase, 1) :: emiss, albedo, z0, tsk, snow, snowc, snowh
  real, dimension(ncase, 1) :: chs, chs2, cqs2, rib, znt, lh, hfx, qfx
  real, dimension(ncase, 1) :: potevp, grdflx, qsfc, acsnow, acsnom, snopcx
  real, dimension(ncase, 1) :: sfcrunoff, noahres, sfcprs
  real, dimension(ncase, nsoil, 1) :: tslb
  real, dimension(ncase, 2, 1) :: b_t_bep, b_q_bep, rho

  ! saved inputs
  real, dimension(ncase, 1) :: emiss0, albedo0, z00, tsk0, snow0, snowc0
  real, dimension(ncase, 1) :: snowh0, chs0, chs20, cqs20, znt0, lh0, hfx0
  real, dimension(ncase, 1) :: qfx0, potevp0, grdflx0, qsfc0, acsnow0
  real, dimension(ncase, 1) :: acsnom0, snopcx0, sfcrunoff0, noahres0
  real, dimension(ncase, nsoil, 1) :: tslb0

  character(len=256) :: out_path, arg
  logical :: frpcpn
  real :: albedo_default, subn
  integer :: i, u

  call get_command_argument(1, out_path)
  if (len_trim(out_path) == 0) then
    write(*, '(A)') 'usage: run_seaice OUT.csv [frpcpn T/F] [seaice_albedo_default]'
    error stop 2
  end if
  frpcpn = .false.
  albedo_default = 0.65
  call get_command_argument(2, arg)
  if (len_trim(arg) > 0) frpcpn = (trim(arg) == 'T')
  call get_command_argument(3, arg)
  if (len_trim(arg) > 0) read(arg, *) albedo_default

  ! the smallest positive float32 subnormal, by bit pattern
  subn = transfer(1, 1.0)

  call build_fixture()
  do i = 1, ncase
    sfcprs(i, 1) = (p8w3d(i, 2, 1) + p8w3d(i, 1, 1)) * 0.5
  end do
  call save_state()

  call seaice_noah(0, albedo_default, 0, 3.0, 0, 1.e10, 0.001,          &
       t3d, qv3d, p8w3d, dz8w, nsoil, dt, frpcpn, sr,                    &
       glw, swdown, rainbl, snoalb, qgh, xice, 0.5,                      &
       albsi, icedepth, snowsi,                                          &
       tslb, emiss, albedo, z0, tsk, snow, snowc, snowh,                 &
       chs, chs2, cqs2,                                                  &
       rib, znt, lh, hfx, qfx, potevp, grdflx, qsfc, acsnow,             &
       acsnom, snopcx, sfcrunoff, noahres,                               &
       0, b_t_bep, b_q_bep, rho,                                         &
       1, ncase, 1, 1, 1, 2,                                             &
       1, ncase, 1, 1, 1, 2,                                             &
       1, ncase, 1, 1, 1, 1)

  call write_csv()

contains

  subroutine build_fixture()
    integer :: i
    ! ---- base column: thick cold sea ice under a 5 cm snow cover ---------
    do i = 1, ncase
      p8w3d(i, 1, 1) = 101000.0
      p8w3d(i, 2, 1) = 100000.0
      t3d(i, 1, 1) = 255.0
      t3d(i, 2, 1) = 254.0
      qv3d(i, 1, 1) = 0.0008
      qv3d(i, 2, 1) = 0.0008
      dz8w(i, 1, 1) = 50.0
      dz8w(i, 2, 1) = 50.0
      sr(i, 1) = 0.0
      glw(i, 1) = 200.0
      swdown(i, 1) = 100.0
      rainbl(i, 1) = 0.0
      snoalb(i, 1) = 0.75
      qgh(i, 1) = 0.0012
      xice(i, 1) = 1.0
      albsi(i, 1) = -1.e36
      icedepth(i, 1) = -1.e36
      snowsi(i, 1) = -1.e36
      emiss(i, 1) = 0.98
      albedo(i, 1) = 0.65
      z0(i, 1) = 0.001
      tsk(i, 1) = 258.0
      snow(i, 1) = 50.0
      snowc(i, 1) = 1.0
      snowh(i, 1) = 0.2
      chs(i, 1) = 0.015
      chs2(i, 1) = 0.02
      cqs2(i, 1) = 0.02
      rib(i, 1) = 0.05
      znt(i, 1) = 0.001
      lh(i, 1) = 0.0
      hfx(i, 1) = 0.0
      qfx(i, 1) = 0.0
      potevp(i, 1) = 0.0
      grdflx(i, 1) = 0.0
      qsfc(i, 1) = 0.001
      acsnow(i, 1) = 0.0
      acsnom(i, 1) = 0.0
      snopcx(i, 1) = 0.0
      sfcrunoff(i, 1) = 0.0
      noahres(i, 1) = 0.0
      tslb(i, 1, 1) = 260.0
      tslb(i, 2, 1) = 264.0
      tslb(i, 3, 1) = 267.5
      tslb(i, 4, 1) = 270.5
      b_t_bep(i, :, 1) = 0.0
      b_q_bep(i, :, 1) = 0.0
      rho(i, :, 1) = 1.3
    end do

    ! 1  base: cold, SNOPAC's T12 <= TFREEZ arm                -- unchanged

    ! 2  melting snow on ice: T12 > TFREEZ, FLX3 > 0
    t3d(2, 1, 1) = 278.0
    tsk(2, 1) = 272.5
    swdown(2, 1) = 600.0
    glw(2, 1) = 320.0
    qv3d(2, 1, 1) = 0.005
    qgh(2, 1) = 0.0066
    snow(2, 1) = 30.0
    snowh(2, 1) = 0.1
    tslb(2, :, 1) = 271.0

    ! 3  thin snow melted out within the step: ESD-SNOMLT < ESDMIN
    t3d(3, 1, 1) = 285.0
    tsk(3, 1) = 273.2
    swdown(3, 1) = 900.0
    glw(3, 1) = 360.0
    qv3d(3, 1, 1) = 0.008
    qgh(3, 1) = 0.010
    snow(3, 1) = 10.05
    snowh(3, 1) = 0.03
    tslb(3, :, 1) = 271.2

    ! 4  SNEQV < 0.01 m on entry: the 0.01 m / 0.05 m floor
    snow(4, 1) = 5.0
    snowh(4, 1) = 0.02

    ! 5  snowfall on cold ice, STC(1) < TFREEZ-5: SNOW_NEW, density reset
    rainbl(5, 1) = 2.0
    sr(5, 1) = 1.0
    t3d(5, 1, 1) = 260.0

    ! 6  freezing rain (SFCTMP > 273.15, T1 <= TFREEZ): FLX2, FRZGRA
    rainbl(6, 1) = 2.0
    t3d(6, 1, 1) = 275.0
    tsk(6, 1) = 270.0
    qv3d(6, 1, 1) = 0.004
    qgh(6, 1) = 0.0035
    tslb(6, :, 1) = 270.5

    ! 7  rain on melting ice, T1 > 273.15: PENMAN's ELCP/LSUBC arm, CPH2O
    rainbl(7, 1) = 2.0
    t3d(7, 1, 1) = 280.0
    tsk(7, 1) = 273.6
    qv3d(7, 1, 1) = 0.006
    qgh(7, 1) = 0.0068
    tslb(7, :, 1) = 271.3

    ! 8  dry air over sunlit ice: ETP > 0, sublimation
    qv3d(8, 1, 1) = 0.0003
    qgh(8, 1) = 0.0025
    t3d(8, 1, 1) = 266.0
    tsk(8, 1) = 264.0
    swdown(8, 1) = 500.0
    glw(8, 1) = 250.0

    ! 9  moist stable air, RIBB >= 0.1, FDOWN > 150, ETP <= 0
    qv3d(9, 1, 1) = 0.004
    qgh(9, 1) = 0.0015
    t3d(9, 1, 1) = 268.0
    tsk(9, 1) = 262.0
    rib(9, 1) = 0.3
    glw(9, 1) = 300.0
    swdown(9, 1) = 100.0

    ! 10 SNOW is the smallest positive subnormal: SNOW > 0 is TRUE in WRF
    snow(10, 1) = subn
    snowh(10, 1) = 0.0
    snowc(10, 1) = 0.0
    t3d(10, 1, 1) = 266.0
    tsk(10, 1) = 265.0

    ! 11 bare ice: SNOW = SNOWH = SNOWC = 0
    snow(11, 1) = 0.0
    snowh(11, 1) = 0.0
    snowc(11, 1) = 0.0
    t3d(11, 1, 1) = 270.0
    tsk(11, 1) = 268.0

    ! 12 warm ground, partial cover: SNOWC blend and the T1 > 273 damping
    t3d(12, 1, 1) = 274.0
    tsk(12, 1) = 273.3
    tslb(12, :, 1) = 271.0
    snow(12, 1) = 40.0
    snowh(12, 1) = 0.15
    snowc(12, 1) = 0.6
    qv3d(12, 1, 1) = 0.004
    qgh(12, 1) = 0.0047

    ! 13 XICE below the threshold: the driver skips the column
    xice(13, 1) = 0.4
    t3d(13, 1, 1) = 275.0

    ! 14 XICE exactly at the threshold: the column runs
    xice(14, 1) = 0.5

    ! 15 snowfall onto ice near 0 C, STC(1) >= TFREEZ-5, T1 and SFCTMP < 273:
    !    the second density reset
    rainbl(15, 1) = 1.5
    sr(15, 1) = 1.0
    t3d(15, 1, 1) = 270.0
    tsk(15, 1) = 271.0
    tslb(15, :, 1) = 270.5

    ! 16 snowfall with SNOWC <= 0.99: no density reset
    rainbl(16, 1) = 1.5
    sr(16, 1) = 1.0
    snowc(16, 1) = 0.5
    t3d(16, 1, 1) = 262.0

    ! 17 snowfall at SFCTMP just under 273.15 onto warm ice: neither reset,
    !    SNOW_NEW's (TEMPC+15)**1.5 density
    rainbl(17, 1) = 1.0
    sr(17, 1) = 1.0
    t3d(17, 1, 1) = 273.0
    tsk(17, 1) = 273.4
    tslb(17, :, 1) = 271.2

    ! 18 QSFC above Q1 on entry: CQS2 keeps its value (Q1 .GT. QSFC false)
    qsfc(18, 1) = 0.05

    ! 19 mixed-phase precipitation, SR = 0.3: snow without frpcpn, freezing
    !    rain with it (discriminates the flag)
    rainbl(19, 1) = 1.0
    sr(19, 1) = 0.3
    t3d(19, 1, 1) = 270.0
    tsk(19, 1) = 268.0

    ! 20 SNOWC is the smallest positive subnormal on warm ground: the
    !    T1 > 273 .AND. SNOWC > 0 test is TRUE in WRF
    snowc(20, 1) = subn
    t3d(20, 1, 1) = 274.0
    tsk(20, 1) = 273.5
    tslb(20, :, 1) = 271.0
    qv3d(20, 1, 1) = 0.004
    qgh(20, 1) = 0.0047

    ! 21 deep snow, strong night-time cooling, weak exchange
    snow(21, 1) = 300.0
    snowh(21, 1) = 1.0
    t3d(21, 1, 1) = 240.0
    tsk(21, 1) = 238.0
    swdown(21, 1) = 0.0
    glw(21, 1) = 150.0
    chs(21, 1) = 0.002
    qv3d(21, 1, 1) = 0.0002
    qgh(21, 1) = 0.0002
    tslb(21, 1, 1) = 245.0
    tslb(21, 2, 1) = 255.0
    tslb(21, 3, 1) = 263.0
    tslb(21, 4, 1) = 269.0

    ! 22 previous accumulators nonzero: POTEVP/ACSNOM/SNOPCX/SFCRUNOFF add
    t3d(22, 1, 1) = 279.0
    tsk(22, 1) = 273.0
    swdown(22, 1) = 500.0
    glw(22, 1) = 320.0
    tslb(22, :, 1) = 271.2
    potevp(22, 1) = 0.25
    acsnom(22, 1) = 3.0
    snopcx(22, 1) = -1000.0
    sfcrunoff(22, 1) = 2.0
    acsnow(22, 1) = 4.0
    rainbl(22, 1) = 0.5

    ! 23 an emissivity and albedo on entry that the scheme overwrites
    emiss(23, 1) = 0.95
    albedo(23, 1) = 0.30
    swdown(23, 1) = 400.0

    ! 24 dense old snow: SNEQV/SNOWH = 0.45 on entry
    snow(24, 1) = 90.0
    snowh(24, 1) = 0.2
  end subroutine build_fixture

  subroutine save_state()
    emiss0 = emiss; albedo0 = albedo; z00 = z0; tsk0 = tsk; snow0 = snow
    snowc0 = snowc; snowh0 = snowh; chs0 = chs; chs20 = chs2; cqs20 = cqs2
    znt0 = znt; lh0 = lh; hfx0 = hfx; qfx0 = qfx; potevp0 = potevp
    grdflx0 = grdflx; qsfc0 = qsfc; acsnow0 = acsnow; acsnom0 = acsnom
    snopcx0 = snopcx; sfcrunoff0 = sfcrunoff; noahres0 = noahres
    tslb0 = tslb
  end subroutine save_state

  subroutine write_csv()
    integer :: i
    open(newunit=u, file=trim(out_path), status='replace', action='write')
    write(u, '(A)') 'case,psfc,sfcprs,sfctmp,qv1,qgh,glw,swdown,rainbl,sr,' // &
      'snoalb,xice,rib,chs,' //                                                &
      'cqs2_in,chs2_in,tsk_in,emiss_in,albedo_in,z0_in,snow_in,snowc_in,' //   &
      'snowh_in,znt_in,lh_in,hfx_in,qfx_in,potevp_in,grdflx_in,qsfc_in,' //    &
      'acsnow_in,acsnom_in,snopcx_in,sfcrunoff_in,noahres_in,' //              &
      'tslb1_in,tslb2_in,tslb3_in,tslb4_in,' //                                &
      'cqs2,chs2,tsk,emiss,albedo,z0,snow,snowc,snowh,znt,lh,hfx,qfx,' //      &
      'potevp,grdflx,qsfc,acsnow,acsnom,snopcx,sfcrunoff,noahres,' //          &
      'tslb1,tslb2,tslb3,tslb4'
    do i = 1, ncase
      write(u, '(I0)', advance='no') i
      write(u, '(13(",",ES24.16E3))', advance='no')                            &
        p8w3d(i, 1, 1), sfcprs(i, 1), t3d(i, 1, 1), qv3d(i, 1, 1), qgh(i, 1), &
        glw(i, 1), swdown(i, 1), rainbl(i, 1), sr(i, 1), snoalb(i, 1),         &
        xice(i, 1), rib(i, 1), chs0(i, 1)
      write(u, '(25(",",ES24.16E3))', advance='no')                            &
        cqs20(i, 1), chs20(i, 1), tsk0(i, 1), emiss0(i, 1), albedo0(i, 1),     &
        z00(i, 1), snow0(i, 1), snowc0(i, 1), snowh0(i, 1), znt0(i, 1),        &
        lh0(i, 1), hfx0(i, 1), qfx0(i, 1), potevp0(i, 1), grdflx0(i, 1),       &
        qsfc0(i, 1), acsnow0(i, 1), acsnom0(i, 1), snopcx0(i, 1),              &
        sfcrunoff0(i, 1), noahres0(i, 1),                                      &
        tslb0(i, 1, 1), tslb0(i, 2, 1), tslb0(i, 3, 1), tslb0(i, 4, 1)
      write(u, '(25(",",ES24.16E3))')                                          &
        cqs2(i, 1), chs2(i, 1), tsk(i, 1), emiss(i, 1), albedo(i, 1),          &
        z0(i, 1), snow(i, 1), snowc(i, 1), snowh(i, 1), znt(i, 1), lh(i, 1),   &
        hfx(i, 1), qfx(i, 1), potevp(i, 1), grdflx(i, 1), qsfc(i, 1),          &
        acsnow(i, 1), acsnom(i, 1), snopcx(i, 1), sfcrunoff(i, 1),             &
        noahres(i, 1),                                                         &
        tslb(i, 1, 1), tslb(i, 2, 1), tslb(i, 3, 1), tslb(i, 4, 1)
    end do
    close(u)
  end subroutine write_csv

end program run_noah_seaice_oracle
