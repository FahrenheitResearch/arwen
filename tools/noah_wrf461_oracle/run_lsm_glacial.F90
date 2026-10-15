program run_noah_lsm_glacial_oracle
  ! The land-ice companion of run_lsm.F90: the same driver call, the same CSV
  ! layout, the same rule that every value is an input this program wrote or
  ! an output `lsm` wrote -- but every column is LAND ICE (ivgtyp == isice =
  ! 15, xland = 1, xice = 0), so `lsm` takes its ICE == -1 arm
  ! (phys/module_sf_noahdrv.F:1137-1190) and calls SFLX_GLACIAL
  ! (phys/module_sf_noahlsm_glacial_only.F) on every one.
  !
  ! run_lsm.F90 carries exactly one land-ice column (its case 28), which was
  ! enough to measure the size of a restriction but is not enough to grade a
  ! port.  The 24 columns below are chosen to reach SFLX_GLACIAL's branches:
  ! the SNEQV < 0.10 m entry reset (with and without snow on entry); snowfall
  ! (SNOWNG) at TEMPC <= -15 and above it, with and without SNCOVR > 0.99 and
  ! each of the two SNDENS = 0.2 overrides; freezing rain (FRZGRA, FLX2) and
  ! plain rain (PRCPF, FLX1, the T1 > 273.15 PENMAN arm); sub-freezing T12
  ! and the melting arm (T1 = TFREEZ, FLX3, SNOMLT, SNOWPACK's TSNOWC >= 0
  ! densification); ETP <= 0 with each arm of the RIBB/FDOWN dew adjustment
  ! and ETP > 0 sublimation; ALCALC's snow aging with SNOALB2 above and below
  ! ALB; SNOWZ0 with BURIAL on both sides of 0.0007; SNOWPACK's 0.05 and 0.40
  ! density clamps; and a steep ice temperature profile under a warm and a
  ! cold TBOT (HRTICE/HSTEP/ROSR12).
  !
  ! build.sh runs it twice: frpcpn = F (FFROZP from SFCTMP <= 273.15) and
  ! frpcpn = T (FFROZP = SR, which the fixture sets per column so that the two
  ! passes take different SNOWNG/FRZGRA arms on some columns).  opt_thcnd,
  ! usemonalb and rdlai2d do not reach SFLX_GLACIAL or the ICE == -1 arm.

  use module_sf_noahdrv, only: lsm, soil_veg_gen_parm
  implicit none

  integer, parameter :: ncase = 24
  integer, parameter :: nsoil = 4
  real, parameter :: dt = 60.0
  real, parameter :: r_d_over_cp = 287.0 / (7.0 * 287.0 / 2.0)   ! WRF ROVCP

  ! --- driver argument arrays, one column per case ---------------------------
  real, dimension(ncase, 2, 1) :: dz8w, qv3d, p8w3d, t3d
  real, dimension(ncase, 1) :: tsk, hfx, qfx, lh, grdflx, qgh, gsw, swdown
  real, dimension(ncase, 1) :: swddir, swddif, glw, smstav, smstot
  real, dimension(ncase, 1) :: sfcrunoff, udrunoff, vegfra, albedo, albbck
  real, dimension(ncase, 1) :: znt, z0, tmn, xland, xice, emiss, embck
  real, dimension(ncase, 1) :: snowc, qsfc, rainbl, snow, canwat
  real, dimension(ncase, 1) :: chs, chs2, cqs2, cpm, sr, chklowq, lai, qz0
  real, dimension(ncase, 1) :: snowh, snoalb, shdmin, shdmax, snotime
  real, dimension(ncase, 1) :: acsnom, acsnow, snopcx, potevp, rib, noahres
  real, dimension(ncase, 1) :: flx4_2d, fvb_2d, fbur_2d, fgsn_2d, ust_urb2d
  real, dimension(ncase, 1) :: frc_urb2d
  real, dimension(ncase, 1) :: sda_hfx, sda_qfx, hfx_both, qfx_both, qnorm
  real, dimension(ncase, nsoil, 1) :: smois, tslb, sh2o, smcrel
  integer, dimension(ncase, 1) :: ivgtyp, isltyp, utype_urb2d
  real, dimension(nsoil) :: dzs

  ! --- saved copies of everything the driver overwrites ----------------------
  real, dimension(ncase, 1) :: tsk0, hfx0, qfx0, lh0, grdflx0, qsfc0
  real, dimension(ncase, 1) :: canwat0, snow0, snowc0, snowh0, albedo0
  real, dimension(ncase, 1) :: albbck0, emiss0, znt0, z00, snotime0, lai0
  real, dimension(ncase, 1) :: smstav0, smstot0, sfcrunoff0, udrunoff0
  real, dimension(ncase, 1) :: acsnom0, acsnow0, snopcx0, potevp0, rib0
  real, dimension(ncase, 1) :: chs0, chs20, cqs20
  real, dimension(ncase, nsoil, 1) :: smois0, tslb0, sh2o0
  real, dimension(ncase, 1) :: sfcprs, zlvl

  character(len=1024) :: out_path
  character(len=32) :: arg
  integer :: icase, ns, u, opt_thcnd, itimestep
  logical :: frpcpn, usemonalb, rdlai2d
  real :: subn, nzero

  call get_command_argument(1, out_path)
  if (len_trim(out_path) == 0) then
    write(*, '(A)') 'usage: run_lsm_glacial OUT.csv [opt_thcnd] [frpcpn] [usemonalb] [rdlai2d]'
    error stop 2
  end if
  opt_thcnd = 1
  frpcpn = .false.
  usemonalb = .false.
  rdlai2d = .false.
  itimestep = 2
  call get_command_argument(2, arg)
  if (len_trim(arg) > 0) read(arg, *) opt_thcnd
  call get_command_argument(3, arg)
  if (len_trim(arg) > 0) frpcpn = (trim(arg) == 'T')
  call get_command_argument(4, arg)
  if (len_trim(arg) > 0) usemonalb = (trim(arg) == 'T')
  call get_command_argument(5, arg)
  if (len_trim(arg) > 0) rdlai2d = (trim(arg) == 'T')

  ! The smallest positive float32 subnormal and a negative zero, by bit
  ! pattern rather than by a decimal literal the compiler is free to fold.
  subn = transfer(1, 1.0)
  nzero = sign(0.0, -1.0)

  call soil_veg_gen_parm('MODIFIED_IGBP_MODIS_NOAH', 'STAS')

  dzs = (/ 0.10, 0.30, 0.60, 1.00 /)

  call build_fixture()

  ! --- echoes of the two quantities the driver derives from its 3-D inputs ---
  do icase = 1, ncase
    sfcprs(icase, 1) = (p8w3d(icase, 2, 1) + p8w3d(icase, 1, 1)) * 0.5
    zlvl(icase, 1) = 0.5 * dz8w(icase, 1, 1)
  end do

  call save_state()

  call lsm(dz8w, qv3d, p8w3d, t3d, tsk,                              &
           hfx, qfx, lh, grdflx, qgh, gsw, swdown, swddir, swddif,   &
           glw, smstav, smstot,                                      &
           sfcrunoff, udrunoff, ivgtyp, isltyp, 13, 15, vegfra,      &
           albedo, albbck, znt, z0, tmn, xland, xice, emiss, embck,  &
           snowc, qsfc, rainbl, 'MODIFIED_IGBP_MODIS_NOAH',          &
           nsoil, dt, dzs, itimestep,                                &
           smois, tslb, snow, canwat,                                &
           chs, chs2, cqs2, cpm, r_d_over_cp, sr, chklowq, lai, qz0, &
           .false., frpcpn,                                          &
           sh2o, snowh,                                              &
           snoalb = snoalb, shdmin = shdmin, shdmax = shdmax,        &
           snotime = snotime,                                        &
           acsnom = acsnom, acsnow = acsnow,                         &
           snopcx = snopcx, potevp = potevp, smcrel = smcrel,        &
           xice_threshold = 0.5,                                     &
           rdlai2d = rdlai2d, usemonalb = usemonalb,                 &
           rib = rib, noahres = noahres, opt_thcnd = opt_thcnd,      &
           ua_phys = .false., flx4_2d = flx4_2d, fvb_2d = fvb_2d,    &
           fbur_2d = fbur_2d, fgsn_2d = fgsn_2d,                     &
           ids = 1, ide = ncase, jds = 1, jde = 1, kds = 1, kde = 2, &
           ims = 1, ime = ncase, jms = 1, jme = 1, kms = 1, kme = 2, &
           its = 1, ite = ncase, jts = 1, jte = 1, kts = 1, kte = 1, &
           sf_urban_physics = 0, ust_urb2d = ust_urb2d,              &
           num_roof_layers = 1, num_wall_layers = 1,                 &
           num_road_layers = 1, julian = 1, julyr = 1974,            &
           frc_urb2d = frc_urb2d, utype_urb2d = utype_urb2d,         &
           num_urban_ndm = 1, urban_map_zrd = 1, urban_map_zwd = 1,  &
           urban_map_gd = 1, urban_map_zd = 1, urban_map_zdf = 1,    &
           urban_map_bd = 1, urban_map_wd = 1, urban_map_gbd = 1,    &
           urban_map_fbd = 1, urban_map_zgrd = 1, num_urban_hi = 1,  &
           sda_hfx = sda_hfx, sda_qfx = sda_qfx,                     &
           hfx_both = hfx_both, qfx_both = qfx_both, qnorm = qnorm,  &
           fasdas = 0)

  call write_csv()

contains

  subroutine build_fixture()
    integer :: i

    ! ---- the base column: warm, vegetated, unfrozen, snow-free land --------
    do i = 1, ncase
      ivgtyp(i, 1) = 10          ! MODIS grassland
      isltyp(i, 1) = 8           ! silty clay loam
      p8w3d(i, 1, 1) = 98000.0   ! surface pressure
      p8w3d(i, 2, 1) = 97000.0   ! -> SFCPRS = 97500 exactly
      t3d(i, 1, 1) = 290.0
      t3d(i, 2, 1) = 289.0
      qv3d(i, 1, 1) = 0.008
      qv3d(i, 2, 1) = 0.008
      dz8w(i, 1, 1) = 60.0
      dz8w(i, 2, 1) = 60.0
      qgh(i, 1) = 0.012
      gsw(i, 1) = 400.0
      swdown(i, 1) = 500.0
      swddir(i, 1) = 400.0
      swddif(i, 1) = 100.0
      glw(i, 1) = 330.0
      rainbl(i, 1) = 0.0
      sr(i, 1) = 0.0
      chs(i, 1) = 0.02
      chs2(i, 1) = 0.03
      cqs2(i, 1) = 0.03
      cpm(i, 1) = 1010.0
      qz0(i, 1) = 0.005
      rib(i, 1) = -0.1
      vegfra(i, 1) = 60.0
      shdmin(i, 1) = 10.0
      shdmax(i, 1) = 80.0
      tmn(i, 1) = 285.0
      xland(i, 1) = 1.0
      xice(i, 1) = 0.0
      snoalb(i, 1) = 0.7
      embck(i, 1) = 0.95
      tsk(i, 1) = 291.0
      hfx(i, 1) = 0.0
      qfx(i, 1) = 0.0
      lh(i, 1) = 0.0
      grdflx(i, 1) = 0.0
      qsfc(i, 1) = 0.010
      canwat(i, 1) = 0.3
      snow(i, 1) = 0.0
      snowc(i, 1) = 0.0
      snowh(i, 1) = 0.0
      albedo(i, 1) = 0.2
      albbck(i, 1) = 0.2
      emiss(i, 1) = 0.95
      znt(i, 1) = 0.1
      z0(i, 1) = 0.1
      snotime(i, 1) = 0.0
      lai(i, 1) = 3.0
      smstav(i, 1) = 0.0
      smstot(i, 1) = 0.0
      sfcrunoff(i, 1) = 0.0
      udrunoff(i, 1) = 0.0
      acsnom(i, 1) = 0.0
      acsnow(i, 1) = 0.0
      snopcx(i, 1) = 0.0
      potevp(i, 1) = 0.0
      frc_urb2d(i, 1) = 0.0
      utype_urb2d(i, 1) = 1
      sda_hfx(i, 1) = 0.0
      sda_qfx(i, 1) = 0.0
      hfx_both(i, 1) = 0.0
      qfx_both(i, 1) = 0.0
      qnorm(i, 1) = 0.0
      do ns = 1, nsoil
        smois(i, ns, 1) = 0.28
        sh2o(i, ns, 1) = 0.28
      end do
      tslb(i, 1, 1) = 289.0
      tslb(i, 2, 1) = 288.0
      tslb(i, 3, 1) = 287.0
      tslb(i, 4, 1) = 286.0
    end do

    ! ---- every column is land ice ----------------------------------------
    do i = 1, ncase
      ivgtyp(i, 1) = 15          ! MODIS snow and ice (ISICE)
      isltyp(i, 1) = 16          ! STAS "OTHER (land-ice)"
      t3d(i, 1, 1) = 258.0
      t3d(i, 2, 1) = 257.5
      qv3d(i, 1, 1) = 0.0008
      qv3d(i, 2, 1) = 0.0008
      qgh(i, 1) = 0.0012
      swdown(i, 1) = 300.0
      gsw(i, 1) = 100.0
      swddir(i, 1) = 240.0
      swddif(i, 1) = 60.0
      glw(i, 1) = 220.0
      chs(i, 1) = 0.01
      chs2(i, 1) = 0.01
      cqs2(i, 1) = 0.01
      rib(i, 1) = 0.05
      vegfra(i, 1) = 0.0
      shdmin(i, 1) = 0.0
      shdmax(i, 1) = 0.0
      tmn(i, 1) = 255.0
      snoalb(i, 1) = 0.75
      embck(i, 1) = 0.95
      albbck(i, 1) = 0.55
      albedo(i, 1) = 0.7
      emiss(i, 1) = 0.95
      z0(i, 1) = 0.001
      znt(i, 1) = 0.001
      tsk(i, 1) = 255.0
      canwat(i, 1) = 0.0
      lai(i, 1) = 0.01
      snow(i, 1) = 200.0
      snowh(i, 1) = 0.6
      snowc(i, 1) = 1.0
      snotime(i, 1) = 3600.0
      qsfc(i, 1) = 0.0007
      sr(i, 1) = 0.0
      do ns = 1, nsoil
        smois(i, ns, 1) = 1.0
        sh2o(i, ns, 1) = 1.0
        tslb(i, ns, 1) = 256.0
      end do
    end do

    ! ---- per-case overrides -------------------------------------------------
    ! 1  cold dry ice under a deep snowpack: T12 <= TFREEZ, ETP > 0 -- base

    ! 2  surface melt: warm sunny air over a 0 C snowpack.  T12 > TFREEZ, so
    !    T1 = TFREEZ, FLX3 > 0, SNOMLT > 0 and SNOWPACK densifies (TSNOWC = 0).
    t3d(2, 1, 1) = 279.0
    qv3d(2, 1, 1) = 0.004
    qgh(2, 1) = 0.0062
    tsk(2, 1) = 273.0
    tslb(2, :, 1) = 272.5
    swdown(2, 1) = 800.0
    glw(2, 1) = 320.0
    snow(2, 1) = 500.0
    snowh(2, 1) = 1.4
    rib(2, 1) = 0.02

    ! 3  thin snow on entry, SNEQV < 0.10 m: reset to 0.10 m / 0.50 m
    snow(3, 1) = 40.0
    snowh(3, 1) = 0.12

    ! 4  bare ice, no snow at all on entry: the same reset from zero
    snow(4, 1) = 0.0
    snowh(4, 1) = 0.0
    snowc(4, 1) = 0.0

    ! 5  snowfall at TEMPC <= -15 (DSNEW = 0.05), SNCOVR > 0.99 and
    !    STC(1) < TFREEZ - 5 -> SNDENS = 0.2
    t3d(5, 1, 1) = 252.0
    rainbl(5, 1) = 0.6
    sr(5, 1) = 1.0

    ! 6  snowfall at -5 C: DSNEW from the power law; T1 < 273 and
    !    SFCTMP < 273 -> SNDENS = 0.2 (STC(1) warm enough to miss the other
    !    override).  SR = 0.3 sends the frpcpn pass to FRZGRA instead.
    t3d(6, 1, 1) = 268.15
    tsk(6, 1) = 267.0
    tslb(6, :, 1) = 270.0
    rainbl(6, 1) = 0.4
    sr(6, 1) = 0.3

    ! 7  snowfall at SFCTMP just below 273.15 with T1 above 273: neither
    !    SNDENS override fires, the SNOW_NEW blend stands
    t3d(7, 1, 1) = 273.1
    qv3d(7, 1, 1) = 0.0035
    qgh(7, 1) = 0.0045
    tsk(7, 1) = 273.05
    tslb(7, :, 1) = 271.0
    rainbl(7, 1) = 0.3
    sr(7, 1) = 0.9

    ! 8  snowfall with SNCOVR <= 0.99 on entry: no override is considered
    snowc(8, 1) = 0.5
    t3d(8, 1, 1) = 262.0
    rainbl(8, 1) = 0.2
    sr(8, 1) = 1.0

    ! 9  freezing rain: SFCTMP > 273.15 (FFROZP = 0), T1 <= TFREEZ ->
    !    FRZGRA, SN_NEW added, FLX2 = -LSUBF * PRCP
    t3d(9, 1, 1) = 275.0
    qv3d(9, 1, 1) = 0.0045
    qgh(9, 1) = 0.0052
    tsk(9, 1) = 271.0
    tslb(9, :, 1) = 268.0
    rainbl(9, 1) = 0.5
    sr(9, 1) = 0.0

    ! 10 rain on a skin above freezing: PRCPF = PRCP, FLX1 = CPH2O*PRCP*dT,
    !    PENMAN's T1 > 273.15 arm (ELCP, LSUBC)
    t3d(10, 1, 1) = 281.0
    qv3d(10, 1, 1) = 0.006
    qgh(10, 1) = 0.0068
    tsk(10, 1) = 273.6
    tslb(10, :, 1) = 272.0
    rainbl(10, 1) = 1.0
    sr(10, 1) = 0.0

    ! 11 deposition (ETP <= 0) with RIBB >= 0.1 and FDOWN > 150: the
    !    stable-layer ETP adjustment
    qv3d(11, 1, 1) = 0.0016
    qgh(11, 1) = 0.0011
    rib(11, 1) = 0.25

    ! 12 deposition with RIBB < 0.1: no adjustment
    qv3d(12, 1, 1) = 0.0016
    qgh(12, 1) = 0.0011
    rib(12, 1) = -0.05

    ! 13 deposition with RIBB >= 0.1 but FDOWN <= 150 (polar night)
    qv3d(13, 1, 1) = 0.0016
    qgh(13, 1) = 0.0011
    rib(13, 1) = 0.3
    swdown(13, 1) = 0.0
    glw(13, 1) = 140.0

    ! 14 aged snow: SNOTIME five days, no snowfall, SNOALB2 decays but stays
    !    above ALB
    snotime(14, 1) = 432000.0

    ! 15 aged snow with ALB above the decayed SNOALB2: MAX(SNOALB2, ALB)
    snotime(15, 1) = 2592000.0
    albbck(15, 1) = 0.72

    ! 16 rough ice: 7*Z0BRD - SNOWH > 0.0007, SNOWZ0's BURIAL/7 arm
    z0(16, 1) = 0.2
    snowh(16, 1) = 0.6

    ! 17 low-density pack: SNDENS = 0.05, SNOWPACK's lower clamp
    snow(17, 1) = 150.0
    snowh(17, 1) = 3.0

    ! 18 dense pack: SNDENS above 0.40, SNOWPACK's upper clamp
    snow(18, 1) = 600.0
    snowh(18, 1) = 1.2

    ! 19 steep ice temperature profile over a warm TBOT
    tslb(19, 1, 1) = 248.0
    tslb(19, 2, 1) = 252.0
    tslb(19, 3, 1) = 258.0
    tslb(19, 4, 1) = 264.0
    tmn(19, 1) = 268.0

    ! 20 the same profile inverted, over a very cold TBOT
    tslb(20, 1, 1) = 266.0
    tslb(20, 2, 1) = 260.0
    tslb(20, 3, 1) = 252.0
    tslb(20, 4, 1) = 245.0
    tmn(20, 1) = 240.0

    ! 21 strong exchange in a windy, warm-advection night
    chs(21, 1) = 0.05
    swdown(21, 1) = 0.0
    glw(21, 1) = 260.0
    t3d(21, 1, 1) = 266.0
    rib(21, 1) = 0.15

    ! 22 melt on a thin pack that the entry reset just created
    t3d(22, 1, 1) = 280.0
    qv3d(22, 1, 1) = 0.0045
    qgh(22, 1) = 0.0062
    tsk(22, 1) = 273.15
    tslb(22, :, 1) = 272.0
    swdown(22, 1) = 900.0
    glw(22, 1) = 330.0
    snow(22, 1) = 20.0
    snowh(22, 1) = 0.05
    rib(22, 1) = -0.02

    ! 23 SR = 0.8 with SFCTMP above 273.15: snowfall in the frpcpn pass only;
    !    without frpcpn FFROZP = 0 and the column sees plain rain on a 0 C skin
    t3d(23, 1, 1) = 274.0
    qv3d(23, 1, 1) = 0.004
    qgh(23, 1) = 0.0049
    tsk(23, 1) = 273.15
    tslb(23, :, 1) = 271.5
    rainbl(23, 1) = 0.8
    sr(23, 1) = 0.8

    ! 24 deep polar ice: very cold, dry, dark
    t3d(24, 1, 1) = 225.0
    t3d(24, 2, 1) = 226.0
    qv3d(24, 1, 1) = 0.00004
    qgh(24, 1) = 0.00006
    tsk(24, 1) = 222.0
    tslb(24, :, 1) = 228.0
    tmn(24, 1) = 230.0
    swdown(24, 1) = 0.0
    glw(24, 1) = 110.0
    rib(24, 1) = 0.4
  end subroutine build_fixture

  subroutine save_state()
    tsk0 = tsk; hfx0 = hfx; qfx0 = qfx; lh0 = lh; grdflx0 = grdflx
    qsfc0 = qsfc; canwat0 = canwat; snow0 = snow; snowc0 = snowc
    snowh0 = snowh; albedo0 = albedo; albbck0 = albbck; emiss0 = emiss
    znt0 = znt; z00 = z0; snotime0 = snotime; lai0 = lai
    smstav0 = smstav; smstot0 = smstot
    sfcrunoff0 = sfcrunoff; udrunoff0 = udrunoff
    acsnom0 = acsnom; acsnow0 = acsnow; snopcx0 = snopcx; potevp0 = potevp
    rib0 = rib; chs0 = chs; chs20 = chs2; cqs20 = cqs2
    smois0 = smois; tslb0 = tslb; sh2o0 = sh2o
  end subroutine save_state

  subroutine write_csv()
    integer :: i
    open(newunit=u, file=trim(out_path), status='replace', action='write')
    write(u, '(A)') 'case,ivgtyp,isltyp,psfc,sfcprs,zlvl,sfctmp,qv1,qgh,' //   &
      'dz8w1,glw,swdown,rainbl,sr,chs_in,chs2_in,cqs2_in,cpm,qz0,rib_in,' //   &
      'vegfra,shdmin,shdmax,tmn,xland,xice,snoalb,embck,' //                   &
      'tsk_in,hfx_in,qfx_in,lh_in,grdflx_in,qsfc_in,canwat_in,snow_in,' //     &
      'snowc_in,snowh_in,albedo_in,albbck_in,emiss_in,znt_in,z0_in,' //        &
      'snotime_in,lai_in,smstav_in,smstot_in,sfcrunoff_in,udrunoff_in,' //     &
      'acsnom_in,acsnow_in,snopcx_in,potevp_in,' //                            &
      'smois1_in,smois2_in,smois3_in,smois4_in,' //                            &
      'tslb1_in,tslb2_in,tslb3_in,tslb4_in,' //                                &
      'sh2o1_in,sh2o2_in,sh2o3_in,sh2o4_in,' //                                &
      'tsk,hfx,qfx,lh,grdflx,qsfc,canwat,snow,snowc,snowh,albedo,albbck,' //   &
      'emiss,znt,z0,snotime,lai,smstav,smstot,sfcrunoff,udrunoff,' //          &
      'acsnom,acsnow,snopcx,potevp,noahres,chklowq,rib_out,' //                &
      'smois1,smois2,smois3,smois4,tslb1,tslb2,tslb3,tslb4,' //                &
      'sh2o1,sh2o2,sh2o3,sh2o4,smcrel1,smcrel2,smcrel3,smcrel4'
    do i = 1, ncase
      write(u, '(I0,",",I0,",",I0)', advance='no') i, ivgtyp(i, 1), isltyp(i, 1)
      write(u, '(25(",",ES24.16E3))', advance='no')                            &
        p8w3d(i, 1, 1), sfcprs(i, 1), zlvl(i, 1), t3d(i, 1, 1),                &
        qv3d(i, 1, 1), qgh(i, 1), dz8w(i, 1, 1), glw(i, 1), swdown(i, 1),      &
        rainbl(i, 1), sr(i, 1), chs0(i, 1), chs20(i, 1), cqs20(i, 1),          &
        cpm(i, 1), qz0(i, 1), rib0(i, 1), vegfra(i, 1), shdmin(i, 1),          &
        shdmax(i, 1), tmn(i, 1), xland(i, 1), xice(i, 1), snoalb(i, 1),        &
        embck(i, 1)
      write(u, '(25(",",ES24.16E3))', advance='no')                            &
        tsk0(i, 1), hfx0(i, 1), qfx0(i, 1), lh0(i, 1), grdflx0(i, 1),          &
        qsfc0(i, 1), canwat0(i, 1), snow0(i, 1), snowc0(i, 1), snowh0(i, 1),   &
        albedo0(i, 1), albbck0(i, 1), emiss0(i, 1), znt0(i, 1), z00(i, 1),     &
        snotime0(i, 1), lai0(i, 1), smstav0(i, 1), smstot0(i, 1),              &
        sfcrunoff0(i, 1), udrunoff0(i, 1), acsnom0(i, 1), acsnow0(i, 1),       &
        snopcx0(i, 1), potevp0(i, 1)
      write(u, '(12(",",ES24.16E3))', advance='no')                            &
        smois0(i, 1, 1), smois0(i, 2, 1), smois0(i, 3, 1), smois0(i, 4, 1),    &
        tslb0(i, 1, 1), tslb0(i, 2, 1), tslb0(i, 3, 1), tslb0(i, 4, 1),        &
        sh2o0(i, 1, 1), sh2o0(i, 2, 1), sh2o0(i, 3, 1), sh2o0(i, 4, 1)
      write(u, '(28(",",ES24.16E3))', advance='no')                            &
        tsk(i, 1), hfx(i, 1), qfx(i, 1), lh(i, 1), grdflx(i, 1), qsfc(i, 1),   &
        canwat(i, 1), snow(i, 1), snowc(i, 1), snowh(i, 1), albedo(i, 1),      &
        albbck(i, 1), emiss(i, 1), znt(i, 1), z0(i, 1), snotime(i, 1),         &
        lai(i, 1), smstav(i, 1), smstot(i, 1), sfcrunoff(i, 1),                &
        udrunoff(i, 1), acsnom(i, 1), acsnow(i, 1), snopcx(i, 1),              &
        potevp(i, 1), noahres(i, 1), chklowq(i, 1), rib(i, 1)
      write(u, '(16(",",ES24.16E3))')                                          &
        smois(i, 1, 1), smois(i, 2, 1), smois(i, 3, 1), smois(i, 4, 1),        &
        tslb(i, 1, 1), tslb(i, 2, 1), tslb(i, 3, 1), tslb(i, 4, 1),            &
        sh2o(i, 1, 1), sh2o(i, 2, 1), sh2o(i, 3, 1), sh2o(i, 4, 1),            &
        smcrel(i, 1, 1), smcrel(i, 2, 1), smcrel(i, 3, 1), smcrel(i, 4, 1)
    end do
    close(u)
  end subroutine write_csv

end program run_noah_lsm_glacial_oracle
