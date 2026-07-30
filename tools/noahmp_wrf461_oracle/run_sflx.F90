! Pin WRF v4.6.1 NOAHMP_SFLX under the first admitted Noah-MP option identity.
!
! Routines exercised, all from unmodified pinned sources:
!   MODULE_SF_NOAHMPLSM::NOAHMP_SFLX   (phys/module_sf_noahmplsm.F line 450)
!   MODULE_SF_NOAHMPLSM::noahmp_options
!   module_sf_noahmpdrv::TRANSFER_MP_PARAMETERS
!   module_sf_noahmpdrv::SNOW_INIT      (builds the snow-layer state exactly as WRF does)
!   NOAHMP_TABLES::read_mp_* readers against the byte-pinned tables
!
! NOAHMP_SFLX and noahmp_options are the only two public entry points of
! MODULE_SF_NOAHMPLSM.  Every leaf routine (THERMOPROP, CSNOW, TDFCND,
! PHASECHANGE, FRH2O, ESAT, ...) carries an explicit `private ::` accessibility
! statement at module_sf_noahmplsm.F lines 26-84 and therefore cannot be called
! from a separate program unit without editing the pinned source.  This harness
! consequently drives the whole column.
!
! Option identity (docs/wrf_noahmp_mynn_port.md):
!   dveg=4 opt_crs=1 opt_btr=1 opt_run=3 opt_sfc=1 opt_frz=1 opt_inf=1
!   opt_rad=3 opt_alb=2 opt_snf=1 opt_tbot=2 opt_stc=1 opt_rsf=1 opt_soil=1
!   opt_pedo=1 opt_crop=0 opt_irr=0 opt_irrm=0 opt_infdv=0 opt_tdrn=0
!   soiltstep=0.0  ->  soil_update_steps=1, calculate_soil=.true.
!
! This slice does NOT admit:
!   * opt_gla / NOAHMP_GLACIER: ICE is 0 in every case here, so the glacier
!     column and the ICE=-1 lake path are untouched;
!   * IST=2 (lake) columns;
!   * crop (opt_crop=0, CROPTYPE=0), irrigation (opt_irr=0, IRRFRA=0) and tile
!     drainage (opt_tdrn=0) branches, and the gecros state vector;
!   * the WRF-level driver `noahmplsm` in module_sf_noahmpdrv.F, its urban
!     coupling, its groundwater/lateral-flow init, or any multi-step
!     accumulation (every case here is a single first step with ACC_* = 0);
!   * OPT_SOIL /= 1, OPT_RUN /= 3, and every non-default option value.

program run_noahmp_sflx_oracle
  use module_sf_noahmplsm, only: noahmp_parameters, noahmp_options, noahmp_sflx, &
      calculate_soil, soil_update_steps
  use noahmp_tables, only: read_mp_veg_parameters, read_mp_soil_parameters, &
      read_mp_rad_parameters, read_mp_global_parameters, &
      read_mp_crop_parameters, read_tiledrain_parameters, &
      read_mp_optional_parameters
  use module_sf_noahmpdrv, only: transfer_mp_parameters, snow_init
  implicit none

  integer, parameter :: nsoil = 4
  integer, parameter :: nsnow = 3
  integer, parameter :: ncase = 4
  character(len=24), parameter :: names(ncase) = [character(len=24) :: &
      'veg_warm_day_dry', 'veg_warm_night_rain', 'snowpack_frozen_soil', &
      'bare_thin_snow_melt']

  character(len=1024) :: output_path
  integer :: icase, iz, unit
  type(noahmp_parameters) :: parameters

  ! --- NOAHMP_SFLX arguments, in declaration order -------------------------
  integer :: iloc, jloc, yearlen, vegtyp, ice, ist, croptype, isnow, pgs
  integer :: ircntsi, ircntmi, ircntfi
  real :: lat, julian, cosz, dt, dx, dz8w
  real :: shdfac, shdmax, sfctmp, sfcprs, psfc, uu, vv, q2, qc, soldn, lwdn
  real :: prcpconv, prcpnonc, prcpshcv, prcpsnow, prcpgrpl, prcphail
  real :: tbot, co2air, o2air, foln, zlvl
  real :: irrfra, sifra, mifra, fifra
  real :: albold, sneqvo, tah, eah, fwet, canliq, canice, tv, tg, qsfc
  real :: qsnow, qrain, snowh, sneqv, zwt, wa, wt, wslake
  real :: lfmass, rtmass, stmass, wood, stblcp, fastcp, lai, sai, cm, ch, tauss
  real :: grain, gdd, smcwtd, deeprech, rech, qtldrn, tdfracmp, z0wrf
  real :: iramtsi, iramtmi, iramtfi, irsirate, irmirate, irfirate, firr, eirr
  real :: fsa, fsr, fira, fsh, ssoil, fcev, fgev, fctr, ecan, etran, edir, trad
  real :: tgb, tgv, t2mv, t2mb, q2v, q2b, runsrf, runsub, apar, psn, sav, sag
  real :: fsno, nee, gpp, npp, fveg, albedo, qsnbot, ponding, ponding1, ponding2
  real :: rssun, rssha, bgap, wgap, chv, chb, emissi
  real :: shg, shc, shb, evg, evb, ghv, ghb, irg, irc, irb, tr, evc
  real :: chleaf, chuc, chv2, chb2, fpice, pahv, pahg, pahb, pah
  real :: laisun, laisha, rb
  real :: qints, qintr, qdrips, qdripr, qthros, qthror
  real :: qsnsub, qsnfro, qsubc, qfroc, qfrzc, qmeltc, qevac, qdewc, qmelt
  real :: rain, snow, acc_ssoil, acc_qinsur, acc_qseva, eflxb, canhs
  real :: acc_dwater, acc_prcp, acc_ecan, acc_etran, acc_edir
  real, dimension(1:nsoil) :: zsoil, smceq, sh2o, smc, acc_etrani
  real, dimension(-nsnow+1:0) :: ficeold, snice, snliq
  real, dimension(-nsnow+1:nsoil) :: stc, zsnso, hcpct
  real, dimension(1:2) :: albsnd, albsni
  real, dimension(1:60) :: gecros1d
  character(len=256) :: llanduse

  ! --- SNOW_INIT scratch (single column) -----------------------------------
  integer :: isnowxy(1, 1)
  real :: swexy(1, 1), tgxy(1, 1), snodepxy(1, 1)
  real :: zsnsoxy(1, -nsnow+1:nsoil, 1)
  real :: tsnoxy(1, -nsnow+1:0, 1), snicexy(1, -nsnow+1:0, 1)
  real :: snliqxy(1, -nsnow+1:0, 1)
  real :: snodep, soilt(nsoil), soilw(nsoil), soilliq(nsoil)
  integer :: soiltype(nsoil), soilcat, slopetype, soilcolor

  call get_command_argument(1, output_path)
  if (len_trim(output_path) == 0) then
    write(*, '(A)') 'usage: run_sflx OUTPUT.csv'
    error stop 2
  end if

  llanduse = 'MODIFIED_IGBP_MODIS_NOAH'
  call read_mp_veg_parameters(trim(llanduse))
  call read_mp_soil_parameters()
  call read_mp_rad_parameters()
  call read_mp_global_parameters()
  call read_mp_crop_parameters()
  call read_tiledrain_parameters()
  call read_mp_optional_parameters()

  ! First admitted option identity.
  call noahmp_options(4, 1, 1, 3, 1, 1, 1, 3, 2, 1, 2, 1, 1, 1, 1, 0, 0, 0, 0, 0)
  soil_update_steps = 1
  calculate_soil = .true.

  open(newunit=unit, file=trim(output_path), status='replace', action='write')
  write(unit, '(A)') 'case,stage,field,index,value'

  do icase = 1, ncase
    ! ---------------- shared configuration ---------------------------------
    iloc     = 1
    jloc     = 1
    yearlen  = 365
    dt       = 60.0
    dx       = 3000.0
    dz8w     = 40.0
    zlvl     = 20.0
    ice      = 0
    ist      = 1
    croptype = 0
    co2air   = 40.0
    o2air    = 21200.0
    foln     = 1.0
    tbot     = 285.0
    irrfra   = 0.0
    sifra    = 0.0
    mifra    = 0.0
    fifra    = 0.0
    ircntsi  = 0
    ircntmi  = 0
    ircntfi  = 0
    iramtsi  = 0.0
    iramtmi  = 0.0
    iramtfi  = 0.0
    irsirate = 0.0
    irmirate = 0.0
    irfirate = 0.0
    firr     = 0.0
    eirr     = 0.0
    qtldrn   = 0.0
    tdfracmp = 0.0
    gecros1d = 0.0
    pgs      = 0
    grain    = 0.0
    gdd      = 0.0
    prcpconv = 0.0
    prcpshcv = 0.0
    prcpgrpl = 0.0
    prcphail = 0.0
    qc       = 0.0
    slopetype = 1
    soilcolor = 4

    ! Four-layer WRF default soil column: 0.10/0.30/0.60/1.00 m.
    zsoil = [-0.10, -0.40, -1.00, -2.00]

    select case (icase)
    case (1)   ! vegetated, warm, sunlit, unsaturated, snow-free
      vegtyp   = 10
      soilcat  = 3
      shdfac   = 0.72
      shdmax   = 0.80
      lat      = 0.6632
      julian   = 200.5
      cosz     = 0.82
      sfctmp   = 296.0
      sfcprs   = 96500.0
      uu       = 3.5
      vv       = -1.2
      q2       = 0.0105
      soldn    = 820.0
      lwdn     = 350.0
      prcpnonc = 0.0
      prcpsnow = 0.0
      tg       = 297.5
      tv       = 296.5
      tah      = 296.2
      eah      = 1500.0
      snodep   = 0.0
      sneqv    = 0.0
      soilt    = [294.0, 292.0, 290.0, 288.0]
      soilw    = [0.220, 0.240, 0.260, 0.280]
      soilliq  = soilw
      canliq   = 0.15
      canice   = 0.0
      lfmass   = 120.0
      rtmass   = 250.0
      stmass   = 90.0
      wood     = 10.0
      lai      = 2.2
      sai      = 0.5
      tbot     = 288.0
    case (2)   ! same vegetation, nocturnal, non-convective rain
      vegtyp   = 10
      soilcat  = 3
      shdfac   = 0.72
      shdmax   = 0.80
      lat      = 0.6632
      julian   = 200.9
      cosz     = 0.0
      sfctmp   = 291.0
      sfcprs   = 96200.0
      uu       = 1.8
      vv       = 0.4
      q2       = 0.0122
      soldn    = 0.0
      lwdn     = 372.0
      prcpnonc = 1.2e-3
      prcpsnow = 0.0
      tg       = 291.5
      tv       = 291.0
      tah      = 291.2
      eah      = 1900.0
      snodep   = 0.0
      sneqv    = 0.0
      soilt    = [292.0, 291.0, 290.0, 288.5]
      soilw    = [0.300, 0.305, 0.300, 0.290]
      soilliq  = soilw
      canliq   = 0.05
      canice   = 0.0
      lfmass   = 120.0
      rtmass   = 250.0
      stmass   = 90.0
      wood     = 10.0
      lai      = 2.2
      sai      = 0.5
      tbot     = 288.0
    case (3)   ! layered snowpack over partly frozen soil, snowing
      vegtyp   = 1
      soilcat  = 6
      shdfac   = 0.55
      shdmax   = 0.70
      lat      = 0.7854
      julian   = 15.4
      cosz     = 0.28
      sfctmp   = 263.0
      sfcprs   = 92000.0
      uu       = 5.5
      vv       = 2.0
      q2       = 0.0016
      soldn    = 190.0
      lwdn     = 225.0
      prcpnonc = 6.0e-4
      prcpsnow = 6.0e-4
      tg       = 264.0
      tv       = 264.5
      tah      = 264.2
      eah      = 190.0
      snodep   = 0.20
      sneqv    = 50.0
      soilt    = [270.5, 272.0, 274.0, 277.0]
      soilw    = [0.300, 0.310, 0.320, 0.330]
      soilliq  = [0.090, 0.140, 0.320, 0.330]
      canliq   = 0.0
      canice   = 0.6
      lfmass   = 300.0
      rtmass   = 500.0
      stmass   = 200.0
      wood     = 3000.0
      lai      = 3.4
      sai      = 0.9
      tbot     = 278.0
    case (4)   ! bare ground, sub-layer snow at the melting point
      vegtyp   = 16
      soilcat  = 1
      shdfac   = 0.02
      shdmax   = 0.05
      lat      = 0.6981
      julian   = 60.5
      cosz     = 0.62
      sfctmp   = 279.0
      sfcprs   = 98000.0
      uu       = 2.4
      vv       = 1.1
      q2       = 0.0042
      soldn    = 610.0
      lwdn     = 300.0
      prcpnonc = 0.0
      prcpsnow = 0.0
      tg       = 274.5
      tv       = 274.5
      tah      = 275.0
      eah      = 610.0
      snodep   = 0.02
      sneqv    = 5.0
      soilt    = [274.6, 274.0, 275.0, 278.0]
      soilw    = [0.150, 0.170, 0.190, 0.210]
      soilliq  = [0.150, 0.170, 0.190, 0.210]
      canliq   = 0.0
      canice   = 0.0
      lfmass   = 20.0
      rtmass   = 40.0
      stmass   = 10.0
      wood     = 0.0
      lai      = 0.2
      sai      = 0.1
      tbot     = 280.0
    end select

    soiltype = soilcat
    call transfer_mp_parameters(nsoil, vegtyp, soiltype, slopetype, &
        soilcolor, croptype, parameters)

    ! Build the snow-layer state with WRF's own initializer.
    swexy(1, 1)    = sneqv
    snodepxy(1, 1) = snodep
    tgxy(1, 1)     = tg
    call snow_init(1, 1, 1, 1, 1, 1, 1, 1, nsnow, nsoil, zsoil, swexy, tgxy, &
        snodepxy, zsnsoxy, tsnoxy, snicexy, snliqxy, isnowxy)

    isnow = isnowxy(1, 1)
    do iz = -nsnow + 1, nsoil
      zsnso(iz) = zsnsoxy(1, iz, 1)
    end do
    do iz = -nsnow + 1, 0
      snice(iz) = snicexy(1, iz, 1)
      snliq(iz) = snliqxy(1, iz, 1)
      stc(iz)   = tsnoxy(1, iz, 1)
      if (snice(iz) + snliq(iz) > 0.0) then
        ficeold(iz) = snice(iz) / (snice(iz) + snliq(iz))
      else
        ficeold(iz) = 0.0
      end if
    end do
    do iz = 1, nsoil
      stc(iz)  = soilt(iz)
      smc(iz)  = soilw(iz)
      sh2o(iz) = soilliq(iz)
    end do

    snowh    = snodep
    sneqvo   = sneqv
    albold   = 0.65
    tauss    = 0.0
    fwet     = 0.0
    qsnow    = 0.0
    qrain    = 0.0
    qsfc     = q2
    psfc     = sfcprs
    cm       = 0.01
    ch       = 0.01
    zwt      = 2.5
    wa       = 4900.0
    wt       = 4900.0
    wslake   = 0.0
    smcwtd   = parameters%SMCMAX(nsoil)
    deeprech = 0.0
    rech     = 0.0
    stblcp   = 1000.0
    fastcp   = 1000.0
    smceq    = smc
    zlvl     = 20.0

    acc_ssoil  = 0.0
    acc_qinsur = 0.0
    acc_qseva  = 0.0
    acc_etrani = 0.0
    acc_dwater = 0.0
    acc_prcp   = 0.0
    acc_ecan   = 0.0
    acc_etran  = 0.0
    acc_edir   = 0.0

    ! ---------------- record the complete entry state ----------------------
    call emit_int(icase, 'input', 'vegtyp', 0, vegtyp)
    call emit_int(icase, 'input', 'soiltype', 0, soilcat)
    call emit_int(icase, 'input', 'slopetype', 0, slopetype)
    call emit_int(icase, 'input', 'soilcolor', 0, soilcolor)
    call emit_int(icase, 'input', 'croptype', 0, croptype)
    call emit_int(icase, 'input', 'ice', 0, ice)
    call emit_int(icase, 'input', 'ist', 0, ist)
    call emit_int(icase, 'input', 'isnow', 0, isnow)
    call emit_int(icase, 'input', 'yearlen', 0, yearlen)
    call emit_int(icase, 'input', 'pgs', 0, pgs)
    call emit(icase, 'input', 'dt', 0, dt)
    call emit(icase, 'input', 'dx', 0, dx)
    call emit(icase, 'input', 'dz8w', 0, dz8w)
    call emit(icase, 'input', 'zlvl', 0, zlvl)
    call emit(icase, 'input', 'lat', 0, lat)
    call emit(icase, 'input', 'julian', 0, julian)
    call emit(icase, 'input', 'cosz', 0, cosz)
    call emit(icase, 'input', 'shdfac', 0, shdfac)
    call emit(icase, 'input', 'shdmax', 0, shdmax)
    call emit(icase, 'input', 'sfctmp', 0, sfctmp)
    call emit(icase, 'input', 'sfcprs', 0, sfcprs)
    call emit(icase, 'input', 'psfc', 0, psfc)
    call emit(icase, 'input', 'uu', 0, uu)
    call emit(icase, 'input', 'vv', 0, vv)
    call emit(icase, 'input', 'q2', 0, q2)
    call emit(icase, 'input', 'qc', 0, qc)
    call emit(icase, 'input', 'soldn', 0, soldn)
    call emit(icase, 'input', 'lwdn', 0, lwdn)
    call emit(icase, 'input', 'prcpconv', 0, prcpconv)
    call emit(icase, 'input', 'prcpnonc', 0, prcpnonc)
    call emit(icase, 'input', 'prcpshcv', 0, prcpshcv)
    call emit(icase, 'input', 'prcpsnow', 0, prcpsnow)
    call emit(icase, 'input', 'prcpgrpl', 0, prcpgrpl)
    call emit(icase, 'input', 'prcphail', 0, prcphail)
    call emit(icase, 'input', 'tbot', 0, tbot)
    call emit(icase, 'input', 'co2air', 0, co2air)
    call emit(icase, 'input', 'o2air', 0, o2air)
    call emit(icase, 'input', 'foln', 0, foln)
    call emit(icase, 'input', 'albold', 0, albold)
    call emit(icase, 'input', 'sneqvo', 0, sneqvo)
    call emit(icase, 'input', 'tah', 0, tah)
    call emit(icase, 'input', 'eah', 0, eah)
    call emit(icase, 'input', 'fwet', 0, fwet)
    call emit(icase, 'input', 'canliq', 0, canliq)
    call emit(icase, 'input', 'canice', 0, canice)
    call emit(icase, 'input', 'tv', 0, tv)
    call emit(icase, 'input', 'tg', 0, tg)
    call emit(icase, 'input', 'qsfc', 0, qsfc)
    call emit(icase, 'input', 'qsnow', 0, qsnow)
    call emit(icase, 'input', 'qrain', 0, qrain)
    call emit(icase, 'input', 'snowh', 0, snowh)
    call emit(icase, 'input', 'sneqv', 0, sneqv)
    call emit(icase, 'input', 'zwt', 0, zwt)
    call emit(icase, 'input', 'wa', 0, wa)
    call emit(icase, 'input', 'wt', 0, wt)
    call emit(icase, 'input', 'wslake', 0, wslake)
    call emit(icase, 'input', 'lfmass', 0, lfmass)
    call emit(icase, 'input', 'rtmass', 0, rtmass)
    call emit(icase, 'input', 'stmass', 0, stmass)
    call emit(icase, 'input', 'wood', 0, wood)
    call emit(icase, 'input', 'stblcp', 0, stblcp)
    call emit(icase, 'input', 'fastcp', 0, fastcp)
    call emit(icase, 'input', 'lai', 0, lai)
    call emit(icase, 'input', 'sai', 0, sai)
    call emit(icase, 'input', 'cm', 0, cm)
    call emit(icase, 'input', 'ch', 0, ch)
    call emit(icase, 'input', 'tauss', 0, tauss)
    call emit(icase, 'input', 'smcwtd', 0, smcwtd)
    call emit(icase, 'input', 'deeprech', 0, deeprech)
    call emit(icase, 'input', 'rech', 0, rech)
    do iz = 1, nsoil
      call emit(icase, 'input', 'zsoil', iz, zsoil(iz))
      call emit(icase, 'input', 'smc', iz, smc(iz))
      call emit(icase, 'input', 'sh2o', iz, sh2o(iz))
      call emit(icase, 'input', 'smceq', iz, smceq(iz))
    end do
    do iz = -nsnow + 1, nsoil
      call emit(icase, 'input', 'stc', iz, stc(iz))
      call emit(icase, 'input', 'zsnso', iz, zsnso(iz))
    end do
    do iz = -nsnow + 1, 0
      call emit(icase, 'input', 'snice', iz, snice(iz))
      call emit(icase, 'input', 'snliq', iz, snliq(iz))
      call emit(icase, 'input', 'ficeold', iz, ficeold(iz))
    end do
    ! Parameters actually consulted by this option identity's soil physics.
    do iz = 1, nsoil
      call emit(icase, 'input', 'par_bexp', iz, parameters%BEXP(iz))
      call emit(icase, 'input', 'par_smcmax', iz, parameters%SMCMAX(iz))
      call emit(icase, 'input', 'par_psisat', iz, parameters%PSISAT(iz))
      call emit(icase, 'input', 'par_dksat', iz, parameters%DKSAT(iz))
      call emit(icase, 'input', 'par_quartz', iz, parameters%QUARTZ(iz))
    end do
    call emit(icase, 'input', 'par_csoil', 0, parameters%CSOIL)
    call emit(icase, 'input', 'par_kdt', 0, parameters%KDT)
    call emit(icase, 'input', 'par_frzx', 0, parameters%FRZX)
    call emit(icase, 'input', 'par_slope', 0, parameters%SLOPE)

    ! ---------------- the pinned WRF call ----------------------------------
    call noahmp_sflx(parameters, &
        iloc, jloc, lat, yearlen, julian, cosz, &
        dt, dx, dz8w, nsoil, zsoil, nsnow, &
        shdfac, shdmax, vegtyp, ice, ist, croptype, &
        smceq, &
        sfctmp, sfcprs, psfc, uu, vv, q2, &
        qc, soldn, lwdn, &
        prcpconv, prcpnonc, prcpshcv, prcpsnow, prcpgrpl, prcphail, &
        tbot, co2air, o2air, foln, ficeold, zlvl, &
        irrfra, sifra, mifra, fifra, llanduse, &
        albold, sneqvo, &
        stc, sh2o, smc, tah, eah, fwet, &
        canliq, canice, tv, tg, qsfc, qsnow, &
        qrain, &
        isnow, zsnso, snowh, sneqv, snice, snliq, &
        zwt, wa, wt, wslake, lfmass, rtmass, &
        stmass, wood, stblcp, fastcp, lai, sai, &
        cm, ch, tauss, &
        grain, gdd, pgs, &
        smcwtd, deeprech, rech, &
        gecros1d, &
        qtldrn, tdfracmp, &
        z0wrf, &
        ircntsi, ircntmi, ircntfi, iramtsi, iramtmi, iramtfi, &
        irsirate, irmirate, irfirate, firr, eirr, &
        fsa, fsr, fira, fsh, ssoil, fcev, &
        fgev, fctr, ecan, etran, edir, trad, &
        tgb, tgv, t2mv, t2mb, q2v, q2b, &
        runsrf, runsub, apar, psn, sav, sag, &
        fsno, nee, gpp, npp, fveg, albedo, &
        qsnbot, ponding, ponding1, ponding2, rssun, rssha, &
        albsnd, albsni, &
        bgap, wgap, chv, chb, emissi, &
        shg, shc, shb, evg, evb, ghv, &
        ghb, irg, irc, irb, tr, evc, &
        chleaf, chuc, chv2, chb2, fpice, pahv, &
        pahg, pahb, pah, laisun, laisha, rb, &
        qints, qintr, qdrips, qdripr, qthros, qthror, &
        qsnsub, qsnfro, qsubc, qfroc, qfrzc, qmeltc, &
        qevac, qdewc, qmelt, &
        rain, snow, acc_ssoil, acc_qinsur, acc_qseva, &
        acc_etrani, hcpct, eflxb, canhs, &
        acc_dwater, acc_prcp, acc_ecan, acc_etran, acc_edir)

    ! ---------------- record the complete exit state -----------------------
    call emit_int(icase, 'output', 'isnow', 0, isnow)
    call emit_int(icase, 'output', 'pgs', 0, pgs)
    call emit(icase, 'output', 'z0wrf', 0, z0wrf)
    call emit(icase, 'output', 'fsa', 0, fsa)
    call emit(icase, 'output', 'fsr', 0, fsr)
    call emit(icase, 'output', 'fira', 0, fira)
    call emit(icase, 'output', 'fsh', 0, fsh)
    call emit(icase, 'output', 'ssoil', 0, ssoil)
    call emit(icase, 'output', 'fcev', 0, fcev)
    call emit(icase, 'output', 'fgev', 0, fgev)
    call emit(icase, 'output', 'fctr', 0, fctr)
    call emit(icase, 'output', 'ecan', 0, ecan)
    call emit(icase, 'output', 'etran', 0, etran)
    call emit(icase, 'output', 'edir', 0, edir)
    call emit(icase, 'output', 'trad', 0, trad)
    call emit(icase, 'output', 'tgb', 0, tgb)
    call emit(icase, 'output', 'tgv', 0, tgv)
    call emit(icase, 'output', 't2mv', 0, t2mv)
    call emit(icase, 'output', 't2mb', 0, t2mb)
    call emit(icase, 'output', 'q2v', 0, q2v)
    call emit(icase, 'output', 'q2b', 0, q2b)
    call emit(icase, 'output', 'runsrf', 0, runsrf)
    call emit(icase, 'output', 'runsub', 0, runsub)
    call emit(icase, 'output', 'apar', 0, apar)
    call emit(icase, 'output', 'psn', 0, psn)
    call emit(icase, 'output', 'sav', 0, sav)
    call emit(icase, 'output', 'sag', 0, sag)
    call emit(icase, 'output', 'fsno', 0, fsno)
    call emit(icase, 'output', 'nee', 0, nee)
    call emit(icase, 'output', 'gpp', 0, gpp)
    call emit(icase, 'output', 'npp', 0, npp)
    call emit(icase, 'output', 'fveg', 0, fveg)
    call emit(icase, 'output', 'albedo', 0, albedo)
    call emit(icase, 'output', 'qsnbot', 0, qsnbot)
    call emit(icase, 'output', 'ponding', 0, ponding)
    call emit(icase, 'output', 'ponding1', 0, ponding1)
    call emit(icase, 'output', 'ponding2', 0, ponding2)
    call emit(icase, 'output', 'rssun', 0, rssun)
    call emit(icase, 'output', 'rssha', 0, rssha)
    call emit(icase, 'output', 'bgap', 0, bgap)
    call emit(icase, 'output', 'wgap', 0, wgap)
    call emit(icase, 'output', 'chv', 0, chv)
    call emit(icase, 'output', 'chb', 0, chb)
    call emit(icase, 'output', 'emissi', 0, emissi)
    call emit(icase, 'output', 'shg', 0, shg)
    call emit(icase, 'output', 'shc', 0, shc)
    call emit(icase, 'output', 'shb', 0, shb)
    call emit(icase, 'output', 'evg', 0, evg)
    call emit(icase, 'output', 'evb', 0, evb)
    call emit(icase, 'output', 'ghv', 0, ghv)
    call emit(icase, 'output', 'ghb', 0, ghb)
    call emit(icase, 'output', 'irg', 0, irg)
    call emit(icase, 'output', 'irc', 0, irc)
    call emit(icase, 'output', 'irb', 0, irb)
    call emit(icase, 'output', 'tr', 0, tr)
    call emit(icase, 'output', 'evc', 0, evc)
    call emit(icase, 'output', 'chleaf', 0, chleaf)
    call emit(icase, 'output', 'chuc', 0, chuc)
    call emit(icase, 'output', 'chv2', 0, chv2)
    call emit(icase, 'output', 'chb2', 0, chb2)
    call emit(icase, 'output', 'fpice', 0, fpice)
    call emit(icase, 'output', 'pahv', 0, pahv)
    call emit(icase, 'output', 'pahg', 0, pahg)
    call emit(icase, 'output', 'pahb', 0, pahb)
    call emit(icase, 'output', 'pah', 0, pah)
    call emit(icase, 'output', 'laisun', 0, laisun)
    call emit(icase, 'output', 'laisha', 0, laisha)
    call emit(icase, 'output', 'rb', 0, rb)
    call emit(icase, 'output', 'qints', 0, qints)
    call emit(icase, 'output', 'qintr', 0, qintr)
    call emit(icase, 'output', 'qdrips', 0, qdrips)
    call emit(icase, 'output', 'qdripr', 0, qdripr)
    call emit(icase, 'output', 'qthros', 0, qthros)
    call emit(icase, 'output', 'qthror', 0, qthror)
    call emit(icase, 'output', 'qsnsub', 0, qsnsub)
    call emit(icase, 'output', 'qsnfro', 0, qsnfro)
    call emit(icase, 'output', 'qsubc', 0, qsubc)
    call emit(icase, 'output', 'qfroc', 0, qfroc)
    call emit(icase, 'output', 'qfrzc', 0, qfrzc)
    call emit(icase, 'output', 'qmeltc', 0, qmeltc)
    call emit(icase, 'output', 'qevac', 0, qevac)
    call emit(icase, 'output', 'qdewc', 0, qdewc)
    call emit(icase, 'output', 'qmelt', 0, qmelt)
    call emit(icase, 'output', 'rain', 0, rain)
    call emit(icase, 'output', 'snow', 0, snow)
    call emit(icase, 'output', 'eflxb', 0, eflxb)
    call emit(icase, 'output', 'canhs', 0, canhs)
    call emit(icase, 'output', 'albold', 0, albold)
    call emit(icase, 'output', 'sneqvo', 0, sneqvo)
    call emit(icase, 'output', 'tah', 0, tah)
    call emit(icase, 'output', 'eah', 0, eah)
    call emit(icase, 'output', 'fwet', 0, fwet)
    call emit(icase, 'output', 'canliq', 0, canliq)
    call emit(icase, 'output', 'canice', 0, canice)
    call emit(icase, 'output', 'tv', 0, tv)
    call emit(icase, 'output', 'tg', 0, tg)
    call emit(icase, 'output', 'qsfc', 0, qsfc)
    call emit(icase, 'output', 'qsnow', 0, qsnow)
    call emit(icase, 'output', 'qrain', 0, qrain)
    call emit(icase, 'output', 'snowh', 0, snowh)
    call emit(icase, 'output', 'sneqv', 0, sneqv)
    call emit(icase, 'output', 'zwt', 0, zwt)
    call emit(icase, 'output', 'wa', 0, wa)
    call emit(icase, 'output', 'wt', 0, wt)
    call emit(icase, 'output', 'wslake', 0, wslake)
    call emit(icase, 'output', 'lfmass', 0, lfmass)
    call emit(icase, 'output', 'rtmass', 0, rtmass)
    call emit(icase, 'output', 'stmass', 0, stmass)
    call emit(icase, 'output', 'wood', 0, wood)
    call emit(icase, 'output', 'stblcp', 0, stblcp)
    call emit(icase, 'output', 'fastcp', 0, fastcp)
    call emit(icase, 'output', 'lai', 0, lai)
    call emit(icase, 'output', 'sai', 0, sai)
    call emit(icase, 'output', 'cm', 0, cm)
    call emit(icase, 'output', 'ch', 0, ch)
    call emit(icase, 'output', 'tauss', 0, tauss)
    call emit(icase, 'output', 'zlvl', 0, zlvl)
    call emit(icase, 'output', 'smcwtd', 0, smcwtd)
    call emit(icase, 'output', 'deeprech', 0, deeprech)
    call emit(icase, 'output', 'rech', 0, rech)
    call emit(icase, 'output', 'acc_ssoil', 0, acc_ssoil)
    call emit(icase, 'output', 'acc_qinsur', 0, acc_qinsur)
    call emit(icase, 'output', 'acc_qseva', 0, acc_qseva)
    call emit(icase, 'output', 'acc_dwater', 0, acc_dwater)
    call emit(icase, 'output', 'acc_prcp', 0, acc_prcp)
    call emit(icase, 'output', 'acc_ecan', 0, acc_ecan)
    call emit(icase, 'output', 'acc_etran', 0, acc_etran)
    call emit(icase, 'output', 'acc_edir', 0, acc_edir)
    do iz = 1, 2
      call emit(icase, 'output', 'albsnd', iz, albsnd(iz))
      call emit(icase, 'output', 'albsni', iz, albsni(iz))
    end do
    do iz = 1, nsoil
      call emit(icase, 'output', 'smc', iz, smc(iz))
      call emit(icase, 'output', 'sh2o', iz, sh2o(iz))
      call emit(icase, 'output', 'acc_etrani', iz, acc_etrani(iz))
    end do
    do iz = -nsnow + 1, nsoil
      call emit(icase, 'output', 'stc', iz, stc(iz))
      call emit(icase, 'output', 'zsnso', iz, zsnso(iz))
      call emit(icase, 'output', 'hcpct', iz, hcpct(iz))
    end do
    do iz = -nsnow + 1, 0
      call emit(icase, 'output', 'snice', iz, snice(iz))
      call emit(icase, 'output', 'snliq', iz, snliq(iz))
    end do
  end do

  close(unit)

contains

  subroutine emit(jcase, stage, field, index, value)
    integer, intent(in) :: jcase, index
    character(len=*), intent(in) :: stage, field
    real, intent(in) :: value
    write(unit, '(*(g0,:,","))') trim(names(jcase)), stage, field, index, value
  end subroutine emit

  subroutine emit_int(jcase, stage, field, index, value)
    integer, intent(in) :: jcase, index, value
    character(len=*), intent(in) :: stage, field
    write(unit, '(*(g0,:,","))') trim(names(jcase)), stage, field, index, value
  end subroutine emit_int

end program run_noahmp_sflx_oracle
