program run_ruc_sfctmp_prep_oracle
  ! Drives the unmodified WRF v4.6.1 module_sf_ruclsm::sfctmp so that its snow
  ! PREPARATION block (phys/module_sf_ruclsm.F:1400-1766) can be pinned.
  !
  ! The preparation block is not a subroutine: it is straight-line code at the
  ! top of sfctmp, and every one of its results is either a local or an
  ! inout dummy that the dispatch below (:1767-2193, which calls soil,
  ! snowsoil, sice and snowseaice) overwrites before sfctmp returns.  There is
  ! therefore no argument list through which the preparation state can be
  ! observed, and the physics module may not be edited.
  !
  ! So this program only supplies forcing and calls sfctmp; the pinned values
  ! are read out of the running, unmodified module by gdb, which breaks at
  ! :1418 (the first executable statement of the block, before any state is
  ! touched) and at :1767 or :2120 (the first statement after the block, on
  ! the snow and no-snow paths respectively).  See sfctmp_prep.gdb.
  !
  ! sfctmp is allowed to run to completion for every case - nothing is
  ! interrupted - so the forcing below also has to be a state the full RUC
  ! column can integrate.
  !
  ! The case set is chosen to KILL MUTANTS, not merely to cover regimes:
  ! every argument the preparation block reads takes at least two distinct
  ! values across the cases, in situations where the value reaches an output.
  ! That is what makes a bitwise match evidence that the port reads the
  ! argument at all.  Hence the deliberately varied delt, c1sn, c2sn, sat,
  ! isice, incoming rhosnfall/emiss/alb/iland, the two land-use datasets
  ! (MODI-RUC with URBAN=13/isice=15, USGS-RUC with URBAN=1/isice=24), the
  ! tsnav>0 guard case and the snwe>0/snhei=0 degenerate case.
  !
  ! The CSV this program writes is NOT the oracle.  It is an independent
  ! record of the forcing that was passed in, used by
  ! validate_sfctmp_prep_oracle.py to confirm that gdb read the intended
  ! variables out of the intended frame.
  use module_sf_ruclsm, only: sfctmp, ruclsm_soilvegparm, drysmc, &
      maxsmc, refsmc, wltsmc, satpsi, satdk, bb, hc, qtz, pctbl, &
      laitbl, lemitbl, z0tbl, cfactr_data
  implicit none

  integer, parameter :: ncase = 17, nzs = 9, nddzs = 14
  character(len=24), parameter :: names(ncase) = [character(len=24) :: &
      'warm_rain_canopy_drip', 'bare_soil_rain', 'deep_pack_densify', &
      'shallow_snow_mosaic', 'aged_pack_no_densify', 'new_snow_mosaic_drip', &
      'fresh_snow_keep_albedo', 'graupel_dense_new_snow', 'urban_snow_cap', &
      'sea_ice_snow_deep', 'sea_ice_snow_partial', 'sea_ice_snow_mosaic', &
      'sea_ice_bare', 'snow_water_no_depth', 'usgs_crop_rain_drip', &
      'usgs_urban_snow_cap', 'usgs_warm_pack_new_snow']
  ! 0 = MODI-RUC (URBAN=13, snow/ice class 15), 1 = USGS-RUC (URBAN=1,
  ! snow/ice class 24).  The tables are re-read when the dataset changes.
  integer, parameter :: dataset(ncase) = [ &
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
  integer, parameter :: isice_case(ncase) = [ &
      15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 24, 24, 24]
  integer, parameter :: veg_category(ncase) = [ &
      10, 16, 10, 10, 1, 10, 10, 12, 13, 15, 15, 15, 15, 16, 2, 1, 8]
  integer, parameter :: soil_category(ncase) = [ &
      6, 16, 6, 6, 4, 6, 6, 8, 6, 16, 16, 16, 16, 16, 6, 6, 6]
  integer, parameter :: nroot_case(ncase) = [ &
      6, 1, 6, 6, 8, 6, 6, 4, 1, 1, 1, 1, 1, 1, 4, 1, 4]
  real, parameter :: zsmain(nzs) = [0.0, 0.01, 0.04, 0.10, 0.30, &
      0.60, 1.00, 1.60, 3.00]
  real, parameter :: xlv = 2.5e6, cp = 1004.5, r_d = 287.0
  real, parameter :: g0_p = 9.81, cw = 4.183e6, stbolt = 5.67051e-8
  character(len=1024) :: output_path
  integer :: n, k, k1, k2, unit, ktau, iland, isoil, nroot, ivgtyp, isltyp
  integer :: isice, loaded
  real :: delt, xgeom, cq, evs, eis, r61, conflx, meltfactor
  real :: xland, prcpms, newsnms, snwe, snhei, snowfrac, snhei_force
  real :: rhosn, rhonewsn, rhosnfall, snowrat, grauprat, icerat, curat
  real :: patm, tabs, qvatm, qcatm, rho, glw, gsw, emiss, qkms, tkms, pc
  real :: mavail, cst, vegfra, alb, znt, alb_snow, alb_snow_free, lai
  real :: seaice, qwrtz, rhocs, dqm, qmin, ref, wilt, psis, bclh, ksat
  real :: sat, cn, rovcp, kqwrtz, kice, kwt, c1sn, c2sn
  real :: snweprint, snheiprint, rsm, soilt, soilt1, tsnav, dew
  real :: qvg, qsg, qcg, smelt, snoh, snflx, snom, snowfallac, acsnow
  real :: edir1, ec1, ett1, eeta, qfx, hfx, s, sublim, evapl, prcpl
  real :: fltot, runoff1, runoff2, infiltr, smf
  real :: zshalf(nzs), dtdzs(nddzs), dtdzs2(nzs), tbq(5001)
  real :: soilm1d(nzs), ts1d(nzs), smfrkeep(nzs), keepfr(nzs)
  real :: soilice(nzs), soiliqw(nzs), rstochcol(nzs), fieldcol_sf(nzs)
  real :: ts1d_in(nzs)
  logical :: myj

  call get_command_argument(1, output_path)
  if (len_trim(output_path) == 0) then
    write(*, '(A)') 'usage: run_sfctmp_prep INPUTS.csv'
    error stop 2
  end if

  zshalf(1) = 0.0
  do k = 2, nzs
    zshalf(k) = 0.5 * (zsmain(k - 1) + zsmain(k))
  end do

  cq = 173.15 - 0.05
  r61 = 6.1153 * 0.62198
  do k = 1, 5001
    cq = cq + 0.05
    evs = exp(17.67 * (cq - 273.15) / (cq - 29.65))
    eis = exp(22.514 - 6.15e3 / cq)
    if (cq >= 273.15) then
      tbq(k) = r61 * evs
    else
      tbq(k) = r61 * eis
    end if
  end do

  rovcp = r_d / cp
  kqwrtz = 7.7
  kice = 2.2
  kwt = 0.57
  rstochcol = 0.0
  fieldcol_sf = 0.0
  ktau = 1
  myj = .false.
  loaded = -1

  open(newunit=unit, file=trim(output_path), status='replace', action='write')
  write(unit, '(A)') 'case,case_index,k,delt,c1sn,c2sn,isice,ivgtyp,' // &
      'seaice,gsw,tabs,tsnav,prcpms,newsnms,vegfra,lai,sat,soilt,' // &
      'snowfallac,alb_snow,alb_snow_free,snowrat,grauprat,icerat,curat,' // &
      'iland_before,snwe_before,snhei_before,snowfrac_before,' // &
      'rhosn_before,rhosnfall_before,cst_before,alb_before,emiss_before,' // &
      'znt_before,ts1d'

  do n = 1, ncase
    if (dataset(n) /= loaded) then
      if (dataset(n) == 0) then
        call ruclsm_soilvegparm('MODI-RUC', 'STAS-RUC')
      else
        call ruclsm_soilvegparm('USGS-RUC', 'STAS-RUC')
      end if
      loaded = dataset(n)
    end if

    isice = isice_case(n)
    ivgtyp = veg_category(n)
    isoil = soil_category(n)
    nroot = nroot_case(n)
    iland = ivgtyp
    isltyp = isoil
    qwrtz = qtz(isoil)
    rhocs = hc(isoil) * 1.0e6
    bclh = bb(isoil)
    dqm = maxsmc(isoil) - drysmc(isoil)
    ksat = satdk(isoil)
    psis = -satpsi(isoil)
    qmin = drysmc(isoil)
    ref = refsmc(isoil)
    wilt = wltsmc(isoil)
    pc = pctbl(ivgtyp)
    lai = laitbl(ivgtyp)
    emiss = lemitbl(ivgtyp)
    ! lsmruc's snow-density constants (:744-745).  Two cases perturb them
    ! so a port that hard-codes them cannot reproduce this fixture.
    c1sn = 0.026
    c2sn = 21.0
    ! lsmruc's canopy-water capacity; varied for the same reason.
    sat = 5.0e-4
    delt = 60.0
    alb_snow_free = 0.18
    alb_snow = 0.70
    alb = alb_snow_free
    znt = z0tbl(ivgtyp)
    conflx = 40.0
    meltfactor = 1.0
    xland = 1.0
    seaice = 0.0
    patm = 0.95
    qcatm = 0.0
    rho = 1.25
    qkms = 0.010
    tkms = 0.009
    mavail = 0.60
    cst = 0.0
    prcpms = 0.0
    newsnms = 0.0
    snowrat = 0.0
    grauprat = 0.0
    icerat = 0.0
    curat = 0.0
    snwe = 0.0
    snhei_force = -1.0
    snowfrac = 0.0
    rhosn = 200.0
    rhonewsn = 100.0
    rhosnfall = 100.0
    snowfallac = 0.0
    tsnav = 0.0
    acsnow = 0.0
    dew = 0.0
    smelt = 0.0
    snoh = 0.0
    snflx = 0.0
    snom = 0.0
    edir1 = 0.0
    ec1 = 0.0
    ett1 = 0.0
    eeta = 0.0
    qfx = 0.0
    hfx = 0.0
    s = 0.0
    sublim = 0.0
    evapl = 0.0
    prcpl = 0.0
    fltot = 0.0
    runoff1 = 0.0
    runoff2 = 0.0
    infiltr = 0.0
    smf = 0.0
    rsm = 0.0
    snweprint = 0.0
    snheiprint = 0.0
    smfrkeep = 0.0
    keepfr = 0.0
    soilt1 = 0.0

    select case (n)
    case (1)
      ! Snow-free warm grassland under rain with a nearly full canopy
      ! reservoir: the vegfrac>0.01 interception branch (:1555-1571) fills
      ! cst past sat and spills drip, rainf latches, and the snhei>0 block
      ! is skipped entirely.
      tabs = 291.0
      qvatm = 0.011
      glw = 340.0
      gsw = 520.0
      soilt = 292.0
      vegfra = 65.0
      prcpms = 4.0e-6
      cst = 4.8e-4
      ! Deliberately not alb_snow_free, so that passing alb through is
      ! distinguishable from resetting it.
      alb = 0.22
      do k = 1, nzs
        ts1d(k) = 291.0 - 0.8 * real(k - 1)
        soilm1d(k) = min(dqm, 0.24 + 0.004 * real(k - 1))
      end do
    case (2)
      ! Snow-free barren point with vegfra=0 on a half-length step: the
      ! vegfrac<=0.01 branch (:1572-1577) zeroes the canopy store and routes
      ! all rain to infwater.
      delt = 30.0
      tabs = 288.0
      qvatm = 0.008
      glw = 330.0
      gsw = 480.0
      soilt = 289.0
      vegfra = 0.0
      prcpms = 6.0e-6
      cst = 1.0e-4
      ! Deliberately not lemitbl(ivgtyp), so that passing emiss through is
      ! distinguishable from resetting it to emiss_snowfree.
      emiss = 0.93
      do k = 1, nzs
        ts1d(k) = 288.0 - 0.5 * real(k - 1)
        soilm1d(k) = min(dqm, 0.10 + 0.002 * real(k - 1))
      end do
    case (3)
      ! Deep low-density pack on a double-length step: bsn*snwe*100 clears
      ! 1.e-4 so the Koren density update at :1497-1498 actually runs,
      ! snhei>4*znt takes the third roughness branch (:1677), full cover
      ! keeps snow_mosaic=0, and a low maximum snow albedo puts albsn under
      ! 0.4 so the first leg of :1731 sets alb=albsn.
      delt = 120.0
      tabs = 271.0
      qvatm = 0.0035
      glw = 260.0
      gsw = 120.0
      soilt = 271.2
      tsnav = -2.0
      soilt1 = 271.0
      vegfra = 0.0
      alb_snow = 0.35
      rhosn = 100.0
      snwe = 0.5
      snowfrac = 1.0
      znt = 0.15
      alb = alb_snow
      do k = 1, nzs
        ts1d(k) = 272.0 + 0.3 * real(k - 1)
        soilm1d(k) = min(dqm, 0.30 + 0.003 * real(k - 1))
      end do
    case (4)
      ! Thin 0.025 m pack: snhei is below 0.0081*1.e3/rhosn so the density
      ! update is skipped, the tanh cover falls under 0.75 so :1656 turns
      ! on the mosaic, and snhei<=2*znt takes the first roughness branch.
      tabs = 266.0
      qvatm = 0.0021
      glw = 240.0
      gsw = 60.0
      soilt = 265.0
      tsnav = -8.0
      soilt1 = 265.0
      vegfra = 20.0
      rhosn = 200.0
      snwe = 0.005
      snowfrac = 0.80
      znt = 0.05
      alb = 0.40
      do k = 1, nzs
        ts1d(k) = 264.0 + 0.8 * real(k - 1)
        soilm1d(k) = min(dqm, 0.28 + 0.003 * real(k - 1))
      end do
    case (5)
      ! Aged dense forest pack: snhei clears 0.0081*1.e3/rhosn but
      ! bsn*snwe*100 stays under 1.e-4, so :1496 jumps to 777 and rhosn is
      ! left alone.  snhei lands between 2*znt and 4*znt for the middle
      ! roughness branch (:1675).  The incoming rhosnfall is a third
      ! distinct value that has to survive to the output untouched.
      tabs = 258.0
      qvatm = 0.0009
      glw = 195.0
      gsw = 25.0
      soilt = 257.0
      tsnav = -12.0
      soilt1 = 260.0
      vegfra = 45.0
      rhosn = 400.0
      rhosnfall = 250.0
      snwe = 0.036
      snowfrac = 0.90
      znt = 0.03
      alb = 0.60
      do k = 1, nzs
        ts1d(k) = 265.0 + 0.9 * real(k - 1)
        soilm1d(k) = min(dqm, 0.26 + 0.003 * real(k - 1))
      end do
    case (6)
      ! New snow and rain together on a partly covered pack: the incoming
      ! snowfrac<0.75 sets snow_mosaic at :1504, the canopy spills drip,
      ! and the mosaic split at :1585-1591 routes dripliq to infwater and
      ! dripsn onto the pack.  All four hydrometeor fractions are nonzero,
      ! including curat.  snowfallac stays small so snowfracnewsn<0.99 and
      ! the mosaic survives.
      tabs = 272.5
      qvatm = 0.0040
      glw = 290.0
      gsw = 90.0
      soilt = 271.8
      tsnav = -1.5
      soilt1 = 271.5
      vegfra = 55.0
      prcpms = 3.0e-6
      newsnms = 2.0e-6
      snowrat = 0.75
      grauprat = 0.10
      icerat = 0.10
      curat = 0.05
      rhosn = 150.0
      snwe = 0.010
      snowfrac = 0.50
      znt = 0.075
      alb = 0.45
      cst = 4.9e-4
      snowfallac = 0.02
      do k = 1, nzs
        ts1d(k) = 271.5 + 0.4 * real(k - 1)
        soilm1d(k) = min(dqm, 0.29 + 0.003 * real(k - 1))
      end do
    case (7)
      ! A full fresh-snow event over a low maximum snow albedo, with a
      ! reduced canopy capacity: snowfracnewsn saturates at 1 and rhosnfall
      ! stays under 450, so keep_snow_albedo latches at :1661, the mosaic is
      ! switched off, and the :1703-1710 correction lifts albsn to 0.7.  The
      ! incoming snowfrac>=0.75 keeps snow_mosaic=0 at :1504, so the drip is
      ! added to snwe through the non-mosaic path at :1593.
      sat = 4.0e-4
      tabs = 268.0
      qvatm = 0.0022
      glw = 235.0
      gsw = 70.0
      soilt = 267.0
      tsnav = -6.0
      soilt1 = 267.0
      vegfra = 35.0
      prcpms = 1.0e-6
      newsnms = 5.0e-6
      snowrat = 1.0
      alb_snow = 0.35
      rhosn = 120.0
      snwe = 0.020
      snowfrac = 0.85
      znt = 0.075
      alb = 0.33
      cst = 4.95e-4
      snowfallac = 8.0
      do k = 1, nzs
        ts1d(k) = 266.0 + 0.7 * real(k - 1)
        soilm1d(k) = min(dqm, 0.27 + 0.003 * real(k - 1))
      end do
    case (8)
      ! Graupel-dominated fall just under freezing: rhonewsn saturates at
      ! the 125 cap (:1520), rhonewgr at the 500 cap (:1521), and the
      ! weighted rhosnfall hits the 500 clamp at :1527, which is >=450 so
      ! keep_snow_albedo stays 0 even though snowfracnewsn saturates.
      tabs = 274.0
      qvatm = 0.0048
      glw = 300.0
      gsw = 110.0
      soilt = 272.9
      tsnav = -0.5
      soilt1 = 272.8
      vegfra = 30.0
      newsnms = 8.0e-6
      grauprat = 1.0
      rhosn = 250.0
      snwe = 0.030
      snowfrac = 0.95
      znt = 0.20
      alb = 0.55
      snowfallac = 5.0
      do k = 1, nzs
        ts1d(k) = 272.0 + 0.3 * real(k - 1)
        soilm1d(k) = min(dqm, 0.31 + 0.003 * real(k - 1))
      end do
    case (9)
      ! MODI-RUC urban snow: the tanh cover exceeds 0.75 and :1645 clamps it
      ! back to exactly 0.75, which then fails the :1656 mosaic test.
      ! znt>0.2 also blocks the roughness blend, and the incoming rhosnfall
      ! is a fourth distinct value.
      tabs = 269.0
      qvatm = 0.0025
      glw = 245.0
      gsw = 100.0
      soilt = 268.0
      tsnav = -0.5
      soilt1 = 268.0
      vegfra = 10.0
      rhosn = 180.0
      rhosnfall = 300.0
      snwe = 0.22
      snowfrac = 0.95
      znt = 0.50
      alb = 0.50
      do k = 1, nzs
        ts1d(k) = 267.0 + 0.6 * real(k - 1)
        soilm1d(k) = min(dqm, 0.20 + 0.003 * real(k - 1))
      end do
    case (10)
      ! Snowfall onto deep cold sea ice: the Zubov ice column at :1472-1478
      ! runs and the alb_snow_free floor of :1483 is the binding leg, full
      ! cover keeps snow_mosaic=0, and the very cold air pins rhonewsn on
      ! the 1000./17. floor of :1520.
      seaice = 1.0
      xland = 2.0
      tabs = 250.0
      qvatm = 0.0006
      glw = 185.0
      gsw = 30.0
      soilt = 252.0
      tsnav = -20.0
      soilt1 = 252.0
      vegfra = 0.0
      dqm = 1.0
      ref = 1.0
      qmin = 0.0
      wilt = 0.0
      newsnms = 1.0e-6
      snowrat = 0.9
      grauprat = 0.1
      rhosn = 300.0
      snwe = 0.15
      snowfrac = 1.0
      znt = 0.011
      alb = 0.70
      alb_snow_free = 0.55
      do k = 1, nzs
        ts1d(k) = min(271.4, 250.0 + 2.0 * real(k - 1))
        soilm1d(k) = 1.0
        soiliqw(k) = 0.0
        soilice(k) = 1.0
        smfrkeep(k) = 1.0
      end do
    case (11)
      ! Partly covered sea ice near the melting point: albice is pulled
      ! down to the alb_snow_free-0.05 floor by the warm ice surface,
      ! snowfrac lands between 0.75 and 1 so the non-mosaic ice blend at
      ! :1744 gives albsn<alb_snow and :1758 takes alb=albsn.
      seaice = 1.0
      xland = 2.0
      tabs = 270.0
      qvatm = 0.0040
      glw = 300.0
      gsw = 150.0
      soilt = 270.8
      tsnav = -1.0
      soilt1 = 270.8
      vegfra = 0.0
      dqm = 1.0
      ref = 1.0
      qmin = 0.0
      wilt = 0.0
      rhosn = 350.0
      snwe = 0.042
      snowfrac = 0.80
      znt = 0.011
      alb = 0.60
      alb_snow_free = 0.55
      do k = 1, nzs
        ts1d(k) = min(271.4, 270.5 + 0.1 * real(k - 1))
        soilm1d(k) = 1.0
        soiliqw(k) = 0.0
        soilice(k) = 1.0
        smfrkeep(k) = 1.0
      end do
    case (12)
      ! Thin snow on melting sea ice: the cover drops under 0.75 so the
      ! mosaic ice branch at :1740-1742 sets albsn=alb_snow and emiss=0.98,
      ! and the warm skin makes the albsn-0.1 floor of :1761 the binding
      ! leg.
      seaice = 1.0
      xland = 2.0
      tabs = 272.0
      qvatm = 0.0042
      glw = 310.0
      gsw = 70.0
      soilt = 272.9
      tsnav = -0.3
      soilt1 = 272.9
      vegfra = 0.0
      dqm = 1.0
      ref = 1.0
      qmin = 0.0
      wilt = 0.0
      rhosn = 250.0
      snwe = 0.004
      snowfrac = 0.85
      znt = 0.011
      alb = 0.65
      alb_snow_free = 0.55
      do k = 1, nzs
        ts1d(k) = min(271.4, 271.0 + 0.05 * real(k - 1))
        soilm1d(k) = 1.0
        soiliqw(k) = 0.0
        soilice(k) = 1.0
        smfrkeep(k) = 1.0
      end do
    case (13)
      ! Bare sea ice under rain: the ice column and albice are built, but
      ! snhei==0 zeroes snowfrac at :1429 and the whole snhei>0 block is
      ! skipped, so alb, emiss, znt and iland pass straight through.
      seaice = 1.0
      xland = 2.0
      tabs = 271.5
      qvatm = 0.0043
      glw = 315.0
      gsw = 95.0
      soilt = 270.9
      soilt1 = 270.9
      vegfra = 0.0
      dqm = 1.0
      ref = 1.0
      qmin = 0.0
      wilt = 0.0
      prcpms = 2.0e-6
      rhosn = 200.0
      snwe = 0.0
      snowfrac = 0.30
      znt = 0.011
      alb = 0.55
      alb_snow_free = 0.55
      do k = 1, nzs
        ts1d(k) = min(271.4, 270.8 + 0.05 * real(k - 1))
        soilm1d(k) = 1.0
        soiliqw(k) = 0.0
        soilice(k) = 1.0
        smfrkeep(k) = 1.0
      end do
    case (14)
      ! Degenerate entry state: snow water on the ground with a reported
      ! depth of exactly zero.  :1429 zeroes snowfrac, :1493 and :1600 both
      ! test snhei rather than snwe, so the whole snow block is skipped and
      ! snwe survives untouched.  A port that recomputed snhei from
      ! snwe/rhosn instead of reading the argument cannot reproduce this.
      delt = 30.0
      tabs = 270.0
      qvatm = 0.0030
      glw = 250.0
      gsw = 150.0
      soilt = 269.5
      vegfra = 0.0
      rhosn = 220.0
      rhosnfall = 400.0
      snwe = 0.012
      snhei_force = 0.0
      snowfrac = 0.60
      znt = 0.065
      alb = 0.30
      emiss = 0.87
      iland = 1
      do k = 1, nzs
        ts1d(k) = 269.0 + 0.5 * real(k - 1)
        soilm1d(k) = min(dqm, 0.15 + 0.002 * real(k - 1))
      end do
    case (15)
      ! USGS-RUC snow-free cropland under rain on a half-length step, with a
      ! reduced canopy capacity: same interception branch as case 1 but the
      ! land-use dataset, isice, z0tbl/lemitbl rows and URBAN index all
      ! differ, so a port that hard-codes any of them fails here.
      delt = 30.0
      sat = 3.5e-4
      tabs = 293.0
      qvatm = 0.012
      glw = 345.0
      gsw = 540.0
      soilt = 294.0
      vegfra = 50.0
      prcpms = 5.0e-6
      cst = 3.0e-4
      alb = 0.25
      emiss = 0.90
      do k = 1, nzs
        ts1d(k) = 293.0 - 0.9 * real(k - 1)
        soilm1d(k) = min(dqm, 0.22 + 0.004 * real(k - 1))
      end do
    case (16)
      ! USGS-RUC urban snow with non-default snow-density constants: the
      ! URBAN index is 1 here rather than 13, the compaction runs with
      ! c1sn/c2sn away from lsmruc's values, and the cover is clamped to
      ! 0.75 by :1645.
      c1sn = 0.030
      c2sn = 18.0
      tabs = 267.0
      qvatm = 0.0020
      glw = 240.0
      gsw = 95.0
      soilt = 266.5
      tsnav = -1.0
      soilt1 = 266.5
      vegfra = 8.0
      rhosn = 160.0
      snwe = 0.20
      snowfrac = 0.92
      znt = 0.45
      alb = 0.48
      alb_snow = 0.65
      alb_snow_free = 0.20
      do k = 1, nzs
        ts1d(k) = 266.0 + 0.5 * real(k - 1)
        soilm1d(k) = min(dqm, 0.18 + 0.003 * real(k - 1))
      end do
    case (17)
      ! USGS-RUC pack whose mean temperature has drifted above freezing on
      ! a 90 s step: min(0.,tsnav) at :1495 has to clamp it, otherwise the
      ! compaction runs away.  All four hydrometeor fractions are nonzero,
      ! snowfracnewsn saturates with rhosnfall<450 so keep_snow_albedo
      ! latches and the :1703-1710 correction fires again on a different
      ! dataset.
      delt = 90.0
      sat = 6.0e-4
      tabs = 265.0
      qvatm = 0.0019
      glw = 232.0
      gsw = 65.0
      soilt = 266.0
      tsnav = 1.0
      soilt1 = 266.0
      vegfra = 25.0
      newsnms = 3.0e-6
      snowrat = 0.60
      grauprat = 0.20
      icerat = 0.15
      curat = 0.05
      rhosn = 200.0
      snwe = 0.20
      snowfrac = 0.88
      znt = 0.10
      alb = 0.36
      alb_snow = 0.38
      alb_snow_free = 0.22
      cst = 2.0e-4
      snowfallac = 6.0
      do k = 1, nzs
        ts1d(k) = 265.0 + 0.6 * real(k - 1)
        soilm1d(k) = min(dqm, 0.24 + 0.003 * real(k - 1))
      end do
    end select

    if (seaice < 0.5) then
      do k = 1, nzs
        soiliqw(k) = soilm1d(k)
        soilice(k) = 0.0
      end do
    end if
    ! Snow depth is the diagnostic lsmruc carries, snwe*rhowater/rhosn,
    ! unless the case deliberately forces a different entry state.
    if (snhei_force >= 0.0) then
      snhei = snhei_force
    else if (snwe > 0.0) then
      snhei = snwe * 1.0e3 / rhosn
    else
      snhei = 0.0
    end if
    ! The solver geometry follows delt, which varies between cases.
    dtdzs = 0.0
    dtdzs2 = 0.0
    do k = 2, nzs - 1
      k1 = 2 * k - 3
      k2 = k1 + 1
      xgeom = delt / 2.0 / (zshalf(k + 1) - zshalf(k))
      dtdzs(k1) = xgeom / (zsmain(k) - zsmain(k - 1))
      dtdzs2(k - 1) = xgeom
      dtdzs(k2) = xgeom / (zsmain(k + 1) - zsmain(k))
    end do
    cn = cfactr_data
    qsg = 0.0
    qvg = 0.0
    qcg = 0.0
    ts1d_in = ts1d

    do k = 1, nzs
      write(unit, '(*(g0,:,","))') trim(names(n)), n, k, delt, c1sn, c2sn, &
          isice, ivgtyp, seaice, gsw, tabs, tsnav, prcpms, newsnms, vegfra, &
          lai, sat, soilt, snowfallac, alb_snow, alb_snow_free, snowrat, &
          grauprat, icerat, curat, iland, snwe, snhei, snowfrac, rhosn, &
          rhosnfall, cst, alb, emiss, znt, ts1d_in(k)
    end do

    call sfctmp(0, rstochcol, fieldcol_sf, &
        delt, ktau, conflx, n, 1, &
        nzs, nddzs, nroot, meltfactor, &
        iland, isoil, xland, ivgtyp, isltyp, prcpms, &
        newsnms, snwe, snhei, snowfrac, &
        rhosn, rhonewsn, rhosnfall, &
        snowrat, grauprat, icerat, curat, &
        patm, tabs, qvatm, qcatm, rho, &
        glw, gsw, emiss, qkms, tkms, pc, &
        mavail, cst, vegfra, alb, znt, &
        alb_snow, alb_snow_free, lai, &
        myj, seaice, isice, &
        qwrtz, rhocs, dqm, qmin, ref, wilt, psis, bclh, ksat, &
        sat, cn, zsmain, zshalf, dtdzs, dtdzs2, tbq, &
        cp, rovcp, g0_p, xlv, stbolt, cw, c1sn, c2sn, &
        kqwrtz, kice, kwt, &
        snweprint, snheiprint, rsm, &
        soilm1d, ts1d, smfrkeep, keepfr, soilt, soilt1, &
        tsnav, dew, qvg, qsg, qcg, &
        smelt, snoh, snflx, snom, snowfallac, acsnow, &
        edir1, ec1, ett1, eeta, qfx, hfx, s, sublim, &
        evapl, prcpl, fltot, runoff1, runoff2, soilice, &
        soiliqw, infiltr, smf)
  end do
  close(unit)
end program run_ruc_sfctmp_prep_oracle
