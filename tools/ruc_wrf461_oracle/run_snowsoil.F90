program run_ruc_snowsoil_oracle
  ! Drives the unmodified WRF v4.6.1 module_sf_ruclsm::snowsoil, the complete
  ! snow-covered land column: soil freezing partition, soilprop, transf,
  ! snowtemp and soilmoist, plus the surface flux diagnostics.  Soil and
  ! vegetation parameters come from the pinned WRF tables through the
  ! unmodified ruclsm_soilvegparm, exactly as run_soil.F90 does for the
  ! snow-free column.
  use module_sf_ruclsm, only: snowsoil, ruclsm_soilvegparm, drysmc, &
      maxsmc, refsmc, wltsmc, satpsi, satdk, bb, hc, qtz, pctbl, &
      laitbl, lemitbl, cfactr_data
  implicit none

  integer, parameter :: ncase = 4, nzs = 9, nddzs = 14
  character(len=24), parameter :: names(ncase) = [character(len=24) :: &
      'fresh_snow_cold_forest', 'deep_aged_snow_grass', &
      'melting_snow_rain', 'thin_snow_frozen_crop']
  integer, parameter :: soil_category(ncase) = [4, 6, 4, 8]
  integer, parameter :: land_category(ncase) = [1, 10, 10, 12]
  integer, parameter :: nroot_case(ncase) = [8, 6, 6, 4]
  real, parameter :: zsmain(nzs) = [0.0, 0.01, 0.04, 0.10, 0.30, &
      0.60, 1.00, 1.60, 3.00]
  real, parameter :: xlv = 2.5e6, cp = 1004.5, r_d = 287.0
  real, parameter :: g0_p = 9.81, cw = 4.183e6, stbolt = 5.67051e-8
  real, parameter :: rhowater = 1000.0
  character(len=1024) :: output_path
  integer :: n, k, k1, k2, unit, ktau, iland, isoil, ilnb, ilnb_before
  real :: delt, xgeom, cq, evs, eis, r61, conflx
  real :: meltfactor, rhonewsn, snhei_crit, prcpms, rainf, newsnow
  real :: snhei, snwe, snowfrac, rhosn, patm, qvatm, qcatm
  real :: glw, gsw, gswin, emiss, rnet, qkms, tkms, pc, cst, drip
  real :: infwater, rho, vegfrac, alb, znt, lai
  real :: qwrtz, rhocs, dqm, qmin, ref, wilt, psis, bclh, ksat
  real :: sat, cn, rovcp, tabs, kqwrtz, kice, kwt
  real :: snweprint, snheiprint, rsm, dew, soilt, soilt1, tsnav
  real :: qvg, qsg, qcg, smelt, snoh, snflx, snom
  real :: edir1, ec1, ett1, eeta, qfx, hfx, s, sublim
  real :: prcpl, fltot, runoff1, runoff2, mavail, infiltrp
  real :: snwe_before, snhei_before, rhosn_before, cst_before
  real :: soilt_before, soilt1_before, tsnav_before, dew_before
  real :: qvg_before, qsg_before, qcg_before, snom_before, mavail_before
  real :: zshalf(nzs), dtdzs(nddzs), dtdzs2(nzs), tbq(5001)
  real :: soilmois(nzs), tso(nzs), smfrkeep(nzs), keepfr(nzs)
  real :: soilice(nzs), soiliqw(nzs), rstochcol(nzs), fieldcol_sf(nzs)
  real :: soilmois_before(nzs), tso_before(nzs)
  real :: smfrkeep_before(nzs), keepfr_before(nzs)

  call get_command_argument(1, output_path)
  if (len_trim(output_path) == 0) then
    write(*, '(A)') 'usage: run_snowsoil OUTPUT.csv'
    error stop 2
  end if

  call ruclsm_soilvegparm('MODI-RUC', 'STAS-RUC')
  delt = 60.0
  zshalf(1) = 0.0
  do k = 2, nzs
    zshalf(k) = 0.5 * (zsmain(k - 1) + zsmain(k))
  end do
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
  sat = 5.0e-4
  cn = cfactr_data
  rstochcol = 0.0
  fieldcol_sf = 0.0
  ktau = 1

  open(newunit=unit, file=trim(output_path), status='replace', action='write')
  write(unit, '(A)') 'case,k,delt,ktau,conflx,nroot,iland,isoil,ivgtyp,myj,meltfactor,rhonewsn,snhei_crit,prcpms,rainf,newsnow,snhei_before,snwe_before,snowfrac,rhosn_before,patm,qvatm,qcatm,glw,gsw,gswin,emiss,rnet,qkms,tkms,pc,cst_before,cst_after,drip,infwater,rho,vegfrac,alb,znt,lai,qwrtz,rhocs,dqm,qmin,ref,wilt,psis,bclh,ksat,sat,cn,xlv,cp,rovcp,g0_p,cw,stbolt,tabs,kqwrtz,kice,kwt,zsmain,zshalf,soilmois_before,soilmois_after,tso_before,tso_after,smfrkeep_before,smfrkeep_after,keepfr_before,keepfr_after,soilice,soiliqw,dew_before,dew_after,soilt_before,soilt_after,soilt1_before,soilt1_after,tsnav_before,tsnav_after,qvg_before,qvg_after,qsg_before,qsg_after,qcg_before,qcg_after,snwe_after,snhei_after,rhosn_after,ilnb_before,ilnb_after,snweprint,snheiprint,rsm,smelt,snoh,snflx,snom_before,snom_after,edir1,ec1,ett1,eeta,qfx,hfx,s,sublim,prcpl,fltot,runoff1,runoff2,mavail_before,mavail_after,infiltrp'

  do n = 1, ncase
    iland = land_category(n)
    isoil = soil_category(n)
    qwrtz = qtz(isoil)
    rhocs = hc(isoil) * 1.0e6
    bclh = bb(isoil)
    dqm = maxsmc(isoil) - drysmc(isoil)
    ksat = satdk(isoil)
    psis = -satpsi(isoil)
    qmin = drysmc(isoil)
    ref = refsmc(isoil)
    wilt = wltsmc(isoil)
    emiss = lemitbl(iland)
    pc = pctbl(iland)
    lai = laitbl(iland)
    conflx = 40.0
    prcpms = 0.0
    rainf = 0.0
    newsnow = 0.0
    rhonewsn = 100.0
    meltfactor = 1.0
    patm = 0.95
    qcatm = 0.0
    rho = 1.25
    alb = 0.70
    znt = 0.02
    cst = 0.0
    drip = 0.0
    infwater = 0.0
    snowfrac = 1.0
    mavail = 0.60
    ilnb = 1
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
    prcpl = 0.0
    fltot = 0.0
    runoff1 = 0.0
    runoff2 = 0.0
    infiltrp = 0.0
    rsm = 0.0
    snweprint = 0.0
    snheiprint = 0.0
    keepfr = 0.0
    smfrkeep = 0.0

    select case (n)
    case (1)
      ! 0.20 m of fresh 100 kg/m^3 snow still falling onto frozen forest
      ! soil: snowtemp takes the one-layer solve, no melt.
      rhosn = 100.0
      snwe = 0.02
      newsnow = 0.002
      rhonewsn = 100.0
      tabs = 265.0
      qvatm = 0.0018
      glw = 230.0
      gsw = 40.0
      gswin = 130.0
      rnet = -20.0
      qkms = 0.010
      tkms = 0.009
      vegfrac = 0.55
      cst = 4.0e-5
      soilt = 264.0
      soilt1 = 264.0
      qvg = 0.0018325
      qsg = 0.0018325
      qcg = 0.0
      do k = 1, nzs
        tso(k) = 263.0 + 1.1 * real(k - 1)
        soilmois(k) = min(dqm, 0.22 + 0.004 * real(k - 1))
      end do
    case (2)
      ! 0.43 m aged 350 kg/m^3 pack over grassland: two snow layers.
      rhosn = 350.0
      snwe = 0.15
      tabs = 258.0
      qvatm = 0.0009
      glw = 195.0
      gsw = 25.0
      gswin = 85.0
      rnet = -35.0
      qkms = 0.008
      tkms = 0.007
      vegfrac = 0.15
      soilt = 257.0
      soilt1 = 260.0
      qvg = 0.00097137
      qsg = 0.00097137
      qcg = 0.0
      do k = 1, nzs
        tso(k) = 265.0 + 0.9 * real(k - 1)
        soilmois(k) = min(dqm, 0.26 + 0.003 * real(k - 1))
      end do
    case (3)
      ! Light rain on a ripe 0.32 m pack over thawed grassland soil: the
      ! melt iteration runs, smelt feeds soilmoist, snom accumulates, and
      ! the warm top soil layer also melts the pack from below.
      rhosn = 250.0
      snwe = 0.08
      prcpms = 3.0e-6
      rainf = 1.0
      infwater = 2.5e-6
      tabs = 274.6
      qvatm = 0.00415
      glw = 305.0
      gsw = 70.0
      gswin = 180.0
      rnet = 20.0
      qkms = 0.012
      tkms = 0.010
      vegfrac = 0.30
      soilt = 273.14
      soilt1 = 273.05
      qvg = 0.0039961
      qsg = 0.0039961
      qcg = 0.0
      do k = 1, nzs
        tso(k) = 273.4 + 0.20 * real(k - 1)
        soilmois(k) = min(dqm, 0.30 + 0.003 * real(k - 1))
      end do
    case (4)
      ! 0.025 m of snow with snth = 0.05 over frozen cropland whose
      ! freeze-thaw balance is already latched (keepfr = 1) with a
      ! smfrkeep low enough that the "ice cannot increase" cap actually
      ! binds; the snow is blended into the top soil layer.
      rhosn = 200.0
      snwe = 0.005
      tabs = 266.0
      qvatm = 0.0021
      glw = 240.0
      gsw = 60.0
      gswin = 200.0
      rnet = -5.0
      qkms = 0.011
      tkms = 0.010
      vegfrac = 0.40
      soilt = 265.0
      soilt1 = 265.0
      qvg = 0.0020011
      qsg = 0.0020011
      qcg = 0.0
      do k = 1, nzs
        tso(k) = 264.0 + 0.8 * real(k - 1)
        soilmois(k) = min(dqm, 0.28 + 0.003 * real(k - 1))
        smfrkeep(k) = 0.5 * soilmois(k) / 0.9
        keepfr(k) = 1.0
      end do
    end select

    snhei = snwe * 1.0e3 / rhosn
    snhei_crit = 0.01601 * rhowater / rhosn

    snwe_before = snwe
    snhei_before = snhei
    rhosn_before = rhosn
    cst_before = cst
    soilmois_before = soilmois
    tso_before = tso
    smfrkeep_before = smfrkeep
    keepfr_before = keepfr
    soilt_before = soilt
    soilt1_before = soilt1
    tsnav = soilt - 273.15
    tsnav_before = tsnav
    dew_before = dew
    qvg_before = qvg
    qsg_before = qsg
    qcg_before = qcg
    snom_before = snom
    mavail_before = mavail
    ilnb_before = ilnb

    call snowsoil(0, rstochcol, fieldcol_sf, 1, n, isoil, delt, ktau, &
        conflx, nzs, nddzs, nroot_case(n), meltfactor, rhonewsn, &
        snhei_crit, iland, prcpms, rainf, newsnow, snhei, snwe, snowfrac, &
        rhosn, patm, qvatm, qcatm, glw, gsw, gswin, emiss, rnet, iland, &
        qkms, tkms, pc, cst, drip, infwater, rho, vegfrac, alb, znt, lai, &
        .false., qwrtz, rhocs, dqm, qmin, ref, wilt, psis, bclh, ksat, &
        sat, cn, zsmain, zshalf, dtdzs, dtdzs2, tbq, xlv, cp, rovcp, &
        g0_p, cw, stbolt, tabs, kqwrtz, kice, kwt, ilnb, snweprint, &
        snheiprint, rsm, soilmois, tso, smfrkeep, keepfr, dew, soilt, &
        soilt1, tsnav, qvg, qsg, qcg, smelt, snoh, snflx, snom, edir1, &
        ec1, ett1, eeta, qfx, hfx, s, sublim, prcpl, fltot, runoff1, &
        runoff2, mavail, soilice, soiliqw, infiltrp)

    do k = 1, nzs
      write(unit, '(*(g0,:,","))') trim(names(n)), k, delt, ktau, conflx, &
          nroot_case(n), iland, isoil, iland, 0, meltfactor, rhonewsn, &
          snhei_crit, prcpms, rainf, newsnow, snhei_before, snwe_before, &
          snowfrac, rhosn_before, patm, qvatm, qcatm, glw, gsw, gswin, &
          emiss, rnet, qkms, tkms, pc, cst_before, cst, drip, infwater, &
          rho, vegfrac, alb, znt, lai, qwrtz, rhocs, dqm, qmin, ref, &
          wilt, psis, bclh, ksat, sat, cn, xlv, cp, rovcp, g0_p, cw, &
          stbolt, tabs, kqwrtz, kice, kwt, zsmain(k), zshalf(k), &
          soilmois_before(k), soilmois(k), tso_before(k), tso(k), &
          smfrkeep_before(k), smfrkeep(k), keepfr_before(k), keepfr(k), &
          soilice(k), soiliqw(k), dew_before, dew, soilt_before, soilt, &
          soilt1_before, soilt1, tsnav_before, tsnav, qvg_before, qvg, &
          qsg_before, qsg, qcg_before, qcg, snwe, snhei, rhosn, &
          ilnb_before, ilnb, snweprint, snheiprint, rsm, smelt, snoh, &
          snflx, snom_before, snom, edir1, ec1, ett1, eeta, qfx, hfx, s, &
          sublim, prcpl, fltot, runoff1, runoff2, mavail_before, mavail, &
          infiltrp
    end do
  end do
  close(unit)
end program run_ruc_snowsoil_oracle
