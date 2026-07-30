program run_ruc_soil_oracle
  use module_sf_ruclsm, only: soil, ruclsm_soilvegparm, drysmc, &
      maxsmc, refsmc, wltsmc, satpsi, satdk, bb, hc, qtz, pctbl, &
      laitbl, lemitbl, cfactr_data
  implicit none

  integer, parameter :: ncase = 4, nzs = 9, nddzs = 14
  character(len=20), parameter :: names(ncase) = [character(len=20) :: &
      'warm_forest_rain', 'dry_grass', 'frozen_crop_rain', 'humid_bare_dew']
  integer, parameter :: soil_category(ncase) = [1, 4, 6, 8]
  integer, parameter :: land_category(ncase) = [1, 10, 12, 16]
  integer, parameter :: nroot_case(ncase) = [8, 6, 6, 6]
  real, parameter :: zsmain(nzs) = [0.0, 0.01, 0.04, 0.10, 0.30, &
      0.60, 1.00, 1.60, 3.00]
  character(len=1024) :: output_path
  integer :: n, k, k1, k2, unit, iland, isoil
  real :: delt, xgeom, cq, evs, eis, r61, conflx
  real :: prcpms, rainf, patm, qvatm, qcatm, glw, gsw, gswin
  real :: emiss, rnet, qkms, tkms, pc, cst, cst_before, drip
  real :: infwater, rho, vegfrac, lai
  real :: qwrtz, rhocs, dqm, qmin, ref, wilt, psis, bclh, ksat
  real :: sat, cn, xlv, cp, rovcp, g0_p, cw, stbolt, tabs
  real :: kqwrtz, kice, kwt
  real :: dew, soilt, qvg, qsg, qcg, edir1, ec1, ett1, eeta
  real :: qfx, hfx, s, evapl, prcpl, fltot, runoff1, runoff2
  real :: mavail, infiltrp, smf
  real :: mavail_before
  real :: soilt_before, qvg_before, qsg_before, qcg_before
  real :: zshalf(nzs), dtdzs(nddzs), dtdzs2(nzs), tbq(5001)
  real :: soilmois(nzs), tso(nzs), smfrkeep(nzs), keepfr(nzs)
  real :: soilice(nzs), soiliqw(nzs), rstochcol(nzs), fieldcol_sf(nzs)
  real :: soilmois_before(nzs), tso_before(nzs)
  real :: smfrkeep_before(nzs), keepfr_before(nzs)

  call get_command_argument(1, output_path)
  if (len_trim(output_path) == 0) then
    write(*, '(A)') 'usage: run_soil OUTPUT.csv'
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

  xlv = 2.5e6
  cp = 1004.5
  rovcp = 287.0 / cp
  g0_p = 9.81
  cw = 4.183e6
  stbolt = 5.67051e-8
  kqwrtz = 7.7
  kice = 2.2
  kwt = 0.57
  sat = 5.0e-4
  cn = cfactr_data
  rstochcol = 0.0
  fieldcol_sf = 0.0

  open(newunit=unit, file=trim(output_path), status='replace', action='write')
  write(unit, '(A)') 'case,k,iland,isoil,nroot,delt,conflx,prcpms,rainf,patm,qvatm,qcatm,glw,gsw,gswin,emiss,rnet,qkms,tkms,pc,cst_before,cst_after,drip,infwater,rho,vegfrac,lai,qwrtz,rhocs,dqm,qmin,ref,wilt,psis,bclh,ksat,sat,cn,tabs,soilmois_before,soilmois_after,tso_before,tso_after,smfrkeep_before,smfrkeep_after,keepfr_before,keepfr_after,soilice,soiliqw,soilt_before,soilt_after,qvg_before,qvg_after,qsg_before,qsg_after,qcg_before,qcg_after,dew,edir1,ec1,ett1,eeta,qfx,hfx,s,evapl,prcpl,fltot,runoff1,runoff2,mavail_before,mavail_after,infiltrp,smf'

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
    patm = 0.95
    qvatm = 0.010
    qcatm = 0.0
    glw = 330.0
    gsw = 420.0
    gswin = 500.0
    rnet = 300.0
    qkms = 0.015
    tkms = 0.010
    cst = 8.0e-5
    drip = 0.0
    infwater = 0.0
    rho = 1.10
    vegfrac = 0.65
    tabs = 295.0
    soilt = 294.0
    qvg = 0.012
    qsg = 0.016
    qcg = 0.0
    mavail = 0.60
    keepfr = 0.0
    smfrkeep = 0.0
    do k = 1, nzs
      tso(k) = 294.0 - 0.75 * real(k - 1)
      soilmois(k) = min(dqm, 0.24 + 0.004 * real(k - 1))
    end do

    select case (n)
    case (1)
      prcpms = 8.0e-6
      rainf = 1.0
      infwater = 6.5e-6
      rnet = 360.0
      vegfrac = 0.75
    case (2)
      tabs = 305.0
      qvatm = 0.003
      rnet = 520.0
      vegfrac = 0.35
      cst = 0.0
      soilt = 302.0
      qvg = 0.006
      qsg = 0.026
      do k = 1, nzs
        tso(k) = 302.0 - 0.55 * real(k - 1)
        soilmois(k) = min(dqm, 0.035 + 0.003 * real(k - 1))
      end do
    case (3)
      tabs = 270.0
      qvatm = 0.0035
      prcpms = 4.0e-6
      rainf = 1.0
      infwater = 4.0e-6
      rnet = 110.0
      vegfrac = 0.55
      soilt = 269.0
      qvg = 0.003
      qsg = 0.004
      do k = 1, nzs
        tso(k) = 268.0 + 0.45 * real(k - 1)
        soilmois(k) = min(dqm, 0.28 + 0.003 * real(k - 1))
        smfrkeep(k) = soilmois(k) / 0.9
      end do
    case (4)
      tabs = 289.0
      qvatm = 0.018
      rnet = 80.0
      vegfrac = 0.10
      cst = 0.0
      soilt = 290.0
      qvg = 0.010
      qsg = 0.012
      qcg = 0.0
      do k = 1, nzs
        tso(k) = 290.0 - 0.35 * real(k - 1)
        soilmois(k) = min(dqm, 0.18 + 0.002 * real(k - 1))
      end do
    end select

    soilmois_before = soilmois
    tso_before = tso
    smfrkeep_before = smfrkeep
    keepfr_before = keepfr
    soilt_before = soilt
    qvg_before = qvg
    qsg_before = qsg
    qcg_before = qcg
    cst_before = cst
    mavail_before = mavail
    dew = -999.0
    edir1 = -999.0
    ec1 = -999.0
    ett1 = -999.0
    eeta = -999.0
    qfx = -999.0
    hfx = -999.0
    s = -999.0
    evapl = -999.0
    prcpl = -999.0
    fltot = -999.0
    runoff1 = -999.0
    runoff2 = -999.0
    infiltrp = -999.0
    smf = -999.0
    call soil(0, rstochcol, fieldcol_sf, 1, n, iland, isoil, delt, 1, &
        conflx, nzs, nddzs, nroot_case(n), prcpms, rainf, patm, qvatm, &
        qcatm, glw, gsw, gswin, emiss, rnet, qkms, tkms, pc, cst, drip, &
        infwater, rho, vegfrac, lai, .false., qwrtz, rhocs, dqm, qmin, &
        ref, wilt, psis, bclh, ksat, sat, cn, zsmain, zshalf, dtdzs, &
        dtdzs2, tbq, xlv, cp, rovcp, g0_p, cw, stbolt, tabs, kqwrtz, &
        kice, kwt, soilmois, tso, smfrkeep, keepfr, dew, soilt, qvg, &
        qsg, qcg, edir1, ec1, ett1, eeta, qfx, hfx, s, evapl, prcpl, &
        fltot, runoff1, runoff2, mavail, soilice, soiliqw, infiltrp, smf)
    do k = 1, nzs
      write(unit, '(*(g0,:,","))') trim(names(n)), k, iland, isoil, &
          nroot_case(n), delt, conflx, prcpms, rainf, patm, qvatm, qcatm, &
          glw, gsw, gswin, emiss, rnet, qkms, tkms, pc, cst_before, cst, &
          drip, infwater, rho, vegfrac, lai, qwrtz, rhocs, dqm, qmin, ref, &
          wilt, psis, bclh, ksat, sat, cn, tabs, soilmois_before(k), &
          soilmois(k), tso_before(k), tso(k), smfrkeep_before(k), &
          smfrkeep(k), keepfr_before(k), keepfr(k), soilice(k), &
          soiliqw(k), soilt_before, soilt, qvg_before, qvg, &
          qsg_before, qsg, qcg_before, qcg, dew, edir1, ec1, ett1, eeta, &
          qfx, hfx, s, evapl, prcpl, fltot, runoff1, runoff2, &
          mavail_before, mavail, &
          infiltrp, smf
    end do
  end do
  close(unit)
end program run_ruc_soil_oracle
