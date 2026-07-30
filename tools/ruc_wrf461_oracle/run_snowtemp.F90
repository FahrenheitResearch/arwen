program run_ruc_snowtemp_oracle
  ! Drives the unmodified WRF v4.6.1 module_sf_ruclsm::snowtemp, the snow
  ! energy-budget and snow/soil heat-diffusion solve.  Everything the harness
  ! computes before the call (deltsn, snth, snwepr, beta) is the arithmetic
  ! snowsoil performs on its own inputs, reproduced verbatim; every one of
  ! those derived quantities is written to the CSV so a row is independently
  ! reproducible without re-deriving anything.
  !
  ! snowtemp declares ilnb as intent(out) yet reads it at the tsnav update
  ! when the snow layer is blended into the top soil layer.  The harness
  ! therefore seeds ilnb and records ilnb_before as part of the input state.
  use module_sf_ruclsm, only: snowtemp
  implicit none

  integer, parameter :: ncase = 6, nzs = 9, nddzs = 14
  character(len=24), parameter :: names(ncase) = [character(len=24) :: &
      'shallow_fresh_1layer', 'deep_aged_2layer', 'melting_2layer', &
      'thin_blended', 'warm_ground_bottom_melt', 'sublimating_all_snow']
  integer, parameter :: nroot_case(ncase) = [6, 4, 6, 8, 6, 4]
  real, parameter :: zsmain(nzs) = [0.0, 0.01, 0.04, 0.10, 0.30, &
      0.60, 1.00, 1.60, 3.00]
  real, parameter :: xlv = 2.5e6, cp = 1004.5, r_d = 287.0
  real, parameter :: g0_p = 9.81, cw = 4.183e6, stbolt = 5.67051e-8
  real, parameter :: xlmelt = 3.35e5
  character(len=1024) :: output_path
  integer :: n, k, k1, k2, unit, ktau, iland, isoil, ilnb, ilnb_before
  real :: delt, xgeom, cq, evs, eis, r61, conflx
  real :: snwe, snwepr, snhei, newsnow, snowfrac, beta, deltsn, snth
  real :: rhosn, rhonewsn, meltfactor, prcpms, rainf, patm, tabs
  real :: qvatm, qcatm, glw, gsw, emiss, rnet, qkms, tkms, pc, rho
  real :: vegfrac, drycan, wetcan, cst, transum, dew, mavail
  real :: dqm, qmin, psis, bclh, xlvm, rovcp, cvw
  real :: snweprint, snheiprint, rsm, soilt, soilt1, tsnav
  real :: qvg, qsg, qcg, smelt, snoh, snflx, s, x
  real :: umveg, epot, epdt, ras
  real :: snwe_before, snhei_before, beta_before, rhosn_before
  real :: soilt_before, soilt1_before, tsnav_before, dew_before
  real :: qvg_before, qsg_before, qcg_before
  real :: zshalf(nzs), dtdzs(nddzs), tbq(5001)
  real :: thdif(nzs), cap(nzs), tranf(nzs), tso(nzs), tso_before(nzs)

  call get_command_argument(1, output_path)
  if (len_trim(output_path) == 0) then
    write(*, '(A)') 'usage: run_snowtemp OUTPUT.csv'
    error stop 2
  end if

  delt = 60.0
  zshalf(1) = 0.0
  do k = 2, nzs
    zshalf(k) = 0.5 * (zsmain(k - 1) + zsmain(k))
  end do
  dtdzs = 0.0
  do k = 2, nzs - 1
    k1 = 2 * k - 3
    k2 = k1 + 1
    xgeom = delt / 2.0 / (zshalf(k + 1) - zshalf(k))
    dtdzs(k1) = xgeom / (zsmain(k) - zsmain(k - 1))
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

  xlvm = xlv + xlmelt
  rovcp = r_d / cp
  cvw = cw
  ktau = 1
  iland = 1
  isoil = 4

  open(newunit=unit, file=trim(output_path), status='replace', action='write')
  write(unit, '(A)') 'case,k,delt,ktau,conflx,nroot,iland,isoil,snwe_before,snwepr,snhei_before,newsnow,snowfrac,beta_before,deltsn,snth,rhosn_before,rhonewsn,meltfactor,prcpms,rainf,patm,tabs,qvatm,qcatm,glw,gsw,emiss,rnet,qkms,tkms,pc,rho,vegfrac,drycan,wetcan,cst,transum,mavail,dqm,qmin,psis,bclh,xlvm,cp,rovcp,g0_p,cvw,stbolt,zsmain,zshalf,thdif,cap,tranf,tso_before,tso_after,soilt_before,soilt_after,soilt1_before,soilt1_after,tsnav_before,tsnav_after,qvg_before,qvg_after,qsg_before,qsg_after,qcg_before,qcg_after,dew_before,dew_after,snwe_after,snhei_after,rhosn_after,beta_after,smelt,snoh,snflx,s,rsm,snweprint,snheiprint,x,ilnb_before,ilnb_after'

  do n = 1, ncase
    conflx = 40.0
    prcpms = 0.0
    rainf = 0.0
    patm = 0.95
    qcatm = 0.0
    pc = 0.7
    rho = 1.25
    vegfrac = 0.20
    wetcan = 0.02
    drycan = 1.0 - wetcan
    cst = 0.0
    mavail = 0.60
    dqm = 0.40
    qmin = 0.02
    psis = -0.35
    bclh = 5.4
    emiss = 0.98
    meltfactor = 1.0
    newsnow = 0.0
    rhonewsn = 100.0
    snowfrac = 1.0
    ilnb = 1
    dew = 0.0
    smelt = 0.0
    snoh = 0.0
    snflx = 0.0
    s = 0.0
    x = 0.0
    rsm = 0.0
    snweprint = 0.0
    snheiprint = 0.0
    do k = 1, nzs
      thdif(k) = 6.5e-7 * (1.0 + 0.03 * real(k - 1))
      cap(k) = 1.90e6 + 3.0e4 * real(k - 1)
      tranf(k) = 0.0
    end do

    select case (n)
    case (1)
      ! Fresh low-density snow, 0.20 m deep: snhei >= snth and
      ! snhei <= deltsn+snth so snowtemp takes the one-layer solve.
      rhosn = 100.0
      snwe = 0.02
      newsnow = 0.002
      rhonewsn = 100.0
      tabs = 265.0
      qvatm = 0.0018
      glw = 230.0
      gsw = 40.0
      rnet = -20.0
      qkms = 0.010
      tkms = 0.009
      soilt = 264.0
      soilt1 = 264.0
      qvg = 0.0018325
      qsg = 0.0018325
      qcg = 0.0
      do k = 1, nzs
        tso(k) = 263.0 + 1.1 * real(k - 1)
      end do
    case (2)
      ! Deep aged pack, 0.43 m: snhei > deltsn+snth so the two-layer
      ! snow solve with an interior soilt1 temperature is taken.
      rhosn = 350.0
      snwe = 0.15
      tabs = 258.0
      qvatm = 0.0009
      glw = 195.0
      gsw = 25.0
      rnet = -35.0
      qkms = 0.008
      tkms = 0.007
      vegfrac = 0.05
      soilt = 257.0
      soilt1 = 260.0
      qvg = 0.00097137
      qsg = 0.00097137
      qcg = 0.0
      do k = 1, nzs
        tso(k) = 265.0 + 0.9 * real(k - 1)
      end do
    case (3)
      ! Light rain on a ripe two-layer pack that is already at the melting
      ! point: soilt clears 273.15 with beta = 1, so the nmelt iteration
      ! and the Egglston melt limiter both run.
      rhosn = 250.0
      snwe = 0.08
      prcpms = 3.0e-6
      rainf = 1.0
      tabs = 274.6
      qvatm = 0.00415
      glw = 305.0
      gsw = 70.0
      rnet = 20.0
      qkms = 0.012
      tkms = 0.010
      soilt = 273.14
      soilt1 = 273.05
      qvg = 0.0039961
      qsg = 0.0039961
      qcg = 0.0
      do k = 1, nzs
        tso(k) = 272.60 + 0.06 * real(k - 1)
      end do
    case (4)
      ! 0.025 m of snow with snth = 0.05: too thin to carry its own
      ! layer, so it is blended with the top soil layer.
      rhosn = 200.0
      snwe = 0.005
      tabs = 266.0
      qvatm = 0.0021
      glw = 240.0
      gsw = 60.0
      rnet = -5.0
      qkms = 0.011
      tkms = 0.010
      vegfrac = 0.35
      soilt = 265.0
      soilt1 = 265.0
      qvg = 0.0020011
      qsg = 0.0020011
      qcg = 0.0
      do k = 1, nzs
        tso(k) = 264.0 + 1.0 * real(k - 1)
      end do
    case (5)
      ! Cold air over thawed ground: the surface stays below freezing so
      ! no top melt happens, but tso(1) > 273.15 drives the bottom-melt
      ! (snohg/smeltg) branch.
      rhosn = 300.0
      snwe = 0.05
      tabs = 268.0
      qvatm = 0.0028
      glw = 250.0
      gsw = 30.0
      rnet = -10.0
      qkms = 0.009
      tkms = 0.008
      soilt = 270.0
      soilt1 = 270.0
      qvg = 0.0030757
      qsg = 0.0030757
      qcg = 0.0
      do k = 1, nzs
        tso(k) = 276.0 + 0.15 * real(k - 1)
      end do
    case (6)
      ! A trace of snow under dry warm air: the potential sublimation over
      ! one step exceeds snwe, so beta < 1 and the whole pack sublimates.
      rhosn = 150.0
      snwe = 1.0e-6
      tabs = 275.0
      qvatm = 0.0005
      glw = 300.0
      gsw = 150.0
      rnet = 160.0
      qkms = 0.050
      tkms = 0.020
      vegfrac = 0.10
      soilt = 272.0
      soilt1 = 272.0
      qvg = 0.0036361
      qsg = 0.0036361
      qcg = 0.0
      do k = 1, nzs
        tso(k) = 271.0 + 0.6 * real(k - 1)
      end do
    end select

    do k = 1, nroot_case(n)
      tranf(k) = 0.02 * (1.0 - 0.1 * real(k - 1))
    end do
    transum = 0.0
    do k = 1, nroot_case(n)
      transum = transum + tranf(k)
    end do

    snhei = snwe * 1.0e3 / rhosn

    ! --- snowsoil's own pre-call arithmetic, reproduced verbatim ---
    deltsn = 0.05 * 1.e3 / rhosn
    snth = 0.01 * 1.e3 / rhosn
    if (snhei >= deltsn + snth) then
      if (snhei - deltsn - snth < snth) deltsn = 0.5 * (snhei - snth)
    end if
    ras = rho * 1.e-3
    umveg = 1.0 - vegfrac
    snwepr = snwe
    beta = 1.0
    epot = -qkms * (qvatm - qsg)
    epdt = epot * ras * delt * umveg
    if (epdt > 0. .and. snwepr <= epdt) then
      beta = snwepr / max(1.e-8, epdt)
      snwe = 0.0
    end if
    ! --------------------------------------------------------------

    snwe_before = snwe
    snhei_before = snhei
    beta_before = beta
    rhosn_before = rhosn
    tso_before = tso
    soilt_before = soilt
    soilt1_before = soilt1
    tsnav = soilt - 273.15
    tsnav_before = tsnav
    dew_before = dew
    qvg_before = qvg
    qsg_before = qsg
    qcg_before = qcg
    ilnb_before = ilnb

    call snowtemp(1, n, iland, isoil, delt, ktau, conflx, nzs, nddzs, &
        nroot_case(n), snwe, snwepr, snhei, newsnow, snowfrac, beta, &
        deltsn, snth, rhosn, rhonewsn, meltfactor, prcpms, rainf, patm, &
        tabs, qvatm, qcatm, glw, gsw, emiss, rnet, qkms, tkms, pc, rho, &
        vegfrac, thdif, cap, drycan, wetcan, cst, tranf, transum, dew, &
        mavail, dqm, qmin, psis, bclh, zsmain, zshalf, dtdzs, tbq, xlvm, &
        cp, rovcp, g0_p, cvw, stbolt, snweprint, snheiprint, rsm, tso, &
        soilt, soilt1, tsnav, qvg, qsg, qcg, smelt, snoh, snflx, s, &
        ilnb, x)

    do k = 1, nzs
      write(unit, '(*(g0,:,","))') trim(names(n)), k, delt, ktau, conflx, &
          nroot_case(n), iland, isoil, snwe_before, snwepr, snhei_before, &
          newsnow, snowfrac, beta_before, deltsn, snth, rhosn_before, &
          rhonewsn, meltfactor, prcpms, rainf, patm, tabs, qvatm, qcatm, &
          glw, gsw, emiss, rnet, qkms, tkms, pc, rho, vegfrac, drycan, &
          wetcan, cst, transum, mavail, dqm, qmin, psis, bclh, xlvm, cp, &
          rovcp, g0_p, cvw, stbolt, zsmain(k), zshalf(k), thdif(k), &
          cap(k), tranf(k), tso_before(k), tso(k), soilt_before, soilt, &
          soilt1_before, soilt1, tsnav_before, tsnav, qvg_before, qvg, &
          qsg_before, qsg, qcg_before, qcg, dew_before, dew, snwe, &
          snhei, rhosn, beta, smelt, snoh, snflx, s, rsm, snweprint, &
          snheiprint, x, ilnb_before, ilnb
    end do
  end do
  close(unit)
end program run_ruc_snowtemp_oracle
