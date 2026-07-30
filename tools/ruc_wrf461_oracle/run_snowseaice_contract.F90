program run_ruc_snowseaice_contract_oracle
  ! Drives the unmodified WRF v4.6.1 module_sf_ruclsm::snowseaice on the
  ! argument-contract regimes that oracle/snowseaice.csv leaves unpinned.
  !
  ! oracle/snowseaice.csv pins the arithmetic of five snow-on-ice regimes, but
  ! it holds myj = .false., snowfrac = 1., meltfactor = 1., rainf = 1.,
  ! ilnb = 1, snwe > 0 and xlv = 2.5e6 across every row.  Because each of
  ! those values is either a multiplicative identity or a branch the file
  ! never enters, a port that dropped the argument entirely still reproduced
  ! the file bit for bit.  These ten cases give each one a value or a branch
  ! where its omission changes an output column:
  !
  !   1 myj_evaporating    myj = .true. with q1 >= 0   (:4455-4462)
  !   2 myj_condensing     myj = .true. with q1 <  0   (:4442-4449)
  !   3 partial_cover_melt snowfrac/meltfactor/rainf off unity (:4260,:4318,
  !                        :4168)
  !   4 blended_ilnb_entry blended pack entered with ilnb = 2 (:4410); lsmruc
  !                        leaves ilnb undefined at :1385 so the incoming
  !                        value is live
  !   5 bare_ice_entry     snwe = 0 so snhei = 0 on entry: the ":4157" all
  !                        sublimated coefficients, the ":4480" snflx form
  !                        that leaves s untouched, and the ":4517" bare-ice
  !                        restore
  !   6 deltsn_halved      snhei in [deltsn+snth, deltsn+2*snth) so :3951
  !                        halves deltsn
  !   7 thin_pack_melt     melting with snhei <= 0.01 so :4346 zeroes rsm
  !   8 melt_evaporating   melting with q1 > 0 (:4277) rather than :4271
  !   9 xlv_offset         xlv /= 2.5e6 so xlvm at :3932 is live
  !  10 snhei_state_drift  incoming snhei /= snwe*1.e3/rhosn.  lsmruc only
  !                        re-syncs the two at :1596 under newsn > 0, so the
  !                        snowh and snow prognostics drift apart; the
  !                        incoming snhei is read once, by the :3950-3951
  !                        deltsn test, before :4004 rebuilds it from snwe
  !
  ! Everything else -- the geometry, the Zubov ice-column relations, the tbq
  ! table and the CSV header -- is identical to run_snowseaice.F90, so both
  ! files are read by one reader.
  use module_sf_ruclsm, only: snowseaice
  implicit none

  integer, parameter :: ncase = 10, nzs = 9, nddzs = 14
  character(len=24), parameter :: names(ncase) = [character(len=24) :: &
      'myj_evaporating', 'myj_condensing', 'partial_cover_melt', &
      'blended_ilnb_entry', 'bare_ice_entry', 'deltsn_halved', &
      'thin_pack_melt', 'melt_evaporating', 'xlv_offset', &
      'snhei_state_drift']
  real, parameter :: zsmain(nzs) = [0.0, 0.01, 0.04, 0.10, 0.30, &
      0.60, 1.00, 1.60, 3.00]
  real, parameter :: cp = 1004.5, r_d = 287.0
  real, parameter :: cw = 4.183e6, stbolt = 5.67051e-8
  real, parameter :: rhowater = 1000.0
  character(len=1024) :: output_path
  integer :: n, k, k1, k2, unit, ktau, iland, isoil, ilnb, ilnb_before
  real :: delt, xgeom, cq, evs, eis, r61, conflx, cice
  real :: meltfactor, rhonewsn, snhei_crit, prcpms, rainf, newsnow
  real :: snhei, snwe, snowfrac, rhosn, patm, qvatm, qcatm
  real :: glw, gsw, emiss, rnet, qkms, tkms, rho, alb, znt
  real :: rovcp, tabs, snweprint, snheiprint, rsm, xlv
  real :: dew, soilt, soilt1, tsnav, qvg, qsg, qcg
  real :: smelt, snoh, snflx, snom, eeta, qfx, hfx, s, sublim
  real :: prcpl, fltot
  real :: snwe_before, snhei_before, rhosn_before, emiss_before
  real :: alb_before, znt_before, soilt_before, soilt1_before
  real :: tsnav_before, dew_before, qvg_before, qsg_before, qcg_before
  real :: snom_before, s_before
  logical :: myj
  integer :: myjout
  real :: zshalf(nzs), dtdzs(nddzs), dtdzs2(nzs), tbq(5001)
  real :: tice(nzs), rhosice(nzs), capice(nzs), thdifice(nzs)
  real :: tso(nzs), tso_before(nzs)

  call get_command_argument(1, output_path)
  if (len_trim(output_path) == 0) then
    write(*, '(A)') 'usage: run_snowseaice_contract OUTPUT.csv'
    error stop 2
  end if

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
  ktau = 1
  iland = 24
  isoil = 16

  open(newunit=unit, file=trim(output_path), status='replace', action='write')
  write(unit, '(A)') 'case,k,delt,ktau,conflx,iland,isoil,myj,meltfactor,rhonewsn,snhei_crit,prcpms,rainf,newsnow,snhei_before,snwe_before,snowfrac,rhosn_before,patm,qvatm,qcatm,glw,gsw,emiss_before,rnet,qkms,tkms,rho,alb_before,znt_before,xlv,cp,rovcp,cw,stbolt,tabs,zsmain,zshalf,tice,rhosice,capice,thdifice,tso_before,tso_after,soilt_before,soilt_after,soilt1_before,soilt1_after,tsnav_before,tsnav_after,qvg_before,qvg_after,qsg_before,qsg_after,qcg_before,qcg_after,dew_before,dew_after,snwe_after,snhei_after,rhosn_after,emiss_after,alb_after,znt_after,ilnb_before,ilnb_after,snweprint,snheiprint,rsm,smelt,snoh,snflx,snom_before,snom_after,eeta,qfx,hfx,s_before,s_after,sublim,prcpl,fltot'

  do n = 1, ncase
    conflx = 40.0
    prcpms = 0.0
    rainf = 0.0
    newsnow = 0.0
    rhonewsn = 100.0
    meltfactor = 1.0
    patm = 0.95
    qcatm = 0.0
    rho = 1.30
    emiss = 0.99
    alb = 0.75
    znt = 0.02
    snowfrac = 1.0
    ilnb = 1
    myj = .false.
    xlv = 2.5e6
    dew = 0.0
    smelt = 0.0
    snoh = 0.0
    snflx = 0.0
    snom = 0.0
    eeta = 0.0
    qfx = 0.0
    hfx = 0.0
    s = 0.0
    sublim = 0.0
    prcpl = 0.0
    fltot = 0.0
    rsm = 0.0
    snweprint = 0.0
    snheiprint = 0.0

    select case (n)
    case (1)
      ! myj moisture flux on the evaporating side: dry air over a 0.12 m
      ! one-layer pack.
      myj = .true.
      rhosn = 250.0
      snwe = 0.03
      tabs = 258.0
      qvatm = 0.0002
      glw = 200.0
      gsw = 35.0
      rnet = -25.0
      qkms = 0.010
      tkms = 0.009
      soilt = 257.0
      soilt1 = 257.0
      qvg = 0.00097137
      qsg = 0.00097137
      qcg = 0.0
      do k = 1, nzs
        tso(k) = 256.0 + 1.6 * real(k - 1)
      end do
    case (2)
      ! myj moisture flux on the condensing side: moist air over a cold
      ! one-layer pack drives q1 < 0.
      myj = .true.
      rhosn = 250.0
      snwe = 0.03
      tabs = 262.0
      qvatm = 0.0040
      glw = 215.0
      gsw = 35.0
      rnet = -20.0
      qkms = 0.010
      tkms = 0.009
      soilt = 257.0
      soilt1 = 257.0
      qvg = 0.00097137
      qsg = 0.00097137
      qcg = 0.0
      do k = 1, nzs
        tso(k) = 256.0 + 1.6 * real(k - 1)
      end do
    case (3)
      ! Fractional snow cover with a damped Egglston limit and half-rain:
      ! snowfrac, meltfactor and rainf all leave unity together.
      snowfrac = 0.6
      meltfactor = 0.4
      rainf = 0.5
      prcpms = 3.0e-6
      rhosn = 250.0
      snwe = 0.05
      tabs = 276.0
      qvatm = 0.00500
      glw = 310.0
      gsw = 70.0
      rnet = 25.0
      qkms = 0.012
      tkms = 0.010
      soilt = 273.10
      soilt1 = 272.60
      qvg = 0.0039961
      qsg = 0.0039961
      qcg = 0.0
      do k = 1, nzs
        tso(k) = 271.0 + 0.05 * real(k - 1)
      end do
    case (4)
      ! Blended thin pack entered with ilnb = 2, which selects the :4411
      ! two-layer tsnav form even though the pack itself is blended.
      ilnb = 2
      rhosn = 150.0
      snwe = 0.004
      tabs = 253.0
      qvatm = 0.0006
      glw = 180.0
      gsw = 20.0
      rnet = -40.0
      qkms = 0.009
      tkms = 0.008
      soilt = 252.0
      soilt1 = 252.0
      qvg = 0.00060441
      qsg = 0.00060441
      qcg = 0.0
      do k = 1, nzs
        tso(k) = 251.0 + 2.0 * real(k - 1)
      end do
    case (5)
      ! No snow at all on entry: snhei = 0 selects the sublimated-pack
      ! coefficients, snflx = d9sn*(soilt-tsob) leaves the incoming s in
      ! place, and the bare sea-ice surface is restored.
      rhosn = 250.0
      snwe = 0.0
      s = 12.5
      dew = 0.25
      tsnav = -3.5
      tabs = 269.0
      qvatm = 0.0002
      glw = 240.0
      gsw = 40.0
      rnet = -8.0
      qkms = 0.050
      tkms = 0.020
      soilt = 271.0
      soilt1 = 271.0
      qvg = 0.0008
      qsg = 0.0010
      qcg = 0.0
      do k = 1, nzs
        tso(k) = 270.5 + 0.10 * real(k - 1)
      end do
    case (6)
      ! 0.16 m pack at 400 kg/m^3: deltsn = 0.125 and snth = 0.025, so the
      ! pack clears deltsn+snth by less than snth and :3951 halves deltsn.
      rhosn = 400.0
      snwe = 0.064
      tabs = 250.0
      qvatm = 0.0005
      glw = 170.0
      gsw = 15.0
      rnet = -45.0
      qkms = 0.008
      tkms = 0.007
      soilt = 249.0
      soilt1 = 254.0
      qvg = 0.00046
      qsg = 0.00046
      qcg = 0.0
      do k = 1, nzs
        tso(k) = 256.0 + 1.5 * real(k - 1)
      end do
    case (7)
      ! 0.008 m pack: melting runs but the pack is below the 1 cm depth at
      ! which Koren retained liquid is kept, so rsm stays zero.
      rhosn = 250.0
      snwe = 0.002
      tabs = 278.0
      qvatm = 0.00500
      glw = 330.0
      gsw = 80.0
      rnet = 60.0
      qkms = 0.020
      tkms = 0.020
      soilt = 273.10
      soilt1 = 273.10
      qvg = 0.0039961
      qsg = 0.0039961
      qcg = 0.0
      do k = 1, nzs
        tso(k) = 271.2 + 0.02 * real(k - 1)
      end do
    case (8)
      ! Melting under dry air: the melt block takes its evaporating side.
      rhosn = 250.0
      snwe = 0.05
      tabs = 284.0
      qvatm = 0.0010
      glw = 320.0
      gsw = 90.0
      rnet = 150.0
      qkms = 0.008
      tkms = 0.014
      soilt = 273.10
      soilt1 = 272.60
      qvg = 0.0039961
      qsg = 0.0039961
      qcg = 0.0
      do k = 1, nzs
        tso(k) = 271.0 + 0.05 * real(k - 1)
      end do
    case (9)
      ! A latent heat of vaporisation away from module_model_constants XLV,
      ! so the xlvm = xlv + xlmelt sum at :3932 is observable.
      xlv = 2.4e6
      rhosn = 300.0
      snwe = 0.10
      newsnow = 0.001
      rhonewsn = 120.0
      tabs = 248.0
      qvatm = 0.0004
      glw = 165.0
      gsw = 15.0
      rnet = -50.0
      qkms = 0.007
      tkms = 0.006
      soilt = 247.0
      soilt1 = 252.0
      qvg = 0.00036879
      qsg = 0.00036879
      qcg = 0.0
      do k = 1, nzs
        tso(k) = 255.0 + 1.8 * real(k - 1)
      end do
    case (10)
      ! snowh has drifted above snow*1.e3/rhosn: with the drifted 0.30 m the
      ! :3951 deltsn halving does not fire, with the rebuilt 0.16 m it does.
      rhosn = 400.0
      snwe = 0.064
      tabs = 250.0
      qvatm = 0.0005
      glw = 170.0
      gsw = 15.0
      rnet = -45.0
      qkms = 0.008
      tkms = 0.007
      soilt = 249.0
      soilt1 = 254.0
      qvg = 0.00046
      qsg = 0.00046
      qcg = 0.0
      do k = 1, nzs
        tso(k) = 256.0 + 1.5 * real(k - 1)
      end do
    end select

    snhei = snwe * 1.0e3 / rhosn
    if (n == 10) snhei = 0.30
    snhei_crit = 0.01601 * rhowater / rhosn

    ! --- sea-ice column properties, exactly as lsmruc builds them ---
    do k = 1, nzs
      tice(k) = tso(k) - 273.15
      rhosice(k) = 917.6 / (1 - 0.000165 * tice(k))
      cice = 2115.85 + 7.7948 * tice(k)
      capice(k) = cice * rhosice(k)
      thdifice(k) = 2.260872 / capice(k)
    end do
    ! ---------------------------------------------------------------

    snwe_before = snwe
    snhei_before = snhei
    rhosn_before = rhosn
    emiss_before = emiss
    alb_before = alb
    znt_before = znt
    tso_before = tso
    soilt_before = soilt
    soilt1_before = soilt1
    if (n /= 5) tsnav = soilt - 273.15
    tsnav_before = tsnav
    dew_before = dew
    qvg_before = qvg
    qsg_before = qsg
    qcg_before = qcg
    snom_before = snom
    s_before = s
    ilnb_before = ilnb
    if (myj) then
      myjout = 1
    else
      myjout = 0
    end if

    call snowseaice(1, n, isoil, delt, ktau, conflx, nzs, nddzs, &
        meltfactor, rhonewsn, snhei_crit, iland, prcpms, rainf, newsnow, &
        snhei, snwe, snowfrac, rhosn, patm, qvatm, qcatm, glw, gsw, &
        emiss, rnet, qkms, tkms, rho, myj, alb, znt, tice, rhosice, &
        capice, thdifice, zsmain, zshalf, dtdzs, dtdzs2, tbq, xlv, cp, &
        rovcp, cw, stbolt, tabs, ilnb, snweprint, snheiprint, rsm, tso, &
        dew, soilt, soilt1, tsnav, qvg, qsg, qcg, smelt, snoh, snflx, &
        snom, eeta, qfx, hfx, s, sublim, prcpl, fltot)

    do k = 1, nzs
      write(unit, '(*(g0,:,","))') trim(names(n)), k, delt, ktau, conflx, &
          iland, isoil, myjout, meltfactor, rhonewsn, snhei_crit, prcpms, &
          rainf, newsnow, snhei_before, snwe_before, snowfrac, &
          rhosn_before, patm, qvatm, qcatm, glw, gsw, emiss_before, rnet, &
          qkms, tkms, rho, alb_before, znt_before, xlv, cp, rovcp, cw, &
          stbolt, tabs, zsmain(k), zshalf(k), tice(k), rhosice(k), &
          capice(k), thdifice(k), tso_before(k), tso(k), soilt_before, &
          soilt, soilt1_before, soilt1, tsnav_before, tsnav, qvg_before, &
          qvg, qsg_before, qsg, qcg_before, qcg, dew_before, dew, snwe, &
          snhei, rhosn, emiss, alb, znt, ilnb_before, ilnb, snweprint, &
          snheiprint, rsm, smelt, snoh, snflx, snom_before, snom, eeta, &
          qfx, hfx, s_before, s, sublim, prcpl, fltot
    end do
  end do
  close(unit)
end program run_ruc_snowseaice_contract_oracle
