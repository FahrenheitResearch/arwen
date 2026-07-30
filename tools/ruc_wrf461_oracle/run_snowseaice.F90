program run_ruc_snowseaice_oracle
  ! Drives the unmodified WRF v4.6.1 module_sf_ruclsm::snowseaice, the snow
  ! energy budget and snow/sea-ice heat diffusion solve.  The sea-ice column
  ! properties (tice, rhosice, capice, thdifice) are built from the ice
  ! temperature profile with the same Zubov relations lsmruc uses, and every
  ! one of them is written to the CSV, so each row is independently
  ! reproducible from the file alone.
  use module_sf_ruclsm, only: snowseaice
  implicit none

  integer, parameter :: ncase = 5, nzs = 9, nddzs = 14
  character(len=24), parameter :: names(ncase) = [character(len=24) :: &
      'blended_thin_on_ice', 'one_layer_on_ice', 'two_layer_deep_on_ice', &
      'melting_on_ice', 'sublimating_on_ice']
  real, parameter :: zsmain(nzs) = [0.0, 0.01, 0.04, 0.10, 0.30, &
      0.60, 1.00, 1.60, 3.00]
  real, parameter :: xlv = 2.5e6, cp = 1004.5, r_d = 287.0
  real, parameter :: cw = 4.183e6, stbolt = 5.67051e-8
  real, parameter :: rhowater = 1000.0
  character(len=1024) :: output_path
  integer :: n, k, k1, k2, unit, ktau, iland, isoil, ilnb, ilnb_before
  real :: delt, xgeom, cq, evs, eis, r61, conflx, cice
  real :: meltfactor, rhonewsn, snhei_crit, prcpms, rainf, newsnow
  real :: snhei, snwe, snowfrac, rhosn, patm, qvatm, qcatm
  real :: glw, gsw, emiss, rnet, qkms, tkms, rho, alb, znt
  real :: rovcp, tabs, snweprint, snheiprint, rsm
  real :: dew, soilt, soilt1, tsnav, qvg, qsg, qcg
  real :: smelt, snoh, snflx, snom, eeta, qfx, hfx, s, sublim
  real :: prcpl, fltot
  real :: snwe_before, snhei_before, rhosn_before, emiss_before
  real :: alb_before, znt_before, soilt_before, soilt1_before
  real :: tsnav_before, dew_before, qvg_before, qsg_before, qcg_before
  real :: snom_before, s_before
  real :: zshalf(nzs), dtdzs(nddzs), dtdzs2(nzs), tbq(5001)
  real :: tice(nzs), rhosice(nzs), capice(nzs), thdifice(nzs)
  real :: tso(nzs), tso_before(nzs)

  call get_command_argument(1, output_path)
  if (len_trim(output_path) == 0) then
    write(*, '(A)') 'usage: run_snowseaice OUTPUT.csv'
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
      ! 0.027 m of light snow with snth = 0.067: the pack is blended into
      ! the top sea-ice layer.
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
    case (2)
      ! 0.12 m pack, snth = 0.04, deltsn = 0.20: single snow layer.
      rhosn = 250.0
      snwe = 0.03
      newsnow = 0.001
      rhonewsn = 120.0
      tabs = 258.0
      qvatm = 0.0010
      glw = 200.0
      gsw = 35.0
      rnet = -25.0
      qkms = 0.010
      tkms = 0.009
      snowfrac = 0.9
      soilt = 257.0
      soilt1 = 257.0
      qvg = 0.00097137
      qsg = 0.00097137
      qcg = 0.0
      do k = 1, nzs
        tso(k) = 256.0 + 1.6 * real(k - 1)
      end do
    case (3)
      ! 0.33 m pack over deltsn + snth = 0.20: two snow layers with an
      ! interior soilt1.
      rhosn = 300.0
      snwe = 0.10
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
    case (4)
      ! Light rain on a ripe 0.20 m pack over sea ice: soilt clears
      ! 273.15 and the nmelt branch runs while the pack survives.
      rhosn = 250.0
      snwe = 0.05
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
      soilt1 = 272.60
      qvg = 0.0039961
      qsg = 0.0039961
      qcg = 0.0
      do k = 1, nzs
        tso(k) = 271.0 + 0.05 * real(k - 1)
      end do
    case (5)
      ! A trace of snow under dry air: the potential sublimation over one
      ! step exceeds snwe, so beta < 1, the pack vanishes entirely and the
      ! bare sea-ice albedo/roughness/emissivity are restored.
      rhosn = 150.0
      snwe = 1.0e-9
      tabs = 269.5
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
    end select

    snhei = snwe * 1.0e3 / rhosn
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
    tsnav = soilt - 273.15
    tsnav_before = tsnav
    dew_before = dew
    qvg_before = qvg
    qsg_before = qsg
    qcg_before = qcg
    snom_before = snom
    s_before = s
    ilnb_before = ilnb

    call snowseaice(1, n, isoil, delt, ktau, conflx, nzs, nddzs, &
        meltfactor, rhonewsn, snhei_crit, iland, prcpms, rainf, newsnow, &
        snhei, snwe, snowfrac, rhosn, patm, qvatm, qcatm, glw, gsw, &
        emiss, rnet, qkms, tkms, rho, .false., alb, znt, tice, rhosice, &
        capice, thdifice, zsmain, zshalf, dtdzs, dtdzs2, tbq, xlv, cp, &
        rovcp, cw, stbolt, tabs, ilnb, snweprint, snheiprint, rsm, tso, &
        dew, soilt, soilt1, tsnav, qvg, qsg, qcg, smelt, snoh, snflx, &
        snom, eeta, qfx, hfx, s, sublim, prcpl, fltot)

    do k = 1, nzs
      write(unit, '(*(g0,:,","))') trim(names(n)), k, delt, ktau, conflx, &
          iland, isoil, 0, meltfactor, rhonewsn, snhei_crit, prcpms, &
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
end program run_ruc_snowseaice_oracle
