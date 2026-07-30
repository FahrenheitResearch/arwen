program run_ruc_sice_oracle
  ! Drives the unmodified WRF v4.6.1 module_sf_ruclsm::sice, the snow-free
  ! sea-ice column: heat diffusion through the nine-level ice slab plus the
  ! surface energy balance closed by vilka.
  !
  ! The ice column properties (tice, rhosice, capice, thdifice) are built
  ! from the ice temperature profile with the same Zubov relations lsmruc
  ! uses, exactly as run_snowseaice.F90 does, and every one of them is
  ! written to the CSV so each row is reproducible from the file alone.
  !
  ! The five forced soil arrays (soilmois=1, soiliqw=0, soilice=1,
  ! smfrkeep=1, keepfr=0) are NOT set inside sice: sfctmp writes them
  ! immediately after each of the two call sites (:1871-1877 after the call
  ! at :1849, and :2184-2190 after the call at :2162), identically.  This
  ! harness reproduces that loop after the
  ! call and pins the result, because the gpuwm port folds the same forcing
  ! into ruc_sea_ice_step.
  use module_sf_ruclsm, only: sice, qsn
  implicit none

  integer, parameter :: ncase = 8, nzs = 9, nddzs = 14
  character(len=24), parameter :: names(ncase) = [character(len=24) :: &
      'cold_thick_ice', 'near_melt_cap', 'thin_ice_warm_base', &
      'strong_forcing', 'dew_condensing', 'myj_condensing', &
      'myj_evaporating', 'rain_on_ice']
  logical, parameter :: myj_case(ncase) = [.false., .false., .false., &
      .false., .false., .true., .true., .false.]
  real, parameter :: zsmain(nzs) = [0.0, 0.01, 0.04, 0.10, 0.30, &
      0.60, 1.00, 1.60, 3.00]
  real, parameter :: xlv = 2.5e6, cp = 1004.5, r_d = 287.0
  real, parameter :: cw = 4.183e6, stbolt = 5.67051e-8
  character(len=1024) :: output_path
  integer :: n, k, k1, k2, unit, ktau, iland, isoil, nroot
  real :: delt, xgeom, cq, evs, eis, r61, conflx, cice, pp
  real :: prcpms, rainf, patm, qvatm, qcatm, glw, gsw
  real :: emiss, rnet, qkms, tkms, rho, rovcp, tabs
  real :: dew, soilt, qvg, qsg, qcg
  real :: eeta, qfx, hfx, s, evapl, prcpl, fltot
  real :: dew_before, soilt_before, qvg_before, qsg_before, qcg_before
  real :: soilmois(nzs), soiliqw(nzs), soilice(nzs)
  real :: smfrkeep(nzs), keepfr(nzs)
  real :: zshalf(nzs), dtdzs(nddzs), dtdzs2(nzs), tbq(5001)
  real :: tice(nzs), rhosice(nzs), capice(nzs), thdifice(nzs)
  real :: tso(nzs), tso_before(nzs)
  logical :: myj

  call get_command_argument(1, output_path)
  if (len_trim(output_path) == 0) then
    write(*, '(A)') 'usage: run_sice OUTPUT.csv'
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
  nroot = 4

  open(newunit=unit, file=trim(output_path), status='replace', action='write')
  write(unit, '(A)') 'case,k,delt,ktau,conflx,iland,isoil,nroot,myj,prcpms,rainf,patm,qvatm,qcatm,glw,gsw,emiss,rnet,qkms,tkms,rho,xlv,cp,rovcp,cw,stbolt,tabs,zsmain,zshalf,tice,rhosice,capice,thdifice,tso_before,tso_after,dew_before,dew_after,soilt_before,soilt_after,qvg_before,qvg_after,qsg_before,qsg_after,qcg_before,qcg_after,eeta,qfx,hfx,s,evapl,prcpl,fltot,soilmois_after,soiliqw_after,soilice_after,smfrkeep_after,keepfr_after'

  do n = 1, ncase
    conflx = 40.0
    prcpms = 0.0
    rainf = 0.0
    patm = 0.95
    qcatm = 0.0
    emiss = 0.98
    rho = 1.30
    qcg = 0.0
    dew = 0.0
    eeta = 0.0
    qfx = 0.0
    hfx = 0.0
    s = 0.0
    evapl = 0.0
    prcpl = 0.0
    fltot = 0.0
    myj = myj_case(n)

    select case (n)
    case (1)
      ! Deep cold ice under a colder atmosphere: net radiative loss, the
      ! whole column stays far below the 271.4 K cap.
      tabs = 238.0
      qvatm = 0.0002
      glw = 170.0
      gsw = 5.0
      rnet = -45.0
      qkms = 0.008
      tkms = 0.007
      rho = 1.35
      soilt = 241.0
      do k = 1, nzs
        tso(k) = 240.0 + 3.5 * real(k - 1)
      end do
    case (2)
      ! Ice already at the melting point with strong absorbed solar: the
      ! vilka skin solution overshoots and the min(271.4,...) cap binds on
      ! tso(1) and on the interior levels.
      tabs = 276.0
      qvatm = 0.0050
      glw = 310.0
      gsw = 250.0
      rnet = 180.0
      qkms = 0.020
      tkms = 0.018
      soilt = 271.30
      do k = 1, nzs
        tso(k) = 271.30 + 0.01 * real(k - 1)
      end do
    case (3)
      ! Thin ice: the base sits at the ocean freezing point while the skin
      ! is 13 K colder, so the conductive term dominates the balance.
      tabs = 255.0
      qvatm = 0.0008
      glw = 195.0
      gsw = 25.0
      rnet = -20.0
      qkms = 0.010
      tkms = 0.009
      soilt = 258.0
      do k = 1, nzs
        tso(k) = 258.0 + (271.35 - 258.0) * real(k - 1) / real(nzs - 1)
      end do
    case (4)
      ! Strongly forced surface balance: shallow constant-flux layer, high
      ! exchange coefficients, a large positive net radiation and dry air,
      ! so the RUC (non-MYJ) evaporation branch carries a real flux.
      conflx = 20.0
      tabs = 268.0
      qvatm = 0.0005
      glw = 260.0
      gsw = 600.0
      rnet = 400.0
      qkms = 0.050
      tkms = 0.045
      rho = 1.20
      soilt = 262.0
      do k = 1, nzs
        tso(k) = 260.0 + 0.5 * real(k - 1)
      end do
    case (5)
      ! Moist air over cold ice: q1 <= 0, so the condensation branch runs
      ! and dew is diagnosed from the RUC (non-MYJ) flux form.
      tabs = 272.0
      qvatm = 0.0045
      glw = 290.0
      gsw = 40.0
      rnet = 30.0
      qkms = 0.015
      tkms = 0.013
      soilt = 265.0
      do k = 1, nzs
        tso(k) = 265.0 - 0.4 * real(k - 1)
      end do
    case (6)
      ! Same condensation state with myj=.true.: qfx keeps the MYJ moisture
      ! flux while eeta is overwritten by -rho*dew with dew still 0.
      tabs = 272.0
      qvatm = 0.0045
      glw = 290.0
      gsw = 40.0
      rnet = 30.0
      qkms = 0.015
      tkms = 0.013
      soilt = 265.0
      do k = 1, nzs
        tso(k) = 265.0 - 0.4 * real(k - 1)
      end do
    case (7)
      ! Dry air over relatively warm ice with myj=.true.: q1 > 0, so the
      ! MYJ evaporation form is used.
      tabs = 270.0
      qvatm = 0.0005
      glw = 250.0
      gsw = 120.0
      rnet = 55.0
      qkms = 0.018
      tkms = 0.016
      soilt = 270.5
      do k = 1, nzs
        tso(k) = 270.5 - 0.15 * real(k - 1)
      end do
    case (8)
      ! Rain falling on near-melting ice: rainf/prcpms activate the rain
      ! heat capacity terms in tdenom, bb and the storage residual x.
      prcpms = 3.0e-6
      rainf = 1.0
      tabs = 275.0
      qvatm = 0.0048
      glw = 320.0
      gsw = 100.0
      rnet = 60.0
      qkms = 0.012
      tkms = 0.011
      soilt = 270.90
      do k = 1, nzs
        tso(k) = 270.80 + 0.05 * real(k - 1)
      end do
    end select

    ! Surface humidity is saturation over sea ice, exactly as lsmruc seeds
    ! it at :504/:841 through qsn.
    pp = patm * 1.e3
    qsg = qsn(soilt, tbq) / pp
    qvg = qsg

    ! --- sea-ice column properties, exactly as lsmruc builds them ---
    do k = 1, nzs
      tice(k) = tso(k) - 273.15
      rhosice(k) = 917.6 / (1 - 0.000165 * tice(k))
      cice = 2115.85 + 7.7948 * tice(k)
      capice(k) = cice * rhosice(k)
      thdifice(k) = 2.260872 / capice(k)
    end do
    ! ---------------------------------------------------------------

    tso_before = tso
    dew_before = dew
    soilt_before = soilt
    qvg_before = qvg
    qsg_before = qsg
    qcg_before = qcg

    call sice(1, n, iland, isoil, delt, ktau, conflx, nzs, nddzs, nroot, &
        prcpms, rainf, patm, qvatm, qcatm, glw, gsw, &
        emiss, rnet, qkms, tkms, rho, myj, &
        tice, rhosice, capice, thdifice, &
        zsmain, zshalf, dtdzs, dtdzs2, tbq, &
        xlv, cp, rovcp, cw, stbolt, tabs, &
        tso, dew, soilt, qvg, qsg, qcg, &
        eeta, qfx, hfx, s, evapl, prcpl, fltot)

    ! --- sfctmp's post-call forcing for the sea-ice column ---
    do k = 1, nzs
      soilmois(k) = 1.
      soiliqw(k) = 0.
      soilice(k) = 1.
      smfrkeep(k) = 1.
      keepfr(k) = 0.
    end do
    ! ---------------------------------------------------------

    do k = 1, nzs
      write(unit, '(*(g0,:,","))') trim(names(n)), k, delt, ktau, conflx, &
          iland, isoil, nroot, merge(1, 0, myj), prcpms, rainf, patm, &
          qvatm, qcatm, glw, gsw, emiss, rnet, qkms, tkms, rho, xlv, cp, &
          rovcp, cw, stbolt, tabs, zsmain(k), zshalf(k), tice(k), &
          rhosice(k), capice(k), thdifice(k), tso_before(k), tso(k), &
          dew_before, dew, soilt_before, soilt, qvg_before, qvg, &
          qsg_before, qsg, qcg_before, qcg, eeta, qfx, hfx, s, evapl, &
          prcpl, fltot, soilmois(k), soiliqw(k), soilice(k), &
          smfrkeep(k), keepfr(k)
    end do
  end do
  close(unit)
end program run_ruc_sice_oracle
