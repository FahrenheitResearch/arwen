program run_mynn_tendencies_oracle
  ! Pins the unmodified WRF v4.6.1 mynn_tendencies under the mass-flux-free
  ! identity: bl_mynn_edmf=0, bl_mynn_edmf_mom=0, bl_mynn_mixqt=0,
  ! bl_mynn_cloudmix=1, bl_mynn_mixscalars=0, FLAG_QC/FLAG_QI true and
  ! FLAG_QS/FLAG_QNC/FLAG_QNI/FLAG_QNWFA/FLAG_QNIFA/FLAG_QNBCA/FLAG_OZONE
  ! false.  Every s_aw*, sd_aw*, sub_* and det_* argument is handed in as
  ! zero, which is exactly what the driver supplies before DMP_mf exists.
  !
  ! dfm, dfh, dfq, tcd and qcd come from a real mym_turbulence call so the
  ! diffusivities are self-consistent; diss_heat reproduces the driver block
  ! at module_bl_mynn.F:1223-1233 (dheat_opt=1).  Both are intent(in) to
  ! mynn_tendencies, so the CSV records them as inputs and the port only has
  ! to consume them.
  !
  ! retrieve_exchange_coeffs is pinned in the same rows because it consumes
  ! the same dfm/dfh/dz and is trivially separable.
  use module_model_constants, only: cp, r_d, xlv, xlf, p608
  use module_bl_mynn, only: mym_turbulence, mynn_tendencies, &
      retrieve_exchange_coeffs, b1
  implicit none

  integer, parameter :: ncase = 4, nz = 16, kts = 1, kte = nz
  character(len=32), parameter :: names(ncase) = [character(len=32) :: &
      'stable', 'convective', 'cloudy', 'depleted_moisture']
  character(len=1024) :: output_path
  integer :: c, k, unit
  real :: dz(nz), zw(nz+1), rho(nz), u(nz), v(nz), th(nz), tk(nz)
  real :: qv(nz), qc(nz), qi(nz), qs(nz), qnc(nz), qni(nz)
  real :: p(nz), exner(nz), thl(nz), sqv(nz), sqc(nz), sqi(nz), sqs(nz)
  real :: sqw(nz), qnwfa(nz), qnifa(nz), qnbca(nz), ozone(nz)
  real :: thl_before(nz), sqv_before(nz), sqc_before(nz), sqi_before(nz)
  real :: sqs_before(nz), sqw_before(nz)
  real :: tsq(nz), qsq(nz), cov(nz), tcd(nz), qcd(nz)
  real :: dfm(nz), dfh(nz), dfq(nz), cldfra_bl1d(nz), diss_heat(nz)
  real :: du(nz), dv(nz), dth(nz), dqv(nz), dqc(nz), dqi(nz), dqs(nz)
  real :: dqnc(nz), dqni(nz), dqnwfa(nz), dqnifa(nz), dqnbca(nz)
  real :: dozone(nz), k_m(nz), k_h(nz)
  real :: s_aw(nz+1), s_awthl(nz+1), s_awqt(nz+1), s_awqv(nz+1)
  real :: s_awqc(nz+1), s_awu(nz+1), s_awv(nz+1), s_awqnc(nz+1)
  real :: s_awqni(nz+1), s_awqnwfa(nz+1), s_awqnifa(nz+1), s_awqnbca(nz+1)
  real :: sd_aw(nz+1), sd_awthl(nz+1), sd_awqt(nz+1), sd_awqv(nz+1)
  real :: sd_awqc(nz+1), sd_awu(nz+1), sd_awv(nz+1)
  real :: sub_thl(nz), sub_sqv(nz), sub_u(nz), sub_v(nz)
  real :: det_thl(nz), det_sqv(nz), det_sqc(nz), det_u(nz), det_v(nz)
  ! mym_turbulence workspace
  real :: thetav(nz), qke(nz), vt(nz), vq(nz), theta(nz)
  real :: sh(nz), sm(nz), el(nz), pdk(nz), pdt(nz), pdq(nz), pdc(nz)
  real :: qwt(nz), qshear(nz), qbuoy(nz), qdiss(nz)
  real :: cldfra(nz), edmf_w(nz), edmf_a(nz), tkeprodtd(nz), rstoch(nz)
  real :: xland, closure, dx, rmo, fltv, zi, psig_bl, psig_shcu
  real :: delt, psfc, ust, wspd, uoce, voce, flt, flq, flqv, flqc
  real :: xlvcp, xlscp, zmid, temperature, saturation_mix, relhum
  logical, parameter :: flag_qc = .true., flag_qi = .true.
  logical, parameter :: flag_qs = .false., flag_qnc = .false.
  logical, parameter :: flag_qni = .false., flag_qnwfa = .false.
  logical, parameter :: flag_qnifa = .false., flag_qnbca = .false.
  logical, parameter :: flag_ozone = .false.
  integer, parameter :: bl_mynn_cloudmix = 1, bl_mynn_mixqt = 0
  integer, parameter :: bl_mynn_edmf = 0, bl_mynn_edmf_mom = 0
  integer, parameter :: bl_mynn_mixscalars = 0

  call get_command_argument(1, output_path)
  if (len_trim(output_path) == 0) then
    write(*, '(A)') 'usage: run_tendencies OUTPUT.csv'
    error stop 2
  end if
  xlvcp = xlv / cp
  xlscp = (xlv + xlf) / cp
  open(newunit=unit, file=trim(output_path), status='replace', action='write')
  write(unit, '(A)') 'case,k,delt,psfc,ust,wspd,uoce,voce,flt,flq,flqv,' // &
      'flqc,dz,rho,u,v,th,tk,qv,qc,qi,qs,qnc,qni,p,exner,thl_before,' // &
      'sqv_before,sqc_before,sqi_before,sqs_before,sqw_before,qnwfa,' // &
      'qnifa,qnbca,ozone,tsq,qsq,cov,tcd,qcd,dfm,dfh,dfq,cldfra_bl1d,' // &
      'diss_heat,s_aw,s_aw_next,s_awthl,s_awthl_next,s_awqt,' // &
      's_awqt_next,s_awqv,s_awqv_next,s_awqc,s_awqc_next,s_awu,' // &
      's_awu_next,s_awv,s_awv_next,s_awqnc,s_awqnc_next,s_awqni,' // &
      's_awqni_next,s_awqnwfa,s_awqnwfa_next,s_awqnifa,' // &
      's_awqnifa_next,s_awqnbca,s_awqnbca_next,sd_aw,sd_aw_next,' // &
      'sd_awthl,sd_awthl_next,sd_awqt,sd_awqt_next,sd_awqv,' // &
      'sd_awqv_next,sd_awqc,sd_awqc_next,sd_awu,sd_awu_next,sd_awv,' // &
      'sd_awv_next,sub_thl,sub_sqv,sub_u,sub_v,det_thl,det_sqv,' // &
      'det_sqc,det_u,det_v,du,dv,dth,dqv,dqc,dqi,dqs,dqnc,dqni,' // &
      'dqnwfa,dqnifa,dqnbca,dozone,thl_after,k_m,k_h'

  do c = 1, ncase
    ! ---- geometry, thermodynamic base state ---------------------------
    zw(1) = 0.0
    do k = 1, nz
      dz(k) = 60.0 + 8.0 * real(k - 1)
      zw(k+1) = zw(k) + dz(k)
    end do
    psfc = 100000.0
    delt = 60.0
    xland = 1.0
    closure = 2.6
    dx = 3000.0
    uoce = 0.0
    voce = 0.0
    flqc = 0.0
    do k = 1, nz
      zmid = zw(k) + 0.5 * dz(k)
      p(k) = psfc * exp(-zmid / 8000.0)
      exner(k) = (p(k) / 100000.0) ** (r_d / cp)
      rho(k) = 1.18 * exp(-zw(k) / 8000.0)
      temperature = 291.0 - 0.0061 * zmid
      th(k) = temperature / exner(k)
      u(k) = 5.0 + 0.8 * real(k - 1)
      v(k) = 1.0 - 0.12 * real(k - 1)
      qc(k) = 0.0
      qi(k) = 0.0
      qs(k) = 0.0
      qnc(k) = 0.0
      qni(k) = 0.0
      qnwfa(k) = 0.0
      qnifa(k) = 0.0
      qnbca(k) = 0.0
      ozone(k) = 0.0
      cldfra(k) = 0.0
      cldfra_bl1d(k) = 0.0
      edmf_w(k) = 0.0
      edmf_a(k) = 0.0
      tkeprodtd(k) = 0.0
      rstoch(k) = 0.0
      vt(k) = 0.0
      vq(k) = 0.0
      qwt(k) = 0.0
      qshear(k) = 0.0
      qbuoy(k) = 0.0
      qdiss(k) = 0.0
      ! Saturation mixing ratio from the Tetens form, kept local to the
      ! harness so no ported routine is involved in building the state.
      saturation_mix = 0.622 * 611.2 * exp(17.67 * (temperature - 273.15) &
          / (temperature - 29.65)) / max(p(k) - 611.2, 1.0)
      relhum = 0.62
      qv(k) = relhum * saturation_mix
      qke(k) = max(1.7 - 0.11 * real(k - 1), 0.06)
      tsq(k) = 1.0e-3 * exp(-real(k - 1) / 5.0)
      qsq(k) = 2.0e-7 * exp(-real(k - 1) / 5.0)
      cov(k) = -8.0e-6 * exp(-real(k - 1) / 5.0)
    end do
    ust = 0.28
    flt = -0.015
    fltv = -0.02
    flqv = -1.0e-5
    rmo = 0.008
    zi = 400.0
    psig_bl = 1.0
    psig_shcu = 1.0

    select case (c)
    case (2)
      ! Convective land: positive surface fluxes, deeper mixed layer.
      ust = 0.45
      flt = 0.08
      fltv = 0.11
      flqv = 6.0e-5
      rmo = -0.006
      zi = 700.0
      psig_bl = 0.94
      psig_shcu = 0.90
      do k = 1, nz
        zmid = zw(k) + 0.5 * dz(k)
        temperature = 297.0 - 0.0098 * min(zmid, 700.0) &
            - 0.0045 * max(zmid - 700.0, 0.0)
        th(k) = temperature / exner(k)
        saturation_mix = 0.622 * 611.2 * exp(17.67 * (temperature - 273.15) &
            / (temperature - 29.65)) / max(p(k) - 611.2, 1.0)
        qv(k) = 0.78 * saturation_mix
        qke(k) = max(3.8 - 0.24 * real(k - 1), 0.12)
      end do
    case (3)
      ! Cloudy marine column: resolved liquid in the middle of the column,
      ! ice above it, a nonzero ocean current, and a nonzero flqc.  The WRF
      ! driver hard-wires flqc = 0.0 (module_bl_mynn.F:1069), so the
      ! flqc term is unreachable through the driver; exercising it here is
      ! what validates the transcription of that term.
      xland = 2.0
      ust = 0.38
      flt = 0.045
      fltv = 0.06
      flqv = 4.0e-5
      flqc = 2.0e-7
      uoce = 0.35
      voce = -0.20
      rmo = -0.003
      zi = 620.0
      psig_bl = 0.86
      psig_shcu = 0.80
      do k = 1, nz
        zmid = zw(k) + 0.5 * dz(k)
        temperature = 288.0 - 0.0072 * zmid
        th(k) = temperature / exner(k)
        saturation_mix = 0.622 * 611.2 * exp(17.67 * (temperature - 273.15) &
            / (temperature - 29.65)) / max(p(k) - 611.2, 1.0)
        qv(k) = 0.93 * saturation_mix
        qke(k) = max(2.4 - 0.13 * real(k - 1), 0.09)
        cldfra(k) = min(0.09 * real(k - 1), 0.70)
        cldfra_bl1d(k) = cldfra(k)
        if (zmid > 250.0 .and. zmid < 900.0) then
          qc(k) = max(2.4e-4 - abs(zmid - 550.0) * 3.0e-7, 5.0e-6)
        end if
        if (zmid > 950.0) then
          qi(k) = max(4.0e-5 - (zmid - 1200.0) * 8.0e-9, 3.0e-6)
        end if
      end do
    case (4)
      ! Depleted-moisture column.  qcd is driven strongly negative at the
      ! surface level and at k=9 so the vapour solve returns negative
      ! specific humidity there.  That is the only way to reach
      ! moisture_check's borrow-from-below repair (k=9) and its column-wide
      ! proportional extraction (k=1, where no lower layer exists), which
      ! are trajectory-affecting and cannot be replaced by a clip.
      ust = 0.33
      flt = 0.02
      fltv = 0.03
      flqv = -8.0e-5
      rmo = -0.001
      zi = 500.0
      psig_bl = 0.90
      psig_shcu = 0.88
      do k = 1, nz
        zmid = zw(k) + 0.5 * dz(k)
        temperature = 285.0 - 0.0065 * zmid
        th(k) = temperature / exner(k)
        saturation_mix = 0.622 * 611.2 * exp(17.67 * (temperature - 273.15) &
            / (temperature - 29.65)) / max(p(k) - 611.2, 1.0)
        qv(k) = 0.45 * saturation_mix
        qke(k) = max(2.0 - 0.12 * real(k - 1), 0.07)
        if (zmid > 400.0 .and. zmid < 800.0) then
          qc(k) = 3.0e-6
        end if
      end do
      ! A small negative cloud-ice value, which is what a microphysics
      ! scheme can hand back, so the qi deficit branch of moisture_check
      ! is reached.  Nothing else in this fixture drives sqi negative.
      qi(12) = -1.0e-7
    end select

    do k = 1, nz
      tk(k) = th(k) * exner(k)
      sqv(k) = qv(k)
      sqc(k) = qc(k)
      sqi(k) = qi(k)
      sqs(k) = 0.0
      sqw(k) = sqv(k) + sqc(k) + sqi(k)
      thl(k) = th(k) - xlvcp / exner(k) * sqc(k) &
          - xlscp / exner(k) * sqi(k)
      thetav(k) = th(k) * (1.0 + p608 * sqv(k))
      theta(k) = th(k)
    end do
    flq = flqv + flqc
    wspd = max(sqrt(u(kts)**2 + v(kts)**2), 1.0)

    ! ---- diffusivities and countergradient terms ----------------------
    call mym_turbulence(kts, kte, xland, closure, dz, dx, zw, u, v, &
        thl, thetav, sqc, sqw, qke, tsq, qsq, cov, vt, vq, rmo, flt, &
        fltv, flq, zi, theta, sh, sm, el, dfm, dfh, dfq, tcd, qcd, &
        pdk, pdt, pdq, pdc, qwt, qshear, qbuoy, qdiss, 0, psig_bl, &
        psig_shcu, cldfra, 1, edmf_w, edmf_a, tkeprodtd, 0, rstoch)

    ! Dissipative heating, dheat_opt = 1 (module_bl_mynn.F:1223-1233).
    do k = kts, kte-1
      diss_heat(k) = min(max(1.0 * (qke(k)**1.5) &
          / (b1 * max(0.5 * (el(k) + el(k+1)), 1.)) / cp, 0.0), 0.002)
      diss_heat(k) = diss_heat(k) * exp(-10000. / max(p(k), 1.))
    end do
    diss_heat(kte) = 0.

    if (c == 3) then
      ! Push the cloud-water solve negative at one level so the qc deficit
      ! branch of moisture_check condenses vapour back into liquid and
      ! feeds the theta tendency.
      qcd(6) = qcd(6) - 6.0e-6
    end if
    if (c == 4) then
      qcd(kts) = qcd(kts) - 5.0e-5
      qcd(9) = qcd(9) - 5.5e-5
    end if

    ! ---- zero mass flux, subsidence and detrainment --------------------
    s_aw = 0.0
    s_awthl = 0.0
    s_awqt = 0.0
    s_awqv = 0.0
    s_awqc = 0.0
    s_awu = 0.0
    s_awv = 0.0
    s_awqnc = 0.0
    s_awqni = 0.0
    s_awqnwfa = 0.0
    s_awqnifa = 0.0
    s_awqnbca = 0.0
    sd_aw = 0.0
    sd_awthl = 0.0
    sd_awqt = 0.0
    sd_awqv = 0.0
    sd_awqc = 0.0
    sd_awu = 0.0
    sd_awv = 0.0
    sub_thl = 0.0
    sub_sqv = 0.0
    sub_u = 0.0
    sub_v = 0.0
    det_thl = 0.0
    det_sqv = 0.0
    det_sqc = 0.0
    det_u = 0.0
    det_v = 0.0

    thl_before = thl
    sqv_before = sqv
    sqc_before = sqc
    sqi_before = sqi
    sqs_before = sqs
    sqw_before = sqw

    call retrieve_exchange_coeffs(kts, kte, dfm, dfh, dz, k_m, k_h)

    call mynn_tendencies(kts, kte, 1,                                    &
        delt, dz, rho,                                                   &
        u, v, th, tk, qv, qc, qi, qs, qnc, qni,                          &
        psfc, p, exner,                                                  &
        thl, sqv, sqc, sqi, sqs, sqw,                                    &
        qnwfa, qnifa, qnbca, ozone,                                      &
        ust, flt, flq, flqv, flqc, wspd,                                 &
        uoce, voce,                                                      &
        tsq, qsq, cov,                                                   &
        tcd, qcd,                                                        &
        dfm, dfh, dfq,                                                   &
        du, dv, dth, dqv, dqc, dqi, dqs, dqnc, dqni,                     &
        dqnwfa, dqnifa, dqnbca, dozone,                                  &
        diss_heat,                                                       &
        s_aw, s_awthl, s_awqt, s_awqv, s_awqc,                           &
        s_awu, s_awv,                                                    &
        s_awqnc, s_awqni,                                                &
        s_awqnwfa, s_awqnifa, s_awqnbca,                                 &
        sd_aw, sd_awthl, sd_awqt, sd_awqv,                               &
        sd_awqc, sd_awu, sd_awv,                                         &
        sub_thl, sub_sqv,                                                &
        sub_u, sub_v,                                                    &
        det_thl, det_sqv, det_sqc,                                       &
        det_u, det_v,                                                    &
        flag_qc, flag_qi, flag_qnc, flag_qni,                            &
        flag_qs,                                                         &
        flag_qnwfa, flag_qnifa, flag_qnbca,                              &
        flag_ozone,                                                      &
        cldfra_bl1d,                                                     &
        bl_mynn_cloudmix,                                                &
        bl_mynn_mixqt,                                                   &
        bl_mynn_edmf,                                                    &
        bl_mynn_edmf_mom,                                                &
        bl_mynn_mixscalars)

    do k = 1, nz
      write(unit, '(*(g0,:,","))') trim(names(c)), k, delt, psfc, ust, &
          wspd, uoce, voce, flt, flq, flqv, flqc, dz(k), rho(k), u(k), &
          v(k), th(k), tk(k), qv(k), qc(k), qi(k), qs(k), qnc(k), &
          qni(k), p(k), exner(k), thl_before(k), sqv_before(k), &
          sqc_before(k), sqi_before(k), sqs_before(k), sqw_before(k), &
          qnwfa(k), qnifa(k), qnbca(k), ozone(k), tsq(k), qsq(k), &
          cov(k), tcd(k), qcd(k), dfm(k), dfh(k), dfq(k), &
          cldfra_bl1d(k), diss_heat(k), &
          s_aw(k), s_aw(k+1), s_awthl(k), s_awthl(k+1), &
          s_awqt(k), s_awqt(k+1), s_awqv(k), s_awqv(k+1), &
          s_awqc(k), s_awqc(k+1), s_awu(k), s_awu(k+1), &
          s_awv(k), s_awv(k+1), s_awqnc(k), s_awqnc(k+1), &
          s_awqni(k), s_awqni(k+1), s_awqnwfa(k), s_awqnwfa(k+1), &
          s_awqnifa(k), s_awqnifa(k+1), s_awqnbca(k), s_awqnbca(k+1), &
          sd_aw(k), sd_aw(k+1), sd_awthl(k), sd_awthl(k+1), &
          sd_awqt(k), sd_awqt(k+1), sd_awqv(k), sd_awqv(k+1), &
          sd_awqc(k), sd_awqc(k+1), sd_awu(k), sd_awu(k+1), &
          sd_awv(k), sd_awv(k+1), &
          sub_thl(k), sub_sqv(k), sub_u(k), sub_v(k), &
          det_thl(k), det_sqv(k), det_sqc(k), det_u(k), det_v(k), &
          du(k), dv(k), dth(k), dqv(k), dqc(k), dqi(k), dqs(k), &
          dqnc(k), dqni(k), dqnwfa(k), dqnifa(k), dqnbca(k), &
          dozone(k), thl(k), k_m(k), k_h(k)
    end do
  end do
  close(unit)
end program run_mynn_tendencies_oracle
