program run_mynn_tendencies_mf_oracle
  ! Pins the unmodified WRF v4.6.1 mynn_tendencies (module_bl_mynn.F:4027-5134)
  ! with the mass flux ADMITTED.  The mass-flux arguments are not synthesised:
  ! every s_aw*, sub_* and det_* handed to mynn_tendencies is produced by a real
  ! DMP_mf call on the same column, and dfm/dfh/tcd/qcd come from a real
  ! mym_turbulence call, so this fixture is the coupled arrangement the driver
  ! builds at module_bl_mynn.F:1129-1275 rather than an invented forcing.
  !
  ! Admitted option identity: bl_mynn_cloudmix=1, bl_mynn_mixqt=0,
  ! bl_mynn_edmf=1, bl_mynn_mixscalars=0, FLAG_QC/FLAG_QI true and
  ! FLAG_QS/FLAG_QNC/FLAG_QNI/FLAG_QNWFA/FLAG_QNIFA/FLAG_QNBCA/FLAG_OZONE
  ! false.  bl_mynn_edmf_mom varies by case and is recorded per row.
  !
  ! Two cases are deliberately NOT reachable through mynn_bl_driver and are
  ! labelled as probes.  They exist because the corresponding transcription is
  ! live code in mynn_tendencies that the driver never exercises:
  !   * momentum_off_probe hands nonzero s_awu/s_awv to a bl_mynn_edmf_mom=0
  !     call.  The driver cannot: the same knob is DMP_mf's momentum_opt, so
  !     s_awu/s_awv are already zero whenever onoff is zero
  !     (module_bl_mynn.F:4127-4134).  Without this probe nothing distinguishes
  !     "onoff multiplies the momentum mass flux" from "onoff is ignored".
  !   * downdraft_probe fills sd_aw* with a synthetic downdraft.  The driver
  !     leaves them zero because bl_mynn_edmf_dd is a compile-time 0 at
  !     module_bl_mynn.F:330, yet the sd_aw* terms are added unconditionally in
  !     every scalar system, so their transcription is otherwise untested.
  !   * subsidence_probe fills sub_*/det_*.  DMP_mf computes those only inside
  !     IF (env_subs) blocks (module_bl_mynn.F:6547-6597) and env_subs is a
  !     compile-time .false. at :336, so DMP_mf always returns them zero; the
  !     terms are nevertheless added unconditionally in mynn_tendencies.
  !
  ! retrieve_exchange_coeffs is pinned in the same rows because it consumes the
  ! same dfm/dfh/dz and is trivially separable.
  use module_model_constants, only: cp, r_d, xlv, xlf, p608, karman
  use module_bl_mynn, only: mym_initialize, get_pblh, scale_aware,             &
      mym_condensation, dmp_mf, mym_turbulence, mym_predict,                   &
      mynn_tendencies, retrieve_exchange_coeffs, b1, phim, phih
  implicit none

  integer, parameter :: ncase = 9, nz = 30, kts = 1, kte = nz
  integer, parameter :: nchem = 2
  character(len=32), parameter :: names(ncase) = [character(len=32) :: &
      'land_cumulus', 'water_cumulus', 'deep_plume', 'fine_grid',      &
      'momentum_off', 'momentum_off_probe', 'downdraft_probe',         &
      'moisture_repair', 'subsidence_probe']
  character(len=1024) :: output_path
  integer :: c, k, unit, kpbl, ktop, edmf_mom, momentum_opt, spp_pbl
  logical :: mix_chem

  real :: dz(nz), zw(nz+1), p(nz), rho(nz), u(nz), v(nz), w(nz)
  real :: th(nz), thl(nz), thetav(nz), tk(nz), sqw(nz), sqv(nz), sqc(nz)
  real :: sqi(nz), sqs(nz), exner(nz), qke(nz), qnc(nz), qni(nz)
  real :: qnwfa(nz), qnifa(nz), qnbca(nz), ozone(nz)
  real :: qv1(nz), qc1(nz), qi1(nz), qs1(nz), kzero(nz)
  real :: vt(nz), vq(nz), sgm(nz), rstoch(nz)
  real :: qc_bl(nz), qi_bl(nz), cldfra_bl(nz)
  real :: qc_bl_old(nz), cldfra_bl_old(nz)
  real :: edmf_a(nz), edmf_w(nz), edmf_qt(nz), edmf_thl(nz)
  real :: edmf_ent(nz), edmf_qc(nz)
  real :: s_aw(nz+1), s_awthl(nz+1), s_awqt(nz+1), s_awqv(nz+1)
  real :: s_awqc(nz+1), s_awu(nz+1), s_awv(nz+1), s_awqke(nz+1)
  real :: s_awqnc(nz+1), s_awqni(nz+1), s_awqnwfa(nz+1), s_awqnifa(nz+1)
  real :: s_awqnbca(nz+1)
  real :: sd_aw(nz+1), sd_awthl(nz+1), sd_awqt(nz+1), sd_awqv(nz+1)
  real :: sd_awqc(nz+1), sd_awu(nz+1), sd_awv(nz+1)
  real :: sub_thl(nz), sub_sqv(nz), sub_u(nz), sub_v(nz)
  real :: det_thl(nz), det_sqv(nz), det_sqc(nz), det_u(nz), det_v(nz)
  real :: chem1(nz, nchem), s_awchem(nz+1, nchem)
  ! turbulence / predictor workspace
  real :: sh(nz), sm(nz), el(nz), dfm(nz), dfh(nz), dfq(nz)
  real :: tcd(nz), qcd(nz), pdk(nz), pdt(nz), pdq(nz), pdc(nz)
  real :: qwt(nz), qshear(nz), qbuoy(nz), qdiss(nz), tkeprodtd(nz)
  real :: tsq(nz), qsq(nz), cov(nz), diss_heat(nz)
  ! tendency outputs
  real :: du(nz), dv(nz), dth(nz), dqv(nz), dqc(nz), dqi(nz), dqs(nz)
  real :: dqnc(nz), dqni(nz), dqnwfa(nz), dqnifa(nz), dqnbca(nz)
  real :: dozone(nz), k_m(nz), k_h(nz)
  ! saved pre-call state
  real :: thl_before(nz), sqv_before(nz), sqc_before(nz), sqi_before(nz)
  real :: sqs_before(nz), tcd_in(nz), qcd_in(nz)
  real :: delt, psfc, ust, wspd, uoce, voce, flt, flq, flqv, flqc, fltv
  real :: pblh, dx, xland, ts, th_sfc, psig_bl, psig_shcu, closure
  real :: maxwidth, maxmf, ztop, rmol, zet, pmz, phh, phi_m
  real :: xlvcp, xlscp
  ! profile controls, host-associated by build_profile
  real :: dz1, dzslope, th_ml, dth_ml, inv_dth, up_lapse, qsurf, dq_ml
  real :: dq_up, zinv, ushear, u0, v0, zmid
  logical, parameter :: flag_qc = .true., flag_qi = .true.
  logical, parameter :: flag_qs = .false., flag_qnc = .false.
  logical, parameter :: flag_qni = .false., flag_qnwfa = .false.
  logical, parameter :: flag_qnifa = .false., flag_qnbca = .false.
  logical, parameter :: flag_ozone = .false.
  integer, parameter :: bl_mynn_cloudmix = 1, bl_mynn_mixqt = 0
  integer, parameter :: bl_mynn_edmf = 1, bl_mynn_mixscalars = 0
  integer, parameter :: bl_mynn_edmf_tke = 0, bl_mynn_mixlength = 1
  integer, parameter :: bl_mynn_cloudpdf = 2, tke_budget = 0

  call get_command_argument(1, output_path)
  if (len_trim(output_path) == 0) then
    write(*, '(A)') 'usage: run_tendencies_mf OUTPUT.csv'
    error stop 2
  end if
  xlvcp = xlv / cp
  xlscp = (xlv + xlf) / cp
  spp_pbl = 0
  mix_chem = .false.
  closure = 2.6
  open(newunit=unit, file=trim(output_path), status='new', action='write')
  write(unit, '(A)') 'case,k,bl_mynn_edmf_mom,delt,psfc,ust,wspd,uoce,' //   &
      'voce,flt,flq,flqv,flqc,dz,rho,u,v,th,tk,qv,p,exner,thl_before,' //    &
      'sqv_before,sqc_before,sqi_before,sqs_before,ozone,tcd,qcd,dfm,' //    &
      'dfh,diss_heat,sub_thl,sub_sqv,sub_u,sub_v,det_thl,det_sqv,' //        &
      'det_sqc,det_u,det_v,s_aw,s_awthl,s_awqv,s_awqc,s_awu,s_awv,' //       &
      'sd_aw,sd_awthl,sd_awqv,sd_awqc,sd_awu,sd_awv,s_aw_next,' //           &
      's_awthl_next,s_awqv_next,s_awqc_next,s_awu_next,s_awv_next,' //       &
      'sd_aw_next,sd_awthl_next,sd_awqv_next,sd_awqc_next,sd_awu_next,' //   &
      'sd_awv_next,du,dv,dth,dqv,dqc,dqi,dqs,dozone,thl_after,k_m,k_h'

  do c = 1, ncase
    ! Baseline: a moist convective land mixed layer under a 6 K inversion,
    ! deep enough that DMP_mf keeps at least one plume alive.
    delt = 20.0
    dz1 = 50.0
    dzslope = 15.0
    th_ml = 301.0
    dth_ml = -0.0006
    inv_dth = 6.0
    up_lapse = 0.0045
    qsurf = 0.0168
    dq_ml = -3.0e-7
    dq_up = -6.0e-6
    zinv = 1100.0
    u0 = 3.0
    v0 = -1.0
    ushear = 0.0015
    ust = 0.35
    flt = 0.14
    flqv = 1.0e-4
    flqc = 0.0
    uoce = 0.0
    voce = 0.0
    pblh = 1100.0
    dx = 3000.0
    xland = 1.0
    ts = 302.3
    psfc = 100000.0
    edmf_mom = 1

    select case (c)
    case (2)
      ! Marine column with a nonzero ocean current and a nonzero flqc.  The
      ! driver hard-wires flqc = 0.0 (module_bl_mynn.F:1069), so the flqc term
      ! in the sqc system is only reachable here.
      xland = 2.0
      flt = 0.05
      flqv = 1.3e-4
      flqc = 2.0e-7
      uoce = 0.35
      voce = -0.20
      qsurf = 0.0175
      zinv = 900.0
      pblh = 900.0
      ts = 301.4
      dx = 6000.0
    case (3)
      ! Deep mixed layer: more interfaces carry a nonzero s_aw, and the
      ! entrainment ramp at module_bl_mynn.F:6191 engages.
      flt = 0.19
      flqv = 1.4e-4
      qsurf = 0.0165
      zinv = 2600.0
      pblh = 2600.0
      ts = 303.0
      dx = 12000.0
    case (4)
      ! Fine grid: dx below the 1.2*dx plume-width cutoff, thin layers, so the
      ! mass flux is large relative to khdz and the stability floors at
      ! module_bl_mynn.F:4163-4169 have their best chance of binding.
      dz1 = 25.0
      dzslope = 5.0
      dx = 400.0
      qsurf = 0.0166
      zinv = 700.0
      pblh = 700.0
      ts = 302.0
    case (5)
      ! Driver-faithful momentum-off: bl_mynn_edmf_mom=0 reaches DMP_mf as
      ! momentum_opt=0, so s_awu/s_awv come back zero and onoff is zero too.
      edmf_mom = 0
    case (6)
      ! PROBE, not reachable through the driver.  DMP_mf runs with
      ! momentum_opt=1 so s_awu/s_awv are nonzero, then mynn_tendencies is
      ! called with bl_mynn_edmf_mom=0.  Discriminates the onoff factor.
      edmf_mom = 0
    case (7)
      ! PROBE, not reachable through the driver (bl_mynn_edmf_dd=0 at
      ! module_bl_mynn.F:330).  A synthetic downdraft is layered onto the
      ! DMP_mf updraft so every sd_aw* term is exercised.
      continue
    case (8)
      ! Live mass flux plus a strongly negative qcd, so the vapour solve goes
      ! negative and moisture_check's borrow-from-below repair runs while
      ! s_aw is nonzero.
      continue
    case (9)
      ! PROBE, not reachable through the driver (env_subs=.false. at
      ! module_bl_mynn.F:336).  Synthetic subsidence and detrainment
      ! tendencies are layered onto the DMP_mf updraft.
      continue
    end select

    momentum_opt = edmf_mom
    if (c == 6) momentum_opt = 1

    zw(1) = 0.0
    do k = 1, nz
      dz(k) = dz1 + dzslope * real(k - 1)
      zw(k+1) = zw(k) + dz(k)
    end do
    call build_profile()

    if (c == 2) then
      ! Resolved liquid inside the mixed layer and ice above the inversion, so
      ! the sqc and sqi diffusion solves and the xlscp term of the theta
      ! tendency are all nontrivial while the mass flux is live.
      do k = 1, nz
        zmid = 0.5 * (zw(k) + zw(k+1))
        if (zmid > 250.0 .and. zmid < 900.0) then
          sqc(k) = max(2.4e-4 - abs(zmid - 550.0) * 3.0e-7, 5.0e-6)
        end if
        if (zmid > 1400.0) then
          sqi(k) = max(4.0e-5 - (zmid - 1600.0) * 8.0e-9, 3.0e-6)
        end if
      end do
    end if

    do k = 1, nz
      w(k) = 0.02
      qke(k) = max(1.4 - 0.0008 * 0.5 * (zw(k) + zw(k+1)), 0.05)
      qnc(k) = 0.0
      qni(k) = 0.0
      qnwfa(k) = 0.0
      qnifa(k) = 0.0
      qnbca(k) = 0.0
      ozone(k) = 0.0
      kzero(k) = 0.0
      sqs(k) = 0.0
      rstoch(k) = 0.0
      tkeprodtd(k) = 0.0
      qc_bl(k) = 0.0
      qi_bl(k) = 0.0
      cldfra_bl(k) = 0.0
      qc_bl_old(k) = 0.0
      cldfra_bl_old(k) = 0.0
      chem1(k, 1) = 0.0
      chem1(k, 2) = 0.0
      sh(k) = 0.0
      sm(k) = 0.0
      el(k) = 0.0
      edmf_a(k) = 0.0
      edmf_w(k) = 0.0
      edmf_qc(k) = 0.0
      vt(k) = 0.0
      vq(k) = 0.0
      sgm(k) = 0.0
    end do
    s_awchem = 0.0

    flq = flqv + flqc
    th_sfc = ts / exner(kts)
    fltv = flt + flqv * p608 * th_sfc
    rmol = -karman * (9.81 / 300.0) * fltv / max(ust**3, 1.0e-6)
    wspd = max(sqrt(u(kts)**2 + v(kts)**2), 1.0)

    ! ---- driver assembly, module_bl_mynn.F:1001-1017 --------------------
    do k = 1, nz
      qv1(k) = sqv(k) / (1.0 - sqv(k))
      qc1(k) = sqc(k) / (1.0 - sqv(k))
      qi1(k) = sqi(k) / (1.0 - sqv(k))
      qs1(k) = sqs(k) / (1.0 - sqv(k))
      sqw(k) = sqv(k) + sqc(k) + sqi(k)
      thl(k) = th(k) - xlvcp / exner(k) * sqc(k) - xlscp / exner(k) * sqi(k)
      thetav(k) = th(k) * (1.0 + p608 * sqv(k))
    end do

    call get_pblh(kts, kte, pblh, thetav, qke, zw, dz, xland, kpbl)
    call scale_aware(dx, pblh, psig_bl, psig_shcu)

    ! ---- first-step initialisation, module_bl_mynn.F:813-824 -------------
    call mym_initialize(kts, kte, xland, dz, dx, zw, u, v, thl, sqw,      &
        pblh, th, thetav, sh, sm, ust, rmol, el, qke, tsq, qsq, cov,      &
        psig_bl, cldfra_bl, bl_mynn_mixlength, edmf_w, edmf_a, .true.,    &
        spp_pbl, rstoch)

    ! ---- subgrid condensation, module_bl_mynn.F:1104-1112 ---------------
    call mym_condensation(kts, kte, dx, dz, zw, xland, thl, sqw, sqv,     &
        sqc, sqi, sqs, p, exner, tsq, qsq, cov, sh, el, bl_mynn_cloudpdf, &
        qc_bl, qi_bl, cldfra_bl, pblh, flt * rho(kts) * cp, vt, vq, th,   &
        sgm, rmol, spp_pbl, rstoch)

    ! ---- mass flux, module_bl_mynn.F:1131-1169 --------------------------
    call dmp_mf(kts, kte, delt, zw, dz, p, rho, momentum_opt,             &
        bl_mynn_edmf_tke, bl_mynn_mixscalars, u, v, w, th, thl, thetav,   &
        tk, sqw, sqv, sqc, qke, qnc, qni, qnwfa, qnifa, qnbca, exner,     &
        vt, vq, sgm, ust, flt, fltv, flq, flqv, pblh, kpbl, dx, xland,    &
        th_sfc, edmf_a, edmf_w, edmf_qt, edmf_thl, edmf_ent, edmf_qc,     &
        s_aw, s_awthl, s_awqt, s_awqv, s_awqc, s_awu, s_awv, s_awqke,     &
        s_awqnc, s_awqni, s_awqnwfa, s_awqnifa, s_awqnbca,                &
        sub_thl, sub_sqv, sub_u, sub_v,                                   &
        det_thl, det_sqv, det_sqc, det_u, det_v,                          &
        nchem, chem1, s_awchem, mix_chem,                                 &
        qc_bl, cldfra_bl, qc_bl_old, cldfra_bl_old,                       &
        flag_qc, flag_qi, flag_qnc, flag_qni, flag_qnwfa, flag_qnifa,     &
        flag_qnbca, psig_shcu, maxwidth, ktop, maxmf, ztop, spp_pbl,      &
        rstoch)

    ! No DDMF_JPL: bl_mynn_edmf_dd is a compile-time 0.
    sd_aw = 0.0
    sd_awthl = 0.0
    sd_awqt = 0.0
    sd_awqv = 0.0
    sd_awqc = 0.0
    sd_awu = 0.0
    sd_awv = 0.0

    ! ---- diffusivities, module_bl_mynn.F:1192-1210 ----------------------
    call mym_turbulence(kts, kte, xland, closure, dz, dx, zw, u, v, thl,  &
        thetav, sqc, sqw, qke, tsq, qsq, cov, vt, vq, rmol, flt, fltv,    &
        flq, pblh, th, sh, sm, el, dfm, dfh, dfq, tcd, qcd, pdk, pdt,     &
        pdq, pdc, qwt, qshear, qbuoy, qdiss, tke_budget, psig_bl,         &
        psig_shcu, cldfra_bl, bl_mynn_mixlength, edmf_w, edmf_a,          &
        tkeprodtd, spp_pbl, rstoch)

    ! ---- predictor, module_bl_mynn.F:1079-1097 and :1215-1221 -----------
    zet = 0.5 * dz(kts) * rmol
    zet = max(zet, -20.0)
    zet = min(zet, 20.0)
    phi_m = phim(zet)
    pmz = phi_m - zet
    phh = phih(zet)
    call mym_predict(kts, kte, closure, delt, dz, ust, flt, flq, pmz,     &
        phh, el, dfq, rho, pdk, pdt, pdq, pdc, qke, tsq, qsq, cov,        &
        s_aw, s_awqke, bl_mynn_edmf_tke, qwt, qdiss, tke_budget)

    ! ---- dissipative heating, dheat_opt=1, module_bl_mynn.F:1223-1233 ---
    do k = kts, kte-1
      diss_heat(k) = min(max(1.0 * (qke(k)**1.5) &
          / (b1 * max(0.5 * (el(k) + el(k+1)), 1.)) / cp, 0.0), 0.002)
      diss_heat(k) = diss_heat(k) * exp(-10000. / max(p(k), 1.))
    end do
    diss_heat(kte) = 0.

    if (c == 7) then
      ! Synthetic downdraft, shaped like a DDMF_JPL return: sd_aw is negative
      ! (downward mass flux) and the scalar products carry the environmental
      ! values of the layer above the interface.
      do k = 2, nz
        if (zw(k) < 0.85 * pblh) then
          sd_aw(k) = -0.010 * (1.0 - zw(k) / max(pblh, 1.0))
          sd_awthl(k) = sd_aw(k) * thl(k)
          sd_awqt(k) = sd_aw(k) * sqw(k)
          sd_awqv(k) = sd_aw(k) * sqv(k)
          sd_awqc(k) = sd_aw(k) * 2.0e-5
          sd_awu(k) = sd_aw(k) * u(k)
          sd_awv(k) = sd_aw(k) * v(k)
        end if
      end do
    end if

    if (c == 9) then
      ! Synthetic subsidence/detrainment, shaped like the env_subs return at
      ! module_bl_mynn.F:6582-6597: subsidence warms and dries where the
      ! compensating descent is strongest, detrainment moistens the plume
      ! layer and deposits condensate.
      do k = 1, nz
        if (zw(k+1) < pblh) then
          sub_thl(k) = 2.0e-4 * (1.0 - zw(k) / max(pblh, 1.0))
          sub_sqv(k) = -3.0e-8 * (1.0 - zw(k) / max(pblh, 1.0))
          sub_u(k) = 1.5e-4 * (1.0 - zw(k) / max(pblh, 1.0))
          sub_v(k) = -9.0e-5 * (1.0 - zw(k) / max(pblh, 1.0))
          det_thl(k) = -1.1e-4 * (1.0 - zw(k) / max(pblh, 1.0))
          det_sqv(k) = 4.0e-8 * (1.0 - zw(k) / max(pblh, 1.0))
          det_sqc(k) = 2.5e-9 * (1.0 - zw(k) / max(pblh, 1.0))
          det_u(k) = -7.0e-5 * (1.0 - zw(k) / max(pblh, 1.0))
          det_v(k) = 5.0e-5 * (1.0 - zw(k) / max(pblh, 1.0))
        end if
      end do
    end if

    tcd_in = tcd
    qcd_in = qcd
    if (c == 8) then
      qcd_in(kts) = qcd(kts) - 6.0e-5
      qcd_in(9) = qcd(9) - 7.0e-5
      qcd_in(14) = qcd(14) - 4.0e-5
    end if
    tcd = tcd_in
    qcd = qcd_in

    thl_before = thl
    sqv_before = sqv
    sqc_before = sqc
    sqi_before = sqi
    sqs_before = sqs

    call retrieve_exchange_coeffs(kts, kte, dfm, dfh, dz, k_m, k_h)

    call mynn_tendencies(kts, kte, 1,                                    &
        delt, dz, rho,                                                   &
        u, v, th, tk, qv1, qc1, qi1, kzero, qnc, qni,                    &
        psfc, p, exner,                                                  &
        thl, sqv, sqc, sqi, kzero, sqw,                                  &
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
        cldfra_bl,                                                       &
        bl_mynn_cloudmix,                                                &
        bl_mynn_mixqt,                                                   &
        bl_mynn_edmf,                                                    &
        edmf_mom,                                                        &
        bl_mynn_mixscalars)

    do k = 1, nz
      write(unit, '(A,",",I0,",",I0)', advance='no') trim(names(c)), k,   &
          edmf_mom
      write(unit, '(41(",",ES24.16E3))', advance='no')                    &
          delt, psfc, ust, wspd, uoce, voce, flt, flq, flqv, flqc,        &
          dz(k), rho(k), u(k), v(k), th(k), tk(k), qv1(k), p(k),          &
          exner(k), thl_before(k), sqv_before(k), sqc_before(k),          &
          sqi_before(k), sqs_before(k), ozone(k), tcd(k), qcd(k),         &
          dfm(k), dfh(k), diss_heat(k), sub_thl(k), sub_sqv(k),           &
          sub_u(k), sub_v(k), det_thl(k), det_sqv(k), det_sqc(k),         &
          det_u(k), det_v(k), s_aw(k), s_awthl(k)
      write(unit, '(33(",",ES24.16E3))')                                  &
          s_awqv(k), s_awqc(k), s_awu(k), s_awv(k),                       &
          sd_aw(k), sd_awthl(k), sd_awqv(k), sd_awqc(k), sd_awu(k),       &
          sd_awv(k), s_aw(k+1), s_awthl(k+1), s_awqv(k+1), s_awqc(k+1),   &
          s_awu(k+1), s_awv(k+1), sd_aw(k+1), sd_awthl(k+1),              &
          sd_awqv(k+1), sd_awqc(k+1), sd_awu(k+1), sd_awv(k+1),           &
          du(k), dv(k), dth(k), dqv(k), dqc(k), dqi(k), dqs(k),           &
          dozone(k), thl(k), k_m(k), k_h(k)
    end do
  end do
  close(unit)

contains

  subroutine build_profile()
    integer :: kk
    real :: zm
    do kk = 1, nz
      zm = 0.5 * (zw(kk) + zw(kk+1))
      p(kk) = 100000.0 * exp(-zm / 8500.0)
      exner(kk) = (p(kk) / 100000.0) ** (287.0 / (7.0 * 287.0 / 2.0))
      if (zm <= zinv) then
        th(kk) = th_ml + dth_ml * zm
        sqv(kk) = max(qsurf + dq_ml * zm, 2.0e-4)
      else
        th(kk) = th_ml + dth_ml * zinv + inv_dth + up_lapse * (zm - zinv)
        sqv(kk) = max(qsurf + dq_ml * zinv + dq_up * (zm - zinv), 2.0e-4)
      end if
      sqc(kk) = 0.0
      sqi(kk) = 0.0
      tk(kk) = th(kk) * exner(kk)
      rho(kk) = p(kk) / (287.0 * tk(kk))
      u(kk) = u0 + ushear * zm
      v(kk) = v0 + 0.0008 * zm
    end do
  end subroutine build_profile

end program run_mynn_tendencies_mf_oracle
