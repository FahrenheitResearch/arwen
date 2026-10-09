program run_mynn_bl_driver_families_oracle
  ! Multi-step oracle for module_bl_mynn.F:360-1453 (mynn_bl_driver) from the
  ! unmodified pinned WRF v4.6.1 physics module, over six column families:
  ! convective day, stable night (dew: downward vapour flux), marine
  ! stratocumulus, shallow cumulus, a sub-freezing cold pool with fog and
  ! ice, and high wind.  Step 1 is the cold start (initflag=1); steps 2..N
  ! are warm.  Between steps the harness applies the driver's own
  ! tendencies to u, v, th and the water species with one rounded FP32
  ! multiply and one add each (x = x + delt*tend), floors the species at 0,
  ! and recomputes t3d = th*exner and rho = p/(r_d*t3d); every other input
  ! (surface fluxes, ust, wspd, qsfc, p, exner, dz) is held.  The state the
  ! driver carries itself (qke, tsq, qsq, cov, el_pbl, Sh3D, Sm3D, QC_BL,
  ! QI_BL, CLDFRA_BL, pblh, kpbl, rmol) is passed straight back.  Each
  ! step's incoming state is recorded, so any step is reproducible from its
  ! own rows, and a port can also be integrated forward from step 1 alone.
  !
  ! Admitted option identity: as run_driver.F90 (WRF registry defaults),
  ! bl_mynn_mixlength from the second argument (1 default; 2 is HRRR's).
  ! Third argument: number of steps (default 12).
  use module_bl_mynn, only: mynn_bl_driver
  implicit none

  integer, parameter :: ncol = 6, nz = 30
  integer, parameter :: ids = 1, ide = ncol + 1, jds = 1, jde = 2
  integer, parameter :: kds = 1, kde = nz + 1
  integer, parameter :: ims = 1, ime = ncol, jms = 1, jme = 1
  integer, parameter :: kms = 1, kme = nz
  integer, parameter :: its = 1, ite = ncol, jts = 1, jte = 1
  integer, parameter :: kts = 1, kte = nz
  integer, parameter :: nchem = 1, kdvel = 1, ndvel = 1
  integer :: nstep = 12
  character(len=32) :: nstep_arg
  character(len=32) :: carry_arg
  real :: initial_qke = 0.0
  character(len=32), parameter :: names(ncol) = [character(len=32) :: &
      'convective_day', 'stable_night', 'stratocumulus', 'shallow_cumulus', &
      'cold_pool', 'high_wind']
  character(len=32) :: mixlength_arg
  character(len=1024) :: output_path
  integer :: c, k, s, unit

  real :: dz(ims:ime, kms:kme), u(ims:ime, kms:kme), v(ims:ime, kms:kme)
  real :: w(ims:ime, kms:kme), th(ims:ime, kms:kme)
  real :: sqv3d(ims:ime, kms:kme), sqc3d(ims:ime, kms:kme)
  real :: sqi3d(ims:ime, kms:kme), sqs3d(ims:ime, kms:kme)
  real :: qnc(ims:ime, kms:kme), qni(ims:ime, kms:kme)
  real :: qnwfa(ims:ime, kms:kme), qnifa(ims:ime, kms:kme)
  real :: qnbca(ims:ime, kms:kme), ozone(ims:ime, kms:kme)
  real :: p(ims:ime, kms:kme), exner(ims:ime, kms:kme)
  real :: rho(ims:ime, kms:kme), t3d(ims:ime, kms:kme)
  real :: rthraten(ims:ime, kms:kme)
  real :: dx(ims:ime), znt(ims:ime), xland(ims:ime), ts(ims:ime)
  real :: qsfc(ims:ime), ps(ims:ime), ust(ims:ime), ch(ims:ime)
  real :: hfx(ims:ime), qfx(ims:ime), rmol(ims:ime), wspd(ims:ime)
  real :: uoce(ims:ime), voce(ims:ime), pblh(ims:ime)
  real :: maxwidth(ims:ime), maxmf(ims:ime), ztop_plume(ims:ime)
  integer :: kpbl(ims:ime), ktop_plume(ims:ime)
  real :: qke(ims:ime, kms:kme), qke_adv(ims:ime, kms:kme)
  real :: sh3d(ims:ime, kms:kme), sm3d(ims:ime, kms:kme)
  real :: tsq(ims:ime, kms:kme), qsq(ims:ime, kms:kme)
  real :: cov(ims:ime, kms:kme), el_pbl(ims:ime, kms:kme)
  real :: qc_bl(ims:ime, kms:kme), qi_bl(ims:ime, kms:kme)
  real :: cldfra_bl(ims:ime, kms:kme)
  real :: rublten(ims:ime, kms:kme), rvblten(ims:ime, kms:kme)
  real :: rthblten(ims:ime, kms:kme), rqvblten(ims:ime, kms:kme)
  real :: rqcblten(ims:ime, kms:kme), rqiblten(ims:ime, kms:kme)
  real :: rqsblten(ims:ime, kms:kme), rqncblten(ims:ime, kms:kme)
  real :: rqniblten(ims:ime, kms:kme), rqnwfablten(ims:ime, kms:kme)
  real :: rqnifablten(ims:ime, kms:kme), rqnbcablten(ims:ime, kms:kme)
  real :: dozone(ims:ime, kms:kme)
  real :: exch_h(ims:ime, kms:kme), exch_m(ims:ime, kms:kme)
  real :: dqke(ims:ime, kms:kme), qwt(ims:ime, kms:kme)
  real :: qshear(ims:ime, kms:kme), qbuoy(ims:ime, kms:kme)
  real :: qdiss(ims:ime, kms:kme)
  real :: edmf_a(ims:ime, kms:kme), edmf_w(ims:ime, kms:kme)
  real :: edmf_qt(ims:ime, kms:kme), edmf_thl(ims:ime, kms:kme)
  real :: edmf_ent(ims:ime, kms:kme), edmf_qc(ims:ime, kms:kme)
  real :: sub_thl3d(ims:ime, kms:kme), sub_sqv3d(ims:ime, kms:kme)
  real :: det_thl3d(ims:ime, kms:kme), det_sqv3d(ims:ime, kms:kme)
  real :: pattern_spp_pbl(ims:ime, kms:kme)
  real :: chem3d(ims:ime, kms:kme, nchem), vdep(ims:ime, ndvel)
  real :: frp(ims:ime), emis_ant_no(ims:ime)
  ! recorded incoming state
  real :: qke_in(ims:ime, kms:kme), tsq_in(ims:ime, kms:kme)
  real :: qsq_in(ims:ime, kms:kme), cov_in(ims:ime, kms:kme)
  real :: el_in(ims:ime, kms:kme), sh_in(ims:ime, kms:kme)
  real :: sm_in(ims:ime, kms:kme), qcbl_in(ims:ime, kms:kme)
  real :: qibl_in(ims:ime, kms:kme), cfbl_in(ims:ime, kms:kme)
  real :: pblh_in(ims:ime), rmol_in(ims:ime)
  integer :: kpbl_in(ims:ime)
  real :: delt, zw, zm, zinv, th_ml, qsurf, dz1, dzslope
  real :: dthml, dthinv, dqinv, gam, qt, es, qs, tk, uref, ushear, cldtop
  integer :: initflag
  logical, parameter :: restart = .false.
  logical :: cycling = .false.
  logical, parameter :: tkeadvect = .false.
  logical, parameter :: mix_chem = .false., enh_mix = .false.
  logical, parameter :: rrfs_sd = .false., smoke_dbg = .false.
  logical, parameter :: flag_qc = .true., flag_qi = .true.
  logical, parameter :: flag_qnc = .false., flag_qni = .false.
  logical, parameter :: flag_qs = .true., flag_qnwfa = .false.
  logical, parameter :: flag_qnifa = .false., flag_qnbca = .false.
  logical, parameter :: flag_ozone = .false.
  integer, parameter :: tke_budget = 0, bl_mynn_cloudpdf = 2
  integer, parameter :: icloud_bl = 1
  integer :: bl_mynn_mixlength = 1
  integer, parameter :: bl_mynn_edmf = 1, bl_mynn_edmf_mom = 1
  integer, parameter :: bl_mynn_edmf_tke = 0, bl_mynn_mixscalars = 0
  integer, parameter :: bl_mynn_output = 0, bl_mynn_cloudmix = 1
  integer, parameter :: bl_mynn_mixqt = 0, spp_pbl = 0
  real, parameter :: closure = 2.6

  call get_command_argument(1, output_path)
  call get_command_argument(2, mixlength_arg)
  if (len_trim(mixlength_arg) > 0) read(mixlength_arg, *) bl_mynn_mixlength
  call get_command_argument(3, nstep_arg)
  if (len_trim(nstep_arg) > 0) read(nstep_arg, *) nstep
  ! Optional cycled-start control. TRANSFER constructs exact adjacent words
  ! without decimal-parser ambiguity. Omission preserves the cold fixtures.
  call get_command_argument(4, carry_arg)
  if (len_trim(carry_arg) > 0) then
    cycling = .true.
    select case (trim(carry_arg))
    case ('empty')
      initial_qke = 0.0
    case ('below', 'mixed')
      initial_qke = transfer(int(z'3951B716'), 0.0)
    case ('equal')
      initial_qke = transfer(int(z'3951B717'), 0.0)
    case ('above')
      initial_qke = transfer(int(z'3951B718'), 0.0)
    case ('high')
      initial_qke = 0.9
    case default
      error stop 'unknown carry control'
    end select
  end if
  if (len_trim(output_path) == 0) then
    write(*, '(A)') 'usage: run_driver_families OUTPUT.csv [MIXLENGTH [NSTEP [CARRY]]]'
    error stop 2
  end if
  open(newunit=unit, file=trim(output_path), status='new', action='write')
  write(unit, '(A)') 'case,step,k,initflag,delt,dx,znt,xland,ts,qsfc,ps,' // &
      'ust,ch,hfx,qfx,wspd,uoce,voce,dz,u,v,w,th,sqv3d,sqc3d,sqi3d,' //      &
      'sqs3d,p,' //                                                          &
      'exner,rho,t3d,qke_in,tsq_in,qsq_in,cov_in,el_in,sh_in,sm_in,' //      &
      'qc_bl_in,qi_bl_in,cldfra_bl_in,pblh_in,kpbl_in,rmol_in,rublten,' //   &
      'rvblten,rthblten,rqvblten,rqcblten,rqiblten,rqsblten,dozone,exch_h,' // &
      'exch_m,qke,tsq,qsq,cov,el_pbl,sh3d,sm3d,qc_bl,qi_bl,cldfra_bl,' //    &
      'pblh,kpbl,rmol,maxwidth,maxmf,ztop_plume,ktop_plume'

  delt = 20.0
  ! ---- atmosphere: six column families, one block --------------------
  do c = 1, ncol
    dz1 = 50.0
    dzslope = 15.0
    dx(c) = 3000.0
    xland(c) = 1.0
    ps(c) = 100000.0
    uoce(c) = 0.0
    voce(c) = 0.0
    dthml = 0.0
    dthinv = 4.0
    dqinv = 2.0e-3
    gam = 0.004
    cldtop = -1.0
    select case (c)
    case (1)
      ! Convective day: strong heating over dry land, deep mixed layer.
      th_ml = 303.0; qsurf = 0.0120; zinv = 1600.0
      ts(c) = 308.0; ust(c) = 0.45; hfx(c) = 320.0; qfx(c) = 9.0e-5
      znt(c) = 0.10; ch(c) = 0.012; uref = 4.0; ushear = 0.0015
    case (2)
      ! Stable night: cooling surface, surface inversion, dew (QFX < 0)
      ! over rough land, light wind with a low-level jet.
      th_ml = 290.0; qsurf = 0.0105; zinv = 250.0; dthml = 0.016
      dthinv = 1.0; dqinv = 5.0e-4
      ts(c) = 286.5; ust(c) = 0.12; hfx(c) = -38.0; qfx(c) = -1.2e-5
      znt(c) = 0.50; ch(c) = 0.004; uref = 2.0; ushear = 0.012
    case (3)
      ! Marine stratocumulus: well mixed, saturated below a strong
      ! inversion, weak surface fluxes, cloud-top radiative cooling.
      xland(c) = 2.0
      th_ml = 289.0; qsurf = 0.0102; zinv = 850.0; dthinv = 9.0
      dqinv = 5.0e-3
      ts(c) = 289.5; ust(c) = 0.25; hfx(c) = 12.0; qfx(c) = 3.5e-5
      znt(c) = 0.0002; ch(c) = 0.010; uref = 7.0; ushear = 0.001
      cldtop = 850.0; dx(c) = 6000.0
    case (4)
      ! Shallow cumulus over land: moist mixed layer, weak inversion,
      ! conditionally unstable cloud layer with sparse resolved liquid.
      th_ml = 300.0; qsurf = 0.0130; zinv = 1100.0; dthinv = 1.0
      dqinv = 1.5e-3; gam = 0.0035
      ts(c) = 303.5; ust(c) = 0.35; hfx(c) = 190.0; qfx(c) = 2.1e-4
      znt(c) = 0.15; ch(c) = 0.011; uref = 5.0; ushear = 0.0010
      cldtop = 2200.0
    case (5)
      ! Cold pool: sub-freezing stable layer with fog, ice and snow above,
      ! frost (QFX < 0) and a weak downward heat flux.
      th_ml = 268.5; qsurf = 0.0029; zinv = 400.0; dthml = 0.006
      dthinv = 5.0; dqinv = 8.0e-4
      ts(c) = 266.0; ust(c) = 0.08; hfx(c) = -12.0; qfx(c) = -3.0e-6
      znt(c) = 0.05; ch(c) = 0.003; uref = 1.5; ushear = 0.004
      ps(c) = 98000.0; cldtop = 300.0
    case (6)
      ! High wind: near neutral, strong shear, mechanical turbulence.
      th_ml = 295.0; qsurf = 0.0090; zinv = 1300.0; dthinv = 2.0
      dqinv = 1.0e-3
      ts(c) = 295.6; ust(c) = 0.95; hfx(c) = 25.0; qfx(c) = 6.0e-5
      znt(c) = 0.40; ch(c) = 0.020; uref = 18.0; ushear = 0.006
      dx(c) = 1000.0; dz1 = 30.0; dzslope = 12.0
    end select
    zw = 0.0
    do k = 1, nz
      dz(c, k) = dz1 + dzslope * real(k - 1)
      zm = zw + 0.5 * dz(c, k)
      p(c, k) = ps(c) * exp(-zm / 8500.0)
      exner(c, k) = (p(c, k) / 100000.0) ** (287.0 / (7.0 * 287.0 / 2.0))
      if (zm <= zinv) then
        th(c, k) = th_ml + dthml * zm
        qt = qsurf - 1.0e-7 * zm
      else
        th(c, k) = th_ml + dthml * zinv + dthinv + gam * (zm - zinv)
        qt = max(qsurf - 1.0e-7 * zinv - dqinv - 2.5e-6 * (zm - zinv), &
            1.0e-4)
      end if
      ! Saturation adjustment of the total water (Bolton over water above
      ! 253 K, Murray over ice below), so cloudy families start saturated
      ! with resolved condensate.
      tk = th(c, k) * exner(c, k)
      if (tk > 253.0) then
        es = 611.2 * exp(17.67 * (tk - 273.15) / (tk - 29.65))
      else
        es = 611.2 * exp(21.8745584 * (tk - 273.16) / (tk - 7.66))
      end if
      qs = 0.622 * es / (p(c, k) - 0.378 * es)
      sqc3d(c, k) = 0.0
      sqi3d(c, k) = 0.0
      sqs3d(c, k) = 0.0
      if (zm <= cldtop .and. qt > qs) then
        sqv3d(c, k) = qs
        if (tk > 263.0) then
          sqc3d(c, k) = qt - qs
        else
          sqc3d(c, k) = 0.5 * (qt - qs)
          sqi3d(c, k) = 0.5 * (qt - qs)
        end if
      else
        sqv3d(c, k) = min(qt, qs)
      end if
      if (c == 4 .and. zm > zinv .and. zm < cldtop) then
        sqc3d(c, k) = 4.0e-5
      end if
      if (c == 5 .and. zm > 1500.0 .and. zm < 3500.0) then
        sqi3d(c, k) = 2.0e-5
        sqs3d(c, k) = 3.0e-5
      end if
      t3d(c, k) = th(c, k) * exner(c, k)
      rho(c, k) = p(c, k) / (287.0 * t3d(c, k))
      u(c, k) = uref + ushear * zm
      if (c == 2 .and. zm < 600.0) u(c, k) = uref + 9.0 * zm / 600.0
      v(c, k) = -1.0 + 0.0008 * zm
      w(c, k) = 0.0
      if (c == 4) w(c, k) = 0.05
      if (c == 3) w(c, k) = -0.004
      qnc(c, k) = 0.0
      qni(c, k) = 0.0
      qnwfa(c, k) = 0.0
      qnifa(c, k) = 0.0
      qnbca(c, k) = 0.0
      ozone(c, k) = 0.0
      rthraten(c, k) = 0.0
      if (c == 3 .and. zm > zinv - 120.0 .and. zm <= zinv) then
        rthraten(c, k) = -1.5e-4
      end if
      pattern_spp_pbl(c, k) = 0.0
      chem3d(c, k, 1) = 0.0
      qke(c, k) = 0.0
      qke_adv(c, k) = 0.0
      sh3d(c, k) = 0.0
      sm3d(c, k) = 0.0
      tsq(c, k) = 0.0
      qsq(c, k) = 0.0
      cov(c, k) = 0.0
      el_pbl(c, k) = 0.0
      qc_bl(c, k) = 0.0
      qi_bl(c, k) = 0.0
      cldfra_bl(c, k) = 0.0
      exch_h(c, k) = 0.0
      exch_m(c, k) = 0.0
      rublten(c, k) = 0.0
      rvblten(c, k) = 0.0
      rthblten(c, k) = 0.0
      rqvblten(c, k) = 0.0
      rqcblten(c, k) = 0.0
      rqiblten(c, k) = 0.0
      rqsblten(c, k) = 0.0
      rqncblten(c, k) = 0.0
      rqniblten(c, k) = 0.0
      rqnwfablten(c, k) = 0.0
      rqnifablten(c, k) = 0.0
      rqnbcablten(c, k) = 0.0
      dozone(c, k) = 0.0
      dqke(c, k) = 0.0
      qwt(c, k) = 0.0
      qshear(c, k) = 0.0
      qbuoy(c, k) = 0.0
      qdiss(c, k) = 0.0
      edmf_a(c, k) = 0.0
      edmf_w(c, k) = 0.0
      edmf_qt(c, k) = 0.0
      edmf_thl(c, k) = 0.0
      edmf_ent(c, k) = 0.0
      edmf_qc(c, k) = 0.0
      sub_thl3d(c, k) = 0.0
      sub_sqv3d(c, k) = 0.0
      det_thl3d(c, k) = 0.0
      det_sqv3d(c, k) = 0.0
      zw = zw + dz(c, k)
    end do
    qsfc(c) = sqv3d(c, 1)
    wspd(c) = max(sqrt(u(c, 1)**2 + v(c, 1)**2), 1.0)
    rmol(c) = 0.0
    pblh(c) = 0.0
    kpbl(c) = 1
    vdep(c, 1) = 0.0
    frp(c) = 0.0
    emis_ant_no(c) = 0.0
  end do

  if (cycling) then
    qke = initial_qke
    if (trim(carry_arg) /= 'empty') then
      qc_bl = 3.0e-4
      cldfra_bl = 0.6
      ! Poison the six reset slots with finite nonzero state too.
      sh3d = 0.2; sm3d = 0.3; el_pbl = 10.0
      tsq = 0.02; qsq = 1.0e-6; cov = 1.0e-5
    end if
    if (trim(carry_arg) == 'mixed') qke(ncol, 1) = transfer(int(z'3951B717'), 0.0)
  end if

  do s = 1, nstep
    initflag = 0
    if (s == 1) initflag = 1
    qke_in = qke
    tsq_in = tsq
    qsq_in = qsq
    cov_in = cov
    el_in = el_pbl
    sh_in = sh3d
    sm_in = sm3d
    qcbl_in = qc_bl
    qibl_in = qi_bl
    cfbl_in = cldfra_bl
    pblh_in = pblh
    kpbl_in = kpbl
    rmol_in = rmol

    call mynn_bl_driver(                                                   &
        initflag, restart, cycling,                                        &
        delt, dz, dx, znt,                                                 &
        u, v, w, th, sqv3d, sqc3d, sqi3d,                                  &
        sqs3d, qnc, qni,                                                   &
        qnwfa, qnifa, qnbca, ozone,                                        &
        p, exner, rho, t3d,                                                &
        xland, ts, qsfc, ps,                                               &
        ust, ch, hfx, qfx, rmol, wspd,                                     &
        uoce, voce,                                                        &
        qke, qke_adv,                                                      &
        sh3d, sm3d,                                                        &
        nchem, kdvel, ndvel,                                               &
        chem3d, vdep,                                                      &
        frp, emis_ant_no,                                                  &
        mix_chem, enh_mix,                                                 &
        rrfs_sd, smoke_dbg,                                                &
        tsq, qsq, cov,                                                     &
        rublten, rvblten, rthblten,                                        &
        rqvblten, rqcblten, rqiblten,                                      &
        rqncblten, rqniblten, rqsblten,                                    &
        rqnwfablten, rqnifablten,                                          &
        rqnbcablten, dozone,                                               &
        exch_h, exch_m,                                                    &
        pblh, kpbl,                                                        &
        el_pbl,                                                            &
        dqke, qwt, qshear, qbuoy, qdiss,                                   &
        qc_bl, qi_bl, cldfra_bl,                                           &
        tkeadvect,                                                         &
        tke_budget,                                                        &
        bl_mynn_cloudpdf,                                                  &
        bl_mynn_mixlength,                                                 &
        icloud_bl,                                                         &
        closure,                                                           &
        bl_mynn_edmf,                                                      &
        bl_mynn_edmf_mom,                                                  &
        bl_mynn_edmf_tke,                                                  &
        bl_mynn_mixscalars,                                                &
        bl_mynn_output,                                                    &
        bl_mynn_cloudmix, bl_mynn_mixqt,                                   &
        edmf_a, edmf_w, edmf_qt,                                           &
        edmf_thl, edmf_ent, edmf_qc,                                       &
        sub_thl3d, sub_sqv3d,                                              &
        det_thl3d, det_sqv3d,                                              &
        maxwidth, maxmf, ztop_plume,                                       &
        ktop_plume,                                                        &
        spp_pbl, pattern_spp_pbl,                                          &
        rthraten,                                                          &
        flag_qc, flag_qi, flag_qnc,                                        &
        flag_qni, flag_qs,                                                 &
        flag_qnwfa, flag_qnifa,                                            &
        flag_qnbca, flag_ozone,                                            &
        ids, ide, jds, jde, kds, kde,                                      &
        ims, ime, jms, jme, kms, kme,                                      &
        its, ite, jts, jte, kts, kte)

    do c = 1, ncol
      do k = 1, nz
        write(unit, '(A,",",I0,",",I0,",",I0)', advance='no')              &
            trim(names(c)), s, k, initflag
        write(unit, '(38(",",ES24.16E3))', advance='no')                   &
            delt, dx(c), znt(c), xland(c), ts(c), qsfc(c), ps(c), ust(c),  &
            ch(c), hfx(c), qfx(c), wspd(c), uoce(c), voce(c), dz(c, k),    &
            u(c, k), v(c, k), w(c, k), th(c, k), sqv3d(c, k),              &
            sqc3d(c, k), sqi3d(c, k), sqs3d(c, k), p(c, k), exner(c, k),   &
            rho(c, k),                                                       &
            t3d(c, k), qke_in(c, k), tsq_in(c, k), qsq_in(c, k),           &
            cov_in(c, k), el_in(c, k), sh_in(c, k), sm_in(c, k),           &
            qcbl_in(c, k), qibl_in(c, k), cfbl_in(c, k), pblh_in(c)
        write(unit, '(",",I0)', advance='no') kpbl_in(c)
        write(unit, '(22(",",ES24.16E3))', advance='no')                   &
            rmol_in(c), rublten(c, k), rvblten(c, k), rthblten(c, k),      &
            rqvblten(c, k), rqcblten(c, k), rqiblten(c, k), rqsblten(c, k), &
            dozone(c, k), exch_h(c, k), exch_m(c, k), qke(c, k),           &
            tsq(c, k), qsq(c, k), cov(c, k), el_pbl(c, k), sh3d(c, k),     &
            sm3d(c, k), qc_bl(c, k), qi_bl(c, k), cldfra_bl(c, k),         &
            pblh(c)
        write(unit, '(",",I0)', advance='no') kpbl(c)
        write(unit, '(4(",",ES24.16E3))', advance='no')                    &
            rmol(c), maxwidth(c), maxmf(c), ztop_plume(c)
        write(unit, '(",",I0)') ktop_plume(c)
      end do
    end do
    ! Advance the atmosphere with the driver's own tendencies.
    do c = 1, ncol
      do k = 1, nz
        u(c, k) = u(c, k) + delt * rublten(c, k)
        v(c, k) = v(c, k) + delt * rvblten(c, k)
        th(c, k) = th(c, k) + delt * rthblten(c, k)
        sqv3d(c, k) = max(sqv3d(c, k) + delt * rqvblten(c, k), 0.0)
        sqc3d(c, k) = max(sqc3d(c, k) + delt * rqcblten(c, k), 0.0)
        sqi3d(c, k) = max(sqi3d(c, k) + delt * rqiblten(c, k), 0.0)
        sqs3d(c, k) = max(sqs3d(c, k) + delt * rqsblten(c, k), 0.0)
        t3d(c, k) = th(c, k) * exner(c, k)
        rho(c, k) = p(c, k) / (287.0 * t3d(c, k))
      end do
    end do
  end do
  close(unit)
end program run_mynn_bl_driver_families_oracle
