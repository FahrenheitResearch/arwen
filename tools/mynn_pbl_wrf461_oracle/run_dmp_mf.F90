program run_mynn_dmp_mf_oracle
  ! Dump module_bl_mynn.F:5679-6823 (DMP_mf) from the unmodified pinned WRF
  ! v4.6.1 physics module.  The admitted identity is bl_mynn_edmf_mom=1,
  ! bl_mynn_edmf_tke=0, bl_mynn_mixscalars=0, mix_chem=.false. and spp_pbl=0;
  ! env_subs=.false. and bl_mynn_edmf_dd=0 are compile-time parameters in
  ! module_bl_mynn.F:336 and :330.
  !
  ! Every column is a mixed layer under a capping inversion, which is the only
  ! regime in which the plume model does anything: a uniformly stable column
  ! kills the smallest plume at its first level and DMP_mf then zeroes all of
  ! its outputs.  Case 'dead_probe' repeats 'land_cumulus' with every argument
  ! that no admitted branch reads moved off its baseline, so their deadness is
  ! demonstrated by the Fortran rather than argued from reading it.
  use module_bl_mynn, only: dmp_mf
  implicit none

  integer, parameter :: ncase = 12, nz = 30, kts = 1, kte = nz
  integer, parameter :: nchem = 2
  character(len=32), parameter :: names(ncase) = [character(len=32) :: &
      'land_dry', 'land_cumulus', 'water_cumulus', 'stable_off', &
      'resolved_w', 'flux_limited', 'stochastic', 'high_wind_thin', &
      'deep_pblh', 'cloud_base_capped', 'fine_grid', 'dead_probe']
  character(len=1024) :: output_path
  integer :: c, k, unit, kpbl, ktop, momentum_opt, tke_opt, scalar_opt
  integer :: spp_pbl
  logical :: mix_chem

  real :: dz(nz), zw(nz+1), p(nz), rho(nz), u(nz), v(nz), w(nz)
  real :: th(nz), thl(nz), thv(nz), tk(nz), qt(nz), qv(nz), qc(nz)
  real :: exner(nz), qke(nz), qnc(nz), qni(nz), qnwfa(nz), qnifa(nz)
  real :: qnbca(nz), vt(nz), vq(nz), sgm(nz), rstoch(nz)
  real :: qc_bl(nz), cldfra_bl(nz), qc_bl_old(nz), cldfra_bl_old(nz)
  real :: edmf_a(nz), edmf_w(nz), edmf_qt(nz), edmf_thl(nz)
  real :: edmf_ent(nz), edmf_qc(nz)
  real :: s_aw(nz+1), s_awthl(nz+1), s_awqt(nz+1), s_awqv(nz+1)
  real :: s_awqc(nz+1), s_awu(nz+1), s_awv(nz+1), s_awqke(nz+1)
  real :: s_awqnc(nz+1), s_awqni(nz+1), s_awqnwfa(nz+1), s_awqnifa(nz+1)
  real :: s_awqnbca(nz+1)
  real :: sub_thl(nz), sub_sqv(nz), sub_u(nz), sub_v(nz)
  real :: det_thl(nz), det_sqv(nz), det_sqc(nz), det_u(nz), det_v(nz)
  real :: chem1(nz, nchem), s_awchem(nz+1, nchem)
  real :: dt, ust, flt, fltv, flq, flqv, pblh, dx, landsea, ts, psig_shcu
  real :: maxwidth, maxmf, ztop
  real :: qc_bl_in(nz), cldfra_bl_in(nz), vt_in(nz), vq_in(nz)
  ! Profile controls, host-associated by build_profile.
  real :: dz1, dzslope, th_ml, dth_ml, inv_dth, up_lapse, qsurf, dq_ml
  real :: dq_up, zinv, ushear, u0, v0

  call get_command_argument(1, output_path)
  if (len_trim(output_path) == 0) then
    write(*, '(A)') 'usage: run_dmp_mf OUTPUT.csv'
    error stop 2
  end if
  open(newunit=unit, file=trim(output_path), status='new', action='write')
  write(unit, '(A)') 'case,k,dz,zw,zw_next,p,rho,u,v,w,th,thl,thv,tk,qt,' // &
      'qv,qc,exner,qke,qnc,qni,qnwfa,qnifa,qnbca,vt_before,vq_before,' // &
      'sgm,rstoch,qc_bl_before,cldfra_bl_before,qc_bl_old,' // &
      'cldfra_bl_old,dt,ust,flt,fltv,flq,flqv,pblh,kpbl,dx,landsea,ts,' // &
      'psig_shcu,edmf_a,edmf_w,edmf_qt,edmf_thl,edmf_ent,edmf_qc,' // &
      's_aw,s_awthl,s_awqt,s_awqv,s_awqc,s_awu,s_awv,s_awqke,s_awqnc,' // &
      's_awqni,s_awqnwfa,s_awqnifa,s_awqnbca,' // &
      's_aw_next,s_awthl_next,s_awqt_next,s_awqv_next,s_awqc_next,' // &
      's_awu_next,s_awv_next,sub_thl,sub_sqv,sub_u,sub_v,det_thl,' // &
      'det_sqv,det_sqc,det_u,det_v,qc_bl,cldfra_bl,vt,vq,' // &
      'maxwidth,ktop,ztop,maxmf'

  momentum_opt = 1
  tke_opt = 0
  scalar_opt = 0
  spp_pbl = 0
  mix_chem = .false.

  do c = 1, ncase
    ! Baseline: a moist convective land mixed layer under a 6 K inversion.
    dt = 20.0
    dz1 = 50.0
    dzslope = 15.0
    th_ml = 301.0
    dth_ml = -0.0006
    inv_dth = 6.0
    up_lapse = 0.0045
    qsurf = 0.0150
    dq_ml = -3.0e-7
    dq_up = -6.0e-6
    zinv = 900.0
    u0 = 3.0
    v0 = -1.0
    ushear = 0.0015
    ust = 0.35
    flt = 0.11
    flqv = 6.0e-5
    pblh = 900.0
    kpbl = 12
    dx = 3000.0
    landsea = 1.0
    ts = 301.9
    psig_shcu = 1.0

    select case (c)
    case (1)
      ! Dry: the plumes never reach saturation, so condensation_edmf keeps
      ! UPQC at zero and the shallow-cumulus block cannot fire.
      qsurf = 0.0060
      ts = 302.1
    case (2, 12)
      ! Moist enough that the plumes saturate, which is what reaches
      ! condensation_edmf with a nonzero QC and fires the shallow-cumulus
      ! cloud-fraction block at module_bl_mynn.F:6633.  dead_probe repeats
      ! this column, so it takes the same profile.
      qsurf = 0.0168
      flt = 0.14
      flqv = 1.0e-4
      zinv = 1100.0
      pblh = 1100.0
      ts = 302.3
    case (3)
      landsea = 2.0
      flt = 0.05
      flqv = 1.3e-4
      qsurf = 0.0175
      zinv = 900.0
      pblh = 900.0
      ts = 301.4
      dx = 6000.0
    case (4)
      ! Stable: fltv2 stays below the 0.002 activation threshold.
      th_ml = 299.0
      dth_ml = 0.004
      flt = -0.03
      flqv = 1.0e-6
      qsurf = 0.0080
      zinv = 250.0
      pblh = 250.0
      ts = 298.0
    case (5)
      ! Strong resolved ascent drives Psig_w to zero, which flips fltv2 and
      ! deactivates the scheme after the cloud-base search has run.
      qsurf = 0.0158
      pblh = 850.0
      zinv = 850.0
    case (6)
      ! A small heat flux with a strong resolved theta step trips the
      ! fluxportion limiter at module_bl_mynn.F:6474.
      flt = 0.0035
      flqv = 2.6e-4
      qsurf = 0.0160
      ts = 302.6
    case (7)
      ! Stochastic entrainment perturbation, module_bl_mynn.F:6195.
      qsurf = 0.0152
    case (8)
      ! wspd_pbl above 10 m/s tapers acfac and exc_fac; the thin surface
      ! layer lifts k50 to 3, which is what reaches the k>=2 arm of the
      ! superadiabatic test at module_bl_mynn.F:5985.
      dz1 = 20.0
      dzslope = 18.0
      u0 = 16.0
      v0 = 4.0
      qsurf = 0.0155
      zinv = 1000.0
      pblh = 1000.0
      ts = 302.0
    case (9)
      ! A deep mixed layer so ZW(k) passes MIN(pblh+1500,4000) and the
      ! entrainment ramp at module_bl_mynn.F:6191 engages.
      flt = 0.19
      flqv = 1.4e-4
      qsurf = 0.0165
      zinv = 2600.0
      pblh = 2600.0
      ts = 303.0
      dx = 12000.0
    case (10)
      ! A resolved stratus deck inside the mixed layer caps maxwidth through
      ! criterion (3) at module_bl_mynn.F:6008.
      qsurf = 0.0166
      pblh = 1300.0
      zinv = 1300.0
      ts = 302.4
    case (11)
      ! dx below the 1.2*dx cutoff so criterion (1) sets maxwidth, and a
      ! finer grid so more interfaces sit inside the plume layer.
      dz1 = 30.0
      dzslope = 8.0
      dx = 400.0
      qsurf = 0.0156
      zinv = 700.0
      pblh = 700.0
      ts = 302.0
    end select

    zw(1) = 0.0
    do k = 1, nz
      dz(k) = dz1 + dzslope * real(k - 1)
      zw(k+1) = zw(k) + dz(k)
    end do
    flq = flqv
    call build_profile()

    do k = 1, nz
      w(k) = 0.02
      qke(k) = max(1.4 - 0.0008 * 0.5 * (zw(k) + zw(k+1)), 0.05)
      qnc(k) = 1.0e8
      qni(k) = 1.0e5
      qnwfa(k) = 1.0e9
      qnifa(k) = 1.0e6
      qnbca(k) = 1.0e7
      vt_in(k) = -0.2
      vq_in(k) = 0.05
      sgm(k) = 1.0e-4
      rstoch(k) = 0.0
      qc_bl_in(k) = 0.0
      cldfra_bl_in(k) = 0.0
      qc_bl_old(k) = 0.0
      cldfra_bl_old(k) = 0.0
      chem1(k, 1) = 1.0
      chem1(k, 2) = 2.0
    end do

    select case (c)
    case (5)
      do k = 1, nz
        w(k) = 2.6 - 0.05 * real(k - 1)
      end do
      do k = 1, 8
        qc_bl_in(k) = 3.0e-5
        cldfra_bl_in(k) = 0.7
      end do
    case (6)
      th(2) = th(2) - 7.5
      thl(2) = th(2)
      tk(2) = th(2) * exner(2)
      thv(2) = th(2) * (1.0 + 0.608 * qv(2) - qc(2))
      rho(2) = p(2) / (287.0 * tk(2))
    case (7)
      do k = 1, nz
        rstoch(k) = 0.18 - 0.004 * real(k - 1)
      end do
    case (10)
      ! The resolved qc has to dominate qc_bl1d and be the only term above
      ! the 1e-5 cloud-base threshold, otherwise MAX(qc,qc_bl1d) at
      ! module_bl_mynn.F:5952 masks it and no case can detect qc at all.
      do k = 8, 12
        qc(k) = 8.0e-5
        qt(k) = qv(k) + qc(k)
        qc_bl_in(k) = 2.0e-6
        cldfra_bl_in(k) = 0.62
      end do
    case (12)
      ! Every argument that no admitted branch reads, moved off baseline.
      dt = 137.0
      ust = 1.9
      flqv = -0.03
      kpbl = 27
      do k = 1, nz
        qke(k) = 9.5 + 0.1 * real(k)
        qnc(k) = 3.3e9
        qni(k) = 7.7e6
        qnwfa(k) = 4.4e10
        qnifa(k) = 5.5e7
        qnbca(k) = 6.6e8
        sgm(k) = 0.37
        qc_bl_old(k) = 1.5e-3
        cldfra_bl_old(k) = 0.85
        chem1(k, 1) = -21.0
        chem1(k, 2) = 43.0
      end do
    end select

    do k = 1, nz
      vt(k) = vt_in(k)
      vq(k) = vq_in(k)
      qc_bl(k) = qc_bl_in(k)
      cldfra_bl(k) = cldfra_bl_in(k)
    end do

    call dmp_mf(kts, kte, dt, zw, dz, p, rho, momentum_opt, tke_opt, &
        scalar_opt, u, v, w, th, thl, thv, tk, qt, qv, qc, qke, &
        qnc, qni, qnwfa, qnifa, qnbca, exner, vt, vq, sgm, &
        ust, flt, fltv, flq, flqv, pblh, kpbl, dx, landsea, ts, &
        edmf_a, edmf_w, edmf_qt, edmf_thl, edmf_ent, edmf_qc, &
        s_aw, s_awthl, s_awqt, s_awqv, s_awqc, s_awu, s_awv, s_awqke, &
        s_awqnc, s_awqni, s_awqnwfa, s_awqnifa, s_awqnbca, &
        sub_thl, sub_sqv, sub_u, sub_v, &
        det_thl, det_sqv, det_sqc, det_u, det_v, &
        nchem, chem1, s_awchem, mix_chem, &
        qc_bl, cldfra_bl, qc_bl_old, cldfra_bl_old, &
        .true., .true., .false., .false., .false., .false., .false., &
        psig_shcu, maxwidth, ktop, maxmf, ztop, spp_pbl, rstoch)

    do k = 1, nz
      write(unit, '(A,",",I0)', advance='no') trim(names(c)), k
      write(unit, '(37(",",ES24.16E3))', advance='no') &
          dz(k), zw(k), zw(k+1), p(k), rho(k), u(k), v(k), w(k), th(k), &
          thl(k), thv(k), tk(k), qt(k), qv(k), qc(k), exner(k), qke(k), &
          qnc(k), qni(k), qnwfa(k), qnifa(k), qnbca(k), vt_in(k), &
          vq_in(k), sgm(k), rstoch(k), qc_bl_in(k), cldfra_bl_in(k), &
          qc_bl_old(k), cldfra_bl_old(k), dt, ust, flt, fltv, flq, flqv, &
          pblh
      write(unit, '(",",I0)', advance='no') kpbl
      write(unit, '(44(",",ES24.16E3))', advance='no') &
          dx, landsea, ts, psig_shcu, &
          edmf_a(k), edmf_w(k), edmf_qt(k), edmf_thl(k), edmf_ent(k), &
          edmf_qc(k), s_aw(k), s_awthl(k), s_awqt(k), s_awqv(k), &
          s_awqc(k), s_awu(k), s_awv(k), s_awqke(k), s_awqnc(k), &
          s_awqni(k), s_awqnwfa(k), s_awqnifa(k), s_awqnbca(k), &
          s_aw(nz+1), s_awthl(nz+1), s_awqt(nz+1), s_awqv(nz+1), &
          s_awqc(nz+1), s_awu(nz+1), s_awv(nz+1), &
          sub_thl(k), sub_sqv(k), sub_u(k), sub_v(k), det_thl(k), &
          det_sqv(k), det_sqc(k), det_u(k), det_v(k), &
          qc_bl(k), cldfra_bl(k), vt(k), vq(k), maxwidth
      write(unit, '(",",I0)', advance='no') ktop
      write(unit, '(2(",",ES24.16E3))') ztop, maxmf
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
        qv(kk) = max(qsurf + dq_ml * zm, 2.0e-4)
      else
        th(kk) = th_ml + dth_ml * zinv + inv_dth + up_lapse * (zm - zinv)
        qv(kk) = max(qsurf + dq_ml * zinv + dq_up * (zm - zinv), 2.0e-4)
      end if
      qc(kk) = 0.0
      qt(kk) = qv(kk) + qc(kk)
      thl(kk) = th(kk)
      tk(kk) = th(kk) * exner(kk)
      thv(kk) = th(kk) * (1.0 + 0.608 * qv(kk) - qc(kk))
      rho(kk) = p(kk) / (287.0 * tk(kk))
      u(kk) = u0 + ushear * zm
      v(kk) = v0 + 0.0008 * zm
    end do
    fltv = flt + flqv * 0.608 * ts
  end subroutine build_profile

end program run_mynn_dmp_mf_oracle
