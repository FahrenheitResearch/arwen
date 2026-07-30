program run_mynn_bl_driver_oracle
  ! Dump module_bl_mynn.F:360-1453 (mynn_bl_driver) from the unmodified pinned
  ! WRF v4.6.1 physics module.  This is the assembled scheme, not a leaf: the
  ! harness hands it a four-column block and lets the driver do the column
  ! extraction, the flux construction, the call ordering and the write-back.
  !
  ! Two calls are made over the same atmosphere.  The first has initflag=1,
  ! which runs the mym_initialize cold start at :658-857; the second has
  ! initflag=0 and consumes the state the first produced.  Both the incoming
  ! and the outgoing state are recorded per step, so each step is
  ! independently reproducible from its own rows.
  !
  ! Admitted option identity, matching the WRF registry defaults:
  ! bl_mynn_cloudpdf=2, bl_mynn_mixlength=1, bl_mynn_edmf=1,
  ! bl_mynn_edmf_mom=1, bl_mynn_edmf_tke=0, bl_mynn_mixscalars=0,
  ! bl_mynn_output=0, bl_mynn_cloudmix=1, bl_mynn_mixqt=0, icloud_bl=1,
  ! closure=2.6, bl_mynn_tkeadvect=.false., tke_budget=0, spp_pbl=0,
  ! mix_chem=.false., restart=.false., cycling=.false., FLAG_QC/FLAG_QI true
  ! and every other species flag false.
  !
  ! Note on Sh3D/Sm3D: WRF declares them intent(out) at :511 and then reads
  ! them back at :952 on every step after the first.  With gfortran and an
  ! explicit-shape array that is a pass-by-reference no-op, so the previous
  ! step's values survive; the port has to carry them as state for the same
  ! reason.  The CSV records them on both sides of each call so the behaviour
  ! is pinned rather than inferred.
  use module_bl_mynn, only: mynn_bl_driver
  implicit none

  integer, parameter :: ncol = 4, nz = 30
  integer, parameter :: ids = 1, ide = ncol + 1, jds = 1, jde = 2
  integer, parameter :: kds = 1, kde = nz + 1
  integer, parameter :: ims = 1, ime = ncol, jms = 1, jme = 1
  integer, parameter :: kms = 1, kme = nz
  integer, parameter :: its = 1, ite = ncol, jts = 1, jte = 1
  integer, parameter :: kts = 1, kte = nz
  integer, parameter :: nchem = 1, kdvel = 1, ndvel = 1
  integer, parameter :: nstep = 2
  character(len=32), parameter :: names(ncol) = [character(len=32) :: &
      'convective_land', 'marine_cumulus', 'stable_land', 'cloudy_deep']
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
  integer :: initflag
  logical, parameter :: restart = .false., cycling = .false.
  logical, parameter :: tkeadvect = .false.
  logical, parameter :: mix_chem = .false., enh_mix = .false.
  logical, parameter :: rrfs_sd = .false., smoke_dbg = .false.
  logical, parameter :: flag_qc = .true., flag_qi = .true.
  logical, parameter :: flag_qnc = .false., flag_qni = .false.
  logical, parameter :: flag_qs = .false., flag_qnwfa = .false.
  logical, parameter :: flag_qnifa = .false., flag_qnbca = .false.
  logical, parameter :: flag_ozone = .false.
  integer, parameter :: tke_budget = 0, bl_mynn_cloudpdf = 2
  integer, parameter :: bl_mynn_mixlength = 1, icloud_bl = 1
  integer, parameter :: bl_mynn_edmf = 1, bl_mynn_edmf_mom = 1
  integer, parameter :: bl_mynn_edmf_tke = 0, bl_mynn_mixscalars = 0
  integer, parameter :: bl_mynn_output = 0, bl_mynn_cloudmix = 1
  integer, parameter :: bl_mynn_mixqt = 0, spp_pbl = 0
  real, parameter :: closure = 2.6

  call get_command_argument(1, output_path)
  if (len_trim(output_path) == 0) then
    write(*, '(A)') 'usage: run_driver OUTPUT.csv'
    error stop 2
  end if
  open(newunit=unit, file=trim(output_path), status='new', action='write')
  write(unit, '(A)') 'case,step,k,initflag,delt,dx,znt,xland,ts,qsfc,ps,' // &
      'ust,ch,hfx,qfx,wspd,uoce,voce,dz,u,v,w,th,sqv3d,sqc3d,sqi3d,p,' //    &
      'exner,rho,t3d,qke_in,tsq_in,qsq_in,cov_in,el_in,sh_in,sm_in,' //      &
      'qc_bl_in,qi_bl_in,cldfra_bl_in,pblh_in,kpbl_in,rmol_in,rublten,' //   &
      'rvblten,rthblten,rqvblten,rqcblten,rqiblten,dozone,exch_h,' //        &
      'exch_m,qke,tsq,qsq,cov,el_pbl,sh3d,sm3d,qc_bl,qi_bl,cldfra_bl,' //    &
      'pblh,kpbl,rmol,maxwidth,maxmf,ztop_plume,ktop_plume'

  delt = 20.0
  ! ---- atmosphere: four columns, one block ----------------------------
  do c = 1, ncol
    dz1 = 50.0
    dzslope = 15.0
    th_ml = 301.0
    qsurf = 0.0168
    zinv = 1100.0
    dx(c) = 3000.0
    znt(c) = 0.15
    xland(c) = 1.0
    ts(c) = 302.3
    ust(c) = 0.35
    ch(c) = 0.01
    hfx(c) = 160.0
    qfx(c) = 1.15e-4
    uoce(c) = 0.0
    voce(c) = 0.0
    ps(c) = 100000.0
    select case (c)
    case (2)
      ! Marine: a weaker heat flux, a stronger moisture flux, an ocean
      ! current, and the water branch of GET_PBLH.
      xland(c) = 2.0
      znt(c) = 0.0002
      ts(c) = 301.4
      hfx(c) = 58.0
      qfx(c) = 1.5e-4
      qsurf = 0.0175
      zinv = 900.0
      uoce(c) = 0.35
      voce(c) = -0.20
      dx(c) = 6000.0
    case (3)
      ! Stable land: a downward heat flux, so fltv2 stays under the DMP_mf
      ! activation threshold and the plume model deactivates.
      th_ml = 299.0
      ts(c) = 297.4
      hfx(c) = -34.0
      qfx(c) = 1.2e-6
      qsurf = 0.0080
      zinv = 250.0
      ust(c) = 0.16
    case (4)
      ! Deep and cloudy: resolved liquid inside the mixed layer, ice above
      ! the inversion, a fine grid.
      hfx(c) = 220.0
      qfx(c) = 1.6e-4
      qsurf = 0.0165
      zinv = 2000.0
      ts(c) = 303.0
      dx(c) = 1000.0
      dz1 = 30.0
      dzslope = 12.0
    end select
    zw = 0.0
    do k = 1, nz
      dz(c, k) = dz1 + dzslope * real(k - 1)
      zm = zw + 0.5 * dz(c, k)
      p(c, k) = 100000.0 * exp(-zm / 8500.0)
      exner(c, k) = (p(c, k) / 100000.0) ** (287.0 / (7.0 * 287.0 / 2.0))
      if (zm <= zinv) then
        th(c, k) = th_ml - 0.0006 * zm
        sqv3d(c, k) = max(qsurf - 3.0e-7 * zm, 2.0e-4)
      else
        th(c, k) = th_ml - 0.0006 * zinv + 6.0 + 0.0045 * (zm - zinv)
        sqv3d(c, k) = max(qsurf - 3.0e-7 * zinv - 6.0e-6 * (zm - zinv), &
            2.0e-4)
      end if
      sqc3d(c, k) = 0.0
      sqi3d(c, k) = 0.0
      if (c == 4) then
        if (zm > 300.0 .and. zm < 1000.0) then
          sqc3d(c, k) = max(2.4e-4 - abs(zm - 650.0) * 3.0e-7, 5.0e-6)
        end if
        if (zm > 2400.0) then
          sqi3d(c, k) = max(4.0e-5 - (zm - 2600.0) * 8.0e-9, 3.0e-6)
        end if
      end if
      sqs3d(c, k) = 0.0
      t3d(c, k) = th(c, k) * exner(c, k)
      rho(c, k) = p(c, k) / (287.0 * t3d(c, k))
      u(c, k) = 3.0 + 0.0015 * zm
      v(c, k) = -1.0 + 0.0008 * zm
      w(c, k) = 0.02
      qnc(c, k) = 0.0
      qni(c, k) = 0.0
      qnwfa(c, k) = 0.0
      qnifa(c, k) = 0.0
      qnbca(c, k) = 0.0
      ozone(c, k) = 0.0
      rthraten(c, k) = 0.0
      pattern_spp_pbl(c, k) = 0.0
      chem3d(c, k, 1) = 0.0
      ! State arrays.  A cold start zeroes them inside the driver; seeding
      ! them with a distinctive nonzero pattern is what proves it does.
      qke(c, k) = 0.7 + 0.01 * real(k)
      qke_adv(c, k) = 0.0
      sh3d(c, k) = 0.31 + 0.011 * real(k - 1)
      sm3d(c, k) = 0.47 - 0.009 * real(k - 1)
      tsq(c, k) = 1.0e-3
      qsq(c, k) = 2.0e-7
      cov(c, k) = -8.0e-6
      el_pbl(c, k) = 30.0 + 2.0 * real(k)
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
        write(unit, '(37(",",ES24.16E3))', advance='no')                   &
            delt, dx(c), znt(c), xland(c), ts(c), qsfc(c), ps(c), ust(c),  &
            ch(c), hfx(c), qfx(c), wspd(c), uoce(c), voce(c), dz(c, k),    &
            u(c, k), v(c, k), w(c, k), th(c, k), sqv3d(c, k),              &
            sqc3d(c, k), sqi3d(c, k), p(c, k), exner(c, k), rho(c, k),     &
            t3d(c, k), qke_in(c, k), tsq_in(c, k), qsq_in(c, k),           &
            cov_in(c, k), el_in(c, k), sh_in(c, k), sm_in(c, k),           &
            qcbl_in(c, k), qibl_in(c, k), cfbl_in(c, k), pblh_in(c)
        write(unit, '(",",I0)', advance='no') kpbl_in(c)
        write(unit, '(21(",",ES24.16E3))', advance='no')                   &
            rmol_in(c), rublten(c, k), rvblten(c, k), rthblten(c, k),      &
            rqvblten(c, k), rqcblten(c, k), rqiblten(c, k),                &
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
  end do
  close(unit)
end program run_mynn_bl_driver_oracle
