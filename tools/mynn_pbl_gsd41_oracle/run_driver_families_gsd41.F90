program run_mynn_bl_driver_families_gsd41_oracle
  ! Multi-step oracle for the GSD MYNN v4.1 mynn_bl_driver of the operational
  ! WRF 3.9 fork (NOAA-EMC/HRRR tag v4.1.21,
  ! sorc/hrrr_wrfarw.fd/WRFV3.9/phys/module_bl_mynn.F, unmodified, or with the
  ! one :995 line build.sh patches), over the same six column families as
  ! tools/mynn_pbl_wrf461_oracle/run_driver_families.F90: convective day,
  ! stable night with dew, marine stratocumulus, shallow cumulus, a
  ! sub-freezing cold pool with fog, ice and frost, and high wind.
  !
  ! HRRR's namelist identity (hrrr_wrf.nl): bl_mynn_mixlength=2,
  ! bl_mynn_cloudpdf=2, bl_mynn_edmf=1, bl_mynn_edmf_mom=1,
  ! bl_mynn_edmf_tke=0, bl_mynn_tkebudget=0, bl_mynn_tkeadvect=.false.,
  ! icloud_bl=1, grav_settling=0; Registry defaults bl_mynn_cloudmix=1,
  ! bl_mynn_mixqt=0, bl_mynn_mixscalars=0; spp_pbl=0; restart and cycling
  ! false.  FLAG_QC and FLAG_QI true, the number flags false.
  !
  ! The fork driver takes dx as one scalar per call, so each family is its
  ! own one-column call.  The fork reads mixing ratios (qv, qc, qi) and
  ! returns mixing-ratio tendencies; between steps the harness applies
  ! them (x = x + delt*tend, species floored at 0) together with u, v and
  ! th, and recomputes t3d = th*exner and rho = p/(r_d*t3d).  Step 1 is the
  ! cold start (initflag=1), steps 2..N warm.  The driver's carried state
  ! (QKE, Tsq, Qsq, Cov, el_pbl, Sh3D, QC_BL, CLDFRA_BL, PBLH, KPBL) is
  ! passed straight back; rmol is the surface layer's and is held.
  !
  ! Arguments: OUTPUT.csv [NSTEP]   (default 12)
  use module_bl_mynn, only: mynn_bl_driver
  implicit none

  integer, parameter :: nfam = 6, nz = 30
  integer, parameter :: ids = 1, ide = 2, jds = 1, jde = 2
  integer, parameter :: kds = 1, kde = nz + 1
  integer, parameter :: ims = 1, ime = 1, jms = 1, jme = 1
  integer, parameter :: kms = 1, kme = nz
  integer, parameter :: its = 1, ite = 1, jts = 1, jte = 1
  integer, parameter :: kts = 1, kte = nz
  character(len=32), parameter :: names(nfam) = [character(len=32) :: &
      'convective_day', 'stable_night', 'stratocumulus', 'shallow_cumulus', &
      'cold_pool', 'high_wind']
  character(len=1024) :: output_path
  character(len=32) :: nstep_arg
  character(len=1024) :: controls_path
  integer :: controls_unit, bl_mynn_mixlength = 2
  logical :: cycling = .false.
  real :: initial_qke = 0.0, initial_qc_bl = 0.0, initial_cldfra_bl = 0.0
  real :: heat_scale = 1.0, vapor_scale = 1.0, wind_scale = 1.0
  namelist /review/ bl_mynn_mixlength, cycling, initial_qke, initial_qc_bl, &
      initial_cldfra_bl, heat_scale, vapor_scale, wind_scale
  integer :: nstep = 12
  integer :: c, k, s, unit, initflag

  ! one family's column, as the driver sees it
  real, dimension(ims:ime, kms:kme, jms:jme) :: dz, u, v, w, th, qv, qc, qi, &
      qnc, qni, qnwfa, qnifa, p, exner, rho, t3d, qke, qke_adv, tsq, qsq, &
      cov, rublten, rvblten, rthblten, rqvblten, rqcblten, rqiblten, &
      rqncblten, rqniblten, rqnwfablten, rqnifablten, exch_h, exch_m, &
      el_pbl, dqke, qwt, qshear, qbuoy, qdiss, sh3d, qc_bl, cldfra_bl, &
      edmf_a, edmf_w, edmf_qt, edmf_thl, edmf_ent, edmf_qc, &
      pattern_spp_pbl, rthraten
  real, dimension(ims:ime, jms:jme) :: znt, xland, ts, qsfc, qcg, ps, ust, &
      ch, hfx, qfx, rmol, wspd, uoce, voce, vdfg, pblh, wstar, delta, &
      maxmf, mf_at_base
  integer, dimension(ims:ime, jms:jme) :: kpbl, nupdraft, ktop_plume, &
      kbot_shallow
  real :: dx

  ! the six families' state across steps
  real, dimension(nfam, nz) :: fdz, fu, fv, fw, fth, fqv, fqc, fqi, fp, &
      fexner, frho, ft3d, fqke, ftsq, fqsq, fcov, fel, fsh, fqcbl, fcfbl, &
      frthraten
  real, dimension(nfam) :: fdx, fznt, fxland, fts, fqsfc, fps, fust, fch, &
      fhfx, fqfx, fwspd, fpblh, frmol
  integer, dimension(nfam) :: fkpbl

  real :: delt, zw, zm, zinv, th_ml, qsurf, dz1, dzslope
  real :: dthml, dthinv, dqinv, gam, qt, es, qs, tk, uref, ushear, cldtop
  real :: sqv, sqc, sqi

  call get_command_argument(1, output_path)
  call get_command_argument(2, nstep_arg)
  if (len_trim(nstep_arg) > 0) read(nstep_arg, *) nstep
  call get_command_argument(3, controls_path)
  if (len_trim(controls_path) > 0) then
    open(newunit=controls_unit, file=trim(controls_path), status='old', action='read')
    read(controls_unit, nml=review)
    close(controls_unit)
  end if
  if (len_trim(output_path) == 0) then
    write(*, '(A)') 'usage: run_driver_families_gsd41 OUTPUT.csv [NSTEP]'
    error stop 2
  end if
  open(newunit=unit, file=trim(output_path), status='new', action='write')
  write(unit, '(A)') 'case,step,k,initflag,delt,dx,znt,xland,ts,qsfc,ps,' // &
      'ust,ch,hfx,qfx,wspd,rmol,dz,u,v,w,th,qv,qc,qi,p,exner,rho,t3d,' //     &
      'rthraten,qke_in,tsq_in,qsq_in,cov_in,el_in,sh_in,qc_bl_in,' //         &
      'cldfra_bl_in,pblh_in,kpbl_in,rublten,rvblten,rthblten,rqvblten,' //    &
      'rqcblten,rqiblten,exch_h,exch_m,qke,tsq,qsq,cov,el_pbl,sh3d,' //       &
      'qc_bl,cldfra_bl,pblh,kpbl,maxmf,ktop_plume'

  delt = 20.0
  ! ---- atmosphere: identical to run_driver_families.F90, mixing ratios --
  do c = 1, nfam
    dz1 = 50.0
    dzslope = 15.0
    fdx(c) = 3000.0
    fxland(c) = 1.0
    fps(c) = 100000.0
    dthml = 0.0
    dthinv = 4.0
    dqinv = 2.0e-3
    gam = 0.004
    cldtop = -1.0
    select case (c)
    case (1)
      th_ml = 303.0; qsurf = 0.0120; zinv = 1600.0
      fts(c) = 308.0; fust(c) = 0.45; fhfx(c) = 320.0; fqfx(c) = 9.0e-5
      fznt(c) = 0.10; fch(c) = 0.012; uref = 4.0; ushear = 0.0015
      frmol(c) = -0.01
    case (2)
      th_ml = 290.0; qsurf = 0.0105; zinv = 250.0; dthml = 0.016
      dthinv = 1.0; dqinv = 5.0e-4
      fts(c) = 286.5; fust(c) = 0.12; fhfx(c) = -38.0; fqfx(c) = -1.2e-5
      fznt(c) = 0.50; fch(c) = 0.004; uref = 2.0; ushear = 0.012
      frmol(c) = 0.02
    case (3)
      fxland(c) = 2.0
      th_ml = 289.0; qsurf = 0.0102; zinv = 850.0; dthinv = 9.0
      dqinv = 5.0e-3
      fts(c) = 289.5; fust(c) = 0.25; fhfx(c) = 12.0; fqfx(c) = 3.5e-5
      fznt(c) = 0.0002; fch(c) = 0.010; uref = 7.0; ushear = 0.001
      cldtop = 850.0; fdx(c) = 6000.0; frmol(c) = -0.002
    case (4)
      th_ml = 300.0; qsurf = 0.0130; zinv = 1100.0; dthinv = 1.0
      dqinv = 1.5e-3; gam = 0.0035
      fts(c) = 303.5; fust(c) = 0.35; fhfx(c) = 190.0; fqfx(c) = 2.1e-4
      fznt(c) = 0.15; fch(c) = 0.011; uref = 5.0; ushear = 0.0010
      cldtop = 2200.0; frmol(c) = -0.012
    case (5)
      th_ml = 268.5; qsurf = 0.0029; zinv = 400.0; dthml = 0.006
      dthinv = 5.0; dqinv = 8.0e-4
      fts(c) = 266.0; fust(c) = 0.08; fhfx(c) = -12.0; fqfx(c) = -3.0e-6
      fznt(c) = 0.05; fch(c) = 0.003; uref = 1.5; ushear = 0.004
      fps(c) = 98000.0; cldtop = 300.0; frmol(c) = 0.05
    case (6)
      th_ml = 295.0; qsurf = 0.0090; zinv = 1300.0; dthinv = 2.0
      dqinv = 1.0e-3
      fts(c) = 295.6; fust(c) = 0.95; fhfx(c) = 25.0; fqfx(c) = 6.0e-5
      fznt(c) = 0.40; fch(c) = 0.020; uref = 18.0; ushear = 0.006
      fdx(c) = 1000.0; dz1 = 30.0; dzslope = 12.0; frmol(c) = -0.0003
    end select
    zw = 0.0
    do k = 1, nz
      fdz(c, k) = dz1 + dzslope * real(k - 1)
      zm = zw + 0.5 * fdz(c, k)
      fp(c, k) = fps(c) * exp(-zm / 8500.0)
      fexner(c, k) = (fp(c, k) / 100000.0) ** (287.0 / (7.0 * 287.0 / 2.0))
      if (zm <= zinv) then
        fth(c, k) = th_ml + dthml * zm
        qt = qsurf - 1.0e-7 * zm
      else
        fth(c, k) = th_ml + dthml * zinv + dthinv + gam * (zm - zinv)
        qt = max(qsurf - 1.0e-7 * zinv - dqinv - 2.5e-6 * (zm - zinv), &
            1.0e-4)
      end if
      tk = fth(c, k) * fexner(c, k)
      if (tk > 253.0) then
        es = 611.2 * exp(17.67 * (tk - 273.15) / (tk - 29.65))
      else
        es = 611.2 * exp(21.8745584 * (tk - 273.16) / (tk - 7.66))
      end if
      qs = 0.622 * es / (fp(c, k) - 0.378 * es)
      sqc = 0.0
      sqi = 0.0
      if (zm <= cldtop .and. qt > qs) then
        sqv = qs
        if (tk > 263.0) then
          sqc = qt - qs
        else
          sqc = 0.5 * (qt - qs)
          sqi = 0.5 * (qt - qs)
        end if
      else
        sqv = min(qt, qs)
      end if
      if (c == 4 .and. zm > zinv .and. zm < cldtop) sqc = 4.0e-5
      if (c == 5 .and. zm > 1500.0 .and. zm < 3500.0) sqi = 2.0e-5
      ! the fork's inputs are mixing ratios
      fqv(c, k) = sqv / (1.0 - sqv)
      fqc(c, k) = sqc / (1.0 - sqv)
      fqi(c, k) = sqi / (1.0 - sqv)
      ft3d(c, k) = fth(c, k) * fexner(c, k)
      frho(c, k) = fp(c, k) / (287.0 * ft3d(c, k))
      fu(c, k) = uref + ushear * zm
      if (c == 2 .and. zm < 600.0) fu(c, k) = uref + 9.0 * zm / 600.0
      fv(c, k) = -1.0 + 0.0008 * zm
      fw(c, k) = 0.0
      if (c == 4) fw(c, k) = 0.05
      if (c == 3) fw(c, k) = -0.004
      frthraten(c, k) = 0.0
      if (c == 3 .and. zm > zinv - 120.0 .and. zm <= zinv) then
        frthraten(c, k) = -1.5e-4
      end if
      fqke(c, k) = initial_qke
      ftsq(c, k) = 0.0
      fqsq(c, k) = 0.0
      fcov(c, k) = 0.0
      fel(c, k) = 0.0
      fsh(c, k) = 0.0
      fqcbl(c, k) = initial_qc_bl
      fcfbl(c, k) = initial_cldfra_bl
      zw = zw + fdz(c, k)
    end do
    fqsfc(c) = fqv(c, 1) / (1.0 + fqv(c, 1))
    fwspd(c) = max(sqrt(fu(c, 1)**2 + fv(c, 1)**2), 1.0)
    fpblh(c) = 0.0
    fkpbl(c) = 1
  end do
  fhfx = fhfx * heat_scale
  fqfx = fqfx * vapor_scale
  fu = fu * wind_scale
  fv = fv * wind_scale
  fwspd = fwspd * wind_scale

  do s = 1, nstep
    initflag = 0
    if (s == 1) initflag = 1
    do c = 1, nfam
      ! load the family into the driver's one-column block
      dz(1, :, 1) = fdz(c, :); u(1, :, 1) = fu(c, :); v(1, :, 1) = fv(c, :)
      w(1, :, 1) = fw(c, :); th(1, :, 1) = fth(c, :)
      qv(1, :, 1) = fqv(c, :); qc(1, :, 1) = fqc(c, :)
      qi(1, :, 1) = fqi(c, :)
      qnc = 0.0; qni = 0.0; qnwfa = 0.0; qnifa = 0.0
      p(1, :, 1) = fp(c, :); exner(1, :, 1) = fexner(c, :)
      rho(1, :, 1) = frho(c, :); t3d(1, :, 1) = ft3d(c, :)
      qke(1, :, 1) = fqke(c, :); qke_adv = 0.0
      tsq(1, :, 1) = ftsq(c, :); qsq(1, :, 1) = fqsq(c, :)
      cov(1, :, 1) = fcov(c, :); el_pbl(1, :, 1) = fel(c, :)
      sh3d(1, :, 1) = fsh(c, :)
      qc_bl(1, :, 1) = fqcbl(c, :); cldfra_bl(1, :, 1) = fcfbl(c, :)
      rthraten(1, :, 1) = frthraten(c, :)
      pattern_spp_pbl = 0.0
      rublten = 0.0; rvblten = 0.0; rthblten = 0.0; rqvblten = 0.0
      rqcblten = 0.0; rqiblten = 0.0; rqncblten = 0.0; rqniblten = 0.0
      rqnwfablten = 0.0; rqnifablten = 0.0
      exch_h = 0.0; exch_m = 0.0
      dqke = 0.0; qwt = 0.0; qshear = 0.0; qbuoy = 0.0; qdiss = 0.0
      edmf_a = 0.0; edmf_w = 0.0; edmf_qt = 0.0; edmf_thl = 0.0
      edmf_ent = 0.0; edmf_qc = 0.0
      dx = fdx(c)
      znt(1, 1) = fznt(c); xland(1, 1) = fxland(c); ts(1, 1) = fts(c)
      qsfc(1, 1) = fqsfc(c); qcg(1, 1) = 0.0; ps(1, 1) = fps(c)
      ust(1, 1) = fust(c); ch(1, 1) = fch(c); hfx(1, 1) = fhfx(c)
      qfx(1, 1) = fqfx(c); rmol(1, 1) = frmol(c); wspd(1, 1) = fwspd(c)
      uoce = 0.0; voce = 0.0; vdfg = 0.0
      pblh(1, 1) = fpblh(c); kpbl(1, 1) = fkpbl(c)
      wstar = 0.0; delta = 0.0
      nupdraft = 0; ktop_plume = 0; kbot_shallow = 0
      maxmf = 0.0; mf_at_base = 0.0

      call mynn_bl_driver(                                                 &
          initflag, .false., cycling,                                     &
          0,                                                               &
          delt, dz, dx, znt,                                               &
          u, v, w, th, qv, qc, qi, qnc, qni,                               &
          qnwfa, qnifa,                                                    &
          p, exner, rho, t3d,                                              &
          xland, ts, qsfc, qcg, ps,                                        &
          ust, ch, hfx, qfx, rmol, wspd,                                   &
          uoce, voce,                                                      &
          vdfg,                                                            &
          qke,                                                             &
          qke_adv, .false.,                                                &
          tsq, qsq, cov,                                                   &
          rublten, rvblten, rthblten,                                      &
          rqvblten, rqcblten, rqiblten,                                    &
          rqncblten, rqniblten,                                            &
          rqnwfablten, rqnifablten,                                        &
          exch_h, exch_m,                                                  &
          pblh, kpbl,                                                      &
          el_pbl,                                                          &
          dqke, qwt, qshear, qbuoy, qdiss,                                 &
          wstar, delta,                                                    &
          0,                                                               &
          2, sh3d,                                                         &
          bl_mynn_mixlength,                                               &
          1, qc_bl, cldfra_bl,                                             &
          1,                                                               &
          1, 0,                                                            &
          0,                                                               &
          1, 0,                                                            &
          edmf_a, edmf_w, edmf_qt,                                         &
          edmf_thl, edmf_ent, edmf_qc,                                     &
          nupdraft, maxmf, ktop_plume,                                     &
          kbot_shallow, mf_at_base,                                        &
          0, pattern_spp_pbl,                                              &
          rthraten,                                                        &
          .true., .true., .false.,                                         &
          .false., .false., .false.,                                       &
          ids, ide, jds, jde, kds, kde,                                    &
          ims, ime, jms, jme, kms, kme,                                    &
          its, ite, jts, jte, kts, kte)

      do k = 1, nz
        write(unit, '(A,",",I0,",",I0,",",I0)', advance='no')              &
            trim(names(c)), s, k, initflag
        write(unit, '(28(",",ES24.16E3))', advance='no')                   &
            delt, fdx(c), fznt(c), fxland(c), fts(c), fqsfc(c), fps(c),    &
            fust(c), fch(c), fhfx(c), fqfx(c), fwspd(c), frmol(c),         &
            fdz(c, k), fu(c, k), fv(c, k), fw(c, k), fth(c, k),            &
            fqv(c, k), fqc(c, k), fqi(c, k), fp(c, k), fexner(c, k),       &
            frho(c, k), ft3d(c, k), frthraten(c, k), fqke(c, k),           &
            ftsq(c, k)
        write(unit, '(7(",",ES24.16E3))', advance='no')                    &
            fqsq(c, k), fcov(c, k), fel(c, k), fsh(c, k), fqcbl(c, k),     &
            fcfbl(c, k), fpblh(c)
        write(unit, '(",",I0)', advance='no') fkpbl(c)
        write(unit, '(17(",",ES24.16E3))', advance='no')                   &
            rublten(1, k, 1), rvblten(1, k, 1), rthblten(1, k, 1),         &
            rqvblten(1, k, 1), rqcblten(1, k, 1), rqiblten(1, k, 1),       &
            exch_h(1, k, 1), exch_m(1, k, 1), qke(1, k, 1),                &
            tsq(1, k, 1), qsq(1, k, 1), cov(1, k, 1), el_pbl(1, k, 1),     &
            sh3d(1, k, 1), qc_bl(1, k, 1), cldfra_bl(1, k, 1), pblh(1, 1)
        write(unit, '(",",I0)', advance='no') kpbl(1, 1)
        write(unit, '(",",ES24.16E3)', advance='no') maxmf(1, 1)
        write(unit, '(",",I0)') ktop_plume(1, 1)
      end do

      ! carry the driver's state and advance the family
      fqke(c, :) = qke(1, :, 1); ftsq(c, :) = tsq(1, :, 1)
      fqsq(c, :) = qsq(1, :, 1); fcov(c, :) = cov(1, :, 1)
      fel(c, :) = el_pbl(1, :, 1); fsh(c, :) = sh3d(1, :, 1)
      fqcbl(c, :) = qc_bl(1, :, 1); fcfbl(c, :) = cldfra_bl(1, :, 1)
      fpblh(c) = pblh(1, 1); fkpbl(c) = kpbl(1, 1)
      do k = 1, nz
        fu(c, k) = fu(c, k) + delt * rublten(1, k, 1)
        fv(c, k) = fv(c, k) + delt * rvblten(1, k, 1)
        fth(c, k) = fth(c, k) + delt * rthblten(1, k, 1)
        fqv(c, k) = max(fqv(c, k) + delt * rqvblten(1, k, 1), 0.0)
        fqc(c, k) = max(fqc(c, k) + delt * rqcblten(1, k, 1), 0.0)
        fqi(c, k) = max(fqi(c, k) + delt * rqiblten(1, k, 1), 0.0)
        ft3d(c, k) = fth(c, k) * fexner(c, k)
        frho(c, k) = fp(c, k) / (287.0 * ft3d(c, k))
      end do
    end do
  end do
  close(unit)
end program run_mynn_bl_driver_families_gsd41_oracle
