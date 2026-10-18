program run_mynn_wrapper_columns
  ! Column oracle for WRF v4.6.1's MYNN PBL entry, mynnedmf_wrapper_run
  ! (phys/module_bl_mynn_wrapper.F), the routine module_pbl_driver.F calls
  ! for bl_pbl_physics = 5.  Every module comes unmodified from the WRF
  ! source tree; there is no stub.
  !
  ! Input (written by wrapper_columns.py gen, raw little-endian words):
  !   int32 magic, version, ncol, nz, nstep, ngroup
  !   per group: int32 start (0-based), count, flag_qs; real32 dx, delt
  !   real32 scalars(ncol) for xland ts qsfc ps ust ch hfx qfx wspd znt
  !   real32 profiles(nz, ncol) for dz u v w th qv qc qi qs p exner rho t3d
  !          rthraten qnc qni qnwfa qnifa spp  (mixing ratios, as WRF
  !          holds them; number concentrations per kg; the SPP pattern)
  ! Arguments: INPUT OUTPUT MIXLENGTH [VARIANT]
  !   VARIANT base (default): bl_mynn_mixscalars = 0, spp_pbl = 0, the
  !     number-concentration flags off (every microphysics without them).
  !   VARIANT mixscalars: bl_mynn_mixscalars = 1 with FLAG_QNC, FLAG_QNI,
  !     FLAG_QNWFA, FLAG_QNIFA and FLAG_QNBCA on, as WRF passes them for
  !     mp_physics = 28 (Registry.EM_COMMON thompsonaero package); qnbca is
  !     held at zero because WOOF's mp = 28 carries no black-carbon number.
  !   VARIANT spp: spp_pbl = 1 with the pattern from the input file.
  !
  ! Each step calls the wrapper once per group (one dxc and one delt per
  ! call, as WRF does), with WRF's physics memory layout: kms:kme =
  ! 1:nz+1, kde = nz+1, kte = nz.  The unused kme slot of every input array
  ! is a quiet NaN, so any read of it would show in the outputs.  Step 1 is
  ! the cold start (initflag = 1).  Between steps the harness applies the
  ! wrapper's own tendencies, x = x + delt*tend (one fp32 multiply and one
  ! add), floors the water species at 0, and recomputes t3d = th*exner and
  ! rho = p/(287*t3d).  Snow has no tendency in WRF (RQSBLTEN is a dummy in
  ! the wrapper) and is held.  The carried MYNN state is passed straight
  ! back.  Output: per step, a marker word, then the step's incoming state
  ! and the wrapper's outputs, in the order wrapper_columns.read_output
  ! names them.
  use, intrinsic :: ieee_arithmetic, only: ieee_value, ieee_quiet_nan
  use module_bl_mynn_wrapper, only: mynnedmf_wrapper_run
  implicit none

  integer, parameter :: magic_expected = int(z'4D594E57')
  integer, parameter :: step_marker = int(z'53544550')
  character(len=1024) :: in_path, out_path, arg
  integer :: uin, uout, magic, version, ncol, nz, nstep, ngroup
  integer :: mixlength, s, g, c0, n, k, variant
  integer, allocatable :: gstart(:), gcount(:), gqs(:)
  real, allocatable :: gdx(:), gdelt(:)
  real, allocatable :: xland(:), ts(:), qsfc(:), ps(:), ust(:), ch(:)
  real, allocatable :: hfx(:), qfx(:), wspd(:), znt(:)
  real, allocatable :: dz(:,:), u(:,:), v(:,:), w(:,:), th(:,:), qv(:,:)
  real, allocatable :: qc(:,:), qi(:,:), qs(:,:), p(:,:), exner(:,:)
  real, allocatable :: rho(:,:), t3d(:,:), rthraten(:,:)
  real, allocatable :: qnc(:,:), qni(:,:), qnwfa(:,:), qnifa(:,:), spp(:,:)
  real, allocatable :: rqnc(:,:), rqni(:,:), rqnwfa(:,:), rqnifa(:,:)
  ! carried state
  real, allocatable :: qke(:,:), tsq(:,:), qsq(:,:), cov(:,:), el(:,:)
  real, allocatable :: sh(:,:), sm(:,:), qcbl(:,:), qibl(:,:), cfbl(:,:)
  real, allocatable :: exh(:,:), exm(:,:)
  real, allocatable :: pblh(:), rmol(:), maxwidth(:), maxmf(:), ztop(:)
  integer, allocatable :: kpbl(:), ktop(:)
  ! outputs
  real, allocatable :: ru(:,:), rv(:,:), rth(:,:), rqv(:,:), rqc(:,:)
  real, allocatable :: rqi(:,:)
  real :: nan
  real :: delt
  logical :: flag_qs

  call get_command_argument(1, in_path)
  call get_command_argument(2, out_path)
  call get_command_argument(3, arg)
  if (len_trim(in_path) == 0 .or. len_trim(out_path) == 0 .or. &
      len_trim(arg) == 0) then
    write(*, '(A)') 'usage: run_wrapper_columns INPUT OUTPUT MIXLENGTH'
    error stop 2
  end if
  read(arg, *) mixlength
  call get_command_argument(4, arg)
  select case (trim(arg))
  case ('', 'base')
    variant = 0
  case ('mixscalars')
    variant = 1
  case ('spp')
    variant = 2
  case default
    error stop 4
  end select
  nan = ieee_value(nan, ieee_quiet_nan)

  open(newunit=uin, file=trim(in_path), access='stream', &
       form='unformatted', status='old', action='read')
  read(uin) magic, version, ncol, nz, nstep, ngroup
  if (magic /= magic_expected .or. version /= 1) error stop 3
  allocate(gstart(ngroup), gcount(ngroup), gqs(ngroup), gdx(ngroup), &
           gdelt(ngroup))
  do g = 1, ngroup
    read(uin) gstart(g), gcount(g), gqs(g), gdx(g), gdelt(g)
  end do
  allocate(xland(ncol), ts(ncol), qsfc(ncol), ps(ncol), ust(ncol), &
           ch(ncol), hfx(ncol), qfx(ncol), wspd(ncol), znt(ncol))
  read(uin) xland
  read(uin) ts
  read(uin) qsfc
  read(uin) ps
  read(uin) ust
  read(uin) ch
  read(uin) hfx
  read(uin) qfx
  read(uin) wspd
  read(uin) znt
  allocate(dz(nz,ncol), u(nz,ncol), v(nz,ncol), w(nz,ncol), th(nz,ncol), &
           qv(nz,ncol), qc(nz,ncol), qi(nz,ncol), qs(nz,ncol), p(nz,ncol), &
           exner(nz,ncol), rho(nz,ncol), t3d(nz,ncol), rthraten(nz,ncol))
  read(uin) dz
  read(uin) u
  read(uin) v
  read(uin) w
  read(uin) th
  read(uin) qv
  read(uin) qc
  read(uin) qi
  read(uin) qs
  read(uin) p
  read(uin) exner
  read(uin) rho
  read(uin) t3d
  read(uin) rthraten
  allocate(qnc(nz,ncol), qni(nz,ncol), qnwfa(nz,ncol), qnifa(nz,ncol), &
           spp(nz,ncol))
  read(uin) qnc
  read(uin) qni
  read(uin) qnwfa
  read(uin) qnifa
  read(uin) spp
  close(uin)

  allocate(qke(nz,ncol), tsq(nz,ncol), qsq(nz,ncol), cov(nz,ncol), &
           el(nz,ncol), sh(nz,ncol), sm(nz,ncol), qcbl(nz,ncol), &
           qibl(nz,ncol), cfbl(nz,ncol), exh(nz,ncol), exm(nz,ncol))
  allocate(ru(nz,ncol), rv(nz,ncol), rth(nz,ncol), rqv(nz,ncol), &
           rqc(nz,ncol), rqi(nz,ncol))
  allocate(rqnc(nz,ncol), rqni(nz,ncol), rqnwfa(nz,ncol), rqnifa(nz,ncol))
  rqnc = 0.0; rqni = 0.0; rqnwfa = 0.0; rqnifa = 0.0
  allocate(pblh(ncol), rmol(ncol), maxwidth(ncol), maxmf(ncol), &
           ztop(ncol), kpbl(ncol), ktop(ncol))
  qke = 0.0; tsq = 0.0; qsq = 0.0; cov = 0.0; el = 0.0; sh = 0.0
  sm = 0.0; qcbl = 0.0; qibl = 0.0; cfbl = 0.0; exh = 0.0; exm = 0.0
  ru = 0.0; rv = 0.0; rth = 0.0; rqv = 0.0; rqc = 0.0; rqi = 0.0
  pblh = 0.0; rmol = 0.0; maxwidth = 0.0; maxmf = 0.0; ztop = 0.0
  kpbl = 1; ktop = 0

  open(newunit=uout, file=trim(out_path), access='stream', &
       form='unformatted', status='replace', action='write')
  write(uout) magic_expected, 1, ncol, nz, nstep, ngroup

  do s = 1, nstep
    ! ---- the step's incoming state --------------------------------------
    write(uout) step_marker
    write(uout) u, v, th, qv, qc, qi, qs, t3d, rho, qnc, qni, qnwfa, qnifa
    write(uout) qke, tsq, qsq, cov, el, sh, sm, qcbl, qibl, cfbl, exh, exm
    write(uout) pblh, rmol, ust, hfx, qfx, wspd, znt
    write(uout) kpbl
    do g = 1, ngroup
      c0 = gstart(g) + 1
      n = gcount(g)
      flag_qs = gqs(g) /= 0
      call one_group(c0, n, gdx(g), gdelt(g), flag_qs, merge(1, 0, s == 1))
    end do
    ! ---- outputs ---------------------------------------------------------
    write(uout) ru, rv, rth, rqv, rqc, rqi, exh, exm
    write(uout) rqnc, rqni, rqnwfa, rqnifa
    write(uout) qke, tsq, qsq, cov, el, sh, sm, qcbl, qibl, cfbl
    write(uout) pblh, rmol, maxwidth, maxmf, ztop, ust, hfx, qfx, wspd, znt
    write(uout) kpbl, ktop
    ! ---- advance with the wrapper's own tendencies -----------------------
    do g = 1, ngroup
      c0 = gstart(g) + 1
      n = gcount(g)
      delt = gdelt(g)
      do k = 1, nz
        u(k, c0:c0+n-1) = u(k, c0:c0+n-1) + delt * ru(k, c0:c0+n-1)
        v(k, c0:c0+n-1) = v(k, c0:c0+n-1) + delt * rv(k, c0:c0+n-1)
        th(k, c0:c0+n-1) = th(k, c0:c0+n-1) + delt * rth(k, c0:c0+n-1)
        qv(k, c0:c0+n-1) = max(qv(k, c0:c0+n-1) + delt * rqv(k, c0:c0+n-1), 0.0)
        qc(k, c0:c0+n-1) = max(qc(k, c0:c0+n-1) + delt * rqc(k, c0:c0+n-1), 0.0)
        qi(k, c0:c0+n-1) = max(qi(k, c0:c0+n-1) + delt * rqi(k, c0:c0+n-1), 0.0)
        qnc(k, c0:c0+n-1) = max(qnc(k, c0:c0+n-1) + delt * rqnc(k, c0:c0+n-1), 0.0)
        qni(k, c0:c0+n-1) = max(qni(k, c0:c0+n-1) + delt * rqni(k, c0:c0+n-1), 0.0)
        qnwfa(k, c0:c0+n-1) = max(qnwfa(k, c0:c0+n-1) + delt * rqnwfa(k, c0:c0+n-1), 0.0)
        qnifa(k, c0:c0+n-1) = max(qnifa(k, c0:c0+n-1) + delt * rqnifa(k, c0:c0+n-1), 0.0)
        t3d(k, c0:c0+n-1) = th(k, c0:c0+n-1) * exner(k, c0:c0+n-1)
        rho(k, c0:c0+n-1) = p(k, c0:c0+n-1) / (287.0 * t3d(k, c0:c0+n-1))
      end do
    end do
  end do
  close(uout)

contains

  subroutine pack3(src, c0, n, dst)
    real, intent(in) :: src(:,:)
    integer, intent(in) :: c0, n
    real, intent(out) :: dst(1:n, 1:nz+1, 1:1)
    integer :: i, kk
    do kk = 1, nz
      do i = 1, n
        dst(i, kk, 1) = src(kk, c0 + i - 1)
      end do
    end do
    dst(:, nz + 1, 1) = nan
  end subroutine pack3

  subroutine unpack3(src, c0, n, dst)
    real, intent(in) :: src(1:n, 1:nz+1, 1:1)
    integer, intent(in) :: c0, n
    real, intent(inout) :: dst(:,:)
    integer :: i, kk
    do kk = 1, nz
      do i = 1, n
        dst(kk, c0 + i - 1) = src(i, kk, 1)
      end do
    end do
  end subroutine unpack3

  subroutine one_group(c0, n, dxc, delt_g, flag_qs_g, initflag)
    integer, intent(in) :: c0, n, initflag
    real, intent(in) :: dxc, delt_g
    logical, intent(in) :: flag_qs_g
    integer :: ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme
    integer :: its, ite, jts, jte, kts, kte
    real, dimension(1:n, 1:nz+1, 1:1) :: l_dz, l_u, l_v, l_w, l_th, l_qv, &
        l_qc, l_qi, l_qs, l_p, l_ex, l_rho, l_t3d, l_rthraten, l_qnc, l_qni, &
        l_qnwfa, l_qnifa, l_qnbca, l_qke, l_qke_adv, l_sh, l_sm, l_tsq, &
        l_qsq, l_cov, l_ru, l_rv, l_rth, l_rqv, l_rqc, l_rqi, l_rqs, &
        l_rqnc, l_rqni, l_rqnwfa, l_rqnifa, l_rqnbca, l_exh, l_exm, l_el, &
        l_dqke, l_qwt, l_qshear, l_qbuoy, l_qdiss, l_qcbl, l_qibl, l_cfbl, &
        l_edmf_a, l_edmf_w, l_edmf_qt, l_edmf_thl, l_edmf_ent, l_edmf_qc, &
        l_sub_thl, l_sub_sqv, l_det_thl, l_det_sqv, l_spp
    real, dimension(1:n, 1:1) :: l_xland, l_ts, l_qsfc, l_ps, l_ch, l_znt, &
        l_pblh, l_maxwidth, l_maxmf, l_ztop, l_rmol, l_hfx, l_qfx, l_ust, &
        l_wspd, l_uoce, l_voce
    integer, dimension(1:n, 1:1) :: l_kpbl, l_ktop
    integer :: i
    logical :: qn_on

    qn_on = variant == 1
    ids = 1; ide = n + 1; jds = 1; jde = 2; kds = 1; kde = nz + 1
    ims = 1; ime = n; jms = 1; jme = 1; kms = 1; kme = nz + 1
    its = 1; ite = n; jts = 1; jte = 1; kts = 1; kte = nz

    call pack3(dz, c0, n, l_dz)
    call pack3(u, c0, n, l_u)
    call pack3(v, c0, n, l_v)
    call pack3(w, c0, n, l_w)
    call pack3(th, c0, n, l_th)
    call pack3(qv, c0, n, l_qv)
    call pack3(qc, c0, n, l_qc)
    call pack3(qi, c0, n, l_qi)
    call pack3(qs, c0, n, l_qs)
    call pack3(p, c0, n, l_p)
    call pack3(exner, c0, n, l_ex)
    call pack3(rho, c0, n, l_rho)
    call pack3(t3d, c0, n, l_t3d)
    call pack3(rthraten, c0, n, l_rthraten)
    call pack3(qke, c0, n, l_qke)
    call pack3(tsq, c0, n, l_tsq)
    call pack3(qsq, c0, n, l_qsq)
    call pack3(cov, c0, n, l_cov)
    call pack3(el, c0, n, l_el)
    call pack3(sh, c0, n, l_sh)
    call pack3(sm, c0, n, l_sm)
    call pack3(qcbl, c0, n, l_qcbl)
    call pack3(qibl, c0, n, l_qibl)
    call pack3(cfbl, c0, n, l_cfbl)
    call pack3(exh, c0, n, l_exh)
    call pack3(exm, c0, n, l_exm)
    call pack3(ru, c0, n, l_ru)
    call pack3(rv, c0, n, l_rv)
    call pack3(rth, c0, n, l_rth)
    call pack3(rqv, c0, n, l_rqv)
    call pack3(rqc, c0, n, l_rqc)
    call pack3(rqi, c0, n, l_rqi)
    call pack3(qnc, c0, n, l_qnc)
    call pack3(qni, c0, n, l_qni)
    call pack3(qnwfa, c0, n, l_qnwfa)
    call pack3(qnifa, c0, n, l_qnifa)
    call pack3(rqnc, c0, n, l_rqnc)
    call pack3(rqni, c0, n, l_rqni)
    call pack3(rqnwfa, c0, n, l_rqnwfa)
    call pack3(rqnifa, c0, n, l_rqnifa)
    l_qnbca = 0.0; l_rqnbca = 0.0
    l_qke_adv = 0.0; l_rqs = 0.0
    l_dqke = 0.0; l_qwt = 0.0; l_qshear = 0.0; l_qbuoy = 0.0; l_qdiss = 0.0
    l_edmf_a = 0.0; l_edmf_w = 0.0; l_edmf_qt = 0.0; l_edmf_thl = 0.0
    l_edmf_ent = 0.0; l_edmf_qc = 0.0; l_sub_thl = 0.0; l_sub_sqv = 0.0
    l_det_thl = 0.0; l_det_sqv = 0.0
    call pack3(spp, c0, n, l_spp)
    do i = 1, n
      l_xland(i, 1) = xland(c0 + i - 1)
      l_ts(i, 1) = ts(c0 + i - 1)
      l_qsfc(i, 1) = qsfc(c0 + i - 1)
      l_ps(i, 1) = ps(c0 + i - 1)
      l_ch(i, 1) = ch(c0 + i - 1)
      l_znt(i, 1) = znt(c0 + i - 1)
      l_pblh(i, 1) = pblh(c0 + i - 1)
      l_rmol(i, 1) = rmol(c0 + i - 1)
      l_hfx(i, 1) = hfx(c0 + i - 1)
      l_qfx(i, 1) = qfx(c0 + i - 1)
      l_ust(i, 1) = ust(c0 + i - 1)
      l_wspd(i, 1) = wspd(c0 + i - 1)
      l_kpbl(i, 1) = kpbl(c0 + i - 1)
      l_maxwidth(i, 1) = maxwidth(c0 + i - 1)
      l_maxmf(i, 1) = maxmf(c0 + i - 1)
      l_ztop(i, 1) = ztop(c0 + i - 1)
      l_ktop(i, 1) = ktop(c0 + i - 1)
    end do
    l_uoce = 0.0
    l_voce = 0.0

    call mynnedmf_wrapper_run(                                           &
        initflag=initflag, restart=.false., cycling=.false.,             &
        delt=delt_g, dz=l_dz, dxc=dxc, znt=l_znt,                        &
        u=l_u, v=l_v, w=l_w, th=l_th,                                    &
        qv=l_qv, qc=l_qc, qi=l_qi, qs=l_qs, qnc=l_qnc, qni=l_qni,        &
        qnwfa=l_qnwfa, qnifa=l_qnifa, qnbca=l_qnbca,                     &
        p=l_p, exner=l_ex, rho=l_rho, t3d=l_t3d,                         &
        xland=l_xland, ts=l_ts, qsfc=l_qsfc, ps=l_ps,                    &
        ust=l_ust, ch=l_ch, hfx=l_hfx, qfx=l_qfx, rmol=l_rmol,           &
        wspd=l_wspd, uoce=l_uoce, voce=l_voce,                           &
        qke=l_qke, qke_adv=l_qke_adv, sh3d=l_sh, sm3d=l_sm,              &
        tsq=l_tsq, qsq=l_qsq, cov=l_cov,                                 &
        rublten=l_ru, rvblten=l_rv, rthblten=l_rth,                      &
        rqvblten=l_rqv, rqcblten=l_rqc, rqiblten=l_rqi, rqsblten=l_rqs,  &
        rqncblten=l_rqnc, rqniblten=l_rqni,                              &
        rqnwfablten=l_rqnwfa, rqnifablten=l_rqnifa,                      &
        rqnbcablten=l_rqnbca,                                            &
        exch_h=l_exh, exch_m=l_exm, pblh=l_pblh, kpbl=l_kpbl,            &
        el_pbl=l_el,                                                     &
        dqke=l_dqke, qwt=l_qwt, qshear=l_qshear, qbuoy=l_qbuoy,          &
        qdiss=l_qdiss,                                                   &
        qc_bl=l_qcbl, qi_bl=l_qibl, cldfra_bl=l_cfbl,                    &
        edmf_a=l_edmf_a, edmf_w=l_edmf_w, edmf_qt=l_edmf_qt,             &
        edmf_thl=l_edmf_thl, edmf_ent=l_edmf_ent, edmf_qc=l_edmf_qc,     &
        sub_thl3d=l_sub_thl, sub_sqv3d=l_sub_sqv,                        &
        det_thl3d=l_det_thl, det_sqv3d=l_det_sqv,                        &
        maxwidth=l_maxwidth, maxmf=l_maxmf, ztop_plume=l_ztop,           &
        ktop_plume=l_ktop,                                               &
        rthraten=l_rthraten,                                             &
        tke_budget=0, bl_mynn_tkeadvect=.false.,                         &
        bl_mynn_cloudpdf=2, bl_mynn_mixlength=mixlength,                 &
        icloud_bl=1, bl_mynn_edmf=1,                                     &
        bl_mynn_edmf_mom=1, bl_mynn_edmf_tke=0,                          &
        bl_mynn_cloudmix=1, bl_mynn_mixqt=0,                             &
        bl_mynn_output=0, bl_mynn_closure=2.6,                           &
        bl_mynn_mixscalars=merge(1, 0, qn_on),                           &
        spp_pbl=merge(1, 0, variant == 2), pattern_spp_pbl=l_spp,        &
        flag_qc=.true., flag_qi=.true., flag_qs=flag_qs_g,               &
        flag_qnc=qn_on, flag_qni=qn_on,                                  &
        flag_qnwfa=qn_on, flag_qnifa=qn_on, flag_qnbca=qn_on,            &
        ids=ids, ide=ide, jds=jds, jde=jde, kds=kds, kde=kde,            &
        ims=ims, ime=ime, jms=jms, jme=jme, kms=kms, kme=kme,            &
        its=its, ite=ite, jts=jts, jte=jte, kts=kts, kte=kte)

    call unpack3(l_qke, c0, n, qke)
    call unpack3(l_tsq, c0, n, tsq)
    call unpack3(l_qsq, c0, n, qsq)
    call unpack3(l_cov, c0, n, cov)
    call unpack3(l_el, c0, n, el)
    call unpack3(l_sh, c0, n, sh)
    call unpack3(l_sm, c0, n, sm)
    call unpack3(l_qcbl, c0, n, qcbl)
    call unpack3(l_qibl, c0, n, qibl)
    call unpack3(l_cfbl, c0, n, cfbl)
    call unpack3(l_exh, c0, n, exh)
    call unpack3(l_exm, c0, n, exm)
    call unpack3(l_ru, c0, n, ru)
    call unpack3(l_rv, c0, n, rv)
    call unpack3(l_rth, c0, n, rth)
    call unpack3(l_rqv, c0, n, rqv)
    call unpack3(l_rqc, c0, n, rqc)
    call unpack3(l_rqi, c0, n, rqi)
    call unpack3(l_rqnc, c0, n, rqnc)
    call unpack3(l_rqni, c0, n, rqni)
    call unpack3(l_rqnwfa, c0, n, rqnwfa)
    call unpack3(l_rqnifa, c0, n, rqnifa)
    do i = 1, n
      pblh(c0 + i - 1) = l_pblh(i, 1)
      rmol(c0 + i - 1) = l_rmol(i, 1)
      kpbl(c0 + i - 1) = l_kpbl(i, 1)
      maxwidth(c0 + i - 1) = l_maxwidth(i, 1)
      maxmf(c0 + i - 1) = l_maxmf(i, 1)
      ztop(c0 + i - 1) = l_ztop(i, 1)
      ktop(c0 + i - 1) = l_ktop(i, 1)
      ust(c0 + i - 1) = l_ust(i, 1)
      hfx(c0 + i - 1) = l_hfx(i, 1)
      qfx(c0 + i - 1) = l_qfx(i, 1)
      wspd(c0 + i - 1) = l_wspd(i, 1)
      znt(c0 + i - 1) = l_znt(i, 1)
    end do
  end subroutine one_group

end program run_mynn_wrapper_columns
