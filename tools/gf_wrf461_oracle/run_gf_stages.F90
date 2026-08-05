program run_gf_stages_oracle
  ! Decompose `cup_gf` itself, one stage per capture.
  !
  ! `run_cu_gf.F90` pins GFDRV; `run_cup_gf.F90` pins `cup_gf`.  Both are
  ! black boxes at the granularity a port is WRITTEN at.  A reference whose
  ! `outt` is wrong at column 137 needs to know whether it lost the answer in
  ! `cup_env`, in the beta-function updraft profile, or in the closure -- and
  ! `cup_gf`'s 60-odd locals never leave the routine.
  !
  ! This program is a statement-order replication of CUP_gf's body
  ! (module_cu_gf_deep.F:359-1868) for the arm WRF can actually reach --
  ! imid = 0, dicycle = 1, ichoice = 0, ipr = 0, nranflag = 0, csum = 0,
  ! WRF_CHEM = 0, WRF_DFI_RADAR = 0 -- calling the module's own public
  ! procedures (`cup_env`, `cup_env_clev`, `cup_kbcon`, `cup_MAXIMI`,
  ! `cup_minimi`, `get_cloud_bc`, `rates_up_pdf`, `get_lateral_massflux`,
  ! `cup_dd_moisture`, `cup_up_moisture`, `cup_up_aa0`, `cup_up_aa1bl`,
  ! `cup_dd_edt`, `cup_forcing_ens_3d`, `cup_output_ens_3d`) and writing out
  ! every intermediate between them.
  !
  ! It does not claim the replication is faithful, it PROVES it per column:
  ! after the replication it calls the real `cup_gf` on the same prepared
  ! column and compares 13 output fields bitwise.  `gf-deep-consistency.csv`
  ! records the differing-word count.  Any nonzero row there means this
  ! decomposition has drifted from the module and its stage numbers must not
  ! be trusted.
  !
  ! What is deliberately NOT replicated, because WRF cannot reach it:
  !   imid_gf = 0 (module_cu_gf_wrfdrv.F:69) kills the mid-level arm -- the
  !     `get_inversion_layers` call at :690, the ktop-from-inversion block at
  !     :708-723, `xff_mid`, and every `if(imid.eq.1)` tuning override.
  !   iversion = 0 (:1205-1288) -- `iversion` is a local initialised to 1 and
  !     never assigned.
  !   irainevap = 0 (:14) kills the SAS rain-evaporation block at :1746-1799.
  !   autoconv = 1 (:29) selects the non-Berry arm of `cup_up_moisture`;
  !     aeroevap = 1 (:30) leaves `cup_dd_edt`'s aerosol arm off.
  !   `gfinit` (:4337) -- module_physics_init.F routes GFSCHEME to `g3init`.
  !   `DERIV3` (:4161) -- defined and never called, by anything, anywhere.
  !
  ! `get_inversion_layers` IS reached, but only by `CUP_gf_sh`.  It is called
  ! here on the deep column purely to capture it; see the kend clamp below.
  use module_cu_gf_deep
  use module_cu_gf_sh, only: cup_gf_sh
  use gf_cases
  implicit none

  integer, parameter :: nz = gf_nz
  integer, parameter :: ncase = gf_ncase
  integer, parameter :: ndx = gf_ndx

  integer, parameter :: its = 1, ite = 1, itf = 1
  integer, parameter :: kts = 1, kte = nz, ktf = nz

  ! The case column is built through the same GFDRV-shaped tile run_cup_gf
  ! uses, so both decompositions consume identical array words.
  integer, parameter :: gids = 1, gide = 10
  integer, parameter :: gits = gids + 4
  integer, parameter :: gjds = 1, gjde = 10
  integer, parameter :: gjts = gjds + 4
  integer, parameter :: gkds = 1, gkde = nz + 1

  ! module_gfs_physcons, at the real(8) width module_gfs_machine.F declares
  ! (see run_cup_gf.F90 for the measurement that settles which double).
  integer, parameter :: kind_phys = selected_real_kind(13, 60)
  real(kind=kind_phys), parameter :: con_g = 9.80665e+0_kind_phys
  real(kind=kind_phys), parameter :: con_cp = 1.0046e+3_kind_phys
  real(kind=kind_phys), parameter :: con_hvap = 2.5000e+6_kind_phys

  ! --- the prepared column ---------------------------------------------------
  real, dimension(its:ite, kts:kte) :: zo, t2d, q2d, po, p2d, us, vs, rhoi
  real, dimension(its:ite, kts:kte) :: tn, qo, tshall, qshall, dhdt, omeg
  real, dimension(its:ite) :: ter11, psur, xlandi, hfxi, qfxi, dxi, ccn
  real, dimension(its:ite) :: mconv
  integer, dimension(its:ite) :: kpbli
  real, dimension(kts:kte) :: pi_col

  ! --- the shallow arm, only so xmbs_in matches cup_gf's caller --------------
  real, dimension(its:ite, kts:kte) :: outts, outqs, outqcs, outus, outvs
  real, dimension(its:ite, kts:kte) :: cupclws, cnvwts, zus
  real, dimension(its:ite) :: xmbs, prets
  integer, dimension(its:ite) :: ierrs, kbcons, ktops, k22s
  character(len=50) :: ierrcs(its:ite)

  ! --- CUP_gf's own locals, by their WRF names -------------------------------
  real, dimension(its:ite, kts:kte) :: entr_rate_2d, mentrd_rate_2d
  real, dimension(its:ite, kts:kte) :: he, hes, qes, z, heo, heso, qeso
  real, dimension(its:ite, kts:kte) :: xhe, xhes, xqes, xz, xt, xq
  real, dimension(its:ite, kts:kte) :: qes_cup, q_cup, he_cup, hes_cup, z_cup
  real, dimension(its:ite, kts:kte) :: p_cup, gamma_cup, t_cup
  real, dimension(its:ite, kts:kte) :: qeso_cup, qo_cup, heo_cup, heso_cup
  real, dimension(its:ite, kts:kte) :: zo_cup, po_cup, gammao_cup, tn_cup
  real, dimension(its:ite, kts:kte) :: xqes_cup, xq_cup, xhe_cup, xhes_cup
  real, dimension(its:ite, kts:kte) :: xz_cup, xt_cup
  real, dimension(its:ite, kts:kte) :: dby, hc, zu, clw_all
  real, dimension(its:ite, kts:kte) :: dbyo, qco, qrcdo, pwdo, pwo, hcdo
  real, dimension(its:ite, kts:kte) :: qcdo, dbydo, hco, qrco
  real, dimension(its:ite, kts:kte) :: dbyt, xdby, xhc, xzu
  real, dimension(its:ite, kts:kte) :: cd, cdd, dellah, dellaq, dellat, dellaqc
  real, dimension(its:ite, kts:kte) :: u_cup, v_cup, uc, vc, ucd, vcd
  real, dimension(its:ite, kts:kte) :: dellu, dellv
  real, dimension(its:ite, kts:kte) :: up_massentr, up_massdetr, c1d
  real, dimension(its:ite, kts:kte) :: up_massentro, up_massdetro
  real, dimension(its:ite, kts:kte) :: dd_massentro, dd_massdetro
  real, dimension(its:ite, kts:kte) :: up_massentru, up_massdetru
  real, dimension(its:ite, kts:kte) :: dd_massentru, dd_massdetru
  real, dimension(its:ite, kts:kte) :: cnvwt, zuo, zdo, cupclw
  real, dimension(its:ite, kts:kte) :: outt, outq, outqc, outu, outv
  real, dimension(its:ite, kts:kte) :: dtempdz
  integer, dimension(its:ite, kts:kte) :: k_inv_layers

  real, dimension(its:ite) :: edt, edto, aa1, aa0, xaa0, hkb, hkbo, xhkb
  real, dimension(its:ite) :: xmb, pwavo, pwevo, bu, bud, cap_max
  real, dimension(its:ite) :: cap_max_increment, closure_n, psum, psumh
  real, dimension(its:ite) :: sig, axx, edtmax, edtmin, entr_rate
  real, dimension(its:ite) :: lambau, flux_tun, zws, ztexec, zqexec
  real, dimension(its:ite) :: aa1_bl, tau_bl, tau_ecmwf, wmean
  real, dimension(its:ite) :: xf_dicycle, pre, xmb_out, xmbm_in
  real, dimension(its:ite) :: rand_mom, rand_vmas
  real, dimension(its:ite, 4) :: rand_clos
  real, dimension(its:ite, 10) :: forcing
  real, dimension(its:ite, 1) :: xaa0_ens, edtc
  real, dimension(its:ite, kts:kte, 1) :: dellat_ens, dellaqc_ens, dellaq_ens
  real, dimension(its:ite, kts:kte, 1) :: pwo_ens
  real, dimension(its:ite, 1:maxens3) :: xf_ens, pr_ens
  real, dimension(its:ite, 2) :: xff_mid
  integer, dimension(its:ite) :: kzdown, kdet, k22, jmin, kstabi, kstabm
  integer, dimension(its:ite) :: k22x, xland1, ktopdby, kbconx, ierr2, ierr3
  integer, dimension(its:ite) :: kbmax, ierr, kbcon, ktop, cactiv
  integer, dimension(its:ite) :: pmin_lev, start_level, ktopkeep
  character(len=50) :: ierrc(its:ite)
  real :: zuh2(40)

  ! stage snapshots the replication overwrites in place
  real, dimension(its:ite, kts:kte) :: entr2d_a, zu_pdf, outt_o, outq_o, outqc_o
  real, dimension(its:ite, kts:kte) :: gamma_cup1
  real, dimension(its:ite) :: hkb0, hkbo0, umean_s, edtc1, mconv2, frh_kb
  integer, dimension(its:ite) :: k22_0, kbcon_1, ierr_1, ktop_pdf, ierr_2
  integer, dimension(its:ite) :: ktop_dbyt, ierr_3, ierr_4, ierr_5, ierr_6
  integer, dimension(its:ite) :: ierr_7, kdet_2, kinv_clamped
  ! get_zu_zd_pdf_fim's internals.  These are the only place the scheme calls
  ! tgammaf, and glibc's float32 gamma is up to 2 ULP off correctly-rounded
  ! (31 of 51 arguments in gf-pow-probe.txt's pgamma table), so a port cannot
  ! model it from a float64 gamma.  Capturing alpha/beta/fzu lets a reference
  ! be graded on everything else and lets the residual be attributed.
  real, dimension(its:ite) :: up_tun, up_alpha, up_beta, up_fzu
  real, dimension(its:ite) :: dn_tun, dn_alpha, dn_beta, dn_fzu
  integer, dimension(its:ite) :: up_kbadj, dn_kbadj, up_kklev, up_kfinal

  ! --- the same column, run through the real cup_gf --------------------------
  real, dimension(its:ite, kts:kte) :: r_outt, r_outq, r_outqc, r_outu, r_outv
  real, dimension(its:ite, kts:kte) :: r_cupclw, r_cnvwt, r_zu, r_zd
  real, dimension(its:ite, kts:kte) :: r_q, r_qo, r_omeg
  real, dimension(its:ite) :: r_pre, r_xmb, r_xmb_out, r_edt, r_mconv
  real, dimension(its:ite, 10) :: r_forcing
  integer, dimension(its:ite) :: r_ierr, r_kbcon, r_ktop, r_k22, r_jmin

  ! --- the case tile ---------------------------------------------------------
  real, dimension(gids:gide, gkds:gkde, gjds:gjde) :: gu, gv, gw, gt, gq, gp
  real, dimension(gids:gide, gkds:gkde, gjds:gjde) :: gpi, grho, gdz, gp8w
  real, dimension(gids:gide, gkds:gkde, gjds:gjde) :: gthften, gqvften
  real, dimension(gids:gide, gkds:gkde, gjds:gjde) :: gthraten, gthblten, gqvblten
  real, dimension(gids:gide, gjds:gjde) :: ght, ghfx, gqfx, gxland
  integer, dimension(gids:gide, gjds:gjde) :: gkpbl

  ! --- CUP_gf's scalar locals ------------------------------------------------
  real :: dz, dzo, mbdt, radius, zcutdown, depth_min, zkbmax, z_detr, zktop
  real :: dh, cap_maxs, frh, sig_thresh, buo_flux, pgeoh, pgcon, pgc
  real :: entdo, dp, subin, detdo, entup, detup, subdown, entdoj, entupk
  real :: detupk, totmas, denom, h_entr, umean, t_star, dq, x_add
  real :: beta, dts, fp, fpi, pmin, g_rain, e_dn, c_up, elocp, el2orc
  real :: cbeg, cmid, cend, const_a, const_b, const_c
  integer :: iloop, nens3, ki, kk, i, k, jmini, start_k22, iversion
  logical :: keep_going

  real :: dtstep, dxv, tcrit_l, cuten, cutens
  integer :: ichoice, ichoice_s, ishallow, dicycle
  integer :: ic, idx, iarm, n, ulev, usfc, ucon, ndiff, nclamp
  integer :: spp0, kend_gil
  character(len=1024) :: level_path, surface_path, cons_path

  call get_command_argument(1, level_path)
  call get_command_argument(2, surface_path)
  call get_command_argument(3, cons_path)
  if (len_trim(level_path) == 0 .or. len_trim(surface_path) == 0 .or. &
      len_trim(cons_path) == 0) then
    write(*, '(A)') 'usage: run_gf_stages LEVELS.csv SURFACE.csv CONSISTENCY.csv'
    error stop 2
  end if

  call gf_build_case_table()

  tcrit_l = 258.
  ichoice = 0
  ichoice_s = 0
  dicycle = 1
  dtstep = 60.0
  spp0 = 0
  nclamp = 0

  open(newunit=ulev, file=trim(level_path), status='replace', action='write')
  write(ulev, '(A)') 'case,idx,arm,k,' //                                     &
    'qes,he,hes,qeso,heo,heso,' //                                            &
    'qes_cup,q_cup,he_cup,hes_cup,gamma_cup1,t_cup,' //                       &
    'qeso_cup,qo_cup,heo_cup,heso_cup,zo_cup,po_cup,gammao_cup,tn_cup,' //    &
    'u_cup,v_cup,entr2d_a,zu_pdf,' //                                         &
    'cd,entr2d_b,upme,upmd,upmeu,upmdu,' //                                   &
    'hc,uc,vc,hco,dby,dbyo,dbyt,' //                                          &
    'cdd,ddme,ddmd,ddmeu,ddmdu,mentrd2d,hcdo,ucd,vcd,dbydo,c1d,' //           &
    'qcdo,qrcdo,pwdo,qco,qrco,pwo,clw_all,' //                                &
    'dellu,dellv,dellah,dellaq,dellaqc,dellat,' //                            &
    'xhe,xq,xt,xqes,xhes,' //                                                 &
    'xqes_cup,xq_cup,xhe_cup,xhes_cup,gamma_cupx,xt_cup,xhc,xdby,' //         &
    'outt_o,outq_o,outqc_o,outt_ke,outu_f,outv_f,dtempdz'
  open(newunit=usfc, file=trim(surface_path), status='replace', action='write')
  write(usfc, '(A)') 'case,idx,arm,' //                                       &
    'zws,ztexec,zqexec,cap_max,xland1,entr_rate,sig,sig_thresh,' //           &
    'kbmax,kdet,k22_0,hkb0,hkbo0,k22_1,kbcon_1,hkb1,ierr_1,' //               &
    'kstabi,kstabm,frh_kb,pmin_lev,start_level,' //                           &
    'ktop_pdf,ktopdby,kbcon_2,ierr_2,ktop_dbyt,ierr_3,' //                    &
    'kzdown,jmin,kdet_2,beta,edtmax,ierr_4,' //                               &
    'bud,pwevo,bu,ierr_5,pwavo,psum,psumh,' //                                &
    'aa0,aa1,ierr_6,tau_ecmwf,tau_bl,aa1_bl,umean,' //                        &
    'edt,edtc1,edto,xhkb,xaa0,pr7,ierr_7,' //                                 &
    'k22x,kbconx,ierr2,ierr3,mconv2,' //                                      &
    'xf1,xf2,xf3,xf4,xf5,xf6,xf7,xf8,xf9,xf10,' //                            &
    'xf11,xf12,xf13,xf14,xf15,xf16,xf_dicycle,closure_n,' //                  &
    'f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,' //                                      &
    'xmb,pre,ktop,ierr,kinv1,kinv2,kinv3,kinv4,kinv5,kinv_clamped,' //       &
    'up_tun,up_alpha,up_beta,up_fzu,up_kbadj,up_kklev,up_kfinal,' //         &
    'dn_tun,dn_alpha,dn_beta,dn_fzu,dn_kbadj'
  open(newunit=ucon, file=trim(cons_path), status='replace', action='write')
  write(ucon, '(A)') 'case,idx,arm,ndiff_words_vs_cup_gf,ierr_repl,ierr_cupgf'

  do idx = 1, ndx
    dxv = gf_dxsweep(idx)
    do iarm = 1, 2
      ishallow = iarm - 1
      do ic = 1, ncase

        call fill_driver_tile(ic)
        call prepare_column(ic, dxv)

        ! ---- the shallow arm, for xmbs_in only ---------------------------
        outts = 0.0; outqs = 0.0; outqcs = 0.0; outus = 0.0; outvs = 0.0
        cupclws = 0.0; cnvwts = 0.0; zus = 0.0
        xmbs = 0.0; prets = 0.0
        ierrs = 0; kbcons = 0; ktops = 0; k22s = 0
        ierrcs(:) = " "
        if (ishallow == 1) then
          call cup_gf_sh(zo, t2d, q2d, ter11, tshall, qshall, p2d, psur,     &
                         dhdt, kpbli, rhoi, hfxi, qfxi, xlandi, ichoice_s,   &
                         tcrit_l, dtstep,                                     &
                         zus, xmbs, kbcons, ktops, k22s, ierrs, ierrcs,      &
                         outts, outqs, outqcs, cnvwts, prets, cupclws,       &
                         itf, ktf, its, ite, kts, kte, 0)
          call neg_check('shallow', 0, dtstep, q2d, outqs, outts, outus,     &
                         outvs, outqcs, prets, its, ite, kts, kte, itf, ktf)
        end if

        call replicate_cup_gf(dxv)
        call run_real_cup_gf(dxv)
        call compare_and_emit(ic, idx, ishallow)
      end do
    end do
  end do

  close(ulev)
  close(usfc)
  close(ucon)
  write(*, '(A,I0,A)') 'gf deep stage oracle written (', nclamp,             &
    ' get_inversion_layers kend clamps)'

contains

  integer function bitdiff(a, b)
    real, intent(in) :: a, b
    if (transfer(a, 0) == transfer(b, 0)) then
      bitdiff = 0
    else
      bitdiff = 1
    end if
  end function bitdiff

  ! ==========================================================================
  ! CUP_gf, module_cu_gf_deep.F:359-1868, statement order, live arm only.
  ! ==========================================================================
  subroutine replicate_cup_gf(dx_in)
    real, intent(in) :: dx_in
    real, dimension(its:ite) :: dx

    dx(:) = dx_in
    xmbm_in(:) = 0.
    rand_mom(:) = 0.
    rand_vmas(:) = 0.
    rand_clos(:, :) = 0.
    cactiv(:) = 0
    forcing(:, :) = 0.
    xff_mid(:, :) = 0.
    ierr(:) = 0
    ierrc(:) = " "
    kbcon(:) = 0
    ktop(:) = 0
    k22(:) = 0
    jmin(:) = 0
    pre(:) = 0.
    xmb(:) = 0.
    outt = 0.; outq = 0.; outqc = 0.; outu = 0.; outv = 0.
    cnvwt = 0.; cupclw = 0.; zuo = 0.; zdo = 0.; edto = 0.
    dn_tun(:) = 0.; dn_alpha(:) = 0.; dn_beta(:) = 0.; dn_fzu(:) = 0.
    dn_kbadj(:) = 0
    ! kdet and pmin_lev are genuinely uninitialised in CUP_gf; zeroing them
    ! here would be a divergence, so they are left carrying the previous
    ! column exactly as the module's stack slots would.

    ! ---- :359-367 ----------------------------------------------------------
    flux_tun(:) = fluxtune
    pmin = 150.
    ktopdby(:) = 0
    elocp = xlv / cp
    el2orc = xlv * xlv / (r_v * cp)
    pgcon = 0.
    lambau(:) = 2.
    ztexec(:) = 0.
    zqexec(:) = 0.
    zws(:) = 0.

    ! ---- :387-406 : the w* / excess block ----------------------------------
    do i = its, itf
      buo_flux = (hfxi(i) / cp + 0.608 * t2d(i, 1) * qfxi(i) / xlv) / rhoi(i, 1)
      pgeoh = zo(i, 2) * g
      zws(i) = max(0., flux_tun(i) * 0.41 * buo_flux * zo(i, 2) * g / t2d(i, 1))
      if (zws(i) > TINY(pgeoh)) then
        zws(i) = 1.2 * zws(i)**.3333
        ztexec(i) = MAX(flux_tun(i) * hfxi(i) / (rhoi(i, 1) * zws(i) * cp), 0.0)
        zqexec(i) = MAX(flux_tun(i) * qfxi(i) / xlv / (rhoi(i, 1) * zws(i)), 0.)
      endif
      zws(i) = max(0., flux_tun(i) * 0.41 * buo_flux * zo(i, kpbli(i)) * g   &
                       / t2d(i, kpbli(i)))
      zws(i) = 1.2 * zws(i)**.3333
      zws(i) = zws(i) * rhoi(i, kpbli(i))
    enddo

    ! ---- :409-433 ----------------------------------------------------------
    cap_maxs = 75.
    do i = its, itf
      edto(i) = 0.
      closure_n(i) = 16.
      xmb_out(i) = 0.
      cap_max(i) = cap_maxs
      cap_max_increment(i) = 20.
      xland1(i) = int(xlandi(i) + .0001)
      if (xlandi(i) .gt. 1.5 .or. xlandi(i) .lt. 0.5) then
        xland1(i) = 0
        cap_max_increment(i) = 20.
      else
        if (ztexec(i) .gt. 0.) cap_max(i) = cap_max(i) + 25.
        if (ztexec(i) .lt. 0.) cap_max(i) = cap_max(i) - 25.
      endif
      ierrc(i) = " "
    enddo

    ! ---- :455-471 : entrainment, radius, the scale-aware sig ---------------
    start_level(:) = kte
    do i = its, ite
      c1d(i, :) = 0.
      entr_rate(i) = 7.e-5 - min(20., float(cactiv(i))) * 3.e-6
      if (xland1(i) == 0) entr_rate(i) = 7.e-5
      radius = .2 / entr_rate(i)
      frh = min(1., 3.14 * radius * radius / dx(i) / dx(i))
      if (frh > frh_thresh) then
        frh = frh_thresh
        radius = sqrt(frh * dx(i) * dx(i) / 3.14)
        entr_rate(i) = .2 / radius
      endif
      sig(i) = (1. - frh)**2
    enddo
    sig_thresh = (1. - frh_thresh)**2

    ! ---- :480-495 ----------------------------------------------------------
    do k = kts, ktf
      do i = its, itf
        cnvwt(i, k) = 0.
        zuo(i, k) = 0.
        zdo(i, k) = 0.
        z(i, k) = zo(i, k)
        xz(i, k) = zo(i, k)
        cupclw(i, k) = 0.
        cd(i, k) = 1.e-9
        cdd(i, k) = 1.e-9
        hcdo(i, k) = 0.
        qrcdo(i, k) = 0.
        dellaqc(i, k) = 0.
      enddo
    enddo

    edtmax(:) = 1.
    edtmin(:) = .1
    depth_min = 1000.
    DO i = its, itf
      kbmax(i) = 1
      aa0(i) = 0.
      aa1(i) = 0.
      edt(i) = 0.
      kstabm(i) = ktf - 1
      IERR2(i) = 0
      IERR3(i) = 0
      x_add = 0.
    enddo
    zkbmax = 4000.
    zcutdown = 4000.
    z_detr = 1000.
    do i = its, itf
      do k = 1, maxens3
        xf_ens(i, k) = 0.
        pr_ens(i, k) = 0.
      enddo
    enddo

    ! ---- :561-568 : cup_env x2 ---------------------------------------------
    call cup_env(z, qes, he, hes, t2d, q2d, po, ter11,                       &
                 psur, ierr, tcrit_l, -1, itf, ktf, its, ite, kts, kte)
    call cup_env(zo, qeso, heo, heso, tn, qo, po, ter11,                     &
                 psur, ierr, tcrit_l, -1, itf, ktf, its, ite, kts, kte)

    ! ---- :573-582 : cup_env_clev x2 ----------------------------------------
    call cup_env_clev(t2d, qes, q2d, he, hes, z, po, qes_cup, q_cup, he_cup, &
                      hes_cup, z_cup, p_cup, gamma_cup, t_cup, psur,         &
                      ierr, ter11, itf, ktf, its, ite, kts, kte)
    call cup_env_clev(tn, qeso, qo, heo, heso, zo, po, qeso_cup, qo_cup,     &
                      heo_cup, heso_cup, zo_cup, po_cup, gammao_cup, tn_cup, &
                      psur, ierr, ter11, itf, ktf, its, ite, kts, kte)
    gamma_cup1 = gamma_cup

    ! ---- :583-615 ----------------------------------------------------------
    do i = its, itf
      if (ierr(i) .eq. 0) then
        u_cup(i, kts) = us(i, kts)
        v_cup(i, kts) = vs(i, kts)
        do k = kts + 1, ktf
          u_cup(i, k) = .5 * (us(i, k - 1) + us(i, k))
          v_cup(i, k) = .5 * (vs(i, k - 1) + vs(i, k))
        enddo
      endif
    enddo
    do i = its, itf
      if (ierr(i) .eq. 0) then
        do k = kts, ktf
          if (zo_cup(i, k) .gt. zkbmax + ter11(i)) then
            kbmax(i) = k
            go to 25
          endif
        enddo
25      continue
        do k = kts, ktf
          if (zo_cup(i, k) .gt. z_detr + ter11(i)) then
            kdet(i) = k
            go to 26
          endif
        enddo
26      continue
      endif
    enddo

    ! ---- :621-633 : k22 ----------------------------------------------------
    start_k22 = 2
    DO 36 i = its, itf
      IF (ierr(I) .eq. 0) THEN
        k22(i) = maxloc(HEO_CUP(i, start_k22:kbmax(i) + 2), 1) + start_k22 - 1
        if (K22(I) .GE. KBMAX(i)) then
          ierr(i) = 2
          ierrc(i) = "could not find k22"
          ktop(i) = 0
          k22(i) = 0
          kbcon(i) = 0
        endif
      endif
36  CONTINUE
    k22_0 = k22

    ! ---- :638-644 : hkb / hkbo --------------------------------------------
    do i = its, itf
      IF (ierr(I) .eq. 0) THEN
        x_add = xlv * zqexec(i) + cp * ztexec(i)
        call get_cloud_bc(kte, he_cup(i, 1:kte), hkb(i), k22(i), x_add)
        call get_cloud_bc(kte, heo_cup(i, 1:kte), hkbo(i), k22(i), x_add)
      endif
    enddo
    hkb0 = hkb
    hkbo0 = hkbo

    ! ---- :648-653 : cup_kbcon ---------------------------------------------
    iloop = 1
    call cup_kbcon(ierrc, cap_max_increment, iloop, k22, kbcon, heo_cup,     &
                   heso_cup, hkbo, ierr, kbmax, po_cup, cap_max,             &
                   ztexec, zqexec, 0, itf, ktf, its, ite, kts, kte,          &
                   z_cup, entr_rate, heo, 0)
    kbcon_1 = kbcon
    ierr_1 = ierr

    ! ---- :657-659 : cup_minimi -> kstabi -----------------------------------
    CALL cup_minimi(HEso_cup, Kbcon, kstabm, kstabi, ierr,                   &
                    itf, ktf, its, ite, kts, kte)

    ! ---- :660-685 ----------------------------------------------------------
    frh_kb(:) = 0.
    DO i = its, itf
      IF (ierr(I) == 0) THEN
        frh = min(qo_cup(i, kbcon(i)) / qeso_cup(i, kbcon(i)), 1.)
        frh_kb(i) = frh
        if (frh >= rh_thresh .and. sig(i) <= sig_thresh) then
          ierr(i) = 231
          cycle
        endif
        x_add = 0.
        do k = kbcon(i) + 1, ktf
          if (po(i, kbcon(i)) - po(i, k) > pmin + x_add) then
            pmin_lev(i) = k
            exit
          endif
        enddo
        start_level(i) = k22(i)
        x_add = xlv * zqexec(i) + cp * ztexec(i)
        call get_cloud_bc(kte, he_cup(i, 1:kte), hkb(i), k22(i), x_add)
      endif
    enddo

    ! ---- :693-726 (imid == 0 arm) ------------------------------------------
    DO i = its, itf
      if (kstabi(i) .lt. kbcon(i)) then
        kbcon(i) = 1
        ierr(i) = 42
      endif
      do k = kts, ktf
        entr_rate_2d(i, k) = entr_rate(i)
      enddo
      IF (ierr(I) .eq. 0) THEN
        kbcon(i) = max(2, kbcon(i))
        do k = kts, ktf
          frh = min(qo_cup(i, k) / qeso_cup(i, k), 1.)
          entr_rate_2d(i, k) = entr_rate(i) * (1.3 - frh)
        enddo
      endif
    ENDDO
    entr2d_a = entr_rate_2d

    ! ---- :737-738 : rates_up_pdf -------------------------------------------
    ! Replicated rather than called, so alpha/beta/fzu become visible.  The
    ! bitwise check against cup_gf is what proves the replication.
    call rates_up_pdf_deep()
    zu_pdf = zuo
    ktop_pdf = ktop
    ierr_2 = ierr

    ! ---- :743-763 ----------------------------------------------------------
    do i = its, itf
      if (ierr(i) .eq. 0) then
        if (k22(i) .gt. 1) then
          do k = 1, k22(i) - 1
            zuo(i, k) = 0.
            zu(i, k) = 0.
            xzu(i, k) = 0.
          enddo
        endif
        do k = k22(i), ktop(i)
          xzu(i, k) = zuo(i, k)
          zu(i, k) = zuo(i, k)
        enddo
        do k = ktop(i) + 1, kte
          zuo(i, k) = 0.
          zu(i, k) = 0.
          xzu(i, k) = 0.
        enddo
      endif
    enddo

    ! ---- :767-770 : get_lateral_massflux -----------------------------------
    CALL get_lateral_massflux(itf, ktf, its, ite, kts, kte,                  &
                              ierr, ktop, zo_cup, zuo, cd, entr_rate_2d,     &
                              up_massentro, up_massdetro, up_massentr,       &
                              up_massdetr, 'deep', kbcon, k22,               &
                              up_massentru, up_massdetru, lambau)

    ! ---- :777-801 ----------------------------------------------------------
    do k = kts, ktf
      do i = its, itf
        uc(i, k) = 0.
        vc(i, k) = 0.
        hc(i, k) = 0.
        dby(i, k) = 0.
        hco(i, k) = 0.
        dbyo(i, k) = 0.
      enddo
    enddo
    do i = its, itf
      IF (ierr(I) .eq. 0) THEN
        do k = 1, start_level(i)
          uc(i, k) = u_cup(i, k)
          vc(i, k) = v_cup(i, k)
        enddo
        do k = 1, start_level(i) - 1
          hc(i, k) = he_cup(i, k)
          hco(i, k) = heo_cup(i, k)
        enddo
        k = start_level(i)
        hc(i, k) = hkb(i)
        hco(i, k) = hkbo(i)
      ENDIF
    enddo

    ! ---- :803-852 : the in-cloud updraft, and ktop's revision --------------
    DO i = its, itf
      ktopkeep(i) = 0
      dbyt(i, :) = 0.
      if (ierr(i) /= 0) cycle
      ktopkeep(i) = ktop(i)
      DO k = start_level(i) + 1, ktop(i)
        denom = zuo(i, k - 1) - .5 * up_massdetro(i, k - 1) + up_massentro(i, k - 1)
        if (denom .lt. 1.e-8) then
          ierr(i) = 51
          exit
        endif
        hc(i, k) = (hc(i, k - 1) * zu(i, k - 1) - .5 * up_massdetr(i, k - 1) * hc(i, k - 1) + &
                    up_massentr(i, k - 1) * he(i, k - 1)) /                   &
                   (zu(i, k - 1) - .5 * up_massdetr(i, k - 1) + up_massentr(i, k - 1))
        uc(i, k) = (uc(i, k - 1) * zu(i, k - 1) - .5 * up_massdetru(i, k - 1) * uc(i, k - 1) + &
                    up_massentru(i, k - 1) * us(i, k - 1)                     &
                    - pgcon * .5 * (zu(i, k) + zu(i, k - 1)) * (u_cup(i, k) - u_cup(i, k - 1))) / &
                   (zu(i, k - 1) - .5 * up_massdetru(i, k - 1) + up_massentru(i, k - 1))
        vc(i, k) = (vc(i, k - 1) * zu(i, k - 1) - .5 * up_massdetru(i, k - 1) * vc(i, k - 1) + &
                    up_massentru(i, k - 1) * vs(i, k - 1)                     &
                    - pgcon * .5 * (zu(i, k) + zu(i, k - 1)) * (v_cup(i, k) - v_cup(i, k - 1))) / &
                   (zu(i, k - 1) - .5 * up_massdetru(i, k - 1) + up_massentru(i, k - 1))
        dby(i, k) = hc(i, k) - hes_cup(i, k)
        hco(i, k) = (hco(i, k - 1) * zuo(i, k - 1) - .5 * up_massdetro(i, k - 1) * hco(i, k - 1) + &
                     up_massentro(i, k - 1) * heo(i, k - 1)) /                &
                    (zuo(i, k - 1) - .5 * up_massdetro(i, k - 1) + up_massentro(i, k - 1))
        dbyo(i, k) = hco(i, k) - heso_cup(i, k)
        DZ = Zo_cup(i, K + 1) - Zo_cup(i, K)
        dbyt(i, k) = dbyt(i, k - 1) + dbyo(i, k) * dz
      ENDDO
      do k = ktop(i) - 1, kbcon(i), -1
        if (dbyo(i, k) .gt. 0.) then
          ktopkeep(i) = k + 1
          exit
        endif
      enddo
      ktop(I) = ktopkeep(i)
      if (ierr(i) .eq. 0) ktop(I) = ktopkeep(i)
    ENDDO
    ktop_dbyt = ktop
    ierr_3 = ierr

    ! ---- :854-881 ----------------------------------------------------------
    DO i = its, itf
      if (ierr(i) /= 0) cycle
      do k = ktop(i) + 1, ktf
        HC(i, K) = hes_cup(i, k)
        UC(i, K) = u_cup(i, k)
        VC(i, K) = v_cup(i, k)
        HCo(i, K) = heso_cup(i, k)
        DBY(I, K) = 0.
        DBYo(I, K) = 0.
        zu(i, k) = 0.
        zuo(i, k) = 0.
        cd(i, k) = 0.
        entr_rate_2d(i, k) = 0.
        up_massentr(i, k) = 0.
        up_massdetr(i, k) = 0.
        up_massentro(i, k) = 0.
        up_massdetro(i, k) = 0.
      enddo
    ENDDO
    DO i = its, itf
      if (ierr(i) /= 0) cycle
      if (ktop(i) .lt. kbcon(i) + 2) then
        ierr(i) = 5
        ierrc(i) = 'ktop too small deep'
        ktop(i) = 0
      endif
    ENDDO

    ! ---- :882-896 : kzdown --------------------------------------------------
    DO 37 i = its, itf
      kzdown(i) = 0
      if (ierr(i) .eq. 0) then
        zktop = (zo_cup(i, ktop(i)) - ter11(i)) * .6
        zktop = min(zktop + ter11(i), zcutdown + ter11(i))
        do k = kts, ktf
          if (zo_cup(i, k) .gt. zktop) then
            kzdown(i) = k
            kzdown(i) = min(kzdown(i), kstabi(i) - 1)
            go to 37
          endif
        enddo
      endif
37  CONTINUE

    ! ---- :900-941 : jmin ---------------------------------------------------
    call cup_minimi(HEso_cup, K22, kzdown, JMIN, ierr,                       &
                    itf, ktf, its, ite, kts, kte)
    DO 100 i = its, itf
      IF (ierr(I) .eq. 0) THEN
        jmini = jmin(i)
        keep_going = .TRUE.
        do while (keep_going)
          keep_going = .FALSE.
          if (jmini - 1 .lt. kdet(i)) kdet(i) = jmini - 1
          if (jmini .ge. ktop(i) - 1) jmini = ktop(i) - 2
          ki = jmini
          hcdo(i, ki) = heso_cup(i, ki)
          DZ = Zo_cup(i, Ki + 1) - Zo_cup(i, Ki)
          dh = 0.
          do k = ki - 1, 1, -1
            hcdo(i, k) = heso_cup(i, jmini)
            DZ = Zo_cup(i, K + 1) - Zo_cup(i, K)
            dh = dh + dz * (HCDo(i, K) - heso_cup(i, k))
            if (dh .gt. 0.) then
              jmini = jmini - 1
              if (jmini .gt. 5) then
                keep_going = .TRUE.
              else
                ierr(i) = 9
                ierrc(i) = "could not find jmini9"
                exit
              endif
            endif
          enddo
        enddo
        jmin(i) = jmini
        if (jmini .le. 5) then
          ierr(i) = 4
          ierrc(i) = "could not find jmini4"
        endif
      ENDIF
100 continue

    ! ---- :946-954 : depth_min ----------------------------------------------
    do i = its, itf
      IF (ierr(I) .eq. 0) THEN
        if (jmin(i) - 1 .lt. kdet(i)) kdet(i) = jmin(i) - 1
        IF (-zo_cup(I, KBCON(I)) + zo_cup(I, KTOP(I)) .LT. depth_min) then
          ierr(i) = 6
          ierrc(i) = "cloud depth very shallow"
        endif
      endif
    enddo
    kdet_2 = kdet
    ierr_4 = ierr

    ! ---- :960-1082 : the downdraft ------------------------------------------
    do k = kts, ktf
      do i = its, itf
        zdo(i, k) = 0.
        cdd(i, k) = 0.
        dd_massentro(i, k) = 0.
        dd_massdetro(i, k) = 0.
        dd_massentru(i, k) = 0.
        dd_massdetru(i, k) = 0.
        hcdo(i, k) = heso_cup(i, k)
        ucd(i, k) = u_cup(i, k)
        vcd(i, k) = v_cup(i, k)
        dbydo(i, k) = 0.
        mentrd_rate_2d(i, k) = entr_rate(i)
      enddo
    enddo
    do i = its, itf
      beta = max(.02, .05 - float(cactiv(i)) * .0015)
      if (xland1(i) == 0) then
        edtmax(i) = max(0.1, .4 - float(cactiv(i)) * .015)
      endif
      bud(i) = 0.
      IF (ierr(I) .eq. 0) then
        cdd(i, 1:jmin(i)) = 1.e-9
        cdd(i, jmin(i)) = 0.
        dd_massdetro(i, :) = 0.
        dd_massentro(i, :) = 0.
        call pdf_down(po_cup(i, :), kdet(i), jmin(i), zdo(i, :))
        if (zdo(i, jmin(i)) .lt. 1.e-8) then
          zdo(i, jmin(i)) = 0.
          jmin(i) = jmin(i) - 1
          if (zdo(i, jmin(i)) .lt. 1.e-8) then
            ierr(i) = 876
            cycle
          endif
        endif
        do ki = jmin(i), maxloc(zdo(i, :), 1), -1
          dzo = zo_cup(i, ki + 1) - zo_cup(i, ki)
          dd_massdetro(i, ki) = cdd(i, ki) * dzo * zdo(i, ki + 1)
          dd_massentro(i, ki) = zdo(i, ki) - zdo(i, ki + 1) + dd_massdetro(i, ki)
          if (dd_massentro(i, ki) .lt. 0.) then
            dd_massentro(i, ki) = 0.
            dd_massdetro(i, ki) = zdo(i, ki + 1) - zdo(i, ki)
            if (zdo(i, ki + 1) .gt. 0.) cdd(i, ki) = dd_massdetro(i, ki) / (dzo * zdo(i, ki + 1))
          endif
          if (zdo(i, ki + 1) .gt. 0.) mentrd_rate_2d(i, ki) = dd_massentro(i, ki) / (dzo * zdo(i, ki + 1))
        enddo
        mentrd_rate_2d(i, 1) = 0.
        do ki = maxloc(zdo(i, :), 1) - 1, 1, -1
          dzo = zo_cup(i, ki + 1) - zo_cup(i, ki)
          dd_massentro(i, ki) = mentrd_rate_2d(i, ki) * dzo * zdo(i, ki + 1)
          dd_massdetro(i, ki) = zdo(i, ki + 1) + dd_massentro(i, ki) - zdo(i, ki)
          if (dd_massdetro(i, ki) .lt. 0.) then
            dd_massdetro(i, ki) = 0.
            dd_massentro(i, ki) = zdo(i, ki) - zdo(i, ki + 1)
            if (zdo(i, ki + 1) .gt. 0.) mentrd_rate_2d(i, ki) = dd_massentro(i, ki) / (dzo * zdo(i, ki + 1))
          endif
          if (zdo(i, ki + 1) .gt. 0.) cdd(i, ki) = dd_massdetro(i, ki) / (dzo * zdo(i, ki + 1))
        enddo
        cbeg = po_cup(i, kbcon(i))
        cend = min(po_cup(i, ktop(i)), 400.)
        cmid = .5 * (cbeg + cend)
        const_b = c1 / ((cmid * cmid - cbeg * cbeg) * (cbeg - cend) / (cend * cend - cbeg * cbeg) + cmid - cbeg)
        const_a = const_b * (cbeg - cend) / (cend * cend - cbeg * cbeg)
        const_c = -const_a * cbeg * cbeg - const_b * cbeg
        do k = kbcon(i) + 1, ktop(i) - 1
          c1d(i, k) = const_a * po_cup(i, k) * po_cup(i, k) + const_b * po_cup(i, k) + const_c
          c1d(i, k) = max(0., c1d(i, k))
          c1d(i, k) = c1
        enddo
        do k = 2, jmin(i) + 1
          dd_massentru(i, k - 1) = dd_massentro(i, k - 1) + lambau(i) * dd_massdetro(i, k - 1)
          dd_massdetru(i, k - 1) = dd_massdetro(i, k - 1) + lambau(i) * dd_massdetro(i, k - 1)
        enddo
        dbydo(i, jmin(i)) = hcdo(i, jmin(i)) - heso_cup(i, jmin(i))
        bud(i) = dbydo(i, jmin(i)) * (zo_cup(i, jmin(i) + 1) - zo_cup(i, jmin(i)))
        do ki = jmin(i), 1, -1
          dzo = zo_cup(i, ki + 1) - zo_cup(i, ki)
          h_entr = .5 * (heo(i, ki) + .5 * (hco(i, ki) + hco(i, ki + 1)))
          ucd(i, ki) = (ucd(i, ki + 1) * zdo(i, ki + 1)                       &
                        - .5 * dd_massdetru(i, ki) * ucd(i, ki + 1) +         &
                        dd_massentru(i, ki) * us(i, ki)                       &
                        - pgcon * zdo(i, ki + 1) * (us(i, ki + 1) - us(i, ki))) / &
                       (zdo(i, ki + 1) - .5 * dd_massdetru(i, ki) + dd_massentru(i, ki))
          vcd(i, ki) = (vcd(i, ki + 1) * zdo(i, ki + 1)                       &
                        - .5 * dd_massdetru(i, ki) * vcd(i, ki + 1) +         &
                        dd_massentru(i, ki) * vs(i, ki)                       &
                        - pgcon * zdo(i, ki + 1) * (vs(i, ki + 1) - vs(i, ki))) / &
                       (zdo(i, ki + 1) - .5 * dd_massdetru(i, ki) + dd_massentru(i, ki))
          hcdo(i, ki) = (hcdo(i, ki + 1) * zdo(i, ki + 1)                     &
                         - .5 * dd_massdetro(i, ki) * hcdo(i, ki + 1) +       &
                         dd_massentro(i, ki) * h_entr) /                      &
                        (zdo(i, ki + 1) - .5 * dd_massdetro(i, ki) + dd_massentro(i, ki))
          dbydo(i, ki) = hcdo(i, ki) - heso_cup(i, ki)
          bud(i) = bud(i) + dbydo(i, ki) * dzo
        enddo
      endif
      if (bud(i) .gt. 0) then
        ierr(i) = 7
        ierrc(i) = 'downdraft is not negatively buoyant '
      endif
    enddo
    ierr_5 = ierr

    ! ---- :1086-1090 : cup_dd_moisture --------------------------------------
    call cup_dd_moisture(ierrc, zdo, hcdo, heso_cup, qcdo, qeso_cup,         &
                         pwdo, qo_cup, zo_cup, dd_massentro, dd_massdetro,   &
                         jmin, ierr, gammao_cup, pwevo, bu, qrcdo, qo, heo,  &
                         1, itf, ktf, its, ite, kts, kte)

    ! ---- :1102-1107 : cup_up_moisture --------------------------------------
    call cup_up_moisture('deep', ierr, zo_cup, qco, qrco, pwo, pwavo,        &
                         p_cup, kbcon, ktop, dbyo, clw_all, xland1,          &
                         qo, GAMMAo_cup, zuo, qeso_cup, k22, qo_cup,         &
                         ZQEXEC, ccn, rhoi, c1d, tn_cup, up_massentr,        &
                         up_massdetr, psum, psumh, 1, itf, ktf,              &
                         its, ite, kts, kte)

    ! ---- :1109-1117 --------------------------------------------------------
    do i = its, itf
      if (ierr(i) .eq. 0) then
        do k = kts + 1, ktop(i)
          dp = 100. * (po_cup(i, 1) - po_cup(i, 2))
          cupclw(i, k) = qrco(i, k)
          cnvwt(i, k) = zuo(i, k) * cupclw(i, k) * g / dp
        enddo
      endif
    enddo

    ! ---- :1121-1136 : cup_up_aa0 x2 ----------------------------------------
    call cup_up_aa0(aa0, z, zu, dby, GAMMA_CUP, t_cup,                       &
                    kbcon, ktop, ierr, itf, ktf, its, ite, kts, kte)
    call cup_up_aa0(aa1, zo, zuo, dbyo, GAMMAo_CUP, tn_cup,                  &
                    kbcon, ktop, ierr, itf, ktf, its, ite, kts, kte)
    do i = its, itf
      if (ierr(i) .eq. 0) then
        if (aa1(i) .eq. 0.) then
          ierr(i) = 17
          ierrc(i) = "cloud work function zero"
        endif
      endif
    enddo
    ierr_6 = ierr

    ! ---- :1141-1203 : the diurnal-cycle closure ----------------------------
    aa1_bl(:) = 0.0
    xf_dicycle(:) = 0.0
    tau_ecmwf(:) = 0.
    iversion = 1
    umean_s(:) = 0.
    DO i = its, itf
      if (ierr(i) .eq. 0) then
        wmean(i) = 7.0
        tau_ecmwf(i) = (zo_cup(i, ktopdby(i)) - zo_cup(i, kbcon(i))) / wmean(i)
        tau_ecmwf(i) = tau_ecmwf(i) * (1.0061 + 1.23E-2 * (dx(i) / 1000.))
      endif
    enddo
    tau_bl(:) = 0.
    DO i = its, itf
      if (ierr(i) .eq. 0) then
        if (xland1(i) == 0) then
          umean = 2.0 + sqrt(2.0 * (US(i, 1)**2 + VS(i, 1)**2 +              &
                                    US(i, kbcon(i))**2 + VS(i, kbcon(i))**2))
          umean_s(i) = umean
          tau_bl(i) = (zo_cup(i, kbcon(i)) - ter11(i)) / umean
        else
          tau_bl(i) = (zo_cup(i, ktopdby(i)) - zo_cup(i, kbcon(i))) / wmean(i)
        endif
      endif
    ENDDO
    t_star = 4.
    call cup_up_aa1bl(aa1_bl, t2d, tn, q2d, qo, dtstep,                      &
                      zo_cup, zuo, dbyo, GAMMAo_CUP, tn_cup,                 &
                      kbcon, ktop, ierr, itf, ktf, its, ite, kts, kte)
    DO i = its, itf
      if (ierr(i) .eq. 0) then
        if (zo_cup(i, kbcon(i)) - ter11(i) > zo(i, min(kte, kpbli(i) + 1))) then
          aa1_bl(i) = 0.0
        else
          aa1_bl(i) = max(0., aa1_bl(i) / t_star * tau_bl(i))
        endif
      endif
    ENDDO
    axx(:) = aa1(:)

    ! ---- :1297-1305 : cup_dd_edt -------------------------------------------
    call cup_dd_edt(ierr, us, vs, zo, ktop, kbcon, edt, po, pwavo,           &
                    pwo, ccn, pwevo, edtmax, edtmin, edtc, psum, psumh,      &
                    rhoi, aeroevap, itf, ktf, its, ite, kts, kte)
    do i = its, itf
      if (ierr(i) .eq. 0) then
        edto(i) = edtc(i, 1)
      endif
    enddo
    edtc1 = edtc(:, 1)
    do k = kts, ktf
      do i = its, itf
        dellat_ens(i, k, 1) = 0.
        dellaq_ens(i, k, 1) = 0.
        dellaqc_ens(i, k, 1) = 0.
        pwo_ens(i, k, 1) = 0.
      enddo
    enddo

    ! ---- :1319-1328 --------------------------------------------------------
    do k = kts, kte
      do i = its, itf
        dellu(i, k) = 0.
        dellv(i, k) = 0.
        dellah(i, k) = 0.
        dellat(i, k) = 0.
        dellaq(i, k) = 0.
        dellaqc(i, k) = 0.
      enddo
    enddo

    ! ---- :1369-1425 : momentum dellas --------------------------------------
    do i = its, itf
      if (ierr(i) .eq. 0) then
        dp = 100. * (po_cup(i, 1) - po_cup(i, 2))
        dellu(i, 1) = pgcd * (edto(i) * zdo(i, 2) * ucd(i, 2)                &
                              - edto(i) * zdo(i, 2) * u_cup(i, 2)) * g / dp
        dellv(i, 1) = pgcd * (edto(i) * zdo(i, 2) * vcd(i, 2)                &
                              - edto(i) * zdo(i, 2) * v_cup(i, 2)) * g / dp
        do k = kts + 1, ktop(i)
          pgc = pgcon
          entupk = 0.
          if (k == k22(i) - 1) entupk = zuo(i, k + 1)
          detupk = 0.
          entdoj = 0.
          detdo = edto(i) * dd_massdetro(i, k)
          entdo = edto(i) * dd_massentro(i, k)
          entup = up_massentro(i, k)
          detup = up_massdetro(i, k)
          subin = -zdo(i, k + 1) * edto(i)
          subdown = -zdo(i, k) * edto(i)
          if (k .eq. ktop(i)) then
            detupk = zuo(i, ktop(i))
            subin = 0.
            subdown = 0.
            detdo = 0.
            entdo = 0.
            entup = 0.
            detup = 0.
          endif
          totmas = subin - subdown + detup - entup - entdo +                 &
                   detdo - entupk - entdoj + detupk + zuo(i, k + 1) - zuo(i, k)
          dp = 100. * (po_cup(i, k) - po_cup(i, k + 1))
          pgc = pgcon
          if (k .ge. ktop(i)) pgc = 0.
          dellu(i, k) = -(zuo(i, k + 1) * (uc(i, k + 1) - u_cup(i, k + 1)) -  &
                          zuo(i, k) * (uc(i, k) - u_cup(i, k))) * g / dp      &
                        + (zdo(i, k + 1) * (ucd(i, k + 1) - u_cup(i, k + 1)) - &
                           zdo(i, k) * (ucd(i, k) - u_cup(i, k))) * g / dp * edto(i) * pgcd
          dellv(i, k) = -(zuo(i, k + 1) * (vc(i, k + 1) - v_cup(i, k + 1)) -  &
                          zuo(i, k) * (vc(i, k) - v_cup(i, k))) * g / dp      &
                        + (zdo(i, k + 1) * (vcd(i, k + 1) - v_cup(i, k + 1)) - &
                           zdo(i, k) * (vcd(i, k) - v_cup(i, k))) * g / dp * edto(i) * pgcd
        enddo
      endif
    enddo

    ! ---- :1428-1495 : thermodynamic dellas ---------------------------------
    do i = its, itf
      if (ierr(i) .eq. 0) then
        dp = 100. * (po_cup(i, 1) - po_cup(i, 2))
        dellah(i, 1) = (edto(i) * zdo(i, 2) * hcdo(i, 2)                      &
                        - edto(i) * zdo(i, 2) * heo_cup(i, 2)) * g / dp
        dellaq(i, 1) = (edto(i) * zdo(i, 2) * qcdo(i, 2)                      &
                        - edto(i) * zdo(i, 2) * qo_cup(i, 2)) * g / dp
        G_rain = 0.5 * (pwo(i, 1) + pwo(i, 2)) * g / dp
        E_dn = -0.5 * (pwdo(i, 1) + pwdo(i, 2)) * g / dp * edto(i)
        dellaq(i, 1) = dellaq(i, 1) + E_dn - G_rain
        do k = kts + 1, ktop(i)
          dp = 100. * (po_cup(i, k) - po_cup(i, k + 1))
          dellah(i, k) = -(zuo(i, k + 1) * (hco(i, k + 1) - heo_cup(i, k + 1)) - &
                           zuo(i, k) * (hco(i, k) - heo_cup(i, k))) * g / dp  &
                         + (zdo(i, k + 1) * (hcdo(i, k + 1) - heo_cup(i, k + 1)) - &
                            zdo(i, k) * (hcdo(i, k) - heo_cup(i, k))) * g / dp * edto(i)
          detup = up_massdetro(i, k)
          dz = zo_cup(i, k) - zo_cup(i, k - 1)
          if (k .lt. ktop(i)) dellaqc(i, k) = zuo(i, k) * c1d(i, k) * qrco(i, k) * dz / dp * g
          if (k .eq. ktop(i)) dellaqc(i, k) = detup * 0.5 * (qrco(i, k + 1) + qrco(i, k)) * g / dp
          G_rain = 0.5 * (pwo(i, k) + pwo(i, k + 1)) * g / dp
          E_dn = -0.5 * (pwdo(i, k) + pwdo(i, k + 1)) * g / dp * edto(i)
          C_up = dellaqc(i, k) + (zuo(i, k + 1) * qrco(i, k + 1) -            &
                                  zuo(i, k) * qrco(i, k)) * g / dp + G_rain
          dellaq(i, k) = -(zuo(i, k + 1) * (qco(i, k + 1) - qo_cup(i, k + 1)) - &
                           zuo(i, k) * (qco(i, k) - qo_cup(i, k))) * g / dp   &
                         + (zdo(i, k + 1) * (qcdo(i, k + 1) - qo_cup(i, k + 1)) - &
                            zdo(i, k) * (qcdo(i, k) - qo_cup(i, k))) * g / dp * edto(i) &
                         - C_up + E_dn
        enddo
      endif
    enddo

    ! ---- :1500-1524 : the mbdt-perturbed state -----------------------------
    mbdt = .1
    do i = its, itf
      xaa0_ens(i, 1) = 0.
    enddo
    do i = its, itf
      if (ierr(i) .eq. 0) then
        do k = kts, ktf
          XHE(I, K) = DELLAH(I, K) * MBDT + HEO(I, K)
          XQ(I, K) = max(1.e-16, DELLAQ(I, K) * MBDT + QO(I, K))
          DELLAT(I, K) = (1. / cp) * (DELLAH(I, K) - xlv * DELLAQ(I, K))
          XT(I, K) = DELLAT(I, K) * MBDT + TN(I, K)
          xt(i, k) = max(190., xt(i, k))
        enddo
      ENDIF
    enddo
    do i = its, itf
      if (ierr(i) .eq. 0) then
        XHE(I, ktf) = HEO(I, ktf)
        XQ(I, ktf) = QO(I, ktf)
        XT(I, ktf) = TN(I, ktf)
      endif
    enddo

    ! ---- :1528-1539 : cup_env / cup_env_clev on the perturbed state --------
    call cup_env(xz, xqes, xhe, xhes, xt, xq, po, ter11,                     &
                 psur, ierr, tcrit_l, -1, itf, ktf, its, ite, kts, kte)
    call cup_env_clev(xt, xqes, xq, xhe, xhes, xz, po, xqes_cup, xq_cup,     &
                      xhe_cup, xhes_cup, xz_cup, po_cup, gamma_cup, xt_cup,  &
                      psur, ierr, ter11, itf, ktf, its, ite, kts, kte)

    ! ---- :1546-1578 --------------------------------------------------------
    do k = kts, ktf
      do i = its, itf
        xhc(i, k) = 0.
        xDBY(I, K) = 0.
      enddo
    enddo
    do i = its, itf
      if (ierr(i) .eq. 0) then
        x_add = xlv * zqexec(i) + cp * ztexec(i)
        call get_cloud_bc(kte, xhe_cup(i, 1:kte), xhkb(i), k22(i), x_add)
        do k = 1, start_level(i) - 1
          xhc(i, k) = xhe_cup(i, k)
        enddo
        k = start_level(i)
        xhc(i, k) = xhkb(i)
      endif
    enddo
    do i = its, itf
      if (ierr(i) .eq. 0) then
        do k = start_level(i) + 1, ktop(i)
          xhc(i, k) = (xhc(i, k - 1) * xzu(i, k - 1) - .5 * up_massdetro(i, k - 1) * xhc(i, k - 1) + &
                       up_massentro(i, k - 1) * xhe(i, k - 1)) /              &
                      (xzu(i, k - 1) - .5 * up_massdetro(i, k - 1) + up_massentro(i, k - 1))
          xdby(i, k) = xhc(i, k) - xhes_cup(i, k)
        enddo
        do k = ktop(i) + 1, ktf
          xHC(i, K) = xhes_cup(i, k)
          xDBY(I, K) = 0.
        enddo
      endif
    enddo

    ! ---- :1583-1586 : cup_up_aa0 on the perturbed state --------------------
    call cup_up_aa0(xaa0, xz, xzu, xdby, GAMMA_CUP, xt_cup,                  &
                    kbcon, ktop, ierr, itf, ktf, its, ite, kts, kte)

    ! ---- :1587-1623 : pr_ens -----------------------------------------------
    do i = its, itf
      if (ierr(i) .eq. 0) then
        xaa0_ens(i, 1) = xaa0(i)
        do k = kts, ktop(i)
          do nens3 = 1, maxens3
            if (nens3 .eq. 7) then
              pr_ens(i, nens3) = pr_ens(i, nens3) + pwo(i, k) + edto(i) * pwdo(i, k)
            else if (nens3 .eq. 8) then
              pr_ens(i, nens3) = pr_ens(i, nens3) + pwo(i, k) + edto(i) * pwdo(i, k)
            else if (nens3 .eq. 9) then
              pr_ens(i, nens3) = pr_ens(i, nens3) + pwo(i, k) + edto(i) * pwdo(i, k)
            else
              pr_ens(i, nens3) = pr_ens(i, nens3) + pwo(i, k) + edto(i) * pwdo(i, k)
            endif
          enddo
        enddo
        if (pr_ens(i, 7) .lt. 1.e-6) then
          ierr(i) = 18
          ierrc(i) = "total normalized condensate too small"
          do nens3 = 1, maxens3
            pr_ens(i, nens3) = 0.
          enddo
        endif
        do nens3 = 1, maxens3
          if (pr_ens(i, nens3) .lt. 1.e-5) then
            pr_ens(i, nens3) = 0.
          endif
        enddo
      endif
    enddo
    ierr_7 = ierr

    ! ---- :1633-1654 : the ierr2 / ierr3 cap probes -------------------------
    do i = its, itf
      ierr2(i) = ierr(i)
      ierr3(i) = ierr(i)
      k22x(i) = k22(i)
    enddo
    CALL cup_MAXIMI(HEO_CUP, 2, KBMAX, K22x, ierr, itf, ktf, its, ite, kts, kte)
    iloop = 2
    call cup_kbcon(ierrc, cap_max_increment, iloop, k22x, kbconx, heo_cup,   &
                   heso_cup, hkbo, ierr2, kbmax, po_cup, cap_max,            &
                   ztexec, zqexec, 0, itf, ktf, its, ite, kts, kte,          &
                   z_cup, entr_rate, heo, 0)
    iloop = 3
    call cup_kbcon(ierrc, cap_max_increment, iloop, k22x, kbconx, heo_cup,   &
                   heso_cup, hkbo, ierr3, kbmax, po_cup, cap_max,            &
                   ztexec, zqexec, 0, itf, ktf, its, ite, kts, kte,          &
                   z_cup, entr_rate, heo, 0)

    ! ---- :1659-1666 : mconv, recomputed on the cloud grid -------------------
    DO I = its, itf
      mconv(i) = 0
      if (ierr(i) /= 0) cycle
      DO K = 1, ktop(i)
        dq = (qo_cup(i, k + 1) - qo_cup(i, k))
        mconv(i) = mconv(i) + omeg(i, k) * dq / g
      ENDDO
    ENDDO
    mconv2 = mconv

    ! ---- :1667-1674 : cup_forcing_ens_3d -----------------------------------
    call cup_forcing_ens_3d(closure_n, xland1, aa0, aa1, xaa0_ens, mbdt,     &
                            dtstep, ierr, ierr2, ierr3, xf_ens, axx,         &
                            forcing, maxens3, mconv, rand_clos,              &
                            po_cup, ktop, omeg, zdo, k22, zuo, pr_ens, edto, &
                            kbcon, ichoice, 0, 0, itf, ktf,                  &
                            its, ite, kts, kte,                              &
                            dicycle, tau_ecmwf, aa1_bl, xf_dicycle)

    ! ---- :1676-1690 --------------------------------------------------------
    do k = kts, ktf
      do i = its, itf
        if (ierr(i) .eq. 0) then
          dellat_ens(i, k, 1) = dellat(i, k)
          dellaq_ens(i, k, 1) = dellaq(i, k)
          dellaqc_ens(i, k, 1) = dellaqc(i, k)
          pwo_ens(i, k, 1) = pwo(i, k)
        else
          dellat_ens(i, k, 1) = 0.
          dellaq_ens(i, k, 1) = 0.
          dellaqc_ens(i, k, 1) = 0.
          pwo_ens(i, k, 1) = 0.
        endif
      enddo
    enddo

    ! ---- :1715-1723 : cup_output_ens_3d ------------------------------------
    call cup_output_ens_3d(xff_mid, xf_ens, ierr, dellat_ens, dellaq_ens,    &
                           dellaqc_ens, outt, outq, outqc, zuo, pre,         &
                           pwo_ens, xmb, ktop, edto, pwdo, 'deep',           &
                           ierr2, ierr3, po_cup, pr_ens, maxens3,            &
                           sig, closure_n, xland1, xmbm_in, xmbs,            &
                           ichoice, 0, 0, itf, ktf, its, ite, kts, kte,      &
                           dicycle, xf_dicycle)
    outt_o = outt
    outq_o = outq
    outqc_o = outqc

    ! ---- :1724-1743 --------------------------------------------------------
    k = 1
    do i = its, itf
      if (ierr(i) .eq. 0 .and. pre(i) .gt. 0.) then
        PRE(I) = MAX(PRE(I), 0.)
        xmb_out(i) = xmb(i)
        do k = kts, ktop(i)
          outu(i, k) = dellu(i, k) * xmb(i)
          outv(i, k) = dellv(i, k) * xmb(i)
        enddo
      elseif (ierr(i) .ne. 0 .or. pre(i) .eq. 0.) then
        ktop(i) = 0
        do k = kts, kte
          outt(i, k) = 0.
          outq(i, k) = 0.
          outqc(i, k) = 0.
          outu(i, k) = 0.
          outv(i, k) = 0.
        enddo
      endif
    enddo

    ! ---- :1803-1821 : dissipative heating ----------------------------------
    do i = its, itf
      if (ierr(i) .eq. 0) then
        dts = 0.
        fpi = 0.
        do k = kts, ktop(i)
          dp = (po_cup(i, k) - po_cup(i, k + 1)) * 100.
          dts = dts - (outu(i, k) * us(i, k) + outv(i, k) * vs(i, k)) * dp / g
          fpi = fpi + sqrt(outu(i, k) * outu(i, k) + outv(i, k) * outv(i, k)) * dp
        enddo
        if (fpi .gt. 0.) then
          do k = kts, ktop(i)
            fp = sqrt((outu(i, k) * outu(i, k) + outv(i, k) * outv(i, k))) / fpi
            outt(i, k) = outt(i, k) + fp * dts * g / cp
          enddo
        endif
      endif
    enddo

    ! ---- get_inversion_layers, captured for the shallow port ---------------
    ! WRF calls this only from CUP_gf_sh (module_cu_gf_sh.F:413) and from the
    ! dead imid == 1 arm (:690), always as (kstart, kend) = (kbcon, kstabi).
    ! It reads t_cup(kend+8), which is out of bounds for kend > ktf-8, so the
    ! capture clamps kend and counts the clamps rather than emitting an
    ! undefined read.
    dtempdz(:, :) = 0.
    k_inv_layers(:, :) = 1
    kinv_clamped(:) = 0
    do i = its, itf
      if (ierr_6(i) == 0) then
        kend_gil = kstabi(i)
        if (kend_gil > ktf - 8) then
          kend_gil = ktf - 8
          kinv_clamped(i) = 1
          nclamp = nclamp + 1
        endif
        call get_inversion_layers(ierr_6, p_cup, t_cup, z_cup, q_cup,        &
                                  qes_cup, k_inv_layers, kbcon,              &
                                  (/kend_gil/), dtempdz, itf, ktf,           &
                                  its, ite, kts, kte)
      endif
    enddo
  end subroutine replicate_cup_gf

  ! ==========================================================================
  ! rates_up_pdf's 'deep' arm, module_cu_gf_deep.F:3697-3823.
  ! ==========================================================================
  subroutine rates_up_pdf_deep()
    real :: dby_l(kts:kte), dbm_l(kts:kte), zux(kts:kte), hcot_l(kts:kte)
    real :: zustart, dbythresh, dz_l, massent, massdetr, zubeg, beta_u
    integer :: kklev, kfinalzu, il, kl

    zustart = .1
    dbythresh = 1.
    dby_l(:) = 0.
    DO il = its, itf
      zux(:) = 0.
      beta_u = max(.1, .2 - float(cactiv(il)) * .01)
      zuo(il, :) = 0.
      dby_l(:) = 0.
      dbm_l(:) = 0.
      kbcon(il) = max(kbcon(il), 2)
      up_tun(il) = 0.; up_alpha(il) = 0.; up_beta(il) = 0.; up_fzu(il) = 0.
      up_kbadj(il) = 0; up_kklev(il) = 0; up_kfinal(il) = 0
      if (ierr(il) .eq. 0) then
        start_level(il) = k22(il)
        zuo(il, start_level(il)) = zustart
        zux(start_level(il)) = zustart
        do kl = start_level(il) + 1, kbcon(il)
          dz_l = zo_cup(il, kl) - zo_cup(il, kl - 1)
          massent = dz_l * entr_rate_2d(il, kl - 1) * zuo(il, kl - 1)
          massdetr = dz_l * 1.e-9 * zuo(il, kl - 1)
          zuo(il, kl) = zuo(il, kl - 1) + massent - massdetr
          zux(kl) = zuo(il, kl)
        enddo
        zubeg = zustart
        ktop(il) = 0
        hcot_l(start_level(il)) = hkbo(il)
        dz_l = zo_cup(il, start_level(il)) - zo_cup(il, start_level(il) - 1)
        do kl = start_level(il) + 1, ktf - 2
          dz_l = zo_cup(il, kl) - zo_cup(il, kl - 1)
          hcot_l(kl) = ((1. - 0.5 * entr_rate_2d(il, kl - 1) * dz_l) * hcot_l(kl - 1) &
                        + entr_rate_2d(il, kl - 1) * dz_l * heo(il, kl - 1)) /  &
                       (1. + 0.5 * entr_rate_2d(il, kl - 1) * dz_l)
          if (kl >= kbcon(il)) dby_l(kl) = dby_l(kl - 1) + (hcot_l(kl) - heso_cup(il, kl)) * dz_l
          if (kl >= kbcon(il)) dbm_l(kl) = hcot_l(kl) - heso_cup(il, kl)
        enddo
        ktopdby(il) = maxloc(dby_l(:), 1)
        kklev = maxloc(dbm_l(:), 1)
        kfinalzu = ktf - 2
        ktop(il) = kfinalzu
        do kl = maxloc(dby_l(:), 1) + 1, ktf - 2
          if (dby_l(kl) .lt. dbythresh * maxval(dby_l)) then
            kfinalzu = kl - 1
            ktop(il) = kfinalzu
            exit
          endif
        enddo
        up_kklev(il) = kklev
        up_kfinal(il) = kfinalzu
        if (kfinalzu .le. kbcon(il) + 2) then
          ierr(il) = 41
          ktop(il) = 0
        else
          call pdf_up(po_cup(il, :), k22(il), kfinalzu, kstabi(il),          &
                      cactiv(il), zubeg, zuo(il, :), il)
        endif
      endif
    ENDDO
  end subroutine rates_up_pdf_deep

  ! get_zu_zd_pdf_fim, draft == "UP" (:3845-3874).
  subroutine pdf_up(p, kb, kt, kpbli, csum_i, zubeg, zu, islot)
    real, intent(in) :: p(kts:kte), zubeg
    integer, intent(in) :: kb, kt, kpbli, csum_i, islot
    real, intent(inout) :: zu(kts:kte)
    integer :: kb_adj, kl
    real :: lev_start, tunning, beta_l, alpha_l, fzu, kratio

    zu = 0.0
    kb_adj = max(kb, 2)
    lev_start = min(.9, .4 + csum_i * .013)
    kb_adj = max(kb, 2)
    tunning = p(kt) + (p(kpbli) - p(kt)) * lev_start
    tunning = min(0.9, (tunning - p(kb_adj)) / (p(kt) - p(kb_adj)))
    tunning = max(0.2, tunning)
    beta_l = 1.3
    alpha_l = (tunning * (beta_l - 2.) + 1.) / (1. - tunning)
    fzu = gamma(alpha_l + beta_l) / (gamma(alpha_l) * gamma(beta_l))
    do kl = kb_adj, min(kte, kt)
      kratio = (p(kl) - p(kb_adj)) / (p(kt) - p(kb_adj))
      zu(kl) = zubeg + FZU * kratio**(alpha_l - 1.0) * (1.0 - kratio)**(beta_l - 1.0)
    enddo
    if (maxval(zu(kts:min(ktf, kt + 1))) .gt. 0.)                            &
      zu(kts:min(ktf, kt + 1)) = zu(kts:min(ktf, kt + 1)) / maxval(zu(kts:min(ktf, kt + 1)))
    do kl = maxloc(zu(:), 1), 1, -1
      if (zu(kl) .lt. 1.e-6) then
        kb_adj = kl + 1
        exit
      endif
    enddo
    kb_adj = max(2, kb_adj)
    do kl = kts, kb_adj - 1
      zu(kl) = 0.
    enddo
    up_tun(islot) = tunning
    up_alpha(islot) = alpha_l
    up_beta(islot) = beta_l
    up_fzu(islot) = fzu
    up_kbadj(islot) = kb_adj
  end subroutine pdf_up

  ! get_zu_zd_pdf_fim, draft == "DOWN" (:3937-3981).  max_mass, kpbli, csum
  ! and pmin_lev are all dead on this branch; so is kb_adj, which is computed
  ! and then not used -- the branch indexes p(kb) directly.
  subroutine pdf_down(p, kb, kt, zu)
    real, intent(in) :: p(kts:kte)
    integer, intent(in) :: kb, kt
    real, intent(inout) :: zu(kts:kte)
    integer :: kb_adj, kl
    real :: tunning, beta_l, alpha_l, fzu, kratio

    zu = 0.0
    kb_adj = max(kb, 2)
    tunning = p(kb)
    tunning = min(0.9, (tunning - p(1)) / (p(kt) - p(1)))
    tunning = max(0.2, tunning)
    beta_l = 4.
    alpha_l = (tunning * (beta_l - 2.) + 1.) / (1. - tunning)
    fzu = gamma(alpha_l + beta_l) / (gamma(alpha_l) * gamma(beta_l))
    dn_tun(its) = tunning
    dn_alpha(its) = alpha_l
    dn_beta(its) = beta_l
    dn_fzu(its) = fzu
    dn_kbadj(its) = kb_adj
    zu(:) = 0.
    do kl = 2, min(kt, ktf)
      kratio = (p(kl) - p(1)) / (p(kt) - p(1))
      zu(kl) = FZU * kratio**(alpha_l - 1.0) * (1.0 - kratio)**(beta_l - 1.0)
    enddo
    fzu = maxval(zu(kts:min(ktf, kt + 1)))
    if (fzu .gt. 0.)                                                         &
      zu(kts:min(ktf, kt + 1)) = zu(kts:min(ktf, kt + 1)) / fzu
    zu(1) = 0.
  end subroutine pdf_down

  ! ==========================================================================
  ! The same prepared column, through the module's own cup_gf.
  ! ==========================================================================
  subroutine run_real_cup_gf(dx_in)
    real, intent(in) :: dx_in
    real, dimension(its:ite) :: rdx, r_xmbm, r_xmbs

    r_outt = 0.; r_outq = 0.; r_outqc = 0.; r_outu = 0.; r_outv = 0.
    r_cupclw = 0.; r_cnvwt = 0.; r_zu = 0.; r_zd = 0.
    r_pre = 0.; r_xmb = 0.; r_xmb_out = 0.; r_edt = 0.
    r_ierr = 0; r_kbcon = 0; r_ktop = 0; r_k22 = 0; r_jmin = 0
    r_forcing = 0.
    r_q = q2d
    r_qo = qo
    r_omeg = omeg
    r_mconv(:) = 0.
    rdx(:) = dx_in
    r_xmbm(:) = 0.
    r_xmbs = xmbs
    ierrc(:) = " "
    ! mconv, as GFDRV hands it over: the driver's column integral.
    call rebuild_mconv(r_mconv)
    call cup_gf(itf, ktf, its, ite, kts, kte, 1, ichoice, 0, ccn, dtstep, 0, &
                kpbli, dhdt, xlandi, zo, r_forcing, t2d, r_q, ter11, tn,     &
                r_qo, p2d, psur, us, vs, rhoi, hfxi, qfxi, rdx, r_mconv,     &
                r_omeg, cactiv, r_cnvwt, r_zu, r_zd, r_edt, r_xmb, r_xmbm,   &
                r_xmbs, r_pre, r_outu, r_outv, r_outt, r_outq, r_outqc,      &
                r_kbcon, r_ktop, r_cupclw, r_ierr, ierrc,                    &
                rand_mom, rand_vmas, rand_clos, 0, r_k22, r_jmin)
    r_xmb_out = r_xmb
  end subroutine run_real_cup_gf

  ! ==========================================================================
  subroutine compare_and_emit(ic, idx, iarm_v)
    integer, intent(in) :: ic, idx, iarm_v
    integer :: kk2

    ndiff = 0
    do kk2 = kts, ktf
      ndiff = ndiff                                                          &
        + bitdiff(outt(its, kk2), r_outt(its, kk2))                          &
        + bitdiff(outq(its, kk2), r_outq(its, kk2))                          &
        + bitdiff(outqc(its, kk2), r_outqc(its, kk2))                        &
        + bitdiff(outu(its, kk2), r_outu(its, kk2))                          &
        + bitdiff(outv(its, kk2), r_outv(its, kk2))                          &
        + bitdiff(cupclw(its, kk2), r_cupclw(its, kk2))                      &
        + bitdiff(cnvwt(its, kk2), r_cnvwt(its, kk2))                        &
        + bitdiff(zuo(its, kk2), r_zu(its, kk2))                             &
        + bitdiff(zdo(its, kk2), r_zd(its, kk2))
    enddo
    ndiff = ndiff + bitdiff(pre(its), r_pre(its))                            &
                  + bitdiff(xmb(its), r_xmb(its))                            &
                  + bitdiff(edto(its), r_edt(its))
    if (ktop(its) /= r_ktop(its)) ndiff = ndiff + 1
    if (kbcon(its) /= r_kbcon(its)) ndiff = ndiff + 1
    if (k22(its) /= r_k22(its)) ndiff = ndiff + 1
    if (jmin(its) /= r_jmin(its)) ndiff = ndiff + 1
    if (ierr(its) /= r_ierr(its)) ndiff = ndiff + 1
    do kk2 = 1, 10
      ndiff = ndiff + bitdiff(forcing(its, kk2), r_forcing(its, kk2))
    enddo
    write(ucon, '(I0,3(",",I0),2(",",I0))') ic, idx, iarm_v, ndiff,          &
      ierr(its), r_ierr(its)

    do k = kts, ktf
      write(ulev, '(I0,3(",",I0))', advance='no') ic, idx, iarm_v, k
      write(ulev, '(20(",",ES24.16E3))', advance='no')                       &
        qes(its, k), he(its, k), hes(its, k),                                &
        qeso(its, k), heo(its, k), heso(its, k),                             &
        qes_cup(its, k), q_cup(its, k), he_cup(its, k), hes_cup(its, k),     &
        gamma_cup1(its, k), t_cup(its, k),                                   &
        qeso_cup(its, k), qo_cup(its, k), heo_cup(its, k), heso_cup(its, k), &
        zo_cup(its, k), po_cup(its, k), gammao_cup(its, k), tn_cup(its, k)
      write(ulev, '(17(",",ES24.16E3))', advance='no')                       &
        u_cup(its, k), v_cup(its, k), entr2d_a(its, k), zu_pdf(its, k),      &
        cd(its, k), entr_rate_2d(its, k), up_massentro(its, k),              &
        up_massdetro(its, k), up_massentru(its, k), up_massdetru(its, k),    &
        hc(its, k), uc(its, k), vc(its, k), hco(its, k), dby(its, k),        &
        dbyo(its, k), dbyt(its, k)
      write(ulev, '(18(",",ES24.16E3))', advance='no')                       &
        cdd(its, k), dd_massentro(its, k), dd_massdetro(its, k),             &
        dd_massentru(its, k), dd_massdetru(its, k), mentrd_rate_2d(its, k),  &
        hcdo(its, k), ucd(its, k), vcd(its, k), dbydo(its, k), c1d(its, k),  &
        qcdo(its, k), qrcdo(its, k), pwdo(its, k),                           &
        qco(its, k), qrco(its, k), pwo(its, k), clw_all(its, k)
      write(ulev, '(11(",",ES24.16E3))', advance='no')                       &
        dellu(its, k), dellv(its, k), dellah(its, k), dellaq(its, k),        &
        dellaqc(its, k), dellat(its, k),                                     &
        xhe(its, k), xq(its, k), xt(its, k), xqes(its, k), xhes(its, k)
      write(ulev, '(15(",",ES24.16E3))')                                     &
        xqes_cup(its, k), xq_cup(its, k), xhe_cup(its, k), xhes_cup(its, k), &
        gamma_cup(its, k), xt_cup(its, k), xhc(its, k), xdby(its, k),        &
        outt_o(its, k), outq_o(its, k), outqc_o(its, k), outt(its, k),       &
        outu(its, k), outv(its, k), dtempdz(its, k)
    enddo

    write(usfc, '(I0,2(",",I0))', advance='no') ic, idx, iarm_v
    write(usfc, '(4(",",ES24.16E3),",",I0,3(",",ES24.16E3))', advance='no')  &
      zws(its), ztexec(its), zqexec(its), cap_max(its), xland1(its),         &
      entr_rate(its), sig(its), sig_thresh
    write(usfc, '(3(",",I0),2(",",ES24.16E3),2(",",I0),",",ES24.16E3,",",I0)', &
      advance='no') kbmax(its), kdet(its), k22_0(its), hkb0(its),            &
      hkbo0(its), k22(its), kbcon_1(its), hkb(its), ierr_1(its)
    write(usfc, '(2(",",I0),",",ES24.16E3,2(",",I0))', advance='no')         &
      kstabi(its), kstabm(its), frh_kb(its), pmin_lev(its), start_level(its)
    write(usfc, '(6(",",I0))', advance='no') ktop_pdf(its), ktopdby(its),    &
      kbcon(its), ierr_2(its), ktop_dbyt(its), ierr_3(its)
    write(usfc, '(3(",",I0),2(",",ES24.16E3),",",I0)', advance='no')         &
      kzdown(its), jmin(its), kdet_2(its), beta, edtmax(its), ierr_4(its)
    write(usfc, '(3(",",ES24.16E3),",",I0,3(",",ES24.16E3))', advance='no')  &
      bud(its), pwevo(its), bu(its), ierr_5(its), pwavo(its), psum(its),     &
      psumh(its)
    write(usfc, '(2(",",ES24.16E3),",",I0,4(",",ES24.16E3))', advance='no')  &
      aa0(its), aa1(its), ierr_6(its), tau_ecmwf(its), tau_bl(its),          &
      aa1_bl(its), umean_s(its)
    write(usfc, '(5(",",ES24.16E3),",",ES24.16E3,",",I0)', advance='no')     &
      edt(its), edtc1(its), edto(its), xhkb(its), xaa0(its),                 &
      pr_ens(its, 7), ierr_7(its)
    write(usfc, '(4(",",I0),",",ES24.16E3)', advance='no')                   &
      k22x(its), kbconx(its), ierr2(its), ierr3(its), mconv2(its)
    write(usfc, '(18(",",ES24.16E3))', advance='no')                         &
      (xf_ens(its, n), n = 1, 16), xf_dicycle(its), closure_n(its)
    write(usfc, '(10(",",ES24.16E3))', advance='no') (forcing(its, n), n = 1, 10)
    write(usfc, '(2(",",ES24.16E3),2(",",I0),6(",",I0))', advance='no')      &
      xmb(its), pre(its), ktop(its), ierr(its),                              &
      (k_inv_layers(its, n), n = 1, 5), kinv_clamped(its)
    write(usfc, '(4(",",ES24.16E3),3(",",I0),4(",",ES24.16E3),",",I0)')      &
      up_tun(its), up_alpha(its), up_beta(its), up_fzu(its),                 &
      up_kbadj(its), up_kklev(its), up_kfinal(its),                          &
      dn_tun(its), dn_alpha(its), dn_beta(its), dn_fzu(its), dn_kbadj(its)
  end subroutine compare_and_emit

  ! GFDRV's mconv (module_cu_gf_wrfdrv.F:484-492), which cup_gf overwrites at
  ! :1660 but reads nowhere before -- rebuilt here so the reference call
  ! receives exactly the word the driver would have handed it.
  subroutine rebuild_mconv(mc)
    real, dimension(its:ite), intent(out) :: mc
    real :: dqv
    integer :: kk3
    mc(its) = 0.
    do kk3 = kts, ktf - 1
      dqv = q2d(its, kk3 + 1) - q2d(its, kk3)
      mc(its) = mc(its) + omeg(its, kk3) * dqv / con_g
    enddo
    if (mc(its) < 0.) mc(its) = 0.
  end subroutine rebuild_mconv

  ! GFDRV's own column preparation, module_cu_gf_wrfdrv.F:383-492.  Identical
  ! to run_cup_gf.F90's, and for the same reason: both decompositions must
  ! consume the same array words as the driver.
  subroutine prepare_column(ic, dxv_in)
    integer, intent(in) :: ic
    real, intent(in) :: dxv_in
    real :: dqv
    integer :: kk4, i0, j0

    i0 = gits
    j0 = gjts

    dxi(its) = dxv_in
    psur(its) = gp8w(i0, 1, j0) * .01
    ter11(its) = max(0., ght(i0, j0))
    hfxi(its) = ghfx(i0, j0)
    qfxi(its) = gqfx(i0, j0)
    xlandi(its) = gxland(i0, j0)
    kpbli(its) = gkpbl(i0, j0)
    ccn(its) = 150.
    mconv(its) = 0.

    zo(its, kts) = ter11(its) + .5 * gdz(i0, 1, j0)
    do kk4 = kts + 1, ktf
      zo(its, kk4) = zo(its, kk4 - 1) + .5 * (gdz(i0, kk4 - 1, j0) + gdz(i0, kk4, j0))
    end do

    do kk4 = kts, ktf
      po(its, kk4) = gp(i0, kk4, j0) * .01
      pi_col(kk4) = gpi(i0, kk4, j0)
      p2d(its, kk4) = po(its, kk4)
      rhoi(its, kk4) = grho(i0, kk4, j0)
      us(its, kk4) = gu(i0, kk4, j0)
      vs(its, kk4) = gv(i0, kk4, j0)
      t2d(its, kk4) = gt(i0, kk4, j0)
      q2d(its, kk4) = gq(i0, kk4, j0)
      if (q2d(its, kk4) < 1.e-08) q2d(its, kk4) = 1.e-08

      tn(its, kk4) = t2d(its, kk4) + (gthften(i0, kk4, j0) + gthraten(i0, kk4, j0) &
                                      + gthblten(i0, kk4, j0))                &
                                     * gpi(i0, kk4, j0) * dtstep
      qo(its, kk4) = q2d(its, kk4) + (gqvften(i0, kk4, j0) + gqvblten(i0, kk4, j0)) &
                                     * dtstep
      tshall(its, kk4) = t2d(its, kk4) + gthblten(i0, kk4, j0) * gpi(i0, kk4, j0) &
                                         * dtstep
      dhdt(its, kk4) = con_cp * gthblten(i0, kk4, j0) * gpi(i0, kk4, j0)      &
                       + con_hvap * gqvblten(i0, kk4, j0)
      qshall(its, kk4) = q2d(its, kk4) + gqvblten(i0, kk4, j0) * dtstep
      if (tn(its, kk4) < 200.) tn(its, kk4) = t2d(its, kk4)
      if (qo(its, kk4) < 1.e-08) qo(its, kk4) = 1.e-08

      omeg(its, kk4) = -con_g * grho(i0, kk4, j0) * gw(i0, kk4, j0)
    end do

    do kk4 = kts, ktf - 1
      dqv = q2d(its, kk4 + 1) - q2d(its, kk4)
      mconv(its) = mconv(its) + omeg(its, kk4) * dqv / con_g
    end do
    if (mconv(its) < 0.) mconv(its) = 0.
  end subroutine prepare_column

  subroutine fill_driver_tile(ic)
    integer, intent(in) :: ic
    real :: zc(nz), dz_l(nz), tt(nz), qq(nz), pp(nz), ppw(nz + 1)
    real :: ppi(nz), rr(nz), uu(nz), vv(nz), ww(nz)
    integer :: ii, kk5, jj

    gu = 0.0; gv = 0.0; gw = 0.0; gt = 0.0; gq = 0.0
    gp = 0.0; gpi = 1.0; grho = 0.0; gdz = 0.0; gp8w = 0.0
    gthften = 0.0; gqvften = 0.0; gthraten = 0.0
    gthblten = 0.0; gqvblten = 0.0
    ght = 0.0; ghfx = 0.0; gqfx = 0.0; gxland = 1.0
    gkpbl = 1

    call gf_column(ic, zc, dz_l, tt, qq, pp, ppw, ppi, rr, uu, vv, ww)
    ii = gits
    do jj = gjds, gjde
      do kk5 = 1, nz
        gu(ii, kk5, jj) = uu(kk5); gv(ii, kk5, jj) = vv(kk5)
        gw(ii, kk5, jj) = ww(kk5); gt(ii, kk5, jj) = tt(kk5)
        gq(ii, kk5, jj) = qq(kk5); gp(ii, kk5, jj) = pp(kk5)
        gpi(ii, kk5, jj) = ppi(kk5); grho(ii, kk5, jj) = rr(kk5)
        gdz(ii, kk5, jj) = dz_l(kk5); gp8w(ii, kk5, jj) = ppw(kk5)
        gthften(ii, kk5, jj) = c_thf(ic) * (1.0 - zc(kk5) / 16000.0)
        gqvften(ii, kk5, jj) = c_qvf(ic) * (1.0 - zc(kk5) / 10000.0)
        gthraten(ii, kk5, jj) = c_thrad(ic)
        if (kk5 <= c_kpbl(ic)) then
          gthblten(ii, kk5, jj) = c_thbl(ic)
          gqvblten(ii, kk5, jj) = c_qvbl(ic)
        end if
      end do
      gp8w(ii, nz + 1, jj) = ppw(nz + 1)
      gpi(ii, nz + 1, jj) = 1.0
      ght(ii, jj) = c_ht(ic); ghfx(ii, jj) = c_hfx(ic)
      gqfx(ii, jj) = c_qfx(ic); gxland(ii, jj) = c_xland(ic)
      gkpbl(ii, jj) = c_kpbl(ic)
    end do
  end subroutine fill_driver_tile

end program run_gf_stages_oracle
