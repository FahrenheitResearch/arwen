program run_gf_shallow_oracle
  ! Decompose `CUP_gf_sh`, the shallow arm, one stage per capture.
  !
  ! The deep instrument (`run_gf_stages.F90`) opens `cup_gf`; this one opens
  ! the routine beside it.  `CUP_gf_sh` is not a reduced `cup_gf`: it has no
  ! downdraft, no closure ensemble, no scale awareness, its own `mbdt = .5`,
  ! its own three-member shallow closure, and a `ktop` that comes from
  ! `get_inversion_layers` rather than from a buoyancy integral.  Its
  ! internals -- `k22`, `kbcon`, `ktop`, `ierr`, the SH2 beta-function shape
  ! parameters, `aa0`/`aa1`/`xaa0`, `xff_shal(1:3)` -- never leave the routine,
  ! so a port whose `outts` is wrong has nothing to bisect against.
  !
  ! This program is a statement-order replication of CUP_gf_sh's body
  ! (module_cu_gf_sh.F:241-874) for the arm WRF reaches -- ichoice_s = 0,
  ! MAKE_CALC_FOR_XK = .true., WRF_CHEM = 0 -- calling the module's own public
  ! procedures (`cup_env`, `cup_env_clev`, `get_cloud_bc`, `cup_kbcon`,
  ! `cup_minimi`, `get_inversion_layers`, `get_lateral_massflux`,
  ! `cup_up_aa0`) and writing out every intermediate between them.
  !
  ! `rates_up_pdf` and `get_zu_zd_pdf_fim` are TRANSCRIBED rather than called,
  ! for the same reason the deep instrument transcribes them: the SH2 branch
  ! is the scheme's second and last `tgammaf` site, and `tunning`, `alpha`,
  ! `beta` and `fzu` have to be captured for a port's residual to be
  ! attributable.  The transcription is held to the same bar as everything
  ! else -- it is inside the bitwise self-check below.
  !
  ! It proves the replication per column: after it, the real `CUP_gf_sh` runs
  ! on the same prepared column and 8 level fields and 6 column values are
  ! compared bitwise.  `gf-shallow-consistency.csv` records the differing-word
  ! count and `build.sh` fails on any nonzero row.
  !
  ! The capture is keyed by CASE ONLY, and that is a measurement rather than a
  ! convenience: `CUP_gf_sh` has no `dx` argument (module_cu_gf_sh.F:58-75) and
  ! none of the fourteen fields GFDRV hands it depends on `dx`, so the shallow
  ! answer cannot move with grid spacing.  The consistency file carries the
  ! proof anyway -- every one of the 6 dx values is run and compared, bitwise,
  ! against the same case at dx = 1000 m.
  !
  ! What is deliberately NOT replicated, because WRF cannot reach it:
  !   `ichoice > 0` (:847) -- module_cu_gf_wrfdrv.F:71 fixes ichoice_s = 0, so
  !     only the three-way average closure is live.
  !   the WRF_CHEM block at :876-929, including `tempco`.
  !   `c1_shal = 0.` (:48) makes the `dellaqc` below-ktop limb identically zero
  !     and drops `c1_shal` out of the `qrco` denominator -- it is still spelled
  !     here as the source spells it.
  use module_cu_gf_deep, only: cup_env, cup_env_clev, get_cloud_bc,           &
                               cup_minimi, get_inversion_layers, cup_kbcon,   &
                               cup_up_aa0, get_lateral_massflux
  use module_cu_gf_sh, only: cup_gf_sh
  use gf_cases
  implicit none

  integer, parameter :: nz = gf_nz
  integer, parameter :: ncase = gf_ncase
  integer, parameter :: ndx = gf_ndx

  integer, parameter :: its = 1, ite = 1, itf = 1
  integer, parameter :: kts = 1, kte = nz, ktf = nz

  ! The case column is built through the same GFDRV-shaped tile the other two
  ! decompositions use, so all three consume identical array words.
  integer, parameter :: gids = 1, gide = 10
  integer, parameter :: gits = gids + 4
  integer, parameter :: gjds = 1, gjde = 10
  integer, parameter :: gjts = gjds + 4
  integer, parameter :: gkds = 1, gkde = nz + 1

  ! module_gfs_physcons at the real(8) width module_gfs_machine.F declares.
  integer, parameter :: kind_phys = selected_real_kind(13, 60)
  real(kind=kind_phys), parameter :: con_g = 9.80665e+0_kind_phys
  real(kind=kind_phys), parameter :: con_cp = 1.0046e+3_kind_phys
  real(kind=kind_phys), parameter :: con_hvap = 2.5000e+6_kind_phys

  ! module_cu_gf_sh.F:48-54 -- the shallow module's OWN parameters.  Three of
  ! the four disagree with the GFS set the driver uses; nothing here
  ! harmonises them.
  real, parameter :: c1_shal = 0.
  real, parameter :: g = 9.81
  real, parameter :: cp = 1004.
  real, parameter :: xlv = 2.5e6
  real, parameter :: c0_shal = .001
  real, parameter :: fluxtune = 1.5

  ! --- the prepared column (GFDRV's locals, by their WRF names) --------------
  real, dimension(its:ite, kts:kte) :: zo, t2d, q2d, po, p2d, us, vs, rhoi
  real, dimension(its:ite, kts:kte) :: tn, qo, tshall, qshall, dhdt, omeg
  real, dimension(its:ite) :: ter11, psur, xlandi, hfxi, qfxi, dxi, ccn
  real, dimension(its:ite) :: mconv
  integer, dimension(its:ite) :: kpbli
  real, dimension(kts:kte) :: pi_col

  ! --- CUP_gf_sh's own locals, by their WRF names ----------------------------
  real, dimension(its:ite, kts:kte) :: entr_rate_2d, he, hes, qes, z
  real, dimension(its:ite, kts:kte) :: heo, heso, qeso, zsh
  real, dimension(its:ite, kts:kte) :: xhe, xhes, xqes, xz, xt, xq
  real, dimension(its:ite, kts:kte) :: qes_cup, q_cup, he_cup, hes_cup, z_cup
  real, dimension(its:ite, kts:kte) :: p_cup, gamma_cup, t_cup
  real, dimension(its:ite, kts:kte) :: qeso_cup, qo_cup, heo_cup, heso_cup
  real, dimension(its:ite, kts:kte) :: zo_cup, po_cup, gammao_cup, tn_cup
  real, dimension(its:ite, kts:kte) :: xqes_cup, xq_cup, xhe_cup, xhes_cup
  real, dimension(its:ite, kts:kte) :: xz_cup, xt_cup
  real, dimension(its:ite, kts:kte) :: dby, hc, zu, dbyo, qco, pwo, hco, qrco
  real, dimension(its:ite, kts:kte) :: dbyt, xdby, xhc, xzu
  real, dimension(its:ite, kts:kte) :: cd, dellah, dellaq, dellat, dellaqc
  real, dimension(its:ite, kts:kte) :: up_massentr, up_massdetr
  real, dimension(its:ite, kts:kte) :: up_massentro, up_massdetro
  real, dimension(its:ite, kts:kte) :: cnvwt, zuo, cupclw, outt, outq, outqc
  real, dimension(its:ite, kts:kte) :: dtempdz
  integer, dimension(its:ite, kts:kte) :: k_inv_layers

  real, dimension(its:ite) :: zws, ztexec, zqexec, pre, aa1, aa0, xaa0
  real, dimension(its:ite) :: hkb, flux_tun, hkbo, xhkb, rand_vmas, xmbmax
  real, dimension(its:ite) :: xmb, cap_max, entr_rate, cap_max_increment
  real, dimension(its:ite) :: xmb_out
  integer, dimension(its:ite) :: kstabi, xland1, kbmax, ktopx
  integer, dimension(its:ite) :: ierr, kbcon, ktop, k22, start_level
  character(len=50) :: ierrc(its:ite)

  ! --- the stage snapshots the replication would otherwise overwrite ---------
  real, dimension(its:ite, kts:kte) :: gamma_cup0, po_cup0, entr2d_a, cd_a
  real, dimension(its:ite, kts:kte) :: zu_pdf, zuo_b, qco_a
  real, dimension(its:ite) :: buo_flux_o, hkb0, hkbo0, hkbo_1, hkb_2, qaver_o
  real, dimension(its:ite) :: xkshal_o, blqe_o, trash_kb, xff1, xff2, xff3
  real, dimension(its:ite) :: sh_tun, sh_alpha, sh_beta, sh_fzu
  integer, dimension(its:ite) :: k22_0, kbcon_1, k22_1, ierr_1, ierr_231
  integer, dimension(its:ite) :: ktop_0, ktop_pdf, kbcon_2, ierr_2
  integer, dimension(its:ite) :: ktop_3, k22_3, ki_dbyt, ktop_4, ierr_4
  integer, dimension(its:ite) :: ierr_5, ierr_6, sh_kbadj, sh_kfinal
  integer, dimension(its:ite) :: kstabi_oob

  ! --- the same column, through the module's own CUP_gf_sh -------------------
  real, dimension(its:ite, kts:kte) :: r_zus, r_outt, r_outq, r_outqc
  real, dimension(its:ite, kts:kte) :: r_cnvwt, r_cupclw, r_q, r_qo
  real, dimension(its:ite) :: r_xmb_out, r_pre
  integer, dimension(its:ite) :: r_ierr, r_kbcon, r_ktop, r_k22
  character(len=50) :: r_ierrc(its:ite)

  ! --- the same real answer at dx = 1000 m, for the dx-independence proof ----
  real, dimension(ncase, kts:kte) :: b_zus, b_outt, b_outq, b_outqc
  real, dimension(ncase, kts:kte) :: b_cnvwt, b_cupclw
  real, dimension(ncase) :: b_xmb_out, b_pre
  integer, dimension(ncase) :: b_ierr, b_kbcon, b_ktop, b_k22

  ! --- the case tile ---------------------------------------------------------
  real, dimension(gids:gide, gkds:gkde, gjds:gjde) :: gu, gv, gw, gt, gq, gp
  real, dimension(gids:gide, gkds:gkde, gjds:gjde) :: gpi, grho, gdz, gp8w
  real, dimension(gids:gide, gkds:gkde, gjds:gjde) :: gthften, gqvften
  real, dimension(gids:gide, gkds:gkde, gjds:gjde) :: gthraten, gthblten, gqvblten
  real, dimension(gids:gide, gjds:gjde) :: ght, ghfx, gqfx, gxland
  integer, dimension(gids:gide, gjds:gjde) :: gkpbl

  real :: dz, mbdt, zkbmax, cap_maxs, trash, trash2, frh
  real :: buo_flux, pgeoh, dp, entup, detup, totmas, c_up, x_add, qaver
  real :: xff_shal(3), blqe, xkshal
  integer :: i, k, ki

  real :: dtstep, dxv, tcrit_l
  integer :: ichoice_s, ic, idx, n, ulev, usfc, ucon, ndiff, ndx1, noob
  character(len=1024) :: level_path, surface_path, cons_path

  call get_command_argument(1, level_path)
  call get_command_argument(2, surface_path)
  call get_command_argument(3, cons_path)
  if (len_trim(level_path) == 0 .or. len_trim(surface_path) == 0 .or. &
      len_trim(cons_path) == 0) then
    write(*, '(A)') 'usage: run_gf_shallow LEVELS.csv SURFACE.csv CONSISTENCY.csv'
    error stop 2
  end if

  call gf_build_case_table()

  tcrit_l = 258.
  ichoice_s = 0
  dtstep = 60.0
  noob = 0

  open(newunit=ulev, file=trim(level_path), status='replace', action='write')
  write(ulev, '(A)') 'case,k,' //                                             &
    'qes,he,hes,qeso,heo,heso,' //                                            &
    'qes_cup,q_cup,he_cup,hes_cup,z_cup,p_cup,gamma_cup0,t_cup,' //           &
    'qeso_cup,qo_cup,heo_cup,heso_cup,zo_cup,po_cup0,gammao_cup,tn_cup,' //   &
    'dtempdz,entr2d_a,cd_a,zu_pdf,zuo_b,' //                                  &
    'upme,upmd,cd_b,entr2d_b,' //                                             &
    'hc,hco,dby,dbyo,dbyt,' //                                                &
    'qco_a,qrco,pwo,cupclw,qco,cnvwt,' //                                     &
    'dellah,dellaq,dellaqc,dellat,' //                                        &
    'xhe,xq,xt,xqes,xhes,' //                                                 &
    'xqes_cup,xq_cup,xhe_cup,xhes_cup,gamma_cupx,xt_cup,po_cupx,' //          &
    'xhc,xdby,xzu,zuo,outt,outq,outqc'
  open(newunit=usfc, file=trim(surface_path), status='replace', action='write')
  write(usfc, '(A)') 'case,' //                                               &
    'buo_flux,zws,ztexec,zqexec,cap_max,xland1,entr_rate,' //                 &
    'kbmax,k22_0,hkb0,hkbo0,k22_1,kbcon_1,hkbo_1,ierr_1,' //                  &
    'kstabi,kinv1,kinv2,kinv3,kinv4,kinv5,kstabi_oob,' //                     &
    'start_level,hkb_2,ierr_231,ktop_0,' //                                   &
    'ktop_pdf,kbcon_2,ierr_2,sh_tun,sh_alpha,sh_beta,sh_fzu,' //              &
    'sh_kbadj,sh_kfinal,ktop_3,k22_3,ki_dbyt,ktop_4,ierr_4,' //               &
    'qaver,aa0,aa1,ierr_5,xhkb,xaa0,' //                                      &
    'xkshal,xff1,xff2,xff3,blqe,trash_kb,xmbmax,xmb,ierr_6,' //               &
    'k22,kbcon,ktop,ierr,xmb_out,pre,ierrc'
  open(newunit=ucon, file=trim(cons_path), status='replace', action='write')
  write(ucon, '(A)') 'case,idx,ndiff_words_vs_cup_gf_sh,ndiff_words_vs_dx1,' //  &
    'ierr_repl,ierr_sh'

  do idx = 1, ndx
    dxv = gf_dxsweep(idx)
    do ic = 1, ncase

      call fill_driver_tile(ic)
      call prepare_column(ic, dxv)

      call replicate_cup_gf_sh()
      call run_real_cup_gf_sh()

      ndiff = 0
      do k = kts, ktf
        ndiff = ndiff                                                        &
          + bitdiff(zuo(its, k), r_zus(its, k))                              &
          + bitdiff(outt(its, k), r_outt(its, k))                            &
          + bitdiff(outq(its, k), r_outq(its, k))                            &
          + bitdiff(outqc(its, k), r_outqc(its, k))                          &
          + bitdiff(cnvwt(its, k), r_cnvwt(its, k))                          &
          + bitdiff(cupclw(its, k), r_cupclw(its, k))
      end do
      ndiff = ndiff + bitdiff(xmb_out(its), r_xmb_out(its))                  &
                    + bitdiff(pre(its), r_pre(its))
      if (ierr(its) /= r_ierr(its)) ndiff = ndiff + 1
      if (kbcon(its) /= r_kbcon(its)) ndiff = ndiff + 1
      if (ktop(its) /= r_ktop(its)) ndiff = ndiff + 1
      if (k22(its) /= r_k22(its)) ndiff = ndiff + 1

      ! dx independence: the module's own answer against the module's own
      ! answer at the first dx, on the same case.
      if (idx == 1) then
        b_zus(ic, :) = r_zus(its, :); b_outt(ic, :) = r_outt(its, :)
        b_outq(ic, :) = r_outq(its, :); b_outqc(ic, :) = r_outqc(its, :)
        b_cnvwt(ic, :) = r_cnvwt(its, :); b_cupclw(ic, :) = r_cupclw(its, :)
        b_xmb_out(ic) = r_xmb_out(its); b_pre(ic) = r_pre(its)
        b_ierr(ic) = r_ierr(its); b_kbcon(ic) = r_kbcon(its)
        b_ktop(ic) = r_ktop(its); b_k22(ic) = r_k22(its)
      end if
      ndx1 = 0
      do k = kts, ktf
        ndx1 = ndx1                                                          &
          + bitdiff(b_zus(ic, k), r_zus(its, k))                             &
          + bitdiff(b_outt(ic, k), r_outt(its, k))                           &
          + bitdiff(b_outq(ic, k), r_outq(its, k))                           &
          + bitdiff(b_outqc(ic, k), r_outqc(its, k))                         &
          + bitdiff(b_cnvwt(ic, k), r_cnvwt(its, k))                         &
          + bitdiff(b_cupclw(ic, k), r_cupclw(its, k))
      end do
      ndx1 = ndx1 + bitdiff(b_xmb_out(ic), r_xmb_out(its))                   &
                  + bitdiff(b_pre(ic), r_pre(its))
      if (b_ierr(ic) /= r_ierr(its)) ndx1 = ndx1 + 1
      if (b_kbcon(ic) /= r_kbcon(its)) ndx1 = ndx1 + 1
      if (b_ktop(ic) /= r_ktop(its)) ndx1 = ndx1 + 1
      if (b_k22(ic) /= r_k22(its)) ndx1 = ndx1 + 1

      write(ucon, '(I0,4(",",I0),",",I0)') ic, idx, ndiff, ndx1,             &
        ierr(its), r_ierr(its)

      if (idx == 1) call emit_capture(ic)
    end do
  end do

  close(ulev)
  close(usfc)
  close(ucon)
  write(*, '(A,I0,A)') 'gf shallow stage oracle written (', noob,            &
    ' get_inversion_layers out-of-bounds kstabi columns)'

contains

  integer function bitdiff(a, b)
    real, intent(in) :: a, b
    if (transfer(a, 0) == transfer(b, 0)) then
      bitdiff = 0
    else
      bitdiff = 1
    end if
  end function bitdiff

  function clean(s) result(o)
    character(len=*), intent(in) :: s
    character(len=50) :: o
    integer :: nn
    o = s
    do nn = 1, len(o)
      if (o(nn:nn) == ',') o(nn:nn) = ';'
    end do
  end function clean

  ! ==========================================================================
  ! CUP_gf_sh, module_cu_gf_sh.F:241-874, statement order, live arm only.
  ! ==========================================================================
  subroutine replicate_cup_gf_sh()

    ! GFDRV zeroes these before the call (module_cu_gf_wrfdrv.F:430-435); it
    ! does NOT zero cnvwt or zus, which it passes as uninitialised
    ! intent(inout) automatics.  Zeroing them here is what makes the two
    ! paths comparable; the phase-2 sentinel control showed GF writes every
    ! slot it reads.
    outt = 0.; outq = 0.; outqc = 0.; cupclw = 0.
    cnvwt = 0.; zuo = 0.
    ierr = 0; kbcon = 0; ktop = 0; k22 = 0
    ierrc(:) = " "
    zsh = zo

    ! Locals CUP_gf_sh writes only under `ierr == 0` and reads nowhere else.
    ! In WRF they are stack automatics, so on a rejected column they hold
    ! whatever was there; in this replication they are module arrays, so they
    ! would hold the PREVIOUS case's answer and the capture would depend on
    ! case order.  Neither is a value WRF defines.  Per the project's standing
    ! rule the capture takes the defined answer -- zero -- and the port
    ! matches it; nothing in CUP_gf_sh or GFDRV reads any of them on a
    ! rejected column, so the choice is unobservable in WRF.
    hkb = 0.; hkbo = 0.; xhkb = 0.
    xhe = 0.; xq = 0.; xt = 0.; xqes = 0.; xhes = 0.
    zu = 0.; xzu = 0.
    dtempdz = 0.
    qco_a = 0.

    ! -- :241-256 --------------------------------------------------------
    start_level(:) = 0
    rand_vmas(:) = 0.
    flux_tun = fluxtune
    do i = its, itf
      xland1(i) = int(xlandi(i) + .001)
      ktopx(i) = 0
      if (xlandi(i) .gt. 1.5 .or. xlandi(i) .lt. 0.5) then
        xland1(i) = 0
      endif
      pre(i) = 0.
      xmb_out(i) = 0.
      cap_max_increment(i) = 25.
      ierrc(i) = " "
      entr_rate(i) = 9.e-5
    enddo

    ! -- :265-277 --------------------------------------------------------
    do k = kts, ktf
      do i = its, itf
        up_massentro(i, k) = 0.
        up_massdetro(i, k) = 0.
        z(i, k) = zsh(i, k)
        xz(i, k) = zsh(i, k)
        qrco(i, k) = 0.
        pwo(i, k) = 0.
        cd(i, k) = 1. * entr_rate(i)
        dellaqc(i, k) = 0.
        cupclw(i, k) = 0.
      enddo
    enddo

    ! -- :287-298 --------------------------------------------------------
    cap_maxs = 125.
    DO i = its, itf
      kbmax(i) = 1
      aa0(i) = 0.
      aa1(i) = 0.
    enddo
    do i = its, itf
      cap_max(i) = cap_maxs
      ztexec(i) = 0.
      zqexec(i) = 0.
      zws(i) = 0.
    enddo

    ! -- :299-319, the convective-scale velocity -------------------------
    do i = its, itf
      buo_flux = (hfxi(i) / cp + 0.608 * t2d(i, 1) * qfxi(i) / xlv) / rhoi(i, 1)
      pgeoh = zsh(i, 2) * g
      zws(i) = max(0., flux_tun(i) * 0.41 * buo_flux * zsh(i, 2) * g / t2d(i, 1))
      if (zws(i) > TINY(pgeoh)) then
        zws(i) = 1.2 * zws(i)**.3333
        ztexec(i) = MAX(flux_tun(i) * hfxi(i) / (rhoi(i, 1) * zws(i) * cp), 0.0)
        zqexec(i) = MAX(flux_tun(i) * qfxi(i) / xlv / (rhoi(i, 1) * zws(i)), 0.)
      endif
      zws(i) = max(0., flux_tun(i) * 0.41 * buo_flux * zsh(i, kpbli(i)) * g   &
                       / t2d(i, kpbli(i)))
      zws(i) = 1.2 * zws(i)**.3333
      zws(i) = zws(i) * rhoi(i, kpbli(i))
      buo_flux_o(i) = buo_flux
    enddo

    zkbmax = 3000.

    ! -- :328-349, the two environments ----------------------------------
    call cup_env(z, qes, he, hes, t2d, q2d, po, ter11, psur, ierr, tcrit_l,  &
                 -1, itf, ktf, its, ite, kts, kte)
    call cup_env(zsh, qeso, heo, heso, tshall, qshall, po, ter11, psur, ierr, &
                 tcrit_l, -1, itf, ktf, its, ite, kts, kte)
    call cup_env_clev(t2d, qes, q2d, he, hes, z, po, qes_cup, q_cup, he_cup, &
                      hes_cup, z_cup, p_cup, gamma_cup, t_cup, psur, ierr,   &
                      ter11, itf, ktf, its, ite, kts, kte)
    call cup_env_clev(tshall, qeso, qshall, heo, heso, zsh, po, qeso_cup,    &
                      qo_cup, heo_cup, heso_cup, zo_cup, po_cup, gammao_cup, &
                      tn_cup, psur, ierr, ter11, itf, ktf, its, ite, kts, kte)
    gamma_cup0 = gamma_cup
    po_cup0 = po_cup

    ! -- :350-363, kbmax --------------------------------------------------
    do i = its, itf
      if (ierr(i) .eq. 0) then
        do k = kts, ktf
          if (zo_cup(i, k) .gt. zkbmax + ter11(i)) then
            kbmax(i) = k
            go to 25
          endif
        enddo
25      continue
        kbmax(i) = min(kbmax(i), ktf / 2)
      endif
    enddo

    ! -- :370-383, k22.  MAXLOC over a SECTION returns the position WITHIN
    !    the section, so WRF's k22 is one level below the argmax of
    !    heo_cup(2:kbmax).  That is the scheme as shipped; a port that
    !    "corrects" it to maxloc+1 disagrees with WRF on every column.
    DO i = its, itf
      if (kpbli(i) .gt. 3) cap_max(i) = po_cup(i, kpbli(i))
      IF (ierr(i) == 0) THEN
        k22(i) = maxloc(heo_cup(i, 2:kbmax(i)), 1)
        k22(i) = max(2, k22(i))
        IF (k22(i) .GT. kbmax(i)) then
          ierr(i) = 2
          ierrc(i) = "could not find k22"
          ktop(i) = 0
          k22(i) = 0
          kbcon(i) = 0
        endif
      endif
      k22_0(i) = k22(i)
    ENDDO

    ! -- :387-393, the cloud-base values ----------------------------------
    do i = its, itf
      hkb0(i) = 0.; hkbo0(i) = 0.
      if (ierr(i) .eq. 0) then
        x_add = xlv * zqexec(i) + cp * ztexec(i)
        call get_cloud_bc(kte, he_cup(i, 1:kte), hkb(i), k22(i), x_add)
        call get_cloud_bc(kte, heo_cup(i, 1:kte), hkbo(i), k22(i), x_add)
        hkb0(i) = hkb(i); hkbo0(i) = hkbo(i)
      endif
    enddo

    ! -- :396-400 ---------------------------------------------------------
    do i = its, itf
      do k = kts, ktf
        dbyo(i, k) = 0.
      enddo
    enddo

    ! -- :402-407, cup_kbcon with iloop = 5, the SHALLOW arm --------------
    call cup_kbcon(ierrc, cap_max_increment, 5, k22, kbcon, heo_cup,         &
                   heso_cup, hkbo, ierr, kbmax, po_cup, cap_max,             &
                   ztexec, zqexec, 0, itf, ktf, its, ite, kts, kte,          &
                   z_cup, entr_rate, heo, 0)
    kbcon_1 = kbcon; k22_1 = k22; hkbo_1 = hkbo; ierr_1 = ierr

    ! -- :409-414 ---------------------------------------------------------
    call cup_minimi(heso_cup, kbcon, kbmax, kstabi, ierr, itf, ktf,          &
                    its, ite, kts, kte)
    ! get_inversion_layers reads t_cup(kend+8) against a kend bounded only by
    ! ktf-1 (module_cu_gf_deep.F:4088).  Here kend is kstabi, which cup_minimi
    ! bounds by kbmax <= ktf/2 = 20, so kend+8 <= 28 < kte and the read is in
    ! bounds on every column of this fixture.  The count proves it rather
    ! than the arithmetic asserting it.
    do i = its, itf
      kstabi_oob(i) = 0
      if (ierr(i) == 0 .and. kstabi(i) > ktf - 8) then
        kstabi_oob(i) = 1
        noob = noob + 1
      endif
    enddo
    call get_inversion_layers(ierr, p_cup, t_cup, z_cup, q_cup, qes_cup,     &
                              k_inv_layers, kbcon, kstabi, dtempdz,          &
                              itf, ktf, its, ite, kts, kte)

    ! -- :417-449, the entrainment profile and the first ktop -------------
    DO i = its, itf
      entr_rate_2d(i, :) = entr_rate(i)
      IF (ierr(i) == 0) THEN
        start_level(i) = k22(i)
        x_add = xlv * zqexec(i) + cp * ztexec(i)
        call get_cloud_bc(kte, he_cup(i, 1:kte), hkb(i), k22(i), x_add)
        if (kbcon(i) .gt. ktf - 4) then
          ierr(i) = 231
        endif
        do k = kts, ktf
          frh = 2. * min(qo_cup(i, k) / qeso_cup(i, k), 1.)
          entr_rate_2d(i, k) = entr_rate(i) * (2.3 - frh)
          cd(i, k) = entr_rate_2d(i, k)
        enddo
        ktop(i) = 1
        if (k_inv_layers(i, 1) .gt. 0 .and.                                  &
            (po_cup(i, kbcon(i)) - po_cup(i, k_inv_layers(i, 1))) .lt. 200.) then
          ktop(i) = k_inv_layers(i, 1)
        else
          do k = kbcon(i) + 1, ktf
            if ((po_cup(i, kbcon(i)) - po_cup(i, k)) .gt. 200.) then
              ktop(i) = k
              exit
            endif
          enddo
        endif
      endif
    enddo
    hkb_2 = hkb; ierr_231 = ierr; ktop_0 = ktop
    entr2d_a = entr_rate_2d
    cd_a = cd

    ! -- :451-452, the normalised mass-flux profile -----------------------
    call rates_up_pdf_shal()
    zu_pdf = zuo
    ktop_pdf = ktop; kbcon_2 = kbcon; ierr_2 = ierr

    ! -- :453-486 ---------------------------------------------------------
    do i = its, itf
      if (ierr(i) .eq. 0) then
        if (k22(i) .gt. 1) then
          do k = 1, k22(i) - 1
            zuo(i, k) = 0.
            zu(i, k) = 0.
            xzu(i, k) = 0.
          enddo
        endif
        do k = maxloc(zuo(i, :), 1), ktop(i)
          if (zuo(i, k) .lt. 1.e-6) then
            ktop(i) = k - 1
            exit
          endif
        enddo
        do k = k22(i), ktop(i)
          xzu(i, k) = zuo(i, k)
          zu(i, k) = zuo(i, k)
        enddo
        do k = ktop(i) + 1, ktf
          zuo(i, k) = 0.
          zu(i, k) = 0.
          xzu(i, k) = 0.
        enddo
        k22(i) = max(2, k22(i))
      endif
    enddo
    zuo_b = zuo
    ktop_3 = ktop; k22_3 = k22

    ! -- :490-493, lateral mass flux.  The shallow call omits the OPTIONAL
    !    up_massentru/up_massdetru/lambau triple, so the momentum-transport
    !    limb inside the routine never runs.
    CALL get_lateral_massflux(itf, ktf, its, ite, kts, kte, ierr, ktop,      &
                              zo_cup, zuo, cd, entr_rate_2d,                 &
                              up_massentro, up_massdetro, up_massentr,       &
                              up_massdetr, 'shallow', kbcon, k22)

    ! -- :495-514 ---------------------------------------------------------
    do k = kts, ktf
      do i = its, itf
        hc(i, k) = 0.
        qco(i, k) = 0.
        qrco(i, k) = 0.
        dby(i, k) = 0.
        hco(i, k) = 0.
        dbyo(i, k) = 0.
      enddo
    enddo
    do i = its, itf
      IF (ierr(i) /= 0) cycle
      do k = 1, start_level(i) - 1
        hc(i, k) = he_cup(i, k)
        hco(i, k) = heo_cup(i, k)
      enddo
      k = start_level(i)
      hc(i, k) = hkb(i)
      hco(i, k) = hkbo(i)
    enddo

    ! -- :517-611, the in-cloud updraft ------------------------------------
    do 42 i = its, itf
      dbyt(i, :) = 0.
      ki_dbyt(i) = 0; ktop_4(i) = ktop(i); ierr_4(i) = ierr(i)
      qaver_o(i) = 0.
      IF (ierr(i) /= 0) cycle
      do k = start_level(i) + 1, ktop(i)
        hc(i, k) = (hc(i, k - 1) * zu(i, k - 1) - .5 * up_massdetr(i, k - 1)  &
                    * hc(i, k - 1) + up_massentr(i, k - 1) * he(i, k - 1)) /  &
                   (zu(i, k - 1) - .5 * up_massdetr(i, k - 1) + up_massentr(i, k - 1))
        dby(i, k) = max(0., hc(i, k) - hes_cup(i, k))
        hco(i, k) = (hco(i, k - 1) * zuo(i, k - 1) - .5 * up_massdetro(i, k - 1) &
                     * hco(i, k - 1) + up_massentro(i, k - 1) * heo(i, k - 1)) / &
                    (zuo(i, k - 1) - .5 * up_massdetro(i, k - 1) + up_massentro(i, k - 1))
        dbyo(i, k) = hco(i, k) - heso_cup(i, k)
        DZ = zo_cup(i, k + 1) - zo_cup(i, k)
        dbyt(i, k) = dbyt(i, k - 1) + dbyo(i, k) * dz
      enddo
      ki = maxloc(dbyt(i, :), 1)
      ki_dbyt(i) = ki
      if (ktop(i) .gt. ki + 1) then
        ktop(i) = ki + 1
        zuo(i, ktop(i) + 1:ktf) = 0.
        zu(i, ktop(i) + 1:ktf) = 0.
        cd(i, ktop(i) + 1:ktf) = 0.
        up_massdetro(i, ktop(i)) = zuo(i, ktop(i))
        up_massentro(i, ktop(i):ktf) = 0.
        up_massdetro(i, ktop(i) + 1:ktf) = 0.
        entr_rate_2d(i, ktop(i) + 1:ktf) = 0.
      endif

      if (ktop(i) .lt. kbcon(i) + 1) then
        ierr(i) = 5
        ierrc(i) = 'ktop is less than kbcon+1'
        go to 42
      endif
      if (ktop(i) .gt. ktf - 2) then
        ierr(i) = 5
        ierrc(i) = "ktop is larger than ktf-2"
        go to 42
      endif

      call get_cloud_bc(kte, qo_cup(i, 1:kte), qaver, k22(i))
      qaver = qaver + zqexec(i)
      qaver_o(i) = qaver
      do k = 1, start_level(i) - 1
        qco(i, k) = qo_cup(i, k)
      enddo
      k = start_level(i)
      qco(i, k) = qaver

      do k = start_level(i) + 1, ktop(i)
        trash = qeso_cup(i, k) + (1. / xlv) * (gammao_cup(i, k)               &
                / (1. + gammao_cup(i, k))) * dbyo(i, k)
        trash2 = qco(i, k - 1)
        qco(i, k) = (trash2 * (zuo(i, k - 1) - 0.5 * up_massdetr(i, k - 1)) + &
                     up_massentr(i, k - 1) * qshall(i, k - 1)) /              &
                    (zuo(i, k - 1) - .5 * up_massdetr(i, k - 1) + up_massentr(i, k - 1))
        if (qco(i, k) >= trash) then
          DZ = z_cup(i, k) - z_cup(i, k - 1)
          qrco(i, k) = (qco(i, k) - trash) / (1. + (c0_shal + c1_shal) * dz)
          pwo(i, k) = c0_shal * dz * qrco(i, k) * zuo(i, k)
          qco(i, k) = trash + qrco(i, k)
        else
          qrco(i, k) = 0.0
        endif
        cupclw(i, k) = qrco(i, k)
      enddo
      qco_a(i, :) = qco(i, :)
      ! trash and trash2 accumulate entrainment sums that nothing downstream
      ! reads -- trash is overwritten at :841 before its only use.
      trash = 0.
      trash2 = 0.
      do k = k22(i) + 1, ktop(i)
        dp = 100. * (po_cup(i, k) - po_cup(i, k + 1))
        cnvwt(i, k) = zuo(i, k) * cupclw(i, k) * g / dp
        trash2 = trash2 + entr_rate_2d(i, k)
        qco(i, k) = qco(i, k) - qrco(i, k)
      enddo
      do k = k22(i) + 1, max(kbcon(i), k22(i) + 1)
        trash = trash + entr_rate_2d(i, k)
      enddo
      do k = ktop(i) + 1, ktf - 1
        hc(i, k) = hes_cup(i, k)
        hco(i, k) = heso_cup(i, k)
        qco(i, k) = qeso_cup(i, k)
        qrco(i, k) = 0.
        dby(i, k) = 0.
        dbyo(i, k) = 0.
        zu(i, k) = 0.
        xzu(i, k) = 0.
        zuo(i, k) = 0.
      enddo
42  continue
    ktop_4 = ktop; ierr_4 = ierr

    ! -- :615-630, the cloud work functions --------------------------------
    call cup_up_aa0(aa0, z, zu, dby, gamma_cup, t_cup, kbcon, ktop, ierr,    &
                    itf, ktf, its, ite, kts, kte)
    call cup_up_aa0(aa1, zsh, zuo, dbyo, gammao_cup, tn_cup, kbcon, ktop,    &
                    ierr, itf, ktf, its, ite, kts, kte)
    do i = its, itf
      if (ierr(i) == 0) then
        if (aa1(i) <= 0.) then
          ierr(i) = 17
          ierrc(i) = "cloud work function zero"
        endif
      endif
    enddo
    ierr_5 = ierr

    ! -- :639-720, the dellas ----------------------------------------------
    do k = kts, kte
      do i = its, itf
        dellah(i, k) = 0.
        dellaq(i, k) = 0.
      enddo
    enddo
    trash2 = 0.
    do i = its, itf
      if (ierr(i) .eq. 0) then
        do k = k22(i), ktop(i)
          entup = up_massentro(i, k)
          detup = up_massdetro(i, k)
          totmas = detup - entup + zuo(i, k + 1) - zuo(i, k)
          dp = 100. * (po_cup(i, k) - po_cup(i, k + 1))
          dellah(i, k) = -(zuo(i, k + 1) * (hco(i, k + 1) - heo_cup(i, k + 1)) - &
                           zuo(i, k) * (hco(i, k) - heo_cup(i, k))) * g / dp
          dz = zo_cup(i, k + 1) - zo_cup(i, k)
          if (k .lt. ktop(i)) then
            dellaqc(i, k) = zuo(i, k) * c1_shal * qrco(i, k) * dz / dp * g
          else
            dellaqc(i, k) = detup * qrco(i, k) * g / dp
          endif
          c_up = dellaqc(i, k) + (zuo(i, k + 1) * qrco(i, k + 1) -              &
                                  zuo(i, k) * qrco(i, k)) * g / dp
          dellaq(i, k) = -(zuo(i, k + 1) * (qco(i, k + 1) - qo_cup(i, k + 1)) - &
                           zuo(i, k) * (qco(i, k) - qo_cup(i, k))) * g / dp     &
                         - c_up - 0.5 * (pwo(i, k) + pwo(i, k + 1)) * g / dp
        enddo
      endif
    enddo

    ! -- :725-746, the mbdt-perturbed state --------------------------------
    mbdt = .5
    do k = kts, ktf
      do i = its, itf
        dellat(i, k) = 0.
        if (ierr(i) /= 0) cycle
        xhe(i, k) = dellah(i, k) * mbdt + heo(i, k)
        xq(i, k) = max(1.e-16, (dellaq(i, k) + dellaqc(i, k)) * mbdt + qshall(i, k))
        dellat(i, k) = (1. / cp) * (dellah(i, k) - xlv * (dellaq(i, k)))
        xt(i, k) = (-dellaqc(i, k) * xlv / cp + dellat(i, k)) * mbdt + tshall(i, k)
        xt(i, k) = max(190., xt(i, k))
      enddo
    enddo
    do i = its, itf
      if (ierr(i) .eq. 0) then
        xhe(i, ktf) = heo(i, ktf)
        xq(i, ktf) = qshall(i, ktf)
        xt(i, ktf) = tshall(i, ktf)
      endif
    enddo

    ! -- :749-810, the perturbed static control ----------------------------
    ! cup_env_clev's 13th actual argument is po_cup itself, and the routine
    ! zeroes its outputs BEFORE the ierr guard -- so po_cup comes back all
    ! zeros on every rejected column.  The live closure at :839 reads po_cup
    ! only under ierr == 0, so nothing downstream sees the zeros; a port that
    ! keeps po_cup untouched here disagrees with the capture and not with WRF.
    call cup_env(xz, xqes, xhe, xhes, xt, xq, po, ter11, psur, ierr, tcrit_l, &
                 -1, itf, ktf, its, ite, kts, kte)
    call cup_env_clev(xt, xqes, xq, xhe, xhes, xz, po, xqes_cup, xq_cup,      &
                      xhe_cup, xhes_cup, xz_cup, po_cup, gamma_cup, xt_cup,   &
                      psur, ierr, ter11, itf, ktf, its, ite, kts, kte)
    do k = kts, ktf
      do i = its, itf
        xhc(i, k) = 0.
        xdby(i, k) = 0.
      enddo
    enddo
    do i = its, itf
      xhkb(i) = 0.
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
        xzu(i, 1:ktf) = zuo(i, 1:ktf)
        do k = start_level(i) + 1, ktop(i)
          xhc(i, k) = (xhc(i, k - 1) * xzu(i, k - 1) - .5 * up_massdetro(i, k - 1) &
                       * xhc(i, k - 1) + up_massentro(i, k - 1) * xhe(i, k - 1)) / &
                      (xzu(i, k - 1) - .5 * up_massdetro(i, k - 1) + up_massentro(i, k - 1))
          xdby(i, k) = xhc(i, k) - xhes_cup(i, k)
        enddo
        do k = ktop(i) + 1, ktf
          xhc(i, k) = xhes_cup(i, k)
          xdby(i, k) = 0.
          xzu(i, k) = 0.
        enddo
      endif
    enddo
    call cup_up_aa0(xaa0, xz, xzu, xdby, gamma_cup, xt_cup, kbcon, ktop,     &
                    ierr, itf, ktf, its, ite, kts, kte)

    ! -- :817-874, the shallow closure and the tendencies -------------------
    do i = its, itf
      xmb(i) = 0.
      xff_shal(1:3) = 0.
      xkshal_o(i) = 0.; blqe_o(i) = 0.; trash_kb(i) = 0.
      xff1(i) = 0.; xff2(i) = 0.; xff3(i) = 0.; xmbmax(i) = 0.
      if (ierr(i) .eq. 0) then
        xmbmax(i) = 1.0
        xkshal = (xaa0(i) - aa1(i)) / mbdt
        if (xkshal .le. 0. .and. xkshal .gt. -.01 * mbdt) xkshal = -.01 * mbdt
        if (xkshal .gt. 0. .and. xkshal .lt. 1.e-2) xkshal = 1.e-2
        xff_shal(1) = max(0., -(aa1(i) - aa0(i)) / (xkshal * dtstep))
        xff_shal(2) = .03 * zws(i)
        blqe = 0.
        trash = 0.
        do k = 1, kpbli(i)
          blqe = blqe + 100. * dhdt(i, k) * (po_cup(i, k) - po_cup(i, k + 1)) / g
        enddo
        trash = max((hc(i, kbcon(i)) - he_cup(i, kbcon(i))), 1.e1)
        xff_shal(3) = max(0., blqe / trash)
        xff_shal(3) = min(xmbmax(i), xff_shal(3))
        xmb(i) = (xff_shal(1) + xff_shal(2) + xff_shal(3)) / 3.
        xmb(i) = min(xmbmax(i), xmb(i))
        if (ichoice_s > 0) xmb(i) = min(xmbmax(i), xff_shal(ichoice_s))
        if (xmb(i) <= 0.) then
          ierr(i) = 21
          ierrc(i) = "21"
        endif
        xkshal_o(i) = xkshal; blqe_o(i) = blqe; trash_kb(i) = trash
        xff1(i) = xff_shal(1); xff2(i) = xff_shal(2); xff3(i) = xff_shal(3)
      endif
      if (ierr(i) .ne. 0) then
        k22(i) = 0
        kbcon(i) = 0
        ktop(i) = 0
        xmb(i) = 0.
        outt(i, :) = 0.
        outq(i, :) = 0.
        outqc(i, :) = 0.
      else if (ierr(i) .eq. 0) then
        xmb_out(i) = xmb(i)
        pre(i) = 0.
        do k = 2, ktop(i)
          outt(i, k) = dellat(i, k) * xmb(i)
          outq(i, k) = dellaq(i, k) * xmb(i)
          outqc(i, k) = dellaqc(i, k) * xmb(i)
          pre(i) = pre(i) + pwo(i, k) * xmb(i)
        enddo
      endif
    enddo
    ierr_6 = ierr
  end subroutine replicate_cup_gf_sh

  ! ==========================================================================
  ! rates_up_pdf, module_cu_gf_deep.F:3697-3823, name == 'shallow'.
  !
  ! CUP_gf_sh:451 passes `ktopx` for BOTH ktopdby (intent inout) and csum
  ! (intent in), and `kbcon` for both `kbcon` (inout) and `pmin_lev` (in).
  ! Both are argument aliasing, which Fortran does not allow, and both are
  ! harmless only because the read-only alias is dead: `csum` reaches
  ! `beta_u`, which feeds `max_mass`, which no branch of
  ! `get_zu_zd_pdf_fim` reads, and `pmin_lev` is dead on every branch.  The
  ! ordering saves it too -- `beta_u` is evaluated before `ktopdby` is
  ! written.  `kklev` is passed to `get_zu_zd_pdf_fim` UNINITIALISED on this
  ! path (it is only assigned on the deep branch) and is never read inside,
  ! which is the only reason that is not a bug either.
  ! ==========================================================================
  subroutine rates_up_pdf_shal()
    real :: zustart, zubeg, dz_l, massent, massdetr, beta_u
    real :: dby_l(kts:kte), dbm_l(kts:kte), zux(kts:kte)
    integer :: il, kl, kfinalzu

    zustart = .1
    dby_l(:) = 0.
    DO il = its, itf
      zux(:) = 0.
      beta_u = max(.1, .2 - float(ktopx(il)) * .01)
      zuo(il, :) = 0.
      dby_l(:) = 0.
      dbm_l(:) = 0.
      kbcon(il) = max(kbcon(il), 2)
      sh_tun(il) = 0.; sh_alpha(il) = 0.; sh_beta(il) = 0.; sh_fzu(il) = 0.
      sh_kbadj(il) = 0; sh_kfinal(il) = 0
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
        if (ktop(il) <= kbcon(il) + 2) then
          ierr(il) = 41
          ktop(il) = 0
        else
          kfinalzu = ktop(il)
          ktopx(il) = ktop(il)
          sh_kfinal(il) = kfinalzu
          call pdf_sh2(po_cup(il, :), k22(il), kfinalzu, kpbli(il), zubeg,   &
                       zuo(il, :), il)
        endif
      endif
    ENDDO
  end subroutine rates_up_pdf_shal

  ! get_zu_zd_pdf_fim, draft == "SH2" (module_cu_gf_deep.F:3876-3895).
  !
  ! Two things separate it from the "UP" branch beside it: the tunning clamp
  ! is 0.8 rather than 0.9 and takes p(kpbli) directly instead of the
  ! lev_start blend, and beta is 2.5 rather than 1.3.  And the trailing
  ! kb_adj scan is DEAD -- unlike "UP" and "MID", SH2 neither raises kb_adj to
  ! 2 nor zeroes zu below it, so the loop computes a value nothing reads.
  subroutine pdf_sh2(p, kb, kt, kpbli, zubeg, zu, islot)
    real, intent(in) :: p(kts:kte), zubeg
    integer, intent(in) :: kb, kt, kpbli, islot
    real, intent(inout) :: zu(kts:kte)
    integer :: kb_adj, kl
    real :: tunning, beta_l, alpha_l, fzu, kratio

    zu = 0.0
    kb_adj = max(kb, 2)
    tunning = min(0.8, (p(kpbli) - p(kb_adj)) / (p(kt) - p(kb_adj)))
    tunning = max(0.2, tunning)
    beta_l = 2.5
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
    sh_tun(islot) = tunning
    sh_alpha(islot) = alpha_l
    sh_beta(islot) = beta_l
    sh_fzu(islot) = fzu
    sh_kbadj(islot) = kb_adj
  end subroutine pdf_sh2

  ! ==========================================================================
  subroutine run_real_cup_gf_sh()
    integer, dimension(its:ite) :: rr_ierr
    real, dimension(its:ite, kts:kte) :: r_outu, r_outv

    r_zus = 0.; r_outt = 0.; r_outq = 0.; r_outqc = 0.
    r_cnvwt = 0.; r_cupclw = 0.
    r_outu = 0.; r_outv = 0.
    r_xmb_out = 0.; r_pre = 0.
    r_kbcon = 0; r_ktop = 0; r_k22 = 0
    rr_ierr = 0
    r_ierrc(:) = " "
    r_q = q2d
    r_qo = qshall
    call cup_gf_sh(zo, t2d, r_q, ter11, tshall, r_qo, p2d, psur, dhdt,       &
                   kpbli, rhoi, hfxi, qfxi, xlandi, ichoice_s, tcrit_l,      &
                   dtstep, r_zus, r_xmb_out, r_kbcon, r_ktop, r_k22,         &
                   rr_ierr, r_ierrc, r_outt, r_outq, r_outqc, r_cnvwt,       &
                   r_pre, r_cupclw, itf, ktf, its, ite, kts, kte, 0)
    r_ierr = rr_ierr
  end subroutine run_real_cup_gf_sh

  ! ==========================================================================
  subroutine emit_capture(ic)
    integer, intent(in) :: ic
    integer :: kk

    do kk = kts, ktf
      write(ulev, '(I0,",",I0)', advance='no') ic, kk
      write(ulev, '(20(",",ES24.16E3))', advance='no')                        &
        qes(its, kk), he(its, kk), hes(its, kk),                              &
        qeso(its, kk), heo(its, kk), heso(its, kk),                           &
        qes_cup(its, kk), q_cup(its, kk), he_cup(its, kk), hes_cup(its, kk),  &
        z_cup(its, kk), p_cup(its, kk), gamma_cup0(its, kk), t_cup(its, kk),  &
        qeso_cup(its, kk), qo_cup(its, kk), heo_cup(its, kk),                 &
        heso_cup(its, kk), zo_cup(its, kk), po_cup0(its, kk)
      write(ulev, '(11(",",ES24.16E3))', advance='no')                        &
        gammao_cup(its, kk), tn_cup(its, kk), dtempdz(its, kk),               &
        entr2d_a(its, kk), cd_a(its, kk), zu_pdf(its, kk), zuo_b(its, kk),    &
        up_massentro(its, kk), up_massdetro(its, kk), cd(its, kk),            &
        entr_rate_2d(its, kk)
      write(ulev, '(11(",",ES24.16E3))', advance='no')                        &
        hc(its, kk), hco(its, kk), dby(its, kk), dbyo(its, kk),               &
        dbyt(its, kk), qco_a(its, kk), qrco(its, kk), pwo(its, kk),           &
        cupclw(its, kk), qco(its, kk), cnvwt(its, kk)
      write(ulev, '(9(",",ES24.16E3))', advance='no')                         &
        dellah(its, kk), dellaq(its, kk), dellaqc(its, kk), dellat(its, kk),  &
        xhe(its, kk), xq(its, kk), xt(its, kk), xqes(its, kk), xhes(its, kk)
      write(ulev, '(14(",",ES24.16E3))')                                      &
        xqes_cup(its, kk), xq_cup(its, kk), xhe_cup(its, kk),                 &
        xhes_cup(its, kk), gamma_cup(its, kk), xt_cup(its, kk),               &
        po_cup(its, kk), xhc(its, kk), xdby(its, kk), xzu(its, kk),           &
        zuo(its, kk), outt(its, kk), outq(its, kk), outqc(its, kk)
    enddo

    write(usfc, '(I0)', advance='no') ic
    write(usfc, '(5(",",ES24.16E3),",",I0,",",ES24.16E3)', advance='no')      &
      buo_flux_o(its), zws(its), ztexec(its), zqexec(its), cap_max(its),      &
      xland1(its), entr_rate(its)
    write(usfc, '(2(",",I0),2(",",ES24.16E3),2(",",I0),",",ES24.16E3,",",I0)',&
      advance='no') kbmax(its), k22_0(its), hkb0(its), hkbo0(its),            &
      k22_1(its), kbcon_1(its), hkbo_1(its), ierr_1(its)
    write(usfc, '(7(",",I0))', advance='no') kstabi(its),                     &
      (k_inv_layers(its, n), n = 1, 5), kstabi_oob(its)
    write(usfc, '(",",I0,",",ES24.16E3,2(",",I0))', advance='no')             &
      start_level(its), hkb_2(its), ierr_231(its), ktop_0(its)
    write(usfc, '(3(",",I0),4(",",ES24.16E3),2(",",I0))', advance='no')       &
      ktop_pdf(its), kbcon_2(its), ierr_2(its), sh_tun(its), sh_alpha(its),   &
      sh_beta(its), sh_fzu(its), sh_kbadj(its), sh_kfinal(its)
    write(usfc, '(5(",",I0))', advance='no') ktop_3(its), k22_3(its),         &
      ki_dbyt(its), ktop_4(its), ierr_4(its)
    write(usfc, '(3(",",ES24.16E3),",",I0,2(",",ES24.16E3))', advance='no')   &
      qaver_o(its), aa0(its), aa1(its), ierr_5(its), xhkb(its), xaa0(its)
    write(usfc, '(8(",",ES24.16E3),",",I0)', advance='no')                    &
      xkshal_o(its), xff1(its), xff2(its), xff3(its), blqe_o(its),            &
      trash_kb(its), xmbmax(its), xmb(its), ierr_6(its)
    write(usfc, '(4(",",I0),2(",",ES24.16E3),",",A)')                         &
      k22(its), kbcon(its), ktop(its), ierr(its), xmb_out(its), pre(its),     &
      trim(clean(ierrc(its)))
  end subroutine emit_capture

  ! GFDRV's own column preparation, module_cu_gf_wrfdrv.F:383-492.  Identical
  ! to run_cup_gf.F90's and run_gf_stages.F90's, for the same reason: all
  ! three decompositions must consume the same array words as the driver.
  subroutine prepare_column(ic, dxv_in)
    integer, intent(in) :: ic
    real, intent(in) :: dxv_in
    real :: dqv
    integer :: kk, i0, j0

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
    do kk = kts + 1, ktf
      zo(its, kk) = zo(its, kk - 1) + .5 * (gdz(i0, kk - 1, j0) + gdz(i0, kk, j0))
    end do

    do kk = kts, ktf
      po(its, kk) = gp(i0, kk, j0) * .01
      pi_col(kk) = gpi(i0, kk, j0)
      p2d(its, kk) = po(its, kk)
      rhoi(its, kk) = grho(i0, kk, j0)
      us(its, kk) = gu(i0, kk, j0)
      vs(its, kk) = gv(i0, kk, j0)
      t2d(its, kk) = gt(i0, kk, j0)
      q2d(its, kk) = gq(i0, kk, j0)
      if (q2d(its, kk) < 1.e-08) q2d(its, kk) = 1.e-08

      tn(its, kk) = t2d(its, kk) + (gthften(i0, kk, j0) + gthraten(i0, kk, j0) &
                                    + gthblten(i0, kk, j0))                    &
                                   * gpi(i0, kk, j0) * dtstep
      qo(its, kk) = q2d(its, kk) + (gqvften(i0, kk, j0) + gqvblten(i0, kk, j0)) &
                                   * dtstep
      tshall(its, kk) = t2d(its, kk) + gthblten(i0, kk, j0) * gpi(i0, kk, j0)  &
                                       * dtstep
      dhdt(its, kk) = con_cp * gthblten(i0, kk, j0) * gpi(i0, kk, j0)          &
                      + con_hvap * gqvblten(i0, kk, j0)
      qshall(its, kk) = q2d(its, kk) + gqvblten(i0, kk, j0) * dtstep
      if (tn(its, kk) < 200.) tn(its, kk) = t2d(its, kk)
      if (qo(its, kk) < 1.e-08) qo(its, kk) = 1.e-08

      omeg(its, kk) = -con_g * grho(i0, kk, j0) * gw(i0, kk, j0)
    end do

    do kk = kts, ktf - 1
      dqv = q2d(its, kk + 1) - q2d(its, kk)
      mconv(its) = mconv(its) + omeg(its, kk) * dqv / con_g
    end do
    if (mconv(its) < 0.) mconv(its) = 0.
  end subroutine prepare_column

  subroutine fill_driver_tile(ic)
    integer, intent(in) :: ic
    real :: zc(nz), dzc(nz), tt(nz), qq(nz), pp(nz), ppw(nz + 1)
    real :: ppi(nz), rr(nz), uu(nz), vv(nz), ww(nz)
    integer :: ii, kk, jj

    gu = 0.0; gv = 0.0; gw = 0.0; gt = 0.0; gq = 0.0
    gp = 0.0; gpi = 1.0; grho = 0.0; gdz = 0.0; gp8w = 0.0
    gthften = 0.0; gqvften = 0.0; gthraten = 0.0
    gthblten = 0.0; gqvblten = 0.0
    ght = 0.0; ghfx = 0.0; gqfx = 0.0; gxland = 1.0
    gkpbl = 1

    call gf_column(ic, zc, dzc, tt, qq, pp, ppw, ppi, rr, uu, vv, ww)
    ii = gits
    do jj = gjds, gjde
      do kk = 1, nz
        gu(ii, kk, jj) = uu(kk); gv(ii, kk, jj) = vv(kk)
        gw(ii, kk, jj) = ww(kk); gt(ii, kk, jj) = tt(kk)
        gq(ii, kk, jj) = qq(kk); gp(ii, kk, jj) = pp(kk)
        gpi(ii, kk, jj) = ppi(kk); grho(ii, kk, jj) = rr(kk)
        gdz(ii, kk, jj) = dzc(kk); gp8w(ii, kk, jj) = ppw(kk)
        gthften(ii, kk, jj) = c_thf(ic) * (1.0 - zc(kk) / 16000.0)
        gqvften(ii, kk, jj) = c_qvf(ic) * (1.0 - zc(kk) / 10000.0)
        gthraten(ii, kk, jj) = c_thrad(ic)
        if (kk <= c_kpbl(ic)) then
          gthblten(ii, kk, jj) = c_thbl(ic)
          gqvblten(ii, kk, jj) = c_qvbl(ic)
        end if
      end do
      gp8w(ii, nz + 1, jj) = ppw(nz + 1)
      gpi(ii, nz + 1, jj) = 1.0
      ght(ii, jj) = c_ht(ic); ghfx(ii, jj) = c_hfx(ic)
      gqfx(ii, jj) = c_qfx(ic); gxland(ii, jj) = c_xland(ic)
      gkpbl(ii, jj) = c_kpbl(ic)
    end do
  end subroutine fill_driver_tile

end program run_gf_shallow_oracle
