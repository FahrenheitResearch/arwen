! Process sweeps and refusal cases for the fast SBM oracle (links the PROBE
! build: the access transform plus inert probe calls, so COAL_BOTT_NEW's
! breakup path can be read back through fsbm_probe_state).
!
! Usage (run where ./SBM_input_33 exists):
!   run_sweeps sweep <family> <outdir>    one family per process, one stream
!                                         <outdir>/sweep_<family>
!   run_sweeps refusal <n> <outdir>       refusal case n (1..6): writes its
!                                         input record to <outdir>/refusal_<n>,
!                                         closes it, then makes the call that
!                                         must not return
!   run_sweeps refusal_cold <n> <outdir>  cold refusal case n (1..13), the same
!                                         way, to <outdir>/refusal_cold_<n>
!
! Families (design 5.3 minimum coverage): polysvp, jerrate, jersupsat,
! jerdfun, jernewf, onecond1, nucleation, coll_xxx, coal_bott, breakup, spont,
! falflux. Cold families (milestone 2): freez, melt, ice_nucl,
! jernucl01_ice, jerrate_ice, jertimesc_ice, jerdfun_ice, onecond2,
! onecond3, coll_cold, modkrn, coal_cold (the last also in the graupel
! build). Every record carries its inputs exactly as passed (in the kinds
! WRF declares) and the outputs the routine returns. Where a record also
! carries a branch label, the label is the harness's classification of the
! inputs with the routine's own predicates evaluated in the same kinds (the
! branches of JERSUPSAT_KS are not observable without editing it); the
! values themselves are the routine's.
!
! Spectra are built from lognormal cloud modes and Marshall-Palmer rain on the
! drop mass grid XL, in the scheme's f-form (number per gram per cm3 per
! ln-mass unit; N_k = 3 COL XL_k f_k).
program run_sweeps
  use module_mp_fast_sbm, only: fast_hucminit, onecond1, coal_bott_new, &
       falfluxhucm_z, kernals_ks, xl, xi, vr1, rlec, ro1bl, cwll, chucm, ima, &
       pkij, qkj, prob, gain_var_new, nnd, ikr_spon_break, rccn, dropradii, &
       fccnr_con, fccnr_mar, col, c1_mey, c2_mey, ro_solute, ions, mwaero, &
       jernucl01_ks, jerrate_ks, jersupsat_ks, jerdfun_ks, polysvp, &
       spont_rain_breakup
  use module_mp_fast_sbm, only: freez, j_w_melt, onecond2, onecond3, jertimesc_ks, &
       xs, xg, xh, vr2, vr3, vr4, vr5, riec, rsec, rgec, rhec, ro2bl, ro3bl, ro4bl, &
       ro5bl, cwls, cwsl, cwlh, cwhl, cwss, cwlg, cwgl, hail_opt, icemax
  use module_mp_SBM_Collision, only: coll_xxx_lwf, coll_breakup_ks
  use module_mp_SBM_Collision, only: coll_xyx_lwf, coll_xyy_lwf, coll_xyz_lwf, &
       modkrn_ks, courant_bott_ks
  use fsbm_io
  use fsbm_probe_state, only: pr_capture, cap_icol_drop, cap_icol_brk, cap_ndiv, &
       cap_path, cap_g1_xxx_in, cap_g1_xxx_out, cap_g1_at11, cap_break_per
  use fsbm_probe_state, only: capc_icenucl_calls, capc_icenucl_band, capc_icenucl_added, &
       capc_icenucl_psi2_in, capc_icenucl_psi2_out, capc_icenucl_scal_in, &
       capc_icenucl_scal_out, capc_homfrz_calls, capc_homfrz, capc_supsat_n, &
       capc_supsat_site, capc_supsat_kcond, capc_supsat_branch, capc_supsat_clamp, &
       capc_supsat_rates, capc_cc_mask, capc_cold_block, capc_hm_fired, capc_cold_icol, &
       capc_hm, capc_rime, capc_cold_g_in, capc_cold_g_out, capc_cold_rf3_out, &
       capc_cold_scal
  implicit none

  integer, parameter :: nb = 33, na = 43
  real(8), parameter :: pi_d = 3.141592653589793d0
  real :: AA1_MY, BB1_MY, AA2_MY, BB2_MY
  DATA AA1_MY, BB1_MY, AA2_MY, BB2_MY /2.53E12, 5.42E3, 3.41E13, 6.13E3/

  character(len=64) :: mode, what
  character(len=512) :: outdir
  type(fsbm_stream) :: s
  real :: dt_init
  real :: fr_lim_z(nb), frh_lim_z(nb)
  integer(8) :: pm_state = 1_8

  call get_command_argument(1, mode)
  call get_command_argument(2, what)
  call get_command_argument(3, outdir)
  if (len_trim(outdir) == 0) call fsbm_fail('usage: run_sweeps sweep|refusal <name> <outdir>')

  dt_init = 12.0
  call fast_hucminit(dt_init)
  fr_lim_z = 0.0
  frh_lim_z = 0.0

  if (trim(mode) == 'sweep') then
    call fsbm_open(s, trim(outdir)//'/sweep_'//trim(what))
    select case (trim(what))
    case ('polysvp')
      call sweep_polysvp()
    case ('jerrate')
      call sweep_jerrate()
    case ('jersupsat')
      call sweep_jersupsat()
    case ('jerdfun')
      call sweep_jerdfun()
    case ('jernewf')
      call sweep_jernewf()
    case ('onecond1')
      call sweep_onecond1()
    case ('nucleation')
      call sweep_nucleation()
    case ('coll_xxx')
      call sweep_coll_xxx()
    case ('coal_bott')
      call sweep_coal_bott()
    case ('breakup')
      call sweep_breakup()
    case ('spont')
      call sweep_spont()
    case ('falflux')
      call sweep_falflux()
    case ('freez')
      call sweep_phase(1)
    case ('melt')
      call sweep_phase(2)
    case ('ice_nucl')
      call sweep_ice_nucl()
    case ('jernucl01_ice')
      call sweep_jernucl01_ice()
    case ('jerrate_ice')
      call sweep_jerrate_ice()
    case ('jertimesc_ice')
      call sweep_jertimesc_ice()
    case ('jerdfun_ice')
      call sweep_jerdfun_ice()
    case ('onecond2')
      call sweep_onecond2()
    case ('onecond3')
      call sweep_onecond3()
    case ('coll_cold')
      call sweep_coll_cold()
    case ('modkrn')
      call sweep_modkrn()
    case ('coal_cold')
      call sweep_coal_cold()
    case default
      call fsbm_fail('unknown sweep family '//trim(what))
    end select
    call fsbm_close(s)
  else if (trim(mode) == 'refusal') then
    call refusal_case(trim(what))
  else if (trim(mode) == 'refusal_cold') then
    call refusal_cold_case(trim(what))
  else
    call fsbm_fail('unknown mode '//trim(mode))
  end if

contains

  ! ------------------------------------------------------------------ helpers
  real(8) function drop_radius(k)
    integer, intent(in) :: k
    drop_radius = (3.0d0*real(xl(k), 8)/(4.0d0*pi_d))**(1.0d0/3.0d0)
  end function drop_radius

  ! Lognormal number mode: nd [cm-3], mode radius rm [cm], geometric sd sig.
  subroutine add_lognormal(nd, rm, sig, ff)
    real(8), intent(in) :: nd, rm, sig
    real(8), intent(inout) :: ff(nb)
    integer :: k
    real(8) :: r, nk, ls
    ls = log(sig)
    do k = 1, nb
      r = drop_radius(k)
      nk = nd/(sqrt(2.0d0*pi_d)*ls)*exp(-(log(r/rm))**2/(2.0d0*ls*ls))*real(col, 8)
      ff(k) = ff(k) + nk/(3.0d0*real(col, 8)*real(xl(k), 8))
    end do
  end subroutine add_lognormal

  ! Marshall-Palmer rain for rate [mm/h] over bins kmin..kmax:
  ! N0 = 0.08 cm-4, lambda = 41 R^-0.21 cm-1, N_k = N0 exp(-lambda D) D COL.
  subroutine add_mp(rate, kmin, kmax, ff)
    real(8), intent(in) :: rate
    integer, intent(in) :: kmin, kmax
    real(8), intent(inout) :: ff(nb)
    integer :: k
    real(8) :: d, lam, nk
    lam = 41.0d0*rate**(-0.21d0)
    do k = kmin, kmax
      d = 2.0d0*drop_radius(k)
      nk = 0.08d0*exp(-lam*d)*d*real(col, 8)
      ff(k) = ff(k) + nk/(3.0d0*real(col, 8)*real(xl(k), 8))
    end do
  end subroutine add_mp

  ! Spectrum catalogue (f-form, double).
  subroutine spectrum(id, ff)
    integer, intent(in) :: id
    real(8), intent(out) :: ff(nb)
    ff = 0.0d0
    select case (id)
    case (1)   ! cloud
      call add_lognormal(200.0d0, 8.0d-4, 1.3d0, ff)
    case (2)   ! cloud plus rain
      call add_lognormal(200.0d0, 8.0d-4, 1.3d0, ff)
      call add_mp(5.0d0, 15, nb, ff)
    case (3)   ! rain only
      call add_mp(20.0d0, 15, nb, ff)
    case (4)   ! dense cloud
      call add_lognormal(1000.0d0, 6.0d-4, 1.25d0, ff)
    case (5)   ! cloud with drizzle mode
      call add_lognormal(200.0d0, 10.0d-4, 1.35d0, ff)
      call add_lognormal(1.0d0, 40.0d-4, 1.4d0, ff)
    case (6)   ! big drops (heavy rain, large bins only)
      call add_mp(100.0d0, 24, nb, ff)
    case (7)   ! small droplets
      call add_lognormal(500.0d0, 3.0d-4, 1.2d0, ff)
    case (8)   ! cloud with a sharp upper tail in bins 9..12 (remap merge)
      call add_lognormal(200.0d0, 5.0d-4, 1.25d0, ff)
      ff(9:12) = ff(9:12)*1.0d-4
    case (9)   ! all bins filled (spontaneous breakup, collisional breakup)
      call add_mp(20.0d0, 1, nb, ff)
    case (10)  ! intense rain over all bins
      call add_mp(100.0d0, 1, nb, ff)
    case (11)  ! light rain over all bins
      call add_mp(5.0d0, 1, nb, ff)
    case (12)  ! rain truncated below the spontaneous breakup bin
      call add_mp(20.0d0, 15, 30, ff)
    case (13)  ! empty
      ff = 0.0d0
    case (14)  ! big drops, ten times the concentration
      call add_mp(100.0d0, 24, nb, ff)
      ff = 10.0d0*ff
    case default
      call fsbm_fail('unknown spectrum id')
    end select
  end subroutine spectrum

  ! State for a target water supersaturation, with FAST_SBM's expressions.
  subroutine thermo(t, sat, pp, qq, del1n, del2n, div1, div2)
    real(8), intent(in) :: t, sat, pp
    real(8), intent(out) :: qq, del1n, del2n, div1, div2
    real(8) :: es1n, es2n, ew1n, ew
    es1n = AA1_MY*DEXP(-BB1_MY/t)
    ew = (1.0d0 + sat)*es1n
    qq = 0.622d0*ew/(pp - 0.378d0*ew)
    es1n = AA1_MY*DEXP(-BB1_MY/t)
    es2n = AA2_MY*DEXP(-BB2_MY/t)
    ew1n = qq*pp/(0.622+0.378*qq)
    div1 = ew1n/es1n
    del1n = ew1n/es1n-1.
    div2 = ew1n/es2n
    del2n = ew1n/es2n-1.
  end subroutine thermo

  ! Pressure-corrected drop fall speeds, as FAST_SBM builds VR1_Z (:4518).
  subroutine fall_speeds(pcgs, vz)
    real, intent(in) :: pcgs
    real, intent(out) :: vz(nb)
    real :: factor_p
    factor_p = DSQRT(1.0D6/pcgs)
    vz(1:nb) = vr1(1:nb)*factor_p
  end subroutine fall_speeds

  real function air_density_cgs(pcgs, t)
    real, intent(in) :: pcgs
    real(8), intent(in) :: t
    air_density_cgs = real(real(pcgs, 8)/(287.0d4*t))
  end function air_density_cgs

  ! ------------------------------------------------------------------ polysvp
  subroutine sweep_polysvp()
    integer :: n, it
    real :: tt
    real :: extra(6)
    extra = (/ 273.15, 273.16, 233.15, 313.15, 250.123, 299.999 /)
    do it = 0, 1
      do n = 0, 320 + 6
        if (n <= 320) then
          tt = 233.0 + 0.25*real(n)
        else
          tt = extra(n - 320)
        end if
        call fsbm_begin(s, 'case')
        call fsbm_put(s, 'tt', tt)
        call fsbm_put(s, 'itype', int(it, 4))
        call fsbm_put(s, 'es', polysvp(tt, it))
        call fsbm_end(s)
      end do
    end do
  end subroutine sweep_polysvp

  ! ------------------------------------------------------------------ jerrate
  subroutine sweep_jerrate()
    real(8) :: tlist(4), r1d(nb), vd(nb), tps, pp
    real :: plist(4), pcgs, vz(nb), b11(nb), fl1(nb)
    integer :: it, ip
    tlist = (/ 278.15d0, 288.15d0, 298.15d0, 308.15d0 /)
    plist = (/ 1000.0, 850.0, 700.0, 500.0 /)
    fl1 = 0.0
    do ip = 1, 4
      pcgs = plist(ip)*1000.0
      call fall_speeds(pcgs, vz)
      do it = 1, 4
        r1d = xl
        vd = vz
        tps = tlist(it)
        pp = pcgs
        b11 = 0.0
        call jerrate_ks(r1d, tps, pp, vd, rlec, ro1bl, b11, 1, 1, fl1, nb, 3)
        call fsbm_begin(s, 'case')
        call fsbm_put(s, 'tp', tps)
        call fsbm_put(s, 'pp', pp)
        call fsbm_put(s, 'xl', r1d)
        call fsbm_put(s, 'vxl', vd)
        call fsbm_put(s, 'b11', b11)
        call fsbm_end(s)
      end do
    end do
  end subroutine sweep_jerrate

  ! ---------------------------------------------------------------- jersupsat
  integer function supsat_branch(rw0, pw0, ri0, pi0, dt, clamp)
    real(8), intent(in) :: rw0, pw0, ri0, pi0
    real, intent(in) :: dt
    integer, intent(out) :: clamp
    real(8) :: rw, pw, ri, pi, alfa, gama, g2
    integer :: irw, iri
    rw = rw0; pw = pw0; ri = ri0; pi = pi0
    irw = 1; iri = 1; clamp = 0
    if (max(rw, pw, ri, pi) <= 1.0d-10) then
      supsat_branch = 0
      return
    end if
    if (dmax1(rw, pw) > 1.0d-10) then
      if (rw < 1.0d-20) then
        rw = 1.0d-20; irw = 0
      end if
      if (pw < 1.0d-20) pw = 1.0d-20
      if (dmax1(pi/pw, ri/rw) <= 1.0d-4) then
        ri = 0.0; iri = 0; pi = 0.0
      end if
      if (dmin1(pi/pw, ri/rw) > 1.0d0/1.0d-4) then
        rw = 0.0; irw = 0; pw = 0.0
      end if
    else
      rw = 0.0; irw = 0; pw = 0.0
    end if
    if (irw == 0 .and. iri == 0) then
      supsat_branch = 1
    else if (iri == 0) then
      if (abs(rw*dt) > 1.0E-6) then
        supsat_branch = 2
      else
        supsat_branch = 3
      end if
    else if (irw == 0) then
      if (abs(pi*dt) > 1.0E-6) then
        supsat_branch = 4
      else
        supsat_branch = 5
      end if
    else
      alfa = dsqrt((rw-pi)*(rw-pi)+4.0d0*pw*ri)
      gama = 0.5d0*(alfa-rw-pi)
      g2 = rw*pi-ri*pw
      if (g2 < 1.0d-20) clamp = 1
      if (dabs(gama*dt) > 1.0E-6) then
        supsat_branch = 6
      else if (abs(ri/rw) > 1.0E-12) then
        if (abs(rw/ri) > 1.0E-12) then
          supsat_branch = 7
        else
          supsat_branch = 8
        end if
      else
        supsat_branch = 9
      end if
    end if
  end function supsat_branch

  subroutine sweep_jersupsat()
    ! columns: intended branch, rw, pw, ri, pi, del1, del2, isym-ice flag
    integer, parameter :: ncase = 17
    real(8) :: tab(8, ncase)
    real :: dts(2), dyns(2, 2)
    real(8) :: del1, del2, del1n, del2n, d1i, d2i, rw, pw, ri, pi
    integer :: c, id, iy, isym1, isym2(3), isym3, isym4, isym5, br, clamp
    integer :: isym1_in, isym2_in(3), isym3_in, isym4_in, isym5_in
    real :: dt, dyn1, dyn2
    tab(:, 1)  = (/ 0d0, 1d-12, 1d-12, 1d-12, 1d-12, 0.01d0, 0.05d0, 0d0 /)
    tab(:, 2)  = (/ 2d0, 0.5d0, 0.4d0, 0d0, 0d0, 0.01d0, 0.05d0, 0d0 /)
    tab(:, 3)  = (/ 2d0, 0.5d0, 0.4d0, 0d0, 0d0, -0.05d0, -0.03d0, 0d0 /)
    tab(:, 4)  = (/ 2d0, 0.5d0, 0.4d0, 0d0, 0d0, 0.01d0, 0.05d0, 1d0 /)
    tab(:, 5)  = (/ 3d0, 1d-8, 2d-8, 0d0, 0d0, 0.01d0, 0.05d0, 0d0 /)
    tab(:, 6)  = (/ 3d0, 1d-8, 2d-8, 0d0, 0d0, -0.02d0, -0.01d0, 0d0 /)
    tab(:, 7)  = (/ 1d0, 0d0, 1d-5, 0d0, 0d0, 0.01d0, 0.05d0, 0d0 /)
    tab(:, 8)  = (/ 4d0, 0d0, 0d0, 0.3d0, 0.6d0, -0.01d0, 0.05d0, 1d0 /)
    tab(:, 9)  = (/ 4d0, 1d-6, 1d-6, 0.3d0, 0.6d0, -0.01d0, 0.05d0, 1d0 /)
    tab(:, 10) = (/ 5d0, 0d0, 0d0, 1d-8, 2d-8, -0.01d0, 0.05d0, 1d0 /)
    tab(:, 11) = (/ 6d0, 0.5d0, 0.4d0, 0.3d0, 0.6d0, 0.01d0, 0.05d0, 1d0 /)
    tab(:, 12) = (/ 6d0, 0.5d0, 0.4d0, 0.3d0, 0.6d0, -0.02d0, 0.02d0, 1d0 /)
    tab(:, 13) = (/ 6d0, 0.5d0, 0.4d0, 2.0d0, 0.6d0, 0.01d0, 0.05d0, 1d0 /)
    tab(:, 14) = (/ 7d0, 0.5d0, 0.4d0, 0.75d0, 0.6d0, 0.01d0, 0.05d0, 1d0 /)
    tab(:, 15) = (/ 7d0, 0.5d0, 0.4d0, 0.75d0, 0.6d0, -0.01d0, 0.02d0, 1d0 /)
    tab(:, 16) = (/ 8d0, 1d-19, 1d-9, 1d-6, 0.9d-5, 0.01d0, 0.05d0, 1d0 /)
    tab(:, 17) = (/ 9d0, 1d-6, 1d-9, 1d-19, 2d-13, 0.01d0, 0.05d0, 1d0 /)
    dts = (/ 4.0, 1.0 /)
    dyns(:, 1) = (/ 0.0, 0.0 /)
    dyns(:, 2) = (/ 1.0e-4, -2.0e-4 /)
    do c = 1, ncase
      do id = 1, 2
        do iy = 1, 2
          dt = dts(id)
          dyn1 = dyns(1, iy)
          dyn2 = dyns(2, iy)
          del1 = tab(6, c)
          del2 = tab(7, c)
          rw = tab(2, c); pw = tab(3, c); ri = tab(4, c); pi = tab(5, c)
          isym1 = 1
          isym2 = 0
          isym3 = 0
          isym4 = 0
          isym5 = 0
          if (tab(8, c) > 0.5d0) then
            isym2(1) = 1
            isym3 = 1
          else if (c == 2) then
            isym3 = 1   ! water-only inputs with a stale ice flag: JERSUPSAT clears it
          end if
          isym1_in = isym1; isym2_in = isym2; isym3_in = isym3
          isym4_in = isym4; isym5_in = isym5
          br = supsat_branch(rw, pw, ri, pi, dt, clamp)
          del1n = 0.0d0; del2n = 0.0d0; d1i = 0.0d0; d2i = 0.0d0
          call fsbm_begin(s, 'case')
          call fsbm_put(s, 'case_id', int(c, 4))
          call fsbm_put(s, 'intended_branch', int(nint(tab(1, c)), 4))
          call fsbm_put(s, 'classified_branch', int(br, 4))
          call fsbm_put(s, 'g2_clamp', int(clamp, 4))
          call fsbm_put(s, 'del1', del1)
          call fsbm_put(s, 'del2', del2)
          call fsbm_put(s, 'rw_in', rw)
          call fsbm_put(s, 'pw_in', pw)
          call fsbm_put(s, 'ri_in', ri)
          call fsbm_put(s, 'pi_in', pi)
          call fsbm_put(s, 'dt', dt)
          call fsbm_put(s, 'dyn1', dyn1)
          call fsbm_put(s, 'dyn2', dyn2)
          call fsbm_put(s, 'isym_in', int((/ isym1_in, isym2_in, isym3_in, isym4_in, isym5_in /), 4))
          call jersupsat_ks(del1, del2, del1n, del2n, rw, pw, ri, pi, dt, d1i, d2i, &
                            dyn1, dyn2, isym1, isym2, isym3, isym4, isym5)
          call fsbm_put(s, 'del1n', del1n)
          call fsbm_put(s, 'del2n', del2n)
          call fsbm_put(s, 'del1int', d1i)
          call fsbm_put(s, 'del2int', d2i)
          call fsbm_put(s, 'rw_out', rw)
          call fsbm_put(s, 'pw_out', pw)
          call fsbm_put(s, 'ri_out', ri)
          call fsbm_put(s, 'pi_out', pi)
          call fsbm_put(s, 'isym_out', int((/ isym1, isym2, isym3, isym4, isym5 /), 4))
          call fsbm_end(s)
        end do
      end do
    end do
  end subroutine sweep_jersupsat

  ! ------------------------------------------------------------------ jerdfun
  subroutine sweep_jerdfun()
    integer :: specs(4), is, id, idr, idrop, idrop_in
    real :: d1ns(5), d1n, b21(nb), fi(nb), psi(nb), fl2(nb), vz(nb), pcgs, fl1(nb)
    real(8) :: xi_d(nb), xin(nb), ffd(nb), tpn, pp, vd(nb)
    specs = (/ 1, 2, 3, 7 /)
    d1ns = (/ 1.0e-3, 4.0e-2, -1.0e-3, -0.4, -2.0 /)
    tpn = 288.15d0
    pcgs = 900.0*1000.0
    pp = pcgs
    call fall_speeds(pcgs, vz)
    vd = vz
    xi_d = xl
    fl1 = 0.0
    b21 = 0.0
    call jerrate_ks(xi_d, tpn, pp, vd, rlec, ro1bl, b21, 1, 1, fl1, nb, 3)
    do is = 1, 4
      call spectrum(specs(is), ffd)
      do id = 1, 5
        do idr = 0, 1
          d1n = d1ns(id)
          fi = real(ffd)
          psi = 0.0
          fl2 = 0.0
          xin = 0.0d0
          idrop = idr
          idrop_in = idr
          call jerdfun_ks(xi_d, xin, b21, fi, psi, fl2, d1n, 1, 1, 1, tpn, idrop, &
                          fr_lim_z, frh_lim_z, 3, nb, col, 1, 1, 1, 1, 1)
          call fsbm_begin(s, 'case')
          call fsbm_put(s, 'spectrum', int(specs(is), 4))
          call fsbm_put(s, 'del2n', d1n)
          call fsbm_put(s, 'idrop_in', int(idrop_in, 4))
          call fsbm_put(s, 'tpn', tpn)
          call fsbm_put(s, 'xi', xi_d)
          call fsbm_put(s, 'b21', b21)
          call fsbm_put(s, 'fi', fi)
          call fsbm_put(s, 'xin', xin)
          call fsbm_put(s, 'psi', psi)
          call fsbm_put(s, 'idrop_out', int(idrop, 4))
          call fsbm_end(s)
        end do
      end do
    end do
  end subroutine sweep_jerdfun

  ! ------------------------------------------------------------------ jernewf
  ! JERNEWF_KS is PRIVATE to module_mp_SBM_Auxiliary (the access transform
  ! does not touch that module), so it is reached through JERDFUN_KS with
  ! IND = 2, whose new masses are xiN = xi + DEL2N*B21 exactly: with
  ! DEL2N = 1 the harness sets RN per bin. The record carries RR, FI and RN
  ! exactly as JERNEWF_KS receives them (RN is JERDFUN_KS's xiN output).
  subroutine sweep_jernewf()
    integer :: specs(3), is, im, idr, idrop, idrop_in, k
    real :: fracs(6), b21(nb), fi(nb), psi(nb), fl2(nb), one
    real(8) :: xi_d(nb), xin(nb), ffd(nb), tpn, f
    specs = (/ 1, 2, 8 /)
    ! modes: small growth, large growth, mixed, evaporation, evaporation below
    ! bin 1, no change
    fracs = (/ 1.0e-3, 0.9, 0.0, -0.5, -0.9, 0.0 /)
    tpn = 288.15d0
    xi_d = xl
    one = 1.0
    do is = 1, 3
      call spectrum(specs(is), ffd)
      do im = 1, 6
        do idr = 0, 1
          do k = 1, nb
            f = fracs(im)
            if (im == 3) then
              if (mod(k, 2) == 1) then
                f = 0.5d0
              else
                f = 0.01d0
              end if
            end if
            b21(k) = real(xi_d(k)*f)
          end do
          fi = real(ffd)
          psi = 0.0
          fl2 = 0.0
          xin = 0.0d0
          idrop = idr
          idrop_in = idr
          call jerdfun_ks(xi_d, xin, b21, fi, psi, fl2, one, 1, 2, 1, tpn, idrop, &
                          fr_lim_z, frh_lim_z, 3, nb, col, 1, 1, 1, 1, 1)
          call fsbm_begin(s, 'case')
          call fsbm_put(s, 'spectrum', int(specs(is), 4))
          call fsbm_put(s, 'mode', int(im, 4))
          call fsbm_put(s, 'idrop_in', int(idrop_in, 4))
          call fsbm_put(s, 'tpn', tpn)
          call fsbm_put(s, 'rr', xi_d)
          call fsbm_put(s, 'fi', fi)
          call fsbm_put(s, 'rn', xin)
          call fsbm_put(s, 'psi', psi)
          call fsbm_put(s, 'idrop_out', int(idrop, 4))
          call fsbm_end(s)
        end do
      end do
    end do
  end subroutine sweep_jernewf

  ! ----------------------------------------------------------------- onecond1
  subroutine sweep_onecond1()
    real(8) :: tlist(2), slist(5), ffd(nb)
    real(8) :: tt, qq, pp, del1n, del2n, div1, div2, tt0, qq0, d1n0, d2n0
    integer :: specs(4), it, isat, is, isym1, isym2(3), isym3, isym4, isym5
    real :: pcgs, ror, vz(nb), ff1(nb), psi1(nb), ff1_in(nb), dtcond, w_in, dx_in
    tlist = (/ 280.15d0, 295.15d0 /)
    slist = (/ 5.0d-3, 5.0d-4, -1.0d-3, -2.0d-2, -2.0d-1 /)
    specs = (/ 1, 4, 2, 3 /)
    pcgs = 900.0*1000.0
    pp = pcgs
    call fall_speeds(pcgs, vz)
    dtcond = 12.0/real(3)
    w_in = 0.0
    dx_in = 1000.0
    do it = 1, 2
      do isat = 1, 5
        do is = 1, 4
          tt = tlist(it)
          call thermo(tt, slist(isat), pp, qq, del1n, del2n, div1, div2)
          ror = air_density_cgs(pcgs, tt)
          call spectrum(specs(is), ffd)
          ff1 = real(ffd)
          ff1_in = ff1
          psi1 = 0.0
          isym1 = 0
          if (maxval(ff1) > 0.0) isym1 = 1
          isym2 = 0; isym3 = 0; isym4 = 0; isym5 = 0
          tt0 = tt; qq0 = qq; d1n0 = del1n; d2n0 = del2n
          call onecond1(tt, qq, pp, ror, vz, pcgs, del1n, del2n, div1, div2, &
                        ff1, psi1, xl, rlec, ro1bl, AA1_MY, BB1_MY, AA2_MY, BB2_MY, &
                        c1_mey, c2_mey, col, dtcond, 3, nb, isym1, isym2, isym3, &
                        isym4, isym5, 1, 1, 1, w_in, dx_in, 2)
          call fsbm_begin(s, 'case')
          call fsbm_put(s, 'spectrum', int(specs(is), 4))
          call fsbm_put(s, 'target_sat', slist(isat))
          call fsbm_put(s, 'tt_in', tt0)
          call fsbm_put(s, 'qq_in', qq0)
          call fsbm_put(s, 'pp', pp)
          call fsbm_put(s, 'ror', ror)
          call fsbm_put(s, 'psingle', pcgs)
          call fsbm_put(s, 'del1n_in', d1n0)
          call fsbm_put(s, 'del2n_in', d2n0)
          call fsbm_put(s, 'div1', div1)
          call fsbm_put(s, 'div2', div2)
          call fsbm_put(s, 'vr1', vz)
          call fsbm_put(s, 'dtcond', dtcond)
          call fsbm_put(s, 'isym1', int(isym1, 4))
          call fsbm_put(s, 'ff1_in', ff1_in)
          call fsbm_put(s, 'tt_out', tt)
          call fsbm_put(s, 'qq_out', qq)
          call fsbm_put(s, 'del1n_out', del1n)
          call fsbm_put(s, 'del2n_out', del2n)
          call fsbm_put(s, 'ff1_out', ff1)
          call fsbm_end(s)
        end do
      end do
    end do
  end subroutine sweep_onecond1

  ! --------------------------------------------------------------- nucleation
  subroutine sweep_nucleation()
    real(8) :: tlist(3), slist(5), ffd(nb)
    real(8) :: tt, qq, pp, sup1, sup2, div1, div2, tt0, qq0, sup10, sup20
    real :: wlist(3), pcgs, ror, psi1(nb), psi2(nb, 3), fccnr(na), psi1_in(nb), &
            fccnr_in(na), sup2_old, dsup, win
    integer :: it, isat, ib, iw, ia, is_cb, k
    tlist = (/ 278.15d0, 290.15d0, 300.15d0 /)
    slist = (/ 1.0d-4, 1.0d-3, 3.0d-3, 1.0d-2, 3.0d-2 /)
    wlist = (/ 10.0, 100.0, 500.0 /)
    pcgs = 900.0*1000.0
    pp = pcgs
    do it = 1, 3
      do isat = 1, 5
        do ib = 0, 1
          do iw = 1, 3
            do ia = 1, 3
              tt = tlist(it)
              call thermo(tt, slist(isat), pp, qq, sup1, sup2, div1, div2)
              ror = air_density_cgs(pcgs, tt)
              is_cb = 1 - ib
              win = wlist(iw)
              psi2 = 0.0
              select case (ia)
              case (1)   ! fresh continental, no drops yet
                fccnr = real(fccnr_con)
                psi1 = 0.0
              case (2)   ! depleted continental (radii above 0.02 um gone), drops present
                fccnr = real(fccnr_con)
                do k = 1, na
                  if (rccn(k) > 2.0e-6) fccnr(k) = 0.0
                end do
                call spectrum(1, ffd)
                psi1 = real(0.5d0*ffd)
              case (3)   ! fresh maritime
                fccnr = real(fccnr_mar)
                psi1 = 0.0
              end select
              psi1_in = psi1
              fccnr_in = fccnr
              sup2_old = real(sup2)
              dsup = 0.0
              tt0 = tt; qq0 = qq; sup10 = sup1; sup20 = sup2
              call jernucl01_ks(psi1, psi2, fccnr, xl, xi, tt, qq, ror, pcgs, sup1, sup2, &
                                col, sup2_old, dsup, rccn, dropradii, nb, na, 3, 1, &
                                win, is_cb, ro_solute, ions, mwaero, 1, 1, 1)
              call fsbm_begin(s, 'case')
              call fsbm_put(s, 'aerosol', int(ia, 4))
              call fsbm_put(s, 'is_cloud_base', int(is_cb, 4))
              call fsbm_put(s, 'win', win)
              call fsbm_put(s, 'target_sat', slist(isat))
              call fsbm_put(s, 'tt_in', tt0)
              call fsbm_put(s, 'qq_in', qq0)
              call fsbm_put(s, 'sup1_in', sup10)
              call fsbm_put(s, 'sup2_in', sup20)
              call fsbm_put(s, 'ror', ror)
              call fsbm_put(s, 'pp', pcgs)
              call fsbm_put(s, 'psi1_in', psi1_in)
              call fsbm_put(s, 'fccnr_in', fccnr_in)
              call fsbm_put(s, 'tt_out', tt)
              call fsbm_put(s, 'qq_out', qq)
              call fsbm_put(s, 'sup1_out', sup1)
              call fsbm_put(s, 'sup2_out', sup2)
              call fsbm_put(s, 'psi1_out', psi1)
              call fsbm_put(s, 'fccnr_out', fccnr)
              call fsbm_end(s)
            end do
          end do
        end do
      end do
    end do
  end subroutine sweep_nucleation

  ! ----------------------------------------------------------------- coll_xxx
  subroutine g1_from_ff(ff, g)
    real, intent(in) :: ff(nb)
    real(8), intent(out) :: g(nb)
    integer :: kr
    do kr = 1, nb
      g(kr) = ff(kr)*3.*xl(kr)*xl(kr)*1.E3
    end do
  end subroutine g1_from_ff

  subroutine xl_mg_grid(x)
    real(8), intent(out) :: x(nb)
    integer :: i
    do i = 1, nb
      x(i) = xl(i)*1.e3
    end do
  end subroutine xl_mg_grid

  subroutine sweep_coll_xxx()
    real :: plist(4), dlist(3), pz, dtc, ff(nb)
    real(8) :: g(nb), gin(nb), fl(nb), x(nb), ffd(nb)
    integer :: ip, id, is, specs(3)
    plist = (/ 1000.0, 900.0, 650.0, 450.0 /)
    dlist = (/ 3.0, 12.0, 30.0 /)
    specs = (/ 1, 2, 3 /)
    call xl_mg_grid(x)
    do ip = 1, 4
      do id = 1, 3
        pz = plist(ip)*1000.0
        dtc = dlist(id)
        call kernals_ks(dtc, nb, pz)
        call fsbm_begin(s, 'kernel')
        call fsbm_put(s, 'p_z', pz)
        call fsbm_put(s, 'dt_coll', dtc)
        call fsbm_put(s, 'cwll', cwll)
        call fsbm_end(s)
        do is = 1, 3
          call spectrum(specs(is), ffd)
          ff = real(ffd)
          call g1_from_ff(ff, g)
          gin = g
          fl = 1.0d0
          call coll_xxx_lwf(g, fl, cwll, x, chucm, ima, 1.d0, nb)
          call fsbm_begin(s, 'case')
          call fsbm_put(s, 'p_z', pz)
          call fsbm_put(s, 'dt_coll', dtc)
          call fsbm_put(s, 'spectrum', int(specs(is), 4))
          call fsbm_put(s, 'x', x)
          call fsbm_put(s, 'g_in', gin)
          call fsbm_put(s, 'g_out', g)
          call fsbm_put(s, 'fl_out', fl)
          call fsbm_end(s)
        end do
      end do
    end do
  end subroutine sweep_coll_xxx

  ! ---------------------------------------------------------------- coal_bott
  subroutine coal_case(pz_hpa, dtc, spec)
    real, intent(in) :: pz_hpa, dtc
    integer, intent(in) :: spec
    real :: ff1r(nb), ff2r(nb, 3), ff3r(nb), ff4r(nb), ff5r(nb), rho, colleff, &
            tcrit, ttcoal, ff1_in(nb)
    real(8) :: tt, qq, pp, del1in, del2in, fls(nb), flg(nb), flh(nb), frs(nb), &
               ffd(nb), tt0, d10, d20
    tcrit = 2.7015E02
    ttcoal = 2.3315E02
    call spectrum(spec, ffd)
    ff1r = real(ffd)
    ff1_in = ff1r
    ff2r = 0.0; ff3r = 0.0; ff4r = 0.0; ff5r = 0.0
    fls = 0.0d0; flg = 0.0d0; flh = 0.0d0; frs = 0.0d0
    tt = 288.15d0
    pp = real(pz_hpa*1000.0, 8)
    qq = 0.010d0
    del1in = 1.0d-3
    del2in = 5.0d-2
    rho = air_density_cgs(pz_hpa*1000.0, tt)
    colleff = 0.0
    tt0 = tt; d10 = del1in; d20 = del2in
    cap_icol_drop = -1; cap_icol_brk = -1; cap_ndiv = -1; cap_path = -1
    cap_g1_xxx_in = 0.0d0; cap_g1_xxx_out = 0.0d0; cap_g1_at11 = 0.0d0
    cap_break_per = 0.0d0
    call coal_bott_new(ff1r, ff2r, ff3r, ff4r, ff5r, tt, qq, pp, rho, dtc, tcrit, &
                       ttcoal, fls, flg, flh, frs, del1in, del2in, 1, 1, 1, 1, colleff)
    call fsbm_begin(s, 'case')
    call fsbm_put(s, 'p_hpa', pz_hpa)
    call fsbm_put(s, 'dt_coll', dtc)
    call fsbm_put(s, 'spectrum', int(spec, 4))
    call fsbm_put(s, 'tt_in', tt0)
    call fsbm_put(s, 'qq', qq)
    call fsbm_put(s, 'pp', pp)
    call fsbm_put(s, 'rho', rho)
    call fsbm_put(s, 'del1in_in', d10)
    call fsbm_put(s, 'del2in_in', d20)
    call fsbm_put(s, 'ff1_in', ff1_in)
    call fsbm_put(s, 'ff1_out', ff1r)
    call fsbm_put(s, 'tt_out', tt)
    call fsbm_put(s, 'del1in_out', del1in)
    call fsbm_put(s, 'del2in_out', del2in)
    call fsbm_put(s, 'icol_drop', cap_icol_drop)
    call fsbm_put(s, 'icol_drop_brk', cap_icol_brk)
    call fsbm_put(s, 'ndiv', cap_ndiv)
    call fsbm_put(s, 'breakup_path', cap_path)
    call fsbm_put(s, 'g1_before_xxx', cap_g1_xxx_in)
    call fsbm_put(s, 'g1_after_xxx', cap_g1_xxx_out)
    call fsbm_put(s, 'g1_after_breakup', cap_g1_at11)
    call fsbm_put(s, 'break_drop_per', cap_break_per)
    call fsbm_end(s)
  end subroutine coal_case

  subroutine sweep_coal_bott()
    real :: plist(4), dlist(3), stress_dt(3)
    integer :: ip, id, is, specs(3)
    plist = (/ 1000.0, 900.0, 650.0, 450.0 /)
    dlist = (/ 3.0, 12.0, 30.0 /)
    stress_dt = (/ 12.0, 30.0, 60.0 /)
    specs = (/ 1, 2, 3 /)
    pr_capture = .true.
    do ip = 1, 4
      do id = 1, 3
        do is = 1, 3
          call coal_case(plist(ip), dlist(id), specs(is))
        end do
      end do
    end do
    ! breakup stress: big drops and ten times as many, long steps
    do id = 1, 3
      call coal_case(1000.0, stress_dt(id), 6)
      call coal_case(1000.0, stress_dt(id), 14)
      call coal_case(1000.0, stress_dt(id), 10)
    end do
    pr_capture = .false.
  end subroutine sweep_coal_bott

  ! ------------------------------------------------------------------ breakup
  subroutine sweep_breakup()
    integer :: specs(4), is, id, n, it, kr
    real :: dts(4), ff(nb), dt_coll_r
    real(8) :: g1(nb), gt(nb), gin(nb), xt(nb), xmg(nb), dtb, ffd(nb), sb, sa
    integer :: ndivs(4)
    specs = (/ 3, 6, 14, 10 /)
    dts = (/ 12.0, 6.0, 3.0, 1.5 /)
    ndivs = (/ 1, 2, 4, 8 /)
    call xl_mg_grid(xmg)
    do kr = 1, nb
      xt(kr) = xmg(kr)*1.D-3
    end do
    dt_coll_r = 12.0
    do is = 1, 4
      call spectrum(specs(is), ffd)
      ff = real(ffd)
      call g1_from_ff(ff, g1)
      do kr = 1, nb
        gin(kr) = g1(kr)*1.D-3
      end do
      ! single calls at several time steps
      do id = 1, 4
        gt = gin
        dtb = dts(id)
        call coll_breakup_ks(gt, xt, nb, dtb, 18, pkij, qkj, nb, nb)
        call fsbm_begin(s, 'case')
        call fsbm_put(s, 'spectrum', int(specs(is), 4))
        call fsbm_put(s, 'ndiv', int(1, 4))
        call fsbm_put(s, 'dt_call', dtb)
        call fsbm_put(s, 'xt', xt)
        call fsbm_put(s, 'gt_in', gin)
        call fsbm_put(s, 'gt_out', gt)
        call fsbm_end(s)
      end do
      ! COAL_BOTT_NEW's substep sequences: ndiv calls of dt_coll/ndiv
      do n = 1, 4
        gt = gin
        dtb = dt_coll_r/ndivs(n)
        do it = 1, ndivs(n)
          call coll_breakup_ks(gt, xt, nb, dtb, 18, pkij, qkj, nb, nb)
        end do
        sb = 0.0d0
        sa = 0.0d0
        do kr = 1, nb
          sb = sb + gin(kr)
          sa = sa + gt(kr)
        end do
        call fsbm_begin(s, 'sequence')
        call fsbm_put(s, 'spectrum', int(specs(is), 4))
        call fsbm_put(s, 'ndiv', int(ndivs(n), 4))
        call fsbm_put(s, 'dt_call', dtb)
        call fsbm_put(s, 'gt_in', gin)
        call fsbm_put(s, 'gt_out', gt)
        call fsbm_put(s, 'mass_ratio', sa/sb)
        call fsbm_end(s)
      end do
    end do
  end subroutine sweep_breakup

  ! -------------------------------------------------------------------- spont
  subroutine sweep_spont()
    integer :: specs(6), is, id, imax, kr, taken
    real :: dts(2)
    real(8) :: ffd(nb), ff(nb), ffin(nb), xld(nb)
    specs = (/ 11, 9, 10, 12, 13, 6 /)
    dts = (/ 12.0, 60.0 /)
    xld = xl
    do is = 1, 6
      do id = 1, 2
        call spectrum(specs(is), ffd)
        ff = real(real(ffd), 8)     ! FF1R (default real) copied into FF1R_D
        ffin = ff
        imax = nb
        do kr = nb, 1, -1
          imax = kr
          if (ff(kr) > 0.0d0) exit
        end do
        taken = 0
        if (sum(ff) > nb*1.0d-30 .and. imax >= ikr_spon_break) taken = 1
        call spont_rain_breakup(dts(id), ff, xld, prob, gain_var_new, nnd, nb, ikr_spon_break)
        call fsbm_begin(s, 'case')
        call fsbm_put(s, 'spectrum', int(specs(is), 4))
        call fsbm_put(s, 'dt', dts(id))
        call fsbm_put(s, 'xl', xld)
        call fsbm_put(s, 'ff_in', ffin)
        call fsbm_put(s, 'ff_out', ff)
        call fsbm_put(s, 'imax', int(imax, 4))
        call fsbm_put(s, 'taken', int(taken, 4))
        call fsbm_end(s)
      end do
    end do
  end subroutine sweep_spont

  ! ------------------------------------------------------------------ falflux
  subroutine falflux_grid(igrid, zc, rhoc, pc)
    integer, intent(in) :: igrid
    real, intent(out) :: zc(40), rhoc(40), pc(40)
    real(8) :: dz(40), zm, t, p
    real :: z_full, dz8w
    integer :: k
    do k = 1, 40
      if (igrid == 1) then
        dz(k) = 100.0d0
      else
        dz(k) = 50.0d0*1.025d0**(k-1)
      end if
    end do
    z_full = 0.
    zm = 0.0d0
    do k = 1, 40
      dz8w = real(dz(k))
      zc(k) = z_full+0.5*dz8w*100
      z_full = z_full+dz8w*100.
      zm = zm + 0.5d0*dz(k)
      t = 300.0d0 - 0.0055d0*zm
      p = 1.0d5*(t/300.0d0)**(9.81d0/(287.0d0*0.0055d0))
      pc(k) = real(p*10.0d0)
      rhoc(k) = real(p/(287.0d0*t)*1.0d-3)
      zm = zm + 0.5d0*dz(k)
    end do
  end subroutine falflux_grid

  subroutine sweep_falflux()
    integer :: ig, ip, id, k
    real :: zc(40), rhoc(40), pc(40), vrx(40, nb), ffx(40, nb), ffx_in(40, nb), vz(nb), dts(2)
    real(8) :: ffd(nb), ffd2(nb)
    dts = (/ 12.0, 60.0 /)
    do ig = 1, 2
      call falflux_grid(ig, zc, rhoc, pc)
      do k = 1, 40
        call fall_speeds(pc(k), vz)
        vrx(k, :) = vz
      end do
      do ip = 1, 2
        do id = 1, 2
          ffx = 0.0
          if (ip == 1) then
            call spectrum(3, ffd)
            do k = 1, 40
              if (zc(k) > 1.2e5) ffx(k, :) = real(ffd)/rhoc(k)
            end do
          else
            call spectrum(5, ffd)
            call spectrum(1, ffd2)
            do k = 10, 25
              ffx(k, :) = real(ffd)/rhoc(k)
            end do
            do k = 26, 30
              ffx(k, :) = real(ffd2)/rhoc(k)
            end do
          end if
          ffx_in = ffx
          call falfluxhucm_z(ffx, vrx, rhoc, pc, zc, dts(id), 1, 40, nb)
          call fsbm_begin(s, 'case')
          call fsbm_put(s, 'grid', int(ig, 4))
          call fsbm_put(s, 'profile', int(ip, 4))
          call fsbm_put(s, 'dt', dts(id))
          call fsbm_put(s, 'zcgs', zc)
          call fsbm_put(s, 'rhocgs', rhoc)
          call fsbm_put(s, 'pcgs', pc)
          call fsbm_put(s, 'vrx', vrx)
          call fsbm_put(s, 'ffx_in', ffx_in)
          call fsbm_put(s, 'ffx_out', ffx)
          call fsbm_end(s)
        end do
      end do
    end do
  end subroutine sweep_falflux

  ! ----------------------------------------------------------------- refusals
  subroutine refusal_case(which)
    character(len=*), intent(in) :: which
    integer :: n
    real(8) :: ffd(nb), tt, qq, pp, del1n, del2n, div1, div2, sup1, sup2, xi_d(nb), xin(nb), tpn
    real(8) :: fls(nb), flg(nb), flh(nb), frs(nb)
    real :: pcgs, ror, vz(nb), ff1(nb), psi1(nb), dtcond, w_in, dx_in, b21(nb), fi(nb), &
            psi(nb), fl2(nb), one, ff2r(nb, 3), ff3r(nb), ff4r(nb), ff5r(nb), colleff, &
            tcrit, ttcoal, dtc, psi2(nb, 3), fccnr(na), sup2_old, dsup, win
    real :: zc(40), rhoc(40), pc(40), vrx(40, nb), ffx(40, nb), dtf
    integer :: isym1, isym2(3), isym3, isym4, isym5, idrop
    read(which, *) n
    call fsbm_open(s, trim(outdir)//'/refusal_'//trim(which))
    select case (n)
    case (1, 2)
      ! ONECOND1 temperature jump over 3 K in one substep (:7070): a dense,
      ! large-droplet cloud at 50 percent supersaturation (case 1) or at
      ! 10 percent relative humidity (case 2).
      tt = 295.15d0
      pcgs = 900.0*1000.0
      pp = pcgs
      if (n == 1) then
        call thermo(tt, 0.5d0, pp, qq, del1n, del2n, div1, div2)
      else
        call thermo(tt, -0.9d0, pp, qq, del1n, del2n, div1, div2)
      end if
      ror = air_density_cgs(pcgs, tt)
      call fall_speeds(pcgs, vz)
      ffd = 0.0d0
      call add_lognormal(2000.0d0, 12.0d-4, 1.2d0, ffd)
      ff1 = real(ffd)
      psi1 = 0.0
      dtcond = 4.0
      w_in = 0.0
      dx_in = 1000.0
      isym1 = 1; isym2 = 0; isym3 = 0; isym4 = 0; isym5 = 0
      call fsbm_begin(s, 'onecond1')
      call fsbm_put(s, 'tt', tt)
      call fsbm_put(s, 'qq', qq)
      call fsbm_put(s, 'pp', pp)
      call fsbm_put(s, 'ror', ror)
      call fsbm_put(s, 'del1n', del1n)
      call fsbm_put(s, 'del2n', del2n)
      call fsbm_put(s, 'div1', div1)
      call fsbm_put(s, 'div2', div2)
      call fsbm_put(s, 'vr1', vz)
      call fsbm_put(s, 'dtcond', dtcond)
      call fsbm_put(s, 'ff1', ff1)
      call fsbm_end(s)
      call fsbm_close(s)
      call announce(n, 'ONECOND1')
      call onecond1(tt, qq, pp, ror, vz, pcgs, del1n, del2n, div1, div2, &
                    ff1, psi1, xl, rlec, ro1bl, AA1_MY, BB1_MY, AA2_MY, BB2_MY, &
                    c1_mey, c2_mey, col, dtcond, 3, nb, isym1, isym2, isym3, &
                    isym4, isym5, 1, 1, 1, w_in, dx_in, 2)
    case (3)
      ! JERNEWF_KS negative spectrum (:2750): a negative bin with no growth.
      call spectrum(1, ffd)
      fi = real(ffd)
      fi(5) = -1.0
      xi_d = xl
      b21 = 0.0
      one = 1.0
      psi = 0.0
      fl2 = 0.0
      xin = 0.0d0
      idrop = 1
      tpn = 288.15d0
      call fsbm_begin(s, 'jernewf')
      call fsbm_put(s, 'rr', xi_d)
      call fsbm_put(s, 'fi', fi)
      call fsbm_put(s, 'b21', b21)
      call fsbm_put(s, 'del2n', one)
      call fsbm_put(s, 'idrop', int(idrop, 4))
      call fsbm_end(s)
      call fsbm_close(s)
      call announce(n, 'JERDFUN_KS/JERNEWF_KS')
      call jerdfun_ks(xi_d, xin, b21, fi, psi, fl2, one, 1, 2, 1, tpn, idrop, &
                      fr_lim_z, frh_lim_z, 3, nb, col, 1, 1, 1, 1, 1)
    case (4)
      ! Negative drops after coalescence (:8610): a negative bin that no
      ! collision pair targets survives to the final check.
      call spectrum(1, ffd)
      ff1 = real(ffd)
      ff1(20) = -1.0e-3
      ff2r = 0.0; ff3r = 0.0; ff4r = 0.0; ff5r = 0.0
      fls = 0.0d0; flg = 0.0d0; flh = 0.0d0; frs = 0.0d0
      tt = 288.15d0
      qq = 0.010d0
      pp = 1.0d6
      del1n = 1.0d-3
      del2n = 5.0d-2
      ror = air_density_cgs(1.0e6, tt)
      dtc = 12.0
      tcrit = 2.7015E02
      ttcoal = 2.3315E02
      colleff = 0.0
      call fsbm_begin(s, 'coal_bott')
      call fsbm_put(s, 'ff1', ff1)
      call fsbm_put(s, 'tt', tt)
      call fsbm_put(s, 'qq', qq)
      call fsbm_put(s, 'pp', pp)
      call fsbm_put(s, 'rho', ror)
      call fsbm_put(s, 'dt_coll', dtc)
      call fsbm_end(s)
      call fsbm_close(s)
      call announce(n, 'COAL_BOTT_NEW')
      call coal_bott_new(ff1, ff2r, ff3r, ff4r, ff5r, tt, qq, pp, ror, dtc, tcrit, &
                         ttcoal, fls, flg, flh, frs, del1n, del2n, 1, 1, 1, 1, colleff)
    case (5)
      ! Cloud-base activation with no solution (:3548): aerosol in every bin
      ! up to 2 um and a vanishing updraft.
      tt = 290.15d0
      pcgs = 900.0*1000.0
      pp = pcgs
      call thermo(tt, 1.0d-2, pp, qq, sup1, sup2, div1, div2)
      ror = air_density_cgs(pcgs, tt)
      psi1 = 0.0
      psi2 = 0.0
      fccnr = 100.0
      sup2_old = real(sup2)
      dsup = 0.0
      win = 1.0e-4
      call fsbm_begin(s, 'nucleation')
      call fsbm_put(s, 'tt', tt)
      call fsbm_put(s, 'qq', qq)
      call fsbm_put(s, 'sup1', sup1)
      call fsbm_put(s, 'sup2', sup2)
      call fsbm_put(s, 'ror', ror)
      call fsbm_put(s, 'pp', pcgs)
      call fsbm_put(s, 'fccnr', fccnr)
      call fsbm_put(s, 'win', win)
      call fsbm_put(s, 'is_cloud_base', int(1, 4))
      call fsbm_end(s)
      call fsbm_close(s)
      call announce(n, 'JERNUCL01_KS/Cloud_Base_Super')
      call jernucl01_ks(psi1, psi2, fccnr, xl, xi, tt, qq, ror, pcgs, sup1, sup2, &
                        col, sup2_old, dsup, rccn, dropradii, nb, na, 3, 1, &
                        win, 1, ro_solute, ions, mwaero, 1, 1, 1)
    case (6)
      ! FALFLUXHUCM_Z with zero fall speeds (:5345): a bare STOP (exit
      ! status 0, no message) rather than wrf_error_fatal.
      call falflux_grid(2, zc, rhoc, pc)
      vrx = 0.0
      ffx = 0.0
      ffx(20, 15) = 1.0
      dtf = 12.0
      call fsbm_begin(s, 'falflux')
      call fsbm_put(s, 'zcgs', zc)
      call fsbm_put(s, 'rhocgs', rhoc)
      call fsbm_put(s, 'pcgs', pc)
      call fsbm_put(s, 'vrx', vrx)
      call fsbm_put(s, 'ffx', ffx)
      call fsbm_put(s, 'dt', dtf)
      call fsbm_end(s)
      call fsbm_close(s)
      call announce(n, 'FALFLUXHUCM_Z')
      call falfluxhucm_z(ffx, vrx, rhoc, pc, zc, dtf, 1, 40, nb)
    case default
      call fsbm_fail('unknown refusal case')
    end select
    ! Reaching this line means the routine returned: the refusal did not fire.
    write(0,'(a,i0)') 'FSBM_REFUSAL_RETURNED ', n
    error stop 3
  end subroutine refusal_case

  subroutine announce(n, name)
    integer, intent(in) :: n
    character(len=*), intent(in) :: name
    write(0,'(a,i0,1x,a)') 'FSBM_REFUSAL_CALLING ', n, name
    flush(0)
  end subroutine announce


  ! ==================================================================
  ! Milestone 2: the cold families and the cold refusal cases.
  !
  ! The cases carried from the 09-13/14 port keep its grids (temperatures,
  ! time steps, habit boundaries, supersaturation pairs, seeds, ranges and
  ! the y = 1.25 x orientation), on the real tables. Its draws came from
  ! numpy's PCG64 in Python; here they come from the Park-Miller minimal
  ! standard generator (x <- 16807 x mod (2^31 - 1), started at the old seed
  ! + 1), so the grids carry over and the drawn values are this harness's.
  ! Every draw is written into the records that use it.
  ! ==================================================================

  subroutine pm_seed(seed)
    integer, intent(in) :: seed
    pm_state = mod(int(seed, 8), 2147483646_8) + 1_8
  end subroutine pm_seed

  real(8) function pm_uniform(a, b)
    real(8), intent(in) :: a, b
    pm_state = mod(16807_8*pm_state, 2147483647_8)
    pm_uniform = a + (b - a)*real(pm_state, 8)/2147483647.0d0
  end function pm_uniform

  ! Exponential spectrum n0 exp(-lam d) (n0 cm-4, lam cm-1, d the bin's
  ! equal-mass drop diameter) over bins kmin..kmax, in the scheme's f-form.
  ! The five mass grids are equal, so this serves every species.
  subroutine add_expo(n0, lam, kmin, kmax, ff)
    real(8), intent(in) :: n0, lam
    integer, intent(in) :: kmin, kmax
    real(8), intent(inout) :: ff(nb)
    integer :: k
    real(8) :: d, nk
    do k = kmin, kmax
      d = 2.0d0*drop_radius(k)
      nk = n0*exp(-lam*d)*d*real(col, 8)
      ff(k) = ff(k) + nk/(3.0d0*real(col, 8)*real(xl(k), 8))
    end do
  end subroutine add_expo

  ! Ice spectrum catalogue (f-form, double).
  subroutine ice_spectrum(id, ff)
    integer, intent(in) :: id
    real(8), intent(out) :: ff(nb)
    ff = 0.0d0
    select case (id)
    case (1)   ! small ice, bins 1-18 (reported as QI)
      call add_expo(2.0d0, 150.0d0, 1, 18, ff)
    case (2)   ! snow over every bin
      call add_expo(5.0d-2, 20.0d0, 1, nb, ff)
    case (3)   ! graupel-sized dense ice, bins 10-28
      call add_expo(4.0d-3, 12.0d0, 10, 28, ff)
    case (4)   ! hail, bins 16-33
      call add_expo(2.0d-3, 6.0d0, 16, nb, ff)
    case (5)   ! dense snow, twenty times case 2
      call add_expo(1.0d0, 20.0d0, 1, nb, ff)
    case (6)   ! dilute snow, a thousandth of case 2
      call add_expo(5.0d-5, 20.0d0, 1, nb, ff)
    case default
      call fsbm_fail('unknown ice spectrum id')
    end select
  end subroutine ice_spectrum

  ! State for a target ice supersaturation, with FAST_SBM's expressions.
  subroutine thermo_ice(t, sat_i, pp, qq, del1n, del2n, div1, div2)
    real(8), intent(in) :: t, sat_i, pp
    real(8), intent(out) :: qq, del1n, del2n, div1, div2
    real(8) :: es1n, es2n, ew1n, e
    es2n = AA2_MY*DEXP(-BB2_MY/t)
    e = (1.0d0 + sat_i)*es2n
    qq = 0.622d0*e/(pp - 0.378d0*e)
    es1n = AA1_MY*DEXP(-BB1_MY/t)
    es2n = AA2_MY*DEXP(-BB2_MY/t)
    ew1n = qq*pp/(0.622+0.378*qq)
    div1 = ew1n/es1n
    del1n = ew1n/es1n-1.
    div2 = ew1n/es2n
    del2n = ew1n/es2n-1.
  end subroutine thermo_ice

  subroutine reset_cold_capture()
    capc_icenucl_calls = 0; capc_icenucl_band = 0; capc_icenucl_added = 0
    capc_icenucl_psi2_in = 0.0d0; capc_icenucl_psi2_out = 0.0d0
    capc_icenucl_scal_in = 0.0d0; capc_icenucl_scal_out = 0.0d0
    capc_homfrz_calls = 0; capc_homfrz = 0.0d0
    capc_supsat_n = 0; capc_supsat_site = 0; capc_supsat_kcond = 0
    capc_supsat_branch = -1; capc_supsat_clamp = 0; capc_supsat_rates = 0.0d0
    capc_cc_mask = 0; capc_cold_block = 0; capc_hm_fired = 0; capc_cold_icol = 0
    capc_hm = 0.0d0; capc_rime = 0.0d0; capc_cold_g_in = 0.0d0; capc_cold_g_out = 0.0d0
    capc_cold_rf3_out = 0.0d0; capc_cold_scal = 0.0d0
  end subroutine reset_cold_capture

  ! The seven phase-change species on their mass grids, in the layout of the
  ! old binding: liquid, the three crystal habits, snow, graupel, hail.
  subroutine phase_grids(x7)
    real, intent(out) :: x7(nb, 7)
    x7(:, 1) = xl
    x7(:, 2:4) = xi
    x7(:, 5) = xs
    x7(:, 6) = xg
    x7(:, 7) = xh
  end subroutine phase_grids

  ! The old port's phase-change draws: uniform(0, 1e-3) for 7 species x 33
  ! bins from seed 6703, species by species (numpy's C order).
  subroutine old_phase_draws(f7)
    real, intent(out) :: f7(nb, 7)
    integer :: sp, k
    call pm_seed(6703)
    do sp = 1, 7
      do k = 1, nb
        f7(k, sp) = real(pm_uniform(0.0d0, 1.0d-3))
      end do
    end do
  end subroutine old_phase_draws

  ! One FREEZ (op 1) or J_W_MELT (op 2) call on the seven species.
  subroutine phase_case(op, c, set, f7in, tin0, dtc, ro)
    integer, intent(in) :: op, c, set
    real, intent(in) :: f7in(nb, 7), tin0, dtc, ro
    real :: f(nb, 7), t, x7(nb, 7), gl(nb), gi(nb, icemax), gs(nb), gg(nb), gh(nb)
    real :: af, bf, bmax
    integer :: kfreez, unchanged
    af = 0.3333E-04
    bf = 0.6600E00
    bmax = 0.6600E00
    kfreez = 21
    call phase_grids(x7)
    gl = xl; gi = xi; gs = xs; gg = xg; gh = xh
    f = f7in
    t = tin0
    if (op == 1) then
      call freez(f(:, 1), gl, f(:, 2:4), gi, f(:, 5), gs, f(:, 6), gg, f(:, 7), gh, &
                 t, dtc, ro, col, af, bf, bmax, kfreez, icemax, nb)
    else
      call j_w_melt(f(:, 1), gl, f(:, 2:4), gi, f(:, 5), gs, f(:, 6), gg, f(:, 7), gh, &
                    t, dtc, ro, col, icemax, nb)
    end if
    unchanged = 0
    if (all(gl == xl) .and. all(gi == xi) .and. all(gs == xs) .and. all(gg == xg) &
        .and. all(gh == xh)) unchanged = 1
    call fsbm_begin(s, 'case')
    call fsbm_put(s, 'case_id', int(c, 4))
    call fsbm_put(s, 'set', int(set, 4))
    call fsbm_put(s, 't_in', tin0)
    call fsbm_put(s, 'dt', dtc)
    call fsbm_put(s, 'rho', ro)
    call fsbm_put(s, 'ff_in', f7in)
    call fsbm_put(s, 'ff_out', f)
    call fsbm_put(s, 't_out', t)
    call fsbm_put(s, 'grids_unchanged', int(unchanged, 4))
    call fsbm_end(s)
  end subroutine phase_case

  ! FREEZ and J_W_MELT. set 1: the old grid, 5 temperatures x 4 time steps on
  ! the old draws. set 2 (FREEZ): the plate-to-hail split at KRFREEZ, no
  ! drops, and physical cloud and rain spectra. set 2 (J_W_MELT): every rate
  ! band of every species, the old dt = 200 s case that drives a snow bin
  ! negative (finding D12), and physical snow and hail.
  subroutine sweep_phase(op)
    integer, intent(in) :: op
    real :: x7(nb, 7), f7(nb, 7), g7(nb, 7), temps(5), dts(4)
    real(8) :: ffd(nb)
    integer :: it, id, c, sp
    call phase_grids(x7)
    call fsbm_begin(s, 'grid')
    call fsbm_put(s, 'masses', x7)
    call fsbm_put(s, 'col', col)
    call fsbm_end(s)
    temps = (/ 233.15, 250.0, 273.15, 274.0, 290.0 /)
    dts = (/ 0.0, 1.0, 20.0, 200.0 /)
    call old_phase_draws(f7)
    c = 0
    do it = 1, 5
      do id = 1, 4
        c = c + 1
        call phase_case(op, c, 1, f7, temps(it), dts(id), 0.001)
      end do
    end do
    if (op == 1) then
      ! drops only in bins 18-26, either side of KRFREEZ = 21
      g7 = 0.0
      g7(18:26, 1) = 1.0e-3
      c = c + 1; call phase_case(op, c, 2, g7, 250.0, 12.0, 0.001)
      c = c + 1; call phase_case(op, c, 2, g7, 263.15, 60.0, 0.001)
      ! no drops: F1_MAX = 0 skips freezing
      g7 = f7
      g7(:, 1) = 0.0
      c = c + 1; call phase_case(op, c, 2, g7, 250.0, 12.0, 0.001)
      ! cloud plus rain at three temperatures
      call spectrum(2, ffd)
      g7 = 0.0
      g7(:, 1) = real(ffd)
      c = c + 1; call phase_case(op, c, 2, g7, 268.15, 12.0, 0.0008)
      c = c + 1; call phase_case(op, c, 2, g7, 253.15, 12.0, 0.0007)
      c = c + 1; call phase_case(op, c, 2, g7, 238.15, 12.0, 0.0006)
    else
      ! each species in each rate band, alone
      do sp = 2, 7
        g7 = 0.0
        g7(:, sp) = 1.0e-3
        c = c + 1; call phase_case(op, c, 2, g7, 275.0, 12.0, 0.001)
      end do
      ! the old dt = 200 s case: snow bin 16 (band 15-21) at 280 K
      g7 = 0.0
      g7(16, 5) = 1.0
      c = c + 1; call phase_case(op, c, 2, g7, 280.0, 200.0, 0.001)
      ! the band edges at the step where a band's rate reaches 1 (D12)
      g7 = 0.0
      g7(:, 5) = 1.0e-3
      g7(:, 7) = 1.0e-3
      c = c + 1; call phase_case(op, c, 2, g7, 273.2, 100.0, 0.001)
      c = c + 1; call phase_case(op, c, 2, g7, 273.2, 176.0, 0.001)
      c = c + 1; call phase_case(op, c, 2, g7, 273.2, 180.0, 0.001)
      ! physical snow and hail, with drops, at 0 C exactly and just above
      g7 = 0.0
      call ice_spectrum(2, ffd)
      g7(:, 5) = real(ffd)
      call ice_spectrum(4, ffd)
      g7(:, 7) = real(ffd)
      call spectrum(1, ffd)
      g7(:, 1) = real(ffd)
      c = c + 1; call phase_case(op, c, 2, g7, 273.15, 12.0, 0.001)
      c = c + 1; call phase_case(op, c, 2, g7, 274.15, 12.0, 0.001)
      c = c + 1; call phase_case(op, c, 2, g7, 280.15, 12.0, 0.001)
    end if
  end subroutine sweep_phase

  ! One JERNUCL01_KS call with ICEPROCS = 1, captured inside through the probe
  ! build: ICE_NUCL's own inputs and outputs, its band, whether it added
  ! crystals, and the homogeneous freezing.
  subroutine nucl_ice_case(c, set, tt0, qq0, sup10, sup20, sup2o, dsup, win, is_cb, &
                           psi1_in, psi2_in, fccnr_in, xi_c, ror, pcgs)
    integer, intent(in) :: c, set, is_cb
    real(8), intent(in) :: tt0, qq0, sup10, sup20
    real, intent(in) :: sup2o, dsup, win, psi1_in(nb), psi2_in(nb, 3), fccnr_in(na), &
                        xi_c(nb, 3), ror, pcgs
    real(8) :: tt, qq, sup1, sup2
    real :: psi1(nb), psi2(nb, 3), fccnr(na)
    tt = tt0; qq = qq0; sup1 = sup10; sup2 = sup20
    psi1 = psi1_in; psi2 = psi2_in; fccnr = fccnr_in
    call reset_cold_capture()
    call jernucl01_ks(psi1, psi2, fccnr, xl, xi_c, tt, qq, ror, pcgs, sup1, sup2, &
                      col, sup2o, dsup, rccn, dropradii, nb, na, 3, 1, &
                      win, is_cb, ro_solute, ions, mwaero, 1, 1, 1)
    call fsbm_begin(s, 'case')
    call fsbm_put(s, 'case_id', int(c, 4))
    call fsbm_put(s, 'set', int(set, 4))
    call fsbm_put(s, 'is_cloud_base', int(is_cb, 4))
    call fsbm_put(s, 'win', win)
    call fsbm_put(s, 'tt_in', tt0)
    call fsbm_put(s, 'qq_in', qq0)
    call fsbm_put(s, 'sup1_in', sup10)
    call fsbm_put(s, 'sup2_in', sup20)
    call fsbm_put(s, 'sup2_old', sup2o)
    call fsbm_put(s, 'dsupice_xyz', dsup)
    call fsbm_put(s, 'ror', ror)
    call fsbm_put(s, 'pp', pcgs)
    call fsbm_put(s, 'xi', xi_c)
    call fsbm_put(s, 'psi1_in', psi1_in)
    call fsbm_put(s, 'psi2_in', psi2_in)
    call fsbm_put(s, 'fccnr_in', fccnr_in)
    call fsbm_put(s, 'tt_out', tt)
    call fsbm_put(s, 'qq_out', qq)
    call fsbm_put(s, 'sup1_out', sup1)
    call fsbm_put(s, 'sup2_out', sup2)
    call fsbm_put(s, 'psi1_out', psi1)
    call fsbm_put(s, 'psi2_out', psi2)
    call fsbm_put(s, 'fccnr_out', fccnr)
    call fsbm_put(s, 'ice_nucl_calls', capc_icenucl_calls)
    call fsbm_put(s, 'ice_nucl_band', capc_icenucl_band)
    call fsbm_put(s, 'ice_nucl_added', capc_icenucl_added)
    call fsbm_put(s, 'ice_nucl_psi2_in', capc_icenucl_psi2_in)
    call fsbm_put(s, 'ice_nucl_psi2_out', capc_icenucl_psi2_out)
    call fsbm_put(s, 'ice_nucl_scalars_in', capc_icenucl_scal_in)
    call fsbm_put(s, 'ice_nucl_scalars_out', capc_icenucl_scal_out)
    call fsbm_put(s, 'homfrz_calls', capc_homfrz_calls)
    call fsbm_put(s, 'homfrz_scalars', capc_homfrz)
    call fsbm_end(s)
  end subroutine nucl_ice_case

  ! ICE_NUCL, reached through JERNUCL01_KS with the water supersaturation
  ! below zero, so no drop activates and nothing freezes homogeneously
  ! (ICE_NUCL is private to module_mp_SBM_Nucleation; the probe records its
  ! own call). set 1: the old grid, the habit and Meyers boundaries -1, -4,
  ! -5, -8.1, -12.7, -17.8, -22.4, -30 and -37 C at SUP2_OLD 0 and 0.12,
  ! SUP2 0.1 and DSUPICE_XYZ 0.005. set 2: either side of each boundary,
  ! the -37 C limit, the DELSUPICE_MAX clamps on DEL2N and DSUP2N, a
  ! falling supersaturation, and crystals already above HELEK1.
  subroutine sweep_ice_nucl()
    real(8) :: tcs(9), sup2olds(2), edges(7), tc
    real :: psi1(nb), psi2(nb, 3), fccnr(na), xi_c(nb, 3)
    integer :: it, io, c, ie
    tcs = (/ -1.0d0, -4.0d0, -5.0d0, -8.1d0, -12.7d0, -17.8d0, -22.4d0, -30.0d0, -37.0d0 /)
    sup2olds = (/ 0.0d0, 0.12d0 /)
    edges = (/ -4.0d0, -5.0d0, -8.1d0, -12.7d0, -17.8d0, -22.4d0, -37.0d0 /)
    psi1 = 0.0
    psi2 = 0.0
    fccnr = 0.0
    xi_c = xi
    pr_capture = .true.
    c = 0
    do it = 1, 9
      do io = 1, 2
        c = c + 1
        call nucl_ice_case(c, 1, 273.15d0 + tcs(it), 0.004d0, -1.0d-2, 0.1d0, &
                           real(sup2olds(io)), 0.005, 200.0, 0, psi1, psi2, fccnr, xi_c, &
                           0.001, 8.0e5)
      end do
    end do
    do ie = 1, 7
      do io = -1, 1, 2
        tc = edges(ie) + real(io, 8)*1.0d-6
        c = c + 1
        call nucl_ice_case(c, 2, 273.15d0 + tc, 0.004d0, -1.0d-2, 0.1d0, 0.0, 0.005, &
                           200.0, 0, psi1, psi2, fccnr, xi_c, 0.001, 8.0e5)
      end do
    end do
    ! above 0 C and at -40 C: ICE_NUCL is not called
    c = c + 1
    call nucl_ice_case(c, 2, 273.65d0, 0.004d0, -1.0d-2, 0.1d0, 0.0, 0.005, 200.0, 0, &
                       psi1, psi2, fccnr, xi_c, 0.001, 8.0e5)
    c = c + 1
    call nucl_ice_case(c, 2, 233.15d0, 0.004d0, -1.0d-2, 0.1d0, 0.0, 0.005, 200.0, 0, &
                       psi1, psi2, fccnr, xi_c, 0.001, 8.0e5)
    ! DEL2N above DELSUPICE_MAX (59 percent) and DSUP2N above it
    c = c + 1
    call nucl_ice_case(c, 2, 253.15d0, 0.004d0, -1.0d-2, 0.7d0, 0.0, 0.0, 200.0, 0, &
                       psi1, psi2, fccnr, xi_c, 0.001, 8.0e5)
    c = c + 1
    call nucl_ice_case(c, 2, 253.15d0, 0.004d0, -1.0d-2, 0.2d0, -0.5, 0.0, 200.0, 0, &
                       psi1, psi2, fccnr, xi_c, 0.001, 8.0e5)
    ! supersaturation falling since the last call: DELTACD <= 0
    c = c + 1
    call nucl_ice_case(c, 2, 253.15d0, 0.004d0, -1.0d-2, 0.1d0, 0.15, 0.0, 200.0, 0, &
                       psi1, psi2, fccnr, xi_c, 0.001, 8.0e5)
    ! crystals of the nucleating habit already above HELEK1 (type 3 at -15 C)
    psi2 = 0.0
    psi2(1:5, 3) = 1.0e12
    c = c + 1
    call nucl_ice_case(c, 2, 258.15d0, 0.004d0, -1.0d-2, 0.2d0, 0.0, 0.0, 200.0, 0, &
                       psi1, psi2, fccnr, xi_c, 0.001, 8.0e5)
    ! the same crystals in another habit: nucleation proceeds
    psi2 = 0.0
    psi2(1:5, 1) = 1.0e12
    c = c + 1
    call nucl_ice_case(c, 2, 258.15d0, 0.004d0, -1.0d-2, 0.2d0, 0.0, 0.0, 200.0, 0, &
                       psi1, psi2, fccnr, xi_c, 0.001, 8.0e5)
    pr_capture = .false.
  end subroutine sweep_ice_nucl

  ! JERNUCL01_KS with ice: the old port's 48 cases (8 temperatures x 3
  ! supersaturation pairs x cloud base or not), on the real grids and the
  ! module's aerosol constants: drops in bins 5-10, no ice, a Gaussian
  ! aerosol spectrum 100 exp(-((k - 23)/9)^2), q 0.004, rho 0.001 g/cm3,
  ! p 800 hPa, SUP2_OLD 0.05, DSUPICE_XYZ 0.002, w 200 cm/s.
  subroutine sweep_jernucl01_ice()
    real(8) :: temps(8), sups(2, 3)
    real :: psi1(nb), psi2(nb, 3), fccnr(na), xi_c(nb, 3)
    integer :: it, ip, ib, c, k
    temps = (/ 193.0d0, 220.0d0, 234.0d0, 236.0d0, 245.0d0, 263.0d0, 274.0d0, 290.0d0 /)
    sups(:, 1) = (/ -0.01d0, -0.02d0 /)
    sups(:, 2) = (/ -0.01d0, 0.1d0 /)
    sups(:, 3) = (/ 0.002d0, 0.1d0 /)
    psi1 = 0.0
    psi1(5:10) = 1.0e8
    psi2 = 0.0
    do k = 1, na
      fccnr(k) = real(100.0d0*exp(-((real(k - 1, 8) - 22.0d0)/9.0d0)**2))
    end do
    xi_c = xi
    pr_capture = .true.
    c = 0
    do it = 1, 8
      do ip = 1, 3
        do ib = 0, 1
          c = c + 1
          call nucl_ice_case(c, 1, temps(it), 0.004d0, sups(1, ip), sups(2, ip), 0.05, &
                             0.002, 200.0, ib, psi1, psi2, fccnr, xi_c, 0.001, 8.0e5)
        end do
      end do
    end do
    pr_capture = .false.
  end subroutine sweep_jernucl01_ice

  ! Ice JERRATE_KS: the three crystal habits (ID 3, IN 2), snow (IN 3, whose
  ! nskin rule switches at 273.15 K), graupel and hail (IN 2) with liquid
  ! fraction 0 and 0.5, and snow with IN 6, at 4 temperatures and 3 pressures.
  subroutine sweep_jerrate_ice()
    real(8) :: temps(4), xls(nb, 3), vxl(nb, 3), tp, pp
    real :: plist(3), pcgs, factor_p, cap(nb, 3), ro(nb, 3), b(nb, 3), fl(nb)
    integer :: it, ip, sp, idn, inn
    temps = (/ 238.15d0, 258.15d0, 272.15d0, 274.15d0 /)
    plist = (/ 800.0, 500.0, 300.0 /)
    do ip = 1, 3
      pcgs = plist(ip)*1000.0
      factor_p = DSQRT(1.0D6/pcgs)
      do it = 1, 4
        tp = temps(it)
        pp = pcgs
        do sp = 1, 7
          xls = 0.0d0; vxl = 0.0d0; cap = 0.0; ro = 0.0; b = 0.0; fl = 0.0
          idn = 1
          inn = 2
          select case (sp)
          case (1)   ! crystals, three habits
            idn = 3
            xls = xi
            vxl = vr2*factor_p
            cap = riec
            ro = ro2bl
          case (2)   ! snow
            inn = 3
            xls(:, 1) = xs; vxl(:, 1) = vr3*factor_p; cap(:, 1) = rsec; ro(:, 1) = ro3bl
          case (3, 4)   ! graupel, liquid fraction 0 then 0.5
            xls(:, 1) = xg; vxl(:, 1) = vr4*factor_p; cap(:, 1) = rgec; ro(:, 1) = ro4bl
            if (sp == 4) fl = 0.5
          case (5, 6)   ! hail
            xls(:, 1) = xh; vxl(:, 1) = vr5*factor_p; cap(:, 1) = rhec; ro(:, 1) = ro5bl
            if (sp == 6) fl = 0.5
          case (7)   ! snow with IN 6
            inn = 6
            xls(:, 1) = xs; vxl(:, 1) = vr3*factor_p; cap(:, 1) = rsec; ro(:, 1) = ro3bl
          end select
          call jerrate_ks(xls(:, 1:idn), tp, pp, vxl(:, 1:idn), cap(:, 1:idn), ro(:, 1:idn), &
                          b(:, 1:idn), idn, inn, fl, nb, 3)
          call fsbm_begin(s, 'case')
          call fsbm_put(s, 'species', int(sp, 4))
          call fsbm_put(s, 'id', int(idn, 4))
          call fsbm_put(s, 'in', int(inn, 4))
          call fsbm_put(s, 'tp', tp)
          call fsbm_put(s, 'pp', pp)
          call fsbm_put(s, 'fl', fl)
          call fsbm_put(s, 'xls', xls)
          call fsbm_put(s, 'vxl', vxl)
          call fsbm_put(s, 'capacity', cap)
          call fsbm_put(s, 'bulk_density', ro)
          call fsbm_put(s, 'b', b)
          call fsbm_end(s)
        end do
      end do
    end do
  end subroutine sweep_jerrate_ice

  ! Ice JERTIMESC_KS: the crystal habits (ID 3) and snow, graupel and hail
  ! (ID 1) on rates from JERRATE_KS at 258.15 K and 500 hPa. Spectra: the old
  ! port's uniform(0, 100) draws (seeds 431-433, one per ID of its grid) and
  ! physical ice spectra; CF is 1/ROR for air of 0.7 kg/m3.
  subroutine sweep_jertimesc_ice()
    real(8) :: xls(nb, 3), vxl(nb, 3), ffd(nb), tp, pp
    real :: pcgs, factor_p, cap(nb, 3), ro(nb, 3), b(nb, 3), fl(nb), fi(nb, 3), sfn(3), cf
    integer :: sp, idn, inn, draw, k, m
    pcgs = 500.0*1000.0
    factor_p = DSQRT(1.0D6/pcgs)
    tp = 258.15d0
    pp = pcgs
    cf = 1.0/7.0e-4
    fl = 0.0
    do sp = 1, 4
      xls = 0.0d0; vxl = 0.0d0; cap = 0.0; ro = 0.0; b = 0.0
      idn = 1
      inn = 2
      select case (sp)
      case (1)
        idn = 3
        xls = xi; vxl = vr2*factor_p; cap = riec; ro = ro2bl
      case (2)
        inn = 3
        xls(:, 1) = xs; vxl(:, 1) = vr3*factor_p; cap(:, 1) = rsec; ro(:, 1) = ro3bl
      case (3)
        xls(:, 1) = xg; vxl(:, 1) = vr4*factor_p; cap(:, 1) = rgec; ro(:, 1) = ro4bl
      case (4)
        xls(:, 1) = xh; vxl(:, 1) = vr5*factor_p; cap(:, 1) = rhec; ro(:, 1) = ro5bl
      end select
      call jerrate_ks(xls(:, 1:idn), tp, pp, vxl(:, 1:idn), cap(:, 1:idn), ro(:, 1:idn), &
                      b(:, 1:idn), idn, inn, fl, nb, 3)
      do draw = 1, 2
        fi = 0.0
        if (draw == 1) then
          call pm_seed(430 + idn)
          do m = 1, idn
            do k = 1, nb
              fi(k, m) = real(pm_uniform(0.0d0, 100.0d0))
            end do
          end do
        else
          do m = 1, idn
            call ice_spectrum(sp, ffd)
            fi(:, m) = real(ffd)
          end do
        end if
        sfn = 0.0
        call jertimesc_ks(fi(:, 1:idn), xls(:, 1:idn), sfn(1:idn), b(:, 1:idn), cf, idn, &
                          nb, 3, col)
        call fsbm_begin(s, 'case')
        call fsbm_put(s, 'species', int(sp, 4))
        call fsbm_put(s, 'id', int(idn, 4))
        call fsbm_put(s, 'draw', int(draw, 4))
        call fsbm_put(s, 'cf', cf)
        call fsbm_put(s, 'x', xls)
        call fsbm_put(s, 'b', b)
        call fsbm_put(s, 'fi', fi)
        call fsbm_put(s, 'sfn', sfn)
        call fsbm_end(s)
      end do
    end do
  end subroutine sweep_jertimesc_ice

  ! One JERDFUN_KS call on an ice grid.
  subroutine jerdfun_ice_case(c, set, ind, ityp, isym, x, b21, fi, d2n, spec)
    integer, intent(in) :: c, set, ind, ityp, isym, spec
    real(8), intent(in) :: x(nb)
    real, intent(in) :: b21(nb), fi(nb), d2n
    real(8) :: xi_d(nb), xin(nb), tpn
    real :: psi(nb), fl2(nb)
    integer :: idrop
    xi_d = x
    xin = x
    psi = fi
    fl2 = 0.0
    idrop = 0
    tpn = 263.15d0
    call jerdfun_ks(xi_d, xin, b21, fi, psi, fl2, d2n, isym, ind, ityp, tpn, idrop, &
                    fr_lim_z, frh_lim_z, 3, nb, col, ityp, 1, 1, 1, 1)
    if (c < 0) return
    call fsbm_begin(s, 'case')
    call fsbm_put(s, 'case_id', int(c, 4))
    call fsbm_put(s, 'set', int(set, 4))
    call fsbm_put(s, 'ind', int(ind, 4))
    call fsbm_put(s, 'itype', int(ityp, 4))
    call fsbm_put(s, 'isym', int(isym, 4))
    call fsbm_put(s, 'spectrum', int(spec, 4))
    call fsbm_put(s, 'del2n', d2n)
    call fsbm_put(s, 'tpn', tpn)
    call fsbm_put(s, 'xi', x)
    call fsbm_put(s, 'b21', b21)
    call fsbm_put(s, 'fi', fi)
    call fsbm_put(s, 'xin', xin)
    call fsbm_put(s, 'psi', psi)
    call fsbm_put(s, 'idrop_out', int(idrop, 4))
    call fsbm_end(s)
  end subroutine jerdfun_ice_case

  ! Ice JERDFUN_KS (ITYPE 2 to 5 and the crystal habits): the new masses are
  ! x + DEL2N B, not the drops' m^(2/3) law. set 1: crystals (IND 3, ITYPE
  ! 1-3), snow (IND 1, ITYPE 3), graupel (4) and hail (5, which reads
  ! FRH_LIM), at 5 values of DEL2N on physical spectra and rates. set 2: the
  ! old port's (IND 2, ITYPE 3) cases, rate x (-0.001, 0, 0.002), bins 7-23
  ! at 3, ISYM 0 and 1, DEL2N 0.01. set 3: a negative snow bin with growth,
  ! which the Kovetz-Olund remap drops without refusing.
  subroutine sweep_jerdfun_ice()
    real(8) :: xls(nb, 3), vxl(nb, 3), x(nb), ffd(nb), tp, pp, facs(3)
    real :: pcgs, factor_p, cap(nb, 3), ro(nb, 3), b(nb, 3), fl(nb), d2ns(5), fi(nb), b21(nb)
    integer :: sp, idn, inn, m, id, c, specid, ia, ir, ind, ityp
    pcgs = 500.0*1000.0
    factor_p = DSQRT(1.0D6/pcgs)
    tp = 258.15d0
    pp = pcgs
    fl = 0.0
    d2ns = (/ 0.05, 0.5, -0.05, -0.5, 0.0 /)
    c = 0
    do sp = 1, 4
      xls = 0.0d0; vxl = 0.0d0; cap = 0.0; ro = 0.0; b = 0.0
      idn = 1
      inn = 2
      select case (sp)
      case (1)
        idn = 3
        xls = xi; vxl = vr2*factor_p; cap = riec; ro = ro2bl
      case (2)
        inn = 3
        xls(:, 1) = xs; vxl(:, 1) = vr3*factor_p; cap(:, 1) = rsec; ro(:, 1) = ro3bl
      case (3)
        xls(:, 1) = xg; vxl(:, 1) = vr4*factor_p; cap(:, 1) = rgec; ro(:, 1) = ro4bl
      case (4)
        xls(:, 1) = xh; vxl(:, 1) = vr5*factor_p; cap(:, 1) = rhec; ro(:, 1) = ro5bl
      end select
      call jerrate_ks(xls(:, 1:idn), tp, pp, vxl(:, 1:idn), cap(:, 1:idn), ro(:, 1:idn), &
                      b(:, 1:idn), idn, inn, fl, nb, 3)
      do m = 1, idn
        if (sp == 1) then
          ind = 3; ityp = m; specid = 1
        else
          ind = 1; ityp = sp + 1
          specid = sp
        end if
        call ice_spectrum(specid, ffd)
        fi = real(ffd)
        x = xls(:, m)
        do id = 1, 5
          b21 = b(:, m)
          c = c + 1
          call jerdfun_ice_case(c, 1, ind, ityp, 1, x, b21, fi, d2ns(id), specid)
        end do
      end do
    end do
    ! the old (IND 2, ITYPE 3) cases on the snow grid
    facs = (/ -0.001d0, 0.0d0, 0.002d0 /)
    x = xs
    fi = 0.0
    fi(7:23) = 3.0
    do ir = 1, 3
      do ia = 0, 1
        b21 = real(x*facs(ir))
        c = c + 1
        call jerdfun_ice_case(c, 2, 2, 3, ia, x, b21, fi, 0.01, 0)
      end do
    end do
    ! a negative snow bin with growth
    call ice_spectrum(2, ffd)
    fi = real(ffd)
    fi(12) = -1.0
    b21 = real(x*1.0d-3)
    c = c + 1
    call jerdfun_ice_case(c, 3, 1, 3, 1, x, b21, fi, 0.05, 2)
  end subroutine sweep_jerdfun_ice

  ! Pressure-corrected ice fall speeds, as FAST_SBM builds VR2_Z..VR5_Z.
  subroutine ice_fall_speeds(pcgs, v2, v3, v4, v5)
    real, intent(in) :: pcgs
    real, intent(out) :: v2(nb, 3), v3(nb), v4(nb), v5(nb)
    real :: factor_p
    factor_p = DSQRT(1.0D6/pcgs)
    v2 = vr2*factor_p
    v3 = vr3*factor_p
    v4 = vr4*factor_p
    v5 = vr5*factor_p
  end subroutine ice_fall_speeds

  subroutine isym_of(f1, f2, f3, f4, f5, isym)
    real, intent(in) :: f1(nb), f2(nb, 3), f3(nb), f4(nb), f5(nb)
    integer, intent(out) :: isym(7)
    ! FAST_SBM :4696-4709: [ISYM1, ISYM2(1:3), ISYM3, ISYM4, ISYM5]
    isym = 0
    if (maxval(f1) > 0) isym(1) = 1
    if (maxval(f2(:, 1)) > 1.E-10) isym(2) = 1
    if (maxval(f2(:, 2)) > 1.E-10) isym(3) = 1
    if (maxval(f2(:, 3)) > 1.E-10) isym(4) = 1
    if (maxval(f3) > 1.E-10) isym(5) = 1
    if (maxval(f4) > 1.E-10) isym(6) = 1
    if (maxval(f5) > 1.E-10) isym(7) = 1
  end subroutine isym_of

  ! One ONECOND2 (op 2) or ONECOND3 (op 3) call. The spectra are passed as
  ! FAST_SBM passes them: FF and its copy PSI (FF2R and FF2IN) equal.
  subroutine cond_case(op, c, set, tt0, pp0, d1, d2, dv1, dv2, qq0, dtcond, &
                       f1, f2, f3, f4, f5)
    integer, intent(in) :: op, c, set
    real(8), intent(in) :: tt0, pp0, d1, d2, dv1, dv2, qq0
    real, intent(in) :: dtcond, f1(nb), f2(nb, 3), f3(nb), f4(nb), f5(nb)
    real(8) :: tt, qq, pp, del1n, del2n, div1, div2
    real :: ff1(nb), ff2(nb, 3), ff3(nb), ff4(nb), ff5(nb), p1(nb), p2(nb, 3), p3(nb), &
            p4(nb), p5(nb), v1(nb), v2(nb, 3), v3(nb), v4(nb), v5(nb), pcgs, ror, w_in, dx_in
    integer :: isym(7), isym2(3)
    tt = tt0; qq = qq0; pp = pp0; del1n = d1; del2n = d2; div1 = dv1; div2 = dv2
    pcgs = real(pp0)
    ror = air_density_cgs(pcgs, tt0)
    call fall_speeds(pcgs, v1)
    call ice_fall_speeds(pcgs, v2, v3, v4, v5)
    ff1 = f1; ff2 = f2; ff3 = f3; ff4 = f4; ff5 = f5
    p1 = f1; p2 = f2; p3 = f3; p4 = f4; p5 = f5
    call isym_of(f1, f2, f3, f4, f5, isym)
    isym2 = isym(2:4)
    w_in = 0.0
    dx_in = 1000.0
    call reset_cold_capture()
    if (op == 2) then
      call onecond2(tt, qq, pp, ror, v2, v3, v4, v5, pcgs, del1n, del2n, div1, div2, &
                    ff2, p2, xi, riec, ro2bl, ff3, p3, xs, rsec, ro3bl, &
                    ff4, p4, xg, rgec, ro4bl, ff5, p5, xh, rhec, ro5bl, &
                    AA1_MY, BB1_MY, AA2_MY, BB2_MY, c1_mey, c2_mey, col, dtcond, icemax, nb, &
                    isym(1), isym2, isym(5), isym(6), isym(7), 1, 1, 1, w_in, dx_in, 2)
    else
      call onecond3(tt, qq, pp, ror, v1, v2, v3, v4, v5, pcgs, del1n, del2n, div1, div2, &
                    ff1, p1, xl, rlec, ro1bl, ff2, p2, xi, riec, ro2bl, ff3, p3, xs, rsec, &
                    ro3bl, ff4, p4, xg, rgec, ro4bl, ff5, p5, xh, rhec, ro5bl, &
                    AA1_MY, BB1_MY, AA2_MY, BB2_MY, c1_mey, c2_mey, col, dtcond, icemax, nb, &
                    isym(1), isym2, isym(5), isym(6), isym(7), 1, 1, 1, w_in, dx_in, 2)
    end if
    if (c < 0) return
    call fsbm_begin(s, 'case')
    call fsbm_put(s, 'case_id', int(c, 4))
    call fsbm_put(s, 'set', int(set, 4))
    call fsbm_put(s, 'tt_in', tt0)
    call fsbm_put(s, 'qq_in', qq0)
    call fsbm_put(s, 'pp', pp0)
    call fsbm_put(s, 'ror', ror)
    call fsbm_put(s, 'del1n_in', d1)
    call fsbm_put(s, 'del2n_in', d2)
    call fsbm_put(s, 'div1', dv1)
    call fsbm_put(s, 'div2', dv2)
    call fsbm_put(s, 'dtcond', dtcond)
    call fsbm_put(s, 'isym', int(isym, 4))
    if (op == 3) call fsbm_put(s, 'ff1_in', f1)
    call fsbm_put(s, 'ff2_in', f2)
    call fsbm_put(s, 'ff3_in', f3)
    call fsbm_put(s, 'ff4_in', f4)
    call fsbm_put(s, 'ff5_in', f5)
    call fsbm_put(s, 'tt_out', tt)
    call fsbm_put(s, 'qq_out', qq)
    call fsbm_put(s, 'del1n_out', del1n)
    call fsbm_put(s, 'del2n_out', del2n)
    if (op == 3) call fsbm_put(s, 'ff1_out', ff1)
    call fsbm_put(s, 'ff2_out', ff2)
    call fsbm_put(s, 'ff3_out', ff3)
    call fsbm_put(s, 'ff4_out', ff4)
    call fsbm_put(s, 'ff5_out', ff5)
    call fsbm_put(s, 'supsat_calls', capc_supsat_n)
    call fsbm_put(s, 'supsat_kcond', capc_supsat_kcond(1))
    call fsbm_put(s, 'supsat_branch', capc_supsat_branch(1))
    call fsbm_put(s, 'supsat_g2_clamp', capc_supsat_clamp(1))
    call fsbm_put(s, 'supsat_rates', capc_supsat_rates(:, 1))
    call fsbm_end(s)
  end subroutine cond_case

  ! ONECOND2 (ice only): KCOND 21 (DEL2N > 0) and 20 (DEL2N <= 0, the exact
  ! zero included) at 4 temperatures, 5 ice supersaturations and 3 ice
  ! populations (snow; hail; two crystal habits with snow and graupel).
  ! set 2: snow diluted a thousandth and a hundred-millionth of the dilute
  ! spectrum, for JERSUPSAT_KS's slow ice branch (5) and its no-rate
  ! branch (0).
  subroutine sweep_onecond2()
    real(8) :: temps(4), plist(4), sats(5), qq, d1, d2, dv1, dv2, ffd(nb), dil(2)
    real :: f1(nb), f2(nb, 3), f3(nb), f4(nb), f5(nb)
    integer :: it, is, ip, c, idl
    temps = (/ 268.15d0, 258.15d0, 243.15d0, 228.15d0 /)
    plist = (/ 700.0d3, 600.0d3, 450.0d3, 350.0d3 /)
    sats = (/ 0.15d0, 0.02d0, 0.0d0, -0.05d0, -0.3d0 /)
    pr_capture = .true.
    c = 0
    do it = 1, 4
      do is = 1, 5
        do ip = 1, 3
          call thermo_ice(temps(it), sats(is), plist(it), qq, d1, d2, dv1, dv2)
          if (is == 3) d2 = 0.0d0
          f1 = 0.0; f2 = 0.0; f3 = 0.0; f4 = 0.0; f5 = 0.0
          select case (ip)
          case (1)
            call ice_spectrum(2, ffd); f3 = real(ffd)
          case (2)
            call ice_spectrum(4, ffd); f5 = real(ffd)
          case (3)
            call ice_spectrum(1, ffd); f2(:, 1) = real(ffd); f2(:, 2) = real(0.5d0*ffd)
            call ice_spectrum(2, ffd); f3 = real(ffd)
            call ice_spectrum(3, ffd); f4 = real(ffd)
          end select
          c = c + 1
          call cond_case(2, c, 1, temps(it), plist(it), d1, d2, dv1, dv2, qq, 4.0, &
                         f1, f2, f3, f4, f5)
        end do
      end do
    end do
    dil = (/ 1.0d-3, 1.0d-8 /)
    do idl = 1, 2
      do is = 1, 2
        call thermo_ice(temps(2), sats(merge(1, 4, is == 1)), plist(2), qq, d1, d2, dv1, dv2)
        f1 = 0.0; f2 = 0.0; f4 = 0.0; f5 = 0.0
        call ice_spectrum(6, ffd); f3 = real(dil(idl)*ffd)
        c = c + 1
        call cond_case(2, c, 2, temps(2), plist(2), d1, d2, dv1, dv2, qq, 4.0, &
                       f1, f2, f3, f4, f5)
      end do
    end do
    pr_capture = .false.
  end subroutine sweep_onecond2

  ! ONECOND3 (mixed phase): KCOND 31 (both supersaturated), 32 (Bergeron),
  ! 30 (both subsaturated) and 50 (DEL1N > 0 > DEL2N, and the exact zeros
  ! DEL2N = 0 with DEL1N < 0 and DEL1N = 0 with DEL2N > 0) at 3 temperatures
  ! and 3 populations (drops and snow; drops and hail; drops, two crystal
  ! habits and snow); a dilute population at each temperature (JERSUPSAT_KS
  ! branch 7, finding S2); and the skip above -0.187 C (:7908).
  subroutine sweep_onecond3()
    real(8) :: temps(3), plist(3), qq, d1, d2, dv1, dv2, ffd(nb), ratio, sw
    real :: f1(nb), f2(nb, 3), f3(nb), f4(nb), f5(nb)
    integer :: it, ir, ip, c
    temps = (/ 272.15d0, 263.15d0, 245.15d0 /)
    plist = (/ 800.0d3, 650.0d3, 450.0d3 /)
    pr_capture = .true.
    c = 0
    do it = 1, 3
      ratio = (AA2_MY*DEXP(-BB2_MY/temps(it)))/(AA1_MY*DEXP(-BB1_MY/temps(it)))
      do ir = 1, 6
        do ip = 1, 3
          select case (ir)
          case (1)   ! KCOND 31
            sw = 5.0d-3
          case (2)   ! KCOND 32: halfway between ice and water saturation
            sw = -0.5d0*(1.0d0 - ratio)
          case (3)   ! KCOND 30
            sw = -0.2d0
          case default
            sw = -0.5d0*(1.0d0 - ratio)
          end select
          call thermo(temps(it), sw, plist(it), qq, d1, d2, dv1, dv2)
          if (ir == 4) then       ! KCOND 50: DEL1N > 0 > DEL2N
            d1 = 1.0d-3
            d2 = -1.0d-3
          else if (ir == 5) then  ! KCOND 50: DEL2N exactly 0, DEL1N < 0
            d2 = 0.0d0
          else if (ir == 6) then  ! KCOND 50: DEL1N exactly 0, DEL2N > 0
            d1 = 0.0d0
          end if
          f1 = 0.0; f2 = 0.0; f3 = 0.0; f4 = 0.0; f5 = 0.0
          call spectrum(2, ffd); f1 = real(ffd)
          select case (ip)
          case (1)
            call ice_spectrum(2, ffd); f3 = real(ffd)
          case (2)
            call ice_spectrum(4, ffd); f5 = real(ffd)
          case (3)
            call ice_spectrum(1, ffd); f2(:, 1) = real(ffd); f2(:, 3) = real(0.5d0*ffd)
            call ice_spectrum(2, ffd); f3 = real(ffd)
          end select
          c = c + 1
          call cond_case(3, c, 1, temps(it), plist(it), d1, d2, dv1, dv2, qq, 4.0, &
                         f1, f2, f3, f4, f5)
        end do
      end do
      ! dilute drops and snow, Bergeron: slow rates, JERSUPSAT_KS branch 7
      sw = -0.5d0*(1.0d0 - ratio)
      call thermo(temps(it), sw, plist(it), qq, d1, d2, dv1, dv2)
      call spectrum(1, ffd); f1 = real(1.0d-4*ffd)
      call ice_spectrum(6, ffd); f3 = real(ffd)
      f2 = 0.0; f4 = 0.0; f5 = 0.0
      c = c + 1
      call cond_case(3, c, 2, temps(it), plist(it), d1, d2, dv1, dv2, qq, 4.0, &
                     f1, f2, f3, f4, f5)
      ! cloud and rain with a trace of snow: ice below RATIO_ICEW_MIN of the
      ! water rates, so JERSUPSAT_KS solves for water only (branch 2)
      call thermo(temps(it), 5.0d-3, plist(it), qq, d1, d2, dv1, dv2)
      call spectrum(2, ffd); f1 = real(ffd)
      call ice_spectrum(6, ffd); f3 = real(1.0d-6*ffd)
      c = c + 1
      call cond_case(3, c, 2, temps(it), plist(it), d1, d2, dv1, dv2, qq, 4.0, &
                     f1, f2, f3, f4, f5)
    end do
    ! above -0.187 C: ONECOND3 returns its input (GO TO 17)
    call thermo(273.05d0, 5.0d-3, 900.0d3, qq, d1, d2, dv1, dv2)
    call spectrum(2, ffd); f1 = real(ffd)
    call ice_spectrum(2, ffd); f3 = real(ffd)
    f2 = 0.0; f4 = 0.0; f5 = 0.0
    c = c + 1
    call cond_case(3, c, 3, 273.05d0, 900.0d3, d1, d2, dv1, dv2, qq, 4.0, f1, f2, f3, f4, f5)
    pr_capture = .false.
  end subroutine sweep_onecond3

  ! --------------------------------------------------------- cold collisions
  ! coll_xyx_lwf, coll_xyy_lwf and coll_xyz_lwf. Records 'orient', set 1, the
  ! old port's 24 live cases: each routine x indc 0 and 1 x seeds 0, 71, 2701
  ! and 7753, a non-symmetric kernel uniform(1e-4, 1e-3), g uniform(0.01, 0.3)
  ! in bins 1-31 of three populations, liquid fractions uniform(0, 1), prdkrn
  ! 1e-9, on the synthetic grid x = 1e-9 2^(k-1) with y = 1.25 x (the
  ! orientation cases: WRF's own five grids are equal, so a swapped x and y
  ! cannot show on them); set 2, the same on the synthetic y = x; set 3, the
  ! same draws on the real tables: x = y = xl_mg (the :6078 expression) with
  ! the module's chucm and ima from FAST_HUCMINIT. Records 'table': the calls
  ! COAL_BOTT_NEW makes, on the real grids and kernels (850 and 500 hPa, dt
  ! 12 s, as the kernels fixture), with physical spectra, and snow-snow
  ! through coll_xxx_lwf.
  subroutine sweep_coll_cold()
    real :: xr(nb)
    real(8) :: xs8(nb), y125(nb), csyn(nb, nb), g(nb, 3), f(nb, 3), ck(nb, nb)
    real(8) :: gg(nb, 3), fl(nb, 3), dm(nb), prd, xlm(nb), xsm(nb), xgm(nb), xhm(nb)
    real(8) :: ffd(nb), gd(nb), gs(nb), gh(nb), qqk, fct
    real(8) :: g_all(nb, 3, 4), f_all(nb, 3, 4), ck_all(nb, nb, 4)
    real(8) :: gx_grid(nb), gy_grid(nb), c_grid(nb, nb)
    integer :: imasyn(nb, nb), ima_grid(nb, nb), seeds(4), is, op, indc, grid, c, k, i, ip
    real :: plist(2), pz, dtk, ff(nb)
    seeds = (/ 0, 71, 2701, 7753 /)
    do k = 1, nb
      xr(k) = 1.0e-9*2.0**(k-1)
    end do
    xs8 = xr
    y125 = xs8*1.25d0
    csyn = 0.0d0
    imasyn = nb - 1
    call courant_bott_ks(xr, nb, csyn, imasyn, 1.0d0)
    call fsbm_begin(s, 'grid')
    call fsbm_put(s, 'x', xs8)
    call fsbm_put(s, 'y125', y125)
    call fsbm_put(s, 'c', csyn)
    call fsbm_put(s, 'ima', int(imasyn, 4))
    call fsbm_end(s)
    ! the real grids (:6078)
    do k = 1, nb
      xlm(k) = xl(k)*1.e3
      xsm(k) = xs(k)*1.e3
      xgm(k) = xg(k)*1.e3
      xhm(k) = xh(k)*1.e3
    end do
    c = 0
    prd = 1.0d-9
    do is = 1, 4
      call pm_seed(seeds(is))
      g = 0.0d0
      do i = 1, 3
        do k = 1, nb - 2
          g(k, i) = pm_uniform(0.01d0, 0.3d0)
        end do
      end do
      do i = 1, 3
        do k = 1, nb
          f(k, i) = pm_uniform(0.0d0, 1.0d0)
        end do
      end do
      do i = 1, nb
        do k = 1, nb
          ck(i, k) = pm_uniform(1.0d-4, 1.0d-3)
        end do
      end do
      g_all(:, :, is) = g
      f_all(:, :, is) = f
      ck_all(:, :, is) = ck
      call fsbm_begin(s, 'draws')
      call fsbm_put(s, 'seed', int(seeds(is), 4))
      call fsbm_put(s, 'g', g)
      call fsbm_put(s, 'fl', f)
      call fsbm_put(s, 'ck', ck)
      call fsbm_end(s)
      do grid = 1, 2
        gx_grid = xs8
        gy_grid = xs8
        if (grid == 1) gy_grid = y125
        c_grid = csyn
        ima_grid = imasyn
        do op = 1, 3
          do indc = 0, 1
            c = c + 1
            call orient_case(c, grid, seeds(is), op, indc, g, f, ck, gx_grid, gy_grid, &
                             c_grid, ima_grid, prd)
          end do
        end do
      end do
    end do
    ! the table records: real grids and kernels
    call fsbm_begin(s, 'table_grid')
    call fsbm_put(s, 'xl_mg', xlm)
    call fsbm_put(s, 'xs_mg', xsm)
    call fsbm_put(s, 'xg_mg', xgm)
    call fsbm_put(s, 'xh_mg', xhm)
    call fsbm_end(s)
    call spectrum(2, ffd); ff = real(ffd); call g1_from_ff(ff, gd)
    call ice_spectrum(2, ffd); ff = real(ffd)
    do k = 1, nb
      gs(k) = ff(k)*3.*xs(k)*xs(k)*1.e3
    end do
    call ice_spectrum(4, ffd); ff = real(ffd)
    do k = 1, nb
      gh(k) = ff(k)*3.*xh(k)*xh(k)*1.e3
    end do
    plist = (/ 850.0, 500.0 /)
    dtk = 12.0
    do ip = 1, 2
      pz = plist(ip)*1000.0
      call kernals_ks(dtk, nb, pz)
      do op = 1, 6
        gg = 0.0d0
        fl = 0.0d0
        dm = -1.0d0
        fl(:, 1) = 1.0d0
        select case (op)
        case (1)   ! drops - snow = hail   (:8452)
          gg(:, 1) = gd; gg(:, 2) = gs
          call coll_xyz_lwf(gg(:, 1), gg(:, 2), gg(:, 3), fl(:, 1), fl(:, 2), fl(:, 3), &
                            cwls, xlm, xsm, chucm, ima, 1.0d0, nb, 0)
        case (2)   ! snow - drops = snow   (:8462)
          gg(:, 1) = gs; gg(:, 2) = gd
          fl(:, 1) = 0.0d0; fl(:, 2) = 1.0d0
          call coll_xyx_lwf(gg(:, 1), gg(:, 2), fl(:, 1), fl(:, 2), cwsl, xsm, xlm, &
                            chucm, ima, 1.0d0, nb, 1, dm)
        case (3)   ! snow - drops = hail   (:8466)
          gg(:, 1) = gs; gg(:, 2) = gd
          fl(:, 1) = 0.0d0; fl(:, 2) = 1.0d0
          call coll_xyz_lwf(gg(:, 1), gg(:, 2), gg(:, 3), fl(:, 1), fl(:, 2), fl(:, 3), &
                            cwsl, xsm, xlm, chucm, ima, 1.0d0, nb, 1)
        case (4)   ! drops - hail = hail   (:8524)
          gg(:, 1) = gd; gg(:, 2) = gh
          call coll_xyy_lwf(gg(:, 1), gg(:, 2), fl(:, 1), fl(:, 2), cwlh, xlm, xhm, &
                            chucm, ima, 1.0d0, nb, 0)
        case (5)   ! hail - drops = hail   (:8534)
          gg(:, 1) = gh; gg(:, 2) = gd
          fl(:, 1) = 0.0d0; fl(:, 2) = 1.0d0
          call coll_xyx_lwf(gg(:, 1), gg(:, 2), fl(:, 1), fl(:, 2), cwhl, xhm, xlm, &
                            chucm, ima, 1.0d0, nb, 1, dm)
        case (6)   ! snow - snow (:8563), prdkrn from modkrn_KS CASE 11 at 258.15 K
          gg(:, 1) = gs
          fl(:, 1) = 0.0d0
          qqk = 0.002d0
          fct = -1.0d0
          call modkrn_ks(258.15d0, qqk, real(pz, 8), 7.0e-4, fct, 2.3315E02, 11, 1, 1, 1, 1)
          call coll_xxx_lwf(gg(:, 1), fl(:, 1), cwss, xsm, chucm, ima, fct, nb)
        end select
        c = c + 1
        call fsbm_begin(s, 'table')
        call fsbm_put(s, 'case_id', int(c, 4))
        call fsbm_put(s, 'op', int(op, 4))
        call fsbm_put(s, 'p_hpa', plist(ip))
        call fsbm_put(s, 'dt_coll', dtk)
        call fsbm_put(s, 'g_drops', gd)
        call fsbm_put(s, 'g_snow', gs)
        call fsbm_put(s, 'g_hail', gh)
        call fsbm_put(s, 'g_out', gg)
        call fsbm_put(s, 'fl_out', fl)
        call fsbm_put(s, 'dm_rime_out', dm)
        call fsbm_end(s)
      end do
    end do
    ! orient set 3, after the table records, so every case number the
    ! synthetic sets and the table records had before is unchanged
    gx_grid = xlm
    gy_grid = xlm
    c_grid = chucm
    ima_grid = ima
    do is = 1, 4
      do op = 1, 3
        do indc = 0, 1
          c = c + 1
          call orient_case(c, 3, seeds(is), op, indc, g_all(:, :, is), f_all(:, :, is), &
                           ck_all(:, :, is), gx_grid, gy_grid, c_grid, ima_grid, prd)
        end do
      end do
    end do
  end subroutine sweep_coll_cold

  ! One orientation case: coll_xyx_lwf (op 1), coll_xyy_lwf (2) or
  ! coll_xyz_lwf (3) on the draws of one seed, on the grid x, y with the
  ! Courant numbers cg and the target bins ig.
  subroutine orient_case(c, set, seed, op, indc, g, f, ck, x, y, cg, ig, prd)
    integer, intent(in) :: c, set, seed, op, indc, ig(nb, nb)
    real(8), intent(in) :: g(nb, 3), f(nb, 3), ck(nb, nb), x(nb), y(nb), cg(nb, nb), prd
    real(8) :: gg(nb, 3), fl(nb, 3), dm(nb)
    gg = g
    fl = f
    dm = -1.0d0
    select case (op)
    case (1)
      call coll_xyx_lwf(gg(:, 1), gg(:, 2), fl(:, 1), fl(:, 2), ck, x, y, cg, ig, prd, nb, &
                        indc, dm)
    case (2)
      call coll_xyy_lwf(gg(:, 1), gg(:, 2), fl(:, 1), fl(:, 2), ck, x, y, cg, ig, prd, nb, indc)
    case (3)
      call coll_xyz_lwf(gg(:, 1), gg(:, 2), gg(:, 3), fl(:, 1), fl(:, 2), fl(:, 3), ck, x, y, &
                        cg, ig, prd, nb, indc)
    end select
    call fsbm_begin(s, 'orient')
    call fsbm_put(s, 'case_id', int(c, 4))
    call fsbm_put(s, 'set', int(set, 4))
    call fsbm_put(s, 'op', int(op, 4))
    call fsbm_put(s, 'indc', int(indc, 4))
    call fsbm_put(s, 'seed', int(seed, 4))
    call fsbm_put(s, 'prdkrn', prd)
    call fsbm_put(s, 'g_out', gg)
    call fsbm_put(s, 'fl_out', fl)
    call fsbm_put(s, 'dm_rime_out', dm)
    call fsbm_end(s)
  end subroutine orient_case

  ! modkrn_KS CASE 11 (the only case COAL_BOTT_NEW uses; INTERPOL_SE, private
  ! to module_mp_SBM_Collision, is reached only through it). set 1, the old
  ! port's 12: 230, 253.15, 259, 268, 273.15 and 278 K x Icondition 0 and 1.
  ! set 2: the seven table nodes, the node temperature being 273.15 plus the
  ! node in the build's real; at the last node (-40 C) INTERPOL_SE sets
  ! nothing. The output starts at -1 in every case, so a record shows whether
  ! the routine set it (above 0 C it does not: finding D6).
  subroutine sweep_modkrn()
    real(8) :: temps(6), tt
    real :: nodes(7)
    integer :: it, ic, c, n
    temps = (/ 230.0d0, 253.15d0, 259.0d0, 268.0d0, 273.15d0, 278.0d0 /)
    nodes = [0.0, -0.813, -5.26, -10.13, -14.63, -20.02, -40.0]
    c = 0
    do it = 1, 6
      do ic = 0, 1
        tt = temps(it)
        c = c + 1
        call modkrn_one(c, 1, tt, ic)
      end do
    end do
    do n = 1, 7
      tt = 273.15 + real(nodes(n), 8)
      c = c + 1
      call modkrn_one(c, 2, tt, 1)
    end do
  end subroutine sweep_modkrn

  subroutine modkrn_one(cid, set, t, icond)
    integer, intent(in) :: cid, set, icond
    real(8), intent(in) :: t
    real(8) :: qq, fct, tc
    qq = 0.002d0
    fct = -1.0d0
    tc = t - 273.15
    call modkrn_ks(t, qq, 8.0d5, 1.0, fct, 2.3315E02, 11, icond, 1, 1, 1)
    call fsbm_begin(s, 'case')
    call fsbm_put(s, 'case_id', int(cid, 4))
    call fsbm_put(s, 'set', int(set, 4))
    call fsbm_put(s, 'tt', t)
    call fsbm_put(s, 'tc', tc)
    call fsbm_put(s, 'qq_out', qq)
    call fsbm_put(s, 'icondition', int(icond, 4))
    call fsbm_put(s, 'factor_in', -1.0d0)
    call fsbm_put(s, 'factor_out', fct)
    call fsbm_end(s)
  end subroutine modkrn_one

  ! One COAL_BOTT_NEW call with ice; the cold block is read back through the
  ! probe build (the collision calls it made, the Hallett-Mossop values, the
  ! riming heat, and the spectra it saw and returned). COAL_BOTT_NEW returns
  ! CollEff = PRDKRN (:8322), which modkrn_KS leaves unset above 0 C (D6)
  ! and at exactly -40 C (INTERPOL_SE sets nothing at its last node), so
  ! the returned value is then uninitialised: the record holds it only where
  ! modkrn_KS set it (colleff_defined 1, decided as modkrn_KS decides, in the
  ! build's arithmetic) and 0 elsewhere.
  subroutine coal_cold_case(c, set, pz_hpa, t0, combo, dtc)
    integer, intent(in) :: c, set, combo
    real, intent(in) :: pz_hpa, dtc
    real(8), intent(in) :: t0
    real :: ff1r(nb), ff2r(nb, 3), ff3r(nb), ff4r(nb), ff5r(nb), rho, colleff, tcrit, &
            ttcoal, f1in(nb), f3in(nb), fdin(nb)
    real(8) :: tt, qq, pp, del1in, del2in, fls(nb), flg(nb), flh(nb), frs(nb), ffd(nb), &
               d10, d20, sw, tcm
    integer :: cdef
    tcrit = 2.7015E02
    ttcoal = 2.3315E02
    ff1r = 0.0; ff2r = 0.0; ff3r = 0.0; ff4r = 0.0; ff5r = 0.0
    fdin = 0.0
    select case (combo)
    case (1)   ! light cloud (ALWC below ALCR) and snow
      call add_lognormal_c(60.0d0, 8.0d-4, 1.3d0, ff1r)
      call ice_spectrum(2, ffd); ff3r = real(ffd)
    case (2)   ! dense cloud with drizzle (ALWC above ALCR) and snow
      call add_lognormal_c(300.0d0, 1.0d-3, 1.3d0, ff1r)
      call add_lognormal_c(1.0d0, 4.0d-3, 1.4d0, ff1r)
      call ice_spectrum(2, ffd); ff3r = real(ffd)
    case (3)   ! cloud with drizzle and dense ice (riming, Hallett-Mossop)
      call add_lognormal_c(300.0d0, 1.0d-3, 1.3d0, ff1r)
      call add_lognormal_c(1.0d0, 4.0d-3, 1.4d0, ff1r)
      call ice_spectrum(3, ffd); fdin = real(ffd)
    case (4)   ! cloud, snow and dense ice
      call add_lognormal_c(300.0d0, 1.0d-3, 1.3d0, ff1r)
      call ice_spectrum(2, ffd); ff3r = real(ffd)
      call ice_spectrum(4, ffd); fdin = real(ffd)
    case (5)   ! snow only (snow-snow)
      call ice_spectrum(2, ffd); ff3r = real(ffd)
    case default
      call fsbm_fail('unknown coal_cold combo')
    end select
    ! Dense ice enters as FAST_SBM passes it: FF5R in the hail build, FF4R in
    ! the graupel build.
    if (hail_opt == 1) then
      ff5r = fdin
    else
      ff4r = fdin
    end if
    f1in = ff1r; f3in = ff3r
    fls = 0.0d0; flg = 0.0d0; flh = 0.0d0; frs = 0.0d0
    tt = t0
    pp = real(pz_hpa*1000.0, 8)
    sw = 2.0d-3
    call thermo(tt, sw, pp, qq, del1in, del2in, d10, d20)
    d10 = del1in; d20 = del2in
    rho = air_density_cgs(pz_hpa*1000.0, tt)
    colleff = -1.0
    call reset_cold_capture()
    call coal_bott_new(ff1r, ff2r, ff3r, ff4r, ff5r, tt, qq, pp, rho, dtc, tcrit, &
                       ttcoal, fls, flg, flh, frs, del1in, del2in, 1, 1, 1, 1, colleff)
    ! modkrn_KS: tc = tt - 273.15 (:1416), return if tc > 0 (:1417); INTERPOL_SE
    ! assigns nothing when H equals its last node, -40.0 (:1372-1390)
    tcm = t0 - 273.15
    cdef = 0
    if (tcm <= 0.0 .and. tcm /= -40.0) cdef = 1
    if (cdef == 0) colleff = 0.0
    call fsbm_begin(s, 'case')
    call fsbm_put(s, 'case_id', int(c, 4))
    call fsbm_put(s, 'set', int(set, 4))
    call fsbm_put(s, 'combo', int(combo, 4))
    call fsbm_put(s, 'hail_opt', int(hail_opt, 4))
    call fsbm_put(s, 'p_hpa', pz_hpa)
    call fsbm_put(s, 'dt_coll', dtc)
    call fsbm_put(s, 'tt_in', t0)
    call fsbm_put(s, 'qq', qq)
    call fsbm_put(s, 'pp', pp)
    call fsbm_put(s, 'rho', rho)
    call fsbm_put(s, 'del1in_in', d10)
    call fsbm_put(s, 'del2in_in', d20)
    call fsbm_put(s, 'ff1_in', f1in)
    call fsbm_put(s, 'ff3_in', f3in)
    call fsbm_put(s, 'ffdense_in', fdin)
    call fsbm_put(s, 'ff1_out', ff1r)
    call fsbm_put(s, 'ff3_out', ff3r)
    call fsbm_put(s, 'ff4_out', ff4r)
    call fsbm_put(s, 'ff5_out', ff5r)
    call fsbm_put(s, 'fliqfr_s_out', fls)
    call fsbm_put(s, 'fliqfr_g_out', flg)
    call fsbm_put(s, 'fliqfr_h_out', flh)
    call fsbm_put(s, 'frimfr_s_out', frs)
    call fsbm_put(s, 'tt_out', tt)
    call fsbm_put(s, 'del1in_out', del1in)
    call fsbm_put(s, 'del2in_out', del2in)
    call fsbm_put(s, 'colleff', colleff)
    call fsbm_put(s, 'colleff_defined', int(cdef, 4))
    call fsbm_put(s, 'cold_block', capc_cold_block)
    call fsbm_put(s, 'icol', capc_cold_icol)
    call fsbm_put(s, 'cold_call_mask', capc_cc_mask)
    call fsbm_put(s, 'cold_scalars', capc_cold_scal)
    call fsbm_put(s, 'g_before_cold', capc_cold_g_in)
    call fsbm_put(s, 'g_after_cold', capc_cold_g_out)
    call fsbm_put(s, 'rf3_after_cold', capc_cold_rf3_out)
    call fsbm_put(s, 'hm_fired', capc_hm_fired)
    call fsbm_put(s, 'hm', capc_hm)
    call fsbm_put(s, 'rime', capc_rime)
    call fsbm_end(s)
  end subroutine coal_cold_case

  subroutine add_lognormal_c(nd, rm, sig, ff)
    real(8), intent(in) :: nd, rm, sig
    real, intent(inout) :: ff(nb)
    real(8) :: fd(nb)
    fd = 0.0d0
    call add_lognormal(nd, rm, sig, fd)
    ff = ff + real(fd)
  end subroutine add_lognormal_c

  ! Cold COAL_BOTT_NEW: 5 populations (light cloud and snow, ALWC below ALCR;
  ! dense cloud with drizzle and snow, above it; cloud with drizzle and dense
  ! ice; cloud, snow and dense ice; snow alone) at 11 temperatures from
  ! 273.15 K (the cold block's edge) down to 233.15 K, through the
  ! Hallett-Mossop ramp edges 265.15, 268.15 and 270.15 K, at 600 hPa; and
  ! the mixed population at 850 and 400 hPa. The same sweep runs in the hail
  ! and graupel builds.
  subroutine sweep_coal_cold()
    real(8) :: temps(11)
    integer :: it, cb, c
    temps = (/ 273.15d0, 272.15d0, 270.15d0, 269.15d0, 268.15d0, 266.65d0, 265.15d0, &
               264.15d0, 253.15d0, 240.15d0, 233.15d0 /)
    pr_capture = .true.
    c = 0
    do it = 1, 11
      do cb = 1, 5
        c = c + 1
        call coal_cold_case(c, 1, 600.0, temps(it), cb, 12.0)
      end do
    end do
    do it = 1, 2
      c = c + 1
      call coal_cold_case(c, 2, 850.0, merge(268.15d0, 253.15d0, it == 1), 4, 12.0)
      c = c + 1
      call coal_cold_case(c, 2, 400.0, merge(268.15d0, 253.15d0, it == 1), 4, 12.0)
    end do
    pr_capture = .false.
  end subroutine sweep_coal_cold

  ! ----------------------------------------------------------- cold refusals
  ! Each case writes its input record, closes it, announces the call, then
  ! makes a call that must not return. The expected stop per case is listed
  ! in pack_fixtures.py (REFUSAL_COLD).
  subroutine refusal_cold_case(which)
    character(len=*), intent(in) :: which
    integer :: n, isym(7), k
    real(8) :: tt, qq, pp, d1, d2, dv1, dv2, sup1, sup2, ffd(nb), x(nb), fls(nb), flg(nb), &
               flh(nb), frs(nb)
    real :: psi1(nb), psi2(nb, 3), fccnr(na), xi_c(nb, 3), f1(nb), f2(nb, 3), f3(nb), &
            f4(nb), f5(nb), b21(nb), fi(nb), rho, colleff, dtc, dtcond
    read(which, *) n
    call fsbm_open(s, trim(outdir)//'/refusal_cold_'//trim(which))
    select case (n)
    case (1)
      ! ICE_NUCL's bare STOP 099 (:3463), through JERNUCL01_KS at -15 C
      ! (habit 3): the grid's first habit-3 mass is the smallest subnormal of
      ! the build's real, so 1/DX overflows and the concentration check sees
      ! an infinite change. From a float32 grid the product cannot underflow
      ! in real(8), so the native build returns.
      xi_c = xi
      xi_c(1, 3) = tiny(1.0)*epsilon(1.0)
      psi1 = 0.0; psi2 = 0.0; fccnr = 0.0
      tt = 258.15d0; qq = 0.004d0; sup1 = -1.0d-2; sup2 = 0.2d0
      call fsbm_begin(s, 'ice_nucl')
      call fsbm_put(s, 'tt', tt)
      call fsbm_put(s, 'sup1', sup1)
      call fsbm_put(s, 'sup2', sup2)
      call fsbm_put(s, 'xi', xi_c)
      call fsbm_end(s)
      call fsbm_close(s)
      call announce_cold(n, 'JERNUCL01_KS/ICE_NUCL')
      call jernucl01_ks(psi1, psi2, fccnr, xl, xi_c, tt, qq, 0.001, 8.0e5, sup1, sup2, &
                        col, 0.0, 0.0, rccn, dropradii, nb, na, 3, 1, &
                        200.0, 0, ro_solute, ions, mwaero, 1, 1, 1)
    case (2)
      ! a negative snow bin and no growth: JERNEWF_KS's negative check (:2750)
      call ice_spectrum(2, ffd)
      fi = real(ffd)
      fi(12) = -1.0
      x = xs
      b21 = 0.0
      call fsbm_begin(s, 'jerdfun')
      call fsbm_put(s, 'xi', x)
      call fsbm_put(s, 'fi', fi)
      call fsbm_put(s, 'b21', b21)
      call fsbm_end(s)
      call fsbm_close(s)
      call announce_cold(n, 'JERDFUN_KS/JERNEWF_KS')
      call jerdfun_ice_case(-1, 1, 1, 3, 1, x, b21, fi, 0.05, 2)
    case (3, 4, 5, 6, 7, 8, 12)
      ! 3, 6: zero vapour makes RW and PW NaN (ONECOND2 :7441, ONECOND3 :7993)
      ! 4, 7: a negative substep (ONECOND2 :7452 on its KCOND 21 path,
      !       ONECOND3 :8049)
      ! 12: a negative substep on ONECOND2's KCOND 20 path (:7535): case 4
      !       below ice saturation
      ! 5, 8: a latent heating above 5 K in one substep (:7694, :8239)
      tt = 268.15d0
      pp = 500.0d3
      dtcond = 4.0
      f1 = 0.0; f2 = 0.0; f3 = 0.0; f4 = 0.0; f5 = 0.0
      call ice_spectrum(5, ffd); f3 = real(ffd)
      if (n >= 6 .and. n <= 8) then
        call spectrum(4, ffd); f1 = real(ffd)
      end if
      select case (n)
      case (3, 6)
        ! no vapour: DEL1N = DEL2N = -1, so DOPL = DOPI = 0 against an
        ! infinite OPER2(0)
        call thermo_ice(tt, 0.1d0, pp, qq, d1, d2, dv1, dv2)
        qq = 0.0d0
        d1 = -1.0d0
        d2 = -1.0d0
        dv1 = 0.0d0
        dv2 = 0.0d0
      case (4, 7)
        call thermo_ice(tt, 0.1d0, pp, qq, d1, d2, dv1, dv2)
        dtcond = -4.0
      case (12)
        ! 10 percent below ice saturation: DEL2N < 0, so KCOND 20
        call thermo_ice(tt, -0.1d0, pp, qq, d1, d2, dv1, dv2)
        dtcond = -4.0
      case (5)
        call thermo_ice(tt, 2.0d0, pp, qq, d1, d2, dv1, dv2)
        dtcond = 3600.0
      case (8)
        call thermo(tt, 2.0d0, pp, qq, d1, d2, dv1, dv2)
        dtcond = 3600.0
      end select
      call isym_of(f1, f2, f3, f4, f5, isym)
      call fsbm_begin(s, 'cond')
      call fsbm_put(s, 'tt', tt)
      call fsbm_put(s, 'qq', qq)
      call fsbm_put(s, 'pp', pp)
      call fsbm_put(s, 'del1n', d1)
      call fsbm_put(s, 'del2n', d2)
      call fsbm_put(s, 'dtcond', dtcond)
      call fsbm_put(s, 'ff1', f1)
      call fsbm_put(s, 'ff3', f3)
      call fsbm_end(s)
      call fsbm_close(s)
      if (n <= 5 .or. n == 12) then
        call announce_cold(n, 'ONECOND2')
        call cond_case(2, -1, 1, tt, pp, d1, d2, dv1, dv2, qq, dtcond, f1, f2, f3, f4, f5)
      else
        call announce_cold(n, 'ONECOND3')
        call cond_case(3, -1, 1, tt, pp, d1, d2, dv1, dv2, qq, dtcond, f1, f2, f3, f4, f5)
      end if
    case (9, 10, 11)
      ! 9: drops gaining mass in a cold collision step (:8598)
      ! 10: a riming heat above 5 K in one step (:8636)
      ! 11: a negative snow bin at the end of the step (:8614)
      f1 = 0.0; f2 = 0.0; f3 = 0.0; f4 = 0.0; f5 = 0.0
      fls = 0.0d0; flg = 0.0d0; flh = 0.0d0; frs = 0.0d0
      tt = 263.15d0
      pp = 600.0d3
      qq = 0.002d0
      d1 = -1.0d-2; d2 = 5.0d-2
      dtc = 12.0
      select case (n)
      case (9)
        ! a flat, very dilute drop spectrum (g = 1.5e-13 in bins 1-10 and 30)
        ! and a very long step: the gmin floor of coll_xxx_lwf adds drop mass
        do k = 1, nb
          if (k <= 10 .or. k == 30) f1(k) = real(1.5d-13/(3.0d0*real(xl(k), 8)**2*1.0d3))
        end do
        dtc = 1.0e4
      case (10)
        ! dense cloud and drizzle (no drop above bin 31, so no breakup) and
        ! much hail over a long step
        call add_lognormal_c(2000.0d0, 1.2d-3, 1.2d0, f1)
        call add_lognormal_c(5.0d0, 5.0d-3, 1.3d0, f1)
        f1(29:nb) = 0.0
        call ice_spectrum(4, ffd); f5 = real(20.0d0*ffd)
        if (hail_opt /= 1) then
          f4 = f5; f5 = 0.0
        end if
        dtc = 600.0
        pp = 400.0d3
      case (11)
        call ice_spectrum(2, ffd); f3 = real(ffd)
        f3(20) = -1.0e-3
        tt = 288.15d0
      end select
      rho = air_density_cgs(real(pp), tt)
      call fsbm_begin(s, 'coal_bott')
      call fsbm_put(s, 'ff1', f1)
      call fsbm_put(s, 'ff3', f3)
      call fsbm_put(s, 'ff4', f4)
      call fsbm_put(s, 'ff5', f5)
      call fsbm_put(s, 'tt', tt)
      call fsbm_put(s, 'pp', pp)
      call fsbm_put(s, 'rho', rho)
      call fsbm_put(s, 'dt_coll', dtc)
      call fsbm_end(s)
      call fsbm_close(s)
      call announce_cold(n, 'COAL_BOTT_NEW')
      call coal_bott_new(f1, f2, f3, f4, f5, tt, qq, pp, rho, dtc, 2.7015E02, 2.3315E02, &
                         fls, flg, flh, frs, d1, d2, 1, 1, 1, 1, colleff)
    case (13)
      ! ICE_NUCL's bare STOP 099 (:3463) on finite inputs, in both builds,
      ! through JERNUCL01_KS at -15 C (habit 3) with 60 percent ice
      ! supersaturation (HELEK1 = DELTAF = 1.106 per cm3 after the 59 percent
      ! clamps). Habit 3 holds a mixed-sign spectrum whose concentration terms
      ! 3 COL PSI2 xi are about 7.5 (bin 1), +1e17 (bin 2) and -1.00001e17
      ! (bin 3). Its running real(8) sum rounds to multiples of 16 once it
      ! passes 2^56, so bin 1 is absorbed before nucleation (7.5 < 8) and adds
      ! 16 after it (7.5 + 1.106 > 8); bin 3 cancels exactly. The concentration
      ! check sees a change of 16 per cm3 against its limit of 10, while the
      ! concentration before nucleation is about -1e12, below HELEK1, so
      ! nucleation runs. FAST_SBM cannot pass such a spectrum: it passes
      ! JERNUCL01_KS no crystals (FF2R is zeroed at :4548-4550 and after every
      ! nucleation call at :4667-4674).
      xi_c = xi
      psi1 = 0.0; psi2 = 0.0; fccnr = 0.0
      psi2(1, 3) = real(7.5d0/(3.0d0*real(col, 8)*real(xi(1, 3), 8)))
      psi2(2, 3) = real(1.0d17/(3.0d0*real(col, 8)*real(xi(2, 3), 8)))
      psi2(3, 3) = real(-1.00001d17/(3.0d0*real(col, 8)*real(xi(3, 3), 8)))
      tt = 258.15d0; qq = 0.004d0; sup1 = -1.0d-2; sup2 = 0.6d0
      call fsbm_begin(s, 'ice_nucl')
      call fsbm_put(s, 'tt', tt)
      call fsbm_put(s, 'sup1', sup1)
      call fsbm_put(s, 'sup2', sup2)
      call fsbm_put(s, 'xi', xi_c)
      call fsbm_put(s, 'psi2', psi2)
      call fsbm_end(s)
      call fsbm_close(s)
      call announce_cold(n, 'JERNUCL01_KS/ICE_NUCL')
      call jernucl01_ks(psi1, psi2, fccnr, xl, xi_c, tt, qq, 0.001, 8.0e5, sup1, sup2, &
                        col, 0.0, 0.0, rccn, dropradii, nb, na, 3, 1, &
                        200.0, 0, ro_solute, ions, mwaero, 1, 1, 1)
    case default
      call fsbm_fail('unknown cold refusal case')
    end select
    write(0,'(a,i0)') 'FSBM_REFUSAL_RETURNED ', n
    error stop 3
  end subroutine refusal_cold_case

  subroutine announce_cold(n, name)
    integer, intent(in) :: n
    character(len=*), intent(in) :: name
    write(0,'(a,i0,1x,a)') 'FSBM_REFUSAL_CALLING ', n, name
    flush(0)
  end subroutine announce_cold

end program run_sweeps
