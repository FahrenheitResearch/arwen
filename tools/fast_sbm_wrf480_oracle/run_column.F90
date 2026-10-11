! Warm kinematic column KC-1 for the fast SBM oracle (design 5.4).
!
! A tile of four columns called together through FAST_SBM, so the
! cross-column behaviour (findings D1, D5) is inside the fixture:
!   c1  land (continental)  1000 hPa  updraft w1 sin(pi z/H), w1 = 2 m/s, 900 s
!   c2  water (maritime)    1005 hPa  as c1
!   c3  water                995 hPa  still air; Marshall-Palmer rain for
!                                     20 mm/h in bins 15-33 above 1.2 km
!   c4  land                 850 hPa  w1 = 0.3 m/s for 1800 s (last in tile)
! 40 levels, lowest layer 50 m, each layer 2.5 percent deeper (top 3369 m);
! T = Tsfc - 5.5 K/km (Tsfc 303 K, c4 296 K); RH 92 percent below 800 m, 80
! to 1.8 km, 55 above (c3: 60 below 1.2 km); dt 12 s, 300 steps.
!
! Sounding adjustment (the design lets the oracle worker tune the sounding
! inside the case's acceptance limits): c4 carries a moist cap, RH 99 percent
! above 2.8 km (levels 37-40). Reason: FAST_SBM's cloud-base test (:4387)
!   if(SUP_WATER > 0.0D0 .and. w_stag_my > 0.1*1.0D2 .and. K_found .eqv.
!      .FALSE. .and. K > 2 .and. zcgs(I,K,J) < 3.0*1.0D5)
! parses as (SUP_WATER > 0 .and. w > 10 cm/s .and. K_found) .eqv.
! (.FALSE. .and. ...), because .EQV. binds more loosely than .AND.; the right
! side is always false, so KZ_Cloud_Base ends as the HIGHEST level that is not
! (supersaturated with w above 10 cm/s), not the lowest level that is. With
! the design's sounding that level is always an unsaturated one and the
! cloud-base activation branch (Cloud_Base_Super) is never reached. In c4 the
! top level's w stays below 10 cm/s (0.3 m/s times sin(pi z/H) at the top
! face), so c4's top level is its KZ at every call; the moist cap lets that
! level saturate under the forcing, which takes the cloud-base branch there.
! With 99 percent the top level first saturates at step 40, which is both a
! replay step and a probe step, and level 40 is one of the probe levels.
! The finding is recorded in the manifest (case record fields rh_cap_*).
!
! Between calls theta, qv and all 142 bins (the advected WRF form in chem) are
! advected by first-order upwind with fixed density and zero flux through the
! bottom and the top, th_old and qv_old are not. The operator is the flux
! form with its divergence term restored,
!   g(k)   = 0.5*(rho(k)+rho(k+1))*w(k+1)          face k (k = 1..39)
!   up(k)  = phi(k) if g(k) >= 0 else phi(k+1)      g(0) = g(40) = 0
!   ten(k) = -((g(k)*(up(k) - phi(k))) - (g(k-1)*(up(k-1) - phi(k)))) / (rho(k)*dz(k))
!   phi(k) = phi(k) + dt*ten(k)
! evaluated in this order in the build's default real. The plain flux form
! the design names (ten = -(F(k) - F(k-1))/(rho dz), F = g*up) is not
! consistent with a fixed density and a height-dependent w: it empties the
! lowest level at the rate w(face 1)/dz(1) and piles everything at the top.
! Measured here on c1's theta over the 900 s updraft it cools the lowest
! level to the value recorded in kc1_case (record literal_advection_check),
! which would break the case's own acceptance (T > 275.15 K); the restored
! divergence term is what keeps a constant field constant.
!
! Modes:
!   run_column kc1 <outdir> probes|noprobes   the case; writes kc1_case,
!       kc1_summary (every step), kc1_replay (pre/post at the replay steps),
!       kc1_control (post every step, for the inertness and determinism
!       controls), kc1_handoff.f64 (pre-call states for the cross-replay) and,
!       with probes, kc1_probe
!   run_column sensitivity <outdir>           the same run with every initial
!       theta moved up by one ULP (nearest(th, +1)); writes kc1_sensitivity
!   run_column crossreplay <outdir> <handoff> this build's FAST_SBM once on
!       each pre-call state in <handoff> (the native build's); writes
!       kc1_crossreplay
! Run where ./SBM_input_33 exists.
program run_column
  use module_mp_fast_sbm, only: fast_sbm, fast_hucminit
  use fsbm_io
  use fsbm_probe_state, only: fsbm_probe_configure, fsbm_probe_finish, pr_cnt, &
       PR_NCNT, CNT_NAME
  implicit none

  integer, parameter :: nx = 4, nz = 40, nchem = 143, nsteps = 300
  integer, parameter :: ids = 1, ide = 5, jds = 1, jde = 2, kds = 1, kde = 41
  integer, parameter :: ims = 0, ime = 5, jms = 0, jme = 2, kms = 1, kme = 41
  integer, parameter :: its = 1, ite = 4, jts = 1, jte = 1, kts = 1, kte = 40
  integer, parameter :: nrep = 16, nprb = 6, nlev = 8
  integer, parameter :: rep_steps(nrep) = (/ 2, 3, 4, 6, 10, 15, 25, 40, 60, 80, &
                                             100, 130, 160, 200, 250, 300 /)
  integer, parameter :: prb_steps(nprb) = (/ 10, 40, 80, 130, 200, 300 /)
  integer, parameter :: prb_levs(nlev) = (/ 1, 5, 10, 15, 20, 25, 30, 40 /)
  real(8), parameter :: pi_d = 3.141592653589793d0
  real(8), parameter :: rd = 287.0d0, cp = 1004.5d0, grav = 9.81d0, p0 = 1.0d5
  real(8), parameter :: lapse = 0.0055d0
  ! c4 moist cap (see the header): RH rh_cap_value at and above rh_cap_z metres
  real(8), parameter :: rh_cap_z = 2800.0d0, rh_cap_value = 0.99d0
  integer, parameter :: rh_cap_column = 4

  real :: dt, dx, dy
  real, dimension(ims:ime, kms:kme, jms:jme) :: w, u, v, th_old, qv_old, qv, qc, qr, &
       qi, qs, qg, qnc, qnr, qni, qns, qng, qna, dz8w, rho, p, pii, th
  real :: chem(ims:ime, kms:kme, jms:jme, nchem), radar(ims:ime, kms:kme, jms:jme, 1)
  real, dimension(ims:ime, jms:jme) :: xland, xlat, xlong, rainnc, rainncv, snownc, &
       snowncv, graupelnc, graupelncv, sr
  integer :: ivgtyp(ims:ime, jms:jme)

  real(8) :: zf(nz+1), zm(nz), dzd(nz), hdom
  real(8) :: tsfc(nx), psfc(nx), w1(nx), t_on(nx), xland_d(nx)
  real(8) :: t_d(nx, nz), p_d(nx, nz), pii_d(nx, nz), th_d(nx, nz), rho_d(nx, nz), &
             qv_d(nx, nz), rh_d(nx, nz), rain_d(nx, nz, 33), w_prof(nx, nz+1)
  real :: xl_tab(33)

  character(len=32) :: mode, popt
  character(len=512) :: outdir, handoff
  type(fsbm_stream) :: s_case, s_sum, s_rep, s_ctl, s_x
  logical :: use_probes

  call get_command_argument(1, mode)
  call get_command_argument(2, outdir)
  if (len_trim(outdir) == 0) call fsbm_fail('usage: run_column kc1|sensitivity|crossreplay <outdir> ...')
  dt = 12.0
  dx = 1000.0
  dy = 1000.0

  call read_masses(xl_tab)
  call define_case()
  call fast_hucminit(dt)

  select case (trim(mode))
  case ('kc1')
    call get_command_argument(3, popt)
    use_probes = trim(popt) == 'probes'
    call run_case(.false., use_probes)
  case ('sensitivity')
    call run_case(.true., .false.)
  case ('crossreplay')
    call get_command_argument(3, handoff)
    call cross_replay(trim(handoff))
  case default
    call fsbm_fail('unknown mode '//trim(mode))
  end select

contains

  ! Drop bin masses as FAST_HUCMINIT reads them (masses33.asc, FORMAT 6E13.5,
  ! first 33 values), so the pristine build needs no private module access.
  subroutine read_masses(xlt)
    real, intent(out) :: xlt(33)
    integer :: u, st
    open(newunit=u, file='SBM_input_33/masses33.asc', form='formatted', status='old', iostat=st)
    if (st /= 0) call fsbm_fail('cannot open SBM_input_33/masses33.asc')
    read(u, '(6E13.5)') xlt
    close(u)
  end subroutine read_masses

  subroutine define_case()
    integer :: c, k, b
    real(8) :: es, e, pc, lam, d, r, nk, rhoc
    do k = 1, nz
      dzd(k) = 50.0d0*1.025d0**(k-1)
    end do
    zf(1) = 0.0d0
    do k = 1, nz
      zf(k+1) = zf(k) + dzd(k)
      zm(k) = zf(k) + 0.5d0*dzd(k)
    end do
    hdom = zf(nz+1)
    tsfc = (/ 303.0d0, 303.0d0, 303.0d0, 296.0d0 /)
    psfc = (/ 1000.0d2, 1005.0d2, 995.0d2, 850.0d2 /)
    w1 = (/ 2.0d0, 2.0d0, 0.0d0, 0.3d0 /)
    t_on = (/ 900.0d0, 900.0d0, 0.0d0, 1800.0d0 /)
    xland_d = (/ 1.0d0, 2.0d0, 2.0d0, 1.0d0 /)
    lam = 41.0d0*20.0d0**(-0.21d0)
    rain_d = 0.0d0
    do c = 1, nx
      do k = 1, nz
        t_d(c, k) = tsfc(c) - lapse*zm(k)
        p_d(c, k) = psfc(c)*(t_d(c, k)/tsfc(c))**(grav/(rd*lapse))
        pii_d(c, k) = (p_d(c, k)/p0)**(rd/cp)
        th_d(c, k) = t_d(c, k)/pii_d(c, k)
        rho_d(c, k) = p_d(c, k)/(rd*t_d(c, k))
        if (c /= 3) then
          if (zm(k) < 800.0d0) then
            rh_d(c, k) = 0.92d0
          else if (zm(k) < 1800.0d0) then
            rh_d(c, k) = 0.80d0
          else if (c == 4 .and. zm(k) >= rh_cap_z) then
            rh_d(c, k) = rh_cap_value
          else
            rh_d(c, k) = 0.55d0
          end if
        else
          if (zm(k) < 1200.0d0) then
            rh_d(c, k) = 0.60d0
          else if (zm(k) < 1800.0d0) then
            rh_d(c, k) = 0.80d0
          else
            rh_d(c, k) = 0.55d0
          end if
        end if
        es = 2.53d12*exp(-5.42d3/t_d(c, k))
        e = rh_d(c, k)*es
        pc = p_d(c, k)*10.0d0
        qv_d(c, k) = 0.622d0*e/(pc - 0.378d0*e)
        if (c == 3 .and. zm(k) > 1200.0d0) then
          rhoc = rho_d(c, k)*1.0d-3
          do b = 15, 33
            r = (3.0d0*real(xl_tab(b), 8)/(4.0d0*pi_d))**(1.0d0/3.0d0)
            d = 2.0d0*r
            nk = 0.08d0*exp(-lam*d)*d*0.23105d0
            rain_d(c, k, b) = nk*real(xl_tab(b), 8)/rhoc
          end do
        end if
      end do
      do k = 1, nz+1
        w_prof(c, k) = w1(c)*sin(pi_d*zf(k)/hdom)
      end do
      w_prof(c, 1) = 0.0d0
      w_prof(c, nz+1) = 0.0d0
    end do
  end subroutine define_case

  subroutine init_state(nudge)
    logical, intent(in) :: nudge
    integer :: c, k, b
    w = 0.0; u = 0.0; v = 0.0; th_old = 0.0; qv_old = 0.0; qv = 0.0
    qc = 0.0; qr = 0.0; qi = 0.0; qs = 0.0; qg = 0.0
    qnc = 0.0; qnr = 0.0; qni = 0.0; qns = 0.0; qng = 0.0; qna = 0.0
    dz8w = 1.0; rho = 1.0; p = 1.0e5; pii = 1.0; th = 300.0
    chem = 0.0; radar = 0.0
    xland = 1.0; xlat = 0.0; xlong = 0.0; ivgtyp = 16
    rainnc = 0.0; rainncv = 0.0; snownc = 0.0; snowncv = 0.0
    graupelnc = 0.0; graupelncv = 0.0; sr = 0.0
    do c = 1, nx
      xland(c, 1) = real(xland_d(c))
      do k = 1, nz
        dz8w(c, k, 1) = real(dzd(k))
        rho(c, k, 1) = real(rho_d(c, k))
        p(c, k, 1) = real(p_d(c, k))
        pii(c, k, 1) = real(pii_d(c, k))
        th(c, k, 1) = real(th_d(c, k))
        qv(c, k, 1) = real(qv_d(c, k))
        do b = 15, 33
          chem(c, k, 1, 1+b) = real(rain_d(c, k, b))
        end do
      end do
      dz8w(c, kme, 1) = dz8w(c, kte, 1)
    end do
    if (nudge) then
      do c = 1, nx
        do k = 1, nz
          th(c, k, 1) = nearest(th(c, k, 1), 1.0)
        end do
      end do
    end if
    th_old = th
    qv_old = qv
  end subroutine init_state

  subroutine set_w(tsec)
    real(8), intent(in) :: tsec
    integer :: c, k
    do c = 1, nx
      do k = 1, nz+1
        if (tsec < t_on(c)) then
          w(c, k, 1) = real(w_prof(c, k))
        else
          w(c, k, 1) = 0.0
        end if
      end do
    end do
  end subroutine set_w

  subroutine advect_column(phi, wc, rhoc, dzc)
    real, intent(inout) :: phi(nz)
    real, intent(in) :: wc(nz+1), rhoc(nz), dzc(nz)
    real :: g(0:nz), up(0:nz), old(nz), ten
    integer :: k
    old = phi
    g(0) = 0.0
    up(0) = 0.0
    g(nz) = 0.0
    up(nz) = 0.0
    do k = 1, nz-1
      g(k) = 0.5*(rhoc(k)+rhoc(k+1))*wc(k+1)
      if (g(k) >= 0.0) then
        up(k) = old(k)
      else
        up(k) = old(k+1)
      end if
    end do
    do k = 1, nz
      ten = -((g(k)*(up(k) - old(k))) - (g(k-1)*(up(k-1) - old(k)))) / (rhoc(k)*dzc(k))
      phi(k) = old(k) + dt*ten
    end do
  end subroutine advect_column

  subroutine advect_all()
    integer :: c, m
    real :: col(nz)
    do c = 1, nx
      col = th(c, kts:kte, 1)
      call advect_column(col, w(c, 1:nz+1, 1), rho(c, kts:kte, 1), dz8w(c, kts:kte, 1))
      th(c, kts:kte, 1) = col
      col = qv(c, kts:kte, 1)
      call advect_column(col, w(c, 1:nz+1, 1), rho(c, kts:kte, 1), dz8w(c, kts:kte, 1))
      qv(c, kts:kte, 1) = col
      do m = 2, nchem
        col = chem(c, kts:kte, 1, m)
        call advect_column(col, w(c, 1:nz+1, 1), rho(c, kts:kte, 1), dz8w(c, kts:kte, 1))
        chem(c, kts:kte, 1, m) = col
      end do
    end do
  end subroutine advect_all

  ! The design's plain flux form on c1's theta, dry, over the 900 s updraft,
  ! beside the operator the case uses; both in double.
  subroutine literal_advection_check(st)
    type(fsbm_stream), intent(inout) :: st
    real(8) :: a(nz), b(nz), g(0:nz), up(0:nz), f(0:nz), olda(nz), oldb(nz), ta, tb
    integer :: n, k, nst
    a = th_d(1, :)
    b = th_d(1, :)
    nst = nint(t_on(1)/12.0d0)
    do n = 1, nst
      olda = a
      oldb = b
      g = 0.0d0
      do k = 1, nz-1
        g(k) = 0.5d0*(rho_d(1, k)+rho_d(1, k+1))*w_prof(1, k+1)
      end do
      up = 0.0d0
      f = 0.0d0
      do k = 1, nz-1
        up(k) = olda(k)
        f(k) = g(k)*up(k)
      end do
      do k = 1, nz
        a(k) = olda(k) - 12.0d0*(f(k) - f(k-1))/(rho_d(1, k)*dzd(k))
      end do
      up = 0.0d0
      do k = 1, nz-1
        up(k) = oldb(k)
      end do
      do k = 1, nz
        b(k) = oldb(k) + 12.0d0*(-((g(k)*(up(k) - oldb(k))) - (g(k-1)*(up(k-1) - oldb(k)))) &
               /(rho_d(1, k)*dzd(k)))
      end do
    end do
    ta = minval(a*pii_d(1, :))
    tb = minval(b*pii_d(1, :))
    call fsbm_begin(st, 'literal_advection_check')
    call fsbm_put(st, 'steps', int(nst, 4))
    call fsbm_put(st, 'th_initial', th_d(1, :))
    call fsbm_put(st, 'th_plain_flux_form', a)
    call fsbm_put(st, 'th_case_operator', b)
    call fsbm_put(st, 'tmin_plain_flux_form', ta)
    call fsbm_put(st, 'tmin_case_operator', tb)
    call fsbm_end(st)
  end subroutine literal_advection_check

  subroutine write_case(st)
    type(fsbm_stream), intent(inout) :: st
    integer :: c
    call fsbm_begin(st, 'case')
    call fsbm_put(st, 'dt', real(dt, 8))
    call fsbm_put(st, 'nsteps', int(nsteps, 4))
    call fsbm_put(st, 'z_face', zf)
    call fsbm_put(st, 'z_mass', zm)
    call fsbm_put(st, 'dz', dzd)
    call fsbm_put(st, 't_surface', tsfc)
    call fsbm_put(st, 'p_surface', psfc)
    call fsbm_put(st, 'w1', w1)
    call fsbm_put(st, 'w_on_seconds', t_on)
    call fsbm_put(st, 'xland', xland_d)
    call fsbm_put(st, 'temperature', t_d)
    call fsbm_put(st, 'pressure', p_d)
    call fsbm_put(st, 'exner', pii_d)
    call fsbm_put(st, 'theta', th_d)
    call fsbm_put(st, 'rho', rho_d)
    call fsbm_put(st, 'relative_humidity', rh_d)
    call fsbm_put(st, 'rh_cap_column', int(rh_cap_column, 4))
    call fsbm_put(st, 'rh_cap_height', rh_cap_z)
    call fsbm_put(st, 'rh_cap_value', rh_cap_value)
    call fsbm_put(st, 'qv', qv_d)
    call fsbm_put(st, 'w_profile_full_levels', w_prof)
    call fsbm_put(st, 'rain_bins_mass_mixing_ratio', rain_d)
    call fsbm_put(st, 'xl_used_for_rain', xl_tab)
    call fsbm_put(st, 'replay_steps', int(rep_steps, 4))
    call fsbm_put(st, 'probe_steps', int(prb_steps, 4))
    call fsbm_put(st, 'probe_levels', int(prb_levs, 4))
    call fsbm_end(st)

  end subroutine write_case

  subroutine write_initial(st)
    type(fsbm_stream), intent(inout) :: st
    call fsbm_begin(st, 'initial')
    call fsbm_put(st, 'dz8w', dz8w(its:ite, kts:kte, 1))
    call fsbm_put(st, 'rho', rho(its:ite, kts:kte, 1))
    call fsbm_put(st, 'p', p(its:ite, kts:kte, 1))
    call fsbm_put(st, 'pii', pii(its:ite, kts:kte, 1))
    call fsbm_put(st, 'th', th(its:ite, kts:kte, 1))
    call fsbm_put(st, 'qv', qv(its:ite, kts:kte, 1))
    call fsbm_put(st, 'xland', xland(its:ite, 1))
    call fsbm_put(st, 'chem', chem(its:ite, kts:kte, 1, 2:nchem))
    call fsbm_end(st)
  end subroutine write_initial

  subroutine write_pre(st, n)
    type(fsbm_stream), intent(inout) :: st
    integer, intent(in) :: n
    call fsbm_begin(st, 'pre')
    call fsbm_put(st, 'step', int(n, 4))
    call fsbm_put(st, 'w', w(its:ite, kts:kte+1, 1))
    call fsbm_put(st, 'th', th(its:ite, kts:kte, 1))
    call fsbm_put(st, 'qv', qv(its:ite, kts:kte, 1))
    call fsbm_put(st, 'th_old', th_old(its:ite, kts:kte, 1))
    call fsbm_put(st, 'qv_old', qv_old(its:ite, kts:kte, 1))
    call fsbm_put(st, 'chem', chem(its:ite, kts:kte, 1, 2:nchem))
    call fsbm_put(st, 'rainnc', rainnc(its:ite, 1))
    call fsbm_put(st, 'snownc', snownc(its:ite, 1))
    call fsbm_put(st, 'graupelnc', graupelnc(its:ite, 1))
    call fsbm_put(st, 'sr', sr(its:ite, 1))
    call fsbm_end(st)
  end subroutine write_pre

  subroutine write_post(st, n)
    type(fsbm_stream), intent(inout) :: st
    integer, intent(in) :: n
    call fsbm_begin(st, 'post')
    call fsbm_put(st, 'step', int(n, 4))
    call fsbm_put(st, 'th', th(its:ite, kts:kte, 1))
    call fsbm_put(st, 'qv', qv(its:ite, kts:kte, 1))
    call fsbm_put(st, 'th_old', th_old(its:ite, kts:kte, 1))
    call fsbm_put(st, 'qv_old', qv_old(its:ite, kts:kte, 1))
    call fsbm_put(st, 'chem', chem(its:ite, kts:kte, 1, 2:nchem))
    call fsbm_put(st, 'qc', qc(its:ite, kts:kte, 1))
    call fsbm_put(st, 'qr', qr(its:ite, kts:kte, 1))
    call fsbm_put(st, 'qi', qi(its:ite, kts:kte, 1))
    call fsbm_put(st, 'qs', qs(its:ite, kts:kte, 1))
    call fsbm_put(st, 'qg', qg(its:ite, kts:kte, 1))
    call fsbm_put(st, 'qnc', qnc(its:ite, kts:kte, 1))
    call fsbm_put(st, 'qnr', qnr(its:ite, kts:kte, 1))
    call fsbm_put(st, 'qni', qni(its:ite, kts:kte, 1))
    call fsbm_put(st, 'qns', qns(its:ite, kts:kte, 1))
    call fsbm_put(st, 'qng', qng(its:ite, kts:kte, 1))
    call fsbm_put(st, 'qna', qna(its:ite, kts:kte, 1))
    call fsbm_put(st, 'rainnc', rainnc(its:ite, 1))
    call fsbm_put(st, 'rainncv', rainncv(its:ite, 1))
    call fsbm_put(st, 'snownc', snownc(its:ite, 1))
    call fsbm_put(st, 'snowncv', snowncv(its:ite, 1))
    call fsbm_put(st, 'graupelnc', graupelnc(its:ite, 1))
    call fsbm_put(st, 'graupelncv', graupelncv(its:ite, 1))
    call fsbm_put(st, 'sr', sr(its:ite, 1))
    call fsbm_end(st)
  end subroutine write_post

  subroutine write_step(st, n, tmin)
    type(fsbm_stream), intent(inout) :: st
    integer, intent(in) :: n
    real, intent(in) :: tmin(nx)
    call fsbm_begin(st, 'step')
    call fsbm_put(st, 'step', int(n, 4))
    call fsbm_put(st, 'time', real(n, 8)*real(dt, 8))
    call fsbm_put(st, 'rainnc', rainnc(its:ite, 1))
    call fsbm_put(st, 'rainncv', rainncv(its:ite, 1))
    call fsbm_put(st, 'tmin', tmin)
    call fsbm_put(st, 'th', th(its:ite, kts:kte, 1))
    call fsbm_put(st, 'qv', qv(its:ite, kts:kte, 1))
    call fsbm_put(st, 'qc', qc(its:ite, kts:kte, 1))
    call fsbm_put(st, 'qr', qr(its:ite, kts:kte, 1))
    call fsbm_put(st, 'qnc', qnc(its:ite, kts:kte, 1))
    call fsbm_end(st)
  end subroutine write_step

  subroutine write_handoff_statics(u)
    integer, intent(in) :: u
    write(u) real(dz8w(its:ite, kts:kte, 1), 8), real(rho(its:ite, kts:kte, 1), 8), &
             real(p(its:ite, kts:kte, 1), 8), real(pii(its:ite, kts:kte, 1), 8), &
             real(xland(its:ite, 1), 8)
  end subroutine write_handoff_statics

  subroutine write_handoff_state(u, n)
    integer, intent(in) :: u, n
    write(u) int(n, 4)
    write(u) real(w(its:ite, kts:kte+1, 1), 8), real(th(its:ite, kts:kte, 1), 8), &
             real(qv(its:ite, kts:kte, 1), 8), real(th_old(its:ite, kts:kte, 1), 8), &
             real(qv_old(its:ite, kts:kte, 1), 8), &
             real(chem(its:ite, kts:kte, 1, 2:nchem), 8), real(rainnc(its:ite, 1), 8), &
             real(snownc(its:ite, 1), 8), real(graupelnc(its:ite, 1), 8), &
             real(sr(its:ite, 1), 8)
  end subroutine write_handoff_state

  subroutine call_sbm(n)
    integer, intent(in) :: n
    call fast_sbm(w, u, v, th_old, chem, nchem, n, dt, dx, dy, dz8w, rho, p, pii, th, &
                  xland, 1, ivgtyp, xlat, xlong, qv, qc, qr, qi, qs, qg, qv_old, &
                  qnc, qnr, qni, qns, qng, qna, ids, ide, jds, jde, kds, kde, &
                  ims, ime, jms, jme, kms, kme, its, ite, jts, jte, kts, kte, &
                  sbmradar=radar, num_sbmradar=1, sbm_diagnostics=0, &
                  RAINNC=rainnc, RAINNCV=rainncv, SNOWNC=snownc, SNOWNCV=snowncv, &
                  GRAUPELNC=graupelnc, GRAUPELNCV=graupelncv, SR=sr)
  end subroutine call_sbm

  subroutine tile_tmin(tmin)
    real, intent(out) :: tmin(nx)
    integer :: c
    do c = 1, nx
      tmin(c) = minval(th(c, kts:kte, 1)*pii(c, kts:kte, 1))
    end do
  end subroutine tile_tmin

  subroutine run_case(nudge, probes)
    logical, intent(in) :: nudge, probes
    integer :: n, c, uh, cnt
    real :: tmin(nx), tmin_pre(nx), tmin_post(nx), tmin_run_pre(nx), tmin_run_post(nx)
    integer(4) :: rain_steps(nx)
    real(8) :: tsec
    character(len=64) :: sumname

    call init_state(nudge)
    if (nudge) then
      sumname = 'kc1_sensitivity'
    else
      sumname = 'kc1_summary'
      call fsbm_open(s_case, trim(outdir)//'/kc1_case')
      call write_case(s_case)
      call literal_advection_check(s_case)
      call fsbm_close(s_case)
      call fsbm_open(s_rep, trim(outdir)//'/kc1_replay')
      call fsbm_open(s_ctl, trim(outdir)//'/kc1_control')
      open(newunit=uh, file=trim(outdir)//'/kc1_handoff.f64', access='stream', &
           form='unformatted', status='replace', action='write')
      call write_handoff_statics(uh)
    end if
    call fsbm_open(s_sum, trim(outdir)//'/'//trim(sumname))
    call write_initial(s_sum)
    call fsbm_probe_configure(trim(outdir)//'/kc1_probe', prb_steps, prb_levs, &
                              probes, probes, .false.)

    tmin_run_pre = 1.0e30
    tmin_run_post = 1.0e30
    rain_steps = 0
    do n = 1, nsteps
      tsec = real(n-1, 8)*real(dt, 8)
      call set_w(tsec)
      if (n > 1) call advect_all()
      call tile_tmin(tmin_pre)
      tmin_run_pre = min(tmin_run_pre, tmin_pre)
      if (.not. nudge .and. any(rep_steps == n)) then
        call write_pre(s_rep, n)
        call write_handoff_state(uh, n)
      end if
      call call_sbm(n)
      call tile_tmin(tmin_post)
      tmin_run_post = min(tmin_run_post, tmin_post)
      do c = 1, nx
        if (rainncv(c, 1) > 0.0) rain_steps(c) = rain_steps(c) + 1
      end do
      call write_step(s_sum, n, tmin_post)
      if (.not. nudge) then
        call write_post(s_ctl, n)
        if (any(rep_steps == n)) call write_post(s_rep, n)
      end if
    end do

    call fsbm_begin(s_sum, 'acceptance')
    call fsbm_put(s_sum, 'tmin_run_pre', tmin_run_pre)
    call fsbm_put(s_sum, 'tmin_run_post', tmin_run_post)
    call fsbm_put(s_sum, 'rainnc_final', rainnc(its:ite, 1))
    call fsbm_put(s_sum, 'rain_steps', rain_steps)
    do cnt = 1, PR_NCNT
      call fsbm_put(s_sum, trim(CNT_NAME(cnt)), pr_cnt(:, cnt))
    end do
    call fsbm_end(s_sum)
    call fsbm_close(s_sum)
    call fsbm_probe_finish()
    if (.not. nudge) then
      call fsbm_close(s_rep)
      call fsbm_close(s_ctl)
      close(uh)
    end if
    tmin = tmin_run_post
  end subroutine run_case

  subroutine cross_replay(path)
    character(len=*), intent(in) :: path
    integer :: uh, st, n, i
    real(8) :: a_dz(nx, nz), a_rho(nx, nz), a_p(nx, nz), a_pii(nx, nz), a_xl(nx)
    real(8) :: a_w(nx, nz+1), a_th(nx, nz), a_qv(nx, nz), a_tho(nx, nz), a_qvo(nx, nz), &
               a_chem(nx, nz, nchem-1), a_rn(nx), a_sn(nx), a_gn(nx), a_sr(nx)
    integer(4) :: step4
    call init_state(.false.)
    open(newunit=uh, file=path, access='stream', form='unformatted', status='old', &
         action='read', iostat=st)
    if (st /= 0) call fsbm_fail('cannot open handoff '//path)
    read(uh) a_dz, a_rho, a_p, a_pii, a_xl
    dz8w(its:ite, kts:kte, 1) = a_dz
    dz8w(its:ite, kme, 1) = dz8w(its:ite, kte, 1)
    rho(its:ite, kts:kte, 1) = a_rho
    p(its:ite, kts:kte, 1) = a_p
    pii(its:ite, kts:kte, 1) = a_pii
    xland(its:ite, 1) = a_xl
    call fsbm_open(s_x, trim(outdir)//'/kc1_crossreplay')
    do i = 1, nrep
      read(uh, iostat=st) step4
      if (st /= 0) call fsbm_fail('handoff ended early')
      read(uh) a_w, a_th, a_qv, a_tho, a_qvo, a_chem, a_rn, a_sn, a_gn, a_sr
      n = step4
      if (n /= rep_steps(i)) call fsbm_fail('handoff step order differs from the replay steps')
      w(its:ite, kts:kte+1, 1) = a_w
      th(its:ite, kts:kte, 1) = a_th
      qv(its:ite, kts:kte, 1) = a_qv
      th_old(its:ite, kts:kte, 1) = a_tho
      qv_old(its:ite, kts:kte, 1) = a_qvo
      chem(its:ite, kts:kte, 1, 2:nchem) = a_chem
      rainnc(its:ite, 1) = a_rn
      snownc(its:ite, 1) = a_sn
      graupelnc(its:ite, 1) = a_gn
      sr(its:ite, 1) = a_sr
      call call_sbm(n)
      call write_post(s_x, n)
    end do
    call fsbm_close(s_x)
    close(uh)
  end subroutine cross_replay

end program run_column
