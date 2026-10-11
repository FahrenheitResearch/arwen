! Cold kinematic column KC-2 for the fast SBM oracle (milestone 2).
!
! A 3 x 3 tile (i, j = 2..4) in the corner of a 4 x 4 domain (ids..ide = 1..5,
! jds..jde = 1..5), called together through FAST_SBM, so the ice pre-pass
! (:4373-4512) reads a halo row and column (i = 1 and j = 1, never called)
! and takes its interior branches (i, j = 2, 3) and its IDE-1 / JDE-1 edge
! branches (i, j = 4). Four of the nine tile columns carry the cold physics:
!   A (2,2) deep mixed-phase updraft: 300 K surface, 7 K/km, w1 = 4 m/s
!           sin(pi z / H) for the whole hour, ice-saturated air aloft
!   B (3,3) Hallett-Mossop updraft: 298 K, 6.5 K/km, w1 = 1.5 m/s for 1800 s,
!           a supercooled cloud with a drizzle and rain tail, dense ice
!           between 4.2 and 5.5 km (-2.45 to -10.9 C on the sounding, so
!           across the -3 to -8 C Hallett-Mossop band), snow above it
!   C (4,4) melting stratiform column: 275.5 K surface (0 C near 390 m),
!           6 K/km, w1 = 0.2 m/s, snow between 1.5 and 5 km falling through
!           0 C to the ground; last in the tile (the D1 column)
!   D (4,2) hail shaft: 299 K, 7 K/km, a 0.8 m/s downdraft, large dense ice
!           between 5 and 8 km over a supercooled cloud and light snow
! The other five tile columns and the seven halo columns are the environment:
! still, subsaturated over ice, no hydrometeors, with a surface temperature
! falling 0.8 K per column in x and 0.5 K in y and a humidity falling in y.
! u and v are prescribed and vary with i, j and k, so the pre-pass averages
! (:4411-4417) differ from its edge values; they enter only the pre-pass.
!
! 50 levels, lowest 134 m, each 2.5 percent deeper (top 13.06 km); 12 s steps,
! 300 steps. Between calls theta, qv and all 142 bins of the tile columns are
! advected vertically by KC-1's operator (the flux form with its divergence
! term restored, first-order upwind, fixed density, zero flux at the bottom
! and top); th_old and qv_old are not advected. The halo columns never
! change: FAST_SBM never writes them after its first call, and the first call
! sets their th_old and qv_old to their th and qv, which they already hold.
!
! Modes:
!   run_kc2 kc2|kc2g <outdir> probes|noprobes   the case in a hail (kc2) or
!       graupel (kc2g) build; writes <mode>_case, <mode>_summary (a record
!       after every call, profiles every 30 calls in kc2, and the acceptance
!       record with the per-column counts), <mode>_replay (pre and post at
!       the replay steps: 7 in kc2, 3 in kc2g), <mode>_control (the post
!       state after every call, for the inertness and determinism controls),
!       <mode>_handoff.f64 and, with probes, <mode>_probe (4 steps, 8 levels,
!       the four cold columns)
!   run_kc2 sensitivity <outdir>                kc2 with every initial theta
!       one ULP higher (nearest(th, +1)); writes kc2_sensitivity
!   run_kc2 crossreplay <outdir> <handoff> [kc2|kc2g]   this build's FAST_SBM
!       once on each pre-call state in <handoff>; writes <mode>_crossreplay
!   run_kc2 kc2edge <outdir> probes|noprobes   the low-edge tile: the same
!       domain with the tile moved to i, j = 1..3 and a Hallett-Mossop column
!       (type B) in the corner (1, 1), for 6 calls, so the pre-pass takes its
!       I = 1 and J = 1 branches: DERIVT_X and DERIVT_Y (:4422-4423,
!       :4429-4430), and the DSUPICE_XYZ differences (:4460-4462, :4468-4469,
!       :4483-4485, :4491-4492), which need 2 percent ice supersaturation below
!       -5 C at the point itself (:4441); KC-2's environment columns never
!       reach it. Writes kc2edge_replay (the static fields, then
!       pre and post at steps 2, 4 and 6), kc2edge_control (the post state
!       after every call) and, with probes, kc2edge_probe (the pre-pass arrays
!       of the tile at steps 2, 4 and 6, and the per-column counts; no
!       per-column records)
! Run where ./SBM_input_33 exists.
program run_kc2
  use module_mp_fast_sbm, only: fast_sbm, fast_hucminit
  use fsbm_io
  use fsbm_probe_state, only: fsbm_probe_configure, fsbm_probe_configure_tile, &
       fsbm_probe_finish, pr_cnt2, PR_NCNT2, CNT2_NAME
  implicit none

  integer, parameter :: nd = 4, nz = 50, nchem = 143, nsteps = 300
  integer, parameter :: ids = 1, ide = 5, jds = 1, jde = 5, kds = 1, kde = 51
  integer, parameter :: ims = 0, ime = 6, jms = 0, jme = 6, kms = 1, kme = 51
  integer, parameter :: kts = 1, kte = 50
  ! The tile: i, j = 2..4 for KC-2, 1..3 for the low-edge run; 3 x 3 either way.
  integer :: its = 2, ite = 4, jts = 2, jte = 4
  integer, parameter :: ntc = 9
  integer, parameter :: nsteps_edge = 6, nrep_edge = 3
  integer, parameter :: rep_steps_edge(nrep_edge) = (/ 2, 4, 6 /)
  integer, parameter :: nrep = 7, nrep_g = 3, nprb = 4, nlev = 8, nprof = 30
  integer, parameter :: rep_steps(nrep) = (/ 2, 12, 30, 60, 120, 200, 300 /)
  integer, parameter :: rep_steps_g(nrep_g) = (/ 2, 150, 300 /)
  integer, parameter :: prb_steps(nprb) = (/ 10, 60, 150, 300 /)
  integer, parameter :: prb_levs(nlev) = (/ 5, 12, 18, 23, 27, 31, 36, 42 /)
  real(8), parameter :: pi_d = 3.141592653589793d0
  real(8), parameter :: rd = 287.0d0, cp = 1004.5d0, grav = 9.81d0, p0 = 1.0d5
  real(8), parameter :: col_d = 0.23105d0
  ! Column types: 0 environment, 1 A, 2 B, 3 C, 4 D.
  integer, parameter :: CT_ENV = 0, CT_A = 1, CT_B = 2, CT_C = 3, CT_D = 4

  real :: dt, dx, dy
  real, dimension(ims:ime, kms:kme, jms:jme) :: w, u, v, th_old, qv_old, qv, qc, qr, &
       qi, qs, qg, qnc, qnr, qni, qns, qng, qna, dz8w, rho, p, pii, th
  real :: chem(ims:ime, kms:kme, jms:jme, nchem), radar(ims:ime, kms:kme, jms:jme, 1)
  real, dimension(ims:ime, jms:jme) :: xland, xlat, xlong, rainnc, rainncv, snownc, &
       snowncv, graupelnc, graupelncv, sr
  integer :: ivgtyp(ims:ime, jms:jme)

  real(8) :: zf(nz+1), zm(nz), dzd(nz), hdom
  integer :: ctype(nd, nd)
  real(8) :: tsfc(nd, nd), lapse(nd, nd), psfc(nd, nd), w1(nd, nd), t_off(nd, nd), &
             xland_d(nd, nd)
  real(8) :: t_d(nd, nd, nz), p_d(nd, nd, nz), pii_d(nd, nd, nz), th_d(nd, nd, nz), &
             rho_d(nd, nd, nz), qv_d(nd, nd, nz), rh_d(nd, nd, nz), &
             u_d(nd, nd, nz), v_d(nd, nd, nz), w_prof(nd, nd, nz+1)
  real(8) :: bins_d(nd, nd, nz, 99)       ! drops 1-33, snow 34-66, dense ice 67-99 (kg/kg)
  real :: xl_tab(33)

  character(len=32) :: mode, popt, cmode
  character(len=512) :: outdir, handoff
  type(fsbm_stream) :: s_case, s_sum, s_rep, s_ctl, s_x
  logical :: use_probes, graupel, edge

  call get_command_argument(1, mode)
  call get_command_argument(2, outdir)
  if (len_trim(outdir) == 0) call fsbm_fail('usage: run_kc2 kc2|kc2g|sensitivity|crossreplay <outdir> ...')
  dt = 12.0
  dx = 1000.0
  dy = 1000.0

  edge = trim(mode) == 'kc2edge'
  if (edge) then
    its = 1; ite = 3; jts = 1; jte = 3
  end if
  call read_masses(xl_tab)
  call define_case()
  call fast_hucminit(dt)

  select case (trim(mode))
  case ('kc2', 'kc2g')
    graupel = trim(mode) == 'kc2g'
    call get_command_argument(3, popt)
    use_probes = trim(popt) == 'probes'
    call run_case(trim(mode), .false., use_probes)
  case ('sensitivity')
    graupel = .false.
    call run_case('kc2', .true., .false.)
  case ('kc2edge')
    graupel = .false.
    call get_command_argument(3, popt)
    call run_edge(trim(popt) == 'probes')
  case ('crossreplay')
    call get_command_argument(3, handoff)
    call get_command_argument(4, cmode)
    if (len_trim(cmode) == 0) cmode = 'kc2'
    graupel = trim(cmode) == 'kc2g'
    call cross_replay(trim(cmode), trim(handoff))
  case default
    call fsbm_fail('unknown mode '//trim(mode))
  end select

contains

  subroutine read_masses(xlt)
    real, intent(out) :: xlt(33)
    integer :: uu, st
    open(newunit=uu, file='SBM_input_33/masses33.asc', form='formatted', status='old', iostat=st)
    if (st /= 0) call fsbm_fail('cannot open SBM_input_33/masses33.asc')
    read(uu, '(6E13.5)') xlt
    close(uu)
  end subroutine read_masses

  real(8) function bin_radius(b)
    integer, intent(in) :: b
    bin_radius = (3.0d0*real(xl_tab(b), 8)/(4.0d0*pi_d))**(1.0d0/3.0d0)
  end function bin_radius

  ! Number per cm3 in bin b of a lognormal mode (nd cm-3, mode radius rm cm,
  ! geometric width sig), and of an exponential n0 exp(-lam d) in diameter
  ! (n0 cm-4, lam cm-1); both per bin of width COL in ln r.
  real(8) function n_lognormal(b, ntot, rm, sig)
    integer, intent(in) :: b
    real(8), intent(in) :: ntot, rm, sig
    real(8) :: ls
    ls = log(sig)
    n_lognormal = ntot/(sqrt(2.0d0*pi_d)*ls)*exp(-(log(bin_radius(b)/rm))**2/(2.0d0*ls*ls))*col_d
  end function n_lognormal

  real(8) function n_exponential(b, n0, lam)
    integer, intent(in) :: b
    real(8), intent(in) :: n0, lam
    real(8) :: d
    d = 2.0d0*bin_radius(b)
    n_exponential = n0*exp(-lam*d)*d*col_d
  end function n_exponential

  ! Marshall-Palmer slope for a rain rate in mm/h.
  real(8) function mp_lambda(rate)
    real(8), intent(in) :: rate
    mp_lambda = 41.0d0*rate**(-0.21d0)
  end function mp_lambda

  ! Saturation vapour pressures FAST_SBM uses (dyn/cm2).
  real(8) function esw(t)
    real(8), intent(in) :: t
    esw = 2.53d12*exp(-5.42d3/t)
  end function esw

  real(8) function esi(t)
    real(8), intent(in) :: t
    esi = 3.41d13*exp(-6.13d3/t)
  end function esi

  subroutine define_case()
    integer :: i, j, k, b, ty
    real(8) :: e, pc, rhoc, z, t, rhw, ramp
    do k = 1, nz
      dzd(k) = 134.0d0*1.025d0**(k-1)
    end do
    zf(1) = 0.0d0
    do k = 1, nz
      zf(k+1) = zf(k) + dzd(k)
      zm(k) = zf(k) + 0.5d0*dzd(k)
    end do
    hdom = zf(nz+1)
    ctype = CT_ENV
    ctype(2, 2) = CT_A
    ctype(3, 3) = CT_B
    ctype(4, 4) = CT_C
    ctype(4, 2) = CT_D
    if (edge) ctype(1, 1) = CT_B     ! the low-edge run's corner column
    bins_d = 0.0d0
    do j = 1, nd
      do i = 1, nd
        ty = ctype(i, j)
        select case (ty)
        case (CT_A)
          tsfc(i, j) = 300.0d0; lapse(i, j) = 0.0070d0; w1(i, j) = 4.0d0; t_off(i, j) = 1.0d9
          xland_d(i, j) = 1.0d0
        case (CT_B)
          tsfc(i, j) = 298.0d0; lapse(i, j) = 0.0065d0; w1(i, j) = 1.5d0; t_off(i, j) = 1800.0d0
          xland_d(i, j) = 1.0d0
        case (CT_C)
          tsfc(i, j) = 275.5d0; lapse(i, j) = 0.0060d0; w1(i, j) = 0.2d0; t_off(i, j) = 1.0d9
          xland_d(i, j) = 2.0d0
        case (CT_D)
          tsfc(i, j) = 299.0d0; lapse(i, j) = 0.0070d0; w1(i, j) = -0.8d0; t_off(i, j) = 1.0d9
          xland_d(i, j) = 1.0d0
        case default
          tsfc(i, j) = 299.0d0 - 0.8d0*(i-1) - 0.5d0*(j-1); lapse(i, j) = 0.0068d0
          w1(i, j) = 0.0d0; t_off(i, j) = 0.0d0
          xland_d(i, j) = real(1 + mod(i+j, 2), 8)
        end select
        psfc(i, j) = 1000.0d2
        do k = 1, nz
          z = zm(k)
          t = tsfc(i, j) - lapse(i, j)*z
          t_d(i, j, k) = t
          p_d(i, j, k) = psfc(i, j)*(t/tsfc(i, j))**(grav/(rd*lapse(i, j)))
          pii_d(i, j, k) = (p_d(i, j, k)/p0)**(rd/cp)
          th_d(i, j, k) = t/pii_d(i, j, k)
          rho_d(i, j, k) = p_d(i, j, k)/(rd*t)
          u_d(i, j, k) = 6.0d0 + 1.5d0*(i-1) + 0.5d0*(j-1) + 0.1d0*k
          v_d(i, j, k) = -2.0d0 + 0.5d0*(i-1) - 1.0d0*(j-1) + 0.05d0*k
          ! relative humidity over water
          select case (ty)
          case (CT_A)
            if (z < 1500.0d0) then
              rhw = 0.92d0
            else if (t >= 273.15d0) then
              rhw = 0.85d0
            else if (z < 11000.0d0) then
              rhw = esi(t)/esw(t)            ! ice-saturated
            else
              rhw = 0.70d0*esi(t)/esw(t)
            end if
          case (CT_B)
            if (z < 3500.0d0) then
              rhw = 0.93d0
            else if (z < 6500.0d0) then
              rhw = 1.0d0                    ! water-saturated cloud layer
            else
              rhw = 0.98d0*esi(t)/esw(t)
            end if
          case (CT_C)
            if (t >= 273.15d0) then
              rhw = 0.93d0
            else if (z < 5500.0d0) then
              rhw = esi(t)/esw(t)
            else
              rhw = 0.60d0*esi(t)/esw(t)
            end if
          case (CT_D)
            if (z < 3000.0d0) then
              rhw = 0.80d0
            else if (z < 4000.0d0) then
              rhw = 0.90d0
            else if (z < 6500.0d0) then
              rhw = 1.0d0
            else
              rhw = 0.97d0*esi(t)/esw(t)
            end if
          case default
            if (z < 1500.0d0) then
              rhw = 0.75d0
            else if (z < 4000.0d0) then
              rhw = 0.60d0
            else
              rhw = 0.45d0
            end if
            rhw = rhw*(1.0d0 - 0.04d0*(j-1))
          end select
          rh_d(i, j, k) = rhw
          e = rhw*esw(t)
          pc = p_d(i, j, k)*10.0d0
          qv_d(i, j, k) = 0.622d0*e/(pc - 0.378d0*e)
          ! initial hydrometeors (number per cm3 per bin -> kg/kg)
          rhoc = rho_d(i, j, k)*1.0d-3
          select case (ty)
          case (CT_B)
            if (z >= 3800.0d0 .and. z <= 6000.0d0) then
              do b = 1, 33
                bins_d(i, j, k, b) = (n_lognormal(b, 250.0d0, 9.0d-4, 1.3d0) &
                                      + merge(n_exponential(b, 0.08d0, mp_lambda(2.0d0)), &
                                              0.0d0, b >= 15))*real(xl_tab(b), 8)/rhoc
              end do
            end if
            if (z >= 4200.0d0 .and. z <= 5500.0d0) then
              do b = 12, 26
                bins_d(i, j, k, 66+b) = n_exponential(b, 4.0d-3, 12.0d0)*real(xl_tab(b), 8)/rhoc
              end do
            end if
            if (z >= 5000.0d0 .and. z <= 6500.0d0) then
              do b = 8, 28
                bins_d(i, j, k, 33+b) = n_exponential(b, 2.0d-2, 25.0d0)*real(xl_tab(b), 8)/rhoc
              end do
            end if
          case (CT_C)
            if (z >= 1500.0d0 .and. z <= 5000.0d0) then
              ramp = min(1.0d0, (z - 1500.0d0)/1000.0d0)
              do b = 1, 33
                bins_d(i, j, k, 33+b) = ramp*n_exponential(b, 5.0d-2, 20.0d0)*real(xl_tab(b), 8)/rhoc
              end do
            end if
          case (CT_D)
            if (z >= 5000.0d0 .and. z <= 8000.0d0) then
              do b = 18, 33
                bins_d(i, j, k, 66+b) = n_exponential(b, 2.0d-3, 6.0d0)*real(xl_tab(b), 8)/rhoc
              end do
            end if
            if (z >= 4000.0d0 .and. z <= 6500.0d0) then
              do b = 1, 20
                bins_d(i, j, k, b) = n_lognormal(b, 150.0d0, 7.0d-4, 1.3d0)*real(xl_tab(b), 8)/rhoc
              end do
            end if
            if (z >= 6000.0d0 .and. z <= 8000.0d0) then
              do b = 5, 24
                bins_d(i, j, k, 33+b) = n_exponential(b, 5.0d-3, 30.0d0)*real(xl_tab(b), 8)/rhoc
              end do
            end if
          end select
        end do
        do k = 1, nz+1
          w_prof(i, j, k) = w1(i, j)*sin(pi_d*zf(k)/hdom)
        end do
        w_prof(i, j, 1) = 0.0d0
        w_prof(i, j, nz+1) = 0.0d0
      end do
    end do
  end subroutine define_case

  subroutine init_state(nudge)
    logical, intent(in) :: nudge
    integer :: i, j, k, b
    w = 0.0; u = 0.0; v = 0.0; th_old = 0.0; qv_old = 0.0; qv = 0.0
    qc = 0.0; qr = 0.0; qi = 0.0; qs = 0.0; qg = 0.0
    qnc = 0.0; qnr = 0.0; qni = 0.0; qns = 0.0; qng = 0.0; qna = 0.0
    dz8w = 1.0; rho = 1.0; p = 1.0e5; pii = 1.0; th = 300.0
    chem = 0.0; radar = 0.0
    xland = 1.0; xlat = 0.0; xlong = 0.0; ivgtyp = 16
    rainnc = 0.0; rainncv = 0.0; snownc = 0.0; snowncv = 0.0
    graupelnc = 0.0; graupelncv = 0.0; sr = 0.0
    do j = 1, nd
      do i = 1, nd
        xland(i, j) = real(xland_d(i, j))
        do k = 1, nz
          dz8w(i, k, j) = real(dzd(k))
          rho(i, k, j) = real(rho_d(i, j, k))
          p(i, k, j) = real(p_d(i, j, k))
          pii(i, k, j) = real(pii_d(i, j, k))
          th(i, k, j) = real(th_d(i, j, k))
          qv(i, k, j) = real(qv_d(i, j, k))
          u(i, k, j) = real(u_d(i, j, k))
          v(i, k, j) = real(v_d(i, j, k))
          do b = 1, 99
            chem(i, k, j, 1+b) = real(bins_d(i, j, k, b))
          end do
        end do
        dz8w(i, kme, j) = dz8w(i, kte, j)
        u(i, kme, j) = u(i, kte, j)
        v(i, kme, j) = v(i, kte, j)
      end do
    end do
    if (nudge) then
      do j = 1, nd
        do i = 1, nd
          do k = 1, nz
            th(i, k, j) = nearest(th(i, k, j), 1.0)
          end do
        end do
      end do
    end if
    th_old = th
    qv_old = qv
  end subroutine init_state

  subroutine set_w(tsec)
    real(8), intent(in) :: tsec
    integer :: i, j, k
    do j = jts, jte
      do i = its, ite
        do k = 1, nz+1
          if (tsec < t_off(i, j)) then
            w(i, k, j) = real(w_prof(i, j, k))
          else
            w(i, k, j) = 0.0
          end if
        end do
      end do
    end do
  end subroutine set_w

  ! KC-1's operator (run_column.F90), unchanged.
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
    integer :: i, j, m
    real :: colv(nz)
    do j = jts, jte
      do i = its, ite
        if (all(w(i, 1:nz+1, j) == 0.0)) cycle
        colv = th(i, kts:kte, j)
        call advect_column(colv, w(i, 1:nz+1, j), rho(i, kts:kte, j), dz8w(i, kts:kte, j))
        th(i, kts:kte, j) = colv
        colv = qv(i, kts:kte, j)
        call advect_column(colv, w(i, 1:nz+1, j), rho(i, kts:kte, j), dz8w(i, kts:kte, j))
        qv(i, kts:kte, j) = colv
        do m = 2, nchem
          colv = chem(i, kts:kte, j, m)
          call advect_column(colv, w(i, 1:nz+1, j), rho(i, kts:kte, j), dz8w(i, kts:kte, j))
          chem(i, kts:kte, j, m) = colv
        end do
      end do
    end do
  end subroutine advect_all

  subroutine write_case(st, cm)
    type(fsbm_stream), intent(inout) :: st
    character(len=*), intent(in) :: cm
    call fsbm_begin(st, 'case')
    call fsbm_put(st, 'dt', real(dt, 8))
    call fsbm_put(st, 'nsteps', int(nsteps, 4))
    call fsbm_put(st, 'graupel_build', int(merge(1, 0, graupel), 4))
    call fsbm_put(st, 'domain', int((/ ids, ide, jds, jde, kds, kde /), 4))
    call fsbm_put(st, 'memory', int((/ ims, ime, jms, jme, kms, kme /), 4))
    call fsbm_put(st, 'tile', int((/ its, ite, jts, jte, kts, kte /), 4))
    call fsbm_put(st, 'column_type', int(ctype, 4))
    call fsbm_put(st, 'z_face', zf)
    call fsbm_put(st, 'z_mass', zm)
    call fsbm_put(st, 'dz', dzd)
    call fsbm_put(st, 't_surface', tsfc)
    call fsbm_put(st, 'lapse_rate', lapse)
    call fsbm_put(st, 'p_surface', psfc)
    call fsbm_put(st, 'w1', w1)
    call fsbm_put(st, 'w_off_seconds', t_off)
    call fsbm_put(st, 'xland', xland_d)
    call fsbm_put(st, 'temperature', t_d)
    call fsbm_put(st, 'pressure', p_d)
    call fsbm_put(st, 'exner', pii_d)
    call fsbm_put(st, 'theta', th_d)
    call fsbm_put(st, 'rho', rho_d)
    call fsbm_put(st, 'relative_humidity', rh_d)
    call fsbm_put(st, 'qv', qv_d)
    call fsbm_put(st, 'u', u_d)
    call fsbm_put(st, 'v', v_d)
    call fsbm_put(st, 'w_profile_full_levels', w_prof)
    call fsbm_put(st, 'bins_mass_mixing_ratio', bins_d)
    call fsbm_put(st, 'xl_used_for_bins', xl_tab)
    if (cm == 'kc2g') then
      call fsbm_put(st, 'replay_steps', int(rep_steps_g, 4))
    else
      call fsbm_put(st, 'replay_steps', int(rep_steps, 4))
    end if
    call fsbm_put(st, 'probe_steps', int(prb_steps, 4))
    call fsbm_put(st, 'probe_levels', int(prb_levs, 4))
    call fsbm_put(st, 'profile_every', int(nprof, 4))
    call fsbm_end(st)
  end subroutine write_case

  ! Static fields of the whole domain (tile and halo), as the build holds them.
  subroutine write_initial(st)
    type(fsbm_stream), intent(inout) :: st
    call fsbm_begin(st, 'initial')
    call fsbm_put(st, 'dz8w', dz8w(1:nd, kts:kte, 1:nd))
    call fsbm_put(st, 'rho', rho(1:nd, kts:kte, 1:nd))
    call fsbm_put(st, 'p', p(1:nd, kts:kte, 1:nd))
    call fsbm_put(st, 'pii', pii(1:nd, kts:kte, 1:nd))
    call fsbm_put(st, 'u', u(1:nd, kts:kte, 1:nd))
    call fsbm_put(st, 'v', v(1:nd, kts:kte, 1:nd))
    call fsbm_put(st, 'th', th(1:nd, kts:kte, 1:nd))
    call fsbm_put(st, 'qv', qv(1:nd, kts:kte, 1:nd))
    call fsbm_put(st, 'xland', xland(1:nd, 1:nd))
    call fsbm_put(st, 'chem', chem(its:ite, kts:kte, jts:jte, 2:nchem))
    call fsbm_end(st)
  end subroutine write_initial

  subroutine write_pre(st, n)
    type(fsbm_stream), intent(inout) :: st
    integer, intent(in) :: n
    call fsbm_begin(st, 'pre')
    call fsbm_put(st, 'step', int(n, 4))
    call fsbm_put(st, 'w', w(its:ite, kts:kte+1, jts:jte))
    call fsbm_put(st, 'th', th(1:nd, kts:kte, 1:nd))
    call fsbm_put(st, 'qv', qv(1:nd, kts:kte, 1:nd))
    call fsbm_put(st, 'th_old', th_old(1:nd, kts:kte, 1:nd))
    call fsbm_put(st, 'qv_old', qv_old(1:nd, kts:kte, 1:nd))
    call fsbm_put(st, 'chem', chem(its:ite, kts:kte, jts:jte, 2:nchem))
    call fsbm_put(st, 'rainnc', rainnc(its:ite, jts:jte))
    call fsbm_put(st, 'snownc', snownc(its:ite, jts:jte))
    call fsbm_put(st, 'graupelnc', graupelnc(its:ite, jts:jte))
    call fsbm_put(st, 'sr', sr(its:ite, jts:jte))
    call fsbm_end(st)
  end subroutine write_pre

  subroutine write_post(st, n)
    type(fsbm_stream), intent(inout) :: st
    integer, intent(in) :: n
    call fsbm_begin(st, 'post')
    call fsbm_put(st, 'step', int(n, 4))
    call fsbm_put(st, 'th', th(its:ite, kts:kte, jts:jte))
    call fsbm_put(st, 'qv', qv(its:ite, kts:kte, jts:jte))
    call fsbm_put(st, 'th_old', th_old(its:ite, kts:kte, jts:jte))
    call fsbm_put(st, 'qv_old', qv_old(its:ite, kts:kte, jts:jte))
    call fsbm_put(st, 'chem', chem(its:ite, kts:kte, jts:jte, 2:nchem))
    call fsbm_put(st, 'qc', qc(its:ite, kts:kte, jts:jte))
    call fsbm_put(st, 'qr', qr(its:ite, kts:kte, jts:jte))
    call fsbm_put(st, 'qi', qi(its:ite, kts:kte, jts:jte))
    call fsbm_put(st, 'qs', qs(its:ite, kts:kte, jts:jte))
    call fsbm_put(st, 'qg', qg(its:ite, kts:kte, jts:jte))
    call fsbm_put(st, 'qnc', qnc(its:ite, kts:kte, jts:jte))
    call fsbm_put(st, 'qnr', qnr(its:ite, kts:kte, jts:jte))
    call fsbm_put(st, 'qni', qni(its:ite, kts:kte, jts:jte))
    call fsbm_put(st, 'qns', qns(its:ite, kts:kte, jts:jte))
    call fsbm_put(st, 'qng', qng(its:ite, kts:kte, jts:jte))
    call fsbm_put(st, 'qna', qna(its:ite, kts:kte, jts:jte))
    call fsbm_put(st, 'rainnc', rainnc(its:ite, jts:jte))
    call fsbm_put(st, 'rainncv', rainncv(its:ite, jts:jte))
    call fsbm_put(st, 'snownc', snownc(its:ite, jts:jte))
    call fsbm_put(st, 'snowncv', snowncv(its:ite, jts:jte))
    call fsbm_put(st, 'graupelnc', graupelnc(its:ite, jts:jte))
    call fsbm_put(st, 'graupelncv', graupelncv(its:ite, jts:jte))
    call fsbm_put(st, 'sr', sr(its:ite, jts:jte))
    call fsbm_end(st)
  end subroutine write_post

  subroutine write_step(st, n, tmin)
    type(fsbm_stream), intent(inout) :: st
    integer, intent(in) :: n
    real, intent(in) :: tmin(its:ite, jts:jte)
    call fsbm_begin(st, 'step')
    call fsbm_put(st, 'step', int(n, 4))
    call fsbm_put(st, 'time', real(n, 8)*real(dt, 8))
    call fsbm_put(st, 'rainnc', rainnc(its:ite, jts:jte))
    call fsbm_put(st, 'rainncv', rainncv(its:ite, jts:jte))
    call fsbm_put(st, 'snownc', snownc(its:ite, jts:jte))
    call fsbm_put(st, 'snowncv', snowncv(its:ite, jts:jte))
    call fsbm_put(st, 'graupelnc', graupelnc(its:ite, jts:jte))
    call fsbm_put(st, 'graupelncv', graupelncv(its:ite, jts:jte))
    call fsbm_put(st, 'tmin', tmin)
    call fsbm_end(st)
  end subroutine write_step

  subroutine write_profile(st, n)
    type(fsbm_stream), intent(inout) :: st
    integer, intent(in) :: n
    call fsbm_begin(st, 'profile')
    call fsbm_put(st, 'step', int(n, 4))
    call fsbm_put(st, 'th', th(its:ite, kts:kte, jts:jte))
    call fsbm_put(st, 'qv', qv(its:ite, kts:kte, jts:jte))
    call fsbm_put(st, 'qc', qc(its:ite, kts:kte, jts:jte))
    call fsbm_put(st, 'qr', qr(its:ite, kts:kte, jts:jte))
    call fsbm_put(st, 'qi', qi(its:ite, kts:kte, jts:jte))
    call fsbm_put(st, 'qs', qs(its:ite, kts:kte, jts:jte))
    call fsbm_put(st, 'qg', qg(its:ite, kts:kte, jts:jte))
    call fsbm_end(st)
  end subroutine write_profile

  subroutine write_handoff_statics(uh)
    integer, intent(in) :: uh
    write(uh) real(dz8w(1:nd, kts:kte, 1:nd), 8), real(rho(1:nd, kts:kte, 1:nd), 8), &
              real(p(1:nd, kts:kte, 1:nd), 8), real(pii(1:nd, kts:kte, 1:nd), 8), &
              real(u(1:nd, kts:kte, 1:nd), 8), real(v(1:nd, kts:kte, 1:nd), 8), &
              real(xland(1:nd, 1:nd), 8)
  end subroutine write_handoff_statics

  subroutine write_handoff_state(uh, n)
    integer, intent(in) :: uh, n
    write(uh) int(n, 4)
    write(uh) real(w(its:ite, kts:kte+1, jts:jte), 8), real(th(1:nd, kts:kte, 1:nd), 8), &
              real(qv(1:nd, kts:kte, 1:nd), 8), real(th_old(1:nd, kts:kte, 1:nd), 8), &
              real(qv_old(1:nd, kts:kte, 1:nd), 8), &
              real(chem(its:ite, kts:kte, jts:jte, 2:nchem), 8), &
              real(rainnc(its:ite, jts:jte), 8), real(snownc(its:ite, jts:jte), 8), &
              real(graupelnc(its:ite, jts:jte), 8), real(sr(its:ite, jts:jte), 8)
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
    real, intent(out) :: tmin(its:ite, jts:jte)
    integer :: i, j
    do j = jts, jte
      do i = its, ite
        tmin(i, j) = minval(th(i, kts:kte, j)*pii(i, kts:kte, j))
      end do
    end do
  end subroutine tile_tmin

  subroutine run_case(cm, nudge, probes)
    character(len=*), intent(in) :: cm
    logical, intent(in) :: nudge, probes
    integer :: n, uh, cnt, i, j, c
    real :: tmin(its:ite, jts:jte), tmin_run(its:ite, jts:jte), tmax_run(its:ite, jts:jte)
    real :: tmax(its:ite, jts:jte)
    integer(4) :: rain_steps(ntc), snow_steps(ntc), hail_steps(ntc)
    real(8) :: tsec
    logical :: colmask(ntc), is_rep
    character(len=64) :: sumname

    call init_state(nudge)
    if (nudge) then
      sumname = 'kc2_sensitivity'
    else
      sumname = cm//'_summary'
      call fsbm_open(s_case, trim(outdir)//'/'//cm//'_case')
      call write_case(s_case, cm)
      call fsbm_close(s_case)
      call fsbm_open(s_rep, trim(outdir)//'/'//cm//'_replay')
      call fsbm_open(s_ctl, trim(outdir)//'/'//cm//'_control')
      open(newunit=uh, file=trim(outdir)//'/'//cm//'_handoff.f64', access='stream', &
           form='unformatted', status='replace', action='write')
      call write_handoff_statics(uh)
    end if
    call fsbm_open(s_sum, trim(outdir)//'/'//trim(sumname))
    call write_initial(s_sum)
    call fsbm_probe_configure(trim(outdir)//'/'//cm//'_probe', prb_steps, prb_levs, &
                              probes, probes, .false.)
    do j = jts, jte
      do i = its, ite
        colmask((j-jts)*(ite-its+1) + (i-its) + 1) = ctype(i, j) /= CT_ENV
      end do
    end do
    call fsbm_probe_configure_tile(its, jts, ite-its+1, jte-jts+1, colmask, 6, 2)

    tmin_run = 1.0e30
    tmax_run = -1.0e30
    rain_steps = 0
    snow_steps = 0
    hail_steps = 0
    do n = 1, nsteps
      tsec = real(n-1, 8)*real(dt, 8)
      call set_w(tsec)
      if (n > 1) call advect_all()
      if (cm == 'kc2g') then
        is_rep = any(rep_steps_g == n)
      else
        is_rep = any(rep_steps == n)
      end if
      if (.not. nudge .and. is_rep) then
        call write_pre(s_rep, n)
        call write_handoff_state(uh, n)
      end if
      call call_sbm(n)
      call tile_tmin(tmin)
      tmin_run = min(tmin_run, tmin)
      do j = jts, jte
        do i = its, ite
          tmax(i, j) = maxval(th(i, kts:kte, j)*pii(i, kts:kte, j))
          c = (j-jts)*(ite-its+1) + (i-its) + 1
          if (rainncv(i, j) > 0.0) rain_steps(c) = rain_steps(c) + 1
          if (snowncv(i, j) > 0.0) snow_steps(c) = snow_steps(c) + 1
          if (graupelncv(i, j) > 0.0) hail_steps(c) = hail_steps(c) + 1
        end do
      end do
      tmax_run = max(tmax_run, tmax)
      call write_step(s_sum, n, tmin)
      if (mod(n, nprof) == 0 .and. cm /= 'kc2g') call write_profile(s_sum, n)
      if (.not. nudge) then
        call write_post(s_ctl, n)
        if (is_rep) call write_post(s_rep, n)
      end if
    end do

    call fsbm_begin(s_sum, 'acceptance')
    call fsbm_put(s_sum, 'column_type', int(ctype(its:ite, jts:jte), 4))
    call fsbm_put(s_sum, 'tmin_run', tmin_run)
    call fsbm_put(s_sum, 'tmax_run', tmax_run)
    call fsbm_put(s_sum, 'rainnc_final', rainnc(its:ite, jts:jte))
    call fsbm_put(s_sum, 'snownc_final', snownc(its:ite, jts:jte))
    call fsbm_put(s_sum, 'graupelnc_final', graupelnc(its:ite, jts:jte))
    call fsbm_put(s_sum, 'rain_steps', rain_steps)
    call fsbm_put(s_sum, 'snow_steps', snow_steps)
    call fsbm_put(s_sum, 'hail_steps', hail_steps)
    do cnt = 1, PR_NCNT2
      call fsbm_put(s_sum, trim(CNT2_NAME(cnt)), pr_cnt2(1:ntc, cnt))
    end do
    call fsbm_end(s_sum)
    call fsbm_close(s_sum)
    call fsbm_probe_finish()
    if (.not. nudge) then
      call fsbm_close(s_rep)
      call fsbm_close(s_ctl)
      close(uh)
    end if
  end subroutine run_case

  ! The low-edge tile (i, j = 1..3): 6 calls from the KC-2 initial state (with
  ! the corner column (1, 1) of type B), advected between calls as in KC-2.
  subroutine run_edge(probes)
    logical, intent(in) :: probes
    integer :: n
    logical :: colmask(ntc)
    call init_state(.false.)
    call fsbm_open(s_rep, trim(outdir)//'/kc2edge_replay')
    call fsbm_open(s_ctl, trim(outdir)//'/kc2edge_control')
    call fsbm_begin(s_rep, 'edge_case')
    call fsbm_put(s_rep, 'dt', real(dt, 8))
    call fsbm_put(s_rep, 'nsteps', int(nsteps_edge, 4))
    call fsbm_put(s_rep, 'domain', int((/ ids, ide, jds, jde, kds, kde /), 4))
    call fsbm_put(s_rep, 'memory', int((/ ims, ime, jms, jme, kms, kme /), 4))
    call fsbm_put(s_rep, 'tile', int((/ its, ite, jts, jte, kts, kte /), 4))
    call fsbm_put(s_rep, 'column_type', int(ctype, 4))
    call fsbm_put(s_rep, 'w_profile_full_levels', w_prof)
    call fsbm_put(s_rep, 'w_off_seconds', t_off)
    call fsbm_put(s_rep, 'replay_steps', int(rep_steps_edge, 4))
    call fsbm_end(s_rep)
    call write_initial(s_rep)
    colmask = .false.
    call fsbm_probe_configure(trim(outdir)//'/kc2edge_probe', rep_steps_edge, prb_levs, &
                              probes, probes, .false.)
    call fsbm_probe_configure_tile(its, jts, ite-its+1, jte-jts+1, colmask, 0, 0)
    do n = 1, nsteps_edge
      call set_w(real(n-1, 8)*real(dt, 8))
      if (n > 1) call advect_all()
      if (any(rep_steps_edge == n)) call write_pre(s_rep, n)
      call call_sbm(n)
      call write_post(s_ctl, n)
      if (any(rep_steps_edge == n)) call write_post(s_rep, n)
    end do
    call fsbm_probe_finish()
    call fsbm_close(s_rep)
    call fsbm_close(s_ctl)
  end subroutine run_edge

  subroutine cross_replay(cm, path)
    character(len=*), intent(in) :: cm, path
    integer :: uh, st, n, r, nr
    real(8) :: a_dz(nd, nz, nd), a_rho(nd, nz, nd), a_p(nd, nz, nd), a_pii(nd, nz, nd), &
               a_u(nd, nz, nd), a_v(nd, nz, nd), a_xl(nd, nd)
    real(8) :: a_w(ite-its+1, nz+1, jte-jts+1), a_th(nd, nz, nd), a_qv(nd, nz, nd), a_tho(nd, nz, nd), &
               a_qvo(nd, nz, nd), a_chem(ite-its+1, nz, jte-jts+1, nchem-1), &
               a_rn(ite-its+1, jte-jts+1), a_sn(ite-its+1, jte-jts+1), &
               a_gn(ite-its+1, jte-jts+1), a_sr(ite-its+1, jte-jts+1)
    integer(4) :: step4
    call init_state(.false.)
    open(newunit=uh, file=path, access='stream', form='unformatted', status='old', &
         action='read', iostat=st)
    if (st /= 0) call fsbm_fail('cannot open handoff '//path)
    read(uh) a_dz, a_rho, a_p, a_pii, a_u, a_v, a_xl
    dz8w(1:nd, kts:kte, 1:nd) = a_dz
    rho(1:nd, kts:kte, 1:nd) = a_rho
    p(1:nd, kts:kte, 1:nd) = a_p
    pii(1:nd, kts:kte, 1:nd) = a_pii
    u(1:nd, kts:kte, 1:nd) = a_u
    v(1:nd, kts:kte, 1:nd) = a_v
    xland(1:nd, 1:nd) = a_xl
    dz8w(1:nd, kme, 1:nd) = dz8w(1:nd, kte, 1:nd)
    u(1:nd, kme, 1:nd) = u(1:nd, kte, 1:nd)
    v(1:nd, kme, 1:nd) = v(1:nd, kte, 1:nd)
    call fsbm_open(s_x, trim(outdir)//'/'//cm//'_crossreplay')
    nr = nrep
    if (cm == 'kc2g') nr = nrep_g
    do r = 1, nr
      read(uh, iostat=st) step4
      if (st /= 0) call fsbm_fail('handoff ended early')
      read(uh) a_w, a_th, a_qv, a_tho, a_qvo, a_chem, a_rn, a_sn, a_gn, a_sr
      n = step4
      if (cm == 'kc2g') then
        if (n /= rep_steps_g(r)) call fsbm_fail('handoff step order differs from the replay steps')
      else
        if (n /= rep_steps(r)) call fsbm_fail('handoff step order differs from the replay steps')
      end if
      w(its:ite, kts:kte+1, jts:jte) = a_w
      th(1:nd, kts:kte, 1:nd) = a_th
      qv(1:nd, kts:kte, 1:nd) = a_qv
      th_old(1:nd, kts:kte, 1:nd) = a_tho
      qv_old(1:nd, kts:kte, 1:nd) = a_qvo
      chem(its:ite, kts:kte, jts:jte, 2:nchem) = a_chem
      rainnc(its:ite, jts:jte) = a_rn
      snownc(its:ite, jts:jte) = a_sn
      graupelnc(its:ite, jts:jte) = a_gn
      sr(its:ite, jts:jte) = a_sr
      call call_sbm(n)
      call write_post(s_x, n)
    end do
    call fsbm_close(s_x)
    close(uh)
  end subroutine cross_replay

end program run_kc2
