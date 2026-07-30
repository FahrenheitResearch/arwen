! Per-leaf oracle for the WRF v4.6.1 Noah-MP *thermal* group:
! TSNOSOI, HRT, HSTEP, PHASECHANGE and FRH2O.
!
! This is run_leaves.F90's method applied to a second, independently owned leaf
! group; it shares no writable file with that harness.  It links the
! *visibility-patched* module (patches/noahmp-lsm-leaf-visibility.patch,
! audited by check_visibility_patch.py before this file is compiled).  Apart
! from the 50 `private ::` -> `public ::` accessibility statements, the physics
! source is the pinned WRF v4.6.1 file byte for byte.
!
! Every leaf is driven as a pure function of a flat FP32 input vector `x` plus
! a small integer topology vector `ix`, producing a flat FP32 output vector
! `y`.  Integer outputs (PHASECHANGE's IMELT) are emitted as REAL(IMELT),
! which is exact for the values 0, 1 and 2 that routine can produce.
!
! Array regions the callee leaves undefined (INTENT(OUT) slots above ISNOW) are
! pre-filled with 0.0 by the harness and flagged live=0.  Regions a leaf must
! never read are filled with POISON = -9999.0 instead of a plausible value, so
! a port that reads them is visibly wrong.  INTENT(INOUT) slots above ISNOW are
! flagged live=1 and must be echoed unchanged -- that is a real assertion, not
! a convention.
!
! Option identity: the pinned WRF Registry default
!   dveg=4 opt_crs=1 opt_btr=1 opt_run=3 opt_sfc=1 opt_frz=1 opt_inf=1
!   opt_rad=3 opt_alb=2 opt_snf=1 opt_tbot=2 opt_stc=1 opt_rsf=1 opt_soil=1
!   opt_pedo=1 opt_crop=0 opt_irr=0 opt_irrm=0 opt_infdv=0 opt_tdrn=0
!
! Which options this group reads:
!   HRT         OPT_TBOT (5440, 5443) and OPT_STC (5455, 5458)
!   PHASECHANGE OPT_FRZ  (5683, 5690)
!   TSNOSOI     OPT_TBOT (5334, 5336) -- but only to fill a local that the
!               unconditional RETURN at 5344 then discards
!   HSTEP, FRH2O  read no option variable at all
!
! FRH2O is DEAD under this identity: its only call site in the module is line
! 5692, inside `IF (OPT_FRZ == 2)`.  It is pinned here anyway because it is a
! self-contained leaf whose fixture costs nothing and whose port would
! otherwise have to be written blind if opt_frz ever moves; nothing in gpuwm
! dispatches to it, and PHASECHANGE's port implements the OPT_FRZ == 1 branch
! only.  See validate_thermal_oracle.py, which asserts that separation.

module noahmp_thermal_oracle

  use module_sf_noahmplsm, only: noahmp_parameters, &
      hrt, hstep, tsnosoi, phasechange, frh2o
  implicit none
  private
  public :: open_outputs, close_outputs, dump_all

  integer, parameter :: LNSNOW = 3
  integer, parameter :: LNSOIL = 4
  integer, parameter :: LNLAY = LNSNOW + LNSOIL
  real, parameter :: POISON = -9999.0

  integer :: unit_leaf = -1
  integer :: unit_disc = -1

  abstract interface
    subroutine leaf_eval(x, ix, y)
      real,    intent(in)  :: x(:)
      integer, intent(in)  :: ix(:)
      real,    intent(out) :: y(:)
    end subroutine leaf_eval
  end interface

contains

! ---------------------------------------------------------------- plumbing --

  subroutine open_outputs(leaf_path, disc_path)
    character(len=*), intent(in) :: leaf_path, disc_path
    open(newunit=unit_leaf, file=trim(leaf_path), status='replace', &
         action='write')
    write(unit_leaf, '(A)') 'leaf,case,role,name,index,slot,live,bits,value'
    open(newunit=unit_disc, file=trim(disc_path), status='replace', &
         action='write')
    write(unit_disc, '(A)') 'leaf,case,slot,name,index,baseline_bits,' // &
        'already_zero,noutputs_changed,max_abs_delta'
  end subroutine open_outputs

  subroutine close_outputs()
    close(unit_leaf)
    close(unit_disc)
  end subroutine close_outputs

  function hexbits(value) result(text)
    real, intent(in) :: value
    character(len=8) :: text
    integer :: raw
    raw = transfer(value, raw)
    write(text, '(Z8.8)') raw
  end function hexbits

  integer function rawbits(value)
    real, intent(in) :: value
    rawbits = transfer(value, rawbits)
  end function rawbits

  logical function isfinite(value)
    real, intent(in) :: value
    isfinite = (value == value) .and. (abs(value) <= huge(value))
  end function isfinite

  subroutine emit_row(leaf, casename, role, name, idx, slot, live, value)
    character(len=*), intent(in) :: leaf, casename, role, name
    integer, intent(in) :: idx, slot
    logical, intent(in) :: live
    real,    intent(in) :: value
    integer :: liveflag
    liveflag = 0
    if (live) liveflag = 1
    write(unit_leaf, '(*(g0,:,","))') trim(leaf), trim(casename), &
        trim(role), trim(name), idx, slot, liveflag, hexbits(value), value
  end subroutine emit_row

! -------------------------------------------------------------- leaf driver --

  subroutine run_leaf(leaf, evaluator, casenames, ixnames, ixcases, &
                      xnames, xindex, xcases, ynames, yindex, ylive)
    character(len=*), intent(in) :: leaf
    procedure(leaf_eval)         :: evaluator
    character(len=*), intent(in) :: casenames(:)
    character(len=*), intent(in) :: ixnames(:)
    integer,          intent(in) :: ixcases(:, :)  ! (nix, ncase)
    character(len=*), intent(in) :: xnames(:)
    integer,          intent(in) :: xindex(:)
    real,             intent(in) :: xcases(:, :)   ! (nx, ncase)
    character(len=*), intent(in) :: ynames(:)
    integer,          intent(in) :: yindex(:)
    logical,          intent(in) :: ylive(:, :)    ! (ny, ncase)

    integer :: nx, ny, ncase, ic, i, j, nchanged, alreadyzero
    real, allocatable :: x0(:), x1(:), y0(:), y1(:)
    real :: maxdelta, delta

    nx = size(xcases, 1)
    ny = size(ynames)
    ncase = size(casenames)
    allocate(x0(nx), x1(nx), y0(ny), y1(ny))

    do ic = 1, ncase
      x0 = xcases(:, ic)
      y0 = 0.0
      call evaluator(x0, ixcases(:, ic), y0)

      do i = 1, size(ixcases, 1)
        call emit_row(leaf, casenames(ic), 'int', ixnames(i), 0, i, .true., &
                      real(ixcases(i, ic)))
      end do
      do i = 1, nx
        call emit_row(leaf, casenames(ic), 'in', xnames(i), xindex(i), i, &
                      .true., x0(i))
      end do
      do j = 1, ny
        call emit_row(leaf, casenames(ic), 'out', ynames(j), yindex(j), j, &
                      ylive(j, ic), y0(j))
      end do

      ! Zero-probe sweep on the oracle side.
      do i = 1, nx
        x1 = x0
        x1(i) = 0.0
        y1 = 0.0
        call evaluator(x1, ixcases(:, ic), y1)
        nchanged = 0
        maxdelta = 0.0
        do j = 1, ny
          if (.not. ylive(j, ic)) cycle
          if (rawbits(y0(j)) /= rawbits(y1(j))) then
            nchanged = nchanged + 1
            if (isfinite(y0(j)) .and. isfinite(y1(j))) then
              delta = abs(y0(j) - y1(j))
              if (delta > maxdelta) maxdelta = delta
            end if
          end if
        end do
        alreadyzero = 0
        if (rawbits(x0(i)) == rawbits(0.0)) alreadyzero = 1
        write(unit_disc, '(*(g0,:,","))') trim(leaf), trim(casenames(ic)), &
            i, trim(xnames(i)), xindex(i), hexbits(x0(i)), alreadyzero, &
            nchanged, maxdelta
      end do
    end do

    deallocate(x0, x1, y0, y1)
  end subroutine run_leaf

! ---------------------------------------------------------------- leaf: hrt --
!
! module_sf_noahmplsm.F:5375-5473.  The body never references `parameters`, so
! the harness passes an untouched handle.
!
! Live branches under the pinned identity: the K == ISNOW+1 head (5425-5430),
! the interior K < NSOIL body (5431-5436), the K == NSOIL foot (5437-5447) with
! OPT_TBOT == 2 (5443-5446), and the OPT_STC == 1 diagonal (5455-5457).  Dead
! and NOT ported: BOTFLX = 0 under OPT_TBOT == 1 (5440-5442) and the OPT_STC ==
! 2 diagonal (5458-5460).
!
! DT is declared INTENT(IN) at 5397 and the body never references it -- HRT
! forms a tendency, HSTEP does the time step.  It carries a distinct non-zero
! value in every case so that inertness is measured rather than assumed.
!
! ZSNSO, STC, DF, HCPCT and PHI are read only over ISNOW+1..NSOIL (the loops at
! 5424 and 5451; the only out-of-loop subscripts are K+1 at 5427/5429 with
! K = ISNOW+1, and K-1 at 5432-5447 with K >= ISNOW+2, both in range).  Layers
! above ISNOW are therefore POISONed per case.
!
! TSNOSOI's only call passes PHI(ISNOW+1:NSOIL) = 0.0 (5310).  The fixture
! gives PHI non-zero values anyway: with PHI == 0 everywhere the term at
! 5430/5436/5447 is undetectable and no PHI mutant could be killed.
!
! Input slots: 1-7 zsnso(-2:4), 8-14 stc(-2:4), 15-21 df(-2:4),
! 22-28 hcpct(-2:4), 29-35 phi(-2:4), 36 tbot, 37 zbot, 38 dt, 39 ssoil.
! Output slots: 1-7 ai(-2:4), 8-14 bi(-2:4), 15-21 ci(-2:4),
! 22-28 rhsts(-2:4), 29 botflx.

  subroutine eval_hrt(x, ix, y)
    real,    intent(in)  :: x(:)
    integer, intent(in)  :: ix(:)
    real,    intent(out) :: y(:)
    type(noahmp_parameters) :: p
    real :: zsnso(-LNSNOW+1:LNSOIL), stc(-LNSNOW+1:LNSOIL)
    real :: df(-LNSNOW+1:LNSOIL), hcpct(-LNSNOW+1:LNSOIL)
    real :: phi(-LNSNOW+1:LNSOIL)
    real :: ai(-LNSNOW+1:LNSOIL), bi(-LNSNOW+1:LNSOIL)
    real :: ci(-LNSNOW+1:LNSOIL), rhsts(-LNSNOW+1:LNSOIL)
    real :: tbot, zbot, dt, ssoil, botflx
    integer :: k, s

    s = 0
    do k = -LNSNOW+1, LNSOIL
      s = s + 1; zsnso(k) = x(s)
    end do
    do k = -LNSNOW+1, LNSOIL
      s = s + 1; stc(k) = x(s)
    end do
    do k = -LNSNOW+1, LNSOIL
      s = s + 1; df(k) = x(s)
    end do
    do k = -LNSNOW+1, LNSOIL
      s = s + 1; hcpct(k) = x(s)
    end do
    do k = -LNSNOW+1, LNSOIL
      s = s + 1; phi(k) = x(s)
    end do
    s = s + 1; tbot  = x(s)
    s = s + 1; zbot  = x(s)
    s = s + 1; dt    = x(s)
    s = s + 1; ssoil = x(s)

    ai = 0.0
    bi = 0.0
    ci = 0.0
    rhsts = 0.0
    botflx = 0.0

    call hrt(p, LNSNOW, LNSOIL, ix(1), zsnso, &
             stc, tbot, zbot, dt, &
             df, hcpct, ssoil, phi, &
             ai, bi, ci, rhsts, &
             botflx)

    s = 0
    do k = -LNSNOW+1, LNSOIL
      s = s + 1
      y(s) = ai(k)
      y(LNLAY + s) = bi(k)
      y(2*LNLAY + s) = ci(k)
      y(3*LNLAY + s) = rhsts(k)
    end do
    y(4*LNLAY + 1) = botflx
  end subroutine eval_hrt

  subroutine dump_hrt()
    integer, parameter :: NX = 5*LNLAY + 4, NY = 4*LNLAY + 1, NCASE = 6
    character(len=12) :: xn(NX), yn(NY)
    integer :: xi(NX), yi(NY)
    character(len=28) :: cn(NCASE)
    character(len=12) :: ixn(1)
    real :: xc(NX, NCASE)
    integer :: ixc(1, NCASE)
    logical :: ylive(NY, NCASE)
    integer :: k, s, ic, isnow
    real :: zs(LNLAY, NCASE), st(LNLAY, NCASE), dfv(LNLAY, NCASE)
    real :: hc(LNLAY, NCASE), ph(LNLAY, NCASE)
    real :: tbotv(NCASE), zbotv(NCASE), dtv(NCASE), ssoilv(NCASE)

    ixn = [character(len=12) :: 'isnow']
    s = 0
    do k = -LNSNOW+1, LNSOIL
      s = s + 1
      xn(s) = 'zsnso';             xi(s) = k
      xn(LNLAY + s) = 'stc';       xi(LNLAY + s) = k
      xn(2*LNLAY + s) = 'df';      xi(2*LNLAY + s) = k
      xn(3*LNLAY + s) = 'hcpct';   xi(3*LNLAY + s) = k
      xn(4*LNLAY + s) = 'phi';     xi(4*LNLAY + s) = k
      yn(s) = 'ai';                yi(s) = k
      yn(LNLAY + s) = 'bi';        yi(LNLAY + s) = k
      yn(2*LNLAY + s) = 'ci';      yi(2*LNLAY + s) = k
      yn(3*LNLAY + s) = 'rhsts';   yi(3*LNLAY + s) = k
    end do
    xn(5*LNLAY + 1) = 'tbot';  xi(5*LNLAY + 1) = 0
    xn(5*LNLAY + 2) = 'zbot';  xi(5*LNLAY + 2) = 0
    xn(5*LNLAY + 3) = 'dt';    xi(5*LNLAY + 3) = 0
    xn(5*LNLAY + 4) = 'ssoil'; xi(5*LNLAY + 4) = 0
    yn(4*LNLAY + 1) = 'botflx'; yi(4*LNLAY + 1) = 0

    cn = [character(len=28) :: 'soil_only_isnow0', 'one_snow_layer', &
          'two_snow_layers', 'three_snow_layers', 'three_snow_layers_alt', &
          'warm_soil_isnow0']
    ixc(1, :) = [0, -1, -2, -3, -3, 0]

    ! ZSNSO is the layer-bottom depth measured from the snow surface, so it is
    ! negative and strictly decreasing; ZSNSO(K-1)-ZSNSO(K) is DZSNSO(K) > 0.
    zs = reshape([ &
        POISON,  POISON,  POISON,  -0.10,  -0.40,  -1.00,  -2.00,  &
        POISON,  POISON,  -0.09,   -0.19,  -0.49,  -1.09,  -2.09,  &
        POISON,  -0.06,   -0.15,   -0.26,  -0.56,  -1.16,  -2.16,  &
        -0.05,   -0.13,   -0.25,   -0.35,  -0.65,  -1.25,  -2.25,  &
        -0.04,   -0.11,   -0.21,   -0.32,  -0.62,  -1.22,  -2.22,  &
        POISON,  POISON,  POISON,  -0.12,  -0.45,  -1.10,  -2.10   ], &
        [LNLAY, NCASE])
    st = reshape([ &
        POISON,  POISON,  POISON,  281.40, 283.10, 284.90, 286.20, &
        POISON,  POISON,  268.30,  270.50, 272.40, 274.80, 277.10, &
        POISON,  261.20,  265.70,  269.40, 271.80, 273.90, 276.30, &
        253.60,  258.90,  263.40,  268.10, 270.70, 273.20, 275.40, &
        249.80,  255.30,  261.90,  266.70, 269.90, 272.60, 274.80, &
        POISON,  POISON,  POISON,  291.70, 290.20, 288.60, 287.10  ], &
        [LNLAY, NCASE])
    dfv = reshape([ &
        POISON,  POISON,  POISON,  1.42,  1.63,  1.85,  1.94, &
        POISON,  POISON,  0.31,    1.38,  1.59,  1.81,  1.90, &
        POISON,  0.24,    0.35,    1.46,  1.67,  1.88,  1.97, &
        0.19,    0.27,    0.41,    1.51,  1.72,  1.92,  2.01, &
        0.22,    0.30,    0.38,    1.33,  1.55,  1.79,  1.88, &
        POISON,  POISON,  POISON,  1.28,  1.49,  1.71,  1.83  ], &
        [LNLAY, NCASE])
    hc = reshape([ &
        POISON,  POISON,  POISON,  2.31e6, 2.44e6, 2.57e6, 2.66e6, &
        POISON,  POISON,  5.12e5,  2.28e6, 2.41e6, 2.54e6, 2.63e6, &
        POISON,  3.94e5,  6.05e5,  2.35e6, 2.48e6, 2.61e6, 2.70e6, &
        2.87e5,  4.63e5,  7.18e5,  2.39e6, 2.52e6, 2.65e6, 2.74e6, &
        3.15e5,  5.27e5,  6.41e5,  2.22e6, 2.36e6, 2.49e6, 2.58e6, &
        POISON,  POISON,  POISON,  2.19e6, 2.33e6, 2.46e6, 2.55e6  ], &
        [LNLAY, NCASE])
    ph = reshape([ &
        POISON,  POISON,  POISON,  4.70,  2.10,  0.80,  0.30, &
        POISON,  POISON,  9.40,    5.20,  2.60,  1.10,  0.40, &
        POISON,  13.80,   7.90,    4.30,  2.20,  0.90,  0.35, &
        21.50,   12.70,   6.80,    3.60,  1.80,  0.75,  0.28, &
        18.20,   10.40,   5.90,    3.10,  1.50,  0.62,  0.24, &
        POISON,  POISON,  POISON,  6.30,  3.40,  1.40,  0.55  ], &
        [LNLAY, NCASE])
    tbotv  = [285.60, 283.20, 281.90, 280.40, 279.70, 287.30]
    zbotv  = [-8.00, -8.09, -8.16, -8.25, -8.22, -8.00]
    dtv    = [1800.0, 900.0, 600.0, 300.0, 120.0, 60.0]
    ssoilv = [-37.40, 22.80, 45.10, 68.30, -19.60, 53.70]

    do ic = 1, NCASE
      do k = 1, LNLAY
        xc(k, ic)           = zs(k, ic)
        xc(LNLAY + k, ic)   = st(k, ic)
        xc(2*LNLAY + k, ic) = dfv(k, ic)
        xc(3*LNLAY + k, ic) = hc(k, ic)
        xc(4*LNLAY + k, ic) = ph(k, ic)
      end do
      xc(5*LNLAY + 1, ic) = tbotv(ic)
      xc(5*LNLAY + 2, ic) = zbotv(ic)
      xc(5*LNLAY + 3, ic) = dtv(ic)
      xc(5*LNLAY + 4, ic) = ssoilv(ic)
    end do

    ylive = .false.
    do ic = 1, NCASE
      isnow = ixc(1, ic)
      s = 0
      do k = -LNSNOW+1, LNSOIL
        s = s + 1
        ylive(s, ic) = (k >= isnow + 1)
        ylive(LNLAY + s, ic) = (k >= isnow + 1)
        ylive(2*LNLAY + s, ic) = (k >= isnow + 1)
        ylive(3*LNLAY + s, ic) = (k >= isnow + 1)
      end do
      ylive(4*LNLAY + 1, ic) = .true.
    end do

    call run_leaf('hrt', eval_hrt, cn, ixn, ixc, xn, xi, xc, yn, yi, ylive)
  end subroutine dump_hrt

! -------------------------------------------------------------- leaf: hstep --
!
! module_sf_noahmplsm.F:5477-5530.  The body never references `parameters`.
!
! Every argument except DT is INTENT(INOUT) over -NSNOW+1:NSOIL, and HSTEP
! touches only ISNOW+1..NSOIL (5506-5511, 5515-5518, 5526-5528).  Slots above
! ISNOW must therefore come back EXACTLY as they went in; they are flagged
! live=1 with ordinary non-zero values so that echo is asserted rather than
! assumed, and they are deliberately not POISONed for the same reason.
!
! CI(NSOIL) is inert: HSTEP scales it at 5510, copies it into the local CIIN at
! 5517, and ROSR12 overwrites C(NSOIL) with 0.0 at 5565 before any read of C.
! The CI(NSOIL) *output* is P(NSOIL) = DELTA(NSOIL) (5582), which does not
! depend on the CI(NSOIL) input.  This is the same inertness the `rosr12` leaf
! declares for ("c", 4), reached one call frame higher up.
!
! AI(ISNOW+1) is NOT inert even though ROSR12 never reads A(NTOP) (5575): HSTEP
! still emits AI(ISNOW+1)*DT at 5508, so the slot is observable.
!
! Input slots: 1-7 ai(-2:4), 8-14 bi(-2:4), 15-21 ci(-2:4), 22-28 rhsts(-2:4),
! 29-35 stc(-2:4), 36 dt.
! Output slots: the same five arrays in the same order, 1-35.

  subroutine eval_hstep(x, ix, y)
    real,    intent(in)  :: x(:)
    integer, intent(in)  :: ix(:)
    real,    intent(out) :: y(:)
    type(noahmp_parameters) :: p
    real :: ai(-LNSNOW+1:LNSOIL), bi(-LNSNOW+1:LNSOIL)
    real :: ci(-LNSNOW+1:LNSOIL), rhsts(-LNSNOW+1:LNSOIL)
    real :: stc(-LNSNOW+1:LNSOIL)
    real :: dt
    integer :: k, s

    s = 0
    do k = -LNSNOW+1, LNSOIL
      s = s + 1; ai(k) = x(s)
    end do
    do k = -LNSNOW+1, LNSOIL
      s = s + 1; bi(k) = x(s)
    end do
    do k = -LNSNOW+1, LNSOIL
      s = s + 1; ci(k) = x(s)
    end do
    do k = -LNSNOW+1, LNSOIL
      s = s + 1; rhsts(k) = x(s)
    end do
    do k = -LNSNOW+1, LNSOIL
      s = s + 1; stc(k) = x(s)
    end do
    s = s + 1; dt = x(s)

    call hstep(p, LNSNOW, LNSOIL, ix(1), dt, ai, bi, ci, rhsts, stc)

    s = 0
    do k = -LNSNOW+1, LNSOIL
      s = s + 1
      y(s) = ai(k)
      y(LNLAY + s) = bi(k)
      y(2*LNLAY + s) = ci(k)
      y(3*LNLAY + s) = rhsts(k)
      y(4*LNLAY + s) = stc(k)
    end do
  end subroutine eval_hstep

  subroutine dump_hstep()
    integer, parameter :: NX = 5*LNLAY + 1, NY = 5*LNLAY, NCASE = 6
    character(len=12) :: xn(NX), yn(NY)
    integer :: xi(NX), yi(NY)
    character(len=28) :: cn(NCASE)
    character(len=12) :: ixn(1)
    real :: xc(NX, NCASE)
    integer :: ixc(1, NCASE)
    logical :: ylive(NY, NCASE)
    integer :: k, s, ic
    real :: av(LNLAY, NCASE), bv(LNLAY, NCASE), cv(LNLAY, NCASE)
    real :: rv(LNLAY, NCASE), sv(LNLAY, NCASE), dtv(NCASE)

    ixn = [character(len=12) :: 'isnow']
    s = 0
    do k = -LNSNOW+1, LNSOIL
      s = s + 1
      xn(s) = 'ai';              xi(s) = k
      xn(LNLAY + s) = 'bi';      xi(LNLAY + s) = k
      xn(2*LNLAY + s) = 'ci';    xi(2*LNLAY + s) = k
      xn(3*LNLAY + s) = 'rhsts'; xi(3*LNLAY + s) = k
      xn(4*LNLAY + s) = 'stc';   xi(4*LNLAY + s) = k
      yn(s) = 'ai';              yi(s) = k
      yn(LNLAY + s) = 'bi';      yi(LNLAY + s) = k
      yn(2*LNLAY + s) = 'ci';    yi(2*LNLAY + s) = k
      yn(3*LNLAY + s) = 'rhsts'; yi(3*LNLAY + s) = k
      yn(4*LNLAY + s) = 'stc';   yi(4*LNLAY + s) = k
    end do
    xn(5*LNLAY + 1) = 'dt'; xi(5*LNLAY + 1) = 0

    cn = [character(len=28) :: 'soil_only_isnow0', 'one_snow_layer', &
          'two_snow_layers', 'three_snow_layers', 'three_snow_layers_alt', &
          'short_step_isnow0']
    ixc(1, :) = [0, -1, -2, -3, -3, 0]

    ! HRT-shaped coefficients: AI and CI negative, BI ~ -(AI+CI), so
    ! 1 + BI*DT dominates AI*DT + CI*DT and the ROSR12 sweep is stable.
    av = reshape([ &
        -8.00e-6, -1.20e-5, -2.00e-5, -3.10e-5, -4.20e-5, -2.60e-5, -1.30e-5, &
        -7.10e-6, -1.05e-5, -1.84e-5, -2.87e-5, -3.95e-5, -2.41e-5, -1.18e-5, &
        -9.30e-6, -1.36e-5, -2.15e-5, -3.34e-5, -4.48e-5, -2.79e-5, -1.42e-5, &
        -6.40e-6, -9.80e-6, -1.71e-5, -2.63e-5, -3.70e-5, -2.24e-5, -1.09e-5, &
        -1.02e-5, -1.47e-5, -2.31e-5, -3.58e-5, -4.73e-5, -2.95e-5, -1.51e-5, &
        -8.70e-6, -1.28e-5, -1.96e-5, -3.02e-5, -4.06e-5, -2.52e-5, -1.24e-5  ], &
        [LNLAY, NCASE])
    cv = reshape([ &
        -9.00e-6, -1.40e-5, -2.20e-5, -5.30e-5, -3.10e-5, -1.50e-5, -7.00e-6, &
        -8.20e-6, -1.27e-5, -2.04e-5, -4.91e-5, -2.88e-5, -1.39e-5, -6.40e-6, &
        -1.04e-5, -1.52e-5, -2.37e-5, -5.68e-5, -3.34e-5, -1.61e-5, -7.50e-6, &
        -7.30e-6, -1.13e-5, -1.88e-5, -4.52e-5, -2.66e-5, -1.28e-5, -5.90e-6, &
        -1.13e-5, -1.61e-5, -2.49e-5, -6.05e-5, -3.57e-5, -1.72e-5, -8.10e-6, &
        -9.60e-6, -1.45e-5, -2.13e-5, -5.14e-5, -3.02e-5, -1.46e-5, -6.80e-6  ], &
        [LNLAY, NCASE])
    bv = reshape([ &
         1.70e-5,  2.60e-5,  4.20e-5,  8.40e-5,  7.30e-5,  4.10e-5,  2.00e-5, &
         1.53e-5,  2.32e-5,  3.88e-5,  7.78e-5,  6.83e-5,  3.80e-5,  1.82e-5, &
         1.97e-5,  2.88e-5,  4.52e-5,  9.02e-5,  7.82e-5,  4.40e-5,  2.17e-5, &
         1.37e-5,  2.11e-5,  3.59e-5,  7.15e-5,  6.36e-5,  3.52e-5,  1.68e-5, &
         2.15e-5,  3.08e-5,  4.80e-5,  9.63e-5,  8.30e-5,  4.67e-5,  2.32e-5, &
         1.83e-5,  2.73e-5,  4.09e-5,  8.16e-5,  7.08e-5,  3.98e-5,  1.92e-5  ], &
        [LNLAY, NCASE])
    rv = reshape([ &
         1.10e-4, -9.00e-5,  1.40e-4,  2.51e-4, -1.32e-4,  8.70e-5, -4.40e-5, &
        -1.03e-4,  8.40e-5, -1.29e-4, -2.34e-4,  1.21e-4, -8.10e-5,  4.10e-5, &
         1.26e-4, -1.05e-4,  1.58e-4,  2.86e-4, -1.49e-4,  9.60e-5, -5.10e-5, &
        -9.20e-5,  7.60e-5, -1.17e-4, -2.09e-4,  1.10e-4, -7.30e-5,  3.70e-5, &
         1.41e-4, -1.18e-4,  1.72e-4,  3.14e-4, -1.63e-4,  1.05e-4, -5.60e-5, &
        -1.17e-4,  9.50e-5, -1.44e-4, -2.62e-4,  1.37e-4, -8.90e-5,  4.60e-5  ], &
        [LNLAY, NCASE])
    sv = reshape([ &
        270.10, 271.40, 272.30, 281.40, 283.10, 284.90, 286.20, &
        265.20, 267.80, 268.30, 270.50, 272.40, 274.80, 277.10, &
        259.40, 261.20, 265.70, 269.40, 271.80, 273.90, 276.30, &
        253.60, 258.90, 263.40, 268.10, 270.70, 273.20, 275.40, &
        249.80, 255.30, 261.90, 266.70, 269.90, 272.60, 274.80, &
        274.30, 275.10, 276.40, 291.70, 290.20, 288.60, 287.10  ], &
        [LNLAY, NCASE])
    dtv = [1800.0, 900.0, 600.0, 300.0, 120.0, 60.0]

    do ic = 1, NCASE
      do k = 1, LNLAY
        xc(k, ic)           = av(k, ic)
        xc(LNLAY + k, ic)   = bv(k, ic)
        xc(2*LNLAY + k, ic) = cv(k, ic)
        xc(3*LNLAY + k, ic) = rv(k, ic)
        xc(4*LNLAY + k, ic) = sv(k, ic)
      end do
      xc(5*LNLAY + 1, ic) = dtv(ic)
    end do

    ylive = .true.
    call run_leaf('hstep', eval_hstep, cn, ixn, ixc, xn, xi, xc, yn, yi, ylive)
  end subroutine dump_hstep

! ------------------------------------------------------------ leaf: tsnosoi --
!
! module_sf_noahmplsm.F:5258-5371.  The observable body ends at the
! unconditional RETURN on line 5346: everything from 5348 to 5369 -- the
! ERR_EST energy-balance check and its diagnostic WRITEs -- is unreachable.
! That RETURN is what makes six of this routine's arguments inert.
!
! What TSNOSOI actually does under the pinned identity:
!   5310  PHI(ISNOW+1:NSOIL) = 0.0            (a local, handed to HRT)
!   5314  ZBOTSNO = parameters%ZBOT - SNOWH
!   5318  TBEG(ISNOW+1:NSOIL) = STC(...)      (a local, read only after 5346)
!   5324  CALL HRT  (..., ZBOTSNO, ..., EFLXB)
!   5330  CALL HSTEP(..., STC)
!   5337  EFLXB2 = ...                        (a local, read only after 5346)
!
! `parameters` is referenced exactly once, for the scalar ZBOT at 5314, so the
! harness carries ZBOT as an ordinary input slot and leaves every other
! component untouched.  ZBOT is a table constant in a real forecast
! (ZBOT_TABLE, :11209) but it varies per case here: with a single value no
! freeze mutant on it could be killed, and the point of the fixture is to
! prove the port reads the argument rather than hard-coding the table.
!
! Inert, all six because of the RETURN at 5346 or because nothing references
! them at all:
!   ICE   (5275) never referenced anywhere in the routine
!   SAG   (5284) never referenced anywhere in the routine
!   IST   (5279) only in the diagnostic WRITE at 5367
!   ILOC  (5273) only in the diagnostic WRITE at 5367
!   JLOC  (5274) only in the diagnostic WRITE at 5367
!   TG    (5286) only in SSOIL2 at 5359
!   DZSNSO(5288) only in the ERR_EST accumulation at 5352
! All of them carry distinct non-zero values per case, so their inertness is
! measured rather than assumed.
!
! STC is INTENT(INOUT); HRT and HSTEP touch only ISNOW+1..NSOIL, so the layers
! above ISNOW must come back bit-identical and are flagged live=1 with ordinary
! values.  ZSNSO, DF and HCPCT above ISNOW are never read and are POISONed.
!
! Input slots: 1-7 zsnso(-2:4), 8-14 stc(-2:4), 15-21 df(-2:4),
! 22-28 hcpct(-2:4), 29-35 dzsnso(-2:4), 36 tbot, 37 ssoil, 38 sag, 39 dt,
! 40 snowh, 41 tg, 42 zbot.
! Output slots: 1-7 stc(-2:4), 8 eflxb.

  subroutine eval_tsnosoi(x, ix, y)
    real,    intent(in)  :: x(:)
    integer, intent(in)  :: ix(:)
    real,    intent(out) :: y(:)
    type(noahmp_parameters) :: p
    real :: zsnso(-LNSNOW+1:LNSOIL), stc(-LNSNOW+1:LNSOIL)
    real :: df(-LNSNOW+1:LNSOIL), hcpct(-LNSNOW+1:LNSOIL)
    real :: dzsnso(-LNSNOW+1:LNSOIL)
    real :: tbot, ssoil, sag, dt, snowh, tg, eflxb
    integer :: k, s

    s = 0
    do k = -LNSNOW+1, LNSOIL
      s = s + 1; zsnso(k) = x(s)
    end do
    do k = -LNSNOW+1, LNSOIL
      s = s + 1; stc(k) = x(s)
    end do
    do k = -LNSNOW+1, LNSOIL
      s = s + 1; df(k) = x(s)
    end do
    do k = -LNSNOW+1, LNSOIL
      s = s + 1; hcpct(k) = x(s)
    end do
    do k = -LNSNOW+1, LNSOIL
      s = s + 1; dzsnso(k) = x(s)
    end do
    s = s + 1; tbot  = x(s)
    s = s + 1; ssoil = x(s)
    s = s + 1; sag   = x(s)
    s = s + 1; dt    = x(s)
    s = s + 1; snowh = x(s)
    s = s + 1; tg    = x(s)
    s = s + 1; p%ZBOT = x(s)

    eflxb = 0.0
    call tsnosoi(p, ix(2), LNSOIL, LNSNOW, ix(1), ix(3), &
                 tbot, zsnso, ssoil, df, hcpct, &
                 sag, dt, snowh, dzsnso, &
                 tg, ix(4), ix(5), &
                 stc, eflxb)

    s = 0
    do k = -LNSNOW+1, LNSOIL
      s = s + 1; y(s) = stc(k)
    end do
    y(LNLAY + 1) = eflxb
  end subroutine eval_tsnosoi

  subroutine dump_tsnosoi()
    integer, parameter :: NX = 5*LNLAY + 7, NY = LNLAY + 1, NCASE = 6
    character(len=12) :: xn(NX), yn(NY)
    integer :: xi(NX), yi(NY)
    character(len=28) :: cn(NCASE)
    character(len=12) :: ixn(5)
    real :: xc(NX, NCASE)
    integer :: ixc(5, NCASE)
    logical :: ylive(NY, NCASE)
    integer :: k, s, ic
    real :: zs(LNLAY, NCASE), st(LNLAY, NCASE), dfv(LNLAY, NCASE)
    real :: hc(LNLAY, NCASE), dz(LNLAY, NCASE)
    real :: tbotv(NCASE), ssoilv(NCASE), sagv(NCASE), dtv(NCASE)
    real :: snowhv(NCASE), tgv(NCASE), zbotv(NCASE)

    ixn = [character(len=12) :: 'isnow', 'ice', 'ist', 'iloc', 'jloc']
    s = 0
    do k = -LNSNOW+1, LNSOIL
      s = s + 1
      xn(s) = 'zsnso';           xi(s) = k
      xn(LNLAY + s) = 'stc';     xi(LNLAY + s) = k
      xn(2*LNLAY + s) = 'df';    xi(2*LNLAY + s) = k
      xn(3*LNLAY + s) = 'hcpct'; xi(3*LNLAY + s) = k
      xn(4*LNLAY + s) = 'dzsnso'; xi(4*LNLAY + s) = k
      yn(s) = 'stc';             yi(s) = k
    end do
    xn(5*LNLAY + 1) = 'tbot';  xi(5*LNLAY + 1) = 0
    xn(5*LNLAY + 2) = 'ssoil'; xi(5*LNLAY + 2) = 0
    xn(5*LNLAY + 3) = 'sag';   xi(5*LNLAY + 3) = 0
    xn(5*LNLAY + 4) = 'dt';    xi(5*LNLAY + 4) = 0
    xn(5*LNLAY + 5) = 'snowh'; xi(5*LNLAY + 5) = 0
    xn(5*LNLAY + 6) = 'tg';    xi(5*LNLAY + 6) = 0
    xn(5*LNLAY + 7) = 'zbot';  xi(5*LNLAY + 7) = 0
    yn(LNLAY + 1) = 'eflxb';   yi(LNLAY + 1) = 0

    cn = [character(len=28) :: 'soil_only_isnow0', 'one_snow_layer', &
          'two_snow_layers', 'three_snow_layers', 'three_snow_layers_alt', &
          'warm_soil_isnow0']
    !            case  1   2   3   4   5   6
    ixc(1, :) = [  0, -1, -2, -3, -3,  0 ]   ! isnow
    ixc(2, :) = [  0,  0,  1, -1,  0,  1 ]   ! ice   (inert)
    ixc(3, :) = [  1,  1,  2,  1,  2,  1 ]   ! ist   (inert)
    ixc(4, :) = [  3, 17, 42,  8, 91,  5 ]   ! iloc  (inert)
    ixc(5, :) = [ 11,  4, 63, 27,  2, 88 ]   ! jloc  (inert)

    zs = reshape([ &
        POISON,  POISON,  POISON,  -0.10,  -0.40,  -1.00,  -2.00,  &
        POISON,  POISON,  -0.09,   -0.19,  -0.49,  -1.09,  -2.09,  &
        POISON,  -0.06,   -0.15,   -0.26,  -0.56,  -1.16,  -2.16,  &
        -0.05,   -0.13,   -0.25,   -0.35,  -0.65,  -1.25,  -2.25,  &
        -0.04,   -0.11,   -0.21,   -0.32,  -0.62,  -1.22,  -2.22,  &
        POISON,  POISON,  POISON,  -0.12,  -0.45,  -1.10,  -2.10   ], &
        [LNLAY, NCASE])
    ! STC above ISNOW is echoed, not read, so it carries ordinary values.
    st = reshape([ &
        270.10, 271.40, 272.30, 281.40, 283.10, 284.90, 286.20, &
        265.20, 267.80, 268.30, 270.50, 272.40, 274.80, 277.10, &
        259.40, 261.20, 265.70, 269.40, 271.80, 273.90, 276.30, &
        253.60, 258.90, 263.40, 268.10, 270.70, 273.20, 275.40, &
        249.80, 255.30, 261.90, 266.70, 269.90, 272.60, 274.80, &
        274.30, 275.10, 276.40, 291.70, 290.20, 288.60, 287.10  ], &
        [LNLAY, NCASE])
    dfv = reshape([ &
        POISON,  POISON,  POISON,  1.42,  1.63,  1.85,  1.94, &
        POISON,  POISON,  0.31,    1.38,  1.59,  1.81,  1.90, &
        POISON,  0.24,    0.35,    1.46,  1.67,  1.88,  1.97, &
        0.19,    0.27,    0.41,    1.51,  1.72,  1.92,  2.01, &
        0.22,    0.30,    0.38,    1.33,  1.55,  1.79,  1.88, &
        POISON,  POISON,  POISON,  1.28,  1.49,  1.71,  1.83  ], &
        [LNLAY, NCASE])
    hc = reshape([ &
        POISON,  POISON,  POISON,  2.31e6, 2.44e6, 2.57e6, 2.66e6, &
        POISON,  POISON,  5.12e5,  2.28e6, 2.41e6, 2.54e6, 2.63e6, &
        POISON,  3.94e5,  6.05e5,  2.35e6, 2.48e6, 2.61e6, 2.70e6, &
        2.87e5,  4.63e5,  7.18e5,  2.39e6, 2.52e6, 2.65e6, 2.74e6, &
        3.15e5,  5.27e5,  6.41e5,  2.22e6, 2.36e6, 2.49e6, 2.58e6, &
        POISON,  POISON,  POISON,  2.19e6, 2.33e6, 2.46e6, 2.55e6  ], &
        [LNLAY, NCASE])
    ! DZSNSO is inert (5352, after the RETURN at 5346), so it carries the
    ! physically correct thicknesses -- distinct in every slot of every case --
    ! and the sweep proves none of them reaches an output.
    dz = reshape([ &
        0.031, 0.042, 0.058, 0.100, 0.300, 0.600, 1.000, &
        0.034, 0.046, 0.090, 0.100, 0.300, 0.600, 1.000, &
        0.037, 0.060, 0.090, 0.110, 0.300, 0.600, 1.000, &
        0.050, 0.080, 0.120, 0.100, 0.300, 0.600, 1.000, &
        0.040, 0.070, 0.100, 0.110, 0.300, 0.600, 1.000, &
        0.029, 0.048, 0.063, 0.120, 0.330, 0.650, 1.000  ], &
        [LNLAY, NCASE])
    tbotv  = [285.60, 283.20, 281.90, 280.40, 279.70, 287.30]
    ssoilv = [-37.40, 22.80, 45.10, 68.30, -19.60, 53.70]
    sagv   = [412.70, 168.30, 95.40, 231.80, 57.20, 604.10]
    dtv    = [1800.0, 900.0, 600.0, 300.0, 120.0, 60.0]
    snowhv = [0.0135, 0.0620, 0.1970, 0.4130, 0.3720, 0.0208]
    tgv    = [288.40, 271.20, 268.90, 274.60, 259.40, 295.10]
    zbotv  = [-8.00, -7.85, -8.20, -8.40, -7.60, -8.10]

    do ic = 1, NCASE
      do k = 1, LNLAY
        xc(k, ic)           = zs(k, ic)
        xc(LNLAY + k, ic)   = st(k, ic)
        xc(2*LNLAY + k, ic) = dfv(k, ic)
        xc(3*LNLAY + k, ic) = hc(k, ic)
        xc(4*LNLAY + k, ic) = dz(k, ic)
      end do
      xc(5*LNLAY + 1, ic) = tbotv(ic)
      xc(5*LNLAY + 2, ic) = ssoilv(ic)
      xc(5*LNLAY + 3, ic) = sagv(ic)
      xc(5*LNLAY + 4, ic) = dtv(ic)
      xc(5*LNLAY + 5, ic) = snowhv(ic)
      xc(5*LNLAY + 6, ic) = tgv(ic)
      xc(5*LNLAY + 7, ic) = zbotv(ic)
    end do

    ylive = .true.
    call run_leaf('tsnosoi', eval_tsnosoi, cn, ixn, ixc, xn, xi, xc, &
                  yn, yi, ylive)
  end subroutine dump_tsnosoi

! --------------------------------------------------------- leaf: phasechange --
!
! module_sf_noahmplsm.F:5595-5810 under the pinned OPT_FRZ = 1.  The
! supercooled-water content comes from the closed form at 5683-5689; the
! `CALL FRH2O` at 5690-5693 is OPT_FRZ == 2 and is dead, so it is neither
! reached nor ported.
!
! `parameters` is referenced only for SMCMAX, PSISAT and BEXP inside that
! closed form, so the harness carries those three per soil layer.
!
! Inert:
!   HCPCT  (5617) declared INTENT(IN); the body never references it
!   ILOC   (5608) declared INTENT(IN); the body never references it
!   JLOC   (5609) declared INTENT(IN); the body never references it
!   DZSNSO(-2:0)  DZSNSO is read only for soil layers (5666-5667, 5687,
!                 5804-5805), so its snow half reaches nothing
! All four carry ordinary, distinct, non-zero values in every case.
!
! XMF (5646, 5751, 5789) accumulates the latent heat of the phase change into
! a local that nothing reads: WRF neither returns it nor uses it again.  It is
! not an output here and it is not transcribed.
!
! Branch notes, all of which the validator asserts:
!   * `IMELT==1 .AND. HM<0` (5723) and `IMELT==2 .AND. HM>0` (5727) cannot fire
!     at any call site, because FACT is DT/(HCPCT*DZSNSO) (2497-2499) and is
!     therefore strictly positive, which ties sign(HM) to the IMELT test that
!     produced it.  `fact_sign_probe` gives FACT a negative sign so both reset
!     bodies are measured; nothing in a forecast can reach them.
!   * `WMASS0(J) < SUPERCOOL(J)` (5768) is UNREACHABLE, not merely untested:
!     that line is inside `XM(J) < 0`, which requires HM(J) < 0, which under a
!     positive FACT requires IMELT(J) == 2, which requires
!     MLIQ(J) > SUPERCOOL(J) (5703); and WMASS0 = MICE + MLIQ with MICE =
!     (SMC-SH2O)*DZ*1000 >= 0, so WMASS0 >= MLIQ > SUPERCOOL.  No case binds it
!     and none can without an SH2O > SMC column, which SOILWATER cannot
!     produce.  The port transcribes it and says so.
!   * `MAX(MICE(J), 0.0)` (5773) is a no-op for the same reason.
!
! Input slots: 1-7 fact(-2:4), 8-14 dzsnso(-2:4), 15-21 hcpct(-2:4),
! 22-28 stc(-2:4), 29-31 snice(-2:0), 32-34 snliq(-2:0), 35-38 smc(1:4),
! 39-42 sh2o(1:4), 43 sneqv, 44 snowh, 45 dt, 46-49 smcmax(1:4),
! 50-53 psisat(1:4), 54-57 bexp(1:4).
! Output slots: 1-7 stc(-2:4), 8-10 snice(-2:0), 11-13 snliq(-2:0),
! 14-17 smc(1:4), 18-21 sh2o(1:4), 22 sneqv, 23 snowh, 24 qmelt, 25 ponding,
! 26-32 imelt(-2:4) as REAL (exact for the values 0, 1, 2).

  subroutine eval_phasechange(x, ix, y)
    real,    intent(in)  :: x(:)
    integer, intent(in)  :: ix(:)
    real,    intent(out) :: y(:)
    type(noahmp_parameters) :: p
    real :: fact(-LNSNOW+1:LNSOIL), dzsnso(-LNSNOW+1:LNSOIL)
    real :: hcpct(-LNSNOW+1:LNSOIL), stc(-LNSNOW+1:LNSOIL)
    real :: snice(-LNSNOW+1:0), snliq(-LNSNOW+1:0)
    real :: smc(LNSOIL), sh2o(LNSOIL)
    real :: sneqv, snowh, dt, qmelt, ponding
    integer :: imelt(-LNSNOW+1:LNSOIL)
    integer :: k, s

    s = 0
    do k = -LNSNOW+1, LNSOIL
      s = s + 1; fact(k) = x(s)
    end do
    do k = -LNSNOW+1, LNSOIL
      s = s + 1; dzsnso(k) = x(s)
    end do
    do k = -LNSNOW+1, LNSOIL
      s = s + 1; hcpct(k) = x(s)
    end do
    do k = -LNSNOW+1, LNSOIL
      s = s + 1; stc(k) = x(s)
    end do
    do k = -LNSNOW+1, 0
      s = s + 1; snice(k) = x(s)
    end do
    do k = -LNSNOW+1, 0
      s = s + 1; snliq(k) = x(s)
    end do
    do k = 1, LNSOIL
      s = s + 1; smc(k) = x(s)
    end do
    do k = 1, LNSOIL
      s = s + 1; sh2o(k) = x(s)
    end do
    s = s + 1; sneqv = x(s)
    s = s + 1; snowh = x(s)
    s = s + 1; dt    = x(s)
    do k = 1, LNSOIL
      s = s + 1; p%SMCMAX(k) = x(s)
    end do
    do k = 1, LNSOIL
      s = s + 1; p%PSISAT(k) = x(s)
    end do
    do k = 1, LNSOIL
      s = s + 1; p%BEXP(k) = x(s)
    end do

    imelt = 0
    qmelt = 0.0
    ponding = 0.0

    call phasechange(p, LNSNOW, LNSOIL, ix(1), dt, fact, &
                     dzsnso, hcpct, ix(2), ix(3), ix(4), &
                     stc, snice, snliq, sneqv, snowh, &
                     smc, sh2o, &
                     qmelt, imelt, ponding)

    s = 0
    do k = -LNSNOW+1, LNSOIL
      s = s + 1; y(s) = stc(k)
    end do
    s = LNLAY
    do k = -LNSNOW+1, 0
      s = s + 1
      y(s) = snice(k)
      y(LNSNOW + s) = snliq(k)
    end do
    s = LNLAY + 2*LNSNOW
    do k = 1, LNSOIL
      s = s + 1
      y(s) = smc(k)
      y(LNSOIL + s) = sh2o(k)
    end do
    s = LNLAY + 2*LNSNOW + 2*LNSOIL
    y(s + 1) = sneqv
    y(s + 2) = snowh
    y(s + 3) = qmelt
    y(s + 4) = ponding
    s = s + 4
    do k = -LNSNOW+1, LNSOIL
      s = s + 1; y(s) = real(imelt(k))
    end do
  end subroutine eval_phasechange

  subroutine dump_phasechange()
    integer, parameter :: NX = 4*LNLAY + 2*LNSNOW + 2*LNSOIL + 3 + 3*LNSOIL
    integer, parameter :: NY = LNLAY + 2*LNSNOW + 2*LNSOIL + 4 + LNLAY
    integer, parameter :: NCASE = 10
    character(len=12) :: xn(NX), yn(NY)
    integer :: xi(NX), yi(NY)
    character(len=28) :: cn(NCASE)
    character(len=12) :: ixn(4)
    real :: xc(NX, NCASE)
    integer :: ixc(4, NCASE)
    logical :: ylive(NY, NCASE)
    integer :: k, s, ic, isnow, base
    real :: fa(LNLAY, NCASE), dz(LNLAY, NCASE), hc(LNLAY, NCASE)
    real :: st(LNLAY, NCASE), sni(LNSNOW, NCASE), snl(LNSNOW, NCASE)
    real :: smcv(LNSOIL, NCASE), sh2ov(LNSOIL, NCASE)
    real :: mxv(LNSOIL, NCASE), psv(LNSOIL, NCASE), bxv(LNSOIL, NCASE)
    real :: sneqvv(NCASE), snowhv(NCASE), dtv(NCASE)

    ixn = [character(len=12) :: 'isnow', 'ist', 'iloc', 'jloc']
    s = 0
    do k = -LNSNOW+1, LNSOIL
      s = s + 1
      xn(s) = 'fact';            xi(s) = k
      xn(LNLAY + s) = 'dzsnso';  xi(LNLAY + s) = k
      xn(2*LNLAY + s) = 'hcpct'; xi(2*LNLAY + s) = k
      xn(3*LNLAY + s) = 'stc';   xi(3*LNLAY + s) = k
      yn(s) = 'stc';             yi(s) = k
    end do
    base = 4*LNLAY
    do k = -LNSNOW+1, 0
      base = base + 1
      xn(base) = 'snice'; xi(base) = k
    end do
    do k = -LNSNOW+1, 0
      base = base + 1
      xn(base) = 'snliq'; xi(base) = k
    end do
    do k = 1, LNSOIL
      base = base + 1
      xn(base) = 'smc'; xi(base) = k
    end do
    do k = 1, LNSOIL
      base = base + 1
      xn(base) = 'sh2o'; xi(base) = k
    end do
    base = base + 1; xn(base) = 'sneqv'; xi(base) = 0
    base = base + 1; xn(base) = 'snowh'; xi(base) = 0
    base = base + 1; xn(base) = 'dt';    xi(base) = 0
    do k = 1, LNSOIL
      base = base + 1
      xn(base) = 'smcmax'; xi(base) = k
    end do
    do k = 1, LNSOIL
      base = base + 1
      xn(base) = 'psisat'; xi(base) = k
    end do
    do k = 1, LNSOIL
      base = base + 1
      xn(base) = 'bexp'; xi(base) = k
    end do

    base = LNLAY
    do k = -LNSNOW+1, 0
      base = base + 1
      yn(base) = 'snice';           yi(base) = k
      yn(LNSNOW + base) = 'snliq';  yi(LNSNOW + base) = k
    end do
    base = LNLAY + 2*LNSNOW
    do k = 1, LNSOIL
      base = base + 1
      yn(base) = 'smc';            yi(base) = k
      yn(LNSOIL + base) = 'sh2o';  yi(LNSOIL + base) = k
    end do
    base = LNLAY + 2*LNSNOW + 2*LNSOIL
    yn(base + 1) = 'sneqv';   yi(base + 1) = 0
    yn(base + 2) = 'snowh';   yi(base + 2) = 0
    yn(base + 3) = 'qmelt';   yi(base + 3) = 0
    yn(base + 4) = 'ponding'; yi(base + 4) = 0
    base = base + 4
    do k = -LNSNOW+1, LNSOIL
      base = base + 1
      yn(base) = 'imelt'; yi(base) = k
    end do

    cn = [character(len=28) :: 'soil_melt_and_freeze', &
          'soil_column_refreeze', 'no_layer_partial_melt', &
          'no_layer_total_melt', 'no_layer_heat_deficit', &
          'three_snow_melt', 'three_snow_freeze', 'lake_two_snow_ist2', &
          'one_snow_mixed', 'fact_sign_probe']
    !            case  1   2   3   4   5   6   7   8   9  10
    ixc(1, :) = [  0,  0,  0,  0,  0, -3, -3, -2, -1,  0 ]   ! isnow
    ixc(2, :) = [  1,  1,  1,  1,  1,  1,  1,  2,  1,  1 ]   ! ist
    ixc(3, :) = [  3, 17, 42,  8, 91,  5, 23, 61, 12, 74 ]   ! iloc  (inert)
    ixc(4, :) = [ 11,  4, 63, 27,  2, 88, 39,  7, 55, 16 ]   ! jloc  (inert)

    ! FACT above ISNOW is never read (the loops all start at ISNOW+1), so it
    ! is POISONed there.  Case 10 gives the two soil layers a NEGATIVE FACT --
    ! see the branch note above.
    fa = reshape([ &
        POISON,  POISON,  POISON,  7.80e-3, 2.60e-3, 1.30e-3, 7.90e-4, &
        POISON,  POISON,  POISON,  6.40e-3, 2.20e-3, 1.10e-3, 6.60e-4, &
        POISON,  POISON,  POISON,  7.80e-3, 2.60e-3, 1.30e-3, 7.90e-4, &
        POISON,  POISON,  POISON,  2.10e-2, 2.60e-3, 1.30e-3, 7.90e-4, &
        POISON,  POISON,  POISON,  4.00e-4, 2.60e-3, 1.30e-3, 7.90e-4, &
        7.40e-2, 4.10e-2, 2.60e-2, 7.80e-3, 2.60e-3, 1.30e-3, 7.90e-4, &
        6.10e-2, 3.50e-2, 2.20e-2, 6.90e-3, 2.30e-3, 1.10e-3, 6.80e-4, &
        POISON,  3.00e-2, 1.90e-2, 5.40e-3, 2.00e-3, 9.80e-4, 6.00e-4, &
        POISON,  POISON,  1.40e-2, 3.10e-3, 1.20e-3, 6.10e-4, 3.70e-4, &
        POISON,  POISON,  POISON, -5.20e-3,-1.80e-3, 9.40e-4, 5.50e-4  ], &
        [LNLAY, NCASE])
    ! DZSNSO(-2:0) is inert: it carries the physically correct snow
    ! thicknesses so its inertness is measured, not vacuous.
    dz = reshape([ &
        0.031, 0.042, 0.058, 0.10, 0.30, 0.60, 1.00, &
        0.036, 0.049, 0.061, 0.09, 0.28, 0.58, 0.95, &
        0.029, 0.048, 0.063, 0.11, 0.31, 0.62, 1.02, &
        0.030, 0.045, 0.060, 0.09, 0.28, 0.58, 0.95, &
        0.032, 0.047, 0.065, 0.12, 0.29, 0.61, 0.98, &
        0.050, 0.080, 0.120, 0.10, 0.30, 0.60, 1.00, &
        0.040, 0.070, 0.100, 0.11, 0.31, 0.62, 1.02, &
        0.037, 0.060, 0.090, 0.12, 0.29, 0.61, 0.98, &
        0.034, 0.046, 0.090, 0.10, 0.30, 0.60, 1.00, &
        0.033, 0.051, 0.067, 0.13, 0.33, 0.65, 1.05  ], &
        [LNLAY, NCASE])
    ! HCPCT is inert (5617): plausible, distinct, non-zero everywhere.
    hc = reshape([ &
        2.87e5, 4.63e5, 7.18e5, 2.31e6, 2.44e6, 2.57e6, 2.66e6, &
        3.15e5, 5.27e5, 6.41e5, 2.28e6, 2.41e6, 2.54e6, 2.63e6, &
        2.94e5, 4.88e5, 6.72e5, 2.35e6, 2.48e6, 2.61e6, 2.70e6, &
        3.06e5, 5.02e5, 7.44e5, 2.39e6, 2.52e6, 2.65e6, 2.74e6, &
        2.71e5, 4.35e5, 6.09e5, 2.22e6, 2.36e6, 2.49e6, 2.58e6, &
        3.38e5, 5.61e5, 7.90e5, 2.19e6, 2.33e6, 2.46e6, 2.55e6, &
        2.55e5, 4.14e5, 6.86e5, 2.42e6, 2.55e6, 2.68e6, 2.77e6, &
        3.22e5, 5.44e5, 7.31e5, 2.26e6, 2.39e6, 2.52e6, 2.61e6, &
        2.79e5, 4.71e5, 6.53e5, 2.33e6, 2.46e6, 2.59e6, 2.68e6, &
        3.41e5, 5.18e5, 7.62e5, 2.20e6, 2.34e6, 2.47e6, 2.56e6  ], &
        [LNLAY, NCASE])
    ! STC, SNICE and SNLIQ above ISNOW are INTENT(INOUT) and echoed, so they
    ! carry ordinary values rather than POISON: the echo is asserted.
    st = reshape([ &
        270.10, 271.40, 272.30, 274.60, 272.10, 271.16, 269.40, &
        269.90, 270.70, 271.50, 272.55, 271.85, 272.95, 270.40, &
        271.20, 272.00, 272.80, 274.20, 273.40, 272.60, 271.90, &
        270.90, 271.70, 272.40, 278.90, 275.10, 274.20, 273.50, &
        270.60, 271.30, 272.20, 273.1601, 275.10, 274.20, 273.50, &
        273.90, 273.42, 273.16, 274.10, 273.60, 272.90, 271.80, &
        269.40, 270.80, 272.10, 272.40, 271.60, 270.90, 269.70, &
        268.70, 272.30, 273.55, 273.90, 272.10, 271.40, 274.60, &
        267.90, 269.10, 273.30, 272.85, 271.95, 273.40, 270.60, &
        269.80, 270.60, 271.90, 275.30, 271.70, 272.40, 274.10  ], &
        [LNLAY, NCASE])
    sni = reshape([ &
         0.31,  0.62,  0.94,  &
         0.28,  0.55,  0.87,  &
         0.35,  0.71,  1.02,  &
         0.24,  0.49,  0.78,  &
         0.41,  0.83,  1.16,  &
         0.04,  8.40, 21.60,  &
        14.70, 26.30, 41.20,  &
         0.52, 18.20, 33.40,  &
         0.19,  0.38, 12.40,  &
         0.46,  0.91,  1.33   ], [LNSNOW, NCASE])
    snl = reshape([ &
        0.11, 0.23, 0.37, &
        0.09, 0.19, 0.31, &
        0.14, 0.27, 0.42, &
        0.07, 0.16, 0.26, &
        0.17, 0.33, 0.48, &
        0.90, 1.35, 3.10, &
        2.40, 3.10, 5.60, &
        0.21, 2.70, 4.90, &
        0.06, 0.13, 1.90, &
        0.19, 0.36, 0.54  ], [LNSNOW, NCASE])
    ! Cases 1, 2 and 9 put SH2O just above the supercooled level that
    ! 5685 computes, so that MLIQ - SUPERCOOL < |XM| and MICE at 5771 is
    ! SUPERCOOL-limited rather than XM-limited.  That is the only regime in
    ! which SMCMAX, PSISAT and BEXP reach an output at all, and three cases
    ! with distinct parameter values are what makes their freeze mutants die.
    smcv = reshape([ &
        0.30000, 0.20648, 0.30127, 0.09314,   0.10169, 0.14202, 0.37039, 0.18297, &
        0.29000, 0.30500, 0.28800, 0.27600,   0.28000, 0.30000, 0.29000, 0.27000, &
        0.26700, 0.28100, 0.29300, 0.30900,   0.30000, 0.31000, 0.29500, 0.28000, &
        0.32000, 0.30000, 0.28500, 0.27000,   0.31000, 0.29500, 0.28800, 0.26500, &
        0.18660, 0.09590, 0.33600, 0.10225,   0.29500, 0.28800, 0.30200, 0.27100  ], &
        [LNSOIL, NCASE])
    sh2ov = reshape([ &
        0.10000, 0.14648, 0.24127, 0.03314,   0.04169, 0.08202, 0.31039, 0.12297, &
        0.18000, 0.24000, 0.15000, 0.07000,   0.19000, 0.25000, 0.16000, 0.08000, &
        0.19000, 0.25000, 0.16000, 0.08000,   0.26000, 0.29000, 0.15000, 0.06000, &
        0.25000, 0.18000, 0.14000, 0.05500,   0.22000, 0.20000, 0.13000, 0.26500, &
        0.12660, 0.03590, 0.27600, 0.04225,   0.14000, 0.23000, 0.17000, 0.27100  ], &
        [LNSOIL, NCASE])
    mxv = reshape([ &
        0.434, 0.439, 0.464, 0.421,   0.404, 0.412, 0.421, 0.430, &
        0.464, 0.455, 0.447, 0.439,   0.421, 0.418, 0.415, 0.412, &
        0.398, 0.406, 0.414, 0.422,   0.339, 0.347, 0.356, 0.364, &
        0.446, 0.441, 0.437, 0.433,   0.476, 0.468, 0.459, 0.451, &
        0.412, 0.424, 0.436, 0.448,   0.443, 0.431, 0.427, 0.419  ], &
        [LNSOIL, NCASE])
    psv = reshape([ &
        0.141, 0.355, 0.617, 0.069,   0.036, 0.141, 0.759, 0.355, &
        0.617, 0.759, 0.263, 0.098,   0.069, 0.036, 0.141, 0.617, &
        0.759, 0.069, 0.098, 0.036,   0.098, 0.263, 0.355, 0.141, &
        0.263, 0.098, 0.036, 0.759,   0.355, 0.617, 0.069, 0.263, &
        0.141, 0.036, 0.617, 0.098,   0.203, 0.311, 0.052, 0.428  ], &
        [LNSOIL, NCASE])
    bxv = reshape([ &
         4.74,  5.25,  8.72,  2.79,    3.31,  4.26, 11.55,  5.25, &
         8.72, 11.55,  6.77,  3.30,    2.79,  3.31,  4.74,  8.72, &
        11.55,  2.79,  3.30,  3.31,    3.30,  6.77,  5.25,  4.74, &
         6.77,  3.30,  3.31, 11.55,    5.25,  8.72,  2.79,  6.77, &
         4.74,  3.31,  8.72,  3.30,    7.12,  4.03,  9.41,  2.95  ], &
        [LNSOIL, NCASE])
    sneqvv = [0.00, 0.00, 12.40, 0.85, 250.00, 35.40, 93.30, 59.20, 14.30, 0.00]
    snowhv = [0.0410, 0.0375, 0.0248, 0.0034, 0.5500, 0.2500, 0.2100, &
              0.1500, 0.0900, 0.0290]
    dtv    = [1800.0, 900.0, 1800.0, 3600.0, 60.0, 1800.0, 900.0, 600.0, &
              300.0, 1200.0]

    do ic = 1, NCASE
      do k = 1, LNLAY
        xc(k, ic)           = fa(k, ic)
        xc(LNLAY + k, ic)   = dz(k, ic)
        xc(2*LNLAY + k, ic) = hc(k, ic)
        xc(3*LNLAY + k, ic) = st(k, ic)
      end do
      base = 4*LNLAY
      do k = 1, LNSNOW
        xc(base + k, ic) = sni(k, ic)
        xc(base + LNSNOW + k, ic) = snl(k, ic)
      end do
      base = base + 2*LNSNOW
      do k = 1, LNSOIL
        xc(base + k, ic) = smcv(k, ic)
        xc(base + LNSOIL + k, ic) = sh2ov(k, ic)
      end do
      base = base + 2*LNSOIL
      xc(base + 1, ic) = sneqvv(ic)
      xc(base + 2, ic) = snowhv(ic)
      xc(base + 3, ic) = dtv(ic)
      base = base + 3
      do k = 1, LNSOIL
        xc(base + k, ic) = mxv(k, ic)
        xc(base + LNSOIL + k, ic) = psv(k, ic)
        xc(base + 2*LNSOIL + k, ic) = bxv(k, ic)
      end do
    end do

    ! Everything is INTENT(INOUT) and therefore defined, except IMELT, which
    ! is INTENT(OUT) and assigned only over ISNOW+1..NSOIL (5663).
    ylive = .true.
    do ic = 1, NCASE
      isnow = ixc(1, ic)
      base = LNLAY + 2*LNSNOW + 2*LNSOIL + 4
      s = 0
      do k = -LNSNOW+1, LNSOIL
        s = s + 1
        ylive(base + s, ic) = (k >= isnow + 1)
      end do
    end do

    call run_leaf('phasechange', eval_phasechange, cn, ixn, ixc, xn, xi, xc, &
                  yn, yi, ylive)
  end subroutine dump_phasechange

! -------------------------------------------------------------- leaf: frh2o --
!
! module_sf_noahmplsm.F:5814-5946.  DEAD under the pinned option identity: the
! module's only CALL FRH2O is line 5692, inside `IF (OPT_FRZ == 2)`, and the
! pinned identity is opt_frz = 1.  It is pinned here because it is a
! self-contained leaf whose fixture costs nothing; nothing in gpuwm calls it,
! and the `phasechange` port implements the 5683-5689 closed form instead.
!
! Reads parameters%BEXP(ISOIL), PSISAT(ISOIL) and SMCMAX(ISOIL) only, so the
! fixture carries those three as ordinary inputs and the harness poisons every
! other layer of all three arrays.  ISOIL is fixed at 1 because the routine is
! scalar in the layer.  DICE (5852) is a named constant the body never uses.
!
! Branch coverage, all asserted by the validator from the inputs:
!   5872  TKELV > TFRZ-1.0E-3  -> FREE = SMC          warm_shortcut
!   5866  BEXP > BLIM(5.5)     -> BX = BLIM           clay_bexp_above_blim
!   5883  SWL > SMC-0.02       (needs SH2O < 0.02)    dry_sh2o_below_two_pct
!   5887  SWL < 0.0            (needs SMC  < 0.02)    trace_moisture_column
!   5899  SWLK > SMC-0.02                             swlk_high_clamp
!   5900  SWLK < 0.0                                  swlk_low_clamp
!   5878  CK /= 0.0 is a comparison against the PARAMETER CK = 8.0 and is
!         therefore always true; the ELSE does not exist.
!
! NOT bound: the Flerchinger fallback at 5928-5935, which needs the Newton
! loop to run all ten iterations without DSWL falling to ERROR = 0.005.  A
! 200000-draw random search over TKELV in [200, 273.16], SMC in (0, SMCMAX],
! SH2O in [0, SMC], BEXP in [2, 14], PSISAT in [0.01, 1.5] and SMCMAX in
! [0.2, 0.6] never reached it.  That is evidence, not a proof, and the port
! says so rather than claiming the branch is unreachable.

  subroutine eval_frh2o(x, ix, y)
    real,    intent(in)  :: x(:)
    integer, intent(in)  :: ix(:)
    real,    intent(out) :: y(:)
    type(noahmp_parameters) :: p
    real :: free
    p%BEXP = POISON
    p%PSISAT = POISON
    p%SMCMAX = POISON
    p%BEXP(1) = x(4)
    p%PSISAT(1) = x(5)
    p%SMCMAX(1) = x(6)
    free = 0.0
    call frh2o(p, 1, free, x(1), x(2), x(3))
    y(1) = free
  end subroutine eval_frh2o

  subroutine dump_frh2o()
    integer, parameter :: NX = 6, NY = 1, NCASE = 8
    character(len=12) :: xn(NX), yn(NY)
    integer :: xi(NX), yi(NY)
    character(len=28) :: cn(NCASE)
    character(len=12) :: ixn(0)
    real :: xc(NX, NCASE)
    integer :: ixc(0, NCASE)
    logical :: ylive(NY, NCASE)

    xn = [character(len=12) :: 'tkelv', 'smc', 'sh2o', 'bexp', 'psisat', &
          'smcmax']
    xi = [0, 0, 0, 1, 1, 1]
    yn = [character(len=12) :: 'free']
    yi = 0
    cn = [character(len=28) :: 'warm_shortcut', 'cold_loam', &
          'clay_bexp_above_blim', 'dry_sh2o_below_two_pct', &
          'trace_moisture_column', 'swlk_high_clamp', 'swlk_low_clamp', &
          'sand_wet_cold']

    !             tkelv      smc     sh2o    bexp   psisat  smcmax
    xc(:, 1) = [ 273.200,  0.3000,  0.3000,  4.740,  0.141,  0.434 ]
    xc(:, 2) = [ 271.160,  0.3000,  0.1000,  4.740,  0.141,  0.434 ]
    xc(:, 3) = [ 268.400,  0.3700,  0.1200,  8.720,  0.617,  0.464 ]
    xc(:, 4) = [ 265.000,  0.2800,  0.0100, 11.550,  0.759,  0.421 ]
    xc(:, 5) = [ 262.300,  0.0120,  0.0040,  3.310,  0.036,  0.404 ]
    xc(:, 6) = [ 250.000,  0.0600,  0.0200,  2.790,  0.069,  0.421 ]
    xc(:, 7) = [ 273.159,  0.3100,  0.1800,  6.770,  0.263,  0.446 ]
    xc(:, 8) = [ 255.000,  0.4300,  0.3000,  5.250,  0.355,  0.439 ]

    ylive = .true.
    call run_leaf('frh2o', eval_frh2o, cn, ixn, ixc, xn, xi, xc, yn, yi, ylive)
  end subroutine dump_frh2o

! -------------------------------------------------------------- entry point --

  subroutine dump_all()
    call dump_hrt()
    call dump_hstep()
    call dump_tsnosoi()
    call dump_phasechange()
    call dump_frh2o()
  end subroutine dump_all

end module noahmp_thermal_oracle


program run_noahmp_thermal_oracle
  use module_sf_noahmplsm, only: noahmp_options
  use noahmp_thermal_oracle, only: open_outputs, close_outputs, dump_all
  implicit none
  character(len=1024) :: leaf_path, disc_path

  call get_command_argument(1, leaf_path)
  call get_command_argument(2, disc_path)
  if (len_trim(leaf_path) == 0 .or. len_trim(disc_path) == 0) then
    write(*, '(A)') 'usage: run_thermal THERMAL.csv DISCRIMINATION.csv'
    error stop 2
  end if

  ! The pinned WRF Registry default identity.
  call noahmp_options(4, 1, 1, 3, 1, 1, &
                      1, 3, 2, 1, 2, 1, &
                      1, 1, 1, 0, 0, 0, &
                      0, 0)

  call open_outputs(leaf_path, disc_path)
  call dump_all()
  call close_outputs()
end program run_noahmp_thermal_oracle
