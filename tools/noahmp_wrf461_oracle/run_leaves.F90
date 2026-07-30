! Per-leaf oracle for WRF v4.6.1 MODULE_SF_NOAHMPLSM.
!
! This program links the *visibility-patched* module (see
! patches/noahmp-lsm-leaf-visibility.patch, audited by
! check_visibility_patch.py before this file is compiled).  Apart from the 50
! `private ::` -> `public ::` accessibility statements, the physics source is
! the pinned WRF v4.6.1 file byte for byte.  The pristine whole-column oracle
! (build.sh / run_sflx.F90) keeps compiling the UNPATCHED module.
!
! Every leaf is driven as a pure function of a flat FP32 input vector `x` plus
! a small integer topology vector `ix`, producing a flat FP32 output vector
! `y`.  That uniformity buys three things:
!
!   * one CSV schema for every leaf, carrying the raw IEEE-754 bit pattern of
!     each value, so `max_ulp 0` comparison is exact rather than a decimal
!     round trip;
!   * a generic zero-probe sweep on the ORACLE side (this file), recording for
!     every case and every input slot how many live outputs move when that slot
!     is zeroed -- evidence about what the Fortran actually reads;
!   * a generic MUTATION study on the PORT side
!     (gpuwm/core/noahmp_leaf_mutation.py), which is the stronger gate: for
!     every input slot and every constant that slot takes anywhere in the
!     fixture, the mutant that freezes the slot to that constant must FAIL to
!     reproduce the fixture.  A fixture that cannot kill a frozen argument
!     cannot detect a port that ignores it.
!
! The mutation criterion is what fixes the case tables below.  It requires,
! for every argument the pinned option identity actually consumes:
!
!   * at least two distinct values across the fixture (otherwise a port could
!     hard-code the constant);
!   * for a threshold-only argument, cases on BOTH sides of the threshold
!     (otherwise no constant flips the test) -- hence `lake_thawed_column`
!     with every soil STC above TFRZ and `lake_frozen_column` with every soil
!     STC below it;
!   * for an argument that a topology gates, at least two cases with that
!     topology (otherwise freezing to the single reader's own value is a
!     no-op) -- hence two ISNOW=-3 cases in `rosr12`, `csnow` and
!     `thermoprop`, and a non-zero SOLDN in `atm`'s nocturnal case.
!
! Array regions the callee leaves undefined (INTENT(OUT) slots above ISNOW,
! ROSR12 entries above NTOP) are pre-filled with 0.0 by the harness and
! flagged live=0.  The CPU/CUDA ports adopt the same pre-fill; the validator
! compares bitwise on live=1 slots and asserts the pre-fill survived on
! live=0 slots.
!
! Regions a leaf must never read are filled with POISON = -9999.0 instead of a
! plausible value, so a port that reads them is visibly wrong.
!
! Option identity: the pinned WRF Registry default
!   dveg=4 opt_crs=1 opt_btr=1 opt_run=3 opt_sfc=1 opt_frz=1 opt_inf=1
!   opt_rad=3 opt_alb=2 opt_snf=1 opt_tbot=2 opt_stc=1 opt_rsf=1 opt_soil=1
!   opt_pedo=1 opt_crop=0 opt_irr=0 opt_irrm=0 opt_infdv=0 opt_tdrn=0
! Only ATM reads an option variable (OPT_SNF); no other leaf in this file
! reads one.

module noahmp_leaf_oracle

  use module_sf_noahmplsm, only: noahmp_parameters, &
      atm, esat, rosr12, csnow, tdfcnd, thermoprop, wdfcnd1, wdfcnd2
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

! ---------------------------------------------------------------- leaf: atm --
!
! module_sf_noahmplsm.F:1083-1251.  The body never references `parameters`
! under any OPT_SNF, so the harness passes an untouched handle.  PRCPSNOW /
! PRCPGRPL / PRCPHAIL are read only inside `IF(OPT_SNF == 4)` (1217-1228) and
! are therefore inert; they carry non-zero values in four cases so their
! inertness is a measurement rather than a vacuum.
!
! `nocturnal_dark` carries SOLDN = 45 W/m2 with COSZ < 0.  That is not physical
! forcing, it is what makes COSZ mutation-detectable: with SOLDN = 0 the
! COSZ<=0 branch and its complement produce the same SWDOWN, so freezing COSZ
! to any positive constant would go unnoticed.

  subroutine eval_atm(x, ix, y)
    real,    intent(in)  :: x(:)
    integer, intent(in)  :: ix(:)
    real,    intent(out) :: y(:)
    type(noahmp_parameters) :: p
    real :: thair, qair, eair, rhoair, qprecc, qprecl
    real :: solad(2), solai(2)
    real :: swdown, bdfall, rain, snow, fp, fpice, prcp
    call atm(p, x(1), x(2), x(3), x(4), x(5), x(6), x(7), x(8), x(9), &
             x(10), x(11), thair, qair, eair, rhoair, qprecc, qprecl, &
             solad, solai, swdown, bdfall, rain, snow, fp, fpice, prcp)
    y(1)  = thair
    y(2)  = qair
    y(3)  = eair
    y(4)  = rhoair
    y(5)  = qprecc
    y(6)  = qprecl
    y(7)  = solad(1)
    y(8)  = solad(2)
    y(9)  = solai(1)
    y(10) = solai(2)
    y(11) = swdown
    y(12) = bdfall
    y(13) = rain
    y(14) = snow
    y(15) = fp
    y(16) = fpice
    y(17) = prcp
  end subroutine eval_atm

  subroutine dump_atm()
    integer, parameter :: NX = 11, NY = 17, NCASE = 8
    character(len=12) :: xn(NX), yn(NY)
    integer :: xi(NX), yi(NY)
    character(len=28) :: cn(NCASE)
    character(len=12) :: ixn(0)
    real :: xc(NX, NCASE)
    integer :: ixc(0, NCASE)
    logical :: ylive(NY, NCASE)

    xn = [character(len=12) :: 'sfcprs', 'sfctmp', 'q2', 'prcpconv', &
          'prcpnonc', 'prcpshcv', 'prcpsnow', 'prcpgrpl', 'prcphail', &
          'soldn', 'cosz']
    xi = 0
    yn = [character(len=12) :: 'thair', 'qair', 'eair', 'rhoair', 'qprecc', &
          'qprecl', 'solad', 'solad', 'solai', 'solai', 'swdown', 'bdfall', &
          'rain', 'snow', 'fp', 'fpice', 'prcp']
    yi = [0, 0, 0, 0, 0, 0, 1, 2, 1, 2, 0, 0, 0, 0, 0, 0, 0]

    cn = [character(len=28) :: 'warm_dry_noon', 'warm_rain_afternoon', &
          'mixed_phase_lower', 'mixed_phase_upper', 'cold_all_snow', &
          'near_freezing_snow', 'nocturnal_dark', 'hot_bright_convective']

    !            sfcprs  sfctmp     q2  prcpconv prcpnonc prcpshcv
    !            prcpsnow prcpgrpl prcphail  soldn   cosz
    xc(:, 1) = [ 96500.0, 298.15, 0.0102, 0.0,    0.0,     0.0,     &
                 0.0,     0.0,    0.0,    780.0,  0.87 ]
    xc(:, 2) = [ 95200.0, 291.00, 0.0121, 1.0e-3, 2.5e-3,  0.0,     &
                 0.0,     0.0,    0.0,    240.0,  0.41 ]
    ! TFRZ+0.84: the 1.0-(-54.632+0.2*SFCTMP) ramp at line 1190.
    xc(:, 3) = [ 97300.0, 274.00, 0.0038, 0.0,    1.4e-3,  0.0,     &
                 0.0,     0.0,    0.0,    120.0,  0.22 ]
    ! TFRZ+2.34: the FPICE=0.6 shelf at line 1192.
    xc(:, 4) = [ 97300.0, 275.50, 0.0044, 3.0e-4, 9.0e-4,  2.0e-4,  &
                 4.0e-4,  2.0e-4, 1.0e-4, 160.0,  0.31 ]
    xc(:, 5) = [ 88400.0, 261.30, 0.0011, 0.0,    8.0e-4,  0.0,     &
                 7.5e-4,  5.0e-5, 3.0e-5,  95.0,  0.18 ]
    ! TFRZ+0.34: the FPICE=1.0 branch at line 1188.
    xc(:, 6) = [ 99100.0, 273.50, 0.0041, 0.0,    3.1e-3,  0.0,     &
                 2.6e-3,  4.0e-4, 1.0e-4,  60.0,  0.12 ]
    xc(:, 7) = [ 96800.0, 284.70, 0.0075, 0.0,    6.0e-4,  0.0,     &
                 1.0e-4,  2.0e-5, 1.0e-5,  45.0, -0.34 ]
    xc(:, 8) = [101100.0, 305.90, 0.0176, 2.2e-3, 4.0e-4,  7.0e-4,  &
                 0.0,     0.0,    0.0,    940.0,  0.96 ]

    ylive = .true.
    call run_leaf('atm', eval_atm, cn, ixn, ixc, xn, xi, xc, yn, yi, ylive)
  end subroutine dump_atm

! --------------------------------------------------------------- leaf: esat --
!
! module_sf_noahmplsm.F:4952-5001.  Its argument is degrees Celsius, not
! Kelvin: every call site passes TDC(T) = MIN(50, MAX(-50, T-TFRZ)) from the
! statement functions at lines 3813 and 4332, so the fixture spans that exact
! clamped range.  No `parameters` argument at all.

  subroutine eval_esat(x, ix, y)
    real,    intent(in)  :: x(:)
    integer, intent(in)  :: ix(:)
    real,    intent(out) :: y(:)
    real :: esw, esi, desw, desi
    call esat(x(1), esw, esi, desw, desi)
    y(1) = esw
    y(2) = esi
    y(3) = desw
    y(4) = desi
  end subroutine eval_esat

  subroutine dump_esat()
    integer, parameter :: NX = 1, NY = 4, NCASE = 7
    character(len=12) :: xn(NX), yn(NY)
    integer :: xi(NX), yi(NY)
    character(len=28) :: cn(NCASE)
    character(len=12) :: ixn(0)
    real :: xc(NX, NCASE)
    integer :: ixc(0, NCASE)
    logical :: ylive(NY, NCASE)

    xn = [character(len=12) :: 'tc']
    xi = 0
    yn = [character(len=12) :: 'esw', 'esi', 'desw', 'desi']
    yi = 0
    cn = [character(len=28) :: 'clamp_low_minus50', 'deep_cold_minus28', &
          'frost_minus7', 'freezing_zero', 'cool_plus6', 'warm_plus27', &
          'clamp_high_plus50']
    xc(1, :) = [-50.0, -28.4, -7.15, 0.0, 6.35, 27.8, 50.0]
    ylive = .true.
    call run_leaf('esat', eval_esat, cn, ixn, ixc, xn, xi, xc, yn, yi, ylive)
  end subroutine dump_esat

! ------------------------------------------------------------- leaf: rosr12 --
!
! module_sf_noahmplsm.F:5534-5591.  Arrays are dimensioned -NSNOW+1:NSOIL and
! only NTOP=ISNOW+1 .. NSOIL participates.  P and DELTA are INTENT(INOUT) but
! ROSR12 assigns both from NTOP down, so the harness pre-fills them with 0.0
! and flags indices above NTOP live=0.  C is INTENT(INOUT): C(NSOIL) is
! overwritten with 0.0 at line 5565 before any read, so the C(NSOIL) *input*
! is inert; A(NTOP) is inert because A is read only for K > NTOP (line 5575).
!
! Two ISNOW=-3 cases are required, not decorative: with only one, A(-1), B(-2)
! and D(-2) have a single reader, and freezing them to that reader's own value
! is a no-op the fixture cannot detect.

  subroutine eval_rosr12(x, ix, y)
    real,    intent(in)  :: x(:)
    integer, intent(in)  :: ix(:)
    real,    intent(out) :: y(:)
    real :: a(-LNSNOW+1:LNSOIL), b(-LNSNOW+1:LNSOIL)
    real :: c(-LNSNOW+1:LNSOIL), d(-LNSNOW+1:LNSOIL)
    real :: pp(-LNSNOW+1:LNSOIL), delta(-LNSNOW+1:LNSOIL)
    integer :: k, s, ntop
    ntop = ix(1) + 1
    s = 0
    do k = -LNSNOW+1, LNSOIL
      s = s + 1
      a(k) = x(s)
    end do
    do k = -LNSNOW+1, LNSOIL
      s = s + 1
      b(k) = x(s)
    end do
    do k = -LNSNOW+1, LNSOIL
      s = s + 1
      c(k) = x(s)
    end do
    do k = -LNSNOW+1, LNSOIL
      s = s + 1
      d(k) = x(s)
    end do
    pp = 0.0
    delta = 0.0
    call rosr12(pp, a, b, c, d, delta, ntop, LNSOIL, LNSNOW)
    s = 0
    do k = -LNSNOW+1, LNSOIL
      s = s + 1
      y(s) = pp(k)
      y(LNLAY + s) = delta(k)
      y(2*LNLAY + s) = c(k)
    end do
  end subroutine eval_rosr12

  subroutine dump_rosr12()
    integer, parameter :: NX = 4 * LNLAY, NY = 3 * LNLAY, NCASE = 5
    character(len=12) :: xn(NX), yn(NY)
    integer :: xi(NX), yi(NY)
    character(len=28) :: cn(NCASE)
    character(len=12) :: ixn(1)
    real :: xc(NX, NCASE)
    integer :: ixc(1, NCASE)
    logical :: ylive(NY, NCASE)
    integer :: k, s, ic, ntop
    real :: base_a(LNLAY), base_b(LNLAY), base_c(LNLAY), base_d(LNLAY)
    real :: fa(NCASE), fb(NCASE), fc(NCASE), fd(NCASE)

    ixn = [character(len=12) :: 'isnow']
    s = 0
    do k = -LNSNOW+1, LNSOIL
      s = s + 1
      xn(s) = 'a';               xi(s) = k
      xn(LNLAY + s) = 'b';       xi(LNLAY + s) = k
      xn(2*LNLAY + s) = 'c';     xi(2*LNLAY + s) = k
      xn(3*LNLAY + s) = 'd';     xi(3*LNLAY + s) = k
      yn(s) = 'p';               yi(s) = k
      yn(LNLAY + s) = 'delta';   yi(LNLAY + s) = k
      yn(2*LNLAY + s) = 'c_out'; yi(2*LNLAY + s) = k
    end do

    ! A diagonally dominant HRT-shaped system: B on the diagonal, A below,
    ! C above, D the right-hand side.
    base_a = [-0.31, -0.47, -0.92, -2.14, -1.63, -0.88, -0.35]
    base_b = [ 1.62,  1.94,  2.71,  4.38,  3.55,  2.41,  1.77]
    base_c = [-0.44, -0.63, -1.15, -1.87, -1.42, -0.71, -0.29]
    base_d = [ 0.021, -0.048, 0.117, -0.263, 0.084, -0.037, 0.009]
    ! Per-case scalings, all distinct, so no slot is constant across cases.
    fa = [1.00, 1.07, 0.91, 1.23, 0.84]
    fb = [1.00, 0.94, 1.11, 0.87, 1.19]
    fc = [1.00, 1.13, 0.88, 1.06, 0.79]
    fd = [1.00, 0.83, 1.21, 0.92, 1.34]

    cn = [character(len=28) :: 'soil_only_isnow0', 'one_snow_layer', &
          'two_snow_layers', 'three_snow_layers', 'three_snow_layers_alt']
    ixc(1, :) = [0, -1, -2, -3, -3]

    do ic = 1, NCASE
      do s = 1, LNLAY
        xc(s, ic)           = base_a(s) * fa(ic)
        xc(LNLAY + s, ic)   = base_b(s) * fb(ic)
        xc(2*LNLAY + s, ic) = base_c(s) * fc(ic)
        xc(3*LNLAY + s, ic) = base_d(s) * fd(ic)
      end do
    end do

    ylive = .false.
    do ic = 1, NCASE
      ntop = ixc(1, ic) + 1
      s = 0
      do k = -LNSNOW+1, LNSOIL
        s = s + 1
        ylive(s, ic) = (k >= ntop)
        ylive(LNLAY + s, ic) = (k >= ntop)
        ylive(2*LNLAY + s, ic) = .true.
      end do
    end do

    call run_leaf('rosr12', eval_rosr12, cn, ixn, ixc, xn, xi, xc, &
                  yn, yi, ylive)
  end subroutine dump_rosr12

! -------------------------------------------------------------- leaf: csnow --
!
! module_sf_noahmplsm.F:2514-2569.  The body never references `parameters`.
! DZSNSO is dimensioned -NSNOW+1:NSOIL but CSNOW reads only ISNOW+1..0, so the
! harness poisons the soil half.  Every output is INTENT(OUT) and assigned
! only for ISNOW+1..0.  Two ISNOW=-3 cases give the bottom snow layer two
! distinct readers.

  subroutine eval_csnow(x, ix, y)
    real,    intent(in)  :: x(:)
    integer, intent(in)  :: ix(:)
    real,    intent(out) :: y(:)
    type(noahmp_parameters) :: p
    real :: snice(-LNSNOW+1:0), snliq(-LNSNOW+1:0)
    real :: dzsnso(-LNSNOW+1:LNSOIL)
    real :: tksno(-LNSNOW+1:0), cvsno(-LNSNOW+1:0)
    real :: snicev(-LNSNOW+1:0), snliqv(-LNSNOW+1:0), epore(-LNSNOW+1:0)
    integer :: k, s
    s = 0
    do k = -LNSNOW+1, 0
      s = s + 1
      snice(k) = x(s)
    end do
    do k = -LNSNOW+1, 0
      s = s + 1
      snliq(k) = x(s)
    end do
    dzsnso = POISON
    do k = -LNSNOW+1, 0
      s = s + 1
      dzsnso(k) = x(s)
    end do
    tksno = 0.0
    cvsno = 0.0
    snicev = 0.0
    snliqv = 0.0
    epore = 0.0
    call csnow(p, ix(1), LNSNOW, LNSOIL, snice, snliq, dzsnso, &
               tksno, cvsno, snicev, snliqv, epore)
    s = 0
    do k = -LNSNOW+1, 0
      s = s + 1
      y(s) = tksno(k)
      y(LNSNOW + s) = cvsno(k)
      y(2*LNSNOW + s) = snicev(k)
      y(3*LNSNOW + s) = snliqv(k)
      y(4*LNSNOW + s) = epore(k)
    end do
  end subroutine eval_csnow

  subroutine dump_csnow()
    integer, parameter :: NX = 3 * LNSNOW, NY = 5 * LNSNOW, NCASE = 6
    character(len=12) :: xn(NX), yn(NY)
    integer :: xi(NX), yi(NY)
    character(len=28) :: cn(NCASE)
    character(len=12) :: ixn(1)
    real :: xc(NX, NCASE)
    integer :: ixc(1, NCASE)
    logical :: ylive(NY, NCASE)
    integer :: k, s, ic, isnow

    ixn = [character(len=12) :: 'isnow']
    s = 0
    do k = -LNSNOW+1, 0
      s = s + 1
      xn(s) = 'snice';             xi(s) = k
      xn(LNSNOW + s) = 'snliq';    xi(LNSNOW + s) = k
      xn(2*LNSNOW + s) = 'dzsnso'; xi(2*LNSNOW + s) = k
      yn(s) = 'tksno';             yi(s) = k
      yn(LNSNOW + s) = 'cvsno';    yi(LNSNOW + s) = k
      yn(2*LNSNOW + s) = 'snicev'; yi(2*LNSNOW + s) = k
      yn(3*LNSNOW + s) = 'snliqv'; yi(3*LNSNOW + s) = k
      yn(4*LNSNOW + s) = 'epore';  yi(4*LNSNOW + s) = k
    end do

    cn = [character(len=28) :: 'snow_free_isnow0', 'single_thin_layer', &
          'two_layer_ripe', 'three_layer_cold', 'ice_saturated_clamp', &
          'three_layer_dense']
    ixc(1, :) = [0, -1, -2, -3, -2, -3]

    ! Slots: 1-3 snice(-2:0), 4-6 snliq(-2:0), 7-9 dzsnso(-2:0).
    xc = 0.0
    ! Case 2: one layer.
    xc(3, 2) = 12.4
    xc(6, 2) = 1.9
    xc(9, 2) = 0.048
    ! Case 3: two layers, ripe (liquid near the effective-porosity clamp).
    xc(2, 3) = 41.0
    xc(3, 3) = 26.5
    xc(5, 3) = 9.4
    xc(6, 3) = 14.8
    xc(8, 3) = 0.135
    xc(9, 3) = 0.071
    ! Case 4: three layers, cold and nearly dry.
    xc(1, 4) = 88.3
    xc(2, 4) = 52.7
    xc(3, 4) = 21.6
    xc(4, 4) = 0.90
    xc(5, 4) = 0.35
    xc(6, 4) = 1.05
    xc(7, 4) = 0.244
    xc(8, 4) = 0.148
    xc(9, 4) = 0.062
    ! Case 5: SNICEV hits the MIN(1.0, ...) clamp at line 2548, driving EPORE
    ! to 0 and SNLIQV to the MIN(EPORE, ...) clamp at line 2550.
    xc(2, 5) = 210.0
    xc(3, 5) = 96.0
    xc(5, 5) = 33.0
    xc(6, 5) = 12.0
    xc(8, 5) = 0.180
    xc(9, 5) = 0.051
    ! Case 6: three layers again, denser, so the bottom layer has a second
    ! reader with a different value.
    xc(1, 6) = 134.7
    xc(2, 6) = 71.2
    xc(3, 6) = 29.8
    xc(4, 6) = 2.40
    xc(5, 6) = 1.60
    xc(6, 6) = 3.10
    xc(7, 6) = 0.196
    xc(8, 6) = 0.121
    xc(9, 6) = 0.055

    ylive = .false.
    do ic = 1, NCASE
      isnow = ixc(1, ic)
      s = 0
      do k = -LNSNOW+1, 0
        s = s + 1
        if (k >= isnow + 1) then
          ylive(s, ic) = .true.
          ylive(LNSNOW + s, ic) = .true.
          ylive(2*LNSNOW + s, ic) = .true.
          ylive(3*LNSNOW + s, ic) = .true.
          ylive(4*LNSNOW + s, ic) = .true.
        end if
      end do
    end do

    call run_leaf('csnow', eval_csnow, cn, ixn, ixc, xn, xi, xc, &
                  yn, yi, ylive)
  end subroutine dump_csnow

! ------------------------------------------------------------- leaf: tdfcnd --
!
! module_sf_noahmplsm.F:2573-2680.  Reads parameters%SMCMAX(ISOIL) and
! parameters%QUARTZ(ISOIL) only, so the fixture carries those two as ordinary
! inputs and the harness poisons every other layer of both arrays.  ISOIL is
! fixed at 1 because the routine is scalar in the layer.

  subroutine eval_tdfcnd(x, ix, y)
    real,    intent(in)  :: x(:)
    integer, intent(in)  :: ix(:)
    real,    intent(out) :: y(:)
    type(noahmp_parameters) :: p
    real :: df
    p%SMCMAX = POISON
    p%QUARTZ = POISON
    p%SMCMAX(1) = x(3)
    p%QUARTZ(1) = x(4)
    df = 0.0
    call tdfcnd(p, 1, df, x(1), x(2))
    y(1) = df
  end subroutine eval_tdfcnd

  subroutine dump_tdfcnd()
    integer, parameter :: NX = 4, NY = 1, NCASE = 7
    character(len=12) :: xn(NX), yn(NY)
    integer :: xi(NX), yi(NY)
    character(len=28) :: cn(NCASE)
    character(len=12) :: ixn(0)
    real :: xc(NX, NCASE)
    integer :: ixc(0, NCASE)
    logical :: ylive(NY, NCASE)

    xn = [character(len=12) :: 'smc', 'sh2o', 'smcmax', 'quartz']
    xi = [0, 0, 1, 1]
    yn = [character(len=12) :: 'df']
    yi = 0
    cn = [character(len=28) :: 'wet_unfrozen_loam', 'dry_below_kersten', &
          'partly_frozen', 'bone_dry_zero_smc', 'saturated_loam', &
          'quartz_rich_sand', 'clay_wilting']

    !                 smc     sh2o   smcmax  quartz
    xc(:, 1) = [ 0.3020, 0.3020, 0.4340, 0.2500 ]
    xc(:, 2) = [ 0.0300, 0.0290, 0.4120, 0.3200 ]
    xc(:, 3) = [ 0.2810, 0.1010, 0.4470, 0.1800 ]
    xc(:, 4) = [ 0.0000, 0.0000, 0.4210, 0.3500 ]
    xc(:, 5) = [ 0.4340, 0.4340, 0.4340, 0.2600 ]
    xc(:, 6) = [ 0.2110, 0.2110, 0.3390, 0.9200 ]
    xc(:, 7) = [ 0.1380, 0.0620, 0.4680, 0.1000 ]

    ylive = .true.
    call run_leaf('tdfcnd', eval_tdfcnd, cn, ixn, ixc, xn, xi, xc, &
                  yn, yi, ylive)
  end subroutine dump_tdfcnd

! --------------------------------------------------------- leaf: thermoprop --
!
! module_sf_noahmplsm.F:2400-2510.  Composes CSNOW and TDFCND.  TG, UR, LAT,
! Z0M, ZLVL and VEGTYP are declared INTENT(IN) but the body never references
! them; they vary across cases so their inertness is measured, not assumed.
! STC is read only as STC(IZ), IZ=1..NSOIL, inside the IST==2 branch
! (2483-2493), so the snow half of STC is inert too.
!
! `lake_thawed_column` puts every soil STC above TFRZ and `lake_frozen_column`
! puts every soil STC below it.  That pair is what makes STC(1:4)
! mutation-detectable: any constant a port might substitute is on one side of
! TFRZ or the other, so it must flip one of those two cases.
!
! Input slots: 1-7 dzsnso(-2:4), 8-10 snice(-2:0), 11-13 snliq(-2:0),
! 14-17 smc(1:4), 18-21 sh2o(1:4), 22-28 stc(-2:4), 29 snowh, 30 dt, 31 tg,
! 32 ur, 33 lat, 34 z0m, 35 zlvl, 36-39 smcmax(1:4), 40 csoil,
! 41-44 quartz(1:4).
! Output slots: 1-7 df(-2:4), 8-14 hcpct(-2:4), 15-17 snicev(-2:0),
! 18-20 snliqv(-2:0), 21-23 epore(-2:0), 24-30 fact(-2:4).

  subroutine eval_thermoprop(x, ix, y)
    real,    intent(in)  :: x(:)
    integer, intent(in)  :: ix(:)
    real,    intent(out) :: y(:)
    type(noahmp_parameters) :: p
    real :: dzsnso(-LNSNOW+1:LNSOIL), stc(-LNSNOW+1:LNSOIL)
    real :: snice(-LNSNOW+1:0), snliq(-LNSNOW+1:0)
    real :: smc(LNSOIL), sh2o(LNSOIL)
    real :: df(-LNSNOW+1:LNSOIL), hcpct(-LNSNOW+1:LNSOIL)
    real :: fact(-LNSNOW+1:LNSOIL)
    real :: snicev(-LNSNOW+1:0), snliqv(-LNSNOW+1:0), epore(-LNSNOW+1:0)
    real :: snowh, dt, tg, ur, lat, z0m, zlvl
    integer :: k, s, isnow, ist, vegtyp

    isnow = ix(1)
    ist = ix(2)
    vegtyp = ix(3)
    p%urban_flag = (ix(4) /= 0)

    s = 0
    do k = -LNSNOW+1, LNSOIL
      s = s + 1
      dzsnso(k) = x(s)
    end do
    do k = -LNSNOW+1, 0
      s = s + 1
      snice(k) = x(s)
    end do
    do k = -LNSNOW+1, 0
      s = s + 1
      snliq(k) = x(s)
    end do
    do k = 1, LNSOIL
      s = s + 1
      smc(k) = x(s)
    end do
    do k = 1, LNSOIL
      s = s + 1
      sh2o(k) = x(s)
    end do
    do k = -LNSNOW+1, LNSOIL
      s = s + 1
      stc(k) = x(s)
    end do
    s = s + 1; snowh = x(s)
    s = s + 1; dt    = x(s)
    s = s + 1; tg    = x(s)
    s = s + 1; ur    = x(s)
    s = s + 1; lat   = x(s)
    s = s + 1; z0m   = x(s)
    s = s + 1; zlvl  = x(s)
    do k = 1, LNSOIL
      s = s + 1
      p%SMCMAX(k) = x(s)
    end do
    s = s + 1; p%CSOIL = x(s)
    do k = 1, LNSOIL
      s = s + 1
      p%QUARTZ(k) = x(s)
    end do

    df = 0.0
    hcpct = 0.0
    fact = 0.0
    snicev = 0.0
    snliqv = 0.0
    epore = 0.0

    call thermoprop(p, LNSOIL, LNSNOW, isnow, ist, dzsnso, &
                    dt, snowh, snice, snliq, &
                    smc, sh2o, tg, stc, ur, &
                    lat, z0m, zlvl, vegtyp, &
                    df, hcpct, snicev, snliqv, epore, &
                    fact)

    s = 0
    do k = -LNSNOW+1, LNSOIL
      s = s + 1
      y(s) = df(k)
      y(LNLAY + s) = hcpct(k)
      y(2*LNLAY + 3*LNSNOW + s) = fact(k)
    end do
    s = 2 * LNLAY
    do k = -LNSNOW+1, 0
      s = s + 1
      y(s) = snicev(k)
      y(LNSNOW + s) = snliqv(k)
      y(2*LNSNOW + s) = epore(k)
    end do
  end subroutine eval_thermoprop

  subroutine dump_thermoprop()
    integer, parameter :: NX = 44, NY = 30, NCASE = 10
    character(len=12) :: xn(NX), yn(NY)
    integer :: xi(NX), yi(NY)
    character(len=28) :: cn(NCASE)
    character(len=12) :: ixn(4)
    real :: xc(NX, NCASE)
    integer :: ixc(4, NCASE)
    logical :: ylive(NY, NCASE)
    integer :: k, s, ic, isnow

    real :: dzn(LNSNOW, NCASE), dzs(LNSOIL, NCASE)
    real :: sni(LNSNOW, NCASE), snl(LNSNOW, NCASE)
    real :: smcv(LNSOIL, NCASE), sh2ov(LNSOIL, NCASE)
    real :: stcv(LNLAY, NCASE)
    real :: smcmaxv(LNSOIL, NCASE), quartzv(LNSOIL, NCASE)
    real :: snowhv(NCASE), dtv(NCASE), tgv(NCASE), urv(NCASE)
    real :: latv(NCASE), z0mv(NCASE), zlvlv(NCASE), csoilv(NCASE)

    ixn = [character(len=12) :: 'isnow', 'ist', 'vegtyp', 'urban_flag']

    s = 0
    do k = -LNSNOW+1, LNSOIL
      s = s + 1; xn(s) = 'dzsnso'; xi(s) = k
    end do
    do k = -LNSNOW+1, 0
      s = s + 1; xn(s) = 'snice'; xi(s) = k
    end do
    do k = -LNSNOW+1, 0
      s = s + 1; xn(s) = 'snliq'; xi(s) = k
    end do
    do k = 1, LNSOIL
      s = s + 1; xn(s) = 'smc'; xi(s) = k
    end do
    do k = 1, LNSOIL
      s = s + 1; xn(s) = 'sh2o'; xi(s) = k
    end do
    do k = -LNSNOW+1, LNSOIL
      s = s + 1; xn(s) = 'stc'; xi(s) = k
    end do
    s = s + 1; xn(s) = 'snowh'; xi(s) = 0
    s = s + 1; xn(s) = 'dt';    xi(s) = 0
    s = s + 1; xn(s) = 'tg';    xi(s) = 0
    s = s + 1; xn(s) = 'ur';    xi(s) = 0
    s = s + 1; xn(s) = 'lat';   xi(s) = 0
    s = s + 1; xn(s) = 'z0m';   xi(s) = 0
    s = s + 1; xn(s) = 'zlvl';  xi(s) = 0
    do k = 1, LNSOIL
      s = s + 1; xn(s) = 'smcmax'; xi(s) = k
    end do
    s = s + 1; xn(s) = 'csoil'; xi(s) = 0
    do k = 1, LNSOIL
      s = s + 1; xn(s) = 'quartz'; xi(s) = k
    end do

    s = 0
    do k = -LNSNOW+1, LNSOIL
      s = s + 1
      yn(s) = 'df';      yi(s) = k
      yn(LNLAY + s) = 'hcpct'; yi(LNLAY + s) = k
      yn(2*LNLAY + 3*LNSNOW + s) = 'fact'
      yi(2*LNLAY + 3*LNSNOW + s) = k
    end do
    s = 2 * LNLAY
    do k = -LNSNOW+1, 0
      s = s + 1
      yn(s) = 'snicev';            yi(s) = k
      yn(LNSNOW + s) = 'snliqv';   yi(LNSNOW + s) = k
      yn(2*LNSNOW + s) = 'epore';  yi(2*LNSNOW + s) = k
    end do

    cn = [character(len=28) :: 'bare_soil_wet', 'bare_soil_frozen', &
          'one_snow_layer', 'two_snow_layers', 'three_snow_layers', &
          'three_snow_layers_dense', 'urban_override', &
          'lake_thawed_column', 'lake_frozen_column', 'lake_mixed_column']
    !            isnow, ist, vegtyp, urban_flag
    ixc(:, 1)  = [  0, 1, 10, 0 ]
    ixc(:, 2)  = [  0, 1,  1, 0 ]
    ixc(:, 3)  = [ -1, 1, 12, 0 ]
    ixc(:, 4)  = [ -2, 1,  7, 0 ]
    ixc(:, 5)  = [ -3, 1,  1, 0 ]
    ixc(:, 6)  = [ -3, 1, 15, 0 ]
    ixc(:, 7)  = [  0, 1, 13, 1 ]
    ixc(:, 8)  = [ -2, 2, 17, 0 ]
    ixc(:, 9)  = [ -1, 2, 17, 0 ]
    ixc(:, 10) = [  0, 2, 16, 0 ]

    ! Snow-layer thickness, layers -2:0.
    dzn = reshape([ &
        0.000, 0.000, 0.000,   0.000, 0.000, 0.000, &
        0.000, 0.000, 0.062,   0.000, 0.126, 0.071, &
        0.221, 0.134, 0.058,   0.196, 0.121, 0.055, &
        0.000, 0.000, 0.000,   0.000, 0.148, 0.069, &
        0.000, 0.000, 0.084,   0.000, 0.000, 0.000  ], [LNSNOW, NCASE])
    ! Soil-layer thickness, layers 1:4.  Distinct in every case.
    dzs = reshape([ &
        0.10, 0.30, 0.60, 1.00,   0.09, 0.28, 0.58, 0.95, &
        0.11, 0.32, 0.63, 1.05,   0.12, 0.29, 0.61, 0.98, &
        0.08, 0.26, 0.55, 0.90,   0.13, 0.31, 0.59, 1.02, &
        0.14, 0.34, 0.66, 1.10,   0.10, 0.27, 0.62, 1.01, &
        0.15, 0.33, 0.57, 0.93,   0.07, 0.25, 0.64, 1.07  ], [LNSOIL, NCASE])
    sni = reshape([ &
          0.0,   0.0,   0.0,     0.0,   0.0,   0.0, &
          0.0,   0.0,  14.8,     0.0,  41.0,  17.5, &
         79.4,  46.1,  18.7,   134.7,  71.2,  29.8, &
          0.0,   0.0,   0.0,     0.0,  52.3,  21.4, &
          0.0,   0.0,  22.6,     0.0,   0.0,   0.0  ], [LNSNOW, NCASE])
    snl = reshape([ &
          0.0,   0.0,   0.0,     0.0,   0.0,   0.0, &
          0.0,   0.0,   2.1,     0.0,   1.1,   3.4, &
          0.7,   0.4,   1.3,     2.4,   1.6,   3.1, &
          0.0,   0.0,   0.0,     0.0,   1.8,   2.9, &
          0.0,   0.0,   2.7,     0.0,   0.0,   0.0  ], [LNSNOW, NCASE])
    smcv = reshape([ &
        0.302, 0.288, 0.271, 0.263,   0.281, 0.294, 0.312, 0.298, &
        0.244, 0.259, 0.276, 0.288,   0.318, 0.305, 0.291, 0.277, &
        0.105, 0.168, 0.221, 0.254,   0.132, 0.191, 0.236, 0.268, &
        0.196, 0.212, 0.238, 0.251,   0.331, 0.318, 0.305, 0.294, &
        0.267, 0.281, 0.293, 0.309,   0.223, 0.247, 0.262, 0.279 ], &
        [LNSOIL, NCASE])
    sh2ov = reshape([ &
        0.302, 0.288, 0.271, 0.263,   0.091, 0.102, 0.155, 0.201, &
        0.244, 0.259, 0.276, 0.288,   0.318, 0.305, 0.291, 0.277, &
        0.048, 0.097, 0.183, 0.239,   0.061, 0.114, 0.198, 0.244, &
        0.196, 0.212, 0.238, 0.251,   0.331, 0.318, 0.305, 0.294, &
        0.187, 0.243, 0.293, 0.309,   0.223, 0.247, 0.262, 0.279 ], &
        [LNSOIL, NCASE])
    ! STC: cases 8 and 9 straddle TFRZ = 273.16 as whole columns; case 10 is
    ! mixed within the column.
    stcv = reshape([ &
        285.00, 285.40, 285.90, 286.20, 287.10, 288.00, 288.60, &
        272.10, 272.40, 272.80, 271.90, 272.60, 273.40, 274.10, &
        268.00, 269.50, 270.90, 271.80, 272.90, 274.00, 275.20, &
        270.20, 271.40, 272.60, 273.90, 274.70, 275.10, 275.80, &
        258.30, 261.70, 265.40, 269.80, 271.20, 272.80, 274.60, &
        254.10, 259.20, 263.80, 268.50, 270.40, 272.10, 273.90, &
        290.40, 291.10, 292.00, 293.20, 294.00, 294.80, 295.50, &
        271.10, 272.40, 273.00, 274.80, 275.20, 276.10, 275.60, &
        270.00, 271.00, 272.50, 272.40, 271.80, 271.10, 270.30, &
        269.40, 270.70, 271.90, 274.50, 273.05, 272.10, 271.30 ], &
        [LNLAY, NCASE])
    smcmaxv = reshape([ &
        0.434, 0.434, 0.434, 0.434,   0.404, 0.412, 0.421, 0.430, &
        0.464, 0.455, 0.447, 0.439,   0.421, 0.418, 0.415, 0.412, &
        0.339, 0.347, 0.356, 0.364,   0.352, 0.361, 0.370, 0.379, &
        0.446, 0.441, 0.437, 0.433,   0.476, 0.468, 0.459, 0.451, &
        0.398, 0.406, 0.414, 0.422,   0.412, 0.424, 0.436, 0.448 ], &
        [LNSOIL, NCASE])
    quartzv = reshape([ &
        0.25, 0.24, 0.23, 0.22,   0.32, 0.30, 0.28, 0.26, &
        0.18, 0.21, 0.24, 0.27,   0.35, 0.37, 0.39, 0.41, &
        0.92, 0.85, 0.78, 0.71,   0.60, 0.57, 0.54, 0.51, &
        0.48, 0.46, 0.44, 0.42,   0.10, 0.13, 0.16, 0.19, &
        0.44, 0.47, 0.50, 0.53,   0.20, 0.29, 0.33, 0.31 ], &
        [LNSOIL, NCASE])
    !          1      2      3      4      5      6      7      8      9     10
    snowhv = [0.0000, 0.0135, 0.062, 0.197, 0.413, 0.372, 0.0208, 0.217, &
              0.084, 0.031]
    dtv    = [60.0, 45.0, 30.0, 120.0, 90.0, 75.0, 20.0, 110.0, 50.0, 100.0]
    tgv    = [288.4, 271.2, 268.9, 274.6, 259.4, 256.1, 295.1, 275.3, &
              272.8, 270.5]
    urv    = [3.7, 5.1, 2.3, 4.4, 6.8, 7.2, 1.9, 3.1, 2.8, 5.6]
    latv   = [0.6981, 0.7854, 0.5236, 0.8727, 0.9599, 1.0123, 0.4363, &
              0.6109, 0.7330, 0.5672]
    z0mv   = [0.15, 0.08, 0.22, 0.02, 0.05, 0.03, 0.35, 0.04, 0.11, 0.28]
    zlvlv  = [20.0, 25.0, 18.0, 22.0, 30.0, 28.0, 12.0, 24.0, 16.0, 26.0]
    csoilv = [2.00e6, 1.86e6, 2.13e6, 2.41e6, 1.72e6, 1.94e6, 3.00e6, &
              2.27e6, 1.95e6, 2.08e6]

    do ic = 1, NCASE
      do k = 1, LNSNOW
        xc(k, ic) = dzn(k, ic)
      end do
      do k = 1, LNSOIL
        xc(LNSNOW + k, ic) = dzs(k, ic)
      end do
      do k = 1, LNSNOW
        xc(LNLAY + k, ic) = sni(k, ic)
        xc(LNLAY + LNSNOW + k, ic) = snl(k, ic)
      end do
      do k = 1, LNSOIL
        xc(LNLAY + 2*LNSNOW + k, ic) = smcv(k, ic)
        xc(LNLAY + 2*LNSNOW + LNSOIL + k, ic) = sh2ov(k, ic)
      end do
      do k = 1, LNLAY
        xc(LNLAY + 2*LNSNOW + 2*LNSOIL + k, ic) = stcv(k, ic)
      end do
      xc(29, ic) = snowhv(ic)
      xc(30, ic) = dtv(ic)
      xc(31, ic) = tgv(ic)
      xc(32, ic) = urv(ic)
      xc(33, ic) = latv(ic)
      xc(34, ic) = z0mv(ic)
      xc(35, ic) = zlvlv(ic)
      do k = 1, LNSOIL
        xc(35 + k, ic) = smcmaxv(k, ic)
      end do
      xc(40, ic) = csoilv(ic)
      do k = 1, LNSOIL
        xc(40 + k, ic) = quartzv(k, ic)
      end do
    end do

    ylive = .false.
    do ic = 1, NCASE
      isnow = ixc(1, ic)
      s = 0
      do k = -LNSNOW+1, LNSOIL
        s = s + 1
        ! DF and HCPCT are zeroed at 2446-2447 then filled, so every entry is
        ! defined.  FACT is assigned only for ISNOW+1..NSOIL.
        ylive(s, ic) = .true.
        ylive(LNLAY + s, ic) = .true.
        ylive(2*LNLAY + 3*LNSNOW + s, ic) = (k >= isnow + 1)
      end do
      s = 2 * LNLAY
      do k = -LNSNOW+1, 0
        s = s + 1
        ylive(s, ic) = (k >= isnow + 1)
        ylive(LNSNOW + s, ic) = (k >= isnow + 1)
        ylive(2*LNSNOW + s, ic) = (k >= isnow + 1)
      end do
    end do

    call run_leaf('thermoprop', eval_thermoprop, cn, ixn, ixc, xn, xi, xc, &
                  yn, yi, ylive)
  end subroutine dump_thermoprop

! ---------------------------------------------------------- leaf: wdfcnd1 --
!
! module_sf_noahmplsm.F:9153-9188.  Live under the pinned OPT_INF=1 through
! SRT line 7773.  Local VKWGT is declared and never used.

  subroutine eval_wdfcnd1(x, ix, y)
    real,    intent(in)  :: x(:)
    integer, intent(in)  :: ix(:)
    real,    intent(out) :: y(:)
    type(noahmp_parameters) :: p
    real :: wdf, wcnd
    p%SMCMAX = POISON
    p%BEXP = POISON
    p%DWSAT = POISON
    p%DKSAT = POISON
    p%SMCMAX(1) = x(3)
    p%BEXP(1) = x(4)
    p%DWSAT(1) = x(5)
    p%DKSAT(1) = x(6)
    wdf = 0.0
    wcnd = 0.0
    call wdfcnd1(p, wdf, wcnd, x(1), x(2), 1)
    y(1) = wdf
    y(2) = wcnd
  end subroutine eval_wdfcnd1

  subroutine dump_wdfcnd1()
    integer, parameter :: NX = 6, NY = 2, NCASE = 6
    character(len=12) :: xn(NX), yn(NY)
    integer :: xi(NX), yi(NY)
    character(len=28) :: cn(NCASE)
    character(len=12) :: ixn(0)
    real :: xc(NX, NCASE)
    integer :: ixc(0, NCASE)
    logical :: ylive(NY, NCASE)

    xn = [character(len=12) :: 'smc', 'fcr', 'smcmax', 'bexp', 'dwsat', &
          'dksat']
    xi = [0, 0, 1, 1, 1, 1]
    yn = [character(len=12) :: 'wdf', 'wcnd']
    yi = 0
    cn = [character(len=28) :: 'loam_unfrozen', 'loam_partly_impermeable', &
          'fully_impermeable', 'dry_factr_clamp', 'saturated', &
          'sand_unfrozen']

    !                 smc     fcr   smcmax  bexp    dwsat      dksat
    xc(:, 1) = [ 0.2810, 0.0000, 0.4340, 5.2500, 5.1420e-6, 3.4700e-6 ]
    xc(:, 2) = [ 0.2640, 0.3400, 0.4120, 4.7400, 4.8700e-6, 2.9100e-6 ]
    xc(:, 3) = [ 0.2930, 1.0000, 0.4470, 5.6100, 5.5300e-6, 3.8200e-6 ]
    xc(:, 4) = [ 0.0011, 0.0700, 0.4210, 4.2600, 4.1500e-6, 2.4600e-6 ]
    xc(:, 5) = [ 0.4340, 0.0000, 0.4340, 5.3300, 5.2100e-6, 3.5500e-6 ]
    xc(:, 6) = [ 0.1170, 0.1900, 0.3390, 2.7900, 0.6080e-5, 1.7600e-5 ]

    ylive = .true.
    call run_leaf('wdfcnd1', eval_wdfcnd1, cn, ixn, ixc, xn, xi, xc, &
                  yn, yi, ylive)
  end subroutine dump_wdfcnd1

! ---------------------------------------------------------- leaf: wdfcnd2 --
!
! module_sf_noahmplsm.F:9192-9232.  Live under the pinned OPT_INF=1 through
! INFIL line 7703: that call is NOT inside an OPT_INF branch, its only gate is
! `IF (QINSUR > 0.0)` at 7655.  Note (500*SICE)**3.0 at line 9223 -- a real
! constant exponent, which gfortran evaluates with powf, so the port must
! round once rather than three times.

  subroutine eval_wdfcnd2(x, ix, y)
    real,    intent(in)  :: x(:)
    integer, intent(in)  :: ix(:)
    real,    intent(out) :: y(:)
    type(noahmp_parameters) :: p
    real :: wdf, wcnd
    p%SMCMAX = POISON
    p%BEXP = POISON
    p%DWSAT = POISON
    p%DKSAT = POISON
    p%SMCMAX(1) = x(3)
    p%BEXP(1) = x(4)
    p%DWSAT(1) = x(5)
    p%DKSAT(1) = x(6)
    wdf = 0.0
    wcnd = 0.0
    call wdfcnd2(p, wdf, wcnd, x(1), x(2), 1)
    y(1) = wdf
    y(2) = wcnd
  end subroutine eval_wdfcnd2

  subroutine dump_wdfcnd2()
    integer, parameter :: NX = 6, NY = 2, NCASE = 6
    character(len=12) :: xn(NX), yn(NY)
    integer :: xi(NX), yi(NY)
    character(len=28) :: cn(NCASE)
    character(len=12) :: ixn(0)
    real :: xc(NX, NCASE)
    integer :: ixc(0, NCASE)
    logical :: ylive(NY, NCASE)

    xn = [character(len=12) :: 'smc', 'sice', 'smcmax', 'bexp', 'dwsat', &
          'dksat']
    xi = [0, 0, 1, 1, 1, 1]
    yn = [character(len=12) :: 'wdf', 'wcnd']
    yi = 0
    cn = [character(len=28) :: 'ice_free', 'trace_ice_vkwgt_high', &
          'heavy_ice_vkwgt_low', 'dry_factr_clamp', 'saturated_ice_free', &
          'sand_moderate_ice']

    !                 smc    sice   smcmax  bexp    dwsat      dksat
    xc(:, 1) = [ 0.2810, 0.0000, 0.4340, 5.2500, 5.1420e-6, 3.4700e-6 ]
    xc(:, 2) = [ 0.2640, 0.0021, 0.4120, 4.7400, 4.8700e-6, 2.9100e-6 ]
    xc(:, 3) = [ 0.2930, 0.1900, 0.4470, 5.6100, 5.5300e-6, 3.8200e-6 ]
    xc(:, 4) = [ 0.0011, 0.0400, 0.4210, 4.2600, 4.1500e-6, 2.4600e-6 ]
    xc(:, 5) = [ 0.4340, 0.0000, 0.4340, 5.3300, 5.2100e-6, 3.5500e-6 ]
    xc(:, 6) = [ 0.1170, 0.0330, 0.3390, 2.7900, 0.6080e-5, 1.7600e-5 ]

    ylive = .true.
    call run_leaf('wdfcnd2', eval_wdfcnd2, cn, ixn, ixc, xn, xi, xc, &
                  yn, yi, ylive)
  end subroutine dump_wdfcnd2

! -------------------------------------------------------------- entry point --

  subroutine dump_all()
    call dump_atm()
    call dump_esat()
    call dump_rosr12()
    call dump_csnow()
    call dump_tdfcnd()
    call dump_thermoprop()
    call dump_wdfcnd1()
    call dump_wdfcnd2()
  end subroutine dump_all

end module noahmp_leaf_oracle


program run_noahmp_leaves_oracle
  use module_sf_noahmplsm, only: noahmp_options
  use noahmp_leaf_oracle, only: open_outputs, close_outputs, dump_all
  implicit none
  character(len=1024) :: leaf_path, disc_path

  call get_command_argument(1, leaf_path)
  call get_command_argument(2, disc_path)
  if (len_trim(leaf_path) == 0 .or. len_trim(disc_path) == 0) then
    write(*, '(A)') 'usage: run_leaves LEAVES.csv DISCRIMINATION.csv'
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
end program run_noahmp_leaves_oracle
