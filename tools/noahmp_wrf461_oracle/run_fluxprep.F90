! Per-leaf oracle for the VEGE_FLUX / BARE_FLUX *flux-preparation* leaves of
! WRF v4.6.1 MODULE_SF_NOAHMPLSM: SFCDIF1, RAGRB and STOMATA.
!
! This program links the *visibility-patched* module (see
! patches/noahmp-lsm-leaf-visibility.patch, audited by
! check_visibility_patch.py before this file is compiled).  Apart from the 50
! `private ::` -> `public ::` accessibility statements, the physics source is
! the pinned WRF v4.6.1 file byte for byte.
!
! It is a sibling of run_leaves.F90 and shares its CSV schema and its generic
! `run_leaf` driver verbatim; the two are separate files only so that parallel
! lanes never edit the same source.  The output CSVs are
! noahmp-fluxprep.csv / noahmp-fluxprep-discrimination.csv.
!
! Option identity: the pinned WRF Registry default
!   dveg=4 opt_crs=1 opt_btr=1 opt_run=3 opt_sfc=1 opt_frz=1 opt_inf=1
!   opt_rad=3 opt_alb=2 opt_snf=1 opt_tbot=2 opt_stc=1 opt_rsf=1 opt_soil=1
!   opt_pedo=1 opt_crop=0 opt_irr=0 opt_irrm=0 opt_infdv=0 opt_tdrn=0
!
! What that identity kills in this subsystem, and is therefore NOT ported:
!   * SFCDIF2 (:4747-4948) -- reached only from `IF(OPT_SFC == 2)` at
!     VEGE_FLUX:3897 and BARE_FLUX:4367.  OPT_SFC is 1.
!   * CANRES (:5141-5220) and CALHUM (:5224-5262) -- reached only from
!     `IF (OPT_CRS == 2)` at VEGE_FLUX:3949.  OPT_CRS is 1, so STOMATA is the
!     live canopy-conductance leaf.
!   * the Gecros crop chain at VEGE_FLUX:3958 -- `IF (opt_crop == 2)`.
! None of the three leaves below reads an option variable itself.
!
! Iteration counts the pinned identity produces, i.e. the ITER range each leaf
! actually sees:
!   * SFCDIF1 and RAGRB are called inside `loop1: DO ITER = 1, NITERC` in
!     VEGE_FLUX (:3877) with `INTEGER, PARAMETER :: NITERC = 20` (:3792), and
!     SFCDIF1 again inside `loop3: DO ITER = 1, NITERB` in BARE_FLUX (:4351)
!     with `DATA NITERB /5/` (:4329).  So ITER runs 1..20; the fixture spans
!     ITER = 1..5, which is the whole *behavioural* range because the body
!     tests only `ITER == 1` and `ITER > 1`.
!   * STOMATA is called only under `IF(ITER == 1)` (VEGE_FLUX:3934), so it
!     never sees ITER at all -- it has no ITER argument.  Its own internal
!     loop is `DO ITER = 1, NITER` with `DATA NITER /3/` (:5044-5046), a
!     compile-time constant: exactly three Ball-Berry sweeps, never more.
!
! FV carries values that separate the two lowerings of FV**3 in four cases
! (RAGRB 0.4196 and 0.5497, SFCDIF1 0.3499 and 0.1999).  gfortran emits
! libgcc's __powisf2, fl(x * fl(x*x)), not powf(x, 3.0); the two disagree on
! about 26% of FP32 friction velocities, and without those values every case
! either clamped MOZ/MOZG afterwards or happened to land where the two agree,
! so a port that used powf was indistinguishable.  The blind spot was found by
! a negative control on the CUDA half, not by inspection.
!
! Fixture philosophy, stated once because it decides many values below: a leaf
! fixture pins a *function*, not a call site.  Where WRF's pinned MPTABLE.TBL
! ships one value of a parameter for all 20 MODIS classes (AKC=2.1, AKO=1.2,
! AVCMX=2.4, KC25=30, KO25=30000, DLEAF=0.04, C3PSN=1.0) a fixture that copied
! the table would leave that argument frozen, and the mutation study in
! gpuwm/core/noahmp_leaf_mutation.py would find that no port could be caught
! hard-coding it.  Those arguments therefore vary across cases.  Each such
! choice is called out in the per-leaf header.

module noahmp_fluxprep_oracle

  use module_sf_noahmplsm, only: noahmp_parameters, &
      sfcdif1, ragrb, stomata
  implicit none
  private
  public :: open_outputs, close_outputs, dump_all

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
! Identical in behaviour to run_leaves.F90's plumbing, so both fixtures share
! one CSV schema and one validator shape.

  subroutine open_outputs(leaf_path, disc_path)
    character(len=*), intent(in) :: leaf_path, disc_path
    open(newunit=unit_leaf, file=trim(leaf_path), status='replace', &
         action='write')
    write(unit_leaf, '(A)') 'leaf,case,role,name,index,slot,live,bits,value'
    open(newunit=unit_disc, file=trim(disc_path), status='replace', &
         action='write')
    write(unit_disc, '(A)') 'leaf,case,slot,name,index,baseline_bits,' // &
        'probe_bits,already_at_probe,noutputs_changed,max_abs_delta'
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
                      xnames, xindex, xcases, xprobe, ynames, yindex, ylive)
    character(len=*), intent(in) :: leaf
    procedure(leaf_eval)         :: evaluator
    character(len=*), intent(in) :: casenames(:)
    character(len=*), intent(in) :: ixnames(:)
    integer,          intent(in) :: ixcases(:, :)  ! (nix, ncase)
    character(len=*), intent(in) :: xnames(:)
    integer,          intent(in) :: xindex(:)
    real,             intent(in) :: xcases(:, :)   ! (nx, ncase)
    ! The substitute value each input slot is probed with.  It is 0.0 for
    ! every slot of every leaf except SFCDIF1's ZLVL, where zero would satisfy
    ! `IF(ZLVL <= ZPD)` at :4650 and take wrf_error_fatal, killing the run.
    ! The value used is recorded in the CSV, so no probe is implicit.
    real,             intent(in) :: xprobe(:)      ! (nx)
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

      ! Substitution-probe sweep on the oracle side.
      do i = 1, nx
        x1 = x0
        x1(i) = xprobe(i)
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
        if (rawbits(x0(i)) == rawbits(xprobe(i))) alreadyzero = 1
        write(unit_disc, '(*(g0,:,","))') trim(leaf), trim(casenames(ic)), &
            i, trim(xnames(i)), xindex(i), hexbits(x0(i)), &
            hexbits(xprobe(i)), alreadyzero, nchanged, maxdelta
      end do
    end do

    deallocate(x0, x1, y0, y1)
  end subroutine run_leaf

! -------------------------------------------------------------- leaf: ragrb --
!
! module_sf_noahmplsm.F:4483-4579.  Under-canopy aerodynamic resistance and
! leaf boundary-layer resistance.  Reads `parameters%DLEAF` (:4576) and nothing
! else off the parameter handle, so DLEAF is carried as an ordinary input and
! every other component is left untouched.
!
! Arguments the body never references, and which the fixture therefore varies
! only so that their inertness is measured:
!   TV     -- INTENT(IN) at :4507, never referenced;
!   VEGTYP -- INTENT(IN) at :4500, never referenced;
!   ILOC   -- INTENT(IN) at :4498, never referenced;
!   JLOC   -- INTENT(IN) at :4499, never referenced;
!   MOZG   -- INTENT(INOUT) at :4519 but assigned 0.0 at :4536 before any
!             read, so the incoming value cannot be consumed.
! FHG is INTENT(INOUT) and IS consumed, but only through `FHG = 0.5*(FHG +
! FHGNEW)` at :4557, i.e. only when ITER > 1.
!
! Branch coverage, each bound by inputs:
!   :4540 ITER > 1                iter{1,8} take the ELSE, the rest enter
!   :4542 ABS(TMP1) <= MPE        mpe_floors_tmp1 (HG = 1e-4 W/m2, MPE = 1e-4)
!   :4545 MIN(...,1.0) clamp      stable_clamp_mozg pins MOZG = 1
!   :4548 MOZG < 0                unstable cases take powf, stable take linear
!   :4556 ITER == 1               neutral_forest, rb_low_clamp
!   :4572 MAX(...,MPE) for KH     mpe_floors_kh (HCAN-ZPD = 2e-5, MPE = 1e-5)
!   :4577 MIN(MAX(RB,5),50)       rb_low_clamp (5), rb_high_clamp (50),
!                                 six interior cases
!
! DLEAF, CWP and MPE vary across cases.  MPTABLE ships DLEAF = 0.04 for all 20
! MODIS classes and every call site passes MPE = 1.E-6 (:3816, :4337), so a
! table-faithful fixture would freeze both and no mutant could be caught
! substituting the constant.  CWP is parameters%CWPVT, which does vary in the
! table (0.18 .. 5.0), and the fixture spans that range.

  subroutine eval_ragrb(x, ix, y)
    real,    intent(in)  :: x(:)
    integer, intent(in)  :: ix(:)
    real,    intent(out) :: y(:)
    type(noahmp_parameters) :: p
    real :: mozg, fhg, ramg, rahg, rawg, rb
    p%CH2OP = POISON
    p%Z0MVT = POISON
    p%HVT = POISON
    p%CWPVT = POISON
    p%DLEAF = x(17)
    mozg = x(15)
    fhg = x(16)
    ramg = POISON
    rahg = POISON
    rawg = POISON
    rb = POISON
    call ragrb(p, ix(1), x(1), x(2), x(3), x(5), &
               x(6), x(7), x(8), x(9), x(10), &
               x(11), x(12), x(13), ix(2), x(14), &
               x(4), mozg, fhg, ix(3), ix(4), &
               ramg, rahg, rawg, rb)
    y(1) = mozg
    y(2) = fhg
    y(3) = ramg
    y(4) = rahg
    y(5) = rawg
    y(6) = rb
  end subroutine eval_ragrb

  subroutine dump_ragrb()
    integer, parameter :: NX = 17, NY = 6, NCASE = 8, NIX = 4
    character(len=12) :: xn(NX), yn(NY), ixn(NIX)
    integer :: xi(NX), yi(NY)
    character(len=28) :: cn(NCASE)
    real :: xc(NX, NCASE), xp(NX)
    integer :: ixc(NIX, NCASE)
    logical :: ylive(NY, NCASE)

    ixn = [character(len=12) :: 'iter', 'vegtyp', 'iloc', 'jloc']
    xn = [character(len=12) :: 'vai', 'rhoair', 'hg', 'tv', 'tah', 'zpd', &
          'z0mg', 'z0hg', 'hcan', 'uc', 'z0h', 'fv', 'cwp', 'mpe', 'mozg', &
          'fhg', 'dleaf']
    xi = 0
    yn = [character(len=12) :: 'mozg', 'fhg', 'ramg', 'rahg', 'rawg', 'rb']
    yi = 0
    cn = [character(len=28) :: 'neutral_forest', 'unstable_forest', &
          'stable_grass', 'stable_clamp_mozg', 'mpe_floors_tmp1', &
          'mpe_floors_kh', 'rb_high_clamp', 'rb_low_clamp']

    ixc(:, 1) = [1,  1, 11, 21]
    ixc(:, 2) = [3,  2, 12, 22]
    ixc(:, 3) = [2, 10, 13, 23]
    ixc(:, 4) = [4,  5, 14, 24]
    ixc(:, 5) = [2,  7, 15, 25]
    ixc(:, 6) = [2, 12, 16, 26]
    ixc(:, 7) = [3, 14, 17, 27]
    ixc(:, 8) = [1, 16, 18, 28]

    !            vai   rhoair       hg      tv     tah      zpd     z0mg
    !           z0hg     hcan       uc      z0h      fv     cwp      mpe
    !           mozg      fhg    dleaf
    xc(:, 1) = [ 4.10, 1.1800,   45.00, 291.50, 290.20, 13.0000, 0.010000, &
               0.012000, 20.000,  2.4000, 1.09000, 0.3400, 0.1800, 1.0e-6, &
               -0.2500, 1.2500, 0.040000 ]
    xc(:, 2) = [ 3.60, 1.1500,  180.00, 298.00, 295.50,  9.7500, 0.008000, &
               0.009000, 15.000,  1.9000, 0.82000, 0.4196, 0.2500, 1.0e-6, &
                0.4000, 1.4000, 0.050000 ]
    xc(:, 3) = [ 2.20, 1.2000,  -30.00, 279.00, 280.00,  0.6500, 0.010000, &
               0.011000,  1.000,  1.2000, 0.12000, 0.1500, 5.0000, 1.0e-6, &
                0.1000, 1.1000, 0.030000 ]
    xc(:, 4) = [ 5.20, 1.3000,  -80.00, 266.00, 265.00, 13.0000, 0.010000, &
               0.014000, 20.000,  6.0000, 1.00000, 0.0500, 0.2000, 1.0e-6, &
               -0.0500, 2.0000, 0.060000 ]
    xc(:, 5) = [ 1.80, 1.1500,  1.0e-4, 288.00, 290.00,  1.3000, 0.005000, &
               0.006000,  2.000,  1.5000, 0.15000, 0.3000, 1.6700, 1.0e-4, &
                0.0700, 0.9000, 0.045000 ]
    xc(:, 6) = [ 0.90, 1.2500,   60.00, 284.00, 286.00, 1.99998, 0.010000, &
               0.013000,  2.000,  3.0000, 0.10000, 0.0500, 0.3000, 1.0e-5, &
               -0.1500, 1.0500, 0.020000 ]
    xc(:, 7) = [ 6.00, 1.0500,  250.00, 305.00, 302.00,  1.3000, 0.020000, &
               0.025000,  2.000,  0.5000, 0.20000, 0.5497, 5.0000, 1.0e-6, &
                0.3000, 1.7000, 0.070000 ]
    xc(:, 8) = [ 0.50, 1.2200,  -10.00, 274.00, 275.00,  0.3250, 0.030000, &
               0.035000,  0.500, 20.0000, 0.05000, 0.6000, 0.1800, 1.0e-6, &
                0.2000, 0.8000, 0.040000 ]

    xp = 0.0
    ylive = .true.
    call run_leaf('ragrb', eval_ragrb, cn, ixn, ixc, xn, xi, xc, xp, &
                  yn, yi, ylive)
  end subroutine dump_ragrb

! ------------------------------------------------------------ leaf: sfcdif1 --
!
! module_sf_noahmplsm.F:4583-4743.  Monin-Obukhov surface exchange
! coefficients.  It never touches `parameters`, so the harness passes an
! untouched handle.
!
! Arguments the body never references: ILOC (:4595), JLOC (:4596).  Every other
! argument is consumed.  MOZ, MOZSGN, FM, FH, FM2, FH2 and FV are INTENT(INOUT)
! and are both read and written; the fixture carries each as an input slot and
! an output slot.  MOZSGN is integer, so it appears in the `int` vector and is
! echoed to the output vector as a real.
!
! The `IF(ZLVL <= ZPD)` guard at :4650 calls wrf_error_fatal and stops the
! model.  It is a live branch that cannot be pinned as a value, so no case
! takes it; the ports raise instead, and the fixture asserts ZLVL > ZPD in
! every case.
!
! Branch coverage, each bound by inputs:
!   :4661 ITER == 1              neutral_start, and the ELSE everywhere else
!   :4669 ABS(TMP1) <= MPE       mpe_floors_tmp1 (H = 2e-4 W/m2, MPE = 1e-4)
!   :4671 MIN(...,1.0) for MOZ   moz_clamped_to_one pins MOZ = MOZ2 = 1
!   :4677 MOZOLD*MOZ < 0.0       sign_flip_counts, sign_flip_resets
!   :4678 MOZSGN >= 2 reset      sign_flip_resets (0->2), mozsgn_already_two
!   :4687 MOZ < 0                unstable cases take the ATAN branch, stable
!                                and reset cases take the linear one
!   :4708 ITER == 1 weighting    neutral_start
!   :4720-4723 MIN(F,0.9*TMPC)   clamps_bind binds all four
!   :4729-4732 ABS(..) <= MPE    degenerate_guards binds all four
!
! MPE varies (1e-6 at both call sites, 1e-4 in mpe_floors_tmp1) for the same
! mutation-detectability reason given in the RAGRB header.

  subroutine eval_sfcdif1(x, ix, y)
    real,    intent(in)  :: x(:)
    integer, intent(in)  :: ix(:)
    real,    intent(out) :: y(:)
    type(noahmp_parameters) :: p
    integer :: mozsgn
    real :: moz, fm, fh, fm2, fh2, fv, cm, ch, ch2
    mozsgn = ix(2)
    moz = x(11)
    fm = x(12)
    fh = x(13)
    fm2 = x(14)
    fh2 = x(15)
    fv = x(16)
    cm = POISON
    ch = POISON
    ch2 = POISON
    call sfcdif1(p, ix(1), x(1), x(2), x(3), x(4), &
                 x(5), x(6), x(7), x(8), x(9), &
                 x(10), ix(3), ix(4), &
                 moz, mozsgn, fm, fh, fm2, fh2, &
                 cm, ch, fv, ch2)
    y(1) = moz
    y(2) = real(mozsgn)
    y(3) = fm
    y(4) = fh
    y(5) = fm2
    y(6) = fh2
    y(7) = fv
    y(8) = cm
    y(9) = ch
    y(10) = ch2
  end subroutine eval_sfcdif1

  subroutine dump_sfcdif1()
    integer, parameter :: NX = 16, NY = 10, NCASE = 10, NIX = 4
    character(len=12) :: xn(NX), yn(NY), ixn(NIX)
    integer :: xi(NX), yi(NY)
    character(len=28) :: cn(NCASE)
    real :: xc(NX, NCASE), xp(NX)
    integer :: ixc(NIX, NCASE)
    logical :: ylive(NY, NCASE)

    ixn = [character(len=12) :: 'iter', 'mozsgn', 'iloc', 'jloc']
    xn = [character(len=12) :: 'sfctmp', 'rhoair', 'h', 'qair', 'zlvl', &
          'zpd', 'z0m', 'z0h', 'ur', 'mpe', 'moz', 'fm', 'fh', 'fm2', &
          'fh2', 'fv']
    xi = 0
    yn = [character(len=12) :: 'moz', 'mozsgn', 'fm', 'fh', 'fm2', 'fh2', &
          'fv', 'cm', 'ch', 'ch2']
    yi = 0
    cn = [character(len=28) :: 'neutral_start', 'unstable_moderate', &
          'stable_moderate', 'sign_flip_counts', 'sign_flip_resets', &
          'mozsgn_already_two', 'mpe_floors_tmp1', 'moz_clamped_to_one', &
          'clamps_bind', 'degenerate_guards']

    ixc(:, 1)  = [1, 0, 31, 41]
    ixc(:, 2)  = [2, 0, 32, 42]
    ixc(:, 3)  = [3, 0, 33, 43]
    ixc(:, 4)  = [2, 0, 34, 44]
    ixc(:, 5)  = [2, 1, 35, 45]
    ixc(:, 6)  = [4, 3, 36, 46]
    ixc(:, 7)  = [2, 0, 37, 47]
    ixc(:, 8)  = [3, 0, 38, 48]
    ixc(:, 9)  = [5, 0, 39, 49]
    ixc(:, 10) = [2, 2, 40, 50]

    !            sfctmp  rhoair          h     qair       zlvl      zpd
    !               z0m      z0h      ur     mpe      moz       fm      fh
    !               fm2      fh2      fv
    xc(:, 1)  = [ 290.00, 1.1800,   120.00, 0.01000,   10.0000, 0.6500, &
                 0.10000, 0.10000, 3.5000, 1.0e-6, -0.2000,  0.5000, 0.4000, &
                  0.3000,  0.2500, 0.3000 ]
    xc(:, 2)  = [ 295.00, 1.1500,   250.00, 0.01200,   10.0000, 0.6500, &
                 0.10000, 0.10000, 3.0000, 1.0e-6, -0.1000,  0.6000, 0.5500, &
                  0.4000,  0.3500, 0.3499 ]
    xc(:, 3)  = [ 278.00, 1.2500,   -45.00, 0.00350,   10.0000, 0.3250, &
                 0.05000, 0.05000, 2.2000, 1.0e-6,  0.0500, -0.3000, -0.2800, &
                 -0.1500, -0.1400, 0.1999 ]
    xc(:, 4)  = [ 292.00, 1.1600,   160.00, 0.00900,   12.0000, 0.5000, &
                 0.08000, 0.08000, 2.8000, 1.0e-6,  0.3000,  0.2000, 0.1800, &
                  0.1200,  0.1000, 0.2500 ]
    xc(:, 5)  = [ 287.00, 1.2000,    95.00, 0.00700,    8.0000, 0.4000, &
                 0.06000, 0.06000, 2.5000, 1.0e-6,  0.2500,  0.4500, 0.4200, &
                  0.2200,  0.2000, 0.2800 ]
    xc(:, 6)  = [ 284.00, 1.2200,    70.00, 0.00550,    6.0000, 0.2000, &
                 0.04000, 0.04000, 2.0000, 1.0e-6, -0.4000,  0.7000, 0.6500, &
                  0.3300,  0.3000, 0.2200 ]
    xc(:, 7)  = [ 286.00, 1.1900,  2.0e-04, 0.00600,   10.0000, 0.5500, &
                 0.07000, 0.07000, 3.2000, 1.0e-4, -0.0500,  0.3000, 0.2900, &
                  0.1600,  0.1500, 0.3300 ]
    xc(:, 8)  = [ 272.00, 1.3000,   -90.00, 0.00200,   10.0000, 0.3000, &
                 0.03000, 0.03000, 1.5000, 1.0e-6,  0.8000, -1.0000, -0.9500, &
                 -0.5000, -0.4800, 0.0400 ]
    xc(:, 9)  = [ 294.00, 1.1200,    86.00, 0.01100,    2.5000, 0.5000, &
                 1.00000, 1.00000, 1.8000, 1.0e-6, -0.3000,  1.2000, 1.0000, &
                  1.1000,  1.0000, 0.1500 ]
    xc(:, 10) = [ 289.00, 1.1700,   140.00, 0.00800, 4000004.0, 2.0000, &
                 4000000.0, 4000000.0, 2.0000, 1.0e-6, -0.1500, 0.9000, &
                 0.8000, 0.6000, 0.5000, 0.4000 ]

    ! Slot 5 is ZLVL.  Probing it with 0.0 would satisfy IF(ZLVL <= ZPD) at
    ! :4650 and take wrf_error_fatal, so it is probed with 3.0 m instead --
    ! above every ZPD in the table and equal to no case's own ZLVL.
    xp = 0.0
    xp(5) = 3.0
    ylive = .true.
    call run_leaf('sfcdif1', eval_sfcdif1, cn, ixn, ixc, xn, xi, xc, xp, &
                  yn, yi, ylive)
  end subroutine dump_sfcdif1

! ------------------------------------------------------------ leaf: stomata --
!
! module_sf_noahmplsm.F:5005-5137.  Ball-Berry stomatal resistance and leaf
! photosynthesis, the live OPT_CRS = 1 canopy-conductance leaf.
!
! Reads eleven `parameters` components -- BP, FOLNMX, QE25, KC25, AKC, KO25,
! AKO, VCMX25, AVCMX, C3PSN, MP -- and nothing else, so each is carried as an
! ordinary input.  The harness poisons the seven neighbouring photosynthesis
! scalars (AQE, RMF25, RMS25, RMR25, ARM, MRP, DLEAF) so that a port reading
! the wrong component of the handle is visibly wrong.
!
! Arguments the body never references: VEGTYP (:5013), ILOC (:5011),
! JLOC (:5012).
!
! Branch coverage, each bound by inputs:
!   :5085 APAR_scale <= 0 RETURN   dark_early_return (APAR = 0)
!   :5083 MAX(FVEG,1.0e-6)         fveg_clamp_folnmx_zero (FVEG = 1e-7)
!   :5088 MAX(MPE,FOLNMX)          fveg_clamp_folnmx_zero (FOLNMX = 0)
!   :5088 MIN(...,1.0) for FNF     forest_wj_limited has FOLN < FOLNMX
!   :5108 MAX / MIN(EA,EI) in CEA  grass_wc_limited: EA below 0.25*EI;
!                                  crop_we_limited: EA above EI;
!                                  the rest: 0.25*EI < EA < EI
!   :5113-5116 which of WJ/WC/WE   forest_wj_limited and
!              is MIN              fveg_clamp_folnmx_zero (WJ, and they carry
!                                  different QE25 so no frozen-QE25 mutant
!                                  survives) / dry_canopy_ci_floor (WC) /
!                                  crop_we_limited (WE)
!   :5113 MAX(CI-CP,0.0)           co2_starved_* drive CI below CP
!   :5118 MAX(...,MPE) for CS      co2_starved_fine (MPE=1e-6) and
!                                  co2_starved_coarse (MPE=1e-3)
!   :5121 B >= 0                   highland_crop_b_nonneg, co2_starved_*
!   :5121 B <  0                   every other case
!   :5128 MAX(...,0.0) for CI       co2_starved_*
!   C3PSN = 0                      c4_pathway exercises every (1-C3PSN) term
!
! MPTABLE ships C3PSN = 1.0, AKC = 2.1, AKO = 1.2, AVCMX = 2.4, KC25 = 30 and
! KO25 = 30000 for all 20 MODIS classes, so the C4 terms are unreachable
! through the pinned table and those five q10/Michaelis constants are frozen
! there.  They vary here because the fixture pins the routine, not the table;
! without that variation the mutation study cannot distinguish a port that
! reads the argument from one that inlines the table value.

  subroutine eval_stomata(x, ix, y)
    real,    intent(in)  :: x(:)
    integer, intent(in)  :: ix(:)
    real,    intent(out) :: y(:)
    type(noahmp_parameters) :: p
    real :: rs, psn
    p%AQE = POISON
    p%RMF25 = POISON
    p%RMS25 = POISON
    p%RMR25 = POISON
    p%ARM = POISON
    p%MRP = POISON
    p%DLEAF = POISON
    p%BP = x(15)
    p%FOLNMX = x(16)
    p%QE25 = x(17)
    p%KC25 = x(18)
    p%AKC = x(19)
    p%KO25 = x(20)
    p%AKO = x(21)
    p%VCMX25 = x(22)
    p%AVCMX = x(23)
    p%C3PSN = x(24)
    p%MP = x(25)
    rs = POISON
    psn = POISON
    call stomata(p, ix(1), x(1), x(2), x(3), ix(2), ix(3), &
                 x(4), x(5), x(6), x(7), x(8), x(9), &
                 x(10), x(11), x(12), x(13), x(14), &
                 rs, psn)
    y(1) = rs
    y(2) = psn
  end subroutine eval_stomata

  subroutine dump_stomata()
    integer, parameter :: NX = 25, NY = 2, NCASE = 10, NIX = 3
    character(len=12) :: xn(NX), yn(NY), ixn(NIX)
    integer :: xi(NX), yi(NY)
    character(len=28) :: cn(NCASE)
    real :: xc(NX, NCASE), xp(NX)
    integer :: ixc(NIX, NCASE)
    logical :: ylive(NY, NCASE)

    ixn = [character(len=12) :: 'vegtyp', 'iloc', 'jloc']
    xn = [character(len=12) :: 'mpe', 'apar', 'foln', 'tv', 'ei', 'ea', &
          'sfctmp', 'sfcprs', 'fveg', 'o2', 'co2', 'igs', 'btran', 'rb', &
          'bp', 'folnmx', 'qe25', 'kc25', 'akc', 'ko25', 'ako', 'vcmx25', &
          'avcmx', 'c3psn', 'mp']
    xi = 0
    yn = [character(len=12) :: 'rs', 'psn']
    yi = 0
    cn = [character(len=28) :: 'forest_wj_limited', 'dry_canopy_ci_floor', &
          'crop_we_limited', 'c4_pathway', 'dark_early_return', &
          'fveg_clamp_folnmx_zero', 'highland_crop_b_nonneg', &
          'co2_starved_fine', 'co2_starved_coarse', 'igs_zero_dormant']

    ixc(:, 1)  = [ 1, 51, 61]
    ixc(:, 2)  = [10, 52, 62]
    ixc(:, 3)  = [12, 53, 63]
    ixc(:, 4)  = [ 6, 54, 64]
    ixc(:, 5)  = [13, 55, 65]
    ixc(:, 6)  = [16, 56, 66]
    ixc(:, 7)  = [12, 57, 67]
    ixc(:, 8)  = [ 5, 58, 68]
    ixc(:, 9)  = [ 2, 59, 69]
    ixc(:, 10) = [14, 60, 70]

    !               mpe     apar    foln      tv       ei       ea   sfctmp
    !            sfcprs     fveg      o2     co2     igs   btran      rb
    !                bp   folnmx    qe25    kc25     akc      ko25    ako
    !            vcmx25    avcmx   c3psn      mp
    xc(:, 1)  = [ 1.0e-6,   5.000, 1.2000, 298.00, 3170.00, 2100.00, 296.00, &
                 97000.0, 0.95000, 20900.0,  40.00, 1.0000, 0.8000, 25.000, &
                 2000.00,  1.5000, 0.06000, 30.000, 2.1000, 30000.00, 1.2000, &
                  50.000,  2.4000, 1.0000, 6.0000 ]
    xc(:, 2)  = [ 1.0e-6, 600.000, 1.8000, 301.00, 3900.00,  400.00, 299.00, &
                100000.0, 0.80000, 20900.0,  40.00, 1.0000, 0.6000,  5.000, &
                 2000.00,  1.5000, 0.05500, 30.000, 2.1000, 30000.00, 1.2000, &
                 120.000,  2.4000, 1.0000, 6.0000 ]
    xc(:, 3)  = [ 1.0e-6, 420.000, 2.4000, 300.00, 3600.00, 3900.00, 297.00, &
                 99000.0, 0.90000, 20900.0,  40.00, 0.9000, 1.0000, 12.000, &
                 2000.00,  1.5000, 0.06500, 10.000, 1.8000, 30000.00, 1.3000, &
                  80.000,  2.2000, 1.0000, 9.0000 ]
    xc(:, 4)  = [ 1.0e-6, 310.000, 1.6000, 303.00, 4250.00, 2600.00, 301.00, &
                 98000.0, 0.85000, 20900.0,  40.00, 1.0000, 0.7000, 20.000, &
                 4000.00,  1.5000, 0.04000, 30.000, 2.1000, 30000.00, 1.2000, &
                  55.000,  2.4000, 0.0000, 9.0000 ]
    xc(:, 5)  = [ 1.0e-6,   0.000, 1.5000, 288.00, 1700.00, 1500.00, 287.00, &
                101000.0, 0.60000, 20900.0,  40.00, 0.5000, 0.9000, 30.000, &
                  1.0e15,  0.0000, 0.00000, 30.000, 2.1000, 30000.00, 1.2000, &
                   0.000,  2.4000, 1.0000, 9.0000 ]
    xc(:, 6)  = [ 1.0e-6,  1.0e-7, 0.9000, 294.00, 2600.00, 1900.00, 293.00, &
                 96000.0,  1.0e-7, 20900.0,  40.00, 1.0000, 0.5000, 35.000, &
                 2000.00,  0.0000, 0.07000, 26.000, 2.0000, 28000.00, 1.1000, &
                  45.000,  2.3000, 1.0000, 8.0000 ]
    xc(:, 7)  = [ 1.0e-6, 380.000, 2.0000, 292.00, 2400.00, 1600.00, 290.00, &
                 60000.0, 0.92000, 12500.0,  24.00, 1.0000, 1.0000, 50.000, &
                 2000.00,  1.5000, 0.06000, 30.000, 2.1000, 30000.00, 1.2000, &
                  80.000,  2.4000, 1.0000, 9.0000 ]
    xc(:, 8)  = [ 1.0e-6, 300.000, 1.4000, 297.00, 3000.00, 2000.00, 295.00, &
                 97000.0, 0.88000,   200.0,   0.50, 1.0000, 0.9000, 50.000, &
                 2000.00,  1.5000, 0.06000,  3.000, 2.1000, 30000.00, 1.2000, &
                  60.000,  2.4000, 1.0000, 6.0000 ]
    xc(:, 9)  = [ 1.0e-3, 290.000, 1.3000, 296.00, 2900.00, 1800.00, 294.00, &
                 95000.0, 0.86000,   180.0,   0.40, 1.0000, 0.8500, 48.000, &
                 2500.00,  1.4000, 0.05800,  2.500, 2.2000, 31000.00, 1.2500, &
                  58.000,  2.5000, 1.0000, 7.0000 ]
    xc(:, 10) = [ 1.0e-6, 150.000, 1.1000, 285.00, 1900.00, 1300.00, 284.00, &
                 99500.0, 0.75000, 20900.0,  38.00, 0.0000, 0.4000, 28.000, &
                 3000.00,  1.6000, 0.05000, 28.000, 1.9000, 29000.00, 1.1500, &
                  35.000,  2.1000, 1.0000, 5.0000 ]

    xp = 0.0
    ylive = .true.
    call run_leaf('stomata', eval_stomata, cn, ixn, ixc, xn, xi, xc, xp, &
                  yn, yi, ylive)
  end subroutine dump_stomata

! -------------------------------------------------------------- entry point --

  subroutine dump_all()
    call dump_ragrb()
    call dump_sfcdif1()
    call dump_stomata()
  end subroutine dump_all

end module noahmp_fluxprep_oracle


program run_noahmp_fluxprep_oracle
  use module_sf_noahmplsm, only: noahmp_options
  use noahmp_fluxprep_oracle, only: open_outputs, close_outputs, dump_all
  implicit none
  character(len=1024) :: leaf_path, disc_path

  call get_command_argument(1, leaf_path)
  call get_command_argument(2, disc_path)
  if (len_trim(leaf_path) == 0 .or. len_trim(disc_path) == 0) then
    write(*, '(A)') 'usage: run_fluxprep LEAVES.csv DISCRIMINATION.csv'
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
end program run_noahmp_fluxprep_oracle
