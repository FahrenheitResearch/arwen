! run_vegeflux.F90 -- bitwise fixture generator for the Noah-MP VEGE_FLUX subtree.
!
! Calls the *unmodified* WRF v4.6.1 phys/module_sf_noahmplsm.F (visibility patch
! only) and writes one CSV row per scalar crossing the interface, as the exact
! IEEE-754 binary32 bit pattern.  Nothing here recomputes any physics; the CSV is
! whatever gfortran + glibc produced.
!
! CSV columns:  leaf,case,role,name,hex,value
!   role  = "in"  argument value on entry (for INOUT, the value written in)
!           "out" argument value on exit
!           "opt" pinned option identity / parameter, echoed for provenance
!   hex   = TRANSFER(value, integer) printed Z8.8 -- the byte-exact fixture
!   value = decimal rendering, for humans only; hex is authoritative
!
! Option identity is set through the module's own noahmp_options entry point at
! the exact WRF Registry defaults, so the fixture cannot silently drift onto a
! branch WRF never takes.
!
! usage:  run_vegeflux <esat|ragrb|sfcdif1|stomata|vegeflux|all>

program run_vegeflux
  use module_sf_noahmplsm
  implicit none

  character(len=32) :: which

  call get_command_argument(1, which)
  if (len_trim(which) == 0) which = 'all'

  ! WRF Registry defaults, verbatim:
  !   dveg=4 opt_crs=1 opt_btr=1 opt_run=3 opt_sfc=1 opt_frz=1 opt_inf=1
  !   opt_rad=3 opt_alb=2 opt_snf=1 opt_tbot=2 opt_stc=1 opt_rsf=1 opt_soil=1
  !   opt_pedo=1 opt_crop=0 opt_irr=0 opt_irrm=0 opt_infdv=0 opt_tdrn=0
  call noahmp_options(4, 1, 1, 3, 1, 1, 1, 3, 2, 1, 2, 1, 1, 1, 1, 0, 0, 0, 0, 0)

  write(*,'(A)') 'leaf,case,role,name,hex,value'
  call emit_options()

  if (which == 'esat'    .or. which == 'all') call do_esat()
  if (which == 'ragrb'   .or. which == 'all') call do_ragrb()
  if (which == 'sfcdif1' .or. which == 'all') call do_sfcdif1()
  if (which == 'stomata' .or. which == 'all') call do_stomata()
  if (which == 'vegeflux'.or. which == 'all') call do_vegeflux()

contains

! ---------------------------------------------------------------------------
  subroutine emitr(leaf, case_, role, name, val)
    character(len=*), intent(in) :: leaf, case_, role, name
    real, intent(in) :: val
    integer :: ib
    ib = transfer(val, ib)
    write(*,'(A,",",A,",",A,",",A,",",Z8.8,",",ES17.9E2)') &
         trim(leaf), trim(case_), trim(role), trim(name), ib, val
  end subroutine emitr

  subroutine emiti(leaf, case_, role, name, val)
    character(len=*), intent(in) :: leaf, case_, role, name
    integer, intent(in) :: val
    write(*,'(A,",",A,",",A,",",A,",",Z8.8,",",I17)') &
         trim(leaf), trim(case_), trim(role), trim(name), val, val
  end subroutine emiti

  subroutine emitr_idx(leaf, case_, role, name, idx, val)
    character(len=*), intent(in) :: leaf, case_, role, name
    integer, intent(in) :: idx
    real, intent(in) :: val
    character(len=48) :: nm
    write(nm,'(A,"_",SP,I0)') trim(name), idx
    call emitr(leaf, case_, role, nm, val)
  end subroutine emitr_idx

  subroutine emit_options()
    call emiti('options','registry','opt','DVEG',     DVEG)
    call emiti('options','registry','opt','OPT_CRS',  OPT_CRS)
    call emiti('options','registry','opt','OPT_BTR',  OPT_BTR)
    call emiti('options','registry','opt','OPT_RUN',  OPT_RUN)
    call emiti('options','registry','opt','OPT_SFC',  OPT_SFC)
    call emiti('options','registry','opt','OPT_STC',  OPT_STC)
    call emiti('options','registry','opt','OPT_CROP', OPT_CROP)
    call emiti('options','registry','opt','OPT_IRR',  OPT_IRR)
    call emiti('options','registry','opt','OPT_TDRN', OPT_TDRN)
    call emiti('options','registry','opt','OPT_ALB',  OPT_ALB)
    call emiti('options','registry','opt','OPT_RAD',  OPT_RAD)
  end subroutine emit_options

! ---------------------------------------------------------------------------
! Parameter block.  MPTABLE.TBL (WRF v4.6.1 run/MPTABLE.TBL), MODIS-IGBP
! "Evergreen Needleleaf Forest"-class values for the photosynthesis group; every
! MPTABLE row shares C3PSN/KC25/AKC/KO25/AKO/AVCMX/FOLNMX, and DLEAF/CBIOM are
! uniform across all rows.  HVT and VCMX25/BP/MP/QE25 are per-row and are set by
! the caller so the fixture can bind more than one row.
  subroutine set_parameters(p, hvt, vcmx25, bp, mp, qe25)
    type(noahmp_parameters), intent(inout) :: p
    real, intent(in) :: hvt, vcmx25, bp, mp, qe25
    p%DLEAF  = 0.04
    p%HVT    = hvt
    p%CBIOM  = 0.02
    p%C3PSN  = 1.0
    p%KC25   = 30.0
    p%AKC    = 2.1
    p%KO25   = 3.E4
    p%AKO    = 1.2
    p%AVCMX  = 2.4
    p%VCMX25 = vcmx25
    p%BP     = bp
    p%MP     = mp
    p%QE25   = qe25
    p%FOLNMX = 1.5
    p%SMCWLT = 0.10       ! gecros only; opt_crop=0 makes it dead
  end subroutine set_parameters

! ---------------------------------------------------------------------------
! ESAT: straight-line polynomial, no branches.  Cases sweep the TDC-clamped
! range [-50, 50] C that VEGE_FLUX can present, plus both signs of the T > 0
! test its callers make on the result.
  subroutine do_esat()
    integer, parameter :: NC = 11
    real :: tv(NC)
    real :: esw, esi, desw, desi
    integer :: i
    character(len=24) :: cs
    tv = (/ -50.0, -40.0, -25.375, -10.0, -0.5, 0.0, 0.5, 12.25, 27.31571, &
            40.0, 50.0 /)
    do i = 1, NC
      write(cs,'("esat",I2.2)') i
      call ESAT(tv(i), esw, esi, desw, desi)
      call emitr('esat', cs, 'in',  'T',    tv(i))
      call emitr('esat', cs, 'out', 'ESW',  esw)
      call emitr('esat', cs, 'out', 'ESI',  esi)
      call emitr('esat', cs, 'out', 'DESW', desw)
      call emitr('esat', cs, 'out', 'DESI', desi)
    end do
  end subroutine do_esat

! ---------------------------------------------------------------------------
! RAGRB branch map (case tag -> branch bound):
!   r01 ITER=1                     -> MOZG=0 skip, FHG = FHGNEW, MOZG>=0 leg
!   r02 ITER=2, HG>0               -> MOLG<0, MOZG<0 leg, FHG averaged
!   r03 ITER=3, HG<0               -> MOLG>0, MOZG>=0 leg (1+4.7*MOZG)
!   r04 ITER=2, HG ~ 0             -> ABS(TMP1) <= MPE -> TMP1 = MPE
!   r05 ITER=2, HG<0 tiny MOLG     -> MOZG capped at 1.0 by MIN
!   r06 ITER=1, large UC           -> RB clamped up to 5.0 by MAX
!   r07 ITER=1, tiny UC            -> RB clamped down to 50.0 by MIN
!   r08 ITER=1, FV ~ 0             -> KH = MPE leg of the MAX
!   r09 ITER=1, CWP tuned          -> CWPC = (..)**0.5 lands on an argument
!                                     where glibc powf and a naive float32 pow
!                                     disagree, so the fixture pins the libm
!                                     identity and not just the algebra
!   r10 ITER=2, HG tuned           -> FHGNEW = (..)**(-0.25) likewise
  subroutine do_ragrb()
    type(noahmp_parameters) :: p
    integer, parameter :: NC = 10
    integer :: iter(NC), i
    real :: vai(NC), rhoair(NC), hg(NC), tah(NC), zpd(NC), z0mg(NC), z0hg(NC)
    real :: hcan(NC), uc(NC), z0h(NC), fv(NC), cwp(NC), tv(NC), mozg0(NC), fhg0(NC)
    real :: mozg, fhg, ramg, rahg, rawg, rb
    character(len=24) :: cs

    iter   = (/    1,     2,     3,      2,       2,     1,      1,      1,        1,      2 /)
    vai    = (/  3.5,   3.5,   3.5,    3.5,     3.5,   3.5,    3.5,    3.5,      3.5,    3.5 /)
    rhoair = (/ 1.18,  1.18,  1.18,   1.18,    1.18,  1.18,   1.18,   1.18,     1.18,   1.18 /)
    hg     = (/  0.0,  85.0, -35.0, 1.0E-9, -2.0E-4,   0.0,    0.0,    0.0,      0.0, 50.206 /)
    tah    = (/295.0, 295.0, 271.0,  295.0,   295.0, 295.0,  295.0,  295.0,    295.0,  295.0 /)
    zpd    = (/ 6.70,  6.70,  6.70,   6.70,    6.70,  6.70,   6.70,   6.70,     6.70,   6.70 /)
    z0mg   = (/ 0.01,  0.01,  0.01,   0.01,    0.01,  0.01,   0.01,   0.01,     0.01,   0.01 /)
    z0hg   = (/ 0.01,  0.01,  0.01,   0.01,    0.01,  0.01,   0.01,   0.01,     0.01,   0.01 /)
    hcan   = (/ 10.0,  10.0,  10.0,   10.0,    10.0,  10.0,   10.0,   6.701,    10.0,   10.0 /)
    uc     = (/  2.4,   2.4,   2.4,    2.4,     2.4, 900.0, 1.0E-4,    2.4,      2.4,    2.4 /)
    z0h    = (/  1.0,   1.0,   1.0,    1.0,     1.0,   1.0,    1.0,    1.0,      1.0,    1.0 /)
    fv     = (/ 0.35,  0.35,  0.35,   0.35,    0.35,  0.35,   0.35, 1.0E-7,     0.35,   0.35 /)
    cwp    = (/  3.0,   3.0,   3.0,    3.0,     3.0,   3.0,    3.0,    3.0,  2.01364,    3.0 /)
    tv     = (/296.0, 296.0, 270.0,  296.0,   296.0, 296.0,  296.0,  296.0,    296.0,  296.0 /)
    mozg0  = (/  0.0,  -0.4,   0.2,    0.0,    -0.1,   0.0,    0.0,    0.0,      0.0,   -0.4 /)
    fhg0   = (/  1.0,   1.3,   1.1,    1.0,     1.2,   1.0,    1.0,    1.0,      1.0,    1.0 /)

    call set_parameters(p, 20.0, 60.0, 2.E3, 9.0, 0.06)

    do i = 1, NC
      write(cs,'("ragrb",I2.2)') i
      mozg = mozg0(i)
      fhg  = fhg0(i)
      call emiti('ragrb', cs, 'in', 'ITER',   iter(i))
      call emitr('ragrb', cs, 'in', 'VAI',    vai(i))
      call emitr('ragrb', cs, 'in', 'RHOAIR', rhoair(i))
      call emitr('ragrb', cs, 'in', 'HG',     hg(i))
      call emitr('ragrb', cs, 'in', 'TAH',    tah(i))
      call emitr('ragrb', cs, 'in', 'ZPD',    zpd(i))
      call emitr('ragrb', cs, 'in', 'Z0MG',   z0mg(i))
      call emitr('ragrb', cs, 'in', 'Z0HG',   z0hg(i))
      call emitr('ragrb', cs, 'in', 'HCAN',   hcan(i))
      call emitr('ragrb', cs, 'in', 'UC',     uc(i))
      call emitr('ragrb', cs, 'in', 'Z0H',    z0h(i))
      call emitr('ragrb', cs, 'in', 'FV',     fv(i))
      call emitr('ragrb', cs, 'in', 'CWP',    cwp(i))
      call emitr('ragrb', cs, 'in', 'MPE',    1.0E-6)
      call emitr('ragrb', cs, 'in', 'TV',     tv(i))
      call emitr('ragrb', cs, 'in', 'MOZG',   mozg)
      call emitr('ragrb', cs, 'in', 'FHG',    fhg)
      call emitr('ragrb', cs, 'in', 'P_DLEAF', p%DLEAF)
      call RAGRB(p, iter(i), vai(i), rhoair(i), hg(i), tah(i), &
                 zpd(i), z0mg(i), z0hg(i), hcan(i), uc(i), &
                 z0h(i), fv(i), cwp(i), 1, 1.0E-6, &
                 tv(i), mozg, fhg, 1, 1, &
                 ramg, rahg, rawg, rb)
      call emitr('ragrb', cs, 'out', 'MOZG', mozg)
      call emitr('ragrb', cs, 'out', 'FHG',  fhg)
      call emitr('ragrb', cs, 'out', 'RAMG', ramg)
      call emitr('ragrb', cs, 'out', 'RAHG', rahg)
      call emitr('ragrb', cs, 'out', 'RAWG', rawg)
      call emitr('ragrb', cs, 'out', 'RB',   rb)
    end do
  end subroutine do_ragrb

! ---------------------------------------------------------------------------
! SFCDIF1 branch map:
!   s01 ITER=1                       -> FV/MOZ/MOL/MOZ2 zeroed, MOZ>=0 leg
!   s02 ITER=2, H>0                  -> MOL<0, MOZ<0 leg (the ATAN branch)
!   s03 ITER=2, H<0                  -> MOL>0, MOZ>=0 leg, MOZ capped by MIN
!   s04 ITER=2, H ~ 0                -> ABS(TMP1) <= MPE -> TMP1 = MPE
!   s05 ITER=2, MOZ sign flip        -> MOZOLD*MOZ<0 -> MOZSGN 1->2 -> reset
!   s06 ITER=2 entering with MOZSGN=1 and a second flip -> reset leg taken
!   s07 ITER=2 strongly stable       -> FH/FM clamped by 0.9*TMPCH / 0.9*TMPCM
!   s08 ZLVL-ZPD barely above Z0M    -> TMPCM/TMPCH ~ 0 -> ABS(CMFM)<=MPE leg
!   s09 ITER=2, H tuned              -> ATAN((1-16*MOZ)**0.25) lands on an
!                                       argument where glibc atanf and a naive
!                                       float32 arctan disagree, so the fixture
!                                       pins the libm identity
!   s10 ITER=2, H tuned              -> (1-16*MOZ)**0.25 likewise for powf
  subroutine do_sfcdif1()
    type(noahmp_parameters) :: p
    integer, parameter :: NC = 10
    integer :: iter(NC), mozsgn0(NC), mozsgn, i
    real :: sfctmp(NC), rhoair(NC), h(NC), qair(NC), zlvl(NC), zpd(NC)
    real :: z0m(NC), z0h(NC), ur(NC), moz0(NC), fm0(NC), fh0(NC), fm20(NC), fh20(NC), fv0(NC)
    real :: moz, fm, fh, fm2, fh2, fv, cm, ch, ch2
    character(len=24) :: cs

    iter    = (/     1,     2,     2,      2,      2,      2,      2,      2,      2,      2 /)
    sfctmp  = (/ 295.0, 295.0, 271.0,  295.0,  295.0,  295.0,  262.0,  295.0,  295.0,  295.0 /)
    rhoair  = (/  1.18,  1.18,  1.30,   1.18,   1.18,   1.18,   1.32,   1.18,   1.18,   1.18 /)
    h       = (/   0.0, 180.0, -60.0, 1.0E-9,  -55.0,   40.0, -140.0,   25.0, 50.057, 50.216 /)
    qair    = (/ 0.010, 0.010, 0.003,  0.010,  0.010,  0.010,  0.002,  0.010,  0.010,  0.010 /)
    zlvl    = (/  30.0,  30.0,  30.0,   30.0,   30.0,   30.0,   30.0, 6.7005,   30.0,   30.0 /)
    zpd     = (/  6.70,  6.70,  6.70,   6.70,   6.70,   6.70,   6.70,   6.70,   6.70,   6.70 /)
    z0m     = (/  1.00,  1.00,  1.00,   1.00,   1.00,   1.00,   1.00, 0.0004,   1.00,   1.00 /)
    z0h     = (/  1.00,  1.00,  1.00,   1.00,   1.00,   1.00,   1.00, 0.0004,   1.00,   1.00 /)
    ur      = (/   4.2,   4.2,   2.1,    4.2,    4.2,    4.2,    1.6,    4.2,    4.2,    4.2 /)
    moz0    = (/   0.0,  -0.2,   0.1,    0.0,    0.3,   -0.3,    0.5,   -0.1,   -0.2,   -0.2 /)
    fm0     = (/   0.0,   0.4,  -0.2,    0.0,   -0.5,    0.6,   -1.0,    0.2,    0.4,    0.4 /)
    fh0     = (/   0.0,   0.5,  -0.3,    0.0,   -0.6,    0.7,   -1.2,    0.3,    0.5,    0.5 /)
    fm20    = (/   0.0,   0.1,  -0.1,    0.0,   -0.2,    0.2,   -0.4,    0.1,    0.1,    0.1 /)
    fh20    = (/   0.0,   0.2,  -0.1,    0.0,   -0.3,    0.3,   -0.5,    0.1,    0.2,    0.2 /)
    fv0     = (/   0.1,  0.35,  0.22,   0.35,   0.35,   0.35,   0.11,   0.35,   0.35,   0.35 /)
    mozsgn0 = (/     0,     0,     0,      0,      1,      1,      0,      0,      0,      0 /)

    call set_parameters(p, 20.0, 60.0, 2.E3, 9.0, 0.06)

    do i = 1, NC
      write(cs,'("sfcdif1",I2.2)') i
      moz = moz0(i); fm = fm0(i); fh = fh0(i); fm2 = fm20(i); fh2 = fh20(i)
      fv = fv0(i);   mozsgn = mozsgn0(i)
      call emiti('sfcdif1', cs, 'in', 'ITER',   iter(i))
      call emitr('sfcdif1', cs, 'in', 'SFCTMP', sfctmp(i))
      call emitr('sfcdif1', cs, 'in', 'RHOAIR', rhoair(i))
      call emitr('sfcdif1', cs, 'in', 'H',      h(i))
      call emitr('sfcdif1', cs, 'in', 'QAIR',   qair(i))
      call emitr('sfcdif1', cs, 'in', 'ZLVL',   zlvl(i))
      call emitr('sfcdif1', cs, 'in', 'ZPD',    zpd(i))
      call emitr('sfcdif1', cs, 'in', 'Z0M',    z0m(i))
      call emitr('sfcdif1', cs, 'in', 'Z0H',    z0h(i))
      call emitr('sfcdif1', cs, 'in', 'UR',     ur(i))
      call emitr('sfcdif1', cs, 'in', 'MPE',    1.0E-6)
      call emitr('sfcdif1', cs, 'in', 'MOZ',    moz)
      call emiti('sfcdif1', cs, 'in', 'MOZSGN', mozsgn)
      call emitr('sfcdif1', cs, 'in', 'FM',     fm)
      call emitr('sfcdif1', cs, 'in', 'FH',     fh)
      call emitr('sfcdif1', cs, 'in', 'FM2',    fm2)
      call emitr('sfcdif1', cs, 'in', 'FH2',    fh2)
      call emitr('sfcdif1', cs, 'in', 'FV',     fv)
      call SFCDIF1(p, iter(i), sfctmp(i), rhoair(i), h(i), qair(i), &
                   zlvl(i), zpd(i), z0m(i), z0h(i), ur(i), &
                   1.0E-6, 1, 1, &
                   moz, mozsgn, fm, fh, fm2, fh2, &
                   cm, ch, fv, ch2)
      call emitr('sfcdif1', cs, 'out', 'MOZ',    moz)
      call emiti('sfcdif1', cs, 'out', 'MOZSGN', mozsgn)
      call emitr('sfcdif1', cs, 'out', 'FM',     fm)
      call emitr('sfcdif1', cs, 'out', 'FH',     fh)
      call emitr('sfcdif1', cs, 'out', 'FM2',    fm2)
      call emitr('sfcdif1', cs, 'out', 'FH2',    fh2)
      call emitr('sfcdif1', cs, 'out', 'CM',     cm)
      call emitr('sfcdif1', cs, 'out', 'CH',     ch)
      call emitr('sfcdif1', cs, 'out', 'FV',     fv)
      call emitr('sfcdif1', cs, 'out', 'CH2',    ch2)
    end do
  end subroutine do_sfcdif1

! ---------------------------------------------------------------------------
! STOMATA branch map:
!   t01 APAR=0                 -> APAR_scale <= 0 early return (RS=CF/BP, PSN=0)
!   t02 bright, warm, wet      -> full 3-iteration path, B < 0 leg
!   t03 dim light              -> WJ limits PSN
!   t04 hot leaf               -> F2 inhibition large, WC/WE limit
!   t05 FOLN >> FOLNMX         -> FNF clamped to 1.0 by MIN
!   t06 EA > EI                -> CEA = MIN(EA,EI) -> EI
!   t07 EA very small          -> CEA = 0.25*EI leg of the MAX
!   t08 BTRAN=0                -> VCMX=0, PSN=0, degenerate quadratic (B>=0 leg)
!   t09 IGS=0                  -> PSN=0 with non-zero VCMX
!   t10 TV tuned               -> the Q10 responses AKC/AKO/AVCMX ** ((TC-25)/10)
!                                 land where glibc powf and a naive float32 pow
!                                 disagree, so the fixture pins the libm
!                                 identity for this leaf too
  subroutine do_stomata()
    type(noahmp_parameters) :: p
    integer, parameter :: NC = 10
    integer :: i
    real :: apar(NC), foln(NC), tv(NC), ei(NC), ea(NC), sfctmp(NC), sfcprs(NC)
    real :: fveg(NC), o2(NC), co2(NC), igs(NC), btran(NC), rb(NC)
    real :: rs, psn
    character(len=24) :: cs

    apar   = (/   0.0, 260.0,   8.0, 260.0, 260.0, 260.0, 260.0, 260.0, 260.0,  260.0 /)
    foln   = (/   1.0,   1.0,   1.0,   1.0,  12.0,   1.0,   1.0,   1.0,   1.0,    1.0 /)
    tv     = (/ 296.0, 296.0, 296.0, 318.0, 296.0, 296.0, 296.0, 296.0, 296.0, 286.224 /)
    ei     = (/2810.0,2810.0,2810.0,8100.0,2810.0,2810.0,2810.0,2810.0,2810.0, 2810.0 /)
    ea     = (/2000.0,2000.0,2000.0,5000.0,2000.0,3600.0,  50.0,2000.0,2000.0, 2000.0 /)
    sfctmp = (/ 295.0, 295.0, 295.0, 312.0, 295.0, 295.0, 295.0, 295.0, 295.0,  295.0 /)
    sfcprs = (/97000.0,97000.0,97000.0,97000.0,97000.0,97000.0,97000.0,97000.0,97000.0,97000.0/)
    fveg   = (/  0.78,  0.78,  0.78,  0.78,  0.78,  0.78,  0.78,  0.78,  0.78,   0.78 /)
    o2     = (/20460.0,20460.0,20460.0,20460.0,20460.0,20460.0,20460.0,20460.0,20460.0,20460.0/)
    co2    = (/  38.0,  38.0,  38.0,  38.0,  38.0,  38.0,  38.0,  38.0,  38.0,   38.0 /)
    igs    = (/   1.0,   1.0,   1.0,   1.0,   1.0,   1.0,   1.0,   1.0,   0.0,    1.0 /)
    btran  = (/  0.80,  0.80,  0.80,  0.80,  0.80,  0.80,  0.80,   0.0,  0.80,   0.80 /)
    rb     = (/  22.0,  22.0,  22.0,  22.0,  22.0,  22.0,  22.0,  22.0,  22.0,   22.0 /)

    call set_parameters(p, 20.0, 60.0, 2.E3, 9.0, 0.06)

    do i = 1, NC
      write(cs,'("stomata",I2.2)') i
      call emitr('stomata', cs, 'in', 'APAR',   apar(i))
      call emitr('stomata', cs, 'in', 'FOLN',   foln(i))
      call emitr('stomata', cs, 'in', 'TV',     tv(i))
      call emitr('stomata', cs, 'in', 'EI',     ei(i))
      call emitr('stomata', cs, 'in', 'EA',     ea(i))
      call emitr('stomata', cs, 'in', 'SFCTMP', sfctmp(i))
      call emitr('stomata', cs, 'in', 'SFCPRS', sfcprs(i))
      call emitr('stomata', cs, 'in', 'FVEG',   fveg(i))
      call emitr('stomata', cs, 'in', 'O2',     o2(i))
      call emitr('stomata', cs, 'in', 'CO2',    co2(i))
      call emitr('stomata', cs, 'in', 'IGS',    igs(i))
      call emitr('stomata', cs, 'in', 'BTRAN',  btran(i))
      call emitr('stomata', cs, 'in', 'RB',     rb(i))
      call emitr('stomata', cs, 'in', 'MPE',    1.0E-6)
      call emitr('stomata', cs, 'in', 'P_C3PSN',  p%C3PSN)
      call emitr('stomata', cs, 'in', 'P_KC25',   p%KC25)
      call emitr('stomata', cs, 'in', 'P_AKC',    p%AKC)
      call emitr('stomata', cs, 'in', 'P_KO25',   p%KO25)
      call emitr('stomata', cs, 'in', 'P_AKO',    p%AKO)
      call emitr('stomata', cs, 'in', 'P_AVCMX',  p%AVCMX)
      call emitr('stomata', cs, 'in', 'P_VCMX25', p%VCMX25)
      call emitr('stomata', cs, 'in', 'P_BP',     p%BP)
      call emitr('stomata', cs, 'in', 'P_MP',     p%MP)
      call emitr('stomata', cs, 'in', 'P_QE25',   p%QE25)
      call emitr('stomata', cs, 'in', 'P_FOLNMX', p%FOLNMX)
      call STOMATA(p, 1, 1.0E-6, apar(i), foln(i), 1, 1, &
                   tv(i), ei(i), ea(i), sfctmp(i), sfcprs(i), fveg(i), &
                   o2(i), co2(i), igs(i), btran(i), rb(i), &
                   rs, psn)
      call emitr('stomata', cs, 'out', 'RS',  rs)
      call emitr('stomata', cs, 'out', 'PSN', psn)
    end do
  end subroutine do_stomata

! ---------------------------------------------------------------------------
! VEGE_FLUX branch map:
!   v01 daytime unstable, warm canopy, no snow -> TV>TFRZ EVC limiter (CANLIQ),
!                                                 loop1 runs to convergence exit
!   v02 cold canopy and ground                 -> T<=0 legs of both ESAT tests,
!                                                 TV<=TFRZ EVC limiter (CANICE)
!   v03 nocturnal stable                       -> negative H, stable SFCDIF1 leg
!   v04 deep snow with TG > TFRZ               -> OPT_STC=1 TG reset block
!   v05 shallow snow (SNOWH<=0.05), TG>TFRZ    -> reset block NOT taken
!   v06 near-neutral, converges at ITER=5      -> LITER latch then exit loop1
!   v07 very small FV                          -> CAH2 < 1.0E-5 leg of the 2 m
!                                                 diagnostic
!   v08 sparse canopy, dry, hot                -> EVC limiter binds, TR large
!   v09 two-layer snowpack (ISNOW=-2)          -> DF/DZSNSO/STC read at index -1
!   v10 three-layer snowpack (ISNOW=-3)        -> DF/DZSNSO/STC read at index -2
  subroutine do_vegeflux()
    integer, parameter :: NSNOW = 3, NSOIL = 4, NC = 10
    type(noahmp_parameters) :: p
    integer :: i, k, isnow
    real :: dzsnso(-NSNOW+1:NSOIL), stc(-NSNOW+1:NSOIL), df(-NSNOW+1:NSOIL)
    real :: sh2o(1:NSOIL), gecros1d(1:60)
    real :: dt, sav, sag, lwdn, ur, uu, vv, sfctmp, thair, qair, eair, rhoair
    real :: snowh, vai, gammav, gammag, fwet, laisun, laisha, cwp, zlvl, zpd
    real :: z0m, fveg, z0mg, emv, emg, canliq, fsno, canice, rssun, rssha
    real :: rsurf, latheav, latheag, parsun, parsha, igs, foln, co2air, o2air
    real :: btran, sfcprs, rhsur, q2, pahv, pahg, eah, tah, tv, tg, cm, ch
    real :: dx, dz8w, tauxv, tauyv, irg, irc, shg, shc, evg, evc, tr, gh
    real :: t2mv, psnsun, psnsha, canhs, qc, qsfc, psfc, q2v, cah2, chleaf, chuc
    real :: julian, swdown, prcp, fb, fsr
    character(len=24) :: cs
    logical :: veg

    call set_parameters(p, 20.0, 60.0, 2.E3, 9.0, 0.06)

    do i = 1, NC
      write(cs,'("vegeflux",I2.2)') i

      ! ---- common base state -------------------------------------------
      isnow   = 0
      dt      = 30.0
      sav     = 320.0
      sag     = 90.0
      lwdn    = 355.0
      ur      = 4.2
      uu      = 3.4
      vv      = -2.5
      sfctmp  = 295.0
      thair   = 296.4
      qair    = 0.0102
      eair    = 1580.0
      rhoair  = 1.1745
      snowh   = 0.0
      vai     = 3.5
      gammav  = 65.0
      gammag  = 65.0
      fwet    = 0.12
      laisun  = 1.4
      laisha  = 2.1
      cwp     = 3.0
      zlvl    = 30.0
      zpd     = 6.70
      z0m     = 1.00
      fveg    = 0.78
      z0mg    = 0.01
      emv     = 0.95
      emg     = 0.96
      canliq  = 0.35
      fsno    = 0.0
      canice  = 0.0
      rssun   = 200.0
      rssha   = 400.0
      rsurf   = 90.0
      latheav = 2.5104E06
      latheag = 2.5104E06
      parsun  = 260.0
      parsha  = 60.0
      igs     = 1.0
      foln    = 1.0
      co2air  = 38.0
      o2air   = 20460.0
      btran   = 0.80
      sfcprs  = 97000.0
      rhsur   = 0.92
      q2      = 0.0102
      pahv    = 0.0
      pahg    = 0.0
      eah     = 1600.0
      tah     = 295.5
      tv      = 297.0
      tg      = 296.0
      cm      = 0.02
      ch      = 0.02
      dx      = 3000.0
      dz8w    = 60.0
      qc      = 0.0
      qsfc    = 0.0102
      psfc    = 97000.0
      julian  = 180.0
      swdown  = 700.0
      prcp    = 0.0
      fb      = 0.0
      fsr     = 0.0
      veg     = .true.
      gecros1d = 0.0
      do k = -NSNOW+1, NSOIL
        dzsnso(k) = 0.10
        stc(k)    = 293.0
        df(k)     = 1.35
      end do
      dzsnso(1) = 0.10; dzsnso(2) = 0.30; dzsnso(3) = 0.60; dzsnso(4) = 1.00
      sh2o = 0.25

      ! ---- per-case perturbations ---------------------------------------
      select case (i)
      case (2)   ! cold canopy and ground: both ESAT T<=0 legs, CANICE limiter
        sfctmp = 262.0; thair = 263.2; qair = 0.0018; eair = 285.0
        rhoair = 1.325; tah = 262.5; tv = 265.0; tg = 263.0
        stc = 264.0; eah = 300.0; canliq = 0.0; canice = 0.60
        latheav = 2.8440E06; latheag = 2.8440E06; gammav = 74.0; gammag = 74.0
        sav = 120.0; sag = 40.0; lwdn = 250.0; rhsur = 0.98
      case (3)   ! nocturnal stable
        sav = 0.0; sag = 0.0; lwdn = 300.0; swdown = 0.0
        parsun = 0.0; parsha = 0.0; sfctmp = 288.0; thair = 289.2
        tah = 287.0; tv = 285.5; tg = 285.0; stc = 288.0; ur = 1.6
      case (4)   ! deep snow, TG above freezing -> OPT_STC=1 reset block
        snowh = 0.35; fsno = 0.95; isnow = -1
        dzsnso(0) = 0.35; stc(0) = 272.0; df(0) = 0.30
        tg = 274.0; tv = 274.5; tah = 273.5; z0mg = 0.002
      case (5)   ! shallow snow: reset block must NOT fire
        snowh = 0.02; fsno = 0.20
        tg = 274.0; tv = 274.5; tah = 273.5
      case (6)   ! near-neutral: loop1 latches LITER and exits early
        sav = 40.0; sag = 12.0; sfctmp = 293.0; thair = 293.2
        tah = 293.0; tv = 293.05; tg = 293.02; stc = 293.0; ur = 3.0
      case (7)   ! UR at the floor ATM imposes (UR = MAX(SQRT(UU^2+VV^2), 1.0),
                 ! module_sf_noahmplsm.F line 2057) with the smallest MPTABLE
                 ! Z0MVT: the weakest-exchange state WRF can actually present.
                 ! This does NOT reach the CAH2 < 1.0E-5 leg of the 2 m
                 ! diagnostic, and nothing can: CAH2 = FV*VKC/(TMPCH2-FH2) with
                 ! FV >= UR*VKC/(TMPCM-FM), FM,FH2 >= -5 (MOZ,MOZ2 <= 1), so
                 ! CAH2 >= UR*VKC^2/((TMPCM+5)*(TMPCH2+5)) ~ 8E-4 at Z0M=0.001,
                 ! two orders above the threshold.  The branch is asserted
                 ! unreachable rather than bound.
        ur = 1.0; uu = 0.71; vv = -0.71; z0m = 0.001; z0mg = 0.001
      case (9)   ! two-layer snowpack: the ISNOW+1 = -1 layer is the one read
        snowh = 0.62; fsno = 0.98; isnow = -2
        dzsnso(-1) = 0.22; dzsnso(0) = 0.40
        stc(-1) = 268.5;   stc(0) = 271.2
        df(-1) = 0.24;     df(0) = 0.31
        sfctmp = 268.0; thair = 269.1; qair = 0.0026; eair = 350.0
        rhoair = 1.30; tah = 268.6; tv = 269.4; tg = 269.0
        canliq = 0.0; canice = 0.45; z0mg = 0.002
        latheav = 2.8440E06; latheag = 2.8440E06; gammav = 74.0; gammag = 74.0
      case (10)  ! three-layer snowpack: the ISNOW+1 = -2 layer is the one read
        snowh = 1.05; fsno = 1.0; isnow = -3
        dzsnso(-2) = 0.18; dzsnso(-1) = 0.35; dzsnso(0) = 0.52
        stc(-2) = 261.0;   stc(-1) = 265.4;   stc(0) = 270.1
        df(-2) = 0.19;     df(-1) = 0.26;     df(0) = 0.33
        sfctmp = 260.0; thair = 261.0; qair = 0.0012; eair = 190.0
        rhoair = 1.35; tah = 260.8; tv = 262.0; tg = 261.5
        canliq = 0.0; canice = 0.80; z0mg = 0.0015
        latheav = 2.8440E06; latheag = 2.8440E06; gammav = 74.0; gammag = 74.0
        sav = 60.0; sag = 18.0; lwdn = 215.0
      case (8)   ! sparse hot canopy, dry air: EVC limiter binds hard
        fveg = 0.22; vai = 0.6; laisun = 0.35; laisha = 0.25
        sfctmp = 308.0; thair = 309.4; qair = 0.004; eair = 640.0
        rhoair = 1.10; tv = 313.0; tg = 316.0; tah = 309.0; eah = 700.0
        canliq = 0.02; rsurf = 2500.0; rhsur = 0.35; stc = 305.0
      end select

      ! ---- record inputs -------------------------------------------------
      call emiti('vegeflux', cs, 'in', 'ISNOW',  isnow)
      call emiti('vegeflux', cs, 'in', 'NSNOW',  NSNOW)
      call emiti('vegeflux', cs, 'in', 'NSOIL',  NSOIL)
      call emiti('vegeflux', cs, 'in', 'VEG',    merge(1, 0, veg))
      call emitr('vegeflux', cs, 'in', 'DT',     dt)
      call emitr('vegeflux', cs, 'in', 'SAV',    sav)
      call emitr('vegeflux', cs, 'in', 'SAG',    sag)
      call emitr('vegeflux', cs, 'in', 'LWDN',   lwdn)
      call emitr('vegeflux', cs, 'in', 'UR',     ur)
      call emitr('vegeflux', cs, 'in', 'UU',     uu)
      call emitr('vegeflux', cs, 'in', 'VV',     vv)
      call emitr('vegeflux', cs, 'in', 'SFCTMP', sfctmp)
      call emitr('vegeflux', cs, 'in', 'THAIR',  thair)
      call emitr('vegeflux', cs, 'in', 'QAIR',   qair)
      call emitr('vegeflux', cs, 'in', 'EAIR',   eair)
      call emitr('vegeflux', cs, 'in', 'RHOAIR', rhoair)
      call emitr('vegeflux', cs, 'in', 'SNOWH',  snowh)
      call emitr('vegeflux', cs, 'in', 'VAI',    vai)
      call emitr('vegeflux', cs, 'in', 'GAMMAV', gammav)
      call emitr('vegeflux', cs, 'in', 'GAMMAG', gammag)
      call emitr('vegeflux', cs, 'in', 'FWET',   fwet)
      call emitr('vegeflux', cs, 'in', 'LAISUN', laisun)
      call emitr('vegeflux', cs, 'in', 'LAISHA', laisha)
      call emitr('vegeflux', cs, 'in', 'CWP',    cwp)
      do k = -NSNOW+1, NSOIL
        call emitr_idx('vegeflux', cs, 'in', 'DZSNSO', k, dzsnso(k))
        call emitr_idx('vegeflux', cs, 'in', 'STC',    k, stc(k))
        call emitr_idx('vegeflux', cs, 'in', 'DF',     k, df(k))
      end do
      call emitr('vegeflux', cs, 'in', 'ZLVL',   zlvl)
      call emitr('vegeflux', cs, 'in', 'ZPD',    zpd)
      call emitr('vegeflux', cs, 'in', 'Z0M',    z0m)
      call emitr('vegeflux', cs, 'in', 'FVEG',   fveg)
      call emitr('vegeflux', cs, 'in', 'Z0MG',   z0mg)
      call emitr('vegeflux', cs, 'in', 'EMV',    emv)
      call emitr('vegeflux', cs, 'in', 'EMG',    emg)
      call emitr('vegeflux', cs, 'in', 'CANLIQ', canliq)
      call emitr('vegeflux', cs, 'in', 'FSNO',   fsno)
      call emitr('vegeflux', cs, 'in', 'CANICE', canice)
      call emitr('vegeflux', cs, 'in', 'RSSUN',  rssun)
      call emitr('vegeflux', cs, 'in', 'RSSHA',  rssha)
      call emitr('vegeflux', cs, 'in', 'RSURF',  rsurf)
      call emitr('vegeflux', cs, 'in', 'LATHEAV',latheav)
      call emitr('vegeflux', cs, 'in', 'LATHEAG',latheag)
      call emitr('vegeflux', cs, 'in', 'PARSUN', parsun)
      call emitr('vegeflux', cs, 'in', 'PARSHA', parsha)
      call emitr('vegeflux', cs, 'in', 'IGS',    igs)
      call emitr('vegeflux', cs, 'in', 'FOLN',   foln)
      call emitr('vegeflux', cs, 'in', 'CO2AIR', co2air)
      call emitr('vegeflux', cs, 'in', 'O2AIR',  o2air)
      call emitr('vegeflux', cs, 'in', 'BTRAN',  btran)
      call emitr('vegeflux', cs, 'in', 'SFCPRS', sfcprs)
      call emitr('vegeflux', cs, 'in', 'RHSUR',  rhsur)
      call emitr('vegeflux', cs, 'in', 'Q2',     q2)
      call emitr('vegeflux', cs, 'in', 'PAHV',   pahv)
      call emitr('vegeflux', cs, 'in', 'PAHG',   pahg)
      call emitr('vegeflux', cs, 'in', 'EAH',    eah)
      call emitr('vegeflux', cs, 'in', 'TAH',    tah)
      call emitr('vegeflux', cs, 'in', 'TV',     tv)
      call emitr('vegeflux', cs, 'in', 'TG',     tg)
      call emitr('vegeflux', cs, 'in', 'CM',     cm)
      call emitr('vegeflux', cs, 'in', 'CH',     ch)
      call emitr('vegeflux', cs, 'in', 'DX',     dx)
      call emitr('vegeflux', cs, 'in', 'DZ8W',   dz8w)
      call emitr('vegeflux', cs, 'in', 'QC',     qc)
      call emitr('vegeflux', cs, 'in', 'QSFC',   qsfc)
      call emitr('vegeflux', cs, 'in', 'PSFC',   psfc)
      do k = 1, NSOIL
        call emitr_idx('vegeflux', cs, 'in', 'SH2O', k, sh2o(k))
      end do
      call emitr('vegeflux', cs, 'in', 'JULIAN', julian)
      call emitr('vegeflux', cs, 'in', 'SWDOWN', swdown)
      call emitr('vegeflux', cs, 'in', 'PRCP',   prcp)
      call emitr('vegeflux', cs, 'in', 'FB',     fb)
      call emitr('vegeflux', cs, 'in', 'FSR',    fsr)
      call emitr('vegeflux', cs, 'in', 'P_HVT',    p%HVT)
      call emitr('vegeflux', cs, 'in', 'P_DLEAF',  p%DLEAF)
      call emitr('vegeflux', cs, 'in', 'P_CBIOM',  p%CBIOM)
      call emitr('vegeflux', cs, 'in', 'P_C3PSN',  p%C3PSN)
      call emitr('vegeflux', cs, 'in', 'P_KC25',   p%KC25)
      call emitr('vegeflux', cs, 'in', 'P_AKC',    p%AKC)
      call emitr('vegeflux', cs, 'in', 'P_KO25',   p%KO25)
      call emitr('vegeflux', cs, 'in', 'P_AKO',    p%AKO)
      call emitr('vegeflux', cs, 'in', 'P_AVCMX',  p%AVCMX)
      call emitr('vegeflux', cs, 'in', 'P_VCMX25', p%VCMX25)
      call emitr('vegeflux', cs, 'in', 'P_BP',     p%BP)
      call emitr('vegeflux', cs, 'in', 'P_MP',     p%MP)
      call emitr('vegeflux', cs, 'in', 'P_QE25',   p%QE25)
      call emitr('vegeflux', cs, 'in', 'P_FOLNMX', p%FOLNMX)

      call VEGE_FLUX(p, NSNOW, NSOIL, isnow, 1, veg, &
                     dt, sav, sag, lwdn, ur, &
                     uu, vv, sfctmp, thair, qair, &
                     eair, rhoair, snowh, vai, gammav, gammag, &
                     fwet, laisun, laisha, cwp, dzsnso, &
                     zlvl, zpd, z0m, fveg, &
                     z0mg, emv, emg, canliq, fsno, &
                     canice, stc, df, rssun, rssha, &
                     rsurf, latheav, latheag, parsun, parsha, igs, &
                     foln, co2air, o2air, btran, sfcprs, &
                     rhsur, 1, 1, q2, pahv, pahg, &
                     eah, tah, tv, tg, cm, &
                     ch, dx, dz8w, &
                     tauxv, tauyv, irg, irc, shg, &
                     shc, evg, evc, tr, gh, &
                     t2mv, psnsun, psnsha, canhs, &
                     qc, qsfc, psfc, &
                     q2v, cah2, chleaf, chuc, &
                     sh2o, julian, swdown, prcp, fb, fsr, gecros1d)

      call emitr('vegeflux', cs, 'out', 'EAH',    eah)
      call emitr('vegeflux', cs, 'out', 'TAH',    tah)
      call emitr('vegeflux', cs, 'out', 'TV',     tv)
      call emitr('vegeflux', cs, 'out', 'TG',     tg)
      call emitr('vegeflux', cs, 'out', 'CM',     cm)
      call emitr('vegeflux', cs, 'out', 'CH',     ch)
      call emitr('vegeflux', cs, 'out', 'TAUXV',  tauxv)
      call emitr('vegeflux', cs, 'out', 'TAUYV',  tauyv)
      call emitr('vegeflux', cs, 'out', 'IRG',    irg)
      call emitr('vegeflux', cs, 'out', 'IRC',    irc)
      call emitr('vegeflux', cs, 'out', 'SHG',    shg)
      call emitr('vegeflux', cs, 'out', 'SHC',    shc)
      call emitr('vegeflux', cs, 'out', 'EVG',    evg)
      call emitr('vegeflux', cs, 'out', 'EVC',    evc)
      call emitr('vegeflux', cs, 'out', 'TR',     tr)
      call emitr('vegeflux', cs, 'out', 'GH',     gh)
      call emitr('vegeflux', cs, 'out', 'T2MV',   t2mv)
      call emitr('vegeflux', cs, 'out', 'PSNSUN', psnsun)
      call emitr('vegeflux', cs, 'out', 'PSNSHA', psnsha)
      call emitr('vegeflux', cs, 'out', 'CANHS',  canhs)
      call emitr('vegeflux', cs, 'out', 'QSFC',   qsfc)
      call emitr('vegeflux', cs, 'out', 'Q2V',    q2v)
      call emitr('vegeflux', cs, 'out', 'CAH2',   cah2)
      call emitr('vegeflux', cs, 'out', 'CHLEAF', chleaf)
      call emitr('vegeflux', cs, 'out', 'CHUC',   chuc)
      call emitr('vegeflux', cs, 'out', 'RSSUN',  rssun)
      call emitr('vegeflux', cs, 'out', 'RSSHA',  rssha)
      call emitr('vegeflux', cs, 'out', 'SAV',    sav)
      call emitr('vegeflux', cs, 'out', 'SAG',    sag)
      call emitr('vegeflux', cs, 'out', 'FSR',    fsr)
    end do
  end subroutine do_vegeflux

end program run_vegeflux
