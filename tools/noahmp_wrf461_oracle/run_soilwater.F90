! Per-leaf oracle for the Noah-MP soil-water group of WRF v4.6.1
! MODULE_SF_NOAHMPLSM: CANWATER, INFIL, SRT, SSTEP and their driver SOILWATER.
!
! The module source is compiled exactly as it ships apart from the
! accessibility lift produced by visibility_patch_leaves.py, whose text diff is
! nothing but `private` -> `public ` and whose object code is proven identical
! to the pristine compile.  No expression in the physics is touched.
!
! Pinned option identity (WRF Registry defaults), asserted at run time against
! the module's own variables rather than assumed:
!
!   dveg=4 opt_crs=1 opt_btr=1 opt_run=3 opt_sfc=1 opt_frz=1 opt_inf=1
!   opt_rad=3 opt_alb=2 opt_snf=1 opt_tbot=2 opt_stc=1 opt_rsf=1 opt_soil=1
!   opt_pedo=1 opt_crop=0 opt_irr=0 opt_irrm=0 opt_infdv=0 opt_tdrn=0
!
! Dead under that identity and NOT exercised here:
!   opt_run=3  kills GROUNDWATER, SHALLOWWATERTABLE, ZWTEQ, COMPUTE_VIC_SURFRUNOFF,
!              COMPUTE_XAJ_SURFRUNOFF and DYNAMIC_VIC, together with SOILWATER's
!              OPT_RUN 1/2/4/5/6/7/8 blocks (7350-7419, 7442-7462), SRT's
!              OPT_RUN 1/2/4/5 drainage forms (7801-7818) and SSTEP's water
!              table block (7923-7947).
!   opt_inf=1  kills SRT's WDFCND2 loop (7781-7787); WDFCND2 stays live through
!              INFIL at 7703.
!   opt_tdrn=0 kills TILE_DRAIN and TILE_HOOGHOUDT (7521-7529).
!   opt_irr=0  kills the five irrigation routines; none is called from this group.
!
! Fixture convention
! ------------------
! Long format, one row per scalar: leaf,case,stage,field,index,dtype,value,bits.
! Every case emits its complete entry state and its complete exit state, so a
! row is reproducible from the CSV alone and any argument a leaf does not touch
! is visibly identical across the two stages.  Reals carry both a round-tripping
! decimal form and the raw IEEE binary32 bit pattern; the consumer compares bit
! patterns, so no parsing tolerance exists anywhere.
!
! Two INTENT(OUT) aliasing hazards, pinned rather than assumed
! ------------------------------------------------------------
! * SOILWATER declares RUNSUB INTENT(OUT) (7280) but under OPT_RUN==3 the only
!   statement that touches it is `RUNSUB = RUNSUB - XS/DT` (7549), which reads
!   it before it is ever assigned.  gfortran passes scalar dummies by reference,
!   so RUNSUB behaves as INOUT.  WATER always enters with RUNSUB=0.0 (6109), so
!   the forecast path is well defined; the fixture drives a non-zero entry value
!   so the aliasing is measured instead of guessed.
! * INFIL declares PDDUM and RUNSRF INTENT(OUT) (7638-7639) but assigns them
!   only inside `IF (QINSUR > 0.0)` (7655).  The QINSUR<=0 case therefore leaves
!   the caller's values standing; SOILWATER zeroes both at 7318-7319 before its
!   first call, so again the forecast path is defined.  Both entry values are
!   recorded.
!
! Soil-parameter set
! ------------------
! Every per-layer parameter varies with layer so that a port which indexes the
! wrong layer cannot reproduce the fixture.  The dead-branch parameters TIMEAN
! and FSATMX carry non-zero values so their inertness is measured, not vacuous.

module soilwater_oracle

  use module_sf_noahmplsm, only : noahmp_parameters,                        &
                                  SOILWATER, INFIL, SRT, SSTEP, CANWATER,   &
                                  TFRZ, HVAP, HSUB, HFUS, CWAT, CICE,       &
                                  DENH2O, DENICE,                           &
                                  OPT_RUN, OPT_INF, OPT_TDRN, OPT_IRR,      &
                                  OPT_INFDV, DVEG, OPT_CROP
  implicit none

  integer, parameter :: NSNOW = 3
  integer, parameter :: NSOIL = 4
  integer, parameter :: CSV   = 21
  integer, parameter :: PRB   = 22
  integer, parameter :: ILOC  = 3
  integer, parameter :: JLOC  = 5

  ! Soil interfaces -0.10/-0.40/-1.00/-2.00 m: the pinned Registry stack.
  real, parameter :: ZSOIL_PIN(NSOIL)  = (/ -0.10, -0.40, -1.00, -2.00 /)
  real, parameter :: DZSOIL_PIN(NSOIL) = (/  0.10,  0.30,  0.60,  1.00 /)

  real, parameter :: SENTINEL = -999.0

contains

! ---------------------------------------------------------------------------
! CSV emitters
! ---------------------------------------------------------------------------

  function itoa(i) result(s)
    integer, intent(in) :: i
    character(len=16) :: s
    write(s, '(I0)') i
  end function itoa

  function rtoa(v) result(s)
    real, intent(in) :: v
    character(len=20) :: s
    write(s, '(ES16.9E2)') v
    s = adjustl(s)
  end function rtoa

  function hex8(v) result(s)
    real, intent(in) :: v
    character(len=8) :: s
    integer :: b
    b = transfer(v, b)
    write(s, '(Z8.8)') b
  end function hex8

  subroutine emit_r(leaf, cname, stage, field, idx, v)
    character(len=*), intent(in) :: leaf, cname, stage, field
    integer, intent(in) :: idx
    real,    intent(in) :: v
    write(CSV, '(A)') leaf//','//cname//','//stage//','//field//','// &
         trim(itoa(idx))//',real,'//trim(rtoa(v))//','//hex8(v)
  end subroutine emit_r

  subroutine emit_i(leaf, cname, stage, field, idx, v)
    character(len=*), intent(in) :: leaf, cname, stage, field
    integer, intent(in) :: idx
    integer, intent(in) :: v
    write(CSV, '(A)') leaf//','//cname//','//stage//','//field//','// &
         trim(itoa(idx))//',int,'//trim(itoa(v))//','
  end subroutine emit_i

  subroutine emit_ra(leaf, cname, stage, field, lo, hi, a)
    character(len=*), intent(in) :: leaf, cname, stage, field
    integer, intent(in) :: lo, hi
    real,    intent(in) :: a(lo:hi)
    integer :: k
    do k = lo, hi
       call emit_r(leaf, cname, stage, field, k, a(k))
    end do
  end subroutine emit_ra

  subroutine emit_probe(name, v)
    character(len=*), intent(in) :: name
    real, intent(in) :: v
    write(PRB, '(A)') name//','//trim(rtoa(v))//','//hex8(v)
  end subroutine emit_probe

! ---------------------------------------------------------------------------
! Parameter handle.  Every per-layer entry varies with layer.
! ---------------------------------------------------------------------------

  subroutine base_parameters(p)
    type(noahmp_parameters), intent(out) :: p
    p%SMCMAX = (/ 0.434, 0.439, 0.445, 0.451 /)
    p%SMCWLT = (/ 0.047, 0.051, 0.055, 0.059 /)
    p%BEXP   = (/ 4.74,  4.90,  5.05,  5.20  /)
    p%DKSAT  = (/ 5.23e-6, 4.90e-6, 4.55e-6, 4.20e-6 /)
    p%DWSAT  = (/ 3.58e-5, 3.40e-5, 3.20e-5, 3.05e-5 /)
    p%KDT    = 3.137
    p%FRZX   = 0.1519
    p%SLOPE  = 0.1
    p%CH2OP  = 0.1
    p%URBAN_FLAG = .false.
    ! Dead under opt_run=3; non-zero so the inertness claim is measured.
    p%TIMEAN = 10.5
    p%FSATMX = 0.38
  end subroutine base_parameters

  subroutine emit_parameters(leaf, cname, p, nlayer)
    character(len=*), intent(in) :: leaf, cname
    type(noahmp_parameters), intent(in) :: p
    integer, intent(in) :: nlayer
    call emit_ra(leaf, cname, 'param', 'smcmax', 1, nlayer, p%SMCMAX(1:nlayer))
    call emit_ra(leaf, cname, 'param', 'smcwlt', 1, nlayer, p%SMCWLT(1:nlayer))
    call emit_ra(leaf, cname, 'param', 'bexp',   1, nlayer, p%BEXP(1:nlayer))
    call emit_ra(leaf, cname, 'param', 'dksat',  1, nlayer, p%DKSAT(1:nlayer))
    call emit_ra(leaf, cname, 'param', 'dwsat',  1, nlayer, p%DWSAT(1:nlayer))
    call emit_r (leaf, cname, 'param', 'kdt',    0, p%KDT)
    call emit_r (leaf, cname, 'param', 'frzx',   0, p%FRZX)
    call emit_r (leaf, cname, 'param', 'slope',  0, p%SLOPE)
    call emit_r (leaf, cname, 'param', 'ch2op',  0, p%CH2OP)
    call emit_r (leaf, cname, 'param', 'timean', 0, p%TIMEAN)
    call emit_r (leaf, cname, 'param', 'fsatmx', 0, p%FSATMX)
    if (p%URBAN_FLAG) then
       call emit_i(leaf, cname, 'param', 'urban_flag', 0, 1)
    else
       call emit_i(leaf, cname, 'param', 'urban_flag', 0, 0)
    end if
  end subroutine emit_parameters

! ===========================================================================
! CANWATER -- module_sf_noahmplsm.F:6265-6394
!
! Reads exactly one parameter component, CH2OP (6323).  VEGTYP, ILOC, JLOC and
! TG are declared INTENT(IN) and never referenced in the body; they are varied
! per case so their inertness is measured.  CANLIQ, CANICE and TV are INOUT.
! ===========================================================================

  subroutine case_canwater(cname, p, vegtyp, dt, fcev, fctr, elai, esai,     &
                           tg, fveg, bdfall, frozen_canopy,                  &
                           canliq0, canice0, tv0)
    character(len=*), intent(in) :: cname
    type(noahmp_parameters), intent(in) :: p
    integer, intent(in) :: vegtyp
    real,    intent(in) :: dt, fcev, fctr, elai, esai, tg, fveg, bdfall
    logical, intent(in) :: frozen_canopy
    real,    intent(in) :: canliq0, canice0, tv0

    real :: canliq, canice, tv
    real :: cmc, ecan, etran, fwet
    real :: qsubc, qfroc, qfrzc, qmeltc, qevac, qdewc
    character(len=*), parameter :: L = 'canwater'

    canliq = canliq0; canice = canice0; tv = tv0
    cmc = SENTINEL; ecan = SENTINEL; etran = SENTINEL; fwet = SENTINEL
    qsubc = SENTINEL; qfroc = SENTINEL; qfrzc = SENTINEL
    qmeltc = SENTINEL; qevac = SENTINEL; qdewc = SENTINEL

    call emit_parameters(L, cname, p, NSOIL)
    call emit_i(L, cname, 'input', 'vegtyp', 0, vegtyp)
    call emit_i(L, cname, 'input', 'iloc',   0, ILOC)
    call emit_i(L, cname, 'input', 'jloc',   0, JLOC)
    if (frozen_canopy) then
       call emit_i(L, cname, 'input', 'frozen_canopy', 0, 1)
    else
       call emit_i(L, cname, 'input', 'frozen_canopy', 0, 0)
    end if
    call emit_r(L, cname, 'input', 'dt',     0, dt)
    call emit_r(L, cname, 'input', 'fcev',   0, fcev)
    call emit_r(L, cname, 'input', 'fctr',   0, fctr)
    call emit_r(L, cname, 'input', 'elai',   0, elai)
    call emit_r(L, cname, 'input', 'esai',   0, esai)
    call emit_r(L, cname, 'input', 'tg',     0, tg)
    call emit_r(L, cname, 'input', 'fveg',   0, fveg)
    call emit_r(L, cname, 'input', 'bdfall', 0, bdfall)
    call emit_r(L, cname, 'input', 'canliq', 0, canliq)
    call emit_r(L, cname, 'input', 'canice', 0, canice)
    call emit_r(L, cname, 'input', 'tv',     0, tv)

    call CANWATER (p, vegtyp, dt, fcev, fctr, elai, esai, tg, fveg,          &
                   ILOC, JLOC, bdfall, frozen_canopy,                        &
                   canliq, canice, tv,                                       &
                   cmc, ecan, etran, fwet,                                   &
                   qsubc, qfroc, qfrzc, qmeltc, qevac, qdewc)

    call emit_r(L, cname, 'output', 'canliq', 0, canliq)
    call emit_r(L, cname, 'output', 'canice', 0, canice)
    call emit_r(L, cname, 'output', 'tv',     0, tv)
    call emit_r(L, cname, 'output', 'cmc',    0, cmc)
    call emit_r(L, cname, 'output', 'ecan',   0, ecan)
    call emit_r(L, cname, 'output', 'etran',  0, etran)
    call emit_r(L, cname, 'output', 'fwet',   0, fwet)
    call emit_r(L, cname, 'output', 'qsubc',  0, qsubc)
    call emit_r(L, cname, 'output', 'qfroc',  0, qfroc)
    call emit_r(L, cname, 'output', 'qfrzc',  0, qfrzc)
    call emit_r(L, cname, 'output', 'qmeltc', 0, qmeltc)
    call emit_r(L, cname, 'output', 'qevac',  0, qevac)
    call emit_r(L, cname, 'output', 'qdewc',  0, qdewc)
  end subroutine case_canwater

! ===========================================================================
! INFIL -- module_sf_noahmplsm.F:7616-7712
!
! Parameters: SMCMAX(1) and SMCWLT(K) (7659-7674), KDT (7676), FRZX (7688) plus
! WDFCND2's BEXP(1)/DWSAT(1)/DKSAT(1)/SMCMAX(1) through the K=1 call at 7703.
! ZSOIL(K) is read for every K.  PDDUM/RUNSRF are INTENT(OUT) but untouched
! when QINSUR<=0, so both entry values are pinned.
! ===========================================================================

  subroutine case_infil(cname, p, dt, zsoil, sh2o, sice, sicemax, qinsur,    &
                        pddum0, runsrf0)
    character(len=*), intent(in) :: cname
    type(noahmp_parameters), intent(in) :: p
    real, intent(in) :: dt, zsoil(NSOIL), sh2o(NSOIL), sice(NSOIL)
    real, intent(in) :: sicemax, qinsur, pddum0, runsrf0

    real :: pddum, runsrf
    character(len=*), parameter :: L = 'infil'

    pddum = pddum0; runsrf = runsrf0

    call emit_parameters(L, cname, p, NSOIL)
    call emit_i (L, cname, 'input', 'nsoil',   0, NSOIL)
    call emit_r (L, cname, 'input', 'dt',      0, dt)
    call emit_ra(L, cname, 'input', 'zsoil',   1, NSOIL, zsoil)
    call emit_ra(L, cname, 'input', 'sh2o',    1, NSOIL, sh2o)
    call emit_ra(L, cname, 'input', 'sice',    1, NSOIL, sice)
    call emit_r (L, cname, 'input', 'sicemax', 0, sicemax)
    call emit_r (L, cname, 'input', 'qinsur',  0, qinsur)
    call emit_r (L, cname, 'input', 'pddum',   0, pddum)
    call emit_r (L, cname, 'input', 'runsrf',  0, runsrf)

    call INFIL (p, NSOIL, dt, zsoil, sh2o, sice, sicemax, qinsur,            &
                pddum, runsrf)

    call emit_r(L, cname, 'output', 'pddum',  0, pddum)
    call emit_r(L, cname, 'output', 'runsrf', 0, runsrf)
  end subroutine case_infil

! ===========================================================================
! SRT -- module_sf_noahmplsm.F:7716-7846
!
! Parameter: SLOPE (7806), plus WDFCND1's BEXP(K)/DWSAT(K)/DKSAT(K)/SMCMAX(K)
! through the OPT_INF==1 loop at 7775-7778.
!
! Inert under the pinned identity, every one varied per case:
!   DT      declared 7739, never referenced in the body
!   ILOC    7731, JLOC 7732, never referenced
!   SH2O    read only in the OPT_INF==2 loop (7783) and at 7789
!   SICEMAX read only in the OPT_INF==2 loop (7783)
!   ZWT     read only in the OPT_RUN==5 block (7812)
!   FCRMAX  read only in the OPT_RUN==4 block (7810)
!   SMCWTD  read only in the OPT_RUN==5 lines 7779 and 7789
! ===========================================================================

  subroutine case_srt(cname, p, dt, zsoil, pddum, etrani, qseva, sh2o, smc,  &
                      zwt, fcr, sicemax, fcrmax, smcwtd, iloc, jloc)
    character(len=*), intent(in) :: cname
    type(noahmp_parameters), intent(in) :: p
    real, intent(in) :: dt, zsoil(NSOIL), pddum, etrani(NSOIL), qseva
    real, intent(in) :: sh2o(NSOIL), smc(NSOIL), zwt, fcr(NSOIL)
    real, intent(in) :: sicemax, fcrmax, smcwtd
    integer, intent(in) :: iloc, jloc

    real :: rhstt(NSOIL), ai(NSOIL), bi(NSOIL), ci(NSOIL), wcnd(NSOIL)
    real :: qdrain
    character(len=*), parameter :: L = 'srt'

    rhstt = SENTINEL; ai = SENTINEL; bi = SENTINEL; ci = SENTINEL
    wcnd = SENTINEL; qdrain = SENTINEL

    call emit_parameters(L, cname, p, NSOIL)
    call emit_i (L, cname, 'input', 'nsoil',   0, NSOIL)
    call emit_i (L, cname, 'input', 'iloc',    0, iloc)
    call emit_i (L, cname, 'input', 'jloc',    0, jloc)
    call emit_r (L, cname, 'input', 'dt',      0, dt)
    call emit_ra(L, cname, 'input', 'zsoil',   1, NSOIL, zsoil)
    call emit_r (L, cname, 'input', 'pddum',   0, pddum)
    call emit_ra(L, cname, 'input', 'etrani',  1, NSOIL, etrani)
    call emit_r (L, cname, 'input', 'qseva',   0, qseva)
    call emit_ra(L, cname, 'input', 'sh2o',    1, NSOIL, sh2o)
    call emit_ra(L, cname, 'input', 'smc',     1, NSOIL, smc)
    call emit_r (L, cname, 'input', 'zwt',     0, zwt)
    call emit_ra(L, cname, 'input', 'fcr',     1, NSOIL, fcr)
    call emit_r (L, cname, 'input', 'sicemax', 0, sicemax)
    call emit_r (L, cname, 'input', 'fcrmax',  0, fcrmax)
    call emit_r (L, cname, 'input', 'smcwtd',  0, smcwtd)

    call SRT (p, NSOIL, zsoil, dt, pddum, etrani, qseva, sh2o, smc, zwt,     &
              fcr, sicemax, fcrmax, iloc, jloc, smcwtd,                      &
              rhstt, ai, bi, ci, qdrain, wcnd)

    call emit_ra(L, cname, 'output', 'rhstt',  1, NSOIL, rhstt)
    call emit_ra(L, cname, 'output', 'ai',     1, NSOIL, ai)
    call emit_ra(L, cname, 'output', 'bi',     1, NSOIL, bi)
    call emit_ra(L, cname, 'output', 'ci',     1, NSOIL, ci)
    call emit_r (L, cname, 'output', 'qdrain', 0, qdrain)
    call emit_ra(L, cname, 'output', 'wcnd',   1, NSOIL, wcnd)
  end subroutine case_srt

! ===========================================================================
! SSTEP -- module_sf_noahmplsm.F:7850-7973
!
! Parameter: SMCMAX(K) (7952, 7959, 7963, 7969).  Calls ROSR12 with NTOP=1,
! NSOIL=NSOIL and NSNOW=0 (7913).
!
! Inert under the pinned identity, every one varied per case:
!   ILOC/JLOC        7859-7860, never referenced
!   ZSOIL            read only in the OPT_RUN==5 block (7927)
!   ZWT              read only in the OPT_RUN==5 block (7927)
!   DZSNSO(-2:0)     the snow slots; the body reads only DZSNSO(1:NSOIL)
!   SMCWTD/DEEPRECH  written only inside the OPT_RUN==5 block (7929-7944)
!   QDRAIN           read/written only inside the OPT_RUN==5 block
!   SMC on entry     unconditionally overwritten by `SMC = SH2O + SICE` (7971)
! ===========================================================================

  subroutine case_sstep(cname, p, dt, zsoil, dzsnso, sice, zwt, sh2o0, smc0, &
                        ai0, bi0, ci0, rhstt0, smcwtd0, qdrain0, deeprech0,  &
                        iloc, jloc)
    character(len=*), intent(in) :: cname
    type(noahmp_parameters), intent(in) :: p
    real, intent(in) :: dt, zsoil(NSOIL), dzsnso(-NSNOW+1:NSOIL)
    real, intent(in) :: sice(NSOIL), zwt
    real, intent(in) :: sh2o0(NSOIL), smc0(NSOIL)
    real, intent(in) :: ai0(NSOIL), bi0(NSOIL), ci0(NSOIL), rhstt0(NSOIL)
    real, intent(in) :: smcwtd0, qdrain0, deeprech0
    integer, intent(in) :: iloc, jloc

    real :: sh2o(NSOIL), smc(NSOIL), ai(NSOIL), bi(NSOIL), ci(NSOIL)
    real :: rhstt(NSOIL), smcwtd, qdrain, deeprech, wplus
    character(len=*), parameter :: L = 'sstep'

    sh2o = sh2o0; smc = smc0; ai = ai0; bi = bi0; ci = ci0; rhstt = rhstt0
    smcwtd = smcwtd0; qdrain = qdrain0; deeprech = deeprech0
    wplus = SENTINEL

    call emit_parameters(L, cname, p, NSOIL)
    call emit_i (L, cname, 'input', 'nsoil',    0, NSOIL)
    call emit_i (L, cname, 'input', 'nsnow',    0, NSNOW)
    call emit_i (L, cname, 'input', 'iloc',     0, iloc)
    call emit_i (L, cname, 'input', 'jloc',     0, jloc)
    call emit_r (L, cname, 'input', 'dt',       0, dt)
    call emit_ra(L, cname, 'input', 'zsoil',    1, NSOIL, zsoil)
    call emit_ra(L, cname, 'input', 'dzsnso',   -NSNOW+1, NSOIL, dzsnso)
    call emit_ra(L, cname, 'input', 'sice',     1, NSOIL, sice)
    call emit_r (L, cname, 'input', 'zwt',      0, zwt)
    call emit_ra(L, cname, 'input', 'sh2o',     1, NSOIL, sh2o)
    call emit_ra(L, cname, 'input', 'smc',      1, NSOIL, smc)
    call emit_ra(L, cname, 'input', 'ai',       1, NSOIL, ai)
    call emit_ra(L, cname, 'input', 'bi',       1, NSOIL, bi)
    call emit_ra(L, cname, 'input', 'ci',       1, NSOIL, ci)
    call emit_ra(L, cname, 'input', 'rhstt',    1, NSOIL, rhstt)
    call emit_r (L, cname, 'input', 'smcwtd',   0, smcwtd)
    call emit_r (L, cname, 'input', 'qdrain',   0, qdrain)
    call emit_r (L, cname, 'input', 'deeprech', 0, deeprech)

    call SSTEP (p, NSOIL, NSNOW, dt, zsoil, dzsnso, sice, iloc, jloc, zwt,   &
                sh2o, smc, ai, bi, ci, rhstt, smcwtd, qdrain, deeprech,      &
                wplus)

    call emit_ra(L, cname, 'output', 'sh2o',     1, NSOIL, sh2o)
    call emit_ra(L, cname, 'output', 'smc',      1, NSOIL, smc)
    call emit_ra(L, cname, 'output', 'ai',       1, NSOIL, ai)
    call emit_ra(L, cname, 'output', 'bi',       1, NSOIL, bi)
    call emit_ra(L, cname, 'output', 'ci',       1, NSOIL, ci)
    call emit_ra(L, cname, 'output', 'rhstt',    1, NSOIL, rhstt)
    call emit_r (L, cname, 'output', 'smcwtd',   0, smcwtd)
    call emit_r (L, cname, 'output', 'qdrain',   0, qdrain)
    call emit_r (L, cname, 'output', 'deeprech', 0, deeprech)
    call emit_r (L, cname, 'output', 'wplus',    0, wplus)
  end subroutine case_sstep

! ===========================================================================
! SOILWATER -- module_sf_noahmplsm.F:7234-7556
!
! Parameters: SMCMAX(K) (7326, 7334, 7343) and URBAN_FLAG (7361), plus
! everything INFIL, SRT and SSTEP consume.
!
! Inert under the pinned identity, every one varied per case:
!   ILOC/JLOC   7253-7254, only forwarded to SRT/SSTEP where they are inert too
!   VEGTYP      7268, declared INOUT and never referenced
!   DX          7267, read only by TILE_HOOGHOUDT (7527)
!   TDFRACMP    7266, read only in the OPT_TDRN gates at 7521 and 7524
!   ZWT         7272, read in the OPT_RUN 1/2/5 blocks and forwarded to SRT
!   SMCWTD      7273, forwarded to SRT/SSTEP where it is OPT_RUN==5 only
!   DEEPRECH    7274, written only in the OPT_RUN==5 branch of SSTEP and at 7544
!   QTLDRN      7275, written only by the tile-drain routines
!   TIMEAN      read only at 7356 (OPT_RUN==2)
!   FSATMX      read only at 7367, 7378, 7389 (OPT_RUN 1/5/2)
!
! RUNSUB is INTENT(OUT) yet read at 7549 before assignment; see the header.
! ===========================================================================

  subroutine case_soilwater(cname, p, dt, zsoil, dzsnso, qinsur, qseva,      &
                            etrani, sice, tdfracmp, dx, sh2o0, smc0, zwt0,   &
                            vegtyp0, smcwtd0, deeprech0, qtldrn0, runsub0,   &
                            iloc, jloc)
    character(len=*), intent(in) :: cname
    type(noahmp_parameters), intent(in) :: p
    real, intent(in) :: dt, zsoil(NSOIL), dzsnso(-NSNOW+1:NSOIL)
    real, intent(in) :: qinsur, qseva, etrani(NSOIL), sice(NSOIL)
    real, intent(in) :: tdfracmp, dx
    real, intent(in) :: sh2o0(NSOIL), smc0(NSOIL), zwt0
    integer, intent(in) :: vegtyp0
    real, intent(in) :: smcwtd0, deeprech0, qtldrn0, runsub0
    integer, intent(in) :: iloc, jloc

    real :: sh2o(NSOIL), smc(NSOIL), zwt, smcwtd, deeprech, qtldrn
    real :: runsrf, qdrain, runsub, wcnd(NSOIL), fcrmax
    integer :: vegtyp
    ! Only for the NITER branch probe below.
    real    :: psh2o(NSOIL), psicemax, ppddum, prunsrf, pepore
    integer :: pk, pniter
    character(len=*), parameter :: L = 'soilwater'

    sh2o = sh2o0; smc = smc0; zwt = zwt0; vegtyp = vegtyp0
    smcwtd = smcwtd0; deeprech = deeprech0; qtldrn = qtldrn0
    runsub = runsub0
    runsrf = SENTINEL; qdrain = SENTINEL; wcnd = SENTINEL; fcrmax = SENTINEL

    call emit_parameters(L, cname, p, NSOIL)
    call emit_i (L, cname, 'input', 'nsoil',    0, NSOIL)
    call emit_i (L, cname, 'input', 'nsnow',    0, NSNOW)
    call emit_i (L, cname, 'input', 'iloc',     0, iloc)
    call emit_i (L, cname, 'input', 'jloc',     0, jloc)
    call emit_i (L, cname, 'input', 'vegtyp',   0, vegtyp)
    call emit_r (L, cname, 'input', 'dt',       0, dt)
    call emit_ra(L, cname, 'input', 'zsoil',    1, NSOIL, zsoil)
    call emit_ra(L, cname, 'input', 'dzsnso',   -NSNOW+1, NSOIL, dzsnso)
    call emit_r (L, cname, 'input', 'qinsur',   0, qinsur)
    call emit_r (L, cname, 'input', 'qseva',    0, qseva)
    call emit_ra(L, cname, 'input', 'etrani',   1, NSOIL, etrani)
    call emit_ra(L, cname, 'input', 'sice',     1, NSOIL, sice)
    call emit_r (L, cname, 'input', 'tdfracmp', 0, tdfracmp)
    call emit_r (L, cname, 'input', 'dx',       0, dx)
    call emit_ra(L, cname, 'input', 'sh2o',     1, NSOIL, sh2o)
    call emit_ra(L, cname, 'input', 'smc',      1, NSOIL, smc)
    call emit_r (L, cname, 'input', 'zwt',      0, zwt)
    call emit_r (L, cname, 'input', 'smcwtd',   0, smcwtd)
    call emit_r (L, cname, 'input', 'deeprech', 0, deeprech)
    call emit_r (L, cname, 'input', 'qtldrn',   0, qtldrn)
    call emit_r (L, cname, 'input', 'runsub',   0, runsub)

    ! ---- NITER branch probe ------------------------------------------------
    ! SOILWATER's iteration count (7440-7445) is a local, so branch coverage of
    ! `PDDUM*DT > DZSNSO(1)*SMCMAX(1)` cannot be read off the outputs.  It is
    ! re-derived here from the inputs alone: the RSAT clamp (7324-7329) and
    ! SICEMAX (7341-7347) are MIN/MAX over the inputs, and PDDUM then comes from
    ! the module's own INFIL -- no physics is duplicated.  Emitted as its own
    ! stage so a consumer can assert which branch each case takes.
    psh2o = sh2o
    do pk = 1, NSOIL
       pepore   = max(1.0e-4, p%SMCMAX(pk) - sice(pk))
       psh2o(pk) = min(pepore, psh2o(pk))
    end do
    psicemax = 0.0
    do pk = 1, NSOIL
       if (sice(pk) > psicemax) psicemax = sice(pk)
    end do
    ppddum = 0.0
    prunsrf = 0.0
    ! These two emits are load-bearing, not diagnostics.  INFIL declares PDDUM
    ! and RUNSRF INTENT(OUT), so with the explicit interface in scope gfortran
    ! at -O2 treats the two stores above as dead and removes them -- and INFIL
    ! leaves both untouched when QINSUR <= 0, which would let the previous
    ! case's stack value stand.  Reading them here forces the stores to be
    ! live.  (This is the same hazard SOILWATER itself carries at 7318-7319.)
    call emit_r(L, cname, 'probe', 'niter_pddum_in',  0, ppddum)
    call emit_r(L, cname, 'probe', 'niter_runsrf_in', 0, prunsrf)
    call INFIL (p, NSOIL, dt, zsoil, psh2o, sice, psicemax, qinsur,          &
                ppddum, prunsrf)
    pniter = 3
    if (ppddum * dt > dzsnso(1) * p%SMCMAX(1)) pniter = pniter * 2
    call emit_r(L, cname, 'probe', 'niter_pddum',   0, ppddum)
    call emit_r(L, cname, 'probe', 'niter_lhs',     0, ppddum * dt)
    call emit_r(L, cname, 'probe', 'niter_rhs',     0, dzsnso(1) * p%SMCMAX(1))
    call emit_r(L, cname, 'probe', 'probe_sicemax', 0, psicemax)
    call emit_i(L, cname, 'probe', 'niter',         0, pniter)

    call SOILWATER (p, NSOIL, NSNOW, dt, zsoil, dzsnso,                      &
                    qinsur, qseva, etrani, sice, iloc, jloc,                 &
                    tdfracmp, dx,                                            &
                    sh2o, smc, zwt, vegtyp,                                  &
                    smcwtd, deeprech,                                        &
                    runsrf, qdrain, runsub, wcnd, fcrmax, qtldrn)

    call emit_ra(L, cname, 'output', 'sh2o',     1, NSOIL, sh2o)
    call emit_ra(L, cname, 'output', 'smc',      1, NSOIL, smc)
    call emit_r (L, cname, 'output', 'zwt',      0, zwt)
    call emit_i (L, cname, 'output', 'vegtyp',   0, vegtyp)
    call emit_r (L, cname, 'output', 'smcwtd',   0, smcwtd)
    call emit_r (L, cname, 'output', 'deeprech', 0, deeprech)
    call emit_r (L, cname, 'output', 'qtldrn',   0, qtldrn)
    call emit_r (L, cname, 'output', 'runsrf',   0, runsrf)
    call emit_r (L, cname, 'output', 'qdrain',   0, qdrain)
    call emit_r (L, cname, 'output', 'runsub',   0, runsub)
    call emit_ra(L, cname, 'output', 'wcnd',     1, NSOIL, wcnd)
    call emit_r (L, cname, 'output', 'fcrmax',   0, fcrmax)
  end subroutine case_soilwater

! ===========================================================================
! libm / constant probe
!
! SOILWATER's FCR expression (7335-7336) contains EXP(-A) with A a REAL
! PARAMETER, which gfortran folds at compile time with MPFR, and
! EXP(-A*(1.-FICE)) which lowers to a runtime expf@plt call.  The two are not
! the same function, so both are emitted and the port pins whichever the
! fixture shows.  The sweeps below give the CUDA gate its reference values on
! exactly the domains this group uses.
! ===========================================================================

  subroutine dump_probe()
    real, parameter :: A = 4.0
    real :: avar, x, y
    integer :: i

    call emit_probe('tfrz            ', TFRZ)
    call emit_probe('hvap            ', HVAP)
    call emit_probe('hsub            ', HSUB)
    call emit_probe('hfus            ', HFUS)
    call emit_probe('cwat            ', CWAT)
    call emit_probe('cice            ', CICE)
    call emit_probe('denh2o          ', DENH2O)
    call emit_probe('denice          ', DENICE)

    call emit_probe('exp_neg_a_folded', EXP(-A))
    avar = -4.0
    call emit_probe('exp_neg_a_runtim', EXP(avar))
    call emit_probe('one_minus_expnega', 1.0 - EXP(-A))

    ! expf over the FCR argument range -A*(1-FICE), FICE in [0,1].
    do i = 0, 256
       x = -4.0 * (1.0 - real(i) / 256.0)
       y = EXP(x)
       call emit_probe('expf_fcr_'//trim(itoa(i)), y)
    end do

    ! powf over CANWATER's FWET**0.667, FWET in [0,1].
    do i = 0, 256
       x = real(i) / 256.0
       y = x ** 0.667
       call emit_probe('powf_fwet_'//trim(itoa(i)), y)
    end do

    ! powf over WDFCND's FACTR**EXPON on the shapes this group produces.
    do i = 0, 128
       x = 0.01 + (1.0 - 0.01) * real(i) / 128.0
       y = x ** 6.74
       call emit_probe('powf_wdf_'//trim(itoa(i)), y)
       y = x ** 12.48
       call emit_probe('powf_wcnd_'//trim(itoa(i)), y)
    end do
  end subroutine dump_probe

end module soilwater_oracle


! ===========================================================================
program run_soilwater

  use module_sf_noahmplsm, only : noahmp_options, noahmp_parameters,         &
                                  OPT_RUN, OPT_INF, OPT_TDRN, OPT_IRR,       &
                                  OPT_INFDV, DVEG, OPT_CROP
  use soilwater_oracle
  implicit none

  character(len=1024) :: csv_path, probe_path
  type(noahmp_parameters) :: p, pu, ps, pz
  real :: zsoil(NSOIL), dzsnso(-NSNOW+1:NSOIL)
  real :: sh2o(NSOIL), smc(NSOIL), sice(NSOIL), etrani(NSOIL)
  real :: fcr(NSOIL), ai(NSOIL), bi(NSOIL), ci(NSOIL), rhstt(NSOIL)
  integer :: k

  call get_command_argument(1, csv_path)
  call get_command_argument(2, probe_path)
  if (len_trim(csv_path) == 0 .or. len_trim(probe_path) == 0) then
     write(*, '(A)') 'usage: run_soilwater SOILWATER.csv PROBE.csv'
     error stop 2
  end if

  ! The pinned WRF Registry default identity.
  call noahmp_options(4, 1, 1, 3, 1, 1, &
                      1, 3, 2, 1, 2, 1, &
                      1, 1, 1, 0, 0, 0, &
                      0, 0)

  ! Assert it against the module's own variables rather than assuming it.
  if (OPT_RUN   /= 3) error stop 'opt_run is not 3'
  if (OPT_INF   /= 1) error stop 'opt_inf is not 1'
  if (OPT_TDRN  /= 0) error stop 'opt_tdrn is not 0'
  if (OPT_IRR   /= 0) error stop 'opt_irr is not 0'
  if (OPT_INFDV /= 0) error stop 'opt_infdv is not 0'
  if (OPT_CROP  /= 0) error stop 'opt_crop is not 0'
  if (DVEG      /= 4) error stop 'dveg is not 4'

  open(CSV, file=trim(csv_path), status='replace', action='write')
  write(CSV, '(A)') 'leaf,case,stage,field,index,dtype,value,bits'
  open(PRB, file=trim(probe_path), status='replace', action='write')
  write(PRB, '(A)') 'name,value,bits'

  call dump_probe()

  call base_parameters(p)
  call base_parameters(pu)
  pu%URBAN_FLAG = .true.

  ! A coarse, highly conductive set.  It exists so that SOILWATER's NITER
  ! doubling at 7443 is reachable: with the base set INFIL's INFMAX is capped
  ! by DD*VAL/DT, which is an order of magnitude below DZSNSO(1)*SMCMAX(1)/DT
  ! for every admissible QINSUR, so the `NITER = NITER*2` branch is dead under
  ! the base set alone.  DKSAT/DWSAT/BEXP/SMCMAX/SMCWLT here are WRF SOILPARM's
  ! sand row; KDT follows REFKDT*DKSAT/REFDK with REFKDT=3, REFDK=2.0e-6.
  call base_parameters(ps)
  ps%SMCMAX = (/ 0.339, 0.339, 0.339, 0.339 /)
  ps%SMCWLT = (/ 0.010, 0.010, 0.010, 0.010 /)
  ps%BEXP   = (/ 2.79,  2.79,  2.79,  2.79  /)
  ps%DKSAT  = (/ 1.7639e-4, 1.7639e-4, 1.7639e-4, 1.7639e-4 /)
  ps%DWSAT  = (/ 6.0808e-5, 6.0808e-5, 6.0808e-5, 6.0808e-5 /)
  ps%KDT    = 264.585

  ! The base set with a small FRZX.  At DICE just above the 1.0E-2 test the
  ! base FRZX=0.1519 makes ACRT=45.6, so EXP(-ACRT)*SUM underflows to 0 and
  ! FCR is exactly 1.0 on both sides of the branch -- the boundary is real but
  ! numerically inert there.  FRZX=6.5E-3 puts ACRT near 2 at the same DICE,
  ! which is what makes the straddle observable.  The physical-FRZX cases
  ! infil_frozen and infil_heavy_ice exercise the same block at DICE 0.105 and
  ! 0.409, where it is observable without any parameter change.
  call base_parameters(pz)
  pz%FRZX = 6.5e-3

  zsoil = ZSOIL_PIN
  dzsnso(-2) = 0.05; dzsnso(-1) = 0.09; dzsnso(0) = 0.13
  dzsnso(1:NSOIL) = DZSOIL_PIN

! ---------------------------------------------------------------------------
! CANWATER
! ---------------------------------------------------------------------------
  ! Unfrozen canopy, evaporation and transpiration both positive, no ice, TV
  ! above freezing.  QEVAC is rate-limited (FCEV/HVAP < CANLIQ/DT), FWET comes
  ! from the liquid branch, neither phase-change block fires.
  call case_canwater('canw_unfrozen_evap', p, 11, 1800.0, 45.0, 30.0,        &
                     3.0, 0.5, 285.0, 0.8, 120.0, .false.,                   &
                     0.40, 0.0, 285.5)

  ! Same but CANLIQ small, so MIN(CANLIQ/DT, QEVAC) picks the canopy store.
  call case_canwater('canw_unfrozen_evap_masslimited', p, 7, 1800.0,         &
                     4.5e4, 30.0, 3.0, 0.5, 285.0, 0.8, 120.0, .false.,      &
                     0.02, 0.0, 285.5)

  ! FCEV < 0: dew.  QDEWC = |FCEV|/HVAP, QEVAC = 0, CANLIQ grows.
  call case_canwater('canw_unfrozen_dew', p, 12, 1800.0, -22.0, 10.0,        &
                     2.5, 0.4, 281.0, 0.7, 110.0, .false.,                   &
                     0.05, 0.0, 280.0)

  ! Frozen canopy, sublimation, ice present and >= liquid so FWET takes the ice
  ! branch; TV below freezing so no melt and CANLIQ is 0 so no freeze.
  call case_canwater('canw_frozen_sublim', p, 14, 1800.0, 35.0, 20.0,        &
                     2.0, 0.6, 264.0, 0.9, 90.0, .true.,                     &
                     0.0, 1.20, 265.0)

  ! MIN(CANICE/DT, QSUBC) picks the canopy store.
  call case_canwater('canw_frozen_sublim_masslimited', p, 14, 1800.0,        &
                     5.0e4, 20.0, 2.0, 0.6, 264.0, 0.9, 90.0, .true.,        &
                     0.0, 0.03, 265.0)

  ! Frozen canopy, FCEV < 0: frost.  QFROC = |FCEV|/HSUB, CANICE grows.
  call case_canwater('canw_frozen_frost', p, 14, 1800.0, -18.0, 12.0,        &
                     2.0, 0.6, 262.0, 0.9, 85.0, .true.,                     &
                     0.0, 0.10, 263.0)

  ! CANLIQ straddling the `CANLIQ <= 1.0E-06` test at 6353.  FCEV and FCTR are
  ! zero so QDEWC and QEVAC are zero and CANLIQ + 0.0*DT is exact; TV is at
  ! TFRZ so neither phase-change block fires.  1.0E-06 is 0x358637BD.
  call case_canwater('canw_canliq_at_limit', p, 5, 1800.0, 0.0, 0.0,         &
                     1.5, 0.3, 288.0, 0.6, 130.0, .false.,                   &
                     9.999999975e-07, 0.0, 273.16)

  call case_canwater('canw_canliq_just_above_limit', p, 5, 1800.0, 0.0, 0.0, &
                     1.5, 0.3, 288.0, 0.6, 130.0, .false.,                   &
                     1.000000111e-06, 0.0, 273.16)

  ! CANICE straddling the `CANICE <= 1.0E-6` test at 6363, same construction.
  call case_canwater('canw_canice_at_limit', p, 5, 1800.0, 0.0, 0.0,         &
                     1.5, 0.3, 268.0, 0.6, 130.0, .true.,                    &
                     0.0, 9.999999975e-07, 273.16)

  call case_canwater('canw_canice_just_above_limit', p, 5, 1800.0, 0.0, 0.0, &
                     1.5, 0.3, 268.0, 0.6, 130.0, .true.,                    &
                     0.0, 1.000000111e-06, 273.16)

  ! LSAI = 0 so MAXLIQ = 0 and MAX(MAXLIQ, 1.0E-06) picks the floor; FWET then
  ! saturates and MIN(FWET,1.0) clamps, so 1.0**0.667 is exercised.
  call case_canwater('canw_fwet_clamped_zero_lsai', p, 16, 1800.0, 0.0,      &
                     0.0, 0.0, 0.0, 290.0, 0.5, 100.0, .false.,              &
                     0.25, 0.0, 290.0)

  ! Melt: CANICE > 1e-6 and TV > TFRZ, energy-limited (the MIN picks the
  ! temperature side), and TV is relaxed toward TFRZ by FWET.
  call case_canwater('canw_melt_energy_limited', p, 11, 1800.0, 5.0, 3.0,    &
                     3.0, 0.5, 275.0, 0.8, 120.0, .false.,                   &
                     0.10, 2.50, 273.30)

  ! Melt: the MIN picks CANICE/DT instead.
  call case_canwater('canw_melt_mass_limited', p, 11, 1800.0, 5.0, 3.0,      &
                     3.0, 0.5, 280.0, 0.8, 120.0, .false.,                   &
                     0.10, 0.02, 283.0)

  ! Freeze: CANLIQ > 1e-6 and TV < TFRZ, energy-limited.
  call case_canwater('canw_freeze_energy_limited', p, 11, 1800.0, 4.0, 2.0,  &
                     3.0, 0.5, 271.0, 0.8, 120.0, .false.,                   &
                     2.00, 0.05, 273.05)

  ! Freeze: the MIN picks CANLIQ/DT instead.
  call case_canwater('canw_freeze_mass_limited', p, 11, 1800.0, 4.0, 2.0,    &
                     3.0, 0.5, 265.0, 0.8, 120.0, .false.,                   &
                     0.03, 0.05, 268.0)

  ! TV exactly at TFRZ: neither phase-change block fires (both tests are strict).
  call case_canwater('canw_tv_at_tfrz', p, 11, 1800.0, 6.0, 2.0,             &
                     3.0, 0.5, 273.16, 0.8, 120.0, .false.,                  &
                     0.60, 0.60, 273.16)

  ! CANICE > 0 but < CANLIQ, so FWET still takes the liquid branch.
  call case_canwater('canw_ice_below_liq', p, 11, 1800.0, 3.0, 1.0,          &
                     3.0, 0.5, 272.0, 0.8, 120.0, .false.,                   &
                     1.40, 0.30, 273.16)

  ! A 60 s step, so DT is discriminated independently of the fluxes.
  call case_canwater('canw_short_step', p, 11, 60.0, 45.0, 30.0,             &
                     3.0, 0.5, 285.0, 0.8, 120.0, .false.,                   &
                     0.40, 0.0, 285.5)

  ! LSAI = 0 with CANICE one ULP above the 1.0E-6 zeroing test.  MAXSNO is
  ! exactly 0, so `MAX(MAXSNO,1.0E-06)` at 6360 takes the floor and FWET lands
  ! at 1.0000001, one ULP clear of the MIN.  This is the only shape in which
  ! the value of that floor is observable: everywhere else the 6355 zeroing has
  ! already forced CANICE to 0 or to strictly more than the floor, and the
  ! MIN(FWET,1.0) erases the difference.  It also puts CANICE exactly on the
  ! melt test at 6372, so TV collapses to TFRZ if and only if both constants
  ! are what WRF says they are.
  call case_canwater('canw_fwet_floor_ice_melt', p, 11, 1800.0, 0.0, 0.0,    &
                     0.0, 0.0, 285.0, 0.5, 120.0, .false.,                   &
                     0.0, 1.000000111e-06, 280.0)

  ! The liquid mirror: MAXLIQ = 0 so 6362's floor is selected, and CANLIQ sits
  ! exactly on the freeze test at 6379.
  call case_canwater('canw_fwet_floor_liq_freeze', p, 11, 1800.0, 0.0, 0.0,  &
                     0.0, 0.0, 265.0, 0.5, 120.0, .false.,                   &
                     1.000000111e-06, 0.0, 265.0)

  ! Inertness probe: identical to canw_unfrozen_evap except VEGTYP, ILOC/JLOC
  ! (compiled in) and TG.  Every output must be bit-identical.
  call case_canwater('canw_inert_probe', p, 27, 1800.0, 45.0, 30.0,          &
                     3.0, 0.5, 301.75, 0.8, 120.0, .false.,                  &
                     0.40, 0.0, 285.5)

! ---------------------------------------------------------------------------
! INFIL
! ---------------------------------------------------------------------------
  sh2o = (/ 0.180, 0.210, 0.240, 0.260 /)
  sice = (/ 0.000, 0.000, 0.000, 0.000 /)

  ! QINSUR <= 0: the whole body is skipped and the caller's PDDUM/RUNSRF stand.
  call case_infil('infil_qinsur_zero', p, 1800.0, zsoil, sh2o, sice,         &
                  0.0, 0.0, SENTINEL, 0.5 * SENTINEL)

  ! Unfrozen soil, moderate rain: DICE = 0 so FCR stays 1 and the CVFRZ loop is
  ! skipped; the MAX picks INFMAX and the MIN picks INFMAX, RUNSRF > 0.
  call case_infil('infil_unfrozen_runoff', p, 1800.0, zsoil, sh2o, sice,     &
                  0.0, 3.0e-5, SENTINEL, 0.5 * SENTINEL)

  ! Light rain: MIN(INFMAX, PX/DT) picks PX/DT, so RUNSRF is exactly 0.
  call case_infil('infil_px_limited', p, 1800.0, zsoil, sh2o, sice,          &
                  0.0, 1.0e-8, SENTINEL, 0.5 * SENTINEL)

  ! Nearly saturated column: DD collapses, INFMAX < WCND, so MAX picks WCND.
  sh2o = (/ 0.4300, 0.4350, 0.4400, 0.4460 /)
  call case_infil('infil_wcnd_floor', p, 1800.0, zsoil, sh2o, sice,          &
                  0.0, 5.0e-5, SENTINEL, 0.5 * SENTINEL)

  ! Frozen soil: DICE > 1.0E-2 so ACRT/SUM/FCR run, and SICEMAX > 0 so
  ! WDFCND2's VKWGT branch is taken.
  sh2o = (/ 0.140, 0.170, 0.200, 0.230 /)
  sice = (/ 0.120, 0.090, 0.060, 0.030 /)
  call case_infil('infil_frozen', p, 1800.0, zsoil, sh2o, sice,              &
                  0.120, 4.0e-5, SENTINEL, 0.5 * SENTINEL)

  ! DICE straddling the strict `DICE > 1.0E-2` test at 7687.  With SICE zero
  ! below layer 1, DICE is -(ZSOIL(1)*SICE(1)) and the deeper terms add exact
  ! zeros, so the straddle is a one-ULP pair in SICE(1):
  !   SICE(1) = 0x3DCCCCCC -> DICE = 0x3C23D70A, exactly 1.0E-2, test FALSE
  !   SICE(1) = 0x3DCCCCCD -> DICE = 0x3C23D70B, one ULP above,  test TRUE
  sice = (/ 0.099999994, 0.0, 0.0, 0.0 /)
  call case_infil('infil_dice_at_limit', pz, 1800.0, zsoil, sh2o, sice,      &
                  0.099999994, 4.0e-5, SENTINEL, 0.5 * SENTINEL)

  sice = (/ 0.100000001, 0.0, 0.0, 0.0 /)
  call case_infil('infil_dice_just_above_limit', pz, 1800.0, zsoil, sh2o,    &
                  sice, 0.100000001, 4.0e-5, SENTINEL, 0.5 * SENTINEL)

  ! Heavy ice: ACRT small, FCR strongly reduced.
  sice = (/ 0.280, 0.250, 0.210, 0.180 /)
  sh2o = (/ 0.120, 0.140, 0.160, 0.180 /)
  call case_infil('infil_heavy_ice', p, 1800.0, zsoil, sh2o, sice,           &
                  0.280, 8.0e-5, SENTINEL, 0.5 * SENTINEL)

  ! Dry column, short step: DT1 and KDT are discriminated separately.
  sh2o = (/ 0.060, 0.080, 0.100, 0.120 /)
  sice = 0.0
  call case_infil('infil_dry_short_step', p, 60.0, zsoil, sh2o, sice,        &
                  0.0, 2.0e-5, SENTINEL, 0.5 * SENTINEL)

  ! Below wilting point in layer 1, so DMAX(1) exceeds its own capacity term.
  sh2o = (/ 0.020, 0.030, 0.040, 0.050 /)
  call case_infil('infil_below_wilting', p, 1800.0, zsoil, sh2o, sice,       &
                  0.0, 6.0e-5, SENTINEL, 0.5 * SENTINEL)

! ---------------------------------------------------------------------------
! SRT
! ---------------------------------------------------------------------------
  sh2o   = (/ 0.180, 0.210, 0.240, 0.260 /)
  smc    = (/ 0.190, 0.220, 0.250, 0.270 /)
  etrani = (/ 2.0e-8, 1.4e-8, 6.0e-9, 1.0e-9 /)
  fcr    = 0.0

  ! Baseline: OPT_INF==1 (WDFCND1 on SMC), OPT_RUN==3 drainage SLOPE*WCND(4).
  call case_srt('srt_wet_baseline', p, 1800.0, zsoil, 8.0e-6, etrani,        &
                1.5e-8, sh2o, smc, -2.5, fcr, 0.0, 0.0, 0.30, ILOC, JLOC)

  ! Inertness probe: DT, ILOC, JLOC, SH2O, ZWT, SICEMAX, FCRMAX and SMCWTD all
  ! changed, everything SRT actually reads held fixed.  Every output must be
  ! bit-identical to srt_wet_baseline.
  call case_srt('srt_inert_probe', p, 71.5, zsoil, 8.0e-6, etrani,           &
                1.5e-8, (/ 0.031, 0.317, 0.113, 0.409 /), smc, 4.75, fcr,    &
                0.83, 0.61, 0.277, 41, 97)

  ! Frozen soil: FCR > 0 per layer, so WDFCND1's (1 - FCR) factor is live and
  ! varies with layer.
  fcr = (/ 0.62, 0.41, 0.23, 0.07 /)
  call case_srt('srt_frozen_fcr', p, 1800.0, zsoil, 5.0e-6, etrani,          &
                1.0e-8, sh2o, smc, -2.5, fcr, 0.12, 0.62, 0.30, ILOC, JLOC)

  ! Very dry: SMC/SMCMAX below 0.01, so WDFCND1's MAX(0.01, .) floor is taken
  ! in every layer.
  fcr = 0.0
  smc = (/ 0.0035, 0.0040, 0.0040, 0.0040 /)
  call case_srt('srt_factr_floor', p, 1800.0, zsoil, 1.0e-7, etrani,         &
                1.0e-9, sh2o, smc, -2.5, fcr, 0.0, 0.0, 0.30, ILOC, JLOC)

  ! Uniform column: every DSMDZ is exactly zero, so WFLUX is conductivity and
  ! sources only.
  smc = (/ 0.250, 0.250, 0.250, 0.250 /)
  call case_srt('srt_uniform_column', p, 1800.0, zsoil, 0.0, etrani,         &
                0.0, sh2o, smc, -2.5, fcr, 0.0, 0.0, 0.30, ILOC, JLOC)

  ! Inverted profile: DSMDZ changes sign, and a large PDDUM drives WFLUX(1)
  ! strongly negative.
  smc = (/ 0.410, 0.330, 0.260, 0.200 /)
  call case_srt('srt_inverted_wet_top', p, 1800.0, zsoil, 4.0e-5, etrani,    &
                3.0e-8, sh2o, smc, -2.5, fcr, 0.0, 0.0, 0.30, ILOC, JLOC)

  ! Saturated: FACTR = 1 in every layer, so WDF = DWSAT and WCND = DKSAT.
  smc = (/ 0.434, 0.439, 0.445, 0.451 /)
  call case_srt('srt_saturated', p, 1800.0, zsoil, 0.0, etrani,              &
                0.0, sh2o, smc, -2.5, fcr, 0.0, 0.0, 0.30, ILOC, JLOC)

  ! Negative ETRANI (condensation into a layer) and negative QSEVA (dew).
  smc = (/ 0.190, 0.220, 0.250, 0.270 /)
  call case_srt('srt_negative_sources', p, 1800.0, zsoil, 0.0,               &
                (/ -3.0e-8, -1.0e-8, 4.0e-9, 1.0e-9 /),                      &
                -2.0e-8, sh2o, smc, -2.5, fcr, 0.0, 0.0, 0.30, ILOC, JLOC)

! ---------------------------------------------------------------------------
! SSTEP
! ---------------------------------------------------------------------------
  sice   = (/ 0.020, 0.015, 0.010, 0.005 /)
  sh2o   = (/ 0.230, 0.250, 0.270, 0.290 /)
  smc    = (/ 0.250, 0.265, 0.280, 0.295 /)
  ai     = (/  0.0,     -9.0e-6, -3.0e-6, -1.2e-6 /)
  bi     = (/  1.5e-5,   1.9e-5,  6.0e-6,  1.2e-6 /)
  ci     = (/ -1.5e-5,  -1.0e-5, -3.0e-6,  0.0    /)
  rhstt  = (/  4.0e-6,  -8.0e-7, -2.0e-7, -5.0e-8 /)

  ! Baseline: no layer reaches EPORE, so both saturation cascades are no-ops
  ! and WPLUS is exactly 0.
  call case_sstep('sstep_baseline', p, 600.0, zsoil, dzsnso, sice, -2.5,     &
                  sh2o, smc, ai, bi, ci, rhstt, 0.30, 3.0e-7, 0.0,           &
                  ILOC, JLOC)

  ! Inertness probe: DT and the tridiagonal held fixed; ZSOIL, ZWT, the snow
  ! slots of DZSNSO, SMC on entry, SMCWTD, QDRAIN, DEEPRECH, ILOC and JLOC all
  ! changed.  Every output must be bit-identical to sstep_baseline.
  call case_sstep('sstep_inert_probe', p, 600.0,                             &
                  (/ -0.37, -0.91, -1.53, -3.11 /),                          &
                  (/ 0.71, 0.29, 0.53, 0.10, 0.30, 0.60, 1.00 /),            &
                  sice, 7.25, sh2o,                                          &
                  (/ 0.017, 0.629, 0.301, 0.884 /),                          &
                  ai, bi, ci, rhstt, 0.911, -5.5e-6, 0.417, 61, 13)

  ! Deep layer oversaturated on exit: the upward cascade moves WPLUS from
  ! layer 4 into 3, and layer 1 stays below EPORE so the downward pass is off.
  call case_sstep('sstep_saturate_upward', p, 600.0, zsoil, dzsnso, sice,    &
                  -2.5, (/ 0.100, 0.150, 0.200, 0.460 /), smc,               &
                  ai, bi, ci, (/ 0.0, 0.0, 0.0, 0.0 /),                      &
                  0.30, 3.0e-7, 0.0, ILOC, JLOC)

  ! Layer 1 oversaturated: WPLUS > 0 after the upward pass, so the downward
  ! cascade at 7959-7969 runs.
  call case_sstep('sstep_saturate_downward', p, 600.0, zsoil, dzsnso, sice,  &
                  -2.5, (/ 0.480, 0.200, 0.230, 0.250 /), smc,               &
                  ai, bi, ci, (/ 0.0, 0.0, 0.0, 0.0 /),                      &
                  0.30, 3.0e-7, 0.0, ILOC, JLOC)

  ! Every layer oversaturated: the downward cascade spills all the way to
  ! NSOIL, which is the only path that reaches 7967-7969.
  call case_sstep('sstep_saturate_all', p, 600.0, zsoil, dzsnso, sice,       &
                  -2.5, (/ 0.480, 0.470, 0.470, 0.470 /), smc,               &
                  ai, bi, ci, (/ 0.0, 0.0, 0.0, 0.0 /),                      &
                  0.30, 3.0e-7, 0.0, ILOC, JLOC)

  ! SICE above SMCMAX in every layer, so MAX(1.0E-4, SMCMAX-SICE) takes the
  ! floor everywhere.
  call case_sstep('sstep_epore_floor', p, 600.0, zsoil, dzsnso,              &
                  (/ 0.500, 0.500, 0.500, 0.500 /), -2.5,                    &
                  (/ 0.010, 0.008, 0.006, 0.004 /), smc,                     &
                  ai, bi, ci, rhstt, 0.30, 3.0e-7, 0.0, ILOC, JLOC)

  ! SICE above SMCMAX in layers 2-4 only, and little enough water that layer 1
  ! stays under its own EPORE.  The upward cascade therefore takes the 1.0E-4
  ! floor in three layers and the downward cascade never fires, so those three
  ! floors survive into the output instead of being overwritten by the
  ! downward pass at 7963-7969.  Without this case the floor at 7952 is
  ! unobservable: every other saturating case runs both cascades, and the
  ! second one re-clamps with its own copy of the constant.  RHSTT is zero so
  ! ROSR12 contributes exactly nothing and the cascade is isolated.
  call case_sstep('sstep_epore_floor_upward_only', p, 600.0, zsoil, dzsnso,  &
                  (/ 0.020, 0.500, 0.500, 0.500 /), -2.5,                    &
                  (/ 0.050, 0.010, 0.008, 0.006 /), smc,                     &
                  ai, bi, ci, (/ 0.0, 0.0, 0.0, 0.0 /),                      &
                  0.30, 3.0e-7, 0.0, ILOC, JLOC)

  ! A large RHSTT that drives SH2O negative, so ROSR12's result is signed and
  ! the MIN/MAX pairs see negative arguments.
  call case_sstep('sstep_negative_solution', p, 600.0, zsoil, dzsnso, sice,  &
                  -2.5, (/ 0.080, 0.090, 0.100, 0.110 /), smc,               &
                  ai, bi, ci, (/ -6.0e-5, -2.0e-5, -8.0e-6, -3.0e-6 /),      &
                  0.30, 3.0e-7, 0.0, ILOC, JLOC)

  ! A 1800 s step against the same tridiagonal, so DT is discriminated.
  call case_sstep('sstep_long_step', p, 1800.0, zsoil, dzsnso, sice, -2.5,   &
                  sh2o, smc, ai, bi, ci, rhstt, 0.30, 3.0e-7, 0.0,           &
                  ILOC, JLOC)

  ! AI(1) and CI(NSOIL) non-zero.  SRT always emits them as exactly 0.0 (7826
  ! and 7838), and ROSR12 reads neither -- A(NTOP) is untouched and C(NSOIL) is
  ! overwritten at 5565 -- but SSTEP scales both by DT and returns them, so
  ! without this case the fixture cannot tell a port that scales them from one
  ! that drops them.  This is a leaf-level discrimination case; the composed
  ! chain never reaches it.
  call case_sstep('sstep_offdiagonal_ends', p, 600.0, zsoil, dzsnso, sice,   &
                  -2.5, sh2o, smc,                                           &
                  (/ 7.0e-6, -9.0e-6, -3.0e-6, -1.2e-6 /), bi,               &
                  (/ -1.5e-5, -1.0e-5, -3.0e-6, -4.0e-6 /), rhstt,           &
                  0.30, 3.0e-7, 0.0, ILOC, JLOC)

! ---------------------------------------------------------------------------
! SOILWATER
! ---------------------------------------------------------------------------
  sh2o   = (/ 0.180, 0.210, 0.240, 0.260 /)
  smc    = (/ 0.180, 0.210, 0.240, 0.260 /)
  sice   = 0.0
  etrani = (/ 2.0e-8, 1.4e-8, 6.0e-9, 1.0e-9 /)

  ! Moderate rain on unfrozen soil.  NITER stays 3 (PDDUM*DT below
  ! DZSNSO(1)*SMCMAX(1)); RUNSUB enters non-zero so the 7549 aliasing is
  ! measured.
  call case_soilwater('slw_moderate_rain', p, 1800.0, zsoil, dzsnso,         &
                      3.0e-6, 1.5e-8, etrani, sice, 0.0, 3000.0,             &
                      sh2o, smc, -2.5, 11, 0.30, 0.0, 0.0, 7.5, ILOC, JLOC)

  ! Identical except RUNSUB on entry, which isolates the aliasing.
  call case_soilwater('slw_runsub_alias', p, 1800.0, zsoil, dzsnso,          &
                      3.0e-6, 1.5e-8, etrani, sice, 0.0, 3000.0,             &
                      sh2o, smc, -2.5, 11, 0.30, 0.0, 0.0, -3.25,            &
                      ILOC, JLOC)

  ! Inertness probe on slw_moderate_rain: VEGTYP, ILOC/JLOC, TDFRACMP, DX,
  ! ZWT, SMCWTD, DEEPRECH and QTLDRN all changed.  Every output must be
  ! bit-identical.
  call case_soilwater('slw_inert_probe', p, 1800.0, zsoil, dzsnso,           &
                      3.0e-6, 1.5e-8, etrani, sice, 0.9, 250.0,              &
                      sh2o, smc, 8.75, 27, 0.717, 0.331, 0.529, 7.5,         &
                      31, 83)

  ! Heavy rain on the coarse set: PDDUM*DT exceeds DZSNSO(1)*SMCMAX(1), so
  ! NITER doubles to 6 and the SRT/SSTEP pair runs six times per call.
  call case_soilwater('slw_heavy_rain_niter6', ps, 1800.0, zsoil, dzsnso,    &
                      1.0e-4, 1.5e-8, etrani, sice, 0.0, 3000.0,             &
                      (/ 0.150, 0.180, 0.200, 0.220 /),                      &
                      (/ 0.150, 0.180, 0.200, 0.220 /),                      &
                      -2.5, 11, 0.30, 0.0, 0.0, 0.0, ILOC, JLOC)

  ! The same coarse set at NITER = 3, so the doubling is isolated from the
  ! parameter change.
  call case_soilwater('slw_coarse_niter3', ps, 1800.0, zsoil, dzsnso,        &
                      1.0e-6, 1.5e-8, etrani, sice, 0.0, 3000.0,             &
                      (/ 0.150, 0.180, 0.200, 0.220 /),                      &
                      (/ 0.150, 0.180, 0.200, 0.220 /),                      &
                      -2.5, 11, 0.30, 0.0, 0.0, 0.0, ILOC, JLOC)

  ! No precipitation: INFIL is entered once with QINSUR = 0 (a no-op) and
  ! skipped inside the iteration loop, so PDDUM stays 0 for all three passes.
  call case_soilwater('slw_no_precip', p, 1800.0, zsoil, dzsnso,             &
                      0.0, 4.0e-8, etrani, sice, 0.0, 3000.0,                &
                      sh2o, smc, -2.5, 11, 0.30, 0.0, 0.0, 0.0, ILOC, JLOC)

  ! Supersaturated entry state: the 7324-7329 pre-pass accumulates RSAT and
  ! clamps SH2O, which is the only path that makes RSAT non-zero.
  call case_soilwater('slw_supersaturated_entry', p, 1800.0, zsoil, dzsnso,  &
                      2.0e-6, 1.0e-8, etrani, sice, 0.0, 3000.0,             &
                      (/ 0.480, 0.470, 0.300, 0.280 /),                      &
                      (/ 0.480, 0.470, 0.300, 0.280 /),                      &
                      -2.5, 11, 0.30, 0.0, 0.0, 0.0, ILOC, JLOC)

  ! Frozen soil: FICE and the FCR EXP form are live, SICEMAX > 0 so INFIL's
  ! WDFCND2 takes its VKWGT branch, and FCRMAX becomes non-zero.
  call case_soilwater('slw_frozen', p, 1800.0, zsoil, dzsnso,                &
                      4.0e-6, 1.0e-8, etrani,                                &
                      (/ 0.120, 0.090, 0.060, 0.030 /), 0.0, 3000.0,         &
                      (/ 0.140, 0.170, 0.200, 0.230 /),                      &
                      (/ 0.260, 0.260, 0.260, 0.260 /),                      &
                      -2.5, 11, 0.30, 0.0, 0.0, 0.0, ILOC, JLOC)

  ! SICE above SMCMAX: FICE saturates at 1, so FCR = 1 exactly and EPORE takes
  ! its 1.0E-4 floor.
  call case_soilwater('slw_ice_saturated', p, 1800.0, zsoil, dzsnso,         &
                      4.0e-6, 1.0e-8, etrani,                                &
                      (/ 0.500, 0.500, 0.500, 0.500 /), 0.0, 3000.0,         &
                      (/ 0.050, 0.050, 0.050, 0.050 /),                      &
                      (/ 0.550, 0.550, 0.550, 0.550 /),                      &
                      -2.5, 11, 0.30, 0.0, 0.0, 0.0, ILOC, JLOC)

  ! Urban: FCR(1) is overwritten with 0.95 at 7361, which under OPT_INF==1
  ! reaches WDFCND1 for layer 1 inside SRT.
  call case_soilwater('slw_urban', pu, 1800.0, zsoil, dzsnso,                &
                      3.0e-6, 1.5e-8, etrani, sice, 0.0, 3000.0,             &
                      sh2o, smc, -2.5, 13, 0.30, 0.0, 0.0, 0.0, ILOC, JLOC)

  ! Strong evaporative demand out of a dry column drives MLIQ negative in the
  ! upper layers and below WATMIN at the bottom, so the 7527-7552 fixup runs
  ! and RUNSUB is reduced.
  call case_soilwater('slw_watmin_fixup', p, 1800.0, zsoil, dzsnso,          &
                      0.0, 3.0e-6, (/ 4.0e-6, 3.0e-6, 2.0e-6, 1.0e-6 /),     &
                      sice, 0.0, 3000.0,                                     &
                      (/ 0.0080, 0.0060, 0.0040, 0.0020 /),                  &
                      (/ 0.0080, 0.0060, 0.0040, 0.0020 /),                  &
                      -2.5, 11, 0.30, 0.0, 0.0, 2.0, ILOC, JLOC)

  ! A 60 s step, so DT and DTFINE are discriminated from the fluxes.
  call case_soilwater('slw_short_step', p, 60.0, zsoil, dzsnso,              &
                      3.0e-6, 1.5e-8, etrani, sice, 0.0, 3000.0,             &
                      sh2o, smc, -2.5, 11, 0.30, 0.0, 0.0, 0.0, ILOC, JLOC)

  ! Saturated column with heavy rain: SSTEP spills through both cascades on
  ! every one of the six iterations, so RSAT accumulates inside the loop.
  call case_soilwater('slw_saturated_spill', p, 1800.0, zsoil, dzsnso,       &
                      2.0e-4, 0.0, (/ 0.0, 0.0, 0.0, 0.0 /), sice,           &
                      0.0, 3000.0,                                           &
                      (/ 0.430, 0.435, 0.440, 0.448 /),                      &
                      (/ 0.430, 0.435, 0.440, 0.448 /),                      &
                      -2.5, 11, 0.30, 0.0, 0.0, 0.0, ILOC, JLOC)

  close(CSV)
  close(PRB)
  write(*, '(A)') 'run_soilwater: wrote '//trim(csv_path)//' and '//trim(probe_path)

  ! Silence the unused-variable warning for k without perturbing anything.
  k = 0

end program run_soilwater
