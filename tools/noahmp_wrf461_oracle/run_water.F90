! Oracle for the Noah-MP WATER assembly of WRF v4.6.1
! MODULE_SF_NOAHMPLSM: WATER (5954-6261), the driver that sequences CANWATER,
! SNOWWATER and SOILWATER and owns the unit conversions between them.
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
! plus the soil-timestep identity soiltstep=0.0, which module_sf_noahmpdrv.F
! turns into soil_update_steps=1 and calculate_soil=.true. (drivers/wrf,
! 648-676).  Both are module variables, set and asserted here the same way.
!
! Dead under that identity and NOT exercised here
! ------------------------------------------------
!   opt_run=3   kills GROUNDWATER (6225-6231) and SHALLOWWATERTABLE
!               (6242-6250), together with everything they reach --
!               ZWTEQ, COMPUTE_VIC_SURFRUNOFF, COMPUTE_XAJ_SURFRUNOFF and
!               DYNAMIC_VIC -- and it selects `RUNSUB = RUNSUB + QDRAIN`
!               at 6233-6236 as the only live baseflow form.
!   opt_irr=0   kills FLOOD_IRRIGATION (6188-6193) and MICRO_IRRIGATION
!               (6196-6202).  Their gates are on IRAMTFI/IRAMTMI, not on
!               OPT_IRR, so the kill runs through the caller: with OPT_IRR<1
!               TRIGGER_IRRIGATION takes the `OPT_IRR .LT. 1` branch at 9291,
!               sets IRR_ACTIVE=.false. and zeroes IRAMTSI/MI/FI at 9344-9346;
!               NOAHMP_SFLX additionally zeroes all three at 919-923 whenever
!               IRRFRA < parameters%IRR_FRAC.  WATER therefore cannot be
!               reached with IRAMTFI>0 or IRAMTMI>0, and the cases below hold
!               both at 0.0 while varying CROPLU, IRRFRA, MIFAC, FIFAC,
!               IRFIRATE and IRMIRATE so the claim is measured.
!   opt_tdrn=0  kills the tile drain inside SOILWATER; QTLDRN is zeroed at
!               6112 and scaled by DT_soil at 6254, so it is identically 0.
!   WRF_HYDRO   not defined, so sfcheadrt/WATBLED do not exist and the
!               QINSUR term at 6174 is absent.
!
! Three INTENT(OUT) hazards, pinned rather than assumed
! -----------------------------------------------------
! * QIN and QDIS are INTENT(OUT) at 6052-6053 and the ONLY statement that
!   assigns either is inside the OPT_RUN==1 GROUNDWATER call.  Under
!   opt_run=3 they are never written at all.  gfortran passes scalar dummies
!   by reference, so the caller's value stands; NOAHMP_SFLX declares both as
!   uninitialised locals at 671-672 and reads neither after the call, so
!   nothing downstream consumes them.  Every case enters with a distinct
!   non-zero value and both entry and exit are recorded, which turns "never
!   written" into a measured row instead of a claim.
! * QSNSUB and QSNFRO are dummy arguments declared without INTENT in the
!   local block (6086-6087).  They are unconditionally assigned at 6126/6132,
!   so their entry values are dead; the fixture drives them non-zero anyway.
! * SOILWATER's own RUNSUB aliasing (7549 reads before assigning) is invisible
!   from here because WATER always zeroes RUNSUB at 6109 first.  That is the
!   fact that makes the forecast path defined, and it is pinned by driving
!   RUNSUB non-zero on entry to WATER and requiring the exit value not to
!   depend on it.
!
! Fixture convention
! ------------------
! Long format, one row per scalar: leaf,case,stage,field,index,dtype,value,bits.
! Every case emits its complete entry state and its complete exit state, so a
! row is reproducible from the CSV alone and any argument WATER does not touch
! is visibly identical across the two stages.  Reals carry both a round-tripping
! decimal form and the raw IEEE binary32 bit pattern; the consumer compares bit
! patterns, so no parsing tolerance exists anywhere.
!
! The `probe` stage
! -----------------
! Four of WATER's branches are decided by locals that never reach an output:
! QSNSUB/QSNFRO's `SNEQV > 0.0` gates, the MIN inside QSNSUB, and SNOWWATER's
! glacier SNOFLOW.  The probe stage makes each assertable from the inputs.
! The first three are re-derived from the emitted inputs by MIN/MAX alone.
! SNOFLOW comes from running **the module's own SNOWWATER** on a copy of the
! entry state -- no physics is duplicated in the harness.  The probe stage is
! never the gate: the port is held to the `output` stage, so a probe that
! agreed with a wrong port could not hide anything.

module water_oracle

  use module_sf_noahmplsm, only : noahmp_parameters, WATER, SNOWWATER,       &
                                  TFRZ, HVAP, HSUB, HFUS, CWAT, CICE,        &
                                  DENH2O, DENICE,                            &
                                  calculate_soil, soil_update_steps
  implicit none

  integer, parameter :: NSNOW = 3
  integer, parameter :: NSOIL = 4
  integer, parameter :: CSV   = 21
  integer, parameter :: PRB   = 22

  ! Soil interfaces -0.10/-0.40/-1.00/-2.00 m: the pinned Registry stack.
  real, parameter :: ZSOIL_PIN(NSOIL)  = (/ -0.10, -0.40, -1.00, -2.00 /)
  real, parameter :: DZSOIL_PIN(NSOIL) = (/  0.10,  0.30,  0.60,  1.00 /)

  real, parameter :: SENTINEL = -999.0

  ! Everything WATER is handed.  Grouped exactly as the dummy list is, so the
  ! call below can be diffed line by line against 5954-5972.
  type :: wstate
     ! --- in
     integer :: vegtyp, ist, iloc, jloc
     integer :: imelt(-NSNOW+1:0)
     real    :: dt, uu, vv, fcev, fctr, qprecc, qprecl, elai, esai, sfctmp
     real    :: qvap, qdew, tg, fveg, bdfall, fp, rain, snow
     real    :: qsnow, qrain, snowhin, latheav, latheag, dx, tdfracmp
     logical :: frozen_canopy, frozen_ground, croplu
     real    :: irrfra, mifac, fifac
     real    :: zsoil(NSOIL), btrani(NSOIL), smceq(NSOIL)
     real    :: ficeold(-NSNOW+1:0)
     ! --- inout
     integer :: isnow
     real    :: ponding
     real    :: canliq, canice, tv, snowh, sneqv
     real    :: snice(-NSNOW+1:0), snliq(-NSNOW+1:0)
     real    :: stc(-NSNOW+1:NSOIL), zsnso(-NSNOW+1:NSOIL)
     real    :: dzsnso(-NSNOW+1:NSOIL)
     real    :: sh2o(NSOIL), sice(NSOIL), smc(NSOIL)
     real    :: zwt, wa, wt, wslake, smcwtd, deeprech, rech, qtldrn
     real    :: iramtfi, iramtmi, irfirate, irmirate
     real    :: acc_qinsur, acc_qseva, acc_etrani(NSOIL)
     ! --- out, but entered with a live value so the aliasing is measured
     real    :: cmc, ecan, etran, fwet, runsrf, runsub
     real    :: qin, qdis, ponding1, ponding2, qsnbot
     real    :: qsnsub, qsnfro, qsubc, qfroc, qfrzc, qmeltc, qevac, qdewc
  end type wstate

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

  subroutine emit_r(cname, stage, field, idx, v)
    character(len=*), intent(in) :: cname, stage, field
    integer, intent(in) :: idx
    real,    intent(in) :: v
    write(CSV, '(A)') 'water,'//cname//','//stage//','//field//','// &
         trim(itoa(idx))//',real,'//trim(rtoa(v))//','//hex8(v)
  end subroutine emit_r

  subroutine emit_i(cname, stage, field, idx, v)
    character(len=*), intent(in) :: cname, stage, field
    integer, intent(in) :: idx
    integer, intent(in) :: v
    write(CSV, '(A)') 'water,'//cname//','//stage//','//field//','// &
         trim(itoa(idx))//',int,'//trim(itoa(v))//','
  end subroutine emit_i

  subroutine emit_l(cname, stage, field, idx, v)
    character(len=*), intent(in) :: cname, stage, field
    integer, intent(in) :: idx
    logical, intent(in) :: v
    if (v) then
       call emit_i(cname, stage, field, idx, 1)
    else
       call emit_i(cname, stage, field, idx, 0)
    end if
  end subroutine emit_l

  subroutine emit_ra(cname, stage, field, lo, hi, a)
    character(len=*), intent(in) :: cname, stage, field
    integer, intent(in) :: lo, hi
    real,    intent(in) :: a(lo:hi)
    integer :: k
    do k = lo, hi
       call emit_r(cname, stage, field, k, a(k))
    end do
  end subroutine emit_ra

  subroutine emit_ia(cname, stage, field, lo, hi, a)
    character(len=*), intent(in) :: cname, stage, field
    integer, intent(in) :: lo, hi
    integer, intent(in) :: a(lo:hi)
    integer :: k
    do k = lo, hi
       call emit_i(cname, stage, field, k, a(k))
    end do
  end subroutine emit_ia

  subroutine emit_probe(name, v)
    character(len=*), intent(in) :: name
    real, intent(in) :: v
    write(PRB, '(A)') name//','//trim(rtoa(v))//','//hex8(v)
  end subroutine emit_probe

! ---------------------------------------------------------------------------
! Parameter handle.  Every per-layer entry varies with layer, so a port that
! indexes the wrong layer cannot reproduce the fixture.
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
    p%NROOT  = 3
    p%SSI    = 0.03
    p%SNOW_RET_FAC = 5.0e-5
    p%URBAN_FLAG = .false.
    ! Dead under opt_run=3; non-zero so the inertness claim is measured.
    p%TIMEAN = 10.5
    p%FSATMX = 0.38
  end subroutine base_parameters

  subroutine emit_parameters(cname, p)
    character(len=*), intent(in) :: cname
    type(noahmp_parameters), intent(in) :: p
    call emit_ra(cname, 'param', 'smcmax', 1, NSOIL, p%SMCMAX(1:NSOIL))
    call emit_ra(cname, 'param', 'smcwlt', 1, NSOIL, p%SMCWLT(1:NSOIL))
    call emit_ra(cname, 'param', 'bexp',   1, NSOIL, p%BEXP(1:NSOIL))
    call emit_ra(cname, 'param', 'dksat',  1, NSOIL, p%DKSAT(1:NSOIL))
    call emit_ra(cname, 'param', 'dwsat',  1, NSOIL, p%DWSAT(1:NSOIL))
    call emit_r (cname, 'param', 'kdt',    0, p%KDT)
    call emit_r (cname, 'param', 'frzx',   0, p%FRZX)
    call emit_r (cname, 'param', 'slope',  0, p%SLOPE)
    call emit_r (cname, 'param', 'ch2op',  0, p%CH2OP)
    call emit_r (cname, 'param', 'ssi',    0, p%SSI)
    call emit_r (cname, 'param', 'snow_ret_fac', 0, p%SNOW_RET_FAC)
    call emit_r (cname, 'param', 'timean', 0, p%TIMEAN)
    call emit_r (cname, 'param', 'fsatmx', 0, p%FSATMX)
    call emit_i (cname, 'param', 'nroot',  0, p%NROOT)
    call emit_l (cname, 'param', 'urban_flag', 0, p%URBAN_FLAG)
  end subroutine emit_parameters

! ---------------------------------------------------------------------------
! State emission
! ---------------------------------------------------------------------------

  subroutine emit_state(cname, stage, s)
    character(len=*), intent(in) :: cname, stage
    type(wstate), intent(in) :: s

    call emit_i (cname, stage, 'nsnow',   0, NSNOW)
    call emit_i (cname, stage, 'nsoil',   0, NSOIL)
    call emit_i (cname, stage, 'vegtyp',  0, s%vegtyp)
    call emit_i (cname, stage, 'ist',     0, s%ist)
    call emit_i (cname, stage, 'iloc',    0, s%iloc)
    call emit_i (cname, stage, 'jloc',    0, s%jloc)
    call emit_ia(cname, stage, 'imelt',   -NSNOW+1, 0, s%imelt)
    call emit_r (cname, stage, 'dt',      0, s%dt)
    call emit_r (cname, stage, 'uu',      0, s%uu)
    call emit_r (cname, stage, 'vv',      0, s%vv)
    call emit_r (cname, stage, 'fcev',    0, s%fcev)
    call emit_r (cname, stage, 'fctr',    0, s%fctr)
    call emit_r (cname, stage, 'qprecc',  0, s%qprecc)
    call emit_r (cname, stage, 'qprecl',  0, s%qprecl)
    call emit_r (cname, stage, 'elai',    0, s%elai)
    call emit_r (cname, stage, 'esai',    0, s%esai)
    call emit_r (cname, stage, 'sfctmp',  0, s%sfctmp)
    call emit_r (cname, stage, 'qvap',    0, s%qvap)
    call emit_r (cname, stage, 'qdew',    0, s%qdew)
    call emit_r (cname, stage, 'tg',      0, s%tg)
    call emit_r (cname, stage, 'fveg',    0, s%fveg)
    call emit_r (cname, stage, 'bdfall',  0, s%bdfall)
    call emit_r (cname, stage, 'fp',      0, s%fp)
    call emit_r (cname, stage, 'rain',    0, s%rain)
    call emit_r (cname, stage, 'snow',    0, s%snow)
    call emit_r (cname, stage, 'qsnow',   0, s%qsnow)
    call emit_r (cname, stage, 'qrain',   0, s%qrain)
    call emit_r (cname, stage, 'snowhin', 0, s%snowhin)
    call emit_r (cname, stage, 'latheav', 0, s%latheav)
    call emit_r (cname, stage, 'latheag', 0, s%latheag)
    call emit_r (cname, stage, 'dx',      0, s%dx)
    call emit_r (cname, stage, 'tdfracmp',0, s%tdfracmp)
    call emit_l (cname, stage, 'frozen_canopy', 0, s%frozen_canopy)
    call emit_l (cname, stage, 'frozen_ground', 0, s%frozen_ground)
    call emit_l (cname, stage, 'croplu',  0, s%croplu)
    call emit_r (cname, stage, 'irrfra',  0, s%irrfra)
    call emit_r (cname, stage, 'mifac',   0, s%mifac)
    call emit_r (cname, stage, 'fifac',   0, s%fifac)
    call emit_ra(cname, stage, 'zsoil',   1, NSOIL, s%zsoil)
    call emit_ra(cname, stage, 'btrani',  1, NSOIL, s%btrani)
    call emit_ra(cname, stage, 'smceq',   1, NSOIL, s%smceq)
    call emit_ra(cname, stage, 'ficeold', -NSNOW+1, 0, s%ficeold)

    call emit_i (cname, stage, 'isnow',   0, s%isnow)
    call emit_r (cname, stage, 'ponding', 0, s%ponding)
    call emit_r (cname, stage, 'canliq',  0, s%canliq)
    call emit_r (cname, stage, 'canice',  0, s%canice)
    call emit_r (cname, stage, 'tv',      0, s%tv)
    call emit_r (cname, stage, 'snowh',   0, s%snowh)
    call emit_r (cname, stage, 'sneqv',   0, s%sneqv)
    call emit_ra(cname, stage, 'snice',   -NSNOW+1, 0, s%snice)
    call emit_ra(cname, stage, 'snliq',   -NSNOW+1, 0, s%snliq)
    call emit_ra(cname, stage, 'stc',     -NSNOW+1, NSOIL, s%stc)
    call emit_ra(cname, stage, 'zsnso',   -NSNOW+1, NSOIL, s%zsnso)
    call emit_ra(cname, stage, 'dzsnso',  -NSNOW+1, NSOIL, s%dzsnso)
    call emit_ra(cname, stage, 'sh2o',    1, NSOIL, s%sh2o)
    call emit_ra(cname, stage, 'sice',    1, NSOIL, s%sice)
    call emit_ra(cname, stage, 'smc',     1, NSOIL, s%smc)
    call emit_r (cname, stage, 'zwt',     0, s%zwt)
    call emit_r (cname, stage, 'wa',      0, s%wa)
    call emit_r (cname, stage, 'wt',      0, s%wt)
    call emit_r (cname, stage, 'wslake',  0, s%wslake)
    call emit_r (cname, stage, 'smcwtd',  0, s%smcwtd)
    call emit_r (cname, stage, 'deeprech',0, s%deeprech)
    call emit_r (cname, stage, 'rech',    0, s%rech)
    call emit_r (cname, stage, 'qtldrn',  0, s%qtldrn)
    call emit_r (cname, stage, 'iramtfi', 0, s%iramtfi)
    call emit_r (cname, stage, 'iramtmi', 0, s%iramtmi)
    call emit_r (cname, stage, 'irfirate',0, s%irfirate)
    call emit_r (cname, stage, 'irmirate',0, s%irmirate)
    call emit_r (cname, stage, 'acc_qinsur', 0, s%acc_qinsur)
    call emit_r (cname, stage, 'acc_qseva',  0, s%acc_qseva)
    call emit_ra(cname, stage, 'acc_etrani', 1, NSOIL, s%acc_etrani)

    call emit_r (cname, stage, 'cmc',     0, s%cmc)
    call emit_r (cname, stage, 'ecan',    0, s%ecan)
    call emit_r (cname, stage, 'etran',   0, s%etran)
    call emit_r (cname, stage, 'fwet',    0, s%fwet)
    call emit_r (cname, stage, 'runsrf',  0, s%runsrf)
    call emit_r (cname, stage, 'runsub',  0, s%runsub)
    call emit_r (cname, stage, 'qin',     0, s%qin)
    call emit_r (cname, stage, 'qdis',    0, s%qdis)
    call emit_r (cname, stage, 'ponding1',0, s%ponding1)
    call emit_r (cname, stage, 'ponding2',0, s%ponding2)
    call emit_r (cname, stage, 'qsnbot',  0, s%qsnbot)
    call emit_r (cname, stage, 'qsnsub',  0, s%qsnsub)
    call emit_r (cname, stage, 'qsnfro',  0, s%qsnfro)
    call emit_r (cname, stage, 'qsubc',   0, s%qsubc)
    call emit_r (cname, stage, 'qfroc',   0, s%qfroc)
    call emit_r (cname, stage, 'qfrzc',   0, s%qfrzc)
    call emit_r (cname, stage, 'qmeltc',  0, s%qmeltc)
    call emit_r (cname, stage, 'qevac',   0, s%qevac)
    call emit_r (cname, stage, 'qdewc',   0, s%qdewc)
  end subroutine emit_state

! ---------------------------------------------------------------------------
! One case
! ---------------------------------------------------------------------------

  subroutine run_case(cname, p, s_in)
    character(len=*), intent(in) :: cname
    type(noahmp_parameters), intent(in) :: p
    type(wstate), intent(in) :: s_in

    type(wstate) :: s
    ! Probe copies.  SNOWWATER is the module's own routine; running it on a
    ! copy of the entry state is what makes SNOFLOW assertable from the inputs.
    integer :: p_isnow
    real    :: p_snowh, p_sneqv
    real    :: p_snice(-NSNOW+1:0), p_snliq(-NSNOW+1:0)
    real    :: p_stc(-NSNOW+1:NSOIL), p_zsnso(-NSNOW+1:NSOIL)
    real    :: p_dzsnso(-NSNOW+1:NSOIL)
    real    :: p_sh2o(NSOIL), p_sice(NSOIL)
    real    :: p_qsnbot, p_snoflow, p_ponding1, p_ponding2
    real    :: p_qsnsub, p_qsnfro, p_qseva, p_qsdew

    s = s_in

    call emit_parameters(cname, p)
    call emit_state(cname, 'input', s)

    ! ---- probe: the four scalars WATER derives at 6126-6136 -----------------
    ! Pure MIN/subtraction over the emitted inputs; no physics is duplicated.
    p_qsnsub = 0.0
    if (s%sneqv > 0.0) p_qsnsub = min(s%qvap, s%sneqv / s%dt)
    p_qseva  = s%qvap - p_qsnsub
    p_qsnfro = 0.0
    if (s%sneqv > 0.0) p_qsnfro = s%qdew
    p_qsdew  = s%qdew - p_qsnfro
    call emit_r(cname, 'probe', 'qsnsub', 0, p_qsnsub)
    call emit_r(cname, 'probe', 'qseva',  0, p_qseva)
    call emit_r(cname, 'probe', 'qsnfro', 0, p_qsnfro)
    call emit_r(cname, 'probe', 'qsdew',  0, p_qsdew)

    ! ---- probe: SNOWWATER on a copy, for SNOFLOW and the pack geometry ------
    p_isnow  = s%isnow
    p_snowh  = s%snowh
    p_sneqv  = s%sneqv
    p_snice  = s%snice
    p_snliq  = s%snliq
    p_stc    = s%stc
    p_zsnso  = s%zsnso
    p_dzsnso = s%dzsnso
    p_sh2o   = s%sh2o
    p_sice   = s%sice
    p_qsnbot = SENTINEL; p_snoflow = SENTINEL
    p_ponding1 = 0.0;    p_ponding2 = 0.0
    ! These two emits are load-bearing, not diagnostics: COMBINE leaves
    ! PONDING1/PONDING2 untouched on some paths, so with the explicit interface
    ! in scope the compiler may treat the stores above as dead before an
    ! INTENT(OUT) call.  Reading them here forces the stores to be live.
    call emit_r(cname, 'probe', 'ponding1_in', 0, p_ponding1)
    call emit_r(cname, 'probe', 'ponding2_in', 0, p_ponding2)
    call SNOWWATER(p, NSNOW, NSOIL, s%imelt, s%dt, s%zsoil, s%sfctmp,        &
                   s%snowhin, s%qsnow, p_qsnfro, p_qsnsub, s%qrain,          &
                   s%ficeold, s%iloc, s%jloc,                                &
                   p_isnow, p_snowh, p_sneqv, p_snice, p_snliq,              &
                   p_sh2o, p_sice, p_stc, p_zsnso, p_dzsnso,                 &
                   p_qsnbot, p_snoflow, p_ponding1, p_ponding2)
    call emit_i(cname, 'probe', 'isnow_post_snow',  0, p_isnow)
    call emit_r(cname, 'probe', 'sneqv_post_snow',  0, p_sneqv)
    call emit_r(cname, 'probe', 'snoflow',  0, p_snoflow)
    call emit_r(cname, 'probe', 'qsnbot',   0, p_qsnbot)
    call emit_r(cname, 'probe', 'ponding1', 0, p_ponding1)
    call emit_r(cname, 'probe', 'ponding2', 0, p_ponding2)

    ! ---- the call under test ----------------------------------------------
    call WATER (p, s%vegtyp, NSNOW, NSOIL, s%imelt, s%dt, s%uu,              &
                s%vv, s%fcev, s%fctr, s%qprecc, s%qprecl, s%elai,            &
                s%esai, s%sfctmp, s%qvap, s%qdew, s%zsoil, s%btrani,         &
                s%irrfra, s%mifac, s%fifac, s%croplu,                        &
                s%ficeold, s%ponding, s%tg, s%ist, s%fveg, s%iloc, s%jloc,   &
                s%smceq,                                                     &
                s%bdfall, s%fp, s%rain, s%snow,                              &
                s%qsnow, s%qrain, s%snowhin, s%latheav, s%latheag,           &
                s%frozen_canopy, s%frozen_ground,                            &
                s%dx, s%tdfracmp,                                            &
                s%isnow, s%canliq, s%canice, s%tv, s%snowh, s%sneqv,         &
                s%snice, s%snliq, s%stc, s%zsnso, s%sh2o, s%smc,             &
                s%sice, s%zwt, s%wa, s%wt, s%dzsnso, s%wslake,               &
                s%smcwtd, s%deeprech, s%rech,                                &
                s%iramtfi, s%iramtmi, s%irfirate, s%irmirate,                &
                s%cmc, s%ecan, s%etran, s%fwet, s%runsrf, s%runsub,          &
                s%qin, s%qdis, s%ponding1, s%ponding2,                       &
                s%qsnbot, s%qtldrn,                                          &
                s%qsnsub, s%qsnfro, s%qsubc, s%qfroc, s%qfrzc, s%qmeltc,     &
                s%qevac, s%qdewc, s%acc_qinsur, s%acc_qseva, s%acc_etrani)

    call emit_state(cname, 'output', s)
  end subroutine run_case

! ---------------------------------------------------------------------------
! Base state.  Everything non-zero unless the case needs it zero, so no input
! slot is zero in every case.
! ---------------------------------------------------------------------------

  function base_state() result(s)
    type(wstate) :: s

    s%vegtyp = 11
    s%ist    = 1
    s%iloc   = 3
    s%jloc   = 5
    s%imelt  = (/ 2, 2, 2 /)
    s%dt     = 1800.0
    s%uu     = 3.5
    s%vv     = -2.25
    s%fcev   = 45.0
    s%fctr   = 30.0
    s%qprecc = 1.5e-5
    s%qprecl = 4.5e-5
    s%elai   = 3.0
    s%esai   = 0.5
    s%sfctmp = 285.0
    s%qvap   = 1.2e-5
    s%qdew   = 0.0
    s%tg     = 285.0
    s%fveg   = 0.8
    s%bdfall = 120.0
    s%fp     = 0.65
    s%rain   = 5.5e-5
    s%snow   = 0.0
    s%qsnow  = 0.0
    s%qrain  = 4.0e-5
    s%snowhin= 0.0
    s%latheav= 2.5104e06
    s%latheag= 2.5104e06
    s%dx     = 3000.0
    s%tdfracmp = 0.0
    s%frozen_canopy = .false.
    s%frozen_ground = .false.
    s%croplu = .false.
    s%irrfra = 0.0
    s%mifac  = 0.0
    s%fifac  = 0.0
    s%zsoil  = ZSOIL_PIN
    s%btrani = (/ 0.55, 0.30, 0.15, 0.0 /)
    s%smceq  = (/ 0.21, 0.23, 0.25, 0.27 /)
    s%ficeold= (/ 0.0, 0.0, 0.0 /)

    s%isnow  = 0
    s%ponding= 0.0
    s%canliq = 0.40
    s%canice = 0.0
    s%tv     = 285.5
    s%snowh  = 0.0
    s%sneqv  = 0.0
    s%snice  = (/ 0.0, 0.0, 0.0 /)
    s%snliq  = (/ 0.0, 0.0, 0.0 /)
    s%stc(-2) = 0.0; s%stc(-1) = 0.0; s%stc(0) = 0.0
    s%stc(1:NSOIL) = (/ 285.0, 284.0, 283.0, 282.0 /)
    s%zsnso(-2) = 0.0; s%zsnso(-1) = 0.0; s%zsnso(0) = 0.0
    s%zsnso(1:NSOIL) = ZSOIL_PIN
    s%dzsnso(-2) = 0.0; s%dzsnso(-1) = 0.0; s%dzsnso(0) = 0.0
    s%dzsnso(1:NSOIL) = DZSOIL_PIN
    s%sh2o   = (/ 0.180, 0.210, 0.240, 0.260 /)
    s%sice   = (/ 0.0, 0.0, 0.0, 0.0 /)
    s%smc    = (/ 0.180, 0.210, 0.240, 0.260 /)
    s%zwt    = -2.5
    s%wa     = 4900.0
    s%wt     = 4900.0
    s%wslake = 0.0
    s%smcwtd = 0.30
    s%deeprech = 0.0
    s%rech   = 0.0
    s%qtldrn = 0.0
    s%iramtfi = 0.0
    s%iramtmi = 0.0
    s%irfirate = 0.0
    s%irmirate = 0.0
    s%acc_qinsur = 0.0
    s%acc_qseva  = 0.0
    s%acc_etrani = (/ 0.0, 0.0, 0.0, 0.0 /)

    ! INTENT(OUT) entry values.  Distinct and non-zero so that anything WATER
    ! fails to write is visible instead of being masked by a zero.
    s%cmc      = SENTINEL
    s%ecan     = SENTINEL
    s%etran    = SENTINEL
    s%fwet     = SENTINEL
    s%runsrf   = 11.5
    s%runsub   = 7.5
    s%qin      = -31.25
    s%qdis     = 17.75
    s%ponding1 = SENTINEL
    s%ponding2 = SENTINEL
    s%qsnbot   = SENTINEL
    s%qsnsub   = -3.5
    s%qsnfro   = -6.25
    s%qsubc    = SENTINEL
    s%qfroc    = SENTINEL
    s%qfrzc    = SENTINEL
    s%qmeltc   = SENTINEL
    s%qevac    = SENTINEL
    s%qdewc    = SENTINEL
  end function base_state

! ---------------------------------------------------------------------------
! A three-layer snowpack on top of the base column.
! ---------------------------------------------------------------------------

  function snow_state() result(s)
    type(wstate) :: s
    s = base_state()
    s%isnow  = -2
    s%snowh  = 0.28
    s%sneqv  = 60.0
    s%snice  = (/ 0.0, 25.0, 33.0 /)
    s%snliq  = (/ 0.0, 0.5, 0.8 /)
    s%stc(-1) = 268.0
    s%stc(0)  = 271.0
    s%dzsnso(-1) = 0.100
    s%dzsnso(0)  = 0.180
    s%zsnso(-1) = -0.180
    s%zsnso(0)  =  0.0
    s%zsnso(1:NSOIL) = ZSOIL_PIN
    s%ficeold = (/ 0.0, 0.98, 0.97 /)
    s%imelt   = (/ 2, 1, 1 /)
    s%sfctmp  = 268.0
    s%qsnow   = 3.0e-5
    s%snowhin = 3.0e-7
    s%qrain   = 0.0
  end function snow_state

! ---------------------------------------------------------------------------
! libm / constant probe.  Same construction as the soil-water harness: the
! sweeps give the device gate its reference values on exactly the domains this
! assembly reaches, and the folded/runtime EXP(-A) pair is pinned because
! gfortran folds one with MPFR and lowers the other to expf@plt.
! ---------------------------------------------------------------------------

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

    ! expf over SOILWATER's FCR argument range -A*(1-FICE), FICE in [0,1].
    do i = 0, 256
       x = -4.0 * (1.0 - real(i) / 256.0)
       y = EXP(x)
       call emit_probe('expf_fcr_'//trim(itoa(i)), y)
    end do

    ! expf over COMPACT's exponents, which run far more negative.
    do i = 0, 256
       x = -40.0 * (real(i) / 256.0)
       y = EXP(x)
       call emit_probe('expf_compact_'//trim(itoa(i)), y)
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

end module water_oracle


! ===========================================================================
program run_water

  use module_sf_noahmplsm, only : noahmp_options, noahmp_parameters,         &
                                  OPT_RUN, OPT_INF, OPT_TDRN, OPT_IRR,       &
                                  OPT_INFDV, DVEG, OPT_CROP,                 &
                                  calculate_soil, soil_update_steps
  use water_oracle
  implicit none

  character(len=1024) :: csv_path, probe_path
  type(noahmp_parameters) :: p, pu, pr4, ps
  type(wstate) :: s

  call get_command_argument(1, csv_path)
  call get_command_argument(2, probe_path)
  if (len_trim(csv_path) == 0 .or. len_trim(probe_path) == 0) then
     write(*, '(A)') 'usage: run_water WATER.csv PROBE.csv'
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

  ! soiltstep=0.0 in the pinned Registry, which module_sf_noahmpdrv.F turns
  ! into soil_update_steps = max(nint(0/DT),1) = 1 and, since
  ! mod(itimestep,1)==0 always, calculate_soil = .true. on every step.
  soil_update_steps = 1
  calculate_soil    = .true.
  if (soil_update_steps /= 1) error stop 'soil_update_steps is not 1'
  if (.not. calculate_soil) error stop 'calculate_soil is not true'

  open(CSV, file=trim(csv_path), status='replace', action='write')
  write(CSV, '(A)') 'leaf,case,stage,field,index,dtype,value,bits'
  open(PRB, file=trim(probe_path), status='replace', action='write')
  write(PRB, '(A)') 'name,value,bits'

  call dump_probe()

  call base_parameters(p)

  call base_parameters(pu)
  pu%URBAN_FLAG = .true.

  ! NROOT = 4, so the ETRANI loop at 6169 reaches the bottom layer.  Without
  ! this the fixture cannot tell a port that stops at NROOT from one that runs
  ! to NSOIL, because BTRANI(4) would only ever multiply into a slot nothing
  ! reads.
  call base_parameters(pr4)
  pr4%NROOT = 4

  ! WRF SOILPARM's sand row, the coarse and highly conductive set.  It exists
  ! so SOILWATER's NITER doubling is reachable; see PROVENANCE-soilwater.md.
  call base_parameters(ps)
  ps%SMCMAX = (/ 0.339, 0.339, 0.339, 0.339 /)
  ps%SMCWLT = (/ 0.010, 0.010, 0.010, 0.010 /)
  ps%BEXP   = (/ 2.79,  2.79,  2.79,  2.79  /)
  ps%DKSAT  = (/ 1.7639e-4, 1.7639e-4, 1.7639e-4, 1.7639e-4 /)
  ps%DWSAT  = (/ 6.0808e-5, 6.0808e-5, 6.0808e-5, 6.0808e-5 /)
  ps%KDT    = 264.585

! ---------------------------------------------------------------------------
! Snow-free soil column
! ---------------------------------------------------------------------------

  ! Baseline.  SNEQV == 0 so both `SNEQV > 0.0` gates are false, QSEVA is the
  ! whole of QVAP and QSDEW the whole of QDEW; ISNOW == 0 so QRAIN enters
  ! QINSUR; FROZEN_GROUND false; IST == 1 so SOILWATER runs and RUNSUB picks
  ! up QDRAIN.
  s = base_state()
  call run_case('wat_bare_rain', p, s)

  ! Same column with the accumulators entering non-zero.  ACC_QINSUR/ACC_QSEVA/
  ! ACC_ETRANI are INTENT(INOUT) and WATER adds to them at 6178-6180 before
  ! dividing by soil_update_steps at 6204-6206, so a non-zero entry changes
  ! what SOILWATER is handed.  The driver zeroes them every step when
  ! soil_update_steps == 1, so this is the aliasing being measured, not a
  ! forecast state.
  s = base_state()
  s%acc_qinsur = 2.5e-6
  s%acc_qseva  = 7.0e-9
  s%acc_etrani = (/ 3.0e-9, 2.0e-9, 1.0e-9, 5.0e-10 /)
  call run_case('wat_acc_nonzero', p, s)

  ! NROOT = 4: BTRANI(4) is non-zero and reaches ETRANI(4).
  s = base_state()
  s%btrani = (/ 0.40, 0.28, 0.20, 0.12 /)
  call run_case('wat_nroot_full', pr4, s)

  ! The same BTRANI at NROOT = 3, which isolates the loop bound: ETRANI(4)
  ! must stay exactly 0.0 although BTRANI(4) is 0.12.
  s = base_state()
  s%btrani = (/ 0.40, 0.28, 0.20, 0.12 /)
  call run_case('wat_nroot_short', p, s)

  ! Dew on a snow-free column: QDEW > 0 with SNEQV == 0, so QSNFRO stays 0 and
  ! the whole of QDEW becomes QSDEW and enters QINSUR.
  s = base_state()
  s%qvap = 0.0
  s%qdew = 8.0e-6
  s%fcev = -22.0
  s%tv   = 281.0
  call run_case('wat_bare_dew', p, s)

  ! Melt-water ponding with no snow layer: PONDING is the only QINSUR source.
  s = base_state()
  s%ponding = 1.75
  s%qrain   = 0.0
  s%qvap    = 0.0
  call run_case('wat_ponding_only', p, s)

  ! Frozen ground, SICE(1) stays positive: the 6145-6154 block moves the
  ! surface exchange onto SICE(1), zeroes QSDEW and QSEVA, and rewrites SMC(1).
  s = base_state()
  s%frozen_ground = .true.
  s%frozen_canopy = .true.
  s%latheag = 2.8440e06
  s%latheav = 2.8440e06
  s%sfctmp  = 268.0
  s%tg      = 268.0
  s%tv      = 268.0
  s%stc(1:NSOIL) = (/ 271.0, 272.0, 274.0, 276.0 /)
  s%sh2o    = (/ 0.140, 0.170, 0.200, 0.230 /)
  s%sice    = (/ 0.120, 0.090, 0.060, 0.030 /)
  s%smc     = (/ 0.260, 0.260, 0.260, 0.260 /)
  s%qvap    = 4.0e-6
  s%qdew    = 0.0
  call run_case('wat_frozen_ground', p, s)

  ! Frozen ground with dew instead of evaporation, so the SICE(1) increment
  ! has the other sign.
  s = base_state()
  s%frozen_ground = .true.
  s%frozen_canopy = .true.
  s%latheag = 2.8440e06
  s%latheav = 2.8440e06
  s%sfctmp  = 266.0
  s%tg      = 266.0
  s%tv      = 266.0
  s%fcev    = -18.0
  s%stc(1:NSOIL) = (/ 270.0, 272.0, 274.0, 276.0 /)
  s%sh2o    = (/ 0.140, 0.170, 0.200, 0.230 /)
  s%sice    = (/ 0.120, 0.090, 0.060, 0.030 /)
  s%smc     = (/ 0.260, 0.260, 0.260, 0.260 /)
  s%qvap    = 0.0
  s%qdew    = 6.0e-6
  call run_case('wat_frozen_ground_dew', p, s)

  ! Frozen ground driven hard enough that SICE(1) goes negative, so the
  ! 6149-6152 fixup charges the deficit to SH2O(1) and clamps SICE(1) to 0.
  s = base_state()
  s%frozen_ground = .true.
  s%frozen_canopy = .true.
  s%latheag = 2.8440e06
  s%latheav = 2.8440e06
  s%sfctmp  = 268.0
  s%tg      = 268.0
  s%tv      = 268.0
  s%stc(1:NSOIL) = (/ 271.0, 272.0, 274.0, 276.0 /)
  s%sh2o    = (/ 0.200, 0.170, 0.200, 0.230 /)
  s%sice    = (/ 0.0005, 0.090, 0.060, 0.030 /)
  s%smc     = (/ 0.2005, 0.260, 0.260, 0.260 /)
  s%qvap    = 6.0e-5
  s%qdew    = 0.0
  call run_case('wat_frozen_ground_sice_deficit', p, s)

  ! Heavy rain on the coarse set: SOILWATER's NITER doubles to 6.
  s = base_state()
  s%qrain = 2.0e-3
  s%sh2o  = (/ 0.150, 0.180, 0.200, 0.220 /)
  s%smc   = (/ 0.150, 0.180, 0.200, 0.220 /)
  call run_case('wat_heavy_rain_coarse', ps, s)

  ! Urban: SOILWATER overwrites FCR(1) at 7361.
  s = base_state()
  s%vegtyp = 13
  call run_case('wat_urban', pu, s)

  ! A 60 s step, so DT is discriminated independently of the fluxes.  DT
  ! enters QSNSUB's MIN, the PONDING conversion, the FROZEN_GROUND increment,
  ! DT_soil and the final SNOFLOW*DT.
  s = base_state()
  s%dt = 60.0
  call run_case('wat_short_step', p, s)

  ! No precipitation and no surface exchange at all: QINSUR is exactly 0 and
  ! SOILWATER runs a pure redistribution.
  s = base_state()
  s%qrain = 0.0
  s%qvap  = 0.0
  s%fcev  = 0.0
  s%fctr  = 0.0
  s%canliq = 0.0
  call run_case('wat_quiescent', p, s)

  ! Strong evaporative demand out of a dry column: SOILWATER's WATMIN fixup
  ! runs, which is the only path that makes RUNSUB depend on SH2O.
  s = base_state()
  s%qvap = 3.0e-3
  s%qrain = 0.0
  s%btrani = (/ 0.55, 0.30, 0.15, 0.0 /)
  s%fctr = 9.0e3
  s%sh2o = (/ 0.0080, 0.0060, 0.0040, 0.0020 /)
  s%smc  = (/ 0.0080, 0.0060, 0.0040, 0.0020 /)
  call run_case('wat_watmin_fixup', p, s)

! ---------------------------------------------------------------------------
! Snow-covered column
! ---------------------------------------------------------------------------

  ! Layered pack, sublimation limited by QVAP rather than by the pack: the
  ! MIN at 6128 picks QVAP.  ISNOW < 0 so QRAIN does NOT enter QINSUR at 6162.
  s = snow_state()
  s%qvap = 1.0e-5
  s%qrain = 2.0e-5
  call run_case('wat_snow_sublimation', p, s)

  ! A trace pack, so SNEQV/DT falls below a perfectly ordinary QVAP and the
  ! MIN at 6128 picks the pack instead.  This is the physically reachable
  ! shape of that branch: for a layered pack SNEQV/DT is orders of magnitude
  ! above any QVAP a surface energy balance can produce.
  s = base_state()
  s%isnow  = 0
  s%snowh  = 3.0e-5
  s%sneqv  = 0.01
  s%sfctmp = 271.0
  s%qsnow  = 0.0
  s%snowhin= 0.0
  s%qrain  = 2.0e-5
  s%qvap   = 1.2e-5
  call run_case('wat_snow_sublimation_masslimited', p, s)

  ! Frost onto a pack: SNEQV > 0 so QSNFRO takes the whole of QDEW and QSDEW
  ! is exactly 0.
  s = snow_state()
  s%qvap = 0.0
  s%qdew = 9.0e-6
  s%fcev = -18.0
  s%frozen_canopy = .true.
  s%latheav = 2.8440e06
  s%latheag = 2.8440e06
  call run_case('wat_snow_frost', p, s)

  ! Sub-layer pack: SNEQV > 0 but ISNOW == 0, so both gates are true AND
  ! QRAIN enters QINSUR at 6162.  This is the only combination the two cases above
  ! cannot reach.
  s = base_state()
  s%isnow  = 0
  s%snowh  = 0.020
  s%sneqv  = 5.0
  s%sfctmp = 271.0
  s%qsnow  = 1.0e-5
  s%snowhin= 1.0e-7
  s%qrain  = 3.0e-5
  s%qvap   = 8.0e-6
  call run_case('wat_sublayer_pack', p, s)

  ! A full three-layer pack under a frozen canopy carrying ice.  This is the
  ! only shape in which the ISNOW = -3 slots of SNICE/SNLIQ/STC/DZSNSO/FICEOLD
  ! are live at all, and the only one in which CANICE is non-zero, so without
  ! it a port that ignored either could reproduce every other case.
  s = snow_state()
  s%isnow  = -3
  s%snowh  = 0.540
  s%sneqv  = 46.0
  s%snice  = (/ 8.0, 15.0, 22.0 /)
  s%snliq  = (/ 0.2, 0.4, 0.4 /)
  s%stc(-2) = 255.0
  s%stc(-1) = 262.0
  s%stc(0)  = 270.0
  s%dzsnso(-2) = 0.11
  s%dzsnso(-1) = 0.17
  s%dzsnso(0)  = 0.26
  s%ficeold = (/ 0.96, 0.97, 0.98 /)
  s%imelt   = (/ 1, 1, 2 /)
  s%sfctmp  = 262.0
  s%tv      = 264.0
  s%tg      = 264.0
  s%canliq  = 0.05
  s%canice  = 1.20
  s%frozen_canopy = .true.
  s%latheav = 2.8440e06
  s%latheag = 2.8440e06
  s%fcev    = 35.0
  s%qsnow   = 2.0e-5
  s%snowhin = 2.0e-7
  s%qrain   = 0.0
  s%qvap    = 8.0e-6
  call run_case('wat_deep_pack_canopy_ice', p, s)

  ! The same pack with FCEV < 0, so CANWATER's frost branch runs and CANICE
  ! grows instead of subliming.
  s = snow_state()
  s%isnow  = -3
  s%snowh  = 0.540
  s%sneqv  = 46.0
  s%snice  = (/ 8.0, 15.0, 22.0 /)
  s%snliq  = (/ 0.2, 0.4, 0.4 /)
  s%stc(-2) = 255.0
  s%stc(-1) = 262.0
  s%stc(0)  = 270.0
  s%dzsnso(-2) = 0.11
  s%dzsnso(-1) = 0.17
  s%dzsnso(0)  = 0.26
  s%ficeold = (/ 0.96, 0.97, 0.98 /)
  s%imelt   = (/ 1, 1, 2 /)
  s%sfctmp  = 262.0
  s%tv      = 264.0
  s%tg      = 264.0
  s%canliq  = 0.05
  s%canice  = 0.30
  s%frozen_canopy = .true.
  s%latheav = 2.8440e06
  s%latheag = 2.8440e06
  s%fcev    = -18.0
  s%qsnow   = 2.0e-5
  s%snowhin = 2.0e-7
  s%qrain   = 0.0
  s%qvap    = 0.0
  s%qdew    = 5.0e-6
  call run_case('wat_deep_pack_canopy_frost', p, s)

  ! A sub-layer pack that crosses SNOWH >= 0.025 m inside the call, so
  ! SNOWFALL creates the first snow layer (ISNOW 0 -> -1) and STC(0) is set
  ! from `MIN(TFRZ, SFCTMP)`.  That assignment is the only statement in the
  ! whole assembly that reads SFCTMP, so without this case a port could drop
  ! the argument entirely and still reproduce every other row.
  s = base_state()
  s%isnow   = 0
  s%snowh   = 0.020
  s%sneqv   = 4.0
  s%sfctmp  = 266.5
  s%tv      = 267.0
  s%tg      = 267.0
  s%qsnow   = 3.0e-5
  s%snowhin = 4.0e-6
  s%qrain   = 0.0
  s%qvap    = 5.0e-6
  call run_case('wat_new_snow_layer', p, s)

  ! A melting pack whose ice fraction has fallen below FICEOLD, so COMPACT's
  ! DDZ3 term at 7052-7056 is strictly positive.  That term is the only reader
  ! of FICEOLD and the only place IMELT is consulted, and it is inert unless
  ! FICEOLD exceeds the current SNICE/(SNICE+SNLIQ); every other pack in this
  ! fixture is drier than it was, so the MAX picks zero and both arguments
  ! become undetectable.
  s = snow_state()
  s%isnow  = -2
  s%snowh  = 0.230
  s%sneqv  = 33.0
  s%snice  = (/ 0.0, 10.0, 12.0 /)
  s%snliq  = (/ 0.0, 5.0, 6.0 /)
  s%stc(-1) = 272.5
  s%stc(0)  = 273.16
  s%dzsnso(-1) = 0.100
  s%dzsnso(0)  = 0.130
  s%zsnso(-1) = -0.130
  s%zsnso(0)  = 0.0
  s%ficeold = (/ 0.95, 0.95, 0.95 /)
  s%imelt   = (/ 1, 1, 1 /)
  s%sfctmp  = 273.0
  s%qsnow   = 0.0
  s%snowhin = 0.0
  s%qrain   = 2.0e-5
  s%qvap    = 1.0e-6
  call run_case('wat_melting_pack_ficeold', p, s)

  ! A wet pack that drains: QSNBOT comes back non-zero from SNOWH2O's
  ! percolation loop, which is the only way the QSNBOT term of QINSUR
  ! (6162/6164) is discriminated from zero.  ISNOW < 0, so QRAIN is excluded
  ! and QINSUR is QSNBOT alone.
  s = snow_state()
  s%snowh  = 0.280
  s%sneqv  = 70.0
  s%snice  = (/ 0.0, 25.0, 33.0 /)
  s%snliq  = (/ 0.0, 6.0, 6.0 /)
  s%stc(-1) = 272.0
  s%stc(0)  = 273.16
  s%dzsnso(-1) = 0.100
  s%dzsnso(0)  = 0.180
  s%ficeold = (/ 0.0, 1.0, 1.0 /)
  s%imelt   = (/ 2, 2, 2 /)
  s%sfctmp  = 274.0
  s%qsnow   = 0.0
  s%snowhin = 0.0
  s%qrain   = 1.0e-4
  s%qvap    = 0.0
  call run_case('wat_snow_percolation', p, s)

  ! The same drainage under a non-zero PONDING, so both terms of QINSUR are
  ! live at once and their sum order (6159 then 6164) is observable.
  s = snow_state()
  s%snowh  = 0.280
  s%sneqv  = 70.0
  s%snice  = (/ 0.0, 25.0, 33.0 /)
  s%snliq  = (/ 0.0, 6.0, 6.0 /)
  s%stc(-1) = 272.0
  s%stc(0)  = 273.16
  s%dzsnso(-1) = 0.100
  s%dzsnso(0)  = 0.180
  s%ficeold = (/ 0.0, 1.0, 1.0 /)
  s%imelt   = (/ 2, 2, 2 /)
  s%sfctmp  = 274.0
  s%qsnow   = 0.0
  s%snowhin = 0.0
  s%qrain   = 1.0e-4
  s%qvap    = 0.0
  s%ponding = 0.85
  call run_case('wat_snow_percolation_ponding', p, s)

  ! A single thin layer melts out entirely, so SNOWWATER's COMBINE returns it
  ! as PONDING1 and ISNOW goes -1 -> 0.  PONDING1 then enters QINSUR at 6159
  ! while the ISNOW==0 branch at 6161 also adds QRAIN, which is the only case
  ! where both happen in the same call.
  s = snow_state()
  s%isnow  = -1
  s%snowh  = 0.006
  s%sneqv  = 0.5
  s%snice  = (/ 0.0, 0.0, 0.07 /)
  s%snliq  = (/ 0.0, 0.0, 0.40 /)
  s%stc(0) = 272.5
  s%dzsnso(-1) = 0.0
  s%dzsnso(0)  = 0.006
  s%zsnso(-1)  = 0.0
  s%zsnso(0)   = 0.0
  s%ficeold = (/ 0.0, 0.0, 1.0 /)
  s%imelt   = (/ 2, 2, 1 /)
  s%sfctmp  = 274.0
  s%qsnow   = 0.0
  s%snowhin = 0.0
  s%qrain   = 2.0e-5
  s%qvap    = 0.0
  call run_case('wat_one_layer_to_bare', p, s)

  ! The same melt-out over frozen ground, so the 6145-6154 block composes with
  ! a pack that disappears inside the same call.
  s = snow_state()
  s%isnow  = -1
  s%snowh  = 0.006
  s%sneqv  = 0.5
  s%snice  = (/ 0.0, 0.0, 0.07 /)
  s%snliq  = (/ 0.0, 0.0, 0.40 /)
  s%stc(0) = 272.5
  s%dzsnso(-1) = 0.0
  s%dzsnso(0)  = 0.006
  s%zsnso(-1)  = 0.0
  s%zsnso(0)   = 0.0
  s%ficeold = (/ 0.0, 0.0, 1.0 /)
  s%imelt   = (/ 2, 2, 1 /)
  s%sfctmp  = 272.0
  s%qsnow   = 0.0
  s%snowhin = 0.0
  s%qrain   = 2.0e-5
  s%qvap    = 2.0e-6
  s%frozen_ground = .true.
  s%frozen_canopy = .true.
  s%latheav = 2.8440e06
  s%latheag = 2.8440e06
  s%tv   = 271.0
  s%tg   = 271.0
  s%sh2o = (/ 0.140, 0.170, 0.200, 0.230 /)
  s%sice = (/ 0.120, 0.090, 0.060, 0.030 /)
  s%smc  = (/ 0.260, 0.260, 0.260, 0.260 /)
  s%stc(1:NSOIL) = (/ 271.0, 272.0, 274.0, 276.0 /)
  call run_case('wat_melt_out_frozen_ground', p, s)

  ! Glacier: SNEQV above 5000 inside SNOWWATER, so SNOFLOW is non-zero and the
  ! final `RUNSUB = RUNSUB + SNOFLOW*DT` at 6259 carries a real value.  It is
  ! the only statement outside the calculate_soil block that writes RUNSUB.
  s = snow_state()
  s%isnow  = -1
  s%snowh  = 12.0
  s%sneqv  = 5400.0
  s%snice  = (/ 0.0, 0.0, 5350.0 /)
  s%snliq  = (/ 0.0, 0.0, 50.0 /)
  s%stc(0) = 270.0
  s%dzsnso(-1) = 0.0
  s%dzsnso(0)  = 12.0
  s%zsnso(-1)  = 0.0
  s%zsnso(0)   = 0.0
  s%ficeold = (/ 0.0, 0.0, 0.99 /)
  s%imelt   = (/ 2, 2, 2 /)
  s%qvap    = 1.0e-6
  call run_case('wat_glacier_snoflow', p, s)

  ! Glacier at a 60 s step, so SNOFLOW*DT is discriminated from SNOFLOW.
  s = snow_state()
  s%isnow  = -1
  s%snowh  = 12.0
  s%sneqv  = 5400.0
  s%snice  = (/ 0.0, 0.0, 5350.0 /)
  s%snliq  = (/ 0.0, 0.0, 50.0 /)
  s%stc(0) = 270.0
  s%dzsnso(-1) = 0.0
  s%dzsnso(0)  = 12.0
  s%zsnso(-1)  = 0.0
  s%zsnso(0)   = 0.0
  s%ficeold = (/ 0.0, 0.0, 0.99 /)
  s%imelt   = (/ 2, 2, 2 /)
  s%qvap    = 1.0e-6
  s%dt      = 60.0
  call run_case('wat_glacier_short_step', p, s)

! ---------------------------------------------------------------------------
! Lake (IST == 2).  SOILWATER is not called at all; RUNSRF/RUNSUB/QTLDRN are
! never scaled by DT_soil, and QDRAIN/WCND/FCRMAX stay undefined and unread.
! ---------------------------------------------------------------------------

  ! Below WSLMAX: RUNSRF is exactly 0 and the whole flux goes into WSLAKE.
  s = base_state()
  s%ist    = 2
  s%wslake = 1200.0
  call run_case('wat_lake_filling', p, s)

  ! At WSLMAX exactly: the test is `>=`, so the spill branch fires.
  s = base_state()
  s%ist    = 2
  s%wslake = 5000.0
  call run_case('wat_lake_at_wslmax', p, s)

  ! Above WSLMAX with evaporation exceeding input, so WSLAKE falls.
  s = base_state()
  s%ist    = 2
  s%wslake = 6400.0
  s%qvap   = 9.0e-5
  s%qrain  = 1.0e-6
  call run_case('wat_lake_spilling', p, s)

  ! A lake under a draining pack, which is the only way a non-zero QSNBOT
  ! reaches the lake branch: SNOWWATER still runs, QINSUR_avg carries its
  ! outflow, and WSLAKE takes it instead of SOILWATER.
  s = snow_state()
  s%ist    = 2
  s%wslake = 2200.0
  s%snowh  = 0.280
  s%sneqv  = 70.0
  s%snice  = (/ 0.0, 25.0, 33.0 /)
  s%snliq  = (/ 0.0, 6.0, 6.0 /)
  s%stc(-1) = 272.0
  s%stc(0)  = 273.16
  s%dzsnso(-1) = 0.100
  s%dzsnso(0)  = 0.180
  s%ficeold = (/ 0.0, 1.0, 1.0 /)
  s%imelt   = (/ 2, 2, 2 /)
  s%sfctmp  = 274.0
  s%qsnow   = 0.0
  s%snowhin = 0.0
  s%qrain   = 1.0e-4
  s%qvap    = 1.0e-5
  call run_case('wat_lake_under_snow', p, s)

! ---------------------------------------------------------------------------
! Inertness probes.  Each is its baseline with every argument the pinned
! identity does not consume perturbed; every output must be bit-identical.
! ---------------------------------------------------------------------------

  ! Never referenced anywhere in WATER's body: UU, VV, QPRECC, QPRECL, FP,
  ! RAIN, SNOW, LATHEAV, LATHEAG, IRRFRA.
  ! Referenced only inside dead branches: SMCEQ (opt_run=5 SHALLOWWATERTABLE),
  ! WA/WT (opt_run=1 GROUNDWATER), RECH (opt_run=5), MIFAC/FIFAC/IRFIRATE/
  ! IRMIRATE/CROPLU (irrigation, unreachable with IRAMTFI=IRAMTMI=0).
  ! Forwarded to routines where they are themselves inert: VEGTYP, TG, ILOC,
  ! JLOC, DX, TDFRACMP, ZWT, SMCWTD, DEEPRECH.
  ! Overwritten before it is read: QTLDRN (6112).
  ! Write-only: ZSNSO, which SNOWWATER rebuilds from ISNOW, DZSNSO and ZSOIL
  ! (6522-6533) without ever reading its entry value.
  ! IRAMTFI/IRAMTMI cannot be positive under opt_irr=0 (see the header), so the
  ! only non-zero value the fixture is allowed to give them is a negative one.
  ! That is not a forecast state and is not claimed to be: it measures that the
  ! 6188/6196 gates are `.GT. 0.0` rather than `.NE. 0.0`, which is the only
  ! part of them a fixture can reach at all.
  ! Baseline: wat_bare_rain.
  s = base_state()
  s%uu = -8.75;      s%vv = 6.5
  s%qprecc = 7.5e-4; s%qprecl = 2.5e-4
  s%fp = 0.11;       s%rain = 9.5e-4;  s%snow = 3.5e-4
  s%latheav = 2.8440e06
  s%latheag = 2.8440e06
  s%irrfra = 0.77;   s%mifac = 0.31;   s%fifac = 0.52
  s%croplu = .true.
  s%irfirate = 0.61; s%irmirate = 0.29
  s%smceq = (/ 0.031, 0.317, 0.113, 0.409 /)
  s%wa = 137.0;      s%wt = 251.0;     s%rech = -0.917
  s%vegtyp = 27;     s%tg = 301.75
  s%iloc = 41;       s%jloc = 97
  s%dx = 250.0;      s%tdfracmp = 0.9
  s%zwt = 8.75;      s%smcwtd = 0.717; s%deeprech = 0.331
  s%qtldrn = 4.25
  s%zsnso = (/ -1.31, -0.77, -0.23, 3.19, 7.41, 5.03, 9.11 /)
  s%iramtfi = -0.5;  s%iramtmi = -0.25
  call run_case('wat_inert_probe', p, s)

  ! The same probe on the lake branch, where a different set of statements is
  ! live: SOILWATER is not called, so ZWT/SMCWTD/DEEPRECH/DX/TDFRACMP are
  ! inert for a second reason and the claim is measured twice.
  ! Baseline: wat_lake_filling.
  s = base_state()
  s%ist    = 2
  s%wslake = 1200.0
  s%uu = -8.75;      s%vv = 6.5
  s%qprecc = 7.5e-4; s%qprecl = 2.5e-4
  s%fp = 0.11;       s%rain = 9.5e-4;  s%snow = 3.5e-4
  s%latheav = 2.8440e06
  s%latheag = 2.8440e06
  s%irrfra = 0.77;   s%mifac = 0.31;   s%fifac = 0.52
  s%croplu = .true.
  s%irfirate = 0.61; s%irmirate = 0.29
  s%smceq = (/ 0.031, 0.317, 0.113, 0.409 /)
  s%wa = 137.0;      s%wt = 251.0;     s%rech = -0.917
  s%vegtyp = 27;     s%tg = 301.75
  s%iloc = 41;       s%jloc = 97
  s%dx = 250.0;      s%tdfracmp = 0.9
  s%zwt = 8.75;      s%smcwtd = 0.717; s%deeprech = 0.331
  s%qtldrn = 4.25
  s%zsnso = (/ -1.31, -0.77, -0.23, 3.19, 7.41, 5.03, 9.11 /)
  s%iramtfi = -0.5;  s%iramtmi = -0.25
  call run_case('wat_lake_inert_probe', p, s)

  ! The INTENT(OUT) entry values, moved.  RUNSRF/RUNSUB are zeroed at
  ! 6109-6110 so their entry values must not survive; QIN/QDIS are never
  ! written at all so theirs must survive exactly; QSNSUB/QSNFRO are
  ! unconditionally assigned so theirs must not.
  ! Baseline: wat_bare_rain.
  s = base_state()
  s%runsrf = -47.5
  s%runsub = -23.25
  s%qin    = 5.5
  s%qdis   = -9.75
  s%qsnsub = 44.0
  s%qsnfro = 88.0
  call run_case('wat_out_entry_probe', p, s)

  ! The same on a pack, where QSNSUB/QSNFRO take their other branch and
  ! SNOFLOW is live.
  ! Baseline: wat_snow_sublimation.
  s = snow_state()
  s%qvap = 1.0e-5
  s%qrain = 2.0e-5
  s%runsrf = -47.5
  s%runsub = -23.25
  s%qin    = 5.5
  s%qdis   = -9.75
  s%qsnsub = 44.0
  s%qsnfro = 88.0
  call run_case('wat_snow_out_entry_probe', p, s)

  close(CSV)
  close(PRB)
  write(*, '(A)') 'run_water: wrote '//trim(csv_path)//' and '//trim(probe_path)

end program run_water
