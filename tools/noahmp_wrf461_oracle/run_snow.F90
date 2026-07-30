! Per-leaf oracle for the Noah-MP snowpack layer routines of WRF v4.6.1
! MODULE_SF_NOAHMPLSM: SNOWFALL, COMPACT, COMBINE, DIVIDE, COMBO, SNOWH2O and
! their driver SNOWWATER.
!
! The module source is compiled exactly as it ships apart from the
! accessibility lift produced by snow_visibility_patch.py, whose text diff is
! nothing but `private` -> `public ` and whose object code is proven
! byte-identical to the pristine compile.  No expression in the physics is
! touched.
!
! Fixture convention
! ------------------
! Every case emits its complete entry state and its complete exit state, so a
! row is reproducible from the CSV alone and any argument the leaf does not
! touch is visibly identical across the two stages.  Reals carry both a
! round-tripping decimal form and the raw IEEE binary32 bit pattern; the
! consumer compares bit patterns, so no parsing tolerance exists anywhere.
!
! DZSNSO sign convention: positive layer thickness on entry to every snow
! leaf, which is what WATER hands SNOWWATER and what SNOWWATER restores before
! returning.  Index 0 is the bottom snow layer, ISNOW+1 the top.
!
! PONDING1/PONDING2 hazard: COMBINE declares both INTENT(OUT) but assigns them
! on only some paths.  gfortran passes scalar dummies by reference, so an
! unassigned path leaves the caller's value standing.  Every COMBINE case is
! therefore entered with a -999.0 sentinel and both the entry and exit values
! are recorded, which pins the observed behaviour instead of assuming it.

module snow_oracle
  use module_sf_noahmplsm, only : noahmp_parameters, &
                                  SNOWFALL, COMPACT, COMBINE, DIVIDE, &
                                  COMBO, SNOWH2O, SNOWWATER
  implicit none

  integer, parameter :: NSNOW = 3
  integer, parameter :: NSOIL = 4
  integer, parameter :: CSV   = 21
  integer, parameter :: ILOC  = 1
  integer, parameter :: JLOC  = 1

  ! Pinned Registry/MPTABLE identity.  Soil interfaces -0.10/-0.40/-1.00/-2.00 m.
  real, parameter :: ZSOIL_PIN(NSOIL) = (/ -0.10, -0.40, -1.00, -2.00 /)
  real, parameter :: SSI_PIN          = 0.03
  real, parameter :: SNOW_RET_FAC_PIN = 5.0e-5

  real, parameter :: SENTINEL = -999.0

  type :: snow_state
     integer :: isnow
     real    :: snowh
     real    :: sneqv
     real    :: snice(-NSNOW+1:0)
     real    :: snliq(-NSNOW+1:0)
     real    :: stc(-NSNOW+1:NSOIL)
     real    :: zsnso(-NSNOW+1:NSOIL)
     real    :: dzsnso(-NSNOW+1:NSOIL)
     real    :: sh2o(NSOIL)
     real    :: sice(NSOIL)
  end type snow_state

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

  subroutine emit_ia(leaf, cname, stage, field, lo, hi, a)
    character(len=*), intent(in) :: leaf, cname, stage, field
    integer, intent(in) :: lo, hi
    integer, intent(in) :: a(lo:hi)
    integer :: k
    do k = lo, hi
       call emit_i(leaf, cname, stage, field, k, a(k))
    end do
  end subroutine emit_ia

  subroutine emit_state(leaf, cname, stage, st)
    character(len=*), intent(in) :: leaf, cname, stage
    type(snow_state), intent(in) :: st
    call emit_i (leaf, cname, stage, 'isnow',  0, st%isnow)
    call emit_r (leaf, cname, stage, 'snowh',  0, st%snowh)
    call emit_r (leaf, cname, stage, 'sneqv',  0, st%sneqv)
    call emit_ra(leaf, cname, stage, 'snice',  -NSNOW+1, 0,     st%snice)
    call emit_ra(leaf, cname, stage, 'snliq',  -NSNOW+1, 0,     st%snliq)
    call emit_ra(leaf, cname, stage, 'stc',    -NSNOW+1, NSOIL, st%stc)
    call emit_ra(leaf, cname, stage, 'zsnso',  -NSNOW+1, NSOIL, st%zsnso)
    call emit_ra(leaf, cname, stage, 'dzsnso', -NSNOW+1, NSOIL, st%dzsnso)
    call emit_ra(leaf, cname, stage, 'sh2o',   1, NSOIL, st%sh2o)
    call emit_ra(leaf, cname, stage, 'sice',   1, NSOIL, st%sice)
  end subroutine emit_state

! ---------------------------------------------------------------------------
! State construction
! ---------------------------------------------------------------------------

  function base_state() result(st)
    type(snow_state) :: st
    st%isnow  = 0
    st%snowh  = 0.0
    st%sneqv  = 0.0
    st%snice  = 0.0
    st%snliq  = 0.0
    st%stc    = (/ 0.0, 0.0, 0.0, 271.0, 272.5, 275.0, 280.0 /)
    st%zsnso  = (/ 0.0, 0.0, 0.0, -0.10, -0.40, -1.00, -2.00 /)
    st%dzsnso = (/ 0.0, 0.0, 0.0, 0.10, 0.30, 0.60, 1.00 /)
    st%sh2o   = (/ 0.20, 0.22, 0.25, 0.28 /)
    st%sice   = (/ 0.05, 0.03, 0.02, 0.01 /)
  end function base_state

! ---------------------------------------------------------------------------
! Leaf drivers
! ---------------------------------------------------------------------------

  subroutine case_snowfall(cname, st_in, dt, qsnow, snowhin, sfctmp)
    character(len=*), intent(in) :: cname
    type(snow_state), intent(in) :: st_in
    real, intent(in) :: dt, qsnow, snowhin, sfctmp
    type(snow_state) :: st
    type(noahmp_parameters) :: p
    character(len=*), parameter :: L = 'snowfall'

    p%SSI = SSI_PIN
    p%SNOW_RET_FAC = SNOW_RET_FAC_PIN
    st = st_in

    call emit_i(L, cname, 'input', 'nsnow',   0, NSNOW)
    call emit_i(L, cname, 'input', 'nsoil',   0, NSOIL)
    call emit_r(L, cname, 'input', 'dt',      0, dt)
    call emit_r(L, cname, 'input', 'qsnow',   0, qsnow)
    call emit_r(L, cname, 'input', 'snowhin', 0, snowhin)
    call emit_r(L, cname, 'input', 'sfctmp',  0, sfctmp)
    call emit_state(L, cname, 'input', st)

    call SNOWFALL(p, NSOIL, NSNOW, dt, qsnow, snowhin, sfctmp, ILOC, JLOC, &
                  st%isnow, st%snowh, st%dzsnso, st%stc, st%snice, &
                  st%snliq, st%sneqv)

    call emit_state(L, cname, 'output', st)
  end subroutine case_snowfall

  subroutine case_compact(cname, st_in, dt, imelt, ficeold)
    character(len=*), intent(in) :: cname
    type(snow_state), intent(in) :: st_in
    real,    intent(in) :: dt
    integer, intent(in) :: imelt(-NSNOW+1:0)
    real,    intent(in) :: ficeold(-NSNOW+1:0)
    type(snow_state) :: st
    type(noahmp_parameters) :: p
    character(len=*), parameter :: L = 'compact'

    p%SSI = SSI_PIN
    p%SNOW_RET_FAC = SNOW_RET_FAC_PIN
    st = st_in

    call emit_i (L, cname, 'input', 'nsnow', 0, NSNOW)
    call emit_i (L, cname, 'input', 'nsoil', 0, NSOIL)
    call emit_r (L, cname, 'input', 'dt',    0, dt)
    call emit_ia(L, cname, 'input', 'imelt',   -NSNOW+1, 0, imelt)
    call emit_ra(L, cname, 'input', 'ficeold', -NSNOW+1, 0, ficeold)
    call emit_ra(L, cname, 'input', 'zsoil',   1, NSOIL, ZSOIL_PIN)
    call emit_state(L, cname, 'input', st)

    call COMPACT(p, NSNOW, NSOIL, dt, st%stc, st%snice, st%snliq, ZSOIL_PIN, &
                 imelt, ficeold, ILOC, JLOC, st%isnow, st%dzsnso, st%zsnso)

    call emit_state(L, cname, 'output', st)
  end subroutine case_compact

  subroutine case_combine(cname, st_in)
    character(len=*), intent(in) :: cname
    type(snow_state), intent(in) :: st_in
    type(snow_state) :: st
    type(noahmp_parameters) :: p
    real :: ponding1, ponding2
    character(len=*), parameter :: L = 'combine'

    p%SSI = SSI_PIN
    p%SNOW_RET_FAC = SNOW_RET_FAC_PIN
    st = st_in
    ponding1 = SENTINEL
    ponding2 = SENTINEL

    call emit_i(L, cname, 'input', 'nsnow', 0, NSNOW)
    call emit_i(L, cname, 'input', 'nsoil', 0, NSOIL)
    call emit_r(L, cname, 'input', 'ponding1', 0, ponding1)
    call emit_r(L, cname, 'input', 'ponding2', 0, ponding2)
    call emit_state(L, cname, 'input', st)

    call COMBINE(p, NSNOW, NSOIL, ILOC, JLOC, st%isnow, st%sh2o, st%stc, &
                 st%snice, st%snliq, st%dzsnso, st%sice, st%snowh, st%sneqv, &
                 ponding1, ponding2)

    call emit_r(L, cname, 'output', 'ponding1', 0, ponding1)
    call emit_r(L, cname, 'output', 'ponding2', 0, ponding2)
    call emit_state(L, cname, 'output', st)
  end subroutine case_combine

  subroutine case_divide(cname, st_in)
    character(len=*), intent(in) :: cname
    type(snow_state), intent(in) :: st_in
    type(snow_state) :: st
    type(noahmp_parameters) :: p
    character(len=*), parameter :: L = 'divide'

    p%SSI = SSI_PIN
    p%SNOW_RET_FAC = SNOW_RET_FAC_PIN
    st = st_in

    call emit_i(L, cname, 'input', 'nsnow', 0, NSNOW)
    call emit_i(L, cname, 'input', 'nsoil', 0, NSOIL)
    call emit_state(L, cname, 'input', st)

    call DIVIDE(p, NSNOW, NSOIL, st%isnow, st%stc, st%snice, st%snliq, st%dzsnso)

    call emit_state(L, cname, 'output', st)
  end subroutine case_divide

  subroutine case_combo(cname, dz, wliq, wice, t, dz2, wliq2, wice2, t2)
    character(len=*), intent(in) :: cname
    real, intent(in) :: dz, wliq, wice, t, dz2, wliq2, wice2, t2
    real :: a_dz, a_wliq, a_wice, a_t
    type(noahmp_parameters) :: p
    character(len=*), parameter :: L = 'combo'

    p%SSI = SSI_PIN
    p%SNOW_RET_FAC = SNOW_RET_FAC_PIN
    a_dz = dz; a_wliq = wliq; a_wice = wice; a_t = t

    call emit_r(L, cname, 'input', 'dz',    0, a_dz)
    call emit_r(L, cname, 'input', 'wliq',  0, a_wliq)
    call emit_r(L, cname, 'input', 'wice',  0, a_wice)
    call emit_r(L, cname, 'input', 't',     0, a_t)
    call emit_r(L, cname, 'input', 'dz2',   0, dz2)
    call emit_r(L, cname, 'input', 'wliq2', 0, wliq2)
    call emit_r(L, cname, 'input', 'wice2', 0, wice2)
    call emit_r(L, cname, 'input', 't2',    0, t2)

    call COMBO(p, a_dz, a_wliq, a_wice, a_t, dz2, wliq2, wice2, t2)

    call emit_r(L, cname, 'output', 'dz',   0, a_dz)
    call emit_r(L, cname, 'output', 'wliq', 0, a_wliq)
    call emit_r(L, cname, 'output', 'wice', 0, a_wice)
    call emit_r(L, cname, 'output', 't',    0, a_t)
  end subroutine case_combo

  subroutine case_snowh2o(cname, st_in, dt, qsnfro, qsnsub, qrain, ssi, srf)
    character(len=*), intent(in) :: cname
    type(snow_state), intent(in) :: st_in
    real, intent(in) :: dt, qsnfro, qsnsub, qrain, ssi, srf
    type(snow_state) :: st
    type(noahmp_parameters) :: p
    real :: qsnbot, ponding1, ponding2
    character(len=*), parameter :: L = 'snowh2o'

    p%SSI = ssi
    p%SNOW_RET_FAC = srf
    st = st_in
    qsnbot = SENTINEL
    ponding1 = SENTINEL
    ponding2 = SENTINEL

    call emit_i(L, cname, 'input', 'nsnow',  0, NSNOW)
    call emit_i(L, cname, 'input', 'nsoil',  0, NSOIL)
    call emit_r(L, cname, 'input', 'dt',     0, dt)
    call emit_r(L, cname, 'input', 'qsnfro', 0, qsnfro)
    call emit_r(L, cname, 'input', 'qsnsub', 0, qsnsub)
    call emit_r(L, cname, 'input', 'qrain',  0, qrain)
    call emit_r(L, cname, 'input', 'ssi',    0, ssi)
    call emit_r(L, cname, 'input', 'snow_ret_fac', 0, srf)
    call emit_r(L, cname, 'input', 'ponding1', 0, ponding1)
    call emit_r(L, cname, 'input', 'ponding2', 0, ponding2)
    call emit_state(L, cname, 'input', st)

    call SNOWH2O(p, NSNOW, NSOIL, dt, qsnfro, qsnsub, qrain, ILOC, JLOC, &
                 st%isnow, st%dzsnso, st%snowh, st%sneqv, st%snice, &
                 st%snliq, st%sh2o, st%sice, st%stc, &
                 qsnbot, ponding1, ponding2)

    call emit_r(L, cname, 'output', 'qsnbot',   0, qsnbot)
    call emit_r(L, cname, 'output', 'ponding1', 0, ponding1)
    call emit_r(L, cname, 'output', 'ponding2', 0, ponding2)
    call emit_state(L, cname, 'output', st)
  end subroutine case_snowh2o

  subroutine case_snowwater(cname, st_in, dt, imelt, ficeold, sfctmp, snowhin, &
                            qsnow, qsnfro, qsnsub, qrain)
    character(len=*), intent(in) :: cname
    type(snow_state), intent(in) :: st_in
    real,    intent(in) :: dt, sfctmp, snowhin, qsnow, qsnfro, qsnsub, qrain
    integer, intent(in) :: imelt(-NSNOW+1:0)
    real,    intent(in) :: ficeold(-NSNOW+1:0)
    type(snow_state) :: st
    type(noahmp_parameters) :: p
    real :: qsnbot, snoflow, ponding1, ponding2
    character(len=*), parameter :: L = 'snowwater'

    p%SSI = SSI_PIN
    p%SNOW_RET_FAC = SNOW_RET_FAC_PIN
    st = st_in
    qsnbot = SENTINEL
    snoflow = SENTINEL
    ponding1 = SENTINEL
    ponding2 = SENTINEL

    call emit_i (L, cname, 'input', 'nsnow',  0, NSNOW)
    call emit_i (L, cname, 'input', 'nsoil',  0, NSOIL)
    call emit_r (L, cname, 'input', 'dt',     0, dt)
    call emit_r (L, cname, 'input', 'sfctmp', 0, sfctmp)
    call emit_r (L, cname, 'input', 'snowhin',0, snowhin)
    call emit_r (L, cname, 'input', 'qsnow',  0, qsnow)
    call emit_r (L, cname, 'input', 'qsnfro', 0, qsnfro)
    call emit_r (L, cname, 'input', 'qsnsub', 0, qsnsub)
    call emit_r (L, cname, 'input', 'qrain',  0, qrain)
    call emit_r (L, cname, 'input', 'ssi',    0, SSI_PIN)
    call emit_r (L, cname, 'input', 'snow_ret_fac', 0, SNOW_RET_FAC_PIN)
    call emit_ia(L, cname, 'input', 'imelt',   -NSNOW+1, 0, imelt)
    call emit_ra(L, cname, 'input', 'ficeold', -NSNOW+1, 0, ficeold)
    call emit_ra(L, cname, 'input', 'zsoil',   1, NSOIL, ZSOIL_PIN)
    call emit_state(L, cname, 'input', st)

    call SNOWWATER(p, NSNOW, NSOIL, imelt, dt, ZSOIL_PIN, sfctmp, snowhin, &
                   qsnow, qsnfro, qsnsub, qrain, ficeold, ILOC, JLOC, &
                   st%isnow, st%snowh, st%sneqv, st%snice, st%snliq, &
                   st%sh2o, st%sice, st%stc, st%zsnso, st%dzsnso, &
                   qsnbot, snoflow, ponding1, ponding2)

    call emit_r(L, cname, 'output', 'qsnbot',   0, qsnbot)
    call emit_r(L, cname, 'output', 'snoflow',  0, snoflow)
    call emit_r(L, cname, 'output', 'ponding1', 0, ponding1)
    call emit_r(L, cname, 'output', 'ponding2', 0, ponding2)
    call emit_state(L, cname, 'output', st)
  end subroutine case_snowwater

end module snow_oracle


program run_snow
  use snow_oracle
  implicit none

  character(len=256) :: outfile
  type(snow_state)   :: s
  integer :: imelt0(-NSNOW+1:0), imelt1(-NSNOW+1:0), imeltm(-NSNOW+1:0)
  real    :: fice_hi(-NSNOW+1:0), fice_zero(-NSNOW+1:0), fice_mid(-NSNOW+1:0)
  real    :: fice_high(-NSNOW+1:0)

  if (command_argument_count() /= 1) then
     write(*, '(A)') 'usage: run_snow OUTPUT.csv'
     error stop 2
  end if
  call get_command_argument(1, outfile)

  open(CSV, file=trim(outfile), status='replace', action='write')
  write(CSV, '(A)') 'leaf,case,stage,field,index,dtype,value,bits'

  imelt0 = 0
  imelt1 = 1
  imeltm = (/ 0, 1, 0 /)
  fice_hi   = 1.0
  fice_zero = 0.0
  fice_mid  = (/ 0.90, 0.75, 0.60 /)
  fice_high = 0.95

! ===========================================================================
! COMBO -- all three enthalpy branches plus both equality boundaries
! ===========================================================================

  ! HC < 0: dry cold ice, branch 1.
  call case_combo('combo_cold_dry', 0.05, 0.0, 20.0, 263.0, 0.03, 0.0, 12.0, 258.0)
  ! HC < 0 with liquid present: branch 1 with non-zero WLIQC.
  call case_combo('combo_cold_wet', 0.05, 0.5, 20.0, 265.0, 0.03, 0.2, 12.0, 260.0)
  ! T == T2 == TFRZ with liquid: HC == HFUS*WLIQC exactly, branch 2 boundary.
  call case_combo('combo_isothermal_wet', 0.06, 1.5, 25.0, 273.16, 0.04, 0.8, 18.0, 273.16)
  ! T == T2 == TFRZ, no liquid: HC == 0 == HFUS*WLIQC, branch 2 at zero.
  call case_combo('combo_isothermal_dry', 0.06, 0.0, 25.0, 273.16, 0.04, 0.0, 18.0, 273.16)
  ! HC > HFUS*WLIQC with liquid: branch 3.
  call case_combo('combo_warm_wet', 0.05, 2.0, 15.0, 274.5, 0.05, 1.0, 10.0, 275.0)
  ! HC > 0, WLIQC == 0: branch 3 with a zero latent term.
  call case_combo('combo_warm_dry', 0.05, 0.0, 15.0, 274.5, 0.05, 0.0, 10.0, 276.0)

! ===========================================================================
! SNOWFALL -- the 0 -> -1 layer creation and both no-transition paths
! ===========================================================================

  ! ISNOW=0, QSNOW>0, SNOWH stays below 0.025: accumulation only, no new node.
  s = base_state()
  s%isnow = 0; s%snowh = 0.010; s%sneqv = 2.0
  call case_snowfall('snowfall_shallow_accum', s, 1800.0, 1.0e-4, 1.0e-6, 265.0)

  ! ISNOW=0, accumulation crosses 0.025: creates layer 0.  TRANSITION 0 -> -1.
  ! SFCTMP < TFRZ so STC(0) takes the SFCTMP side of MIN(273.16, SFCTMP).
  s = base_state()
  s%isnow = 0; s%snowh = 0.024; s%sneqv = 5.0
  call case_snowfall('snowfall_new_layer_cold', s, 1800.0, 2.0e-4, 2.0e-6, 265.0)

  ! Same transition with SFCTMP > TFRZ: STC(0) takes the 273.16 side of MIN.
  s = base_state()
  s%isnow = 0; s%snowh = 0.024; s%sneqv = 5.0
  call case_snowfall('snowfall_new_layer_warm', s, 1800.0, 2.0e-4, 2.0e-6, 280.0)

  ! ISNOW=0, QSNOW=0, SNOWH already >= 0.025: C.He's removal of the QSNOW>0
  ! guard means a layer is still created.  TRANSITION 0 -> -1 with no snowfall.
  s = base_state()
  s%isnow = 0; s%snowh = 0.030; s%sneqv = 6.0
  call case_snowfall('snowfall_new_layer_no_precip', s, 1800.0, 0.0, 0.0, 270.0)

  ! ISNOW<0 and SNOWH >= 0.025: proves the ISNOW==0 guard blocks a second
  ! creation, and exercises the add-to-top-layer branch.
  s = base_state()
  s%isnow = -2; s%snowh = 0.28; s%sneqv = 60.0
  s%snice = (/ 0.0, 25.0, 30.0 /)
  s%snliq = (/ 0.0, 1.0, 2.0 /)
  s%stc(-1) = 264.0; s%stc(0) = 268.0
  s%dzsnso(-1) = 0.10; s%dzsnso(0) = 0.18
  call case_snowfall('snowfall_add_to_top', s, 1800.0, 1.5e-4, 1.2e-6, 262.0)

  ! ISNOW<0, QSNOW=0: every branch off.  Negative control.
  s = base_state()
  s%isnow = -1; s%snowh = 0.0; s%sneqv = 40.0
  s%snice(0) = 38.0; s%snliq(0) = 2.0
  s%stc(0) = 266.0; s%dzsnso(0) = 0.20
  call case_snowfall('snowfall_layered_dry', s, 1800.0, 0.0, 0.0, 260.0)

  ! ISNOW=0, no snow, no snowfall: total no-op.  Negative control.
  s = base_state()
  call case_snowfall('snowfall_bare_dry', s, 1800.0, 0.0, 0.0, 285.0)

! ===========================================================================
! COMPACT -- every guard, both clamps, the melt term and its FICEOLD guard
! ===========================================================================

  ! Three cold dry layers, no melt, BI below DM: base settling path, and the
  ! burden accumulates down the pack so layer 0 sees a different DDZ2.
  s = base_state()
  s%isnow = -3
  s%snice = (/ 8.0, 14.0, 22.0 /)
  s%snliq = (/ 0.0, 0.0, 0.0 /)
  s%stc(-2) = 253.0; s%stc(-1) = 258.0; s%stc(0) = 265.0
  s%dzsnso(-2) = 0.10; s%dzsnso(-1) = 0.16; s%dzsnso(0) = 0.25
  call case_compact('compact_cold_dry_3layer', s, 1800.0, imelt0, fice_hi)

  ! BI = SNICE/DZ above DM = 100: the exp(-46e-3*(BI-DM)) attenuation fires on
  ! layers -1 and 0 but not on layer -2.
  s = base_state()
  s%isnow = -3
  s%snice = (/ 8.0, 30.0, 60.0 /)
  s%snliq = (/ 0.0, 0.0, 0.0 /)
  s%stc(-2) = 255.0; s%stc(-1) = 262.0; s%stc(0) = 270.0
  s%dzsnso(-2) = 0.12; s%dzsnso(-1) = 0.20; s%dzsnso(0) = 0.30
  call case_compact('compact_dense_burden', s, 1800.0, imelt0, fice_hi)

  ! SNLIQ(J) > 0.01*DZSNSO(J) on every layer: the C5 liquid-water multiplier.
  s = base_state()
  s%isnow = -3
  s%snice = (/ 9.0, 15.0, 24.0 /)
  s%snliq = (/ 1.2, 2.0, 3.0 /)
  s%stc(-2) = 268.0; s%stc(-1) = 271.0; s%stc(0) = 273.16
  s%dzsnso(-2) = 0.11; s%dzsnso(-1) = 0.17; s%dzsnso(0) = 0.26
  call case_compact('compact_wet_c5', s, 1800.0, imelt0, fice_hi)

  ! IMELT=1 with FICEOLD above the current ice fraction: DDZ3 negative.
  s = base_state()
  s%isnow = -3
  s%snice = (/ 9.0, 15.0, 24.0 /)
  s%snliq = (/ 1.5, 2.5, 4.0 /)
  s%stc(-2) = 271.0; s%stc(-1) = 272.5; s%stc(0) = 273.16
  s%dzsnso(-2) = 0.11; s%dzsnso(-1) = 0.17; s%dzsnso(0) = 0.26
  call case_compact('compact_melting', s, 1800.0, imelt1, fice_hi)

  ! Mixed IMELT: only layer -1 melts, so DDZ3 is zero on -2 and 0 and non-zero
  ! on -1.  Binds the IMELT(J) indexing, not just the flag.
  s = base_state()
  s%isnow = -3
  s%snice = (/ 9.0, 15.0, 24.0 /)
  s%snliq = (/ 1.5, 2.5, 4.0 /)
  s%stc(-2) = 271.0; s%stc(-1) = 272.5; s%stc(0) = 273.16
  s%dzsnso(-2) = 0.11; s%dzsnso(-1) = 0.17; s%dzsnso(0) = 0.26
  call case_compact('compact_melt_middle_only', s, 1800.0, imeltm, fice_mid)

  ! IMELT=1 with FICEOLD=0: MAX(1.0E-6,FICEOLD) divides, and FICEOLD-FICE is
  ! negative so MAX(0.0,...) returns zero.  Both guards bound at once.
  s = base_state()
  s%isnow = -2
  s%snice = (/ 0.0, 12.0, 20.0 /)
  s%snliq = (/ 0.0, 1.0, 2.0 /)
  s%stc(-1) = 270.0; s%stc(0) = 272.0
  s%dzsnso(-1) = 0.14; s%dzsnso(0) = 0.22
  call case_compact('compact_melt_ficeold_zero', s, 1800.0, imelt1, fice_zero)

  ! FICEOLD far above FICE with a short DT: PDZDTC below -0.5 and clamped.
  s = base_state()
  s%isnow = -1
  s%snice = (/ 0.0, 0.0, 6.0 /)
  s%snliq = (/ 0.0, 0.0, 6.0 /)
  s%stc(0) = 273.0
  s%dzsnso(0) = 0.12
  call case_compact('compact_pdzdtc_clamp', s, 60.0, imelt1, fice_hi)

  ! VOID <= 0.001: a fully saturated layer is skipped entirely.
  s = base_state()
  s%isnow = -1
  s%snice = (/ 0.0, 0.0, 90.0 /)
  s%snliq = (/ 0.0, 0.0, 6.0 /)
  s%stc(0) = 270.0
  s%dzsnso(0) = 0.1041439
  call case_compact('compact_skip_saturated', s, 1800.0, imelt0, fice_hi)

  ! SNICE(J) <= 0.1: the ice-mass guard skips the layer.
  s = base_state()
  s%isnow = -1
  s%snice = (/ 0.0, 0.0, 0.08 /)
  s%snliq = (/ 0.0, 0.0, 0.02 /)
  s%stc(0) = 268.0
  s%dzsnso(0) = 0.030
  call case_compact('compact_skip_thin_ice', s, 1800.0, imelt0, fice_hi)

  ! Fluffy layer: the MIN(...,(SNICE+SNLIQ)/50) upper clamp binds.
  s = base_state()
  s%isnow = -1
  s%snice = (/ 0.0, 0.0, 3.0 /)
  s%snliq = (/ 0.0, 0.0, 0.0 /)
  s%stc(0) = 250.0
  s%dzsnso(0) = 0.40
  call case_compact('compact_density_floor_50', s, 1800.0, imelt0, fice_hi)

  ! Very dense layer: the MAX(...,(SNICE+SNLIQ)/500) lower clamp binds.
  s = base_state()
  s%isnow = -1
  s%snice = (/ 0.0, 0.0, 45.0 /)
  s%snliq = (/ 0.0, 0.0, 0.5 /)
  s%stc(0) = 272.0
  s%dzsnso(0) = 0.080
  call case_compact('compact_density_cap_500', s, 1800.0, imelt0, fice_hi)

! ===========================================================================
! COMBINE -- layer-count transitions downward, both merge mechanisms
! ===========================================================================

  ! ISNOW=-3, nothing thin: no merge at all, MSSI walks 1->2->3 so DZMIN(2)
  ! and DZMIN(3) are both read.  Layer count unchanged.
  s = base_state()
  s%isnow = -3
  s%snice = (/ 6.0, 12.0, 30.0 /)
  s%snliq = (/ 0.2, 0.4, 1.0 /)
  s%stc(-2) = 255.0; s%stc(-1) = 262.0; s%stc(0) = 270.0
  s%dzsnso(-2) = 0.04; s%dzsnso(-1) = 0.06; s%dzsnso(0) = 0.15
  call case_combine('combine_noop_3layer', s)

  ! SNICE(-1) <= 0.1: mid-pack thin-ice merge into layer 0 plus the shift.
  ! TRANSITION -3 -> -2.
  s = base_state()
  s%isnow = -3
  s%snice = (/ 6.0, 0.05, 30.0 /)
  s%snliq = (/ 0.2, 0.03, 1.0 /)
  s%stc(-2) = 255.0; s%stc(-1) = 262.0; s%stc(0) = 270.0
  s%dzsnso(-2) = 0.05; s%dzsnso(-1) = 0.004; s%dzsnso(0) = 0.15
  call case_combine('combine_thin_ice_mid', s)

  ! SNICE(0) <= 0.1 with ISNOW < -1: bottom layer merges upward into -1 and
  ! the shift runs over two indices.  TRANSITION -3 -> -2.
  s = base_state()
  s%isnow = -3
  s%snice = (/ 6.0, 12.0, 0.06 /)
  s%snliq = (/ 0.2, 0.4, 0.02 /)
  s%stc(-2) = 255.0; s%stc(-1) = 262.0; s%stc(0) = 270.0
  s%dzsnso(-2) = 0.05; s%dzsnso(-1) = 0.07; s%dzsnso(0) = 0.005
  call case_combine('combine_thin_ice_bottom_multi', s)

  ! SNICE(0) <= 0.1 with ISNOW == -1 and SNICE >= 0: the pack becomes ponding
  ! and PONDING1 is assigned while PONDING2 is not.  TRANSITION -1 -> 0.
  s = base_state()
  s%isnow = -1
  s%snice = (/ 0.0, 0.0, 0.07 /)
  s%snliq = (/ 0.0, 0.0, 0.4 /)
  s%stc(0) = 272.0
  s%dzsnso(0) = 0.006
  call case_combine('combine_last_layer_to_ponding', s)

  ! SNICE(0) < 0 (over-sublimated) but SNLIQ covers it: PONDING1 > 0, no soil
  ! adjustment, SNEQV and SNOWH both zeroed.  TRANSITION -1 -> 0.
  s = base_state()
  s%isnow = -1
  s%snice = (/ 0.0, 0.0, -0.05 /)
  s%snliq = (/ 0.0, 0.0, 0.30 /)
  s%stc(0) = 272.0
  s%dzsnso(0) = 0.004
  call case_combine('combine_oversub_ponding_positive', s)

  ! SNICE(0) < 0 and SNLIQ cannot cover it: PONDING1 < 0 is charged to SICE(1),
  ! which stays positive.  TRANSITION -1 -> 0.
  s = base_state()
  s%isnow = -1
  s%snice = (/ 0.0, 0.0, -0.50 /)
  s%snliq = (/ 0.0, 0.0, 0.20 /)
  s%stc(0) = 272.0
  s%dzsnso(0) = 0.004
  call case_combine('combine_oversub_charges_sice', s)

  ! Same, but the soil ice cannot absorb the debit: SICE(1) goes negative and
  ! the SH2O(1) fixup fires.  TRANSITION -1 -> 0.
  s = base_state()
  s%isnow = -1
  s%snice = (/ 0.0, 0.0, -0.50 /)
  s%snliq = (/ 0.0, 0.0, 0.20 /)
  s%sice = (/ 0.001, 0.03, 0.02, 0.01 /)
  s%stc(0) = 272.0
  s%dzsnso(0) = 0.004
  call case_combine('combine_oversub_sh2o_fixup', s)

  ! No thin layer, but the pack is shallower than 0.025 m: the whole pack
  ! collapses to ponding with SNEQV = ZWICE > 0.  TRANSITION -2 -> 0.
  s = base_state()
  s%isnow = -2
  s%snice = (/ 0.0, 1.0, 1.0 /)
  s%snliq = (/ 0.0, 0.05, 0.05 /)
  s%stc(-1) = 271.0; s%stc(0) = 272.0
  s%dzsnso(-1) = 0.010; s%dzsnso(0) = 0.010
  call case_combine('combine_collapse_thin_pack', s)

  ! The bottom layer is over-sublimated and merges upward into an already
  ! visited slot, so the surviving pack carries SNICE < 0 and ZWICE <= 0.
  ! This is the only route to `IF(SNEQV <= 0.0) SNOWH = 0.0`.
  ! TRANSITION -3 -> -2 -> 0.
  s = base_state()
  s%isnow = -3
  s%snice = (/ 0.11, 0.15, -0.40 /)
  s%snliq = (/ 0.02, 0.03, 0.01 /)
  s%stc(-2) = 268.0; s%stc(-1) = 270.0; s%stc(0) = 272.0
  s%dzsnso(-2) = 0.008; s%dzsnso(-1) = 0.009; s%dzsnso(0) = 0.006
  call case_combine('combine_collapse_negative_ice', s)

  ! Thin top layer by thickness (not ice mass): I == ISNOW+1 takes NEIBOR=I+1.
  ! TRANSITION -3 -> -2.
  s = base_state()
  s%isnow = -3
  s%snice = (/ 2.0, 12.0, 30.0 /)
  s%snliq = (/ 0.1, 0.4, 1.0 /)
  s%stc(-2) = 255.0; s%stc(-1) = 262.0; s%stc(0) = 270.0
  s%dzsnso(-2) = 0.024; s%dzsnso(-1) = 0.060; s%dzsnso(0) = 0.150
  call case_combine('combine_thin_dz_top', s)

  ! Thin bottom layer by thickness: I == 0 takes NEIBOR=I-1 and the K-shift
  ! runs.  DZMIN(3)=0.1 is the threshold that fires.  TRANSITION -3 -> -2.
  s = base_state()
  s%isnow = -3
  s%snice = (/ 6.0, 12.0, 4.0 /)
  s%snliq = (/ 0.2, 0.4, 0.2 /)
  s%stc(-2) = 255.0; s%stc(-1) = 262.0; s%stc(0) = 270.0
  s%dzsnso(-2) = 0.040; s%dzsnso(-1) = 0.060; s%dzsnso(0) = 0.030
  call case_combine('combine_thin_dz_bottom', s)

  ! Thin middle layer, the neighbour test picks the layer ABOVE (NEIBOR=I-1).
  ! TRANSITION -3 -> -2.
  s = base_state()
  s%isnow = -3
  s%snice = (/ 6.0, 2.0, 30.0 /)
  s%snliq = (/ 0.2, 0.1, 1.0 /)
  s%stc(-2) = 255.0; s%stc(-1) = 262.0; s%stc(0) = 270.0
  s%dzsnso(-2) = 0.030; s%dzsnso(-1) = 0.024; s%dzsnso(0) = 0.200
  call case_combine('combine_thin_dz_mid_up', s)

  ! Thin middle layer, the neighbour test picks the layer BELOW (NEIBOR=I+1),
  ! which also triggers the K-shift.  TRANSITION -3 -> -2.
  s = base_state()
  s%isnow = -3
  s%snice = (/ 30.0, 2.0, 6.0 /)
  s%snliq = (/ 1.0, 0.1, 0.2 /)
  s%stc(-2) = 255.0; s%stc(-1) = 262.0; s%stc(0) = 270.0
  s%dzsnso(-2) = 0.300; s%dzsnso(-1) = 0.024; s%dzsnso(0) = 0.050
  call case_combine('combine_thin_dz_mid_down', s)

  ! Two successive thickness merges drive ISNOW to -1 and the EXIT fires
  ! before layer 0 is examined.  TRANSITION -3 -> -1.
  s = base_state()
  s%isnow = -3
  s%snice = (/ 1.0, 2.0, 30.0 /)
  s%snliq = (/ 0.05, 0.1, 1.0 /)
  s%stc(-2) = 255.0; s%stc(-1) = 262.0; s%stc(0) = 270.0
  s%dzsnso(-2) = 0.005; s%dzsnso(-1) = 0.010; s%dzsnso(0) = 0.200
  call case_combine('combine_double_merge_exit', s)

  ! ISNOW = -2 with both layers thick: no merge, MSSI walks 1->2, ISNOW held.
  s = base_state()
  s%isnow = -2
  s%snice = (/ 0.0, 12.0, 30.0 /)
  s%snliq = (/ 0.0, 0.4, 1.0 /)
  s%stc(-1) = 262.0; s%stc(0) = 270.0
  s%dzsnso(-1) = 0.060; s%dzsnso(0) = 0.150
  call case_combine('combine_noop_2layer', s)

  ! ISNOW = 0 on entry: the J loop is empty and the early RETURN fires, so
  ! neither PONDING is ever assigned.  Negative control for the sentinel.
  s = base_state()
  s%isnow = 0; s%snowh = 0.012; s%sneqv = 3.0
  call case_combine('combine_already_zero', s)

! ===========================================================================
! DIVIDE -- layer-count transitions upward
! ===========================================================================

  ! Single layer thinner than 0.05: nothing happens.  Negative control.
  s = base_state()
  s%isnow = -1
  s%snice = (/ 0.0, 0.0, 10.0 /)
  s%snliq = (/ 0.0, 0.0, 0.5 /)
  s%stc(0) = 268.0
  s%dzsnso(0) = 0.040
  call case_divide('divide_single_thin_noop', s)

  ! Single layer above 0.05: split in two, then the top is trimmed to 0.05 and
  ! the remainder combined downward.  TRANSITION -1 -> -2.
  s = base_state()
  s%isnow = -1
  s%snice = (/ 0.0, 0.0, 24.0 /)
  s%snliq = (/ 0.0, 0.0, 1.2 /)
  s%stc(0) = 266.0
  s%dzsnso(0) = 0.120
  call case_divide('divide_single_to_two', s)

  ! Deep single layer: split to two, trim, then DZ(2) > 0.20 promotes to three
  ! and the DZ(2) > 0.2 trim also fires.  TRANSITION -1 -> -3.
  s = base_state()
  s%isnow = -1
  s%snice = (/ 0.0, 0.0, 150.0 /)
  s%snliq = (/ 0.0, 0.0, 3.0 /)
  s%stc(0) = 261.0
  s%dzsnso(0) = 0.600
  call case_divide('divide_single_to_three', s)

  ! Two layers, top above 0.05: trim and combine, DZ(2) stays below 0.20 so
  ! the layer count is unchanged.
  s = base_state()
  s%isnow = -2
  s%snice = (/ 0.0, 18.0, 22.0 /)
  s%snliq = (/ 0.0, 0.9, 1.1 /)
  s%stc(-1) = 262.0; s%stc(0) = 269.0
  s%dzsnso(-1) = 0.100; s%dzsnso(0) = 0.110
  call case_divide('divide_two_trim_top', s)

  ! Two layers, DZ(2) crosses 0.20 after the trim: promotion to three with
  ! TSNO(3) below TFRZ, so the ELSE re-warms TSNO(2).  TRANSITION -2 -> -3.
  s = base_state()
  s%isnow = -2
  s%snice = (/ 0.0, 60.0, 20.0 /)
  s%snliq = (/ 0.0, 1.0, 0.5 /)
  s%stc(-1) = 272.0; s%stc(0) = 260.0
  s%dzsnso(-1) = 0.300; s%dzsnso(0) = 0.100
  call case_divide('divide_two_to_three_cold', s)

  ! Same promotion but with a strongly inverted profile, so TSNO(3) >= TFRZ
  ! and the IF side of that test fires instead.  TRANSITION -2 -> -3.
  s = base_state()
  s%isnow = -2
  s%snice = (/ 0.0, 60.0, 20.0 /)
  s%snliq = (/ 0.0, 1.0, 0.5 /)
  s%stc(-1) = 240.0; s%stc(0) = 272.6
  s%dzsnso(-1) = 0.300; s%dzsnso(0) = 0.100
  call case_divide('divide_two_to_three_inverted', s)

  ! Three layers, top already at the 0.05 limit so the MSNO>1 block is skipped,
  ! but DZ(2) > 0.2 so the MSNO>2 block runs on its own.
  s = base_state()
  s%isnow = -3
  s%snice = (/ 10.0, 70.0, 40.0 /)
  s%snliq = (/ 0.5, 2.0, 1.5 /)
  s%stc(-2) = 258.0; s%stc(-1) = 264.0; s%stc(0) = 270.0
  s%dzsnso(-2) = 0.050; s%dzsnso(-1) = 0.300; s%dzsnso(0) = 0.250
  call case_divide('divide_three_trim_mid', s)

  ! Three layers already within limits: nothing happens.  Negative control.
  s = base_state()
  s%isnow = -3
  s%snice = (/ 10.0, 40.0, 40.0 /)
  s%snliq = (/ 0.5, 1.5, 1.5 /)
  s%stc(-2) = 258.0; s%stc(-1) = 264.0; s%stc(0) = 270.0
  s%dzsnso(-2) = 0.050; s%dzsnso(-1) = 0.200; s%dzsnso(0) = 0.250
  call case_divide('divide_three_noop', s)

  ! Three layers with a thick top: the MSNO>1 trim runs, the MSNO<=2 promotion
  ! does not (MSNO is already 3), then the MSNO>2 trim runs.
  s = base_state()
  s%isnow = -3
  s%snice = (/ 30.0, 40.0, 40.0 /)
  s%snliq = (/ 1.0, 1.5, 1.5 /)
  s%stc(-2) = 256.0; s%stc(-1) = 263.0; s%stc(0) = 270.0
  s%dzsnso(-2) = 0.150; s%dzsnso(-1) = 0.180; s%dzsnso(0) = 0.250
  call case_divide('divide_three_trim_top_and_mid', s)

! ===========================================================================
! SNOWH2O -- sublimation bookkeeping, the internal COMBINE, and percolation
! ===========================================================================

  ! SNEQV == 0: the frost/sublimation debit lands directly on SICE(1).
  s = base_state()
  s%isnow = 0; s%snowh = 0.0; s%sneqv = 0.0
  call case_snowh2o('snowh2o_sneqv_zero_frost', s, 1800.0, 2.0e-5, 0.0, 0.0, &
                    SSI_PIN, SNOW_RET_FAC_PIN)

  ! SNEQV == 0 and the debit exceeds SICE(1): the SH2O(1) fixup fires.
  s = base_state()
  s%isnow = 0; s%snowh = 0.0; s%sneqv = 0.0
  s%sice = (/ 0.002, 0.03, 0.02, 0.01 /)
  call case_snowh2o('snowh2o_sneqv_zero_deficit', s, 1800.0, 0.0, 5.0e-4, 0.0, &
                    SSI_PIN, SNOW_RET_FAC_PIN)

  ! ISNOW==0 with a sub-layer pack: proportional depth loss, both density
  ! clamps inactive.
  s = base_state()
  s%isnow = 0; s%snowh = 0.020; s%sneqv = 5.0
  call case_snowh2o('snowh2o_shallow_sublimation', s, 1800.0, 0.0, 1.0e-4, 0.0, &
                    SSI_PIN, SNOW_RET_FAC_PIN)

  ! Same path but the pack is fluffier than 50 kg/m3: MIN clamps to SNEQV/50.
  s = base_state()
  s%isnow = 0; s%snowh = 0.200; s%sneqv = 5.0
  call case_snowh2o('snowh2o_shallow_density_floor', s, 1800.0, 0.0, 1.0e-4, 0.0, &
                    SSI_PIN, SNOW_RET_FAC_PIN)

  ! Same path but denser than 500 kg/m3: MAX clamps to SNEQV/500.
  s = base_state()
  s%isnow = 0; s%snowh = 0.004; s%sneqv = 5.0
  call case_snowh2o('snowh2o_shallow_density_cap', s, 1800.0, 0.0, 1.0e-4, 0.0, &
                    SSI_PIN, SNOW_RET_FAC_PIN)

  ! Sublimation exceeds the whole sub-layer pack: SNEQV goes negative, is
  ! charged to SICE(1), and both SNEQV and SNOWH are zeroed.
  s = base_state()
  s%isnow = 0; s%snowh = 0.020; s%sneqv = 5.0
  call case_snowh2o('snowh2o_shallow_oversublimation', s, 1800.0, 0.0, 4.0e-3, 0.0, &
                    SSI_PIN, SNOW_RET_FAC_PIN)

  ! Same, with SICE(1) too small to absorb it: the SH2O(1) fixup fires too.
  s = base_state()
  s%isnow = 0; s%snowh = 0.020; s%sneqv = 5.0
  s%sice = (/ 0.001, 0.03, 0.02, 0.01 /)
  call case_snowh2o('snowh2o_shallow_oversub_sh2o', s, 1800.0, 0.0, 4.0e-3, 0.0, &
                    SSI_PIN, SNOW_RET_FAC_PIN)

  ! SNEQV below 1.0e-6 with a finite SNOWH: the joint zeroing test fires.
  s = base_state()
  s%isnow = 0; s%snowh = 0.001; s%sneqv = 5.0e-7
  call case_snowh2o('snowh2o_trace_zeroed', s, 1800.0, 0.0, 0.0, 0.0, &
                    SSI_PIN, SNOW_RET_FAC_PIN)

  ! Layered pack, no COMBINE: rain loads the top layer and percolates.  Both
  ! layers hold liquid below SSI so QOUT is zero above the base.
  s = base_state()
  s%isnow = -2; s%snowh = 0.28; s%sneqv = 60.0
  s%snice = (/ 0.0, 25.0, 33.0 /)
  s%snliq = (/ 0.0, 0.5, 0.8 /)
  s%stc(-1) = 268.0; s%stc(0) = 271.0
  s%dzsnso(-1) = 0.100; s%dzsnso(0) = 0.180
  call case_snowh2o('snowh2o_layered_rain', s, 1800.0, 0.0, 0.0, 1.0e-4, &
                    SSI_PIN, SNOW_RET_FAC_PIN)

  ! Wet pack: VOL_LIQ exceeds SSI*EPORE so QOUT is positive in both layers and
  ! the J==0 SNOW_RET_FAC branch takes the retention side of its MAX.
  s = base_state()
  s%isnow = -2; s%snowh = 0.28; s%sneqv = 70.0
  s%snice = (/ 0.0, 25.0, 33.0 /)
  s%snliq = (/ 0.0, 6.0, 6.0 /)
  s%stc(-1) = 272.0; s%stc(0) = 273.16
  s%dzsnso(-1) = 0.100; s%dzsnso(0) = 0.180
  call case_snowh2o('snowh2o_percolation_retention', s, 1800.0, 0.0, 0.0, 0.0, &
                    SSI_PIN, SNOW_RET_FAC_PIN)

  ! Same pack with SNOW_RET_FAC raised so the other side of the J==0 MAX wins.
  s = base_state()
  s%isnow = -2; s%snowh = 0.28; s%sneqv = 70.0
  s%snice = (/ 0.0, 25.0, 33.0 /)
  s%snliq = (/ 0.0, 6.0, 6.0 /)
  s%stc(-1) = 272.0; s%stc(0) = 273.16
  s%dzsnso(-1) = 0.100; s%dzsnso(0) = 0.180
  call case_snowh2o('snowh2o_percolation_release', s, 1800.0, 0.0, 0.0, 0.0, &
                    SSI_PIN, 1.0e-2)

  ! SSI raised so nothing drains: negative control on the SSI argument.
  s = base_state()
  s%isnow = -2; s%snowh = 0.28; s%sneqv = 70.0
  s%snice = (/ 0.0, 25.0, 33.0 /)
  s%snliq = (/ 0.0, 6.0, 6.0 /)
  s%stc(-1) = 272.0; s%stc(0) = 273.16
  s%dzsnso(-1) = 0.100; s%dzsnso(0) = 0.180
  call case_snowh2o('snowh2o_percolation_held', s, 1800.0, 0.0, 0.0, 0.0, &
                    0.30, SNOW_RET_FAC_PIN)

  ! Very wet top layer: SNLIQ/(SNICE+SNLIQ) exceeds 0.4 and the extra QOUT
  ! term fires.
  s = base_state()
  s%isnow = -2; s%snowh = 0.28; s%sneqv = 40.0
  s%snice = (/ 0.0, 6.0, 20.0 /)
  s%snliq = (/ 0.0, 9.0, 4.0 /)
  s%stc(-1) = 273.16; s%stc(0) = 273.16
  s%dzsnso(-1) = 0.100; s%dzsnso(0) = 0.180
  call case_snowh2o('snowh2o_max_liq_fraction', s, 1800.0, 0.0, 0.0, 0.0, &
                    SSI_PIN, SNOW_RET_FAC_PIN)

  ! Ice-saturated layer: MIN(1.0, SNICE/(DZ*DENICE)) hits the clamp, EPORE=0.
  s = base_state()
  s%isnow = -1; s%snowh = 0.10; s%sneqv = 95.0
  s%snice = (/ 0.0, 0.0, 95.0 /)
  s%snliq = (/ 0.0, 0.0, 0.5 /)
  s%stc(0) = 270.0
  s%dzsnso(0) = 0.100
  call case_snowh2o('snowh2o_vol_ice_clamped', s, 1800.0, 0.0, 0.0, 0.0, &
                    SSI_PIN, SNOW_RET_FAC_PIN)

  ! Sublimation drives WGDIF below 1.0e-6 in a two-layer pack: SNOWH2O calls
  ! COMBINE internally.  TRANSITION -2 -> -1 inside the leaf.
  s = base_state()
  s%isnow = -2; s%snowh = 0.21; s%sneqv = 33.0
  s%snice = (/ 0.0, 0.05, 32.0 /)
  s%snliq = (/ 0.0, 0.02, 0.9 /)
  s%stc(-1) = 265.0; s%stc(0) = 270.0
  s%dzsnso(-1) = 0.030; s%dzsnso(0) = 0.180
  call case_snowh2o('snowh2o_internal_combine', s, 1800.0, 0.0, 1.0e-4, 0.0, &
                    SSI_PIN, SNOW_RET_FAC_PIN)

  ! Same mechanism on a single thin layer: the internal COMBINE drives ISNOW to
  ! 0, and the KWM guard then blocks the SNLIQ rain update.
  ! TRANSITION -1 -> 0 inside the leaf.
  s = base_state()
  s%isnow = -1; s%snowh = 0.006; s%sneqv = 0.5
  s%snice = (/ 0.0, 0.0, 0.4 /)
  s%snliq = (/ 0.0, 0.0, 0.1 /)
  s%stc(0) = 272.0
  s%dzsnso(0) = 0.006
  call case_snowh2o('snowh2o_internal_combine_to_zero', s, 1800.0, 0.0, 3.0e-4, &
                    2.0e-4, SSI_PIN, SNOW_RET_FAC_PIN)

  ! Frost onto a layered pack: WGDIF grows, no COMBINE, and the final
  ! DZSNSO = MAX(DZSNSO, SNLIQ/DENH2O + SNICE/DENICE) expansion fires.
  s = base_state()
  s%isnow = -1; s%snowh = 0.030; s%sneqv = 26.0
  s%snice = (/ 0.0, 0.0, 25.0 /)
  s%snliq = (/ 0.0, 0.0, 1.0 /)
  s%stc(0) = 268.0
  s%dzsnso(0) = 0.026
  call case_snowh2o('snowh2o_dz_expansion', s, 1800.0, 1.0e-3, 0.0, 0.0, &
                    SSI_PIN, SNOW_RET_FAC_PIN)

! ===========================================================================
! SNOWWATER -- the composed driver, both transition directions
! ===========================================================================

  ! Bare ground receiving snow: SNOWFALL creates layer 0, then COMPACT,
  ! COMBINE and DIVIDE all run on the fresh layer.  TRANSITION 0 -> -1.
  s = base_state()
  s%isnow = 0; s%snowh = 0.024; s%sneqv = 5.0
  call case_snowwater('snowwater_bare_to_one_layer', s, 1800.0, imelt0, fice_hi, &
                      265.0, 2.0e-6, 2.0e-4, 0.0, 0.0, 0.0)

  ! Deep single layer: DIVIDE splits it.  TRANSITION -1 -> -2.
  s = base_state()
  s%isnow = -1; s%snowh = 0.120; s%sneqv = 25.0
  s%snice = (/ 0.0, 0.0, 24.0 /)
  s%snliq = (/ 0.0, 0.0, 1.0 /)
  s%stc(0) = 266.0
  s%dzsnso(0) = 0.120
  call case_snowwater('snowwater_one_to_two_layers', s, 1800.0, imelt0, fice_hi, &
                      264.0, 0.0, 0.0, 0.0, 0.0, 0.0)

  ! Two layers, the upper one thin in ice: COMBINE merges it away and the
  ! merged layer stays under the 0.05 m DIVIDE limit, so the reduction stands.
  ! TRANSITION -2 -> -1.
  s = base_state()
  s%isnow = -2; s%snowh = 0.040; s%sneqv = 8.37
  s%snice = (/ 0.0, 0.05, 8.0 /)
  s%snliq = (/ 0.0, 0.02, 0.30 /)
  s%stc(-1) = 265.0; s%stc(0) = 270.0
  s%dzsnso(-1) = 0.010; s%dzsnso(0) = 0.030
  call case_snowwater('snowwater_two_to_one_layer', s, 1800.0, imelt0, fice_hi, &
                      266.0, 0.0, 0.0, 0.0, 0.0, 0.0)

  ! The same merge on a deep pack: COMBINE takes ISNOW to -1 and DIVIDE
  ! immediately splits the 0.21 m survivor back to -2.  The composed layer
  ! count is unchanged, which is exactly why SNOWWATER cannot be validated by
  ! its layer count alone.
  s = base_state()
  s%isnow = -2; s%snowh = 0.21; s%sneqv = 33.0
  s%snice = (/ 0.0, 0.05, 32.0 /)
  s%snliq = (/ 0.0, 0.02, 0.9 /)
  s%stc(-1) = 265.0; s%stc(0) = 270.0
  s%dzsnso(-1) = 0.030; s%dzsnso(0) = 0.180
  call case_snowwater('snowwater_combine_then_divide', s, 1800.0, imelt0, fice_hi, &
                      266.0, 0.0, 0.0, 0.0, 0.0, 0.0)

  ! A single thin layer melts out entirely: COMBINE returns it as ponding.
  ! TRANSITION -1 -> 0.
  s = base_state()
  s%isnow = -1; s%snowh = 0.006; s%sneqv = 0.5
  s%snice = (/ 0.0, 0.0, 0.07 /)
  s%snliq = (/ 0.0, 0.0, 0.40 /)
  s%stc(0) = 272.5
  s%dzsnso(0) = 0.006
  call case_snowwater('snowwater_one_layer_to_bare', s, 1800.0, imelt1, fice_hi, &
                      274.0, 0.0, 0.0, 0.0, 0.0, 1.0e-4)

  ! SNEQV above 5000 mm: the glacier SNOFLOW branch fires.
  s = base_state()
  s%isnow = -3; s%snowh = 12.05; s%sneqv = 5400.0
  s%snice = (/ 40.0, 360.0, 4980.0 /)
  s%snliq = (/ 1.0, 4.0, 15.0 /)
  s%stc(-2) = 250.0; s%stc(-1) = 258.0; s%stc(0) = 266.0
  s%dzsnso(-2) = 0.150; s%dzsnso(-1) = 0.900; s%dzsnso(0) = 11.000
  call case_snowwater('snowwater_glacier_snoflow', s, 1800.0, imelt0, fice_hi, &
                      252.0, 0.0, 0.0, 0.0, 0.0, 0.0)

  ! Three-layer pack in steady state: no transition, but the ZSNSO/DZSNSO
  ! rebuild and the multi-layer SNEQV/SNOWH re-sum all run.
  s = base_state()
  s%isnow = -3; s%snowh = 0.41; s%sneqv = 66.0
  s%snice = (/ 6.0, 18.0, 40.0 /)
  s%snliq = (/ 0.2, 0.6, 1.2 /)
  s%stc(-2) = 255.0; s%stc(-1) = 262.0; s%stc(0) = 269.0
  s%dzsnso(-2) = 0.050; s%dzsnso(-1) = 0.110; s%dzsnso(0) = 0.250
  call case_snowwater('snowwater_three_layer_steady', s, 1800.0, imeltm, fice_mid, &
                      258.0, 1.0e-6, 1.0e-4, 1.0e-6, 2.0e-6, 5.0e-5)

  ! No snow anywhere: only the DZSNSO/ZSNSO rebuild runs.  Negative control.
  s = base_state()
  call case_snowwater('snowwater_bare_dry', s, 1800.0, imelt0, fice_hi, &
                      285.0, 0.0, 0.0, 0.0, 0.0, 0.0)

! ===========================================================================
! Threshold boundary cases.  Every constant these leaves compare against gets
! a pair of cases that straddle it, so a mutant that moves the threshold is
! detected rather than merely untested.  The mutation study in
! tests/test_noahmp_snow.py is what forced these to exist.
! ===========================================================================

  ! SNOWH exactly at the 0.025 m layer-creation limit: `>=` creates a layer.
  s = base_state()
  s%isnow = 0; s%snowh = 0.025; s%sneqv = 5.0
  call case_snowfall('snowfall_limit_exact', s, 1800.0, 0.0, 0.0, 268.0)

  ! One float32 ULP below it: no layer.  The pair pins the comparison itself.
  s = base_state()
  s%isnow = 0; s%snowh = 2.499999851e-02; s%sneqv = 5.0
  call case_snowfall('snowfall_limit_just_below', s, 1800.0, 0.0, 0.0, 268.0)

  ! SNICE exactly at the 0.1 mm merge limit: `<=` merges the layer away.
  s = base_state()
  s%isnow = -2
  s%snice = (/ 0.0, 0.1, 30.0 /)
  s%snliq = (/ 0.0, 0.03, 1.0 /)
  s%stc(-1) = 262.0; s%stc(0) = 270.0
  s%dzsnso(-1) = 0.030; s%dzsnso(0) = 0.150
  call case_combine('combine_ice_limit_exact', s)

  ! Just above it: the layer survives.
  s = base_state()
  s%isnow = -2
  s%snice = (/ 0.0, 0.1001, 30.0 /)
  s%snliq = (/ 0.0, 0.03, 1.0 /)
  s%stc(-1) = 262.0; s%stc(0) = 270.0
  s%dzsnso(-1) = 0.030; s%dzsnso(0) = 0.150
  call case_combine('combine_ice_limit_just_above', s)

  ! DZSNSO(0) between 0.05 and DZMIN(3)=0.1: merges only because the third
  ! DZMIN entry is 0.1, so a mutant that reuses 0.025 or 0.05 is caught.
  s = base_state()
  s%isnow = -3
  s%snice = (/ 6.0, 12.0, 8.0 /)
  s%snliq = (/ 0.2, 0.4, 0.3 /)
  s%stc(-2) = 255.0; s%stc(-1) = 262.0; s%stc(0) = 270.0
  s%dzsnso(-2) = 0.040; s%dzsnso(-1) = 0.060; s%dzsnso(0) = 0.070
  call case_combine('combine_dzmin3_boundary', s)

  ! Pack depth exactly at the 0.025 m collapse limit: `<` does not collapse.
  s = base_state()
  s%isnow = -1
  s%snice = (/ 0.0, 0.0, 4.0 /)
  s%snliq = (/ 0.0, 0.0, 0.2 /)
  s%stc(0) = 271.0
  s%dzsnso(0) = 0.025
  call case_combine('combine_collapse_limit_exact', s)

  ! DIVIDE's 0.05 m limit, straddled from above by two ULP of margin.
  s = base_state()
  s%isnow = -1
  s%snice = (/ 0.0, 0.0, 12.0 /)
  s%snliq = (/ 0.0, 0.0, 0.6 /)
  s%stc(0) = 267.0
  s%dzsnso(0) = 5.000000447e-02
  call case_divide('divide_limit_just_above', s)

  ! Exactly at it: `>` does not split.
  s = base_state()
  s%isnow = -1
  s%snice = (/ 0.0, 0.0, 12.0 /)
  s%snliq = (/ 0.0, 0.0, 0.6 /)
  s%stc(0) = 267.0
  s%dzsnso(0) = 0.05
  call case_divide('divide_limit_exact', s)

  ! WGDIF above the 1.0e-6 guard on a layer thin enough that COMBINE would
  ! have merged it: proves the guard, not the merge, is what held.
  s = base_state()
  s%isnow = -2; s%snowh = 0.21; s%sneqv = 33.0
  s%snice = (/ 0.0, 0.002, 32.0 /)
  s%snliq = (/ 0.0, 0.02, 0.9 /)
  s%stc(-1) = 265.0; s%stc(0) = 270.0
  s%dzsnso(-1) = 0.030; s%dzsnso(0) = 0.180
  call case_snowh2o('snowh2o_wgdif_above_limit', s, 1800.0, 0.0, 0.0, 0.0, &
                    SSI_PIN, SNOW_RET_FAC_PIN)

  ! SNEQV exactly at the 1.0e-6 trace limit: `<=` zeroes the pack.
  s = base_state()
  s%isnow = 0; s%snowh = 0.001; s%sneqv = 1.0e-6
  call case_snowh2o('snowh2o_trace_limit_exact', s, 1800.0, 0.0, 0.0, 0.0, &
                    SSI_PIN, SNOW_RET_FAC_PIN)

  ! Just above it, with SNOWH also above 1.0e-8: the pack survives.
  s = base_state()
  s%isnow = 0; s%snowh = 0.001; s%sneqv = 1.1e-6
  call case_snowh2o('snowh2o_trace_limit_just_above', s, 1800.0, 0.0, 0.0, 0.0, &
                    SSI_PIN, SNOW_RET_FAC_PIN)

! ===========================================================================
! One-ULP threshold straddles.  Every comparison constant these leaves use is
! bracketed by a pair of cases one float32 ULP apart, so any change to the
! constant -- down to its last bit -- flips a branch and is detected.  The
! mutation study is what established these were needed: a 0.1% perturbation of
! each constant survived the looser brackets that preceded them.
! ===========================================================================

  ! COMBINE DZMIN(1) = 0.025, tested at I = ISNOW+1 with MSSI = 1.
  s = base_state()
  s%isnow = -3
  s%snice = (/ 6.0, 12.0, 30.0 /)
  s%snliq = (/ 0.2, 0.4, 1.0 /)
  s%stc(-2) = 255.0; s%stc(-1) = 262.0; s%stc(0) = 270.0
  s%dzsnso(-2) = 2.499999851e-02; s%dzsnso(-1) = 0.060; s%dzsnso(0) = 0.150
  call case_combine('combine_dzmin1_below', s)

  s%dzsnso(-2) = 2.500000037e-02
  call case_combine('combine_dzmin1_exact', s)

  ! COMBINE DZMIN(2) = 0.025, reached at I = -1 only after I = -2 declines to
  ! merge and increments MSSI.
  s = base_state()
  s%isnow = -3
  s%snice = (/ 6.0, 12.0, 30.0 /)
  s%snliq = (/ 0.2, 0.4, 1.0 /)
  s%stc(-2) = 255.0; s%stc(-1) = 262.0; s%stc(0) = 270.0
  s%dzsnso(-2) = 0.040; s%dzsnso(-1) = 2.499999851e-02; s%dzsnso(0) = 0.150
  call case_combine('combine_dzmin2_below', s)

  s%dzsnso(-1) = 2.500000037e-02
  call case_combine('combine_dzmin2_exact', s)

  ! COMBINE DZMIN(3) = 0.1, reached at I = 0 with MSSI = 3.
  s = base_state()
  s%isnow = -3
  s%snice = (/ 6.0, 12.0, 30.0 /)
  s%snliq = (/ 0.2, 0.4, 1.0 /)
  s%stc(-2) = 255.0; s%stc(-1) = 262.0; s%stc(0) = 270.0
  s%dzsnso(-2) = 0.040; s%dzsnso(-1) = 0.060; s%dzsnso(0) = 9.999999404e-02
  call case_combine('combine_dzmin3_below', s)

  s%dzsnso(0) = 1.000000015e-01
  call case_combine('combine_dzmin3_exact', s)

  ! COMPACT's SNICE(J) > 0.1 ice-mass guard.
  s = base_state()
  s%isnow = -1
  s%snice = (/ 0.0, 0.0, 1.000000015e-01 /)
  s%snliq = (/ 0.0, 0.0, 0.0 /)
  s%stc(0) = 263.0
  s%dzsnso(0) = 0.050
  call case_compact('compact_ice_limit_exact', s, 1800.0, imelt0, fice_hi)

  s%snice = (/ 0.0, 0.0, 1.000000089e-01 /)
  call case_compact('compact_ice_limit_just_above', s, 1800.0, imelt0, fice_hi)

  ! COMPACT's VOID > 0.001 saturation guard.  DZSNSO is chosen so that VOID
  ! lands one ULP either side of 0.001 for SNICE = 20, SNLIQ = 0.
  s = base_state()
  s%isnow = -1
  s%snice = (/ 0.0, 0.0, 20.0 /)
  s%snliq = (/ 0.0, 0.0, 0.0 /)
  s%stc(0) = 263.0
  s%dzsnso(0) = 2.183208242e-02
  call case_compact('compact_void_limit_below', s, 1800.0, imelt0, fice_hi)

  s%dzsnso(0) = 2.183208428e-02
  call case_compact('compact_void_limit_above', s, 1800.0, imelt0, fice_hi)

  ! COMPACT's SNLIQ(J) > 0.01*DZSNSO(J) liquid-water test, straddled at
  ! DZSNSO = 0.15 where the threshold is exactly 1.500000013e-03.
  s = base_state()
  s%isnow = -1
  s%snice = (/ 0.0, 0.0, 15.0 /)
  s%snliq = (/ 0.0, 0.0, 1.500000013e-03 /)
  s%stc(0) = 263.0
  s%dzsnso(0) = 0.150
  call case_compact('compact_c5_limit_exact', s, 1800.0, imelt0, fice_hi)

  s%snliq = (/ 0.0, 0.0, 1.500000129e-03 /)
  call case_compact('compact_c5_limit_just_above', s, 1800.0, imelt0, fice_hi)

  ! DIVIDE's 0.05 m limit at the MSNO > 1 site.  ISNOW = -2 skips the
  ! MSNO == 1 block, so this is a different comparison from divide_limit_exact.
  s = base_state()
  s%isnow = -2
  s%snice = (/ 0.0, 12.0, 22.0 /)
  s%snliq = (/ 0.0, 0.6, 1.1 /)
  s%stc(-1) = 262.0; s%stc(0) = 269.0
  s%dzsnso(-1) = 5.000000075e-02; s%dzsnso(0) = 0.110
  call case_divide('divide_two_limit_exact', s)

  s%dzsnso(-1) = 5.000000447e-02
  call case_divide('divide_two_limit_just_above', s)

  ! DIVIDE's 0.20 m promotion limit.  DZ(1) = 0.15 trims to 0.05 leaving
  ! DRR = 1.000000089e-01, so DZ(2) lands exactly on 0.2 and one ULP above it.
  s = base_state()
  s%isnow = -2
  s%snice = (/ 0.0, 20.0, 31.0 /)
  s%snliq = (/ 0.0, 0.8, 1.2 /)
  s%stc(-1) = 268.0; s%stc(0) = 264.0
  s%dzsnso(-1) = 0.150; s%dzsnso(0) = 9.999999404e-02
  call case_divide('divide_promote_limit_exact', s)

  s%dzsnso(0) = 1.000000015e-01
  call case_divide('divide_promote_limit_just_above', s)

  ! DIVIDE's second 0.2 m limit, in the MSNO > 2 block.  DZ(1) is exactly at
  ! 0.05 so the MSNO > 1 block is skipped and DZ(2) is read straight through.
  s = base_state()
  s%isnow = -3
  s%snice = (/ 10.0, 70.0, 40.0 /)
  s%snliq = (/ 0.5, 2.0, 1.5 /)
  s%stc(-2) = 258.0; s%stc(-1) = 264.0; s%stc(0) = 270.0
  s%dzsnso(-2) = 0.050; s%dzsnso(-1) = 2.000000030e-01; s%dzsnso(0) = 0.250
  call case_divide('divide_three_mid_limit_exact', s)

  s%dzsnso(-1) = 2.000000179e-01
  call case_divide('divide_three_mid_limit_just_above', s)

  ! SNOWH2O's SNOWH <= 1.0e-8 trace test.  ISNOW < 0 so the shallow-snow block
  ! is skipped and SNOWH reaches the test unmodified.
  s = base_state()
  s%isnow = -1; s%snowh = 9.999999939e-09; s%sneqv = 1.0
  s%snice = (/ 0.0, 0.0, 20.0 /)
  s%snliq = (/ 0.0, 0.0, 0.5 /)
  s%stc(0) = 268.0
  s%dzsnso(0) = 0.100
  call case_snowh2o('snowh2o_snowh_limit_exact', s, 1800.0, 0.0, 0.0, 0.0, &
                    SSI_PIN, SNOW_RET_FAC_PIN)

  s%snowh = 1.000000083e-08
  call case_snowh2o('snowh2o_snowh_limit_just_above', s, 1800.0, 0.0, 0.0, 0.0, &
                    SSI_PIN, SNOW_RET_FAC_PIN)

  ! SNOWH2O's SNEQV <= 1.0e-6 trace test, likewise reached unmodified.
  s = base_state()
  s%isnow = -1; s%snowh = 0.100; s%sneqv = 9.999999975e-07
  s%snice = (/ 0.0, 0.0, 20.0 /)
  s%snliq = (/ 0.0, 0.0, 0.5 /)
  s%stc(0) = 268.0
  s%dzsnso(0) = 0.100
  call case_snowh2o('snowh2o_sneqv_limit_exact', s, 1800.0, 0.0, 0.0, 0.0, &
                    SSI_PIN, SNOW_RET_FAC_PIN)

  s%sneqv = 1.000000111e-06
  call case_snowh2o('snowh2o_sneqv_limit_just_above', s, 1800.0, 0.0, 0.0, 0.0, &
                    SSI_PIN, SNOW_RET_FAC_PIN)

  ! SNOWH2O's WGDIF < 1.0e-6 guard.  With no sublimation or frost WGDIF is
  ! SNICE(ISNOW+1) exactly, so the straddle is a straight one-ULP pair.
  s = base_state()
  s%isnow = -2; s%snowh = 0.200; s%sneqv = 30.0
  s%snice = (/ 0.0, 9.999999975e-07, 29.0 /)
  s%snliq = (/ 0.0, 0.020, 0.900 /)
  s%stc(-1) = 265.0; s%stc(0) = 270.0
  s%dzsnso(-1) = 0.030; s%dzsnso(0) = 0.170
  call case_snowh2o('snowh2o_wgdif_limit_exact', s, 1800.0, 0.0, 0.0, 0.0, &
                    SSI_PIN, SNOW_RET_FAC_PIN)

  s%snice = (/ 0.0, 9.999998838e-07, 29.0 /)
  call case_snowh2o('snowh2o_wgdif_limit_just_below', s, 1800.0, 0.0, 0.0, 0.0, &
                    SSI_PIN, SNOW_RET_FAC_PIN)

  ! SNOWWATER's SNEQV > 5000 mm glacier limit.  COMBINE re-sums SNEQV from a
  ! single layer, so it reaches the test as exactly SNICE(0).
  s = base_state()
  s%isnow = -1; s%snowh = 11.0; s%sneqv = 5000.0
  s%snice = (/ 0.0, 0.0, 5.000000000e+03 /)
  s%snliq = (/ 0.0, 0.0, 0.0 /)
  s%stc(0) = 258.0
  s%dzsnso(0) = 11.000
  call case_snowwater('snowwater_glacier_limit_exact', s, 1800.0, imelt0, fice_hi, &
                      252.0, 0.0, 0.0, 0.0, 0.0, 0.0)

  s%snice = (/ 0.0, 0.0, 5.000000488e+03 /)
  call case_snowwater('snowwater_glacier_limit_just_above', s, 1800.0, imelt0, &
                      fice_hi, 252.0, 0.0, 0.0, 0.0, 0.0, 0.0)

  ! SNOWWATER with a draining pack, so SSI and SNOW_RET_FAC actually reach the
  ! percolation loop.  Without this the composed driver cannot see either.
  s = base_state()
  s%isnow = -2; s%snowh = 0.280; s%sneqv = 70.0
  s%snice = (/ 0.0, 25.0, 33.0 /)
  s%snliq = (/ 0.0, 6.0, 6.0 /)
  s%stc(-1) = 272.0; s%stc(0) = 273.16
  s%dzsnso(-1) = 0.100; s%dzsnso(0) = 0.180
  call case_snowwater('snowwater_wet_percolation', s, 1800.0, imelt0, fice_hi, &
                      274.0, 0.0, 0.0, 0.0, 0.0, 1.0e-4)

  ! SNOWWATER with FICEOLD above the current ice fraction on a melting pack, so
  ! COMPACT's DDZ3 melt term is non-zero and FICEOLD becomes observable.
  s = base_state()
  s%isnow = -2; s%snowh = 0.230; s%sneqv = 33.0
  s%snice = (/ 0.0, 10.0, 12.0 /)
  s%snliq = (/ 0.0, 5.0, 6.0 /)
  s%stc(-1) = 272.5; s%stc(0) = 273.16
  s%dzsnso(-1) = 0.100; s%dzsnso(0) = 0.130
  call case_snowwater('snowwater_melt_ficeold', s, 1800.0, imelt1, fice_high, &
                      274.0, 0.0, 0.0, 0.0, 0.0, 0.0)

  close(CSV)
  write(*, '(A)') 'run_snow: wrote '//trim(outfile)

end program run_snow
