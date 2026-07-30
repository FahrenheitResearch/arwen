! Pin WRF v4.6.1 MODULE_SF_NOAHMPLSM::ERROR (module_sf_noahmplsm.F:1560-1737),
! the balance check NOAHMP_SFLX runs at 1049-1058, as raw FP32 bit patterns.
!
! Why this exists when noahmp-sflx.csv already exists
! ---------------------------------------------------
! The whole-column fixture pins ERROR only through the five ACC_* accumulators
! it mutates, and only on four columns that all balance.  Three things ERROR
! owns are invisible there:
!
!   * ERRWAT.  NOAHMP_SFLX declares it a local (625) and never reads it after
!     the call, so no whole-column comparison can constrain it.  It is the
!     quantity the abort at 1713 tests.
!   * The IST /= 1 lake branch (1733-1734), which zeroes ERRWAT and leaves all
!     five accumulators untouched.  Every noahmp-sflx.csv column is IST = 1.
!   * IRFIRATE and IRMIRATE, which enter ERRWAT scaled by 1000.0 (1709) and are
!     identically zero under the pinned opt_irr = 0.  A transcription that
!     dropped or mis-signed either would pass every existing check.
!
! Cases 1-16 are ordinary calls and produce the fixture.  Cases 90-92 each trip
! one of the three `wrf_error_fatal` gates; they are run one per process by
! build_sflx_compose.sh, which requires each to terminate.  A gate that cannot
! be shown to fire is not a gate, and a port that returned instead of aborting
! would be strictly more permissive than the module it claims to reproduce.
!
! `parameters` is ERROR's first dummy argument and its body contains no
! `parameters%` reference at all, so an undefined derived type is passed
! deliberately: nothing can read it, and giving it values would suggest
! otherwise.
!
! Option identity: dveg=4 opt_crs=1 opt_btr=1 opt_run=3 opt_sfc=1 opt_frz=1
! opt_inf=1 opt_rad=3 opt_alb=2 opt_snf=1 opt_tbot=2 opt_stc=1 opt_rsf=1
! opt_soil=1 opt_pedo=1 opt_crop=0 opt_irr=0 opt_irrm=0 opt_infdv=0 opt_tdrn=0,
! soiltstep=0.0 -> soil_update_steps=1, calculate_soil=.true.  ERROR reads only
! `calculate_soil` (1708) out of that set; case 16 drives it .false., which is
! the non-soil substep the ACC_* variables exist for.

program run_noahmp_sflx_compose_oracle
  use module_sf_noahmplsm, only: noahmp_parameters, noahmp_options, &
      noahmp_error => error, calculate_soil, soil_update_steps
  implicit none

  integer, parameter :: NSOIL = 4
  integer, parameter :: NSNOW = 3
  integer, parameter :: NCASE = 16
  real,    parameter :: ERRWAT_POISON = -9.0e30

  character(len=32), parameter :: NAMES(NCASE) = [character(len=32) :: &
      'err_balanced',           &
      'err_gaining_inside',     &
      'err_losing_inside',      &
      'err_lake',               &
      'err_lake_dirty_wb',      &
      'err_second_substep',     &
      'err_flood_irrigation',   &
      'err_micro_irrigation',   &
      'err_both_irrigation',    &
      'err_tile_drain',         &
      'err_runoff_both',        &
      'err_snow_and_canopy',    &
      'err_mixed_layer_signs',  &
      'err_negative_pah',       &
      'err_sprinkler_heat',     &
      'err_no_soil_substep']

  type(noahmp_parameters) :: parameters
  integer :: CSV, icase, k, nrun
  character(len=1024) :: path
  character(len=32)   :: arg
  logical :: single

  ! ERROR's argument list, in declaration order.
  integer :: nsoil_a, nsnow_a, ist, iloc, jloc
  real :: swdown, fsa, fsr, fira, fsh, fcev, fgev, fctr, ssoil
  real :: fveg, sav, sag, fsrv, fsrg, zwt
  real :: prcp, ecan, etran, edir, runsrf, runsub, qtldrn
  real :: canliq, canice, sneqv, wa, dt, beg_wb, errwat
  real :: pah, pahv, pahg, pahb, firr, canhs, irmirate, irfirate
  real :: acc_dwater, acc_prcp, acc_ecan, acc_etran, acc_edir
  real, dimension(1:NSOIL) :: smc
  real, dimension(-NSNOW+1:NSOIL) :: dzsnso

  call get_command_argument(1, path)
  if (len_trim(path) == 0) then
    write(*, '(A)') 'usage: run_sflx_compose OUTPUT.csv [ABORT_CASE]'
    error stop 2
  end if
  call get_command_argument(2, arg)
  single = len_trim(arg) > 0

  ! The first admitted option identity.  ERROR reads only calculate_soil out of
  ! it, but setting the whole thing keeps every harness in this directory
  ! describing the same model.
  call noahmp_options(4, 1, 1, 3, 1, 1, 1, 3, 2, 1, 2, 1, 1, 1, 1, 0, 0, 0, 0, 0)
  soil_update_steps = 1
  calculate_soil = .true.

  if (single) then
    read(arg, *) icase
    call build(icase)
    call invoke()
    write(*, '(A,I0,A)') 'case ', icase, ' returned without aborting'
    error stop 3
  end if

  open(newunit=CSV, file=trim(path), status='replace', action='write')
  write(CSV, '(A)') 'leaf,case,stage,field,index,dtype,value,bits'

  nrun = 0
  do icase = 1, NCASE
    call build(icase)
    call emit_inputs(NAMES(icase))
    call invoke()
    call emit_outputs(NAMES(icase))
    nrun = nrun + 1
  end do

  close(CSV)
  write(*, '(A,I0,A)') 'run_sflx_compose: ', nrun, ' ERROR cases written'

contains

! ---------------------------------------------------------------------------
! case construction
! ---------------------------------------------------------------------------
  subroutine build(j)
    integer, intent(in) :: j

    ! Shared reference column.  SMC*DZ*1000 summed over the four soil layers is
    ! 0.25 * 2.0 * 1000 = 500.0 mm, so BEG_WB = 500.0 closes the water budget
    ! exactly and every deviation below is a deliberate, named one.
    nsoil_a = NSOIL
    nsnow_a = NSNOW
    ist     = 1
    iloc    = 1
    jloc    = 1
    swdown  = 800.0
    fsa     = 600.0
    fsr     = 200.0
    sav     = 400.0
    sag     = 200.0
    fira    = 100.0
    fsh     = 200.0
    fcev    = 50.0
    fgev    = 150.0
    fctr    = 60.0
    ssoil   = 40.0
    firr    = 0.0
    canhs   = 0.0
    pah     = 0.0
    pahv    = 0.0
    pahg    = 0.0
    pahb    = 0.0
    fveg    = 0.8
    fsrv    = 40.0
    fsrg    = 160.0
    zwt     = 2.5
    dt      = 60.0
    prcp    = 0.0
    ecan    = 0.0
    etran   = 0.0
    edir    = 0.0
    runsrf  = 0.0
    runsub  = 0.0
    qtldrn  = 0.0
    canliq  = 0.0
    canice  = 0.0
    sneqv   = 0.0
    wa      = 0.0
    beg_wb  = 500.0
    smc     = 0.25
    dzsnso  = 0.0
    dzsnso(1) = 0.10
    dzsnso(2) = 0.30
    dzsnso(3) = 0.60
    dzsnso(4) = 1.00
    irmirate = 0.0
    irfirate = 0.0
    acc_dwater = 0.0
    acc_prcp   = 0.0
    acc_ecan   = 0.0
    acc_etran  = 0.0
    acc_edir   = 0.0
    calculate_soil = .true.
    ! ERRWAT is INTENT(OUT) and ERROR leaves it unassigned whenever IST == 1
    ! and calculate_soil is .false. (1708-1731).  gfortran passes scalars by
    ! reference, so the caller's value stands.  Poison it before every call, so
    ! case err_no_soil_substep records the poison and the fixture *says* that
    ! WRF wrote nothing there.
    errwat = ERRWAT_POISON

    select case (j)
    case (1)                    ! the reference column, all three residuals 0
    case (2)                    ! ERRWAT = +0.09, inside the 0.1 gate
      beg_wb = 500.0 - 0.09
    case (3)                    ! ERRWAT = -0.09, inside the 0.1 gate
      beg_wb = 500.0 + 0.09
    case (4)                    ! lake: the whole water block is skipped
      ist = 2
    case (5)                    ! lake with a water balance that would abort
      ist = 2
      beg_wb = 1.0e6
      acc_prcp = 7.0
      acc_dwater = 3.0
    case (6)                    ! a second soil substep on a live accumulator
      prcp   = 1.0e-3
      ecan   = 1.0e-4
      etran  = 2.0e-4
      edir   = 3.0e-4
      runsrf = 0.03
      acc_dwater = -0.03
      acc_prcp   = 0.06
      acc_ecan   = 0.006
      acc_etran  = 0.012
      acc_edir   = 0.018
      beg_wb = 500.0 + 0.006
    case (7)                    ! IRFIRATE alone, m/timestep scaled by 1000
      irfirate = 1.0e-5
      beg_wb = 500.0 - 0.01
    case (8)                    ! IRMIRATE alone
      irmirate = 2.0e-5
      beg_wb = 500.0 - 0.02
    case (9)                    ! both, so the two 1000.0 scalings cannot swap
      irfirate = 1.0e-5
      irmirate = 3.0e-5
      beg_wb = 500.0 - 0.04
    case (10)                   ! tile drainage, the last term of the residual
      qtldrn = 0.05
      beg_wb = 500.0 + 0.05
    case (11)                   ! surface and subsurface runoff together
      runsrf = 0.04
      runsub = 0.02
      beg_wb = 500.0 + 0.06
    case (12)                   ! END_WB's canopy and snow terms
      canliq = 0.15
      canice = 0.05
      sneqv  = 12.5
      wa     = 4900.0
      beg_wb = 500.0 + 0.15 + 0.05 + 12.5 + 4900.0
    case (13)                   ! layer contributions of unequal magnitude
      smc(1) = 0.05
      smc(2) = 0.44
      smc(3) = 0.19
      smc(4) = 0.31
      beg_wb = 0.05*0.10*1000.0 + 0.44*0.30*1000.0 &
             + 0.19*0.60*1000.0 + 0.31*1.00*1000.0
    case (14)                   ! PAH negative, the term added outside the sum
      pah  = -35.0
      pahv = -20.0
      pahg = -10.0
      pahb = -5.0
      ssoil = 40.0 - 35.0
    case (15)                   ! FIRR and CANHS, the two youngest sink terms
      firr  = 12.0
      canhs = -7.0
      ssoil = 40.0 - 12.0 + 7.0
    case (16)                   ! a non-soil substep: accumulate, do not close
      calculate_soil = .false.
      prcp   = 2.0e-3
      ecan   = 5.0e-5
      beg_wb = 480.0
    case (90)                   ! ABORT: shortwave, |ERRSW| = 0.02 > 0.01
      fsr = 200.0 + 0.02
    case (91)                   ! ABORT: energy, |ERRENG| = 0.02 > 0.01
      ssoil = 40.0 + 0.02
    case (92)                   ! ABORT: water, |ERRWAT| = 0.2 > 0.1
      beg_wb = 500.0 - 0.2
    case default
      write(*, '(A,I0)') 'unknown case ', j
      error stop 4
    end select
  end subroutine build

  subroutine invoke()
    call noahmp_error(parameters, swdown, fsa, fsr, fira, fsh, fcev, &
               fgev, fctr, ssoil, beg_wb, canliq, canice, &
               sneqv, wa, smc, dzsnso, prcp, ecan, &
               etran, edir, runsrf, runsub, dt, nsoil_a, &
               qtldrn, &
               nsnow_a, ist, errwat, iloc, jloc, fveg, &
               sav, sag, fsrv, fsrg, zwt, pah, &
               pahv, pahg, pahb, firr, canhs, &
               irmirate, irfirate, acc_dwater, acc_prcp, acc_ecan, &
               acc_etran, acc_edir)
  end subroutine invoke

! ---------------------------------------------------------------------------
! emission
! ---------------------------------------------------------------------------
  subroutine emit_inputs(cname)
    character(len=*), intent(in) :: cname
    call emit_i(cname, 'input', 'ist',    0, ist)
    call emit_i(cname, 'input', 'nsoil',  0, nsoil_a)
    call emit_i(cname, 'input', 'nsnow',  0, nsnow_a)
    call emit_l(cname, 'input', 'calculate_soil', 0, calculate_soil)
    call emit_r(cname, 'input', 'swdown', 0, swdown)
    call emit_r(cname, 'input', 'fsa',    0, fsa)
    call emit_r(cname, 'input', 'fsr',    0, fsr)
    call emit_r(cname, 'input', 'fira',   0, fira)
    call emit_r(cname, 'input', 'fsh',    0, fsh)
    call emit_r(cname, 'input', 'fcev',   0, fcev)
    call emit_r(cname, 'input', 'fgev',   0, fgev)
    call emit_r(cname, 'input', 'fctr',   0, fctr)
    call emit_r(cname, 'input', 'ssoil',  0, ssoil)
    call emit_r(cname, 'input', 'sav',    0, sav)
    call emit_r(cname, 'input', 'sag',    0, sag)
    call emit_r(cname, 'input', 'firr',   0, firr)
    call emit_r(cname, 'input', 'canhs',  0, canhs)
    call emit_r(cname, 'input', 'pah',    0, pah)
    call emit_r(cname, 'input', 'beg_wb', 0, beg_wb)
    call emit_r(cname, 'input', 'canliq', 0, canliq)
    call emit_r(cname, 'input', 'canice', 0, canice)
    call emit_r(cname, 'input', 'sneqv',  0, sneqv)
    call emit_r(cname, 'input', 'wa',     0, wa)
    call emit_r(cname, 'input', 'prcp',   0, prcp)
    call emit_r(cname, 'input', 'ecan',   0, ecan)
    call emit_r(cname, 'input', 'etran',  0, etran)
    call emit_r(cname, 'input', 'edir',   0, edir)
    call emit_r(cname, 'input', 'runsrf', 0, runsrf)
    call emit_r(cname, 'input', 'runsub', 0, runsub)
    call emit_r(cname, 'input', 'qtldrn', 0, qtldrn)
    call emit_r(cname, 'input', 'dt',     0, dt)
    call emit_r(cname, 'input', 'irmirate', 0, irmirate)
    call emit_r(cname, 'input', 'irfirate', 0, irfirate)
    call emit_r(cname, 'input', 'acc_dwater', 0, acc_dwater)
    call emit_r(cname, 'input', 'acc_prcp',   0, acc_prcp)
    call emit_r(cname, 'input', 'acc_ecan',   0, acc_ecan)
    call emit_r(cname, 'input', 'acc_etran',  0, acc_etran)
    call emit_r(cname, 'input', 'acc_edir',   0, acc_edir)
    do k = 1, NSOIL
      call emit_r(cname, 'input', 'smc',    k, smc(k))
      call emit_r(cname, 'input', 'dzsnso', k, dzsnso(k))
    end do
    ! The arguments ERROR reads only inside a block that ends in
    ! wrf_error_fatal.  Recorded so the fixture says what they were.
    call emit_r(cname, 'input', 'fveg', 0, fveg)
    call emit_r(cname, 'input', 'fsrv', 0, fsrv)
    call emit_r(cname, 'input', 'fsrg', 0, fsrg)
    call emit_r(cname, 'input', 'zwt',  0, zwt)
    call emit_r(cname, 'input', 'pahv', 0, pahv)
    call emit_r(cname, 'input', 'pahg', 0, pahg)
    call emit_r(cname, 'input', 'pahb', 0, pahb)
  end subroutine emit_inputs

  subroutine emit_outputs(cname)
    character(len=*), intent(in) :: cname
    call emit_r(cname, 'output', 'errwat',     0, errwat)
    call emit_r(cname, 'output', 'acc_dwater', 0, acc_dwater)
    call emit_r(cname, 'output', 'acc_prcp',   0, acc_prcp)
    call emit_r(cname, 'output', 'acc_ecan',   0, acc_ecan)
    call emit_r(cname, 'output', 'acc_etran',  0, acc_etran)
    call emit_r(cname, 'output', 'acc_edir',   0, acc_edir)
  end subroutine emit_outputs

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
    write(CSV, '(A)') 'sflx_error,'//trim(cname)//','//stage//','//field//','// &
         trim(itoa(idx))//',real,'//trim(rtoa(v))//','//hex8(v)
  end subroutine emit_r

  subroutine emit_i(cname, stage, field, idx, v)
    character(len=*), intent(in) :: cname, stage, field
    integer, intent(in) :: idx, v
    write(CSV, '(A)') 'sflx_error,'//trim(cname)//','//stage//','//field//','// &
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

end program run_noahmp_sflx_compose_oracle
