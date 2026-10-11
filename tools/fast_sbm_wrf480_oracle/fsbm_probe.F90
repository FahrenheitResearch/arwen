! Record writer and probe module for the fast SBM oracle harness.
!
! This file is compiled twice per build:
!
!   -DFSBM_PART_IO     module fsbm_io, ALWAYS without -freal-4-real-8, so its
!                      real(4) and real(8) specifics are genuine and a generic
!                      call resolves on the actual argument's kind: a WRF
!                      REAL(KIND=4) variable writes as f4 in the native build
!                      and as f8 in the promoted build.
!   -DFSBM_PART_PROBE  module fsbm_probe_state and the external subroutines
!                      fsbm_probe_*, with the SAME flags as the WRF module, so
!                      their real(4) dummies are promoted exactly when WRF's
!                      REAL(KIND=R4SIZE) actuals are.
!
! Record format (every oracle stream): a sequence of records, each an int32
! type tag followed by that type's fields, raw little-endian, Fortran order.
! The first record of each type defines its layout, written to <prefix>.layout
! as "type <tag> <name> <nfields>" and "field <name> <kind> <rank> <dims...>"
! lines; every later record of that type must match it field for field or the
! harness stops (FSBM_HARNESS_FAILURE, exit 70). "count <tag> <n>" lines and
! "end" close the layout. pack_fixtures.py reads nothing else.
!
! The probe subroutines are called from lines that instrument_probes.py
! inserts into FAST_SBM and COAL_BOTT_NEW (implicit interface, so the
! transform adds CALL lines only). They never write to their arguments. They
! record for the selected steps and levels only, count branch events over the
! whole run, and keep the last COAL_BOTT_NEW breakup path for the sweeps.
!
! The cold anchors (milestone 2) add probe routines for the ice pre-pass, the
! ice branch of nucleation (inside JERNUCL01_KS), the regime dispatch,
! ONECOND2 and ONECOND3 (and the JERSUPSAT_KS call inside them), the cold
! block of COAL_BOTT_NEW (with a counting probe before each of its twelve
! collision calls, the Hallett-Mossop line and the riming heat), FREEZ,
! J_W_MELT, the merges, the snow-breakup guard and the exit. A run that calls
! fsbm_probe_configure_tile (the cold column KC-2) also gets a column mask for
! the records and per-column counters indexed by (i, j), and only such a run
! gets cold records; a run that does not (KC-1, the sweeps) records exactly
! what milestone 1 recorded (the sweeps still read the cold anchors' capture
! values).

#if defined(FSBM_PART_IO)
module fsbm_io
  implicit none
  private
  integer, parameter, public :: FSBM_MAXT = 80, FSBM_MAXF = 128, FSBM_NL = 40

  type, public :: fsbm_stream
    integer :: u_raw = -1, u_lay = -1
    logical :: is_open = .false.
    integer :: ntypes = 0, cur = 0, curf = 0
    logical :: defining = .false.
    integer :: nf(FSBM_MAXT) = 0
    integer(8) :: nrec(FSBM_MAXT) = 0_8
    character(len=FSBM_NL) :: tname(FSBM_MAXT) = ' '
    character(len=FSBM_NL) :: fname(FSBM_MAXF, FSBM_MAXT) = ' '
    character(len=2) :: fkind(FSBM_MAXF, FSBM_MAXT) = '  '
    integer :: frank(FSBM_MAXF, FSBM_MAXT) = 0
    integer :: fshape(4, FSBM_MAXF, FSBM_MAXT) = 0
  end type fsbm_stream

  public :: fsbm_open, fsbm_close, fsbm_begin, fsbm_end, fsbm_put, fsbm_fail

  interface fsbm_put
    module procedure put_f8_0, put_f8_1, put_f8_2, put_f8_3, put_f8_4
    module procedure put_f4_0, put_f4_1, put_f4_2, put_f4_3, put_f4_4
    module procedure put_i4_0, put_i4_1, put_i4_2
  end interface fsbm_put

contains

  subroutine fsbm_fail(msg)
    character(len=*), intent(in) :: msg
    write(0,'(a)') 'FSBM_HARNESS_FAILURE: '//trim(msg)
    error stop 70
  end subroutine fsbm_fail

  subroutine fsbm_open(s, prefix)
    type(fsbm_stream), intent(inout) :: s
    character(len=*), intent(in) :: prefix
    integer :: st
    if (s%is_open) call fsbm_fail('stream already open: '//trim(prefix))
    open(newunit=s%u_raw, file=trim(prefix)//'.raw', access='stream', &
         form='unformatted', status='replace', action='write', iostat=st)
    if (st /= 0) call fsbm_fail('cannot open '//trim(prefix)//'.raw')
    open(newunit=s%u_lay, file=trim(prefix)//'.layout', form='formatted', &
         status='replace', action='write', iostat=st)
    if (st /= 0) call fsbm_fail('cannot open '//trim(prefix)//'.layout')
    write(s%u_lay,'(a)') 'fsbm-layout 1'
    s%is_open = .true.
    s%ntypes = 0
    s%cur = 0
    s%curf = 0
    s%defining = .false.
    s%nf = 0
    s%nrec = 0_8
  end subroutine fsbm_open

  subroutine fsbm_close(s)
    type(fsbm_stream), intent(inout) :: s
    integer :: t
    if (.not. s%is_open) return
    if (s%cur /= 0) call fsbm_fail('stream closed inside a record')
    do t = 1, s%ntypes
      write(s%u_lay,'(a,1x,i0,1x,i0)') 'count', t, s%nrec(t)
    end do
    write(s%u_lay,'(a)') 'end'
    close(s%u_raw)
    close(s%u_lay)
    s%is_open = .false.
  end subroutine fsbm_close

  subroutine fsbm_begin(s, tname)
    type(fsbm_stream), intent(inout) :: s
    character(len=*), intent(in) :: tname
    integer :: t, i
    if (.not. s%is_open) call fsbm_fail('record on a closed stream: '//tname)
    if (s%cur /= 0) call fsbm_fail('nested record: '//tname)
    if (len_trim(tname) > FSBM_NL .or. index(trim(tname), ' ') > 0) &
      call fsbm_fail('bad record type name: '//tname)
    t = 0
    do i = 1, s%ntypes
      if (s%tname(i) == tname) t = i
    end do
    if (t == 0) then
      if (s%ntypes >= FSBM_MAXT) call fsbm_fail('too many record types')
      s%ntypes = s%ntypes + 1
      t = s%ntypes
      s%tname(t) = tname
      s%nf(t) = 0
      s%defining = .true.
    else
      s%defining = .false.
    end if
    s%cur = t
    s%curf = 0
    write(s%u_raw) int(t, 4)
  end subroutine fsbm_begin

  subroutine fsbm_end(s)
    type(fsbm_stream), intent(inout) :: s
    integer :: t, f, r
    character(len=256) :: line
    t = s%cur
    if (t == 0) call fsbm_fail('end without begin')
    if (s%defining) then
      s%nf(t) = s%curf
      write(s%u_lay,'(a,1x,i0,1x,a,1x,i0)') 'type', t, trim(s%tname(t)), s%nf(t)
      do f = 1, s%nf(t)
        write(line,'(a,1x,a,1x,a,1x,i0)') 'field', trim(s%fname(f,t)), &
             s%fkind(f,t), s%frank(f,t)
        do r = 1, s%frank(f,t)
          write(line,'(a,1x,i0)') trim(line), s%fshape(r,f,t)
        end do
        write(s%u_lay,'(a)') trim(line)
      end do
      flush(s%u_lay)
    else if (s%curf /= s%nf(t)) then
      call fsbm_fail('record of type '//trim(s%tname(t))//' has the wrong field count')
    end if
    s%nrec(t) = s%nrec(t) + 1_8
    s%cur = 0
    s%curf = 0
    s%defining = .false.
  end subroutine fsbm_end

  subroutine field(s, name, kind, rank, shp)
    type(fsbm_stream), intent(inout) :: s
    character(len=*), intent(in) :: name
    character(len=2), intent(in) :: kind
    integer, intent(in) :: rank
    integer, intent(in) :: shp(:)
    integer :: t, f
    t = s%cur
    if (t == 0) call fsbm_fail('field outside a record: '//name)
    if (len_trim(name) > FSBM_NL .or. index(trim(name), ' ') > 0) &
      call fsbm_fail('bad field name: '//name)
    s%curf = s%curf + 1
    f = s%curf
    if (s%defining) then
      if (f > FSBM_MAXF) call fsbm_fail('too many fields in '//trim(s%tname(t)))
      s%fname(f,t) = name
      s%fkind(f,t) = kind
      s%frank(f,t) = rank
      s%fshape(:,f,t) = 0
      if (rank > 0) s%fshape(1:rank,f,t) = shp(1:rank)
    else
      if (f > s%nf(t)) call fsbm_fail('extra field '//name//' in '//trim(s%tname(t)))
      if (s%fname(f,t) /= name .or. s%fkind(f,t) /= kind .or. s%frank(f,t) /= rank) &
        call fsbm_fail('field '//name//' does not match layout of '//trim(s%tname(t)))
      if (rank > 0) then
        if (any(s%fshape(1:rank,f,t) /= shp(1:rank))) &
          call fsbm_fail('shape of '//name//' does not match layout of '//trim(s%tname(t)))
      end if
    end if
  end subroutine field

  subroutine put_f8_0(s, name, a)
    type(fsbm_stream), intent(inout) :: s
    character(len=*), intent(in) :: name
    real(8), intent(in) :: a
    integer :: none(1)
    none = 0
    call field(s, name, 'f8', 0, none)
    write(s%u_raw) a
  end subroutine put_f8_0

  subroutine put_f8_1(s, name, a)
    type(fsbm_stream), intent(inout) :: s
    character(len=*), intent(in) :: name
    real(8), intent(in) :: a(:)
    call field(s, name, 'f8', 1, shape(a))
    write(s%u_raw) a
  end subroutine put_f8_1

  subroutine put_f8_2(s, name, a)
    type(fsbm_stream), intent(inout) :: s
    character(len=*), intent(in) :: name
    real(8), intent(in) :: a(:,:)
    call field(s, name, 'f8', 2, shape(a))
    write(s%u_raw) a
  end subroutine put_f8_2

  subroutine put_f8_3(s, name, a)
    type(fsbm_stream), intent(inout) :: s
    character(len=*), intent(in) :: name
    real(8), intent(in) :: a(:,:,:)
    call field(s, name, 'f8', 3, shape(a))
    write(s%u_raw) a
  end subroutine put_f8_3

  subroutine put_f8_4(s, name, a)
    type(fsbm_stream), intent(inout) :: s
    character(len=*), intent(in) :: name
    real(8), intent(in) :: a(:,:,:,:)
    call field(s, name, 'f8', 4, shape(a))
    write(s%u_raw) a
  end subroutine put_f8_4

  subroutine put_f4_0(s, name, a)
    type(fsbm_stream), intent(inout) :: s
    character(len=*), intent(in) :: name
    real(4), intent(in) :: a
    integer :: none(1)
    none = 0
    call field(s, name, 'f4', 0, none)
    write(s%u_raw) a
  end subroutine put_f4_0

  subroutine put_f4_1(s, name, a)
    type(fsbm_stream), intent(inout) :: s
    character(len=*), intent(in) :: name
    real(4), intent(in) :: a(:)
    call field(s, name, 'f4', 1, shape(a))
    write(s%u_raw) a
  end subroutine put_f4_1

  subroutine put_f4_2(s, name, a)
    type(fsbm_stream), intent(inout) :: s
    character(len=*), intent(in) :: name
    real(4), intent(in) :: a(:,:)
    call field(s, name, 'f4', 2, shape(a))
    write(s%u_raw) a
  end subroutine put_f4_2

  subroutine put_f4_3(s, name, a)
    type(fsbm_stream), intent(inout) :: s
    character(len=*), intent(in) :: name
    real(4), intent(in) :: a(:,:,:)
    call field(s, name, 'f4', 3, shape(a))
    write(s%u_raw) a
  end subroutine put_f4_3

  subroutine put_f4_4(s, name, a)
    type(fsbm_stream), intent(inout) :: s
    character(len=*), intent(in) :: name
    real(4), intent(in) :: a(:,:,:,:)
    call field(s, name, 'f4', 4, shape(a))
    write(s%u_raw) a
  end subroutine put_f4_4

  subroutine put_i4_0(s, name, a)
    type(fsbm_stream), intent(inout) :: s
    character(len=*), intent(in) :: name
    integer(4), intent(in) :: a
    integer :: none(1)
    none = 0
    call field(s, name, 'i4', 0, none)
    write(s%u_raw) a
  end subroutine put_i4_0

  subroutine put_i4_1(s, name, a)
    type(fsbm_stream), intent(inout) :: s
    character(len=*), intent(in) :: name
    integer(4), intent(in) :: a(:)
    call field(s, name, 'i4', 1, shape(a))
    write(s%u_raw) a
  end subroutine put_i4_1

  subroutine put_i4_2(s, name, a)
    type(fsbm_stream), intent(inout) :: s
    character(len=*), intent(in) :: name
    integer(4), intent(in) :: a(:,:)
    call field(s, name, 'i4', 2, shape(a))
    write(s%u_raw) a
  end subroutine put_i4_2

end module fsbm_io
#endif

#if defined(FSBM_PART_PROBE)
module fsbm_probe_state
  use fsbm_io
  implicit none
  public

  integer, parameter :: PR_NKR = 33, PR_NCCN = 43, PR_MAXCOL = 4
  integer, parameter :: PR_KERNEL_CAP = 40

  ! Counter ids; names below are the field names of the "counts" record.
  integer, parameter :: C_FORCING = 1, C_NUCL_CB_CALL = 2, C_NUCL_CB_ACT = 3, &
       C_NUCL_IC_CALL = 4, C_NUCL_IC_ACT = 5, C_COND1_CALL = 6, C_COND1_EVAP = 7, &
       C_COND1_COND = 8, C_COAL_CALL = 9, C_COAL_BRK = 10, C_COAL_BRK_DONE = 11, &
       C_COAL_BRK_SKIP = 12, C_NDIV1 = 13, C_NDIV2 = 14, C_NDIV4 = 15, C_NDIV8 = 16, &
       C_NDIV_MORE = 17, C_SPONT_CALL = 18, C_SPONT_TAKEN = 19, C_FALL_DROP = 20, &
       PR_NCNT = 20
  character(len=16), parameter :: CNT_NAME(PR_NCNT) = [character(len=16) :: &
       'forcing', 'nucl_cb_calls', 'nucl_cb_act', 'nucl_ic_calls', 'nucl_ic_act', &
       'cond1_calls', 'cond1_evap', 'cond1_cond', 'coal_calls', 'coal_breakup', &
       'coal_brk_done', 'coal_brk_skip', 'ndiv_1', 'ndiv_2', 'ndiv_4', 'ndiv_8', &
       'ndiv_more', 'spont_calls', 'spont_taken', 'fall_drop_calls']

  logical, save :: pr_record = .false., pr_count = .false., pr_capture = .false.
  integer, save :: pr_nsteps = 0, pr_steps(64) = 0
  integer, save :: pr_nlevs = 0, pr_levs(64) = 0
  integer, save :: pr_kernel_written = 0
  type(fsbm_stream), save :: pr_stream
  integer(4), save :: pr_cnt(PR_MAXCOL, PR_NCNT) = 0

  real(8), save :: pr_nucl_before(PR_NKR) = 0.0d0

  ! ---- milestone 2: tiles of several rows, per-column counters, cold capture
  integer, parameter :: PR_ICEMAX = 3, PR_MAXC2 = 16
  integer, save :: pr_kcap_ll = PR_KERNEL_CAP
  integer, save :: pr_ci0 = 0, pr_cj0 = 0, pr_cni = 0, pr_cnj = 0
  logical, save :: pr_colmask(PR_MAXC2) = .true.
  integer, save :: pr_cur_step = 0
  integer, save :: pr_kcold_cap = 0, pr_kcold_written = 0

  ! Per-column counter ids (record "counts_tile", one field per counter,
  ! each field one value per tile column, i fastest, then j).
  integer, parameter :: K_FORCING = 1, K_NUCL_CB_CALL = 2, K_NUCL_CB_ACT = 3, &
       K_NUCL_IC_CALL = 4, K_NUCL_IC_ACT = 5, K_COND1_CALL = 6, K_COND1_EVAP = 7, &
       K_COND1_COND = 8, K_COAL_CALL = 9, K_COAL_BRK = 10, K_COAL_BRK_DONE = 11, &
       K_COAL_BRK_SKIP = 12, K_SPONT_CALL = 13, K_SPONT_TAKEN = 14, K_FALL_DROP = 15, &
       K_HOMFRZ = 16, K_ICENUCL = 17, K_ICENUCL_BAND = 17, K_ICENUCL_ADD = 23, &
       K_ICENUCL_DEL2_CLAMP = 30, K_ICENUCL_DSUP_CLAMP = 31, K_DISP1 = 32, K_DISP2 = 33, &
       K_DISP3 = 34, K_DISP_D16_COLD = 35, K_DISP_D16_WARM = 36, K_DISP_OTHER = 37, &
       K_COND2 = 38, K_COND2_K20 = 39, K_COND2_K21 = 40, K_COND3 = 41, K_COND3_K30 = 42, &
       K_COND3_K31 = 43, K_COND3_K32 = 44, K_COND3_K50 = 45, K_SUPSAT_B = 46, &
       K_SUPSAT_CLAMP = 56, K_COAL_COLD = 57, K_CC = 57, K_HM_BLOCK = 70, &
       K_HM_SPLINTERS = 71, K_RIME_HEAT = 72, K_FREEZ = 73, K_FREEZ_PLATES = 74, &
       K_FREEZ_HAIL = 75, K_MELT = 76, K_MELT_SNOW_B = 76, K_MELT_DENSE_B = 79, &
       K_MELT_NEG = 83, K_SNOWBRK_GUARD = 84, K_SNOWBRK_CALL = 85, K_D11 = 86, &
       K_D15 = 87, K_FALL_SNOW = 88, K_FALL_DENSE = 89, K_EXIT_SNOW = 90, &
       K_EXIT_DENSE = 91, PR_NCNT2 = 91
  ! K_ICENUCL_BAND + b (b = 1..6), K_ICENUCL_ADD + b, K_SUPSAT_B + branch
  ! (branch 0..9), K_CC + id (id = 1..12), K_MELT_SNOW_B + b and
  ! K_MELT_DENSE_B + b (b = 1..3) address the families of counters.
  character(len=24), parameter :: CNT2_NAME(PR_NCNT2) = [character(len=24) :: &
       'forcing', 'nucl_cb_calls', 'nucl_cb_act', 'nucl_ic_calls', 'nucl_ic_act', &
       'cond1_calls', 'cond1_evap', 'cond1_cond', 'coal_calls', 'coal_breakup', &
       'coal_brk_done', 'coal_brk_skip', 'spont_calls', 'spont_taken', 'fall_drop_calls', &
       'homfrz', 'icenucl_calls', &
       'icenucl_band1', 'icenucl_band2', 'icenucl_band3', 'icenucl_band4', &
       'icenucl_band5', 'icenucl_band6', &
       'icenucl_add_band1', 'icenucl_add_band2', 'icenucl_add_band3', &
       'icenucl_add_band4', 'icenucl_add_band5', 'icenucl_add_band6', &
       'icenucl_del2_clamp', 'icenucl_dsup_clamp', &
       'dispatch_cond1', 'dispatch_cond2', 'dispatch_cond3', &
       'dispatch_none_d16_cold', 'dispatch_none_d16_warm', 'dispatch_none_other', &
       'cond2_calls', 'cond2_kcond20', 'cond2_kcond21', &
       'cond3_calls', 'cond3_kcond30', 'cond3_kcond31', 'cond3_kcond32', 'cond3_kcond50', &
       'supsat_branch0', 'supsat_branch1', 'supsat_branch2', 'supsat_branch3', &
       'supsat_branch4', 'supsat_branch5', 'supsat_branch6', 'supsat_branch7', &
       'supsat_branch8', 'supsat_branch9', 'supsat_g2_clamp', &
       'coal_cold_block', &
       'cc01_xyz_drop_snow_hail', 'cc02_xyz_drop_snow_grp', 'cc03_xyx_snow_drop', &
       'cc04_xyz_snow_drop_hail', 'cc05_xyz_snow_drop_grp', 'cc06_xyy_drop_grp', &
       'cc07_xyx_grp_drop', 'cc08_xyz_drop_grp_hail', 'cc09_xyz_grp_drop_hail', &
       'cc10_xyy_drop_hail', 'cc11_xyx_hail_drop', 'cc12_xxx_snow_snow', &
       'hm_block', 'hm_splinters', 'rime_heat', &
       'freez_calls', 'freez_to_plates', 'freez_to_hail', &
       'melt_calls', 'melt_snow_band1', 'melt_snow_band2', 'melt_snow_band3', &
       'melt_dense_band1', 'melt_dense_band2', 'melt_dense_band3', 'melt_negative_bin', &
       'snowbrk_guard_true', 'snowbrk_call', 'd11_levels', 'd15_points', &
       'fall_snow_calls', 'fall_dense_calls', 'exit_snow_levels', 'exit_dense_levels']
  integer(4), save :: pr_cnt2(PR_MAXC2, PR_NCNT2) = 0

  ! Per-call state kept between a before and an after anchor.
  real(8), save :: pr_psi2_before(PR_NKR, PR_ICEMAX) = 0.0d0
  real(8), save :: pr_tt_coal_before = 0.0d0
  real(8), save :: pr_ff_before(PR_NKR, 7) = 0.0d0

  ! Capture for the cold sweeps: the last call's inner values.
  integer(4), save :: capc_icenucl_calls = 0, capc_icenucl_band = 0, capc_icenucl_added = 0
  real(8), save :: capc_icenucl_psi2_in(PR_NKR, PR_ICEMAX) = 0.0d0
  real(8), save :: capc_icenucl_psi2_out(PR_NKR, PR_ICEMAX) = 0.0d0
  real(8), save :: capc_icenucl_scal_in(4) = 0.0d0, capc_icenucl_scal_out(4) = 0.0d0
  integer(4), save :: capc_homfrz_calls = 0
  real(8), save :: capc_homfrz(3) = 0.0d0
  integer(4), save :: capc_supsat_n = 0, capc_supsat_site(4) = 0, capc_supsat_kcond(4) = 0
  integer(4), save :: capc_supsat_branch(4) = -1, capc_supsat_clamp(4) = 0
  real(8), save :: capc_supsat_rates(7, 4) = 0.0d0
  integer(4), save :: capc_cc_mask = 0, capc_cold_block = 0, capc_hm_fired = 0
  integer(4), save :: capc_cold_icol(4) = 0
  real(8), save :: capc_hm(5) = 0.0d0, capc_rime(8) = 0.0d0
  real(8), save :: capc_cold_g_in(PR_NKR, 4) = 0.0d0, capc_cold_g_out(PR_NKR, 4) = 0.0d0
  real(8), save :: capc_cold_rf3_out(PR_NKR) = 0.0d0, capc_cold_scal(3) = 0.0d0

  ! Last COAL_BOTT_NEW call, kept for the sweeps when pr_capture is on.
  integer(4), save :: cap_icol_drop = -1, cap_icol_brk = -1, cap_ndiv = -1, cap_path = -1
  real(8), save :: cap_g1_xxx_in(PR_NKR) = 0.0d0, cap_g1_xxx_out(PR_NKR) = 0.0d0
  real(8), save :: cap_g1_at11(PR_NKR) = 0.0d0, cap_break_per = 0.0d0

contains

  subroutine fsbm_probe_configure(prefix, steps, levs, record, count, capture)
    character(len=*), intent(in) :: prefix
    integer, intent(in) :: steps(:), levs(:)
    logical, intent(in) :: record, count, capture
    pr_nsteps = size(steps)
    pr_steps = 0
    pr_steps(1:pr_nsteps) = steps
    pr_nlevs = size(levs)
    pr_levs = 0
    pr_levs(1:pr_nlevs) = levs
    pr_record = record
    pr_count = count
    pr_capture = capture
    pr_cnt = 0
    pr_kernel_written = 0
    pr_kcap_ll = PR_KERNEL_CAP
    pr_ci0 = 0; pr_cj0 = 0; pr_cni = 0; pr_cnj = 0
    pr_colmask = .true.
    pr_cnt2 = 0
    pr_kcold_cap = 0
    pr_kcold_written = 0
    if (record) then
      call fsbm_open(pr_stream, prefix)
      call fsbm_begin(pr_stream, 'config')
      call fsbm_put(pr_stream, 'steps', int(steps, 4))
      call fsbm_put(pr_stream, 'levels', int(levs, 4))
      call fsbm_put(pr_stream, 'kernel_cap', int(PR_KERNEL_CAP, 4))
      call fsbm_end(pr_stream)
    end if
  end subroutine fsbm_probe_configure

  ! Milestone 2: a tile of ni x nj columns starting at (i0, j0). Records are
  ! written only for the columns whose mask entry is true; the per-column
  ! counters count every tile column. kernel_cap bounds the drop-drop kernel
  ! records (milestone 1's 'kernel'), kcold_cap the cold-block kernel slice
  ! records ('kernel_cold').
  subroutine fsbm_probe_configure_tile(i0, j0, ni, nj, colmask, kernel_cap, kcold_cap)
    integer, intent(in) :: i0, j0, ni, nj, kernel_cap, kcold_cap
    logical, intent(in) :: colmask(:)
    integer :: n
    if (ni*nj > PR_MAXC2 .or. size(colmask) /= ni*nj) call fsbm_fail('bad tile configuration')
    pr_ci0 = i0; pr_cj0 = j0; pr_cni = ni; pr_cnj = nj
    pr_colmask = .false.
    pr_colmask(1:ni*nj) = colmask
    pr_kcap_ll = kernel_cap
    pr_kcold_cap = kcold_cap
    pr_cnt2 = 0
    if (pr_record) then
      call fsbm_begin(pr_stream, 'tile_config')
      call fsbm_put(pr_stream, 'i0', int(i0, 4))
      call fsbm_put(pr_stream, 'j0', int(j0, 4))
      call fsbm_put(pr_stream, 'ni', int(ni, 4))
      call fsbm_put(pr_stream, 'nj', int(nj, 4))
      call fsbm_put(pr_stream, 'column_recorded', [(merge(1_4, 0_4, colmask(n)), n = 1, ni*nj)])
      call fsbm_put(pr_stream, 'kernel_cap', int(kernel_cap, 4))
      call fsbm_put(pr_stream, 'kernel_cold_cap', int(kcold_cap, 4))
      call fsbm_end(pr_stream)
    end if
  end subroutine fsbm_probe_configure_tile

  subroutine fsbm_probe_finish()
    integer :: c
    if (pr_record) then
      call fsbm_begin(pr_stream, 'counts')
      do c = 1, PR_NCNT
        call fsbm_put(pr_stream, trim(CNT_NAME(c)), pr_cnt(:, c))
      end do
      call fsbm_end(pr_stream)
      if (pr_cni > 0) then
        call fsbm_begin(pr_stream, 'counts_tile')
        do c = 1, PR_NCNT2
          call fsbm_put(pr_stream, trim(CNT2_NAME(c)), pr_cnt2(1:pr_cni*pr_cnj, c))
        end do
        call fsbm_end(pr_stream)
      end if
      call fsbm_close(pr_stream)
    end if
    pr_record = .false.
    pr_count = .false.
  end subroutine fsbm_probe_finish

  logical function pr_step_on(step)
    integer, intent(in) :: step
    pr_step_on = pr_record .and. any(pr_steps(1:pr_nsteps) == step)
  end function pr_step_on

  logical function pr_point_on(step, k)
    integer, intent(in) :: step, k
    pr_point_on = pr_step_on(step) .and. any(pr_levs(1:pr_nlevs) == k)
  end function pr_point_on

  subroutine pr_bump(i, c)
    integer, intent(in) :: i, c
    if (.not. pr_count) return
    if (i < 1 .or. i > PR_MAXCOL) return
    pr_cnt(i, c) = pr_cnt(i, c) + 1
  end subroutine pr_bump

  ! Tile column number of (i, j), 0 when no tile is configured or (i, j) is
  ! outside it.
  integer function pr_colid(i, j)
    integer, intent(in) :: i, j
    pr_colid = 0
    if (pr_cni <= 0) return
    if (i < pr_ci0 .or. i >= pr_ci0 + pr_cni) return
    if (j < pr_cj0 .or. j >= pr_cj0 + pr_cnj) return
    pr_colid = (j - pr_cj0)*pr_cni + (i - pr_ci0) + 1
  end function pr_colid

  logical function pr_col_on(i, j)
    integer, intent(in) :: i, j
    integer :: c
    pr_col_on = .true.
    if (pr_cni <= 0) return
    c = pr_colid(i, j)
    pr_col_on = c > 0
    if (c > 0) pr_col_on = pr_colmask(c)
  end function pr_col_on

  logical function pr_point_on3(step, i, k, j)
    integer, intent(in) :: step, i, k, j
    pr_point_on3 = pr_point_on(step, k) .and. pr_col_on(i, j)
  end function pr_point_on3

  ! The cold anchors record only in a run that configured a tile (KC-2), so a
  ! run that did not (KC-1) writes exactly the records it wrote in milestone 1.
  logical function pr_cold_on(step, i, k, j)
    integer, intent(in) :: step, i, k, j
    pr_cold_on = pr_cni > 0
    if (pr_cold_on) pr_cold_on = pr_point_on3(step, i, k, j)
  end function pr_cold_on

  subroutine pr_bump2(i, j, c)
    integer, intent(in) :: i, j, c
    integer :: col
    if (.not. pr_count) return
    col = pr_colid(i, j)
    if (col <= 0) return
    pr_cnt2(col, c) = pr_cnt2(col, c) + 1
  end subroutine pr_bump2

  ! ICE_NUCL's temperature split (:3385-3398) as a band number: 1 above
  ! -4 C, 2 to -8.1, 3 to -12.7, 4 to -17.8, 5 to -22.4, 6 below; 0 for NaN.
  integer function pr_icenucl_band(tt)
    real(8), intent(in) :: tt
    real(8) :: tpc
    tpc = tt - 273.15d0
    pr_icenucl_band = 0
    if (tpc > -4.0d0) then
      pr_icenucl_band = 1
    else if (tpc <= -4.0d0 .and. tpc > -8.1d0) then
      pr_icenucl_band = 2
    else if (tpc <= -8.1d0 .and. tpc > -12.7d0) then
      pr_icenucl_band = 3
    else if (tpc <= -12.7d0 .and. tpc > -17.8d0) then
      pr_icenucl_band = 4
    else if (tpc <= -17.8d0 .and. tpc > -22.4d0) then
      pr_icenucl_band = 5
    else if (tpc <= -22.4d0) then
      pr_icenucl_band = 6
    end if
  end function pr_icenucl_band

  ! The branch JERSUPSAT_KS takes for these rates (:1995-2197), with its own
  ! predicates and kinds; clamp = 1 where the G2 clamp of :2178 fires. The
  ! same classification as the milestone 1 jersupsat sweep.
  integer function pr_supsat_branch(rw0, pw0, ri0, pi0, dt, clamp)
    real(8), intent(in) :: rw0, pw0, ri0, pi0
    real, intent(in) :: dt
    integer, intent(out) :: clamp
    real(8) :: rw, pw, ri, pi, alfa, gama, g2
    integer :: irw, iri
    rw = rw0; pw = pw0; ri = ri0; pi = pi0
    irw = 1; iri = 1; clamp = 0
    if (max(rw, pw, ri, pi) <= 1.0d-10) then
      pr_supsat_branch = 0
      return
    end if
    if (dmax1(rw, pw) > 1.0d-10) then
      if (rw < 1.0d-20) then
        rw = 1.0d-20; irw = 0
      end if
      if (pw < 1.0d-20) pw = 1.0d-20
      if (dmax1(pi/pw, ri/rw) <= 1.0d-4) then
        ri = 0.0; iri = 0; pi = 0.0
      end if
      if (dmin1(pi/pw, ri/rw) > 1.0d0/1.0d-4) then
        rw = 0.0; irw = 0; pw = 0.0
      end if
    else
      rw = 0.0; irw = 0; pw = 0.0
    end if
    if (irw == 0 .and. iri == 0) then
      pr_supsat_branch = 1
    else if (iri == 0) then
      if (abs(rw*dt) > 1.0E-6) then
        pr_supsat_branch = 2
      else
        pr_supsat_branch = 3
      end if
    else if (irw == 0) then
      if (abs(pi*dt) > 1.0E-6) then
        pr_supsat_branch = 4
      else
        pr_supsat_branch = 5
      end if
    else
      alfa = dsqrt((rw-pi)*(rw-pi)+4.0d0*pw*ri)
      gama = 0.5d0*(alfa-rw-pi)
      g2 = rw*pi-ri*pw
      if (g2 < 1.0d-20) clamp = 1
      if (dabs(gama*dt) > 1.0E-6) then
        pr_supsat_branch = 6
      else if (abs(ri/rw) > 1.0E-12) then
        if (abs(rw/ri) > 1.0E-12) then
          pr_supsat_branch = 7
        else
          pr_supsat_branch = 8
        end if
      else
        pr_supsat_branch = 9
      end if
    end if
  end function pr_supsat_branch

  subroutine pr_head(tname, step, i, k, j)
    character(len=*), intent(in) :: tname
    integer, intent(in) :: step, i, k, j
    call fsbm_begin(pr_stream, tname)
    call fsbm_put(pr_stream, 'step', int(step, 4))
    call fsbm_put(pr_stream, 'i', int(i, 4))
    call fsbm_put(pr_stream, 'k', int(k, 4))
    call fsbm_put(pr_stream, 'j', int(j, 4))
  end subroutine pr_head

end module fsbm_probe_state

! ---------------------------------------------------------------------------
! Anchor: after the entry conversion and the t_new/t_old loop in FAST_SBM.
subroutine fsbm_probe_entry(itimestep, chem, n_chem, t_new, t_old, qv, qv_old, &
                            pcgs, rhocgs, zcgs, ims, ime, kms, kme, jms, jme, &
                            its, ite, kts, kte, jts, jte)
  use fsbm_io
  use fsbm_probe_state
  implicit none
  integer, intent(in) :: itimestep, n_chem, ims, ime, kms, kme, jms, jme
  integer, intent(in) :: its, ite, kts, kte, jts, jte
  real(4), intent(in) :: chem(ims:ime, kms:kme, jms:jme, n_chem)
  real(4), intent(in) :: qv(ims:ime, kms:kme, jms:jme), qv_old(ims:ime, kms:kme, jms:jme)
  real(4), intent(in), dimension(its-1:ite+1, kts:kte, jts-1:jte+1) :: &
       t_new, t_old, pcgs, rhocgs, zcgs
  integer :: i, k, j
  if (.not. pr_step_on(itimestep)) return
  do j = jts, jte
    do i = its, ite
      do k = kts, kte
        if (.not. pr_point_on3(itimestep, i, k, j)) cycle
        call pr_head('entry', itimestep, i, k, j)
        call fsbm_put(pr_stream, 'ff1', chem(i, k, j, 2:34))
        call fsbm_put(pr_stream, 'ff5', chem(i, k, j, 35:67))
        call fsbm_put(pr_stream, 'ff6', chem(i, k, j, 68:100))
        call fsbm_put(pr_stream, 'fccn', chem(i, k, j, 101:143))
        call fsbm_put(pr_stream, 't_new', t_new(i, k, j))
        call fsbm_put(pr_stream, 't_old', t_old(i, k, j))
        call fsbm_put(pr_stream, 'qv', qv(i, k, j))
        call fsbm_put(pr_stream, 'qv_old', qv_old(i, k, j))
        call fsbm_put(pr_stream, 'pcgs', pcgs(i, k, j))
        call fsbm_put(pr_stream, 'rhocgs', rhocgs(i, k, j))
        call fsbm_put(pr_stream, 'zcgs', zcgs(i, k, j))
        call fsbm_end(pr_stream)
      end do
    end do
  end do
end subroutine fsbm_probe_entry

! Anchor: each substep's forcing update (after DEL2IN is recomputed).
subroutine fsbm_probe_forcing(itimestep, i, k, j, ikl, tt, qq, del1in, del2in, &
                              div1, div2, dtcond, del_t, del_q)
  use fsbm_io
  use fsbm_probe_state
  implicit none
  integer, intent(in) :: itimestep, i, k, j, ikl
  real(8), intent(in) :: tt, qq, del1in, del2in, div1, div2, del_t, del_q
  real(4), intent(in) :: dtcond
  call pr_bump(i, C_FORCING)
  call pr_bump2(i, j, K_FORCING)
  if (.not. pr_point_on3(itimestep, i, k, j)) return
  call pr_head('forcing', itimestep, i, k, j)
  call fsbm_put(pr_stream, 'ikl', int(ikl, 4))
  call fsbm_put(pr_stream, 'tt', tt)
  call fsbm_put(pr_stream, 'qq', qq)
  call fsbm_put(pr_stream, 'del1in', del1in)
  call fsbm_put(pr_stream, 'del2in', del2in)
  call fsbm_put(pr_stream, 'div1', div1)
  call fsbm_put(pr_stream, 'div2', div2)
  call fsbm_put(pr_stream, 'dtcond', dtcond)
  call fsbm_put(pr_stream, 'del_t', del_t)
  call fsbm_put(pr_stream, 'del_q', del_q)
  call fsbm_end(pr_stream)
end subroutine fsbm_probe_forcing

! Anchors: immediately before (phase 0) and after (phase 1) JERNUCL01_KS.
subroutine fsbm_probe_nucl(phase, itimestep, i, k, j, ff1in, fccn, tt, qq, &
                           del1in, del2in, is_cb, kz_cb, w_stag, rho, p)
  use fsbm_io
  use fsbm_probe_state
  implicit none
  integer, intent(in) :: phase, itimestep, i, k, j, is_cb, kz_cb
  real(4), intent(in) :: ff1in(PR_NKR), fccn(PR_NCCN), w_stag, rho, p
  real(8), intent(in) :: tt, qq, del1in, del2in
  logical :: activated
  if (phase == 0) then
    pr_nucl_before = real(ff1in, 8)
    if (is_cb == 1) then
      call pr_bump(i, C_NUCL_CB_CALL)
      call pr_bump2(i, j, K_NUCL_CB_CALL)
    else
      call pr_bump(i, C_NUCL_IC_CALL)
      call pr_bump2(i, j, K_NUCL_IC_CALL)
    end if
  else
    activated = any(real(ff1in, 8) > pr_nucl_before)
    if (activated) then
      if (is_cb == 1) then
        call pr_bump(i, C_NUCL_CB_ACT)
        call pr_bump2(i, j, K_NUCL_CB_ACT)
      else
        call pr_bump(i, C_NUCL_IC_ACT)
        call pr_bump2(i, j, K_NUCL_IC_ACT)
      end if
    end if
  end if
  if (.not. pr_point_on3(itimestep, i, k, j)) return
  if (phase == 0) then
    call pr_head('nucl_before', itimestep, i, k, j)
  else
    call pr_head('nucl_after', itimestep, i, k, j)
  end if
  call fsbm_put(pr_stream, 'ff1', ff1in)
  call fsbm_put(pr_stream, 'fccn', fccn)
  call fsbm_put(pr_stream, 'tt', tt)
  call fsbm_put(pr_stream, 'qq', qq)
  call fsbm_put(pr_stream, 'del1in', del1in)
  call fsbm_put(pr_stream, 'del2in', del2in)
  call fsbm_put(pr_stream, 'is_cloud_base', int(is_cb, 4))
  call fsbm_put(pr_stream, 'kz_cloud_base', int(kz_cb, 4))
  call fsbm_put(pr_stream, 'w_stag', w_stag)
  call fsbm_put(pr_stream, 'rhocgs', rho)
  call fsbm_put(pr_stream, 'pcgs', p)
  call fsbm_end(pr_stream)
end subroutine fsbm_probe_nucl

! Anchors: immediately before (phase 0) and after (phase 1) ONECOND1.
subroutine fsbm_probe_cond1(phase, itimestep, i, k, j, ff1r, tt, qq, del1in, &
                            del2in, div1, div2, dtcond, vr1z, pcgs, rho)
  use fsbm_io
  use fsbm_probe_state
  implicit none
  integer, intent(in) :: phase, itimestep, i, k, j
  real(4), intent(in) :: ff1r(PR_NKR), dtcond, vr1z(PR_NKR), pcgs, rho
  real(8), intent(in) :: tt, qq, del1in, del2in, div1, div2
  if (phase == 0) then
    call pr_bump(i, C_COND1_CALL)
    call pr_bump2(i, j, K_COND1_CALL)
    ! ONECOND1 takes its evaporation branch (KCOND = 10) when DEL1N < 0.
    if (del1in < 0.0d0) then
      call pr_bump(i, C_COND1_EVAP)
      call pr_bump2(i, j, K_COND1_EVAP)
    else
      call pr_bump(i, C_COND1_COND)
      call pr_bump2(i, j, K_COND1_COND)
    end if
  end if
  if (.not. pr_point_on3(itimestep, i, k, j)) return
  if (phase == 0) then
    call pr_head('cond1_before', itimestep, i, k, j)
  else
    call pr_head('cond1_after', itimestep, i, k, j)
  end if
  call fsbm_put(pr_stream, 'ff1', ff1r)
  call fsbm_put(pr_stream, 'tt', tt)
  call fsbm_put(pr_stream, 'qq', qq)
  call fsbm_put(pr_stream, 'del1in', del1in)
  call fsbm_put(pr_stream, 'del2in', del2in)
  call fsbm_put(pr_stream, 'div1', div1)
  call fsbm_put(pr_stream, 'div2', div2)
  call fsbm_put(pr_stream, 'dtcond', dtcond)
  call fsbm_put(pr_stream, 'vr1', vr1z)
  call fsbm_put(pr_stream, 'pcgs', pcgs)
  call fsbm_put(pr_stream, 'rhocgs', rho)
  call fsbm_end(pr_stream)
end subroutine fsbm_probe_cond1

! Anchors: immediately before (phase 0) and after (phase 1) COAL_BOTT_NEW.
subroutine fsbm_probe_coal(phase, itimestep, i, k, j, ff1r, tt, qq, del1in, &
                           del2in, pp, rho, dt_coll)
  use fsbm_io
  use fsbm_probe_state
  implicit none
  integer, intent(in) :: phase, itimestep, i, k, j
  real(4), intent(in) :: ff1r(PR_NKR), rho, dt_coll
  real(8), intent(in) :: tt, qq, del1in, del2in, pp
  if (phase == 0) call pr_bump(i, C_COAL_CALL)
  if (phase == 0) call pr_bump2(i, j, K_COAL_CALL)
  if (.not. pr_point_on3(itimestep, i, k, j)) return
  if (phase == 0) then
    call pr_head('coal_before', itimestep, i, k, j)
  else
    call pr_head('coal_after', itimestep, i, k, j)
  end if
  call fsbm_put(pr_stream, 'ff1', ff1r)
  call fsbm_put(pr_stream, 'tt', tt)
  call fsbm_put(pr_stream, 'qq', qq)
  call fsbm_put(pr_stream, 'del1in', del1in)
  call fsbm_put(pr_stream, 'del2in', del2in)
  call fsbm_put(pr_stream, 'pp', pp)
  call fsbm_put(pr_stream, 'rhocgs', rho)
  call fsbm_put(pr_stream, 'dt_coll', dt_coll)
  call fsbm_end(pr_stream)
end subroutine fsbm_probe_coal

! Anchors inside COAL_BOTT_NEW: before (phase 0) and after (phase 1) the
! drop-drop coll_xxx_lwf call. Phase 0 also writes the module's cwll (after
! the call's Kernals_KS) and xl_mg for the first PR_KERNEL_CAP selected calls.
subroutine fsbm_probe_xxx(phase, itimestep, i, k, j, g1, cwll, xl_mg, pp)
  use fsbm_io
  use fsbm_probe_state
  implicit none
  integer, intent(in) :: phase, itimestep, i, k, j
  real(8), intent(in) :: g1(PR_NKR), cwll(PR_NKR, PR_NKR), xl_mg(PR_NKR), pp
  if (pr_capture) then
    if (phase == 0) then
      cap_g1_xxx_in = g1
    else
      cap_g1_xxx_out = g1
    end if
  end if
  if (.not. pr_point_on3(itimestep, i, k, j)) return
  if (phase == 0) then
    call pr_head('xxx_before', itimestep, i, k, j)
  else
    call pr_head('xxx_after', itimestep, i, k, j)
  end if
  call fsbm_put(pr_stream, 'g1', g1)
  call fsbm_put(pr_stream, 'pp', pp)
  call fsbm_end(pr_stream)
  if (phase == 0 .and. pr_kernel_written < pr_kcap_ll) then
    call pr_head('kernel', itimestep, i, k, j)
    call fsbm_put(pr_stream, 'pp', pp)
    call fsbm_put(pr_stream, 'cwll', cwll)
    call fsbm_put(pr_stream, 'xl_mg', xl_mg)
    call fsbm_end(pr_stream)
    pr_kernel_written = pr_kernel_written + 1
  end if
end subroutine fsbm_probe_xxx

! Anchor inside COAL_BOTT_NEW: after label 11, where the collisional breakup
! block (entered when icol_drop and icol_drop_brk are both 1) has finished,
! whether it completed (path 1) or gave up at ndiv >= 8 and skipped breakup
! (path 2, FF1R holds the negative value that stopped it). Path 0: no breakup.
subroutine fsbm_probe_brk(itimestep, i, k, j, icol_drop, icol_drop_brk, ndiv, &
                          g1, ff1r, break_drop_per)
  use fsbm_io
  use fsbm_probe_state
  implicit none
  integer, intent(in) :: itimestep, i, k, j, icol_drop, icol_drop_brk, ndiv
  real(8), intent(in) :: g1(PR_NKR), break_drop_per
  real(4), intent(in) :: ff1r(PR_NKR)
  integer :: path, ndiv_rec
  real(8) :: per_rec
  path = 0
  ndiv_rec = -1
  per_rec = 0.0d0
  if (icol_drop == 1 .and. icol_drop_brk == 1) then
    ndiv_rec = ndiv
    if (any(ff1r < 0.0)) then
      path = 2
    else
      path = 1
      per_rec = break_drop_per
    end if
    call pr_bump(i, C_COAL_BRK)
    call pr_bump2(i, j, K_COAL_BRK)
    if (path == 1) call pr_bump(i, C_COAL_BRK_DONE)
    if (path == 1) call pr_bump2(i, j, K_COAL_BRK_DONE)
    if (path == 2) call pr_bump(i, C_COAL_BRK_SKIP)
    if (path == 2) call pr_bump2(i, j, K_COAL_BRK_SKIP)
    select case (ndiv)
    case (1)
      call pr_bump(i, C_NDIV1)
    case (2)
      call pr_bump(i, C_NDIV2)
    case (4)
      call pr_bump(i, C_NDIV4)
    case (8)
      call pr_bump(i, C_NDIV8)
    case default
      call pr_bump(i, C_NDIV_MORE)
    end select
  end if
  if (pr_capture) then
    cap_icol_drop = icol_drop
    cap_icol_brk = icol_drop_brk
    cap_ndiv = ndiv_rec
    cap_path = path
    cap_g1_at11 = g1
    cap_break_per = per_rec
  end if
  if (.not. pr_point_on3(itimestep, i, k, j)) return
  call pr_head('breakup', itimestep, i, k, j)
  call fsbm_put(pr_stream, 'icol_drop', int(icol_drop, 4))
  call fsbm_put(pr_stream, 'icol_drop_brk', int(icol_drop_brk, 4))
  call fsbm_put(pr_stream, 'ndiv', int(ndiv_rec, 4))
  call fsbm_put(pr_stream, 'path', int(path, 4))
  call fsbm_put(pr_stream, 'g1', g1)
  call fsbm_put(pr_stream, 'break_drop_per', per_rec)
  call fsbm_end(pr_stream)
end subroutine fsbm_probe_brk

! Anchors: immediately before (phase 0) and after (phase 1) Spont_Rain_BreakUp.
subroutine fsbm_probe_spont(phase, itimestep, i, k, j, ff1r_d, ikr_spon_break)
  use fsbm_io
  use fsbm_probe_state
  implicit none
  integer, intent(in) :: phase, itimestep, i, k, j, ikr_spon_break
  real(8), intent(in) :: ff1r_d(PR_NKR)
  integer :: imax, kr
  if (phase == 0) then
    call pr_bump(i, C_SPONT_CALL)
    call pr_bump2(i, j, K_SPONT_CALL)
    imax = PR_NKR
    do kr = PR_NKR, 1, -1
      imax = kr
      if (ff1r_d(kr) > 0.0d0) exit
    end do
    ! Spont_Rain_BreakUp returns early unless SUM(FF1R) > NKR*1D-30 and the
    ! largest non-empty bin reaches ikr_spon_break.
    if (sum(ff1r_d) > PR_NKR*1.0d-30 .and. imax >= ikr_spon_break) then
      call pr_bump(i, C_SPONT_TAKEN)
      call pr_bump2(i, j, K_SPONT_TAKEN)
    end if
  end if
  if (.not. pr_point_on3(itimestep, i, k, j)) return
  if (phase == 0) then
    call pr_head('spont_before', itimestep, i, k, j)
  else
    call pr_head('spont_after', itimestep, i, k, j)
  end if
  call fsbm_put(pr_stream, 'ff1', ff1r_d)
  call fsbm_put(pr_stream, 'ikr_spon_break', int(ikr_spon_break, 4))
  call fsbm_end(pr_stream)
end subroutine fsbm_probe_spont

! Anchors: immediately before (phase 0) and after (phase 1) each of the three
! FALFLUXHUCM_Z calls (species 1 drops, 2 snow, 3 hail or graupel).
subroutine fsbm_probe_fall(phase, species, itimestep, i, j, ffx, vrx, rho, p, z, kts, kte)
  use fsbm_io
  use fsbm_probe_state
  implicit none
  integer, intent(in) :: phase, species, itimestep, i, j, kts, kte
  real(4), intent(in) :: ffx(kts:kte, PR_NKR), vrx(kts:kte, PR_NKR)
  real(4), intent(in) :: rho(kts:kte), p(kts:kte), z(kts:kte)
  if (phase == 0 .and. species == 1) call pr_bump(i, C_FALL_DROP)
  if (phase == 0 .and. species == 1) call pr_bump2(i, j, K_FALL_DROP)
  if (phase == 0 .and. species == 2) call pr_bump2(i, j, K_FALL_SNOW)
  if (phase == 0 .and. species == 3) call pr_bump2(i, j, K_FALL_DENSE)
  if (.not. (pr_step_on(itimestep) .and. pr_col_on(i, j))) return
  if (species == 1) then
    if (phase == 0) then
      call pr_head('fall_before', itimestep, i, 0, j)
    else
      call pr_head('fall_after', itimestep, i, 0, j)
    end if
    call fsbm_put(pr_stream, 'ffx', ffx)
    call fsbm_put(pr_stream, 'vrx', vrx)
    call fsbm_put(pr_stream, 'rhocgs', rho)
    call fsbm_put(pr_stream, 'pcgs', p)
    call fsbm_put(pr_stream, 'zcgs', z)
  else
    if (phase == 0) then
      call pr_head('fall_ice_before', itimestep, i, 0, j)
    else
      call pr_head('fall_ice_after', itimestep, i, 0, j)
    end if
    call fsbm_put(pr_stream, 'species', int(species, 4))
    call fsbm_put(pr_stream, 'ffx', ffx)
  end if
  call fsbm_end(pr_stream)
end subroutine fsbm_probe_fall

! Anchor: after the diagnostics and surface precipitation loop of FAST_SBM.
! vr1z is FAST_SBM's VR1_Z(NKR,KTS:KTE), which has no column index: after the
! loop it holds the tile's LAST column (design finding D1).
subroutine fsbm_probe_diag(itimestep, qc, qr, qnc, qnr, qna, qv, th, rainnc, &
                           rainncv, vr1z, ims, ime, kms, kme, jms, jme, &
                           its, ite, kts, kte, jts, jte)
  use fsbm_io
  use fsbm_probe_state
  implicit none
  integer, intent(in) :: itimestep, ims, ime, kms, kme, jms, jme
  integer, intent(in) :: its, ite, kts, kte, jts, jte
  real(4), intent(in), dimension(ims:ime, kms:kme, jms:jme) :: qc, qr, qnc, qnr, qna, qv, th
  real(4), intent(in) :: rainnc(ims:ime, jms:jme), rainncv(ims:ime, jms:jme)
  real(4), intent(in) :: vr1z(PR_NKR, kts:kte)
  integer :: i, j
  if (.not. pr_step_on(itimestep)) return
  do j = jts, jte
    do i = its, ite
      if (.not. pr_col_on(i, j)) cycle
      call pr_head('diag', itimestep, i, 0, j)
      call fsbm_put(pr_stream, 'qc', qc(i, kts:kte, j))
      call fsbm_put(pr_stream, 'qr', qr(i, kts:kte, j))
      call fsbm_put(pr_stream, 'qnc', qnc(i, kts:kte, j))
      call fsbm_put(pr_stream, 'qnr', qnr(i, kts:kte, j))
      call fsbm_put(pr_stream, 'qna', qna(i, kts:kte, j))
      call fsbm_put(pr_stream, 'qv', qv(i, kts:kte, j))
      call fsbm_put(pr_stream, 'th', th(i, kts:kte, j))
      call fsbm_put(pr_stream, 'rainnc', rainnc(i, j))
      call fsbm_put(pr_stream, 'rainncv', rainncv(i, j))
      call fsbm_put(pr_stream, 'vr1_z_level1_after_loop', vr1z(:, kts))
      call fsbm_end(pr_stream)
    end do
  end do
end subroutine fsbm_probe_diag
! ===========================================================================
! Milestone 2: the cold anchors. Every routine below is inert: it reads its
! arguments, bumps counters, writes records for the selected points and keeps
! capture values for the sweeps. Kinds follow the actuals at each call site:
! real(4) where WRF declares REAL(KIND=R4SIZE) (promoted with the WRF source
! in the O64 build), real where WRF declares default REAL (likewise), and
! real(8) where WRF declares REAL(KIND=R8SIZE) or DOUBLE PRECISION.
! ===========================================================================

! Anchor: after the ice pre-pass of FAST_SBM (:4512). Called on every call,
! before anything else in the point loop; sets pr_cur_step for the anchors
! inside JERNUCL01_KS, which has no step argument. Counts the points where
! the DSUPICE_XYZ branch (:4441) is taken with T_OLD below 238.15 K, where
! w_stag is the previous point's (finding D15). Records DSUPICE_XYZ,
! DTFREEZ_XYZ and the cloud base of the whole tile, and T_OLD, QV_OLD and
! pcgs over the filled range i_start..i_end, j_start..j_end (tile plus halo).
subroutine fsbm_probe_prepass(itimestep, dsupice, dtfreez, kzcb, t_old, qv_old, pcgs, &
                              i_start, i_end, j_start, j_end, ims, ime, kms, kme, &
                              jms, jme, its, ite, kts, kte, jts, jte)
  use fsbm_io
  use fsbm_probe_state
  implicit none
  integer, intent(in) :: itimestep, i_start, i_end, j_start, j_end
  integer, intent(in) :: ims, ime, kms, kme, jms, jme, its, ite, kts, kte, jts, jte
  real(4), intent(in) :: dsupice(ite, kte, jte), dtfreez(ite, kte, jte)
  integer, intent(in) :: kzcb(its:ite, jts:jte)
  real(4), intent(in), dimension(its-1:ite+1, kts:kte, jts-1:jte+1) :: t_old, pcgs
  real, intent(in) :: qv_old(ims:ime, kms:kme, jms:jme)
  real(4) :: aa2_my, bb2_my, supice
  real(8) :: es2n, ew1n
  integer :: i, j, k
  data aa2_my, bb2_my /3.41E13, 6.13E3/
  pr_cur_step = itimestep
  if (pr_count .and. pr_cni > 0) then
    do j = jts, jte
      do i = its, ite
        do k = kts, kte
          ! FAST_SBM :4399-4402, then the branch condition of :4441.
          es2n = aa2_my*exp(-bb2_my/t_old(i, k, j))
          ew1n = qv_old(i, k, j)*pcgs(i, k, j)/(0.622+0.378*qv_old(i, k, j))
          supice = ew1n/es2n - 1.
          if (supice > 0.5) supice = .5
          if (supice >= 0.02 .and. t_old(i, k, j) < 268.15 .and. t_old(i, k, j) < 238.15) &
            call pr_bump2(i, j, K_D15)
        end do
      end do
    end do
  end if
  if (.not. (pr_cni > 0 .and. pr_step_on(itimestep))) return
  call fsbm_begin(pr_stream, 'prepass')
  call fsbm_put(pr_stream, 'step', int(itimestep, 4))
  call fsbm_put(pr_stream, 'i_range', int([i_start, i_end], 4))
  call fsbm_put(pr_stream, 'j_range', int([j_start, j_end], 4))
  call fsbm_put(pr_stream, 'dsupice_xyz', dsupice(its:ite, kts:kte, jts:jte))
  call fsbm_put(pr_stream, 'dtfreez_xyz', dtfreez(its:ite, kts:kte, jts:jte))
  call fsbm_put(pr_stream, 'kz_cloud_base', int(kzcb, 4))
  call fsbm_put(pr_stream, 't_old', t_old(i_start:i_end, kts:kte, j_start:j_end))
  call fsbm_put(pr_stream, 'qv_old', qv_old(i_start:i_end, kts:kte, j_start:j_end))
  call fsbm_put(pr_stream, 'pcgs', pcgs(i_start:i_end, kts:kte, j_start:j_end))
  call fsbm_end(pr_stream)
end subroutine fsbm_probe_prepass

! Anchor: after JERNUCL01_KS in FAST_SBM (:4664), before its crystals are
! moved into snow (:4667-4674).
subroutine fsbm_probe_nuclice(itimestep, i, k, j, ff2in, tt, qq, del1in, del2in, &
                              sup2_old, dsupice)
  use fsbm_io
  use fsbm_probe_state
  implicit none
  integer, intent(in) :: itimestep, i, k, j
  real(4), intent(in) :: ff2in(PR_NKR, PR_ICEMAX), sup2_old, dsupice
  real(8), intent(in) :: tt, qq, del1in, del2in
  if (.not. pr_cold_on(itimestep, i, k, j)) return
  call pr_head('nucl_ice', itimestep, i, k, j)
  call fsbm_put(pr_stream, 'ff2', ff2in)
  call fsbm_put(pr_stream, 'tt', tt)
  call fsbm_put(pr_stream, 'qq', qq)
  call fsbm_put(pr_stream, 'del1in', del1in)
  call fsbm_put(pr_stream, 'del2in', del2in)
  call fsbm_put(pr_stream, 'sup2_old', sup2_old)
  call fsbm_put(pr_stream, 'dsupice_xyz', dsupice)
  call fsbm_end(pr_stream)
end subroutine fsbm_probe_nuclice

! Anchor inside JERNUCL01_KS: just before the heat of homogeneous freezing
! (:3108), after the drops have been moved to plates.
subroutine fsbm_probe_homfrz(i, k, j, tt, sum_ice, rori)
  use fsbm_io
  use fsbm_probe_state
  implicit none
  integer, intent(in) :: i, k, j
  real(8), intent(in) :: tt, sum_ice, rori
  call pr_bump2(i, j, K_HOMFRZ)
  if (pr_capture) then
    capc_homfrz_calls = capc_homfrz_calls + 1
    capc_homfrz = [tt, sum_ice, rori]
  end if
  if (.not. pr_cold_on(pr_cur_step, i, k, j)) return
  call pr_head('homfrz', pr_cur_step, i, k, j)
  call fsbm_put(pr_stream, 'tt', tt)
  call fsbm_put(pr_stream, 'sum_ice', sum_ice)
  call fsbm_put(pr_stream, 'rori', rori)
  call fsbm_end(pr_stream)
end subroutine fsbm_probe_homfrz

! Anchors inside JERNUCL01_KS: before (phase 0) and after (phase 1) ICE_NUCL
! (:3126). ICE_NUCL is private to module_mp_SBM_Nucleation, so these two
! records are its only direct observation. The band is ICE_NUCL's own
! temperature split (:3388-3398) of its entry temperature: 1 above -4 C,
! 2 to -8.1, 3 to -12.7, 4 to -17.8, 5 to -22.4, 6 below.
subroutine fsbm_probe_icenucl(phase, i, k, j, psi2, sup2, tt, dsupice, sup2_old)
  use fsbm_io
  use fsbm_probe_state
  implicit none
  integer, intent(in) :: phase, i, k, j
  real(8), intent(in) :: psi2(PR_NKR, PR_ICEMAX), sup2, tt, dsupice, sup2_old
  integer, save :: band = 0
  real(8) :: tpc
  integer :: ityp
  integer, parameter :: band_type(6) = [2, 1, 2, 3, 2, 1]
  if (phase == 0) then
    pr_psi2_before = psi2
    band = pr_icenucl_band(tt)
    call pr_bump2(i, j, K_ICENUCL)
    if (band > 0) call pr_bump2(i, j, K_ICENUCL_BAND + band)
    tpc = tt - 273.15d0
    if (tpc < -5.0d0 .and. 100.0d0*sup2 > 59.0d0) call pr_bump2(i, j, K_ICENUCL_DEL2_CLAMP)
    if ((sup2 - sup2_old + dsupice)*100.0d0 > 59.0d0) call pr_bump2(i, j, K_ICENUCL_DSUP_CLAMP)
    if (pr_capture) then
      capc_icenucl_calls = capc_icenucl_calls + 1
      capc_icenucl_band = band
      capc_icenucl_added = 0
      capc_icenucl_psi2_in = psi2
      capc_icenucl_scal_in = [sup2, tt, dsupice, sup2_old]
    end if
  else
    ityp = 0
    if (band > 0) ityp = band_type(band)
    if (ityp > 0) then
      if (any(psi2(:, ityp) /= pr_psi2_before(:, ityp))) then
        call pr_bump2(i, j, K_ICENUCL_ADD + band)
        if (pr_capture) capc_icenucl_added = 1
      end if
    end if
    if (pr_capture) then
      capc_icenucl_psi2_out = psi2
      capc_icenucl_scal_out = [sup2, tt, dsupice, sup2_old]
    end if
  end if
  if (.not. pr_cold_on(pr_cur_step, i, k, j)) return
  if (phase == 0) then
    call pr_head('icenucl_before', pr_cur_step, i, k, j)
  else
    call pr_head('icenucl_after', pr_cur_step, i, k, j)
  end if
  call fsbm_put(pr_stream, 'psi2', psi2)
  call fsbm_put(pr_stream, 'sup2', sup2)
  call fsbm_put(pr_stream, 'tt', tt)
  call fsbm_put(pr_stream, 'dsupice_xyz', dsupice)
  call fsbm_put(pr_stream, 'sup2_old', sup2_old)
  call fsbm_put(pr_stream, 'band', int(band, 4))
  call fsbm_end(pr_stream)
end subroutine fsbm_probe_icenucl

! Anchor: before the regime dispatch of FAST_SBM (:4711). Counts which
! diffusional-growth routine the dispatch will call, with its own
! predicates, and the two no-branch cases of finding D16: mixed phase at or
! below 233.15 K (:4744) and ice without drops at or above -0.187 C (:4724).
subroutine fsbm_probe_dispatch(itimestep, i, k, j, tt, isym1, isym2, isym3, isym4, isym5)
  use fsbm_io
  use fsbm_probe_state
  implicit none
  integer, intent(in) :: itimestep, i, k, j, isym1, isym2(PR_ICEMAX), isym3, isym4, isym5
  real(8), intent(in) :: tt
  logical :: ice
  ice = sum(isym2) > 1 .or. isym3 == 1 .or. isym4 == 1 .or. isym5 == 1
  if (isym1 == 1 .and. ((tt-273.15) > -0.187 .or. (sum(isym2) == 0 .and. isym3 == 0 .and. &
      isym4 == 0 .and. isym5 == 0))) then
    call pr_bump2(i, j, K_DISP1)
  else if (isym1 == 0 .and. (tt-273.15) < -0.187 .and. ice) then
    call pr_bump2(i, j, K_DISP2)
  else if (isym1 == 1 .and. (tt-273.15) < -0.187 .and. ice) then
    if (tt > 233.15) then
      call pr_bump2(i, j, K_DISP3)
    else
      call pr_bump2(i, j, K_DISP_D16_COLD)
    end if
  else if (isym1 == 0 .and. ice .and. .not. ((tt-273.15) < -0.187)) then
    call pr_bump2(i, j, K_DISP_D16_WARM)
  else
    call pr_bump2(i, j, K_DISP_OTHER)
  end if
end subroutine fsbm_probe_dispatch

! Anchors: before (phase 0) and after (phase 1) ONECOND2 (:4730-4740).
subroutine fsbm_probe_cond2(phase, itimestep, i, k, j, ff2r, ff3r, ff4r, ff5r, tt, qq, &
                            del1in, del2in, div1, div2, dtcond, pcgs, rho, isym2, isym3, &
                            isym4, isym5)
  use fsbm_io
  use fsbm_probe_state
  implicit none
  integer, intent(in) :: phase, itimestep, i, k, j, isym2(PR_ICEMAX), isym3, isym4, isym5
  real(4), intent(in) :: ff2r(PR_NKR, PR_ICEMAX), ff3r(PR_NKR), ff4r(PR_NKR), ff5r(PR_NKR)
  real(4), intent(in) :: dtcond, pcgs, rho
  real(8), intent(in) :: tt, qq, del1in, del2in, div1, div2
  if (phase == 0) then
    call pr_bump2(i, j, K_COND2)
    ! ONECOND2 :7436-7437
    if (del2in > 0.0) then
      call pr_bump2(i, j, K_COND2_K21)
    else
      call pr_bump2(i, j, K_COND2_K20)
    end if
  end if
  if (.not. pr_cold_on(itimestep, i, k, j)) return
  if (phase == 0) then
    call pr_head('cond2_before', itimestep, i, k, j)
  else
    call pr_head('cond2_after', itimestep, i, k, j)
  end if
  call fsbm_put(pr_stream, 'ff2', ff2r)
  call fsbm_put(pr_stream, 'ff3', ff3r)
  call fsbm_put(pr_stream, 'ff4', ff4r)
  call fsbm_put(pr_stream, 'ff5', ff5r)
  call fsbm_put(pr_stream, 'tt', tt)
  call fsbm_put(pr_stream, 'qq', qq)
  call fsbm_put(pr_stream, 'del1in', del1in)
  call fsbm_put(pr_stream, 'del2in', del2in)
  call fsbm_put(pr_stream, 'div1', div1)
  call fsbm_put(pr_stream, 'div2', div2)
  call fsbm_put(pr_stream, 'dtcond', dtcond)
  call fsbm_put(pr_stream, 'pcgs', pcgs)
  call fsbm_put(pr_stream, 'rhocgs', rho)
  call fsbm_put(pr_stream, 'isym', int([0, isym2, isym3, isym4, isym5], 4))
  call fsbm_end(pr_stream)
end subroutine fsbm_probe_cond2

! Anchors: before (phase 0) and after (phase 1) ONECOND3 (:4748-4759).
subroutine fsbm_probe_cond3(phase, itimestep, i, k, j, ff1r, ff2r, ff3r, ff4r, ff5r, tt, qq, &
                            del1in, del2in, div1, div2, dtcond, pcgs, rho, isym1, isym2, &
                            isym3, isym4, isym5)
  use fsbm_io
  use fsbm_probe_state
  implicit none
  integer, intent(in) :: phase, itimestep, i, k, j, isym1, isym2(PR_ICEMAX), isym3, isym4, isym5
  real(4), intent(in) :: ff1r(PR_NKR), ff2r(PR_NKR, PR_ICEMAX), ff3r(PR_NKR), ff4r(PR_NKR), &
                         ff5r(PR_NKR)
  real(4), intent(in) :: dtcond, pcgs, rho
  real(8), intent(in) :: tt, qq, del1in, del2in, div1, div2
  integer :: kcond
  if (phase == 0) then
    call pr_bump2(i, j, K_COND3)
    ! ONECOND3 :8001-8004
    kcond = 50
    if (del1in < 0.0 .and. del2in < 0.0) kcond = 30
    if (del1in > 0.0 .and. del2in > 0.0) kcond = 31
    if (del1in < 0.0 .and. del2in > 0.0) kcond = 32
    select case (kcond)
    case (30)
      call pr_bump2(i, j, K_COND3_K30)
    case (31)
      call pr_bump2(i, j, K_COND3_K31)
    case (32)
      call pr_bump2(i, j, K_COND3_K32)
    case default
      call pr_bump2(i, j, K_COND3_K50)
    end select
  end if
  if (.not. pr_cold_on(itimestep, i, k, j)) return
  if (phase == 0) then
    call pr_head('cond3_before', itimestep, i, k, j)
  else
    call pr_head('cond3_after', itimestep, i, k, j)
  end if
  call fsbm_put(pr_stream, 'ff1', ff1r)
  call fsbm_put(pr_stream, 'ff2', ff2r)
  call fsbm_put(pr_stream, 'ff3', ff3r)
  call fsbm_put(pr_stream, 'ff4', ff4r)
  call fsbm_put(pr_stream, 'ff5', ff5r)
  call fsbm_put(pr_stream, 'tt', tt)
  call fsbm_put(pr_stream, 'qq', qq)
  call fsbm_put(pr_stream, 'del1in', del1in)
  call fsbm_put(pr_stream, 'del2in', del2in)
  call fsbm_put(pr_stream, 'div1', div1)
  call fsbm_put(pr_stream, 'div2', div2)
  call fsbm_put(pr_stream, 'dtcond', dtcond)
  call fsbm_put(pr_stream, 'pcgs', pcgs)
  call fsbm_put(pr_stream, 'rhocgs', rho)
  call fsbm_put(pr_stream, 'isym', int([isym1, isym2, isym3, isym4, isym5], 4))
  call fsbm_end(pr_stream)
end subroutine fsbm_probe_cond3

! Anchors inside ONECOND2 (sites 21 and 20, :7460 and :7543) and ONECOND3
! (site 3, :8057): the rates JERSUPSAT_KS receives and the branch it takes,
! classified with its own predicates (the same classification as the
! jersupsat sweep of milestone 1).
subroutine fsbm_probe_supsat(site, itimestep, i, k, j, kcond, del1, del2, rw, pw, ri, pi, &
                             dtt, isym1, isym2, isym3, isym4, isym5)
  use fsbm_io
  use fsbm_probe_state
  implicit none
  integer, intent(in) :: site, itimestep, i, k, j, kcond, isym1, isym2(PR_ICEMAX), isym3, &
                         isym4, isym5
  real(8), intent(in) :: del1, del2, rw, pw, ri, pi
  real, intent(in) :: dtt
  integer :: br, clamp, n
  br = pr_supsat_branch(rw, pw, ri, pi, dtt, clamp)
  call pr_bump2(i, j, K_SUPSAT_B + br)
  if (clamp == 1) call pr_bump2(i, j, K_SUPSAT_CLAMP)
  if (pr_capture) then
    capc_supsat_n = capc_supsat_n + 1
    n = min(capc_supsat_n, 4)
    capc_supsat_site(n) = site
    capc_supsat_kcond(n) = kcond
    capc_supsat_branch(n) = br
    capc_supsat_clamp(n) = clamp
    capc_supsat_rates(:, n) = [del1, del2, rw, pw, ri, pi, real(dtt, 8)]
  end if
  if (.not. pr_cold_on(itimestep, i, k, j)) return
  call pr_head('supsat', itimestep, i, k, j)
  call fsbm_put(pr_stream, 'site', int(site, 4))
  call fsbm_put(pr_stream, 'kcond', int(kcond, 4))
  call fsbm_put(pr_stream, 'del1', del1)
  call fsbm_put(pr_stream, 'del2', del2)
  call fsbm_put(pr_stream, 'rw', rw)
  call fsbm_put(pr_stream, 'pw', pw)
  call fsbm_put(pr_stream, 'ri', ri)
  call fsbm_put(pr_stream, 'pi', pi)
  call fsbm_put(pr_stream, 'dtt', dtt)
  call fsbm_put(pr_stream, 'isym', int([isym1, isym2, isym3, isym4, isym5], 4))
  call fsbm_put(pr_stream, 'branch', int(br, 4))
  call fsbm_put(pr_stream, 'g2_clamp', int(clamp, 4))
  call fsbm_end(pr_stream)
end subroutine fsbm_probe_supsat

! Anchors: before (phase 0) and after (phase 1) COAL_BOTT_NEW in FAST_SBM
! (:4774-4779), beside milestone 1's coal_before/coal_after: the ice spectra,
! the liquid and rimed fractions the call receives and returns.
subroutine fsbm_probe_coalice(phase, itimestep, i, k, j, ff3r, ff4r, ff5r, fliqfr_s, &
                              fliqfr_g, fliqfr_h, frimfr_s)
  use fsbm_io
  use fsbm_probe_state
  implicit none
  integer, intent(in) :: phase, itimestep, i, k, j
  real(4), intent(in) :: ff3r(PR_NKR), ff4r(PR_NKR), ff5r(PR_NKR)
  real(8), intent(in) :: fliqfr_s(PR_NKR), fliqfr_g(PR_NKR), fliqfr_h(PR_NKR), frimfr_s(PR_NKR)
  if (.not. pr_cold_on(itimestep, i, k, j)) return
  if (phase == 0) then
    call pr_head('coal_ice_before', itimestep, i, k, j)
  else
    call pr_head('coal_ice_after', itimestep, i, k, j)
  end if
  call fsbm_put(pr_stream, 'ff3', ff3r)
  call fsbm_put(pr_stream, 'ff4', ff4r)
  call fsbm_put(pr_stream, 'ff5', ff5r)
  call fsbm_put(pr_stream, 'fliqfr_s', fliqfr_s)
  call fsbm_put(pr_stream, 'fliqfr_g', fliqfr_g)
  call fsbm_put(pr_stream, 'fliqfr_h', fliqfr_h)
  call fsbm_put(pr_stream, 'frimfr_s', frimfr_s)
  call fsbm_end(pr_stream)
end subroutine fsbm_probe_coalice

! Anchors inside COAL_BOTT_NEW: before the cold block (phase 0, :8440) and
! after it (phase 1, :8568). prdkrn is MODKRN_KS's output, which is left
! unset above 0 C (finding D6) and at exactly -40 C (INTERPOL_SE leaves X
! unset when H equals its last node); it is recorded only where it is set.
! Phase 0 also writes the seven kernel slices the block reads, for the first
! pr_kcold_cap recorded calls that enter the block.
subroutine fsbm_probe_coldcoal(phase, itimestep, i, k, j, g1, g3, g4, g5, rf3, tt, alwc, &
                               prdkrn, icol_drop, icol_snow, icol_graupel, icol_hail, pp, &
                               cwls, cwsl, cwlh, cwhl, cwss, cwlg, cwgl)
  use fsbm_io
  use fsbm_probe_state
  implicit none
  integer, intent(in) :: phase, itimestep, i, k, j, icol_drop, icol_snow, icol_graupel, icol_hail
  real(8), intent(in) :: g1(PR_NKR), g3(PR_NKR), g4(PR_NKR), g5(PR_NKR), rf3(PR_NKR)
  real(8), intent(in) :: tt, prdkrn, pp
  real(4), intent(in) :: alwc
  real(8), intent(in), dimension(PR_NKR, PR_NKR) :: cwls, cwsl, cwlh, cwhl, cwss, cwlg, cwgl
  real(8) :: tc, prd
  logical :: set, entered
  tc = tt - 273.15
  set = tc <= 0.0 .and. tc /= -40.0
  prd = 0.0d0
  if (set) prd = prdkrn
  entered = tt <= 273.15 .and. (icol_drop == 1 .or. icol_snow == 1)
  if (phase == 0) then
    if (entered) call pr_bump2(i, j, K_COAL_COLD)
    if (pr_capture) then
      capc_cold_block = merge(1, 0, entered)
      capc_cold_icol = [icol_drop, icol_snow, icol_graupel, icol_hail]
      capc_cold_g_in(:, 1) = g1
      capc_cold_g_in(:, 2) = g3
      capc_cold_g_in(:, 3) = g4
      capc_cold_g_in(:, 4) = g5
      capc_cold_scal = [real(alwc, 8), prd, merge(1.0d0, 0.0d0, set)]
    end if
  else if (pr_capture) then
    capc_cold_g_out(:, 1) = g1
    capc_cold_g_out(:, 2) = g3
    capc_cold_g_out(:, 3) = g4
    capc_cold_g_out(:, 4) = g5
    capc_cold_rf3_out = rf3
  end if
  if (.not. pr_cold_on(itimestep, i, k, j)) return
  if (phase == 0) then
    call pr_head('coal_cold_before', itimestep, i, k, j)
  else
    call pr_head('coal_cold_after', itimestep, i, k, j)
  end if
  call fsbm_put(pr_stream, 'g1', g1)
  call fsbm_put(pr_stream, 'g3', g3)
  call fsbm_put(pr_stream, 'g4', g4)
  call fsbm_put(pr_stream, 'g5', g5)
  call fsbm_put(pr_stream, 'rf3', rf3)
  call fsbm_put(pr_stream, 'tt', tt)
  call fsbm_put(pr_stream, 'alwc', alwc)
  call fsbm_put(pr_stream, 'prdkrn', prd)
  call fsbm_put(pr_stream, 'prdkrn_set', int(merge(1, 0, set), 4))
  call fsbm_put(pr_stream, 'icol', int([icol_drop, icol_snow, icol_graupel, icol_hail], 4))
  call fsbm_end(pr_stream)
  if (phase == 0 .and. entered .and. pr_kcold_written < pr_kcold_cap) then
    call pr_head('kernel_cold', itimestep, i, k, j)
    call fsbm_put(pr_stream, 'pp', pp)
    call fsbm_put(pr_stream, 'cwls', cwls)
    call fsbm_put(pr_stream, 'cwsl', cwsl)
    call fsbm_put(pr_stream, 'cwlh', cwlh)
    call fsbm_put(pr_stream, 'cwhl', cwhl)
    call fsbm_put(pr_stream, 'cwss', cwss)
    call fsbm_put(pr_stream, 'cwlg', cwlg)
    call fsbm_put(pr_stream, 'cwgl', cwgl)
    call fsbm_end(pr_stream)
    pr_kcold_written = pr_kcold_written + 1
  end if
end subroutine fsbm_probe_coldcoal

! Anchors inside COAL_BOTT_NEW: one before each of the twelve collision calls
! of the cold block (:8452 to :8563). Counting only.
!  1 xyz drop-snow to hail   2 xyz drop-snow to graupel   3 xyx snow-drop
!  4 xyz snow-drop to hail   5 xyz snow-drop to graupel   6 xyy drop-graupel
!  7 xyx graupel-drop        8 xyz drop-graupel to hail   9 xyz graupel-drop to hail
! 10 xyy drop-hail          11 xyx hail-drop             12 xxx snow-snow
subroutine fsbm_probe_cc(id, itimestep, i, k, j)
  use fsbm_probe_state
  implicit none
  integer, intent(in) :: id, itimestep, i, k, j
  call pr_bump2(i, j, K_CC + id)
  if (pr_capture) capc_cc_mask = ior(capc_cc_mask, ishft(1, id - 1))
end subroutine fsbm_probe_cc

! Anchor inside COAL_BOTT_NEW: after the Hallett-Mossop splinters are added
! to snow bin 1 (:8553), inside the block that sets every value recorded.
subroutine fsbm_probe_hm(itimestep, i, k, j, tt, conc_old, conc_new, dconc, conc_icempl)
  use fsbm_io
  use fsbm_probe_state
  implicit none
  integer, intent(in) :: itimestep, i, k, j
  real(8), intent(in) :: tt
  real(4), intent(in) :: conc_old, conc_new, dconc, conc_icempl
  call pr_bump2(i, j, K_HM_BLOCK)
  if (conc_icempl > 0.0) call pr_bump2(i, j, K_HM_SPLINTERS)
  if (pr_capture) then
    capc_hm_fired = 1
    capc_hm = [tt, real(conc_old, 8), real(conc_new, 8), real(dconc, 8), real(conc_icempl, 8)]
  end if
  if (.not. pr_cold_on(itimestep, i, k, j)) return
  call pr_head('hm', itimestep, i, k, j)
  call fsbm_put(pr_stream, 'tt', tt)
  call fsbm_put(pr_stream, 'conc_old', conc_old)
  call fsbm_put(pr_stream, 'conc_new', conc_new)
  call fsbm_put(pr_stream, 'dconc', dconc)
  call fsbm_put(pr_stream, 'conc_icempl', conc_icempl)
  call fsbm_end(pr_stream)
end subroutine fsbm_probe_hm

! Anchor inside COAL_BOTT_NEW: before label 61 (:8604), after the riming
! heat (:8586-8602).
subroutine fsbm_probe_rime(itimestep, i, k, j, tt, t_new, deldrop, cont_init_drop, &
                           cont_fin_drop, cont_init_ice, cont_fin_ice, del1in, del2in)
  use fsbm_io
  use fsbm_probe_state
  implicit none
  integer, intent(in) :: itimestep, i, k, j
  real(8), intent(in) :: tt, del1in, del2in
  real(4), intent(in) :: t_new, deldrop, cont_init_drop, cont_fin_drop, cont_init_ice, &
                         cont_fin_ice
  real(4) :: t0
  t0 = tt
  if (t0 <= 273.15 .and. deldrop > 0.0) call pr_bump2(i, j, K_RIME_HEAT)
  if (pr_capture) capc_rime = [tt, real(t_new, 8), real(deldrop, 8), real(cont_init_drop, 8), &
                               real(cont_fin_drop, 8), real(cont_init_ice, 8), &
                               real(cont_fin_ice, 8), del2in]
  if (.not. pr_cold_on(itimestep, i, k, j)) return
  call pr_head('rime', itimestep, i, k, j)
  call fsbm_put(pr_stream, 'tt', tt)
  call fsbm_put(pr_stream, 't_new', t_new)
  call fsbm_put(pr_stream, 'deldrop', deldrop)
  call fsbm_put(pr_stream, 'cont_init_drop', cont_init_drop)
  call fsbm_put(pr_stream, 'cont_fin_drop', cont_fin_drop)
  call fsbm_put(pr_stream, 'cont_init_ice', cont_init_ice)
  call fsbm_put(pr_stream, 'cont_fin_ice', cont_fin_ice)
  call fsbm_put(pr_stream, 'del1in', del1in)
  call fsbm_put(pr_stream, 'del2in', del2in)
  call fsbm_end(pr_stream)
end subroutine fsbm_probe_rime

! Anchors: before (phase 0) and after (phase 1) FREEZ (which 1, :4795-4799)
! and J_W_MELT (which 2, :4805-4806).
subroutine fsbm_probe_phase(which, phase, itimestep, i, k, j, ff1r, ff2r, ff3r, ff4r, ff5r, &
                            t_new, rho)
  use fsbm_io
  use fsbm_probe_state
  implicit none
  integer, intent(in) :: which, phase, itimestep, i, k, j
  real(4), intent(in) :: ff1r(PR_NKR), ff2r(PR_NKR, PR_ICEMAX), ff3r(PR_NKR), ff4r(PR_NKR), &
                         ff5r(PR_NKR), t_new, rho
  if (phase == 0) then
    pr_ff_before(:, 1) = ff1r
    pr_ff_before(:, 2:4) = ff2r
    pr_ff_before(:, 5) = ff3r
    pr_ff_before(:, 6) = ff4r
    pr_ff_before(:, 7) = ff5r
    if (which == 1) then
      call pr_bump2(i, j, K_FREEZ)
    else
      call pr_bump2(i, j, K_MELT)
      ! J_W_MELT's bands: snow :6708-6719, graupel and hail :6721-6735.
      if (any(ff3r(1:14) > 0.0)) call pr_bump2(i, j, K_MELT_SNOW_B + 1)
      if (any(ff3r(15:21) > 0.0)) call pr_bump2(i, j, K_MELT_SNOW_B + 2)
      if (any(ff3r(22:PR_NKR) > 0.0)) call pr_bump2(i, j, K_MELT_SNOW_B + 3)
      if (any(ff4r(1:13) > 0.0) .or. any(ff5r(1:13) > 0.0)) call pr_bump2(i, j, K_MELT_DENSE_B + 1)
      if (any(ff4r(14:22) > 0.0) .or. any(ff5r(14:22) > 0.0)) &
        call pr_bump2(i, j, K_MELT_DENSE_B + 2)
      if (any(ff4r(23:PR_NKR) > 0.0) .or. any(ff5r(23:PR_NKR) > 0.0)) &
        call pr_bump2(i, j, K_MELT_DENSE_B + 3)
    end if
  else
    if (which == 1) then
      ! FREEZ :6614-6618: bins up to KRFREEZ (21) to plates, the rest to hail.
      if (any(real(ff2r(1:21, 2), 8) > pr_ff_before(1:21, 3))) call pr_bump2(i, j, K_FREEZ_PLATES)
      if (any(real(ff5r(22:PR_NKR), 8) > pr_ff_before(22:PR_NKR, 7))) &
        call pr_bump2(i, j, K_FREEZ_HAIL)
    else
      if (any(ff2r < 0.0) .or. any(ff3r < 0.0) .or. any(ff4r < 0.0) .or. any(ff5r < 0.0)) &
        call pr_bump2(i, j, K_MELT_NEG)
    end if
  end if
  if (.not. pr_cold_on(itimestep, i, k, j)) return
  if (which == 1) then
    if (phase == 0) then
      call pr_head('freez_before', itimestep, i, k, j)
    else
      call pr_head('freez_after', itimestep, i, k, j)
    end if
  else
    if (phase == 0) then
      call pr_head('melt_before', itimestep, i, k, j)
    else
      call pr_head('melt_after', itimestep, i, k, j)
    end if
  end if
  call fsbm_put(pr_stream, 'ff1', ff1r)
  call fsbm_put(pr_stream, 'ff2', ff2r)
  call fsbm_put(pr_stream, 'ff3', ff3r)
  call fsbm_put(pr_stream, 'ff4', ff4r)
  call fsbm_put(pr_stream, 'ff5', ff5r)
  call fsbm_put(pr_stream, 't_new', t_new)
  call fsbm_put(pr_stream, 'rhocgs', rho)
  call fsbm_end(pr_stream)
end subroutine fsbm_probe_phase

! Anchor: after the merges (:4809-4821), before spontaneous breakup (:4826).
subroutine fsbm_probe_merge(itimestep, i, k, j, ff1r, ff2r, ff3r, ff4r, ff5r)
  use fsbm_io
  use fsbm_probe_state
  implicit none
  integer, intent(in) :: itimestep, i, k, j
  real(4), intent(in) :: ff1r(PR_NKR), ff2r(PR_NKR, PR_ICEMAX), ff3r(PR_NKR), ff4r(PR_NKR), &
                         ff5r(PR_NKR)
  if (.not. pr_cold_on(itimestep, i, k, j)) return
  call pr_head('merge', itimestep, i, k, j)
  call fsbm_put(pr_stream, 'ff1', ff1r)
  call fsbm_put(pr_stream, 'ff3', ff3r)
  call fsbm_put(pr_stream, 'ff4', ff4r)
  call fsbm_put(pr_stream, 'ff5', ff5r)
  call fsbm_put(pr_stream, 'ff2_max', maxval(abs(ff2r)))
  call fsbm_end(pr_stream)
end subroutine fsbm_probe_merge

! Anchor: before the snow-breakup guard (:4836). The guard sums the slice
! FF3R(KR_SNOW_MIN:NKR), which is empty on the 33-bin grid (KR_SNOW_MIN is
! 34), so its sum is 0 and the guard 0 > (33 - 34) 1e-30 is true at every
! point; BreakUp_Snow is then called only if KR_SNOW_MAX (35) <= NKR.
subroutine fsbm_probe_snowbrk(itimestep, i, k, j, ff3r, kr_snow_min, kr_snow_max)
  use fsbm_io
  use fsbm_probe_state
  implicit none
  integer, intent(in) :: itimestep, i, k, j, kr_snow_min, kr_snow_max
  real(4), intent(in) :: ff3r(PR_NKR)
  logical :: guard, calls
  guard = sum(ff3r(kr_snow_min:PR_NKR)) > (PR_NKR-kr_snow_min)*1.0D-30
  calls = guard .and. kr_snow_max <= PR_NKR
  if (guard) call pr_bump2(i, j, K_SNOWBRK_GUARD)
  if (calls) call pr_bump2(i, j, K_SNOWBRK_CALL)
  if (.not. pr_cold_on(itimestep, i, k, j)) return
  call pr_head('snowbrk', itimestep, i, k, j)
  call fsbm_put(pr_stream, 'guard_sum', real(sum(ff3r(kr_snow_min:PR_NKR)), 8))
  call fsbm_put(pr_stream, 'guard', int(merge(1, 0, guard), 4))
  call fsbm_put(pr_stream, 'calls_breakup_snow', int(merge(1, 0, calls), 4))
  call fsbm_put(pr_stream, 'kr_snow_min', int(kr_snow_min, 4))
  call fsbm_put(pr_stream, 'kr_snow_max', int(kr_snow_max, 4))
  call fsbm_end(pr_stream)
end subroutine fsbm_probe_snowbrk

! Anchor: before the exit conversion (:5276). The spectra are still in the
! scheme's f-form here; the exit then zeroes the snow bins of every level
! whose QS (bins 19-33 only) is below 1e-13 (:5291, finding D11), counted
! where any of the small bins 1-18 holds snow.
subroutine fsbm_probe_exit(itimestep, chem, n_chem, qi, qs, qg, qni, qns, qng, snownc, &
                           snowncv, graupelnc, graupelncv, ims, ime, kms, kme, jms, jme, &
                           its, ite, kts, kte, jts, jte)
  use fsbm_io
  use fsbm_probe_state
  implicit none
  integer, intent(in) :: itimestep, n_chem, ims, ime, kms, kme, jms, jme
  integer, intent(in) :: its, ite, kts, kte, jts, jte
  real, intent(in) :: chem(ims:ime, kms:kme, jms:jme, n_chem)
  real, intent(in), dimension(ims:ime, kms:kme, jms:jme) :: qi, qs, qg, qni, qns, qng
  real, intent(in), dimension(ims:ime, jms:jme) :: snownc, snowncv, graupelnc, graupelncv
  integer :: i, j, k, n
  integer(4) :: d11(kts:kte)
  real :: ff5(PR_NKR, 64), ff6(PR_NKR, 64)
  do j = jts, jte
    do i = its, ite
      d11 = 0
      do k = kts, kte
        if (qs(i, k, j) < 1.e-13 .and. any(chem(i, k, j, 35:52) > 0.0)) then
          d11(k) = 1
          call pr_bump2(i, j, K_D11)
        end if
        if (any(chem(i, k, j, 35:67) > 0.0)) call pr_bump2(i, j, K_EXIT_SNOW)
        if (any(chem(i, k, j, 68:100) > 0.0)) call pr_bump2(i, j, K_EXIT_DENSE)
      end do
      if (.not. (pr_cni > 0 .and. pr_step_on(itimestep) .and. pr_col_on(i, j))) cycle
      do n = 1, pr_nlevs
        ff5(:, n) = chem(i, pr_levs(n), j, 35:67)
        ff6(:, n) = chem(i, pr_levs(n), j, 68:100)
      end do
      call pr_head('exit', itimestep, i, 0, j)
      call fsbm_put(pr_stream, 'qi', qi(i, kts:kte, j))
      call fsbm_put(pr_stream, 'qs', qs(i, kts:kte, j))
      call fsbm_put(pr_stream, 'qg', qg(i, kts:kte, j))
      call fsbm_put(pr_stream, 'qni', qni(i, kts:kte, j))
      call fsbm_put(pr_stream, 'qns', qns(i, kts:kte, j))
      call fsbm_put(pr_stream, 'qng', qng(i, kts:kte, j))
      call fsbm_put(pr_stream, 'snownc', snownc(i, j))
      call fsbm_put(pr_stream, 'snowncv', snowncv(i, j))
      call fsbm_put(pr_stream, 'graupelnc', graupelnc(i, j))
      call fsbm_put(pr_stream, 'graupelncv', graupelncv(i, j))
      call fsbm_put(pr_stream, 'd11_level', d11)
      call fsbm_put(pr_stream, 'ff5_probe_levels', ff5(:, 1:pr_nlevs))
      call fsbm_put(pr_stream, 'ff6_probe_levels', ff6(:, 1:pr_nlevs))
      call fsbm_end(pr_stream)
    end do
  end do
end subroutine fsbm_probe_exit
#endif
