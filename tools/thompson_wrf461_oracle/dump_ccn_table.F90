! Dump WRF's in-memory tnccn_act after an aerosol-aware thompson_init so
! ArWen's independent parse of CCN_ACTIVATE.BIN can be diffed against the
! array WRF actually uses.
!
! The dump is deliberately a plain unformatted sequential record of the
! REAL(4) array in Fortran storage order, i.e. the same payload
! CCN_ACTIVATE.BIN carries, but written NATIVE-endian.  Comparing the
! widened values (not the bytes) is the point: a C/F order flip does not
! change the file SHA-256 and does not raise on this all-distinct shape.
!
! It also asserts the array is not still WRF's all-ones prefill
! (module_mp_thompson.F:993-1002).  table_ccnAct is called inside the
! one-time `if (micro_init)` block while is_aerosol_aware is reset on every
! thompson_init call, so a second init in the same process silently leaves
! tnccn_act at 1.0 and activ_ncloud then returns 100% activation with no
! error anywhere.  That is why build_aero.sh runs one process per scenario.

program dump_ccn_activation_table
  use module_mp_thompson, only: thompson_init, tnccn_act
  implicit none

  integer, parameter :: nx = 2, ny = 2, nz = 2
  real :: hgt(nx,nz,ny)
  real :: nwfa(nx,nz,ny), nifa(nx,nz,ny)
  real :: nwfa2d(nx,ny)
  integer :: i, j, k, l, m, ones, total, free_unit
  logical :: unit_opened
  real :: vmin, vmax
  character(len=512) :: output_path

  call get_command_argument(1, output_path)
  if (len_trim(output_path) == 0) then
     error stop 'usage: dump_ccn_table OUTPUT_FILE'
  endif

  ! build_aero.sh scopes GFORTRAN_CONVERT_UNIT to unit 20 only, on the
  ! premise that table_ccnAct's 20..99 search (5123-5133) lands there in a
  ! fresh process.  Assert it rather than assume it.
  free_unit = -1
  do i = 20, 99
     inquire(unit=i, opened=unit_opened)
     if (.not. unit_opened) then
        free_unit = i
        exit
     endif
  enddo
  if (free_unit /= 20) then
     print '(A,I0)', 'FATAL: lowest free Fortran unit is not 20 but ', free_unit
     error stop 'CCN_ACTIVATE.BIN endian conversion assumption violated'
  endif

  hgt = 0.0
  hgt(:,1,:) = 100.0
  hgt(:,2,:) = 200.0
  ! Nonzero aerosol so thompson_init does not spend time on the profile
  ! fill; is_aerosol_aware is set purely by argument presence.
  nwfa = 3.0e8
  nifa = 1.0e6
  nwfa2d = 0.0

  call thompson_init(                                                &
       hgt=hgt,                                                      &
       nwfa2d=nwfa2d, nwfa=nwfa, nifa=nifa,                          &
       wif_input_opt=0,                                              &
       ids=1, ide=3, jds=1, jde=3, kds=1, kde=2,                    &
       ims=1, ime=nx, jms=1, jme=ny, kms=1, kme=nz,                 &
       its=1, ite=nx, jts=1, jte=ny, kts=1, kte=nz)

  if (.not. allocated(tnccn_act)) then
     error stop 'tnccn_act was never allocated'
  endif

  ones = 0
  total = 0
  vmin = huge(1.0)
  vmax = -huge(1.0)
  do m = 1, size(tnccn_act, 5)
     do l = 1, size(tnccn_act, 4)
        do k = 1, size(tnccn_act, 3)
           do j = 1, size(tnccn_act, 2)
              do i = 1, size(tnccn_act, 1)
                 total = total + 1
                 if (tnccn_act(i,j,k,l,m) == 1.0) ones = ones + 1
                 vmin = min(vmin, tnccn_act(i,j,k,l,m))
                 vmax = max(vmax, tnccn_act(i,j,k,l,m))
              enddo
           enddo
        enddo
     enddo
  enddo

  print '(A,5(1X,I0))', 'TNCCN_ACT_SHAPE', size(tnccn_act,1),          &
       size(tnccn_act,2), size(tnccn_act,3), size(tnccn_act,4),        &
       size(tnccn_act,5)
  print '(A,1X,I0,1X,I0)', 'TNCCN_ACT_ONES_OF_TOTAL', ones, total
  print '(A,2(1X,ES24.16E3))', 'TNCCN_ACT_RANGE', vmin, vmax
  ! Reference probe used by the ArWen reader test: Fortran 1-based
  ! (1,1,4,3,2), i.e. zero-based [0,0,3,2,1].
  print '(A,1X,ES24.16E3)', 'TNCCN_ACT_1_1_4_3_2', tnccn_act(1,1,4,3,2)
  do j = 1, size(tnccn_act, 2)
     print '(A,1X,I0,1X,ES24.16E3)', 'TNCCN_ACT_SLAB_W', j,            &
          tnccn_act(1,j,4,3,2)
  enddo

  if (ones == total) then
     error stop 'tnccn_act is still the all-ones prefill: CCN_ACTIVATE.BIN was not read'
  endif
  if (vmin <= 0.0 .or. vmax > 1.0) then
     error stop 'tnccn_act outside (0,1]: endianness or record layout is wrong'
  endif

  open(71, file=trim(output_path), form='unformatted',                 &
       status='replace', action='write')
  write(71) tnccn_act
  close(71)

  print '(A)', 'THOMPSON_CCN_TABLE_DUMP_COMPLETE'
end program dump_ccn_activation_table
