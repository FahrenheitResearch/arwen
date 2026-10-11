! TEMPO column oracle: the referee side.
! Built against unmodified TEMPO 9adf9ef (tempo_v3.0.3, the RRFS v2 pin) with
! TEMPO's own Makefile.gfortran flags (-O0 -fcheck=all -fdefault-real-8
! -fdefault-double-8).
!
!   tempo_oracle_driver columns_in.bin ref_out.bin
!
! Runs tempo_init with TEMPO's default configuration (aerosol-aware,
! hail-aware, RRFS v2 flags), then every column set of columns_in.bin (see
! make_columns.py) through tempo_run, all columns of a set in one call per
! step, exactly as a host model calls it.  The SAVE flag first_call_main is
! therefore true for the first column of the first call only.
!
! Output (little-endian stream), per set and step:
!   14 updated fields, each float64[ncol][nz], in the order
!      t qv qc qr qi qs qg ni nr nc nwfa nifa ng qb
!   8 surface diagnostics, each float64[ncol], in the order
!      rain_precip ice_liquid_equiv_precip snow_liquid_equiv_precip
!      graupel_liquid_equiv_precip frz_rain_precip frozen_fraction
!      max_hail_diameter_sfc max_hail_diameter_column
!   4 column diagnostics, each float64[ncol][nz]:
!      refl10cm re_cloud re_ice re_snow
program tempo_refute_driver
  use module_mp_tempo_cfgs, only : ty_tempo_cfgs
  use module_mp_tempo_driver, only : tempo_init, tempo_run, ty_tempo_driver_diags
  use, intrinsic :: ieee_arithmetic, only : ieee_value, ieee_quiet_nan
  implicit none

  type(ty_tempo_cfgs) :: cfgs
  character(len=1024) :: in_path, out_path
  integer :: ui, uo, s
  integer(8) :: nsets, hdr(3)
  real(8) :: dt
  character(len=16) :: envv
  integer :: st

  call get_command_argument(1, in_path)
  call get_command_argument(2, out_path)
  call tempo_init(tempo_cfgs=cfgs)
  ! output-only flags are not tempo_init arguments: set them after init
  ! (tempo_init's tempo_cfgs is intent(out)).  The engine's adapter runs
  ! refl10cm only on reflectivity-output steps.
  call get_environment_variable('TEMPO_REFL10CM', envv, status=st)
  if (st == 0 .and. trim(envv) == '0') cfgs%refl10cm_flag = .false.
  call get_environment_variable('TEMPO_MAXHAIL', envv, status=st)
  if (st == 0 .and. trim(envv) == '0') cfgs%max_hail_diameter_flag = .false.
  write(*,'(A,L1,A,L1)') 'tempo_refute_driver: refl10cm_flag=', cfgs%refl10cm_flag, &
    ' max_hail_diameter_flag=', cfgs%max_hail_diameter_flag

  open(newunit=ui, file=trim(in_path), form='unformatted', access='stream', status='old')
  open(newunit=uo, file=trim(out_path), form='unformatted', access='stream', status='replace')
  read(ui) nsets
  do s = 1, int(nsets)
    read(ui) hdr
    read(ui) dt
    call run_set(ui, uo, int(hdr(1)), int(hdr(2)), int(hdr(3)), dt)
  enddo
  close(ui)
  close(uo)
  write(*,'(A,I0,A)') 'tempo_oracle_driver: ', nsets, ' sets written'

contains

  subroutine read_field(u, a, ncol, nz)
    integer, intent(in) :: u, ncol, nz
    real(8), intent(out) :: a(ncol, nz, 1)
    real(8) :: buf(nz, ncol)
    integer :: i
    read(u) buf
    do i = 1, ncol
      a(i, :, 1) = buf(:, i)
    enddo
  end subroutine read_field

  subroutine write_field(u, a, ncol, nz)
    integer, intent(in) :: u, ncol, nz
    real(8), intent(in) :: a(ncol, nz, 1)
    real(8) :: buf(nz, ncol)
    integer :: i
    do i = 1, ncol
      buf(:, i) = a(i, :, 1)
    enddo
    write(u) buf
  end subroutine write_field

  subroutine write_diag3(u, a, ncol, nz)
    integer, intent(in) :: u, ncol, nz
    real(8), allocatable, intent(in) :: a(:, :, :)
    real(8) :: buf(nz, ncol)
    integer :: i
    if (.not. allocated(a)) then
      buf = ieee_value(1.0_8, ieee_quiet_nan)   ! diagnostic switched off: placeholder
      write(u) buf
      return
    endif
    do i = 1, ncol
      buf(:, i) = a(i, :, 1)
    enddo
    write(u) buf
  end subroutine write_diag3

  subroutine write_diag2(u, a, ncol)
    integer, intent(in) :: u, ncol
    real(8), allocatable, intent(in) :: a(:, :)
    real(8) :: nanbuf(ncol)
    if (.not. allocated(a)) then
      nanbuf = ieee_value(1.0_8, ieee_quiet_nan)
      write(u) nanbuf
      return
    endif
    write(u) a(1:ncol, 1)
  end subroutine write_diag2

  subroutine run_set(ui, uo, ncol, nz, nsteps, dt)
    integer, intent(in) :: ui, uo, ncol, nz, nsteps
    real(8), intent(in) :: dt
    type(ty_tempo_driver_diags) :: d
    real(8), dimension(ncol, nz, 1) :: t, p, w, dz, qv, qc, qr, qi, qs, qg, ni, nr, nc, &
      nwfa, nifa, ng, qb
    integer :: it

    call read_field(ui, t, ncol, nz)
    call read_field(ui, p, ncol, nz)
    call read_field(ui, w, ncol, nz)
    call read_field(ui, dz, ncol, nz)
    call read_field(ui, qv, ncol, nz)
    call read_field(ui, qc, ncol, nz)
    call read_field(ui, qr, ncol, nz)
    call read_field(ui, qi, ncol, nz)
    call read_field(ui, qs, ncol, nz)
    call read_field(ui, qg, ncol, nz)
    call read_field(ui, ni, ncol, nz)
    call read_field(ui, nr, ncol, nz)
    call read_field(ui, nc, ncol, nz)
    call read_field(ui, nwfa, ncol, nz)
    call read_field(ui, nifa, ncol, nz)
    call read_field(ui, ng, ncol, nz)
    call read_field(ui, qb, ncol, nz)

    do it = 1, nsteps
      call tempo_run(tempo_cfgs=cfgs, dt=dt, itimestep=it, t=t, p=p, w=w, dz=dz, &
        qv=qv, qc=qc, qr=qr, qi=qi, qs=qs, qg=qg, ni=ni, nr=nr, &
        nc=nc, nwfa=nwfa, nifa=nifa, ng=ng, qb=qb, &
        ids=1, ide=ncol, jds=1, jde=1, kds=1, kde=nz, &
        ims=1, ime=ncol, jms=1, jme=1, kms=1, kme=nz, &
        its=1, ite=ncol, jts=1, jte=1, kts=1, kte=nz, tempo_diags=d)
      call write_field(uo, t, ncol, nz)
      call write_field(uo, qv, ncol, nz)
      call write_field(uo, qc, ncol, nz)
      call write_field(uo, qr, ncol, nz)
      call write_field(uo, qi, ncol, nz)
      call write_field(uo, qs, ncol, nz)
      call write_field(uo, qg, ncol, nz)
      call write_field(uo, ni, ncol, nz)
      call write_field(uo, nr, ncol, nz)
      call write_field(uo, nc, ncol, nz)
      call write_field(uo, nwfa, ncol, nz)
      call write_field(uo, nifa, ncol, nz)
      call write_field(uo, ng, ncol, nz)
      call write_field(uo, qb, ncol, nz)
      call write_diag2(uo, d%rain_precip, ncol)
      call write_diag2(uo, d%ice_liquid_equiv_precip, ncol)
      call write_diag2(uo, d%snow_liquid_equiv_precip, ncol)
      call write_diag2(uo, d%graupel_liquid_equiv_precip, ncol)
      call write_diag2(uo, d%frz_rain_precip, ncol)
      call write_diag2(uo, d%frozen_fraction, ncol)
      call write_diag2(uo, d%max_hail_diameter_sfc, ncol)
      call write_diag2(uo, d%max_hail_diameter_column, ncol)
      call write_diag3(uo, d%refl10cm, ncol, nz)
      call write_diag3(uo, d%re_cloud, ncol, nz)
      call write_diag3(uo, d%re_ice, ncol, nz)
      call write_diag3(uo, d%re_snow, ncol, nz)
    enddo
  end subroutine run_set

end program tempo_refute_driver
