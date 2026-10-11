! Float32 model storage around the byte-unmodified TEMPO 9adf9ef driver.
! Input: adapter/make_adapter_columns.py. Output: the same 26 slots as the
! column oracle, with t=theta perturbation, qb=m3/kg, radii=microns and
! reflectivity/radii/state rounded to host storage. Surface outputs stay wp.
! TEMPO itself uses wp=8; do not change its source or compile precision.
program tempo_adapter_driver
  use module_mp_tempo_cfgs, only : ty_tempo_cfgs
  use module_mp_tempo_driver, only : tempo_init, tempo_run, ty_tempo_driver_diags, tempo_aerosol_surface_emissions
  implicit none

  type(ty_tempo_cfgs) :: cfgs
  character(len=1024) :: in_path, out_path, mode
  logical :: cloud_mode
  integer :: ui, uo, s
  integer(8) :: nsets, hdr(3)
  real(8) :: dt

  call get_command_argument(1, in_path)
  call get_command_argument(2, out_path)
  call get_command_argument(3, mode)
  cloud_mode = trim(mode) == 'cloud'
  call tempo_init(tempo_cfgs=cfgs, ml_for_bl_nc_flag=cloud_mode)

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
  write(*,'(A,I0,A)') 'tempo_adapter_driver: ', nsets, ' sets written'

contains

  subroutine read_field(u, a, ncol, nz)
    integer, intent(in) :: u, ncol, nz
    real(8), intent(out) :: a(ncol, nz, 1)
    real(8) :: buf(nz, ncol)
    integer :: i
    read(u) buf
    do i = 1, ncol
      a(i, :, 1) = real(buf(:, i),4)
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
    if (.not. allocated(a)) error stop 'tempo_adapter_driver: column diagnostic not allocated'
    do i = 1, ncol
      buf(:, i) = a(i, :, 1)
    enddo
    write(u) buf
  end subroutine write_diag3

  subroutine write_diag2(u, a, ncol)
    integer, intent(in) :: u, ncol
    real(8), allocatable, intent(in) :: a(:, :)
    if (.not. allocated(a)) error stop 'tempo_adapter_driver: surface diagnostic not allocated'
    write(u) a(1:ncol, 1)
  end subroutine write_diag2

  subroutine run_set(ui, uo, ncol, nz, nsteps, dt)
    integer, intent(in) :: ui, uo, ncol, nz, nsteps
    real(8), intent(in) :: dt
    type(ty_tempo_driver_diags) :: d
    real(8), dimension(ncol, nz, 1) :: t, p, w, dz, qv, qc, qr, qi, qs, qg, ni, nr, nc, &
      nwfa, nifa, ng, qb
    real(8), dimension(ncol, nz, 1) :: th, pii, saved
    real(8), dimension(ncol, nz+1, 1) :: phi
    real(8), dimension(ncol, 1) :: emission
    real(8), allocatable :: qc_bl(:,:,:), fraction_bl(:,:,:)
    real(4), dimension(ncol, nz, 1) :: pi4, zlo, zhi, increment
    integer :: it, i, k

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

    ! Float32 host inputs, float64 TEMPO unchanged. t stores theta perturbation;
    ! qb stores m3/kg. Geopotential and Exner follow the WRF host operations.
    call read_field(ui, phi, ncol, nz+1)
    read(ui) emission
    phi = real(phi,4)
    emission = real(emission,4)
    pi4 = (real(p,4) / 100000._4)**real(2._8/7._8,4)
    pii = real(pi4,8)
    zlo = real(phi(:,1:nz,:),4) / 9.81_4
    zhi = real(phi(:,2:nz+1,:),4) / 9.81_4
    dz = real(zhi-zlo,8)
    do it = 1, nsteps
      th = real(300._4 + real(t,4),8)
      saved = th
      qb = qb * 1000._8
      if (cloud_mode) then
        if (.not. allocated(qc_bl)) allocate(qc_bl(ncol,nz,1), fraction_bl(ncol,nz,1))
        do i=1,ncol
          do k=1,nz
            qc_bl(i,k,1)=real(1.e-5_4 * real(1+mod(i+k,4),4),8)
            fraction_bl(i,k,1)=real(.25_4 + .125_4*real(mod(i+2*k,5),4),8)
          enddo
        enddo
        call tempo_aerosol_surface_emissions(dt, nwfa, emission, 1, ncol, 1, 1, 1, nz, 1)
      endif
      call tempo_run(tempo_cfgs=cfgs, dt=dt, itimestep=it, th=th, pii=pii, p=p, w=w, dz=dz, &
        qv=qv, qc=qc, qr=qr, qi=qi, qs=qs, qg=qg, ni=ni, nr=nr, &
        nc=nc, nwfa=nwfa, nifa=nifa, ng=ng, qb=qb, &
        ids=1, ide=ncol, jds=1, jde=1, kds=1, kde=nz, &
        ims=1, ime=ncol, jms=1, jme=1, kms=1, kme=nz, &
        its=1, ite=ncol, jts=1, jte=1, kts=1, kte=nz, tempo_diags=d, &
        qc_bl=qc_bl, qcfrac_bl=fraction_bl)
      ! The public emissions call follows tempo_run, before host writeback.
      if (.not. cloud_mode) call tempo_aerosol_surface_emissions(dt, nwfa, emission, 1, ncol, 1, 1, 1, nz, 1)
      increment = real(th,4) - real(saved,4)
      increment = min(real(10._8*dt,4), max(-real(10._8*dt,4), increment))
      t = real(real(t,4)+increment,8)
      qb = real(qb * 1.e-3_8,4)
      qv=real(qv,4); qc=real(qc,4); qr=real(qr,4); qi=real(qi,4)
      qs=real(qs,4); qg=real(qg,4); ni=real(ni,4); nr=real(nr,4)
      nc=real(nc,4); nwfa=real(nwfa,4); nifa=real(nifa,4); ng=real(ng,4)
      d%refl10cm = real(d%refl10cm,4)
      d%re_cloud = real(d%re_cloud * 1.e6_8,4)
      d%re_ice = real(d%re_ice * 1.e6_8,4)
      d%re_snow = real(d%re_snow * 1.e6_8,4)
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

end program tempo_adapter_driver
