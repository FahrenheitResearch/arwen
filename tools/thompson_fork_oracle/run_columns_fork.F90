! Batch column driver for the UNMODIFIED HRRR v4.1.21 fork Thompson
! (NOAA-EMC/HRRR sorc/hrrr_wrfarw.fd/WRFV3.9/phys/module_mp_thompson.F),
! aerosol aware (mp_physics=28): one thompson_init, one mp_gt_driver call
! over N saved model columns, raw float32 in and out.
!
! The file formats are those of tools/thompson_real_column_parity/
! run_columns_aero.F90 (the WRF v4.6.1 driver), so one saved column file
! runs through both generations and the outputs compare field by field.
! The fork's argument set differs, by the fork's own interface
! (module_mp_thompson.F:1078-1096, :399-402):
!   * no nifa2d, nbca, nbca2d, aer_init_opt, wif_input_opt or ke_diag: the
!     fork has none of them.  nifa2d is read from the input and ignored
!     (the fork has no ice-friendly surface emission, audit row T27).
!   * frain is passed and written as an extra trailing field.
!   * rand_perturb_on = 0 and kme_stoch = 1: the operational namelist has
!     no stochastic block, so every perturbation is zero (audit row T26).
!   * thompson_init gets is_start = .false., which keeps the input nwfa2d.
!     At a cycled start the fork's wrf.exe recomputes nwfa2d from the
!     lowest-level nwfa (row T17, module_mp_thompson.F:587-603); that is a
!     start-time rule, not a per-step one, and the column file carries
!     whatever nwfa2d the saved state holds.
!   * itimestep is the third argument (default 2).  On itimestep 1 the
!     fork switches the melting-snow reflectivity term off (row T28,
!     :5374-5380); a saved mid-run column is not step 1.
!
! LAYOUT.  As run_columns_aero.F90: arrays (ncol+1, nz, 2), ids=its=1,
! ide=ite=ncol+1, jds=jts=1, jde=jte=2; the pad column and row carry copies
! of column 1 and are never read back.
!
! INPUT  (stream, little endian): int32 ncol, int32 nz, float32 dt, then
!        17 fields of ncol*nz float32, column index fastest:
!        th pii p w dz hgt qv qc qr qi qs qg ni nr nc nwfa nifa
!        then nwfa2d and nifa2d of ncol float32 each.
! OUTPUT (stream): 16 fields of ncol*nz float32
!        qv qc qr qi qs qg ni nr nc nwfa nifa th refl re_cloud re_ice re_snow
!        then 8 fields of ncol float32:
!        rainnc rainncv snownc snowncv graupelnc graupelncv sr frain
!
! CCN_ACTIVATE.BIN is big-endian and table_ccnAct opens it on the lowest
! free unit in 20..99; run with GFORTRAN_CONVERT_UNIT='big_endian:20' and
! this program aborts unless that unit really is 20.

program run_thompson_columns_fork
  use module_mp_thompson, only: thompson_init, mp_gt_driver
  implicit none

  integer :: ncol, nz, nxp, i, k, u, free_unit, nfield, itimestep
  logical :: unit_opened
  real :: dt
  real, allocatable, dimension(:,:,:) :: th, pii, p, w, dz, hgt
  real, allocatable, dimension(:,:,:) :: qv, qc, qr, qi, qs, qg
  real, allocatable, dimension(:,:,:) :: ni, nr, nc, nwfa, nifa
  real, allocatable, dimension(:,:,:) :: refl, re_cloud, re_ice, re_snow
  real, allocatable, dimension(:,:,:) :: rand_pert
  real, allocatable, dimension(:,:) :: nwfa2d, nifa2d
  real, allocatable, dimension(:,:) :: rainnc, rainncv, snownc, snowncv
  real, allocatable, dimension(:,:) :: graupelnc, graupelncv, sr, frain
  real, allocatable, dimension(:,:) :: buf
  real, allocatable, dimension(:) :: buf2
  character(len=1024) :: in_path, out_path, step_arg

  call get_command_argument(1, in_path)
  call get_command_argument(2, out_path)
  call get_command_argument(3, step_arg)
  if (len_trim(in_path) == 0 .or. len_trim(out_path) == 0) then
     error stop 'usage: run_columns_fork INPUT.bin OUTPUT.bin [ITIMESTEP]'
  endif
  itimestep = 2
  if (len_trim(step_arg) > 0) read(step_arg, *) itimestep

  open(newunit=u, file=trim(in_path), access='stream', form='unformatted', &
       status='old', action='read')
  read(u) ncol, nz, dt
  nxp = ncol + 1
  allocate(buf(ncol, nz), buf2(ncol))

  allocate(th(nxp,nz,2), pii(nxp,nz,2), p(nxp,nz,2), w(nxp,nz,2))
  allocate(dz(nxp,nz,2), hgt(nxp,nz,2))
  allocate(qv(nxp,nz,2), qc(nxp,nz,2), qr(nxp,nz,2), qi(nxp,nz,2))
  allocate(qs(nxp,nz,2), qg(nxp,nz,2), ni(nxp,nz,2), nr(nxp,nz,2))
  allocate(nc(nxp,nz,2), nwfa(nxp,nz,2), nifa(nxp,nz,2))
  allocate(refl(nxp,nz,2), re_cloud(nxp,nz,2), re_ice(nxp,nz,2))
  allocate(re_snow(nxp,nz,2), rand_pert(nxp,1,2))
  allocate(nwfa2d(nxp,2), nifa2d(nxp,2))
  allocate(rainnc(nxp,2), rainncv(nxp,2), snownc(nxp,2), snowncv(nxp,2))
  allocate(graupelnc(nxp,2), graupelncv(nxp,2), sr(nxp,2), frain(nxp,2))

  call read3(th); call read3(pii); call read3(p); call read3(w)
  call read3(dz); call read3(hgt)
  call read3(qv); call read3(qc); call read3(qr); call read3(qi)
  call read3(qs); call read3(qg); call read3(ni); call read3(nr)
  call read3(nc); call read3(nwfa); call read3(nifa)
  call read2(nwfa2d); call read2(nifa2d)
  close(u)

  rand_pert = 0.0
  refl = -35.0
  re_cloud = 2.51e-6
  re_ice = 10.01e-6
  re_snow = 25.e-6
  rainnc = 0.0; rainncv = 0.0; snownc = 0.0; snowncv = 0.0
  graupelnc = 0.0; graupelncv = 0.0; sr = 0.0; frain = 0.0

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
     error stop 3
  endif

  call thompson_init(                                                 &
       hgt=hgt, nwfa2d=nwfa2d, nwfa=nwfa, nifa=nifa,                  &
       dx=3000.0, dy=3000.0, is_start=.false.,                        &
       ids=1, ide=nxp, jds=1, jde=2, kds=1, kde=nz,                   &
       ims=1, ime=nxp, jms=1, jme=2, kms=1, kme=nz,                   &
       its=1, ite=nxp, jts=1, jte=2, kts=1, kte=nz)

  call mp_gt_driver(                                                  &
       qv=qv, qc=qc, qr=qr, qi=qi, qs=qs, qg=qg, ni=ni, nr=nr,        &
       nc=nc, nwfa=nwfa, nifa=nifa, nwfa2d=nwfa2d,                    &
       th=th, pii=pii, p=p, w=w, dz=dz,                               &
       dt_in=dt, itimestep=itimestep,                                 &
       RAINNC=rainnc, RAINNCV=rainncv,                                &
       SNOWNC=snownc, SNOWNCV=snowncv,                                &
       GRAUPELNC=graupelnc, GRAUPELNCV=graupelncv, SR=sr,             &
       frain=frain,                                                   &
       refl_10cm=refl, diagflag=.true., do_radar_ref=1,               &
       re_cloud=re_cloud, re_ice=re_ice, re_snow=re_snow,             &
       has_reqc=1, has_reqi=1, has_reqs=1,                            &
       rand_perturb_on=0, kme_stoch=1, rand_pert=rand_pert,           &
       ids=1, ide=nxp, jds=1, jde=2, kds=1, kde=nz,                   &
       ims=1, ime=nxp, jms=1, jme=2, kms=1, kme=nz,                   &
       its=1, ite=nxp, jts=1, jte=2, kts=1, kte=nz)

  open(newunit=u, file=trim(out_path), access='stream', form='unformatted', &
       status='replace', action='write')
  nfield = 0
  call write3(qv); call write3(qc); call write3(qr); call write3(qi)
  call write3(qs); call write3(qg); call write3(ni); call write3(nr)
  call write3(nc); call write3(nwfa); call write3(nifa); call write3(th)
  call write3(refl); call write3(re_cloud); call write3(re_ice)
  call write3(re_snow)
  call write2(rainnc); call write2(rainncv); call write2(snownc)
  call write2(snowncv); call write2(graupelnc); call write2(graupelncv)
  call write2(sr); call write2(frain)
  close(u)
  print '(A,I0,A,I0,A,I0)', 'columns ', ncol, ' levels ', nz, ' fields ', nfield

contains

  subroutine read3(a)
    real, intent(inout) :: a(:,:,:)
    read(u) buf
    a(1:ncol, :, 1) = buf
    do k = 1, nz
       a(nxp, k, 1) = buf(1, k)
       a(:, k, 2) = a(:, k, 1)
    enddo
  end subroutine read3

  subroutine read2(a)
    real, intent(inout) :: a(:,:)
    read(u) buf2
    a(1:ncol, 1) = buf2
    a(nxp, 1) = buf2(1)
    a(:, 2) = a(:, 1)
  end subroutine read2

  subroutine write3(a)
    real, intent(in) :: a(:,:,:)
    buf = a(1:ncol, :, 1)
    write(u) buf
    nfield = nfield + 1
  end subroutine write3

  subroutine write2(a)
    real, intent(in) :: a(:,:)
    buf2 = a(1:ncol, 1)
    write(u) buf2
    nfield = nfield + 1
  end subroutine write2

end program run_thompson_columns_fork
