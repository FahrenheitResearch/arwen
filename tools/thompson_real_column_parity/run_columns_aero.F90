! Batch column driver for UNMODIFIED WRF v4.6.1 aerosol-aware Thompson
! (mp_physics=28): one thompson_init, one mp_gt_driver call over N saved
! model columns, raw float32 in and out.
!
! The single-column fixture harness is tools/thompson_wrf461_oracle/
! run_column_aero.F90; this program reuses its argument conventions (the
! THOMPSONAERO argument set of module_microphysics_driver.F, nbca/nbca2d
! supplied because mp_gt_driver:1337 writes nbca unguarded) and differs only
! in reading its columns from a file instead of building them.
!
! LAYOUT.  Arrays are dimensioned (ncol+1, nz, 2) with ids=its=1,
! ide=ite=ncol+1, jds=jts=1, jde=jte=2.  mp_gt_driver loops
! i = its..MIN(ite, ide-1) and j = jts..MIN(jte, jde-1), i.e. exactly the
! ncol real columns in row 1, and thompson_init's aerosol presence test
! reduces over nwfa(its:ite-1, :, jts:jte-1), which is the same ncol
! columns: real aerosol, so the synthetic profile fill cannot fire.  The pad
! column and the pad row carry copies of column 1 and are never read back.
!
! INPUT  (stream, little endian): int32 ncol, int32 nz, float32 dt, then
!        17 fields of ncol*nz float32, column index fastest:
!        th pii p w dz hgt qv qc qr qi qs qg ni nr nc nwfa nifa
!        (w is the lower-face vertical velocity, hgt the lower-face height,
!        read by thompson_init only), then nwfa2d and nifa2d of ncol
!        float32 each.
! OUTPUT (stream): 16 fields of ncol*nz float32
!        qv qc qr qi qs qg ni nr nc nwfa nifa th refl re_cloud re_ice re_snow
!        then 7 fields of ncol float32:
!        rainnc rainncv snownc snowncv graupelnc graupelncv sr
!
! CCN_ACTIVATE.BIN is big-endian and table_ccnAct opens it on the lowest
! free unit in 20..99; run with GFORTRAN_CONVERT_UNIT='big_endian:20' and
! this program aborts unless that unit really is 20.

program run_thompson_columns_aero
  use module_mp_thompson, only: thompson_init, mp_gt_driver
  implicit none

  integer :: ncol, nz, nxp, i, k, u, free_unit, nfield
  logical :: unit_opened
  real :: dt
  real, allocatable, dimension(:,:,:) :: th, pii, p, w, dz, hgt
  real, allocatable, dimension(:,:,:) :: qv, qc, qr, qi, qs, qg
  real, allocatable, dimension(:,:,:) :: ni, nr, nc, nwfa, nifa, nbca
  real, allocatable, dimension(:,:,:) :: refl, re_cloud, re_ice, re_snow
  real, allocatable, dimension(:,:) :: nwfa2d, nifa2d, nbca2d
  real, allocatable, dimension(:,:) :: rainnc, rainncv, snownc, snowncv
  real, allocatable, dimension(:,:) :: graupelnc, graupelncv, sr
  real, allocatable, dimension(:,:) :: buf
  real, allocatable, dimension(:) :: buf2
  character(len=1024) :: in_path, out_path

  call get_command_argument(1, in_path)
  call get_command_argument(2, out_path)
  if (len_trim(in_path) == 0 .or. len_trim(out_path) == 0) then
     error stop 'usage: run_columns_aero INPUT.bin OUTPUT.bin'
  endif

  open(newunit=u, file=trim(in_path), access='stream', form='unformatted', &
       status='old', action='read')
  read(u) ncol, nz, dt
  nxp = ncol + 1
  allocate(buf(ncol, nz), buf2(ncol))

  allocate(th(nxp,nz,2), pii(nxp,nz,2), p(nxp,nz,2), w(nxp,nz,2))
  allocate(dz(nxp,nz,2), hgt(nxp,nz,2))
  allocate(qv(nxp,nz,2), qc(nxp,nz,2), qr(nxp,nz,2), qi(nxp,nz,2))
  allocate(qs(nxp,nz,2), qg(nxp,nz,2), ni(nxp,nz,2), nr(nxp,nz,2))
  allocate(nc(nxp,nz,2), nwfa(nxp,nz,2), nifa(nxp,nz,2), nbca(nxp,nz,2))
  allocate(refl(nxp,nz,2), re_cloud(nxp,nz,2), re_ice(nxp,nz,2))
  allocate(re_snow(nxp,nz,2))
  allocate(nwfa2d(nxp,2), nifa2d(nxp,2), nbca2d(nxp,2))
  allocate(rainnc(nxp,2), rainncv(nxp,2), snownc(nxp,2), snowncv(nxp,2))
  allocate(graupelnc(nxp,2), graupelncv(nxp,2), sr(nxp,2))

  call read3(th); call read3(pii); call read3(p); call read3(w)
  call read3(dz); call read3(hgt)
  call read3(qv); call read3(qc); call read3(qr); call read3(qi)
  call read3(qs); call read3(qg); call read3(ni); call read3(nr)
  call read3(nc); call read3(nwfa); call read3(nifa)
  call read2(nwfa2d); call read2(nifa2d)
  close(u)

  nbca = 0.0
  nbca2d = 0.0
  refl = -35.0
  re_cloud = 2.49e-6
  re_ice = 4.99e-6
  re_snow = 9.99e-6
  rainnc = 0.0; rainncv = 0.0; snownc = 0.0; snowncv = 0.0
  graupelnc = 0.0; graupelncv = 0.0; sr = 0.0

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
       hgt=hgt,                                                       &
       nwfa2d=nwfa2d, nbca2d=nbca2d,                                  &
       nwfa=nwfa, nifa=nifa, nbca=nbca,                               &
       wif_input_opt=0,                                               &
       ids=1, ide=nxp, jds=1, jde=2, kds=1, kde=nz,                   &
       ims=1, ime=nxp, jms=1, jme=2, kms=1, kme=nz,                   &
       its=1, ite=nxp, jts=1, jte=2, kts=1, kte=nz)

  call mp_gt_driver(                                                  &
       qv=qv, qc=qc, qr=qr, qi=qi, qs=qs, qg=qg, ni=ni, nr=nr,        &
       nc=nc, nwfa=nwfa, nifa=nifa, nbca=nbca,                        &
       nwfa2d=nwfa2d, nifa2d=nifa2d, nbca2d=nbca2d,                   &
       aer_init_opt=0, wif_input_opt=0,                               &
       th=th, pii=pii, p=p, w=w, dz=dz,                               &
       dt_in=dt, itimestep=1,                                         &
       RAINNC=rainnc, RAINNCV=rainncv,                                &
       SNOWNC=snownc, SNOWNCV=snowncv,                                &
       GRAUPELNC=graupelnc, GRAUPELNCV=graupelncv, SR=sr,             &
       refl_10cm=refl, diagflag=.true., ke_diag=nz, do_radar_ref=1,   &
       re_cloud=re_cloud, re_ice=re_ice, re_snow=re_snow,             &
       has_reqc=1, has_reqi=1, has_reqs=1,                            &
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
  call write2(sr)
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

end program run_thompson_columns_aero
