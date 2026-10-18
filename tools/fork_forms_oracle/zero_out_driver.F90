! Column driver for the operational RAP/HRRR branch's microphysics zero-out.
!
! zero_out_stubs.F90 stands in for the three WRF framework modules that
! phys/module_microphysics_zero_out.F USEs.  They supply only what the
! routines read: PARAM_FIRST_SCALAR and P_QV (module_state_description;
! both are 2 in every WRF build: index 1 is the dummy slot, qv the first
! moist species), the two config_flags members, and wrf_debug.  The
! zero-out module itself is compiled unchanged.
!
! The program reproduces the branch's solve_em.F:4070-4141: zero_outb on
! every array over the whole domain, then zero_outa over the interior that
! leaves out the spec_zone ring (sz = spec_zone under specified or nested
! boundaries, else 0; :3819-3823).  The scalar array is passed exactly as
! the moist one is: the routines test the array INDEX against P_QV.
!
! Stream input (little-endian, as written by zero_out_oracle.py):
!   int32  nx ny nz n_moist n_scalar mp_zero_out sz
!   real32 thresh
!   real32 moist (nx, nz, ny, n_moist)    Fortran order, index 1 is a dummy
!   real32 scalar(nx, nz, ny, n_scalar)
! Output: the two arrays after the calls, same layout.

program zero_out_driver
  use module_configure
  use module_microphysics_zero_out, only: microphysics_zero_outa, &
                                          microphysics_zero_outb
  implicit none
  integer :: hdr(7), nx, ny, nz, n_moist, n_scalar, sz
  integer :: ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme
  integer :: its, ite, jts, jte
  real :: thresh
  real, allocatable :: moist(:,:,:,:), scalar(:,:,:,:)
  type(grid_config_rec_type) :: config_flags
  character(len=1024) :: inpath, outpath

  call get_command_argument(1, inpath)
  call get_command_argument(2, outpath)
  open(unit=11, file=trim(inpath), access='stream', form='unformatted', status='old')
  read(11) hdr
  read(11) thresh
  nx = hdr(1); ny = hdr(2); nz = hdr(3); n_moist = hdr(4); n_scalar = hdr(5)
  config_flags%mp_zero_out = hdr(6)
  config_flags%mp_zero_out_thresh = thresh
  sz = hdr(7)
  ! Mass points 1..nx, 1..ny, 1..nz: ide/jde/kde are the staggered ends.
  ids = 1; ide = nx + 1; jds = 1; jde = ny + 1; kds = 1; kde = nz + 1
  ims = 1; ime = nx; jms = 1; jme = ny; kms = 1; kme = nz
  allocate(moist(ims:ime, kms:kme, jms:jme, n_moist))
  allocate(scalar(ims:ime, kms:kme, jms:jme, n_scalar))
  read(11) moist
  read(11) scalar
  close(11)

  ! solve_em.F:4072-4075, one tile covering the domain.
  its = ids; ite = ide - 1; jts = jds; jte = jde - 1
  call microphysics_zero_outb(moist, n_moist, config_flags, &
       ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
       its, ite, jts, jte, 1, nz)
  call microphysics_zero_outb(scalar, n_scalar, config_flags, &
       ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
       its, ite, jts, jte, 1, nz)
  ! :4103-4111 (periodic_x is off on every case here).
  its = ids + sz; ite = ide - 1 - sz; jts = jds + sz; jte = jde - 1 - sz
  call microphysics_zero_outa(moist, n_moist, config_flags, &
       ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
       its, ite, jts, jte, 1, nz)
  call microphysics_zero_outa(scalar, n_scalar, config_flags, &
       ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
       its, ite, jts, jte, 1, nz)

  open(unit=12, file=trim(outpath), access='stream', form='unformatted', status='replace')
  write(12) moist
  write(12) scalar
  close(12)
end program zero_out_driver
