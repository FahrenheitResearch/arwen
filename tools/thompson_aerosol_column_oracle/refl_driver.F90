! WRF v4.6.1's calc_refl10cm on N saved columns, for the mp=28 reflectivity
! check (refl_check.py).  calc_refl10cm is a module procedure of
! module_mp_thompson.F with no PUBLIC statement of its own, so this driver
! links against a copy of the pristine module whose PRIVATE attributes are
! dropped (refl_check.py builds it); nothing in the arithmetic changes.
!
! INPUT  (stream): int32 ncol, int32 nz, then 9 fields of ncol*nz float32,
!        column index fastest: qv qc qr nr qs qg ng t p
! OUTPUT (stream): ncol*nz float32, MAX(-35., dBZ) as mp_gt_driver stores it.
! Run in a directory holding the four Thompson tables, with
! GFORTRAN_CONVERT_UNIT='big_endian:20' (CCN_ACTIVATE.BIN).

program refl_driver
  use module_mp_thompson
  implicit none
  integer, parameter :: nzi = 3, nxp = 2
  real :: hgt(nxp,nzi,2), nwfa(nxp,nzi,2), nifa(nxp,nzi,2), nbca(nxp,nzi,2)
  real :: nwfa2d(nxp,2), nbca2d(nxp,2)
  integer :: ncol, nz, u, c, k
  real, allocatable :: f(:,:,:), dbz(:), qb(:), outv(:,:)
  character(len=1024) :: in_path, out_path

  call get_command_argument(1, in_path)
  call get_command_argument(2, out_path)
  hgt = 0.; nwfa = 1.e9; nifa = 1.e6; nbca = 0.; nwfa2d = 0.; nbca2d = 0.
  do k = 1, nzi
     hgt(:,k,:) = 100.*k
  enddo
  call thompson_init(hgt=hgt, nwfa2d=nwfa2d, nbca2d=nbca2d, nwfa=nwfa,     &
       nifa=nifa, nbca=nbca, wif_input_opt=0,                              &
       ids=1, ide=nxp, jds=1, jde=2, kds=1, kde=nzi,                       &
       ims=1, ime=nxp, jms=1, jme=2, kms=1, kme=nzi,                       &
       its=1, ite=nxp, jts=1, jte=2, kts=1, kte=nzi)

  open(newunit=u, file=trim(in_path), access='stream', form='unformatted', &
       status='old', action='read')
  read(u) ncol, nz
  allocate(f(ncol, nz, 9), dbz(nz), qb(nz), outv(ncol, nz))
  do k = 1, 9
     read(u) f(:, :, k)
  enddo
  close(u)
  qb = 0.
  do c = 1, ncol
     call calc_refl10cm(f(c,:,1), f(c,:,2), f(c,:,3), f(c,:,4), f(c,:,5),  &
          f(c,:,6), f(c,:,7), qb, f(c,:,8), f(c,:,9), dbz, 1, nz, c, 1, nz)
     do k = 1, nz
        outv(c, k) = MAX(-35., dbz(k))
     enddo
  enddo
  open(newunit=u, file=trim(out_path), access='stream', form='unformatted', &
       status='replace', action='write')
  write(u) outv
  close(u)
end program refl_driver
