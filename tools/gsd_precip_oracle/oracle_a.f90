! Harness for NOAA's hydro_mxr_thompson (NOAA-EMC/HRRR tag v4.1.21,
! sorc/hrrr_gsi.fd/libsrc/GSD/gsdcloud/hydro_mxr_thompson.f90).
!
! This file is ours.  NOAA's hydro_mxr_thompson.f90 and kinds.f90 are
! compiled unchanged beside it (build.sh fetches them at the tag and checks
! their sha256).  The harness reads raw arrays, sets the three outputs to the
! value gsdcloudanalysis.F90:613-615 gives them before the retrieval runs
! (miss_obs_real, -99999), calls the subroutine once and writes the outputs.
!
! File layout (stream, native endian).  Input: nx, ny, nz (int32); t (real32,
! K); p (real32, Pa); ref (real64, dBZ); each (nx, ny, nz) in Fortran order,
! which is the C order of a (nz, ny, nx) array.  Output: nx, ny, nz, istatus
! (int32); qr (g/kg), qnr (/kg), qs (g/kg), real32, same order.
program oracle_a
  use kinds, only: r_single, i_kind, r_kind
  implicit none
  integer(i_kind) :: nx, ny, nz, istatus, u
  real(r_single), allocatable :: t(:,:,:), p(:,:,:)
  real(r_single), allocatable :: qr(:,:,:), qnr(:,:,:), qs(:,:,:)
  real(r_kind), allocatable :: ref(:,:,:)
  real(r_kind), parameter :: miss_obs_real = -99999.0_r_kind
  character(len=1024) :: fin, fout

  call get_command_argument(1, fin)
  call get_command_argument(2, fout)
  open(newunit=u, file=trim(fin), access='stream', form='unformatted', &
       status='old', action='read')
  read(u) nx, ny, nz
  allocate(t(nx,ny,nz), p(nx,ny,nz), ref(nx,ny,nz))
  allocate(qr(nx,ny,nz), qnr(nx,ny,nz), qs(nx,ny,nz))
  read(u) t
  read(u) p
  read(u) ref
  close(u)

  qr = miss_obs_real
  qnr = miss_obs_real
  qs = miss_obs_real
  call hydro_mxr_thompson(nx, ny, nz, t, p, ref, qr, qnr, qs, istatus, 0)

  open(newunit=u, file=trim(fout), access='stream', form='unformatted', &
       status='replace', action='write')
  write(u) nx, ny, nz, istatus
  write(u) qr
  write(u) qnr
  write(u) qs
  close(u)
end program oracle_a
