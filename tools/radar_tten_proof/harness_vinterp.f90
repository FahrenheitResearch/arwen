! Oracle harness for NOAA's vinterp_radar_ref (HRRR v4.1.21,
! sorc/hrrr_ref2tten.fd/vinterp_radar_ref.f90), called unchanged.
!
! Usage: oracle_vinterp <dir> <nlon> <nlat> <nsig> <nmsclvl>
!   reads  <dir>/mosaic.bin (nlon, nlat, nmsclvl), h.bin (nlon, nlat, nsig),
!          zh.bin (nlon, nlat)
!   writes <dir>/vinterp.bin (nlon, nlat, nsig)
program oracle_vinterp
  use kinds, only: r_single, i_kind
  implicit none
  integer(i_kind) :: nlon, nlat, nsig, nmsc, u
  character(len=1024) :: dir, arg
  real(r_single), allocatable :: mosaic(:,:,:), h_bk(:,:,:), zh(:,:)
  real(r_single), allocatable :: ref_mos_3d(:,:,:)
  call get_command_argument(1, dir)
  call get_command_argument(2, arg); read(arg, *) nlon
  call get_command_argument(3, arg); read(arg, *) nlat
  call get_command_argument(4, arg); read(arg, *) nsig
  call get_command_argument(5, arg); read(arg, *) nmsc
  allocate(mosaic(nlon,nlat,nmsc), h_bk(nlon,nlat,nsig), zh(nlon,nlat))
  allocate(ref_mos_3d(nlon,nlat,nsig))
  open(newunit=u, file=trim(dir)//'/mosaic.bin', access='stream', &
       form='unformatted', status='old', action='read')
  read(u) mosaic
  close(u)
  open(newunit=u, file=trim(dir)//'/h.bin', access='stream', &
       form='unformatted', status='old', action='read')
  read(u) h_bk
  close(u)
  open(newunit=u, file=trim(dir)//'/zh.bin', access='stream', &
       form='unformatted', status='old', action='read')
  read(u) zh
  close(u)
  call vinterp_radar_ref(nlon,nlat,nsig,nmsc,ref_mos_3d,mosaic,h_bk,zh)
  open(newunit=u, file=trim(dir)//'/vinterp.bin', access='stream', &
       form='unformatted', status='replace', action='write')
  write(u) ref_mos_3d
  close(u)
end program oracle_vinterp
