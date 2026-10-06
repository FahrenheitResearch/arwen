! Oracle harness for NOAA's SMOOTH alone (HRRR v4.1.21,
! sorc/hrrr_ref2tten.fd/smooth.f90), so a field with non-zero edges is
! graded too: inside radar_ref2tten the edges are always zero.
!
! Usage: oracle_smooth <dir> <ix> <iy> <passes>
!   reads <dir>/smooth_in.bin (float64, ix*iy), writes <dir>/smooth_out.bin
program oracle_smooth
  use kinds, only: r_kind, i_kind
  implicit none
  integer(i_kind) :: ix, iy, passes, n, u
  character(len=1024) :: dir, arg
  real(r_kind), allocatable :: field(:,:), hold(:,:)
  call get_command_argument(1, dir)
  call get_command_argument(2, arg); read(arg, *) ix
  call get_command_argument(3, arg); read(arg, *) iy
  call get_command_argument(4, arg); read(arg, *) passes
  allocate(field(ix,iy), hold(ix,2))
  open(newunit=u, file=trim(dir)//'/smooth_in.bin', access='stream', &
       form='unformatted', status='old', action='read')
  read(u) field
  close(u)
  do n = 1, passes
     call smooth(field, hold, ix, iy, 0.5_r_kind)
  enddo
  open(newunit=u, file=trim(dir)//'/smooth_out.bin', access='stream', &
       form='unformatted', status='replace', action='write')
  write(u) field
  close(u)
end program oracle_smooth
