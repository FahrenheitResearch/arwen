! Drives WPS v4.6.0's own geogrid smoothers (geogrid/src/smooth_module.F,
! compiled byte-unmodified by build.sh) over float32 input planes.
!
! Input file (little-endian stream): int32 ny, int32 nx, int32 ncase, then
! ncase pairs of int32 (option, npass) with option 1 = one_two_one,
! 2 = smth_desmth, 3 = smth_desmth_special, then ny*nx float32 values in
! numpy C order, which is Fortran array(ix, iy) with ix fastest: the same
! memory, so no transpose.
!
! Output file: ncase planes of ny*nx float32 values, one per case, each
! computed from a fresh copy of the input.
!
! The smoother arguments mirror process_tile_module.F:886-914 on a single
! process: the "dom" bounds are the patch (only exchange_halo_r reads them,
! and it is a no-op without _MPI) and the array bounds are the whole plane.
program wps_smooth_driver
   use smooth_module
   implicit none
   integer :: ny, nx, ncase, icase, unit_in, unit_out
   integer, allocatable :: options(:), passes(:)
   real, allocatable :: plane(:,:,:), work(:,:,:)
   character(len=1024) :: path_in, path_out
   real, parameter :: msgval = -1.e30

   call get_command_argument(1, path_in)
   call get_command_argument(2, path_out)
   open(newunit=unit_in, file=trim(path_in), access='stream', &
        form='unformatted', convert='little_endian', status='old')
   read(unit_in) ny, nx, ncase
   allocate(options(ncase), passes(ncase))
   do icase = 1, ncase
      read(unit_in) options(icase), passes(icase)
   end do
   allocate(plane(nx, ny, 1), work(nx, ny, 1))
   read(unit_in) plane
   close(unit_in)

   open(newunit=unit_out, file=trim(path_out), access='stream', &
        form='unformatted', convert='little_endian', status='replace')
   do icase = 1, ncase
      work = plane
      select case (options(icase))
      case (1)
         call one_two_one(work, 1, nx, 1, ny, 1, nx, 1, ny, 1, 1, &
                          passes(icase), msgval)
      case (2)
         call smth_desmth(work, 1, nx, 1, ny, 1, nx, 1, ny, 1, 1, &
                          passes(icase), msgval)
      case (3)
         call smth_desmth_special(work, 1, nx, 1, ny, 1, nx, 1, ny, 1, 1, &
                                  passes(icase), msgval)
      case default
         stop 'unknown option code'
      end select
      write(unit_out) work
   end do
   close(unit_out)
end program wps_smooth_driver
