! Shared tagged-stream dump writers for the RRTMG WRF v4.6.1 oracle.
!
! Entry layout (every integer/real is byte-swapped to big-endian by the
! oracle build's -fconvert=big-endian, matching WRF's BYTESWAPIO):
!   int32   name length L
!   L bytes ASCII name (character data is never byte-swapped)
!   int32   dtype code: 4 = 32-bit real, 2 = 32-bit integer
!   int32   rank (0 for scalars)
!   int32 * rank  extents, Fortran dimension order
!   payload, Fortran storage order
!
! The kg-module arrays keep WRF's declared kinds (kind_rb/kind_im); the
! FP32 oracle contract requires both to be 32-bit, which check_kinds
! enforces before anything is written.
module dump_kit
   use parkind, only: im => kind_im, rb => kind_rb
   implicit none
   private
   public :: check_kinds, wr0, wr1, wr2, wr3, wr4, wi0

contains

   subroutine check_kinds()
      if (storage_size(1.0_rb) /= 32 .or. storage_size(1_im) /= 32) then
         write (*, '(A)') &
            'dump_kit: kind_rb/kind_im are not 32-bit; not the FP32 build'
         error stop 4
      end if
   end subroutine check_kinds

   subroutine whdr(u, name, dtype, dims)
      integer, intent(in) :: u, dtype
      character(len=*), intent(in) :: name
      integer, intent(in) :: dims(:)
      integer :: i
      write (u) int(len_trim(name), 4)
      write (u) name(1:len_trim(name))
      write (u) int(dtype, 4)
      write (u) int(size(dims), 4)
      do i = 1, size(dims)
         write (u) int(dims(i), 4)
      end do
   end subroutine whdr

   subroutine wr0(u, name, a)
      integer, intent(in) :: u
      character(len=*), intent(in) :: name
      real(kind=rb), intent(in) :: a
      call whdr(u, name, 4, [integer ::])
      write (u) a
   end subroutine wr0

   subroutine wr1(u, name, a)
      integer, intent(in) :: u
      character(len=*), intent(in) :: name
      real(kind=rb), intent(in) :: a(:)
      call whdr(u, name, 4, shape(a))
      write (u) a
   end subroutine wr1

   subroutine wr2(u, name, a)
      integer, intent(in) :: u
      character(len=*), intent(in) :: name
      real(kind=rb), intent(in) :: a(:, :)
      call whdr(u, name, 4, shape(a))
      write (u) a
   end subroutine wr2

   subroutine wr3(u, name, a)
      integer, intent(in) :: u
      character(len=*), intent(in) :: name
      real(kind=rb), intent(in) :: a(:, :, :)
      call whdr(u, name, 4, shape(a))
      write (u) a
   end subroutine wr3

   subroutine wr4(u, name, a)
      integer, intent(in) :: u
      character(len=*), intent(in) :: name
      real(kind=rb), intent(in) :: a(:, :, :, :)
      call whdr(u, name, 4, shape(a))
      write (u) a
   end subroutine wr4

   subroutine wi0(u, name, a)
      integer, intent(in) :: u
      character(len=*), intent(in) :: name
      integer(kind=im), intent(in) :: a
      call whdr(u, name, 2, [integer ::])
      write (u) a
   end subroutine wi0

end module dump_kit
