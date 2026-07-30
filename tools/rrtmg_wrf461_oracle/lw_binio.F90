! Little-endian binary record writer for the RRTMG LW oracle fixtures.
!
! The oracle build compiles the physics module with -fconvert=big-endian so
! rrtmg_lwlookuptable reads RRTMG_LW_DATA with WRF's byte order; fixture
! files therefore pin convert='little_endian' explicitly so Python reads
! them with plain numpy little-endian dtypes.
!
! Record layout (all int32 little-endian):
!   name_len, name bytes, dtype (0=float32, 1=int32), rank, dims(rank),
!   payload (Fortran order)

module lw_binio
  implicit none
  private
  public :: bio_open, bio_close, wr_r0, wr_r1, wr_r2, wr_r3, wr_r4, &
            wr_i0, wr_i1

  integer, save :: bu = -1

contains

  subroutine bio_open(path)
    character(len=*), intent(in) :: path
    open(newunit=bu, file=trim(path), form='unformatted', access='stream', &
         status='replace', action='write', convert='little_endian')
  end subroutine bio_open

  subroutine bio_close()
    close(bu)
    bu = -1
  end subroutine bio_close

  subroutine wr_head(name, dtype, dims)
    character(len=*), intent(in) :: name
    integer, intent(in) :: dtype
    integer, intent(in) :: dims(:)
    integer :: n
    n = len_trim(name)
    write(bu) n
    write(bu) name(1:n)
    write(bu) dtype
    write(bu) size(dims)
    if (size(dims) > 0) write(bu) dims
  end subroutine wr_head

  subroutine wr_r0(name, v)
    character(len=*), intent(in) :: name
    real, intent(in) :: v
    integer :: nodims(0)
    call wr_head(name, 0, nodims)
    write(bu) v
  end subroutine wr_r0

  subroutine wr_r1(name, v)
    character(len=*), intent(in) :: name
    real, intent(in) :: v(:)
    call wr_head(name, 0, shape(v))
    write(bu) v
  end subroutine wr_r1

  subroutine wr_r2(name, v)
    character(len=*), intent(in) :: name
    real, intent(in) :: v(:,:)
    call wr_head(name, 0, shape(v))
    write(bu) v
  end subroutine wr_r2

  subroutine wr_r3(name, v)
    character(len=*), intent(in) :: name
    real, intent(in) :: v(:,:,:)
    call wr_head(name, 0, shape(v))
    write(bu) v
  end subroutine wr_r3

  subroutine wr_r4(name, v)
    character(len=*), intent(in) :: name
    real, intent(in) :: v(:,:,:,:)
    call wr_head(name, 0, shape(v))
    write(bu) v
  end subroutine wr_r4

  subroutine wr_i0(name, v)
    character(len=*), intent(in) :: name
    integer, intent(in) :: v
    integer :: nodims(0)
    call wr_head(name, 1, nodims)
    write(bu) v
  end subroutine wr_i0

  subroutine wr_i1(name, v)
    character(len=*), intent(in) :: name
    integer, intent(in) :: v(:)
    call wr_head(name, 1, shape(v))
    write(bu) v
  end subroutine wr_i1

end module lw_binio
