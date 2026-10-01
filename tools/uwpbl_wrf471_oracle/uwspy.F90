! Stage recorder for the SPY build of the UW PBL oracle (see make_spy.py).
!
! The spy build compiles copies of module_cam_bl_eddy_diff.F and
! module_bl_camuwpbl_driver.F into which make_spy.py has inserted calls to
! this module around the scheme's internal calls (trbintd, caleddy, the
! in-loop compute_vdiff, compute_eddy_diff and the outer compute_vdiff).
! The calls only READ their arguments, and build.sh proves that by
! requiring the spy build's step outputs to equal the pristine build's
! bit for bit.  The stage records are debugging fixtures for the port's
! private routines; the fixture of record is the pristine build's.
module uwspy
  implicit none
  integer, parameter, private :: r8 = selected_real_kind(12)
  integer, private :: ubin = 0, uman = 0
  logical, private :: isopen = .false.
  integer(8), private :: offset = 0
  integer, private :: ccol = 0, cstep = 0, citer = 0
contains
  subroutine uwspy_open(stem)
    character(len=*), intent(in) :: stem
    open(newunit=ubin, file=trim(stem)//'.bin', access='stream', &
         form='unformatted', status='replace', action='write')
    open(newunit=uman, file=trim(stem)//'.manifest', status='replace', &
         action='write')
    offset = 0
    isopen = .true.
  end subroutine uwspy_open

  subroutine uwspy_close()
    ! newunit= numbers are negative, so openness is tracked, not inferred
    if (isopen) then
       close(ubin)
       close(uman)
    end if
    isopen = .false.
  end subroutine uwspy_close

  subroutine uwspy_col(i)
    integer, intent(in) :: i
    ccol = i
  end subroutine uwspy_col

  subroutine uwspy_step(s)
    integer, intent(in) :: s
    cstep = s
  end subroutine uwspy_step

  subroutine uwspy_iter(it)
    integer, intent(in) :: it
    citer = it
  end subroutine uwspy_iter

  subroutine uwspy_r8(name, arr, n)
    character(len=*), intent(in) :: name
    integer, intent(in) :: n
    real(r8), intent(in) :: arr(*)
    character(len=64) :: ctx
    if (.not. isopen) return
    write(ctx, '(a,i0,a,i0,a,i0,a)') 'c', ccol, '/s', cstep, '/it', citer, '/'
    write(uman, '(a,1x,a,1x,i0,1x,i0)') trim(ctx)//trim(name), 'f8', n, offset
    offset = offset + int(n, 8) * 8_8
    write(ubin) arr(1:n)
  end subroutine uwspy_r8

  subroutine uwspy_i4(name, arr, n)
    character(len=*), intent(in) :: name
    integer, intent(in) :: n
    integer, intent(in) :: arr(*)
    character(len=64) :: ctx
    if (.not. isopen) return
    write(ctx, '(a,i0,a,i0,a,i0,a)') 'c', ccol, '/s', cstep, '/it', citer, '/'
    write(uman, '(a,1x,a,1x,i0,1x,i0)') trim(ctx)//trim(name), 'i4', n, offset
    offset = offset + int(n, 8) * 4_8
    write(ubin) arr(1:n)
  end subroutine uwspy_i4

  subroutine uwspy_l(name, arr, n)
    ! logicals are recorded as i4 0/1
    character(len=*), intent(in) :: name
    integer, intent(in) :: n
    logical, intent(in) :: arr(*)
    integer :: tmp(n), j
    do j = 1, n
       tmp(j) = merge(1, 0, arr(j))
    end do
    call uwspy_i4(name, tmp, n)
  end subroutine uwspy_l

  subroutine uwspy_l1(name, x)
    character(len=*), intent(in) :: name
    logical, intent(in) :: x
    integer :: tmp(1)
    tmp(1) = merge(1, 0, x)
    call uwspy_i4(name, tmp, 1)
  end subroutine uwspy_l1

  subroutine uwspy_s(name, x)
    character(len=*), intent(in) :: name
    real(r8), intent(in) :: x
    real(r8) :: a(1)
    a(1) = x
    call uwspy_r8(name, a, 1)
  end subroutine uwspy_s
end module uwspy
