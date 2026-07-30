! Reference sweep of the float32 EXP that COMPACT calls.
!
! gfortran lowers REAL(4) EXP to expf@plt, so this program's output is the
! live glibc 2.39 expf on the build machine.  The CUDA kernel carries its own
! transcription of that function (glibc_expf in noahmp_snow.cu) because CUDA's
! expf, __expf and exp2f are all different functions; this sweep is what lets
! the device gate catch a mis-transcribed table entry that the eleven COMPACT
! cases happen not to reach.
!
! Domain.  COMPACT evaluates EXP at three sites:
!   -C4*TD                with TD = MAX(0, TFRZ-STC)  -> [-11, 0]
!   -46.0e-3*(BI-DM)      with BI <= DENICE           -> [-38, 0]
!   -0.08*TD - C2*BI                                  -> [-41, 0]
! so [-45, 0] strictly contains every reachable argument.  The sweep is dense
! enough that every one of the 32 __exp2f_data table entries is selected many
! times over: consecutive samples are far closer than the ln(2)/32 spacing at
! which the table index advances.

program run_snow_expf
  implicit none
  integer, parameter :: CSV = 22
  integer, parameter :: NCOARSE = 24001   ! [-45, 0]
  integer, parameter :: NFINE   = 12001   ! [-2.5, 0], where TD is small
  character(len=256) :: outfile
  integer :: i, bx, by
  real    :: x, y
  character(len=20) :: dec

  if (command_argument_count() /= 1) then
     write(*, '(A)') 'usage: run_snow_expf OUTPUT.csv'
     error stop 2
  end if
  call get_command_argument(1, outfile)

  open(CSV, file=trim(outfile), status='replace', action='write')
  write(CSV, '(A)') 'band,i,x_bits,y_bits,x,y'

  do i = 0, NCOARSE - 1
     x = -45.0 + 45.0 * real(i) / real(NCOARSE - 1)
     y = EXP(x)
     bx = transfer(x, bx)
     by = transfer(y, by)
     write(dec, '(ES16.9E2)') x
     write(CSV, '(A,I0,A)') 'coarse,', i, ','//hex8(bx)//','//hex8(by)//','// &
          trim(adjustl(dec))//','//trim(rtoa(y))
  end do

  do i = 0, NFINE - 1
     x = -2.5 + 2.5 * real(i) / real(NFINE - 1)
     y = EXP(x)
     bx = transfer(x, bx)
     by = transfer(y, by)
     write(dec, '(ES16.9E2)') x
     write(CSV, '(A,I0,A)') 'fine,', i, ','//hex8(bx)//','//hex8(by)//','// &
          trim(adjustl(dec))//','//trim(rtoa(y))
  end do

  close(CSV)
  write(*, '(A,I0,A)') 'run_snow_expf: wrote ', NCOARSE + NFINE, ' samples'

contains

  function hex8(b) result(s)
    integer, intent(in) :: b
    character(len=8) :: s
    write(s, '(Z8.8)') b
  end function hex8

  function rtoa(v) result(s)
    real, intent(in) :: v
    character(len=20) :: s
    write(s, '(ES16.9E2)') v
    s = adjustl(s)
  end function rtoa

end program run_snow_expf
