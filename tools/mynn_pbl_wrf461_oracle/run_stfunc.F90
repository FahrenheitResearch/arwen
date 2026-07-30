program run_mynn_stfunc_oracle
  ! Dump module_bl_mynn.F:7525-7623 (phim, phih) from the unmodified pinned
  ! WRF v4.6.1 physics module.  bl_mynn_stfunc is a compile-time 1 at :340,
  ! so these two functions -- not the Kansas forms at :1085-1091 -- are what
  ! the driver uses to build pmz and phh for mym_predict.
  !
  ! They are the only place in the MYNN PBL lane where FP32 log, atan and a
  ! real-exponent ** meet in one expression, so they are pinned as pure
  ! functions of z/L over the range the driver can produce.  The driver clamps
  ! zet to [-20, 20] at :1080-1081, and the sweep covers that whole interval
  ! plus the exact endpoints and the zet=0 branch boundary.
  use module_bl_mynn, only: phim, phih
  implicit none

  integer, parameter :: nsweep = 801
  character(len=1024) :: output_path
  integer :: i, unit
  real :: zet

  call get_command_argument(1, output_path)
  if (len_trim(output_path) == 0) then
    write(*, '(A)') 'usage: run_stfunc OUTPUT.csv'
    error stop 2
  end if
  open(newunit=unit, file=trim(output_path), status='new', action='write')
  write(unit, '(A)') 'i,zet,phim,phih'
  do i = 1, nsweep
    zet = -20.0 + 40.0 * real(i - 1) / real(nsweep - 1)
    write(unit, '(I0)', advance='no') i
    write(unit, '(3(",",ES24.16E3))') zet, phim(zet), phih(zet)
  end do
  ! Exact boundary and small-magnitude probes: zet=0 takes the stable arm,
  ! and the unstable arm divides by zet, so the near-zero side matters.
  do i = 1, 13
    select case (i)
    case (1); zet = 0.0
    case (2); zet = -1.0e-7
    case (3); zet = 1.0e-7
    case (4); zet = -1.0e-4
    case (5); zet = 1.0e-4
    case (6); zet = -0.5
    case (7); zet = 0.5
    case (8); zet = -1.0
    case (9); zet = 1.0
    case (10); zet = -20.0
    case (11); zet = 20.0
    case (12); zet = -1.0e-30
    case (13); zet = 1.0e-30
    end select
    write(unit, '(I0)', advance='no') nsweep + i
    write(unit, '(3(",",ES24.16E3))') zet, phim(zet), phih(zet)
  end do
  close(unit)
end program run_mynn_stfunc_oracle
