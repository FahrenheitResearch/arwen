program sfcdiags_sweep
  ! WRF v4.6.1 SFCDIAGS (phys/module_sf_sfcdiags.F, no HWRF, ua_phys false)
  ! over N independent columns read from a stream file, for
  ! sfcdiags_sweep.py.  The run_lsm oracle reaches SFCDIAGS on 80 columns,
  ! nearly all at one surface pressure, which says almost nothing about
  ! TH2 = T2*(1.E5/PSFC)**ROVCP: that power is a libm call whose argument
  ! is the pressure ratio, so it is graded here on every float32 PSFC from
  ! 20 kPa to 110 kPa instead.
  !
  ! Input  (stream, native float32): N as int32, then PSFC, TSK, HFX, QFX,
  !        QSFC, CHS2, CQS2 (N words each).
  ! Output (stream, native float32): T2, TH2, Q2 (N words each).
  use module_sf_sfcdiags, only: sfcdiags
  use module_model_constants, only: cp, r_d, rcp
  implicit none
  integer :: n, u
  real, allocatable :: psfc(:, :), tsk(:, :), hfx(:, :), qfx(:, :)
  real, allocatable :: qsfc(:, :), chs2(:, :), cqs2(:, :), chs(:, :)
  real, allocatable :: t2(:, :), th2(:, :), q2(:, :)
  real, allocatable :: t3d(:, :, :), qv3d(:, :, :)
  character(len=1024) :: in_path, out_path

  call get_command_argument(1, in_path)
  call get_command_argument(2, out_path)
  open(newunit=u, file=trim(in_path), access='stream', form='unformatted', &
       status='old', action='read')
  read(u) n
  allocate(psfc(n, 1), tsk(n, 1), hfx(n, 1), qfx(n, 1), qsfc(n, 1))
  allocate(chs2(n, 1), cqs2(n, 1), chs(n, 1), t2(n, 1), th2(n, 1), q2(n, 1))
  allocate(t3d(n, 1, 1), qv3d(n, 1, 1))
  read(u) psfc, tsk, hfx, qfx, qsfc, chs2, cqs2
  close(u)
  ! CHS, T3D and QV3D are read only on the UA_PHYS and HWRF arms.
  chs = 0.02; t3d = 0.0; qv3d = 0.0
  t2 = 0.0; th2 = 0.0; q2 = 0.0
  call sfcdiags(hfx, qfx, tsk, qsfc, chs2, cqs2, t2, th2, q2,        &
                psfc, cp, r_d, rcp, chs, t3d, qv3d, .false.,        &
                1, n, 1, 1, 1, 1,                                   &
                1, n, 1, 1, 1, 1,                                   &
                1, n, 1, 1, 1, 1)
  open(newunit=u, file=trim(out_path), access='stream', form='unformatted', &
       status='replace', action='write')
  write(u) t2, th2, q2
  close(u)
end program sfcdiags_sweep
