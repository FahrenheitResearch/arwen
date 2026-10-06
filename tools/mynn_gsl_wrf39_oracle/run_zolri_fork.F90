! Leaf oracle for the fork's secant search and its in-place sign reset.
program run_zolri_fork
  use module_sf_mynn, only: mynn_sf_init_driver, zolri, zolri2
  implicit none
  integer :: iu, ou, ios
  real :: ri, za, z0, zt, guess, result, reset_arg, residual
  character(len=1024) :: input_path, output_path
  call get_command_argument(1, input_path)
  call get_command_argument(2, output_path)
  call mynn_sf_init_driver(.false.)
  open(newunit=iu, file=trim(input_path), status='old', action='read')
  open(newunit=ou, file=trim(output_path), status='new', action='write')
  write(ou, '(A)') 'ri,za,z0,zt,guess,zol,reset_arg,residual'
  do
    read(iu, *, iostat=ios) ri, za, z0, zt, guess
    if (ios < 0) exit
    if (ios /= 0) error stop 'invalid leaf oracle input'
    result = zolri(ri, za, z0, zt, guess)
    reset_arg = -guess
    residual = zolri2(reset_arg, ri, za, z0, zt)
    write(ou, '(ES24.16E3,7(",",ES24.16E3))') &
      ri, za, z0, zt, guess, result, reset_arg, residual
  end do
  close(iu)
  close(ou)
end program run_zolri_fork
