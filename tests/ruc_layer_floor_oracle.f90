! Scalar check of WRF 4.6.1 module_initialize_real.F's RUCLSMSCHEME
! flag_soil_layers arm: grid%smois(i,:,j) = MAX(grid%smois(i,:,j), 0.005).
! The caller supplies real prepared f32 words, without an array header.
program layer_floor_oracle
  use iso_fortran_env, only: real32
  implicit none
  real(real32), allocatable :: moisture(:)
  character(len=1024) :: input_path, output_path, count_text
  integer :: count, i, ios, n_floored
  call get_command_argument(1, input_path)
  call get_command_argument(2, output_path)
  call get_command_argument(3, count_text)
  read(count_text, *, iostat=ios) count
  if (ios /= 0 .or. count <= 0) stop 2
  allocate(moisture(count))
  open(unit=10,file=trim(input_path),access='stream',form='unformatted',status='old')
  read(10) moisture
  close(10)
  n_floored = 0
  do i = 1, count
    if (moisture(i) < 0.005_real32) n_floored = n_floored + 1
    moisture(i) = max(moisture(i), 0.005_real32)
  end do
  open(unit=11,file=trim(output_path),access='stream',form='unformatted',status='replace')
  write(11) moisture
  close(11)
  write(*,'(a,i0)') 'floored_values=', n_floored
end program layer_floor_oracle
