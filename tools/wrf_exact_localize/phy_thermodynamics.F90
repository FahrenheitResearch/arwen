! Standalone evaluation of phy_prep's two thermodynamic statements.
! Constants are imported from the box's unchanged WRF 4.6.1 source.
program phy_thermodynamics
  use module_model_constants, only: p1000mb, rcp
  implicit none
  integer :: n, i, unit
  real, allocatable :: pressure(:), theta(:), exner(:), temperature(:)
  character(len=1024) :: input, output
  call get_command_argument(1, input)
  call get_command_argument(2, output)
  open(newunit=unit, file=trim(input), access='stream', form='unformatted', &
       convert='little_endian', status='old')
  read(unit) n
  allocate(pressure(n), theta(n), exner(n), temperature(n))
  read(unit) pressure, theta
  close(unit)
  do i=1,n
    exner(i) = (pressure(i)/p1000mb)**rcp
    temperature(i) = theta(i)*exner(i)
  enddo
  open(newunit=unit, file=trim(output), access='stream', form='unformatted', &
       convert='little_endian', status='replace')
  write(unit) exner, temperature
  close(unit)
end program phy_thermodynamics
