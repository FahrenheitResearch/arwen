program run_mynn_esat_blend_oracle
  ! Pins the three phase-blend helper functions as pure functions of state.
  ! esat_blend is never reached by the mym_condensation CASE(2) lane, so it
  ! has no coverage from any column fixture; qsat_blend and xl_blend are
  ! reached there but only over the temperature range those columns span.
  ! The sweep below crosses every branch boundary of all three: the -80 K
  ! XC clamp, the pure-ice branch at or below tice, the blended band, the
  ! pure-liquid branch, and the qsat 0.15*p vapour-pressure ceiling.
  use module_bl_mynn, only: esat_blend, qsat_blend, xl_blend
  implicit none

  integer, parameter :: nt = 37, np = 3
  real, parameter :: temperature(nt) = [ &
      180.00, 185.00, 190.00, 193.14, 193.15, 193.16, 200.00, 210.00, &
      220.00, 230.00, 235.00, 239.99, 240.00, 240.01, 245.00, 250.00, &
      255.00, 260.00, 265.00, 267.10, 267.15, 267.20, 270.00, 272.00, &
      273.14, 273.15, 273.16, 275.00, 280.00, 285.00, 290.00, 295.00, &
      300.00, 305.00, 310.00, 315.00, 320.00]
  real, parameter :: pressure(np) = [100000.0, 50000.0, 15000.0]
  character(len=32), parameter :: names(np) = [character(len=32) :: &
      'surface_pressure', 'midlevel_pressure', 'upper_pressure']
  character(len=1024) :: output_path
  integer :: i, j, unit

  call get_command_argument(1, output_path)
  if (len_trim(output_path) == 0) then
    write(*, '(A)') 'usage: run_esat_blend OUTPUT.csv'
    error stop 2
  end if
  open(newunit=unit, file=trim(output_path), status='replace', action='write')
  write(unit, '(A)') 'case,i,t,p,esat_blend,qsat_blend,xl_blend'

  do j = 1, np
    do i = 1, nt
      write(unit, '(*(g0,:,","))') trim(names(j)), i, temperature(i), &
          pressure(j), esat_blend(temperature(i)), &
          qsat_blend(temperature(i), pressure(j)), xl_blend(temperature(i))
    end do
  end do
  close(unit)
end program run_mynn_esat_blend_oracle
