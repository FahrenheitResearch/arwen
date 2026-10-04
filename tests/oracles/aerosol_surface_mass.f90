! Operational WRF module_initialize_real.F:4428-4430, NOAA-EMC/HRRR WRFV3.9.
! WRF public-domain notice: licenses/LICENSE-WRF-public-domain.txt.
! The three numerical statements preserve the upstream REAL expression order.
subroutine aerosol_surface_oracle(number, lower, upper, alt, result, n, g, dx, dy) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int), value :: n
  real(c_float), value :: g, dx, dy
  real(c_float), intent(in) :: number(n), lower(n), upper(n), alt(n)
  real(c_float), intent(out) :: result(n)
  real(c_float) :: z1, airmass
  integer :: i
  do i=1,n
    z1 = (upper(i)-lower(i))/g
    airmass = 1./alt(i) * z1 * dx*dy
    result(i) = number(i) * 0.000196 * (airmass*2.E-10)
  enddo
end subroutine aerosol_surface_oracle
