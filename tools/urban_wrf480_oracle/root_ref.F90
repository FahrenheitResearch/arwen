! ABI-only wrapper. All numerical calculations use the release module.
subroutine root_ref(out,n,smc,smax,sdry,radius,lai,dz,zsoil,ele,fvg,width) bind(C)
  use iso_c_binding
  use module_sf_urban, only: root_uptake
  implicit none
  integer(c_int), value :: n
  real(c_float), intent(out) :: out(n)
  real(c_float), intent(in) :: smc(n), dz(n), zsoil(n)
  real(c_float), value :: smax, sdry, radius, lai, ele, fvg, width
  call root_uptake(out,n,smc,smax,sdry,radius,lai,dz,zsoil,ele,fvg,width)
end subroutine root_ref
