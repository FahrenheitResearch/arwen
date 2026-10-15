! Service-only stub for the standalone Milbrandt-Yau oracle.
!
! phys/module_mp_milbrandt2mom.F USEs module_wrf_error but neither the
! wrapper nor the scheme calls any WRF logging or fatal-error routine (its
! only stops are bare Fortran STOP statements).  The empty module supplies
! the compile-time dependency and contains no constants or arithmetic.
module module_wrf_error
  implicit none
end module module_wrf_error
