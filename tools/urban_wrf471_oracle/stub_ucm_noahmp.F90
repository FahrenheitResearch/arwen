! Service-only stubs for the UCM Noah-MP coupling oracle.  module_domain is
! USEd for its TYPE name only by GROUNDWATER_INIT, which this harness never
! calls; the rest is what module_wrf_error.o leaves undefined.  Nothing here
! computes a physical quantity.

module module_domain
  implicit none
  type :: domain
    integer :: unused
  end type domain
end module module_domain

subroutine wrf_abort()
  implicit none
  error stop 1
end subroutine wrf_abort

subroutine wrf_debug(level, msg)
  implicit none
  integer, intent(in) :: level
  character(len=*), intent(in) :: msg
end subroutine wrf_debug
