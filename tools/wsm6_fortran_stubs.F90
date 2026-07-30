module ccpp_kind_types
  implicit none
  integer, parameter :: kind_phys=kind(1.0)
end module

module module_libmassv
  use ccpp_kind_types, only: kind_phys
  implicit none
contains
  subroutine vrec(out,inp,n)
    integer,intent(in):: n
    real(kind_phys),intent(out):: out(*)
    real(kind_phys),intent(in):: inp(*)
    integer i
    do i=1,n; out(i)=1.0_kind_phys/inp(i); enddo
  end subroutine
  subroutine vsqrt(out,inp,n)
    integer,intent(in):: n
    real(kind_phys),intent(out):: out(*)
    real(kind_phys),intent(in):: inp(*)
    integer i
    do i=1,n; out(i)=sqrt(inp(i)); enddo
  end subroutine
end module
