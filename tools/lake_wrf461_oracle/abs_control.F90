! Native Fortran ABS words: zeros, subnormals, finite values and NaNs.
program lake_abs_control
  use iso_c_binding, only: c_float,c_double,c_int32_t,c_int64_t
  implicit none
  integer(c_int32_t),parameter :: single(12) = [ &
    int(z'80000000',c_int32_t),int(z'00000000',c_int32_t), &
    int(z'00000001',c_int32_t),int(z'80000001',c_int32_t), &
    int(z'3F800000',c_int32_t),int(z'BF800000',c_int32_t), &
    int(z'7F800000',c_int32_t),int(z'FF800000',c_int32_t), &
    int(z'7FC12345',c_int32_t),int(z'FFC12345',c_int32_t), &
    int(z'7F812345',c_int32_t),int(z'FF812345',c_int32_t)]
  integer(c_int64_t),parameter :: dual(12) = [ &
    int(z'8000000000000000',c_int64_t),int(z'0000000000000000',c_int64_t), &
    int(z'0000000000000001',c_int64_t),int(z'8000000000000001',c_int64_t), &
    int(z'3FF0000000000000',c_int64_t),int(z'BFF0000000000000',c_int64_t), &
    int(z'7FF0000000000000',c_int64_t),int(z'FFF0000000000000',c_int64_t), &
    int(z'7FF8123456789ABC',c_int64_t),int(z'FFF8123456789ABC',c_int64_t), &
    int(z'7FF0123456789ABC',c_int64_t),int(z'FFF0123456789ABC',c_int64_t)]
  real(c_float) :: f,fo
  real(c_double) :: d,dout
  integer :: i
  do i=1,12
    f=transfer(single(i),f);d=transfer(dual(i),d)
    fo=abs(f);dout=abs(d)
    write(*,'(Z8.8,1X,Z8.8,1X,Z16.16,1X,Z16.16)') &
      single(i),transfer(fo,0_c_int32_t),dual(i),transfer(dout,0_c_int64_t)
  enddo
end program
