program run_controls
  use module_fr_fire_util
  use module_fr_fire_core
  use oracle_io
  implicit none
  integer, parameter :: n=20
  real :: x(n), a(n), b(n), c(n), d(n), e(n), ff(n), area(n), ti(n), tj(n), out1(n), out2(n)
  integer :: i
  character(len=1024) :: root
  call oracle_root(root)
  fire_print_msg=0; fire_print_file=0
  do i=1,n
    x(i)=real(i-10)*0.25
    a(i)=real(mod(i,5)-2)*0.2
    b(i)=real(mod(i,7)-3)*0.1
    c(i)=real(mod(i,3)-1)*0.4
    d(i)=real(mod(i,9)-4)*0.3
    ti(i)=real(i)*0.25
    tj(i)=real(mod(i,4))
    out1(i)=select_upwind(a(i),b(i))
    out2(i)=select_godunov(a(i),b(i))
    call fuel_left_cell_1(ff(i),area(i),a(i),b(i),c(i),d(i),ti(i),tj(i),ti(i),tj(i),10.,30.)
  enddo
  call oracle_open('selectors')
  call oracle_put('left_in',a); call oracle_put('right_in',b)
  call oracle_put('upwind_out',out1); call oracle_put('godunov_out',out2)
  do i=1,n
    out1(i)=select_eno(a(i),b(i))
    out2(i)=select_2nd(0.1,5.,a(i),b(i),c(i))
  enddo
  call oracle_put('eno_out',out1); call oracle_put('second_out',out2)
  call oracle_put('m1_in',b); call oracle_put('p1_in',c)
  call oracle_close()
  call oracle_open('fuel_cell')
  call oracle_put('lfn00_in',a); call oracle_put('lfn01_in',b)
  call oracle_put('lfn10_in',c); call oracle_put('lfn11_in',d)
  call oracle_put('tign00_in',ti); call oracle_put('tign01_in',tj)
  call oracle_put('tign10_in',ti); call oracle_put('tign11_in',tj)
  call oracle_put('time_now',10.); call oracle_put('fuel_time',30.)
  call oracle_put('fuel_frac_out',ff); call oracle_put('fire_area_out',area)
  call oracle_close()
end program run_controls
