program run_math
  use oracle_io
  implicit none
  integer,parameter :: n=120
  integer :: i
  real :: x(n),cube(n),arg(n),th(n),cdf(n)
  real,parameter :: a=167./148.,b=11./109.
  character(len=1024) :: root
  call oracle_root(root)
  do i=1,n
    x(i)=real(i-65)/30.
    cube(i)=x(i)**3
    arg(i)=a*x(i)+b*cube(i)
    th(i)=tanh(arg(i))
    cdf(i)=0.5*(1.+th(i))
  enddo
  call oracle_open('math_tg')
  call oracle_put('x_in',x);call oracle_put('cube_out',cube)
  call oracle_put('arg_out',arg);call oracle_put('tanh_out',th)
  call oracle_put('cdf_out',cdf);call oracle_close()
end program run_math
