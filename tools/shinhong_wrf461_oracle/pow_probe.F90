program pow_probe
  ! Establish, adversarially and on the oracle's own toolchain at -O0, how
  ! gfortran lowers every real-exponent power form that appears in
  ! module_bl_shinhong.F -- BEFORE the port spells any of them.  The trap
  ! cookbook's pow(x,1.0) lesson generalises: `x**2.` may be powf(x,2.0) or
  ! x*x depending on the compiler's constant-exponent folding, and the two
  ! can differ in the last ULP for x**3. and beyond.  This program prints the
  ! bit pattern of each expression next to its candidate decompositions so
  ! the port matches measured semantics, not guessed ones.
  !
  ! Exponent forms in the pinned module (with their sites):
  !   **2.        entfac/entfacmf/prnumfac/rigs/deltaoh (lines 971, 990-993,
  !               1011), incl. NEGATIVE bases (dux/dvx at 971)
  !   **3.        ust**3. (776), zfacent (1006), zfacentk (2185: **3.0)
  !   **4.        wscale**4. (794)
  !   **pfac==2.0 zfac**pfac (1022)
  !   **(pfac_q-pfac)==0.0  zfac**0. (1024)
  !   **h1 (0.33333333)     wstar/wscale/wm2 cube roots (769, 777, 1007)
  !   **h2 (0.6666667)      wm2 (937)
  !   **(-1./4.), **(-1./2.) phim/phih (763-764)
  !   **(-0.18)   brcr_sbro Rossby fit (872)
  !   **(1./3.)   ckp in mixlen (1980)
  !   **1.5       q2l**1.5 in prodq2 dissipation (2112)
  implicit none

  real, parameter :: h1 = 0.33333333, h2 = 0.6666667
  real, parameter :: pfac = 2.0, pfac_q = 2.0
  integer, parameter :: nx = 9
  real :: xs(nx)
  integer :: i

  xs = (/ 3.0e-4, 0.05, 0.3, 0.9973, 1.0, 1.5, 7.25, 137.5, 2.5e5 /)

  write(*, '(A)') '# pow_probe: each line = expr, x bits, result bits.'
  write(*, '(A)') '# built at -O0 beside the oracle; nm -u pow_probe tells'
  write(*, '(A)') '# which forms stayed powf calls.'
  do i = 1, nx
    call probe(xs(i))
  end do
  ! Negative bases with integral real exponents: prohibited by the Fortran
  ! standard, compiled by gfortran anyway (dux**2. at line 971 reaches this
  ! whenever du < 0).  Record what the toolchain actually does.
  call probe_neg(-1.7)
  call probe_neg(-3.0e-4)
  ! x = 0 with exponent 0.0 (zfac**0. via pfac_q-pfac) and with 1.5.
  call probe_zero()

contains

  subroutine emit(name, x, v)
    character(len=*), intent(in) :: name
    real, intent(in) :: x, v
    write(*, '(A,T26,Z8.8,1X,Z8.8,1X,ES16.8E3)') name, transfer(x, 0), &
         transfer(v, 0), v
  end subroutine emit

  subroutine probe(x)
    real, intent(in) :: x
    real :: v
    write(*, '(A,ES16.8E3)') '## x = ', x
    v = x ** 2.;            call emit('x**2.', x, v)
    v = x * x;              call emit('x*x', x, v)
    v = x ** 3.;            call emit('x**3.', x, v)
    v = x * x * x;          call emit('x*x*x', x, v)
    v = (x * x) * x;        call emit('(x*x)*x', x, v)
    v = x ** 4.;            call emit('x**4.', x, v)
    v = (x * x) * (x * x);  call emit('(x*x)*(x*x)', x, v)
    v = x ** pfac;          call emit('x**pfac(2.0)', x, v)
    v = x ** (pfac_q - pfac); call emit('x**(pfac_q-pfac)', x, v)
    v = x ** h1;            call emit('x**h1', x, v)
    v = x ** h2;            call emit('x**h2', x, v)
    v = x ** (1. / 3.);     call emit('x**(1./3.)', x, v)
    v = x ** (-1. / 4.);    call emit('x**(-1./4.)', x, v)
    v = x ** (-1. / 2.);    call emit('x**(-1./2.)', x, v)
    v = 1.0 / sqrt(x);      call emit('1/sqrt(x)', x, v)
    v = x ** (-0.18);       call emit('x**(-0.18)', x, v)
    v = x ** 1.5;           call emit('x**1.5', x, v)
    v = x * sqrt(x);        call emit('x*sqrt(x)', x, v)
    v = x ** 0.875;         call emit('x**0.875', x, v)
    v = x ** 0.5;           call emit('x**0.5', x, v)
    v = sqrt(x);            call emit('sqrt(x)', x, v)
    v = x ** 0.6666667;     call emit('x**0.6666667lit', x, v)
  end subroutine probe

  subroutine probe_neg(x)
    real, intent(in) :: x
    real :: v
    write(*, '(A,ES16.8E3)') '## negative base, x = ', x
    v = x ** 2.;            call emit('negx**2.', x, v)
    v = x * x;              call emit('negx*negx', x, v)
    v = x ** 3.;            call emit('negx**3.', x, v)
    v = x * x * x;          call emit('negx cubed', x, v)
  end subroutine probe_neg

  subroutine probe_zero()
    real :: z, v
    z = 0.0
    write(*, '(A)') '## x = +0.0'
    v = z ** (pfac_q - pfac); call emit('0**(pfac_q-pfac)', z, v)
    v = z ** 1.5;             call emit('0**1.5', z, v)
    v = z ** 2.;              call emit('0**2.', z, v)
  end subroutine probe_zero

end program pow_probe
