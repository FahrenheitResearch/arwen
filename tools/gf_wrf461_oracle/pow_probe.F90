program gf_pow_probe
  use module_gfs_physcons, only: con_g, con_cp, con_hvap, con_rv
  use module_gfs_machine, only: kind_phys
  ! Print the bit pattern of every real- and integer-exponent power form that
  ! appears in module_cu_gf_deep.F / module_cu_gf_sh.F, beside its algebraic
  ! decompositions, on THIS toolchain at -O0.
  !
  ! The Shin-Hong oracle needed this because gfortran folds `x**2.` to `x*x`
  ! bitwise but leaves `x**3.` as a correctly-rounded powf, so a port that
  ! spells the second as a multiply chain is wrong in the last ULP.  GF has
  ! its own list, and two entries no prior oracle in this repo has had to
  ! answer for:
  !
  !   * `gamma()` -- the F2008 intrinsic, compiled to tgammaf, used at
  !     module_cu_gf_deep.F:3854/:3882/:3918/:3952 to normalise the
  !     beta-function updraft mass-flux profile.  Its arguments are runtime
  !     values (alpha depends on the pressure-derived `tunning`), so there is
  !     nothing to fold.
  !   * `zws**.3333` (deep :395,:404; sh :307,:316).  `.3333` is NOT 1/3 in
  !     float32 -- it is 0.33329999446868896... -- so a port that spells this
  !     as cbrt is wrong, and by more than a rounding.
  !
  ! Nothing here is checked against an expectation.  Every line is a
  ! measurement printed as hex so the reference implementation can be built
  ! against it.
  implicit none

  integer, parameter :: np = 12
  real, parameter :: probe(np) = (/ &
       1.0e-6, 1.0e-3, 0.05, 0.2333, 0.5, 0.75, &
       0.9999, 1.0, 1.7, 3.25, 47.5, 1234.5 /)
  ! alpha/beta pairs reachable in get_zu_zd_pdf_fim.  beta is a literal per
  ! draft type (1.3 "UP", 2.5 "DOWN"/"SH2", 3.5 the shallow branch); alpha
  ! follows from tunning in [0.2, 0.9].
  integer, parameter :: ng = 10
  real, parameter :: gprobe(ng) = (/ &
       1.075, 1.3, 1.5, 2.0, 2.375, 2.5, 3.0, 3.5, 3.7, 5.0 /)

  integer :: i, j
  real :: x, y, a, b

  write(*, '(A)') '# gf pow/gamma probe, -O0, this toolchain'
  write(*, '(A)') '# every value is a float32 bit pattern in hex'

  ! ---- x**.3333, the zws cube-rootish form --------------------------------
  write(*, '(A)') '#'
  write(*, '(A)') '# form: x**.3333 (deep:395,:404 sh:307,:316)'
  write(*, '(A)') '#   columns: x  x**.3333  x**(1./3.)  cbrt(x)  literal(.3333)'
  do i = 1, np
    x = probe(i)
    write(*, '(A,4(1X,A),1X,A)') 'p3333', h(x), h(x ** 0.3333), &
      h(x ** (1.0 / 3.0)), h(x ** (1.0 / 3.0)), h(0.3333)
  end do
  ! cbrt is not a Fortran intrinsic; the third and fourth columns above are
  ! the same expression on purpose, so the file records that the port has no
  ! cbrt arm to consider -- only powf(x, 0.3333) vs powf(x, 1./3.).

  ! ---- integer exponents: what __powisf2 does -----------------------------
  write(*, '(A)') '#'
  write(*, '(A)') '# form: x**2 and x**3, INTEGER exponents (deep:469,:471,'
  write(*, '(A)') '#       :1171,:1774,:1946,:1947) -- gfortran emits __powisf2'
  write(*, '(A)') '#   columns: x  x**2  x*x  x**3  x*x*x'
  do i = 1, np
    x = probe(i)
    write(*, '(A,5(1X,A))') 'pint', h(x), h(x ** 2), h(x * x), &
      h(x ** 3), h(x * x * x)
  end do
  ! negative bases too: VSHEAR**3 at :1947 is signed
  do i = 1, np
    x = -probe(i)
    write(*, '(A,5(1X,A))') 'pintneg', h(x), h(x ** 2), h(x * x), &
      h(x ** 3), h(x * x * x)
  end do

  ! ---- real exponents: the beta-function profile --------------------------
  write(*, '(A)') '#'
  write(*, '(A)') '# form: kratio**(alpha-1.) * (1-kratio)**(beta-1.)'
  write(*, '(A)') '#       (deep:3860,:3886,:3906,:3922,:3957)'
  write(*, '(A)') '#   columns: kratio  alpha  beta  k**(a-1)  (1-k)**(b-1)  product'
  do i = 1, np
    x = min(0.999999, probe(i))
    if (x >= 1.0) cycle
    do j = 1, ng
      a = gprobe(j)
      b = 1.3
      write(*, '(A,6(1X,A))') 'pbeta', h(x), h(a), h(b), &
        h(x ** (a - 1.0)), h((1.0 - x) ** (b - 1.0)), &
        h(x ** (a - 1.0) * (1.0 - x) ** (b - 1.0))
    end do
  end do

  ! ---- gamma: tgammaf ------------------------------------------------------
  write(*, '(A)') '#'
  write(*, '(A)') '# form: gamma(alpha+beta)/(gamma(alpha)*gamma(beta))'
  write(*, '(A)') '#       (deep:3854,:3882,:3918,:3952)'
  write(*, '(A)') '#   columns: alpha  beta  gamma(a)  gamma(b)  gamma(a+b)  fzu'
  do i = 1, ng
    a = gprobe(i)
    do j = 1, ng
      b = gprobe(j)
      write(*, '(A,6(1X,A))') 'pgamma', h(a), h(b), &
        h(gamma(a)), h(gamma(b)), h(gamma(a + b)), &
        h(gamma(a + b) / (gamma(a) * gamma(b)))
    end do
  end do

  ! ---- what tgammaf is NOT ------------------------------------------------
  ! Phase 2 measured glibc's tgammaf against three models and none of them
  ! reproduce it: over the 51 distinct arguments the pgamma table above
  ! reaches, float32(tgamma_float64(x)) misses 31, expf(lgammaf(x)) misses 39,
  ! and the exp-lgamma-times-product-recurrence shape glibc's own
  ! e_gammaf_r.c uses misses 32.  All misses are 1-2 ULP.  The tell is at
  ! integer arguments, where gamma is exactly representable: gamma(3) = 2 but
  ! tgammaf returns 40000001, gamma(4) = 6 but tgammaf returns 40C00001.
  ! Something in glibc's path is systematically a hair high.
  !
  ! These rows print the decomposition so a later phase can close it without
  ! re-measuring: for each reachable argument, lgammaf, expf(lgammaf), the
  ! product recurrence's split point, and tgammaf itself.
  write(*, '(A)') '#'
  write(*, '(A)') '# form: tgammaf decomposition (evidence, not an identity)'
  write(*, '(A)') '#   columns: x  gamma(x)  log_gamma(x)  exp(log_gamma(x))' // &
                  '  x_adj  n'
  do i = 1, ng
    a = gprobe(i)
    b = a
    j = 0
    if (a > 1.5) then
      j = int(ceiling(a - 1.5))
      b = a - real(j)
    end if
    write(*, '(A,5(1X,A),1X,I0)') 'plgamma', h(a), h(gamma(a)), &
      h(log_gamma(a)), h(exp(log_gamma(a))), h(b), j
  end do

  ! ---- powf is not correctly rounded, and it matters ----------------------
  ! The float32 reference models powf as a correctly-rounded power.  glibc's
  ! is a double-precision exp2(y*log2(x)) with ~0.82 ULP of worst-case error,
  ! so on an argument whose true value sits within a whisker of a float32
  ! rounding boundary the two disagree.  This is the pair the GF fixture
  ! found, at level 17 of one column, in (1-kratio)**(beta-1.0):
  !
  !   x = 3F0D923B, y = 3E999998
  !   true value  0.83718320727036911868...  (80 significant digits)
  !   midpoint    0.83718320727348327636...
  !   so correctly rounded is 3F5651A3, by 3.1e-12 -- 5.2e-5 of a ULP.
  !   glibc returns 3F5651A4.
  !
  ! The rows below are glibc's answer on this toolchain, so the divergence is
  ! recorded rather than argued.
  !
  ! The operands MUST arrive through variables.  A `transfer(int(z'...'),1.0)
  ! ** transfer(...)` written inline is a constant expression and gfortran's
  ! front end folds it at MPFR precision -- which prints the CORRECTLY ROUNDED
  ! answer and hides the very thing being measured.  This is the same trap the
  ! GFS-constants section below records for `-con_g*rho*w`, and it caught this
  ! probe once already.
  write(*, '(A)') '#'
  write(*, '(A)') '# form: powf near a rounding boundary (deep:3860,:3886,:3922,:3957)'
  write(*, '(A)') '#   columns: x  y  powf(x,y)   [operands via variables, NOT folded]'
  do i = 1, 2
    if (i == 1) then
      x = transfer(int(z'3F0D923B'), 1.0)
      y = transfer(int(z'3E999998'), 1.0)
    else
      x = transfer(int(z'3EE4DB8A'), 1.0)
      y = transfer(int(z'402CCCC7'), 1.0)
    end if
    a = x ** y
    write(*, '(A,3(1X,A))') 'ppowhard', h(x), h(y), h(a)
  end do

  ! ---- satvap's Goff-Gratch forms -----------------------------------------
  ! satvap IS live: module_cu_gf_deep.F:2213 calls it for every (i,k) with
  ! ierr == 0, so cup_env's qes comes through these forms and not through the
  ! commented-out Tetens block above it.
  write(*, '(A)') '#'
  write(*, '(A)') '# form: satvap Goff-Gratch (deep:3646-3667)'
  write(*, '(A)') '#   columns: t  10**y  exp(y*log(10))  log(t)/log(10.)  log10(t)'
  do i = 1, np
    x = 200.0 + 10.0 * real(i)
    y = -3.0 + 0.5 * real(i)
    write(*, '(A,5(1X,A))') 'psat', h(x), h(10 ** y), &
      h(exp(y * log(10.0))), h(log(x) / log(10.0)), h(log10(x))
  end do

  ! ---- the GFS constants, as stored --------------------------------------
  ! module_gfs_physcons.F declares these real(kind_phys) = real(8) and
  ! initialises them from default-real literals -- `con_g = 9.80665e+0`, not
  ! `9.80665d+0`.  Assigned to a real(8) and written out, con_g reads
  ! 40239D0140000000 = float64(float32(9.80665)), NOT the honest
  ! 40239D013A92A305.
  !
  ! That stored word is NOT what a port should build against.  Measured on
  ! the full fixture -- 8640 lanes of `omeg` and `dhdt` from
  ! gf-stage-levels.csv -- GFDRV's arithmetic reproduces exactly when the
  ! constants are the honest doubles and misses 1488 (`omeg`) and 204
  ! (`dhdt`) lanes by exactly 1 ULP when they are the float32-widened ones.
  ! The expression behaviour and the stored word disagree, and the
  ! expression is what matters.
  !
  ! There is deliberately no micro-probe of the expression here.  A probe
  ! that computes `-con_g*rho*w` on local operands gets constant-folded by
  ! gfortran's front end at MPFR precision and answers a different question
  ! than the runtime array arithmetic GFDRV does -- it disagrees with the
  ! oracle by 1 ULP on exactly the lane that discriminates.  The fixture is
  ! the instrument; a two-line probe is not.
  write(*, '(A)') '#'
  write(*, '(A)') '# GFS physcons: the stored real(8) word.  NOTE: the port'
  write(*, '(A)') '# must use float64(v), not this -- see the header comment.'
  write(*, '(A)') '#   columns: name  real8_hex  float64(float32(v))_hex  ' // &
                  'float64(v)_hex'
  call d_probe('con_g   ', con_g, 9.80665d0)
  call d_probe('con_cp  ', con_cp, 1004.6d0)
  call d_probe('con_hvap', con_hvap, 2.5d6)
  call d_probe('con_rv  ', con_rv, 461.50d0)

  ! ---- the constants the port must reproduce exactly -----------------------
  write(*, '(A)') '#'
  write(*, '(A)') '# literals'
  write(*, '(A,1X,A)') 'lit_0.3333', h(0.3333)
  write(*, '(A,1X,A)') 'lit_third', h(1.0 / 3.0)
  write(*, '(A,1X,A)') 'lit_1.2', h(1.2)
  write(*, '(A,1X,A)') 'lit_3.14', h(3.14)
  write(*, '(A,1X,A)') 'lit_frh_thresh', h(0.9)
  write(*, '(A,1X,A)') 'lit_sig_thresh', h((1.0 - 0.9) ** 2)
  write(*, '(A,1X,A)') 'lit_entr_rate', h(7.0e-5)
  write(*, '(A,1X,A)') 'lit_radius', h(0.2 / 7.0e-5)
  write(*, '(A,1X,A)') 'lit_betajb', h(1.5)
  write(*, '(A,1X,A)') 'lit_beta_up', h(1.3)
  ! the two constant sets that coexist in one call
  write(*, '(A,1X,A)') 'gfs_con_g', h(9.80665e+0)
  write(*, '(A,1X,A)') 'gfs_con_cp', h(1.0046e+3)
  write(*, '(A,1X,A)') 'gfs_con_rv', h(4.6150e+2)
  write(*, '(A,1X,A)') 'gfs_con_hvap', h(2.5000e+6)
  write(*, '(A,1X,A)') 'deep_g', h(9.81)
  write(*, '(A,1X,A)') 'deep_cp', h(1004.)
  write(*, '(A,1X,A)') 'deep_r_v', h(461.)
  write(*, '(A,1X,A)') 'deep_xlv', h(2.5e6)

contains

  ! One GFS constant at its declared real(8) width, beside the two spellings
  ! a port might reach for.
  subroutine d_probe(name, stored, honest)
    character(len=*), intent(in) :: name
    real(kind=kind_phys), intent(in) :: stored, honest
    real(kind=kind_phys) :: widened
    widened = real(real(honest, 4), kind_phys)
    write(*, '(A,3(1X,A))') 'gfsconst_' // trim(name), &
      hd(stored), hd(widened), hd(honest)
  end subroutine d_probe

  character(len=16) function hd(v)
    real(kind=kind_phys), intent(in) :: v
    write(hd, '(Z16.16)') transfer(v, 0_8)
  end function hd

  character(len=8) function h(v)
    real, intent(in) :: v
    write(h, '(Z8.8)') transfer(v, 0)
  end function h

end program gf_pow_probe
