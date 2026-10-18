  real :: a
  real, parameter :: vr3mm_p = 5.23599e-10*(3.0/1.)**3
  call probe_init()
  a = 0.523599*(1.5e-3)**3; call pr("vshd15", a)
  a = 0.523599*(3.0e-3)**3; call pr("vshd30", a)
  a = (40.e-6)**3; call pr("cube40", a)
  call pr("vr3mm", vr3mm_p)
  call pr("rzs", rzs)
  a = 6./pi; call pr("6/pi", a)
  a = (0.8-0.05)/(40.e-6); call pr("eswslope", a)
  a = 4.*piinv/3.; call pr("4pii3", a)
