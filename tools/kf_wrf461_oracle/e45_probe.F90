! PROF5 (module_cu_kfeta.F:2957) evaluates E45=EXP(-4.5): a constant
! argument, which gfortran simplifies at compile time.  This prints the
! folded word beside glibc's runtime expf of the same argument, so the port
! uses the word the Fortran actually ran with.
PROGRAM e45_probe
  IMPLICIT NONE
  REAL :: x, folded, runtime
  folded = EXP(-4.5)
  x = -4.5
  CALL opaque(x)
  runtime = EXP(x)
  WRITE(*,'(A,Z8.8)') 'folded  EXP(-4.5) = 0x', TRANSFER(folded, 0)
  WRITE(*,'(A,Z8.8)') 'runtime expf(-4.5) = 0x', TRANSFER(runtime, 0)
CONTAINS
  SUBROUTINE opaque(v)
    REAL, INTENT(INOUT) :: v
    v = v * 1.0
  END SUBROUTINE opaque
END PROGRAM e45_probe
