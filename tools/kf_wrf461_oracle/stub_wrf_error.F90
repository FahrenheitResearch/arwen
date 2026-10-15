! Minimal stand-in for WRF's frame/module_wrf_error.F, the only module
! phys/module_cu_kfeta.F USEs.  KF calls exactly two of its procedures:
! wrf_message (diagnostic prints) and wrf_error_fatal (the QG<0-at-the-
! surface guard, module_cu_kfeta.F:2033, and the mass-budget stop, :2498).
!
! wrf_error_fatal does NOT stop here.  It records the event so the driver can
! flag the column and keep going; the KF source continues past both call
! sites with defined values, so a column that hits one is still a complete,
! deterministic run of the Fortran -- it is simply a column real WRF would
! have aborted on.  The oracle reports such columns separately and never
! counts them as matches.
MODULE module_wrf_error
  IMPLICIT NONE
  INTEGER :: wrf_debug_level = 0
  LOGICAL :: kf_oracle_fatal = .FALSE.
  INTEGER :: kf_oracle_messages = 0
CONTAINS
  SUBROUTINE wrf_message(str)
    CHARACTER(LEN=*), INTENT(IN) :: str
    kf_oracle_messages = kf_oracle_messages + 1
  END SUBROUTINE wrf_message

  SUBROUTINE wrf_error_fatal(str)
    CHARACTER(LEN=*), INTENT(IN) :: str
    kf_oracle_fatal = .TRUE.
    WRITE(0, '(A,A)') 'KF ORACLE: wrf_error_fatal: ', TRIM(str)
  END SUBROUTINE wrf_error_fatal
END MODULE module_wrf_error
