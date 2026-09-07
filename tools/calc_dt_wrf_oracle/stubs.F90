! Minimal WRF framework stubs: libesmf_time.a calls these on error paths
! only.  A real trip here means the oracle asked something invalid, so
! they abort loudly rather than returning.
      SUBROUTINE wrf_error_fatal( str )
        CHARACTER(LEN=*), INTENT(IN) :: str
        WRITE(0,*) 'wrf_error_fatal: ', TRIM(str)
        STOP 1
      END SUBROUTINE wrf_error_fatal

      SUBROUTINE wrf_error_fatal3( file, line, str )
        CHARACTER(LEN=*), INTENT(IN) :: file, str
        INTEGER, INTENT(IN) :: line
        WRITE(0,*) 'wrf_error_fatal3: ', TRIM(file), line, TRIM(str)
        STOP 1
      END SUBROUTINE wrf_error_fatal3

      SUBROUTINE wrf_message( str )
        CHARACTER(LEN=*), INTENT(IN) :: str
        WRITE(0,*) TRIM(str)
      END SUBROUTINE wrf_message

      SUBROUTINE wrf_debug( level, str )
        INTEGER, INTENT(IN) :: level
        CHARACTER(LEN=*), INTENT(IN) :: str
      END SUBROUTINE wrf_debug
