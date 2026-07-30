! Dependency stubs for driving share/module_soil_pre.F byte-unmodified.
!
! `nm -u share/module_soil_pre.o` in a full WRF v4.6.1 build lists exactly
! seven WRF-side undefined symbols beyond libm/libgfortran:
!
!   __module_date_time_MOD_current_date   __module_date_time_MOD_start_date
!   nl_get_aggregate_lu_                  nl_get_mminlu_
!   wrf_debug_      wrf_error_fatal3_     wrf_message_
!
! and one module USE that is resolved at compile time only,
! module_state_description.  That module is NOT stubbed here: the real,
! registry-generated frame/module_state_description.o from the pinned tree is
! compiled and linked, because a stub would be free to give SLABSCHEME or
! RUCLSMSCHEME the wrong integer and process_soil_real branches on those.
!
! module_date_time IS stubbed, down to the two CHARACTER variables the module
! actually references (module_soil_pre.F:1547, inside adjust_soil_temp_new's
! diagnostic write).  The real module_date_time drags in ESMF; these two are
! data, not behaviour, and nothing on the init_soil_depth_3 /
! init_soil_3_real path reads them.

module module_date_time
  implicit none
  character(len=24) :: current_date = '0000-00-00_00:00:00     '
  character(len=24) :: start_date   = '0000-00-00_00:00:00     '
end module module_date_time

! wrf_message / wrf_debug print; init_soil_3_real calls wrf_message on every
! branch it takes, so silencing them would hide which arm ran.  They go to
! stderr so the CSV on stdout stays clean.
subroutine wrf_message(message)
  implicit none
  character(len=*), intent(in) :: message
  write(0, '(A)') trim(message)
end subroutine wrf_message

subroutine wrf_debug(level, message)
  implicit none
  integer, intent(in) :: level
  character(len=*), intent(in) :: message
  write(0, '(A,I0,A,A)') 'debug(', level, '): ', trim(message)
end subroutine wrf_debug

! WRF's wrf_error_fatal is a macro onto wrf_error_fatal3 ( file, line, str ).
! Aborting is the correct behaviour: every call site in module_soil_pre.F that
! reaches it is a geometry the harness must not have asked for.
subroutine wrf_error_fatal3(file_str, line, str)
  implicit none
  character(len=*), intent(in) :: file_str
  integer, intent(in) :: line
  character(len=*), intent(in) :: str
  write(0, '(A,A,I0,A,A)') 'FATAL ', trim(file_str), line, ': ', trim(str)
  error stop 1
end subroutine wrf_error_fatal3

! Namelist accessors, reached only from process_percent_cat_new /
! aggregate_categories, never from the soil-depth path.
subroutine nl_get_mminlu(id_id, mminlu)
  implicit none
  integer, intent(in) :: id_id
  character(len=*), intent(out) :: mminlu
  mminlu = 'MODIFIED_IGBP_MODIS_NOAH'
end subroutine nl_get_mminlu

subroutine nl_get_aggregate_lu(id_id, aggregate_lu)
  implicit none
  integer, intent(in) :: id_id
  integer, intent(out) :: aggregate_lu
  aggregate_lu = 0
end subroutine nl_get_aggregate_lu
