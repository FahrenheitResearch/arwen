! Framework logging stub only; complete container sources remain pristine.
module mpas_kind_types
 integer,parameter::RKIND=kind(1.0)
end module
module mpas_derived_types
 integer,parameter::MPAS_LOG_WARN=1,MPAS_LOG_CRIT=2,MPAS_LOG_ERR=3
end module
module mpas_log
 use iso_fortran_env,only:error_unit
contains
 subroutine mpas_log_write(message,messageType)
 character(*),intent(in)::message
 integer,optional,intent(in)::messageType
 write(error_unit,'(a)')message
 end subroutine
end module
