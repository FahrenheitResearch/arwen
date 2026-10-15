module mpas_kind_types
 integer, parameter :: RKIND=kind(1.0), R8KIND=kind(1.d0)
end module
module mpas_derived_types
 integer, parameter :: MPAS_LOG_WARN=1, MPAS_LOG_CRIT=2, MPAS_LOG_ERR=3
 type mpas_pool_type
  integer::unused
 end type
end module
module mpas_log
contains
 subroutine mpas_log_write(message,messageType,intArgs,realArgs)
 character(*),intent(in)::message
 integer,optional,intent(in)::messageType
 integer,optional,intent(in)::intArgs(:)
 real,optional,intent(in)::realArgs(:)
 write(*,*) message
 end subroutine
end module
