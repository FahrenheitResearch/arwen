module mpas_kind_types
 integer,parameter::RKIND=kind(1.0),StrKIND=512
end module
module mpas_derived_types
 use mpas_kind_types
 integer,parameter::MPAS_LOG_ERR=1,MPAS_LOG_WARN=2
 type mpas_pool_type
 integer::unused
 end type
end module
module mpas_log
contains
 subroutine mpas_log_write(message,messageType,intArgs)
 character(*)::message
 integer,optional::messageType,intArgs(:)
 end subroutine
end module
module mpas_io_units
contains
 subroutine mpas_new_unit(u)
 integer::u
 u=89
 end subroutine
 subroutine mpas_release_unit(u)
 integer::u
 end subroutine
end module
module mpas_pool_routines
 use mpas_kind_types
 use mpas_derived_types
 implicit none
 integer::count=0
 character(512)::names(128),chars(128)
 character::kinds(128)
 integer::ints(128)
 real::reals(128)
 interface mpas_pool_add_config
 module procedure add_char,add_real,add_int
 end interface
contains
 subroutine add_char(pool,key,value)
 type(mpas_pool_type)::pool
 character(*)::key,value
 count=count+1;names(count)=key;kinds(count)='C';chars(count)=value
 end subroutine
 subroutine add_real(pool,key,value)
 type(mpas_pool_type)::pool
 character(*)::key
 real::value
 count=count+1;names(count)=key;kinds(count)='R';reals(count)=value
 end subroutine
 subroutine add_int(pool,key,value)
 type(mpas_pool_type)::pool
 character(*)::key
 integer::value
 count=count+1;names(count)=key;kinds(count)='I';ints(count)=value
 end subroutine
end module
