! Framework-only fixtures. Mathematical routines are extracted unchanged.
module native_geometry
 use mpas_kind_types
 implicit none
 real,parameter::pii=3.141592653589793
 type mpas_geotile_mgr_type
  integer::pool
 end type
 type mpas_geotile_type
  integer::x,y
 end type
 integer,target::border=3
 real,target::origin_lat,origin_lon,step_x,step_y
 interface mpas_pool_get_config
  module procedure get_int,get_real
 end interface
contains
 subroutine get_int(pool,key,p)
 integer::pool
 character(*)::key
 integer,pointer::p
 p=>border
 end subroutine
 subroutine get_real(pool,key,p)
 integer::pool
 character(*)::key
 real,pointer::p
 select case(key)
 case('known_lat')
 p=>origin_lat
 case('known_lon')
 p=>origin_lon
 case('dx')
 p=>step_x
 case('dy')
 p=>step_y
 end select
 end subroutine
