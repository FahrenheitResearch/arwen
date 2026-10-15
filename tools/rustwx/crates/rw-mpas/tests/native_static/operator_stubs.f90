module mpas_constants
end module
module mpas_abort
contains
 subroutine mpas_dmpar_global_abort()
 stop 1
 end subroutine
end module
module mpas_pool_routines
 use mpas_derived_types
 implicit none
 real,allocatable,target::xCell(:),yCell(:),zCell(:),xVertex(:),yVertex(:),zVertex(:),angleEdge(:),dcEdge(:)
 real,allocatable,target::gradient_x(:,:),gradient_y(:,:),deriv(:,:,:)
 integer,allocatable,target::adv(:,:),counts(:),coc(:,:),eoc(:,:),voc(:,:),voe(:,:),coe(:,:)
 interface mpas_pool_get_array
  module procedure real1,real2,real3,int1,int2
 end interface
contains
 subroutine real1(pool,key,p)
 type(mpas_pool_type)::pool
 character(*)::key
 real,pointer::p(:)
 select case(key)
 case('xCell');p=>xCell
 case('yCell');p=>yCell
 case('zCell');p=>zCell
 case('xVertex');p=>xVertex
 case('yVertex');p=>yVertex
 case('zVertex');p=>zVertex
 case('angleEdge');p=>angleEdge
 case('dcEdge');p=>dcEdge
 case default;stop 11
 end select
 end subroutine
 subroutine real2(pool,key,p)
 type(mpas_pool_type)::pool
 character(*)::key
 real,pointer::p(:,:)
 select case(key)
 case('cell_gradient_coef_x');p=>gradient_x
 case('cell_gradient_coef_y');p=>gradient_y
 case default;stop 12
 end select
 end subroutine
 subroutine real3(pool,key,p)
 type(mpas_pool_type)::pool
 character(*)::key
 real,pointer::p(:,:,:)
 p=>deriv
 end subroutine
 subroutine int1(pool,key,p)
 type(mpas_pool_type)::pool
 character(*)::key
 integer,pointer::p(:)
 p=>counts
 end subroutine
 subroutine int2(pool,key,p)
 type(mpas_pool_type)::pool
 character(*)::key
 integer,pointer::p(:,:)
 select case(key)
 case('advCells');p=>adv
 case('cellsOnCell');p=>coc
 case('edgesOnCell');p=>eoc
 case('verticesOnCell');p=>voc
 case('verticesOnEdge');p=>voe
 case('cellsOnEdge');p=>coe
 case default;stop 13
 end select
 end subroutine
end module
