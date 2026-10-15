! The accumulator routines and final mean blocks are copied from pinned MPAS.
module native_counts
 use iso_fortran_env,only:int64,real32,real64
 implicit none
 integer,parameter::I8KIND=int64,RKIND=real32,R8KIND=real64
 integer,allocatable::nhs(:),landmask(:)
 integer,allocatable::ncat(:,:)
 integer::category_min=1,category_max=4
 integer(kind=I8KIND),allocatable::ter_integer(:),soilcomp_int(:,:)
 real(kind=RKIND)::soilcomp_msgval=-9999.0
contains
