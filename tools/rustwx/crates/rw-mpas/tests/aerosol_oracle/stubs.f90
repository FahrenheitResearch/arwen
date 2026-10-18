! Test-only MPAS framework pools. Numerical routines are compiled pristine.
module mpas_kind_types
 integer, parameter :: RKIND=4, StrKIND=64
end module
module mpas_derived_types
 use mpas_kind_types
 integer, parameter :: MPAS_LOG_ERR=1, MPAS_LOG_CRIT=2, MPAS_LOG_WARN=3
 type dm_info
 integer :: dummy
 end type
 type domain_type
 type(dm_info), pointer :: dminfo
 end type
 type block_type
 type(domain_type), pointer :: domain
 end type
 type mpas_pool_type
 integer, pointer :: nc=>null(), ng=>null(), nv=>null(), nm=>null(), inifa=>null(), inwfa=>null()
 integer, pointer :: land(:)=>null()
 character(len=StrKIND), pointer :: date=>null()
 real(RKIND), pointer :: lat(:)=>null(), lon(:)=>null(), pressure(:,:)=>null()
 real(RKIND), pointer :: nifa(:,:,:)=>null(), nwfa(:,:,:)=>null(), pwif(:,:,:)=>null(), scalars(:,:,:)=>null()
 end type
end module
! Fault injection only at framework boundaries; pristine aerosol/read_met code
! retains its native file-open failure paths (thompson_aerosols.F:257-262,298-303).
module aerosol_file_fault
 contains
 subroutine remove_climatology_at_boundary(boundary)
 character(len=*), intent(in) :: boundary
 character(len=8) :: requested
 integer :: unit,status
 call get_environment_variable('MPAS_AEROSOL_OPEN_FAILURE',requested)
 if(trim(requested)/=boundary) return
 open(newunit=unit,file='QNWFA_QNIFA_SIGMA_MONTHLY.dat',status='old',iostat=status)
 if(status==0) close(unit,status='delete')
 end subroutine
end module
module mpas_log
 use mpas_kind_types
 use aerosol_file_fault
 contains
 subroutine mpas_log_write(message, messageType, realArgs, intArgs)
 character(len=*), intent(in) :: message
 integer, optional :: messageType, intArgs(:)
 real(RKIND), optional :: realArgs(:)
 if(index(message,'Landsea mask not available')==1) call remove_climatology_at_boundary('second')
 if(index(message,'Error opening gocart file')==1) print *,trim(message)
 if(present(messageType)) then
 if(messageType==2) then
 print *, trim(message)
 stop 2
 endif
 endif
 end subroutine
end module
module mpas_dmpar
 use mpas_derived_types
 contains
 subroutine mpas_dmpar_abort(info)
 type(dm_info) :: info
 stop 3
 end subroutine
end module
module mpas_io_units
 contains
 subroutine mpas_new_unit(unit, unformatted)
 integer, intent(out) :: unit
 logical, optional :: unformatted
 integer :: i
 logical :: opened
 do i=20,200
 inquire(unit=i,opened=opened)
 if(.not.opened) then
 unit=i
 return
 endif
 enddo
 stop 4
 end subroutine
 subroutine mpas_release_unit(unit)
 integer :: unit
 end subroutine
end module
module mpas_atmphys_utilities
end module
module mpas_pool_routines
 use mpas_derived_types
 use aerosol_file_fault
 interface mpas_pool_get_array
 module procedure array1, array2, array3, arrayint
 end interface
 contains
 subroutine mpas_pool_get_dimension(pool,name,p)
 type(mpas_pool_type), intent(in) :: pool
 character(len=*), intent(in) :: name
 integer, pointer :: p
 select case(name)
 case('nCells'); p=>pool%nc
 case('nGocartLevels'); p=>pool%ng
 case('nVertLevels'); p=>pool%nv
 case('nMonths'); p=>pool%nm
 case('index_nifa','index_lbc_nifa'); p=>pool%inifa
 case('index_nwfa','index_lbc_nwfa'); p=>pool%inwfa
 case default; stop 5
 end select
 end subroutine
 subroutine mpas_pool_get_config(pool,name,p)
 type(mpas_pool_type), intent(in) :: pool
 character(len=*), intent(in) :: name
 character(len=StrKIND), pointer :: p
 p=>pool%date
 call remove_climatology_at_boundary('first')
 end subroutine
 subroutine array1(pool,name,p)
 type(mpas_pool_type), intent(in) :: pool
 character(len=*), intent(in) :: name
 real(RKIND), pointer :: p(:)
 select case(name)
 case('latCell'); p=>pool%lat
 case('lonCell'); p=>pool%lon
 case default; stop 6
 end select
 end subroutine
 subroutine array2(pool,name,p)
 type(mpas_pool_type), intent(in) :: pool
 character(len=*), intent(in) :: name
 real(RKIND), pointer :: p(:,:)
 p=>pool%pressure
 end subroutine
 subroutine array3(pool,name,p)
 type(mpas_pool_type), intent(in) :: pool
 character(len=*), intent(in) :: name
 real(RKIND), pointer :: p(:,:,:)
 select case(name)
 case('nifa_gocart_clim'); p=>pool%nifa
 case('nwfa_gocart_clim'); p=>pool%nwfa
 case('pwif_gocart_clim'); p=>pool%pwif
 case('scalars','lbc_scalars'); p=>pool%scalars
 case default; stop 7
 end select
 end subroutine
 subroutine arrayint(pool,name,p)
 type(mpas_pool_type), intent(in) :: pool
 character(len=*), intent(in) :: name
 integer, pointer :: p(:)
 p=>pool%land
 end subroutine
end module
