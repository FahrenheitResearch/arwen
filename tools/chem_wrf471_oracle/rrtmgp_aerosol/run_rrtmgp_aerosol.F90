program run_rrtmgp_aerosol
 use mo_rte_kind, only: wp, wl
 use mo_rte_config, only: rte_config_checks
 use mo_aerosol_optics_rrtmgp_merra
 use mo_optical_props, only: ty_optical_props_1scl, ty_optical_props_2str
 use netcdf
 use oracle_io
 implicit none
 type(ty_aerosol_optics_rrtmgp_merra) :: lut
 type(ty_optical_props_2str) :: op, mix, component
 type(ty_optical_props_1scl) :: absorption, mixabs
 integer, parameter :: nc=25,nz=8
 integer :: n,nb,nrh,c,k,b,t,bin,r,mode,u
 integer :: types(nc,nz), rowtypes(3), rowbins(3)
 real(wp) :: sizes(nc,nz),mass(nc,nz),rh(nc,nz),q(nc,nz,3),dp(nc,nz),scales(3)
 real(wp), allocatable :: bands(:,:),lims(:,:),rhs(:),dust(:,:,:),salt(:,:,:,:),sulf(:,:,:)
 real(wp), allocatable :: bc(:,:),bcr(:,:,:),oc(:,:),ocr(:,:,:)
 real(wp), parameter :: rhvals(nz)=[0.1_wp,0.5_wp,0.8_wp,0.95_wp,0.99_wp,1._wp,0._wp,0.5_wp]
 character(len=1024) :: root,table_root
 character(len=2) :: kind
 character(len=128) :: err
 call oracle_root(root)
 call get_command_argument(2,table_root)
 call rte_config_checks(.true._wl,.true._wl)
 do mode=1,2
  if(mode==1) then
   kind='sw'
  else
   kind='lw'
  endif
  call check(nf90_open(trim(table_root)//'/rrtmgp-aerosols-merra-'//kind//'.nc',nf90_nowrite,n))
  call dimension('nband',nb); call dimension('nrh',nrh)
  allocate(bands(2,nb),lims(2,5),rhs(nrh),dust(3,5,nb),salt(3,nrh,5,nb),sulf(3,nrh,nb), &
           bc(3,nb),bcr(3,nrh,nb),oc(3,nb),ocr(3,nrh,nb))
  call read2('bnd_limits_wavenumber',bands); call read2('merra_aero_bin_lims',lims)
  call check(nf90_inq_varid(n,'aero_rh',u)); call check(nf90_get_var(n,u,rhs))
  call read3('aero_dust_tbl',dust); call read3('aero_sulf_tbl',sulf)
  call check(nf90_inq_varid(n,'aero_salt_tbl',u)); call check(nf90_get_var(n,u,salt))
  call read2('aero_bcar_tbl',bc); call read3('aero_bcar_rh_tbl',bcr)
  call read2('aero_ocar_tbl',oc); call read3('aero_ocar_rh_tbl',ocr)
  call check(nf90_close(n))
  call die(lut%load(bands,lims,rhs,dust,salt,sulf,bc,bcr,oc,ocr))
  call die(op%alloc_2str(nc,nz,bands)); call die(absorption%alloc_1scl(nc,nz,bands))
  c=0
  do t=1,7
   do bin=1,5
    if(t>2.and.bin>1) cycle
    c=c+1
    types(c,:)=t; sizes(c,:)=(lims(1,bin)+lims(2,bin))/2._wp
   enddo
  enddo
  ! Both mineral types on all shared bin edges, including endpoints.
  do t=1,2
   do bin=1,5
    c=c+1; types(c,:)=t; sizes(c,:)=lims(1,bin)
   enddo
  enddo
  if(c/=nc) error stop 'case column count'
  do k=1,nz
   rh(:,k)=rhvals(k); mass(:,k)=real(1.e-5*(nz-k),wp)
  enddo
  mass(:,7)=1.e-12_wp
  ! Inputs are explicitly rounded to engine precision for both builds.
  mass=real(real(mass,4),wp); rh=real(real(rh,4),wp); sizes=real(real(sizes,4),wp)
  call die(lut%aerosol_optics(types,sizes,mass,rh,op))
  call die(lut%aerosol_optics(types,sizes,mass,rh,absorption))
  call oracle_open(kind//'_types')
  call oracle_put('types',types); call oracle_put('sizes',real(sizes,4))
  call oracle_put('mass',real(mass,4)); call oracle_put('rh',real(rh,4))
  call write_outputs(op,absorption)
  call oracle_close()
  ! Columns 1..12 are dust storm, 13..25 are marine, with carbon/sulfate.
  call die(mix%alloc_2str(nc,nz,bands)); call die(component%alloc_2str(nc,nz,bands))
  call die(mixabs%alloc_1scl(nc,nz,bands))
  mix%tau=0; mix%ssa=0; mix%g=0; mixabs%tau=0
  dp=real(8000.,wp); scales=[1.e-9_wp,1.e-9_wp,1.e-6_wp*96.0576_wp/28.966_wp]
  scales=real(real(scales,4),wp)
  rowtypes=[1,2,3]; rowbins=[3,4,1]
  do r=1,3
   do k=1,nz
    do c=1,nc
     q(c,k,r)=real(10.*(nz-k),wp)
     if(r==3) q(c,k,r)=real(0.005*(nz-k),wp)
     if((r==1.and.c<=12).or.(r==2.and.c>12)) q(c,k,r)=real(500.*(nz-k),wp)
    enddo
   enddo
   types=rowtypes(r); sizes=(lims(1,rowbins(r))+lims(2,rowbins(r)))/2._wp
   mass=(q(:,:,r)*scales(r))*(dp/real(9.81,wp))
   call die(lut%aerosol_optics(types,sizes,mass,rh,component))
   call die(component%increment(mix))
   call die(lut%aerosol_optics(types,sizes,mass,rh,absorption))
   call die(absorption%increment(mixabs))
  enddo
  call oracle_open(kind//'_mixture')
  call oracle_put('rowtypes',rowtypes); call oracle_put('rowbins',rowbins)
  call oracle_put('scales',real(scales,4)); call oracle_put('q',real(q,4))
  call oracle_put('dp',real(dp,4)); call oracle_put('rh',real(rh,4))
  call write_outputs(mix,mixabs); call oracle_close()
  ! Exercise exact source validation errors, written as a reproducible log.
  types=8; err=lut%aerosol_optics(types,sizes,mass,rh,op); print *,trim(err)
  types=1; sizes=11; err=lut%aerosol_optics(types,sizes,mass,rh,op); print *,trim(err)
  sizes=1; rh=1.01_wp; err=lut%aerosol_optics(types,sizes,mass,rh,op); print *,trim(err)
  call lut%finalize(); call die(op%finalize_2str()); call die(absorption%finalize_1scl())
  call die(mix%finalize_2str()); call die(mixabs%finalize_1scl()); call die(component%finalize_2str())
  deallocate(bands,lims,rhs,dust,salt,sulf,bc,bcr,oc,ocr)
 enddo
contains
 subroutine check(status)
  integer,intent(in)::status
  if(status/=nf90_noerr) then
   print *,nf90_strerror(status); error stop
  endif
 end subroutine
 subroutine die(message)
  character(len=*),intent(in)::message
  if(len_trim(message)>0) then
   print *,trim(message); error stop
  endif
 end subroutine
 subroutine dimension(name,length)
  character(len=*),intent(in)::name
  integer,intent(out)::length
  integer::id
  call check(nf90_inq_dimid(n,name,id)); call check(nf90_inquire_dimension(n,id,len=length))
 end subroutine
 subroutine read2(name,a)
  character(len=*),intent(in)::name
  real(wp),intent(out)::a(:,:)
  call check(nf90_inq_varid(n,name,u)); call check(nf90_get_var(n,u,a))
 end subroutine
 subroutine read3(name,a)
  character(len=*),intent(in)::name
  real(wp),intent(out)::a(:,:,:)
  call check(nf90_inq_varid(n,name,u)); call check(nf90_get_var(n,u,a))
 end subroutine
 subroutine write_outputs(x,y)
  type(ty_optical_props_2str),intent(in)::x
  type(ty_optical_props_1scl),intent(in)::y
  call oracle_put('tau',real(x%tau,4)); call oracle_put('ssa',real(x%ssa,4))
  call oracle_put('g',real(x%g,4)); call oracle_put('absorption',real(y%tau,4))
 end subroutine
end program
