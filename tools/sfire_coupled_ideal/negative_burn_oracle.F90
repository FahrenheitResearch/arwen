! Captured live fuel update through compiled immutable WRF v4.7.1 fuel_left.
program negative_burn_oracle
 use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
 use module_fr_fire_core
 use module_fr_fire_util
 implicit none
 integer,parameter:: n=480,h=n+1
 integer:: i,j,ndiff,negative,ncorrect
 real:: old(0:h,0:h),lfn(0:h,0:h),tign(0:h,0:h),tau(0:h,0:h)
 real:: native(0:h,0:h),gpu(0:h,0:h),area(0:h,0:h),burnt(0:h,0:h),gpu_burnt(0:h,0:h)
 real:: fixed(0:h,0:h),fixed_burnt(0:h,0:h)
 real:: eold(8),efuture(8),efixed(8),eburn(8),nan,pinf,ninf
 character(4096):: root
 call get_command_argument(1,root)
 call read_array('before_fuel_frac',old)
 call read_array('after_lfn',lfn)
 call read_array('after_tign',tign)
 call read_array('after_fuel_time',tau)
 call read_array('after_fuel_frac',gpu)
 call read_array('after_burnt_area_dt',gpu_burnt)
 fire_print_msg=0;fire_print_file=0;fuel_left_method=1
 native=0.;area=0.;burnt=0.;fixed=old;fixed_burnt=0.
 call fuel_left(0,h,0,h,1,n,1,n,0,h,0,h,lfn,tign,tau,1177.5,native,area)
 ndiff=0;negative=0;ncorrect=0
 do j=1,n
  do i=1,n
   if(transfer(native(i,j),0)/=transfer(gpu(i,j),0))ndiff=ndiff+1
   burnt(i,j)=old(i,j)-native(i,j)
   if(burnt(i,j)<0.)negative=negative+1
   ! Default correction: consumed fuel cannot return to its initial state.
   fixed(i,j)=native(i,j)
   if(ieee_is_finite(fixed(i,j)).and.fixed(i,j)>old(i,j))fixed(i,j)=old(i,j)
   fixed_burnt(i,j)=old(i,j)-fixed(i,j)
   if(transfer(fixed(i,j),0)/=transfer(native(i,j),0))ncorrect=ncorrect+1
  enddo
 enddo
 print *, 'native_fuel_different_words',ndiff
 print *, 'native_burn_different_words',count(transfer(burnt(1:n,1:n),[0],n*n)/=&
                                                    transfer(gpu_burnt(1:n,1:n),[0],n*n))
 print *, 'native_negative_consumption_cells',negative
 print *, 'native_negative_min',minval(burnt(1:n,1:n))
 print *, 'old_fuel_bad_cell',old(283,237),transfer(old(283,237),0)
 print *, 'native_fuel_bad_cell',native(283,237),transfer(native(283,237),0)
 print *, 'corrected_fuel_bad_cell',fixed(283,237),transfer(fixed(283,237),0)
 print *, 'corrected_fuel_changed_words',ncorrect
 print *, 'corrected_negative_consumption_cells',count(fixed_burnt(1:n,1:n)<0.)
 call write_array('native_fuel',native)
 call write_array('native_burnt',burnt)
 call write_array('corrected_fuel',fixed)
 call write_array('corrected_burnt',fixed_burnt)
 nan=transfer(int(z'7fc01234'),nan)
 pinf=transfer(int(z'7f800000'),pinf)
 ninf=transfer(int(z'ff800000'),ninf)
 eold=[.5,.5,.5,0.,-0.,nan,.5,.2]
 efuture=[nan,pinf,ninf,-0.,0.,.5,.6,.1]
 efixed=efuture
 do i=1,8
  if(ieee_is_finite(efixed(i)).and.efixed(i)>eold(i))efixed(i)=eold(i)
 enddo
 eburn=eold-efixed
 open(17,file=trim(root)//'/edge.bin',access='stream',form='unformatted',status='replace')
 write(17)eold,efuture,efixed,eburn
 close(17)
 if(ndiff/=0.or.negative/=1.or.ncorrect/=1.or.any(fixed_burnt<0.))stop 1
contains
 subroutine read_array(name,a)
  character(*),intent(in)::name
  real,intent(out)::a(0:h,0:h)
  open(17,file=trim(root)//'/native-input/'//name//'.bin',access='stream',form='unformatted',status='old')
  read(17)a
  close(17)
 end subroutine
 subroutine write_array(name,a)
  character(*),intent(in)::name
  real,intent(in)::a(0:h,0:h)
  open(17,file=trim(root)//'/'//name//'.bin',access='stream',form='unformatted',status='replace')
  write(17)a
  close(17)
 end subroutine
end program
