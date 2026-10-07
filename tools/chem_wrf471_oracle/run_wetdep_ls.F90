program run_wetdep_ls
 use oracle_io
 use module_wetdep_oracle
 implicit none
 integer, parameter :: nc=12,nr=19
 integer :: nz,nn,k,i,r,step
 real, allocatable :: var(:,:,:,:),initial(:,:,:,:),before(:,:,:,:),moist(:,:,:,:),rho(:,:,:),dz(:,:,:),w(:,:,:)
 real :: removed(nc,1,nr)
 real :: rain(nc,1),dt
 character(1024) :: root
 character(32) :: name
 call oracle_root(root)
 dt=36.
 do nn=1,2
  nz=49+10*(nn-1)
  allocate(var(nc,nz,1,nr),initial(nc,nz,1,nr),before(nc,nz,1,nr),moist(nc,nz,1,2),rho(nc,nz,1),dz(nc,nz,1),w(nc,nz,1))
  do k=1,nz
   do i=1,nc
    rho(i,k,1)=1.2-real(k)*.01
    dz(i,k,1)=50.+real(k)
    w(i,k,1)=.2+real(k)*.03
    moist(i,k,1,1)=.01
    moist(i,k,1,2)=.001
    do r=1,nr
     var(i,k,1,r)=real(r)*.1+real(k)*.03
    enddo
   enddo
  enddo
  rain(:,1)=(/0.,1.e-8,20.,1.,.01,1.e-8,100.,1.,1.,1.,1.,1./)
  moist(4,:,1,2)=0.
  moist(5,:,1,2)=0.; moist(5,3,1,2)=.001
  moist(6,:,1,2)=1.
  moist(7,:,1,2)=1.e-8
  var(8,:,:,:)=1.e-16
  var(9,:,:,:)=1.e-17
  w(10,:,1)=-1.
  w(11,1:10,1)=-1.
  var(12,1:10,:,:)=1.e-16
  initial=var
  write(name,'(A,I0)') 'nz',nz
  call oracle_open(trim(name))
  call oracle_put('initial',initial)
  call oracle_put('qc',moist(:,:,:,2))
  call oracle_put('rho',rho); call oracle_put('dz',dz); call oracle_put('w',w)
  call oracle_put('rain',rain); call oracle_put('dt',dt)
  do step=1,5
   before=var
   call wetdep_ls(dt,var,rain,moist,rho,2,nr,4,dz,w,300, &
       1,nc+1,1,2,1,nz+1,1,nc,1,1,1,nz,1,nc,1,1,1,nz)
   write(name,'(A,I0)') 'step',step
   call oracle_put(trim(name),var)
   removed=0.
   do r=1,nr
    do k=1,nz-2
     do i=1,nc
      removed(i,1,r)=removed(i,1,r)+(before(i,k,1,r)-var(i,k,1,r))*rho(i,k,1)*dz(i,k,1)
     enddo
    enddo
   enddo
   write(name,'(A,I0)') 'removed',step
   call oracle_put(trim(name),removed)
  enddo
  call oracle_close()
  deallocate(var,initial,before,moist,rho,dz,w)
 enddo
end program
