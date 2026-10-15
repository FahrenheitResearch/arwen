program run_lightning
 use module_lightning_driver
 use module_utility
 implicit none
 integer :: nx,ny,nz,opt,method,cell,month,nsteps,s,k
 integer :: u,o
 character(1024) :: infile,outfile
 real :: dt,dx,dy,interval,start,factor,num,den,adjust,activation
 real(8),allocatable :: times(:)
 real,allocatable :: lat(:,:),lon(:,:),land(:,:),ht(:,:),t(:,:,:),z(:,:,:),w(:,:,:),refl(:,:,:)
 real,allocatable :: dz(:,:,:),theta(:,:,:),pi(:,:,:),p(:,:,:),rho(:,:,:),moist(:,:,:,:),inputnum(:,:,:),inputden(:,:,:)
 real,allocatable :: ic(:,:),cg(:,:),icc(:,:),cgc(:,:),lpi(:,:)
 integer,allocatable :: top(:,:)
 type(WRFU_Time) :: clock
 call get_command_argument(1,infile)
 call get_command_argument(2,outfile)
 open(newunit=u,file=trim(infile),access='stream',form='unformatted',status='old')
 read(u) nx,ny,nz,opt,method,cell,month,nsteps
 read(u) dt,dx,dy,interval,start,factor,num,den,adjust
 allocate(times(nsteps));read(u) times
 allocate(lat(nx,ny),lon(nx,ny),land(nx,ny),ht(nx,ny),top(nx,ny))
 allocate(t(nx,nz,ny),z(nx,nz,ny),w(nx,nz,ny),refl(nx,nz,ny),dz(nx,nz,ny),theta(nx,nz,ny),pi(nx,nz,ny))
 allocate(p(nx,nz,ny),rho(nx,nz,ny),moist(nx,nz,ny,6),inputnum(nx,ny,12),inputden(nx,ny,12))
 allocate(ic(nx,ny),cg(nx,ny),icc(nx,ny),cgc(nx,ny),lpi(nx,ny))
 read(u) lat,lon,land,ht,top,t,z,w,refl,dz,theta,pi,moist,inputnum,inputden
 close(u)
 p=75000.;rho=1.
 ic=.03125;cg=.0625;icc=.125;cgc=.25;lpi=.75
 activation=start
 clock%month=month
 open(newunit=o,file=trim(outfile),access='stream',form='unformatted',status='replace')
 do s=1,nsteps
 call lightning_driver(times(s),dt,dx,dy,lat,lon,land,ht,t,p,rho,w,w,w,theta,pi,dz,z,moist,top,refl,clock, &
 opt,interval,start,activation,factor,method,num,den,inputnum,inputden,cell,adjust, &
 1,nx+1,1,ny+1,1,nz,1,nx,1,ny,1,nz,1,nx,1,ny,1,nz,icc,ic,cgc,cg,lpi)
 write(o) ic,cg,icc,cgc,lpi,activation
 enddo
 close(o)
end program
