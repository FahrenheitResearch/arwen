program run_diag
 use module_convtrans_prep, only: convtrans_prep
 implicit none
 integer :: nx,ny,nz,ns,fqi,adapt,i,j,k,n,pqi
 real :: avg
 character(len=1024) :: fin,fout
 real, allocatable :: t(:,:,:,:),p(:,:,:,:),q(:,:,:,:),qc(:,:,:,:),qi(:,:,:,:),cloud(:,:,:,:),ice(:,:,:,:),rain(:,:,:),dt(:),curr(:)
 real, allocatable :: ca(:,:,:),cb(:,:,:),ia(:,:,:),ib(:,:,:),frac(:,:,:),ra(:,:),rb(:,:),moist(:,:,:,:)
 call get_command_argument(1,fin); call get_command_argument(2,fout)
 open(10,file=trim(fin),access='stream',form='unformatted',status='old')
 read(10) nx,ny,nz,ns,fqi,adapt,avg
 allocate(t(nx,nz,ny,ns),p(nx,nz,ny,ns),q(nx,nz,ny,ns),qc(nx,nz,ny,ns),qi(nx,nz,ny,ns),cloud(nx,nz,ny,ns),ice(nx,nz,ny,ns),rain(nx,ny,ns),dt(ns),curr(ns))
 read(10) dt,curr,t,p,q,qc,qi,cloud,ice,rain
 close(10)
 allocate(ca(nx,nz,ny),cb(nx,nz,ny),ia(nx,nz,ny),ib(nx,nz,ny),frac(nx,nz,ny),ra(nx,ny),rb(nx,ny),moist(nx,nz,ny,3))
 ca=0.; cb=0.; ia=0.; ib=0.; frac=0.; ra=0.; rb=0.
 pqi=1; if(fqi/=0) pqi=3
 open(20,file=trim(fout),access='stream',form='unformatted',status='replace')
 do n=1,ns
  moist(:,:,:,1)=q(:,:,:,n); moist(:,:,:,2)=qc(:,:,:,n); moist(:,:,:,3)=qi(:,:,:,n)
  call convtrans_prep(cloud(:,:,:,n),ice(:,:,:,n),ca,cb,rain(:,:,n),ra,rb,frac,moist,1,2,pqi,t(:,:,:,n),p(:,:,:,n),3,ia,ib,avg,adapt/=0,curr(n),n,dt(n),5,1,nx+1,1,ny+1,1,nz+1,1,nx,1,ny,1,nz,1,nx,1,ny,1,nz)
  write(20) ca,cb,ia,ib,frac,ra,rb
 enddo
 close(20)
end program run_diag
