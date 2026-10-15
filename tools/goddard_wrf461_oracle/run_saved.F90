program run_saved
 use module_ra_goddard, only: lwrad,swrad,fp_kind,fast_overcast
 use, intrinsic :: ieee_arithmetic
 implicit none
 integer :: n,u,v,at,k,ict(16),icb(16),flag(3)
 real,allocatable :: input(:),optics(:),lw(:,:),ld(:,:),lu(:,:),sw(:,:),sd(:,:),su(:,:)
 real(fp_kind),allocatable :: p(:,:),t(:,:),w(:,:),o(:,:),fc(:,:),tc(:,:,:),sc(:,:,:),gc(:,:,:),tl(:,:,:),sl(:,:,:),gl(:,:,:),z(:,:,:)
 real(fp_kind) :: cosine(16),tb(16),ts(16),emis(16,10),albedo(16)
 real :: sf(16,4)
 logical :: mask(16)
 character(512) :: source,target
 call get_command_argument(1,source);call get_command_argument(2,target)
 open(newunit=u,file=trim(source),access='stream',form='unformatted',convert='little_endian')
 read(u)n;allocate(input(5*n+15),optics(63*n));read(u)input,flag,optics;close(u)
 allocate(p(16,n+1),t(16,n),w(16,n),o(16,n),fc(16,n),tc(16,n,11),sc(16,n,11),gc(16,n,11),tl(16,n,10),sl(16,n,10),gl(16,n,10),z(16,n,11))
 allocate(lw(16,n+1),ld(16,n+1),lu(16,n+1),sw(16,n+1),sd(16,n+1),su(16,n+1))
 at=1
 do k=1,n+1;p(:,k)=input(at);at=at+1;enddo
 do k=1,n;t(:,k)=input(at);at=at+1;enddo
 do k=1,n;w(:,k)=input(at);at=at+1;enddo
 do k=1,n;o(:,k)=input(at);at=at+1;enddo
 do k=1,n;fc(:,k)=input(at);at=at+1;enddo
 cosine=input(at);tb=input(at+1);ts=input(at+2);at=at+3
 do k=1,10;emis(:,k)=input(at);at=at+1;enddo
 albedo=input(at);ict=flag(1);icb=flag(2);mask=flag(3)==1
 tc=spread(reshape(real(optics(1:11*n),fp_kind),[n,11]),1,16)
 sc=spread(reshape(real(optics(11*n+1:22*n),fp_kind),[n,11]),1,16)
 gc=spread(reshape(real(optics(22*n+1:33*n),fp_kind),[n,11]),1,16)
 tl=spread(reshape(real(optics(33*n+1:43*n),fp_kind),[n,10]),1,16)
 sl=spread(reshape(real(optics(43*n+1:53*n),fp_kind),[n,10]),1,16)
 gl=spread(reshape(real(optics(53*n+1:63*n),fp_kind),[n,10]),1,16)
 z=0.;fast_overcast=.true.;sw=0.;sd=0.;su=0.;sf=0.
 call swrad(n,cosine,p,t,w,o,fc,ict,icb,tc,sc,gc,z,z,z,albedo,albedo,albedo,albedo,sw,sd,su,sf,mask,1)
 call lwrad(n,emis,tb,ts,ict,icb,p,t,w,o,fc,tl,sl,gl,z(:,:,1:10),z(:,:,1:10),z(:,:,1:10),lw,ld,lu,1)
 open(newunit=v,file=trim(target),access='stream',form='unformatted',convert='little_endian',status='replace')
 write(v)sw(1,:),sd(1,:),su(1,:),sf(1,:),lw(1,:),ld(1,:),lu(1,:);close(v)
 print*,'LW nonfinite',count(.not.ieee_is_finite(lw(1,:))), 'SW nonfinite',count(.not.ieee_is_finite(sw(1,:)))
end program
