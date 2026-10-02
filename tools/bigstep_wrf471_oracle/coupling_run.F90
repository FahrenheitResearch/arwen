! Stream inputs and outputs retain every IEEE real32 word.
program coupling_run
 use coupling_wrf
 implicit none
 type(grid_config_rec_type) :: cfg
 integer :: nx,ny,nz,nmoist,periodic,ieva,wdamping,inp,out,k,i,j,ia,ib,ja,jb
 integer :: ims,ime,jms,jme,kms,kme
 character(len=1024) :: infile,outfile
 real :: rdx,rdy,dt,dampcoef,zdamp,maxv,maxh
 real,allocatable :: mu(:,:),mub(:,:),mut(:,:),muu(:,:),muv(:,:)
 real,allocatable :: msftx(:,:),msfty(:,:),msfux(:,:),msfuy(:,:),msfvx(:,:),msfvy(:,:),msfvxi(:,:)
 real,allocatable :: u(:,:,:),v(:,:,:),w(:,:,:),ww(:,:,:),wwd(:,:,:),ph(:,:,:),phb(:,:,:),php(:,:,:)
 real,allocatable :: t(:,:,:),tinit(:,:,:),cqu(:,:,:),cqv(:,:,:),cqw(:,:,:),cqwr(:,:,:),moist(:,:,:,:)
 real,allocatable :: ru(:,:,:),rv(:,:,:),rw(:,:,:),rt(:,:,:),rwd(:,:,:)
 real,allocatable :: php_ru(:,:,:),php_rv(:,:,:)
 real,allocatable :: c1h(:),c2h(:),c1f(:),c2f(:),dnw(:),rdnw(:),ub(:),vb(:),tb(:),zb(:)
 call get_command_argument(1,infile)
 call get_command_argument(2,outfile)
 open(newunit=inp,file=trim(infile),access='stream',form='unformatted',status='old')
 read(inp) nx,ny,nz,nmoist,periodic,ieva,wdamping
 ims=0; ime=nx+1; jms=0; jme=ny+1; kms=1; kme=nz+1
 allocate(mu(ims:ime,jms:jme),mub(ims:ime,jms:jme),mut(ims:ime,jms:jme),muu(ims:ime,jms:jme),muv(ims:ime,jms:jme))
 allocate(msftx(ims:ime,jms:jme),msfty(ims:ime,jms:jme),msfux(ims:ime,jms:jme),msfuy(ims:ime,jms:jme),msfvx(ims:ime,jms:jme),msfvy(ims:ime,jms:jme),msfvxi(ims:ime,jms:jme))
 allocate(u(ims:ime,kms:kme,jms:jme),v(ims:ime,kms:kme,jms:jme),w(ims:ime,kms:kme,jms:jme),ww(ims:ime,kms:kme,jms:jme),wwd(ims:ime,kms:kme,jms:jme))
 allocate(ph(ims:ime,kms:kme,jms:jme),phb(ims:ime,kms:kme,jms:jme),php(ims:ime,kms:kme,jms:jme),t(ims:ime,kms:kme,jms:jme),tinit(ims:ime,kms:kme,jms:jme))
 allocate(cqu(ims:ime,kms:kme,jms:jme),cqv(ims:ime,kms:kme,jms:jme),cqw(ims:ime,kms:kme,jms:jme),cqwr(ims:ime,kms:kme,jms:jme),moist(ims:ime,kms:kme,jms:jme,nmoist))
 allocate(ru(ims:ime,kms:kme,jms:jme),rv(ims:ime,kms:kme,jms:jme),rw(ims:ime,kms:kme,jms:jme),rt(ims:ime,kms:kme,jms:jme),rwd(ims:ime,kms:kme,jms:jme))
 allocate(php_ru(ims:ime,kms:kme,jms:jme),php_rv(ims:ime,kms:kme,jms:jme))
 allocate(c1h(kms:kme),c2h(kms:kme),c1f(kms:kme),c2f(kms:kme),dnw(kms:kme),rdnw(kms:kme),ub(kms:kme),vb(kms:kme),tb(kms:kme),zb(kms:kme))
 read(inp) rdx,rdy,dt,dampcoef,zdamp,cfg%w_crit_cfl
 read(inp) mu,mub,msftx,msfty,msfux,msfuy,msfvx,msfvy,msfvxi
 read(inp) c1h,c2h,c1f,c2f,dnw,rdnw,ub,vb,tb,zb
 read(inp) u,v,w,wwd,ph,phb,t,tinit,moist,ru,rv,rw,rt,rwd
 close(inp)
 cfg%periodic_x=periodic==1; cfg%periodic_y=periodic==1
 cfg%polar=.false.; cfg%zadvect_implicit=ieva; cfg%w_damping=wdamping
 mut=mu+mub; muu=-999.; muv=-999.; ww=-999.; php=-999.; cqu=-999.; cqv=-999.; cqw=-999.; cqwr=1.
 call calc_mu_uv(cfg,mu,mub,muu,muv,1,nx+1,1,ny+1,1,nz+1,ims,ime,jms,jme,kms,kme,1,nx+1,1,ny+1,1,nz+1)
 call calc_cq(moist,cqu,cqv,cqw,nmoist,1,nx+1,1,ny+1,1,nz+1,ims,ime,jms,jme,kms,kme,1,nx+1,1,ny+1,1,nz+1)
 cqwr(1:nx,2:nz,1:ny)=1./(1.+cqw(1:nx,2:nz,1:ny))
 call calc_ww_cp(u,v,mu,mub,c1h,c2h,ww,rdx,rdy,msftx,msfty,msfux,msfuy,msfvx,msfvxi,msfvy,dnw,1,nx+1,1,ny+1,1,nz+1,ims,ime,jms,jme,kms,kme,1,nx,1,ny,1,nz+1)
 call calc_php(php,ph,phb,1,nx+1,1,ny+1,1,nz+1,ims,ime,jms,jme,kms,kme,1,nx,1,ny,1,nz+1)
 ! Isolated consuming response for advance_uv: zero acoustic p/phi and
 ! fixed mu_pp=1 leave only d(full half-level phi) * (-c1h) * spacing.
 ! This supplementary response is distinct from calc_php's own output.
 php_ru=0.; php_rv=0.
 do j=1,ny
 do k=1,nz
 do i=1,nx+1
   if(periodic==0.and.(j==1.or.j==ny.or.i==1.or.i==nx+1)) cycle
   ia=i; ib=i-1
   if(ia==nx+1) ia=1
   if(ib==0) ib=nx
   php_ru(i,k,j)=0.-((rdx*(php(ia,k,j)-php(ib,k,j)))*(-c1h(k)))
 enddo
 enddo
 enddo
 do j=1,ny+1
 do k=1,nz
 do i=1,nx
   if(periodic==0.and.(j==1.or.j==ny+1.or.i==1.or.i==nx)) cycle
   ja=j; jb=j-1
   if(ja==ny+1) ja=1
   if(jb==0) jb=ny
   php_rv(i,k,j)=0.-((rdy*(php(i,k,ja)-php(i,k,jb)))*(-c1h(k)))
 enddo
 enddo
 enddo
 call w_damp(rwd,maxv,maxh,u,v,wwd,w,mut,c1f,c2f,rdnw,rdx,rdy,msfux,msfuy,msfvx,msfvy,dt,cfg,1,nx+1,1,ny+1,1,nz+1,ims,ime,jms,jme,kms,kme,1,nx,1,ny,1,nz+1)
 call rk_rayleigh_damp(ru,rv,rw,rt,u,v,w,t,tinit,c1h,c2h,c1f,c2f,mut,muu,muv,ph,phb,ub,vb,tb,zb,dampcoef,zdamp,1,nx+1,1,ny+1,1,nz+1,ims,ime,jms,jme,kms,kme,1,nx+1,1,ny+1,1,nz+1)
 open(newunit=out,file=trim(outfile),access='stream',form='unformatted',status='replace')
 write(out) muu,muv,cqu,cqv,cqw,cqwr,ww,php,rwd,maxv,maxh,ru,rv,rw,rt,php_ru,php_rv
 close(out)
end program
