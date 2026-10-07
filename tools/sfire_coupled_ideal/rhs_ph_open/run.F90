program run_open_phi
  use rhs_open_native, only: native => rhs_ph
  use module_configure, only: grid_config_rec_type
  use oracle_io
  implicit none
  integer, parameter :: nx=14,ny=12,nz=7,ml=-3,mh=nx+4,mb=-3,mt=ny+4
  real :: u(ml:mh,1:nz+1,mb:mt),v(ml:mh,1:nz+1,mb:mt),ww(ml:mh,1:nz+1,mb:mt)
  real :: ph(ml:mh,1:nz+1,mb:mt),phb(ml:mh,1:nz+1,mb:mt),w(ml:mh,1:nz+1,mb:mt)
  real :: seed(ml:mh,1:nz+1,mb:mt),a(ml:mh,1:nz+1,mb:mt)
  real :: mut(ml:mh,mb:mt),muu(ml:mh,mb:mt),muv(ml:mh,mb:mt)
  real :: mfx(ml:mh,mb:mt),mfy(ml:mh,mb:mt),mfu(ml:mh,mb:mt),mfv(ml:mh,mb:mt)
  real :: invfv(ml:mh,mb:mt),fnm(nz+1),fnp(nz+1),rdnw(nz+1),c1(nz+1),c2(nz+1)
  real :: cfn,cfn1,rdx,rdy,ghost
  integer :: m,mask,order,vertical,control,i,j,k,ii,jj
  character(len=1024) :: root
  character(len=80) :: name
  type(grid_config_rec_type) :: config
  call oracle_root(root)
  cfn=1.5;cfn1=-0.5;rdx=1./90.;rdy=1./120.
  do k=1,nz+1
    fnm(k)=0.53+0.011*k;fnp(k)=1.-fnm(k)
    rdnw(k)=-real(nz)-0.03*k
    c1(k)=1.-0.087*(k-1);c2(k)=13.*(k-1)
  enddo
  ! WRF assigns FNM/FNP only for k=2..kde-1; kde keeps its zero allocation value.
  fnm(nz+1)=0.;fnp(nz+1)=0.
  do m=1,16
    mask=modulo(m-1,4);order=merge(2,5,modulo((m-1)/4,2)==0)
    vertical=(m-1)/8
    config%open_xs=btest(mask,0);config%open_xe=btest(mask,0)
    config%open_ys=btest(mask,1);config%open_ye=btest(mask,1)
    config%specified=.false.;config%nested=.false.;config%phi_adv_z=1
    config%h_sca_adv_order=order
    do j=mb,mt
      jj=merge(j,modulo(j-1,ny)+1,btest(mask,1))
      do i=ml,mh
        ii=merge(i,modulo(i-1,nx)+1,btest(mask,0))
        mut(i,j)=91000.+31.*ii+17.*jj
        mfx(i,j)=1.+0.0007*ii+0.0004*jj
        mfy(i,j)=mfx(i,j)
        mfu(i,j)=1.+0.0003*ii+0.0002*jj
        mfv(i,j)=1.+0.0002*ii+0.0003*jj
        invfv(i,j)=1./mfv(i,j)
        do k=1,nz+1
          u(i,k,j)=-2.1+0.21*ii+0.009*jj+0.05*k
          v(i,k,j)=-2.0+0.31*jj+0.007*ii-0.02*k
          ph(i,k,j)=13.+0.2*ii*ii+0.15*jj+0.004*k*ii*jj
          phb(i,k,j)=9.81*(300.+50.*(k-1)+0.08*ii*ii+0.13*jj*jj)
          ww(i,k,j)=merge(0.02*k*ii-0.015*k*jj,0.,vertical==1)
          w(i,k,j)=merge(0.01*ii-0.02*jj+0.005*k,0.,vertical==1)
          seed(i,k,j)=0.125*ii-0.375*jj+0.03125*k
        enddo
      enddo
    enddo
    do j=mb+1,mt
      do i=ml+1,mh
        muu(i,j)=0.5*(mut(i-1,j)+mut(i,j))
        muv(i,j)=0.5*(mut(i,j-1)+mut(i,j))
      enddo
    enddo
    do control=1,2
      ! Control 1 holds the unassigned top U level at its zero allocation
      ! value, as WRF does at run time; control 2 is a sentinel.
      ghost=merge(0.,-17.,control==1)
      u(:,nz+1,:)=ghost
      a=seed
      call native(a,u,v,ww,ph,ph,phb,w,mut,muu,muv,c1,c2,fnm,fnp,rdnw,cfn,cfn1,rdx,rdy, &
        mfu,mfu,mfv,invfv,mfv,mfx,mfy,vertical==1,config, &
        1,nx+1,1,ny+1,1,nz+1,ml,mh,mb,mt,1,nz+1,1,nx+1,1,ny+1,1,nz+1)
      write(name,'(a,i0,a,i0)') 'rhs_ph_open/case_',m,'_ghost_',control
      call oracle_open(trim(name))
      call oracle_put('u',u(1:nx+1,1:nz,1:ny));call oracle_put('v',v(1:nx,1:nz,1:ny+1))
      call oracle_put('ww',ww(1:nx,1:nz+1,1:ny));call oracle_put('w',w(1:nx,1:nz+1,1:ny))
      call oracle_put('ph',ph(1:nx,1:nz+1,1:ny));call oracle_put('phb',phb(1:nx,1:nz+1,1:ny))
      call oracle_put('seed',seed(1:nx,1:nz+1,1:ny));call oracle_put('mut',mut(1:nx,1:ny))
      call oracle_put('muu',muu(1:nx+1,1:ny));call oracle_put('muv',muv(1:nx,1:ny+1))
      call oracle_put('msft',mfy(1:nx,1:ny));call oracle_put('msfu',mfu(1:nx+1,1:ny))
      call oracle_put('msfv',mfv(1:nx,1:ny+1));call oracle_put('fnm',fnm);call oracle_put('fnp',fnp)
      call oracle_put('rdnw',rdnw);call oracle_put('c1f',c1);call oracle_put('c2f',c2)
      call oracle_put('cfn',cfn);call oracle_put('cfn1',cfn1)
      call oracle_put('rdx',rdx);call oracle_put('rdy',rdy)
      call oracle_put('mask',mask);call oracle_put('order',order)
      call oracle_put('vertical',vertical);call oracle_put('top_ghost_u',ghost)
      call oracle_put('native',a(1:nx,1:nz+1,1:ny))
      call oracle_close()
    enddo
  enddo
end program
