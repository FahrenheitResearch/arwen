! Full-domain transport, WRF dyn_em/module_advect_em.F:9495-10560.
! Halo values are filled analytically, then the pinned routine runs unchanged.
program run_advect_mono
  use mono_wrf
  use oracle_io
  implicit none
  integer, parameter :: nx=12, ny=10, nz=8, lo=-4, ix=nx+5, jy=ny+5
  real :: q(lo:ix,1:nz+1,lo:jy), q0(lo:ix,1:nz+1,lo:jy)
  real :: ru(lo:ix,1:nz+1,lo:jy), rv(lo:ix,1:nz+1,lo:jy)
  real :: ww(lo:ix,1:nz+1,lo:jy), wi(lo:ix,1:nz+1,lo:jy)
  real :: t(lo:ix,1:nz+1,lo:jy), ht(lo:ix,1:nz+1,lo:jy), zt(lo:ix,1:nz+1,lo:jy)
  real :: pd(lo:ix,1:nz+1,lo:jy)
  real :: mut(lo:ix,lo:jy), mub(lo:ix,lo:jy), mu0(lo:ix,lo:jy)
  real :: mx(lo:ix,lo:jy), my(lo:ix,lo:jy)
  real :: c1(nz+1), c2(nz+1), rd(nz+1), fnm(nz+1), fnp(nz+1)
  real :: dt, cr, mass, base, velx, vely, amp, dx, dy
  integer :: b, m, f, v, n, i,j,k,a,d, ii,jj, horder, status
  character(len=1024) :: root, mode
  character(len=96) :: name
  type(grid_config_rec_type) :: cfg
  call oracle_root(root)
  call get_command_argument(2,mode)
  horder=5
  if (trim(mode)=='reject_h3') horder=3
  dx=1000.; dy=1200.; dt=20.
  do b=0,2
  do m=0,1
  do f=0,1
  do v=3,5,2
  do n=1,3
    cfg=grid_config_rec_type()
    cfg%h_sca_adv_order=horder; cfg%v_sca_adv_order=v
    cfg%periodic_x=b==0; cfg%periodic_y=b==0
    cfg%specified=b==1
    cfg%open_xs=b==2; cfg%open_xe=b==2
    cfg%open_ys=b==2; cfg%open_ye=b==2
    cr=0.1
    if(n==2) cr=0.4
    if(n==3) cr=0.8
    do k=1,nz+1
      c1(k)=1.-0.08*real(k-1); c2(k)=3000.*real(k-1)
      rd(k)=-real(nz)*(1.+0.03*real(k-1))
      fnm(k)=0.45+0.01*real(k-1); fnp(k)=1.-fnm(k)
    enddo
    do j=lo,jy
    do i=lo,ix
      ii=modulo(i-1,nx)+1; jj=modulo(j-1,ny)+1
      mub(i,j)=80000.+16.*real(ii)+8.*real(jj)
      mu0(i,j)=100.+4.*real(ii-jj)
      mut(i,j)=mub(i,j)+mu0(i,j)+8.*real(modulo(ii+jj,3)-1)
      mx(i,j)=1.; my(i,j)=1.
      if(m==1) then
        mx(i,j)=0.9+0.01*real(ii); my(i,j)=1.05+0.005*real(jj)
      endif
      do k=1,nz+1
        q0(i,k,j)=0.3+0.08*sin(6.2831853*real(ii-1)/real(nx)) &
          +0.04*cos(6.2831853*real(jj-1)/real(ny))+0.01*real(k-1)
        if(f==1) then
          q0(i,k,j)=0.
          if(ii>nx/2 .and. jj>ny/3 .and. k>nz/3) q0(i,k,j)=1.
        endif
        q(i,k,j)=q0(i,k,j)+0.03*real(modulo(3*ii+jj+k,5)-2)
        mass=c1(k)*80000.+c2(k)
        ru(i,k,j)=mass*(dx/dt)*cr*(0.7+0.02*real(modulo(jj+k,5)))
        if(modulo(jj+k,3)==0) ru(i,k,j)=-ru(i,k,j)
        rv(i,k,j)=mass*(dy/dt)*cr*(0.08+0.01*real(modulo(ii+k,5)))
        if(modulo(ii+k,3)==0) rv(i,k,j)=-rv(i,k,j)
        ww(i,k,j)=mass/(real(nz)*dt)*cr*0.04*real(modulo(ii+jj+k,3)-1)
        wi(i,k,j)=0.
        if(n==2) wi(i,k,j)=0.2*ww(i,k,j)
        if(k==1 .or. k==nz+1) then
          ww(i,k,j)=0.; wi(i,k,j)=0.
        endif
        t(i,k,j)=0.03125*real(modulo(ii+jj+k,3)-1)
      enddo
    enddo
    enddo
    write(name,'("b",I0,"_map",I0,"_front",I0,"_v",I0,"_c",I0)') b,m,f,v,n
    call oracle_open(trim(name))
    call oracle_put('q',q(1:nx,1:nz,1:ny)); call oracle_put('q0',q0(1:nx,1:nz,1:ny))
    call oracle_put('ru',ru(1:nx+1,1:nz,1:ny)); call oracle_put('rv',rv(1:nx,1:nz,1:ny+1))
    call oracle_put('ww',ww(1:nx,1:nz+1,1:ny)); call oracle_put('wi',wi(1:nx,1:nz+1,1:ny))
    call oracle_put('mut',mut(1:nx,1:ny)); call oracle_put('mub',mub(1:nx,1:ny))
    call oracle_put('mu0',mu0(1:nx,1:ny))
    call oracle_put('mx',mx(1:nx,1:ny)); call oracle_put('my',my(1:nx,1:ny))
    call oracle_put('c1',c1(1:nz)); call oracle_put('c2',c2(1:nz))
    call oracle_put('rd',rd(1:nz)); call oracle_put('fnm',fnm(1:nz)); call oracle_put('fnp',fnp(1:nz))
    call oracle_put('dt',dt); call oracle_put('dx',dx); call oracle_put('dy',dy)
    call oracle_put('boundary',b); call oracle_put('vorder',v)
    call oracle_put('initial_tendency',t(1:nx,1:nz,1:ny))
    ht=-999.; zt=-999.
    call advect_scalar_mono(q,q0,t,ht,zt,ru,rv,ww,wi,c1,c2,mut,mub,mu0,cfg,.true., &
      mx,my,mx,my,mx,my,fnm,fnp,1./dx,1./dy,rd,dt, &
      1,nx+1,1,ny+1,1,nz+1,lo,ix,lo,jy,1,nz+1,1,nx+1,1,ny+1,1,nz+1)
    call oracle_put('tendency',t(1:nx,1:nz,1:ny))
    ! h_tendency is defined only where the x assignment precedes the y addition.
    a=1; d=nx; ii=1; jj=ny
    if(b/=0) then
      a=2; d=nx-1; ii=2; jj=ny-1
    endif
    call oracle_put('h_tendency',ht(a:d,1:nz,ii:jj))
    call oracle_put('z_tendency',zt(1:nx,1:nz,1:ny))
    call oracle_close()
  enddo
  enddo
  enddo
  enddo
  enddo
  ! A divergence-free plateau separates monotonicity from positivity.
  cfg=grid_config_rec_type()
  cfg%periodic_x=.true.; cfg%periodic_y=.true.
  dx=1024.; dy=1024.; dt=16.
  do v=3,5,2
    cfg%v_sca_adv_order=v
    c1=1.; c2=0.; rd=-real(nz); fnm=0.5; fnp=0.5
    mub=65536.; mu0=0.; mut=65536.; mx=1.; my=1.
    ru=65536.*(dx/dt)*0.25; rv=0.; ww=0.; wi=0.; t=0.; pd=0.
    do j=lo,jy
    do k=1,nz+1
    do i=lo,ix
      ii=modulo(i-1,nx)+1
      q0(i,k,j)=0.25
      if(ii>nx/2) q0(i,k,j)=0.75
    enddo
    enddo
    enddo
    q=q0; ht=0.; zt=0.
    write(name,'("bounds_v",I0)') v
    call oracle_open(trim(name))
    call oracle_put('q',q(1:nx,1:nz,1:ny)); call oracle_put('q0',q0(1:nx,1:nz,1:ny))
    call oracle_put('ru',ru(1:nx+1,1:nz,1:ny)); call oracle_put('rv',rv(1:nx,1:nz,1:ny+1))
    call oracle_put('ww',ww(1:nx,1:nz+1,1:ny)); call oracle_put('wi',wi(1:nx,1:nz+1,1:ny))
    call oracle_put('mut',mut(1:nx,1:ny)); call oracle_put('mub',mub(1:nx,1:ny))
    call oracle_put('mu0',mu0(1:nx,1:ny)); call oracle_put('mx',mx(1:nx,1:ny)); call oracle_put('my',my(1:nx,1:ny))
    call oracle_put('c1',c1(1:nz)); call oracle_put('c2',c2(1:nz)); call oracle_put('rd',rd(1:nz))
    call oracle_put('fnm',fnm(1:nz)); call oracle_put('fnp',fnp(1:nz))
    call oracle_put('dt',dt); call oracle_put('dx',dx); call oracle_put('dy',dy)
    call oracle_put('boundary',0); call oracle_put('vorder',v)
    call oracle_put('initial_tendency',t(1:nx,1:nz,1:ny))
    call advect_scalar_mono(q,q0,t,ht,zt,ru,rv,ww,wi,c1,c2,mut,mub,mu0,cfg,.true., &
      mx,my,mx,my,mx,my,fnm,fnp,1./dx,1./dy,rd,dt, &
      1,nx+1,1,ny+1,1,nz+1,lo,ix,lo,jy,1,nz+1,1,nx+1,1,ny+1,1,nz+1)
    call oracle_put('tendency',t(1:nx,1:nz,1:ny))
    call oracle_put('h_tendency',ht(1:nx,1:nz,1:ny)); call oracle_put('z_tendency',zt(1:nx,1:nz,1:ny))
    call advect_scalar_pd(q,q0,pd,ht,zt,ru,rv,ww,c1,c2,mut,mub,mu0,1,cfg,.false., &
      mx,my,mx,my,mx,my,fnm,fnp,1./dx,1./dy,rd,dt, &
      1,nx+1,1,ny+1,1,nz+1,lo,ix,lo,jy,1,nz+1,1,nx+1,1,ny+1,1,nz+1)
    call oracle_put('pd_tendency',pd(1:nx,1:nz,1:ny))
    call oracle_close()
  enddo
end program
