program run_initializer_atmos
 use sfire_ideal_atmos_oracle
 use oracle_io
 implicit none
 integer,parameter::nx=12,ny=10,nz=6,nmax=1000
 type(ideal_grid)::grid
 type(ideal_config)::cfg
 real::moist(nx+1,nz+1,ny+1,1),z(nmax),pm(nmax),pd(nmax),th(nmax),rho(nmax),u(nmax),v(nmax),q(nmax)
 real::eta(nz+1),znu(nz),dn(nz),fnp(nz),fnm(nz),cof1,cof2
 real::vals(6),coords(6),targets(263),result(263)
 integer::nl,i,j,k,c,dryflag,fixflag
 character(4096)::root
 character(60)::name
 call oracle_root(root)
 do dryflag=0,1
  call get_sounding(z,pm,pd,th,rho,u,v,q,dryflag==1,nmax,nl)
  write(name,'(a,i0)')'sounding_dry',dryflag
  call oracle_open(trim(name))
  call oracle_put('height',z(1:nl));call oracle_put('p_moist',pm(1:nl));call oracle_put('p_dry',pd(1:nl))
  call oracle_put('theta',th(1:nl));call oracle_put('rho',rho(1:nl));call oracle_put('u',u(1:nl));call oracle_put('v',v(1:nl));call oracle_put('qv',q(1:nl))
  call oracle_close()
 enddo
 vals=[-0.,2.,-3.,7.,9.,.5]
 do c=1,2
  if(c==1)then
   coords=[0.,100.,700.,1800.,4000.,7000.]
   do i=1,257;targets(i)=-250.+real(i-1)*35.;enddo
  else
   coords=[100000.,92000.,80000.,61000.,42000.,21000.]
   do i=1,257;targets(i)=15000.+real(i-1)*400.;enddo
  endif
  targets(258:263)=coords
  do i=1,263;result(i)=interp_0(vals,coords,targets(i),6);enddo
  write(name,'(a,i0)')'interpolation',c
  call oracle_open(trim(name));call oracle_put('values',vals);call oracle_put('coordinates',coords)
  call oracle_put('targets',targets);call oracle_put('out',result);call oracle_close()
 enddo
 call allocate_grid(grid,nx,ny,nz)
 eta=[1.,.92,.71,.46,.23,.08,0.]
 grid%dnw(1:nz)=eta(2:nz+1)-eta(1:nz)
 grid%rdnw(1:nz)=1./grid%dnw(1:nz)
 znu=.5*(eta(2:nz+1)+eta(1:nz))
 dn=0.;fnp=0.;fnm=0.;grid%rdn=0.
 do k=2,nz
  dn(k)=znu(k)-znu(k-1);grid%rdn(k)=1./dn(k)
  fnp(k)=.5*grid%dnw(k)/dn(k);fnm(k)=.5*grid%dnw(k-1)/dn(k)
 enddo
 cof1=(2.*dn(2)+dn(3))/(dn(2)+dn(3))*grid%dnw(1)/dn(2)
 cof2=dn(2)/(dn(2)+dn(3))*grid%dnw(1)/dn(3)
 grid%cf1=fnp(2)+cof1;grid%cf2=fnm(2)-cof1-cof2;grid%cf3=cof2
 grid%c1f=1.;grid%c2f=0.;grid%c1h=1.;grid%c2h=0.;grid%c3h(1:nz)=znu;grid%c4h=0.
 grid%p_top=interp_0(pm,z,cfg%ztop,nl)
 do c=1,6
  cfg%delt=0.;cfg%xr=0.;cfg%yr=0.;cfg%zr=0.;cfg%height=0.;cfg%use_theta_m=1;cfg%sfc_full_init=.false.
  grid%ht=0.;grid%tsk=280.;grid%tmn=279.5
  if(c>=2)then
   do j=1,ny+1
    do i=1,nx+1
     grid%ht(i,j)=20.+real(i)*13.+real(j)*7.+real(modulo(i*j,5))*3.
    enddo
   enddo
  endif
  if(c==2)cfg%use_theta_m=0
  if(c==3.or.c==4.or.c==6)then
   cfg%delt=3.;if(c==4)cfg%delt=-2.
   cfg%xr=280.;cfg%yr=350.;cfg%zr=1600.;cfg%height=1200.
  endif
  if(c==5)cfg%sfc_full_init=.true.
  if(c==6)then
   grid%ht=grid%ht*2.;cfg%height=750.;cfg%xr=185.;cfg%yr=230.;cfg%zr=850.
  endif
  do fixflag=0,1
   moist=0.
   call initialize_columns(grid,moist,cfg,nx,ny,nz,fixflag==1)
   write(name,'(a,i0,a,i0)')'columns',c,'_corrected',fixflag
   call oracle_open(trim(name))
   call oracle_put('terrain',grid%ht(1:nx,1:ny))
   call oracle_put('settings',[cfg%dx,cfg%dy,cfg%ztop,cfg%delt,cfg%xr,cfg%yr,cfg%zr,cfg%height])
   call oracle_put('flags',[cfg%use_theta_m,merge(1,0,cfg%sfc_full_init)])
   call oracle_put('p_top',grid%p_top);call oracle_put('cf1',grid%cf1);call oracle_put('cf2',grid%cf2);call oracle_put('cf3',grid%cf3)
   call oracle_put('dnw',grid%dnw(1:nz));call oracle_put('rdnw',grid%rdnw(1:nz));call oracle_put('rdn',grid%rdn(1:nz))
   call oracle_put('c1f',grid%c1f);call oracle_put('c2f',grid%c2f)
   call oracle_put('c1h',grid%c1h(1:nz));call oracle_put('c2h',grid%c2h(1:nz));call oracle_put('c3h',grid%c3h(1:nz));call oracle_put('c4h',grid%c4h(1:nz))
   call oracle_put('U',grid%u_2(1:nx+1,1:nz,1:ny));call oracle_put('V',grid%v_2(1:nx,1:nz,1:ny+1));call oracle_put('W',grid%w_2(1:nx,1:nz+1,1:ny))
   call oracle_put('PH',grid%ph_2(1:nx,1:nz+1,1:ny));call oracle_put('PHB',grid%phb(1:nx,1:nz+1,1:ny))
   call oracle_put('MU',grid%mu_2(1:nx,1:ny));call oracle_put('MUB',grid%mub(1:nx,1:ny))
   call oracle_put('T_DRY',grid%t_1(1:nx,1:nz,1:ny));call oracle_put('T',grid%t_2(1:nx,1:nz,1:ny));call oracle_put('T_INIT',grid%t_init(1:nx,1:nz,1:ny))
   call oracle_put('P',grid%p(1:nx,1:nz,1:ny));call oracle_put('PB',grid%pb(1:nx,1:nz,1:ny));call oracle_put('AL',grid%al(1:nx,1:nz,1:ny));call oracle_put('ALB',grid%alb(1:nx,1:nz,1:ny));call oracle_put('ALT',grid%alt(1:nx,1:nz,1:ny))
   call oracle_put('QVAPOR',moist(1:nx,1:nz,1:ny,1));call oracle_put('H_DIABATIC',grid%h_diabatic(1:nx,1:nz,1:ny))
   call oracle_put('TSK',grid%tsk(1:nx,1:ny));call oracle_put('TMN',grid%tmn(1:nx,1:ny))
   call oracle_close()
  enddo
 enddo
end program
