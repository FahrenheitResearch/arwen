program run_chem_prep
 use oracle_chem_prep
 use oracle_io
 implicit none
 include 'eta-vertmx.inc'
 type(grid_config_rec_type) :: cfg
 integer :: nz,k,c,nn,l
 real :: x,w,ps,theta,zz,dn
 real, allocatable :: p(:,:,:),pb(:,:,:),alt(:,:,:),ph(:,:,:),phb(:,:,:),t(:,:,:),u(:,:,:),v(:,:,:),moist(:,:,:,:)
 real, allocatable :: rho(:,:,:),pp(:,:,:),tt(:,:,:),up(:,:,:),vp(:,:,:),zw(:,:,:),dz(:,:,:),z(:,:,:),rh(:,:,:),pw(:,:,:),tw(:,:,:),fnm(:),fnp(:),eta(:)
 character(1024) :: out
 character(64) :: name
 call oracle_root(out)
 do nn=1,2
 nz=49+10*(nn-1)
 allocate(p(2,nz+1,2),pb(2,nz+1,2),alt(2,nz+1,2),ph(2,nz+1,2),phb(2,nz+1,2),t(2,nz+1,2),u(2,nz+1,2),v(2,nz+1,2),moist(2,nz+1,2,1))
 allocate(rho(2,nz+1,2),pp(2,nz+1,2),tt(2,nz+1,2),up(2,nz+1,2),vp(2,nz+1,2),zw(2,nz+1,2),dz(2,nz+1,2),z(2,nz+1,2),rh(2,nz+1,2),pw(2,nz+1,2),tw(2,nz+1,2),fnm(nz+1),fnp(nz+1),eta(nz+1))
 ! Resample the pinned practical eta ladder to each requested extent.
 do k=1,nz+1
 x=real(k-1)*real(size(practical)-1)/real(nz)
 l=min(int(x)+1,size(practical)-1); w=x-real(l-1)
 eta(k)=(1.-w)*practical(l)+w*practical(l+1)
 enddo
 fnm=0.; fnp=0.
 do k=2,nz
 dn=0.5*((eta(k+1)-eta(k))+(eta(k)-eta(k-1)))
 fnp(k)=.5*(eta(k+1)-eta(k))/dn
 fnm(k)=.5*(eta(k)-eta(k-1))/dn
 enddo
 do c=1,4
 ps=100000.; if(c==3)ps=60000.
 do k=1,nz+1
 zz=-7000.*log((5000.+(ps-5000.)*eta(k))/ps)
 if(c==3)zz=zz+4000.
 ph(:,k,:)=real(k)*0.125; phb(:,k,:)=zz*g
 p(:,k,:)=100.+real(k)*.25
 pb(:,k,:)=5000.+(ps-5000.)*eta(k)
 theta=max(216.,302.-.0065*zz)/((p(1,k,1)+pb(1,k,1))/p1000mb)**rcp
 if(c==2)theta=max(210.,275.-.006*zz)/((p(1,k,1)+pb(1,k,1))/p1000mb)**rcp
 t(:,k,:)=theta-t0
 alt(:,k,:)=r_d*theta*((p(1,k,1)+pb(1,k,1))/p1000mb)**rcp/(p(1,k,1)+pb(1,k,1))
 moist(:,k,:,1)=.018*exp(-zz/2500.)
 if(c==2)moist(:,k,:,1)=.002*exp(-zz/2500.)
 if(c==4)moist(:,k,:,1)=.95*3.80*exp(17.27*(theta*((p(1,k,1)+pb(1,k,1))/p1000mb)**rcp-273.)/(theta*((p(1,k,1)+pb(1,k,1))/p1000mb)**rcp-36.))/(.01*(p(1,k,1)+pb(1,k,1)))
 u(1,k,:)=real(k)*.2; u(2,k,:)=real(k)*.3
 v(:,k,1)=-real(k)*.1; v(:,k,2)=real(k)*.15
 enddo
 rho=-999.; pp=-999.; tt=-999.; up=-999.; vp=-999.; zw=-999.; dz=-999.; z=-999.; rh=-999.; pw=-999.; tw=-999.
 write(name,'("family",I0,"_nz",I0)')c,nz
 call oracle_open(trim(name))
 call oracle_put('p',p(1:1,:,1:1)); call oracle_put('pb',pb(1:1,:,1:1))
 call oracle_put('alt',alt(1:1,:,1:1)); call oracle_put('ph',ph(1:1,:,1:1)); call oracle_put('phb',phb(1:1,:,1:1)); call oracle_put('t',t(1:1,:,1:1))
 call oracle_put('qv',moist(1:1,:,1:1,1)); call oracle_put('u',u(:,:,1:1)); call oracle_put('v',v(1:1,:,:)); call oracle_put('fnm',fnm); call oracle_put('fnp',fnp); call oracle_put('eta',eta)
 call chem_prep(cfg,u,v,p,pb,alt,ph,phb,t,moist,1,rho,pp,up,vp,pw,tt,tw,z,zw,dz,rh,fnm,fnp,1,2,1,2,1,nz+1,1,2,1,2,1,nz+1,1,1,1,1,1,nz+1)
 call oracle_put('p_phy',pp(1:1,:,1:1)); call oracle_put('t_phy',tt(1:1,:,1:1)); call oracle_put('rho',rho(1:1,:,1:1)); call oracle_put('u_phy',up(1:1,:,1:1)); call oracle_put('v_phy',vp(1:1,:,1:1))
 call oracle_put('z_at_w',zw(1:1,:,1:1)); call oracle_put('dz8w',dz(1:1,:,1:1)); call oracle_put('z',z(1:1,:,1:1)); call oracle_put('rh',rh(1:1,:,1:1)); call oracle_put('p8w',pw(1:1,:,1:1)); call oracle_put('t8w',tw(1:1,:,1:1))
 call oracle_close()
 enddo
 deallocate(p,pb,alt,ph,phb,t,u,v,moist,rho,pp,tt,up,vp,zw,dz,z,rh,pw,tw,fnm,fnp,eta)
 enddo
end program
