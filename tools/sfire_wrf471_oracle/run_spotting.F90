program run_spotting
use module_spotting_oracle
use oracle_io
implicit none
integer,parameter::n=32
integer,parameter::ax=9,ay=7,nz=8,halo=4,ml=1-halo,mh=ax+halo,mb=1-halo,mt=ay+halo
integer::i,c
integer::j,k
type(p_properties)::prop(n),old(n)
real::inputs(n,4),pressure(n),density(n),temperature(n),wind(n),height(n),dt
real::out(n,5)
real::coords(n,3),corners(n,8),box(n),samples(n,7),zw(nz)
real::u(ml:mh,nz,mb:mt),v(ml:mh,nz,mb:mt),w(ml:mh,nz,mb:mt)
real::p(ml:mh,nz,mb:mt),t(ml:mh,nz,mb:mt),d(ml:mh,nz,mb:mt),z(ml:mh,nz,mb:mt)
real::uvw(3),lp,lt,ld,zp
real::metric(ml:mh,mb:mt),xout(n),yout(n),hout(n)
integer::life(n),ids_array(n)
type(domain)::grid
character(len=200)::message
real,allocatable::ri(:),rj(:),rk(:),ri0(:),rj0(:),rk0(:)
type(p_properties),allocatable::rprop(:),rprop0(:)
integer,allocatable::rnum(:),rage(:),rsrc(:)
integer::pid(n),psrc(n),page(n),idmax,active,points,levels,seed,info(3)
real::px(n),py(n),pz(n),rank_arr(n),rank_value
character(len=1024)::root
character(len=64)::name
call oracle_root(root)
firebrand_dens=513000.;firebrand_dens_char=299000.
do i=1,n
  inputs(i,:)=[.5+real(i)*.625,.35+real(i)*.4,425.+real(i)*15.,real(mod(i,4))*1.5]
  prop(i)=firebrand_property(inputs(i,:))
  pressure(i)=68000.+real(i)*813.
  density(i)=710.+real(i)*13.
  temperature(i)=265.+real(i)*1.1
  wind(i)=-3.5+real(i)*.41
  height(i)=1.+real(i*i)*.125
enddo
call oracle_open('spotting/property')
call oracle_put('inputs',inputs)
call save_prop('prop',prop)
call oracle_close()
call oracle_open('spotting/property_corrected')
call oracle_put('inputs',inputs)
do i=1,n
 prop(i)=firebrand_property_corrected(inputs(i,:))
enddo
call save_prop('prop',prop)
call oracle_close()
do i=1,n
 prop(i)=firebrand_property(inputs(i,:))
enddo
old=prop
do c=1,4
  dt=.015625
  if(c==2)dt=.125
  if(c==3)dt=1.
  if(c==4)dt=120.
  prop=old
  write(name,'(a,i0)')'spotting/burnout_',c
  call oracle_open(trim(name))
  call save_prop('input',prop)
  call save_input()
  do i=1,n
    call burnout(p_mass=prop(i)%p_mass,p_diam=prop(i)%p_diam,p_effd=prop(i)%p_effd, &
      p_temp=prop(i)%p_temp,p_tvel=prop(i)%p_tvel,dt=dt,pres=pressure(i), &
      aird=density(i),temp=temperature(i),loc_w=wind(i))
  enddo
  call save_prop('output',prop)
  call oracle_close()
  prop=old
  write(name,'(a,i0)')'spotting/termvel_',c
  call oracle_open(trim(name))
  call save_prop('input',prop)
  call save_input()
  call oracle_put('height_in',height)
  do i=1,n
    call termvel(p_diam=prop(i)%p_diam,p_effd=prop(i)%p_effd,p_temp=prop(i)%p_temp, &
      p_tvel=prop(i)%p_tvel,dt=dt,hgt=height(i),pres=pressure(i), &
      aird=density(i),temp=temperature(i))
  enddo
  call save_prop('output',prop)
  call oracle_put('height_out',height)
  call oracle_close()
enddo
do i=1,n
 coords(i,:)=[1.5+.0625*real(i),2.125+.03125*real(i),.5+.0625*real(i)]
 do j=1,8
  corners(i,j)=.1375*real(i*j)+.03125*real(j*j)
 enddo
 box(i)=u_3d_interp(coords(i,1),coords(i,2),coords(i,3), &
  corners(i,1),corners(i,2),corners(i,3),corners(i,4),corners(i,5),corners(i,6),corners(i,7),corners(i,8))
enddo
call oracle_open('spotting/interp_box')
call oracle_put('coordinates',coords);call oracle_put('corners',corners);call oracle_put('output',box)
call oracle_close()
ims=ml;ime=mh;jms=mb;jme=mt;kms=1;kme=nz
ids=1;ide=ax+1;jds=1;jde=ay+1;kde=nz
is=1;ie=ax;js=1;je=ay;ks=1;ke=nz
do j=mb,mt
 do k=1,nz
  do i=ml,mh
   u(i,k,j)=2.+.025*real(i*i)+.125*real(k)+.035*real(j*k)
   v(i,k,j)=-1.5+.051*real(j*j)-.0625*real(k)+.00125*real(i*j)
   w(i,k,j)=.0625*real(modulo(i+2*j,4))+.0125*real(k)
   p(i,k,j)=95000.-1750.*real(k)+31.*real(i*i+j*j)
   t(i,k,j)=282.+1.125*real(k)+.0125*real(i*j)
   d(i,k,j)=1.25-.055*real(k)+.00025*real(i*i+j*j)
   z(i,k,j)=(15.+.021*real(i*i)+.015*real(j*j))*real((k-1)**2)
  enddo
 enddo
enddo
zw=[(1.-real(k-1)/real(nz-1),k=1,nz)]
do i=1,n
 coords(i,:)=[1.5+.125*real(mod(i,24)),1.5+.125*real(mod(i,16)),17.5+2.75*real(i)]
 zp=hgt2k(coords(i,1),coords(i,2),coords(i,3),z,zw)
 uvw=uvw_3d_interp(coords(i,1),coords(i,2),zp,u,v,w,is-4,js-4,ie-1+4,je-1+4)
 call get_local_met(coords(i,1),coords(i,2),zp,lp,ld,lt,p,t,d,is-4,js-4,ie-1+4,je-1+4)
 samples(i,:)=[zp,uvw,lp,lt,ld]
enddo
call oracle_open('spotting/sampling')
call oracle_put('coordinates',coords)
call oracle_put('u',u);call oracle_put('v',v);call oracle_put('w',w)
call oracle_put('pressure',p);call oracle_put('theta',t);call oracle_put('density',d);call oracle_put('height',z)
call oracle_put('output',samples)
call oracle_close()
do j=mb,mt
 do i=ml,mh
  metric(i,j)=(.85+.0125*real(i*i+j))*0.01
 enddo
enddo
do c=1,3
 prop=old
 life=[(mod(i,6),i=1,n)]
 ids_array=[(i,i=1,n)]
 dt=.125
 if(c==2)dt=1.
 if(c==3)dt=12.
 write(name,'(a,i0)')'spotting/advection_',c
 call oracle_open(trim(name))
 call save_prop('input',prop)
 call oracle_put('coordinates',coords);call oracle_put('life',life);call oracle_put('dt',dt)
 call oracle_put('u',u);call oracle_put('v',v);call oracle_put('w',w)
 call oracle_put('pressure',p);call oracle_put('theta',t);call oracle_put('density',d);call oracle_put('height',z)
 call oracle_put('metric_x',metric);call oracle_put('metric_y',metric)
 call advect_xyz_m(grid=grid,xp=coords(:,1),yp=coords(:,2),hgt=coords(:,3),dtp=life,idp=ids_array, &
  u=u,v=v,w=w,dt=dt,mf=metric,z_at_w=z,znw=zw,ims=ml,jms=mb,kms=1,phyd=p,thet=t,rho=d, &
  xout=xout,yout=yout,zout=hout,fs_p_prop=prop,land_hgt=.15,start_mom3d_dt=4,msg=message)
 call save_prop('output',prop)
 call oracle_put('x_out',xout);call oracle_put('y_out',yout);call oracle_put('h_out',hout)
 call oracle_close()
enddo
do i=1,n
 rank_arr(i)=.0125*real(mod(i*7,23))+real(mod(i,4))*.03125
enddo
call oracle_open('spotting/ranking')
call oracle_put('values',rank_arr)
do c=0,4
 rank_value=order_val(rank_arr,ord=c*6)
 write(name,'(a,i0)')'order_',c*6
 call oracle_put(trim(name),rank_value)
enddo
call oracle_close()
fs_array_maxsize=n
do c=1,6
 points=4;levels=1;seed=0
 if(c==2)levels=3
 if(c==3)then
  levels=5;seed=11
 endif
 if(c==4)then
  levels=3;seed=3
 endif
 if(c==5)levels=3
 fs_gen_levels=levels
 allocate(ri(points*levels),rj(points*levels),rk(points*levels),rprop(points*levels), &
          ri0(points*levels),rj0(points*levels),rk0(points*levels),rprop0(points*levels), &
          rnum(points),rage(points*levels),rsrc(points*levels))
 ri=0.;rj=0.;rk=0.;rnum=1
 rprop=p_properties(0.,0.,0.,0.,0.)
 do i=1,points
  ri(i)=1.375+.5*real(i)
  rj(i)=2.125+.25*real(i)
  rk(i)=40.+real(i)
  if(c==4)rk(i)=.5+.0625*real(i)
  rprop(i)=old(i+4)
 enddo
 write(name,'(a,i0)')'spotting/release_',c
 call oracle_open(trim(name))
 call oracle_put('points',points);call oracle_put('levels',levels);call oracle_put('seed',seed)
 call oracle_put('x_in',ri);call oracle_put('y_in',rj);call oracle_put('h_in',rk)
 call save_prop('prop_in',rprop)
 call prep_release_hgt(ri,rj,rk,rnum,rprop,seed,.15)
 call oracle_put('x_out',ri);call oracle_put('y_out',rj);call oracle_put('h_out',rk)
 call save_prop('prop_out',rprop)
 call oracle_close()
 ri0=ri;rj0=rj;rk0=rk;rprop0=rprop
 pid=0;psrc=0;page=0;px=0.;py=0.;pz=0.;prop=p_properties(0.,0.,0.,0.,0.)
 active=3
 if(c==5)active=30
 do i=1,active
  pid(i)=i;psrc(i)=1000000+i;page(i)=2
  px(i)=1.75;py(i)=2.25;pz(i)=12.5;prop(i)=old(i)
 enddo
 idmax=active
 if(c==3)idmax=huge(1)-11
 rage=[(i+3,i=1,points*levels)]
 rsrc=[(9000000+i,i=1,points*levels)]
 write(name,'(a,i0)')'spotting/generate_',c
 call oracle_open(trim(name))
 call oracle_put('x_in',px);call oracle_put('y_in',py);call oracle_put('h_in',pz)
 call oracle_put('id_in',pid);call oracle_put('source_in',psrc);call oracle_put('life_in',page)
 call save_prop('prop_in',prop)
 call oracle_put('release_x',ri);call oracle_put('release_y',rj);call oracle_put('release_h',rk)
 call save_prop('release_prop',rprop)
 call oracle_put('idmax_in',idmax)
 call oracle_put('imported',merge(1,0,c==6))
 call oracle_put('release_life',rage);call oracle_put('release_source',rsrc)
 if(c==6)then
  call generate_firebrands(pid,psrc,page,pz,px,py,ri,rj,rk,rage,rsrc,active,idmax,1000000,rprop,prop)
 else
  call generate_firebrands(fs_p_id=pid,fs_p_src=psrc,fs_p_dt=page,fs_p_z=pz,fs_p_x=px,fs_p_y=py, &
   release_i=ri,release_j=rj,release_k=rk,active_br=active,fs_gen_idmax=idmax,myprocid=1000000, &
   release_prop=rprop,fs_p_prop=prop)
 endif
 call oracle_put('x_out',px);call oracle_put('y_out',py);call oracle_put('h_out',pz)
 call oracle_put('id_out',pid);call oracle_put('source_out',psrc);call oracle_put('life_out',page)
 call save_prop('prop_out',prop)
 call oracle_put('release_x_out',ri);call oracle_put('release_y_out',rj);call oracle_put('release_h_out',rk)
 info=[idmax,active,0];call oracle_put('control_out',info)
 call oracle_close()
 deallocate(ri,rj,rk,rprop,ri0,rj0,rk0,rprop0,rnum,rage,rsrc)
enddo
contains
subroutine save_input()
call oracle_put('pressure',pressure);call oracle_put('density',density)
call oracle_put('temperature',temperature);call oracle_put('wind',wind);call oracle_put('dt',dt)
end subroutine
subroutine save_prop(key,p)
character(len=*),intent(in)::key
type(p_properties),intent(in)::p(:)
real::temporary(size(p),5)
temporary(:,1)=p%p_mass;temporary(:,2)=p%p_diam;temporary(:,3)=p%p_effd;temporary(:,4)=p%p_temp;temporary(:,5)=p%p_tvel
call oracle_put(key,temporary)
end subroutine
end program
