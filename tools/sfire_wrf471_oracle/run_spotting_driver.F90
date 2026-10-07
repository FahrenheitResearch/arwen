program run_spotting_driver
use module_domain
use module_configure
use module_firebrand_spotting
use module_spotting_columns_oracle,only:prepare_columns
use oracle_io
implicit none
integer,parameter::nx=6,ny=4,nz=7,h=4,ml=1-h,mh=nx+h,mb=1-h,mt=ny+h,n=256,fx=12,fy=12
type(domain)::grid
type(grid_config_rec_type)::cf
integer::pid(n),psrc(n),page(n),last_gen,idmax,mask(ml:mh,mb:mt),gen(ml:mh,mb:mt)
real::px(n),py(n),pz(n),mass(n),diam(n),effd(n),ptemp(n),tvel(n)
real::rosdt(fx,fy),area(ml:mh,mb:mt),risk(ml:mh,mb:mt),all_count(ml:mh,mb:mt)
real::hist_count(ml:mh,mb:mt),fraction(ml:mh,mb:mt),likelihood(ml:mh,mb:mt)
logical::reset
integer::i,j,k,c,s
real::terrain
real::pu(ml:mh,nz+1,mb:mt),pv(ml:mh,nz+1,mb:mt),pw(ml:mh,nz+1,mb:mt)
real::pp(ml:mh,nz+1,mb:mt),pt(ml:mh,nz+1,mb:mt),pd(ml:mh,nz+1,mb:mt),pz8(ml:mh,nz+1,mb:mt)
real::pm(ml:mh,mb:mt),p8(ml:mh,nz+1,mb:mt)
real::ptmass(ml:mh,nz+1,mb:mt)
character(len=1024)::root
character(len=64)::name
call oracle_root(root)
allocate(grid%u_2(ml:mh,nz+1,mb:mt),grid%v_2(ml:mh,nz+1,mb:mt),grid%w_2(ml:mh,nz+1,mb:mt), &
 grid%p(ml:mh,nz+1,mb:mt),grid%pb(ml:mh,nz+1,mb:mt),grid%t_2(ml:mh,nz+1,mb:mt), &
 grid%phb(ml:mh,nz+1,mb:mt),grid%ph_2(ml:mh,nz+1,mb:mt),grid%moist(ml:mh,nz+1,mb:mt,3), &
 grid%al(ml:mh,nz+1,mb:mt),grid%alb(ml:mh,nz+1,mb:mt),grid%msftx(ml:mh,mb:mt),grid%msfty(ml:mh,mb:mt),grid%muts(ml:mh,mb:mt), &
 grid%znw(nz+1),grid%c1h(nz),grid%c2h(nz),grid%dnw(nz),grid%fnm(nz+1),grid%fnp(nz+1), &
 grid%burnt_area_dt(fx,fy),grid%fmc_g(fx,fy),grid%nfuel_cat(fx,fy),grid%fgip(fx,fy),grid%fire_area(fx,fy))
grid%nx=nx;grid%ny=ny;grid%nz=nz;grid%halo=h
grid%dt=.125;grid%rdx=.01;grid%rdy=.01
grid%znw=[(1.-real(k-1)/real(nz),k=1,nz+1)]
grid%c1h=1.;grid%c2h=0.;grid%dnw=-1./real(nz);grid%fnm=.5;grid%fnp=.5
do c=1,4
 grid%rdy=.01
 if(c==4)grid%rdy=1./150.
 grid%use_theta_m=0
 if(c==3)grid%use_theta_m=1
 cf%fs_firebrand_gen_levrand=c==3
 do j=mb,mt
  do i=ml,mh
   terrain=0.
   if(c>=2)terrain=310.+.25*real(i*i+j*j)
   grid%msftx(i,j)=.95+.001*real(i*i+j*j)
   grid%msfty(i,j)=grid%msftx(i,j)
   if(c==4)grid%msfty(i,j)=.92+.0015*real(i*j)
   grid%muts(i,j)=46000.+31.*real(i+j)
   do k=1,nz+1
    grid%u_2(i,k,j)=3.+.0125*real(i*j)+.025*real(k)
    grid%v_2(i,k,j)=1.+.0075*real(i*i+j)+.015*real(k)
    grid%w_2(i,k,j)=.05+.00625*real(k)
    grid%p(i,k,j)=7.5*real(i*i+j)
    grid%pb(i,k,j)=100000.-6250.*real(k-1)
    grid%t_2(i,k,j)=2.+.25*real(k)+.001*real(i*j)
    grid%phb(i,k,j)=terrain*9.80616
    grid%ph_2(i,k,j)=15.*real((k-1)**2)*9.80616
    grid%moist(i,k,j,1)=.004+.00001*real(i+j+k)
    grid%moist(i,k,j,2)=.00001*real(modulo(i+j,3))
    grid%moist(i,k,j,3)=.00001*real(modulo(i+j,2))
    grid%al(i,k,j)=.0075*real(k)
    grid%alb(i,k,j)=.9+.025*real(k)
   enddo
  enddo
 enddo
 do j=1,fy
  do i=1,fx
   grid%nfuel_cat(i,j)=real(1+mod(i+j,13))
   grid%fgip(i,j)=.25+.05*real(mod(i+2*j,5))
   grid%fmc_g(i,j)=.05+.0025*real(mod(i+j,5))
   grid%fire_area(i,j)=0.
   if(i<7.and.j<7)grid%fire_area(i,j)=.25
  enddo
 enddo
 rosdt=0.;gen=0;area=0.;risk=0.
 call firebrand_spotting_em_init(grid,cf,pid,psrc,page,px,py,pz,last_gen,idmax,reset, &
  mass,diam,effd,ptemp,tvel,all_count,hist_count,mask,likelihood,fraction)
 ! A carried particle on a nonburning cell exercises deposited history.
 pid(1)=1;psrc(1)=1000001;px(1)=5.75;py(1)=3.75;pz(1)=.1;idmax=1
 mass(1)=.25;diam(1)=10.;effd(1)=10.;ptemp(1)=900.
 write(name,'(a,i0)')'spotting_driver/input_',c
 call oracle_open(trim(name))
 call oracle_put('u',grid%u_2);call oracle_put('v',grid%v_2);call oracle_put('w',grid%w_2)
 call oracle_put('p',grid%p);call oracle_put('pb',grid%pb);call oracle_put('theta_pert',grid%t_2)
 call oracle_put('ph',grid%ph_2);call oracle_put('phb',grid%phb);call oracle_put('moist',grid%moist)
 call oracle_put('al',grid%al);call oracle_put('alb',grid%alb);call oracle_put('msftx',grid%msftx);call oracle_put('muts',grid%muts)
 call oracle_put('msfty',grid%msfty);call oracle_put('rdx',grid%rdx);call oracle_put('rdy',grid%rdy)
 call oracle_put('znw',grid%znw);call oracle_put('c1h',grid%c1h);call oracle_put('c2h',grid%c2h)
 call oracle_put('dnw',grid%dnw);call oracle_put('fnm',grid%fnm);call oracle_put('fnp',grid%fnp)
 call oracle_put('nfuel_cat',grid%nfuel_cat);call oracle_put('fgip',grid%fgip);call oracle_put('fmc_g',grid%fmc_g)
 call oracle_put('fire_area',grid%fire_area)
 call prepare_columns(grid,mh-ml+1,mt-mb+1,nz+1,ml,mb,pu,pv,pw,pp,pt,pd,pz8,pm,p8,ptmass)
 call oracle_put('prepared_u',pu);call oracle_put('prepared_v',pv);call oracle_put('prepared_w',pw)
 call oracle_put('prepared_pressure',pp);call oracle_put('prepared_theta',pt);call oracle_put('prepared_density',pd)
 call oracle_put('prepared_height',pz8);call oracle_put('prepared_msft',pm);call oracle_put('prepared_p8w',p8)
 call oracle_put('prepared_theta_mass',ptmass)
 call save_state()
 call oracle_close()
 do s=1,12
  grid%itimestep=s;grid%domain_clock=s
  do j=1,fy
   do i=1,fx
    grid%burnt_area_dt(i,j)=0.
    if(i<7.and.j<7)grid%burnt_area_dt(i,j)=.015625*real(modulo(i*7+j*3+s,16))
   enddo
  enddo
  call firebrand_spotting_em_driver(cf,grid,pid,psrc,page,px,py,pz,gen,mass,diam,effd,ptemp,tvel, &
   last_gen,idmax,rosdt,area,all_count,hist_count,mask,likelihood,fraction,risk,reset)
  write(name,'(a,i0,a,i0)')'spotting_driver/frame_',c,'_',s
  call oracle_open(trim(name))
  call oracle_put('burnt_area_dt',grid%burnt_area_dt)
  call save_state()
  call oracle_close()
 enddo
enddo
contains
subroutine save_state()
call oracle_put('fs_p_id',pid);call oracle_put('fs_p_src',psrc);call oracle_put('fs_p_dt',page)
call oracle_put('fs_p_x',px);call oracle_put('fs_p_y',py);call oracle_put('fs_p_z',pz)
call oracle_put('fs_p_mass',mass);call oracle_put('fs_p_diam',diam);call oracle_put('fs_p_effd',effd)
call oracle_put('fs_p_temp',ptemp);call oracle_put('fs_p_tvel',tvel)
call oracle_put('fs_gen_inst',gen);call oracle_put('fs_last_gen_dt',last_gen);call oracle_put('fs_gen_idmax',idmax)
call oracle_put('fs_fire_rosdt',rosdt);call oracle_put('fs_fire_area',area)
call oracle_put('fs_count_landed_all',all_count);call oracle_put('fs_count_landed_hist',hist_count)
call oracle_put('fs_landing_mask',mask);call oracle_put('fs_spotting_lkhd',likelihood)
call oracle_put('fs_frac_landed',fraction);call oracle_put('fs_fuel_spotting_risk',risk)
call oracle_put('fs_count_reset',merge(1,0,reset))
end subroutine
end program
