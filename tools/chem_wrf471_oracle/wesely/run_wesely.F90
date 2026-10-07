! Driver calls chem/module_dep_simple.F:71,1618 without physics stubs.
program run_wesely
 use module_dep_simple
 use module_state_description
 use module_configure
 use oracle_io
 implicit none
 integer,parameter::nx=24,ny=18,nz=2,ng=9
 type(grid_config_rec_type)::cfg
 real::t(nx,nz,ny),p(nx,nz,ny),pw(nx,nz,ny),tw(nx,nz,ny),rho(nx,nz,ny)
 real::dz(nx,nz,ny),zlev(nx,nz,ny),zw(nx,nz,ny),moist(nx,nz,ny,4),chem(nx,nz,ny,ng)
 real::vel(nx,ny,ng),adef(nx,ny),acen(nx,ny),saved(nx,ny,ng)
 real::skin(nx,ny),sw(nx,ny),veg(nx,ny),pbl(nx,ny),mol(nx,ny),fric(nx,ny)
 real::rough(nx,ny),rain(nx,ny),lat(nx,ny),lon(nx,ny),snow(nx,ny)
 integer::land(nx,ny),i,j,c,day,nrow,pass
 real::temp(9)=(/-10.,-5.,-3.,-1.,0.,1.,2.,3.,45./)
 character(len=48)::case_name,dataset
 do c=1,11
  dataset='USGS'
  cfg%num_land_cat=25
  if(c==5.or.c==6) then
   dataset='MODIFIED_IGBP_MODIS_NOAH'
   cfg%num_land_cat=20
  endif
  day=180
  if(c==2.or.c==6.or.c==8)day=89
  if(c==3)day=270
  if(c==4)day=271
  if(c==1)day=90
  nrow=4
  p_nh3=1
  if(c==7.or.c==8)then
   nrow=6
   p_nh3=6
  endif
  if(allocated(luse2usgs))deallocate(luse2usgs)
  call dep_init(1,cfg,ng,trim(dataset),1,nx,1,ny,nx+1,ny+1)
  write(case_name,'(a,i2.2)')'case_',c
  call oracle_open(trim(case_name))
  call oracle_put('julday',day)
  call oracle_put('nrow',nrow)
  call oracle_put('map_kind',merge(1,0,c==5.or.c==6))
  call oracle_put('ri',ri)
  call oracle_put('rlu',rlu)
  call oracle_put('rac',rac)
  call oracle_put('rgss',rgss)
  call oracle_put('rgso',rgso)
  call oracle_put('rcls',rcls)
  call oracle_put('rclo',rclo)
  call oracle_put('kpart',kpart)
  call oracle_put('ixxxlu',ixxxlu)
  call oracle_put('luse2usgs',luse2usgs)
  call oracle_put('iswater_temp',iswater_temp)
  call oracle_put('isice_temp',isice_temp)
  call oracle_put('hstar',hstar(2:7))
  call oracle_put('dhr',dhr(2:7))
  call oracle_put('f0',f0(2:7))
  call oracle_put('dratio',dratio(2:7))
  call oracle_put('scpr23',scpr23(2:7))
  t=290.;p=98000.;pw=100000.;tw=290.;rho=1.;dz=50.;zlev=25.
  moist=0.;chem=.01;chem(:,:,:,6)=.03
  chem(1:nx:2,:,:,6)=.005
  zw(:,1,:)=0.;zw(:,2,:)=50.
  veg=50.;pbl=500.;rain=0.;lat=35.;lon=0.;snow=0.
  do j=1,ny
   do i=1,nx
    land(i,j)=mod(i-1,cfg%num_land_cat)+1
    skin(i,j)=273.15+temp(mod(j-1,9)+1)
    sw(i,j)=0.
    if(mod(j,3)==1)sw(i,j)=50.
    if(mod(j,3)==2)sw(i,j)=800.
    mol(i,j)=0.
    if(mod(i+j,3)==0)mol(i,j)=.02
    if(mod(i+j,3)==1)mol(i,j)=-.02
    if(i==1)mol(i,j)=5.e-7
    fric(i,j)=.05
    if(mod(j,2)==0)fric(i,j)=1.2
    rough(i,j)=.001
    if(mod(i,2)==0)rough(i,j)=1.
    moist(i,1,j,p_qv)=.001
    if(mod(i+j,8)==0)moist(i,1,j,p_qv)=.1
    if(mod(i+j,8)==1)moist(i,1,j,p_qr)=1.e-5
    if(mod(i+j,8)==2)rain(i,j)=.1
    if(mod(i+j,8)==3)moist(i,1,j,p_qv)=.85*(3.80*exp(17.27*(290.-273.)/(290.-36.))/(.01*98000.))
    if(mod(i+j,8)==4)moist(i,1,j,p_qv)=.80*(3.80*exp(17.27*(290.-273.)/(290.-36.))/(.01*98000.))
    if(i==1)fric(i,j)=1.e-5
    moist(i,1,j,p_qc)=1.e-4
   enddo
  enddo
  ! Three calls A,B,A, chem/module_dep_simple.F:178-257 has no carry.
  if(c==10)sw=sw+100.
  vel=-999.;adef=-999.;acen=-999.
  call invoke()
  saved=vel
  if(c==11)then
   sw=sw+100.
   call invoke()
   call oracle_put('changed_ddvel',vel(:,:,2:5))
   sw=sw-100.
   call invoke()
   if(any(vel/=saved))error stop 'scheme unexpectedly retained memory'
  endif
  call oracle_put('t_phy_k1',t(:,1,:))
  call oracle_put('p_phy_k1',p(:,1,:))
  call oracle_put('p8w_k1',pw(:,1,:))
  call oracle_put('qv_k1',moist(:,1,:,p_qv))
  call oracle_put('qc_k1',moist(:,1,:,p_qc))
  call oracle_put('qr_k1',moist(:,1,:,p_qr))
  call oracle_put('dz1',zw(:,2,:)-zw(:,1,:))
  call oracle_put('tsk',skin)
  call oracle_put('gsw',sw)
  call oracle_put('vegfra',veg)
  call oracle_put('ust',fric)
  ! Original met, before WRF snaps |rmol|<1e-6 to zero.
  where(abs(mol)<1.e-6)mol=5.e-7
  call oracle_put('rmol',mol)
  call oracle_put('znt',rough)
  call oracle_put('raincv',rain)
  call oracle_put('ivgtyp',land)
  call oracle_put('chem_k1',chem(:,1,:,2:1+nrow))
  call oracle_put('ddvel',vel(:,:,2:1+nrow))
  call oracle_put('aer_res_def',adef)
  call oracle_put('aer_res_zcen',acen)
 enddo
contains
 subroutine invoke()
 call wesely_driver(1,1,30.,cfg,6,12.,day,t,moist,pw,tw,rain,p,chem,rho,dz, &
 vel,adef,acen,land,skin,sw,veg,pbl,mol,fric,rough,lat,lon,zlev,zw,snow,ng, &
 1,nx+1,1,ny+1,1,nz,1,nx,1,ny,1,nz,1,nx,1,ny,1,nz)
 end subroutine
end program
