program run_geometry
use ideal_geometry_control
use ideal_geometry_control_original,only:original_soil=>process_soil_ideal
use module_fr_fire_util,only:set_ideal_coord,continue_at_boundary,interpolate_2d
use oracle_io
implicit none
type(grid_record)::grid
type(configuration_record)::cfg
type(model_record)::model
integer::nz,c,stretch,hybrid,k,m,scheme,layers,i,j
real::scale,scalars(7),dx,dy,fdx,fdy,zs(9),dzs(9),original_dzs(9),xs,ys,xe,ye
real::tsk(8,7),tmn(8,7),tslb(8,9,7),smois(8,9,7),xland(8,7),xice(8,7),vegfra(8,7),snow(8,7),canwat(8,7)
integer::ivgtyp(8,7),isltyp(8,7)
integer::xl,xh,yl,yh
real::coordsx(11,9),coordsy(11,9),coarse(0:12,0:10),fine0(33,36),fine1(33,36),finefixed(33,36)
character(len=1024)::root
character(len=80)::name
call oracle_root(root)
c=0
do m=1,3
 nz=7+m*5
 allocate(grid%znw(nz+1),grid%znu(nz+1),grid%dnw(nz+1),grid%rdnw(nz+1),grid%dn(nz+1),grid%rdn(nz+1))
 allocate(grid%fnp(nz+1),grid%fnm(nz+1),grid%c1f(nz+1),grid%c2f(nz+1),grid%c3f(nz+1),grid%c4f(nz+1))
 allocate(grid%c1h(nz+1),grid%c2h(nz+1),grid%c3h(nz+1),grid%c4h(nz+1))
 do stretch=0,3
  do hybrid=0,3
   c=c+1;write(name,'(a,i0)')'ideal_geometry/vertical_',c
   call oracle_open(trim(name))
   grid%znw=0.;grid%znu=0.;grid%dnw=0.;grid%rdnw=0.;grid%dn=0.;grid%rdn=0.
   grid%fnp=0.;grid%fnm=0.;grid%c1f=0.;grid%c2f=0.;grid%c3f=0.;grid%c4f=0.
   grid%c1h=0.;grid%c2h=0.;grid%c3h=0.;grid%c4h=0.
   cfg%dx=90.;cfg%dy=73.5;cfg%hybrid_opt=hybrid;grid%p_top=39750.625;grid%etac=.32
   scale=.4;if(m==2)scale=1.5;if(m==3)scale=8.3
   model%eta_levels=-1.;model%e_vert(1)=nz+1
   if(stretch==3)then
    do k=1,nz+1
     model%eta_levels(k)=1.-(float(k-1)/float(nz))**2
    enddo
   endif
   call initialize_vertical(grid,nz,stretch>0,stretch==2,scale,cfg,model)
   scalars=(/grid%cf1,grid%cf2,grid%cf3,grid%cfn,grid%cfn1,grid%rdx,grid%rdy/)
   call oracle_put('nz',nz);call oracle_put('stretch',stretch);call oracle_put('hybrid',hybrid)
   call oracle_put('scale',scale);call oracle_put('etac',grid%etac);call oracle_put('p_top',grid%p_top)
   call oracle_put('dx',cfg%dx);call oracle_put('dy',cfg%dy);call oracle_put('scalars',scalars)
   call oracle_put('eta_in',model%eta_levels(:nz+1))
   call oracle_put('znw',grid%znw);call oracle_put('znu',grid%znu(:nz));call oracle_put('dnw',grid%dnw(:nz));call oracle_put('rdnw',grid%rdnw(:nz))
   call oracle_put('dn',grid%dn(:nz));call oracle_put('rdn',grid%rdn(:nz));call oracle_put('fnp',grid%fnp(:nz));call oracle_put('fnm',grid%fnm(:nz))
   call oracle_put('c1f',grid%c1f);call oracle_put('c2f',grid%c2f);call oracle_put('c3f',grid%c3f);call oracle_put('c4f',grid%c4f)
   call oracle_put('c1h',grid%c1h(:nz));call oracle_put('c2h',grid%c2h(:nz));call oracle_put('c3h',grid%c3h(:nz));call oracle_put('c4h',grid%c4h(:nz))
   call oracle_close()
  enddo
 enddo
 deallocate(grid%znw,grid%znu,grid%dnw,grid%rdnw,grid%dn,grid%rdn,grid%fnp,grid%fnm)
 deallocate(grid%c1f,grid%c2f,grid%c3f,grid%c4f,grid%c1h,grid%c2h,grid%c3h,grid%c4h)
enddo
grid%dx=73.5;grid%dy=90.;fdx=grid%dx/3.;fdy=grid%dy/4.
allocate(grid%ht(11,9),grid%zsf(0:34,0:37),grid%dzdxf(33,36),grid%dzdyf(33,36))
do m=1,5
 k=m;xs=50.;ys=-125.;xe=650.;ye=575.
 if(m==4)then
  k=2;xs=100.;xe=100.
 endif
 if(m==5)then
  k=3;ys=100.;ye=100.
 endif
 write(name,'(a,i0)')'ideal_geometry/mountain_',m
 call oracle_open(trim(name));grid%ht=0.;grid%zsf=0.
 call initialize_mountain(grid,11,9,33,36,fdx,fdy,k,xs,ys,xe,ye,375.)
 call oracle_put('kind',k);call oracle_put('dx',grid%dx);call oracle_put('dy',grid%dy)
 call oracle_put('xs',xs);call oracle_put('ys',ys);call oracle_put('xe',xe);call oracle_put('ye',ye)
 call oracle_put('fdx',fdx);call oracle_put('fdy',fdy);call oracle_put('height',grid%ht);call oracle_put('zsf',grid%zsf(1:33,1:36))
 call oracle_close()
 call initialize_gradient(grid,33,36,fdx,fdy)
 write(name,'(a,i0)')'ideal_geometry/gradient_',m
 call oracle_open(trim(name));call oracle_put('fdx',fdx);call oracle_put('fdy',fdy)
 call oracle_put('zsf',grid%zsf(1:33,1:36));call oracle_put('gx',grid%dzdxf);call oracle_put('gy',grid%dzdyf)
 call oracle_close()
enddo
call set_ideal_coord(grid%dx,grid%dy,1,11,1,9,1,11,1,9,1,11,1,9,coordsx,coordsy)
call oracle_open('ideal_geometry/coordinates');call oracle_put('dx',grid%dx);call oracle_put('dy',grid%dy)
call oracle_put('x',coordsx);call oracle_put('y',coordsy);call oracle_close()
coarse=0.;coarse(1:11,1:9)=grid%ht;fine0=0.;fine1=-999.
call interpolate_2d(0,12,0,10,1,11,1,9,1,33,1,36,1,33,1,36,3,4,1.,1.,2.,2.5,coarse,fine0)
call interpolate_2d(0,12,0,10,1,11,1,9,1,33,1,36,1,33,1,36,3,4,1.,1.,2.,2.5,coarse,fine1)
call continue_at_boundary(1,1,0.,0,12,0,10,1,11,1,9,1,11,1,9,1,11,1,9,xl,xh,yl,yh,coarse)
finefixed=-999.
call interpolate_2d(0,12,0,10,0,12,0,10,1,33,1,36,1,33,1,36,3,4,1.,1.,2.,2.5,coarse,finefixed)
call oracle_open('ideal_geometry/interpolation_edges');call oracle_put('coarse',grid%ht)
call oracle_put('original_zero',fine0);call oracle_put('original_sentinel',fine1);call oracle_put('corrected',finefixed)
call oracle_close()
do m=1,5
 scheme=m;if(m==5)scheme=3
 layers=4;if(scheme==1)layers=5;if(scheme==3)layers=6
 if(m==5)layers=9
 do j=1,7
  do i=1,8
   tsk(i,j)=280.+float(i+j)*.375;tmn(i,j)=275.+float(i-j)*.25
  enddo
 enddo
 xland=1.;xice=0.;vegfra=.5;snow=0.;canwat=0.;ivgtyp=18;isltyp=7;zs=0.;dzs=0.;tslb=0.;smois=0.
 call original_soil(xland,xice,vegfra,snow,canwat,ivgtyp,isltyp,tslb(:,:layers,:),smois(:,:layers,:), &
  tsk,tmn,zs(:layers),dzs(:layers),layers,scheme,1,9,1,8,1,2,1,8,1,7,1,2,1,8,1,7,1,2)
 original_dzs=dzs
 call process_soil_ideal(xland,xice,vegfra,snow,canwat,ivgtyp,isltyp,tslb(:,:layers,:),smois(:,:layers,:), &
  tsk,tmn,zs(:layers),dzs(:layers),layers,scheme,1,9,1,8,1,2,1,8,1,7,1,2,1,8,1,7,1,2)
 write(name,'(a,i0)')'ideal_geometry/soil_',m
 call oracle_open(trim(name));call oracle_put('scheme',scheme);call oracle_put('layers',layers)
 call oracle_put('tsk',tsk);call oracle_put('tmn',tmn);call oracle_put('zs',zs(:layers));call oracle_put('dzs',dzs(:layers))
 call oracle_put('original_dzs',original_dzs(:layers))
 call oracle_put('tslb',tslb(:,:layers,:));call oracle_put('smois',smois(:,:layers,:));call oracle_close()
enddo
end program
