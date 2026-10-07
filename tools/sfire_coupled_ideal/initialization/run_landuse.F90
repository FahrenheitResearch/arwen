program run_landuse
use native_landuse_control
use oracle_io
implicit none
integer,parameter::nx=9,ny=7
real::lu(nx,ny),snow(nx,ny),xice(nx,ny),snoalb(nx,ny),albedo(nx,ny),albbck(nx,ny),mavail(nx,ny),emiss(nx,ny),embck(nx,ny)
real::znt(nx,ny),z0(nx,ny),thc(nx,ny),xland(nx,ny),xicem(nx,ny),state(7501),table(7,100,12),cenlat
integer::isice,cats,seas,season,fractional,day,iswater,c,i,j,k,l,cursor,reuse
logical::monalb,negative,correct_zero
character(len=1024)::root
character(len=64)::name,arg
call oracle_root(root);call get_command_argument(2,arg);negative=trim(arg)=='--zero-water'
correct_zero=trim(arg)=='--correct-zero'
do c=1,17
 if(c==17.and..not.negative.and..not.correct_zero)exit
 if(c/=17.and.(negative.or.correct_zero))cycle
 fractional=mod(c-1,2);monalb=mod((c-1)/2,2)==1;day=150;cenlat=40.
 if(mod((c-1)/4,2)==1)day=50
 if(c>8)cenlat=-40.
 iswater=16;if(c>=15)iswater=0
 do j=1,ny
  do i=1,nx
   lu(i,j)=float(mod(i+3*j,24)+1)
   if(mod(i+j,3)==0)lu(i,j)=2.5
   if(mod(i+2*j,7)==0)lu(i,j)=16.
   if(c<15.and.mod(i+j,9)==0)lu(i,j)=0.
   snow(i,j)=merge(.75,0.,mod(i+j,4)==0)
   xice(i,j)=0.
   select case(mod(i+j,6))
    case(0);xice(i,j)=1.
    case(1);xice(i,j)=.5
    case(2);xice(i,j)=.02
    case(3);xice(i,j)=.01
   end select
   snoalb(i,j)=.8;albbck(i,j)=.215
  enddo
 enddo
 if(c==17)lu(1,1)=0.
 if(correct_zero)lu(1,1)=28.
 state=0.;isice=24;cats=0;seas=0
 call landuse_init(lu,snow,albedo,albbck,snoalb,mavail,emiss,embck,znt,z0,thc,xland,xice,xicem,day,cenlat,iswater,'USGS', &
   isice,cats,seas,season,fractional,state,.true.,monalb,1,nx+1,1,ny+1,1,2,1,nx,1,ny,1,2,1,nx,1,ny,1,2)
 table=0.;cursor=1
 do k=1,100
  table(6,k,seas)=state(cursor);cursor=cursor+1
  do l=1,12
   table(1,k,l)=state(cursor);table(2,k,l)=state(cursor+1);table(3,k,l)=state(cursor+2);table(4,k,l)=state(cursor+3)
   table(7,k,l)=state(cursor+4);table(5,k,l)=state(cursor+5);cursor=cursor+6
  enddo
 enddo
 do reuse=0,1
  if(reuse==1)call landuse_init(lu,snow,albedo,albbck,snoalb,mavail,emiss,embck,znt,z0,thc,xland,xice,xicem,day,cenlat,iswater,'USGS', &
    isice,cats,seas,season,fractional,state,.false.,monalb,1,nx+1,1,ny+1,1,2,1,nx,1,ny,1,2,1,nx,1,ny,1,2)
  write(name,'(a,i0,a,i0)')'landuse/case_',c,'_reuse_',reuse
  call oracle_open(trim(name));call oracle_put('lu',lu);call oracle_put('snow',snow);call oracle_put('xice',xice)
  call oracle_put('snoalb',snoalb);call oracle_put('initial_albbck',.215);call oracle_put('table',table(:,:cats,:seas))
  call oracle_put('julday',day);call oracle_put('cen_lat',cenlat);call oracle_put('iswater',iswater);call oracle_put('isice',isice)
  call oracle_put('fractional',fractional);call oracle_put('monalb',merge(1,0,monalb));call oracle_put('season',season)
  call oracle_put('albedo',albedo);call oracle_put('albbck',albbck);call oracle_put('mavail',mavail);call oracle_put('emiss',emiss)
  call oracle_put('embck',embck);call oracle_put('znt',znt);call oracle_put('z0',z0);call oracle_put('thc',thc)
  call oracle_put('xland',xland);call oracle_put('xicem',xicem);call oracle_close()
 enddo
enddo
end program
