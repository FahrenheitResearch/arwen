program aerosol_oracle
 use mpas_derived_types
 use init_atm_llxy, only: gausll
 use mpas_init_atm_thompson_aerosols
 implicit none
 integer, parameter :: nx=8, ny=8
 integer :: ng=4
 integer :: nc=16, nv=7
 type(mpas_pool_type) :: mesh, diag, state, configs, lbc
 type(block_type) :: block
 type(dm_info), target :: info
 type(domain_type), target :: domain
 integer, target :: tn,tg,tv,tm=12,tif=1,twf=2
 integer, allocatable, target :: land(:)
 real(RKIND), allocatable, target :: lat(:),lon(:),pressure(:,:),nifa(:,:,:),nwfa(:,:,:),pwif(:,:,:)
 real(RKIND), allocatable, target :: scalars(:,:,:),lbc_scalars(:,:,:)
 real(RKIND), allocatable :: zgrid(:,:),metric(:,:),native_rho_base(:,:)
 character(len=StrKIND), target :: date
 character(len=StrKIND) :: timestamp
 character(len=1024) :: arg,real_input
 character(len=3), parameter :: months(12)=[character(len=3)::'JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
 character(len=24) :: hdate="2000-01-01_00:00:00.0000"
 character(len=32) :: map_source="oracle"
 character(len=25) :: units="nb kg-1"
 character(len=46) :: description="synthetic branch corpus"
 character(len=9) :: field
 real(RKIND) :: slab(nx,ny),gauss(ny),lat1,delta_lat,x,y,pi,deg,ztemp
 ! MPAS mpas_constants.F:46-55; mpas_init_atm_cases.F:5271-5283.
 real(RKIND), parameter :: gravity=9.80616,rgas=287.,t0b=250.,p0=1.e5
 integer :: i,j,k,c,m,f,mode,unit,version,iproj
 logical :: wind
 call get_command_argument(1,arg); read(arg,*) mode
 call get_command_argument(2,date)
 call get_command_argument(3,timestamp)
 call get_command_argument(4,real_input)
 ! Mode 4 consumes the published table linked into the run directory.
 if(mode==4) then
 call get_command_argument(5,arg); read(arg,*) ng
 endif
 tg=ng
 if(len_trim(real_input)>0) then
 open(newunit=unit,file=trim(real_input),access='stream',form='unformatted',status='old')
 read(unit) nc,nv
 endif
 allocate(land(nc),lat(nc),lon(nc),pressure(nv,nc),nifa(12,ng,nc),nwfa(12,ng,nc),pwif(12,ng,nc))
 allocate(scalars(2,nv,nc),lbc_scalars(2,nv,nc))
 if(len_trim(real_input)>0) then
 allocate(zgrid(nv+1,nc),metric(nv,nc),native_rho_base(nv,nc))
 read(unit) lat,lon,land,zgrid,metric,native_rho_base
 close(unit)
 ! Pristine case7 driver slice, mpas_init_atm_cases.F:5274-5277.
 ! The init stream stores zgrid and rho_base, but not diagnostic pressure_base.
 do c=1,nc
 do k=1,nv
 ztemp=0.5*(zgrid(k+1,c)+zgrid(k,c))
 pressure(k,c)=p0*exp(-gravity*ztemp/(rgas*t0b))
 ! Native couples rho_base with the vertical metric at mpas_init_atm_cases.F:5294.
 if(transfer((pressure(k,c)/(rgas*t0b))/metric(k,c),0)/=transfer(native_rho_base(k,c),0)) then
 print *, 'Reconstructed native rho_base differs at cell/level',c,k
 stop 11
 endif
 enddo
 enddo
 deallocate(zgrid,metric,native_rho_base)
 endif
 tn=nc;tv=nv
 domain%dminfo=>info; block%domain=>domain
 mesh%nc=>tn; mesh%ng=>tg; mesh%nv=>tv; mesh%nm=>tm
 mesh%lat=>lat; mesh%lon=>lon; mesh%land=>land
 mesh%nifa=>nifa; mesh%nwfa=>nwfa; mesh%pwif=>pwif
 diag%pressure=>pressure
 state%scalars=>scalars; state%inifa=>tif; state%inwfa=>twf
 lbc%scalars=>lbc_scalars; lbc%inifa=>tif; lbc%inwfa=>twf
 configs%date=>date
 pi=3.141592653589793; deg=180./pi
 call gausll(ny,gauss)
 if(mode==3) gauss=-gauss
 if(mode==1.or.mode==3) then
 iproj=4; lat1=gauss(1); delta_lat=real(ny/2)
 else
 iproj=0; lat1=70.; delta_lat=-20.
 endif
 ! Points include exact nodes, both periodic edges, poles, missing neighbourhoods and -1 mesh mask.
 if(len_trim(real_input)==0) then
 do c=1,nc
 x=1.+mod(c*7,16)*0.5
 y=1.+mod(c*3,14)*0.5
 if(mode==1.or.mode==3) then
 j=min(int(y),ny-1)
 lat(c)=(gauss(j)+(y-j)*(gauss(j+1)-gauss(j)))/deg
 else
 lat(c)=(lat1+(y-1.)*delta_lat)/deg
 endif
 lon(c)=((x-1.)*45.)/deg
 land(c)=mod(c,2)
 pressure(:,c)=[120000.,105000.,75000.,50000.,30000.,10000.,1.]+real(c)*17.
 enddo
 lat(1)=90./deg;lat(2)=-90./deg
 lon(3)=-30./deg;lon(4)=390./deg;lon(5)=337.5/deg
 land(6)=-1
 endif
 if(mode/=4) then
 version=5;wind=.false.
 open(newunit=unit,file='QNWFA_QNIFA_SIGMA_MONTHLY.dat',form='unformatted',convert='big_endian',status='replace')
 do f=0,3
 do m=1,12
 do k=1,ng
 if(f==0.and.(m/=1.or.k/=1.or.mode==2)) cycle
 select case(f)
 case(0);field='LANDSEA'
 case(1);field='QNIFA_'//months(m)
 case(2);field='QNWFA_'//months(m)
 case(3);field='P_WIF_'//months(m)
 end select
 do j=1,ny
 do i=1,nx
 if(f==3) then
 slab(i,j)=120000.-real(k-1)*35000.+real(m*100+i*31+j*17)
 else if(f==0) then
 slab(i,j)=real(mod(i+j,2))
 else
 slab(i,j)=real((f*1000+m*71+k*13+i*7+j*3))/7.
 ! Missing points force the full interpolation sequence, including SEARCH.
 if(k==2.and.(i<=4.or.j<=4)) slab(i,j)=0.
 if(k==3.and.mod(i+j,3)==0) slab(i,j)=0.
 if(k==4) slab(i,j)=0.
 endif
 enddo
 enddo
 write(unit) version
 write(unit) hdate,0.,map_source,field,units,description,real(k),nx,ny,iproj
 write(unit) 'SWCORNER',lat1,0.,delta_lat,45.,6371.229
 write(unit) wind
 write(unit) slab
 enddo
 enddo
 enddo
 close(unit)
 endif
 nifa=0.; nwfa=0.; pwif=0.; scalars=-999.; lbc_scalars=-999.
 call init_atm_thompson_aerosols(block,mesh,configs,diag,state)
 ! Native wire format for Rust replay; no Python decoding/interpolation.
 open(newunit=unit,file='replay.bin',access='stream',form='unformatted',status='replace')
 write(unit) nc,ng,nv,date,timestamp,lat,lon,land,pressure,nifa,nwfa,pwif,scalars
 call init_atm_thompson_aerosols_lbc(date,date,block,mesh,diag,lbc)
 write(unit) lbc_scalars
 call init_atm_thompson_aerosols_lbc(timestamp,date,block,mesh,diag,lbc)
 write(unit) lbc_scalars
 call rename('QNWFA_QNIFA_SIGMA_MONTHLY.dat','saved-climatology')
 scalars=0.; lbc_scalars=321.
 call init_atm_thompson_aerosols(block,mesh,configs,diag,state)
 call init_atm_thompson_aerosols_lbc(timestamp,date,block,mesh,diag,lbc)
 write(unit) scalars,lbc_scalars
 call rename('saved-climatology','QNWFA_QNIFA_SIGMA_MONTHLY.dat')
 close(unit)
end program
