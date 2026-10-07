program run_gocart_sulfur
  use module_gocart_chem
  use module_configure
  use module_state_description
  use oracle_io
  implicit none
  integer, parameter :: nx=8, nz=5, ny=1
  type(grid_config_rec_type) :: config
  real :: t(nx,nz,ny),rho(nx,nz,ny),dz(nx,nz,ny),pw(nx,nz,ny)
  real :: moist(nx,nz,ny,num_moist),chem(nx,nz,ny,num_chem)
  real :: oh(nx,nz,ny),h2o2(nx,nz,ny),no3(nx,nz,ny),cloud(nx,nz,ny)
  real :: lat(nx,ny),lon(nx,ny),tcosz(nx,ny),ttday(nx,ny),cosz(nx,ny)
  real :: sza(1,1),cx(1,1),rlat,gmt,dt,gmtp,xmin
  real(kind=8) :: secs,xtime,xhour
  integer(kind=8) :: ixhour
  real, parameter :: dts(4)=[30.,36.,60.,7.5]
  integer :: id,clock,zero,optional_cloud,i,k,step,julday
  integer :: branches(nx,nz,ny)
  character(len=1024) :: root
  character(len=100) :: case_name,key
  call oracle_root(root)
  lat(:,1)=[-80.,-65.,-40.,0.,35.,65.,80.,20.]
  lon(:,1)=[-180.,-90.,0.,45.,90.,135.,170.,-45.]
  do id=1,4
  do clock=1,3
  do zero=0,1
  do optional_cloud=0,1
    dt=dts(id)
    select case(clock)
    case(1)
      julday=172; gmt=12.
    case(2)
      julday=355; gmt=0.
    case(3)
      julday=80; gmt=23.75
    end select
    call init_solar(dt,gmt,julday,lat,lon,tcosz,ttday)
    chem=0.; moist=0.; cloud=0.
    do k=1,nz
    do i=1,nx
      t(i,k,1)=290.-8.*real(k-1)
      if (mod(i+k,4)==0) t(i,k,1)=250.
      rho(i,k,1)=1.2-0.15*real(k-1)
      dz(i,k,1)=200.+100.*real(k-1)
      pw(i,k,1)=100000.-8000.*real(k-1)
      oh(i,k,1)=4.e-14*(1.+0.1*real(i))
      no3(i,k,1)=2.e-12*(1.+0.1*real(k))
      h2o2(i,k,1)=1.e-8
      if(mod(i,2)==0) h2o2(i,k,1)=2.e-10
      if(mod(i+k,3)==0) cloud(i,k,1)=0.4
      if(mod(i+k,5)==0) moist(i,k,1,p_qc)=1.e-4
      if(mod(i+k,7)==0) moist(i,k,1,p_qi)=2.e-5
      chem(i,k,1,p_dms)=0.002+0.0001*real(i)
      chem(i,k,1,p_so2)=0.005+0.0002*real(k)
      chem(i,k,1,p_sulf)=0.001+0.0001*real(i)
      chem(i,k,1,p_msa)=0.00005
    enddo
    enddo
    if(zero==1) then
      chem(:,:,:,p_dms)=0.; chem(:,:,:,p_msa)=0.
    endif
    write(case_name,'("dt",I0,"_clock",I0,"_zero",I0,"_cloud",I0)') id,clock,zero,optional_cloud
    call oracle_open(trim(case_name))
    call oracle_put('dt',dt); call oracle_put('gmt',gmt); call oracle_put('julday',julday)
    call oracle_put('optional_cloud',optional_cloud); call oracle_put('zero_roles',zero)
    call oracle_put('latitude',lat); call oracle_put('longitude',lon)
    call oracle_put('tcosz',tcosz); call oracle_put('ttday',ttday)
    call oracle_put('temp',t); call oracle_put('rho',rho)
    call oracle_put('qc',moist(:,:,:,p_qc)); call oracle_put('qi',moist(:,:,:,p_qi))
    call oracle_put('gd_cldf',cloud)
    call oracle_put('backg_oh',oh); call oracle_put('backg_h2o2',h2o2); call oracle_put('backg_no3',no3)
    call dump('input')
    do step=1,3
      secs=real(step-1,8)*real(dt,8)
      if(clock==3) secs=899._8+secs
      write(key,'("curr_secs_",I0)') step
      call oracle_put(trim(key),real(secs))
      xtime=secs/60._8
      ixhour=int(gmt+.01,8)+int(xtime/60._8,8)
      xhour=real(ixhour,8)
      xmin=60.*gmt+real(xtime-xhour*60._8,8)
      gmtp=mod(xhour,24._8)
      gmtp=gmtp+xmin/60.
      do i=1,nx
        rlat=lat(i,1)*3.1415926535590/180.
        call szangle(1,1,julday,gmtp,sza,cx,lon(i,1),rlat)
        cosz(i,1)=cx(1,1)
      enddo
      write(key,'("cossza_",I0)') step
      call oracle_put(trim(key),cosz)
      branches=0
      do k=1,nz-1
      do i=1,nx
        if(t(i,k,1)<=258.) cycle
        branches(i,k,1)=1
        if((optional_cloud==1.and.cloud(i,k,1)>0.).or. &
            moist(i,k,1,p_qc)>0..or.moist(i,k,1,p_qi)>0.) then
          branches(i,k,1)=2
          if(chem(i,k,1,p_so2)*1.d-6<=h2o2(i,k,1)) branches(i,k,1)=3
        endif
      enddo
      enddo
      write(key,'("branch_input_",I0)') step
      call oracle_put(trim(key),branches)
      if(optional_cloud==1) then
        call gocart_chem_driver(secs,dt,config,gmt,julday,t,moist,chem,rho,dz,pw,oh,h2o2,no3, &
            cloud,3000.,9.81,lat,lon,ttday,tcosz, &
            1,nx+1,1,ny+1,1,nz+1, 1,nx,1,ny,1,nz, 1,nx,1,ny,1,nz)
      else
        call gocart_chem_driver(curr_secs=secs,dt=dt,config_flags=config,gmt=gmt,julday=julday, &
            t_phy=t,moist=moist,chem=chem,rho_phy=rho,dz8w=dz,p8w=pw, &
            backg_oh=oh,backg_h2o2=h2o2,backg_no3=no3,dx=3000.,g=9.81,xlat=lat,xlong=lon, &
            ttday=ttday,tcosz=tcosz,ids=1,ide=nx+1,jds=1,jde=ny+1,kds=1,kde=nz+1, &
            ims=1,ime=nx,jms=1,jme=ny,kms=1,kme=nz,its=1,ite=nx,jts=1,jte=ny,kts=1,kte=nz)
      endif
      write(key,'("step",I0)') step
      call dump(trim(key))
    enddo
    call oracle_close()
  enddo
  enddo
  enddo
  enddo
contains
  subroutine dump(prefix)
    character(len=*),intent(in) :: prefix
    call oracle_put(trim(prefix)//'_dms',chem(:,:,:,p_dms))
    call oracle_put(trim(prefix)//'_so2',chem(:,:,:,p_so2))
    call oracle_put(trim(prefix)//'_so4',chem(:,:,:,p_sulf))
    call oracle_put(trim(prefix)//'_msa',chem(:,:,:,p_msa))
  end subroutine
  subroutine init_solar(dt,gmt,julday,xlat,xlong,tcosz,ttday)
    real,intent(in) :: dt,gmt,xlat(nx,ny),xlong(nx,ny)
    integer,intent(in) :: julday
    real,intent(out) :: tcosz(nx,ny),ttday(nx,ny)
    integer :: i,j,n,ndystep,ixhour,its,ite,jts,jte
    real :: xtime,xhour,xmin,gmtp,xlonn,rlat,sza(1,1),cosszax(1,1)
    its=1; ite=nx; jts=1; jte=ny
    ! The exact extract begins with ENDIF and ends before the outer ENDDO.
    if(.false.) then
    include 'chemics_init_solar.inc'
    enddo
  end subroutine
end program
