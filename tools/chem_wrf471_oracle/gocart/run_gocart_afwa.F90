program run_gocart_afwa
  use gocart_dust_afwa
  use module_configure
  use module_state_description
  use module_data_gocart_dust, only: ch_dust,porosity
  use oracle_io
  implicit none
  integer,parameter :: nx=38,ny=2,nz=3
  type(grid_config_rec_type) :: cfg
  real :: chem(nx,nz,ny,num_chem), emis(nx,1,ny,num_emis_dust), es(nx,1,ny,num_emis_seas)
  real :: alt(nx,nz,ny),temp(nx,nz,ny),rho(nx,nz,ny),dz(nx,nz,ny),u(nx,nz,ny),v(nx,nz,ny)
  real :: moist(nx,nz,ny,num_moist),smois(nx,4,ny),erod(nx,ny,3),dri(nx,ny,3),dustin(nx,ny,5)
  real :: snow(nx,ny),veg(nx,ny),lai(nx,ny),land(nx,ny),ust(nx,ny),znt(nx,ny)
  real :: clay(nx,ny),sand(nx,ny),clayn(nx,ny),sandn(nx,ny),u10(nx,ny),v10(nx,ny)
  real :: loft(nx,ny),total(nx,nz,ny),te(nx,ny),vis(nx,nz,ny),p(nx,nz,ny),z(nx,nz,ny)
  real :: lat(nx,ny),lon(nx,ny),seasin(nx,ny,5),dt,alpha,gamma,smtune,ustune
  integer :: soil(nx,ny),i,j,k,variant,step,nvariants,nsteps,q
  ! A covering set of switch variants, not the full 192-way factorial (which
  ! made a 101 MB fixture): every dust_smois x sf_surface_physics pair (the
  ! two switches that interact), then each other switch moved alone from a
  ! volumetric Noah base, then all of them moved together.  Columns:
  ! dust_smois, sf_surface_physics, dust_veg, dust_dsr, dust_soils, tuning.
  integer, parameter :: nvar = 14
  integer, parameter :: vtab(6, nvar) = reshape((/ &
      0, 2, 0, 0, 0, 0,   1, 2, 0, 0, 0, 0,   0, 3, 0, 0, 0, 0, &
      1, 3, 0, 0, 0, 0,   0, 7, 0, 0, 0, 0,   1, 7, 0, 0, 0, 0, &
      0, 1, 0, 0, 0, 0,   1, 1, 0, 0, 0, 0,   1, 2, 1, 0, 0, 0, &
      1, 2, 2, 0, 0, 0,   1, 2, 0, 1, 0, 0,   1, 2, 0, 0, 1, 0, &
      1, 2, 0, 0, 0, 1,   1, 3, 2, 1, 1, 1 /), (/ 6, nvar /))
  character(len=1024) :: root
  character(len=80) :: name
  call oracle_root(root)
  ch_dust=0.8D-9
  cfg%dust_opt=3
  cfg%seas_opt=0
  nvariants=nvar
  do variant=0,nvariants-1
    chem=0.25; emis=0.00001; es=7.; te=0.00005
    alt=1./1.2; temp=290.; rho=1.2; dz=20.; u=18.; v=2.; moist=0.; smois=0.02
    erod=0.15; dri=0.2; dustin=0.; snow=0.; veg=0.; lai=1.; land=1.; ust=0.8; znt=0.01
    clay=0.15; sand=0.4; clayn=0.3; sandn=0.2; u10=18.; v10=2.; soil=1
    loft=0.; total=0.; vis=0.; lat=30.; lon=0.; seasin=0.
    do j=1,ny
      do i=1,nx
        soil(i,j)=mod(i-1,19)+1
        smois(i,1,j)=porosity(soil(i,j))*0.1
        if(j==2) smois(i,1,j)=porosity(soil(i,j))*0.7
        if(i==1) smois(i,1,j)=0.
        if(i==2) smois(i,1,j)=porosity(soil(i,j))*0.5
        if(i==3 .or. soil(i,j)==14) land(i,j)=2.
        if(i==4) u10(i,j)=0.
        if(i==5) u10(i,j)=3.
        if(i==6) u10(i,j)=25.
        if(i==7) dz(i,:,j)=8.
        if(i==8) erod(i,j,:)=0.
        if(i==9) erod(i,j,1)=0.
        if(i==10) erod(i,j,2)=0.
        if(i==11) erod(i,j,3)=0.
        if(i==12) snow(i,j)=0.05
        if(i==13) veg(i,j)=10.
        if(i==20) lai(i,j)=0.
        if(i==15) znt(i,j)=0.3
        if(i==16) dri(i,j,1)=-1.
        if(i==17) clayn(i,j)=-1.
        if(i==18) ust(i,j)=0.1
        if(i==19) ust(i,j)=0.
        do k=1,nz
          p(i,k,j)=100000.-1000.*(k-1)
          z(i,k,j)=20.*(k-1)
        enddo
      enddo
    enddo
    do j=1,ny
      do k=1,nz
        do i=1,nx
          rho(i,k,j)=1.25-0.08*(k-1)+0.001*i
          do q=1,num_chem
            chem(i,k,j,q)=0.02*q+0.003*k+0.001*i
          enddo
        enddo
      enddo
    enddo
    chem(3,:,:,p_dust_1:p_dust_5)=0.
    chem(22,2:nz,:,p_dust_1:p_dust_5)=0.
    dt=30.; alpha=1.; gamma=1.; smtune=1.; ustune=1.
    cfg%dust_smois=vtab(1,variant+1)
    cfg%sf_surface_physics=vtab(2,variant+1)
    cfg%dust_veg=vtab(3,variant+1)
    cfg%dust_dsr=vtab(4,variant+1)
    cfg%dust_soils=vtab(5,variant+1)
    if(vtab(6,variant+1)==1) then
      alpha=0.7; gamma=1.3; smtune=0.8; ustune=1.2
    endif
    ! Three consecutive steps on the base variant carry the accumulators;
    ! the other variants take one step.
    nsteps=1
    if(variant==0) nsteps=3
    do step=1,nsteps
      write(name,'("variant_",I3.3,"_step_",I1)') variant,step
      call oracle_open(trim(name))
      call oracle_put('alt',alt)
      call oracle_put('t_phy',temp)
      call oracle_put('moist',moist)
      call oracle_put('xlat',lat)
      call oracle_put('xlong',lon)
      call oracle_put('dustin_before',dustin)
      call oracle_put('seasin_before',seasin)
      call oracle_put('step',step)
      call oracle_put('start_month',cfg%start_month)
      call oracle_put('num_soil_layers',cfg%num_soil_layers)
      call oracle_put('chem_before',chem)
      call oracle_put('edust_before',emis)
      call oracle_put('eseas_before',es)
      call oracle_put('tot_edust_before',te)
      call oracle_put('xland',land)
      call oracle_put('isltyp',soil)
      call oracle_put('u10',u10)
      call oracle_put('v10',v10)
      call oracle_put('smois',smois)
      call oracle_put('erod',erod)
      call oracle_put('erod_dri',dri)
      call oracle_put('rho_phy',rho)
      call oracle_put('dz8w',dz)
      call oracle_put('u_phy',u)
      call oracle_put('v_phy',v)
      call oracle_put('snowh',snow)
      call oracle_put('vegfra',veg)
      call oracle_put('lai_vegmask',lai)
      call oracle_put('ust',ust)
      call oracle_put('znt',znt)
      call oracle_put('clay_wrf',clay)
      call oracle_put('sand_wrf',sand)
      call oracle_put('clay_nga',clayn)
      call oracle_put('sand_nga',sandn)
      call oracle_put('p8w',p)
      call oracle_put('z_at_w',z)
      call oracle_put('dt',dt)
      call oracle_put('g',9.81)
      call oracle_put('dx',6000.)
      call oracle_put('alpha',alpha)
      call oracle_put('gamma',gamma)
      call oracle_put('smtune',smtune)
      call oracle_put('ustune',ustune)
      call oracle_put('switches',(/cfg%dust_dsr,cfg%dust_veg,cfg%dust_soils,cfg%dust_smois,cfg%sf_surface_physics/))
      call oracle_put('chem_opt',cfg%chem_opt)
      call oracle_put('dust_opt',cfg%dust_opt)
      call oracle_put('seas_opt',cfg%seas_opt)
      call gocart_dust_afwa_driver(dt,cfg,alt,chem,rho,smois,u10,v10,dz,erod,dri,dustin,snow, &
        soil,veg,lai,land,9.81,emis,ust,znt,clay,sand,clayn,sandn,loft,total,te,vis,alpha,gamma,smtune,ustune, &
        1,nx,1,ny,1,nz,1,nx,1,ny,1,nz,1,nx,1,ny,1,nz)
      call oracle_put('chem_after',chem)
      call oracle_put('edust_after',emis)
      call oracle_put('eseas_after',es)
      call oracle_put('afwa_dustloft',loft)
      call oracle_put('tot_dust',total)
      call oracle_put('tot_edust_after',te)
      call oracle_put('vis_dust',vis)
      call oracle_close()
    enddo
  enddo
end program
