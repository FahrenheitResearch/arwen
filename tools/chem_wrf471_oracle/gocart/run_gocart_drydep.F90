! The routine below is extracted verbatim with sed by build.sh.
! scpr23 only feeds gas depv, which this driver does not use.
module sink_wesely_oracle
  implicit none
  real :: scpr23(4)=1.1
contains
  include 'sink_depvel.inc'
end module sink_wesely_oracle

program run_gocart_drydep
  use module_configure
  use module_state_description
  use module_gocart_drydep
  use sink_wesely_oracle, only: depvel
  use oracle_io
  implicit none
  integer, parameter :: nx=6, ny=6, nz=50
  type(grid_config_rec_type) :: cfg
  real :: t(nx,nz,ny),p(nx,nz,ny),pw(nx,nz,ny),tw(nx,nz,ny),dz(nx,nz,ny),rho(nx,nz,ny)
  real :: moist(nx,nz,ny,num_moist),chem(nx,nz,ny,num_chem)
  real :: rmol(nx,ny), rmol_in(nx,ny), aer(nx,ny), ust(nx,ny),znt(nx,ny),pbl(nx,ny)
  real :: xland(nx,ny),hfx(nx,ny),tsk(nx,ny),veg(nx,ny),lat(nx,ny),lon(nx,ny)
  real :: ddvel(nx,ny,num_chem), accum(nx,ny,5), depvelocity(nx,ny)
  integer :: ivg(nx,ny),i,j,k,r,arm,step,ic,numgas
  real :: dt,depv(4),vgpart,z
  real, parameter :: rms(6)=(/0.,5.e-7,-5.e-7,0.02,-0.02,-0.1/)
  real, parameter :: us(6)=(/0.,0.02,0.099,0.1,0.5,2.0/)
  character(len=1024) :: root
  character(len=80) :: name
  call oracle_root(root)
  dt=36.
  do arm=1,2
    cfg%chem_opt=300
    numgas=p_msa ! module_chem_share.F:41-43
    if (arm==2) cfg%chem_opt=401
    if (arm==2) numgas=0 ! module_chem_share.F:57-58
    do j=1,ny
      do i=1,nx
        ic=i+(j-1)*nx
        rmol(i,j)=rms(i); ust(i,j)=us(j)
        znt(i,j)=0.03+0.005*real(j)
        pbl(i,j)=200.+200.*real(j)
        xland(i,j)=1.
        if (mod(ic,2)==0) xland(i,j)=2.
        hfx(i,j)=250.
        if (mod(ic,3)==0) hfx(i,j)=1.e-7
        if (mod(ic,3)==1) hfx(i,j)=-20.
        tsk(i,j)=290.+real(j); ivg(i,j)=7
        veg(i,j)=25.; lat(i,j)=40.; lon(i,j)=-100.
        do k=1,nz
          z=real(k-1)*150.
          t(i,k,j)=max(230.,tsk(i,j)-0.006*z)
          p(i,k,j)=100000.*exp(-z/8000.); pw(i,k,j)=p(i,k,j)+400.
          tw(i,k,j)=t(i,k,j)+0.2; dz(i,k,j)=60.+real(k)
          rho(i,k,j)=p(i,k,j)/(287.*t(i,k,j))
          moist(i,k,j,:)=0.; moist(i,k,j,p_qv)=0.005
          do r=1,num_chem
            chem(i,k,j,r)=real(r)*0.7*exp(-z/4000.)
          enddo
        enddo
      enddo
    enddo
    ! Make every rmol arm appear in the domain interior as well as edges.
    rmol(2,2)=0.; rmol(2,3)=5.e-7; rmol(2,4)=-5.e-7
    rmol(3,2)=0.02; rmol(3,3)=-0.02; rmol(3,4)=-0.1
    ust(4,2)=0.; ust(4,3)=2.0
    rmol_in=rmol; aer=0.
    if (arm==1) then
      do j=1,ny
        do i=1,nx
          call depvel(4,rmol(i,j),2.0,znt(i,j),ust(i,j),depv,vgpart,aer(i,j))
        enddo
      enddo
    endif
    accum=2.e-10
    do step=1,3
      ddvel=0.; depvelocity=-77.
      write(name,'("arm",I0,"_step",I0)') cfg%chem_opt,step
      call oracle_open(trim(name))
      call oracle_put('chem',chem); call oracle_put('rho',rho)
      call oracle_put('rmol_in',rmol_in); call oracle_put('rmol_used',rmol)
      call oracle_put('aer_res',aer); call oracle_put('ust',ust)
      call oracle_put('znt',znt); call oracle_put('pbl',pbl)
      call oracle_put('hfx',hfx); call oracle_put('xland',xland)
      call oracle_put('dt',dt); call oracle_put('chem_opt',cfg%chem_opt)
      call oracle_put('numgas',numgas)
      call oracle_put('accum_in',accum)
      call gocart_drydep_driver(dt,cfg,numgas,t,moist,pw,tw,rmol,aer,p,chem,rho,dz, &
        ddvel,xland,hfx,ivg,tsk,veg,pbl,ust,znt,lat,lon, &
        accum(:,:,1),accum(:,:,2),accum(:,:,3),accum(:,:,4),accum(:,:,5),depvelocity, &
        1,nx,1,ny,1,nz+1,1,nx,1,ny,1,nz,1,nx,1,ny,1,nz)
      call oracle_put('ddvel',ddvel)
      call oracle_put('accum_out',accum)
      call oracle_put('depvelocity',depvelocity)
      call oracle_close()
    enddo
  enddo
end program run_gocart_drydep
