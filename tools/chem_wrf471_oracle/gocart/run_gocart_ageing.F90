program run_gocart_ageing
  use module_gocart_aerosols
  use module_configure
  use module_state_description
  use oracle_io
  implicit none
  integer,parameter :: nx=8,nz=4,ny=2
  type(grid_config_rec_type) :: config
  real :: t(nx,nz,ny),rho(nx,nz,ny),dz(nx,nz,ny),pw(nx,nz,ny)
  real :: moist(nx,nz,ny,num_moist),chem(nx,nz,ny,num_chem)
  real,parameter :: dts(7)=[30.,36.,60.,7.5,1.9,3600.5,0.5]
  real,parameter :: concentrations(8)=[0.,1.e-26,1.e-22,1.e-6,0.01,1.,30.,1000.]
  real :: dt
  integer :: id,i,j,k,step
  character(len=1024) :: root
  character(len=80) :: key,case_name
  call oracle_root(root)
  t=280.; rho=1.; dz=300.; moist=0.
  do k=1,nz
    pw(:,k,:)=100000.-10000.*real(k-1)
  enddo
  do id=1,size(dts)
    dt=dts(id); chem=0.
    do j=1,ny
    do k=1,nz
    do i=1,nx
      chem(i,k,j,p_bc1)=concentrations(i)*real(k)
      chem(i,k,j,p_oc1)=concentrations(i)*2.*real(k)
      chem(i,k,j,p_bc2)=concentrations(i)*real(j)*0.2
      chem(i,k,j,p_oc2)=concentrations(i)*real(j)*0.3
    enddo
    enddo
    enddo
    ! A floor in a hydrophilic row changes tt2 even when its source is zero.
    chem(3,1,1,p_bc2)=0.; chem(3,1,1,p_oc2)=0.
    write(case_name,'("dt",I0)') id
    call oracle_open(trim(case_name))
    call oracle_put('dt',dt); call dump('input')
    do step=1,3
      call gocart_aerosols_driver(step,dt,config,t,moist,chem,rho,dz,pw,3000.,9.81, &
           1,nx+1,1,ny+1,1,nz+1, 1,nx,1,ny,1,nz, 1,nx,1,ny,1,nz)
      write(key,'("step",I0)') step
      call dump(trim(key))
    enddo
    call oracle_close()
  enddo
contains
  subroutine dump(prefix)
    character(len=*),intent(in) :: prefix
    call oracle_put(trim(prefix)//'_bc1',chem(:,:,:,p_bc1))
    call oracle_put(trim(prefix)//'_oc1',chem(:,:,:,p_oc1))
    call oracle_put(trim(prefix)//'_bc2',chem(:,:,:,p_bc2))
    call oracle_put(trim(prefix)//'_oc2',chem(:,:,:,p_oc2))
  end subroutine
end program
