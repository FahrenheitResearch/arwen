program run_corrected_tg
  use module_fr_fire_atm
  use module_fr_fire_util
  use oracle_io
  implicit none
  integer,parameter :: nx=6,ny=5,nz=7
  real :: z(nx,nz+1,ny),rho(nx,nz+1,ny),dz(nx,nz+1,ny),zs(nx,ny),mu(nx,ny)
  real :: hg(nx,ny),qg(nx,ny),hc(nx,ny),qc(nx,ny),c1(nz+1),c2(nz+1)
  real :: th(nx,nz+1,ny),qv(nx,nz+1,ny),prop(nx,nz+1,ny)
  integer :: i,j,k
  character(len=1024) :: root
  call oracle_root(root)
  fire_print_msg=0;fire_print_file=0
  do j=1,ny
    do i=1,nx
      zs(i,j)=100.+real(i+j)*11.;mu(i,j)=89000.+real(i*j)*37.
      hg(i,j)=1200.+real(i*j)*80.;qg(i,j)=900.+real(i+j)*10.
      hc(i,j)=1700.+real(i*j)*30.;qc(i,j)=800.+real(i+j)*40.
      z(i,1,j)=zs(i,j)
      do k=1,nz+1
        dz(i,k,j)=7.+real(k*k)+real(mod(i+j,3))
        rho(i,k,j)=1.2-real(k)*.03
        if(k<nz+1)z(i,k+1,j)=z(i,k,j)+dz(i,k,j)
      enddo
    enddo
  enddo
  do k=1,nz+1
    c1(k)=1.-real(k)*.08;c2(k)=real(k)*73.
  enddo
  th=-999.;qv=-999.;prop=-999.
  call oracle_open('corrected_tg')
  call oracle_put('z_at_w',z);call oracle_put('dz8w',dz);call oracle_put('rho',rho)
  call oracle_put('terrain',zs);call oracle_put('mu',mu)
  call oracle_put('grnhfx',hg);call oracle_put('grnqfx',qg)
  call oracle_put('canhfx',hc);call oracle_put('canqfx',qc)
  call oracle_put('c1h',c1);call oracle_put('c2h',c2)
  call tg_dist(1,nx,1,nz+1,1,ny,1,nx,1,ny,1,nz-1,dz,32.,1000.,60.,z,zs,prop)
  call oracle_put('prop_heat_out',prop)
  call fire_tendency(1,nx+1,1,nz+1,1,ny+1,1,nx,1,nz+1,1,ny, &
                     1,nx,1,nz,1,ny,hg,qg,hc,qc,60.,90.,20.,1,32.,1000.,zs,z,dz,mu,c1,c2,rho,th,qv)
  call oracle_put('rthfrten_out',th);call oracle_put('rqvfrten_out',qv);call oracle_close()
end program run_corrected_tg
