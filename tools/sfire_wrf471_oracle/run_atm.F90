program run_atm
  use module_fr_fire_atm
  use module_fr_fire_util
  use oracle_io
  implicit none
  integer, parameter :: nx=8,ny=6,nz=7,rx=3,ry=2
  real :: hg(nx,ny),qg(nx,ny),hc(nx,ny),qc(nx,ny),terrain(nx,ny),mu(nx,ny)
  real :: zw(nx,nz+1,ny),dz(nx,nz+1,ny),rho(nx,nz+1,ny)
  real :: c1(nz+1),c2(nz+1),th(nx,nz+1,ny),qv(nx,nz+1,ny)
  real :: fine(nx*rx,ny*ry),coarse(nx,ny),coarse0(nx,ny)
  real :: fgip(nx*rx,ny*ry),tracer(nx,nz+1,ny,1),tracer0(nx,nz+1,ny,1)
  character(len=1024) :: root
  character(len=64) :: name
  integer :: i,j,k,m
  real :: eg,ec,crown
  call oracle_root(root)
  fire_print_msg=0
  do j=1,ny
    do i=1,nx
      terrain(i,j)=230.+13.*i+7.*j
      mu(i,j)=90000.+23.*i+11.*j
      hg(i,j)=1000.*i+300.*j
      qg(i,j)=400.*i+70.*j
      hc(i,j)=800.*i+240.*j
      qc(i,j)=300.*i+60.*j
      do k=1,nz+1
        zw(i,k,j)=terrain(i,j)+18.*(k-1)*(k-1)
        dz(i,k,j)=18.*(2*k-1)
        rho(i,k,j)=1.2-0.09*(k-1)+0.001*i
      end do
    end do
  end do
  do k=1,nz+1
    c1(k)=1.-0.11*(k-1)
    c2(k)=80.*(k-1)
  end do
  do m=1,4
    eg=6.+23.*m
    ec=13.+11.*m
    crown=30.*m
    if(m==4) then
      hg=0.; qg=0.; hc=0.; qc=0.
    endif
    th=-777.;qv=-777.
    call fire_tendency(1,nx+1,1,nz+1,1,ny+1, 1,nx,1,nz+1,1,ny, &
         1,nx,1,nz,1,ny, hg,qg,hc,qc,eg,ec,crown,0,100.,1000., &
         terrain,zw,dz,mu,c1,c2,rho,th,qv)
    write(name,'(a,i0)') 'atm/exponential_',m
    call oracle_open(trim(name))
    call oracle_put('grnhfx',hg);call oracle_put('grnqfx',qg)
    call oracle_put('canhfx',hc);call oracle_put('canqfx',qc)
    call oracle_put('terrain',terrain);call oracle_put('mu',mu)
    call oracle_put('z_at_w',zw);call oracle_put('dz8w',dz)
    call oracle_put('rho',rho);call oracle_put('c1h',c1);call oracle_put('c2h',c2)
    call oracle_put('ext_grnd',eg);call oracle_put('ext_crwn',ec)
    call oracle_put('crown_height',crown)
    call oracle_put('rthfrten',th);call oracle_put('rqvfrten',qv)
    call oracle_close()
  end do
  do j=1,ny*ry
    do i=1,nx*rx
      fine(i,j)=real(mod(i*17+j*29,97))*0.017+0.001*i
      fgip(i,j)=0.3+real(mod(i+j,13))*0.21
    end do
  end do
  call sum_2d_cells(1,nx*rx,1,ny*ry,1,nx*rx,1,ny*ry,fine, &
                   1,nx,1,ny,1,nx,1,ny,coarse)
  call oracle_open('atm/sum_cells')
  call oracle_put('fine',fine);call oracle_put('sum',coarse)
  call oracle_put('sr_x',rx);call oracle_put('sr_y',ry)
  call oracle_close()
  do j=1,ny
    do i=1,nx
      coarse0(i,j)=0.15*i*i-0.37*j+0.029*i*j
    enddo
  enddo
  fine=-999.
  call interpolate_2d(1,nx,1,ny,1,nx,1,ny, &
      1,nx*rx,1,ny*ry,1,(nx-1)*rx+1,1,(ny-1)*ry+1,rx,ry, &
      1.,1.,1.,1.,coarse0,fine)
  call oracle_open('atm/interpolate_2d')
  call oracle_put('coarse',coarse0)
  call oracle_put('fine',fine(1:(nx-1)*rx+1,1:(ny-1)*ry+1))
  call oracle_put('sr_x',rx);call oracle_put('sr_y',ry)
  call oracle_close()
  do j=1,ny*ry
    do i=1,nx*rx
      fine(i,j)=real(mod(i*17+j*29,97))*0.001
    enddo
  enddo
  tracer=0.;tracer0=tracer
  call add_fire_tracer_emissions(2,1.,30.,30.,1,nx*rx,1,ny*ry, &
      1,nx*rx,1,ny*ry,1,nx+1,1,nz+1,1,ny+1, &
      1,nx,1,nz+1,1,ny,1,nx,1,nz,1,ny,rho,dz,fine,fgip, &
      tracer,0.015,0,100.,50.,1000.,terrain,zw)
  call oracle_open('atm/original_smoke_column_defect')
  call oracle_put('burnt',fine);call oracle_put('fuel',fgip)
  call oracle_put('rho',rho);call oracle_put('dz8w',dz)
  call oracle_put('tracer',tracer);call oracle_put('tracer0',tracer0)
  call oracle_put('sr_x',rx);call oracle_put('sr_y',ry)
  call oracle_close()
end program run_atm
