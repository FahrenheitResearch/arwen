! WRF chem/module_input_chem_data.F:1531-2031,2331-2357.
! Registry/registry.chem:3940,4022 supplies the two package row orders.
program run_flowdep_bdy
 use bdy_reference
 use oracle_io
 implicit none
 integer, parameter :: nx=10,ny=9,nz=3,nfmax=19,bw=3
 real :: chem(nx+1,nz+1,ny+1,nfmax), initial(nx,nz,ny,nfmax)
 real :: u(nx+1,nz+1,ny+1),v(nx+1,nz+1,ny+1),dummy(nx+1,nz+1,ny+1)
 real :: plane(nx+1,ny+1),sigma(nz+1)
 real :: xs(ny+1,nz+1,bw,nfmax),xt(ny+1,nz+1,bw,nfmax)
 real :: xe(ny+1,nz+1,bw,nfmax),xet(ny+1,nz+1,bw,nfmax)
 real :: ys(nx+1,nz+1,bw,nfmax),yt(nx+1,nz+1,bw,nfmax)
 real :: ye(nx+1,nz+1,bw,nfmax),yet(nx+1,nz+1,bw,nfmax)
 real :: defaults(nfmax), dt, times(3)=[0.,0.375,37.25]
 integer :: has(nfmax),mode,w,ti,n,i,j,k,r,nf
 logical :: have
 type(grid_config_rec_type) :: config
 character(len=1024) :: root
 character(len=80) :: label
 call oracle_root(root)
 dummy=0.;plane=1.;sigma=1.
 do mode=1,4
  nf=19
  config%chem_opt=GOCART_SIMPLE
  if(mode==3) then
   nf=6
   config%chem_opt=CHEM_TRACER
  endif
  do w=1,3,2
   do ti=1,3
    dt=times(ti)
    do j=1,ny+1
     do k=1,nz+1
      do i=1,nx+1
       u(i,k,j)=real(mod(i+2*j+k,3)-1)
       v(i,k,j)=real(mod(2*i+j+k,3)-1)
       do n=1,nf
        chem(i,k,j,n)=real(n*1000+k*100+j*10+i)*0.001
       enddo
      enddo
     enddo
    enddo
    initial(:,:,:,1:nf)=chem(1:nx,1:nz,1:ny,1:nf)
    do n=1,nf
     has(n)=0
     if(mode==1.or.(mode==4.and.mod(n,2)==0)) has(n)=1
     if(mode==3) then
      call bdy_chem_value_tracer(defaults(n),n+1)
     else
      call bdy_chem_value_gocart(defaults(n),n+1)
     endif
     do r=1,bw
      do k=1,nz+1
       do j=1,ny+1
        xs(j,k,r,n)=real(mod(j+k+n,5)-2)*0.03125+real(r-1)*100.
        xt(j,k,r,n)=real(mod(j+n,3)-1)*0.0078125
        xe(j,k,r,n)=real(mod(j+2*k+n,5)-2)*0.0625+real(r-1)*100.
        xet(j,k,r,n)=real(mod(k+n,3)-1)*0.00390625
       enddo
       do i=1,nx+1
        ys(i,k,r,n)=real(mod(i+k+n,5)-2)*0.015625+real(r-1)*100.
        yt(i,k,r,n)=real(mod(i+n,3)-1)*0.0078125
        ye(i,k,r,n)=real(mod(i+2*k+n,5)-2)*0.125+real(r-1)*100.
        yet(i,k,r,n)=real(mod(k+n,3)-1)*0.001953125
       enddo
      enddo
     enddo
     ! Separate multiply/add differs by one ULP from fused evaluation at dt=0.375.
     ys(1,1,1,n)=0.7148477435112
     yt(1,1,1,n)=0.19165858626365662
     have=has(n)==1
     call flow_dep_bdy_chem(chem(:,:,:,n), &
      xs(:,:,:,n),xt(:,:,:,n),xe(:,:,:,n),xet(:,:,:,n), &
      ys(:,:,:,n),yt(:,:,:,n),ye(:,:,:,n),yet(:,:,:,n), &
      dt,bw,dummy,have,u,v,config,dummy, &
      dummy,dummy,dummy,300.,100000.,0.2854,dummy,dummy,9.81, &
      w,n+1,100, &
      1,nx+1,1,ny+1,1,nz+1, &
      1,nx+1,1,ny+1,1,nz+1, &
      1,nx,1,ny,1,nz, &
      1,nx,1,ny,1,nz, &
      dummy,dummy,dummy,sigma,plane,plane,plane,plane,plane,1000.,plane,dummy)
    enddo
    write(label,'("mode",I0,"_w",I0,"_dt",I0)') mode,w,ti
    call oracle_open(trim(label))
    call oracle_put('initial',initial(:,:,:,1:nf))
    call oracle_put('chem',chem(1:nx,1:nz,1:ny,1:nf))
    call oracle_put('u',u(1:nx+1,1:nz,1:ny))
    call oracle_put('v',v(1:nx,1:nz,1:ny+1))
    call oracle_put('has_bc',has(1:nf))
    call oracle_put('default_inflow',defaults(1:nf))
    call oracle_put('bxs',xs(1:ny,1:nz,1,1:nf))
    call oracle_put('btxs',xt(1:ny,1:nz,1,1:nf))
    call oracle_put('bxe',xe(1:ny,1:nz,1,1:nf))
    call oracle_put('btxe',xet(1:ny,1:nz,1,1:nf))
    call oracle_put('bys',ys(1:nx,1:nz,1,1:nf))
    call oracle_put('btys',yt(1:nx,1:nz,1,1:nf))
    call oracle_put('bye',ye(1:nx,1:nz,1,1:nf))
    call oracle_put('btye',yet(1:nx,1:nz,1,1:nf))
    call oracle_put('spec_zone',w)
    call oracle_put('dt',dt)
    call oracle_close()
   enddo
  enddo
 enddo
end program

#include "bdy_stubs.inc"
