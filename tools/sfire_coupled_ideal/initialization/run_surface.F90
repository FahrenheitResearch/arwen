program run_surface
use native_surface_control
use oracle_io
implicit none
integer,parameter::nx=9,ny=7,nz=3
real::temperature(nx,nz,ny),theta(nx,nz,ny),qv(nx,nz,ny),pressure(nx,ny),coeff(3),t2(nx,ny),th2(nx,ny)
integer::i,j,k,c
character(len=1024)::root
character(len=64)::name
call oracle_root(root)
do c=1,4
 coeff=(/1.875,-1.25,.375/)
 if(c==2)coeff=(/1.76345,-1.08327,.31982/)
 if(c==3)coeff=(/1.5,-.5,0./)
 if(c==4)coeff=(/.875,.375,-.25/)
 do j=1,ny
  do i=1,nx
   pressure(i,j)=101200.+float(i-j)*13.25
   do k=1,nz
    temperature(i,k,j)=280.+float(3*i+2*j+k)*.1375
    theta(i,k,j)=300.+float(i+3*j-k)*.0625
    qv(i,k,j)=.005+float(i+j+2*k)*.0000125
   enddo
  enddo
 enddo
 call extrapolate(temperature,coeff,t2,nx,ny)
 call extrapolate(theta,coeff,th2,nx,ny)
 write(name,'(a,i0)')'moisture_surface/case_',c
 call oracle_open(trim(name));call oracle_put('temperature',temperature);call oracle_put('theta',theta)
 call oracle_put('qv',qv);call oracle_put('psfc',pressure);call oracle_put('coeff',coeff)
 call oracle_put('t2',t2);call oracle_put('th2',th2);call oracle_put('q2',qv(:,1,:));call oracle_close()
enddo
end program
