program run_aer_opt2
use module_ra_aerosol
implicit none
integer,parameter::nz=8,nc=96
real::p(1,nz+1,1),t(1,nz+1,1),q(1,nz+1,1),dz(1,nz+1,1),ht(1,1)
real::aod(1,1),ang(1,1),ssa(1,1),asy(1,1),v(4)
real::tau(1,nz+1,1,14),ss(1,nz+1,1,14),aa(1,nz+1,1,14),rh(1,nz+1,1)
integer::c,k,typ,opt(4),mode,scheme,nb,u
character(len=1024)::path
call get_command_argument(1,path)
open(newunit=u,file=trim(path),form='unformatted',access='stream',status='replace')
write(u) int(nc,4),int(nz,4)
do scheme=1,2
nb=14
if(scheme==2) nb=11
do c=1,nc
typ=1+mod(c-1,3)
mode=mod((c-1)/3,4)
opt=(/1,1,1,1/)
if(mode==1) opt=(/2,2,2,2/)
if(mode==2) opt=(/2,3,3,3/)
if(mode==3) opt=(/1,2,3,2/)
v=(/0.0,1.3,0.85,0.9/)
if(mod(c,7)==0) v(2)=1.0
if(mod(c,7)==1) v(2)=0.999
if(mod(c,7)==2) v(2)=1.001
if(mod(c,7)==3) v(2)=-0.3
if(mod(c,7)==4) v(2)=2.5
ht=real(mod(c,5))*1000.
aod=0.0
ang=v(2)
ssa=0.5+0.04*mod(c,12)
asy=-0.5+0.1*mod(c,15)
do k=1,nz+1
p(1,k,1)=95000.-8500.*(k-1)-100.*c
t(1,k,1)=295.-6.*(k-1)+0.05*c
q(1,k,1)=0.001*mod(c,25)*exp(-0.3*(k-1))
if(mod(c,11)==0) q(1,k,1)=-0.001
dz(1,k,1)=300.+80.*k+mod(c,3)
end do
tau=-999.;ss=-999.;aa=-999.;rh=-999.
write(u) int(typ,4),int(opt,4),v,ht,aod,ang,ssa,asy,p,t,q,dz
if(scheme==1) then
call calc_aerosol_rrtmg_sw(ht,dz,p,t,q,typ,opt(1),opt(2),opt(3),opt(4), &
v(1),v(2),v(3),v(4),aod,ang,ssa,asy,1,1,1,1,1,nz+1,1,1,1,1,1,nz,tau,ss,aa)
else
call calc_aerosol_goddard_sw(ht,dz,p,t,q,typ,opt(1),opt(2),opt(3),opt(4), &
v(1),v(2),v(3),v(4),aod,ang,ssa,asy,1,1,1,1,1,nz+1,1,1,1,1,1,nz,tau(:,:,:,1:11),ss(:,:,:,1:11),aa(:,:,:,1:11))
endif
call calc_relative_humidity(p,t,q,1,1,1,1,1,nz+1,1,1,1,1,1,nz,rh)
write(u) tau(:,:,:,1:nb),ss(:,:,:,1:nb),aa(:,:,:,1:nb),rh,aod,ang,ssa,asy
enddo
enddo
close(u)
end program
