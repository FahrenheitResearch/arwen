! Apache-2.0. Wrapper for the pinned, unmodified CCPP reference.
module satmedmf_oracle_api
use iso_c_binding
use machine, only: kind_phys
use physcons
use funcphys, only: gpvs,fpvsx,fpvs
use satmedmfvdifq, only: satmedmfvdifq_run
use tridi_mod
implicit none
contains
subroutine satmedmf_table_oracle(table) bind(C)
real(c_double),intent(out) :: table(7501)
real(kind_phys) :: xmin,xmax,xinc,x
integer :: j
xmin=180.0_kind_phys; xmax=330.0_kind_phys
xinc=(xmax-xmin)/(7501-1)
do j=1,7501
 x=xmin+(j-1)*xinc
 table(j)=fpvsx(x)
enddo
end subroutine
subroutine satmedmf_column_oracle(im,km,nt,dt,u,v,t,q,swh,hlw,prsi,prsl,prslk, &
  phii,phil,sfc,du,dv,tdt,rtg,kpbl,hpbl,dkt,dku,sflux) bind(C)
integer(c_int), value :: im,km,nt
real(c_double), value :: dt
real(c_double), intent(in) :: u(im,km),v(im,km),t(im,km),q(im,km,nt)
real(c_double), intent(in) :: swh(im,km),hlw(im,km)
real(c_double), intent(in) :: prsi(im,km+1),prsl(im,km),prslk(im,km)
real(c_double), intent(in) :: phii(im,km+1),phil(im,km),sfc(im,16)
real(c_double), intent(out) :: du(im,km),dv(im,km),tdt(im,km),rtg(im,km,nt)
integer(c_int), intent(out) :: kpbl(im)
real(c_double), intent(out) :: hpbl(im),dkt(im,km),dku(im,km),sflux(im,4)
real(kind_phys) :: del(im,km),usfco(im),vsfco(im),dtend(im,km,nt)
integer :: kinver(im),dtidx(100+nt,1),errflg
character(512) :: errmsg
if(kind_phys /= c_double) error stop 'oracle requires kind_phys=8'
call gpvs()
du=0; dv=0; tdt=0; rtg=0; kpbl=0; hpbl=0; dkt=0; dku=0; sflux=0
dtend=0; dtidx=0; usfco=0; vsfco=0; kinver=km
del=prsi(:,1:km)-prsi(:,2:km+1)
call satmedmfvdifq_run(im=im,km=km,ntrac=nt,ntcw=2,ntrw=3,ntiw=4,ntke=nt, &
 grav=con_g,rd=con_rd,cp=con_cp,rv=con_rv,hvap=con_hvap,hfus=con_hfus, &
 fv=con_fvirt,eps=con_eps,epsm1=con_epsm1, &
 dv=dv,du=du,tdt=tdt,rtg=rtg,u1=u,v1=v,t1=t,q1=q,usfco=usfco,vsfco=vsfco, &
 icplocn2atm=0,swh=swh,hlw=hlw,xmu=sfc(:,13),garea=sfc(:,14), &
 zvfun=sfc(:,15),sigmaf=sfc(:,16),psk=sfc(:,1),rbsoil=sfc(:,2), &
 zorl=sfc(:,3),u10m=sfc(:,9),v10m=sfc(:,10),fm=sfc(:,11),fh=sfc(:,12), &
 tsea=sfc(:,4),heat=sfc(:,5),evap=sfc(:,6),stress=sfc(:,7),spd1=sfc(:,8), &
 kpbl=kpbl,prsi=prsi,del=del,prsl=prsl,prslk=prslk,phii=phii,phil=phil, &
 delt=dt,dspheat=.false.,dusfc=sflux(:,1),dvsfc=sflux(:,2), &
 dtsfc=sflux(:,3),dqsfc=sflux(:,4),hpbl=hpbl,dkt=dkt,dku=dku,kinver=kinver, &
 xkzm_m=1.0_kind_phys,xkzm_h=0.01_kind_phys,xkzm_s=0.2_kind_phys, &
 dspfac=1.0_kind_phys,bl_upfr=0.13_kind_phys,bl_dnfr=0.13_kind_phys, &
 rlmx=300.0_kind_phys,elmx=300.0_kind_phys,sfc_rlm=1,tc_pbl=0, &
 ntqv=1,dtend=dtend,dtidx=dtidx,index_of_temperature=1,index_of_x_wind=2, &
 index_of_y_wind=3,index_of_process_pbl=1,gen_tend=.false.,ldiag3d=.false., &
 errmsg=errmsg,errflg=errflg)
if(errflg /= 0) then
 print *,trim(errmsg)
 error stop 'CCPP column call failed'
endif
end subroutine
subroutine satmedmf_solver_oracle(im,km,nt,which,cl,cm,cu,r1,r2,au,a1,a2) bind(C)
integer(c_int),value :: im,km,nt,which
real(c_double),intent(in) :: cl(im,km-1),cm(im,km),cu(im,km-1)
real(c_double),intent(in) :: r1(im,km),r2(im,km*nt)
real(c_double),intent(out) :: au(im,km-1),a1(im,km),a2(im,km*nt)
au=0; a1=0; a2=0
select case(which)
case(1)
 call tridi1(im,km,cl,cm,cu,r1,au,a1)
case(2)
 call tridi2(im,km,cl,cm,cu,r1,r2,au,a1,a2)
case(3)
 call tridin(im,km,nt,cl,cm,cu,r1,r2,au,a1,a2)
case(4)
 call tridit(im,km,nt,cl,cm,cu,r2,au,a2)
case default
 error stop 'unknown solver'
end select
end subroutine
end module
