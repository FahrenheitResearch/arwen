program run_driver
 use module_ra_goddard, only: init_goddardrad, swrad, lwrad, opt_cloud, fp_kind, fast_overcast, goddardrad
 use, intrinsic :: ieee_arithmetic
 implicit none
 integer,parameter::n=32,c=16,nb=11
 integer::batch,ic,k,s,case_id,ict(c),icb(c),unit,input_unit,cloud_unit
 real(fp_kind)::pl(c,n+1),ta(c,n),wa(c,n),oa(c,n),fcld(c,n)
 real(fp_kind)::cosz(c),tb(c),ts(c),emiss(c,10),albedo(c)
 real(fp_kind)::tc(c,n,nb),sc(c,n,nb),gc(c,n,nb),tl(c,n,10),sl(c,n,10),gl(c,n,10)
 real(fp_kind)::taz(c,n,nb),saz(c,n,nb),gaz(c,n,nb)
 real::dz(c,n),q(c,n,6),re(c,n,6),ct(c,n,nb),cs(c,n,nb),cg(c,n,nb)
 real::lt(c,n,10),ls(c,n,10),lg(c,n,10)
 real::sw(c,n+1),swd(c,n+1),swu(c,n+1),sf(c,4),lw(c,n+1),lwd(c,n+1),lwu(c,n+1)
 real :: a(c,n+1,1,19),pw(c,n+1,1),tw(c,n+1,1),heat(c,n+1,1),heatlw(c,n),heatsw(c,n)
 real :: lat(c,1),lon(c,1),alb(c,1),em(c,1),skin(c,1),cosine(c,1),obscur(c,1)
 real :: erbe(c,1,8),gsw(c,1),glw(c,1),sflux(c,1,4),dni(c,1),direct(c,1),diffuse(c,1),cod(c,1),ctop(c,1)
 real :: surface(c,15)
 integer :: du,dr,km,h
 real::p,scale
 logical::mask(c)
 character(512)::output
 call get_command_argument(1,output)
 if(len_trim(output)==0) error stop 'output path required'
 call init_goddardrad(.true.,1,c,1,1,1,n+1,1,c,1,1,1,n+1,1,c,1,1,1,n)
 fast_overcast=.true.
 open(newunit=unit,file=trim(output),access='stream',form='unformatted',convert='little_endian',status='new')
 open(newunit=input_unit,file=trim(output)//'.inputs',access='stream',form='unformatted',convert='little_endian',status='new')
 write(input_unit) int(z'47524935'),64,n,4
 open(newunit=cloud_unit,file=trim(output)//'.cloud-inputs',access='stream',form='unformatted',convert='little_endian',status='new')
 write(cloud_unit) int(z'47524335'),64,n,4
 write(unit) int(z'47524435'),64,n,4,storage_size(1._fp_kind)
 open(newunit=du,file=trim(output)//'.driver-inputs',access='stream',form='unformatted',convert='little_endian',status='new')
 open(newunit=dr,file=trim(output)//'.driver',access='stream',form='unformatted',convert='little_endian',status='new')
 write(du) int(z'47524936'),64,n,4
 write(dr) int(z'47524436'),64,n,4
 do batch=0,3
  q=0.; fcld=0.; re=10.; dz=400.; emiss=.98; taz=0.; saz=0.; gaz=0.
  do ic=1,c
   case_id=batch*c+ic-1
   s=case_id/8
   scale=1.+.01*real(mod(case_id,8))
   tb(ic)=288.+real(mod(case_id,8)); ts(ic)=tb(ic)+2.
   cosz(ic)=.7; albedo(ic)=.15
   if(s==4) cosz(ic)=-.2
   if(s==5) cosz(ic)=.001+real(mod(case_id,8))*.01
   if(s==6) albedo(ic)=.65
   if(s==7) albedo(ic)=real(mod(case_id,8))*.13
   do k=1,n+1
    p=10.+(1000.-10.)*real(k-1)/real(n)
    if(s==6) p=10.+(650.-10.)*real(k-1)/real(n)
    pl(ic,k)=real(p,fp_kind)
   enddo
   ict(ic)=1; icb(ic)=1
   do k=1,n
    if(pl(ic,k)<400.) ict(ic)=k+1
    if(pl(ic,k)<700.) icb(ic)=k+1
    ta(ic,k)=real(215.+73.*real(k)/real(n),fp_kind)
    wa(ic,k)=real(.00001+.012*(real(k)/real(n))**3,fp_kind)
    oa(ic,k)=real(.0000001+.000007*(1.-real(k)/real(n))**2,fp_kind)
    if(s==7 .and. mod(case_id,8)==0) wa(ic,k)=0.
    if(s==1 .and. k>=25 .and. k<=28) then
     q(ic,k,2)=.25*scale; re(ic,k,2)=10.+.37*real(mod(case_id,8)); fcld(ic,k)=1.
    endif
    if(s==2 .and. k>=8 .and. k<=24) then
     q(ic,k,1)=.05*scale; re(ic,k,1)=70.+.123*real(mod(case_id,8))
     q(ic,k,4)=.1*scale; re(ic,k,4)=300.+.75*real(mod(case_id,8))
     q(ic,k,5)=.08*scale; re(ic,k,5)=700.+2.17*real(mod(case_id,8))
     fcld(ic,k)=1.
    endif
    if(s==3 .and. k>=8 .and. k<=10) then
     q(ic,k,1)=.002*scale; re(ic,k,1)=25.+.43*real(mod(case_id,8)); fcld(ic,k)=.2
    endif
    if(s==7 .and. mod(case_id,8)>=4 .and. k==20) then
     q(ic,k,2)=.1*scale; re(ic,k,2)=real(mod(case_id,8)-4)*1000.; fcld(ic,k)=.5
     if(mod(case_id,8)==6) re(ic,k,2)=20000.
     if(mod(case_id,8)==7) re(ic,k,2)=ieee_value(0.,ieee_quiet_nan)
    endif
   enddo
   icb(ic)=min(icb(ic),n)
   ict(ic)=min(ict(ic),icb(ic))
   mask(ic)=cosz(ic)>0.0001
  enddo
  write(cloud_unit) dz,q,re
  call opt_cloud('sw',n,nb,dz,q,re,ct,cs,cg,c)
  call opt_cloud('lw',n,10,dz,q,re,lt,ls,lg,c)
  if(.not.all(ieee_is_finite(ct)) .or. .not.all(ieee_is_finite(cs)) .or. &
     .not.all(ieee_is_finite(cg)) .or. .not.all(ieee_is_finite(lt)) .or. &
     .not.all(ieee_is_finite(ls)) .or. .not.all(ieee_is_finite(lg))) error stop 'non-finite cloud optics'
  tc=real(ct,fp_kind); sc=real(cs,fp_kind); gc=real(cg,fp_kind)
  tl=real(lt,fp_kind); sl=real(ls,fp_kind); gl=real(lg,fp_kind)
  write(input_unit) real(pl),real(ta),real(wa),real(oa),real(fcld),real(cosz),real(tb),real(ts),real(emiss),real(albedo)
  write(input_unit) ict,icb,mask
  write(input_unit) ct,cs,cg,lt,ls,lg
  ! The public driver initializes night outputs before the masked SW solver.
  sw=0.; swd=0.; swu=0.; sf=0.
  call swrad(n,cosz,pl,ta,wa,oa,fcld,ict,icb,tc,sc,gc,taz,saz,gaz, &
             albedo,albedo,albedo,albedo,sw,swd,swu,sf,mask,c)
  call lwrad(n,emiss,tb,ts,ict,icb,pl,ta,wa,oa,fcld,tl,sl,gl, &
             taz(:,:,1:10),saz(:,:,1:10),gaz(:,:,1:10),lw,lwd,lwu,c)
  if(.not.all(ieee_is_finite(sw)) .or. .not.all(ieee_is_finite(swd)) .or. &
     .not.all(ieee_is_finite(swu)) .or. .not.all(ieee_is_finite(sf)) .or. &
     .not.all(ieee_is_finite(lw)) .or. .not.all(ieee_is_finite(lwd)) .or. &
     .not.all(ieee_is_finite(lwu))) error stop 'non-finite band output'
  write(unit) sw,swd,swu,sf,lw,lwd,lwu,ct,cs,cg,lt,ls,lg

  a=0.; heat=0.; obscur=0.; lon=0.; tw=0.; erbe=0.; gsw=0.; glw=0.
  do ic=1,c
   lat(ic,1)=2.+real(mod(batch*c+ic-1,8))*12.
   if(mod(ic,2)==0)lat(ic,1)=-lat(ic,1)
   alb(ic,1)=real(albedo(ic));em(ic,1)=real(emiss(ic,1));skin(ic,1)=real(ts(ic));cosine(ic,1)=real(cosz(ic))
   do k=1,n+1
    pw(ic,k,1)=real(pl(ic,n+2-k))*100.
    tw(ic,k,1)=real(tb(ic))
   enddo
   do k=1,n
    km=n+1-k
    a(ic,k,1,1)=real(ta(ic,km));a(ic,k,1,2)=.5*(real(pl(ic,km))+real(pl(ic,km+1)))*100.
    a(ic,k,1,3)=real(wa(ic,km))/(1.-real(wa(ic,km)));a(ic,k,1,4)=real(fcld(ic,km))
    a(ic,k,1,5)=dz(ic,km);a(ic,k,1,6)=.9;a(ic,k,1,19)=.8+.2*real(k)/real(n)
    do h=1,6
     a(ic,k,1,6+h)=q(ic,km,h)/.9/1000.;a(ic,k,1,12+h)=re(ic,km,h)
    enddo
   enddo
  enddo
  write(du) pw(:,:,1),(a(:,1:n,1,h),h=1,18),lat(:,1),cosine(:,1),real(tb),skin(:,1),em(:,1),alb(:,1),a(:,1:n,1,19)
  call goddardrad(sw_or_lw='lw',dx=12000.,rthraten=heat,gsf=glw,xlat=lat,xlong=lon, &
    dz8w=a(:,:,:,5),t8w=tw,rho_phy=a(:,:,:,6),alb=alb,emiss=em,tsk=skin,t3d=a(:,:,:,1),p3d=a(:,:,:,2), &
    p8w3d=pw,pi3d=a(:,:,:,19),cldfra3d=a(:,:,:,4),qv=a(:,:,:,3),qc=a(:,:,:,8),qi=a(:,:,:,7), &
    qr=a(:,:,:,9),qs=a(:,:,:,10),qg=a(:,:,:,11),qh=a(:,:,:,12),rec3d=a(:,:,:,14),rei3d=a(:,:,:,13), &
    rer3d=a(:,:,:,15),res3d=a(:,:,:,16),reg3d=a(:,:,:,17),reh3d=a(:,:,:,18), &
    f_qv=.true.,f_qc=.true.,f_qr=.true.,f_qi=.true.,f_qs=.true.,f_qg=.true.,f_qh=.true., &
    gmt=12.,cp=1004.5,g=9.81,julday=160,xtime=0.,declin=0.,solcon=1365.,radfrq=10.,degrad=.017453292, &
    cod2d_out=cod,ctop2d_out=ctop,warm_rain=.false.,ids=1,ide=c,jds=1,jde=1,kds=1,kde=n+1, &
    ims=1,ime=c,jms=1,jme=1,kms=1,kme=n+1,its=1,ite=c,jts=1,jte=1,kts=1,kte=n, &
    erbe_out=erbe,itimestep=1,dt_in=10.,sflxd=sflux,swddir=direct,swddni=dni,swddif=diffuse,coszen=cosine,obscur=obscur)
  heatlw=heat(:,1:n,1)
  call goddardrad(sw_or_lw='sw',dx=12000.,rthraten=heat,gsf=gsw,xlat=lat,xlong=lon, &
    dz8w=a(:,:,:,5),t8w=tw,rho_phy=a(:,:,:,6),alb=alb,emiss=em,tsk=skin,t3d=a(:,:,:,1),p3d=a(:,:,:,2), &
    p8w3d=pw,pi3d=a(:,:,:,19),cldfra3d=a(:,:,:,4),qv=a(:,:,:,3),qc=a(:,:,:,8),qi=a(:,:,:,7), &
    qr=a(:,:,:,9),qs=a(:,:,:,10),qg=a(:,:,:,11),qh=a(:,:,:,12),rec3d=a(:,:,:,14),rei3d=a(:,:,:,13), &
    rer3d=a(:,:,:,15),res3d=a(:,:,:,16),reg3d=a(:,:,:,17),reh3d=a(:,:,:,18), &
    f_qv=.true.,f_qc=.true.,f_qr=.true.,f_qi=.true.,f_qs=.true.,f_qg=.true.,f_qh=.true., &
    gmt=12.,cp=1004.5,g=9.81,julday=160,xtime=0.,declin=0.,solcon=1365.,radfrq=10.,degrad=.017453292, &
    cod2d_out=cod,ctop2d_out=ctop,warm_rain=.false.,ids=1,ide=c,jds=1,jde=1,kds=1,kde=n+1, &
    ims=1,ime=c,jms=1,jme=1,kms=1,kme=n+1,its=1,ite=c,jts=1,jte=1,kts=1,kte=n, &
    erbe_out=erbe,itimestep=1,dt_in=10.,sflxd=sflux,swddir=direct,swddni=dni,swddif=diffuse,coszen=cosine,obscur=obscur)
  heatsw=heat(:,1:n,1)-heatlw
  surface(:,1)=gsw(:,1)/(1.-alb(:,1));surface(:,2)=glw(:,1);surface(:,3)=gsw(:,1);surface(:,4)=-erbe(:,1,2)
  surface(:,5)=dni(:,1);surface(:,6)=direct(:,1);surface(:,7)=diffuse(:,1)
  surface(:,8:15)=erbe(:,1,:)
  if(.not.all(ieee_is_finite(heatlw)) .or. .not.all(ieee_is_finite(heatsw)) .or. .not.all(ieee_is_finite(surface)))error stop 'driver nonfinite'
  write(dr) heatlw,heatsw,surface

 enddo
 close(du)
 close(dr)
 close(unit)
 close(input_unit)
 close(cloud_unit)
 print *, 'columns=64 levels=32 bands=2 output_real_bits=32 solver_real_bits=',storage_size(1._fp_kind)
end program
