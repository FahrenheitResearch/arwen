"""Reuse the qualified scenarios while calling WRF's public goddardrad driver."""
from pathlib import Path
p=Path(__file__).parent
s=(p/'run_columns.F90').read_text().replace('program run_columns','program run_driver').replace('fast_overcast\n','fast_overcast, goddardrad\n')
s=s.replace(' real::p,scale',''' real :: a(c,n+1,1,19),pw(c,n+1,1),tw(c,n+1,1),heat(c,n+1,1),heatlw(c,n),heatsw(c,n)
 real :: lat(c,1),lon(c,1),alb(c,1),em(c,1),skin(c,1),cosine(c,1),obscur(c,1)
 real :: erbe(c,1,8),gsw(c,1),glw(c,1),sflux(c,1,4),dni(c,1),direct(c,1),diffuse(c,1),cod(c,1),ctop(c,1)
 real :: surface(c,15)
 integer :: du,dr,km,h
 real::p,scale''')
s=s.replace(' do batch=0,3', ''' open(newunit=du,file=trim(output)//'.driver-inputs',access='stream',form='unformatted',convert='little_endian',status='new')
 open(newunit=dr,file=trim(output)//'.driver',access='stream',form='unformatted',convert='little_endian',status='new')
 write(du) int(z'47524936'),64,n,4
 write(dr) int(z'47524436'),64,n,4
 do batch=0,3''',1)
inject='''
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
'''
for arm in ('lw','sw'):
    inject+=f'''  call goddardrad(sw_or_lw='{arm}',dx=12000.,rthraten=heat,gsf={'glw' if arm=='lw' else 'gsw'},xlat=lat,xlong=lon, &
    dz8w=a(:,:,:,5),t8w=tw,rho_phy=a(:,:,:,6),alb=alb,emiss=em,tsk=skin,t3d=a(:,:,:,1),p3d=a(:,:,:,2), &
    p8w3d=pw,pi3d=a(:,:,:,19),cldfra3d=a(:,:,:,4),qv=a(:,:,:,3),qc=a(:,:,:,8),qi=a(:,:,:,7), &
    qr=a(:,:,:,9),qs=a(:,:,:,10),qg=a(:,:,:,11),qh=a(:,:,:,12),rec3d=a(:,:,:,14),rei3d=a(:,:,:,13), &
    rer3d=a(:,:,:,15),res3d=a(:,:,:,16),reg3d=a(:,:,:,17),reh3d=a(:,:,:,18), &
    f_qv=.true.,f_qc=.true.,f_qr=.true.,f_qi=.true.,f_qs=.true.,f_qg=.true.,f_qh=.true., &
    gmt=12.,cp=1004.5,g=9.81,julday=160,xtime=0.,declin=0.,solcon=1365.,radfrq=10.,degrad=.017453292, &
    cod2d_out=cod,ctop2d_out=ctop,warm_rain=.false.,ids=1,ide=c,jds=1,jde=1,kds=1,kde=n+1, &
    ims=1,ime=c,jms=1,jme=1,kms=1,kme=n+1,its=1,ite=c,jts=1,jte=1,kts=1,kte=n, &
    erbe_out=erbe,itimestep=1,dt_in=10.,sflxd=sflux,swddir=direct,swddni=dni,swddif=diffuse,coszen=cosine,obscur=obscur)
'''
    if arm=='lw':inject+='  heatlw=heat(:,1:n,1)\n'
inject+='''  heatsw=heat(:,1:n,1)-heatlw
  surface(:,1)=gsw(:,1)/(1.-alb(:,1));surface(:,2)=glw(:,1);surface(:,3)=gsw(:,1);surface(:,4)=-erbe(:,1,2)
  surface(:,5)=dni(:,1);surface(:,6)=direct(:,1);surface(:,7)=diffuse(:,1)
  surface(:,8:15)=erbe(:,1,:)
  if(.not.all(ieee_is_finite(heatlw)) .or. .not.all(ieee_is_finite(heatsw)) .or. .not.all(ieee_is_finite(surface)))error stop 'driver nonfinite'
  write(dr) heatlw,heatsw,surface
'''
s=s.replace('  write(unit) sw,swd,swu,sf,lw,lwd,lwu,ct,cs,cg,lt,ls,lg','  write(unit) sw,swd,swu,sf,lw,lwd,lwu,ct,cs,cg,lt,ls,lg\n'+inject)
s=s.replace(' close(unit)',' close(du)\n close(dr)\n close(unit)',1)
(p/'run_driver.F90').write_text(s)
