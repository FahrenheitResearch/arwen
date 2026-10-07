program gsl_run_smoke_prep
 use machine
 use oracle_io
 implicit none
 integer,parameter :: nc=12,nz=59,its=1,ite=nc,jts=1,jte=1,kts=1,kte=nz,p_qv=1
 integer :: i,j,k,k1,hwp_method,hour_int,poison
 integer :: kpbl(nc,1),kpbl_thetav(nc,1),raw_kpbl(nc,1)
 real(kind_phys) :: z_at_w(nc,nz+1,1),pbl(nc,1),t_phy(nc,nz,1),p_phy(nc,nz,1),moist(nc,nz,1,1)
 real(kind_phys) :: us3d(nc,nz),vs3d(nc,nz),zmid(nc,nz,1),oro(nc),u10m(nc),v10m(nc)
 real(kind_phys) :: thetav(nc,nz,1),theta,sfcwind,sfcwind2,wind,delwind,dz,wdgust,snoweq
 real(kind_phys) :: windgustpot(nc,1),uspdavg2d(nc,1),hpbl2d(nc,1),hwp_local(nc,1),precip_factor
 real(kind_phys) :: totprcp(nc),totprcp_24hrs(nc,1),wetness(nc),t2m(nc),dpt2m(nc),dswsfc(nc),snow_cpl(nc)
 real(kind_phys),parameter :: delta_theta4gust=.5
 character(1024) :: root
 character(32) :: name
 call oracle_root(root)
 hour_int=13
 do i=1,nc
  oro(i)=0.;u10m(i)=real(i,kind_phys)*.7_kind_phys;v10m(i)=1.
  pbl(i,1)=100._kind_phys+real(i,kind_phys)*100._kind_phys
  totprcp(i)=real(mod(i,4),kind_phys)*.002_kind_phys;totprcp_24hrs(i,1)=.0001
  wetness(i)=real(mod(i,4),kind_phys)*.3_kind_phys
  t2m(i)=300.;dpt2m(i)=280._kind_phys+real(i,kind_phys)
  dswsfc(i)=real(i,kind_phys)*100._kind_phys;snow_cpl(i)=real(mod(i,3)*20,kind_phys)
  do k=1,nz+1
   z_at_w(i,k,1)=real(k-1,kind_phys)*100._kind_phys
  enddo
  do k=1,nz
   zmid(i,k,1)=real(k,kind_phys)*100._kind_phys-50._kind_phys
   p_phy(i,k,1)=100000._kind_phys-real(k-1,kind_phys)*1000._kind_phys
   t_phy(i,k,1)=300._kind_phys-real(k-1,kind_phys)*.5_kind_phys
   moist(i,k,1,1)=.01
   us3d(i,k)=real(k,kind_phys)*.3_kind_phys;vs3d(i,k)=2.
  enddo
 enddo
 pbl(1,1)=-1.
 pbl(2,1)=100000.
 pbl(3,1)=5750.
 ! Defined edge policy: above-column kpbl=nz-1, so kpbl+1 remains a mass level.
 do poison=1,2
 do hwp_method=1,4
  if(poison==1) then
   kpbl_thetav=-777;thetav=-huge(1._kind_phys);windgustpot=-huge(1._kind_phys)
   uspdavg2d=-huge(1._kind_phys);hpbl2d=-huge(1._kind_phys);hwp_local=-huge(1._kind_phys)
  else
   kpbl_thetav=777;thetav=huge(1._kind_phys);windgustpot=huge(1._kind_phys)
   uspdavg2d=huge(1._kind_phys);hpbl2d=huge(1._kind_phys);hwp_local=huge(1._kind_phys)
  endif
  kpbl=merge(-777,777,poison==1)
  include 'prep_kpbl.inc'
  raw_kpbl=kpbl
  kpbl=nz-1
  include 'prep_kpbl.inc'
  ! GSL can set kpbl=kte and then read us3d(kte+1); cap the undefined edge.
  kpbl=min(kpbl,nz-1)
  include 'prep_columns.inc'
  include 'prep_hwp.inc'
  write(name,'(A,I0,A,I0)') 'method',hwp_method,'_poison',poison
  call oracle_open(trim(name))
  call oracle_put('z_at_w',real(z_at_w,4));call oracle_put('pbl',real(pbl,4))
  call oracle_put('t',real(t_phy,4));call oracle_put('p',real(p_phy,4));call oracle_put('qv',real(moist(:,:,:,1),4))
  call oracle_put('u',real(us3d,4));call oracle_put('v',real(vs3d,4));call oracle_put('z',real(zmid,4))
  call oracle_put('oro',real(oro,4));call oracle_put('u10',real(u10m,4));call oracle_put('v10',real(v10m,4))
  call oracle_put('totprcp',real(totprcp,4));call oracle_put('totprcp_24hrs',real(totprcp_24hrs,4))
  call oracle_put('wetness',real(wetness,4));call oracle_put('t2m',real(t2m,4));call oracle_put('dpt2m',real(dpt2m,4))
  call oracle_put('swdown',real(dswsfc,4));call oracle_put('snow',real(snow_cpl,4))
  call oracle_put('hour',hour_int)
  call oracle_put('kpbl',kpbl);call oracle_put('kpbl_thetav',kpbl_thetav)
  call oracle_put('raw_kpbl',raw_kpbl)
  call oracle_put('uspdavg2d',real(uspdavg2d,4));call oracle_put('windgustpot',real(windgustpot,4))
  call oracle_put('hpbl2d',real(hpbl2d,4));call oracle_put('hwp',real(hwp_local,4))
  call oracle_close()
 enddo
 enddo
end program
