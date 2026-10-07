program gsl_run_add_emiss_burn
 use machine
 use rrfs_smoke_config, only: num_chem,p_smoke,dbg_opt
 use module_add_emiss_burn
 use oracle_io
 implicit none
 integer, parameter :: nc=20,nz=59,kts=1,kte=nz,its=1,ite=nc
 real(kind_phys) :: chem(nc,nz,1,num_chem), ebu(nc,nz,1),q(nc,nz,1),rho(nc,nz,1),dz(nc,nz,1)
 real(kind_phys) :: coef(nc,1),hist(nc,1),hwp(nc,1),prev(nc,1),sw(nc,1),ends(nc,1),peak(nc,1)
 real(kind_phys) :: lat(nc,1),lon(nc,1),ein(nc,1),dt,pi,time,minimum
 real(kind_phys) :: before(nc,nz,1),emitted(nc,1)
 real(kind_phys) :: ebu_smoke(nc,nz),coef_bb_dc(nc,1)
 real(kind_phys) :: frp_in(nc,1),frp_mw(nc,1),frp_output(nc)
 real(kind_phys),parameter :: conv_frpi=1.e-6_kind_phys
 integer :: typ(nc,1),mode,step,i,k
 character(1024) :: root
 character(32) :: name,key
 call oracle_root(root)
 dbg_opt=.false.
 dt=36.; pi=acos(-1._kind_phys); minimum=1.e-3
 do mode=1,2
  chem=1.;q=0.;coef=1.;hist=1.;lat=0.;lon=0.;peak=0.;ein=1.
  do i=1,nc
   frp_mw(i,1)=real(i,kind_phys)*.3_kind_phys
   frp_in(i,1)=frp_mw(i,1)*1.e6_kind_phys
   typ(i,1)=mod(i-1,5)
   ends(i,1)=real(mod(i,5)*12,kind_phys)+1.99_kind_phys
   hwp(i,1)=real(i*15,kind_phys); prev(i,1)=real(mod(i,3)*9,kind_phys)
   sw(i,1)=merge(0._kind_phys,250._kind_phys,mod(i,2)==0)
   do k=1,nz
    rho(i,k,1)=1.2_kind_phys-real(k,kind_phys)*.01_kind_phys
    dz(i,k,1)=50._kind_phys+real(k,kind_phys)
    ebu(i,k,1)=real(i,kind_phys)*.03_kind_phys
   enddo
  enddo
  ebu(1,:,1)=minimum*.5_kind_phys
  ebu(2,:,1)=minimum
  ebu(3,:,1)=1.e+7_kind_phys
  chem(4,:,1,p_smoke)=-1.
  chem(5,:,1,p_smoke)=4999.
  write(name,'(A,I0)') 'mode',mode
  call oracle_open(trim(name))
  call oracle_put('initial',real(chem(:,:,:,p_smoke),4))
  call oracle_put('ebu',real(ebu,4));call oracle_put('rho',real(rho,4));call oracle_put('dz',real(dz,4))
  call oracle_put('hwp',real(hwp,4));call oracle_put('prev',real(prev,4));call oracle_put('sw',real(sw,4))
  call oracle_put('ends',real(ends,4));call oracle_put('type',typ)
  call oracle_put('frp_mw',real(frp_mw,4))
  call oracle_put('dt',real(dt,4));call oracle_put('minimum',real(minimum,4))
  do step=1,6
   coef_bb_dc=coef
   if(step>1) then
    include 'fire_carry_restore.inc'
   endif
   before=chem(:,:,:,p_smoke)
   time=real((step-1)*13*3600,kind_phys)
   call add_emis_burn(dt,dz,rho,pi,minimum,chem,1,0._kind_phys,lat,lon, &
        ends,peak,time,coef,hist,hwp,prev,sw,mode,ein,ebu,typ,q,.false.,1._kind_phys, &
        1,nc+1,1,2,1,nz+1,1,nc,1,1,1,nz,1,nc,1,1,1,nz,0)
   write(key,'(A,I0)') 'chem',step
   call oracle_put(trim(key),real(chem(:,:,:,p_smoke),4))
   write(key,'(A,I0)') 'coef',step
   call oracle_put(trim(key),real(coef,4))
   write(key,'(A,I0)') 'hist',step
   call oracle_put(trim(key),real(hist,4))
   emitted=0.
   do k=1,51
    do i=1,nc
     emitted(i,1)=emitted(i,1)+(chem(i,k,1,p_smoke)-before(i,k,1))*rho(i,k,1)*dz(i,k,1)
    enddo
   enddo
   write(key,'(A,I0)') 'emitted',step
   call oracle_put(trim(key),real(emitted,4))
   coef_bb_dc=coef
   include 'fire_carry_store.inc'
   write(key,'(A,I0)') 'carry',step
   call oracle_put(trim(key),real(ebu_smoke,4))
   do i=1,nc
    include 'fire_frp_diag.inc'
   enddo
   write(key,'(A,I0)') 'frp',step
   call oracle_put(trim(key),real(frp_output,4))
  enddo
  call oracle_close()
 enddo
end program
