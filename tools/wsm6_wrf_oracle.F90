program wsm6_wrf_oracle
  use ccpp_kind_types, only: kind_phys
  use mp_wsm6, only: mp_wsm6_init,mp_wsm6_run,refl10cm_wsm6
  use mp_wsm6_effectrad, only: mp_wsm6_effectRad_run
  implicit none
  integer,parameter:: nz=8
  integer scenario,k,n,nsteps,errflg
  character(len=256):: errmsg,arg
  real(kind_phys):: t(1,nz),q(1,nz),qc(1,nz),qi(1,nz),qr(1,nz),qs(1,nz),qg(1,nz)
  real(kind_phys):: den(1,nz),p(1,nz),dz(1,nz)
  real(kind_phys):: rain(1),raincv(1),sr(1),snow(1),snowcv(1),graup(1),graupcv(1)
  real(kind_phys):: dbz(nz)
  real(kind_phys):: rec(1,nz),rei(1,nz),res(1,nz)
  real(kind_phys):: dt
  scenario=0
  nsteps=1
  dt=30.0_kind_phys
  if(command_argument_count()>0)then
    call get_command_argument(1,arg); read(arg,*)scenario
  endif
  if(command_argument_count()>1)then
    call get_command_argument(2,arg); read(arg,*)nsteps
  endif
  if(command_argument_count()>2)then
    call get_command_argument(3,arg); read(arg,*)dt
  endif
  if(nsteps<1 .or. dt<=0.0_kind_phys) error stop 2
  do k=1,nz
    if(scenario==0)then
      t(1,k)=286.0_kind_phys-0.7_kind_phys*(k-1)
      q(1,k)=0.011_kind_phys-0.0007_kind_phys*(k-1)
      qc(1,k)=7.0e-4_kind_phys+2.0e-5_kind_phys*(k-1)
      qi(1,k)=0.0_kind_phys
    else if(scenario==1)then
      t(1,k)=269.0_kind_phys-1.8_kind_phys*(k-1)
      q(1,k)=0.0038_kind_phys-0.00025_kind_phys*(k-1)
      qc(1,k)=2.0e-5_kind_phys
      ! Process-only cold oracle: sedimenting species start at zero.
      qi(1,k)=0.0_kind_phys
    else
      t(1,k)=280.0_kind_phys-2.0_kind_phys*(k-1)
      q(1,k)=0.0045_kind_phys-0.0002_kind_phys*(k-1)
      qc(1,k)=1.0e-4_kind_phys
      qi(1,k)=2.0e-5_kind_phys+1.0e-6_kind_phys*(k-1)
    endif
    p(1,k)=96000.0_kind_phys-7000.0_kind_phys*(k-1)
    den(1,k)=p(1,k)/(287.0_kind_phys*t(1,k))
    dz(1,k)=500.0_kind_phys+25.0_kind_phys*(k-1)
    if(scenario>=2)then
      qr(1,k)=2.0e-4_kind_phys+1.0e-5_kind_phys*(k-1)
      qs(1,k)=1.2e-4_kind_phys+8.0e-6_kind_phys*(k-1)
      qg(1,k)=5.0e-5_kind_phys+5.0e-6_kind_phys*(k-1)
    else
      qr(1,k)=0;qs(1,k)=0;qg(1,k)=0
    endif
  enddo
  rain=0;raincv=0;sr=0;snow=0;snowcv=0;graup=0;graupcv=0
  call mp_wsm6_init(1.28_kind_phys,1000.0_kind_phys,100.0_kind_phys, &
       4190.0_kind_phys,1846.4_kind_phys,merge(1,0,scenario==3),errmsg,errflg)
  call refl10cm_wsm6(q(1,:),qr(1,:),qs(1,:),qg(1,:),t(1,:),p(1,:), &
       dbz,1,nz)
  write(*,'(a,8(1x,es16.8))')'refl_before',dbz
  do n=1,nsteps
    call mp_wsm6_run(t,q,qc,qi,qr,qs,qg,den,p,dz,dt, &
         9.81_kind_phys,1004.5_kind_phys,1846.4_kind_phys,287.0_kind_phys, &
         461.6_kind_phys,273.15_kind_phys,461.6_kind_phys/287.0_kind_phys-1.0_kind_phys, &
         287.0_kind_phys/461.6_kind_phys,1.0e-15_kind_phys,2.85e6_kind_phys, &
         2.5e6_kind_phys,3.5e5_kind_phys,1.28_kind_phys,1000.0_kind_phys, &
         4190.0_kind_phys,2106.0_kind_phys,610.78_kind_phys,rain,raincv,sr, &
         snow,snowcv,graup,graupcv,its=1,ite=1,kts=1,kte=nz,errmsg=errmsg,errflg=errflg)
  enddo
  rec=2.49e-6_kind_phys;rei=4.99e-6_kind_phys;res=9.99e-6_kind_phys
  call mp_wsm6_effectRad_run(.true.,t,qc,qi,qs,den,1.0e-15_kind_phys, &
       273.15_kind_phys,2.49e-6_kind_phys,4.99e-6_kind_phys, &
       9.99e-6_kind_phys,50.0e-6_kind_phys,125.0e-6_kind_phys, &
       999.0e-6_kind_phys,rec,rei,res,1,1,1,nz,errmsg,errflg)
  write(*,'(a,7(1x,es16.8))')'surface',rain(1),raincv(1),snow(1),snowcv(1), &
       graup(1),graupcv(1),sr(1)
  do k=1,nz
    write(*,'(i0,10(1x,es16.8))')k-1,t(1,k),q(1,k),qc(1,k),qi(1,k), &
         qr(1,k),qs(1,k),qg(1,k),rec(1,k)*1.0e6_kind_phys, &
         rei(1,k)*1.0e6_kind_phys,res(1,k)*1.0e6_kind_phys
  enddo
end program
