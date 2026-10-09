program run_variance_gsd41
  ! mym_predict of GSD MYNN v4.1 (NOAA-EMC WRF 3.9 module_bl_mynn.F) at the
  ! operational closure, level 2.5 (levflag 2): the TKE solve plus the
  ! diagnosed temperature, moisture and covariance variances
  ! (module_bl_mynn.F:2232-2253), on eight 12-level columns with nonzero
  ! production terms.  run_predict_gsd41 drives the same routine with zero
  ! variance production; this driver binds the variance rows (audit P23).
  use module_bl_mynn, only: mym_predict
  implicit none
  integer,parameter :: nz=12
  integer :: c,k,f
  real :: dz(nz),el(nz),dfq(nz),pdk(nz),pdt(nz),pdq(nz),pdc(nz)
  real :: pdk0(nz),pdt0(nz),pdq0(nz),pdc0(nz)
  real :: qke(nz),qke0(nz),tsq(nz),qsq(nz),cov(nz),s_aw(nz+1),awqke(nz+1)
  real :: tsq0(nz),qsq0(nz),cov0(nz)
  real :: dt,ust,flt,flq,pmz,phh
  character(1024) :: path
  call get_command_argument(1,path)
  open(newunit=f,file=trim(path),status='replace')
  write(f,'(A)') 'case,k,dz,el,dfq,pdk,pdt,pdq,pdc,qke0,tsq0,qsq0,cov0,' // &
      'delt,ust,flt,flq,pmz,phh,qke,tsq,qsq,cov'
  do c=1,8
    dt=20.; ust=0.3; flt=0.05; flq=0.00002; pmz=1.; phh=1.
    s_aw=0.; awqke=0.
    do k=1,nz
      dz(k)=50.+10.*k; el(k)=15.+2.*k; dfq(k)=0.1
      pdk(k)=0.005
      pdt(k)=1.e-5*real(mod(7*k+c,11))
      pdq(k)=3.e-11*real(mod(5*k+2*c,9))
      pdc(k)=2.e-8*real(mod(3*k+c,7)-3)
      qke(k)=1.-0.05*k
      tsq(k)=0.01*k; qsq(k)=1.e-8*k; cov(k)=-1.e-6*k
      if(c==2)then
        qke(k)=0.00001; pdk(k)=0.; ust=0.0001
      endif
      if(c==3)then
        qke(k)=180.+k; dt=0.001
      endif
      if(c==4.and.k==nz)qke(k)=4.
      if(c==5.and.k==4)qke(k)=-0.02
      if(c==6)pdk(k)=-0.1
      if(c==7)dfq(k)=1.2
      if(c==8)dt=60.
    enddo
    qke0=qke; tsq0=tsq; qsq0=qsq; cov0=cov
    pdk0=pdk; pdt0=pdt; pdq0=pdq; pdc0=pdc
    call mym_predict(1,nz,2,dt,dz,ust,flt,flq,pmz,phh,el,dfq, &
      pdk,pdt,pdq,pdc,qke,tsq,qsq,cov,s_aw,awqke,0)
    do k=1,nz
      write(f,'(I0,",",I0,21(",",ES24.16E3))')c,k,dz(k),el(k),dfq(k), &
        pdk0(k),pdt0(k),pdq0(k),pdc0(k),qke0(k),tsq0(k),qsq0(k),cov0(k), &
        dt,ust,flt,flq,pmz,phh,qke(k),tsq(k),qsq(k),cov(k)
    enddo
  enddo
  close(f)
end program
