program run_predict_gsd41
  use module_bl_mynn, only: mym_predict
  implicit none
  integer,parameter :: nz=12
  integer :: c,k,f
  real :: dz(nz),el(nz),dfq(nz),pdk(nz),pdt(nz),pdq(nz),pdc(nz)
  real :: qke(nz),qke0(nz),tsq(nz),qsq(nz),cov(nz),s_aw(nz+1),awqke(nz+1)
  real :: dt,ust,flt,flq,pmz,phh
  character(1024) :: path
  call get_command_argument(1,path)
  open(newunit=f,file=trim(path),status='replace')
  write(f,'(A)') 'case,k,dz,el,dfq,pdk,pdt,pdq,pdc,qke0,s_aw,s_aw_next,awqke,awqke_next,delt,ust,flt,flq,pmz,phh,qke'
  do c=1,8
    dt=20.; ust=0.3; flt=0.05; flq=0.00002; pmz=1.; phh=1.
    s_aw=0.; awqke=0.; tsq=0.; qsq=0.; cov=0.
    do k=1,nz
      dz(k)=50.+10.*k; el(k)=15.+2.*k; dfq(k)=0.1
      pdk(k)=0.005; pdt(k)=0.; pdq(k)=0.; pdc(k)=0.
      qke(k)=1.-0.05*k
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
    qke0=qke
    call mym_predict(1,nz,2,dt,dz,ust,flt,flq,pmz,phh,el,dfq, &
      pdk,pdt,pdq,pdc,qke,tsq,qsq,cov,s_aw,awqke,0)
    do k=1,nz
      write(f,'(I0,",",I0,19(",",ES24.16E3))')c,k,dz(k),el(k),dfq(k), &
        pdk(k),pdt(k),pdq(k),pdc(k),qke0(k),s_aw(k),s_aw(k+1),awqke(k),awqke(k+1), &
        dt,ust,flt,flq,pmz,phh,qke(k)
    enddo
  enddo
  close(f)
end program