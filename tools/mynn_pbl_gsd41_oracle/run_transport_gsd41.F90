program run_transport_gsd41
  use module_bl_mynn, only: mynn_tendencies, scale_aware
  implicit none
  integer, parameter :: nz=12
  integer :: c,k,f
  character(1024) :: path
  real :: dz(nz),rho(nz),u(nz),v(nz),th(nz),tk(nz),qv(nz),p(nz),exner(nz)
  real :: thl(nz),sqv(nz),sqw(nz),zero(nz),dfm(nz),dfh(nz),dfq(nz)
  real :: qc(nz),qi(nz),sqc(nz),sqi(nz),qnwfa(nz),qnifa(nz)
  real :: du(nz),dv(nz),dth(nz),dqv(nz),dqc(nz),dqi(nz)
  real :: dqnc(nz),dqni(nz),dqnwfa(nz),dqnifa(nz),s_aw(nz+1)
  real :: awthl(nz+1),awqv(nz+1),awu(nz+1),awv(nz+1),facezero(nz+1)
  real :: dt,ust,flt,flqv,wspd,dx,pblh,psig_bl,psig_shcu,vdfg
  call get_command_argument(1,path)
  open(newunit=f,file=trim(path),status='replace')
  write(f,'(A)') 'case,k,dz,rho,u,v,th,tk,qv,p,exner,thl,sqv,dfm,dfh,dfq,s_aw,s_aw_next,s_awthl,s_awthl_next,s_awqv,s_awqv_next,s_awu,s_awu_next,s_awv,s_awv_next,delt,ust,flt,flqv,wspd,dx,pblh,psig_bl,psig_shcu,du,dv,dth,dqv,sqv_solved'
  do c=1,8
    zero=0.; qc=0.; qi=0.; sqc=0.; sqi=0.; qnwfa=0.; qnifa=0.; facezero=0.
    dt=20.; ust=0.3; flt=0.05; flqv=2.e-5; wspd=4.
    dx=100.*c; pblh=200.*c
    if(c==2) flqv=-flqv
    if(c==3) flt=-0.02
    do k=1,nz
      dz(k)=40.+10.*(k-1); rho(k)=1.15-0.01*(k-1)
      if(c==4) rho(k)=rho(k)*0.25
      u(k)=3.+0.2*(k-1); v(k)=1.-0.05*(k-1); th(k)=285.+0.3*(k-1)
      p(k)=95000.-450.*(k-1); exner(k)=(p(k)/100000.)**(2./7.)
      tk(k)=th(k)*exner(k); thl(k)=th(k); sqv(k)=0.005-0.0001*(k-1)
      qv(k)=sqv(k)/(1.-sqv(k)); sqw(k)=sqv(k)
      dfm(k)=0.12; dfh(k)=0.08; dfq(k)=0.07
    enddo
    s_aw=0.; awthl=0.; awqv=0.; awu=0.; awv=0.
    if(c>=5)then
      do k=2,nz-1
        s_aw(k)=0.06*real(nz-k)/real(nz)
        awthl(k)=s_aw(k)*(th(k)+0.2); awqv(k)=s_aw(k)*(sqv(k)+0.0001)
        awu(k)=s_aw(k)*u(k); awv(k)=s_aw(k)*v(k)
      enddo
    endif
    call scale_aware(dx,pblh,psig_bl,psig_shcu)
    call mynn_tendencies(1,nz,2,0,dt,dz,rho,u,v,th,tk,qv,qc,qi,zero,zero,p,exner, &
      thl,sqv,sqc,sqi,sqw,qnwfa,qnifa,ust,flt,flqv,flqv,0.,wspd,0.,0.,0., &
      zero,zero,zero,zero,zero,dfm,dfh,dfq,du,dv,dth,dqv,dqc,dqi,dqnc,dqni, &
      dqnwfa,dqnifa,vdfg,zero,s_aw,awthl,awqv,awqv,facezero,awu,awv, &
      facezero,facezero,facezero,facezero,.true.,.true.,.false.,.false., &
      .false.,.false.,zero,1,0,1,1,0)
    do k=1,nz
      write(f,'(I0,",",I0,38(",",ES24.16E3))') c,k,dz(k),rho(k),u(k),v(k),th(k),tk(k),qv(k),p(k), &
        exner(k),th(k),sqv(k),dfm(k),dfh(k),dfq(k),s_aw(k),s_aw(k+1), &
        awthl(k),awthl(k+1),awqv(k),awqv(k+1),awu(k),awu(k+1),awv(k),awv(k+1), &
        dt,ust,flt,flqv,wspd,dx,pblh,psig_bl,psig_shcu,du(k),dv(k),dth(k),dqv(k), &
        (qv(k)+dqv(k)*dt)/(1.+qv(k)+dqv(k)*dt)
    enddo
  enddo
  close(f)
end program
