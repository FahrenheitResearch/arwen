module cu6_oracle_boundary
  use iso_c_binding, only: c_int, c_float
  use module_cu_tiedtke, only: cu_tiedtke, cuflx, tlucua, tlucub, tlucuc, c2es, c5les, c5ies, vtmpc1, cevapcu1, cevapcu2, cprcon
  use module_model_constants, only: r_d, r_v, cp, g
  implicit none
contains
  subroutine cu6_unsafe_flux() bind(C)
    real :: a(1,5,24),sig(4)
    integer :: bot(1),top(1),ddtop(1),ktype(1),ktopm2
    logical :: dd(1),active(1)
    a=0.;sig=0.5;bot=4;top=3;ddtop=4;ktype=0;dd=.false.;active=.false.
    call cuflx(1,4,5,a(:,:,1),a(:,:,2),a(:,:,3),a(:,:,4),a(:,:,5),a(:,:,6), &
      bot,top,ddtop,ktype,dd,active,a(:,:,7),a(:,:,8),a(:,:,9),a(:,:,10), &
      a(:,:,11),a(:,:,12),a(:,:,13),a(:,:,14),a(:,:,15),a(:,:,16), &
      a(:,1,17),a(:,1,18),a(:,:,19),a(:,1,20),a(:,:,21),ktopm2,60.,sig)
  end subroutine
  subroutine cu6_constants(values) bind(C)
    real(c_float), intent(out) :: values(7)
    values=(/c2es,c5les,c5ies,vtmpc1,cevapcu1,cevapcu2,cprcon/)
  end subroutine
  subroutine cu6_thermo(n,t,a,b,d) bind(C)
    integer(c_int), value :: n
    real(c_float), intent(in) :: t(n)
    real(c_float), intent(out) :: a(n),b(n),d(n)
    integer :: k
    do k=1,n
      a(k)=tlucua(t(k)); b(k)=tlucub(t(k)); d(k)=tlucuc(t(k))
    enddo
  end subroutine
  subroutine cu6_tan(n,t,a) bind(C)
    integer(c_int), value :: n
    real(c_float), intent(in) :: t(n)
    real(c_float), intent(out) :: a(n)
    integer :: k
    do k=1,n
      a(k)=tan(t(k))
    enddo
  end subroutine
  subroutine cu6_oracle(nc,nz,dt,itimestep,stepcu, &
      u,v,w,t,qv,qc,qi,pi,rho,qvf,qvbl,dz,p,p8,xland,qfx,znu, &
      rt,rqv,rqc,rqi,ru,rv,rain,prate,active,fqc,fqi) bind(C)
    integer(c_int), value :: nc,nz,itimestep,stepcu,fqc,fqi
    real(c_float), value :: dt
    real(c_float), intent(in) :: u(nc,nz+1,1),v(nc,nz+1,1),w(nc,nz+1,1)
    real(c_float), intent(in) :: t(nc,nz+1,1),qv(nc,nz+1,1),qc(nc,nz+1,1),qi(nc,nz+1,1)
    real(c_float), intent(in) :: pi(nc,nz+1,1),rho(nc,nz+1,1)
    real(c_float), intent(in) :: qvf(nc,nz+1,1),qvbl(nc,nz+1,1),dz(nc,nz+1,1)
    real(c_float), intent(in) :: p(nc,nz+1,1),p8(nc,nz+1,1),xland(nc,1),qfx(nc,1),znu(nz+1)
    real(c_float), intent(inout) :: rt(nc,nz+1,1),rqv(nc,nz+1,1),rqc(nc,nz+1,1)
    real(c_float), intent(inout) :: rqi(nc,nz+1,1),ru(nc,nz+1,1),rv(nc,nz+1,1)
    real(c_float), intent(inout) :: rain(nc,1),prate(nc,1)
    integer(c_int), intent(out) :: active(nc,1)
    logical :: flag(nc,1)
    flag=.false.
    if (kind(1.0) /= c_float) error stop 'cu6 requires default REAL32'
    call cu_tiedtke(dt,itimestep,stepcu,rain,prate,qfx,znu, &
      u,v,w,t,qv,qc,qi,pi,rho,qvf,qvbl,dz,p,p8,xland,flag, &
      1,nc+1,1,2,1,nz+1,1,nc,1,1,1,nz+1,1,nc,1,1,1,nz, &
      rt,rqv,rqc,rqi,ru,rv,.true.,fqc/=0,.true.,fqi/=0,.true.)
    active=merge(1_c_int,0_c_int,flag)
  end subroutine

  ! Synthetic inputs are made by Fortran and consumed unchanged by both
  ! implementations. These regimes are fixture coverage, not observations.
  subroutine cu6_fixture(nc,nz,u,v,w,t,qv,qc,qi,pi,rho,qvf,qvbl,dz,p,p8,xland,qfx,znu,regime,unsafe_top) bind(C)
    integer(c_int), value :: nc,nz,unsafe_top
    real(c_float), intent(out) :: u(nc,nz+1),v(nc,nz+1),w(nc,nz+1),t(nc,nz+1)
    real(c_float), intent(out) :: qv(nc,nz+1),qc(nc,nz+1),qi(nc,nz+1),pi(nc,nz+1),rho(nc,nz+1)
    real(c_float), intent(out) :: qvf(nc,nz+1),qvbl(nc,nz+1),dz(nc,nz+1),p(nc,nz+1),p8(nc,nz+1)
    real(c_float), intent(out) :: xland(nc),qfx(nc),znu(nz+1)
    integer(c_int), intent(out) :: regime(nc)
    integer :: c,k,r,variant
    real :: ts,lapse,ps,qs,h,zh,dzz,variation,saturation,rh
    u=0.;v=0.;w=0.;t=0.;qv=0.;qc=0.;qi=0.;pi=1.;rho=1.
    qvf=0.;qvbl=0.;dz=0.;p=0.;p8=0.;xland=1.;qfx=0.
    dzz=16000./real(nz)
    do k=1,nz+1
      znu(k)=1.-(real(k)-0.5)/real(nz)
    end do
    do c=1,nc
      r=1+mod((c-1)/12,6)
      variant=mod(c-1,12)
      variation=0.7+0.06*real(variant)
      regime(c)=r
      ts=304.;lapse=0.007;ps=100000.;qs=0.020
      select case(r)
      case(2)
        ts=299.;lapse=0.007;qs=0.016
      case(3)
        ts=286.;lapse=0.003;qs=0.003
      case(4)
        ts=303.;lapse=0.0065;qs=0.023;xland(c)=2.
      case(5)
        ts=294.;lapse=0.0075;ps=68000.;qs=0.014
      case(6)
        ts=300.;lapse=0.0065;qs=0.018
      end select
      ts=ts+(real(variant)-5.)*0.25
      qfx(c)=0.00008*variation
      if(r==3) qfx(c)=0.
      do k=1,nz+1
        h=real(k-1)*dzz
        p8(c,k)=ps*exp(-h/7800.)
        w(c,k)=0.05*variation*sin(3.14159265*h/16000.)
      end do
      do k=1,nz
        zh=(real(k)-0.5)*dzz
        t(c,k)=max(205.,ts-lapse*min(zh,10000.))
        if(zh>10000.) t(c,k)=t(c,k)+(zh-10000.)*0.003
        if(unsafe_top/=0) t(c,k)=max(205.,ts-lapse*zh)
        if(r==2.and.zh>2000.) t(c,k)=t(c,k)+min(12.,(zh-2000.)*0.01)
        p(c,k)=0.5*(p8(c,k)+p8(c,k+1))
        rh=0.75+0.01*real(variant)
        if(zh>10000.) rh=0.20
        if(unsafe_top/=0) rh=0.75+0.01*real(variant)
        if(r==2.and.zh>2000.) rh=0.40
        if(r==3) rh=0.35
        saturation=tlucua(t(c,k))/p(c,k)
        saturation=saturation/(1.-(r_v/r_d-1.)*saturation)
        qv(c,k)=rh*saturation/(1.-saturation)
        p8(c,k+1)=p8(c,k)*exp(-g*dzz/(r_d*t(c,k)*(1.+(r_v/r_d-1.)*qv(c,k))))
        p(c,k)=0.5*(p8(c,k)+p8(c,k+1))
        pi(c,k)=(p(c,k)/100000.)**(r_d/cp)
        rho(c,k)=p(c,k)/(r_d*t(c,k)*(1.+(r_v/r_d-1.)*qv(c,k)))
        dz(c,k)=dzz
        u(c,k)=5.+0.001*zh
        v(c,k)=-2.+0.0003*zh
        if(zh<3000.) then
          qvf(c,k)=1.e-7*variation
          qvbl(c,k)=2.e-8*variation
        endif
        if(r==3) then
          qvf(c,k)=0.;qvbl(c,k)=0.;w(c,k)=0.
        endif
        if(r==6) then
          select case(variant)
          case(0)
            qv(c,k)=0.;qvf(c,k)=0.;qvbl(c,k)=0.;qfx(c)=0.
          case(1)
            w(c,k)=0.;qvf(c,k)=0.;qvbl(c,k)=0.;qfx(c)=0.
          case(2)
            qc(c,k)=1.e-4
          case(3)
            qi(c,k)=1.e-5
          case(4)
            qvf(c,k)=-abs(qvf(c,k))
          case(5)
            xland(c)=2.;qfx(c)=0.
          case(6)
            t(c,k)=273.16
          case(7)
            t(c,k)=233.15
          case(8)
            qv(c,k)=0.03*exp(-zh/3000.)
          case(9)
            w(c,k)=-abs(w(c,k))
          case(10)
            qfx(c)=-0.00001
          case(11)
            qvf(c,k)=qvf(c,k)*5.
          end select
        endif
      end do
    end do
  end subroutine
end module
