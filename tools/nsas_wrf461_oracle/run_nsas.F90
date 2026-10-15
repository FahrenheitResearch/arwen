! WRF v4.6.1 NSAS (cu_physics = 96) column oracle.
!
! Reads the case file written by gpuwm/verify/nsas_oracle.py, and for every
! case
!
!   1. calls module_cu_nsas::cu_nsas, the WRF driver entry, once over the
!      case's i-row, exactly as module_cumulus_driver.F:1483 calls it, and
!      writes RTHCUTEN/RQVCUTEN/RQCCUTEN/RQICUTEN/RUCUTEN/RVCUTEN, RAINCV,
!      PRATEC, HBOT and HTOP per column;
!   2. for every column ALONE, forms cu_nsas's own column inputs (its lines
!      168-209, restated below operation for operation) and calls nsas2d,
!      then nscv2d, writing each stage's state so a disagreement can be
!      placed in the deep or the shallow routine.
!
! Linked against the module_cu_nsas.o of the scalar WRF build (build.sh), so
! the arithmetic graded is the arithmetic of that wrf.exe.  The harness owns
! no physics: its only arithmetic is the restated prep of step 2, and the
! stage chain's tendencies are checked against step 1's by the test suite.
program run_nsas
   use module_cu_nsas, only : cu_nsas, nsas2d, nscv2d, nsasinit
   implicit none

   integer, parameter :: MAGIC = 1314079059   ! 0x4E534153 "NSAS"
   ! WRF module_model_constants.F, as REAL PARAMETERs folded in kind 4.
   real, parameter :: r_d = 287., r_v = 461.6, cp = 7.*r_d/2., g = 9.81
   real, parameter :: cpv = 4.*r_v, cliq = 4190., cice = 2106., psat = 610.78
   real, parameter :: xlv = 2.5e6, xls = 2.85e6
   real, parameter :: ep_1 = r_v/r_d - 1., ep_2 = r_d/r_v

   character(len=512) :: cases_path, out_path
   integer :: hdr(4), ih(8), ncase, nz, ic, ncol, c, k
   real    :: rh(8), ch(8)
   real, allocatable :: p8w(:,:,:), w(:,:,:), p(:,:,:), pi3(:,:,:),        &
        t(:,:,:), qv(:,:,:), qc(:,:,:), qi(:,:,:), rho(:,:,:), dz8w(:,:,:), &
        u(:,:,:), v(:,:,:)
   real, allocatable :: rthcuten(:,:,:), rqvcuten(:,:,:), rqccuten(:,:,:), &
        rqicuten(:,:,:), rucuten(:,:,:), rvcuten(:,:,:)
   real, allocatable :: raincv(:,:), pratec(:,:), hbot(:,:), htop(:,:),    &
        xland(:,:), hpbl(:,:), hfx(:,:), qfx(:,:)
   logical, allocatable :: cu_act_flag(:,:)
   integer :: mp_physics, dx_factor, stepcu, itimestep
   real    :: dt, dx, pgcon
   logical :: f_qc, f_qi
   integer :: ncloud

   call get_command_argument(1, cases_path)
   call get_command_argument(2, out_path)
   open(10, file=trim(cases_path), access='stream', form='unformatted',   &
        status='old', convert='little_endian')
   open(11, file=trim(out_path), access='stream', form='unformatted',     &
        status='replace', convert='little_endian')
   read(10) hdr
   if (hdr(1) /= MAGIC) stop 'not an NSAS case file'
   ncase = hdr(2)
   nz = hdr(3)

   do ic = 1, ncase
      read(10) ih
      read(10) rh
      mp_physics = ih(1); dx_factor = ih(2); stepcu = ih(3)
      itimestep = ih(4); ncol = ih(5)
      dt = rh(1); dx = rh(2); pgcon = rh(3)
      call alloc_row(ncol, nz)
      do c = 1, ncol
         read(10) ch
         xland(c,1) = ch(1); hpbl(c,1) = ch(2)
         hfx(c,1) = ch(3);   qfx(c,1) = ch(4)
         read(10) (p8w(c,k,1), k = 1, nz+1)
         read(10) (w(c,k,1), k = 1, nz+1)
         read(10) (p(c,k,1), k = 1, nz)
         read(10) (pi3(c,k,1), k = 1, nz)
         read(10) (t(c,k,1), k = 1, nz)
         read(10) (qv(c,k,1), k = 1, nz)
         read(10) (qc(c,k,1), k = 1, nz)
         read(10) (qi(c,k,1), k = 1, nz)
         read(10) (rho(c,k,1), k = 1, nz)
         read(10) (dz8w(c,k,1), k = 1, nz)
         read(10) (u(c,k,1), k = 1, nz)
         read(10) (v(c,k,1), k = 1, nz)
      end do

      ! module_cu_nsas.F:141-147 -- the same mapping decides which of
      ! WRF's F_QC/F_QI flags a scheme with that microphysics would carry.
      if (mp_physics == 0) then
         ncloud = 0
      elseif (mp_physics == 1 .or. mp_physics == 3) then
         ncloud = 1
      else
         ncloud = 2
      endif
      f_qc = ncloud >= 1
      f_qi = ncloud == 2

      ! Cold initialization follows category presence; restart preserves rates.
      call check_initialization(.false.)
      call check_initialization(.true.)

      ! ---- step 1: the WRF driver entry over the whole i-row ------------
      rthcuten = 0.; rqvcuten = 0.; rqccuten = 0.; rqicuten = 0.
      rucuten = 0.; rvcuten = 0.; raincv = 0.; pratec = 0.
      hbot = 0.; htop = 0.
      call cu_nsas(dt=dt, dx=dx, p3di=p8w, p3d=p, pi3d=pi3, qc3d=qc,      &
           qi3d=qi, rho3d=rho, itimestep=itimestep, stepcu=stepcu,         &
           hbot=hbot, htop=htop, cu_act_flag=cu_act_flag,                  &
           rthcuten=rthcuten, rqvcuten=rqvcuten, rqccuten=rqccuten,        &
           rqicuten=rqicuten, rucuten=rucuten, rvcuten=rvcuten,            &
           qv3d=qv, t3d=t, raincv=raincv, pratec=pratec, xland=xland,      &
           dz8w=dz8w, w=w, u3d=u, v3d=v, hpbl=hpbl, hfx=hfx, qfx=qfx,      &
           mp_physics=mp_physics, dx_factor_nsas=dx_factor,                &
           p_qc=2, p_qi=3, p_first_scalar=2, pgcon=pgcon,                  &
           cp=cp, cliq=cliq, cpv=cpv, g=g, xlv=xlv, r_d=r_d, r_v=r_v,      &
           ep_1=ep_1, ep_2=ep_2, cice=cice, xls=xls, psat=psat,            &
           f_qi=f_qi, f_qc=f_qc,                                           &
           ids=1, ide=ncol+1, jds=1, jde=2, kds=1, kde=nz+1,               &
           ims=1, ime=ncol, jms=1, jme=1, kms=1, kme=nz+1,                 &
           its=1, ite=ncol, jts=1, jte=1, kts=1, kte=nz)

      do c = 1, ncol
         write(11) (rthcuten(c,k,1), k = 1, nz)
         write(11) (rqvcuten(c,k,1), k = 1, nz)
         write(11) (rqccuten(c,k,1), k = 1, nz)
         write(11) (rqicuten(c,k,1), k = 1, nz)
         write(11) (rucuten(c,k,1), k = 1, nz)
         write(11) (rvcuten(c,k,1), k = 1, nz)
         write(11) raincv(c,1), pratec(c,1), hbot(c,1), htop(c,1)
         call stage_chain(c)
      end do
   end do
   close(10)
   close(11)

contains

   subroutine check_initialization(restarting)
      logical, intent(in) :: restarting
      integer :: pqc, pqi
      pqc = 0; pqi = 0
      if (f_qc) pqc = 2
      if (f_qi) pqi = 3
      rthcuten = 42.; rqvcuten = 42.; rqccuten = 42.; rqicuten = 42.
      rucuten = 42.; rvcuten = 42.
      call nsasinit(rthcuten, rqvcuten, rqccuten, rqicuten, rucuten, rvcuten, &
           restarting, pqc, pqi, 2, .true.,                               &
           1,ncol+1,1,2,1,nz+1, 1,ncol,1,1,1,nz+1, 1,ncol,1,1,1,nz)
      if (restarting) then
         if (any(rthcuten /= 42.) .or. any(rqvcuten /= 42.) .or.           &
             any(rqccuten /= 42.) .or. any(rqicuten /= 42.) .or.           &
             any(rucuten /= 42.) .or. any(rvcuten /= 42.))                 &
              stop 'nsasinit overwrote restart rates'
      else
         if (any(rthcuten(:,1:nz,:) /= 0.) .or.                            &
             any(rqvcuten(:,1:nz,:) /= 0.) .or.                            &
             any(rucuten(:,1:nz,:) /= 0.) .or.                             &
             any(rvcuten(:,1:nz,:) /= 0.)) stop 'nsasinit cold rates'
         if (f_qc .and. any(rqccuten(:,1:nz,:) /= 0.)) stop 'nsasinit QC'
         if (.not.f_qc .and. any(rqccuten /= 42.)) stop 'nsasinit absent QC'
         if (f_qi .and. any(rqicuten(:,1:nz,:) /= 0.)) stop 'nsasinit QI'
         if (.not.f_qi .and. any(rqicuten /= 42.)) stop 'nsasinit absent QI'
      endif
   end subroutine check_initialization

   subroutine alloc_row(n, nzz)
      integer, intent(in) :: n, nzz
      if (allocated(p8w)) then
         deallocate(p8w, w, p, pi3, t, qv, qc, qi, rho, dz8w, u, v,         &
              rthcuten, rqvcuten, rqccuten, rqicuten, rucuten, rvcuten,     &
              raincv, pratec, hbot, htop, xland, hpbl, hfx, qfx, cu_act_flag)
      end if
      allocate(p8w(n,nzz+1,1), w(n,nzz+1,1), p(n,nzz+1,1), pi3(n,nzz+1,1), &
           t(n,nzz+1,1), qv(n,nzz+1,1), qc(n,nzz+1,1), qi(n,nzz+1,1),       &
           rho(n,nzz+1,1), dz8w(n,nzz+1,1), u(n,nzz+1,1), v(n,nzz+1,1))
      allocate(rthcuten(n,nzz+1,1), rqvcuten(n,nzz+1,1),                   &
           rqccuten(n,nzz+1,1), rqicuten(n,nzz+1,1), rucuten(n,nzz+1,1),    &
           rvcuten(n,nzz+1,1))
      allocate(raincv(n,1), pratec(n,1), hbot(n,1), htop(n,1), xland(n,1), &
           hpbl(n,1), hfx(n,1), qfx(n,1), cu_act_flag(n,1))
      ! Level kde of the mass arrays is never read by the scheme; give it
      ! a defined value so nothing reads garbage.
      p = 0.; pi3 = 1.; t = 0.; qv = 0.; qc = 0.; qi = 0.; rho = 0.
      dz8w = 0.; u = 0.; v = 0.
   end subroutine alloc_row

   ! cu_nsas's column inputs for column c alone, then nsas2d and nscv2d.
   ! The prep below is module_cu_nsas.F:163-209 restated for one column.
   subroutine stage_chain(c)
      integer, intent(in) :: c
      real :: del(1,nz), prsll(1,nz), dot(1,nz), u1(1,nz), v1(1,nz),       &
              t1(1,nz), q1(1,nz), qc2(1,nz), qi2(1,nz), zll(1,nz)
      real :: prsii(1,nz+1), zii(1,nz+1)
      real :: prslk(1,nz+1), slimsk(1), hpbl1(1), hfx1(1), qfx1(1)
      real :: rain(1), delt
      integer :: kbot(1), ktop(1), icps(1), kk, kp

      delt = dt*stepcu
      do kk = 1, nz
         kp = kk + 1
         dot(1,kk) = -5.0e-4*g*rho(c,kk,1)*(w(c,kk,1)+w(c,kp,1))
         prsll(1,kk) = p(c,kk,1)*0.001
         prsii(1,kk) = p8w(c,kk,1)*0.001
      end do
      prsii(1,nz+1) = p8w(c,nz+1,1)*0.001
      zii(1,1) = 0.0
      do kk = 1, nz
         zii(1,kk+1) = zii(1,kk) + dz8w(c,kk,1)
      end do
      do kk = 1, nz
         zll(1,kk) = 0.5*(zii(1,kk)+zii(1,kk+1))
      end do
      do kk = 1, nz
         del(1,kk) = prsll(1,kk)*g/r_d*dz8w(c,kk,1)/t(c,kk,1)
         u1(1,kk) = u(c,kk,1)
         v1(1,kk) = v(c,kk,1)
         q1(1,kk) = qv(c,kk,1)
         t1(1,kk) = t(c,kk,1)
         qi2(1,kk) = qi(c,kk,1)
         qc2(1,kk) = qc(c,kk,1)
      end do
      do kk = 1, nz + 1
         prslk(1,kk) = pi3(c,kk,1)
      end do
      slimsk(1) = xland(c,1)
      hpbl1(1) = hpbl(c,1); hfx1(1) = hfx(c,1); qfx1(1) = qfx(c,1)

      write(11) (del(1,kk), kk = 1, nz)
      write(11) (prsll(1,kk), kk = 1, nz)
      write(11) (zll(1,kk), kk = 1, nz)
      write(11) (dot(1,kk), kk = 1, nz)
      write(11) (prsii(1,kk), kk = 1, nz+1)

      call nsas2d(delt=delt, delx=dx, del=del, prsl=prsll, prsi=prsii,      &
           prslk=prslk, zl=zll, ncloud=ncloud, qc2=qc2, qi2=qi2, q1=q1,      &
           t1=t1, rain=rain, kbot=kbot, ktop=ktop, icps=icps, lat=1,         &
           slimsk=slimsk, dot=dot, u1=u1, v1=v1, cp_=cp, cliq_=cliq,          &
           cvap_=cpv, g_=g, hvap_=xlv, rd_=r_d, rv_=r_v, fv_=ep_1, ep2=ep_2, &
           cice=cice, xls=xls, psat=psat, pgcon=pgcon,                       &
           dx_factor_nsas=dx_factor,                                         &
           ids=1, ide=2, jds=1, jde=2, kds=1, kde=nz+1,                      &
           ims=1, ime=1, jms=1, jme=1, kms=1, kme=nz+1,                      &
           its=1, ite=1, jts=1, jte=1, kts=1, kte=nz)
      call dump_stage(t1, q1, u1, v1, qc2, qi2, rain, kbot, ktop, icps)

      call nscv2d(delt=delt, del=del, prsl=prsll, prsi=prsii, prslk=prslk,   &
           zl=zll, ncloud=ncloud, qc2=qc2, qi2=qi2, q1=q1, t1=t1, rain=rain, &
           kbot=kbot, ktop=ktop, icps=icps, slimsk=slimsk, dot=dot, u1=u1,   &
           v1=v1, cp_=cp, cliq_=cliq, cvap_=cpv, g_=g, hvap_=xlv, rd_=r_d,   &
           rv_=r_v, fv_=ep_1, ep2=ep_2, cice=cice, xls=xls, psat=psat,       &
           hpbl=hpbl1, hfx=hfx1, qfx=qfx1, pgcon=pgcon,                      &
           ids=1, ide=2, jds=1, jde=2, kds=1, kde=nz+1,                      &
           ims=1, ime=1, jms=1, jme=1, kms=1, kme=nz+1,                      &
           its=1, ite=1, jts=1, jte=1, kts=1, kte=nz)
      call dump_stage(t1, q1, u1, v1, qc2, qi2, rain, kbot, ktop, icps)
   end subroutine stage_chain

   subroutine dump_stage(t1, q1, u1, v1, qc2, qi2, rain, kbot, ktop, icps)
      real, intent(in) :: t1(1,nz), q1(1,nz), u1(1,nz), v1(1,nz),           &
                          qc2(1,nz), qi2(1,nz), rain(1)
      integer, intent(in) :: kbot(1), ktop(1), icps(1)
      write(11) t1(1,:)
      write(11) q1(1,:)
      write(11) u1(1,:)
      write(11) v1(1,:)
      write(11) qc2(1,:)
      write(11) qi2(1,:)
      write(11) rain(1), real(kbot(1)), real(ktop(1)), real(icps(1))
   end subroutine dump_stage

end program run_nsas
