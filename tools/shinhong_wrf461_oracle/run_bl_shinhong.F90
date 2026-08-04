program run_bl_shinhong_oracle
  ! Drive the byte-unmodified WRF v4.6.1 phys/module_bl_shinhong.F through its
  ! own 3-D entry point `shinhong` and dump every input and every output as
  ! float32 CSV.  module_bl_shinhong.F is entirely self-contained: no USE, and
  ! every CALL (shinhong2d, tridi1n, tridin_ysu, mixlen, prodq2, vdifq) plus
  ! the five partition functions (pu, pq, pthnl, pthl, ptke) live in the same
  ! module, so -- like the YSU oracle and unlike RUC/Noah-MP -- there is no
  ! stub to write and `nm -u` on the object is the receipt.
  !
  ! The 3-D wrapper is used rather than shinhong2d so the fixture pins the
  ! whole WRF entry: the ndiff-packed moisture array, the rthblten = ttnp/pi
  ! identity (module_bl_shinhong.F:253), and the u10/v10 ctopo2 blend.
  !
  ! Optional-argument contract, established against the pinned source: the
  ! wrapper subscripts ctopo(ims,j), ctopo2(ims,j) and regime(ims,j)
  ! unconditionally (lines 232/242) and forwards wstar/delta to NON-optional
  ! dummies of shinhong2d, so a call without them is illegal Fortran, and
  ! WRF's own driver (module_pbl_driver.F:1302) always passes every one of
  ! them.  The oracle therefore always passes them too; formal absence is not
  ! a state WRF can reach.
  !
  ! The scheme is called TWICE per fixture row:
  !   * arm A: ctopo = ctopo2 = 1.0  -- the Registry default (topo_wind=0),
  !            the path every ArWen run takes today.
  !   * arm B: ctopo = 0.85, ctopo2 = 0.7 -- legal topo_wind values, so the
  !            ctopo momentum algebra (line 1387) and the u10/v10 blend
  !            (lines 1472-1473) are measured rather than argued about.
  !
  ! dx is a fixture axis, not a constant: Shin-Hong's entire reason to exist
  ! is that pu/pq/pthnl/pthl/ptke move with sqrt(dx*dy)/h, so every case runs
  ! at NDX grid spacings spanning d/h from the pmin clamp to the pmax clamp.
  !
  ! Nothing here invents an expected value: every number in the CSVs is
  ! either an input this program constructed or a word the pinned module
  ! wrote.
  use module_bl_shinhong, only: shinhong, pu, pq, pthnl, pthl, ptke
  implicit none

  integer, parameter :: nz = 40
  integer, parameter :: ncase = 30
  integer, parameter :: ndx = 6
  real, parameter :: dxsweep(ndx) = &
       (/ 100.0, 250.0, 500.0, 1000.0, 3000.0, 9000.0 /)

  ! WRF module_model_constants values for the default single-precision build.
  real, parameter :: g = 9.81
  real, parameter :: r_d = 287.0
  real, parameter :: r_v = 461.6
  real, parameter :: cp = 7.0 * r_d / 2.0
  real, parameter :: rovcp = r_d / cp
  real, parameter :: rovg = r_d / g
  real, parameter :: xlv = 2.5e6
  real, parameter :: ep1 = r_v / r_d - 1.0
  real, parameter :: ep2 = r_d / r_v
  real, parameter :: karman = 0.4

  ! One column per call, memory dims == tile dims, kme = kte+1 exactly as the
  ! scheme requires for p3di.  its=ite=jts=jte=1 so a case cannot borrow a
  ! neighbour's scratch.
  real, dimension(1, nz + 1, 1) :: u3d, v3d, th3d, t3d, qv3d, qc3d, qi3d
  real, dimension(1, nz + 1, 1) :: p3d, p3di, pi3d, dz8w
  real, dimension(1, nz + 1, 1) :: rublten, rvblten, rthblten
  real, dimension(1, nz + 1, 1) :: rqvblten, rqcblten, rqiblten
  real, dimension(1, nz + 1, 1) :: exch_h, tke_pbl, el_pbl
  real, dimension(1, nz + 1, 1) :: rublten_b, rvblten_b, rthblten_b
  real, dimension(1, nz + 1, 1) :: rqvblten_b, rqcblten_b, rqiblten_b
  real, dimension(1, nz + 1, 1) :: exch_h_b, tke_pbl_b, el_pbl_b
  real, dimension(1, 1) :: psfc, znt, ust, hpbl, psim, psih
  real, dimension(1, 1) :: xland, hfx, qfx, wspd, br, u10, v10
  real, dimension(1, 1) :: ctopo, ctopo2, corf, wstar, delta, regime
  real, dimension(1, 1) :: hpbl_b, u10_b, v10_b, wspd_b, ust_b, znt_b
  real, dimension(1, 1) :: wstar_b, delta_b, regime_b
  integer, dimension(1, 1) :: kpbl2d, kpbl2d_b
  real, dimension(nz + 1) :: znu, znw
  real :: p_top

  real, dimension(nz) :: tke_in
  real :: dtstep, dxv, dyv
  integer :: tkediag
  real :: in_psfc, in_znt, in_ust, in_hfx, in_qfx, in_wspd, in_br
  real :: in_psim, in_psih, in_xland, in_corf, in_u10, in_v10
  real :: outa_u10, outa_v10, outa_wspd
  character(len=1024) :: level_path, surface_path, partition_path
  integer :: icase, idx, k, ulev, usfc, upart
  real :: nzero, subn, minnorm

  call get_command_argument(1, level_path)
  call get_command_argument(2, surface_path)
  call get_command_argument(3, partition_path)
  if (len_trim(level_path) == 0 .or. len_trim(surface_path) == 0 .or. &
      len_trim(partition_path) == 0) then
    write(*, '(A)') 'usage: run_bl_shinhong LEVELS.csv SURFACE.csv PARTITION.csv'
    error stop 2
  end if

  nzero = sign(0.0, -1.0)
  subn = transfer(1, 0.0)
  minnorm = transfer(8388608, 0.0)

  ! ---- partition-function micro-fixture -------------------------------------
  ! The five scale-dependency curves probed directly, including the h == 0
  ! early-out and both clamps.  h >= 12 m elsewhere because hpbl in the scheme
  ! is bounded below by za(1) (module_bl_shinhong.F:746), so tiny-h NaN routes
  ! are unreachable from the scheme; h = 0 exactly IS reachable (cslen of a
  ! non-convective column, line 650) and returns 1 by construction.
  call write_partition_fixture()

  open(newunit=ulev, file=trim(level_path), status='replace', action='write')
  write(ulev, '(A)') 'case,dxi,k,u,v,t,qv,qc,qi,p2d,pi2d,p2di_k,p2di_kp1,' // &
    'dz8w,tke_in,rublten,rvblten,rthblten,rqvblten,rqcblten,rqiblten,' // &
    'exch_h,tke_out,el_out,rublten_tw,rvblten_tw,exch_h_tw'
  open(newunit=usfc, file=trim(surface_path), status='replace', action='write')
  write(usfc, '(A)') 'case,dxi,nz,dt,tke_diag,dx,dy,psfc,znt,ust,hfx,qfx,' // &
    'wspd_in,br,psim,psih,xland,corf,u10_in,v10_in,' // &
    'hpbl,kpbl,wstar,delta,u10_out,v10_out,wspd_out,' // &
    'hpbl_tw,kpbl_tw,wstar_tw,delta_tw,u10_tw,v10_tw'

  do icase = 1, ncase
    do idx = 1, ndx
      call build_case(icase, idx, dtstep, dxv, dyv, tkediag, tke_in)
      in_psfc = psfc(1, 1); in_znt = znt(1, 1); in_ust = ust(1, 1)
      in_hfx = hfx(1, 1); in_qfx = qfx(1, 1); in_wspd = wspd(1, 1)
      in_br = br(1, 1); in_psim = psim(1, 1); in_psih = psih(1, 1)
      in_xland = xland(1, 1); in_corf = corf(1, 1)
      in_u10 = u10(1, 1); in_v10 = v10(1, 1)

      ! ---- arm A: ctopo = ctopo2 = 1 (topo_wind=0, the WRF default) --------
      rublten = 0.0; rvblten = 0.0; rthblten = 0.0
      rqvblten = 0.0; rqcblten = 0.0; rqiblten = 0.0
      exch_h = 0.0
      el_pbl = 0.0
      tke_pbl = 0.0
      do k = 1, nz
        tke_pbl(1, k, 1) = tke_in(k)
      end do
      hpbl = 0.0; kpbl2d = 0; wstar = 0.0; delta = 0.0; regime = 0.0
      ctopo = 1.0; ctopo2 = 1.0

      call shinhong(u3d=u3d, v3d=v3d, th3d=th3d, t3d=t3d,                     &
                    qv3d=qv3d, qc3d=qc3d, qi3d=qi3d,                          &
                    p3d=p3d, p3di=p3di, pi3d=pi3d,                            &
                    rublten=rublten, rvblten=rvblten, rthblten=rthblten,      &
                    rqvblten=rqvblten, rqcblten=rqcblten, rqiblten=rqiblten,  &
                    flag_qi=.true.,                                           &
                    cp=cp, g=g, rovcp=rovcp, rd=r_d, rovg=rovg,               &
                    ep1=ep1, ep2=ep2, karman=karman, xlv=xlv, rv=r_v,         &
                    dz8w=dz8w, psfc=psfc,                                     &
                    znu=znu, znw=znw, p_top=p_top,                            &
                    znt=znt, ust=ust, hpbl=hpbl, psim=psim, psih=psih,        &
                    xland=xland, hfx=hfx, qfx=qfx, wspd=wspd, br=br,          &
                    dt=dtstep, kpbl2d=kpbl2d,                                 &
                    exch_h=exch_h,                                            &
                    u10=u10, v10=v10,                                         &
                    shinhong_tke_diag=tkediag,                                &
                    tke_pbl=tke_pbl, el_pbl=el_pbl, corf=corf,                &
                    dx=dxv, dy=dyv,                                           &
                    ids=1, ide=1, jds=1, jde=1, kds=1, kde=nz + 1,            &
                    ims=1, ime=1, jms=1, jme=1, kms=1, kme=nz + 1,            &
                    its=1, ite=1, jts=1, jte=1, kts=1, kte=nz,                &
                    ctopo=ctopo, ctopo2=ctopo2,                               &
                    wstar=wstar, delta=delta,                                 &
                    regime=regime)
      outa_u10 = u10(1, 1); outa_v10 = v10(1, 1); outa_wspd = wspd(1, 1)

      ! ---- arm B: ctopo = 0.85, ctopo2 = 0.7 (a legal topo_wind path) ------
      call build_case(icase, idx, dtstep, dxv, dyv, tkediag, tke_in)
      rublten_b = 0.0; rvblten_b = 0.0; rthblten_b = 0.0
      rqvblten_b = 0.0; rqcblten_b = 0.0; rqiblten_b = 0.0
      exch_h_b = 0.0
      el_pbl_b = 0.0
      tke_pbl_b = 0.0
      do k = 1, nz
        tke_pbl_b(1, k, 1) = tke_in(k)
      end do
      hpbl_b = 0.0; kpbl2d_b = 0; wstar_b = 0.0; delta_b = 0.0
      regime_b = 0.0
      ctopo = 0.85; ctopo2 = 0.7

      call shinhong(u3d=u3d, v3d=v3d, th3d=th3d, t3d=t3d,                     &
                    qv3d=qv3d, qc3d=qc3d, qi3d=qi3d,                          &
                    p3d=p3d, p3di=p3di, pi3d=pi3d,                            &
                    rublten=rublten_b, rvblten=rvblten_b,                     &
                    rthblten=rthblten_b,                                      &
                    rqvblten=rqvblten_b, rqcblten=rqcblten_b,                 &
                    rqiblten=rqiblten_b,                                      &
                    flag_qi=.true.,                                           &
                    cp=cp, g=g, rovcp=rovcp, rd=r_d, rovg=rovg,               &
                    ep1=ep1, ep2=ep2, karman=karman, xlv=xlv, rv=r_v,         &
                    dz8w=dz8w, psfc=psfc,                                     &
                    znu=znu, znw=znw, p_top=p_top,                            &
                    znt=znt_b, ust=ust_b, hpbl=hpbl_b, psim=psim, psih=psih,  &
                    xland=xland, hfx=hfx, qfx=qfx, wspd=wspd_b, br=br,        &
                    dt=dtstep, kpbl2d=kpbl2d_b,                               &
                    exch_h=exch_h_b,                                          &
                    u10=u10_b, v10=v10_b,                                     &
                    shinhong_tke_diag=tkediag,                                &
                    tke_pbl=tke_pbl_b, el_pbl=el_pbl_b, corf=corf,            &
                    dx=dxv, dy=dyv,                                           &
                    ids=1, ide=1, jds=1, jde=1, kds=1, kde=nz + 1,            &
                    ims=1, ime=1, jms=1, jme=1, kms=1, kme=nz + 1,            &
                    its=1, ite=1, jts=1, jte=1, kts=1, kte=nz,                &
                    ctopo=ctopo, ctopo2=ctopo2,                               &
                    wstar=wstar_b, delta=delta_b,                             &
                    regime=regime_b)

      do k = 1, nz
        write(ulev, '(I0,",",I0,",",I0)', advance='no') icase, idx, k
        write(ulev, '(24(",",ES24.16E3))')                                    &
          u3d(1, k, 1), v3d(1, k, 1), t3d(1, k, 1),                           &
          qv3d(1, k, 1), qc3d(1, k, 1), qi3d(1, k, 1),                        &
          p3d(1, k, 1), pi3d(1, k, 1), p3di(1, k, 1), p3di(1, k + 1, 1),      &
          dz8w(1, k, 1), tke_in(k),                                           &
          rublten(1, k, 1), rvblten(1, k, 1), rthblten(1, k, 1),              &
          rqvblten(1, k, 1), rqcblten(1, k, 1), rqiblten(1, k, 1),            &
          exch_h(1, k, 1), tke_pbl(1, k, 1), el_pbl(1, k, 1),                 &
          rublten_b(1, k, 1), rvblten_b(1, k, 1), exch_h_b(1, k, 1)
      end do

      write(usfc, '(I0,",",I0,",",I0,",",ES24.16E3,",",I0)', advance='no')    &
        icase, idx, nz, dtstep, tkediag
      write(usfc, '(15(",",ES24.16E3))', advance='no')                        &
        dxv, dyv, in_psfc, in_znt, in_ust, in_hfx, in_qfx,                    &
        in_wspd, in_br, in_psim, in_psih, in_xland,                           &
        in_corf, in_u10, in_v10
      write(usfc, '(",",ES24.16E3,",",I0,4(",",ES24.16E3))', advance='no')    &
        hpbl(1, 1), kpbl2d(1, 1), wstar(1, 1), delta(1, 1),                   &
        outa_u10, outa_v10
      write(usfc, '(",",ES24.16E3)', advance='no') outa_wspd
      write(usfc, '(",",ES24.16E3,",",I0,4(",",ES24.16E3))')                  &
        hpbl_b(1, 1), kpbl2d_b(1, 1), wstar_b(1, 1), delta_b(1, 1),           &
        u10_b(1, 1), v10_b(1, 1)
    end do
  end do

  close(ulev)
  close(usfc)

contains

  subroutine write_partition_fixture()
    integer :: ih, id
    real :: hgrid(8), dgrid(14), dval, hval
    hgrid = (/ 0.0, 12.0, 200.0, 500.0, 1000.0, 1500.0, 2100.0, 4200.0 /)
    dgrid = (/ 50.0, 100.0, 158.0, 250.0, 500.0, 707.0, 1000.0, 1414.0,     &
               2000.0, 3162.0, 5000.0, 10000.0, 20000.0, 50000.0 /)
    open(newunit=upart, file=trim(partition_path), status='replace',        &
         action='write')
    write(upart, '(A)') 'd,h,pu,pq,pthnl,pthl,ptke'
    do ih = 1, size(hgrid)
      do id = 1, size(dgrid)
        dval = dgrid(id)
        hval = hgrid(ih)
        write(upart, '(ES24.16E3,6(",",ES24.16E3))')                        &
          dval, hval, pu(dval, hval), pq(dval, hval), pthnl(dval, hval),    &
          pthl(dval, hval), ptke(dval, hval)
      end do
    end do
    close(upart)
  end subroutine write_partition_fixture

  subroutine build_case(ic, jdx, dt_out, dx_out, dy_out, tkediag_out, tke0)
    integer, intent(in) :: ic, jdx
    real, intent(out) :: dt_out, dx_out, dy_out
    integer, intent(out) :: tkediag_out
    real, intent(out) :: tke0(nz)
    real :: theta(nz), zl(nz + 1), zc(nz)
    real :: th0, lapse_lo, lapse_hi, zinv, jump
    real :: ubase, ushear, vbase, vshear, qv0, qvscale
    real :: dzbase, dzgrow, tkeamp, tkescale, zfrac
    integer :: kk, kcl_lo, kcl_hi
    real :: qcval, qival

    dx_out = dxsweep(jdx)
    dy_out = dxsweep(jdx)
    dt_out = 45.0
    tkediag_out = 1

    ! --- grid -----------------------------------------------------------
    dzbase = 25.0
    dzgrow = 20.0
    if (ic == 23) then
      dzbase = 500.0            ! rlamdz saturates at its 300 m cap
      dzgrow = 0.0
    end if
    do kk = 1, nz
      dz8w(1, kk, 1) = dzbase + dzgrow * real(kk - 1)
    end do
    dz8w(1, nz + 1, 1) = 0.0
    zl(1) = 0.0
    do kk = 1, nz
      zl(kk + 1) = zl(kk) + dz8w(1, kk, 1)
      zc(kk) = 0.5 * (zl(kk) + zl(kk + 1))
    end do

    ! --- thermodynamic profile -----------------------------------------
    th0 = 300.0
    lapse_lo = 0.002
    lapse_hi = 0.006
    zinv = 1200.0
    jump = 3.0
    ubase = 6.0; ushear = 0.0015; vbase = -1.5; vshear = 0.0008
    qv0 = 0.012; qvscale = 2500.0
    kcl_lo = 0; kcl_hi = -1
    qcval = 0.0; qival = 0.0
    tkeamp = 0.0; tkescale = zinv

    select case (ic)
    case (1)                      ! deep dry convective, developed TKE input
      lapse_lo = 0.0005; zinv = 1800.0; jump = 4.0
      tkeamp = 0.9; tkescale = 1800.0
    case (2)                      ! shallow weak convective (shallow-h edge);
                                  ! ust/wstar ~ 0.30 -> csfac tanh ramp
      zinv = 500.0; jump = 2.0
    case (3, 4)                   ! stable land, weak and strong
      lapse_lo = 0.012; zinv = 300.0; jump = 0.5
    case (5, 6, 7, 8, 9)          ! near-neutral, br sign probes
      lapse_lo = 0.0; zinv = 900.0; jump = 1.5
    case (10, 11)                 ! ocean stable, brcr_sbro Rossby form
      lapse_lo = 0.010; zinv = 200.0; jump = 0.5
      ubase = 3.0; ushear = 0.0005; vbase = -2.0; vshear = 0.0003
    case (12, 13)                 ! zero / subnormal surface coupling
      lapse_lo = 0.001; zinv = 800.0; jump = 2.0
    case (14)                     ! strong capping inversion (deltaoh clamps)
      lapse_lo = 0.0005; zinv = 1000.0; jump = 8.0
    case (15)                     ! weak capping: deltaoh -> hpbl cap,
                                  ! rigs < -cpent, entfmin floor binds
      lapse_lo = 0.0005; zinv = 1000.0; jump = 0.3
    case (16)                     ! qc exactly on WRF's 0.01e-3 imvdif test
      lapse_lo = 0.0002; zinv = 900.0; jump = 6.0
      kcl_lo = 8; kcl_hi = 11; qcval = 0.01e-3
    case (17)                     ! qc one float32 step above the test
      lapse_lo = 0.0002; zinv = 900.0; jump = 6.0
      kcl_lo = 8; kcl_hi = 11
      qcval = nearest(0.01e-3, 1.0)
    case (18)                     ! imvdif in-cloud Ri above the PBL, and ice
      lapse_lo = 0.004; zinv = 400.0; jump = 1.0
      kcl_lo = 28; kcl_hi = 33; qcval = 6.0e-4; qival = 2.0e-4
    case (19)                     ! strong shear, unstable free-atmosphere Ri
      lapse_lo = 0.0008; zinv = 600.0; jump = 0.2
      ubase = 2.0; ushear = 0.010; vbase = 0.0; vshear = 0.004
    case (20)                     ! very stable free atmosphere, prmax clamp
      lapse_lo = 0.002; lapse_hi = 0.030; zinv = 400.0; jump = 1.0
      ubase = 4.0; ushear = 0.0002; vbase = 0.0; vshear = 0.0001
    case (21)                     ! gamcrt / gamcrq saturation, developed TKE
      lapse_lo = 0.0002; zinv = 2500.0; jump = 5.0
      qv0 = 0.020
      tkeamp = 1.5; tkescale = 2500.0
    case (22)                     ! PBL fills the column: kpbl == kte.
                                  ! tke_diag = 0 here, deliberately: with
                                  ! tke_diag = 1 the pinned source reads
                                  ! q2xk(kpbl+1) == q2xk(kte+1), one past the
                                  ! end (module_bl_shinhong.F:1557), and the
                                  ! CSV would record stack garbage.  See
                                  ! README "Not covered, deliberately".
      lapse_lo = 0.0; lapse_hi = 0.0; zinv = 1.0e6; jump = 0.0
      tkediag_out = 0
    case (23)                     ! coarse 500 m dz grid
      lapse_lo = 0.001; zinv = 1500.0; jump = 3.0
    case (24)                     ! subnormal and signed-zero moisture
      lapse_lo = 0.001; zinv = 800.0; jump = 2.0
      qv0 = 0.0
    case (25)                     ! ust/wstar ~ 0.5: csfac at its 2.0 peak.
                                  ! TKE input decays LINEARLY through the
                                  ! whole column so q2 still carries a
                                  ! gradient across kpbl: dex /= 0 at :1557
                                  ! and efxpbl -- the entrainment TKE flux
                                  ! vdifq consumes -- is nonzero.  Every
                                  ! parabolic profile dies below the PBL top
                                  ! (zfrac 0.95) and left that arm at -0.0.
      lapse_lo = 0.001; zinv = 900.0; jump = 2.5
      tkeamp = -0.8; tkescale = 1350.0
    case (26)                     ! uniform wind: dux = dvx = 0 exactly ->
                                  ! rigs = rigsmax, ufxpbl/vfxpbl zero branch
      lapse_lo = 0.001; zinv = 900.0; jump = 2.5
      ubase = 5.0; ushear = 0.0; vbase = -2.0; vshear = 0.0
    case (27)                     ! case 1 with tke_diag = 0: pins that the
                                  ! OFF path leaves tke untouched and still
                                  ! zeroes el_pbl (line 673)
      lapse_lo = 0.0005; zinv = 1800.0; jump = 4.0
      tkeamp = 0.9; tkescale = 1800.0
      tkediag_out = 0
    case (28)                     ! anisotropic grid: dy = 4 dx pins the
                                  ! sqrt(dx*dy) contract
      lapse_lo = 0.0005; zinv = 1400.0; jump = 3.0
      dy_out = 4.0 * dx_out
    case (29)                     ! short acoustic-step dt
      lapse_lo = 0.0005; zinv = 1400.0; jump = 3.0
      dt_out = 6.0
    case (30)                     ! negative corf: mixlen f = max(corf,eps1)
                                  ! clamps a southern-hemisphere f to 1e-12;
                                  ! high ust/wstar limb
      lapse_lo = 0.001; zinv = 700.0; jump = 2.0
      ubase = 12.0; ushear = 0.002; vbase = 4.0; vshear = 0.001
    end select

    do kk = 1, nz
      if (zc(kk) <= zinv) then
        theta(kk) = th0 + lapse_lo * zc(kk)
      else
        theta(kk) = th0 + lapse_lo * zinv + jump + lapse_hi * (zc(kk) - zinv)
      end if
      u3d(1, kk, 1) = ubase + ushear * zc(kk)
      v3d(1, kk, 1) = vbase + vshear * zc(kk)
      qv3d(1, kk, 1) = qv0 * exp(-zc(kk) / qvscale)
      qc3d(1, kk, 1) = 0.0
      qi3d(1, kk, 1) = 0.0
    end do
    do kk = max(kcl_lo, 1), min(kcl_hi, nz)
      qc3d(1, kk, 1) = qcval
      qi3d(1, kk, 1) = qival
    end do

    ! --- input SGS TKE ---------------------------------------------------
    ! epsq2l/2 = 0.005 is shinhonginit's cold-start value; the developed
    ! parabolic profile (tkeamp > 0) exercises prodq2/vdifq away from the
    ! floor; the linear ramp (tkeamp < 0, case 25) keeps a q2 gradient
    ! across the PBL top so dex and efxpbl (:1555-1559) are nonzero.
    do kk = 1, nz
      zfrac = zc(kk) / tkescale
      tke0(kk) = 0.005
      if (tkeamp > 0.0 .and. zfrac < 1.2) then
        tke0(kk) = 0.005 + tkeamp * max(0.0, 1.0 - ((zfrac - 0.35) / 0.60) ** 2)
      else if (tkeamp < 0.0) then
        tke0(kk) = 0.005 - tkeamp * max(0.0, 1.0 - zfrac)
      end if
    end do

    ! --- pressure, Exner, temperature ----------------------------------
    psfc(1, 1) = 100000.0
    if (ic == 23) psfc(1, 1) = 98000.0
    do kk = 1, nz + 1
      p3di(1, kk, 1) = psfc(1, 1) * exp(-zl(kk) / 8500.0)
    end do
    do kk = 1, nz
      p3d(1, kk, 1) = 0.5 * (p3di(1, kk, 1) + p3di(1, kk + 1, 1))
      pi3d(1, kk, 1) = (p3d(1, kk, 1) / 100000.0) ** rovcp
      t3d(1, kk, 1) = theta(kk) * pi3d(1, kk, 1)
      th3d(1, kk, 1) = theta(kk)
    end do
    u3d(1, nz + 1, 1) = 0.0; v3d(1, nz + 1, 1) = 0.0
    t3d(1, nz + 1, 1) = 0.0; th3d(1, nz + 1, 1) = 0.0
    qv3d(1, nz + 1, 1) = 0.0; qc3d(1, nz + 1, 1) = 0.0; qi3d(1, nz + 1, 1) = 0.0
    p3d(1, nz + 1, 1) = 0.0; pi3d(1, nz + 1, 1) = 1.0

    ! znu/znw/p_top are declared by the wrapper and never referenced; filled
    ! with plausible values so the call is the WRF call.
    p_top = p3di(1, nz + 1, 1)
    do kk = 1, nz + 1
      znw(kk) = (p3di(1, kk, 1) - p_top) / (psfc(1, 1) - p_top)
    end do
    do kk = 1, nz
      znu(kk) = 0.5 * (znw(kk) + znw(kk + 1))
    end do
    znu(nz + 1) = 0.0

    ! --- surface coupling ----------------------------------------------
    znt(1, 1) = 0.10
    xland(1, 1) = 1.0
    psim(1, 1) = 6.5
    psih(1, 1) = 8.5
    corf(1, 1) = 1.0e-4
    u10(1, 1) = u3d(1, 1, 1)
    v10(1, 1) = v3d(1, 1, 1)
    select case (ic)
    case (1, 27)
      hfx(1, 1) = 250.0; qfx(1, 1) = 1.5e-4; ust(1, 1) = 0.55; br(1, 1) = -0.35
    case (2)
      hfx(1, 1) = 40.0; qfx(1, 1) = 2.0e-5; ust(1, 1) = 0.25; br(1, 1) = -0.05
    case (3)
      hfx(1, 1) = -25.0; qfx(1, 1) = 0.0; ust(1, 1) = 0.12; br(1, 1) = 0.35
    case (4)
      hfx(1, 1) = -60.0; qfx(1, 1) = -1.0e-6; ust(1, 1) = 0.08; br(1, 1) = 1.20
    case (5)
      hfx(1, 1) = 0.5; qfx(1, 1) = 1.0e-7; ust(1, 1) = 0.30; br(1, 1) = 0.0
    case (6)
      hfx(1, 1) = 0.5; qfx(1, 1) = 1.0e-7; ust(1, 1) = 0.30; br(1, 1) = nzero
    case (7)
      hfx(1, 1) = 0.5; qfx(1, 1) = 1.0e-7; ust(1, 1) = 0.30; br(1, 1) = subn
    case (8)
      hfx(1, 1) = 0.5; qfx(1, 1) = 1.0e-7; ust(1, 1) = 0.30; br(1, 1) = -subn
    case (9)
      hfx(1, 1) = 0.5; qfx(1, 1) = 1.0e-7; ust(1, 1) = 0.30; br(1, 1) = minnorm
    case (10)
      hfx(1, 1) = -8.0; qfx(1, 1) = 3.0e-6; ust(1, 1) = 0.05; br(1, 1) = 0.60
      xland(1, 1) = 2.0; znt(1, 1) = 1.0e-4
      u10(1, 1) = 3.0; v10(1, 1) = -2.0
    case (11)
      hfx(1, 1) = -8.0; qfx(1, 1) = 3.0e-6; ust(1, 1) = 0.05; br(1, 1) = 0.60
      xland(1, 1) = 2.0; znt(1, 1) = 1.0e-4
      u10(1, 1) = subn; v10(1, 1) = nzero
    case (12)
      hfx(1, 1) = 0.0; qfx(1, 1) = 0.0; ust(1, 1) = 0.0; br(1, 1) = 0.02
    case (13)
      hfx(1, 1) = 0.0; qfx(1, 1) = nzero; ust(1, 1) = subn; br(1, 1) = -0.02
    case (14, 15)
      hfx(1, 1) = 180.0; qfx(1, 1) = 1.0e-4; ust(1, 1) = 0.45; br(1, 1) = -0.25
    case (16, 17)
      hfx(1, 1) = 80.0; qfx(1, 1) = 5.0e-5; ust(1, 1) = 0.35; br(1, 1) = -0.10
    case (18)
      hfx(1, 1) = 15.0; qfx(1, 1) = 1.0e-5; ust(1, 1) = 0.20; br(1, 1) = -0.02
    case (19)
      hfx(1, 1) = 60.0; qfx(1, 1) = 3.0e-5; ust(1, 1) = 0.60; br(1, 1) = -0.04
    case (20)
      hfx(1, 1) = -15.0; qfx(1, 1) = 0.0; ust(1, 1) = 0.10; br(1, 1) = 0.80
    case (21)
      hfx(1, 1) = 600.0; qfx(1, 1) = 8.0e-4; ust(1, 1) = 0.90; br(1, 1) = -1.50
    case (22)
      hfx(1, 1) = 300.0; qfx(1, 1) = 2.0e-4; ust(1, 1) = 0.70; br(1, 1) = -0.90
    case (23)
      hfx(1, 1) = 120.0; qfx(1, 1) = 6.0e-5; ust(1, 1) = 0.40; br(1, 1) = -0.20
    case (24)
      hfx(1, 1) = 20.0; qfx(1, 1) = subn; ust(1, 1) = 0.18; br(1, 1) = -0.01
      qv3d(1, 1, 1) = subn
      qv3d(1, 2, 1) = nzero
      qv3d(1, 3, 1) = minnorm
      qc3d(1, 4, 1) = subn
      qi3d(1, 5, 1) = -subn
      qc3d(1, 6, 1) = nzero
    case (25)
      hfx(1, 1) = 100.0; qfx(1, 1) = 6.0e-5; ust(1, 1) = 0.68; br(1, 1) = -0.15
    case (26)
      hfx(1, 1) = 100.0; qfx(1, 1) = 6.0e-5; ust(1, 1) = 0.40; br(1, 1) = -0.15
    case (28, 29)
      hfx(1, 1) = 200.0; qfx(1, 1) = 1.2e-4; ust(1, 1) = 0.50; br(1, 1) = -0.30
    case (30)
      hfx(1, 1) = 60.0; qfx(1, 1) = 4.0e-5; ust(1, 1) = 1.10; br(1, 1) = -0.02
      corf(1, 1) = -1.0e-4
    end select
    if (ic == 26) corf(1, 1) = 0.0

    wspd(1, 1) = sqrt(u3d(1, 1, 1) ** 2 + v3d(1, 1, 1) ** 2)
    if (wspd(1, 1) < 0.1) wspd(1, 1) = 0.1
    ! One probe below any port-side wspd guard so a guard's effect is
    ! measured rather than assumed harmless (the scheme divides by wspd at
    ! module_bl_shinhong.F:794 and by wspd1 = |V1|+1e-9 at :1387).
    if (ic == 13) wspd(1, 1) = 1.0e-10

    ! Arm-B fresh copies of every INOUT surface field.
    znt_b = znt; ust_b = ust; wspd_b = wspd
    u10_b = u10; v10_b = v10
  end subroutine build_case

end program run_bl_shinhong_oracle
