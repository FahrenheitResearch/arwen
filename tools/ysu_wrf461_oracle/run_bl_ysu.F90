program run_bl_ysu_oracle
  ! Drive bl_ysu_run from the byte-unmodified WRF v4.6.1
  ! phys/physics_mmm/bl_ysu.F90 and dump every input and every output as
  ! float32 CSV.  bl_ysu_run is PUBLIC and calls nothing outside its own
  ! module (tridin_ysu, tridi2n, get_pblh), so unlike the RUC and Noah-MP
  ! leaves there is no block to extract and no WRF logging or MPI shim to
  ! stub: the whole scheme compiles as it ships.  Its only USE is
  ! ccpp_kind_types, which build.sh compiles from the pinned tree with
  ! WRF's own default -DRWORDSIZE=4, i.e. kind_phys == real(4).
  !
  ! The scheme is called TWICE per fixture.
  !   * ctopo absent   -- the contract gpuwm/core/kernels/ysu.cu was written
  !                       against: bl_ysu.F90:1315 takes ad(i,1) = 1+fric.
  !   * ctopo = ctopo2 = 1 -- what WRF's OWN driver does.  module_bl_ysu.F:404
  !                       always passes ctopo_hv/ctopo2_hv, and the Registry
  !                       default (topo_wind=0) fills both with 1.0, so
  !                       bl_ysu.F90:1308 takes
  !                         ad(i,1) = 1+fric*vconvlim+ctopo*fric*(1-vconvlim)
  !                       which is NOT 1+fric in float32 and additionally
  !                       depends on the paj TKE block and get_pblh.
  ! Both momentum tendencies are written so the gap can be measured rather
  ! than argued about.
  !
  ! Nothing here invents an expected value: every number in the two CSVs is
  ! either an input this program constructed or a word bl_ysu_run wrote.
  use ccpp_kind_types, only: kind_phys
  use bl_ysu, only: bl_ysu_run
  implicit none

  integer, parameter :: nz = 40
  integer, parameter :: ncase = 24
  integer, parameter :: nmix = 1
  real(kind=kind_phys), parameter :: dtstep = 45.0

  ! WRF module_model_constants values for the default single-precision build.
  real(kind=kind_phys), parameter :: g = 9.81
  real(kind=kind_phys), parameter :: r_d = 287.0
  real(kind=kind_phys), parameter :: r_v = 461.6
  real(kind=kind_phys), parameter :: cp = 7.0 * r_d / 2.0
  real(kind=kind_phys), parameter :: rovcp = r_d / cp
  real(kind=kind_phys), parameter :: rovg = r_d / g
  real(kind=kind_phys), parameter :: xlv = 2.5e6
  real(kind=kind_phys), parameter :: ep1 = r_v / r_d - 1.0
  real(kind=kind_phys), parameter :: ep2 = r_d / r_v
  real(kind=kind_phys), parameter :: karman = 0.4

  ! One column per fixture case; bl_ysu_run is called with its=ite=1 so that
  ! a case cannot borrow a neighbour's scratch.
  real(kind=kind_phys), dimension(1, nz) :: ux, vx, tx, qvx, qcx, qix
  real(kind=kind_phys), dimension(1, nz) :: p2d, pi2d, dz8w2d, rthraten
  real(kind=kind_phys), dimension(1, nz + 1) :: p2di
  real(kind=kind_phys), dimension(1, nz, nmix) :: qmix, qmixtnp
  real(kind=kind_phys), dimension(1, nz) :: utnp, vtnp, ttnp
  real(kind=kind_phys), dimension(1, nz) :: qvtnp, qctnp, qitnp
  real(kind=kind_phys), dimension(1, nz) :: exch_hx, exch_mx
  real(kind=kind_phys), dimension(1, nz) :: utnp_ct, vtnp_ct
  real(kind=kind_phys), dimension(1) :: psfcpa, znt, ust, hpbl, psim, psih
  real(kind=kind_phys), dimension(1) :: xland, hfx, qfx, wspd, br
  real(kind=kind_phys), dimension(1) :: wstar, delta, u10, v10
  real(kind=kind_phys), dimension(1) :: ctopo, ctopo2
  real(kind=kind_phys), dimension(1) :: hpbl_ct, wstar_ct, delta_ct
  integer, dimension(1) :: kpbl1d, kpbl1d_ct

  character(len=1024) :: level_path, surface_path
  character(len=256) :: errmsg
  integer :: errflg
  integer :: icase, k, ulev, usfc
  logical :: topdown

  real(kind=kind_phys) :: zq(nz + 1), za(nz)
  real(kind=kind_phys) :: nzero, subn, minnorm

  call get_command_argument(1, level_path)
  call get_command_argument(2, surface_path)
  if (len_trim(level_path) == 0 .or. len_trim(surface_path) == 0) then
    write(*, '(A)') 'usage: run_bl_ysu LEVELS.csv SURFACE.csv'
    error stop 2
  end if

  nzero = sign(0.0_kind_phys, -1.0_kind_phys)
  subn = transfer(1, 0.0_kind_phys)
  minnorm = transfer(8388608, 0.0_kind_phys)

  open(newunit=ulev, file=trim(level_path), status='replace', action='write')
  write(ulev, '(A)') 'case,k,ux,vx,tx,qvx,qcx,qix,p2d,pi2d,p2di_k,p2di_kp1,' // &
    'dz8w,rthraten,qmix,utnp,vtnp,ttnp,rthblten,qvtnp,qctnp,qitnp,' // &
    'exch_hx,exch_mx,qmixtnp,utnp_ctopo,vtnp_ctopo'
  open(newunit=usfc, file=trim(surface_path), status='replace', action='write')
  write(usfc, '(A)') 'case,nz,dt,topdown,psfcpa,znt,ust,hfx,qfx,wspd,br,' // &
    'psim,psih,xland,u10,v10,hpbl,kpbl,wstar,delta,' // &
    'hpbl_ctopo,kpbl_ctopo,wstar_ctopo,delta_ctopo'

  do icase = 1, ncase
    call build_case(icase, topdown)

    ! zq/za are recomputed here only to place the fixture's cloud layer and
    ! are not handed to the scheme; bl_ysu_run rebuilds them from dz8w2d.
    zq(1) = 0.0
    do k = 1, nz
      zq(k + 1) = zq(k) + dz8w2d(1, k)
      za(k) = 0.5 * (zq(k) + zq(k + 1))
    end do

    qmix(1, :, 1) = qvx(1, :) * 0.5
    utnp = 0.0; vtnp = 0.0; ttnp = 0.0
    qvtnp = 0.0; qctnp = 0.0; qitnp = 0.0
    exch_hx = 0.0; exch_mx = 0.0; qmixtnp = 0.0
    hpbl = 0.0; kpbl1d = 0; wstar = 0.0; delta = 0.0
    errmsg = ''; errflg = -1

    call bl_ysu_run(ux=ux, vx=vx, tx=tx, qvx=qvx, qcx=qcx, qix=qix,          &
                    nmix=nmix, qmix=qmix, p2d=p2d, p2di=p2di, pi2d=pi2d,     &
                    f_qc=.true., f_qi=.true.,                               &
                    utnp=utnp, vtnp=vtnp, ttnp=ttnp, qvtnp=qvtnp,            &
                    qctnp=qctnp, qitnp=qitnp, qmixtnp=qmixtnp,               &
                    cp=cp, g=g, rovcp=rovcp, rd=r_d, rovg=rovg,              &
                    ep1=ep1, ep2=ep2, karman=karman, xlv=xlv, rv=r_v,        &
                    dz8w2d=dz8w2d, psfcpa=psfcpa,                            &
                    znt=znt, ust=ust, hpbl=hpbl,                             &
                    psim=psim, psih=psih, xland=xland,                       &
                    hfx=hfx, qfx=qfx, wspd=wspd, br=br,                      &
                    dt=dtstep, kpbl1d=kpbl1d,                                &
                    exch_hx=exch_hx, exch_mx=exch_mx,                        &
                    wstar=wstar, delta=delta, u10=u10, v10=v10,              &
                    rthraten=rthraten, ysu_topdown_pblmix=topdown,           &
                    flag_bep=.false.,                                        &
                    its=1, ite=1, kte=nz, kme=nz + 1,                        &
                    errmsg=errmsg, errflg=errflg)
    if (errflg /= 0) then
      write(*, '(A,I0,A,A)') 'case ', icase, ' errmsg: ', trim(errmsg)
      error stop 3
    end if

    ! Second call: the ctopo path WRF's own driver always takes.
    ctopo(1) = 1.0
    ctopo2(1) = 1.0
    utnp_ct = 0.0; vtnp_ct = 0.0
    hpbl_ct = 0.0; kpbl1d_ct = 0; wstar_ct = 0.0; delta_ct = 0.0
    errmsg = ''; errflg = -1
    call bl_ysu_run(ux=ux, vx=vx, tx=tx, qvx=qvx, qcx=qcx, qix=qix,          &
                    nmix=nmix, qmix=qmix, p2d=p2d, p2di=p2di, pi2d=pi2d,     &
                    f_qc=.true., f_qi=.true.,                               &
                    utnp=utnp_ct, vtnp=vtnp_ct, ttnp=ttnp, qvtnp=qvtnp,      &
                    qctnp=qctnp, qitnp=qitnp, qmixtnp=qmixtnp,               &
                    cp=cp, g=g, rovcp=rovcp, rd=r_d, rovg=rovg,              &
                    ep1=ep1, ep2=ep2, karman=karman, xlv=xlv, rv=r_v,        &
                    dz8w2d=dz8w2d, psfcpa=psfcpa,                            &
                    znt=znt, ust=ust, hpbl=hpbl_ct,                          &
                    psim=psim, psih=psih, xland=xland,                       &
                    hfx=hfx, qfx=qfx, wspd=wspd, br=br,                      &
                    dt=dtstep, kpbl1d=kpbl1d_ct,                             &
                    exch_hx=exch_hx, exch_mx=exch_mx,                        &
                    wstar=wstar_ct, delta=delta_ct, u10=u10, v10=v10,        &
                    rthraten=rthraten, ysu_topdown_pblmix=topdown,           &
                    ctopo=ctopo, ctopo2=ctopo2,                              &
                    flag_bep=.false.,                                        &
                    its=1, ite=1, kte=nz, kme=nz + 1,                        &
                    errmsg=errmsg, errflg=errflg)
    if (errflg /= 0) then
      write(*, '(A,I0,A,A)') 'ctopo case ', icase, ' errmsg: ', trim(errmsg)
      error stop 4
    end if

    do k = 1, nz
      write(ulev, '(I0,",",I0)', advance='no') icase, k
      write(ulev, '(25(",",ES24.16E3))')                                     &
        ux(1, k), vx(1, k), tx(1, k), qvx(1, k), qcx(1, k), qix(1, k),        &
        p2d(1, k), pi2d(1, k), p2di(1, k), p2di(1, k + 1), dz8w2d(1, k),      &
        rthraten(1, k), qmix(1, k, 1),                                        &
        utnp(1, k), vtnp(1, k), ttnp(1, k),                                   &
        ! module_bl_ysu.F:452 -- the tendency the WRF driver hands the solver.
        ttnp(1, k) / pi2d(1, k),                                              &
        qvtnp(1, k), qctnp(1, k), qitnp(1, k),                                &
        exch_hx(1, k), exch_mx(1, k), qmixtnp(1, k, 1),                       &
        utnp_ct(1, k), vtnp_ct(1, k)
    end do

    write(usfc, '(I0,",",I0,",",ES24.16E3,",",I0)', advance='no')             &
      icase, nz, dtstep, merge(1, 0, topdown)
    write(usfc, '(12(",",ES24.16E3))', advance='no')                          &
      psfcpa(1), znt(1), ust(1), hfx(1), qfx(1), wspd(1), br(1),              &
      psim(1), psih(1), xland(1), u10(1), v10(1)
    write(usfc, '(",",ES24.16E3,",",I0,2(",",ES24.16E3))', advance='no')      &
      hpbl(1), kpbl1d(1), wstar(1), delta(1)
    write(usfc, '(",",ES24.16E3,",",I0,2(",",ES24.16E3))')                    &
      hpbl_ct(1), kpbl1d_ct(1), wstar_ct(1), delta_ct(1)
  end do

  close(ulev)
  close(usfc)

contains

  subroutine build_case(ic, want_topdown)
    integer, intent(in) :: ic
    logical, intent(out) :: want_topdown
    real(kind=kind_phys) :: theta(nz), zl(nz + 1), zc(nz)
    real(kind=kind_phys) :: th0, lapse_lo, lapse_hi, zinv, jump
    real(kind=kind_phys) :: ubase, ushear, vbase, vshear, qv0, qvscale
    real(kind=kind_phys) :: dzbase, dzgrow
    integer :: kk, kcl_lo, kcl_hi
    real(kind=kind_phys) :: qcval, qival, radval

    ! --- grid -----------------------------------------------------------
    dzbase = 25.0
    dzgrow = 20.0
    if (ic == 22) then
      dzbase = 500.0            ! rlamdz saturates at its 300 m cap
      dzgrow = 0.0
    end if
    do kk = 1, nz
      dz8w2d(1, kk) = dzbase + dzgrow * real(kk - 1, kind_phys)
    end do
    zl(1) = 0.0
    do kk = 1, nz
      zl(kk + 1) = zl(kk) + dz8w2d(1, kk)
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
    qcval = 0.0; qival = 0.0; radval = 0.0
    want_topdown = .true.

    select case (ic)
    case (1)                      ! deep dry convective
      lapse_lo = 0.0005; zinv = 1800.0; jump = 4.0
    case (2)                      ! shallow weak convective
      zinv = 500.0; jump = 2.0
    case (3, 4)                   ! stable, surface inversion
      lapse_lo = 0.012; zinv = 300.0; jump = 0.5
    case (5, 6, 7, 8, 9)          ! near-neutral, br probes
      lapse_lo = 0.0; zinv = 900.0; jump = 1.5
    case (10, 11)                 ! ocean stable
      lapse_lo = 0.010; zinv = 200.0; jump = 0.5
      ubase = 3.0; ushear = 0.0005; vbase = -2.0; vshear = 0.0003
    case (12, 13)                 ! zero / subnormal surface coupling
      lapse_lo = 0.001; zinv = 800.0; jump = 2.0
    case (14)                     ! stratocumulus, liquid
      lapse_lo = 0.0002; zinv = 900.0; jump = 6.0
      kcl_lo = 7; kcl_hi = 11; qcval = 4.0e-4; radval = -1.2e-4
    case (15)                     ! stratocumulus, ice
      lapse_lo = 0.0002; zinv = 900.0; jump = 6.0
      kcl_lo = 7; kcl_hi = 11; qival = 3.0e-4; radval = -9.0e-5
    case (16)                     ! qc exactly on WRF's 0.01e-3 test
      lapse_lo = 0.0002; zinv = 900.0; jump = 6.0
      kcl_lo = 8; kcl_hi = 11; qcval = 0.01e-3; radval = -5.0e-5
    case (17)                     ! qc one float32 step above the test
      lapse_lo = 0.0002; zinv = 900.0; jump = 6.0
      kcl_lo = 8; kcl_hi = 11
      qcval = nearest(0.01e-3_kind_phys, 1.0_kind_phys); radval = -5.0e-5
    case (18)                     ! imvdif in-cloud Ri, above the PBL
      lapse_lo = 0.004; zinv = 400.0; jump = 1.0
      kcl_lo = 28; kcl_hi = 33; qcval = 8.0e-4; radval = -2.0e-5
    case (19)                     ! strong shear, unstable free-atmosphere Ri
      lapse_lo = 0.0008; zinv = 600.0; jump = 0.2
      ubase = 2.0; ushear = 0.010; vbase = 0.0; vshear = 0.004
    case (20)                     ! very stable free atmosphere, prmax clamp
      lapse_lo = 0.002; lapse_hi = 0.030; zinv = 400.0; jump = 1.0
      ubase = 4.0; ushear = 0.0002; vbase = 0.0; vshear = 0.0001
    case (21)                     ! gamcrt / gamcrq saturation
      lapse_lo = 0.0002; zinv = 2500.0; jump = 5.0
      qv0 = 0.020
    case (22)                     ! coarse 500 m grid
      lapse_lo = 0.001; zinv = 1500.0; jump = 3.0
    case (23)                     ! PBL fills the column
      lapse_lo = 0.0; lapse_hi = 0.0; zinv = 1.0e6; jump = 0.0
    case (24)                     ! subnormal and signed-zero moisture
      lapse_lo = 0.001; zinv = 800.0; jump = 2.0
      qv0 = 0.0
    end select

    do kk = 1, nz
      if (zc(kk) <= zinv) then
        theta(kk) = th0 + lapse_lo * zc(kk)
      else
        theta(kk) = th0 + lapse_lo * zinv + jump + lapse_hi * (zc(kk) - zinv)
      end if
      ux(1, kk) = ubase + ushear * zc(kk)
      vx(1, kk) = vbase + vshear * zc(kk)
      qvx(1, kk) = qv0 * exp(-zc(kk) / qvscale)
      qcx(1, kk) = 0.0
      qix(1, kk) = 0.0
      rthraten(1, kk) = 0.0
    end do
    do kk = max(kcl_lo, 1), min(kcl_hi, nz)
      qcx(1, kk) = qcval
      qix(1, kk) = qival
      rthraten(1, kk) = radval
    end do

    ! --- pressure, Exner, temperature ----------------------------------
    psfcpa(1) = 100000.0
    if (ic == 22) psfcpa(1) = 98000.0
    do kk = 1, nz + 1
      p2di(1, kk) = psfcpa(1) * exp(-zl(kk) / 8500.0)
    end do
    do kk = 1, nz
      p2d(1, kk) = 0.5 * (p2di(1, kk) + p2di(1, kk + 1))
      pi2d(1, kk) = (p2d(1, kk) / 100000.0) ** rovcp
      tx(1, kk) = theta(kk) * pi2d(1, kk)
    end do

    ! --- surface coupling ----------------------------------------------
    znt(1) = 0.10
    xland(1) = 1.0
    psim(1) = 6.5
    psih(1) = 8.5
    u10(1) = ux(1, 1)
    v10(1) = vx(1, 1)
    select case (ic)
    case (1)
      hfx(1) = 250.0; qfx(1) = 1.5e-4; ust(1) = 0.55; br(1) = -0.35
    case (2)
      hfx(1) = 40.0; qfx(1) = 2.0e-5; ust(1) = 0.25; br(1) = -0.05
    case (3)
      hfx(1) = -25.0; qfx(1) = 0.0; ust(1) = 0.12; br(1) = 0.35
    case (4)
      hfx(1) = -60.0; qfx(1) = -1.0e-6; ust(1) = 0.08; br(1) = 1.20
    case (5)
      hfx(1) = 0.5; qfx(1) = 1.0e-7; ust(1) = 0.30; br(1) = 0.0
    case (6)
      hfx(1) = 0.5; qfx(1) = 1.0e-7; ust(1) = 0.30; br(1) = nzero
    case (7)
      hfx(1) = 0.5; qfx(1) = 1.0e-7; ust(1) = 0.30; br(1) = subn
    case (8)
      hfx(1) = 0.5; qfx(1) = 1.0e-7; ust(1) = 0.30; br(1) = -subn
    case (9)
      hfx(1) = 0.5; qfx(1) = 1.0e-7; ust(1) = 0.30; br(1) = minnorm
    case (10)
      hfx(1) = -8.0; qfx(1) = 3.0e-6; ust(1) = 0.05; br(1) = 0.60
      xland(1) = 2.0; znt(1) = 1.0e-4
      u10(1) = 3.0; v10(1) = -2.0
    case (11)
      hfx(1) = -8.0; qfx(1) = 3.0e-6; ust(1) = 0.05; br(1) = 0.60
      xland(1) = 2.0; znt(1) = 1.0e-4
      u10(1) = subn; v10(1) = nzero
    case (12)
      hfx(1) = 0.0; qfx(1) = 0.0; ust(1) = 0.0; br(1) = 0.02
    case (13)
      hfx(1) = 0.0; qfx(1) = nzero; ust(1) = subn; br(1) = -0.02
    case (14, 15, 16, 17)
      hfx(1) = 80.0; qfx(1) = 5.0e-5; ust(1) = 0.35; br(1) = -0.10
    case (18)
      hfx(1) = 15.0; qfx(1) = 1.0e-5; ust(1) = 0.20; br(1) = -0.02
    case (19)
      hfx(1) = 60.0; qfx(1) = 3.0e-5; ust(1) = 0.60; br(1) = -0.04
    case (20)
      hfx(1) = -15.0; qfx(1) = 0.0; ust(1) = 0.10; br(1) = 0.80
    case (21)
      hfx(1) = 600.0; qfx(1) = 8.0e-4; ust(1) = 0.90; br(1) = -1.50
    case (22)
      hfx(1) = 120.0; qfx(1) = 6.0e-5; ust(1) = 0.40; br(1) = -0.20
    case (23)
      hfx(1) = 300.0; qfx(1) = 2.0e-4; ust(1) = 0.70; br(1) = -0.90
    case (24)
      hfx(1) = 20.0; qfx(1) = subn; ust(1) = 0.18; br(1) = -0.01
      qvx(1, 1) = subn
      qvx(1, 2) = nzero
      qvx(1, 3) = minnorm
      qcx(1, 4) = subn
      qix(1, 5) = -subn
      qcx(1, 6) = nzero
    end select

    wspd(1) = sqrt(ux(1, 1) ** 2 + vx(1, 1) ** 2)
    if (wspd(1) < 0.1) wspd(1) = 0.1
    ! One probe below the port's max(wspd,1e-9) guard so the guard's effect
    ! is measured rather than assumed harmless.
    if (ic == 13) wspd(1) = 1.0e-10
    if (ic == 20) want_topdown = .false.
  end subroutine build_case

end program run_bl_ysu_oracle
