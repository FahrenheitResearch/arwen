program run_ysu_topo
  ! WRF v4.7.1 YSU (phys/physics_mmm/bl_ysu.F90, byte-unmodified) on the
  ! topo_wind arm: bl_ysu_run called with CTOPO and CTOPO2 present, as WRF's
  ! own driver always calls it (module_bl_ysu.F passes ctopo/ctopo2), and with
  ! the values topo_wind = 1 and 2 put there (start_em.F:1579-1626):
  !
  !   variant 1  ctopo = 1,         ctopo2 = 1     the topo_wind = 0 fill
  !   variant 2  ctopo = ln(100),   ctopo2 = 1     topo_wind = 1, 100 m sub-grid
  !                                                 terrain, no hill top
  !   variant 3  ctopo = 1.575**2,  ctopo2 = 1     topo_wind = 2 at its VAR cap
  !   variant 4  ctopo = 0.5,       ctopo2 = 0.5   topo_wind = 1 hill top,
  !                                                 LAP_HGT = -25
  !   variant 5  ctopo = 0,         ctopo2 = 0     topo_wind = 1 peak,
  !                                                 LAP_HGT < -30
  !
  ! ctopo scales the first-level momentum drag in the implicit solve
  ! (bl_ysu.F90:1306-1314, through the paj TKE block and get_pblh, :1254-1304)
  ! and ctopo2 blends U10/V10 toward the first-level wind (:1402-1408).  Every
  ! column of the WRF v4.6.1 YSU oracle (tools/ysu_wrf461_oracle, the same 24
  ! cases, build_case below copied from its run_bl_ysu.F90) runs under each
  ! variant, with U10/V10 set apart from the first-level wind on land so the
  ! blend is measurable.  The no-ctopo call is recorded beside them.
  !
  ! Outputs through oracle_io, fixtures/ysu_topo/columns: (nz, ncol) and (ncol)
  ! arrays, column = (case-1)*nvar + variant.
  use ccpp_kind_types, only: kind_phys
  use bl_ysu, only: bl_ysu_run
  use oracle_io
  implicit none

  integer, parameter :: nz = 40
  integer, parameter :: ncase = 24
  integer, parameter :: nvar = 5
  integer, parameter :: ncol = ncase * nvar
  integer, parameter :: nmix = 1
  real(kind=kind_phys), parameter :: dtstep = 45.0

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

  real(kind=kind_phys), dimension(1, nz) :: ux, vx, tx, qvx, qcx, qix
  real(kind=kind_phys), dimension(1, nz) :: p2d, pi2d, dz8w2d, rthraten
  real(kind=kind_phys), dimension(1, nz + 1) :: p2di
  real(kind=kind_phys), dimension(1, nz, nmix) :: qmix, qmixtnp
  real(kind=kind_phys), dimension(1, nz) :: utnp, vtnp, ttnp
  real(kind=kind_phys), dimension(1, nz) :: qvtnp, qctnp, qitnp
  real(kind=kind_phys), dimension(1, nz) :: exch_hx, exch_mx
  real(kind=kind_phys), dimension(1) :: psfcpa, znt, ust, hpbl, psim, psih
  real(kind=kind_phys), dimension(1) :: xland, hfx, qfx, wspd, br
  real(kind=kind_phys), dimension(1) :: wstar, delta, u10, v10
  real(kind=kind_phys), dimension(1) :: ctopo, ctopo2
  integer, dimension(1) :: kpbl1d

  ! Recorded arrays.
  real(4), dimension(nz, ncol) :: o_ux, o_vx, o_tx, o_qvx, o_qcx, o_qix
  real(4), dimension(nz, ncol) :: o_p2d, o_pi2d, o_dz8w, o_rthraten
  real(4), dimension(nz + 1, ncol) :: o_p2di
  real(4), dimension(nz, ncol) :: o_ut, o_vt, o_tt, o_qvt, o_qct, o_qit
  real(4), dimension(nz, ncol) :: o_exh, o_exm, o_ut_noctopo, o_vt_noctopo
  real(4), dimension(ncol) :: o_psfc, o_znt, o_ust, o_hfx, o_qfx, o_wspd
  real(4), dimension(ncol) :: o_br, o_psim, o_psih, o_xland
  real(4), dimension(ncol) :: o_u10_in, o_v10_in, o_u10, o_v10
  real(4), dimension(ncol) :: o_ctopo, o_ctopo2, o_hpbl, o_wstar, o_delta
  integer(4), dimension(ncol) :: o_kpbl, o_topdown, o_case

  real(kind=kind_phys) :: var_ctopo(nvar), var_ctopo2(nvar)
  real(kind=kind_phys) :: u10_in, v10_in
  character(len=1024) :: root
  character(len=256) :: errmsg
  integer :: errflg
  integer :: icase, iv, col
  logical :: topdown

  real(kind=kind_phys) :: nzero, subn, minnorm

  call oracle_root(root)
  nzero = sign(0.0_kind_phys, -1.0_kind_phys)
  subn = transfer(1, 0.0_kind_phys)
  minnorm = transfer(8388608, 0.0_kind_phys)

  var_ctopo = (/ 1.0_kind_phys, log(100.0_kind_phys), 1.575_kind_phys * 1.575_kind_phys, &
                 0.5_kind_phys, 0.0_kind_phys /)
  var_ctopo2 = (/ 1.0_kind_phys, 1.0_kind_phys, 1.0_kind_phys, 0.5_kind_phys, 0.0_kind_phys /)

  do icase = 1, ncase
    call build_case(icase, topdown)
    qmix(1, :, 1) = qvx(1, :) * 0.5
    ! On land, 10 m wind apart from the first-level wind, so the ctopo2
    ! blend is visible; the ocean cases keep build_case's own (they feed the
    ! Rossby-number critical Richardson number over water).
    if (xland(1) < 1.5) then
      u10(1) = 0.62 * ux(1, 1)
      v10(1) = 0.62 * vx(1, 1)
    end if
    u10_in = u10(1); v10_in = v10(1)

    ! The no-ctopo call (bl_ysu.F90:1315 takes ad(i,1) = 1 + fric).
    call zero_outputs()
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
    do iv = 1, nvar
      col = (icase - 1) * nvar + iv
      o_ut_noctopo(:, col) = utnp(1, :)
      o_vt_noctopo(:, col) = vtnp(1, :)
    end do

    do iv = 1, nvar
      col = (icase - 1) * nvar + iv
      ctopo(1) = var_ctopo(iv)
      ctopo2(1) = var_ctopo2(iv)
      u10(1) = u10_in; v10(1) = v10_in
      call zero_outputs()
      call bl_ysu_run(ux=ux, vx=vx, tx=tx, qvx=qvx, qcx=qcx, qix=qix,        &
                      nmix=nmix, qmix=qmix, p2d=p2d, p2di=p2di, pi2d=pi2d,   &
                      f_qc=.true., f_qi=.true.,                             &
                      utnp=utnp, vtnp=vtnp, ttnp=ttnp, qvtnp=qvtnp,          &
                      qctnp=qctnp, qitnp=qitnp, qmixtnp=qmixtnp,             &
                      cp=cp, g=g, rovcp=rovcp, rd=r_d, rovg=rovg,            &
                      ep1=ep1, ep2=ep2, karman=karman, xlv=xlv, rv=r_v,      &
                      dz8w2d=dz8w2d, psfcpa=psfcpa,                          &
                      znt=znt, ust=ust, hpbl=hpbl,                           &
                      psim=psim, psih=psih, xland=xland,                     &
                      hfx=hfx, qfx=qfx, wspd=wspd, br=br,                    &
                      dt=dtstep, kpbl1d=kpbl1d,                              &
                      exch_hx=exch_hx, exch_mx=exch_mx,                      &
                      wstar=wstar, delta=delta, u10=u10, v10=v10,            &
                      ctopo=ctopo, ctopo2=ctopo2,                            &
                      rthraten=rthraten, ysu_topdown_pblmix=topdown,         &
                      flag_bep=.false.,                                      &
                      its=1, ite=1, kte=nz, kme=nz + 1,                      &
                      errmsg=errmsg, errflg=errflg)
      if (errflg /= 0) then
        write(*, '(A,I0,A,A)') 'ctopo case ', icase, ' errmsg: ', trim(errmsg)
        error stop 4
      end if
      o_ux(:, col) = ux(1, :); o_vx(:, col) = vx(1, :); o_tx(:, col) = tx(1, :)
      o_qvx(:, col) = qvx(1, :); o_qcx(:, col) = qcx(1, :); o_qix(:, col) = qix(1, :)
      o_p2d(:, col) = p2d(1, :); o_pi2d(:, col) = pi2d(1, :)
      o_dz8w(:, col) = dz8w2d(1, :); o_rthraten(:, col) = rthraten(1, :)
      o_p2di(:, col) = p2di(1, :)
      o_ut(:, col) = utnp(1, :); o_vt(:, col) = vtnp(1, :); o_tt(:, col) = ttnp(1, :)
      o_qvt(:, col) = qvtnp(1, :); o_qct(:, col) = qctnp(1, :); o_qit(:, col) = qitnp(1, :)
      o_exh(:, col) = exch_hx(1, :); o_exm(:, col) = exch_mx(1, :)
      o_psfc(col) = psfcpa(1); o_znt(col) = znt(1); o_ust(col) = ust(1)
      o_hfx(col) = hfx(1); o_qfx(col) = qfx(1); o_wspd(col) = wspd(1)
      o_br(col) = br(1); o_psim(col) = psim(1); o_psih(col) = psih(1)
      o_xland(col) = xland(1)
      o_u10_in(col) = u10_in; o_v10_in(col) = v10_in
      o_u10(col) = u10(1); o_v10(col) = v10(1)
      o_ctopo(col) = ctopo(1); o_ctopo2(col) = ctopo2(1)
      o_hpbl(col) = hpbl(1); o_wstar(col) = wstar(1); o_delta(col) = delta(1)
      o_kpbl(col) = kpbl1d(1)
      o_topdown(col) = merge(1, 0, topdown)
      o_case(col) = icase
    end do
  end do

  call oracle_open('ysu_topo/columns')
  call oracle_put('dt', real(dtstep, 4))
  call oracle_put('ux', o_ux); call oracle_put('vx', o_vx); call oracle_put('tx', o_tx)
  call oracle_put('qvx', o_qvx); call oracle_put('qcx', o_qcx); call oracle_put('qix', o_qix)
  call oracle_put('p2d', o_p2d); call oracle_put('pi2d', o_pi2d)
  call oracle_put('p2di', o_p2di); call oracle_put('dz8w', o_dz8w)
  call oracle_put('rthraten', o_rthraten)
  call oracle_put('psfcpa', o_psfc); call oracle_put('znt', o_znt)
  call oracle_put('ust', o_ust); call oracle_put('hfx', o_hfx)
  call oracle_put('qfx', o_qfx); call oracle_put('wspd', o_wspd)
  call oracle_put('br', o_br); call oracle_put('psim', o_psim)
  call oracle_put('psih', o_psih); call oracle_put('xland', o_xland)
  call oracle_put('u10_in', o_u10_in); call oracle_put('v10_in', o_v10_in)
  call oracle_put('ctopo', o_ctopo); call oracle_put('ctopo2', o_ctopo2)
  call oracle_put('topdown', o_topdown); call oracle_put('case', o_case)
  call oracle_put('utnp', o_ut); call oracle_put('vtnp', o_vt)
  call oracle_put('ttnp', o_tt); call oracle_put('qvtnp', o_qvt)
  call oracle_put('qctnp', o_qct); call oracle_put('qitnp', o_qit)
  call oracle_put('exch_hx', o_exh); call oracle_put('exch_mx', o_exm)
  call oracle_put('utnp_noctopo', o_ut_noctopo)
  call oracle_put('vtnp_noctopo', o_vt_noctopo)
  call oracle_put('u10', o_u10); call oracle_put('v10', o_v10)
  call oracle_put('hpbl', o_hpbl); call oracle_put('wstar', o_wstar)
  call oracle_put('delta', o_delta); call oracle_put('kpbl', o_kpbl)
  call oracle_close()

contains

  subroutine zero_outputs()
    utnp = 0.0; vtnp = 0.0; ttnp = 0.0
    qvtnp = 0.0; qctnp = 0.0; qitnp = 0.0
    exch_hx = 0.0; exch_mx = 0.0; qmixtnp = 0.0
    hpbl = 0.0; kpbl1d = 0; wstar = 0.0; delta = 0.0
    errmsg = ''; errflg = -1
  end subroutine zero_outputs

  ! build_case: copied unchanged from tools/ysu_wrf461_oracle/run_bl_ysu.F90.
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

end program run_ysu_topo
