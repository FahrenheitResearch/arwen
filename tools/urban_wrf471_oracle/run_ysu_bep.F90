program run_ysu_bep_oracle
  ! YSU with the BEP/BEP+BEM source terms (flag_bep = .true.), WRF v4.7.1
  ! phys/physics_mmm/bl_ysu.F90 at MMM-physics 20240626-MPASv8.2
  ! (arch/Externals.cfg), byte-unmodified.  That file differs from the
  ! v4.6.1 copy tools/ysu_wrf461_oracle compiled (sha256 a695daa7...) by ONE
  ! line, `we(i) = 0.` at :605, which gpuwm/core/kernels/ysu.cu already
  ! does (its `we` is a zero-initialised column local); every flag_bep
  ! expression (:450-497, :1045-1085, :1135-1170, :1305-1368) is identical.
  !
  ! The 24 columns are tools/ysu_wrf461_oracle/run_bl_ysu.F90's build_case,
  ! copied verbatim below, so the non-BEP inputs are the shipped fixture's.
  ! On top of each, a BEP forcing set shaped like the one
  ! module_sf_noahdrv.F:1679-1720 hands YSU: frc-weighted canopy drag and
  ! heat/moisture sources on the lowest levels, sf/vl below one inside the
  ! canopy, and the rural surface flux folded into level 1.
  !
  ! Two calls per column, both flag_bep = .true.:
  !   * ctopo = ctopo2 = 1 -- WRF's own driver path (module_bl_ysu.F:404),
  !     where :1313 removes the urban fraction of the surface drag;
  !   * ctopo absent       -- the arm kernels/ysu.cu transcribes, where WRF
  !     removes nothing (:1318).  Recorded so the difference between the two
  !     is a measured WRF-against-WRF number.
  use ccpp_kind_types, only: kind_phys
  use bl_ysu, only: bl_ysu_run
  use oracle_io
  implicit none

  integer, parameter :: nz = 40
  integer, parameter :: ncase = 24
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
  real(kind=kind_phys), dimension(1, nz) :: a_u, a_v, a_t, a_q, a_e
  real(kind=kind_phys), dimension(1, nz) :: b_u, b_v, b_t, b_q, b_e
  real(kind=kind_phys), dimension(1, nz) :: dlg, dlu, sfk, vlk
  real(kind=kind_phys), dimension(1) :: frcurb

  real(kind=kind_phys), dimension(ncase, nz) :: c_ux, c_vx, c_tx, c_qvx, c_qcx, c_qix
  real(kind=kind_phys), dimension(ncase, nz) :: c_p2d, c_pi2d, c_dz, c_rth
  real(kind=kind_phys), dimension(ncase, nz + 1) :: c_p2di
  real(kind=kind_phys), dimension(ncase, nz) :: c_au, c_av, c_at, c_aq, c_bu, c_bv, c_bt, c_bq
  real(kind=kind_phys), dimension(ncase, nz) :: c_sfk, c_vlk
  real(kind=kind_phys), dimension(ncase) :: c_frc, c_psfc, c_znt, c_ust, c_hfx, c_qfx
  real(kind=kind_phys), dimension(ncase) :: c_wspd, c_br, c_psim, c_psih, c_xland, c_u10, c_v10
  real(kind=kind_phys), dimension(ncase) :: c_topdown
  real(kind=kind_phys), dimension(ncase, nz, 2) :: o_ut, o_vt, o_tt, o_qvt, o_qct, o_qit, o_exh, o_exm
  real(kind=kind_phys), dimension(ncase, 2) :: o_hpbl, o_kpbl, o_wstar, o_delta

  character(len=1024) :: outdir
  character(len=256) :: errmsg
  integer :: errflg, icase, k, iarm
  logical :: topdown
  real(kind=kind_phys) :: zq(nz + 1), za(nz)
  real(kind=kind_phys) :: nzero, subn, minnorm, frc, rho1, spd1

  call oracle_root(outdir)
  nzero = sign(0.0_kind_phys, -1.0_kind_phys)
  subn = transfer(1, 0.0_kind_phys)
  minnorm = transfer(8388608, 0.0_kind_phys)

  do icase = 1, ncase
    call build_case(icase, topdown)
    zq(1) = 0.0
    do k = 1, nz
      zq(k + 1) = zq(k) + dz8w2d(1, k)
      za(k) = 0.5 * (zq(k) + zq(k + 1))
    end do
    call build_bep(icase)

    c_ux(icase, :) = ux(1, :); c_vx(icase, :) = vx(1, :); c_tx(icase, :) = tx(1, :)
    c_qvx(icase, :) = qvx(1, :); c_qcx(icase, :) = qcx(1, :); c_qix(icase, :) = qix(1, :)
    c_p2d(icase, :) = p2d(1, :); c_pi2d(icase, :) = pi2d(1, :); c_dz(icase, :) = dz8w2d(1, :)
    c_rth(icase, :) = rthraten(1, :); c_p2di(icase, :) = p2di(1, :)
    c_au(icase, :) = a_u(1, :); c_av(icase, :) = a_v(1, :); c_at(icase, :) = a_t(1, :)
    c_aq(icase, :) = a_q(1, :); c_bu(icase, :) = b_u(1, :); c_bv(icase, :) = b_v(1, :)
    c_bt(icase, :) = b_t(1, :); c_bq(icase, :) = b_q(1, :)
    c_sfk(icase, :) = sfk(1, :); c_vlk(icase, :) = vlk(1, :)
    c_frc(icase) = frcurb(1); c_psfc(icase) = psfcpa(1); c_znt(icase) = znt(1)
    c_ust(icase) = ust(1); c_hfx(icase) = hfx(1); c_qfx(icase) = qfx(1)
    c_wspd(icase) = wspd(1); c_br(icase) = br(1); c_psim(icase) = psim(1)
    c_psih(icase) = psih(1); c_xland(icase) = xland(1); c_u10(icase) = u10(1)
    c_v10(icase) = v10(1); c_topdown(icase) = merge(1.0, 0.0, topdown)

    do iarm = 1, 2
      qmix(1, :, 1) = qvx(1, :) * 0.5
      utnp = 0.0; vtnp = 0.0; ttnp = 0.0
      qvtnp = 0.0; qctnp = 0.0; qitnp = 0.0
      exch_hx = 0.0; exch_mx = 0.0; qmixtnp = 0.0
      hpbl = 0.0; kpbl1d = 0; wstar = 0.0; delta = 0.0
      errmsg = ''; errflg = -1
      ctopo(1) = 1.0; ctopo2(1) = 1.0
      if (iarm == 1) then
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
                        a_u=a_u, a_v=a_v, a_t=a_t, a_q=a_q, a_e=a_e,           &
                        b_u=b_u, b_v=b_v, b_t=b_t, b_q=b_q, b_e=b_e,           &
                        sfk=sfk, vlk=vlk, dlu=dlu, dlg=dlg, frcurb=frcurb,     &
                        flag_bep=.true.,                                       &
                        rthraten=rthraten, ysu_topdown_pblmix=topdown,         &
                        its=1, ite=1, kte=nz, kme=nz + 1,                      &
                        errmsg=errmsg, errflg=errflg)
      else
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
                        a_u=a_u, a_v=a_v, a_t=a_t, a_q=a_q, a_e=a_e,           &
                        b_u=b_u, b_v=b_v, b_t=b_t, b_q=b_q, b_e=b_e,           &
                        sfk=sfk, vlk=vlk, dlu=dlu, dlg=dlg, frcurb=frcurb,     &
                        flag_bep=.true.,                                       &
                        rthraten=rthraten, ysu_topdown_pblmix=topdown,         &
                        its=1, ite=1, kte=nz, kme=nz + 1,                      &
                        errmsg=errmsg, errflg=errflg)
      end if
      if (errflg /= 0) then
        write(*, '(A,I0,A,A)') 'case ', icase, ' errmsg: ', trim(errmsg)
        error stop 3
      end if
      o_ut(icase, :, iarm) = utnp(1, :); o_vt(icase, :, iarm) = vtnp(1, :)
      o_tt(icase, :, iarm) = ttnp(1, :); o_qvt(icase, :, iarm) = qvtnp(1, :)
      o_qct(icase, :, iarm) = qctnp(1, :); o_qit(icase, :, iarm) = qitnp(1, :)
      o_exh(icase, :, iarm) = exch_hx(1, :); o_exm(icase, :, iarm) = exch_mx(1, :)
      o_hpbl(icase, iarm) = hpbl(1); o_kpbl(icase, iarm) = real(kpbl1d(1))
      o_wstar(icase, iarm) = wstar(1); o_delta(icase, iarm) = delta(1)
    end do
  end do

  call oracle_open('columns')
  call oracle_put('meta_nz', real(nz)); call oracle_put('meta_ncase', real(ncase))
  call oracle_put('meta_dt', dtstep)
  call oracle_put('ux', c_ux); call oracle_put('vx', c_vx); call oracle_put('tx', c_tx)
  call oracle_put('qvx', c_qvx); call oracle_put('qcx', c_qcx); call oracle_put('qix', c_qix)
  call oracle_put('p2d', c_p2d); call oracle_put('pi2d', c_pi2d); call oracle_put('dz8w', c_dz)
  call oracle_put('rthraten', c_rth); call oracle_put('p2di', c_p2di)
  call oracle_put('a_u_bep', c_au); call oracle_put('a_v_bep', c_av); call oracle_put('a_t_bep', c_at)
  call oracle_put('a_q_bep', c_aq); call oracle_put('b_u_bep', c_bu); call oracle_put('b_v_bep', c_bv)
  call oracle_put('b_t_bep', c_bt); call oracle_put('b_q_bep', c_bq)
  call oracle_put('sf_bep', c_sfk); call oracle_put('vl_bep', c_vlk)
  call oracle_put('frc_urb2d', c_frc); call oracle_put('psfc', c_psfc); call oracle_put('znt', c_znt)
  call oracle_put('ust', c_ust); call oracle_put('hfx', c_hfx); call oracle_put('qfx', c_qfx)
  call oracle_put('wspd', c_wspd); call oracle_put('br', c_br); call oracle_put('psim', c_psim)
  call oracle_put('psih', c_psih); call oracle_put('xland', c_xland); call oracle_put('u10', c_u10)
  call oracle_put('v10', c_v10); call oracle_put('topdown', c_topdown)
  do iarm = 1, 2
    if (iarm == 1) then
      call put_arm('ctopo_')
    else
      call put_arm('noctopo_')
    end if
  end do
  call oracle_close()
  write(*, '(A)') 'ysu bep oracle written'

contains

  subroutine put_arm(prefix)
    character(len=*), intent(in) :: prefix
    call oracle_put(prefix//'utnp', o_ut(:, :, iarm)); call oracle_put(prefix//'vtnp', o_vt(:, :, iarm))
    call oracle_put(prefix//'ttnp', o_tt(:, :, iarm)); call oracle_put(prefix//'qvtnp', o_qvt(:, :, iarm))
    call oracle_put(prefix//'qctnp', o_qct(:, :, iarm)); call oracle_put(prefix//'qitnp', o_qit(:, :, iarm))
    call oracle_put(prefix//'exch_hx', o_exh(:, :, iarm)); call oracle_put(prefix//'exch_mx', o_exm(:, :, iarm))
    call oracle_put(prefix//'hpbl', o_hpbl(:, iarm)); call oracle_put(prefix//'kpbl', o_kpbl(:, iarm))
    call oracle_put(prefix//'wstar', o_wstar(:, iarm)); call oracle_put(prefix//'delta', o_delta(:, iarm))
  end subroutine put_arm

  subroutine build_bep(ic)
    ! BEP-shaped forcing; frc cycles through 0 (rural column: the BEP terms
    ! carry only the rural surface flux) up to 1 (all urban).
    integer, intent(in) :: ic
    integer :: kk, ncan
    real(kind=kind_phys) :: w
    select case (mod(ic, 6))
    case (0); frc = 0.0
    case (1); frc = 0.9
    case (2); frc = 0.5
    case (3); frc = 0.99
    case (4); frc = 0.25
    case default; frc = 1.0
    end select
    frcurb(1) = frc
    ncan = 2 + mod(ic, 3)             ! canopy spans 2..4 model levels
    a_u = 0.0; a_v = 0.0; a_t = 0.0; a_q = 0.0; a_e = 0.0
    b_u = 0.0; b_v = 0.0; b_t = 0.0; b_q = 0.0; b_e = 0.0
    dlg = 0.0; dlu = 0.0; sfk = 1.0; vlk = 1.0
    do kk = 1, ncan
      w = 1.0 / real(kk, kind_phys)
      a_u(1, kk) = -frc * 2.0e-3 * w * (1.0 + 0.1 * real(ic, kind_phys) / 24.0)
      a_v(1, kk) = -frc * 1.6e-3 * w
      a_t(1, kk) = -frc * 3.0e-4 * w
      b_u(1, kk) = frc * 1.0e-4 * w * ux(1, kk) / max(abs(ux(1, kk)), 1.0)
      b_v(1, kk) = -frc * 0.5e-4 * w
      b_t(1, kk) = frc * (8.0e-2 + 2.0e-3 * real(ic, kind_phys)) * w
      b_q(1, kk) = frc * 2.0e-7 * w
      b_e(1, kk) = frc * 1.0e-3 * w
      dlg(1, kk) = za(kk)
      dlu(1, kk) = 5.0 * w
      vlk(1, kk) = (1.0 - frc) + (0.55 + 0.1 * real(kk, kind_phys)) * frc
      sfk(1, kk) = (1.0 - frc) + (0.45 + 0.12 * real(kk, kind_phys)) * frc
    end do
    sfk(1, 1) = 1.0                   ! noahdrv.F:1720 sets sf_bep(i,1,j)=1
    ! Rural part folded into level 1 as noahdrv.F:1703-1714 does.
    rho1 = psfcpa(1) / (r_d * tx(1, 1))
    spd1 = max((ux(1, 1)**2 + vx(1, 1)**2)**0.5, 0.1)
    a_u(1, 1) = (1.0 - frc) * (-ust(1) * ust(1)) / dz8w2d(1, 1) / spd1 + a_u(1, 1)
    a_v(1, 1) = (1.0 - frc) * (-ust(1) * ust(1)) / dz8w2d(1, 1) / spd1 + a_v(1, 1)
    b_t(1, 1) = (1.0 - frc) * hfx(1) / dz8w2d(1, 1) / rho1 / cp + b_t(1, 1)
    b_q(1, 1) = (1.0 - frc) * qfx(1) / dz8w2d(1, 1) / rho1 + b_q(1, 1)
  end subroutine build_bep

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

end program run_ysu_bep_oracle
