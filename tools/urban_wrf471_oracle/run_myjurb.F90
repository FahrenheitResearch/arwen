program run_myjurb_oracle
  ! MYJURB, WRF v4.7.1 phys/module_bl_myjurb.F byte-unmodified: the MYJ PBL
  ! with the BEP/BEP+BEM source terms, which module_pbl_driver.F:1462-1492
  ! calls instead of MYJPBL whenever sf_urban_physics is 2 or 3 (idiff is
  ! hard-wired to 0 at :997, flag_bep is true).  DT is the driver's DTBL
  ! (= dt with bldt = 0, :975) and STEPBL = 1.
  !
  ! 18 columns x NSTEP consecutive calls, TKE/THZ0/QZ0/UZ0/VZ0/QSFC carried
  ! between calls as WRF carries them.  Column regimes: unstable land,
  ! stable land, sea (PQ0SEA/EXP arm of QSFC), snow (RLIVWV), cloudy (CWM),
  ! terrain height 0 and 350 m (ZINT starts at HT), urban fraction 0 (the
  ! rural column still carries the rural surface flux in the BEP terms),
  ! 0.5, 0.9 and 1, and TKE-source-bearing BEP canopies of 2-4 levels.
  use module_model_constants, only: r_d, cp, g, xlv, ep_1
  use module_bl_myjurb, only: myjurb, epsq2
  use oracle_io
  implicit none

  integer, parameter :: nz = 35, ncol = 18, nstep = 2
  real, parameter :: dtstep = 30.
  integer :: icol, k, istep, lowlyr(ncol, 1), kpbl(ncol, 1)
  real, dimension(ncol, nz + 1, 1) :: dz, pmid, pint, th, t, exner, qv, cwm
  real, dimension(ncol, nz + 1, 1) :: u, v, rho, tke, exch_h, exch_m, el
  real, dimension(ncol, nz + 1, 1) :: rublten, rvblten, rthblten, rqvblten, rqcblten
  real, dimension(ncol, nz + 1, 1) :: a_u, a_v, a_t, a_q, a_e, b_u, b_v, b_t
  real, dimension(ncol, nz + 1, 1) :: b_q, b_e, dlg, dl_u, sf, vl
  real, dimension(ncol, 1) :: ht, tsk, qsfc, chklowq, thz0, qz0, uz0, vz0
  real, dimension(ncol, 1) :: xland, sice, snow, ustar, znt, pblh, ct
  real, dimension(ncol, 1) :: akhs, akms, elflx, frc
  character(len=1024) :: outdir
  character(len=64) :: tag

  call oracle_root(outdir)
  call build()
  call oracle_open('columns')
  call oracle_put('meta_nz', real(nz)); call oracle_put('meta_ncol', real(ncol))
  call oracle_put('meta_nstep', real(nstep)); call oracle_put('meta_dt', dtstep)
  call oracle_put('dz', dz(:, :, 1)); call oracle_put('pmid', pmid(:, :, 1))
  call oracle_put('pint', pint(:, :, 1)); call oracle_put('th', th(:, :, 1))
  call oracle_put('t', t(:, :, 1)); call oracle_put('exner', exner(:, :, 1))
  call oracle_put('qv', qv(:, :, 1)); call oracle_put('cwm', cwm(:, :, 1))
  call oracle_put('u', u(:, :, 1)); call oracle_put('v', v(:, :, 1))
  call oracle_put('rho', rho(:, :, 1))
  call oracle_put('a_u_bep', a_u(:, :, 1)); call oracle_put('a_v_bep', a_v(:, :, 1))
  call oracle_put('a_t_bep', a_t(:, :, 1)); call oracle_put('a_q_bep', a_q(:, :, 1))
  call oracle_put('a_e_bep', a_e(:, :, 1)); call oracle_put('b_u_bep', b_u(:, :, 1))
  call oracle_put('b_v_bep', b_v(:, :, 1)); call oracle_put('b_t_bep', b_t(:, :, 1))
  call oracle_put('b_q_bep', b_q(:, :, 1)); call oracle_put('b_e_bep', b_e(:, :, 1))
  call oracle_put('dlg_bep', dlg(:, :, 1)); call oracle_put('dl_u_bep', dl_u(:, :, 1))
  call oracle_put('sf_bep', sf(:, :, 1)); call oracle_put('vl_bep', vl(:, :, 1))
  call oracle_put('frc_urb2d', frc(:, 1)); call oracle_put('ht', ht(:, 1))
  call oracle_put('tsk', tsk(:, 1)); call oracle_put('chklowq', chklowq(:, 1))
  call oracle_put('xland', xland(:, 1)); call oracle_put('sice', sice(:, 1))
  call oracle_put('snow', snow(:, 1)); call oracle_put('ustar', ustar(:, 1))
  call oracle_put('znt', znt(:, 1)); call oracle_put('akhs', akhs(:, 1))
  call oracle_put('akms', akms(:, 1)); call oracle_put('elflx', elflx(:, 1))
  call oracle_put('ct', ct(:, 1))

  do istep = 1, nstep
    write(tag, '(A,I0,A)') 's', istep, '_in_'
    call oracle_put(trim(tag)//'tke', tke(:, :, 1))
    call oracle_put(trim(tag)//'qsfc', qsfc(:, 1)); call oracle_put(trim(tag)//'thz0', thz0(:, 1))
    call oracle_put(trim(tag)//'qz0', qz0(:, 1)); call oracle_put(trim(tag)//'uz0', uz0(:, 1))
    call oracle_put(trim(tag)//'vz0', vz0(:, 1))
    call oracle_put(trim(tag)//'exch_h', exch_h(:, :, 1))
    call oracle_put(trim(tag)//'exch_m', exch_m(:, :, 1))
    rublten = -999.; rvblten = -999.; rthblten = -999.; rqvblten = -999.
    rqcblten = -999.; pblh = -999.; kpbl = -999
    call myjurb(IDIFF=0, FLAG_BEP=.true., DT=dtstep, STEPBL=1, HT=ht, DZ=dz,  &
                PMID=pmid, PINT=pint, TH=th, T=t, EXNER=exner, QV=qv, CWM=cwm, &
                U=u, V=v, RHO=rho, TSK=tsk, QSFC=qsfc, CHKLOWQ=chklowq,       &
                THZ0=thz0, QZ0=qz0, UZ0=uz0, VZ0=vz0, LOWLYR=lowlyr,          &
                XLAND=xland, SICE=sice, SNOW=snow, TKE_MYJ=tke,               &
                EXCH_H=exch_h, EXCH_M=exch_m, USTAR=ustar, ZNT=znt,           &
                EL_MYJ=el, PBLH=pblh, KPBL=kpbl, CT=ct, AKHS=akhs, AKMS=akms, &
                ELFLX=elflx, RUBLTEN=rublten, RVBLTEN=rvblten,                &
                RTHBLTEN=rthblten, RQVBLTEN=rqvblten, RQCBLTEN=rqcblten,      &
                FRC_URB2D=frc, A_U_BEP=a_u, A_V_BEP=a_v, A_T_BEP=a_t,         &
                A_Q_BEP=a_q, A_E_BEP=a_e, B_U_BEP=b_u, B_V_BEP=b_v,           &
                B_T_BEP=b_t, B_Q_BEP=b_q, B_E_BEP=b_e, DLG_BEP=dlg,           &
                DL_U_BEP=dl_u, SF_BEP=sf, VL_BEP=vl,                          &
                IDS=1, IDE=ncol + 1, JDS=1, JDE=2, KDS=1, KDE=nz + 1,         &
                IMS=1, IME=ncol, JMS=1, JME=1, KMS=1, KME=nz + 1,             &
                ITS=1, ITE=ncol, JTS=1, JTE=1, KTS=1, KTE=nz)
    write(tag, '(A,I0,A)') 's', istep, '_out_'
    call oracle_put(trim(tag)//'tke', tke(:, :, 1)); call oracle_put(trim(tag)//'el', el(:, :, 1))
    call oracle_put(trim(tag)//'exch_h', exch_h(:, :, 1))
    call oracle_put(trim(tag)//'exch_m', exch_m(:, :, 1))
    call oracle_put(trim(tag)//'rublten', rublten(:, :, 1))
    call oracle_put(trim(tag)//'rvblten', rvblten(:, :, 1))
    call oracle_put(trim(tag)//'rthblten', rthblten(:, :, 1))
    call oracle_put(trim(tag)//'rqvblten', rqvblten(:, :, 1))
    call oracle_put(trim(tag)//'rqcblten', rqcblten(:, :, 1))
    call oracle_put(trim(tag)//'pblh', pblh(:, 1))
    call oracle_put(trim(tag)//'kpbl', kpbl(:, 1))
    call oracle_put(trim(tag)//'qsfc', qsfc(:, 1)); call oracle_put(trim(tag)//'thz0', thz0(:, 1))
    call oracle_put(trim(tag)//'qz0', qz0(:, 1)); call oracle_put(trim(tag)//'uz0', uz0(:, 1))
    call oracle_put(trim(tag)//'vz0', vz0(:, 1)); call oracle_put(trim(tag)//'ct', ct(:, 1))
  end do
  call oracle_close()
  write(*, '(A)') 'myjurb oracle written'

contains

  subroutine build()
    integer :: ncan, ir
    real :: zc, zbot, th0, lapse, ws, wd, f, w, p, rcp_, rd_, cp_
    rcp_ = r_d / cp
    lowlyr = 1
    do icol = 1, ncol
      ir = mod(icol - 1, 6) + 1
      select case (ir)
      case (1); th0 = 298.; lapse = 0.001; ws = 6.;  wd = 0.4
      case (2); th0 = 290.; lapse = 0.012; ws = 2.;  wd = 2.2
      case (3); th0 = 295.; lapse = 0.004; ws = 11.; wd = 4.0
      case (4); th0 = 300.; lapse = 0.0005; ws = 4.; wd = 5.1
      case (5); th0 = 286.; lapse = 0.006; ws = 8.;  wd = 1.3
      case default; th0 = 293.; lapse = 0.003; ws = 3.; wd = 3.0
      end select
      ht(icol, 1) = merge(350., 0., mod(icol, 4) == 3)
      zbot = 0.
      pint(icol, 1, 1) = 99500. - 30. * ht(icol, 1) / 3.
      do k = 1, nz
        if (k <= 5) then
          dz(icol, k, 1) = 10. + 4. * real(k - 1)
        else
          dz(icol, k, 1) = 26. * 1.13**(k - 5)
        end if
        zc = zbot + 0.5 * dz(icol, k, 1)
        th(icol, k, 1) = th0 + lapse * zc + 0.2 * sin(0.5 * real(k + icol))
        qv(icol, k, 1) = 0.011 * exp(-zc / 2600.) * (1. + 0.05 * cos(real(icol)))
        cwm(icol, k, 1) = 0.
        if (ir == 5 .and. k >= 9 .and. k <= 12) cwm(icol, k, 1) = 2.5e-4
        u(icol, k, 1) = ws * log(1. + zc / 1.5) / log(1. + 500. / 1.5) * cos(wd + 0.001 * zc)
        v(icol, k, 1) = ws * log(1. + zc / 1.5) / log(1. + 500. / 1.5) * sin(wd + 0.001 * zc)
        pint(icol, k + 1, 1) = pint(icol, k, 1) * exp(-dz(icol, k, 1) / 8400.)
        pmid(icol, k, 1) = 0.5 * (pint(icol, k, 1) + pint(icol, k + 1, 1))
        exner(icol, k, 1) = (pmid(icol, k, 1) / 1.e5)**rcp_
        t(icol, k, 1) = th(icol, k, 1) * exner(icol, k, 1)
        rho(icol, k, 1) = pmid(icol, k, 1) / (r_d * t(icol, k, 1) * (1. + ep_1 * qv(icol, k, 1)))
        tke(icol, k, 1) = max(epsq2 * 0.5, 0.6 * exp(-zc / 600.) * (1. + 0.3 * real(ir - 3)))
        zbot = zbot + dz(icol, k, 1)
      end do
      dz(icol, nz + 1, 1) = 0.; pmid(icol, nz + 1, 1) = 0.; th(icol, nz + 1, 1) = 0.
      t(icol, nz + 1, 1) = 0.; exner(icol, nz + 1, 1) = 0.; qv(icol, nz + 1, 1) = 0.
      cwm(icol, nz + 1, 1) = 0.; u(icol, nz + 1, 1) = 0.; v(icol, nz + 1, 1) = 0.
      rho(icol, nz + 1, 1) = 0.; tke(icol, nz + 1, 1) = 0.
      exch_h(icol, :, 1) = 0.; exch_m(icol, :, 1) = 0.; el(icol, :, 1) = 0.

      xland(icol, 1) = merge(2., 1., ir == 3)
      sice(icol, 1) = 0.
      snow(icol, 1) = merge(0.02, 0., ir == 2)
      tsk(icol, 1) = th(icol, 1, 1) * exner(icol, 1, 1) + merge(4., -3., ir /= 2)
      ustar(icol, 1) = 0.1 + 0.04 * ws
      znt(icol, 1) = merge(2.e-4, 0.8, ir == 3)
      akhs(icol, 1) = 0.004 + 0.002 * real(ir)
      akms(icol, 1) = 0.006 + 0.002 * real(ir)
      elflx(icol, 1) = 40. + 20. * real(ir)
      chklowq(icol, 1) = 1.
      ct(icol, 1) = 0.
      thz0(icol, 1) = tsk(icol, 1) * (1.e5 / pint(icol, 1, 1))**rcp_
      qz0(icol, 1) = qv(icol, 1, 1) * 1.1
      qsfc(icol, 1) = qz0(icol, 1)
      uz0(icol, 1) = 0.; vz0(icol, 1) = 0.

      ! BEP terms after the couple: frc-weighted, a_q = a_e = 0, rural
      ! surface flux folded into level 1 (noahdrv.F:1679-1720).
      select case (mod(icol - 1, 5))
      case (0); f = 0.
      case (1); f = 0.5
      case (2); f = 0.9
      case (3); f = 1.
      case default; f = 0.25
      end select
      frc(icol, 1) = f
      a_u(icol, :, 1) = 0.; a_v(icol, :, 1) = 0.; a_t(icol, :, 1) = 0.
      a_q(icol, :, 1) = 0.; a_e(icol, :, 1) = 0.; b_u(icol, :, 1) = 0.
      b_v(icol, :, 1) = 0.; b_t(icol, :, 1) = 0.; b_q(icol, :, 1) = 0.
      b_e(icol, :, 1) = 0.; dlg(icol, :, 1) = 0.; dl_u(icol, :, 1) = 0.
      sf(icol, :, 1) = 1.; vl(icol, :, 1) = 1.
      ncan = 2 + mod(icol, 3)
      do k = 1, ncan
        w = 1. / real(k)
        a_u(icol, k, 1) = -f * 2.2e-3 * w
        a_v(icol, k, 1) = -f * 1.9e-3 * w
        a_t(icol, k, 1) = -f * 2.5e-4 * w
        b_u(icol, k, 1) = f * 1.2e-4 * w
        b_v(icol, k, 1) = -f * 0.7e-4 * w
        b_t(icol, k, 1) = f * 0.09 * w
        b_q(icol, k, 1) = f * 1.5e-7 * w
        b_e(icol, k, 1) = f * 2.0e-3 * w
        dl_u(icol, k, 1) = f * 6. * w
        dlg(icol, k, 1) = 5. * real(k)
        vl(icol, k, 1) = (1. - f) + (0.55 + 0.1 * real(k)) * f
        sf(icol, k, 1) = (1. - f) + (0.45 + 0.12 * real(k)) * f
      end do
      sf(icol, 1, 1) = 1.
      a_u(icol, 1, 1) = (1. - f) * (-ustar(icol, 1)**2) / dz(icol, 1, 1) / &
                        max(sqrt(u(icol, 1, 1)**2 + v(icol, 1, 1)**2), 0.1) + a_u(icol, 1, 1)
      a_v(icol, 1, 1) = a_u(icol, 1, 1) * 0.97
      b_t(icol, 1, 1) = (1. - f) * (60. + 10. * real(ir)) / dz(icol, 1, 1) / rho(icol, 1, 1) / cp &
                        + b_t(icol, 1, 1)
      b_q(icol, 1, 1) = (1. - f) * 4.e-5 / dz(icol, 1, 1) / rho(icol, 1, 1) + b_q(icol, 1, 1)
    end do
  end subroutine build

end program run_myjurb_oracle
