program run_mynn_predict_oracle
  use module_bl_mynn, only: mym_turbulence, mym_predict
  implicit none

  integer, parameter :: ncase = 4, nz = 12, kts = 1, kte = nz
  character(len=32), parameter :: names(ncase) = [character(len=32) :: &
      'stable', 'convective', 'cloudy', 'edmf_active']
  character(len=1024) :: output_path
  integer :: c, k, unit
  real :: dz(nz), zw(nz+1), u(nz), v(nz), thl(nz), thetav(nz)
  real :: ql(nz), qw(nz), qke(nz), tsq(nz), qsq(nz), cov(nz)
  real :: qke_before(nz), tsq_before(nz), qsq_before(nz), cov_before(nz)
  real :: vt(nz), vq(nz), theta(nz), sh(nz), sm(nz), el(nz), rho(nz)
  real :: dfm(nz), dfh(nz), dfq(nz), tcd(nz), qcd(nz)
  real :: pdk(nz), pdt(nz), pdq(nz), pdc(nz)
  real :: pdk_in(nz), pdt_in(nz), pdq_in(nz), pdc_in(nz)
  real :: qwt(nz), qshear(nz), qbuoy(nz), qdiss(nz)
  real :: cldfra(nz), edmf_w(nz), edmf_a(nz), tkeprodtd(nz)
  real :: rstoch(nz), s_aw(nz+1), s_awqke(nz+1)
  real :: xland, closure, dx, rmo, flt, fltv, flq, zi
  real :: psig_bl, psig_shcu, ust, pmz, phh, delt

  call get_command_argument(1, output_path)
  if (len_trim(output_path) == 0) then
    write(*, '(A)') 'usage: run_predict OUTPUT.csv'
    error stop 2
  end if
  open(newunit=unit, file=trim(output_path), status='replace', action='write')
  write(unit, '(A)') 'case,k,dz,rho,dfq,pdk,pdt,pdq,pdc,el,s_aw,' // &
      's_aw_next,s_awqke,s_awqke_next,ust,flt,flq,pmz,phh,delt,' // &
      'qke_before,tsq_before,qsq_before,cov_before,qke_after,' // &
      'tsq_after,qsq_after,cov_after'

  do c = 1, ncase
    zw(1) = 0.0
    do k = 1, nz
      dz(k) = 70.0 + 6.0 * real(k - 1)
      zw(k+1) = zw(k) + dz(k)
      rho(k) = 1.18 * exp(-zw(k) / 8000.0)
      u(k) = 5.0 + 0.8 * real(k - 1)
      v(k) = 1.0 - 0.12 * real(k - 1)
      theta(k) = 290.0 + 0.55 * real(k - 1)
      thl(k) = theta(k)
      qw(k) = max(0.009 - 0.00055 * real(k - 1), 0.001)
      ql(k) = 0.0
      thetav(k) = theta(k) * (1.0 + 0.61 * qw(k))
      qke(k) = max(1.7 - 0.11 * real(k - 1), 0.06)
      tsq(k) = 1.0e-3 * exp(-real(k - 1) / 5.0)
      qsq(k) = 2.0e-7 * exp(-real(k - 1) / 5.0)
      cov(k) = -8.0e-6 * exp(-real(k - 1) / 5.0)
      vt(k) = 0.0
      vq(k) = 0.0
      cldfra(k) = 0.0
      edmf_w(k) = 0.0
      edmf_a(k) = 0.0
      tkeprodtd(k) = 0.0
      rstoch(k) = 0.0
      qwt(k) = 0.0
      qshear(k) = 0.0
      qbuoy(k) = 0.0
      qdiss(k) = 0.0
      s_aw(k) = 0.0
      s_awqke(k) = 0.0
    end do
    s_aw(nz+1) = 0.0
    s_awqke(nz+1) = 0.0
    xland = 1.0
    closure = 2.6
    dx = 3000.0
    rmo = 0.008
    flt = -0.015
    fltv = -0.02
    flq = -1.0e-5
    zi = 400.0
    psig_bl = 1.0
    psig_shcu = 1.0
    ust = 0.28
    pmz = 1.15
    phh = 0.90
    delt = 6.0
    select case (c)
    case (2)
      rmo = -0.006
      flt = 0.08
      fltv = 0.11
      flq = 6.0e-5
      zi = 700.0
      psig_bl = 0.94
      psig_shcu = 0.90
      ust = 0.45
      pmz = 0.85
      phh = 0.75
      do k = 1, nz
        theta(k) = 301.0 - 0.18 * min(real(k - 1), 5.0) &
            + 0.72 * max(real(k - 6), 0.0)
        thl(k) = theta(k)
        thetav(k) = theta(k) * (1.0 + 0.61 * qw(k))
        qke(k) = max(3.8 - 0.24 * real(k - 1), 0.12)
      end do
    case (3)
      rmo = -0.002
      flt = 0.045
      fltv = 0.06
      flq = 4.0e-5
      zi = 620.0
      psig_bl = 0.86
      psig_shcu = 0.80
      ust = 0.38
      pmz = 0.91
      phh = 0.80
      do k = 1, nz
        cldfra(k) = min(0.10 * real(k - 1), 0.75)
        ql(k) = max(0.00035 - 0.000025 * abs(real(k - 6)), 0.0)
        qw(k) = qw(k) + ql(k)
        thl(k) = theta(k) - 2500.0 / 1004.5 * ql(k)
        vt(k) = 0.10 * cldfra(k)
        vq(k) = 12.0 * cldfra(k)
      end do
    case (4)
      rmo = -0.004
      flt = 0.065
      fltv = 0.09
      flq = 5.0e-5
      zi = 680.0
      psig_bl = 0.78
      psig_shcu = 0.72
      ust = 0.50
      pmz = 0.82
      phh = 0.70
      do k = 1, nz
        edmf_a(k) = max(0.040 - 0.0028 * real(k - 1), 0.003)
        edmf_w(k) = max(2.8 - 0.14 * real(k - 1), 0.5)
        cldfra(k) = min(0.06 * real(k - 1), 0.55)
        tkeprodtd(k) = 0.00008 * exp(-real(k - 1) / 4.0)
        s_aw(k) = max(0.08 - 0.005 * real(k - 1), 0.0)
        s_awqke(k) = s_aw(k) * qke(k)
      end do
      s_aw(nz+1) = 0.0
      s_awqke(nz+1) = 0.0
    end select

    call mym_turbulence(kts, kte, xland, closure, dz, dx, zw, u, v, &
        thl, thetav, ql, qw, qke, tsq, qsq, cov, vt, vq, rmo, flt, &
        fltv, flq, zi, theta, sh, sm, el, dfm, dfh, dfq, tcd, qcd, &
        pdk, pdt, pdq, pdc, qwt, qshear, qbuoy, qdiss, 0, psig_bl, &
        psig_shcu, cldfra, 1, edmf_w, edmf_a, tkeprodtd, 0, rstoch)
    dfq(kts) = 0.0
    pdk(kts) = 0.0
    pdt(kts) = 0.0
    pdq(kts) = 0.0
    pdc(kts) = 0.0
    qke_before = qke
    tsq_before = tsq
    qsq_before = qsq
    cov_before = cov
    pdk_in = pdk
    pdt_in = pdt
    pdq_in = pdq
    pdc_in = pdc

    call mym_predict(kts, kte, closure, delt, dz, ust, flt, flq, pmz, &
        phh, el, dfq, rho, pdk, pdt, pdq, pdc, qke, tsq, qsq, cov, &
        s_aw, s_awqke, 0, qwt, qdiss, 0)

    do k = 1, nz
      write(unit, '(*(g0,:,","))') trim(names(c)), k, dz(k), rho(k), &
          dfq(k), pdk_in(k), pdt_in(k), pdq_in(k), pdc_in(k), el(k), &
          s_aw(k), s_aw(k+1), s_awqke(k), s_awqke(k+1), ust, flt, flq, &
          pmz, phh, delt, qke_before(k), tsq_before(k), qsq_before(k), &
          cov_before(k), qke(k), tsq(k), qsq(k), cov(k)
    end do
  end do
  close(unit)
end program run_mynn_predict_oracle
