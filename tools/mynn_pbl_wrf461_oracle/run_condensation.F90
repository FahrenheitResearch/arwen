program run_mynn_condensation_oracle
  use module_model_constants, only: cp, r_d, xlv, xlf
  use module_bl_mynn, only: mym_condensation, qsat_blend
  implicit none

  ! Cases 1-4 keep sigma far below the qsat_tk*qpct floor, so they pin the
  ! floor alone.  Cases 5-7 raise the moisture variance until SQRT(qsq)
  ! dominates, and each one holds dz in a different band of the coarse-grid
  ! weight wt = max(500 - max(dz-100,0), 0)/500 so the inflation ramp itself
  ! is observable: wt = 1 (no inflation), wt = 0.6 (partial), wt = 0 (full).
  integer, parameter :: ncase = 7, nz = 20, kts = 1, kte = nz
  character(len=32), parameter :: names(ncase) = [character(len=32) :: &
      'dry_land', 'humid_land', 'liquid_cloud', 'ice_anvil_water', &
      'high_variance_fine_grid', 'high_variance_transition_grid', &
      'high_variance_coarse_grid']
  character(len=1024) :: output_path
  integer :: c, k, unit
  real :: dz(nz), zw(nz+1), th(nz), thl(nz), qw(nz), qv(nz), qc(nz), qi(nz), qs(nz)
  real :: p(nz), exner(nz), tsq(nz), qsq(nz), cov(nz), sh(nz), el(nz)
  real :: qc_bl(nz), qi_bl(nz), cldfra(nz), vt(nz), vq(nz), sgm(nz)
  real :: vt_before(nz), vq_before(nz), sgm_before(nz), rstoch(nz)
  real :: zmid, temperature, saturation, relative_humidity
  real :: xland, dx, pblh, hfx, rmo
  integer :: spp_pbl

  call get_command_argument(1, output_path)
  if (len_trim(output_path) == 0) then
    write(*, '(A)') 'usage: run_condensation OUTPUT.csv'
    error stop 2
  end if
  open(newunit=unit, file=trim(output_path), status='replace', action='write')
  write(unit, '(A)') 'case,k,dz,zw,zw_next,xland,dx,pblh,hfx,rmo,' // &
      'spp_pbl,th,thl,qw,qv,qc,qi,qs,p,exner,tsq,qsq,cov,sh,el,' // &
      'vt_before,vq_before,sgm_before,rstoch,qc_bl,qi_bl,cldfra,' // &
      'vt_after,vq_after,sgm_after'

  do c = 1, ncase
    zw(1) = 0.0
    do k = 1, nz
      select case (c)
      case (5)
        dz(k) = 60.0
      case (6)
        dz(k) = 300.0
      case (7)
        dz(k) = 900.0
      case default
        dz(k) = 420.0 + 24.0 * real(k - 1)
      end select
      zw(k+1) = zw(k) + dz(k)
    end do
    xland = 1.0
    dx = 3000.0
    pblh = 850.0
    hfx = -18.0
    rmo = 0.008
    spp_pbl = 0
    select case (c)
    case (2)
      pblh = 1200.0
      hfx = 145.0
      rmo = -0.006
    case (3)
      pblh = 900.0
      hfx = 85.0
      rmo = -0.003
    case (4)
      xland = 2.0
      pblh = 700.0
      hfx = 30.0
      rmo = -0.001
    case (5)
      xland = 2.0
      dx = 750.0
      pblh = 600.0
      hfx = 60.0
      rmo = -0.004
    case (6)
      dx = 12000.0
      pblh = 1500.0
      hfx = 110.0
      rmo = -0.005
    case (7)
      dx = 36000.0
      pblh = 1800.0
      hfx = 95.0
      rmo = -0.002
    end select

    do k = 1, nz
      zmid = zw(k) + 0.5 * dz(k)
      p(k) = 100000.0 * exp(-zmid / 8000.0)
      exner(k) = (p(k) / 100000.0) ** (r_d / cp)
      if (zmid <= 11000.0) then
        temperature = 291.0 - 0.0061 * zmid
      else
        temperature = 223.9 + 0.0010 * (zmid - 11000.0)
      end if
      th(k) = temperature / exner(k)
      saturation = qsat_blend(temperature, p(k))
      select case (c)
      case (1)
        relative_humidity = max(0.28, 0.52 - 0.000015 * zmid)
      case (2)
        relative_humidity = merge(0.975, 0.78, zmid < 2600.0)
      case (3)
        relative_humidity = merge(0.94, 0.72, zmid < 4800.0)
      case (4)
        relative_humidity = merge(0.82, 0.70, zmid < 5500.0)
      case (5)
        relative_humidity = merge(0.90, 0.66, zmid < 700.0)
      case (6)
        relative_humidity = 0.86 - 0.000012 * zmid
      case (7)
        relative_humidity = merge(0.88, 0.60, zmid < 4000.0)
      end select
      qv(k) = relative_humidity * saturation
      qc(k) = 0.0
      qi(k) = 0.0
      qs(k) = 0.0
      if (c == 3 .and. zmid > 1500.0 .and. zmid < 4300.0) then
        qc(k) = max(2.2e-4 - abs(zmid - 2900.0) * 6.0e-8, 2.0e-6)
      end if
      if (c == 4 .and. zmid > 5600.0 .and. zmid < 10500.0) then
        qi(k) = max(5.0e-5 - abs(zmid - 8000.0) * 1.2e-8, 2.0e-6)
        qs(k) = 0.7 * qi(k)
      end if
      qw(k) = qv(k) + qc(k) + qi(k)
      thl(k) = th(k) - (xlv / cp) / exner(k) * qc(k) &
          - ((xlv + xlf) / cp) / exner(k) * qi(k)
      tsq(k) = 2.0e-3 * exp(-zmid / 3500.0)
      ! The moisture-variance amplitude decides which of the three sigma
      ! branches wins.  0.018 always loses to the qsat_tk*qpct floor; the
      ! decaying high-variance profiles start above the qsat_tk*0.666 clip
      ! and fall through the un-clipped square-root regime as they rise.
      select case (c)
      case (5)
        qsq(k) = (1.40 * exp(-0.19 * real(k - 1)) * saturation) ** 2
      case (6)
        qsq(k) = (1.55 * exp(-0.22 * real(k - 1)) * saturation) ** 2
      case (7)
        qsq(k) = (1.20 * exp(-0.17 * real(k - 1)) * saturation) ** 2
      case default
        qsq(k) = (0.018 * saturation) ** 2
      end select
      cov(k) = -0.15 * sqrt(max(tsq(k) * qsq(k), 0.0))
      sh(k) = 0.35
      el(k) = max(8.0, min(180.0, 0.08 * zmid + 8.0))
      vt(k) = 0.0
      vq(k) = 0.0
      sgm(k) = 0.0
      rstoch(k) = 0.0
    end do
    ! The WRF loop runs kts..kte-1 and the copy-down block never assigns
    ! sgm(kte), so a nonzero entry value has to survive the call verbatim.
    ! Cases 1-4 hand in 0.0 there, which cannot tell "left alone" from
    ! "zeroed"; these three can.
    select case (c)
    case (5)
      sgm(kte) = 7.5e-4
    case (6)
      sgm(kte) = 1.25e-3
    case (7)
      sgm(kte) = 3.0e-3
    end select
    vt_before = vt
    vq_before = vq
    sgm_before = sgm

    call mym_condensation(kts, kte, dx, dz, zw, xland, thl, qw, qv, &
        qc, qi, qs, p, exner, tsq, qsq, cov, sh, el, 2, qc_bl, qi_bl, &
        cldfra, pblh, hfx, vt, vq, th, sgm, rmo, spp_pbl, rstoch)

    do k = 1, nz
      write(unit, '(*(g0,:,","))') trim(names(c)), k, dz(k), zw(k), &
          zw(k+1), xland, dx, pblh, hfx, rmo, spp_pbl, th(k), thl(k), &
          qw(k), qv(k), qc(k), qi(k), qs(k), p(k), exner(k), tsq(k), qsq(k), &
          cov(k), sh(k), el(k), vt_before(k), vq_before(k), &
          sgm_before(k), rstoch(k), qc_bl(k), qi_bl(k), cldfra(k), &
          vt(k), vq(k), sgm(k)
    end do
  end do
  close(unit)
end program run_mynn_condensation_oracle
