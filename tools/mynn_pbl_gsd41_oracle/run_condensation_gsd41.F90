! mym_condensation CASE(2) of GSD MYNN v4.1 (NOAA-EMC WRF 3.9
! module_bl_mynn.F) over eight 30-level columns: every input and the
! qc_bl, cldfra_bl, vt, vq and sgm outputs per level.
program run_mynn_condensation_gsd41_oracle
  use module_bl_mynn, only: mym_condensation
  implicit none

  integer, parameter :: ncase = 8, nz = 30, kts = 1, kte = nz
  character(len=32), parameter :: names(ncase) = [character(len=32) :: &
      'dry_mixed', 'stratus_inversion', 'humid_mixed', 'cold_cloud', &
      'night_fog', 'deep_moist', 'zero_hfx', 'high_pblh']
  character(len=1024) :: output_path
  integer :: c, k, unit
  real :: dz(nz), zw(nz+1), thl(nz), qw(nz), p(nz), exner(nz)
  real :: tsq(nz), qsq(nz), cov(nz), sh(nz), el(nz), th(nz)
  real :: qc_bl(nz), cldfra(nz), vt(nz), vq(nz), sgm(nz), rstoch(nz)
  real :: dx, pblh, hfx, rmo, rh_top, z

  call get_command_argument(1, output_path)
  if (len_trim(output_path) == 0) then
    write(*, '(A)') 'usage: run_condensation_gsd41 OUTPUT.csv'
    error stop 2
  end if
  open(newunit=unit, file=trim(output_path), status='new', action='write')
  write(unit, '(A)') 'case,k,dz,zw,zw_next,thl,qw,p,exner,th,el,sgm_in,' // &
      'dx,pblh,hfx,qc_bl,cldfra_bl,vt,vq,sgm'

  do c = 1, ncase
    zw(1) = 0.0
    do k = 1, nz
      dz(k) = 30.0 + 25.0 * real(k - 1)
      zw(k+1) = zw(k) + dz(k)
    end do
    dx = 3000.0
    pblh = 1200.0
    hfx = 150.0
    rmo = -0.01
    rh_top = 0.6
    do k = 1, nz
      z = 0.5 * (zw(k) + zw(k+1))
      p(k) = 95000.0 * exp(-z / 8000.0)
      exner(k) = (p(k) / 100000.0)**(287.0 / 1004.5)
      th(k) = 295.0 + merge(0.0005 * z, 0.0005 * pblh + 0.006 * (z - pblh), z < pblh)
      qw(k) = max(0.012 * exp(-z / 2500.0), 1.0e-5)
      el(k) = max(80.0 - 0.02 * z, 10.0)
      sh(k) = 0.5
      tsq(k) = 0.0
      qsq(k) = 0.0
      cov(k) = 0.0
      sgm(k) = 0.0
      vt(k) = 0.0
      vq(k) = 0.0
      rstoch(k) = 0.0
    end do
    select case (c)
    case (2)  ! moist mixed layer under a sharp inversion near 900 m
      pblh = 900.0
      hfx = 40.0
      do k = 1, nz
        z = 0.5 * (zw(k) + zw(k+1))
        th(k) = 288.0 + merge(0.0, 8.0 + 0.004 * (z - 900.0), z < 900.0)
        qw(k) = merge(0.0095, 0.004 * exp(-(z - 900.0) / 2000.0), z < 900.0)
      end do
    case (3)
      hfx = 250.0
      do k = 1, nz
        qw(k) = qw(k) * 1.35
      end do
    case (4)  ! cold column through the 253-273 K blend
      pblh = 600.0
      do k = 1, nz
        z = 0.5 * (zw(k) + zw(k+1))
        th(k) = 262.0 + 0.004 * z
        qw(k) = max(0.0028 * exp(-z / 3000.0), 1.0e-6)
      end do
    case (5)  ! night, near-saturated surface layer
      pblh = 150.0
      hfx = -30.0
      rmo = 0.05
      do k = 1, nz
        z = 0.5 * (zw(k) + zw(k+1))
        th(k) = 284.0 + 0.012 * min(z, 400.0) + 0.004 * max(z - 400.0, 0.0)
        qw(k) = merge(0.0094, 0.007 * exp(-z / 2500.0), z < 200.0)
        el(k) = max(15.0 - 0.01 * z, 2.0)
      end do
    case (6)
      pblh = 2200.0
      hfx = 320.0
      do k = 1, nz
        qw(k) = qw(k) * 1.5
        el(k) = 150.0
      end do
    case (7)
      hfx = 0.0
      do k = 1, nz
        qw(k) = qw(k) * 1.2
      end do
    case (8)
      pblh = 3500.0
      hfx = 400.0
      dx = 13000.0
    end select
    do k = 1, nz
      thl(k) = th(k)
      sgm(k) = 1.0e-4 * real(k)
    end do

    call mym_condensation(kts, kte, dx, dz, zw, thl, qw, p, exner, &
        tsq, qsq, cov, sh, el, 2, qc_bl, cldfra, pblh, hfx, &
        vt, vq, th, sgm, rmo, 0, rstoch)
    do k = 1, nz
      write(unit, '(A,",",I0,18(",",ES24.16E3))') trim(names(c)), k, &
          dz(k), zw(k), zw(k+1), thl(k), qw(k), p(k), exner(k), th(k), &
          el(k), 1.0e-4 * real(k), dx, pblh, hfx, qc_bl(k), cldfra(k), &
          vt(k), vq(k), sgm(k)
    end do
  end do
  close(unit)
end program run_mynn_condensation_gsd41_oracle
