program run_mynn_initialize_oracle
  ! Dump module_bl_mynn.F:1514-1674 (mym_initialize) from the unmodified
  ! pinned WRF v4.6.1 physics module.  bl_mynn_mixlength=1 and spp_pbl=0
  ! pin the admitted option identity.
  use module_bl_mynn, only: mym_initialize
  implicit none

  integer, parameter :: ncase = 5, nz = 16, kts = 1, kte = nz
  character(len=32), parameter :: names(ncase) = [character(len=32) :: &
      'stable_land', 'convective_land', 'restart_water', 'edmf_active', &
      'calm_weak_ust']
  character(len=1024) :: output_path
  integer :: c, k, unit, spp_pbl, init_flag
  real :: dz(nz), zw(nz+1), u(nz), v(nz), thl(nz), qw(nz)
  real :: theta(nz), thetav(nz), cldfra(nz), edmf_w(nz), edmf_a(nz)
  real :: sm(nz), sh(nz), el(nz), qke(nz), tsq(nz), qsq(nz), cov(nz)
  real :: rstoch(nz)
  real :: sm_in(nz), sh_in(nz), qke_in(nz)
  real :: xland, dx, rmo, ust, zi, psig_bl
  logical :: initialize_qke

  call get_command_argument(1, output_path)
  if (len_trim(output_path) == 0) then
    write(*, '(A)') 'usage: run_initialize OUTPUT.csv'
    error stop 2
  end if
  open(newunit=unit, file=trim(output_path), status='new', action='write')
  write(unit, '(A)') 'case,k,dz,zw,zw_next,u,v,thl,qw,theta,thetav,' // &
      'cldfra,edmf_w,edmf_a,sm_before,sh_before,qke_before,xland,dx,' // &
      'rmo,ust,zi,psig_bl,initialize_qke,el,qke,tsq,qsq,cov,sm,sh'

  spp_pbl = 0
  do c = 1, ncase
    zw(1) = 0.0
    do k = 1, nz
      dz(k) = 60.0 + 12.0 * real(k - 1)
      zw(k+1) = zw(k) + dz(k)
      u(k) = 3.0 + 0.85 * real(k - 1)
      v(k) = -1.5 + 0.30 * real(k - 1)
      theta(k) = 291.0 + 0.42 * real(k - 1)
      thl(k) = theta(k) - 0.05 * real(k - 1)
      qw(k) = max(0.0115 - 0.00055 * real(k - 1), 0.0004)
      thetav(k) = theta(k) * (1.0 + 0.61 * qw(k))
      cldfra(k) = 0.0
      edmf_w(k) = 0.0
      edmf_a(k) = 0.0
      rstoch(k) = 0.0
      ! Distinctive seeds: sm(kts)/sh(kts) are never written by
      ! mym_level2, so WRF hands them straight back to the caller.
      sm_in(k) = 0.31 + 0.011 * real(k - 1)
      sh_in(k) = 0.47 - 0.009 * real(k - 1)
      qke_in(k) = max(1.4 - 0.09 * real(k - 1), 0.05)
    end do
    xland = 1.0
    dx = 3000.0
    rmo = 0.011
    ust = 0.24
    zi = 320.0
    psig_bl = 1.0
    initialize_qke = .true.

    select case (c)
    case (2)
      rmo = -0.0085
      ust = 0.52
      zi = 900.0
      psig_bl = 0.94
      do k = 1, nz
        theta(k) = 303.0 - 0.22 * min(real(k - 1), 7.0) &
            + 0.70 * max(real(k - 8), 0.0)
        thl(k) = theta(k) - 0.04 * real(k - 1)
        qw(k) = max(0.0148 - 0.00070 * real(k - 1), 0.0006)
        thetav(k) = theta(k) * (1.0 + 0.61 * qw(k))
      end do
    case (3)
      xland = 2.0
      rmo = -0.0032
      ust = 0.36
      zi = 640.0
      psig_bl = 0.88
      initialize_qke = .false.
      do k = 1, nz
        qke_in(k) = max(2.7 - 0.16 * real(k - 1), 0.02)
        theta(k) = 295.0 - 0.10 * min(real(k - 1), 4.0) &
            + 0.55 * max(real(k - 5), 0.0)
        thl(k) = theta(k) - 0.03 * real(k - 1)
        thetav(k) = theta(k) * (1.0 + 0.61 * qw(k))
      end do
    case (4)
      rmo = -0.0051
      ust = 0.44
      zi = 780.0
      psig_bl = 0.79
      do k = 1, nz
        edmf_a(k) = max(0.040 - 0.0022 * real(k - 1), 0.001)
        edmf_w(k) = max(2.7 - 0.14 * real(k - 1), 0.30)
        cldfra(k) = min(0.06 * real(k - 1), 0.60)
        theta(k) = 299.0 - 0.16 * min(real(k - 1), 6.0) &
            + 0.62 * max(real(k - 7), 0.0)
        thl(k) = theta(k) - 0.045 * real(k - 1)
        thetav(k) = theta(k) * (1.0 + 0.61 * qw(k))
      end do
    case (5)
      ! ust below the 0.02 floor used by the second qke(kts) seed and
      ! below the 0.01 floor of the linear taper.
      ust = 0.006
      rmo = 0.031
      zi = 300.0
      psig_bl = 0.97
      do k = 1, nz
        u(k) = 0.4 + 0.05 * real(k - 1)
        v(k) = 0.1
      end do
    end select

    do k = 1, nz
      sm(k) = sm_in(k)
      sh(k) = sh_in(k)
      qke(k) = qke_in(k)
      el(k) = 0.0
      tsq(k) = 0.0
      qsq(k) = 0.0
      cov(k) = 0.0
    end do

    call mym_initialize(kts, kte, xland, dz, dx, zw, u, v, thl, qw, &
        zi, theta, thetav, sh, sm, ust, rmo, el, qke, tsq, qsq, cov, &
        psig_bl, cldfra, 1, edmf_w, edmf_a, initialize_qke, &
        spp_pbl, rstoch)

    init_flag = 0
    if (initialize_qke) init_flag = 1
    do k = 1, nz
      write(unit, '(A,",",I0,21(",",ES24.16E3),",",I0,7(",",ES24.16E3))') &
          trim(names(c)), k, &
          dz(k), zw(k), zw(k+1), u(k), v(k), thl(k), qw(k), theta(k), &
          thetav(k), cldfra(k), edmf_w(k), edmf_a(k), sm_in(k), sh_in(k), &
          qke_in(k), xland, dx, rmo, ust, zi, psig_bl, init_flag, &
          el(k), qke(k), tsq(k), qsq(k), cov(k), sm(k), sh(k)
    end do
  end do
  close(unit)
end program run_mynn_initialize_oracle
