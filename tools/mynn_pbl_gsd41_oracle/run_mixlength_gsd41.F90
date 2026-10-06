! mym_length CASE(2) of GSD MYNN v4.1 (NOAA-EMC WRF 3.9 module_bl_mynn.F)
! over twelve columns; every input and the el and qkw outputs per level.
program run_mynn_mixlength_gsd41_oracle
  use module_bl_mynn, only: mym_length
  implicit none

  integer, parameter :: ncase = 12, nz = 14, kts = 1, kte = nz
  character(len=32), parameter :: names(ncase) = [character(len=32) :: &
      'stable', 'convective', 'high_shear', 'edmf_active', &
      'weak_buoyancy', 'neutral_les', 'deep_pbl', 'high_tke', &
      'moist_flux', 'night_inversion', 'shallow_pbl', 'cloud_vt']
  character(len=1024) :: output_path
  integer :: c, k, unit
  real :: dz(nz), zw(nz+1), u(nz), v(nz), qke(nz), dtv(nz), theta(nz)
  real :: vt(nz), vq(nz), cldfra(nz), edmf_w(nz), edmf_a(nz), edmf_qc(nz)
  real :: el(nz), qkw(nz), rmo, flt, flq, zi, psig_bl

  call get_command_argument(1, output_path)
  if (len_trim(output_path) == 0) then
    write(*, '(A)') 'usage: run_mixlength_gsd41 OUTPUT.csv'
    error stop 2
  end if
  open(newunit=unit, file=trim(output_path), status='new', action='write')
  write(unit, '(A)') 'case,k,dz,zw,zw_next,u,v,qke,dtv,theta,vt,vq,' // &
      'cldfra,edmf_w,edmf_a,rmo,flt,flq,zi,psig_bl,el,qkw'

  do c = 1, ncase
    zw(1) = 0.0
    do k = 1, nz
      dz(k) = 40.0 + 12.0 * real(k - 1)
      zw(k+1) = zw(k) + dz(k)
      u(k) = 4.0 + 0.7 * real(k - 1)
      v(k) = 1.0 - 0.15 * real(k - 1)
      theta(k) = 290.0 + 0.45 * real(k - 1)
      qke(k) = max(1.8 - 0.12 * real(k - 1), 0.08)
      dtv(k) = 0.006
      vt(k) = 0.0
      vq(k) = 0.0
      cldfra(k) = 0.0
      edmf_w(k) = 0.0
      edmf_a(k) = 0.0
      edmf_qc(k) = 0.0
    end do
    rmo = 0.008
    flt = -0.02
    flq = 0.0
    zi = 350.0
    psig_bl = 1.0
    select case (c)
    case (2)
      rmo = -0.006
      flt = 0.10
      flq = 2.0e-5
      zi = 650.0
      psig_bl = 0.96
      do k = 1, nz
        dtv(k) = merge(-0.004, 0.008, k <= 6)
        qke(k) = max(3.5 - 0.22 * real(k - 1), 0.12)
      end do
    case (3)
      do k = 1, nz
        u(k) = 22.0 + 1.8 * real(k - 1)
      end do
      rmo = 0.002
      flt = 0.01
      zi = 500.0
      psig_bl = 0.88
    case (4)
      rmo = -0.003
      flt = 0.07
      flq = 4.0e-5
      zi = 550.0
      psig_bl = 0.82
      do k = 1, nz
        edmf_a(k) = max(0.035 - 0.0025 * real(k - 1), 0.002)
        edmf_w(k) = max(2.4 - 0.12 * real(k - 1), 0.4)
        cldfra(k) = min(0.08 * real(k - 1), 0.65)
        dtv(k) = merge(-0.002, 0.007, k <= 5)
      end do
    case (5)
      dtv = 1.0e-7
      qke = 1.0e-5
      zi = 60.0
      flt = -0.01
      psig_bl = 0.35
    case (6)
      dtv = 0.0
      rmo = 0.0
      flt = 0.0
      psig_bl = 0.0
    case (7)
      dz = dz*4.0
      do k = 1, nz
        zw(k+1) = zw(k) + dz(k)
      end do
      zi = 2500.0
      rmo = -0.002
      flt = 0.25
      flq = 1.0e-4
      dtv = -0.002
      psig_bl = 0.65
    case (8)
      qke = 120.0
      flt = 3.0
      zi = 600.0
      do k = 1, nz
        dtv(k) = merge(0.004, -0.004, mod(k,2) == 0)
      end do
    case (9)
      rmo = -0.01
      flt = 0.02
      flq = 1.5e-4
      vt(1) = 0.3
      vq(1) = 12.0
      zi = 900.0
      psig_bl = 0.9
      do k = 1, nz
        dtv(k) = merge(-0.001, 0.003, k <= 8)
      end do
    case (10)
      rmo = 0.05
      flt = -0.03
      flq = -1.0e-5
      zi = 120.0
      do k = 1, nz
        dtv(k) = 0.02 - 0.001 * real(k - 1)
        qke(k) = max(0.3 - 0.04 * real(k - 1), 1.0e-4)
      end do
    case (11)
      rmo = -0.02
      flt = 0.05
      zi = 180.0
      psig_bl = 0.99
      do k = 1, nz
        dtv(k) = merge(-0.003, 0.01, k <= 3)
      end do
    case (12)
      rmo = -0.004
      flt = 0.04
      flq = 6.0e-5
      vt(1) = -0.8
      vq(1) = 40.0
      zi = 1200.0
      psig_bl = 0.93
      do k = 1, nz
        edmf_a(k) = merge(0.05, 0.0, k <= 9)
        edmf_w(k) = merge(1.2, 0.0, k <= 9)
        dtv(k) = merge(-0.0005, 0.004, k <= 9)
        qke(k) = max(2.5 - 0.15 * real(k - 1), 0.05)
      end do
    end select

    call mym_length(kts, kte, dz, zw, rmo, flt, flq, vt, vq, qke, dtv, &
        el, zi, theta, qkw, psig_bl, cldfra, 2, edmf_w, edmf_a, edmf_qc, 1)
    do k = 1, nz
      write(unit, '(A,",",I0,20(",",ES24.16E3))') trim(names(c)), k, &
          dz(k), zw(k), zw(k+1), u(k), v(k), qke(k), dtv(k), theta(k), &
          vt(k), vq(k), cldfra(k), edmf_w(k), edmf_a(k), rmo, flt, flq, &
          zi, psig_bl, el(k), qkw(k)
    end do
  end do
  close(unit)
end program run_mynn_mixlength_gsd41_oracle
