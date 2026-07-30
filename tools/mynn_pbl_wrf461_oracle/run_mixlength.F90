program run_mynn_mixlength_oracle
  use module_bl_mynn, only: mym_length
  implicit none

  integer, parameter :: ncase = 4, nz = 12, kts = 1, kte = nz
  character(len=32), parameter :: names(ncase) = [character(len=32) :: &
      'stable', 'convective', 'high_shear', 'edmf_active']
  character(len=1024) :: output_path
  integer :: c, k, unit
  real :: dz(nz), zw(nz+1), u(nz), v(nz), qke(nz), dtv(nz), theta(nz)
  real :: vt(nz), vq(nz), cldfra(nz), edmf_w(nz), edmf_a(nz)
  real :: el(nz), qkw(nz), xland, dx, rmo, flt, fltv, flq, zi, psig_bl

  call get_command_argument(1, output_path)
  if (len_trim(output_path) == 0) then
    write(*, '(A)') 'usage: run_mixlength OUTPUT.csv'
    error stop 2
  end if
  open(newunit=unit, file=trim(output_path), status='new', action='write')
  write(unit, '(A)') 'case,k,dz,zw,zw_next,u,v,qke,dtv,theta,vt,vq,' // &
      'cldfra,edmf_w,edmf_a,xland,dx,rmo,flt,fltv,flq,zi,psig_bl,el,qkw'

  do c = 1, ncase
    zw(1) = 0.0
    do k = 1, nz
      dz(k) = 80.0 + 5.0 * real(k - 1)
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
    end do
    xland = 1.0
    dx = 3000.0
    rmo = 0.008
    flt = 0.0
    fltv = -0.02
    flq = 0.0
    zi = 350.0
    psig_bl = 1.0
    select case (c)
    case (2)
      rmo = -0.006
      fltv = 0.10
      zi = 650.0
      psig_bl = 0.96
      do k = 1, nz
        theta(k) = 302.0 - 0.20 * min(real(k - 1), 5.0) &
            + 0.65 * max(real(k - 6), 0.0)
        dtv(k) = merge(-0.004, 0.008, k <= 6)
        qke(k) = max(3.5 - 0.22 * real(k - 1), 0.12)
      end do
    case (3)
      u = 22.0
      do k = 1, nz
        u(k) = u(k) + 1.8 * real(k - 1)
      end do
      rmo = 0.002
      fltv = 0.01
      zi = 500.0
      psig_bl = 0.88
    case (4)
      rmo = -0.003
      fltv = 0.07
      zi = 550.0
      psig_bl = 0.82
      do k = 1, nz
        edmf_a(k) = max(0.035 - 0.0025 * real(k - 1), 0.002)
        edmf_w(k) = max(2.4 - 0.12 * real(k - 1), 0.4)
        cldfra(k) = min(0.08 * real(k - 1), 0.65)
        dtv(k) = merge(-0.002, 0.007, k <= 5)
      end do
    end select

    call mym_length(kts, kte, xland, dz, dx, zw, rmo, flt, fltv, flq, &
        vt, vq, u, v, qke, dtv, el, zi, theta, qkw, psig_bl, cldfra, &
        1, edmf_w, edmf_a)
    do k = 1, nz
      write(unit, '(A,",",I0,23(",",ES24.16E3))') trim(names(c)), k, &
          dz(k), zw(k), zw(k+1), u(k), v(k), qke(k), dtv(k), theta(k), &
          vt(k), vq(k), cldfra(k), edmf_w(k), edmf_a(k), xland, dx, &
          rmo, flt, fltv, flq, zi, psig_bl, el(k), qkw(k)
    end do
  end do
  close(unit)
end program run_mynn_mixlength_oracle
