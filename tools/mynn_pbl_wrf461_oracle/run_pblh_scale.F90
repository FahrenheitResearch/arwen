program run_mynn_pblh_scale_oracle
  use module_bl_mynn, only: get_pblh, scale_aware
  implicit none

  integer, parameter :: ncase = 4, nz = 10, kts = 1, kte = nz
  character(len=32), parameter :: names(ncase) = [character(len=32) :: &
      'convective_land', 'stable_land', 'marine', 'cold_pool']
  character(len=1024) :: output_path
  integer :: c, k, kzi, unit
  real :: dz(nz), zw(nz+1), thetav(nz), qke(nz)
  real :: landsea, dx, zi, psig_bl, psig_shcu

  call get_command_argument(1, output_path)
  if (len_trim(output_path) == 0) then
    write(*, '(A)') 'usage: run_pblh_scale OUTPUT.csv'
    error stop 2
  end if

  open(newunit=unit, file=trim(output_path), status='new', action='write')
  write(unit, '(A)') 'case,k,dz,zw,zw_next,thetav,qke,landsea,dx,zi,' // &
      'kzi,psig_bl,psig_shcu'
  do c = 1, ncase
    zw(1) = 0.0
    do k = 1, nz
      dz(k) = 40.0 + 5.0 * real(k - 1)
      zw(k+1) = zw(k) + dz(k)
      select case (c)
      case (1)
        thetav(k) = 300.0 + 0.12 * real(k - 1)
        if (k >= 7) thetav(k) = thetav(k) + 1.6 * real(k - 6)
        qke(k) = max(3.0 - 0.38 * real(k - 1), 0.01)
        landsea = 1.0
        dx = 3000.0
      case (2)
        thetav(k) = 290.0 + 0.80 * real(k - 1)
        qke(k) = max(0.80 - 0.15 * real(k - 1), 0.01)
        landsea = 1.0
        dx = 1000.0
      case (3)
        thetav(k) = 295.0 + 0.40 * real(k - 1)
        qke(k) = max(1.80 - 0.25 * real(k - 1), 0.01)
        landsea = 2.0
        dx = 500.0
      case (4)
        thetav(k) = 299.0 - 0.20 * min(real(k - 1), 3.0) &
            + 0.65 * max(real(k - 4), 0.0)
        qke(k) = max(0.04 - 0.004 * real(k - 1), 0.001)
        landsea = 1.0
        dx = 250.0
      end select
    end do
    call get_pblh(kts, kte, zi, thetav, qke, zw, dz, landsea, kzi)
    call scale_aware(dx, zi, psig_bl, psig_shcu)
    do k = 1, nz
      write(unit, '(A,",",I0,8(",",ES24.16E3),",",I0,2(",",ES24.16E3))') &
          trim(names(c)), k, dz(k), zw(k), zw(k+1), thetav(k), qke(k), &
          landsea, dx, zi, kzi, psig_bl, psig_shcu
    end do
  end do
  close(unit)
end program run_mynn_pblh_scale_oracle
