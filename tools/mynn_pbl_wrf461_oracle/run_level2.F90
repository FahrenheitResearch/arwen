program run_mynn_pbl_level2_oracle
  use module_bl_mynn, only: mym_level2
  implicit none

  integer, parameter :: ncase = 4, nz = 8, kts = 1, kte = nz
  character(len=32), parameter :: names(ncase) = [character(len=32) :: &
      'stable_dry', 'convective_dry', 'neutral_shear', 'moist_cloud']
  character(len=1024) :: output_path
  integer :: c, k, unit
  real :: dz(nz), u(nz), v(nz), thl(nz), thetav(nz), qw(nz)
  real :: ql(nz), vt(nz), vq(nz)
  real :: dtl(nz), dqw(nz), dtv(nz), gm(nz), gh(nz), sm(nz), sh(nz)

  call get_command_argument(1, output_path)
  if (len_trim(output_path) == 0) then
    write(*, '(A)') 'usage: run_level2 OUTPUT.csv'
    error stop 2
  end if

  open(newunit=unit, file=trim(output_path), status='new', action='write')
  write(unit, '(A)') 'case,k,dz,u,v,thl,thetav,qw,ql,vt,vq,' // &
      'dz_prev,u_prev,v_prev,thl_prev,thetav_prev,qw_prev,ql_prev,' // &
      'vt_prev,vq_prev,dtl,dqw,dtv,gm,gh,sm,sh'

  do c = 1, ncase
    do k = 1, nz
      dz(k) = 25.0 + 4.0 * real(k - 1)
      u(k) = 2.0 + 1.1 * real(k - 1)
      v(k) = -1.0 + 0.35 * real(k - 1)
      ql(k) = 0.0
      vt(k) = 0.0
      vq(k) = 0.0
      select case (c)
      case (1)
        thl(k) = 288.0 + 0.8 * real(k - 1)
        qw(k) = 0.0060 - 0.00015 * real(k - 1)
      case (2)
        thl(k) = 302.0 - 0.65 * real(k - 1)
        qw(k) = 0.0120 - 0.00005 * real(k - 1)
      case (3)
        thl(k) = 295.0
        qw(k) = 0.0080
      case (4)
        thl(k) = 291.0 + 0.25 * real(k - 1)
        qw(k) = 0.0100 + 0.00020 * real(k - 1)
        ql(k) = max(0.0, 0.00025 - 0.000035 * real(k - 1))
        vt(k) = 0.015 * real(k - 1)
        vq(k) = -12.0 + 2.0 * real(k - 1)
      end select
      thetav(k) = thl(k) * (1.0 + 0.608 * qw(k))
    end do
    if (c == 3) then
      ! Preserve nonzero shear while making the buoyancy gradient exact zero.
      v = 0.0
    end if

    dtl = -999.0
    dqw = -999.0
    dtv = -999.0
    gm = -999.0
    gh = -999.0
    sm = -999.0
    sh = -999.0
    call mym_level2(kts, kte, dz, u, v, thl, thetav, qw, ql, vt, vq, &
        dtl, dqw, dtv, gm, gh, sm, sh)

    do k = kts + 1, kte
      write(unit, '(A,",",I0,25(",",ES24.16E3))') trim(names(c)), k, &
          dz(k), u(k), v(k), thl(k), thetav(k), qw(k), ql(k), vt(k), &
          vq(k), dz(k-1), u(k-1), v(k-1), thl(k-1), thetav(k-1), &
          qw(k-1), ql(k-1), vt(k-1), vq(k-1), &
          dtl(k), dqw(k), dtv(k), gm(k), gh(k), sm(k), sh(k)
    end do
  end do
  close(unit)
end program run_mynn_pbl_level2_oracle
