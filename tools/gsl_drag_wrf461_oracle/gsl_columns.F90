module gsl_columns
  ! Input columns for the gwd_opt = 3 (GSL drag suite) oracle against WRF
  ! v4.6.1.  Nothing here is WRF code and nothing here computes an expected
  ! value: it builds physically consistent inputs, which run_gsl_oracle.F90
  ! writes beside the words WRF's own gwdo_gsl returns, so the port reads its
  ! inputs from the fixture rather than regenerating them.
  !
  ! Two vertical grids, each its own gwdo_gsl call (kte is per call in WRF):
  !
  !   grid A  50 mass levels on an HRRR-style eta set, p_top = 2000 Pa,
  !           NCOL_A columns: the first NEDGE are hand-built edge cases, the
  !           rest cycle through the nine weather families below;
  !   grid B  40 mass levels on a smooth stretched eta set, p_top = 5000 Pa,
  !           NCOL_B columns cycling the families with another seed.
  !
  ! Families (FAMILY array, 0-8; edge cases are 100 + their number):
  !   0 convective day     1 stable night      2 near-neutral windy
  !   3 blocking (deep stable, weak flow, tall terrain)
  !   4 ocean              5 high terrain (plateau, mountain-wave flow)
  !   6 snow/ice arctic inversion               7 tropical moist
  !   8 jet with a critical level aloft
  !
  ! Each column is hydrostatic: interface pressures from the eta set,
  ! heights integrated upward from the terrain with the virtual temperature
  ! of each layer (double precision, rounded once to REAL at the end).
  ! Layout is WRF's memory order (ims:ime, kms:kme, jms:jme) with one j row;
  ! kme = nz + 1 (p3di uses it; the mass arrays carry a zero top level).
  implicit none
  integer, parameter :: nedge = 40
  integer, parameter :: ncol_a = 192, nz_a = 50
  integer, parameter :: ncol_b = 64, nz_b = 40
  real, parameter :: g = 9.81, r_d = 287.0, r_v = 461.6
  real, parameter :: cp = 7.0 * r_d / 2.0, ep_1 = r_v / r_d - 1.0

  type column_set
    integer :: ncol, nz
    real :: p_top
    real, allocatable :: u3d(:,:,:), v3d(:,:,:), t3d(:,:,:), qv3d(:,:,:)
    real, allocatable :: p3d(:,:,:), p3di(:,:,:), pi3d(:,:,:), z(:,:,:)
    real, allocatable :: dz(:,:,:), rub0(:,:,:), rvb0(:,:,:), rthb0(:,:,:)
    real, allocatable :: spp(:,:,:)
    real, allocatable :: znu(:), znw(:)
    real, allocatable :: var2d(:,:), oc12d(:,:), oa(:,:,:), ol(:,:,:)
    real, allocatable :: varss(:,:), ocss(:,:), oass(:,:,:), olss(:,:,:)
    real, allocatable :: sina(:,:), cosa(:,:), xland(:,:), br(:,:)
    real, allocatable :: pblh(:,:), ht(:)
    integer, allocatable :: kpbl(:,:), family(:)
  end type column_set

  ! The HRRR-style eta interfaces (51 values, surface to top).
  real(8), parameter :: eta_hrrr(51) = (/ &
    1.0000d0, 0.9980d0, 0.9940d0, 0.9870d0, 0.9750d0, 0.9590d0, 0.9390d0, &
    0.9160d0, 0.8920d0, 0.8650d0, 0.8350d0, 0.8020d0, 0.7660d0, 0.7270d0, &
    0.6850d0, 0.6400d0, 0.5920d0, 0.5420d0, 0.4970d0, 0.4565d0, 0.4205d0, &
    0.3877d0, 0.3582d0, 0.3317d0, 0.3078d0, 0.2863d0, 0.2670d0, 0.2496d0, &
    0.2329d0, 0.2188d0, 0.2047d0, 0.1906d0, 0.1765d0, 0.1624d0, 0.1483d0, &
    0.1342d0, 0.1201d0, 0.1060d0, 0.0919d0, 0.0778d0, 0.0657d0, 0.0568d0, &
    0.0486d0, 0.0409d0, 0.0337d0, 0.0271d0, 0.0209d0, 0.0151d0, 0.0097d0, &
    0.0047d0, 0.0000d0 /)

  ! Per-column profile controls (double precision).
  type profile
    real(8) :: ht, psfc_fac
    real(8) :: th0, zi, lapse_ml, jump, lapse_ft, dth_inv, h_inv
    real(8) :: spd, alpha, zr, shear, jet, zj, wj, dir0, veer, zcrit
    real(8) :: rh0, rh_top, qv_scale
  end type profile

  integer(8), save :: lcg_state = 1_8

contains

  real(8) function rnd()
    ! Park-Miller minimal standard generator, in [0, 1).
    lcg_state = modulo(16807_8 * lcg_state, 2147483647_8)
    rnd = real(lcg_state, 8) / 2147483647.0d0
  end function rnd

  real(8) function between(a, b)
    real(8), intent(in) :: a, b
    between = a + (b - a) * rnd()
  end function between

  subroutine allocate_set(s, ncol, nz, p_top)
    type(column_set), intent(inout) :: s
    integer, intent(in) :: ncol, nz
    real, intent(in) :: p_top
    integer :: nzi
    nzi = nz + 1
    s%ncol = ncol; s%nz = nz; s%p_top = p_top
    allocate(s%u3d(ncol,nzi,1), s%v3d(ncol,nzi,1), s%t3d(ncol,nzi,1), &
             s%qv3d(ncol,nzi,1), s%p3d(ncol,nzi,1), s%p3di(ncol,nzi,1), &
             s%pi3d(ncol,nzi,1), s%z(ncol,nzi,1), s%dz(ncol,nzi,1), &
             s%rub0(ncol,nzi,1), s%rvb0(ncol,nzi,1), s%rthb0(ncol,nzi,1), &
             s%spp(ncol,nzi,1))
    allocate(s%znu(nzi), s%znw(nzi))
    allocate(s%var2d(ncol,1), s%oc12d(ncol,1), s%oa(ncol,1,4), s%ol(ncol,1,4), &
             s%varss(ncol,1), s%ocss(ncol,1), s%oass(ncol,1,4), s%olss(ncol,1,4), &
             s%sina(ncol,1), s%cosa(ncol,1), s%xland(ncol,1), s%br(ncol,1), &
             s%pblh(ncol,1), s%ht(ncol), s%kpbl(ncol,1), s%family(ncol))
  end subroutine allocate_set

  ! ------------------------------------------------------------------------
  ! Weather families: draw the controls of one column.
  ! ------------------------------------------------------------------------
  subroutine draw_family(fam, p, pblh, br, xland)
    integer, intent(in) :: fam
    type(profile), intent(out) :: p
    real(8), intent(out) :: pblh, br, xland
    p%ht = between(0.0d0, 900.0d0); p%psfc_fac = between(0.985d0, 1.015d0)
    p%th0 = between(288.0d0, 300.0d0); p%zi = 1000.0d0
    p%lapse_ml = 0.0d0; p%jump = 1.0d0; p%lapse_ft = 0.0040d0
    p%dth_inv = 0.0d0; p%h_inv = 200.0d0
    p%spd = between(4.0d0, 12.0d0); p%alpha = 0.14d0; p%zr = 100.0d0
    p%shear = between(0.0d0, 0.0015d0); p%jet = 0.0d0; p%zj = 400.0d0
    p%wj = 150.0d0; p%dir0 = between(0.0d0, 360.0d0)
    p%veer = between(-0.004d0, 0.012d0); p%zcrit = -1.0d0
    p%rh0 = between(0.4d0, 0.8d0); p%rh_top = 0.2d0; p%qv_scale = 1.0d0
    xland = 1.0d0
    select case (fam)
    case (0)   ! convective day: deep mixed layer, unstable surface layer
      p%th0 = between(296.0d0, 312.0d0); p%zi = between(1200.0d0, 3200.0d0)
      p%lapse_ml = between(-0.0012d0, 0.0d0); p%jump = between(1.0d0, 4.0d0)
      p%lapse_ft = between(0.0035d0, 0.0055d0); p%alpha = 0.08d0
      pblh = p%zi * between(0.9d0, 1.05d0); br = between(-2.5d0, -0.05d0)
      p%rh0 = between(0.25d0, 0.6d0)
    case (1)   ! stable night: surface inversion and a low-level jet
      p%th0 = between(278.0d0, 292.0d0); p%zi = between(250.0d0, 600.0d0)
      p%dth_inv = between(4.0d0, 14.0d0); p%h_inv = between(80.0d0, 300.0d0)
      p%lapse_ft = between(0.0030d0, 0.0050d0); p%spd = between(2.0d0, 7.0d0)
      p%jet = between(6.0d0, 16.0d0); p%zj = between(200.0d0, 600.0d0)
      p%alpha = 0.3d0
      pblh = between(40.0d0, 300.0d0); br = between(0.02d0, 1.5d0)
    case (2)   ! near-neutral and windy (synoptic flow)
      p%th0 = between(282.0d0, 296.0d0); p%zi = between(700.0d0, 1500.0d0)
      p%lapse_ml = between(0.0d0, 0.0010d0); p%jump = between(0.5d0, 2.0d0)
      p%spd = between(12.0d0, 24.0d0); p%shear = between(0.0010d0, 0.0030d0)
      pblh = p%zi * between(0.8d0, 1.1d0); br = between(-0.05d0, 0.08d0)
    case (3)   ! blocking: deep stable layer, weak flow, tall terrain
      p%ht = between(800.0d0, 2600.0d0); p%th0 = between(272.0d0, 290.0d0)
      p%zi = between(1500.0d0, 3000.0d0); p%lapse_ml = between(0.010d0, 0.022d0)
      p%jump = 0.0d0; p%lapse_ft = between(0.005d0, 0.009d0)
      p%spd = between(1.5d0, 5.0d0); p%shear = between(0.0d0, 0.0008d0)
      pblh = between(60.0d0, 400.0d0); br = between(0.1d0, 3.0d0)
    case (4)   ! ocean: marine boundary layer
      p%ht = 0.0d0; xland = 2.0d0; p%th0 = between(285.0d0, 301.0d0)
      p%zi = between(400.0d0, 1200.0d0); p%jump = between(2.0d0, 8.0d0)
      p%spd = between(5.0d0, 18.0d0); p%alpha = 0.09d0
      p%rh0 = between(0.75d0, 0.95d0)
      pblh = p%zi; br = between(-0.4d0, 0.2d0)
    case (5)   ! high terrain: plateau, stable cross-barrier flow
      p%ht = between(2400.0d0, 4600.0d0); p%th0 = between(296.0d0, 318.0d0)
      p%zi = between(300.0d0, 1500.0d0); p%lapse_ml = between(0.002d0, 0.008d0)
      p%lapse_ft = between(0.004d0, 0.007d0); p%spd = between(8.0d0, 26.0d0)
      p%shear = between(0.0010d0, 0.0040d0)
      pblh = between(100.0d0, 1500.0d0); br = between(-0.3d0, 0.8d0)
      p%rh0 = between(0.2d0, 0.6d0)
    case (6)   ! snow/ice arctic: cold, dry, strong surface inversion
      p%ht = between(0.0d0, 1200.0d0); p%th0 = between(238.0d0, 262.0d0)
      p%zi = between(100.0d0, 400.0d0); p%dth_inv = between(8.0d0, 22.0d0)
      p%h_inv = between(50.0d0, 400.0d0); p%lapse_ft = between(0.004d0, 0.008d0)
      p%spd = between(1.0d0, 9.0d0); p%qv_scale = 0.6d0
      p%rh0 = between(0.7d0, 0.95d0)
      pblh = between(20.0d0, 200.0d0); br = between(0.05d0, 2.5d0)
    case (7)   ! tropical moist: warm, humid, weak flow
      p%ht = between(0.0d0, 400.0d0); p%th0 = between(298.0d0, 304.0d0)
      p%zi = between(500.0d0, 1200.0d0); p%lapse_ml = 0.0005d0
      p%jump = between(0.5d0, 1.5d0); p%lapse_ft = between(0.0030d0, 0.0040d0)
      p%spd = between(1.0d0, 8.0d0); p%rh0 = between(0.8d0, 0.97d0)
      p%rh_top = 0.5d0
      pblh = p%zi; br = between(-1.0d0, 0.02d0)
    case default   ! jet with a critical level (flow reversal) aloft
      p%th0 = between(285.0d0, 300.0d0); p%zi = between(500.0d0, 1500.0d0)
      p%lapse_ml = between(0.0d0, 0.004d0); p%spd = between(6.0d0, 18.0d0)
      p%shear = between(0.002d0, 0.006d0); p%zcrit = between(1500.0d0, 7000.0d0)
      p%veer = between(-0.01d0, 0.03d0)
      pblh = between(100.0d0, 1500.0d0); br = between(-0.3d0, 0.6d0)
    end select
  end subroutine draw_family

  ! Orographic statistics in realistic ranges.  Over water WPS masks the GSL
  ! statistics to zero; most ocean columns keep that, a few do not, so the
  ! scheme's own xland guards are exercised.
  subroutine draw_statistics(s, i, fam)
    type(column_set), intent(inout) :: s
    integer, intent(in) :: i, fam
    integer :: m
    real(8) :: scale
    scale = 1.0d0
    if (fam == 3 .or. fam == 5) scale = 2.5d0
    if (fam == 4 .or. fam == 7) scale = 0.3d0
    s%var2d(i,1) = real(scale * 260.0d0 * rnd()**2)
    s%oc12d(i,1) = real(between(0.0d0, 1.0d0)**2 * 12.0d0)
    s%varss(i,1) = real(scale * 90.0d0 * rnd()**2)
    s%ocss(i,1) = real(between(0.0d0, 1.0d0)**2 * 12.0d0)
    do m = 1, 4
      s%oa(i,1,m) = real(between(-1.0d0, 1.0d0))
      s%ol(i,1,m) = real(rnd())
      s%oass(i,1,m) = real(between(-1.0d0, 1.0d0))
      s%olss(i,1,m) = real(rnd())
    end do
    if (fam == 4 .and. modulo(i, 4) /= 1) then
      s%var2d(i,1) = 0.0; s%oc12d(i,1) = 0.0; s%oa(i,1,:) = 0.0
      s%ol(i,1,:) = 0.0; s%varss(i,1) = 0.0; s%ocss(i,1) = 0.0
      s%oass(i,1,:) = 0.0; s%olss(i,1,:) = 0.0
    end if
  end subroutine draw_statistics

  ! ------------------------------------------------------------------------
  ! One hydrostatic column from its controls.
  ! ------------------------------------------------------------------------
  real(8) function theta_at(p, zag, zasl)
    type(profile), intent(in) :: p
    real(8), intent(in) :: zag, zasl
    real(8) :: th
    if (zag <= p%zi) then
      th = p%th0 + p%lapse_ml * zag
    else
      th = p%th0 + p%lapse_ml * p%zi + p%jump + p%lapse_ft * (zag - p%zi)
    end if
    th = th + p%dth_inv * (1.0d0 - exp(-zag / p%h_inv))
    if (zasl > 11500.0d0) th = th + 0.012d0 * (zasl - 11500.0d0)
    theta_at = th
  end function theta_at

  subroutine wind_at(p, zag, u, v)
    type(profile), intent(in) :: p
    real(8), intent(in) :: zag
    real(8), intent(out) :: u, v
    real(8) :: s, d
    s = p%spd * min(1.0d0, (max(zag, 1.0d0) / p%zr)**p%alpha) + p%shear * zag &
        + p%jet * exp(-((zag - p%zj) / p%wj)**2)
    d = (p%dir0 + p%veer * zag) * 3.14159265358979d0 / 180.0d0
    u = -s * sin(d)
    v = -s * cos(d)
    if (p%zcrit > 0.0d0 .and. zag > p%zcrit) then
      u = -0.6d0 * u; v = -0.6d0 * v
    end if
  end subroutine wind_at

  real(8) function qsat(t, pr)
    real(8), intent(in) :: t, pr
    real(8) :: es
    if (t > 273.15d0) then
      es = 611.2d0 * exp(17.67d0 * (t - 273.15d0) / (t - 29.65d0))
    else
      es = 611.2d0 * exp(21.875d0 * (t - 273.15d0) / (t - 7.66d0))
    end if
    qsat = 0.622d0 * es / max(pr - es, 1.0d0)
  end function qsat

  subroutine build_column(s, i, p)
    type(column_set), intent(inout) :: s
    integer, intent(in) :: i
    type(profile), intent(in) :: p
    integer :: k, it, nz
    real(8) :: psfc, pt, zw(s%nz + 1), pw(s%nz + 1), pm(s%nz), zm, ex, th, t
    real(8) :: qv, tv, rh, u, v, dzk
    nz = s%nz
    pt = real(s%p_top, 8)
    psfc = 101325.0d0 * (1.0d0 - 2.25577d-5 * p%ht)**5.25588d0 * p%psfc_fac
    do k = 1, nz + 1
      pw(k) = real(s%znw(k), 8) * (psfc - pt) + pt
    end do
    do k = 1, nz
      pm(k) = real(s%znu(k), 8) * (psfc - pt) + pt
    end do
    zw(1) = p%ht
    do k = 1, nz
      zm = zw(k) + 30.0d0
      do it = 1, 4
        ex = (pm(k) / 1.0d5)**(287.0d0 / 1004.5d0)
        th = theta_at(p, zm - p%ht, zm)
        t = th * ex
        rh = p%rh0 + (p%rh_top - p%rh0) * min(1.0d0, (zm - p%ht) / 9000.0d0)
        qv = min(p%qv_scale * rh * qsat(t, pm(k)), 0.03d0)
        if (zm > 11000.0d0) qv = min(qv, 5.0d-6)   ! stratosphere: dry
        tv = t * (1.0d0 + 0.608d0 * qv)
        dzk = 287.0d0 * tv / 9.81d0 * log(pw(k) / pw(k + 1))
        zm = zw(k) + 0.5d0 * dzk
      end do
      zw(k + 1) = zw(k) + dzk
      call wind_at(p, zm - p%ht, u, v)
      s%u3d(i,k,1) = real(u); s%v3d(i,k,1) = real(v)
      s%t3d(i,k,1) = real(t); s%qv3d(i,k,1) = real(qv)
      s%p3d(i,k,1) = real(pm(k)); s%pi3d(i,k,1) = real(ex)
      s%z(i,k,1) = real(zm); s%dz(i,k,1) = real(dzk)
    end do
    do k = 1, nz + 1
      s%p3di(i,k,1) = real(pw(k))
    end do
    s%u3d(i,nz+1,1) = 0.0; s%v3d(i,nz+1,1) = 0.0; s%t3d(i,nz+1,1) = 0.0
    s%qv3d(i,nz+1,1) = 0.0; s%p3d(i,nz+1,1) = 0.0; s%pi3d(i,nz+1,1) = 0.0
    s%z(i,nz+1,1) = 0.0; s%dz(i,nz+1,1) = 0.0
    s%ht(i) = real(p%ht)
  end subroutine build_column

  ! KPBL as a PBL scheme reports it: the first mass level whose height above
  ! ground exceeds PBLH (1-based), at most nz.
  subroutine set_kpbl(s, i)
    type(column_set), intent(inout) :: s
    integer, intent(in) :: i
    integer :: k
    s%kpbl(i,1) = s%nz
    do k = 1, s%nz
      if (s%z(i,k,1) - s%ht(i) > s%pblh(i,1)) then
        s%kpbl(i,1) = k
        exit
      end if
    end do
  end subroutine set_kpbl

  ! The small-scale scheme's kpbl2: the first level >= 2 whose layer-centre
  ! height (from dz) exceeds 300 m, searched up to max(kpbl, 2); else kpbl.
  integer function ss_kpbl2(s, i)
    type(column_set), intent(in) :: s
    integer, intent(in) :: i
    integer :: k
    real(8) :: zq, za
    ss_kpbl2 = s%kpbl(i,1)
    zq = 0.0d0
    do k = 1, max(s%kpbl(i,1), 2)
      za = zq + 0.5d0 * real(s%dz(i,k,1), 8)
      zq = zq + real(s%dz(i,k,1), 8)
      if (k >= 2 .and. za > 300.0d0) then
        ss_kpbl2 = k
        return
      end if
    end do
  end function ss_kpbl2

  subroutine finish_column(s, i)
    type(column_set), intent(inout) :: s
    integer, intent(in) :: i
    integer :: k
    real(8) :: alpha
    alpha = between(-0.40d0, 0.40d0)
    s%sina(i,1) = real(sin(alpha)); s%cosa(i,1) = real(cos(alpha))
    do k = 1, s%nz
      s%rub0(i,k,1) = real(between(-3.0d-4, 3.0d-4))
      s%rvb0(i,k,1) = real(between(-3.0d-4, 3.0d-4))
      s%rthb0(i,k,1) = real(between(-1.0d-4, 1.0d-4))
      s%spp(i,k,1) = real(between(-1.0d0, 1.0d0))
    end do
    s%rub0(i,s%nz+1,1) = 0.0; s%rvb0(i,s%nz+1,1) = 0.0
    s%rthb0(i,s%nz+1,1) = 0.0; s%spp(i,s%nz+1,1) = 0.0
  end subroutine finish_column

  subroutine family_column(s, i, fam)
    type(column_set), intent(inout) :: s
    integer, intent(in) :: i, fam
    type(profile) :: p
    real(8) :: pblh, br, xland
    call draw_family(fam, p, pblh, br, xland)
    s%family(i) = fam
    s%pblh(i,1) = real(pblh); s%br(i,1) = real(br); s%xland(i,1) = real(xland)
    call build_column(s, i, p)
    call set_kpbl(s, i)
    call draw_statistics(s, i, fam)
    call finish_column(s, i)
  end subroutine family_column

  ! ------------------------------------------------------------------------
  ! Edge cases (grid A columns 1..NEDGE), each from a base family column.
  ! ------------------------------------------------------------------------
  subroutine edge_column(s, i)
    type(column_set), intent(inout) :: s
    integer, intent(in) :: i
    type(profile) :: p
    real(8) :: pblh, br, xland, dirdeg, d, spd
    integer :: base, k, k2
    real(8), parameter :: nudge(6) = (/ -2.0d-5, 2.0d-5, -2.0d-4, 2.0d-4, &
                                        -1.0d-6, 1.0d-6 /)
    base = 1
    if (i == 2 .or. i == 32) base = 2
    if (i == 38) base = 0
    call draw_family(base, p, pblh, br, xland)
    s%pblh(i,1) = real(pblh); s%br(i,1) = real(br); s%xland(i,1) = real(xland)
    ! Wind-direction columns: a height-constant wind, so the PBL mean wind
    ! keeps the direction to within rounding.
    select case (i)
    case (3:18, 39:40)
      p%shear = 0.0d0; p%jet = 0.0d0; p%veer = 0.0d0; p%alpha = 0.0d0
      p%spd = 10.0d0
    end select
    select case (i)
    case (3); p%dir0 = 270.0d0    ! westerly: v = 0
    case (4); p%dir0 = 180.0d0    ! southerly: u = 0
    case (5); p%dir0 = 90.0d0     ! easterly
    case (6); p%dir0 = 0.0d0      ! northerly
    case (7:14)                   ! the eight sector boundaries
      p%dir0 = 22.5d0 + 45.0d0 * real(i - 7, 8)
    case (15:18, 39:40)           ! next to a boundary, both sides
      p%dir0 = 22.5d0 + 45.0d0 * real(modulo(i, 8), 8)
      p%dir0 = p%dir0 + nudge(merge(i - 14, i - 34, i <= 18))
    end select
    call build_column(s, i, p)
    s%family(i) = 100 + i
    call set_kpbl(s, i)
    call draw_statistics(s, i, base)
    call finish_column(s, i)
    if (i >= 3 .and. i <= 18 .or. i >= 39) then
      s%sina(i,1) = 0.0; s%cosa(i,1) = 1.0
      dirdeg = p%dir0
      d = dirdeg * 3.14159265358979d0 / 180.0d0
      spd = 10.0d0
      do k = 1, s%nz
        s%u3d(i,k,1) = real(-spd * sin(d)); s%v3d(i,k,1) = real(-spd * cos(d))
        if (i == 3 .or. i == 5) s%v3d(i,k,1) = 0.0
        if (i == 4 .or. i == 6) s%u3d(i,k,1) = 0.0
      end do
    end if
    select case (i)
    case (1)            ! calm: no wind anywhere
      s%u3d(i,:,1) = 0.0; s%v3d(i,:,1) = 0.0
    case (2)            ! wind below 1 m/s: ulow clamps to exactly 1
      s%u3d(i,1:s%nz,1) = 0.35 * s%u3d(i,1:s%nz,1) / max(1.0e-3, &
          sqrt(s%u3d(i,1:s%nz,1)**2 + s%v3d(i,1:s%nz,1)**2))
      s%v3d(i,1:s%nz,1) = 0.0
    case (19)           ! flat land: no sub-grid variance at all
      s%var2d(i,1) = 0.0; s%varss(i,1) = 0.0
    case (20); s%varss(i,1) = 35.0      ! varmax_ss exactly
    case (21); s%varss(i,1) = 160.0     ! varmax_fd exactly
    case (22); s%varss(i,1) = 400.0; s%var2d(i,1) = 600.0
    case (23)           ! 2*var above the model top: klowtop stays 0
      s%var2d(i,1) = 20000.0
    case (24)           ! kpbl = 1
      s%pblh(i,1) = 1.0; s%kpbl(i,1) = 1
    case (25)           ! kpbl = nz
      s%pblh(i,1) = 30000.0; s%kpbl(i,1) = s%nz
    case (26)           ! hpbl = 0 and varss = 0: H_efold = 0
      s%pblh(i,1) = 0.0; s%varss(i,1) = 0.0; s%kpbl(i,1) = 1
    case (27)           ! oa = -1, ol = 0 (olmin floor, od clamp)
      s%oa(i,1,:) = -1.0; s%ol(i,1,:) = 0.0; s%oass(i,1,:) = -1.0
      s%olss(i,1,:) = 0.0
    case (28)
      s%oa(i,1,:) = 1.0; s%ol(i,1,:) = 1.0; s%oass(i,1,:) = 1.0
      s%olss(i,1,:) = 1.0
    case (29)           ! tiny varss: subnormal form-drag tendencies
      s%varss(i,1) = 1.0e-3
    case (30); s%br(i,1) = 0.0
    case (31)           ! reversal between the two lowest levels
      s%u3d(i,2:s%nz,1) = -s%u3d(i,2:s%nz,1)
      s%v3d(i,2:s%nz,1) = -s%v3d(i,2:s%nz,1)
    case (32)           ! very strong stable low-level flow: frmax, efmax
      s%u3d(i,1:s%nz,1) = 4.0 * s%u3d(i,1:s%nz,1)
      s%v3d(i,1:s%nz,1) = 4.0 * s%v3d(i,1:s%nz,1)
      s%br(i,1) = 0.4; s%var2d(i,1) = 900.0; s%oc12d(i,1) = 10.0
    case (33)           ! water with large statistics
      s%xland(i,1) = 2.0; s%varss(i,1) = 200.0; s%var2d(i,1) = 500.0
    case (34)           ! superadiabatic lowest layers with br > 0
      do k = 2, 6     ! theta falls 0.5 K per level
        s%t3d(i,k,1) = (s%t3d(i,1,1) / s%pi3d(i,1,1) - 0.5 * real(k - 1)) &
                       * s%pi3d(i,k,1)
      end do
      s%br(i,1) = 0.3
    case (35, 36)       ! u (35) or v (36) exactly zero at kpbl2
      s%sina(i,1) = 0.0; s%cosa(i,1) = 1.0
      s%varss(i,1) = 20.0; s%br(i,1) = 0.5
      s%xland(i,1) = 1.0
      s%pblh(i,1) = max(s%pblh(i,1), 2.0 * s%varss(i,1) + 50.0)
      call set_kpbl(s, i)
      k2 = ss_kpbl2(s, i)
      if (i == 35) s%u3d(i,k2,1) = 0.0
      if (i == 36) s%v3d(i,k2,1) = 0.0
    case (37)           ! negative-zero wind components
      do k = 1, s%nz, 3
        s%u3d(i,k,1) = -0.0
      end do
      s%v3d(i,2,1) = -0.0
    case (38)           ! deep desert PBL above kpblmax
      s%pblh(i,1) = 5200.0; call set_kpbl(s, i)
    end select
  end subroutine edge_column

  ! ------------------------------------------------------------------------
  subroutine build_grid_a(s)
    type(column_set), intent(inout) :: s
    integer :: i, k
    call allocate_set(s, ncol_a, nz_a, 2000.0)
    do k = 1, nz_a + 1
      s%znw(k) = real(eta_hrrr(k))
    end do
    do k = 1, nz_a
      s%znu(k) = 0.5 * (s%znw(k) + s%znw(k + 1))
    end do
    s%znu(nz_a + 1) = 0.0
    lcg_state = 20261007_8
    do i = 1, ncol_a
      if (i <= nedge) then
        call edge_column(s, i)
      else
        call family_column(s, i, modulo(i - nedge - 1, 9))
      end if
    end do
  end subroutine build_grid_a

  subroutine build_grid_b(s)
    type(column_set), intent(inout) :: s
    integer :: i, k
    real(8) :: x
    call allocate_set(s, ncol_b, nz_b, 5000.0)
    do k = 1, nz_b + 1
      x = real(k - 1, 8) / real(nz_b, 8)
      s%znw(k) = real(1.0d0 - (0.55d0 * x + 0.45d0 * x**1.6d0))
    end do
    s%znw(nz_b + 1) = 0.0
    do k = 1, nz_b
      s%znu(k) = 0.5 * (s%znw(k) + s%znw(k + 1))
    end do
    s%znu(nz_b + 1) = 0.0
    lcg_state = 4610001_8
    do i = 1, ncol_b
      call family_column(s, i, modulo(i * 5, 9))
    end do
  end subroutine build_grid_b

end module gsl_columns
