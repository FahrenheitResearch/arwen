program run_topo_static
  ! CTOPO and CTOPO2 from WRF v4.7.1 start_em.F:1539-1626, cut verbatim by
  ! extract.py, on one synthetic terrain patch for topo_wind = 0, 1 and 2.
  ! Writes, per option, LAP_HGT, CTOPO and CTOPO2 (and the inputs) through
  ! oracle_io: fixtures/topo_static/topo_wind_<n>/.
  !
  ! The patch is built to reach every branch of the two blocks:
  !   * sqrt(VAR_SSO) at, below and above WRF's 2.718 test (ctopo = 1 or
  !     alog(sqrt(VAR_SSO))), including the float32 neighbours of 2.718;
  !   * LAP_HGT above -10, in [-20, -10), in [-30, -20) and below -30 (the
  !     hill-top taper and the zero-drag peak), and exactly -10, -20, -30;
  !   * water cells (XLAND = 2: no VAR_SSO term, topo_wind = 2 leaves them);
  !   * VAR on both sides of topo_wind = 2's 1.575 cap;
  !   * the domain edges, where WRF's neighbour indices clamp to the cell.
  use oracle_io
  use wrf_topo_wind_inline
  implicit none
  integer, parameter :: nx = 24, ny = 20
  integer :: ids, ide, jds, jde, i, j, opt
  type(grid_t) :: grid
  type(config_t) :: config_flags
  character(len=1024) :: root
  character(len=32) :: name
  real :: s
  integer :: seed

  call oracle_root(root)
  ids = 1; ide = nx + 1; jds = 1; jde = ny + 1
  allocate(grid%toposlpx(nx, ny), grid%toposlpy(nx, ny), grid%lap_hgt(nx, ny))
  allocate(grid%slope(nx, ny), grid%slp_azi(nx, ny), grid%ht(nx, ny))
  allocate(grid%msftx(nx, ny), grid%msfty(nx, ny), grid%sina(nx, ny))
  allocate(grid%cosa(nx, ny), grid%ctopo(nx, ny), grid%ctopo2(nx, ny))
  allocate(grid%var_sso(nx, ny), grid%var2d(nx, ny), grid%xland(nx, ny))
  grid%rdx = 1.0 / 3000.0
  grid%rdy = 1.0 / 3000.0
  grid%msftx = 1.0
  grid%msfty = 1.0
  grid%sina = 0.0
  grid%cosa = 1.0

  ! --- terrain, land mask, sub-grid statistics -----------------------------
  seed = 12345
  do j = 1, ny
    do i = 1, nx
      grid%ht(i, j) = 600.0 + 350.0 * sin(0.61 * real(i)) * cos(0.47 * real(j)) &
                      + 9.0 * real(j)
      grid%xland(i, j) = 1.0
      s = lcg(seed)
      grid%var_sso(i, j) = (300.0 * s) ** 2
      grid%var2d(i, j) = 900.0 * lcg(seed)
    end do
  end do
  ! Ocean on the first three columns, flat at sea level.
  do j = 1, ny
    do i = 1, 3
      grid%ht(i, j) = 0.0
      grid%xland(i, j) = 2.0
    end do
  end do
  ! Isolated peaks on flat ground, one per LAP_HGT band (lap = -bump at a
  ! single raised cell on a flat plateau).
  do j = 14, ny
    do i = 4, nx
      grid%ht(i, j) = 1000.0
    end do
  end do
  grid%ht(6, 16) = 1000.0 + 5.0       ! lap = -5   (> -10, unchanged)
  grid%ht(9, 16) = 1000.0 + 10.0      ! lap = -10  (boundary, > -10 false)
  grid%ht(12, 16) = 1000.0 + 15.0     ! lap = -15  (taper alpha)
  grid%ht(15, 16) = 1000.0 + 20.0     ! lap = -20  (boundary)
  grid%ht(18, 16) = 1000.0 + 25.0     ! lap = -25  (ctopo = ctopo2 = 0.5)
  grid%ht(21, 16) = 1000.0 + 30.0     ! lap = -30  (boundary)
  grid%ht(6, 19) = 1000.0 + 45.0      ! lap = -45  (zero drag)
  grid%ht(12, 19) = 1000.0 - 40.0     ! a pit: lap = +40
  ! VAR_SSO on WRF's sqrt(VAR_SSO) <= 2.718 test and its float32 neighbours.
  grid%var_sso(6, 16) = 2.718 ** 2
  grid%var_sso(9, 16) = nearest(2.718, 1.0) ** 2
  grid%var_sso(12, 16) = nearest(2.718, -1.0) ** 2
  grid%var_sso(15, 16) = 0.0
  grid%var_sso(18, 16) = 40000.0
  grid%var_sso(21, 16) = 7.389056
  ! VAR on both sides of topo_wind = 2's 1.575 cap (var*0.4/200 + 1.175).
  grid%var2d(6, 16) = 200.0
  grid%var2d(9, 16) = 199.99998
  grid%var2d(12, 16) = 200.00002
  grid%var2d(15, 16) = 0.0
  ! A water cell carrying VAR_SSO and VAR: both must be ignored there.
  grid%var_sso(2, 5) = 90000.0
  grid%var2d(2, 5) = 500.0

  call oracle_open('topo_static/inputs')
  call oracle_put('ht', grid%ht)
  call oracle_put('var_sso', grid%var_sso)
  call oracle_put('var2d', grid%var2d)
  call oracle_put('xland', grid%xland)
  call oracle_close()

  do opt = 0, 2
    config_flags%topo_wind = opt
    grid%ctopo = -999.0
    grid%ctopo2 = -999.0
    grid%lap_hgt = -999.0
    call topo_wind_static(grid, config_flags, ids, ide, jds, jde, &
                          1, nx, 1, ny)
    write(name, '(A,I0)') 'topo_static/topo_wind_', opt
    call oracle_open(trim(name))
    call oracle_put('lap_hgt', grid%lap_hgt)
    call oracle_put('ctopo', grid%ctopo)
    call oracle_put('ctopo2', grid%ctopo2)
    call oracle_close()
  end do

contains

  ! A small deterministic generator in [0, 1): the fixture's inputs are
  ! written out, so nothing downstream regenerates them.
  real function lcg(state)
    integer, intent(inout) :: state
    integer(8) :: wide
    wide = modulo(1103515245_8 * int(state, 8) + 12345_8, 2147483647_8)
    state = int(wide)
    lcg = real(modulo(state, 1000000)) / 1000000.0
  end function lcg

end program run_topo_static
