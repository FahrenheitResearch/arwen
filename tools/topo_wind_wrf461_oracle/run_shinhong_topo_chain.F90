program run_shinhong_topo_chain
  ! The topo_wind chain oracle's second consumer: WRF v4.6.1 Shin-Hong.
  !
  ! module_pbl_driver.F hands CTOPO/CTOPO2 to Shin-Hong (:1318) exactly as it
  ! hands them to YSU, and phys/module_bl_shinhong.F reads both: the surface
  ! drag diagonal (:1387) and the 10 m wind blend (:1472-1473).  On the same
  ! patches and cells as run_topo_wind_chain.F90:
  !
  !   1. dyn_em/start_em.F:1539-1626 (wrf_topo_wind_inline, cut verbatim)
  !      forms CTOPO/CTOPO2 on the patch for topo_wind = 0, 1 and 2;
  !   2. the byte-unmodified `shinhong` entry runs on every cell with them,
  !      with the call tools/shinhong_column_oracle/run_shinhong_columns.F90
  !      makes (the call module_pbl_driver.F:1302 makes: t3d = th*pi as
  !      phy_prep forms it, psfc = p8w(1), tendencies zeroed, every optional
  !      argument present), except that ctopo/ctopo2 are the patch's.
  !
  ! Usage: run_shinhong_topo_chain PATCHES COLUMNS SHINHONG_EXTRAS OUT
  !
  ! OUT, per patch, per topo_wind in 0, 1, 2, per cell (j outer, i inner),
  ! the Shin-Hong column oracle's output record:
  !   int32[4]      nz, kpbl, patch, cell
  !   float32[nz]   rublten rvblten rthblten rqvblten rqcblten rqiblten
  !                 exch_h tke el
  !   float32[9]    hpbl wstar delta u10 v10 ust znt wspd regime
  use module_bl_shinhong, only: shinhong
  use wrf_topo_wind_inline
  implicit none

  real, parameter :: g = 9.81
  real, parameter :: r_d = 287.0
  real, parameter :: r_v = 461.6
  real, parameter :: cp = 7.0 * r_d / 2.0
  real, parameter :: rovcp = r_d / cp
  real, parameter :: rovg = r_d / g
  real, parameter :: xlv = 2.5e6
  real, parameter :: ep1 = r_v / r_d - 1.0
  real, parameter :: ep2 = r_d / r_v
  real, parameter :: karman = 0.4

  type col_t
    integer :: nz, tke_diag
    real :: sc(18), corf, dx, dy
    real, allocatable :: lev(:,:)    ! (nz, 11) make_wide_columns.py order
    real, allocatable :: pin(:), tke(:)
  end type col_t

  character(len=1024) :: patch_path, col_path, ext_path, out_path
  integer :: up, uc, ue, uo, ios, hdr(4), chdr(4), nx, ny, opt, i, j, n
  integer :: npatch, ncall, tkd
  real :: ext3(3)
  type(grid_t) :: grid
  type(config_t) :: config_flags
  type(col_t), allocatable :: cols(:)

  call get_command_argument(1, patch_path)
  call get_command_argument(2, col_path)
  call get_command_argument(3, ext_path)
  call get_command_argument(4, out_path)
  if (len_trim(out_path) == 0) then
    write(*, '(A)') 'usage: run_shinhong_topo_chain PATCHES COLUMNS EXTRAS OUT'
    error stop 2
  end if
  open(newunit=up, file=trim(patch_path), access='stream', form='unformatted', &
       status='old', action='read', convert='little_endian')
  open(newunit=uc, file=trim(col_path), access='stream', form='unformatted', &
       status='old', action='read', convert='little_endian')
  open(newunit=ue, file=trim(ext_path), access='stream', form='unformatted', &
       status='old', action='read', convert='little_endian')
  open(newunit=uo, file=trim(out_path), access='stream', form='unformatted', &
       status='replace', action='write', convert='little_endian')

  npatch = 0
  ncall = 0
  do
    read(up, iostat=ios) hdr
    if (ios /= 0) exit
    nx = hdr(1); ny = hdr(2)
    call alloc_grid(nx, ny)
    read(up) grid%ht, grid%xland, grid%var_sso, grid%var2d
    if (allocated(cols)) deallocate(cols)
    allocate(cols(nx * ny))
    do n = 1, nx * ny
      read(uc) chdr
      cols(n)%nz = chdr(1)
      read(uc) cols(n)%sc
      allocate(cols(n)%lev(chdr(1), 11), cols(n)%pin(chdr(1) + 1))
      allocate(cols(n)%tke(chdr(1)))
      read(uc) cols(n)%lev
      read(uc) cols(n)%pin
      read(ue) tkd
      read(ue) ext3
      read(ue) cols(n)%tke
      cols(n)%tke_diag = tkd
      cols(n)%corf = ext3(1); cols(n)%dx = ext3(2); cols(n)%dy = ext3(3)
    end do
    do opt = 0, 2
      config_flags%topo_wind = opt
      grid%ctopo = -999.0
      grid%ctopo2 = -999.0
      grid%lap_hgt = -999.0
      call topo_wind_static(grid, config_flags, 1, nx + 1, 1, ny + 1, &
                            1, nx, 1, ny)
      do j = 1, ny
        do i = 1, nx
          n = (j - 1) * nx + i
          call run_cell(cols(n), grid%ctopo(i, j), grid%ctopo2(i, j), &
                        npatch, n - 1)
          ncall = ncall + 1
        end do
      end do
    end do
    npatch = npatch + 1
  end do
  close(up); close(uc); close(ue); close(uo)
  write(*, '(A,I0,A,I0,A)') 'wrote ', npatch, ' patches, ', ncall, &
                            ' Shin-Hong calls'

contains

  subroutine alloc_grid(nx_, ny_)
    integer, intent(in) :: nx_, ny_
    if (allocated(grid%ht)) then
      deallocate(grid%toposlpx, grid%toposlpy, grid%lap_hgt, grid%slope, &
                 grid%slp_azi, grid%ht, grid%msftx, grid%msfty, grid%sina, &
                 grid%cosa, grid%ctopo, grid%ctopo2, grid%var_sso, &
                 grid%var2d, grid%xland)
    end if
    allocate(grid%toposlpx(nx_, ny_), grid%toposlpy(nx_, ny_))
    allocate(grid%lap_hgt(nx_, ny_), grid%slope(nx_, ny_))
    allocate(grid%slp_azi(nx_, ny_), grid%ht(nx_, ny_))
    allocate(grid%msftx(nx_, ny_), grid%msfty(nx_, ny_))
    allocate(grid%sina(nx_, ny_), grid%cosa(nx_, ny_))
    allocate(grid%ctopo(nx_, ny_), grid%ctopo2(nx_, ny_))
    allocate(grid%var_sso(nx_, ny_), grid%var2d(nx_, ny_))
    allocate(grid%xland(nx_, ny_))
    grid%rdx = 1.0 / 3000.0
    grid%rdy = 1.0 / 3000.0
    grid%msftx = 1.0
    grid%msfty = 1.0
    grid%sina = 0.0
    grid%cosa = 1.0
  end subroutine alloc_grid

  subroutine run_cell(c, ct, ct2, ipatch, icell)
    type(col_t), intent(in) :: c
    real, intent(in) :: ct, ct2
    integer, intent(in) :: ipatch, icell
    integer :: nz, k
    real, dimension(1, c%nz + 1, 1) :: u3d, v3d, th3d, t3d, qv3d, qc3d, &
         qi3d, p3d, p3di, pi3d, dz8w, rublten, rvblten, rthblten, rqvblten, &
         rqcblten, rqiblten, exch_h, tke_pbl, el_pbl
    real :: znu(c%nz + 1), znw(c%nz + 1)
    real, dimension(1, 1) :: psfc, znt, ust, hpbl, psim, psih, xland, hfx, &
         qfx, wspd, br, u10, v10, ctopo, ctopo2, corf, wstar, delta, regime
    integer, dimension(1, 1) :: kpbl2d
    real :: p_top

    nz = c%nz
    u3d = 0.0; v3d = 0.0; th3d = 0.0; t3d = 0.0
    qv3d = 0.0; qc3d = 0.0; qi3d = 0.0; p3d = 0.0; pi3d = 1.0; dz8w = 0.0
    p3di = 0.0
    do k = 1, nz
      u3d(1, k, 1) = c%lev(k, 1)
      v3d(1, k, 1) = c%lev(k, 2)
      th3d(1, k, 1) = c%lev(k, 3)
      pi3d(1, k, 1) = c%lev(k, 4)
      t3d(1, k, 1) = c%lev(k, 5)
      qv3d(1, k, 1) = c%lev(k, 6)
      qc3d(1, k, 1) = c%lev(k, 7)
      qi3d(1, k, 1) = c%lev(k, 8)
      p3d(1, k, 1) = c%lev(k, 9)
      dz8w(1, k, 1) = c%lev(k, 10)
    end do
    do k = 1, nz + 1
      p3di(1, k, 1) = c%pin(k)
    end do
    rublten = 0.0; rvblten = 0.0; rthblten = 0.0
    rqvblten = 0.0; rqcblten = 0.0; rqiblten = 0.0
    exch_h = 0.0; el_pbl = 0.0; tke_pbl = 0.0
    do k = 1, nz
      tke_pbl(1, k, 1) = c%tke(k)
    end do
    ! make_wide_columns.py scalars: dt psfc znt ust hfx qfx wspd br psim psih
    ! xland u10 v10 ctopo ctopo2 uox vox pad.  psfc = p8w(1), as the driver.
    psfc = c%pin(1); znt = c%sc(3); ust = c%sc(4); hfx = c%sc(5)
    qfx = c%sc(6); wspd = c%sc(7); br = c%sc(8); psim = c%sc(9)
    psih = c%sc(10); xland = c%sc(11); u10 = c%sc(12); v10 = c%sc(13)
    corf = c%corf
    hpbl = 0.0; kpbl2d = 0; wstar = 0.0; delta = 0.0; regime = 0.0
    ctopo = ct; ctopo2 = ct2
    p_top = c%pin(nz + 1)
    do k = 1, nz + 1
      znw(k) = (c%pin(k) - p_top) / (c%pin(1) - p_top)
    end do
    do k = 1, nz
      znu(k) = 0.5 * (znw(k) + znw(k + 1))
    end do
    znu(nz + 1) = 0.0

    call shinhong(u3d=u3d, v3d=v3d, th3d=th3d, t3d=t3d,                       &
                  qv3d=qv3d, qc3d=qc3d, qi3d=qi3d,                            &
                  p3d=p3d, p3di=p3di, pi3d=pi3d,                              &
                  rublten=rublten, rvblten=rvblten, rthblten=rthblten,        &
                  rqvblten=rqvblten, rqcblten=rqcblten, rqiblten=rqiblten,    &
                  flag_qi=.true.,                                             &
                  cp=cp, g=g, rovcp=rovcp, rd=r_d, rovg=rovg,                 &
                  ep1=ep1, ep2=ep2, karman=karman, xlv=xlv, rv=r_v,           &
                  dz8w=dz8w, psfc=psfc,                                       &
                  znu=znu, znw=znw, p_top=p_top,                              &
                  znt=znt, ust=ust, hpbl=hpbl, psim=psim, psih=psih,          &
                  xland=xland, hfx=hfx, qfx=qfx, wspd=wspd, br=br,            &
                  dt=c%sc(1), kpbl2d=kpbl2d,                                  &
                  exch_h=exch_h,                                              &
                  u10=u10, v10=v10,                                           &
                  shinhong_tke_diag=c%tke_diag,                               &
                  tke_pbl=tke_pbl, el_pbl=el_pbl, corf=corf,                  &
                  dx=c%dx, dy=c%dy,                                           &
                  ids=1, ide=1, jds=1, jde=1, kds=1, kde=nz + 1,              &
                  ims=1, ime=1, jms=1, jme=1, kms=1, kme=nz + 1,              &
                  its=1, ite=1, jts=1, jte=1, kts=1, kte=nz,                  &
                  ctopo=ctopo, ctopo2=ctopo2,                                 &
                  wstar=wstar, delta=delta,                                   &
                  regime=regime)

    write(uo) nz, kpbl2d(1, 1), ipatch, icell
    write(uo) rublten(1, 1:nz, 1)
    write(uo) rvblten(1, 1:nz, 1)
    write(uo) rthblten(1, 1:nz, 1)
    write(uo) rqvblten(1, 1:nz, 1)
    write(uo) rqcblten(1, 1:nz, 1)
    write(uo) rqiblten(1, 1:nz, 1)
    write(uo) exch_h(1, 1:nz, 1)
    write(uo) tke_pbl(1, 1:nz, 1)
    write(uo) el_pbl(1, 1:nz, 1)
    write(uo) hpbl(1, 1), wstar(1, 1), delta(1, 1), u10(1, 1), v10(1, 1), &
              ust(1, 1), znt(1, 1), wspd(1, 1), regime(1, 1)
  end subroutine run_cell

end program run_shinhong_topo_chain
