program run_topo_wind_chain
  ! The topo_wind chain oracle: WRF v4.6.1's own statements end to end.
  !
  !   1. dyn_em/start_em.F:1539-1626 (cut verbatim by extract.py into
  !      wrf_topo_wind_inline) forms LAP_HGT, CTOPO and CTOPO2 on a whole
  !      terrain patch, for topo_wind = 0, 1 and 2, with the non-periodic
  !      edge clamp (ids = jds = 1, ide = nx+1, jde = ny+1: a patch is a
  !      whole domain);
  !   2. phys/physics_mmm/bl_ysu.F90 (byte-unmodified) runs on every cell of
  !      the patch with the CTOPO/CTOPO2 step 1 produced for that cell, with
  !      the argument set module_bl_ysu.F:404-435 passes in a WRF run (the
  !      same call as tools/ysu_wrf461_oracle/run_bl_ysu_wide.F90: ocean
  !      currents zero but present, BEP arrays present with flag_bep false,
  !      f_qc = f_qi = .true., nmix = 1, qmix = qv/2).
  !
  ! Usage: run_topo_wind_chain PATCHES.bin COLUMNS.bin STATIC_OUT.bin YSU_OUT.bin
  !
  ! STATIC_OUT, per patch and per topo_wind in 0, 1, 2:
  !   float32[nx*ny] lap_hgt, ctopo, ctopo2       (i fastest)
  ! YSU_OUT, per patch, per topo_wind in 0, 1, 2, per cell (j outer, i inner):
  !   the wide oracle's output record: int32 kpbl; float32 hpbl, wstar, delta,
  !   u10, v10; float32[nz] utnp, vtnp, ttnp, rthblten, qvtnp, qctnp, qitnp,
  !   exch_hx, exch_mx, qmixtnp
  use ccpp_kind_types, only: kind_phys
  use bl_ysu, only: bl_ysu_run
  use wrf_topo_wind_inline
  implicit none

  real(kind=kind_phys), parameter :: g = 9.81
  real(kind=kind_phys), parameter :: r_d = 287.0
  real(kind=kind_phys), parameter :: r_v = 461.6
  real(kind=kind_phys), parameter :: cp = 7.0 * r_d / 2.0
  real(kind=kind_phys), parameter :: rovcp = r_d / cp
  real(kind=kind_phys), parameter :: rovg = r_d / g
  real(kind=kind_phys), parameter :: xlv = 2.5e6
  real(kind=kind_phys), parameter :: ep1 = r_v / r_d - 1.0
  real(kind=kind_phys), parameter :: ep2 = r_d / r_v
  real(kind=kind_phys), parameter :: karman = 0.4

  type col_t
    integer :: nz, topdown
    real(kind=kind_phys) :: sc(18)
    real(kind=kind_phys), allocatable :: lev(:,:)    ! (nz, 11)
    real(kind=kind_phys), allocatable :: pin(:)      ! (nz+1)
  end type col_t

  character(len=1024) :: patch_path, col_path, static_path, ysu_path
  integer :: up, uc, us, uy, ios, hdr(4), chdr(4), nx, ny, opt, i, j, n
  integer :: npatch, ncall
  type(grid_t) :: grid
  type(config_t) :: config_flags
  type(col_t), allocatable :: cols(:)

  call get_command_argument(1, patch_path)
  call get_command_argument(2, col_path)
  call get_command_argument(3, static_path)
  call get_command_argument(4, ysu_path)
  if (len_trim(ysu_path) == 0) then
    write(*, '(A)') 'usage: run_topo_wind_chain PATCHES COLUMNS STATIC_OUT YSU_OUT'
    error stop 2
  end if
  open(newunit=up, file=trim(patch_path), access='stream', form='unformatted', &
       status='old', action='read', convert='little_endian')
  open(newunit=uc, file=trim(col_path), access='stream', form='unformatted', &
       status='old', action='read', convert='little_endian')
  open(newunit=us, file=trim(static_path), access='stream', &
       form='unformatted', status='replace', action='write', &
       convert='little_endian')
  open(newunit=uy, file=trim(ysu_path), access='stream', form='unformatted', &
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
      cols(n)%topdown = chdr(2)
      read(uc) cols(n)%sc
      allocate(cols(n)%lev(chdr(1), 11), cols(n)%pin(chdr(1) + 1))
      read(uc) cols(n)%lev
      read(uc) cols(n)%pin
    end do

    do opt = 0, 2
      config_flags%topo_wind = opt
      grid%ctopo = -999.0
      grid%ctopo2 = -999.0
      grid%lap_hgt = -999.0
      call topo_wind_static(grid, config_flags, 1, nx + 1, 1, ny + 1, &
                            1, nx, 1, ny)
      write(us) grid%lap_hgt, grid%ctopo, grid%ctopo2
      do j = 1, ny
        do i = 1, nx
          n = (j - 1) * nx + i
          if (cols(n)%sc(11) /= grid%xland(i, j)) then
            write(*, '(A,I0,A,I0)') 'xland differs between patch and column ', n, &
                                     ' of patch ', npatch
            error stop 5
          end if
          call run_cell(cols(n), grid%ctopo(i, j), grid%ctopo2(i, j))
          ncall = ncall + 1
        end do
      end do
    end do
    npatch = npatch + 1
  end do
  read(uc, iostat=ios) chdr
  if (ios == 0) then
    write(*, '(A)') 'columns file has more records than the patches have cells'
    error stop 6
  end if
  close(up); close(uc); close(us); close(uy)
  write(*, '(A,I0,A,I0,A)') 'wrote ', npatch, ' patches, ', ncall, ' YSU calls'

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
    ! The slope fields the same loop forms are not topo_wind's; give their
    ! inputs ordinary values so the loop runs as in a model.
    grid%rdx = 1.0 / 3000.0
    grid%rdy = 1.0 / 3000.0
    grid%msftx = 1.0
    grid%msfty = 1.0
    grid%sina = 0.0
    grid%cosa = 1.0
  end subroutine alloc_grid

  subroutine run_cell(c, ct, ct2)
    type(col_t), intent(in) :: c
    real, intent(in) :: ct, ct2
    integer :: nz, k, errflg
    character(len=256) :: errmsg
    real(kind=kind_phys) :: ux(1, c%nz), vx(1, c%nz), tx(1, c%nz)
    real(kind=kind_phys) :: pi2d(1, c%nz), qvx(1, c%nz), qcx(1, c%nz)
    real(kind=kind_phys) :: qix(1, c%nz), p2d(1, c%nz), dz8w2d(1, c%nz)
    real(kind=kind_phys) :: rthraten(1, c%nz), p2di(1, c%nz + 1)
    real(kind=kind_phys) :: qmix(1, c%nz, 1), qmixtnp(1, c%nz, 1)
    real(kind=kind_phys) :: utnp(1, c%nz), vtnp(1, c%nz), ttnp(1, c%nz)
    real(kind=kind_phys) :: qvtnp(1, c%nz), qctnp(1, c%nz), qitnp(1, c%nz)
    real(kind=kind_phys) :: exch_hx(1, c%nz), exch_mx(1, c%nz)
    real(kind=kind_phys) :: bep0(1, c%nz), bep1(1, c%nz), rthblten(c%nz)
    real(kind=kind_phys) :: dt, psfcpa(1), znt(1), ust(1), hfx(1), qfx(1)
    real(kind=kind_phys) :: wspd(1), br(1), psim(1), psih(1), xland(1)
    real(kind=kind_phys) :: u10(1), v10(1), ctopo(1), ctopo2(1)
    real(kind=kind_phys) :: uox(1), vox(1), hpbl(1), wstar(1), delta(1)
    real(kind=kind_phys) :: frcurb(1)
    integer :: kpbl1d(1)

    nz = c%nz
    ! Level order of make_wide_columns.py: u, v, theta, pi, tx, qv, qc, qi,
    ! p, dz, rthraten.  tx is phy_prep's float32 theta*pi.
    ux(1, :) = c%lev(:, 1); vx(1, :) = c%lev(:, 2)
    pi2d(1, :) = c%lev(:, 4); tx(1, :) = c%lev(:, 5)
    qvx(1, :) = c%lev(:, 6); qcx(1, :) = c%lev(:, 7); qix(1, :) = c%lev(:, 8)
    p2d(1, :) = c%lev(:, 9); dz8w2d(1, :) = c%lev(:, 10)
    rthraten(1, :) = c%lev(:, 11)
    p2di(1, :) = c%pin
    dt = c%sc(1);  psfcpa(1) = c%sc(2);  znt(1) = c%sc(3);  ust(1) = c%sc(4)
    hfx(1) = c%sc(5);  qfx(1) = c%sc(6);  wspd(1) = c%sc(7);  br(1) = c%sc(8)
    psim(1) = c%sc(9);  psih(1) = c%sc(10);  xland(1) = c%sc(11)
    u10(1) = c%sc(12);  v10(1) = c%sc(13)
    ctopo(1) = ct;  ctopo2(1) = ct2
    uox(1) = c%sc(16);  vox(1) = c%sc(17)
    frcurb(1) = 0.0

    qmix(1, :, 1) = qvx(1, :) * 0.5
    utnp = 0.0; vtnp = 0.0; ttnp = 0.0
    qvtnp = 0.0; qctnp = 0.0; qitnp = 0.0
    exch_hx = 0.0; exch_mx = 0.0; qmixtnp = 0.0
    hpbl = 0.0; kpbl1d = 0; wstar = 0.0; delta = 0.0
    bep0 = 0.0; bep1 = 1.0
    errmsg = ''; errflg = -1

    call bl_ysu_run(ux=ux, vx=vx, tx=tx, qvx=qvx, qcx=qcx, qix=qix,          &
                    nmix=1, qmix=qmix, p2d=p2d, p2di=p2di, pi2d=pi2d,        &
                    f_qc=.true., f_qi=.true.,                               &
                    utnp=utnp, vtnp=vtnp, ttnp=ttnp, qvtnp=qvtnp,            &
                    qctnp=qctnp, qitnp=qitnp, qmixtnp=qmixtnp,               &
                    cp=cp, g=g, rovcp=rovcp, rd=r_d, rovg=rovg,              &
                    ep1=ep1, ep2=ep2, karman=karman, xlv=xlv, rv=r_v,        &
                    dz8w2d=dz8w2d, psfcpa=psfcpa,                            &
                    znt=znt, ust=ust, hpbl=hpbl,                             &
                    psim=psim, psih=psih, xland=xland,                       &
                    hfx=hfx, qfx=qfx, wspd=wspd, br=br,                      &
                    dt=dt, kpbl1d=kpbl1d,                                    &
                    exch_hx=exch_hx, exch_mx=exch_mx,                        &
                    wstar=wstar, delta=delta, u10=u10, v10=v10,              &
                    uox=uox, vox=vox,                                        &
                    rthraten=rthraten, ysu_topdown_pblmix=(c%topdown == 1),  &
                    ctopo=ctopo, ctopo2=ctopo2,                              &
                    a_u=bep0, a_v=bep0, a_t=bep0, a_q=bep0, a_e=bep0,        &
                    b_u=bep0, b_v=bep0, b_t=bep0, b_q=bep0, b_e=bep0,        &
                    sfk=bep1, vlk=bep1, dlu=bep0, dlg=bep0, frcurb=frcurb,   &
                    flag_bep=.false.,                                        &
                    its=1, ite=1, kte=nz, kme=nz + 1,                        &
                    errmsg=errmsg, errflg=errflg)
    if (errflg /= 0) then
      write(*, '(A,A)') 'bl_ysu_run errmsg: ', trim(errmsg)
      error stop 3
    end if
    do k = 1, nz
      rthblten(k) = ttnp(1, k) / pi2d(1, k)       ! module_bl_ysu.F:452
    end do
    write(uy) kpbl1d(1)
    write(uy) hpbl(1), wstar(1), delta(1), u10(1), v10(1)
    write(uy) utnp(1, :), vtnp(1, :), ttnp(1, :), rthblten(1:nz), &
              qvtnp(1, :), qctnp(1, :), qitnp(1, :), exch_hx(1, :), &
              exch_mx(1, :), qmixtnp(1, :, 1)
  end subroutine run_cell

end program run_topo_wind_chain
