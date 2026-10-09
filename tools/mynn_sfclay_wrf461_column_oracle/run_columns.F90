! WRF v4.6.1 MYNN surface-layer column oracle (sf_sfclay_physics=5).
!
! Drives the unmodified phys/module_sf_mynn.F entry point SFCLAY1D_mynn
! (:370) over every column of columns.bin, for every option set of
! configs.txt, for several successive timesteps.  The INOUT state WRF keeps
! between calls (UST, MOL, QSFC, ZNT, USTM, HFX, QFX) is carried from one
! step to the next in place, as WRF carries it.  A seeded config first runs
! the SFCLAY_mynn wrapper's itimestep==1 block (:330-337), written here with
! the wrapper's own statements; an unseeded config is a restart that enters
! with the column's own state.
!
! Output (stream, little-endian):
!   int32 ncfg, ncol, nstate, nout
!   per config, per step: int32 cfg, step, itimestep;
!                         float32 state_in(ncol, nstate)  [state that ENTERED]
!                         float32 out(ncol, nout)         [WOOF output order]

program run_mynn_sfclay_columns
  use module_sf_mynn, only: mynn_sf_init_driver, SFCLAY1D_mynn
  implicit none

  integer, parameter :: nfield = 23, nstate = 7, nout = 35
  integer, parameter :: jds = 1, jde = 2, kds = 1, kde = 3
  integer, parameter :: jms = 1, jme = 1, kms = 1, kme = 2
  integer, parameter :: jts = 1, jte = 1, kts = 1, kte = 2
  integer, parameter :: j = 1, iz0tlnd = 0
  ! module_model_constants.F values SFCLAY_mynn receives from the driver.
  real, parameter :: r_d = 287.0, r_v = 461.6
  real, parameter :: cp = 7. * r_d / 2., grav = 9.81
  real, parameter :: rovcp = r_d / cp, xlv = 2.5e6
  real, parameter :: svp1 = 0.6112, svp2 = 17.67, svp3 = 29.65
  real, parameter :: svpt0 = 273.15
  real, parameter :: ep1 = r_v / r_d - 1., ep2 = r_d / r_v
  real, parameter :: karman = 0.4

  character(len=1024) :: in_path, cfg_path, out_path
  character(len=64) :: cfg_name
  integer :: ncol, nf, ncfg, icfg, istep, itimestep, i, uin, uout, ucfg
  integer :: isftcflx, isfflx, it0, seeded, nsteps, spp_pbl
  integer :: ids, ide, ims, ime, its, ite
  real :: dx
  real, allocatable :: tab(:, :)
  real, allocatable :: u1d(:), v1d(:), t1d(:), qv1d(:), p1d(:), rho1d(:)
  real, allocatable :: dz8w1d(:), u1d2(:), v1d2(:), dz2w1d(:), rstoch1d(:)
  real, allocatable :: psfcpa(:), tsk(:), pblh(:), mavail(:), xland(:)
  real, allocatable :: snowh(:), qcg(:)
  real, allocatable :: hfx(:), qfx(:), znt(:), qsfc(:), ust(:), mol(:)
  real, allocatable :: ustm(:)
  real, allocatable :: regime(:), zol(:), rmol(:), psim(:), psih(:)
  real, allocatable :: chs(:), chs2(:), cqs2(:), ch(:), flhc(:), flqc(:)
  real, allocatable :: qgh(:), lh(:), u10(:), v10(:), th2(:), t2(:), q2(:)
  real, allocatable :: gz1oz0(:), wspd(:), br(:), ck(:), cka(:), cd(:)
  real, allocatable :: cda(:), wstar(:), qstar(:), cpm(:)

  call get_command_argument(1, in_path)
  call get_command_argument(2, cfg_path)
  call get_command_argument(3, out_path)
  if (len_trim(out_path) == 0) then
    write(*, '(A)') 'usage: run_columns COLUMNS.bin CONFIGS.txt OUT.bin'
    error stop 2
  end if

  open(newunit=uin, file=trim(in_path), access='stream', &
       form='unformatted', status='old', action='read')
  read(uin) ncol, nf
  if (nf /= nfield) error stop 'columns.bin field count mismatch'
  allocate(tab(nfield, ncol))
  read(uin) tab
  close(uin)

  ids = 1; ide = ncol + 1; ims = 1; ime = ncol; its = 1; ite = ncol
  allocate(u1d(ncol), v1d(ncol), t1d(ncol), qv1d(ncol), p1d(ncol), &
           rho1d(ncol), dz8w1d(ncol), u1d2(ncol), v1d2(ncol), &
           dz2w1d(ncol), rstoch1d(ncol), psfcpa(ncol), tsk(ncol), &
           pblh(ncol), mavail(ncol), xland(ncol), snowh(ncol), qcg(ncol), &
           hfx(ncol), qfx(ncol), znt(ncol), qsfc(ncol), ust(ncol), &
           mol(ncol), ustm(ncol), regime(ncol), zol(ncol), rmol(ncol), &
           psim(ncol), psih(ncol), chs(ncol), chs2(ncol), cqs2(ncol), &
           ch(ncol), flhc(ncol), flqc(ncol), qgh(ncol), lh(ncol), &
           u10(ncol), v10(ncol), th2(ncol), t2(ncol), q2(ncol), &
           gz1oz0(ncol), wspd(ncol), br(ncol), ck(ncol), cka(ncol), &
           cd(ncol), cda(ncol), wstar(ncol), qstar(ncol), cpm(ncol))

  ! columns.py STATIC_FIELDS then STATE_FIELDS.
  u1d = tab(1, :);  v1d = tab(2, :);  t1d = tab(3, :);  qv1d = tab(4, :)
  p1d = tab(5, :);  rho1d = tab(6, :); dz8w1d = tab(7, :)
  u1d2 = tab(8, :); v1d2 = tab(9, :); dz2w1d = tab(10, :)
  psfcpa = tab(11, :); tsk = tab(12, :); pblh = tab(13, :)
  mavail = tab(14, :); xland = tab(15, :); snowh = tab(16, :)
  rstoch1d = 0.0
  qcg = 0.0

  call mynn_sf_init_driver(.false.)

  open(newunit=ucfg, file=trim(cfg_path), status='old', action='read')
  read(ucfg, *) ncfg
  open(newunit=uout, file=trim(out_path), access='stream', &
       form='unformatted', status='new', action='write')
  write(uout) ncfg, ncol, nstate, nout

  do icfg = 1, ncfg
    read(ucfg, *) isftcflx, isfflx, dx, it0, seeded, nsteps, spp_pbl, cfg_name
    ! SPP option sets (spp_pbl=1): a fixed pattern, 0.15*(mod(i-1,7)-3) in
    ! default REAL, which columns.py spp_pattern() reproduces bit for bit.
    ! SFCLAY_mynn passes pattern_spp_pbl(i,kts,j) through as rstoch1D (:323-327).
    do i = 1, ncol
      if (spp_pbl == 1) then
        rstoch1d(i) = 0.15 * real(mod(i - 1, 7) - 3)
      else
        rstoch1d(i) = 0.0
      end if
    end do
    call reset_state()
    do istep = 1, nsteps
      itimestep = it0 + istep - 1
      if (seeded == 1 .and. itimestep == 1) then
        ! module_sf_mynn.F:330-337, the wrapper's first-step seeding.
        do i = 1, ncol
          ust(i) = MAX(0.04*SQRT(u1d(i)*u1d(i) + v1d(i)*v1d(i)), 0.001)
          mol(i) = 0.
          qsfc(i) = qv1d(i)/(1.+qv1d(i))
          qstar(i) = 0.0
        end do
      end if
      write(uout) icfg - 1, istep - 1, itimestep
      write(uout) hfx, qfx, znt, qsfc, ust, mol, ustm
      call SFCLAY1D_mynn( &
          j, u1d, v1d, t1d, qv1d, p1d, dz8w1d, rho1d, &
          u1d2, v1d2, dz2w1d, &
          cp, grav, rovcp, r_d, xlv, psfcpa, chs, chs2, cqs2, cpm, &
          pblh, rmol, znt, ust, mavail, zol, mol, regime, &
          psim, psih, xland, hfx, qfx, tsk, &
          u10, v10, th2, t2, q2, flhc, flqc, snowh, qgh, &
          qsfc, lh, gz1oz0, wspd, br, isfflx, dx, &
          svp1, svp2, svp3, svpt0, ep1, ep2, &
          karman, ch, qcg, itimestep, wstar, qstar, &
          spp_pbl, rstoch1d, &
          ids, ide, jds, jde, kds, kde, &
          ims, ime, jms, jme, kms, kme, &
          its, ite, jts, jte, kts, kte, &
          isftcflx, iz0tlnd, ustm, ck, cka, cd, cda)
      write(uout) regime, zol, rmol, ust, ustm, mol, psim, psih, &
          chs, chs2, cqs2, ch, flhc, flqc, qgh, qsfc, &
          hfx, qfx, lh, u10, v10, th2, t2, q2, &
          gz1oz0, wspd, br, ck, cka, cd, cda, wstar, &
          qstar, cpm, znt
    end do
  end do
  close(uout)
  close(ucfg)

contains

  subroutine reset_state()
    hfx = tab(17, :); qfx = tab(18, :); znt = tab(19, :)
    qsfc = tab(20, :); ust = tab(21, :); mol = tab(22, :)
    ustm = tab(23, :)
    lh = 0.0; regime = 0.0; zol = 0.0; rmol = 0.0; psim = 0.0; psih = 0.0
    chs = 0.0; chs2 = 0.0; cqs2 = 0.0; ch = 0.0; flhc = 0.0; flqc = 0.0
    qgh = 0.0; u10 = 0.0; v10 = 0.0; th2 = 0.0; t2 = 0.0; q2 = 0.0
    gz1oz0 = 0.0; wspd = 0.0; br = 0.0; ck = 0.0; cka = 0.0; cd = 0.0
    cda = 0.0; wstar = 0.0; qstar = 0.0; cpm = 0.0
  end subroutine reset_state

end program run_mynn_sfclay_columns
