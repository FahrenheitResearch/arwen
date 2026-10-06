! GSL WRF 3.9 fork MYNN surface-layer oracle.
!
! Drives the unmodified fork phys/module_sf_mynn.F (NOAA-EMC/HRRR tag
! v4.1.21, sorc/hrrr_wrfarw.fd/WRFV3.9) through its SFCLAY1D_mynn entry
! point.  The columns come from a text file written by make_columns.py (one
! header line with the column count and step count, then one line per
! column); every column is advanced over NSTEP steps of the same atmosphere,
! carrying UST, USTM, MOL, QSFC, ZNT, HFX, QFX and LH from step to step, as
! the model does.  Step 1 enters with the SFCLAY_mynn wrapper's
! itimestep == 1 seeding (fork :329-337) already applied by the generator.
!
! Output: one CSV row per (step, column) with every column-solver output.

program run_mynn_surface_layer_fork_oracle
  use module_sf_mynn, only: mynn_sf_init_driver, SFCLAY1D_mynn
  implicit none

  integer :: n, nstep, step, i, iu, ou
  integer, parameter :: j = 1, isfflx = 1, isftcflx = 0, iz0tlnd = 0
  integer, parameter :: spp_pbl = 0
  real, parameter :: cp = 1004.5, grav = 9.81, r = 287.0
  real, parameter :: rovcp = r / cp, xlv = 2.5e6, dx = 3000.0
  real, parameter :: svp1 = 0.6112, svp2 = 17.67, svp3 = 29.65
  real, parameter :: svpt0 = 273.15, rv = 461.6
  real, parameter :: ep1 = rv / r - 1.0, ep2 = r / rv
  real, parameter :: karman = 0.4
  character(len=1024) :: in_path, out_path

  real, allocatable :: u1d(:), v1d(:), t1d(:), qv1d(:), p1d(:), dz8w1d(:)
  real, allocatable :: rho1d(:), u1d2(:), v1d2(:), dz2w1d(:), qc1d(:)
  real, allocatable :: rstoch1d(:)
  real, allocatable :: psfcpa(:), tsk(:), pblh(:), mavail(:), xland(:)
  real, allocatable :: snowh(:), qcg(:)
  real, allocatable :: chs(:), chs2(:), cqs2(:), cpm(:), rmol(:), znt(:)
  real, allocatable :: ust(:), zol(:), mol(:), regime(:), psim(:), psih(:)
  real, allocatable :: hfx(:), qfx(:), u10(:), v10(:), th2(:), t2(:), q2(:)
  real, allocatable :: flhc(:), flqc(:), qgh(:), qsfc(:), lh(:), gz1oz0(:)
  real, allocatable :: wspd(:), br(:), ch(:), wstar(:), qstar(:), ustm(:)
  real, allocatable :: ck(:), cka(:), cd(:), cda(:)
  real, allocatable :: hfx_in(:), qfx_in(:), znt_in(:), qsfc_in(:)
  real, allocatable :: ust_in(:), mol_in(:), ustm_in(:)

  call get_command_argument(1, in_path)
  call get_command_argument(2, out_path)
  if (len_trim(in_path) == 0 .or. len_trim(out_path) == 0) then
    write(*, '(A)') 'usage: run_surface_layer_fork COLUMNS.txt OUT.csv'
    error stop 2
  end if

  open(newunit=iu, file=trim(in_path), status='old', action='read')
  read(iu, *) n, nstep
  allocate(u1d(n), v1d(n), t1d(n), qv1d(n), p1d(n), dz8w1d(n), rho1d(n), &
           u1d2(n), v1d2(n), dz2w1d(n), qc1d(n), rstoch1d(n), &
           psfcpa(n), tsk(n), pblh(n), mavail(n), xland(n), snowh(n), &
           qcg(n), chs(n), chs2(n), cqs2(n), cpm(n), rmol(n), znt(n), &
           ust(n), zol(n), mol(n), regime(n), psim(n), psih(n), hfx(n), &
           qfx(n), u10(n), v10(n), th2(n), t2(n), q2(n), flhc(n), flqc(n), &
           qgh(n), qsfc(n), lh(n), gz1oz0(n), wspd(n), br(n), ch(n), &
           wstar(n), qstar(n), ustm(n), ck(n), cka(n), cd(n), cda(n), &
           hfx_in(n), qfx_in(n), znt_in(n), qsfc_in(n), ust_in(n), &
           mol_in(n), ustm_in(n))
  do i = 1, n
    read(iu, *) xland(i), snowh(i), u1d(i), v1d(i), t1d(i), qv1d(i), &
        p1d(i), rho1d(i), dz8w1d(i), u1d2(i), v1d2(i), dz2w1d(i), &
        psfcpa(i), tsk(i), pblh(i), mavail(i), hfx(i), qfx(i), znt(i), &
        qsfc(i), ust(i), mol(i)
  end do
  close(iu)

  qc1d = 0.0
  rstoch1d = 0.0
  qcg = qv1d / (1.0 + qv1d)
  ustm = ust
  lh = xlv * qfx
  chs = 0.0; chs2 = 0.0; cqs2 = 0.0; cpm = 0.0; rmol = 0.0; zol = 0.0
  regime = 0.0; psim = 0.0; psih = 0.0; u10 = 0.0; v10 = 0.0; th2 = 0.0
  t2 = 0.0; q2 = 0.0; flhc = 0.0; flqc = 0.0; qgh = 0.0; gz1oz0 = 0.0
  wspd = 0.0; br = 0.0; ch = 0.0; wstar = 0.0; qstar = 0.0
  ck = 0.0; cka = 0.0; cd = 0.0; cda = 0.0

  call mynn_sf_init_driver(.false.)

  open(newunit=ou, file=trim(out_path), status='new', action='write')
  write(ou, '(A)') 'step,column,hfx_input,qfx_input,znt_input,' // &
      'qsfc_input,ust_input,mol_input,ustm_input,' // &
      'regime,zol,rmol,ust,ustm,mol,psim,psih,chs,chs2,cqs2,ch,flhc,' // &
      'flqc,qgh,qsfc,hfx,qfx,lh,u10,v10,th2,t2,q2,gz1oz0,wspd,br,ck,' // &
      'cka,cd,cda,wstar,qstar,cpm,znt'
  do step = 1, nstep
    hfx_in = hfx; qfx_in = qfx; znt_in = znt; qsfc_in = qsfc
    ust_in = ust; mol_in = mol; ustm_in = ustm
    call SFCLAY1D_mynn( &
        j, u1d, v1d, t1d, qv1d, p1d, dz8w1d, rho1d, &
        u1d2, v1d2, dz2w1d, &
        cp, grav, rovcp, r, xlv, psfcpa, chs, chs2, cqs2, cpm, &
        pblh, rmol, znt, ust, mavail, zol, mol, regime, &
        psim, psih, xland, hfx, qfx, tsk, &
        u10, v10, th2, t2, q2, flhc, flqc, snowh, qgh, &
        qsfc, lh, gz1oz0, wspd, br, isfflx, dx, &
        svp1, svp2, svp3, svpt0, ep1, ep2, &
        karman, ch, qc1d, qcg, &
        step, &
        wstar, qstar, &
        spp_pbl, rstoch1d, &
        1, n + 1, 1, 2, 1, 3, &
        1, n, 1, 1, 1, 2, &
        1, n, 1, 1, 1, 2, &
        isftcflx, iz0tlnd, &
        ustm, ck, cka, cd, cda)
    do i = 1, n
      write(ou, '(I0,",",I0,42(",",ES24.16E3))') step, i - 1, &
          hfx_in(i), qfx_in(i), znt_in(i), qsfc_in(i), ust_in(i), &
          mol_in(i), ustm_in(i), &
          regime(i), zol(i), rmol(i), ust(i), ustm(i), mol(i), psim(i), &
          psih(i), chs(i), chs2(i), cqs2(i), ch(i), flhc(i), flqc(i), &
          qgh(i), qsfc(i), hfx(i), qfx(i), lh(i), u10(i), v10(i), th2(i), &
          t2(i), q2(i), gz1oz0(i), wspd(i), br(i), ck(i), cka(i), cd(i), &
          cda(i), wstar(i), qstar(i), cpm(i), znt(i)
    end do
  end do
  close(ou)
end program run_mynn_surface_layer_fork_oracle
