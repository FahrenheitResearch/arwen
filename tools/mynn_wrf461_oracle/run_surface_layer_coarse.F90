! Coarse-spacing WRF v4.6.1 MYNN surface-layer oracle (the DX > 5 km branch).
!
! module_sf_mynn.F:584 adds a Beljaars (1995) subgrid-scale gust velocity
!
!     VSGD = 0.32 * (max(DX/5000. - 1., 0.))**.33
!     WSPD = SQRT(WSPD*WSPD + WSTAR*WSTAR + VSGD*VSGD)
!
! which is identically zero for every DX <= 5000 m and grows without bound
! above it.  run_surface_layer.F90 and run_surface_layer_wide.F90 both fix
! DX = 3000.0, so every existing fixture pins only VSGD == 0 and the whole
! branch above the threshold is unmeasured -- yet every 12 km parent domain
! runs it on every column of every timestep.
!
! This program drives the SAME unmodified SFCLAY1D_mynn on the SAME ten
! columns as the widened fixture, sweeping DX across the threshold:
!
!     DX = 3000  -- VSGD == 0, reproduces the existing branch as a control
!     DX = 5000  -- exactly at the threshold, max() still clamps to 0
!     DX = 5001  -- one metre above it, the smallest live VSGD
!     DX = 12000 -- a real parent-domain spacing, VSGD ~ 0.3576 m s-1
!     DX = 27000 -- a coarse outer domain, VSGD ~ 0.5674 m s-1
!
! Each DX gets a fresh reset_state() and is then advanced over two timesteps
! so the ITIMESTEP>1 ZOL-from-MOL first guess (:792-796, :874-878) and the
! BR clip limit change (2 -> 4) are both exercised with VSGD live.
!
! It then resets again and runs ITIMESTEP=1 with ISFFLX=0, which is the third
! stage run_surface_layer_wide.F90 records and the only one where WRF drives
! the surface fluxes from the exchange coefficients (:1002-1006) instead of
! taking HFX/QFX as given.  Both arms consume WSPD, so leaving ISFFLX=0 out
! of the DX sweep left the flux-diagnosing half of the module unmeasured
! above the 5 km threshold.  The stage set is now exactly the widened
! fixture's -- (1,1), (2,1), (1,0) -- at every spacing.
!
! The narrow and widened fixtures and their byte-pinned CSVs are untouched
! by this program; it only ever writes the file named on the command line.

program run_mynn_surface_layer_coarse_oracle
  use module_sf_mynn, only: mynn_sf_init_driver, SFCLAY1D_mynn
  implicit none

  integer, parameter :: ncase = 10
  integer, parameter :: ndx = 5
  integer, parameter :: ids = 1, ide = ncase + 1
  integer, parameter :: jds = 1, jde = 2, kds = 1, kde = 3
  integer, parameter :: ims = 1, ime = ncase
  integer, parameter :: jms = 1, jme = 1, kms = 1, kme = 2
  integer, parameter :: its = 1, ite = ncase
  integer, parameter :: jts = 1, jte = 1, kts = 1, kte = 2
  integer, parameter :: j = 1
  integer, parameter :: isftcflx = 0, iz0tlnd = 0, spp_pbl = 0
  real, parameter :: cp = 1004.5, grav = 9.81, r = 287.0
  real, parameter :: rovcp = r / cp, xlv = 2.5e6
  real, parameter :: svp1 = 0.6112, svp2 = 17.67, svp3 = 29.65
  real, parameter :: svpt0 = 273.15, rv = 461.6
  real, parameter :: ep1 = rv / r - 1.0, ep2 = r / rv
  real, parameter :: karman = 0.4

  real, parameter :: dx_sweep(ndx) = &
      [3000.0, 5000.0, 5001.0, 12000.0, 27000.0]

  character(len=32) :: case_name(ncase)
  character(len=1024) :: output_path
  integer :: i, unit, idx

  real :: u1d(ncase), v1d(ncase), t1d(ncase), qv1d(ncase)
  real :: p1d(ncase), dz8w1d(ncase), rho1d(ncase)
  real :: u1d2(ncase), v1d2(ncase), dz2w1d(ncase)
  real :: rstoch1d(ncase)
  real :: znt0(ncase), hfx0(ncase), qfx0(ncase)

  real :: psfcpa(ims:ime, jms:jme), tsk(ims:ime, jms:jme)
  real :: pblh(ims:ime, jms:jme), mavail(ims:ime, jms:jme)
  real :: xland(ims:ime, jms:jme), snowh(ims:ime, jms:jme)
  real :: qcg(ims:ime, jms:jme)
  real :: chs(ims:ime, jms:jme), chs2(ims:ime, jms:jme)
  real :: cqs2(ims:ime, jms:jme), cpm(ims:ime, jms:jme)
  real :: rmol(ims:ime, jms:jme), znt(ims:ime, jms:jme)
  real :: ust(ims:ime, jms:jme), zol(ims:ime, jms:jme)
  real :: mol(ims:ime, jms:jme), regime(ims:ime, jms:jme)
  real :: psim(ims:ime, jms:jme), psih(ims:ime, jms:jme)
  real :: hfx(ims:ime, jms:jme), qfx(ims:ime, jms:jme)
  real :: u10(ims:ime, jms:jme), v10(ims:ime, jms:jme)
  real :: th2(ims:ime, jms:jme), t2(ims:ime, jms:jme)
  real :: q2(ims:ime, jms:jme), flhc(ims:ime, jms:jme)
  real :: flqc(ims:ime, jms:jme), qgh(ims:ime, jms:jme)
  real :: qsfc(ims:ime, jms:jme), lh(ims:ime, jms:jme)
  real :: gz1oz0(ims:ime, jms:jme), wspd(ims:ime, jms:jme)
  real :: br(ims:ime, jms:jme), ch(ims:ime, jms:jme)
  real :: wstar(ims:ime, jms:jme), qstar(ims:ime, jms:jme)
  real :: ustm(ims:ime, jms:jme), ck(ims:ime, jms:jme)
  real :: cka(ims:ime, jms:jme), cd(ims:ime, jms:jme)
  real :: cda(ims:ime, jms:jme)

  real :: hfx_in(ncase), qfx_in(ncase), znt_in(ncase), qsfc_in(ncase)
  real :: ust_in(ncase), mol_in(ncase), ustm_in(ncase)

  call get_command_argument(1, output_path)
  if (len_trim(output_path) == 0) then
    write(*, '(A)') 'usage: run_surface_layer_coarse OUTPUT.csv'
    error stop 2
  end if

  case_name = [character(len=32) :: &
      'strong_stable_land', 'clipped_stable_land', 'damped_stable_land', &
      'neutral_land', 'free_convective_land', 'land_qsfc_unset', &
      'thin_land_level2_wind', 'thin_land_log10_wind', 'midres_water', &
      'coarse_water']

  u1d  = [6.00, 1.20, 5.00, 4.00, 3.00, 5.00, 2.50, 2.00, 9.00, 11.00]
  v1d  = [1.00, 0.40, 1.00, 0.00, 1.00, 2.00, 0.80, 0.50, 3.00, 2.00]
  u1d2 = [8.00, 2.00, 6.50, 5.00, 4.00, 6.50, 3.20, 2.60, 11.00, 13.00]
  v1d2 = [1.40, 0.70, 1.30, 0.00, 1.40, 2.60, 1.00, 0.70, 3.60, 2.40]
  t1d  = [295.0, 295.0, 293.0, 295.0, 299.0, 297.0, 294.0, 296.0, 292.0, 299.0]
  qv1d = [0.0060, 0.0050, 0.0070, 0.0080, 0.0120, 0.0090, 0.0065, 0.0075, &
          0.0085, 0.0130]
  p1d  = [99000.0, 99200.0, 99500.0, 100000.0, 98500.0, 99000.0, 99800.0, &
          99850.0, 99000.0, 98500.0]
  dz8w1d = [40.0, 40.0, 40.0, 40.0, 44.0, 40.0, 6.0, 3.0, 18.0, 60.0]
  dz2w1d = [60.0, 60.0, 60.0, 60.0, 66.0, 60.0, 10.0, 40.0, 34.0, 90.0]
  znt0 = [0.10, 0.10, 0.10, 0.10, 0.12, 0.10, 0.05, 0.02, 0.0002, 0.0002]
  hfx0 = [-25.0, -15.0, -10.0, 0.0, 240.0, 150.0, -18.0, -14.0, -10.0, 130.0]
  qfx0 = [-1.0e-6, -8.0e-7, -5.0e-7, 0.0, 1.4e-4, 8.0e-5, -7.0e-7, &
          -6.0e-7, 2.0e-5, 9.0e-5]

  tsk(:, j) = [280.0, 278.0, 291.0, 295.0, 309.0, 302.0, 290.0, 291.0, &
               289.0, 304.0]
  psfcpa(:, j) = 100000.0
  pblh(:, j) = [250.0, 180.0, 400.0, 600.0, 1500.0, 900.0, 300.0, 260.0, &
                420.0, 1200.0]
  xland(:, j) = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0]
  snowh(:, j) = 0.0
  mavail(:, j) = [0.45, 0.40, 0.55, 0.55, 0.75, 0.50, 0.50, 0.50, 1.0, 1.0]
  qcg(:, j) = qv1d / (1.0 + qv1d)

  do i = 1, ncase
    rho1d(i) = p1d(i) / (r * t1d(i) * &
        (1.0 + ep1 * qv1d(i) / (1.0 + qv1d(i))))
  end do

  rstoch1d = 0.0
  chs(:, j) = 0.0
  chs2(:, j) = 0.0
  cqs2(:, j) = 0.0
  cpm(:, j) = 0.0
  psim(:, j) = 0.0
  psih(:, j) = 0.0
  gz1oz0(:, j) = 0.0
  wspd(:, j) = 0.0
  br(:, j) = 0.0
  ch(:, j) = 0.0
  flhc(:, j) = 0.0
  flqc(:, j) = 0.0
  u10(:, j) = 0.0
  v10(:, j) = 0.0
  th2(:, j) = 0.0
  t2(:, j) = 0.0
  q2(:, j) = 0.0
  wstar(:, j) = 0.0
  ck(:, j) = 0.0
  cka(:, j) = 0.0
  cd(:, j) = 0.0
  cda(:, j) = 0.0

  call mynn_sf_init_driver(.false.)

  open(newunit=unit, file=trim(output_path), status='new', action='write')
  call write_header(unit)
  do idx = 1, ndx
    call reset_state()
    call run_stage(unit, dx_sweep(idx), 1, 1)
    call run_stage(unit, dx_sweep(idx), 2, 1)
    ! Same entry state as the ISFFLX=1 first step, so the two (1,*) rows at
    ! each DX differ only in the flux branch -- exactly as the widened
    ! fixture's third stage does.
    call reset_state()
    call run_stage(unit, dx_sweep(idx), 1, 0)
  end do
  close(unit)

contains

  subroutine reset_state()
    qsfc(:, j) = qv1d / (1.0 + qv1d)
    qsfc(6, j) = 0.0
    qsfc(9:10, j) = 0.0
    znt(:, j) = znt0
    hfx(:, j) = hfx0
    qfx(:, j) = qfx0
    lh(:, j) = qfx0 * xlv
    ust(:, j) = 1.0e-4
    ustm(:, j) = 1.0e-4
    mol(:, j) = 0.0
    qstar(:, j) = 0.0
    zol(:, j) = 0.0
    rmol(:, j) = 0.0
    regime(:, j) = 0.0
    qgh(:, j) = 0.0
  end subroutine reset_state

  subroutine snapshot_inputs()
    hfx_in = hfx(:, j)
    qfx_in = qfx(:, j)
    znt_in = znt(:, j)
    qsfc_in = qsfc(:, j)
    ust_in = ust(:, j)
    mol_in = mol(:, j)
    ustm_in = ustm(:, j)
  end subroutine snapshot_inputs

  subroutine run_stage(out_unit, dx, itimestep, isfflx)
    integer, intent(in) :: out_unit, itimestep, isfflx
    real, intent(in) :: dx

    call snapshot_inputs()
    call SFCLAY1D_mynn( &
        j, u1d, v1d, t1d, qv1d, p1d, dz8w1d, rho1d, &
        u1d2, v1d2, dz2w1d, &
        cp, grav, rovcp, r, xlv, psfcpa(:, j), chs(:, j), chs2(:, j), &
        cqs2(:, j), cpm(:, j), &
        pblh(:, j), rmol(:, j), znt(:, j), ust(:, j), mavail(:, j), &
        zol(:, j), mol(:, j), regime(:, j), &
        psim(:, j), psih(:, j), xland(:, j), hfx(:, j), qfx(:, j), &
        tsk(:, j), &
        u10(:, j), v10(:, j), th2(:, j), t2(:, j), q2(:, j), flhc(:, j), &
        flqc(:, j), snowh(:, j), qgh(:, j), &
        qsfc(:, j), lh(:, j), gz1oz0(:, j), wspd(:, j), br(:, j), &
        isfflx, dx, &
        svp1, svp2, svp3, svpt0, ep1, ep2, &
        karman, ch(:, j), qcg(:, j), itimestep, wstar(:, j), qstar(:, j), &
        spp_pbl, rstoch1d, &
        ids, ide, jds, jde, kds, kde, &
        ims, ime, jms, jme, kms, kme, &
        its, ite, jts, jte, kts, kte, &
        isftcflx, iz0tlnd, ustm(:, j), ck(:, j), cka(:, j), cd(:, j), &
        cda(:, j))

    do i = 1, ncase
      write(out_unit, '(A,",",I0,",",I0,59(",",ES24.16E3))') &
          trim(case_name(i)), itimestep, isfflx, dx, &
          xland(i, j), snowh(i, j), u1d(i), v1d(i), t1d(i), qv1d(i), &
          p1d(i), rho1d(i), dz8w1d(i), u1d2(i), v1d2(i), dz2w1d(i), &
          psfcpa(i, j), tsk(i, j), pblh(i, j), mavail(i, j), &
          hfx_in(i), qfx_in(i), znt_in(i), qsfc_in(i), ust_in(i), &
          mol_in(i), ustm_in(i), &
          regime(i, j), zol(i, j), rmol(i, j), ust(i, j), ustm(i, j), &
          mol(i, j), psim(i, j), psih(i, j), chs(i, j), chs2(i, j), &
          cqs2(i, j), ch(i, j), flhc(i, j), flqc(i, j), qgh(i, j), &
          qsfc(i, j), hfx(i, j), qfx(i, j), lh(i, j), u10(i, j), &
          v10(i, j), th2(i, j), t2(i, j), q2(i, j), gz1oz0(i, j), &
          wspd(i, j), br(i, j), ck(i, j), cka(i, j), cd(i, j), cda(i, j), &
          wstar(i, j), qstar(i, j), cpm(i, j), znt(i, j)
    end do
  end subroutine run_stage

  subroutine write_header(out_unit)
    integer, intent(in) :: out_unit
    write(out_unit, '(A)') 'case,itimestep,isfflx,dx,' // &
        'xland,snowh,u1,v1,t1,qv1,p1,rho1,dz1,u2,v2,dz2,psfc,tsk,pblh,' // &
        'mavail,hfx_input,qfx_input,znt_input,qsfc_input,ust_input,' // &
        'mol_input,ustm_input,' // &
        'regime,zol,rmol,ust,ustm,mol,psim,psih,chs,chs2,cqs2,ch,flhc,' // &
        'flqc,qgh,qsfc,hfx,qfx,lh,u10,v10,th2,t2,q2,gz1oz0,wspd,br,ck,' // &
        'cka,cd,cda,wstar,qstar,cpm,znt'
  end subroutine write_header

end program run_mynn_surface_layer_coarse_oracle
