! WRF v4.6.1 MYNN surface-layer oracle for the over-water ISFTCFLX branches.
!
! module_sf_mynn.F:631-710 selects the over-water aerodynamic roughness and
! the thermal/moisture roughness lengths from ISFTCFLX.  Read from the source,
! not from the header comment, the selection is:
!
!   z0  (:631-662)   ISFTCFLX=0 -> charnock_1955        (COARE_OPT==3.0)
!                    ISFTCFLX=1 -> davis_etal_2008
!                    ISFTCFLX=2 -> davis_etal_2008
!                    ISFTCFLX=3 -> Taylor_Yelland_2001
!                    ISFTCFLX=4 -> charnock_1955        (COARE_OPT==3.0)
!                    otherwise  -> ZNT left as it entered
!   zt,zq (:680-710) ISFTCFLX=0 -> fairall_etal_2003    (COARE_OPT==3.0)
!                    ISFTCFLX=1 -> fairall_etal_2003    (COARE_OPT==3.0)
!                    ISFTCFLX=2 -> garratt_1992, water arm
!                    ISFTCFLX=3 -> fairall_etal_2003    (COARE_OPT==3.0)
!                    otherwise  -> z_t/z_q NEVER ASSIGNED
!
! z_t and z_q are undecorated local automatic arrays (:474), so ISFTCFLX=4 and
! any ISFTCFLX>4 read them uninitialized.  This program therefore sweeps only
! the four defined identities 0, 1, 2 and 3; the gpuwm port fails closed on the
! rest rather than inventing a value WRF does not define.
!
! COARE_OPT is a REAL PARAMETER fixed at 3.0 (:85), so edson_etal_2013 and
! fairall_etal_2014 -- the COARE 3.5 half of every branch above -- cannot be
! reached through SFCLAY1D_mynn at all.  They are still module procedures of
! the same unmodified source, so the leaf sweep below calls them directly and
! pins them at their own entry points.  That is the only oracle the unmodified
! module can give them, and it is why the gpuwm column solver keeps COARE 3.0
! wired in instead of exposing a COARE switch WRF does not have.
!
! Two files are written:
!
!   argv(1)  the column sweep -- SFCLAY1D_mynn over twelve columns for
!            ISFTCFLX = 0/1/2/3, advanced over two timesteps so the in-place
!            ZNT rewrite of every water branch persists into the next step's
!            restar/z_t/z_q.  Nine are water; one bare land and one
!            snow-covered land are negative controls that must not move with
!            ISFTCFLX at all; and one sits at XLAND exactly 1.5, which takes
!            the water roughness (:625 tests .GE. 0) but neither arm of the
!            HFX branch (:1065 tests .GT. and :1073 .LT.), so its HFX must
!            come back exactly as it went in.
!   argv(2)  the leaf sweep -- charnock_1955, edson_etal_2013, davis_etal_2008,
!            Taylor_Yelland_2001, fairall_etal_2003, fairall_etal_2014 and the
!            water arm of garratt_1992, each called directly over 32 samples
!            chosen to bind every clamp and every internal arm those routines
!            have.
!
! The narrow, widened and coarse fixtures and their byte-pinned CSVs are
! untouched; this program only ever writes the two files named on its command
! line.

program run_mynn_surface_layer_water_oracle
  use module_sf_mynn, only: mynn_sf_init_driver, SFCLAY1D_mynn, &
      charnock_1955, edson_etal_2013, davis_etal_2008, &
      Taylor_Yelland_2001, fairall_etal_2003, fairall_etal_2014, &
      garratt_1992
  implicit none

  integer, parameter :: ncase = 12
  integer, parameter :: nflx = 4
  integer, parameter :: nsample = 32
  integer, parameter :: ids = 1, ide = ncase + 1
  integer, parameter :: jds = 1, jde = 2, kds = 1, kde = 3
  integer, parameter :: ims = 1, ime = ncase
  integer, parameter :: jms = 1, jme = 1, kms = 1, kme = 2
  integer, parameter :: its = 1, ite = ncase
  integer, parameter :: jts = 1, jte = 1, kts = 1, kte = 2
  integer, parameter :: j = 1
  integer, parameter :: iz0tlnd = 0, spp_pbl = 0
  real, parameter :: cp = 1004.5, grav = 9.81, r = 287.0
  real, parameter :: rovcp = r / cp, xlv = 2.5e6
  real, parameter :: svp1 = 0.6112, svp2 = 17.67, svp3 = 29.65
  real, parameter :: svpt0 = 273.15, rv = 461.6
  real, parameter :: ep1 = rv / r - 1.0, ep2 = r / rv
  real, parameter :: karman = 0.4
  real, parameter :: dx = 3000.0

  integer, parameter :: isftcflx_sweep(nflx) = [0, 1, 2, 3]

  character(len=32) :: case_name(ncase)
  character(len=1024) :: column_path, leaf_path
  integer :: i, unit, iflx

  real :: u1d(ncase), v1d(ncase), t1d(ncase), qv1d(ncase)
  real :: p1d(ncase), dz8w1d(ncase), rho1d(ncase)
  real :: u1d2(ncase), v1d2(ncase), dz2w1d(ncase)
  real :: rstoch1d(ncase)
  real :: znt0(ncase), hfx0(ncase), qfx0(ncase), ust0(ncase)

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

  real :: s_ustar(nsample), s_wsp10(nsample), s_visc(nsample)
  real :: s_zu(nsample), s_ren(nsample), s_z0in(nsample)

  call get_command_argument(1, column_path)
  call get_command_argument(2, leaf_path)
  if (len_trim(column_path) == 0 .or. len_trim(leaf_path) == 0) then
    write(*, '(A)') 'usage: run_surface_layer_water COLUMNS.csv LEAVES.csv'
    error stop 2
  end if

  case_name = [character(len=32) :: &
      'calm_water', 'light_water', 'moderate_water', 'breezy_water', &
      'windy_water', 'gale_water', 'hurricane_water', 'cold_water', &
      'extreme_water', 'xland_exactly_1p5', 'control_land', &
      'control_snow_land']

  ! Nine water columns spanning the whole wind/ustar range the branches key
  ! on, then the XLAND=1.5 column and the two land controls.  UST enters each
  ! first step at the value in ust0: the roughness leaves read UST *before*
  ! :952 rewrites it, so the entry value is what selects davis' ZW arm and
  ! drives every restar.
  u1d  = [0.05,  3.00,  8.00, 12.00, 17.00, 22.00, 34.00,  6.00, 26.00, &
          7.00,  5.00,  4.00]
  v1d  = [0.00,  1.00,  2.00,  3.00,  4.00,  5.00,  8.00,  1.50,  6.00, &
          2.00,  1.00,  1.00]
  u1d2 = [0.10,  3.90, 10.00, 15.00, 21.00, 27.00, 41.00,  7.50, 32.00, &
          9.00,  6.50,  5.20]
  v1d2 = [0.00,  1.30,  2.60,  3.90,  5.00,  6.20, 10.00,  2.00,  7.50, &
          2.60,  1.30,  1.30]
  t1d  = [292.0, 292.0, 293.0, 291.0, 290.0, 289.0, 299.0, 268.0, 295.0, &
          294.0, 295.0, 293.0]
  qv1d = [0.0060, 0.0075, 0.0090, 0.0085, 0.0080, 0.0078, 0.0150, 0.0025, &
          0.0110, 0.0095, 0.0070, 0.0040]
  p1d  = [99000.0, 99200.0, 99400.0, 99100.0, 98900.0, 98600.0, 96500.0, &
          99700.0, 98800.0, 99150.0, 99000.0, 99300.0]
  dz8w1d = [18.0, 20.0, 24.0, 30.0, 36.0, 40.0, 50.0, 16.0, 44.0, 22.0, &
            40.0, 40.0]
  dz2w1d = [28.0, 32.0, 38.0, 46.0, 54.0, 60.0, 76.0, 26.0, 66.0, 34.0, &
            60.0, 60.0]
  znt0 = [2.0e-4, 2.0e-4, 2.0e-4, 2.0e-4, 2.0e-4, 2.0e-4, 2.0e-4, 2.0e-4, &
          2.0e-4, 2.0e-4, 0.10, 0.10]
  hfx0 = [-8.0, 20.0, 60.0, 120.0, 180.0, 240.0, 600.0, -30.0, 350.0, &
          77.0, -20.0, -12.0]
  qfx0 = [-5.0e-7, 1.0e-5, 3.0e-5, 6.0e-5, 9.0e-5, 1.2e-4, 4.0e-4, &
          -8.0e-7, 2.0e-4, 4.0e-5, -9.0e-7, -6.0e-7]
  ust0 = [0.01, 0.05, 0.30, 0.60, 1.00, 1.50, 3.50, 0.20, 9.00, 0.40, &
          0.20, 0.20]

  tsk(:, j) = [289.0, 293.0, 295.0, 292.0, 291.0, 290.0, 303.0, 271.0, &
               298.0, 297.0, 291.0, 288.0]
  psfcpa(:, j) = 100000.0
  pblh(:, j) = [200.0, 400.0, 700.0, 900.0, 1100.0, 1300.0, 2200.0, 300.0, &
                1600.0, 800.0, 250.0, 220.0]
  xland(:, j) = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.5, 1.0, 1.0]
  snowh(:, j) = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2]
  mavail(:, j) = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 0.5, 0.6]
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

  open(newunit=unit, file=trim(column_path), status='new', action='write')
  call write_column_header(unit)
  do iflx = 1, nflx
    call reset_state()
    call run_stage(unit, isftcflx_sweep(iflx), 1, 1)
    call run_stage(unit, isftcflx_sweep(iflx), 2, 1)
  end do
  close(unit)

  call fill_leaf_samples()
  open(newunit=unit, file=trim(leaf_path), status='new', action='write')
  call write_leaf_header(unit)
  call run_leaves(unit)
  close(unit)

contains

  subroutine reset_state()
    qsfc(:, j) = qv1d / (1.0 + qv1d)
    znt(:, j) = znt0
    hfx(:, j) = hfx0
    qfx(:, j) = qfx0
    lh(:, j) = qfx0 * xlv
    ust(:, j) = ust0
    ustm(:, j) = ust0
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

  subroutine run_stage(out_unit, isftcflx, itimestep, isfflx)
    integer, intent(in) :: out_unit, isftcflx, itimestep, isfflx

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
      write(out_unit, '(A,",",I0,",",I0,",",I0,59(",",ES24.16E3))') &
          trim(case_name(i)), isftcflx, itimestep, isfflx, dx, &
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

  subroutine write_column_header(out_unit)
    integer, intent(in) :: out_unit
    write(out_unit, '(A)') 'case,isftcflx,itimestep,isfflx,dx,' // &
        'xland,snowh,u1,v1,t1,qv1,p1,rho1,dz1,u2,v2,dz2,psfc,tsk,pblh,' // &
        'mavail,hfx_input,qfx_input,znt_input,qsfc_input,ust_input,' // &
        'mol_input,ustm_input,' // &
        'regime,zol,rmol,ust,ustm,mol,psim,psih,chs,chs2,cqs2,ch,flhc,' // &
        'flqc,qgh,qsfc,hfx,qfx,lh,u10,v10,th2,t2,q2,gz1oz0,wspd,br,ck,' // &
        'cka,cd,cda,wstar,qstar,cpm,znt'
  end subroutine write_column_header

  subroutine fill_leaf_samples()
    ! ustar spans below and above davis' ZW=1 knee at 1.06 and both sides of
    ! charnock's MAX(ustar,0.05) and edson's MAX(ustar,0.07) floors; wsp10
    ! spans below the 0.1 floor Taylor_Yelland re-applies, through charnock's
    ! CZC ramp (10 to 18 m/s at 10 m) and edson's MIN(19.) cap, to a speed
    ! that drives Taylor_Yelland into its 2.85e-3 ceiling; Ren spans
    ! fairall_etal_2003's Ren<=2 test, both arms of fairall_etal_2014's
    ! MIN(1.6e-4, ...) and enough range to bind every clamp in all three
    ! zt/zq leaves; zu moves charnock's and edson's log-law 10 m reduction
    ! off the identity it takes at zu=10.
    s_ustar = [0.0001, 0.001, 0.005, 0.01, 0.02, 0.05, 0.06, 0.07, &
               0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, &
               0.60, 0.70, 0.80, 0.90, 1.00, 1.05, 1.06, 1.07, &
               1.20, 1.50, 2.00, 2.50, 3.00, 3.50, 5.00, 9.00]
    s_wsp10 = [0.05, 0.10, 1.00, 2.00, 3.00, 5.00, 6.00, 7.00, &
               8.00, 9.00, 9.90, 10.00, 10.10, 12.00, 14.00, 16.00, &
               17.90, 18.00, 18.10, 19.00, 19.10, 20.00, 22.00, 24.00, &
               26.00, 26.30, 28.00, 30.00, 35.00, 40.00, 45.00, 60.00]
    s_visc = [1.326e-5, 1.30e-5, 1.35e-5, 1.40e-5, 1.45e-5, 1.50e-5, &
              1.20e-5, 1.25e-5, 1.30e-5, 1.35e-5, 1.40e-5, 1.45e-5, &
              1.50e-5, 1.326e-5, 1.30e-5, 1.35e-5, 1.40e-5, 1.45e-5, &
              1.50e-5, 1.20e-5, 1.25e-5, 1.30e-5, 1.35e-5, 1.40e-5, &
              1.45e-5, 1.50e-5, 1.326e-5, 1.30e-5, 1.35e-5, 1.40e-5, &
              1.45e-5, 1.50e-5]
    s_zu = [9.0, 10.0, 12.0, 15.0, 20.0, 8.0, 5.0, 4.0, &
            3.0, 2.0, 1.5, 25.0, 30.0, 9.0, 10.0, 12.0, &
            15.0, 20.0, 8.0, 5.0, 4.0, 3.0, 2.0, 1.5, &
            25.0, 30.0, 9.0, 10.0, 12.0, 15.0, 20.0, 30.0]
    s_ren = [0.10, 0.20, 0.50, 1.00, 1.50, 1.99, 2.00, 2.0000005, &
             2.10, 3.00, 5.00, 8.00, 12.0, 20.0, 35.0, 50.0, &
             80.0, 120.0, 200.0, 300.0, 407.0, 600.0, 900.0, 1200.0, &
             1806.0, 2500.0, 5000.0, 1.0e4, 1.0e5, 1.0e6, 1.0e7, 1.0e8]
    s_z0in = [1.27e-7, 1.59e-5, 3.00e-5, 5.27e-5, 1.00e-4, 2.00e-4, &
              5.00e-4, 1.00e-3, 2.85e-3, 1.27e-7, 1.59e-5, 3.00e-5, &
              5.27e-5, 1.00e-4, 2.00e-4, 5.00e-4, 1.00e-3, 2.85e-3, &
              1.27e-7, 1.59e-5, 3.00e-5, 5.27e-5, 1.00e-4, 2.00e-4, &
              5.00e-4, 1.00e-3, 2.85e-3, 1.27e-7, 1.59e-5, 2.85e-3, &
              1.00e-4, 2.85e-3]
  end subroutine fill_leaf_samples

  subroutine write_leaf_header(out_unit)
    integer, intent(in) :: out_unit
    write(out_unit, '(A)') 'leaf,sample,ustar,wsp10,visc,zu,ren,z0_in,' // &
        'landsea,z0_out,zt_out,zq_out'
  end subroutine write_leaf_header

  subroutine write_leaf_row(out_unit, leaf, sample, landsea, z0_out, &
                            zt_out, zq_out)
    integer, intent(in) :: out_unit, sample
    character(len=*), intent(in) :: leaf
    real, intent(in) :: landsea, z0_out, zt_out, zq_out
    write(out_unit, '(A,",",I0,10(",",ES24.16E3))') &
        trim(leaf), sample, s_ustar(sample), s_wsp10(sample), &
        s_visc(sample), s_zu(sample), s_ren(sample), s_z0in(sample), &
        landsea, z0_out, zt_out, zq_out
  end subroutine write_leaf_row

  subroutine run_leaves(out_unit)
    integer, intent(in) :: out_unit
    integer :: k
    real :: z0_out, zt_out, zq_out
    ! Outputs a leaf does not produce are written as exactly 0.0 and are not
    ! compared: charnock/edson/davis/Taylor_Yelland are z0-only, and the
    ! zt/zq leaves do not touch z0.
    do k = 1, nsample
      call charnock_1955(z0_out, s_ustar(k), s_wsp10(k), s_visc(k), s_zu(k))
      call write_leaf_row(out_unit, 'charnock_1955', k, 2.0, z0_out, 0.0, 0.0)

      call edson_etal_2013(z0_out, s_ustar(k), s_wsp10(k), s_visc(k), &
                           s_zu(k))
      call write_leaf_row(out_unit, 'edson_etal_2013', k, 2.0, z0_out, &
                          0.0, 0.0)

      call davis_etal_2008(z0_out, s_ustar(k))
      call write_leaf_row(out_unit, 'davis_etal_2008', k, 2.0, z0_out, &
                          0.0, 0.0)

      call Taylor_Yelland_2001(z0_out, s_ustar(k), s_wsp10(k))
      call write_leaf_row(out_unit, 'taylor_yelland_2001', k, 2.0, z0_out, &
                          0.0, 0.0)

      call fairall_etal_2003(zt_out, zq_out, s_ren(k), s_ustar(k), &
                             s_visc(k), 0.0, spp_pbl)
      call write_leaf_row(out_unit, 'fairall_etal_2003', k, 2.0, 0.0, &
                          zt_out, zq_out)

      call fairall_etal_2014(zt_out, zq_out, s_ren(k), s_ustar(k), &
                             s_visc(k), 0.0, spp_pbl)
      call write_leaf_row(out_unit, 'fairall_etal_2014', k, 2.0, 0.0, &
                          zt_out, zq_out)

      ! garratt_1992 branches internally on landsea-1.5 .GT. 0, and its
      ! ISFTCFLX=2 caller passes XLAND straight through, so a column at
      ! XLAND exactly 1.5 takes the water ROUGHNESS selection and the LAND
      ! arm of this leaf.  Both arms are swept for that reason.
      call garratt_1992(zt_out, zq_out, s_z0in(k), s_ren(k), 2.0)
      call write_leaf_row(out_unit, 'garratt_1992', k, 2.0, 0.0, zt_out, &
                          zq_out)

      call garratt_1992(zt_out, zq_out, s_z0in(k), s_ren(k), 1.5)
      call write_leaf_row(out_unit, 'garratt_1992', k, 1.5, 0.0, zt_out, &
                          zq_out)

      call garratt_1992(zt_out, zq_out, s_z0in(k), s_ren(k), 1.0)
      call write_leaf_row(out_unit, 'garratt_1992', k, 1.0, 0.0, zt_out, &
                          zq_out)
    end do
  end subroutine run_leaves

end program run_mynn_surface_layer_water_oracle
