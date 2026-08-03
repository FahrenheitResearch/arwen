! Aerosol-aware (mp_physics=28) companion to run_column.F90.
!
! This program is deliberately SEPARATE from run_column.F90.  WRF's
! is_aerosol_aware flag is a module-SAVEd LOGICAL set purely by optional
! argument presence at module_mp_thompson.F:480, so adding the aerosol
! optionals to the classic program would flip nc1d/nwfa1d/nifa1d for every
! classic scenario (mp_gt_driver:1236-1256) and rewrite all 92 committed
! mp=8 fixtures.  Those fixtures are the evidence behind ArWen's
! model-validated mp=8 port and must not move.
!
! The environment skeleton (nx=2, ny=2, nz=24, z(k)=(k-0.5)*500,
! p=p0*exp(-z/8000)) is copied from run_column.F90 so the two families are
! directly comparable.  Every seeded aerosol/droplet number is expressed as
! a per-cubic-metre target and divided by the local density estimate,
! because WRF's state arrays carry per-kilogram values and mp_thompson
! multiplies by rho on entry (module_mp_thompson.F:1805-1830).
!
! IMPORTANT ORDERING: the "before" snapshot is taken AFTER thompson_init,
! not before.  thompson_init overwrites nwfa/nifa/nwfa2d in place
! (module_mp_thompson.F:493-522, 536-551) whenever
! MAXVAL(nwfa(its:ite-1,:,jts:jte-1)) < eps, and with its=1,ite=2 that
! MAXVAL scans exactly the column this program dumps.  Recording the
! pre-init zeros would produce a silently wrong input state.

program run_thompson_column_aero
  use module_mp_thompson, only: thompson_init, mp_gt_driver, RSLF, RSIF
  implicit none

  integer, parameter :: nx = 2, ny = 2, nz = 24
  integer :: i, j, k, scenario_id, ios, free_unit
  logical :: unit_opened
  ! WRF'S OWN R/cp, NOT AN APPROXIMATION OF IT.
  ! share/module_model_constants.F:19 gives r_d = 287., :20 gives
  ! cp = 7.*r_d/2. (= 1004.5, NOT 1004.) and :31 gives rcp = r_d/cp; the
  ! Exner function this program builds is WRF's own
  ! dyn_em/module_big_step_utilities_em.F:4854
  !     pi_phy(i,k,j) = (p_phy(i,k,j)/p1000mb)**rcp
  ! with p1000mb = 100000. (module_model_constants.F:36).  Spelling the
  ! divide rather than its answer keeps this DERIVED from WRF's two
  ! parameters.  In float32 r_d/cp == 2./7. == 0x3E924925 bitwise, which is
  ! also gpuwm/core/constants.py:32 RCP; the 287.0/1004.0 this file used
  ! before is 0x3E925BCB, 4774 ulps away, and it made the fixtures'
  ! (p, theta) pair impossible to invert exactly on the ArWen side.
  real, parameter :: p0 = 100000.0
  real, parameter :: r_d_wrf = 287.0, cp_wrf = 7.0 * r_d_wrf / 2.0
  real, parameter :: rd_over_cp = r_d_wrf / cp_wrf
  real, parameter :: r_dry = 287.04
  real, parameter :: re_qc_bg = 2.49e-6
  real, parameter :: re_qi_bg = 4.99e-6
  real, parameter :: re_qs_bg = 9.99e-6
  real :: z(nz), temperature(nz), dt
  real :: hgt(nx,nz,ny)
  real :: qv(nx,nz,ny), qc(nx,nz,ny), qr(nx,nz,ny)
  real :: qi(nx,nz,ny), qs(nx,nz,ny), qg(nx,nz,ny)
  real :: ni(nx,nz,ny), nr(nx,nz,ny), th(nx,nz,ny)
  real :: nc(nx,nz,ny), nwfa(nx,nz,ny), nifa(nx,nz,ny)
  real :: nwfa2d(nx,ny), nifa2d(nx,ny)
  ! Black-carbon aerosol is out of scope (wif_input_opt=0) but the arrays
  ! are NOT optional in practice: mp_gt_driver writes nbca(i,k,j)=0.0 with
  ! no PRESENT() guard at module_mp_thompson.F:1337, and WRF's own
  ! module_microphysics_driver.F CASE(THOMPSONAERO) always supplies them.
  ! Omitting them segfaults.  They stay identically zero throughout.
  real :: nbca(nx,nz,ny), nbca2d(nx,ny)
  real :: pii(nx,nz,ny), p(nx,nz,ny), w(nx,nz,ny), dz(nx,nz,ny)
  real :: refl(nx,nz,ny), re_cloud(nx,nz,ny)
  real :: re_ice(nx,nz,ny), re_snow(nx,nz,ny)
  real :: rainnc(nx,ny), rainncv(nx,ny), snownc(nx,ny)
  real :: snowncv(nx,ny), graupelnc(nx,ny), graupelncv(nx,ny)
  real :: sr(nx,ny)
  real :: qv_before(nz), qc_before(nz), qr_before(nz)
  real :: qi_before(nz), qs_before(nz), qg_before(nz)
  real :: ni_before(nz), nr_before(nz), th_before(nz)
  real :: nc_before(nz), nwfa_before(nz), nifa_before(nz)
  real :: temp_before(nz), temp_after(nz)
  real :: nwfa2d_before, nifa2d_before
  character(len=48) :: scenario
  character(len=512) :: output_dir, column_path, surface_path

  call get_command_argument(1, scenario)
  call get_command_argument(2, output_dir)
  if (len_trim(scenario) == 0 .or. len_trim(output_dir) == 0) then
     error stop 'usage: run_column_aero SCENARIO OUTPUT_DIRECTORY'
  endif

  ! Scenario identifiers start at 101 so the aerosol-aware fixture space
  ! never collides with the classic 1..46 space.
  select case (trim(scenario))
  case ('aero-init-profile')
     scenario_id = 101
  case ('aero-sfc-emit')
     scenario_id = 102
  case ('aero-ccn-activate')
     scenario_id = 103
  case ('aero-ccn-sweep')
     scenario_id = 104
  case ('aero-drop-evap')
     scenario_id = 105
  case ('aero-nc-auto')
     scenario_id = 106
  case ('aero-nc-accrete')
     scenario_id = 107
  case ('aero-nc-effrad')
     scenario_id = 108
  case ('aero-nc-sed')
     scenario_id = 109
  case ('aero-scav-rain')
     scenario_id = 110
  case ('aero-scav-frozen')
     scenario_id = 111
  case ('aero-ice-demott-dep')
     scenario_id = 112
  case ('aero-ice-demott-idxin')
     scenario_id = 113
  case ('aero-ice-koop')
     scenario_id = 114
  case ('aero-cloud-freeze-nc')
     scenario_id = 115
  case ('aero-nc-cap')
     scenario_id = 116
  case ('aero-warm-overlap')
     scenario_id = 117
  case ('aero-cold-overlap')
     scenario_id = 118
  case ('aero-reduces-to-classic')
     scenario_id = 119
  ! WP-08 SCRATCH SCENARIOS, PROMOTED TO COMMITTED FIXTURES.
  ! tests/test_thompson_aerosol_sed_gpu.py's SED_NU_SWEEP, CLEAN_MELT and
  ! CLEAN_FREEZE tables were captured from scratch scenarios that lived only
  ! in a private copy of this file, which left three of its five intermediate
  ! tables with no committed producer (gpuwm/data/thompson/PROVENANCE.md,
  ! "Still ungated by a committed producer").  They are ordinary members of
  ! this driver now, run through the same unmodified physics, so
  ! build_aero_instrumented.sh + check_instrumented_tables_aero.py can
  ! regenerate all five.
  case ('wp08-nusweep')
     scenario_id = 120
  case ('wp08-melt')
     scenario_id = 121
  case ('wp08-freeze')
     scenario_id = 122
  case default
     error stop 'unknown aerosol-aware Thompson column scenario'
  end select

  ! table_ccnAct (module_mp_thompson.F:5123-5133) selects the lowest unit
  ! in 20..99 that is not already OPEN.  build_aero.sh relies on that being
  ! unit 20 so it can request big-endian conversion for CCN_ACTIVATE.BIN
  ! alone via GFORTRAN_CONVERT_UNIT, leaving the native-endian classic
  ! caches on unit 63 untouched.  Assert it rather than assume it.
  free_unit = -1
  do i = 20, 99
     inquire(unit=i, opened=unit_opened)
     if (.not. unit_opened) then
        free_unit = i
        exit
     endif
  enddo
  if (free_unit /= 20) then
     print '(A,I0)', 'FATAL: lowest free Fortran unit is not 20 but ', free_unit
     error stop 'CCN_ACTIVATE.BIN endian conversion assumption violated'
  endif

  if (scenario_id == 117 .or. scenario_id == 118) then
     ! Overlap columns: a longer step and a source-isolation geometry so
     ! the shared ncten/nwfaten/nifaten reconciliation dominates transport.
     dt = 50.0
  else
     dt = 10.0
  endif

  do k = 1, nz
     z(k) = (real(k) - 0.5) * 500.0
     select case (scenario_id)
     case (111)
        temperature(k) = 260.0
     case (112, 113, 115, 118)
        temperature(k) = 240.0
     case (114, 122)
        temperature(k) = 230.0
     case (121)
        ! Above freezing: the phase cleanup's MELT branch (:3947-3953).
        temperature(k) = 280.0
     case (119)
        temperature(k) = max(205.0, 292.0 - 0.0065 * z(k))
     case default
        temperature(k) = 285.0
     end select
  enddo

  qv = 0.0
  qc = 0.0
  qr = 0.0
  qi = 0.0
  qs = 0.0
  qg = 0.0
  ni = 0.0
  nr = 0.0
  nc = 0.0
  nwfa = 0.0
  nifa = 0.0
  nwfa2d = 0.0
  nifa2d = 0.0
  nbca = 0.0
  nbca2d = 0.0
  th = 0.0
  pii = 0.0
  p = 0.0
  w = 0.0
  dz = 500.0
  if (scenario_id == 117 .or. scenario_id == 118) dz = 1.0e8
  ! wp08-nusweep: 20 m layers put all 24 levels inside WRF's 500 m cloud
  ! fallout-depth search (module_mp_thompson.F:3800-3823), so every level
  ! participates in the fallout the nu_c sweep is measuring.
  if (scenario_id == 120) dz = 20.0
  refl = -35.0
  re_cloud = re_qc_bg
  re_ice = re_qi_bg
  re_snow = re_qs_bg
  rainnc = 0.0
  rainncv = 0.0
  snownc = 0.0
  snowncv = 0.0
  graupelnc = 0.0
  graupelncv = 0.0
  sr = 0.0

  do j = 1, ny
     do i = 1, nx
        do k = 1, nz
           hgt(i,k,j) = z(k)
           p(i,k,j) = p0 * exp(-z(k) / 8000.0)
           pii(i,k,j) = (p(i,k,j) / p0) ** rd_over_cp
           th(i,k,j) = temperature(k) / pii(i,k,j)
           select case (scenario_id)
           case (101, 103, 104)
              ! Two-tenths-percent liquid supersaturation drives the
              ! condensation branch, which is the only caller of
              ! activ_ncloud (module_mp_thompson.F:3416).
              qv(i,k,j) = 1.002 * RSLF(p(i,k,j), temperature(k))
           case (102)
              ! Deeply subsaturated and hydrometeor free: nothing but the
              ! surface aerosol emission may move.
              qv(i,k,j) = 0.80 * RSLF(p(i,k,j), temperature(k))
           case (105)
              ! Two-percent subsaturation with cloud present selects the
              ! aerosol-only droplet-evaporation branch at 3423-3471.
              qv(i,k,j) = 0.98 * RSLF(p(i,k,j), temperature(k))
           case (106, 107, 108, 109, 110, 116, 117, 120, 121)
              qv(i,k,j) = RSLF(p(i,k,j), temperature(k))
           case (111, 115, 122)
              qv(i,k,j) = RSIF(p(i,k,j), temperature(k))
           case (112, 113)
              ! Thirty-percent ice supersaturation at 240 K activates
              ! deposition nucleation, which under mp=28 is iceDeMott
              ! rather than Cooper.
              qv(i,k,j) = 1.30 * RSIF(p(i,k,j), temperature(k))
           case (114)
              ! iceKoop is a very steep function of liquid saturation.  At
              ! 230 K, RSIF/RSLF = 0.65313, so this multiplier puts
              ! satw = qv/qvs at 0.97507 - inside the productive band
              ! (below 0.96 the freezing probability underflows to zero,
              ! above about 0.99 it saturates the 1000 per litre cap) -
              ! while keeping ssati = 0.493, above the 0.4 Koop gate and
              ! below liquid saturation so no cloud is condensed.
              qv(i,k,j) = 1.493 * RSIF(p(i,k,j), temperature(k))
           case (118)
              if (z(k) >= 1250.0 .and. z(k) <= 4250.0) then
                 qv(i,k,j) = 1.30 * RSIF(p(i,k,j), temperature(k))
              else
                 qv(i,k,j) = RSIF(p(i,k,j), temperature(k))
              endif
           case (119)
              qv(i,k,j) = max(1.0e-5, 0.014 * exp(-z(k) / 2500.0))
           case default
              error stop 'invalid aerosol scenario id in vapor setup'
           end select
        enddo
     enddo
  enddo

  call initialize_case(scenario_id, z, p, temperature, qv, qc, qr, qi,  &
                       qs, qg, ni, nr, nc, nwfa, nifa, nwfa2d, nifa2d, w)

  ! thompson_init is aerosol-aware purely because nwfa2d, nwfa and nifa
  ! are PRESENT.  wif_input_opt must always be supplied: it is
  ! dereferenced without PRESENT() at module_mp_thompson.F:561.  nbca and
  ! nbca2d are supplied to mirror WRF's own THOMPSONAERO init call
  ! (module_physics_init.F:4526-4538); with wif_input_opt=0 they are only
  ! read inside a guarded block and are never written.  orho, frc_urb2d,
  ! dx, dy and is_start are declared but never referenced in the v4.6.1
  ! init body, so omitting them is exact, not an approximation.
  call thompson_init(                                                 &
       hgt=hgt,                                                       &
       nwfa2d=nwfa2d, nbca2d=nbca2d,                                  &
       nwfa=nwfa, nifa=nifa, nbca=nbca,                               &
       wif_input_opt=0,                                               &
       ids=1, ide=2, jds=1, jde=2, kds=1, kde=nz,                   &
       ims=1, ime=nx, jms=1, jme=ny, kms=1, kme=nz,                 &
       its=1, ite=nx, jts=1, jte=ny, kts=1, kte=nz)

  ! Snapshot AFTER init: thompson_init may have written nwfa/nifa/nwfa2d.
  qv_before = qv(1,:,1)
  qc_before = qc(1,:,1)
  qr_before = qr(1,:,1)
  qi_before = qi(1,:,1)
  qs_before = qs(1,:,1)
  qg_before = qg(1,:,1)
  ni_before = ni(1,:,1)
  nr_before = nr(1,:,1)
  nc_before = nc(1,:,1)
  nwfa_before = nwfa(1,:,1)
  nifa_before = nifa(1,:,1)
  th_before = th(1,:,1)
  nwfa2d_before = nwfa2d(1,1)
  nifa2d_before = nifa2d(1,1)

  ! THE TEMPERATURE COLUMN IS CARRIED, NOT RECONSTRUCTED INSIDE write_row.
  ! mp_gt_driver's working temperature is the LOCAL array t1d (declared at
  ! module_mp_thompson.F:1117), filled at :1222 with
  !     t1d(k) = th(i,k,j)*pii(i,k,j)
  ! and consumed by mp_thompson (:1290), calc_refl10cm (:1459) and
  ! calc_effectRad (:1472).  The entry value is therefore EXACTLY the
  ! product below -- verified bitwise at all 24 levels of all 19 scenarios
  ! against an additive t1d dump inside mp_gt_driver (see
  ! instrument_exit_temperature_aero.py / build_exit_temperature_aero.sh /
  ! check_exit_temperature_aero.py; measured 528 of 528 rows bitwise).
  do k = 1, nz
     temp_before(k) = th_before(k) * pii(1,k,1)
  enddo

  ! Argument list mirrors module_microphysics_driver.F CASE(THOMPSONAERO):
  ! nc/nwfa/nifa/nbca/nwfa2d/nifa2d/nbca2d present, qb/ng absent (mp=28 is
  ! not hail-aware).  Both aer_init_opt and wif_input_opt are dereferenced
  ! without PRESENT() at mp_gt_driver:1241/1322/1334 and
  ! mp_thompson:1804/1807/2968/3978/3983; nifa2d without PRESENT() at
  ! mp_gt_driver:1321; nbca without PRESENT() at mp_gt_driver:1337.
  call mp_gt_driver(                                                  &
       qv=qv, qc=qc, qr=qr, qi=qi, qs=qs, qg=qg, ni=ni, nr=nr,      &
       nc=nc, nwfa=nwfa, nifa=nifa, nbca=nbca,                      &
       nwfa2d=nwfa2d, nifa2d=nifa2d, nbca2d=nbca2d,                 &
       aer_init_opt=0, wif_input_opt=0,                              &
       th=th, pii=pii, p=p, w=w, dz=dz,                             &
       dt_in=dt, itimestep=1,                                        &
       RAINNC=rainnc, RAINNCV=rainncv,                              &
       SNOWNC=snownc, SNOWNCV=snowncv,                              &
       GRAUPELNC=graupelnc, GRAUPELNCV=graupelncv, SR=sr,           &
       refl_10cm=refl, diagflag=.true., ke_diag=nz, do_radar_ref=1, &
       re_cloud=re_cloud, re_ice=re_ice, re_snow=re_snow,           &
       has_reqc=1, has_reqi=1, has_reqs=1,                          &
       ids=1, ide=2, jds=1, jde=2, kds=1, kde=nz,                   &
       ims=1, ime=nx, jms=1, jme=ny, kms=1, kme=nz,                 &
       its=1, ite=nx, jts=1, jte=ny, kts=1, kte=nz)

  ! THE EXIT TEMPERATURE IS THE ONE MEASURED LIMIT OF THIS HARNESS.
  ! mp_gt_driver returns nothing but th: :1358 writes th(i,k,j) =
  ! t1d(k)/pii(i,k,j) and t1d dies with the routine.  A caller of PRISTINE
  ! WRF therefore cannot read the exit t1d, and multiplying th back by pii
  ! is a float32 round trip that is NOT lossless: measured on the committed
  ! set, 37 of 456 after-rows (8.1%, spread over 6 of the 19 scenarios)
  ! come back one ulp away from the t1d calc_effectRad and calc_refl10cm
  ! actually saw.  Inverting :1358 is not possible either -- for 158 of 912
  ! rows TWO float32 values of t1d map to the same th.
  !
  ! The exact exit temperature is therefore published separately, in
  ! gpuwm/data/thompson/oracle-aero/aero-exit-temperature.csv, produced by
  ! an ADDITIVE t1d dump whose neutrality is proven by
  ! check_exit_temperature_aero.py: with the dump compiled in, all 44
  ! pristine fixture files reproduce byte for byte.  The column below stays
  ! the round trip so that the fixtures themselves remain the product of an
  ! unmodified module_mp_thompson.F.
  do k = 1, nz
     temp_after(k) = th(1,k,1) * pii(1,k,1)
  enddo

  column_path = trim(output_dir) // '/' // trim(scenario) // '-column.csv'
  surface_path = trim(output_dir) // '/' // trim(scenario) // '-surface.csv'
  open(newunit=ios, file=trim(column_path), status='replace', action='write')
  write(ios,'(A)') 'phase,k,z_m,p_pa,pii,w_m_s,dz_m,theta_k,temp_k,qv,qc,qr,qi,qs,qg,ni_per_kg,nr_per_kg,nc_per_kg,nwfa_per_kg,nifa_per_kg,effc_m,effi_m,effs_m,refl_dbz'
  do k = 1, nz
     call write_row(ios, 'before', k, z(k), p(1,k,1), pii(1,k,1),      &
          w(1,k,1), dz(1,k,1), th_before(k), temp_before(k),          &
          qv_before(k),                                               &
          qc_before(k), qr_before(k), qi_before(k), qs_before(k),     &
          qg_before(k), ni_before(k), nr_before(k), nc_before(k),     &
          nwfa_before(k), nifa_before(k), re_qc_bg,                   &
          re_qi_bg, re_qs_bg, -35.0)
  enddo
  do k = 1, nz
     call write_row(ios, 'after', k, z(k), p(1,k,1), pii(1,k,1),       &
          w(1,k,1), dz(1,k,1), th(1,k,1), temp_after(k),             &
          qv(1,k,1), qc(1,k,1),                                      &
          qr(1,k,1), qi(1,k,1), qs(1,k,1), qg(1,k,1), ni(1,k,1),     &
          nr(1,k,1), nc(1,k,1), nwfa(1,k,1), nifa(1,k,1),            &
          re_cloud(1,k,1), re_ice(1,k,1),                            &
          re_snow(1,k,1), refl(1,k,1))
  enddo
  close(ios)

  open(newunit=ios, file=trim(surface_path), status='replace', action='write')
  write(ios,'(A)') 'scenario,dt_s,rainnc_mm,rainncv_mm,snownc_mm,snowncv_mm,graupelnc_mm,graupelncv_mm,sr,nwfa2d_kg_s,nifa2d_kg_s'
  write(ios,'(A,10(",",ES24.16E3))') trim(scenario), dt, rainnc(1,1), &
       rainncv(1,1), snownc(1,1), snowncv(1,1), graupelnc(1,1),       &
       graupelncv(1,1), sr(1,1), nwfa2d_before, nifa2d_before
  close(ios)

  print '(A,1X,A)', 'THOMPSON_AERO_COLUMN_ORACLE_COMPLETE', trim(scenario)

contains

  ! Per-kilogram seed for a per-cubic-metre target, using the same density
  ! definition mp_thompson uses at line 1802.
  real function per_kg(target_per_m3, pressure, temp, vapor)
    real, intent(in) :: target_per_m3, pressure, temp, vapor
    real :: rho_local
    rho_local = 0.622 * pressure / (r_dry * temp * (vapor + 0.622))
    per_kg = target_per_m3 / rho_local
  end function per_kg

  subroutine initialize_case(which, height, pres3, temp1, qv3, qc3, qr3, &
                             qi3, qs3, qg3, ni3, nr3, nc3, nwfa3, nifa3, &
                             nwfa2, nifa2, w3)
    integer, intent(in) :: which
    real, intent(in) :: height(nz), temp1(nz)
    real, intent(in) :: pres3(nx,nz,ny), qv3(nx,nz,ny)
    real, intent(inout) :: qc3(nx,nz,ny)
    real, intent(inout) :: qr3(nx,nz,ny), qi3(nx,nz,ny)
    real, intent(inout) :: qs3(nx,nz,ny), qg3(nx,nz,ny)
    real, intent(inout) :: ni3(nx,nz,ny), nr3(nx,nz,ny)
    real, intent(inout) :: nc3(nx,nz,ny), nwfa3(nx,nz,ny), nifa3(nx,nz,ny)
    real, intent(inout) :: nwfa2(nx,ny), nifa2(nx,ny)
    real, intent(inout) :: w3(nx,nz,ny)
    real :: cloud, rain, ice, snow, graupel, velocity
    real :: ccn_m3, in_m3, ncloud_m3
    integer :: ii, jj, kk, slot
    ! Sweep ladders.  ta_Na spans 10..10000 cm-3 and ta_Ww spans 0.01..100
    ! m/s, so the outermost entries here sit past both clamp ends of
    ! activ_ncloud's bracket search (module_mp_thompson.F:5194-5215).
    real, parameter :: w_ladder(10) = (/0.005, 0.02, 0.05, 0.2, 0.5,    &
                                        2.0, 5.0, 20.0, 50.0, 200.0/)
    real, parameter :: ccn_ladder(5) = (/5.0e6, 3.0e7, 3.0e8, 3.0e9,    &
                                         2.0e10/)
    real, parameter :: nc_ladder(5) = (/30.0e6, 100.0e6, 300.0e6,       &
                                        1000.0e6, 1800.0e6/)
    real, parameter :: nc_wide(6) = (/2.0, 50.0, 5.0e6, 1.0e8, 5.0e9,   &
                                      2.0e10/)
    real, parameter :: in_ladder(4) = (/5.0e3, 5.0e5, 5.0e7, 5.0e9/)
    real, parameter :: koop_ladder(5) = (/1.11e7, 1.0e8, 1.0e9, 3.0e9,  &
                                          9.999e9/)
    ! wp08-nusweep's droplet ladder: 1000e6/n for n = 1..13, written to six
    ! significant figures so every rung is a literal rather than a float32
    ! quotient.  mp_thompson:3218's nu_c = MIN(15, NINT(1000.E6/nc)+2) then
    ! takes the values 3..15 down the column -- every reachable one.  These
    ! are the exact rungs tests/test_thompson_aerosol_sed_gpu.py's
    ! SED_NU_SWEEP was captured on, recovered from that table's own nc1d and
    ! rho columns and verified bitwise by check_instrumented_tables_aero.py.
    real, parameter :: nu_c_ladder(13) = (/1000.0e6, 500.0e6, 333.33e6, &
                                           250.0e6, 200.0e6, 166.667e6, &
                                           142.857e6, 125.0e6,          &
                                           111.111e6, 100.0e6,          &
                                           90.909e6, 83.333e6,          &
                                           76.923e6/)

    do jj = 1, ny
       do ii = 1, nx
          do kk = 1, nz
             cloud = 0.0
             rain = 0.0
             ice = 0.0
             snow = 0.0
             graupel = 0.0
             velocity = 1.0
             ccn_m3 = 3.0e8
             in_m3 = 1.0e6
             ncloud_m3 = 0.0

             select case (which)
             case (101)
                ! Everything aerosol is zero so thompson_init must build
                ! its synthetic CCN/IN profile and derive nwfa2d.
                ccn_m3 = 0.0
                in_m3 = 0.0
                ncloud_m3 = 0.0
             case (102)
                ! Seeded aerosol suppresses the profile fill; the only
                ! aerosol change permitted is the surface emission, which
                ! WRF applies at kts only and deliberately does not clamp.
                ncloud_m3 = 1.0e8
             case (103)
                ! Near-empty droplet population plus abundant CCN: the
                ! activation source dominates the droplet budget.
                ncloud_m3 = 2.0
             case (104)
                slot = mod(kk - 1, 10) + 1
                velocity = w_ladder(slot)
                slot = min(5, (kk - 1) / 5 + 1)
                ccn_m3 = ccn_ladder(slot)
                ncloud_m3 = 2.0
             case (105)
                if (height(kk) <= 4250.0) cloud = 3.0e-4
                ncloud_m3 = 1.0e8
             case (106)
                if (height(kk) <= 4250.0) then
                   cloud = 1.0e-3 * exp(-((height(kk)-1250.0)/1100.0)**2)
                endif
                ncloud_m3 = nc_ladder(mod(kk - 1, 5) + 1)
             case (107)
                if (height(kk) <= 4250.0) then
                   cloud = 6.0e-6 * exp(-((height(kk)-1250.0)/1100.0)**2)
                   rain = 3.0e-4 * exp(-((height(kk)-1000.0)/1100.0)**2)
                endif
                ncloud_m3 = nc_ladder(mod(kk - 1, 5) + 1)
             case (108)
                slot = mod(kk - 1, 6) + 1
                ncloud_m3 = nc_wide(slot)
                if (slot <= 2) then
                   ! Barely-above-R1 cloud keeps WRF's terminal droplet
                   ! rediagnosis below 100 m-3, the only way through
                   ! mp_gt_driver to reach calc_effectRad's nc<100 branch.
                   cloud = 2.0e-12
                else
                   cloud = 3.0e-4
                endif
             case (109)
                if (kk <= 3) cloud = 5.0e-4
                ! Cloud sedimentation is gated on w1d(k) < 0.1 m/s.
                velocity = 0.0
                ncloud_m3 = 1.0e8
             case (110)
                if (height(kk) <= 4250.0) then
                   rain = 3.0e-4 * exp(-((height(kk)-1000.0)/1100.0)**2)
                endif
                ccn_m3 = 3.0e9
                in_m3 = 5.0e9
                ncloud_m3 = 0.0
             case (111)
                if (height(kk) <= 4250.0) then
                   snow = 2.0e-4 * exp(-((height(kk)-1200.0)/1200.0)**2)
                   graupel = 2.0e-4 * exp(-((height(kk)-1200.0)/1200.0)**2)
                endif
                ccn_m3 = 3.0e9
                in_m3 = 5.0e9
                ncloud_m3 = 1.0e8
             case (112)
                ncloud_m3 = 0.0
             case (113)
                ! idx_IN indexes the freezeH2O tables (tpi_qcfz, tni_qcfz,
                ! tpi_qrfz, tpg_qrfz, tni_qrfz, tnr_qrfz), which are only
                ! read when supercooled cloud or rain is present.  Without
                ! condensate an IN ladder changes iceDeMott but never
                ! reaches the table axis, so this scenario carries both.
                if (height(kk) <= 4250.0) then
                   cloud = 3.0e-4 * exp(-((height(kk)-1000.0)/1100.0)**2)
                   rain = 3.0e-4 * exp(-((height(kk)-1000.0)/1100.0)**2)
                endif
                in_m3 = in_ladder(mod(kk - 1, 4) + 1)
                ncloud_m3 = 1.0e8
             case (114)
                ! Water-friendly aerosol drives iceKoop; the IN floor keeps
                ! iceDeMott small so the homogeneous term is visible.  The
                ! CCN ladder is what makes the fixture self-evidencing:
                ! every level shares one temperature and one saturation, so
                ! any level-to-level variation in the ice produced is
                ! attributable to iceKoop alone.
                ccn_m3 = koop_ladder(mod(kk - 1, 5) + 1)
                in_m3 = 5.0e3
                ncloud_m3 = 0.0
             case (115)
                if (height(kk) <= 4250.0) then
                   cloud = 3.0e-4 * exp(-((height(kk)-1000.0)/1100.0)**2)
                endif
                ncloud_m3 = nc_ladder(mod(kk - 1, 5) + 1)
             case (116)
                if (height(kk) <= 4250.0) cloud = 3.0e-4
                if (mod(kk, 2) == 1) then
                   ncloud_m3 = 5.0e9
                else
                   ncloud_m3 = 1.0
                endif
             case (117)
                if (height(kk) <= 4250.0) then
                   cloud = 1.0e-3 * exp(-((height(kk)-1250.0)/1100.0)**2)
                   rain = 3.0e-4 * exp(-((height(kk)-1000.0)/1100.0)**2)
                endif
                ccn_m3 = 3.0e9
                ncloud_m3 = 1.0e8
             case (118)
                if (height(kk) >= 1250.0 .and. height(kk) <= 4250.0) then
                   cloud = 5.0e-4 * exp(-((height(kk)-1200.0)/1200.0)**2)
                   rain = 3.0e-4 * exp(-((height(kk)-1200.0)/1200.0)**2)
                   ice = 1.0e-8 * exp(-((height(kk)-1000.0)/1100.0)**2)
                   snow = 6.0e-3 * exp(-((height(kk)-1200.0)/1200.0)**2)
                   graupel = 2.0e-4 * exp(-((height(kk)-1200.0)/1200.0)**2)
                endif
                ccn_m3 = 3.0e9
                in_m3 = 5.0e7
                ncloud_m3 = 1.0e8
             case (119)
                ! Classic 'warm' geometry with the droplet number seeded at
                ! exactly the mp=8 constant Nt_c.  The result must NOT equal
                ! warm-column.csv; the divergence is the aerosol physics.
                cloud = 9.0e-4 * exp(-((height(kk)-1750.0)/900.0)**2)
                rain = 3.5e-4 * exp(-((height(kk)-1000.0)/850.0)**2)
                velocity = 1.8 * exp(-((height(kk)-1800.0)/1200.0)**2)
                ncloud_m3 = 100.0e6
             case (120)
                ! nu_c SWEEP.  mp_thompson:3218 sets
                !     nu_c = MIN(15, NINT(1000.E6/nc1d(k)) + 2)
                ! on the per-cubic-metre working number, so seeding
                ! 1000e6/slot for slot = 1..13 walks nu_c over 3..15 -- every
                ! reachable value.  nu_c = 2 is UNREACHABLE: :3217 clamps nc
                ! to Nt_c_max = 1.999e9 first and NINT(1000e6/1.999e9)+2 = 3.
                ! w = 0 keeps the fallout gate (:3798, w1d < 0.1) open at
                ! every level, and qc is uniform so the only thing that
                ! varies down the column is the shape parameter.
                cloud = 5.0e-4
                velocity = 0.0
                slot = mod(kk - 1, 13) + 1
                ncloud_m3 = nu_c_ladder(slot)
             case (121)
                ! MELT.  Cloud ice above freezing with NO cloud water, so the
                ! phase cleanup's melt branch (:3947-3953) fires with a
                ! non-zero entry ice number.  No aero-* fixture reaches it:
                ! aero-reduces-to-classic is the only committed column whose
                ! cleanup does anything, and it only freezes.
                if (height(kk) <= 4250.0) ice = 1.0e-5
                ncloud_m3 = 0.0
             case (122)
                ! FREEZE.  Cloud water at 230 K across the same five-rung
                ! droplet ladder aero-nc-auto uses, so the cleanup's freeze
                ! branch (:3956-3965) runs at several distinct xnc.
                if (height(kk) <= 4250.0) cloud = 3.0e-4
                ncloud_m3 = nc_ladder(mod(kk - 1, 5) + 1)
             case default
                error stop 'invalid aerosol Thompson scenario id'
             end select

             qc3(ii,kk,jj) = cloud
             qr3(ii,kk,jj) = rain
             qi3(ii,kk,jj) = ice
             qs3(ii,kk,jj) = snow
             qg3(ii,kk,jj) = graupel

             if (ice > 1.0e-12) then
                ! 200-micron mass-weighted diameter, inside the explicit
                ! ice-to-snow autoconversion lookup-table range.
                ni3(ii,kk,jj) = ice * (4.0/200.0e-6)**3                 &
                     / (3.1415926536*890.0)
             else
                ni3(ii,kk,jj) = 0.0
             endif

             if (rain > 1.0e-12) then
                if (which == 118) then
                   ! One-millimetre MVD activates rain self-collection and
                   ! exceeds four times the 200-micron ice diameter.
                   nr3(ii,kk,jj) = rain * (3.672/1000.0e-6)**3          &
                        / (3.1415926536*1000.0)
                elseif (which == 119) then
                   nr3(ii,kk,jj) = 3.0e5
                else
                   ! 500-micron MVD, above D0r, so rain self-collection is
                   ! active alongside the aerosol scavenging terms.
                   nr3(ii,kk,jj) = rain * (3.672/500.0e-6)**3           &
                        / (3.1415926536*1000.0)
                endif
             else
                nr3(ii,kk,jj) = 0.0
             endif

             if (ncloud_m3 > 0.0) then
                nc3(ii,kk,jj) = per_kg(ncloud_m3, pres3(ii,kk,jj),      &
                                       temp1(kk), qv3(ii,kk,jj))
             else
                nc3(ii,kk,jj) = 0.0
             endif
             if (ccn_m3 > 0.0) then
                nwfa3(ii,kk,jj) = per_kg(ccn_m3, pres3(ii,kk,jj),       &
                                         temp1(kk), qv3(ii,kk,jj))
             else
                nwfa3(ii,kk,jj) = 0.0
             endif
             if (in_m3 > 0.0) then
                nifa3(ii,kk,jj) = per_kg(in_m3, pres3(ii,kk,jj),        &
                                         temp1(kk), qv3(ii,kk,jj))
             else
                nifa3(ii,kk,jj) = 0.0
             endif
             w3(ii,kk,jj) = velocity
          enddo
       enddo
    enddo

    if (which == 102) then
       nwfa2 = 1.0e6
       nifa2 = 1.0e4
    else
       nwfa2 = 0.0
       nifa2 = 0.0
    endif
  end subroutine initialize_case

  ! ``temp`` IS AN ARGUMENT, NOT A RECOMPUTATION.  It used to be formed
  ! here as theta*exner, which for the AFTER phase is a second float32
  ! round trip through :1358's divide and lands one ulp away from WRF's own
  ! t1d at 8.1% of rows.  The caller now decides, and says which value it
  ! is passing and why.
  subroutine write_row(unit, phase, level, height, pressure, exner,     &
                       velocity, layer_dz, theta, temp, vapor, cloud,  &
                       rain,                                           &
                       ice, snow, graupel, ice_number, rain_number,    &
                       cloud_number, ccn_number, in_number,            &
                       reqc, reqi, reqs, dbz)
    integer, intent(in) :: unit, level
    character(len=*), intent(in) :: phase
    real, intent(in) :: height, pressure, exner, velocity, layer_dz
    real, intent(in) :: theta, temp
    real, intent(in) :: vapor, cloud, rain, ice, snow, graupel
    real, intent(in) :: ice_number, rain_number, cloud_number
    real, intent(in) :: ccn_number, in_number
    real, intent(in) :: reqc, reqi, reqs, dbz

    write(unit,'(A,",",I0,22(",",ES24.16E3))') trim(phase), level,    &
         height, pressure, exner, velocity, layer_dz, theta, temp,     &
         vapor, cloud, rain, ice, snow, graupel, ice_number,           &
         rain_number, cloud_number, ccn_number, in_number,             &
         reqc, reqi, reqs, dbz
  end subroutine write_row

end program run_thompson_column_aero
