! Oracle for the Noah-MP driver-side cold start of WRF v4.6.1.
!
!   module_sf_noahmpdrv::SNOW_INIT    (phys/module_sf_noahmpdrv.F 2340-2440)
!   module_sf_noahmpdrv::NOAHMP_INIT  (phys/module_sf_noahmpdrv.F 1828-2335)
!
! Both are compiled from the **byte-unmodified** pinned source.  Unlike
! module_sf_noahmplsm.F, module_sf_noahmpdrv.F declares no `private ::`
! statement anywhere: `grep -c '^ *private' phys/module_sf_noahmpdrv.F` is 0,
! so every module procedure is public by Fortran's default accessibility and
! this harness calls both routines directly.  There is no visibility patch in
! this build and none is needed -- see build_driver.sh stage [2], which asserts
! that absence rather than assuming it.
!
! Consequently the ALBEDO collision documented in ADDING_A_LEAF.md section 9
! does not arise here: the *patched* module_sf_noahmplsm.F cannot be linked
! with this driver, and this harness links the **pristine** one.
!
! Pinned option identity (WRF Registry defaults)
! ----------------------------------------------
!   dveg=4 opt_crs=1 opt_btr=1 opt_run=3 opt_sfc=1 opt_frz=1 opt_inf=1
!   opt_rad=3 opt_alb=2 opt_snf=1 opt_tbot=2 opt_stc=1 opt_rsf=1 opt_soil=1
!   opt_pedo=1 opt_crop=0 opt_irr=0 opt_irrm=0 opt_infdv=0 opt_tdrn=0
!   sf_urban_physics=0  restart=.false.  allowed_to_read=.true.
!
! NOAHMP_INIT only reads five of those: iopt_run, iopt_crop, iopt_irr,
! iopt_irrm and sf_urban_physics.  They are dummy arguments, not module state,
! so they are emitted per case and are part of the fixture.
!
! Dead under that identity and NOT exercised here
! -----------------------------------------------
!   iopt_run/=5  kills the groundwater init block (2299-2331) and with it
!                STEPWTD, areaxy and every OPTIONAL groundwater argument;
!                2145-2148 selects waxy=4900 / wtxy=waxy / zwtxy=27-waxy/1000/0.2
!                as the only live aquifer cold start.
!   iopt_crop=0  kills the Liu crop block (2201-2232), the gecros block
!                (2236-2260) and therefore gecros_init and every symbol
!                NOAHMP_INIT imports from module_sf_gecros.  cropcat is then
!                assigned ONLY on the barren/ice/urban/water branch (2178) and
!                is left at its entry value on vegetated land -- an
!                INTENT(OUT) dummy that a live path does not write.  That is
!                pinned, not tidied: every case enters cropcat with a distinct
!                non-zero value and both stages are emitted.
!   iopt_irr=0   kills the irrigation cold start (2263-2278), so irnumsi,
!                irnummi, irnumfi, irwatsi, irwatmi, irwatfi, ireloss,
!                irsivol, irmivol, irfivol and irrsplh are pass-through.
!                Every case drives all eleven non-zero so the claim is
!                measured rather than vacuous.
!   sf_urban=0   makes urbanpt_flag route ISURBAN/LCZ_1..LCZ_11 into the
!                barren/ice/water zeroing branch (2164-2178), so the
!                NATURAL_TABLE masslai form at 2185 is unreachable.  Case
!                nia_urban (ISURBAN_TABLE) and case nia_lcz (LCZ_1_TABLE)
!                prove the routing from the inputs.
!   restart=T    skips the whole body; not a physics branch.
!
! Two INTENT(OUT) hazards, pinned rather than assumed
! ---------------------------------------------------
! * SNOW_INIT declares ZSNSOXY INTENT(OUT) over -NSNOW+1:NSOIL but writes only
!   ISNOWXY+1..NSOIL (2432-2435).  With ISNOWXY=0 the three snow slots are
!   never assigned at all.  gfortran passes the array by reference and does
!   not clobber it, so the caller's values stand.  Every case enters those
!   slots with a distinct non-zero pattern and both stages are emitted, which
!   turns "not written" into a measured row.
! * SNOW_INIT's local DZSNO is zeroed only on the `SNODEP < 0.025` branch
!   (2383).  On every other branch just the slots the case uses are assigned,
!   so DZSNO(-2)/DZSNO(-1) can carry a previous column's value.  They are
!   read only by the `DO IZ = ISNOWXY+1, 0` loops, which never reach an
!   unassigned slot; the `snan` control in build_driver.sh is what proves it.
!
! The transcendental
! ------------------
! NOAHMP_INIT's only transcendental is the supercooled-liquid initial guess
! at 2095-2096:
!     FK = ((HLICE/(GRAV*(-PSISAT))) * ((TSLB-T0)/TSLB))**(-1/BEXP) * SMCMAX
! `-1` is a default INTEGER and BEXP a REAL, so `-1/BEXP` is REAL division,
! and `x**y` with both REAL compiles to a scalar `powf` call at -O0.  The
! fixture is built at -O0 and build_driver.sh runs `nm -u` on the object to
! prove no libmvec `_ZGVbN4v_powf` was substituted (ADDING_A_LEAF.md trap 3).
!
! Fixture convention
! ------------------
! Long format, one row per scalar:  leaf,case,stage,field,index,dtype,value,bits
! Every case emits its complete entry state and its complete exit state, so a
! row is reproducible from the CSV alone and any argument the routine does not
! touch is visibly identical across the two stages.  Reals carry both a
! round-tripping decimal form and the raw IEEE binary32 bit pattern; the
! consumer compares bit patterns, so no parsing tolerance exists anywhere.

module driver_oracle

  use module_sf_noahmpdrv, only : snow_init, noahmp_init
  use noahmp_tables,       only : isice_table, isurban_table, iswater_table,  &
                                  isbarren_table, natural_table, lcz_1_table, &
                                  sla_table, bexp_table, smcmax_table,        &
                                  psisat_table, read_mp_veg_parameters,       &
                                  read_mp_soil_parameters

  implicit none

  ! NOT a parameter, and NOT a hard-coded unit number.  NOAHMP_TABLES'
  ! read_mp_soil_parameters opens Fortran unit 21 on SOILPARM.TBL
  ! (module_sf_noahmplsm.F 11683) and read_mp_veg_parameters opens unit 15;
  ! NOAHMP_INIT calls both.  A hard-coded CSV unit that collides with either
  ! is silently disconnected mid-run and the fixture loses every row emitted
  ! after the first NOAHMP_INIT call.  `newunit=` is the only collision-proof
  ! choice, so the unit is a module variable assigned at open time.
  integer :: CSV

contains

! ---------------------------------------------------------------------------
! CSV emitters
! ---------------------------------------------------------------------------

  function itoa(i) result(s)
    integer, intent(in) :: i
    character(len=16) :: s
    write(s, '(I0)') i
  end function itoa

  function rtoa(v) result(s)
    real, intent(in) :: v
    character(len=20) :: s
    write(s, '(ES16.9E2)') v
    s = adjustl(s)
  end function rtoa

  function hex8(v) result(s)
    real, intent(in) :: v
    character(len=8) :: s
    integer :: b
    b = transfer(v, b)
    write(s, '(Z8.8)') b
  end function hex8

  subroutine emit_r(leaf, cname, stage, field, idx, v)
    character(len=*), intent(in) :: leaf, cname, stage, field
    integer, intent(in) :: idx
    real,    intent(in) :: v
    write(CSV, '(A)') leaf//','//cname//','//stage//','//field//','// &
         trim(itoa(idx))//',real,'//trim(rtoa(v))//','//hex8(v)
  end subroutine emit_r

  subroutine emit_i(leaf, cname, stage, field, idx, v)
    character(len=*), intent(in) :: leaf, cname, stage, field
    integer, intent(in) :: idx
    integer, intent(in) :: v
    write(CSV, '(A)') leaf//','//cname//','//stage//','//field//','// &
         trim(itoa(idx))//',int,'//trim(itoa(v))//','
  end subroutine emit_i

  subroutine emit_l(leaf, cname, stage, field, idx, v)
    character(len=*), intent(in) :: leaf, cname, stage, field
    integer, intent(in) :: idx
    logical, intent(in) :: v
    if (v) then
       call emit_i(leaf, cname, stage, field, idx, 1)
    else
       call emit_i(leaf, cname, stage, field, idx, 0)
    end if
  end subroutine emit_l

! ===========================================================================
! SNOW_INIT -- phys/module_sf_noahmpdrv.F 2340-2440
!
! Six live branches of the snow-depth ladder (2381-2408), all decidable from
! SNODEP alone, plus the two closed edges of every interval so a port that
! swaps `<` for `<=` cannot pass:
!
!   2381  SNODEP <  0.025               ISNOW =  0
!   2385  0.025 <= SNODEP <= 0.05       ISNOW = -1
!   2388  0.05  <  SNODEP <= 0.10       ISNOW = -2, halves
!   2392  0.10  <  SNODEP <= 0.25       ISNOW = -2, 0.05 + remainder
!   2396  0.25  <  SNODEP <= 0.45       ISNOW = -3, 0.05 + two halves
!   2401  SNODEP >  0.45                ISNOW = -3, 0.05 + 0.20 + remainder
!   2406  the `else` is unreachable -- the five tests above are exhaustive
!         over the reals once SNODEP >= 0.025, so wrf_error_fatal at 2407 can
!         only fire on a NaN.  No case reaches it and none can.
!
! Two soil stacks and two NSOIL values are exercised, because ZSOIL enters
! only through DZSNSO(1)=ZSOIL(1) and DZSNSO(k)=ZSOIL(k)-ZSOIL(k-1) (2426-2429)
! and a port that hard-codes four layers must fail.
! ===========================================================================

  subroutine dump_snow_init()
    integer, parameter :: NSNOW = 3
    integer, parameter :: N4 = 4, N9 = 9
    integer, parameter :: NC = 12

    ! WRF Registry default four-layer stack, and a nine-layer stack.
    real, parameter :: ZS4(N4) = (/ -0.10, -0.40, -1.00, -2.00 /)
    real, parameter :: ZS9(N9) = (/ -0.05, -0.15, -0.30, -0.50, -0.75,       &
                                    -1.05, -1.40, -1.80, -2.25 /)

    character(len=28), parameter :: CN(NC) = (/ character(len=28) ::         &
         'si_bare_zero',      'si_bare_thin',      'si_one_layer_low',       &
         'si_one_layer_high', 'si_two_halves_low', 'si_two_halves_high',     &
         'si_two_split_low',  'si_two_split_high', 'si_three_even_low',      &
         'si_three_even_high','si_three_deep',     'si_three_very_deep' /)

    ! SNODEP straddles every interval edge in both directions.
    real, parameter :: SNODEP_C(NC) = (/                                     &
         0.000,  0.024900,  0.025000,  0.050000,  0.050001,  0.100000,       &
         0.100001, 0.250000, 0.250001,  0.450000,  0.450001,  1.237000 /)
    real, parameter :: SWE_C(NC) = (/                                        &
         0.000,  4.310000,  4.900000, 10.700000, 11.100000, 22.400000,       &
         22.900000, 61.500000, 62.300000, 118.900000, 119.400000, 371.100000 /)
    real, parameter :: TG_C(NC) = (/                                         &
         281.37, 272.11, 271.63, 269.84, 268.22, 265.79,                     &
         264.13, 261.48, 259.92, 256.31, 254.77, 247.06 /)

    integer :: isnowxy(NC, 1)
    real    :: swe(NC, 1), tgxy(NC, 1), snodep(NC, 1)
    real    :: zsnso4(NC, -NSNOW+1:N4, 1), zsnso9(NC, -NSNOW+1:N9, 1)
    real    :: tsno(NC, -NSNOW+1:0, 1), snice(NC, -NSNOW+1:0, 1)
    real    :: snliq(NC, -NSNOW+1:0, 1)
    integer :: i

    ! --- four-layer stack -------------------------------------------------
    call seed(NC, NSNOW, N4, isnowxy, swe, tgxy, snodep, zsnso4, tsno,       &
              snice, snliq, SNODEP_C, SWE_C, TG_C)
    call emit_si(NC, NSNOW, N4, 'input ', CN, ZS4, isnowxy, swe, tgxy,       &
                 snodep, zsnso4, tsno, snice, snliq, 'a')
    call snow_init(1, NC, 1, 1, 1, NC, 1, 1, NSNOW, N4, ZS4, swe, tgxy,      &
                   snodep, zsnso4, tsno, snice, snliq, isnowxy)
    call emit_si(NC, NSNOW, N4, 'output', CN, ZS4, isnowxy, swe, tgxy,       &
                 snodep, zsnso4, tsno, snice, snliq, 'a')

    ! --- nine-layer stack -------------------------------------------------
    call seed(NC, NSNOW, N9, isnowxy, swe, tgxy, snodep, zsnso9, tsno,       &
              snice, snliq, SNODEP_C, SWE_C, TG_C)
    ! Perturb the forcing so the two stacks are not the same experiment.
    do i = 1, NC
       swe(i, 1)  = swe(i, 1)  * 1.113
       tgxy(i, 1) = tgxy(i, 1) - 1.37
    end do
    call emit_si(NC, NSNOW, N9, 'input ', CN, ZS9, isnowxy, swe, tgxy,       &
                 snodep, zsnso9, tsno, snice, snliq, 'b')
    call snow_init(1, NC, 1, 1, 1, NC, 1, 1, NSNOW, N9, ZS9, swe, tgxy,      &
                   snodep, zsnso9, tsno, snice, snliq, isnowxy)
    call emit_si(NC, NSNOW, N9, 'output', CN, ZS9, isnowxy, swe, tgxy,       &
                 snodep, zsnso9, tsno, snice, snliq, 'b')
  end subroutine dump_snow_init

  ! Entry state.  The INTENT(OUT) arrays are seeded with a distinct, non-zero,
  ! per-column, per-layer pattern so "SNOW_INIT left this slot alone" is a
  ! measurable statement instead of an indistinguishable zero.
  subroutine seed(nc, nsnow, nsoil, isnowxy, swe, tgxy, snodep, zsnso,       &
                  tsno, snice, snliq, snodep_c, swe_c, tg_c)
    integer, intent(in)  :: nc, nsnow, nsoil
    integer, intent(out) :: isnowxy(nc, 1)
    real,    intent(out) :: swe(nc, 1), tgxy(nc, 1), snodep(nc, 1)
    real,    intent(out) :: zsnso(nc, -nsnow+1:nsoil, 1)
    real,    intent(out) :: tsno(nc, -nsnow+1:0, 1), snice(nc, -nsnow+1:0, 1)
    real,    intent(out) :: snliq(nc, -nsnow+1:0, 1)
    real,    intent(in)  :: snodep_c(nc), swe_c(nc), tg_c(nc)
    integer :: i, k
    do i = 1, nc
       isnowxy(i, 1) = -7 - i
       swe(i, 1)     = swe_c(i)
       tgxy(i, 1)    = tg_c(i)
       snodep(i, 1)  = snodep_c(i)
       do k = -nsnow+1, nsoil
          zsnso(i, k, 1) = -3.125 - 0.5 * real(i) + 0.25 * real(k)
       end do
       do k = -nsnow+1, 0
          tsno(i, k, 1)  = 233.75 + 3.5 * real(i) - 2.25 * real(k)
          snice(i, k, 1) = 17.5 + 1.75 * real(i) - 0.5 * real(k)
          snliq(i, k, 1) = 3.25 + 0.125 * real(i) - 0.375 * real(k)
       end do
    end do
  end subroutine seed

  subroutine emit_si(nc, nsnow, nsoil, stage, cn, zsoil, isnowxy, swe, tgxy, &
                     snodep, zsnso, tsno, snice, snliq, tag)
    integer, intent(in) :: nc, nsnow, nsoil
    character(len=*), intent(in) :: stage, tag
    character(len=*), intent(in) :: cn(nc)
    real,    intent(in) :: zsoil(nsoil)
    integer, intent(in) :: isnowxy(nc, 1)
    real,    intent(in) :: swe(nc, 1), tgxy(nc, 1), snodep(nc, 1)
    real,    intent(in) :: zsnso(nc, -nsnow+1:nsoil, 1)
    real,    intent(in) :: tsno(nc, -nsnow+1:0, 1), snice(nc, -nsnow+1:0, 1)
    real,    intent(in) :: snliq(nc, -nsnow+1:0, 1)
    character(len=32) :: name
    integer :: i, k
    do i = 1, nc
       name = trim(cn(i))//'_'//tag
       call emit_i('snow_init', trim(name), trim(stage), 'nsnow', 0, nsnow)
       call emit_i('snow_init', trim(name), trim(stage), 'nsoil', 0, nsoil)
       call emit_r('snow_init', trim(name), trim(stage), 'swe', 0, swe(i, 1))
       call emit_r('snow_init', trim(name), trim(stage), 'tgxy', 0, tgxy(i, 1))
       call emit_r('snow_init', trim(name), trim(stage), 'snodep', 0,        &
                   snodep(i, 1))
       call emit_i('snow_init', trim(name), trim(stage), 'isnowxy', 0,       &
                   isnowxy(i, 1))
       do k = 1, nsoil
          call emit_r('snow_init', trim(name), trim(stage), 'zsoil', k,      &
                      zsoil(k))
       end do
       do k = -nsnow+1, nsoil
          call emit_r('snow_init', trim(name), trim(stage), 'zsnsoxy', k,    &
                      zsnso(i, k, 1))
       end do
       do k = -nsnow+1, 0
          call emit_r('snow_init', trim(name), trim(stage), 'tsnoxy', k,     &
                      tsno(i, k, 1))
          call emit_r('snow_init', trim(name), trim(stage), 'snicexy', k,    &
                      snice(i, k, 1))
          call emit_r('snow_init', trim(name), trim(stage), 'snliqxy', k,    &
                      snliq(i, k, 1))
       end do
    end do
  end subroutine emit_si

! ===========================================================================
! NOAHMP_INIT -- phys/module_sf_noahmpdrv.F 1828-2335
! ===========================================================================

  subroutine dump_noahmp_init()
    integer, parameter :: NX = 15, NSOIL = 4, NSNOW = 3

    character(len=24), parameter :: COL(NX) = (/ character(len=24) ::        &
         'veg_warm_soil',    'veg_mixed_freeze', 'veg_over_smcmax',          &
         'glacier_cold',     'water_point',      'barren_point',             &
         'urban_point',      'swe_over_cap',     'snow_warm_skin',           &
         'glacier_seaice',   'lcz_point',        'veg_freeze_edge',           &
         'veg_deep_freeze',  'veg_dry_freeze',   'swe_at_cap' /)

    integer, parameter :: IVG(NX) = (/  1,  4, 10, 15, 17, 16, 13,  5,  8,   &
                                       15, 51,  2,  7,  9,  6 /)
    integer, parameter :: ISL(NX) = (/  3,  8, 12, 16, 14, 15,  6,  4,  9,   &
                                       16,  7,  2,  1, 12,  5 /)
    real,    parameter :: XIC(NX) = (/ 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,   &
                                       0.00, 0.00, 0.00, 0.40, 0.00, 0.00,   &
                                       0.00, 0.00, 0.00 /)
    real,    parameter :: TSKC(NX) = (/ 285.30, 262.40, 291.60, 251.20,      &
                                        279.80, 297.40, 293.70, 271.90,      &
                                        281.40, 258.60, 296.10, 302.40,      &
                                        243.70, 270.10, 268.90 /)
    real,    parameter :: SNW(NX) = (/ 0.00, 45.00, 0.00, 6.00, 0.00, 0.00,  &
                                       0.00, 2450.00, 12.50, 90.00, 0.00,    &
                                       0.00, 3.20, 0.00, 2000.00 /)
    real,    parameter :: SNH(NX) = (/ 0.00, 0.140, 0.00, 0.030, 0.00, 0.00, &
                                       0.00, 6.300, 0.060, 0.400, 0.00,      &
                                       0.00, 0.016, 0.00, 3.8112843 /)
    real,    parameter :: LAI0(NX) = (/ 2.70, 1.10, 0.03, 0.20, 0.60, 0.40,  &
                                        1.40, 3.20, 0.90, 0.15, 1.80, 5.40,  &
                                        0.70, 0.45, 1.25 /)
    real,    parameter :: TMN0(NX) = (/ 285.00, 277.20, 288.00, 262.00,      &
                                        281.00, 294.00, 292.00, 274.00,      &
                                        279.00, 260.00, 294.50, 300.00,      &
                                        251.00, 268.00, 265.00 /)
    real,    parameter :: XLA(NX) = (/ 0.70, 0.90, 0.40, 1.10, 0.20, 0.30,   &
                                       0.60, 1.00, 0.50, 1.20, 0.35, 0.10,   &
                                       1.05, 0.85, 0.55 /)

    real, parameter :: TSL(NSOIL, NX) = reshape( (/                          &
         288.10, 287.40, 286.20, 285.00,                                     &
         265.30, 268.10, 271.90, 274.60,                                     &
         293.00, 292.10, 290.40, 288.80,                                     &
         259.40, 261.80, 264.70, 266.10,                                     &
         280.10, 280.40, 280.90, 281.30,                                     &
         299.20, 298.10, 296.30, 294.70,                                     &
         294.60, 294.10, 293.20, 292.40,                                     &
         264.10, 266.70, 269.30, 272.80,                                     &
         275.20, 276.40, 277.10, 278.30,                                     &
         256.30, 257.90, 259.40, 261.20,                                     &
         297.30, 296.80, 295.90, 294.60,                                     &
         271.80, 272.40, 273.148, 273.1495,                                    &
         240.50, 245.20, 250.80, 255.60,                                     &
         272.90, 272.60, 272.30, 272.00,                                     &
         266.20, 267.10, 268.40, 269.60 /), (/ NSOIL, NX /) )

    real, parameter :: SMO(NSOIL, NX) = reshape( (/                          &
         0.280, 0.300, 0.310, 0.330,                                         &
         0.360, 0.380, 0.400, 0.420,                                         &
         0.520, 0.490, 0.470, 0.440,                                         &
         0.310, 0.330, 0.350, 0.370,                                         &
         0.850, 0.860, 0.870, 0.880,                                         &
         0.110, 0.120, 0.130, 0.140,                                         &
         0.290, 0.310, 0.320, 0.340,                                         &
         0.410, 0.430, 0.440, 0.460,                                         &
         0.340, 0.360, 0.370, 0.390,                                         &
         0.260, 0.280, 0.290, 0.310,                                         &
         0.240, 0.260, 0.270, 0.290,                                         &
         0.180, 0.220, 0.400, 0.400,                                         &
         0.050, 0.060, 0.070, 0.080,                                         &
         0.030, 0.035, 0.040, 0.045,                                         &
         0.300, 0.320, 0.330, 0.350 /), (/ NSOIL, NX /) )

    real, parameter :: DZS(NSOIL) = (/ 0.10, 0.30, 0.60, 1.00 /)

    ! --- NOAHMP_INIT dummy arguments --------------------------------------
    character(len=256) :: mminlu
    real    :: snow(NX,1), snowh(NX,1), canwat(NX,1), xlat(NX,1)
    integer :: isltyp(NX,1), ivgtyp(NX,1)
    real    :: tslb(NX,NSOIL,1), smois(NX,NSOIL,1), sh2o(NX,NSOIL,1)
    logical :: fndsoilw, fndsnowh
    real    :: tsk(NX,1), tmn(NX,1), xice(NX,1)
    integer :: isnowxy(NX,1)
    real    :: tvxy(NX,1), tgxy(NX,1), canicexy(NX,1), canliqxy(NX,1)
    real    :: eahxy(NX,1), tahxy(NX,1), cmxy(NX,1), chxy(NX,1)
    real    :: fwetxy(NX,1), sneqvoxy(NX,1), alboldxy(NX,1)
    real    :: qsnowxy(NX,1), qrainxy(NX,1), wslakexy(NX,1)
    real    :: zwtxy(NX,1), waxy(NX,1), wtxy(NX,1)
    real    :: tsnoxy(NX,-2:0,1), zsnsoxy(NX,-2:NSOIL,1)
    real    :: snicexy(NX,-2:0,1), snliqxy(NX,-2:0,1)
    real    :: lfmassxy(NX,1), rtmassxy(NX,1), stmassxy(NX,1), woodxy(NX,1)
    real    :: stblcpxy(NX,1), fastcpxy(NX,1), xsaixy(NX,1), lai(NX,1)
    real    :: grainxy(NX,1), gddxy(NX,1)
    real    :: croptype(NX,5,1)
    integer :: cropcat(NX,1)
    integer :: irnumsi(NX,1), irnummi(NX,1), irnumfi(NX,1)
    real    :: irwatsi(NX,1), irwatmi(NX,1), irwatfi(NX,1), ireloss(NX,1)
    real    :: irsivol(NX,1), irmivol(NX,1), irfivol(NX,1), irrsplh(NX,1)
    real    :: t2mvxy(NX,1), t2mbxy(NX,1), chstarxy(NX,1)
    real    :: qtdrain(NX,1)
    logical :: restart, allowed_to_read
    integer :: iopt_run, iopt_crop, iopt_irr, iopt_irrm, sf_urban_physics

    integer :: scen

    mminlu          = 'MODIFIED_IGBP_MODIS_NOAH'

    ! BEXP_TABLE / SMCMAX_TABLE / PSISAT_TABLE / SLA_TABLE are NOAHMP_TABLES
    ! module state that only the read_mp_* readers populate.  NOAHMP_INIT calls
    ! them itself (2000-2007), but the `input` stage is emitted BEFORE the first
    ! call, so without this the first scenario would report the readers'
    ! pre-initialisation values as if they were the parameters the branch used.
    ! These are the module's own readers against the byte-pinned tables -- the
    ! same two calls run_sflx.F90 makes -- and they are idempotent.  The `snan`
    ! control in build_driver.sh is what caught the omission.
    call read_mp_veg_parameters(trim(mminlu))
    call read_mp_soil_parameters()

    restart         = .false.
    allowed_to_read = .true.
    iopt_run        = 3
    iopt_crop       = 0
    iopt_irr        = 0
    iopt_irrm       = 0
    sf_urban_physics = 0
    fndsoilw        = .true.

    do scen = 1, 2
       ! scen 1: FNDSNOWH = .true.  -- SNOWH arrives from the input file.
       ! scen 2: FNDSNOWH = .false. -- SNOWH is derived as SNOW*0.005 (2022),
       !         which also re-decides the >2000 mm cap at 2037-2040.
       fndsnowh = (scen == 1)

       call seed_ni()
       call emit_ni('input ', scen)
       call noahmp_init(trim(mminlu), snow, snowh, canwat, isltyp, ivgtyp,   &
            xlat, tslb, smois, sh2o, DZS, fndsoilw, fndsnowh,                &
            tsk, isnowxy, tvxy, tgxy, canicexy, tmn, xice,                   &
            canliqxy, eahxy, tahxy, cmxy, chxy,                              &
            fwetxy, sneqvoxy, alboldxy, qsnowxy, qrainxy, wslakexy, zwtxy,   &
            waxy, wtxy, tsnoxy, zsnsoxy, snicexy, snliqxy, lfmassxy,         &
            rtmassxy, stmassxy, woodxy, stblcpxy, fastcpxy, xsaixy, lai,     &
            grainxy, gddxy, croptype, cropcat,                               &
            irnumsi, irnummi, irnumfi, irwatsi, irwatmi, irwatfi, ireloss,   &
            irsivol, irmivol, irfivol, irrsplh,                              &
            t2mvxy, t2mbxy, chstarxy,                                        &
            NSOIL, restart, allowed_to_read,                                 &
            iopt_run, iopt_crop, iopt_irr, iopt_irrm, sf_urban_physics,      &
            1, NX+1, 1, 2, 1, 2,                                             &
            1, NX,   1, 1, 1, 2,                                             &
            1, NX,   1, 1, 1, 1,                                             &
            qtdrain = qtdrain)
       call emit_ni('output', scen)
    end do

    ! The table identities the branch structure depends on, emitted once so a
    ! consumer can assert the routing without re-parsing MPTABLE.TBL.
    call emit_i('noahmp_init', 'table_identity', 'probe', 'isice_table',   0, &
                isice_table)
    call emit_i('noahmp_init', 'table_identity', 'probe', 'isurban_table', 0, &
                isurban_table)
    call emit_i('noahmp_init', 'table_identity', 'probe', 'iswater_table', 0, &
                iswater_table)
    call emit_i('noahmp_init', 'table_identity', 'probe', 'isbarren_table',0, &
                isbarren_table)
    call emit_i('noahmp_init', 'table_identity', 'probe', 'natural_table', 0, &
                natural_table)
    call emit_i('noahmp_init', 'table_identity', 'probe', 'lcz_1_table',   0, &
                lcz_1_table)

  contains

    subroutine seed_ni()
      integer :: i, k
      do i = 1, NX
         snow(i,1)     = SNW(i)
         snowh(i,1)    = SNH(i)
         ! CANWAT is unconditionally zeroed at 2121; enter non-zero so the
         ! kill is measured.
         canwat(i,1)   = 0.75 + 0.05 * real(i)
         xlat(i,1)     = XLA(i)
         isltyp(i,1)   = ISL(i)
         ivgtyp(i,1)   = IVG(i)
         tsk(i,1)      = TSKC(i)
         tmn(i,1)      = TMN0(i)
         xice(i,1)     = XIC(i)
         lai(i,1)      = LAI0(i)
         do k = 1, NSOIL
            tslb(i,k,1)  = TSL(k,i)
            smois(i,k,1) = SMO(k,i)
            ! SH2O is INTENT(INOUT) and every live path overwrites it; enter
            ! with a value no branch could produce.
            sh2o(i,k,1)  = -1.0 - 0.125 * real(k) - 0.03125 * real(i)
         end do
         ! Everything below is INOUT and cold-start assigned.  Non-zero,
         ! per-column distinct entries make each assignment observable.
         isnowxy(i,1)  = -7 - i
         tvxy(i,1)     = 199.5 + real(i)
         tgxy(i,1)     = 198.5 + real(i)
         canicexy(i,1) = 11.25 + 0.5 * real(i)
         canliqxy(i,1) = 12.25 + 0.5 * real(i)
         eahxy(i,1)    = 1301.0 + 7.0 * real(i)
         tahxy(i,1)    = 197.5 + real(i)
         cmxy(i,1)     = 0.0125 + 0.00125 * real(i)
         chxy(i,1)     = 0.0225 + 0.00125 * real(i)
         fwetxy(i,1)   = 0.125 + 0.03125 * real(i)
         sneqvoxy(i,1) = 6.5 + 0.25 * real(i)
         alboldxy(i,1) = 0.15 + 0.01 * real(i)
         qsnowxy(i,1)  = 1.5e-4 + 1.0e-5 * real(i)
         qrainxy(i,1)  = 2.5e-4 + 1.0e-5 * real(i)
         wslakexy(i,1) = 3.5 + 0.125 * real(i)
         zwtxy(i,1)    = -8.25 - 0.5 * real(i)
         waxy(i,1)     = 101.0 + 3.0 * real(i)
         wtxy(i,1)     = 202.0 + 3.0 * real(i)
         lfmassxy(i,1) = 41.0 + 2.0 * real(i)
         rtmassxy(i,1) = 42.0 + 2.0 * real(i)
         stmassxy(i,1) = 43.0 + 2.0 * real(i)
         woodxy(i,1)   = 44.0 + 2.0 * real(i)
         stblcpxy(i,1) = 45.0 + 2.0 * real(i)
         fastcpxy(i,1) = 46.0 + 2.0 * real(i)
         xsaixy(i,1)   = 0.625 + 0.0625 * real(i)
         grainxy(i,1)  = 0.5 + 0.125 * real(i)
         gddxy(i,1)    = 17.0 + real(i)
         t2mvxy(i,1)   = 196.5 + real(i)
         t2mbxy(i,1)   = 195.5 + real(i)
         chstarxy(i,1) = 0.55 + 0.01 * real(i)
         qtdrain(i,1)  = 9.5 + 0.25 * real(i)
         ! cropcat is INTENT(OUT); with iopt_crop=0 the vegetated branch
         ! never writes it.  Seed distinct so the omission is visible.
         cropcat(i,1)  = 300 + i
         ! Irrigation state: inert at iopt_irr=0, driven non-zero anyway.
         irnumsi(i,1)  = 11 + i
         irnummi(i,1)  = 21 + i
         irnumfi(i,1)  = 31 + i
         irwatsi(i,1)  = 0.125 + 0.0625 * real(i)
         irwatmi(i,1)  = 0.250 + 0.0625 * real(i)
         irwatfi(i,1)  = 0.375 + 0.0625 * real(i)
         ireloss(i,1)  = 1.125 + 0.0625 * real(i)
         irsivol(i,1)  = 2.125 + 0.0625 * real(i)
         irmivol(i,1)  = 3.125 + 0.0625 * real(i)
         irfivol(i,1)  = 4.125 + 0.0625 * real(i)
         irrsplh(i,1)  = 5.125 + 0.0625 * real(i)
         ! croptype is read only at iopt_crop>=1; vary it so inertness is
         ! measured, and make column 5 exceed the 0.5 gate at 2203/2238.
         do k = 1, 5
            croptype(i,k,1) = 0.0625 * real(k) + 0.015625 * real(i)
         end do
         croptype(i,5,1) = 0.75 + 0.015625 * real(i)
         do k = -2, 0
            tsnoxy(i,k,1)  = 231.25 + 3.5 * real(i) - 2.25 * real(k)
            snicexy(i,k,1) = 15.5 + 1.75 * real(i) - 0.5 * real(k)
            snliqxy(i,k,1) = 2.25 + 0.125 * real(i) - 0.375 * real(k)
         end do
         do k = -2, NSOIL
            zsnsoxy(i,k,1) = -3.125 - 0.5 * real(i) + 0.25 * real(k)
         end do
      end do
    end subroutine seed_ni

    subroutine emit_ni(stage, scen)
      character(len=*), intent(in) :: stage
      integer, intent(in) :: scen
      character(len=40) :: name
      character(len=4)  :: tag
      integer :: i, k
      if (scen == 1) then
         tag = '_sh1'
      else
         tag = '_sh0'
      end if
      do i = 1, NX
         name = trim(COL(i))//tag
         ! --- pure inputs and the option identity ------------------------
         call emit_i('noahmp_init', trim(name), trim(stage), 'nsoil', 0, NSOIL)
         call emit_i('noahmp_init', trim(name), trim(stage), 'nsnow', 0, NSNOW)
         call emit_l('noahmp_init', trim(name), trim(stage), 'fndsnowh', 0, fndsnowh)
         call emit_l('noahmp_init', trim(name), trim(stage), 'fndsoilw', 0, fndsoilw)
         call emit_l('noahmp_init', trim(name), trim(stage), 'restart', 0, restart)
         call emit_l('noahmp_init', trim(name), trim(stage), 'allowed_to_read', 0,  &
                     allowed_to_read)
         call emit_i('noahmp_init', trim(name), trim(stage), 'iopt_run', 0, iopt_run)
         call emit_i('noahmp_init', trim(name), trim(stage), 'iopt_crop', 0,        &
                     iopt_crop)
         call emit_i('noahmp_init', trim(name), trim(stage), 'iopt_irr', 0,         &
                     iopt_irr)
         call emit_i('noahmp_init', trim(name), trim(stage), 'iopt_irrm', 0,        &
                     iopt_irrm)
         call emit_i('noahmp_init', trim(name), trim(stage), 'sf_urban_physics', 0, &
                     sf_urban_physics)
         call emit_i('noahmp_init', trim(name), trim(stage), 'isltyp', 0,           &
                     isltyp(i,1))
         call emit_i('noahmp_init', trim(name), trim(stage), 'ivgtyp', 0,           &
                     ivgtyp(i,1))
         call emit_r('noahmp_init', trim(name), trim(stage), 'tsk', 0, tsk(i,1))
         call emit_r('noahmp_init', trim(name), trim(stage), 'xice', 0, xice(i,1))
         call emit_r('noahmp_init', trim(name), trim(stage), 'xlat', 0, xlat(i,1))
         do k = 1, NSOIL
            call emit_r('noahmp_init', trim(name), trim(stage), 'dzs', k, DZS(k))
         end do
         ! The three soil-table entries the FK branch reads, so the fixture
         ! is closed over its own parameter dependence.
         call emit_r('noahmp_init', trim(name), trim(stage), 'bexp_table', 0,       &
                     bexp_table(isltyp(i,1)))
         call emit_r('noahmp_init', trim(name), trim(stage), 'smcmax_table', 0,     &
                     smcmax_table(isltyp(i,1)))
         call emit_r('noahmp_init', trim(name), trim(stage), 'psisat_table', 0,     &
                     psisat_table(isltyp(i,1)))
         if (ivgtyp(i,1) >= 1 .and. ivgtyp(i,1) <= 20) then
            call emit_r('noahmp_init', trim(name), trim(stage), 'sla_table', 0,     &
                        sla_table(ivgtyp(i,1)))
         end if
         ! --- inout / out state ------------------------------------------
         call emit_r('noahmp_init', trim(name), trim(stage), 'snow', 0, snow(i,1))
         call emit_r('noahmp_init', trim(name), trim(stage), 'snowh', 0, snowh(i,1))
         call emit_r('noahmp_init', trim(name), trim(stage), 'canwat', 0,           &
                     canwat(i,1))
         call emit_r('noahmp_init', trim(name), trim(stage), 'tmn', 0, tmn(i,1))
         call emit_r('noahmp_init', trim(name), trim(stage), 'lai', 0, lai(i,1))
         call emit_r('noahmp_init', trim(name), trim(stage), 'xsaixy', 0,           &
                     xsaixy(i,1))
         call emit_i('noahmp_init', trim(name), trim(stage), 'isnowxy', 0,          &
                     isnowxy(i,1))
         call emit_i('noahmp_init', trim(name), trim(stage), 'cropcat', 0,          &
                     cropcat(i,1))
         call emit_r('noahmp_init', trim(name), trim(stage), 'tvxy', 0, tvxy(i,1))
         call emit_r('noahmp_init', trim(name), trim(stage), 'tgxy', 0, tgxy(i,1))
         call emit_r('noahmp_init', trim(name), trim(stage), 'canicexy', 0,         &
                     canicexy(i,1))
         call emit_r('noahmp_init', trim(name), trim(stage), 'canliqxy', 0,         &
                     canliqxy(i,1))
         call emit_r('noahmp_init', trim(name), trim(stage), 'eahxy', 0, eahxy(i,1))
         call emit_r('noahmp_init', trim(name), trim(stage), 'tahxy', 0, tahxy(i,1))
         call emit_r('noahmp_init', trim(name), trim(stage), 'cmxy', 0, cmxy(i,1))
         call emit_r('noahmp_init', trim(name), trim(stage), 'chxy', 0, chxy(i,1))
         call emit_r('noahmp_init', trim(name), trim(stage), 'fwetxy', 0,           &
                     fwetxy(i,1))
         call emit_r('noahmp_init', trim(name), trim(stage), 'sneqvoxy', 0,         &
                     sneqvoxy(i,1))
         call emit_r('noahmp_init', trim(name), trim(stage), 'alboldxy', 0,         &
                     alboldxy(i,1))
         call emit_r('noahmp_init', trim(name), trim(stage), 'qsnowxy', 0,          &
                     qsnowxy(i,1))
         call emit_r('noahmp_init', trim(name), trim(stage), 'qrainxy', 0,          &
                     qrainxy(i,1))
         call emit_r('noahmp_init', trim(name), trim(stage), 'wslakexy', 0,         &
                     wslakexy(i,1))
         call emit_r('noahmp_init', trim(name), trim(stage), 'zwtxy', 0, zwtxy(i,1))
         call emit_r('noahmp_init', trim(name), trim(stage), 'waxy', 0, waxy(i,1))
         call emit_r('noahmp_init', trim(name), trim(stage), 'wtxy', 0, wtxy(i,1))
         call emit_r('noahmp_init', trim(name), trim(stage), 'lfmassxy', 0,         &
                     lfmassxy(i,1))
         call emit_r('noahmp_init', trim(name), trim(stage), 'rtmassxy', 0,         &
                     rtmassxy(i,1))
         call emit_r('noahmp_init', trim(name), trim(stage), 'stmassxy', 0,         &
                     stmassxy(i,1))
         call emit_r('noahmp_init', trim(name), trim(stage), 'woodxy', 0,           &
                     woodxy(i,1))
         call emit_r('noahmp_init', trim(name), trim(stage), 'stblcpxy', 0,         &
                     stblcpxy(i,1))
         call emit_r('noahmp_init', trim(name), trim(stage), 'fastcpxy', 0,         &
                     fastcpxy(i,1))
         call emit_r('noahmp_init', trim(name), trim(stage), 'grainxy', 0,          &
                     grainxy(i,1))
         call emit_r('noahmp_init', trim(name), trim(stage), 'gddxy', 0,            &
                     gddxy(i,1))
         call emit_r('noahmp_init', trim(name), trim(stage), 't2mvxy', 0,           &
                     t2mvxy(i,1))
         call emit_r('noahmp_init', trim(name), trim(stage), 't2mbxy', 0,           &
                     t2mbxy(i,1))
         call emit_r('noahmp_init', trim(name), trim(stage), 'chstarxy', 0,         &
                     chstarxy(i,1))
         call emit_r('noahmp_init', trim(name), trim(stage), 'qtdrain', 0,          &
                     qtdrain(i,1))
         call emit_i('noahmp_init', trim(name), trim(stage), 'irnumsi', 0,          &
                     irnumsi(i,1))
         call emit_i('noahmp_init', trim(name), trim(stage), 'irnummi', 0,          &
                     irnummi(i,1))
         call emit_i('noahmp_init', trim(name), trim(stage), 'irnumfi', 0,          &
                     irnumfi(i,1))
         call emit_r('noahmp_init', trim(name), trim(stage), 'irwatsi', 0,          &
                     irwatsi(i,1))
         call emit_r('noahmp_init', trim(name), trim(stage), 'irwatmi', 0,          &
                     irwatmi(i,1))
         call emit_r('noahmp_init', trim(name), trim(stage), 'irwatfi', 0,          &
                     irwatfi(i,1))
         call emit_r('noahmp_init', trim(name), trim(stage), 'ireloss', 0,          &
                     ireloss(i,1))
         call emit_r('noahmp_init', trim(name), trim(stage), 'irsivol', 0,          &
                     irsivol(i,1))
         call emit_r('noahmp_init', trim(name), trim(stage), 'irmivol', 0,          &
                     irmivol(i,1))
         call emit_r('noahmp_init', trim(name), trim(stage), 'irfivol', 0,          &
                     irfivol(i,1))
         call emit_r('noahmp_init', trim(name), trim(stage), 'irrsplh', 0,          &
                     irrsplh(i,1))
         do k = 1, 5
            call emit_r('noahmp_init', trim(name), trim(stage), 'croptype', k,      &
                        croptype(i,k,1))
         end do
         do k = 1, NSOIL
            call emit_r('noahmp_init', trim(name), trim(stage), 'tslb', k,          &
                        tslb(i,k,1))
            call emit_r('noahmp_init', trim(name), trim(stage), 'smois', k,         &
                        smois(i,k,1))
            call emit_r('noahmp_init', trim(name), trim(stage), 'sh2o', k,          &
                        sh2o(i,k,1))
         end do
         do k = -2, NSOIL
            call emit_r('noahmp_init', trim(name), trim(stage), 'zsnsoxy', k,       &
                        zsnsoxy(i,k,1))
         end do
         do k = -2, 0
            call emit_r('noahmp_init', trim(name), trim(stage), 'tsnoxy', k,        &
                        tsnoxy(i,k,1))
            call emit_r('noahmp_init', trim(name), trim(stage), 'snicexy', k,       &
                        snicexy(i,k,1))
            call emit_r('noahmp_init', trim(name), trim(stage), 'snliqxy', k,       &
                        snliqxy(i,k,1))
         end do
      end do
    end subroutine emit_ni

  end subroutine dump_noahmp_init

end module driver_oracle

! ===========================================================================

program run_noahmp_driver_oracle

  use driver_oracle
  implicit none

  character(len=1024) :: csv_path

  if (command_argument_count() /= 1) then
     write(*, '(A)') 'usage: run_driver OUTPUT.csv'
     error stop 2
  end if
  call get_command_argument(1, csv_path)

  open(newunit=CSV, file=trim(csv_path), status='replace', action='write')
  write(CSV, '(A)') 'leaf,case,stage,field,index,dtype,value,bits'

  call dump_snow_init()
  call dump_noahmp_init()

  close(CSV)

end program run_noahmp_driver_oracle
