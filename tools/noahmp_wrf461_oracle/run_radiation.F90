! tools/noahmp_wrf461_oracle/run_radiation.F90
!
! Bitwise fixture generator for the Noah-MP shortwave-radiation leaves:
!   SNOW_AGE, SNOWALB_CLASS, GROUNDALB, SURRAD, TWOSTREAM, ALBEDO
!
! Linked against the *visibility-patched only* pinned WRF-4.6.1 module
! (phys/module_sf_noahmplsm.F, tree d66e442fccc04111067e29274c9f9eaccc3cef28,
! sha256 bd592a5b7db29000e715250e3a7c779ffb5e0dcc356f6b5a7d9e1c9f69c55282).
! make_public_radiation.py performs the patch; compare_object_code_radiation.py
! proves the patch changes no generated code.
!
! Every REAL is written as the hexadecimal IEEE-754 binary32 bit pattern of
! the value, so the fixture round-trips exactly and no decimal parsing is
! involved on the consuming side.  Case inputs are decimal literals here; the
! consumer never sees them, only their emitted bit patterns.
!
! Option identity is set once, from the WRF Registry defaults, and asserted.
!
! Owned by the `radiation` lane.  Do not add other subsystems' leaves here.

program run_radiation

  use module_sf_noahmplsm

  implicit none

  character(len=512) :: outdir
  integer            :: u

  call get_command_argument(1, outdir)
  if (len_trim(outdir) == 0) outdir = '.'

  ! --- pinned option identity: exact WRF Registry defaults -----------------
  !            dveg crs btr run sfc frz inf rad alb snf tbot stc rsf soil
  call noahmp_options( 4,  1,  1,  3,  1,  1,  1,  3,  2,  1,  2,  1,  1,  1, &
  !                   pedo crop irr irrm infdv tdrn
                        1,   0,  0,   0,    1,   0)

  if (opt_rad  /= 3) stop 'opt_rad  identity failed'
  if (opt_alb  /= 2) stop 'opt_alb  identity failed'
  if (dveg     /= 4) stop 'dveg     identity failed'
  if (opt_run  /= 3) stop 'opt_run  identity failed'
  if (opt_sfc  /= 1) stop 'opt_sfc  identity failed'
  if (opt_crs  /= 1) stop 'opt_crs  identity failed'
  if (opt_crop /= 0) stop 'opt_crop identity failed'
  if (opt_irr  /= 0) stop 'opt_irr  identity failed'
  if (opt_tdrn /= 0) stop 'opt_tdrn identity failed'

  u = 21
  call emit_snow_age(u, outdir)
  call emit_snowalb_class(u, outdir)
  call emit_groundalb(u, outdir)
  call emit_surrad(u, outdir)
  call emit_twostream(u, outdir)
  call emit_albedo(u, outdir)

  write(*,'(A)') 'radiation oracle written'

contains

! ---------------------------------------------------------------------------
! helpers
! ---------------------------------------------------------------------------

  function hx(x) result(s)
    real, intent(in) :: x
    character(len=10) :: s
    integer :: b
    b = transfer(x, 0)
    write(s, '("0x",Z8.8)') b
  end function hx

  function ix(i) result(s)
    integer, intent(in) :: i
    character(len=12) :: s
    write(s, '(I0)') i
    s = adjustl(s)
  end function ix

  subroutine open_leaf(u, outdir, leaf, header)
    integer, intent(in) :: u
    character(len=*), intent(in) :: outdir, leaf, header
    open(u, file=trim(outdir)//'/noahmp-radiation-'//trim(leaf)//'.csv', &
         status='replace', action='write')
    write(u,'(A)') trim(header)
  end subroutine open_leaf

! ---------------------------------------------------------------------------
! SNOW_AGE  (module_sf_noahmplsm.F:3119-3167)
!
! parameters used: TAU0, GRAIN_GROWTH, EXTRA_GROWTH, DIRT_SOOT, SWEMX
! branches bound:
!   sneqv<=0 zero-reset ; sneqv>0 normal ; arg<0 / arg=0 / arg>0 ;
!   AMIN1 both arms ; AMAX1(0,sneqv-sneqvo) both arms ; AMAX1(0,sge) both arms
! ---------------------------------------------------------------------------
  subroutine emit_snow_age(u, outdir)
    integer, intent(in) :: u
    character(len=*), intent(in) :: outdir
    integer, parameter :: NC = 13
    character(len=28), parameter :: nm(NC) = [ character(len=28) :: &
        'sa_nosnow_zero        ', 'sa_nosnow_negsneqv    ', &
        'sa_cold_accum         ', 'sa_cold_noaccum       ', &
        'sa_warm_arg_pos       ', 'sa_at_tfrz_arg_zero   ', &
        'sa_deep_cold          ', 'sa_bigdump_sge_neg    ', &
        'sa_bigdump_sge_zero   ', 'sa_old_snow_large_tau ', &
        'sa_tiny_dt            ', 'sa_hot_ground         ', &
        'sa_near_tfrz_age2_live' ]
    ! rows: TAU0, GRAIN_GROWTH, EXTRA_GROWTH, DIRT_SOOT, SWEMX,
    !       DT, TG, SNEQVO, SNEQV, TAUSS_IN
    ! Case 1 carries the exact MPTABLE.TBL defaults; every other case varies
    ! all five parameters, so a transcription that hardcoded any of them is
    ! killed by the mutation study (tools/.../mutate_radiation.py).
    real, parameter :: R(10,NC) = reshape( [ &
      1.00e6, 5000.0, 10.0,  0.30, 1.00,   90.0, 268.15,  10.0,   0.0,   3.5, &
      0.90e6, 4800.0, 11.0,  0.28, 1.10,   90.0, 268.15,  10.0,  -1.0,   3.5, &
      1.10e6, 5200.0,  9.0,  0.32, 0.90,   90.0, 258.15,  10.0,  10.4,   3.5, &
      0.85e6, 5400.0, 12.0,  0.27, 1.15,   90.0, 258.15,  10.6,  10.0,   3.5, &
      1.20e6, 4600.0,  8.0,  0.35, 0.80,   90.0, 278.15,   2.0,   2.0,   0.25, &
      0.95e6, 5100.0, 10.5,  0.31, 1.25,   90.0, 273.16,   2.0,   2.0,   0.25, &
      1.05e6, 4900.0,  9.5,  0.29, 1.05,  180.0, 233.15, 120.0, 120.0,  55.0, &
      0.80e6, 5300.0, 11.5,  0.33, 1.35,   90.0, 263.15,   1.0,   9.0,   2.0, &
      1.15e6, 4700.0, 10.2,  0.26, 0.95,   90.0, 263.15,   1.0,   2.0,   0.0, &
      1.25e6, 5050.0,  9.8,  0.34, 1.05, 3600.0, 271.00, 300.0, 300.0, 900.0, &
      0.88e6, 5150.0, 10.7,  0.24, 1.18,    1.0, 250.00,   5.0,   5.0,   0.125, &
      1.02e6, 4950.0,  9.3,  0.36, 0.92,   90.0, 300.00,   0.5,   0.5,   1.0, &
    ! TG a quarter kelvin below TFRZ: ARG ~ -0.017, so AGE2 = EXP(EG*ARG) is
    ! O(1) and EXTRA_GROWTH actually reaches the output.  Every other case has
    ! |EG*ARG| large enough that AGE2 underflows against AGE1+AGE3 in FP32.
      1.00e6, 5000.0, 13.5,  0.30, 1.00, 1800.0, 272.90,   4.0,   4.0,   0.5 ], &
      [10,NC] )
    type(noahmp_parameters) :: p
    real :: tauss, fage
    integer :: c
    call open_leaf(u, outdir, 'snow_age', &
      'case,tau0,grain_growth,extra_growth,dirt_soot,swemx,dt,tg,sneqvo,'// &
      'sneqv,tauss_in,tauss_out,fage')
    do c = 1, NC
      p%TAU0         = R(1,c)
      p%GRAIN_GROWTH = R(2,c)
      p%EXTRA_GROWTH = R(3,c)
      p%DIRT_SOOT    = R(4,c)
      p%SWEMX        = R(5,c)
      tauss = R(10,c)
      call SNOW_AGE(p, R(6,c), R(7,c), R(8,c), R(9,c), tauss, fage)
      write(u,'(30A)') trim(nm(c)), &
        ',', hx(R(1,c)), ',', hx(R(2,c)), ',', hx(R(3,c)), ',', hx(R(4,c)), &
        ',', hx(R(5,c)), ',', hx(R(6,c)), ',', hx(R(7,c)), ',', hx(R(8,c)), &
        ',', hx(R(9,c)), ',', hx(R(10,c)), ',', hx(tauss), ',', hx(fage)
    end do
    close(u)
  end subroutine emit_snow_age

! ---------------------------------------------------------------------------
! SNOWALB_CLASS  (module_sf_noahmplsm.F:3226-3275)
!
! parameters used: SWEMX.  ILOC/JLOC/NBAND carry no arithmetic (see the
! mutation study).  branches: QSNOW>0 / QSNOW<=0 ; MIN(QSNOW,SWEMX/DT) arms.
! ---------------------------------------------------------------------------
  subroutine emit_snowalb_class(u, outdir)
    integer, intent(in) :: u
    character(len=*), intent(in) :: outdir
    integer, parameter :: NC = 8
    character(len=28), parameter :: nm(NC) = [ character(len=28) :: &
        'sc_noqsnow_fresh      ', 'sc_noqsnow_aged       ', &
        'sc_qsnow_below_cap    ', 'sc_qsnow_above_cap    ', &
        'sc_qsnow_at_cap       ', 'sc_long_dt_decay      ', &
        'sc_short_dt           ', 'sc_albold_above_084   ' ]
    ! rows: SWEMX, QSNOW, DT, ALBOLD
    ! SWEMX is varied across the QSNOW>0 cases (it is dead when QSNOW<=0), so
    ! a transcription that hardcoded it is killed.  Case 5 keeps SWEMX=1.0
    ! because that is what makes QSNOW == SWEMX/DT exactly.
    real, parameter :: R(4,NC) = reshape( [ &
      1.00, 0.0,      90.0,  0.55, &
      1.00, 0.0,      90.0,  0.32, &
      0.75, 0.002,    90.0,  0.40, &
      1.40, 0.5,      90.0,  0.40, &
      1.00, 0.0111111,90.0,  0.40, &
      1.00, 0.0,    3600.0,  0.80, &
      0.60, 0.001,     1.0,  0.60, &
      1.25, 0.004,    90.0,  0.92 ], [4,NC] )
    integer, parameter :: NBAND = 2, ILOC = 3, JLOC = 7
    type(noahmp_parameters) :: p
    real :: alb, albsnd(2), albsni(2)
    integer :: c
    call open_leaf(u, outdir, 'snowalb_class', &
      'case,nband,iloc,jloc,swemx,qsnow,dt,albold,alb,albsnd1,albsnd2,'// &
      'albsni1,albsni2')
    do c = 1, NC
      p%SWEMX = R(1,c)
      alb = -999.0
      albsnd = -999.0
      albsni = -999.0
      call SNOWALB_CLASS(p, NBAND, R(2,c), R(3,c), alb, R(4,c), &
                         albsnd, albsni, ILOC, JLOC)
      write(u,'(30A)') trim(nm(c)), &
        ',', trim(ix(NBAND)), ',', trim(ix(ILOC)), ',', trim(ix(JLOC)), &
        ',', hx(R(1,c)), ',', hx(R(2,c)), ',', hx(R(3,c)), ',', hx(R(4,c)), &
        ',', hx(alb), ',', hx(albsnd(1)), ',', hx(albsnd(2)), &
        ',', hx(albsni(1)), ',', hx(albsni(2))
    end do
    close(u)
  end subroutine emit_snowalb_class

! ---------------------------------------------------------------------------
! GROUNDALB  (module_sf_noahmplsm.F:3279-3332)
!
! parameters used: ALBSAT(2), ALBDRY(2), ALBLAK(2)
! branches: IST=1 soil with MIN picking either arm ; INC clamp both arms ;
!           IST=2 & TG>TFRZ unfrozen lake (powf) with MAX(0.01,COSZ) both arms;
!           IST=2 & TG<=TFRZ frozen lake
! ---------------------------------------------------------------------------
  subroutine emit_groundalb(u, outdir)
    integer, intent(in) :: u
    character(len=*), intent(in) :: outdir
    integer, parameter :: NC = 13
    character(len=28), parameter :: nm(NC) = [ character(len=28) :: &
        'ga_soil_wet_inc_zero  ', 'ga_soil_dry_inc_pos   ', &
        'ga_soil_inc_hits_dry  ', 'ga_soil_full_snow     ', &
        'ga_soil_no_snow       ', 'ga_lake_warm_highsun  ', &
        'ga_lake_warm_lowsun   ', 'ga_lake_warm_cosz_clamp', &
        'ga_lake_frozen        ', 'ga_lake_at_tfrz       ', &
        'ga_soil_col8_min_albdry', 'ga_soil_col5_min_albsat', &
        'ga_lake_powf_discrim   ' ]
    ! rows: ALBSAT1, ALBSAT2, ALBDRY1, ALBDRY2, ALBLAK1, ALBLAK2,
    !       FSNO, SMC1, ALBSND1, ALBSND2, ALBSNI1, ALBSNI2, COSZ, TG
    real, parameter :: R(14,NC) = reshape( [ &
      0.15,0.30,0.27,0.54,0.60,0.40, 0.35, 0.400, 0.72,0.68,0.70,0.66, 0.62, 265.0, &
      0.15,0.30,0.27,0.54,0.60,0.40, 0.00, 0.080, 0.00,0.00,0.00,0.00, 0.62, 290.0, &
      0.15,0.30,0.27,0.54,0.60,0.40, 0.10, 0.005, 0.80,0.75,0.78,0.73, 0.40, 280.0, &
      0.15,0.30,0.27,0.54,0.60,0.40, 1.00, 0.250, 0.65,0.61,0.63,0.59, 0.90, 260.0, &
      0.15,0.30,0.27,0.54,0.60,0.40, 0.00, 0.320, 0.00,0.00,0.00,0.00, 0.15, 295.0, &
      0.15,0.30,0.27,0.54,0.60,0.40, 0.00, 0.410, 0.00,0.00,0.00,0.00, 0.95, 292.0, &
      0.15,0.30,0.27,0.54,0.60,0.40, 0.20, 0.410, 0.55,0.52,0.54,0.51, 0.05, 285.0, &
      0.15,0.30,0.27,0.54,0.60,0.40, 0.00, 0.410, 0.00,0.00,0.00,0.00, 0.003,283.0, &
      0.15,0.30,0.27,0.54,0.55,0.35, 0.45, 0.410, 0.70,0.66,0.68,0.64, 0.50, 262.0, &
      0.15,0.30,0.27,0.54,0.62,0.42, 0.00, 0.410, 0.00,0.00,0.00,0.00, 0.50, 273.16, &
    ! soil colour class 8 (ALBDRY-ALBSAT = 0.05 < INC_max) -> MIN picks ALBDRY
      0.05,0.10,0.10,0.20,0.60,0.40, 0.00, 0.000, 0.00,0.00,0.00,0.00, 0.55, 288.0, &
    ! soil colour class 5 at moderate wetness -> MIN picks ALBSAT+INC
      0.08,0.16,0.16,0.32,0.60,0.40, 0.15, 0.100, 0.71,0.67,0.69,0.65, 0.55, 271.0, &
    ! COSZ chosen so that the *final* ALBSOD -- not merely the raw
    ! MAX(0.01,COSZ)**1.7 -- differs between glibc powf and CUDA powf.  Picking
    ! a COSZ where only the raw powf differs is not enough: the 1-ulp
    ! difference is attenuated by 0.06/(x+0.15) and usually rounds away.  This
    ! row is what makes the CUDA negative control for powf fire; without it the
    ! fixture passes a kernel that calls CUDA powf.
      0.15,0.30,0.27,0.54,0.60,0.40, 0.00, 0.410, 0.00,0.00,0.00,0.00, &
      0.300001085, 289.0 ], &
      [14,NC] )
    integer, parameter :: IST_(NC) = [ 1,1,1,1,1, 2,2,2, 2,2, 1,1, 2 ]
    integer, parameter :: NSOIL = 4, NBAND = 2, ICE = 0, ILOC = 3, JLOC = 7
    type(noahmp_parameters) :: p
    real :: smc(NSOIL), albsnd(2), albsni(2), albgrd(2), albgri(2)
    integer :: c, k
    call open_leaf(u, outdir, 'groundalb', &
      'case,nsoil,nband,ice,ist,iloc,jloc,albsat1,albsat2,albdry1,albdry2,'// &
      'alblak1,alblak2,fsno,smc1,albsnd1,albsnd2,albsni1,albsni2,cosz,tg,'// &
      'albgrd1,albgrd2,albgri1,albgri2')
    do c = 1, NC
      p%ALBSAT(1) = R(1,c) ; p%ALBSAT(2) = R(2,c)
      p%ALBDRY(1) = R(3,c) ; p%ALBDRY(2) = R(4,c)
      p%ALBLAK(1) = R(5,c) ; p%ALBLAK(2) = R(6,c)
      smc = 0.0
      smc(1) = R(8,c)
      do k = 2, NSOIL
        smc(k) = 0.11 * real(k)      ! must not influence the answer
      end do
      albsnd(1) = R(9,c)  ; albsnd(2) = R(10,c)
      albsni(1) = R(11,c) ; albsni(2) = R(12,c)
      albgrd = -999.0 ; albgri = -999.0
      call GROUNDALB(p, NSOIL, NBAND, ICE, IST_(c), R(7,c), smc, &
                     albsnd, albsni, R(13,c), R(14,c), ILOC, JLOC, &
                     albgrd, albgri)
      write(u,'(60A)') trim(nm(c)), &
        ',', trim(ix(NSOIL)), ',', trim(ix(NBAND)), ',', trim(ix(ICE)), &
        ',', trim(ix(IST_(c))), ',', trim(ix(ILOC)), ',', trim(ix(JLOC)), &
        ',', hx(R(1,c)), ',', hx(R(2,c)), ',', hx(R(3,c)), ',', hx(R(4,c)), &
        ',', hx(R(5,c)), ',', hx(R(6,c)), ',', hx(R(7,c)), ',', hx(R(8,c)), &
        ',', hx(R(9,c)), ',', hx(R(10,c)),',', hx(R(11,c)),',', hx(R(12,c)), &
        ',', hx(R(13,c)),',', hx(R(14,c)), &
        ',', hx(albgrd(1)), ',', hx(albgrd(2)), &
        ',', hx(albgri(1)), ',', hx(albgri(2))
    end do
    close(u)
  end subroutine emit_groundalb

! ---------------------------------------------------------------------------
! SURRAD  (module_sf_noahmplsm.F:2994-3115)
!
! pure arithmetic, no libm.  branches: FSUN>0 / FSUN<=0 ; MAX(VAI,MPE),
! MAX(LAISUN,MPE), MAX(LAISHA,MPE) each on both arms.
! ---------------------------------------------------------------------------
  subroutine emit_surrad(u, outdir)
    integer, intent(in) :: u
    character(len=*), intent(in) :: outdir
    integer, parameter :: NC = 8, NR = 37
    character(len=28), parameter :: nm(NC) = [ character(len=28) :: &
        'sr_sunlit_forest      ', 'sr_fsun_zero_shade    ', &
        'sr_bare_vai_zero      ', 'sr_laisun_below_mpe   ', &
        'sr_laisha_below_mpe   ', 'sr_night_zero_flux    ', &
        'sr_high_sun_grass     ', 'sr_snow_bright_ground ' ]
    ! rows 1..37:
    !  1 MPE  2 FSUN  3 FSHA  4 ELAI  5 VAI  6 LAISUN  7 LAISHA
    !  8- 9 SOLAD  10-11 SOLAI  12-13 FABD  14-15 FABI
    ! 16-17 FTDD   18-19 FTID  20-21 FTII  22-23 ALBGRD  24-25 ALBGRI
    ! 26-27 ALBD   28-29 ALBI  30-31 FREVD 32-33 FREVI  34-35 FREGD
    ! 36-37 FREGI
    real, parameter :: R(NR,NC) = reshape( [ &
      1.0e-6, 0.62, 0.38, 3.40, 3.90, 2.108, 1.292, &
        420.0, 380.0, 90.0, 70.0, 0.72, 0.41, 0.68, 0.38, &
        0.081, 0.152, 0.140, 0.201, 0.190, 0.271, 0.121, 0.223, &
        0.118, 0.219, 0.098, 0.281, 0.101, 0.288, 0.061, 0.192, &
        0.064, 0.198, 0.037, 0.089, 0.038, 0.091, &
      1.0e-6, 0.00, 1.00, 2.10, 2.60, 0.000, 2.100, &
        260.0, 240.0, 140.0, 120.0, 0.61, 0.35, 0.59, 0.33, &
        0.121, 0.196, 0.180, 0.240, 0.230, 0.310, 0.161, 0.263, &
        0.158, 0.259, 0.130, 0.301, 0.133, 0.308, 0.081, 0.212, &
        0.084, 0.218, 0.049, 0.109, 0.050, 0.111, &
      1.0e-6, 0.00, 1.00, 0.00, 0.00, 0.000, 0.000, &
        700.0, 620.0, 60.0, 50.0, 0.00, 0.00, 0.00, 0.00, &
        1.000, 1.000, 0.000, 0.000, 1.000, 1.000, 0.211, 0.402, &
        0.211, 0.402, 0.211, 0.402, 0.211, 0.402, 0.000, 0.000, &
        0.000, 0.000, 0.211, 0.402, 0.211, 0.402, &
      2.0e-6, 0.004, 0.996, 0.030, 0.045, 1.2e-7, 0.0299, &
        510.0, 470.0, 75.0, 62.0, 0.021, 0.013, 0.020, 0.012, &
        0.951, 0.962, 0.020, 0.017, 0.958, 0.966, 0.181, 0.352, &
        0.181, 0.352, 0.190, 0.362, 0.192, 0.366, 0.009, 0.010, &
        0.011, 0.013, 0.181, 0.352, 0.181, 0.352, &
      5.0e-7, 0.00, 1.00, 0.020, 0.030, 0.000, 3.0e-8, &
        330.0, 300.0, 110.0, 95.0, 0.014, 0.009, 0.013, 0.008, &
        0.967, 0.974, 0.014, 0.012, 0.971, 0.977, 0.161, 0.322, &
        0.161, 0.322, 0.168, 0.330, 0.170, 0.333, 0.006, 0.007, &
        0.008, 0.009, 0.161, 0.322, 0.161, 0.322, &
      1.0e-6, 0.00, 1.00, 1.80, 2.20, 0.000, 1.800, &
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, &
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, &
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, &
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, &
      1.0e-6, 0.93, 0.07, 1.20, 1.35, 1.116, 0.084, &
        880.0, 790.0, 45.0, 38.0, 0.51, 0.24, 0.47, 0.21, &
        0.310, 0.421, 0.190, 0.241, 0.410, 0.521, 0.141, 0.263, &
        0.138, 0.259, 0.152, 0.311, 0.155, 0.318, 0.101, 0.212, &
        0.104, 0.218, 0.037, 0.089, 0.038, 0.091, &
      1.0e-6, 0.44, 0.56, 0.60, 1.60, 0.264, 0.336, &
        600.0, 540.0, 200.0, 180.0, 0.28, 0.16, 0.26, 0.15, &
        0.520, 0.611, 0.150, 0.171, 0.610, 0.701, 0.780, 0.740, &
        0.760, 0.720, 0.630, 0.590, 0.640, 0.600, 0.140, 0.130, &
        0.150, 0.140, 0.490, 0.460, 0.490, 0.460 ], [NR,NC] )
    integer, parameter :: ILOC = 3, JLOC = 7
    type(noahmp_parameters) :: p
    real :: solad(2), solai(2), fabd(2), fabi(2), ftdd(2), ftid(2), ftii(2)
    real :: albgrd(2), albgri(2), albd(2), albi(2)
    real :: frevd(2), frevi(2), fregd(2), fregi(2)
    real :: parsun, parsha, sav, sag, fsa, fsr, fsrv, fsrg
    integer :: c, k
    call open_leaf(u, outdir, 'surrad', &
      'case,iloc,jloc,mpe,fsun,fsha,elai,vai,laisun,laisha,'// &
      'solad1,solad2,solai1,solai2,fabd1,fabd2,fabi1,fabi2,'// &
      'ftdd1,ftdd2,ftid1,ftid2,ftii1,ftii2,albgrd1,albgrd2,albgri1,albgri2,'// &
      'albd1,albd2,albi1,albi2,frevd1,frevd2,frevi1,frevi2,'// &
      'fregd1,fregd2,fregi1,fregi2,'// &
      'parsun,parsha,sav,sag,fsa,fsr,fsrv,fsrg')
    do c = 1, NC
      solad = R( 8:9 ,c) ; solai = R(10:11,c)
      fabd  = R(12:13,c) ; fabi  = R(14:15,c)
      ftdd  = R(16:17,c) ; ftid  = R(18:19,c) ; ftii = R(20:21,c)
      albgrd= R(22:23,c) ; albgri= R(24:25,c)
      albd  = R(26:27,c) ; albi  = R(28:29,c)
      frevd = R(30:31,c) ; frevi = R(32:33,c)
      fregd = R(34:35,c) ; fregi = R(36:37,c)
      parsun = -999.0 ; parsha = -999.0
      sav = -999.0 ; sag = -999.0 ; fsa = -999.0 ; fsr = -999.0
      fsrv = -999.0 ; fsrg = -999.0
      call SURRAD(p, R(1,c), R(2,c), R(3,c), R(4,c), R(5,c), &
                  R(6,c), R(7,c), solad, solai, fabd, &
                  fabi, ftdd, ftid, ftii, albgrd, &
                  albgri, albd, albi, ILOC, JLOC, &
                  parsun, parsha, sav, sag, fsa, &
                  fsr, &
                  frevi, frevd, fregd, fregi, fsrv, &
                  fsrg)
      write(u,'(A)',advance='no') trim(nm(c))
      write(u,'(4A)',advance='no') ',', trim(ix(ILOC)), ',', trim(ix(JLOC))
      do k = 1, NR
        write(u,'(2A)',advance='no') ',', hx(R(k,c))
      end do
      write(u,'(16A)') &
        ',', hx(parsun), ',', hx(parsha), ',', hx(sav), ',', hx(sag), &
        ',', hx(fsa), ',', hx(fsr), ',', hx(fsrv), ',', hx(fsrg)
    end do
    close(u)
  end subroutine emit_surrad

! ---------------------------------------------------------------------------
! TWOSTREAM  (module_sf_noahmplsm.F:3336-3574) under OPT_RAD=3
!
! parameters used: XL, OMEGAS(2), BETADS, BETAIS.
! OPT_RAD=1 (RC/HVT/HVB/DEN, ACOS/ATAN/TAN/COS) and OPT_RAD=2 are dead under
! the pinned identity and are not exercised.
! branches: VAI=0 gap ; VAI>0 gap=1-FVEG ; CHIL clamps low/high/|chil|<=0.01 ;
!           T>TFRZ / T<=TFRZ snow adjustment ; |SIGMA|<1e-6 both signs ;
!           IC=0 / IC=1 ; IB=1 / IB=2.
! ---------------------------------------------------------------------------
  subroutine emit_twostream(u, outdir)
    integer, intent(in) :: u
    character(len=*), intent(in) :: outdir
    integer, parameter :: NC = 24, NR = 25
    character(len=28), parameter :: nm(NC) = [ character(len=28) :: &
        'ts_vis_dir_forest_nosnow', 'ts_vis_dif_forest_nosnow', &
        'ts_nir_dir_forest_nosnow', 'ts_nir_dif_forest_nosnow', &
        'ts_vis_dir_snow_wet     ', 'ts_vis_dif_snow_wet     ', &
        'ts_nir_dir_snow_lowfwet ', 'ts_nir_dif_snow_fwet0   ', &
        'ts_vai_zero_dir         ', 'ts_vai_zero_dif         ', &
        'ts_chil_clamp_low       ', 'ts_chil_clamp_high      ', &
        'ts_chil_eps_from_zero   ', 'ts_chil_eps_from_small  ', &
        'ts_sigma_neg_far        ', 'ts_sigma_neg_mid        ', &
        'ts_sigma_neg_near       ', 'ts_sigma_pos_near       ', &
        'ts_sigma_pos_mid        ', 'ts_sigma_pos_far        ', &
        'ts_fveg_one_nogap       ', 'ts_lowsun_cosz_clamp    ', &
        'ts_logf_discrim_asu     ', 'ts_logf_discrim_avmu    ' ]
    ! rows 1..25:
    !  1 XL  2 OMEGAS1  3 OMEGAS2  4 BETADS  5 BETAIS
    !  6 COSZ  7 VAI  8 FWET  9 T  10 FVEG
    ! 11-12 ALBGRD  13-14 ALBGRI  15-16 RHO  17-18 TAU
    ! 19-20 FAB_in  21-22 FRE_in  23 GDIR_in  24 BGAP_in  25 WGAP_in
    real, parameter :: R(NR,NC) = reshape( [ &
    ! 1 vis direct, no snow
      -0.30,0.8,0.4,0.5,0.5, 0.62,3.90,0.05,285.0,0.78, &
        0.171,0.322,0.169,0.318, 0.1276,0.5545, 0.0885,0.2510, &
        -9.0,-9.0,-9.0,-9.0,-9.0,0.0,0.0, &
    ! 2 vis diffuse, no snow
      -0.30,0.8,0.4,0.5,0.5, 0.62,3.90,0.05,285.0,0.78, &
        0.171,0.322,0.169,0.318, 0.1276,0.5545, 0.0885,0.2510, &
        -9.0,-9.0,-9.0,-9.0,-9.0,0.0,0.0, &
    ! 3 nir direct, no snow
      -0.30,0.8,0.4,0.5,0.5, 0.62,3.90,0.05,285.0,0.78, &
        0.171,0.322,0.169,0.318, 0.1276,0.5545, 0.0885,0.2510, &
        -9.0,-9.0,-9.0,-9.0,-9.0,0.0,0.0, &
    ! 4 nir diffuse, no snow
      -0.30,0.8,0.4,0.5,0.5, 0.62,3.90,0.05,285.0,0.78, &
        0.171,0.322,0.169,0.318, 0.1276,0.5545, 0.0885,0.2510, &
        -9.0,-9.0,-9.0,-9.0,-9.0,0.0,0.0, &
    ! 5 vis direct, intercepted snow, wet canopy (OMEGAS/BETADS/BETAIS live)
      -0.30,0.72,0.46,0.55,0.44, 0.41,2.55,0.83,266.0,0.62, &
        0.610,0.520,0.600,0.510, 0.1601,0.5652, 0.1105,0.2790, &
        -9.0,-9.0,-9.0,-9.0,-9.0,0.0,0.0, &
    ! 6 vis diffuse, intercepted snow, wet canopy (OMEGAS/BETADS/BETAIS live)
      -0.30,0.72,0.46,0.55,0.44, 0.41,2.55,0.83,266.0,0.62, &
        0.610,0.520,0.600,0.510, 0.1601,0.5652, 0.1105,0.2790, &
        -9.0,-9.0,-9.0,-9.0,-9.0,0.0,0.0, &
    ! 7 nir direct, cold canopy, lightly wetted
      0.25,0.86,0.33,0.47,0.58, 0.88,1.15,0.18,262.0,0.35, &
        0.480,0.430,0.470,0.420, 0.1024,0.4530, 0.0510,0.1055, &
        -9.0,-9.0,-9.0,-9.0,-9.0,0.0,0.0, &
    ! 8 nir diffuse, cold canopy, FWET == 0 under the snow branch
      0.25,0.86,0.33,0.47,0.58, 0.88,1.15,0.0,262.0,0.35, &
        0.480,0.430,0.470,0.420, 0.1024,0.4530, 0.0510,0.1055, &
        -9.0,-9.0,-9.0,-9.0,-9.0,0.0,0.0, &
    ! 9 VAI == 0 direct
      -0.30,0.8,0.4,0.5,0.5, 0.55,0.00,0.0,290.0,0.10, &
        0.190,0.360,0.188,0.356, 0.000001,0.000001, 0.000001,0.000001, &
        -9.0,-9.0,-9.0,-9.0,-9.0,0.0,0.0, &
    ! 10 VAI == 0 diffuse
      -0.30,0.8,0.4,0.5,0.5, 0.55,0.00,0.0,290.0,0.10, &
        0.190,0.360,0.188,0.356, 0.000001,0.000001, 0.000001,0.000001, &
        -9.0,-9.0,-9.0,-9.0,-9.0,0.0,0.0, &
    ! 11 CHIL clamped at -0.4
      -0.90,0.8,0.4,0.5,0.5, 0.70,2.80,0.10,288.0,0.55, &
        0.150,0.300,0.148,0.296, 0.1100,0.5800, 0.0700,0.2500, &
        -9.0,-9.0,-9.0,-9.0,-9.0,0.0,0.0, &
    ! 12 CHIL clamped at +0.6
      0.95,0.8,0.4,0.5,0.5, 0.70,2.80,0.10,288.0,0.55, &
        0.150,0.300,0.148,0.296, 0.1100,0.5800, 0.0700,0.2500, &
        -9.0,-9.0,-9.0,-9.0,-9.0,0.0,0.0, &
    ! 13 CHIL forced to 0.01 from exactly 0
      0.00,0.8,0.4,0.5,0.5, 0.70,2.80,0.10,288.0,0.55, &
        0.150,0.300,0.148,0.296, 0.1100,0.5800, 0.0700,0.2500, &
        -9.0,-9.0,-9.0,-9.0,-9.0,0.0,0.0, &
    ! 14 CHIL forced to 0.01 from -0.005
      -0.005,0.8,0.4,0.5,0.5, 0.70,2.80,0.10,288.0,0.55, &
        0.150,0.300,0.148,0.296, 0.1100,0.5800, 0.0700,0.2500, &
        -9.0,-9.0,-9.0,-9.0,-9.0,0.0,0.0, &
    ! 15 SIGMA ~ -3e-6 (outside the clamp; boundary control)
      -0.30,0.8,0.4,0.5,0.5, 0.5875767469,2.10,0.0,290.0,0.60, &
        0.180,0.340,0.178,0.336, 0.1100,0.5800, 0.0700,0.2500, &
        -9.0,-9.0,-9.0,-9.0,-9.0,0.0,0.0, &
    ! 16 SIGMA ~ -5e-7 (inside the clamp)
      -0.30,0.8,0.4,0.5,0.5, 0.5875760913,2.10,0.0,290.0,0.60, &
        0.180,0.340,0.178,0.336, 0.1100,0.5800, 0.0700,0.2500, &
        -9.0,-9.0,-9.0,-9.0,-9.0,0.0,0.0, &
    ! 17 SIGMA ~ -2.7e-7 (inside the clamp)
      -0.30,0.8,0.4,0.5,0.5, 0.5875760317,2.10,0.0,290.0,0.60, &
        0.180,0.340,0.178,0.336, 0.1100,0.5800, 0.0700,0.2500, &
        -9.0,-9.0,-9.0,-9.0,-9.0,0.0,0.0, &
    ! 18 SIGMA ~ +4e-7 (inside the clamp)
      -0.30,0.8,0.4,0.5,0.5, 0.5875758529,2.10,0.0,290.0,0.60, &
        0.180,0.340,0.178,0.336, 0.1100,0.5800, 0.0700,0.2500, &
        -9.0,-9.0,-9.0,-9.0,-9.0,0.0,0.0, &
    ! 19 SIGMA ~ +6e-7 (inside the clamp)
      -0.30,0.8,0.4,0.5,0.5, 0.5875757933,2.10,0.0,290.0,0.60, &
        0.180,0.340,0.178,0.336, 0.1100,0.5800, 0.0700,0.2500, &
        -9.0,-9.0,-9.0,-9.0,-9.0,0.0,0.0, &
    ! 20 SIGMA ~ +3e-6 (outside the clamp; boundary control)
      -0.30,0.8,0.4,0.5,0.5, 0.5875751376,2.10,0.0,290.0,0.60, &
        0.180,0.340,0.178,0.336, 0.1100,0.5800, 0.0700,0.2500, &
        -9.0,-9.0,-9.0,-9.0,-9.0,0.0,0.0, &
    ! 21 FVEG = 1 -> GAP = KOPEN = 0
      -0.30,0.8,0.4,0.5,0.5, 0.75,4.50,0.25,281.0,1.00, &
        0.200,0.380,0.198,0.376, 0.1276,0.5545, 0.0885,0.2510, &
        -9.0,-9.0,-9.0,-9.0,-9.0,0.0,0.0, &
    ! 22 COSZ below 0.001 -> COSZI clamp
      -0.30,0.8,0.4,0.5,0.5, 0.0002,1.90,0.0,287.0,0.45, &
        0.220,0.400,0.218,0.396, 0.1100,0.5800, 0.0700,0.2500, &
        -9.0,-9.0,-9.0,-9.0,-9.0,0.0,0.0, &
    ! 23 COSZ chosen so ASU's LOG((TMP1+TMP0)/TMP1) lands where CUDA logf and
    !    glibc logf disagree; without it the CUDA logf negative control does
    !    not fire and the fixture would pass a kernel calling CUDA logf.
      -0.30,0.8,0.4,0.5,0.5, 0.550001085,2.30,0.0,288.0,0.58, &
        0.190,0.350,0.188,0.346, 0.1100,0.5800, 0.0700,0.2500, &
        -9.0,-9.0,-9.0,-9.0,-9.0,0.0,0.0, &
    ! 24 XL chosen so AVMU's LOG((PHI1+PHI2)/PHI1) -- which depends on CHIL
    !    alone -- lands on a discriminating argument.
      0.200001121,0.8,0.4,0.5,0.5, 0.64,2.05,0.0,288.0,0.52, &
        0.205,0.365,0.203,0.361, 0.1100,0.5800, 0.0700,0.2500, &
        -9.0,-9.0,-9.0,-9.0,-9.0,0.0,0.0 ], [NR,NC] )
    integer, parameter :: IB_(NC) = &
      [ 1,1,2,2, 1,1,2,2, 1,1, 1,1,1,1, 1,1,1,1,1,1, 2,1, 1,1 ]
    integer, parameter :: IC_(NC) = &
      [ 0,1,0,1, 0,1,0,1, 0,1, 0,0,0,0, 0,0,0,0,0,0, 1,0, 0,1 ]
    integer, parameter :: VEGTYP = 11, IST = 1, ILOC = 3, JLOC = 7
    type(noahmp_parameters) :: p
    real :: albgrd(2), albgri(2), rho(2), tau(2)
    real :: fab(2), fre(2), ftd(2), fti(2), frev(2), freg(2)
    real :: fab_in(2), fre_in(2), ftd_in(2), fti_in(2)
    real :: frev_in(2), freg_in(2)
    real :: gdir, bgap, wgap, gdir_in, bgap_in, wgap_in, cc
    integer :: c, k
    call open_leaf(u, outdir, 'twostream', &
      'case,ib,ic,vegtyp,ist,iloc,jloc,'// &
      'xl,omegas1,omegas2,betads,betais,cosz,vai,fwet,t,fveg,'// &
      'albgrd1,albgrd2,albgri1,albgri2,rho1,rho2,tau1,tau2,'// &
      'fab_in1,fab_in2,fre_in1,fre_in2,ftd_in1,ftd_in2,fti_in1,fti_in2,'// &
      'gdir_in,frev_in1,frev_in2,freg_in1,freg_in2,bgap_in,wgap_in,'// &
      'fab1,fab2,fre1,fre2,ftd1,ftd2,fti1,fti2,gdir,'// &
      'frev1,frev2,freg1,freg2,bgap,wgap')
    do c = 1, NC
      p%XL        = R(1,c)
      p%OMEGAS(1) = R(2,c) ; p%OMEGAS(2) = R(3,c)
      p%BETADS    = R(4,c) ; p%BETAIS    = R(5,c)
      albgrd = R(11:12,c) ; albgri = R(13:14,c)
      rho    = R(15:16,c) ; tau    = R(17:18,c)
      ! Every INOUT slot is seeded with a distinct value that also varies from
      ! case to case.  Distinct across slots catches a transcription that
      ! writes the wrong band; varying down the fixture is what lets the
      ! mutation study kill a transcription that ignores the entry value of a
      ! pass-through slot (an earlier fixture held them constant and every one
      ! of those mutants survived).
      cc = real(c)
      fab  = [  -9.0 -  1.0*cc, -10.0 -  2.0*cc ]
      fre  = [ -11.0 -  3.0*cc, -12.0 -  4.0*cc ]
      ftd  = [  -7.0 -  5.0*cc,  -8.0 -  6.0*cc ]
      fti  = [  -6.0 -  7.0*cc,  -5.0 -  8.0*cc ]
      frev = [  -4.0 -  9.0*cc,  -3.0 - 10.0*cc ]
      freg = [  -2.0 - 11.0*cc,  -1.0 - 12.0*cc ]
      gdir = -9.0 - cc
      bgap = 0.375 + 0.01*cc
      wgap = 0.625 - 0.01*cc
      fab_in = fab ; fre_in = fre ; ftd_in = ftd ; fti_in = fti
      frev_in = frev ; freg_in = freg
      gdir_in = gdir ; bgap_in = bgap ; wgap_in = wgap
      call TWOSTREAM(p, IB_(c), IC_(c), VEGTYP, R(6,c), R(7,c), &
                     R(8,c), R(9,c), albgrd, albgri, rho, &
                     tau, R(10,c), IST, ILOC, JLOC, &
                     fab, fre, ftd, fti, gdir, &
                     frev, freg, bgap, wgap)
      write(u,'(A)',advance='no') trim(nm(c))
      write(u,'(12A)',advance='no') ',', trim(ix(IB_(c))), ',', trim(ix(IC_(c))), &
        ',', trim(ix(VEGTYP)), ',', trim(ix(IST)), ',', trim(ix(ILOC)), &
        ',', trim(ix(JLOC))
      do k = 1, 18
        write(u,'(2A)',advance='no') ',', hx(R(k,c))
      end do
      write(u,'(30A)',advance='no') &
        ',', hx(fab_in(1)),  ',', hx(fab_in(2)), &
        ',', hx(fre_in(1)),  ',', hx(fre_in(2)), &
        ',', hx(ftd_in(1)),  ',', hx(ftd_in(2)), &
        ',', hx(fti_in(1)),  ',', hx(fti_in(2)), &
        ',', hx(gdir_in), &
        ',', hx(frev_in(1)), ',', hx(frev_in(2)), &
        ',', hx(freg_in(1)), ',', hx(freg_in(2)), &
        ',', hx(bgap_in),    ',', hx(wgap_in)
      write(u,'(32A)') &
        ',', hx(fab(1)), ',', hx(fab(2)), ',', hx(fre(1)), ',', hx(fre(2)), &
        ',', hx(ftd(1)), ',', hx(ftd(2)), ',', hx(fti(1)), ',', hx(fti(2)), &
        ',', hx(gdir), &
        ',', hx(frev(1)), ',', hx(frev(2)), ',', hx(freg(1)), ',', hx(freg(2)), &
        ',', hx(bgap), ',', hx(wgap)
    end do
    close(u)
  end subroutine emit_twostream

! ---------------------------------------------------------------------------
! ALBEDO  (module_sf_noahmplsm.F:2810-2990) under OPT_ALB=2, OPT_RAD=3
!
! composite: calls SNOW_AGE, SNOWALB_CLASS, GROUNDALB and TWOSTREAM x4.
! SNOWALB_BATS is dead (OPT_ALB=2) and is asserted off, never called.
! branches: COSZ<=0 early exit ; COSZ>0 full path ; FSUN<0.01 zeroing ;
!           IST=1 soil / IST=2 lake ground albedo ; T>TFRZ / T<=TFRZ.
! ---------------------------------------------------------------------------
  subroutine emit_albedo(u, outdir)
    integer, intent(in) :: u
    character(len=*), intent(in) :: outdir
    integer, parameter :: NC = 10, NR = 43, NSOIL = 4
    character(len=28), parameter :: nm(NC) = [ character(len=28) :: &
        'ab_night_cosz_zero    ', 'ab_night_cosz_neg     ', &
        'ab_day_forest_warm    ', 'ab_day_forest_cold    ', &
        'ab_day_snowpack       ', 'ab_day_fresh_snowfall ', &
        'ab_day_bare_lowvai    ', 'ab_day_lake_unfrozen  ', &
        'ab_day_lake_frozen    ', 'ab_lowsun_fsun_clamp  ' ]
    !  1- 5 TAU0 GRAIN_GROWTH EXTRA_GROWTH DIRT_SOOT SWEMX
    !  6- 7 ALBSAT   8- 9 ALBDRY  10-11 ALBLAK
    ! 12-13 RHOL    14-15 RHOS    16-17 TAUL   18-19 TAUS
    ! 20    XL      21-22 OMEGAS  23 BETADS    24 BETAIS
    ! 25 DT  26 COSZ  27 ELAI  28 ESAI  29 TG  30 TV  31 SNOWH
    ! 32 FSNO  33 FWET  34 SNEQVO  35 SNEQV  36 QSNOW  37 FVEG
    ! 38-41 SMC(1:4)  42 ALBOLD  43 TAUSS
    real, parameter :: R(NR,NC) = reshape( [ &
    ! 1 night, cosz == 0
      1.00e6,5000.0,10.0,0.30,1.00, 0.15,0.30, 0.27,0.54, 0.60,0.40, &
      0.11,0.58, 0.36,0.58, 0.07,0.25, 0.220,0.380, -0.30, 0.80,0.40, 0.50,0.50, &
      90.0, 0.0, 3.4,0.5, 281.0,282.0, 0.0, 0.0,0.05, 0.0,0.0, 0.0, 0.78, &
      0.28,0.30,0.32,0.34, 0.55, 0.0, &
    ! 2 night, cosz < 0
      0.90e6,4800.0,11.0,0.28,1.10, 0.11,0.22, 0.22,0.44, 0.55,0.35, &
      0.10,0.45, 0.16,0.39, 0.05,0.25, 0.001,0.001, 0.25, 0.75,0.45, 0.52,0.48, &
      90.0, -0.12, 3.4,0.5, 281.0,282.0, 0.0, 0.0,0.05, 0.0,0.0, 0.0, 0.78, &
      0.28,0.30,0.32,0.34, 0.55, 0.0, &
    ! 3 day, warm forest, no snow
      1.10e6,5200.0, 9.0,0.32,0.90, 0.10,0.20, 0.20,0.40, 0.58,0.38, &
      0.11,0.58, 0.36,0.58, 0.07,0.25, 0.220,0.380, -0.30, 0.82,0.42, 0.51,0.49, &
      90.0, 0.62, 3.4,0.5, 289.0,290.0, 0.0, 0.0,0.05, 0.0,0.0, 0.0, 0.78, &
      0.28,0.30,0.32,0.34, 0.55, 0.0, &
    ! 4 day, cold forest canopy (TWOSTREAM snow adjustment on)
      0.85e6,5400.0,12.0,0.27,1.15, 0.09,0.18, 0.18,0.36, 0.62,0.42, &
      0.105,0.515, 0.26,0.485, 0.06,0.25, 0.1105,0.1905, -0.025, 0.78,0.38, 0.47,0.53, &
      90.0, 0.48, 3.4,0.5, 265.0,266.0, 120.0, 0.55,0.63, 40.0,40.0, 0.0, 0.78, &
      0.28,0.30,0.32,0.34, 0.42, 61.0, &
    ! 5 day, deep snowpack, no new snow
      1.20e6,4600.0, 8.0,0.35,0.80, 0.08,0.16, 0.16,0.32, 0.56,0.36, &
      0.10,0.45, 0.16,0.39, 0.05,0.10, 0.001,0.001, 0.010, 0.85,0.35, 0.55,0.45, &
      1800.0, 0.35, 0.9,0.4, 258.0,259.0, 900.0, 1.0,0.0, 210.0,210.0, 0.0, 0.55, &
      0.30,0.31,0.32,0.33, 0.38, 240.0, &
    ! 6 day, fresh snowfall (QSNOW > 0 branch of SNOWALB_CLASS)
      0.95e6,5100.0,10.5,0.31,1.25, 0.07,0.14, 0.14,0.28, 0.59,0.39, &
      0.07,0.35, 0.16,0.39, 0.05,0.10, 0.001,0.001, 0.010, 0.80,0.40, 0.50,0.50, &
      90.0, 0.51, 1.6,0.4, 268.0,269.0, 45.0, 0.72,0.40, 12.0,14.5, 0.004, 0.62, &
      0.29,0.30,0.31,0.32, 0.47, 8.0, &
    ! 7 day, near-bare, tiny VAI
      1.05e6,4900.0, 9.5,0.29,1.05, 0.06,0.12, 0.12,0.16, 0.57,0.37, &
      0.10,0.45, 0.16,0.39, 0.05,0.25, 0.001,0.001, 0.25, 0.77,0.43, 0.53,0.47, &
      90.0, 0.74, 0.02,0.01, 297.0,298.0, 0.0, 0.0,0.0, 0.0,0.0, 0.0, 0.05, &
      0.06,0.08,0.10,0.12, 0.55, 0.0, &
    ! 8 day, unfrozen lake (GROUNDALB powf branch)
      1.00e6,5000.0,10.0,0.30,1.00, 0.05,0.10, 0.10,0.20, 0.60,0.40, &
      0.11,0.58, 0.36,0.58, 0.07,0.25, 0.220,0.380, -0.30, 0.80,0.40, 0.50,0.50, &
      90.0, 0.66, 0.5,0.2, 291.0,292.0, 0.0, 0.0,0.0, 0.0,0.0, 0.0, 0.20, &
      0.41,0.41,0.41,0.41, 0.55, 0.0, &
    ! 9 day, frozen lake
      0.80e6,5300.0,11.5,0.33,1.35, 0.15,0.30, 0.27,0.54, 0.52,0.32, &
      0.10,0.45, 0.16,0.39, 0.05,0.25, 0.001,0.001, 0.25, 0.79,0.41, 0.49,0.51, &
      1800.0, 0.44, 0.5,0.2, 272.90,262.0, 60.0, 0.9,0.0, 30.0,30.0, 0.0, 0.20, &
      0.41,0.41,0.41,0.41, 0.50, 0.5, &
    ! 10 very low sun, FSUN below 0.01 -> clamp to zero
      1.15e6,4700.0,10.2,0.26,0.95, 0.11,0.22, 0.22,0.44, 0.61,0.41, &
      0.11,0.58, 0.36,0.58, 0.07,0.25, 0.220,0.380, -0.30, 0.83,0.37, 0.54,0.46, &
      90.0, 0.0009, 6.0,1.2, 283.0,284.0, 0.0, 0.0,0.0, 0.0,0.0, 0.0, 0.95, &
      0.28,0.30,0.32,0.34, 0.55, 0.0 ], [NR,NC] )
    integer, parameter :: IST_(NC) = [ 1,1,1,1,1, 1,1, 2,2, 1 ]
    integer, parameter :: VEGTYP = 11, ICE = 0, ILOC = 3, JLOC = 7
    type(noahmp_parameters) :: p
    real :: smc(NSOIL)
    real :: albgrd(2), albgri(2), albd(2), albi(2)
    real :: fabd(2), fabi(2), ftdd(2), ftid(2), ftii(2)
    real :: frevd(2), frevi(2), fregd(2), fregi(2)
    real :: frevd_in(2), frevi_in(2), fregd_in(2), fregi_in(2)
    real :: albsnd(2), albsni(2)
    real :: fage, albold, tauss, fsun, bgap, wgap, fage_in, cc
    integer :: c, k
    call open_leaf(u, outdir, 'albedo', &
      'case,vegtyp,ist,ice,nsoil,iloc,jloc,'// &
      'tau0,grain_growth,extra_growth,dirt_soot,swemx,'// &
      'albsat1,albsat2,albdry1,albdry2,alblak1,alblak2,'// &
      'rhol1,rhol2,rhos1,rhos2,taul1,taul2,taus1,taus2,xl,'// &
      'omegas1,omegas2,betads,betais,'// &
      'dt,cosz,elai,esai,tg,tv,snowh,fsno,fwet,sneqvo,sneqv,qsnow,fveg,'// &
      'smc1,smc2,smc3,smc4,albold_in,tauss_in,fage_in,'// &
      'frevd_in1,frevd_in2,frevi_in1,frevi_in2,'// &
      'fregd_in1,fregd_in2,fregi_in1,fregi_in2,'// &
      'fage,albold,tauss,fsun,bgap,wgap,'// &
      'albgrd1,albgrd2,albgri1,albgri2,albd1,albd2,albi1,albi2,'// &
      'fabd1,fabd2,fabi1,fabi2,ftdd1,ftdd2,ftid1,ftid2,ftii1,ftii2,'// &
      'frevd1,frevd2,frevi1,frevi2,fregd1,fregd2,fregi1,fregi2,'// &
      'albsnd1,albsnd2,albsni1,albsni2')
    do c = 1, NC
      p%TAU0         = R(1,c)
      p%GRAIN_GROWTH = R(2,c)
      p%EXTRA_GROWTH = R(3,c)
      p%DIRT_SOOT    = R(4,c)
      p%SWEMX        = R(5,c)
      p%ALBSAT(1)=R(6,c)  ; p%ALBSAT(2)=R(7,c)
      p%ALBDRY(1)=R(8,c)  ; p%ALBDRY(2)=R(9,c)
      p%ALBLAK(1)=R(10,c) ; p%ALBLAK(2)=R(11,c)
      p%RHOL(1)=R(12,c) ; p%RHOL(2)=R(13,c)
      p%RHOS(1)=R(14,c) ; p%RHOS(2)=R(15,c)
      p%TAUL(1)=R(16,c) ; p%TAUL(2)=R(17,c)
      p%TAUS(1)=R(18,c) ; p%TAUS(2)=R(19,c)
      p%XL = R(20,c)
      p%OMEGAS(1)=R(21,c) ; p%OMEGAS(2)=R(22,c)
      p%BETADS = R(23,c) ; p%BETAIS = R(24,c)
      do k = 1, NSOIL
        smc(k) = R(37+k,c)
      end do
      albold = R(42,c)
      tauss  = R(43,c)
      ! FAGE is a dummy argument with no INTENT that ALBEDO leaves untouched
      ! whenever COSZ <= 0, so its entry value is part of the contract.
      fage = -999.0 - cc ; fage_in = fage
      fsun = -999.0 ; bgap = -999.0 ; wgap = -999.0
      albgrd=-999.0 ; albgri=-999.0 ; albd=-999.0 ; albi=-999.0
      fabd=-999.0 ; fabi=-999.0 ; ftdd=-999.0 ; ftid=-999.0 ; ftii=-999.0
      ! ALBEDO's init loop (module_sf_noahmplsm.F:2829-2842) zeroes every
      ! INTENT(OUT) array EXCEPT FREVD/FREVI/FREGD/FREGI, so on the COSZ<=0
      ! early exit those four are left exactly as the caller passed them.
      ! Seed each slot distinctly, and vary the seeds down the fixture, so the
      ! mutation study can kill a transcription that ignores the entry value.
      cc = real(c)
      frevd = [ -101.0 - cc, -102.0 - 2.0*cc ]
      frevi = [ -103.0 - 3.0*cc, -104.0 - 4.0*cc ]
      fregd = [ -105.0 - 5.0*cc, -106.0 - 6.0*cc ]
      fregi = [ -107.0 - 7.0*cc, -108.0 - 8.0*cc ]
      frevd_in = frevd ; frevi_in = frevi
      fregd_in = fregd ; fregi_in = fregi
      albsnd=-999.0 ; albsni=-999.0
      call ALBEDO(p, VEGTYP, IST_(c), ICE, NSOIL, &
                  R(25,c), R(26,c), fage, R(27,c), R(28,c), &
                  R(29,c), R(30,c), R(31,c), R(32,c), R(33,c), &
                  smc, R(34,c), R(35,c), R(36,c), R(37,c), &
                  ILOC, JLOC, &
                  albold, tauss, &
                  albgrd, albgri, albd, albi, fabd, &
                  fabi, ftdd, ftid, ftii, fsun, &
                  frevi, frevd, fregd, fregi, bgap, &
                  wgap, albsnd, albsni)
      write(u,'(A)',advance='no') trim(nm(c))
      write(u,'(14A)',advance='no') ',', trim(ix(VEGTYP)), ',', trim(ix(IST_(c))), &
        ',', trim(ix(ICE)), ',', trim(ix(NSOIL)), ',', trim(ix(ILOC)), &
        ',', trim(ix(JLOC))
      do k = 1, NR
        write(u,'(2A)',advance='no') ',', hx(R(k,c))
      end do
      write(u,'(30A)',advance='no') &
        ',', hx(fage_in), &
        ',', hx(frevd_in(1)), ',', hx(frevd_in(2)), &
        ',', hx(frevi_in(1)), ',', hx(frevi_in(2)), &
        ',', hx(fregd_in(1)), ',', hx(fregd_in(2)), &
        ',', hx(fregi_in(1)), ',', hx(fregi_in(2)), &
        ',', hx(fage), ',', hx(albold), ',', hx(tauss), ',', hx(fsun), &
        ',', hx(bgap), ',', hx(wgap)
      write(u,'(16A)',advance='no') &
        ',', hx(albgrd(1)), ',', hx(albgrd(2)), ',', hx(albgri(1)), ',', hx(albgri(2)), &
        ',', hx(albd(1)), ',', hx(albd(2)), ',', hx(albi(1)), ',', hx(albi(2))
      write(u,'(20A)',advance='no') &
        ',', hx(fabd(1)), ',', hx(fabd(2)), ',', hx(fabi(1)), ',', hx(fabi(2)), &
        ',', hx(ftdd(1)), ',', hx(ftdd(2)), ',', hx(ftid(1)), ',', hx(ftid(2)), &
        ',', hx(ftii(1)), ',', hx(ftii(2))
      write(u,'(24A)') &
        ',', hx(frevd(1)), ',', hx(frevd(2)), ',', hx(frevi(1)), ',', hx(frevi(2)), &
        ',', hx(fregd(1)), ',', hx(fregd(2)), ',', hx(fregi(1)), ',', hx(fregi(2)), &
        ',', hx(albsnd(1)), ',', hx(albsnd(2)), ',', hx(albsni(1)), ',', hx(albsni(2))
    end do
    close(u)
  end subroutine emit_albedo

end program run_radiation
