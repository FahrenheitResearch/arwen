! run_energy.F90 -- bitwise fixture generator for the Noah-MP ENERGY assembly.
!
! ENERGY is phys/module_sf_noahmplsm.F lines 1741-2396 of the pinned WRF v4.6.1
! tree.  It is not a leaf: it is the composition that owns the surface energy
! balance.  Everything it calls is already pinned by its own fixture --
! THERMOPROP (noahmp-leaves.csv), RADIATION (noahmp-radiation-*.csv), VEGE_FLUX
! (noahmp-vegeflux.csv), BARE_FLUX (noahmp-bareflux.csv), TSNOSOI/PHASECHANGE
! (noahmp-thermal.csv) -- so what this fixture is *for* is the branching, the
! tile-average bookkeeping and the state ENERGY derives before and after those
! calls.
!
! The module is the unmodified pinned source with only the accessibility lift
! from visibility_patch_leaves.py applied.  Nothing here recomputes any physics.
!
! CSV columns:  case,role,name,hex,value
!   role = "opt"  pinned option identity, echoed for provenance
!          "cfg"  the integer land/soil identity the parameter block came from
!          "par"  a parameters component ENERGY itself reads, echoed so the
!                 fixture states which MPTABLE row it is standing on
!          "seed" the pre-ENERGY column state this driver starts from
!          "in"   ENERGY argument value on entry (for INOUT, the value in)
!          "out"  ENERGY argument value on exit
!   hex  = TRANSFER(value, integer) printed Z8.8 -- the byte-exact fixture
!   value= decimal rendering, for humans; hex is authoritative
!
! Cases 1-4 reproduce, argument for argument, the four columns already pinned
! whole-column in gpuwm/data/noahmp/oracle/noahmp-sflx.csv: the same MPTABLE
! row, the same forcing, the same SNOW_INIT topology, and the same NOAHMP_SFLX
! prologue (ATM, the DZSNSO/TROOT reductions, PHENOLOGY, the DVEG=4 FVEG rule,
! PRECIP_HEAT) evaluated by calling WRF's own routines.  That makes the ENERGY
! outputs of those four cases directly comparable to the NOAHMP_SFLX outputs of
! the same four columns for every quantity ENERGY writes and the rest of the
! column does not touch -- validate_energy_oracle.py performs that comparison.
! Cases 5-7 are not in the whole-column fixture; they exist to reach branches
! the four realistic columns do not.
!
! usage:  run_energy OUTPUT.csv

program run_energy
  use module_sf_noahmplsm, only: noahmp_parameters, noahmp_options, &
      energy, atm, phenology, precip_heat, &
      calculate_soil, soil_update_steps, &
      dveg, opt_crs, opt_btr, opt_run, opt_sfc, opt_frz, opt_inf, opt_rad, &
      opt_alb, opt_snf, opt_tbot, opt_stc, opt_rsf, opt_soil, opt_pedo, &
      opt_crop, opt_irr, opt_irrm, opt_infdv, opt_tdrn
  use noahmp_tables, only: read_mp_veg_parameters, read_mp_soil_parameters, &
      read_mp_rad_parameters, read_mp_global_parameters, &
      read_mp_crop_parameters, read_tiledrain_parameters, &
      read_mp_optional_parameters
  use noahmp_drv_helpers, only: transfer_mp_parameters, snow_init
  implicit none

  integer, parameter :: nsnow = 3
  integer, parameter :: nsoil = 4
  integer, parameter :: ncase = 9
  character(len=28), parameter :: case_name(ncase) = [character(len=28) :: &
      'veg_warm_day_dry', 'veg_warm_night_rain', 'snowpack_frozen_soil', &
      'bare_thin_snow_melt', 'veg_calm_desert_dry', 'veg_deep_snow_saturated', &
      'veg_subfreezing_canopy', 'urban_snowfree', 'veg_single_snow_layer']

  character(len=1024) :: output_path
  character(len=256)  :: llanduse
  integer :: icase, iz, unit
  type(noahmp_parameters) :: parameters

  ! ---- ENERGY arguments, declaration order --------------------------------
  integer :: iloc, jloc, ice, vegtyp, ist, isnow
  real    :: dt, rhoair, sfcprs, qair, sfctmp, thair, lwdn, uu, vv, zref
  real    :: co2air, o2air, cosz, igs, eair, tbot
  real    :: elai, esai, fwet, foln, fveg, pahv, pahg, pahb
  real    :: qsnow, lat, canliq, canice, z0wrf
  real, dimension(1:2)              :: solad, solai, albsnd, albsni
  real, dimension(-nsnow+1:nsoil)   :: zsnso, dzsnso, stc, hcpct
  real, dimension(1:nsoil)          :: zsoil, btrani, sh2o, smc
  integer, dimension(-nsnow+1:nsoil):: imelt
  real, dimension(-nsnow+1:0)       :: snicev, snliqv, epore, snice, snliq
  real    :: t2m, fsno, sav, sag, qmelt, fsa, fsr, taux, tauy
  real    :: fira, fsh, fcev, fgev, fctr, trad, psn, apar, ssoil, btran
  real    :: ponding, ts, latheav, latheag
  logical :: frozen_canopy, frozen_ground
  real    :: tv, tg, snowh, eah, tah, sneqvo, sneqv, albold, cm, ch
  real    :: dx, dz8w, q2, tauss, laisun, laisha, rb
  real    :: qc, qsfc, psfc
  real    :: t2mv, t2mb, fsrv, fsrg, rssun, rssha, bgap, wgap, tgv, tgb
  real    :: q1, q2v, q2b, q2e, chv, chb, emissi, pah, canhs
  real    :: shg, shc, shb, evg, evb, ghv, ghb, irg, irc, irb, tr, evc
  real    :: chleaf, chuc, chv2, chb2, eflxb, acc_ssoil
  real    :: julian, swdown, prcp, fb
  real, dimension(1:60) :: gecros1d

  ! ---- NOAHMP_SFLX prologue scratch (module_sf_noahmplsm.F:806-947) -------
  integer :: yearlen, croptype, pgs, slopetype, soilcolor, soilcat
  integer :: soiltype(nsoil)
  real    :: prcpconv, prcpnonc, prcpshcv, prcpsnow, prcpgrpl, prcphail
  real    :: soldn, qprecc, qprecl, bdfall, rain, snowfall, fp, fpice
  real    :: troot, lai, sai, shdfac, shdmax
  real    :: qintr, qdripr, qthror, qints, qdrips, qthros, qrain, snowhin, cmc
  real    :: snodep, soilt(nsoil), soilw(nsoil), soilliq(nsoil)

  ! ---- SNOW_INIT scratch (single column) ----------------------------------
  integer :: isnowxy(1, 1)
  real :: swexy(1, 1), tgxy(1, 1), snodepxy(1, 1)
  real :: zsnsoxy(1, -nsnow+1:nsoil, 1)
  real :: tsnoxy(1, -nsnow+1:0, 1), snicexy(1, -nsnow+1:0, 1)
  real :: snliqxy(1, -nsnow+1:0, 1)

  call get_command_argument(1, output_path)
  if (len_trim(output_path) == 0) then
    write(*, '(A)') 'usage: run_energy OUTPUT.csv'
    error stop 2
  end if

  llanduse = 'MODIFIED_IGBP_MODIS_NOAH'
  call read_mp_veg_parameters(trim(llanduse))
  call read_mp_soil_parameters()
  call read_mp_rad_parameters()
  call read_mp_global_parameters()
  call read_mp_crop_parameters()
  call read_tiledrain_parameters()
  call read_mp_optional_parameters()

  ! WRF Registry defaults, verbatim:
  !   dveg=4 opt_crs=1 opt_btr=1 opt_run=3 opt_sfc=1 opt_frz=1 opt_inf=1
  !   opt_rad=3 opt_alb=2 opt_snf=1 opt_tbot=2 opt_stc=1 opt_rsf=1 opt_soil=1
  !   opt_pedo=1 opt_crop=0 opt_irr=0 opt_irrm=0 opt_infdv=0 opt_tdrn=0
  call noahmp_options(4, 1, 1, 3, 1, 1, 1, 3, 2, 1, 2, 1, 1, 1, 1, 0, 0, 0, 0, 0)
  ! soiltstep = 0.0 in the Registry, which module_sf_noahmpdrv.F turns into a
  ! soil update on every step.
  soil_update_steps = 1
  calculate_soil    = .true.

  open(newunit=unit, file=trim(output_path), status='replace', action='write')
  write(unit, '(A)') 'case,role,name,hex,value'
  call emit_options()

  do icase = 1, ncase
    call build_case(icase)
    call emit_entry(icase)
    call energy(parameters, ice, vegtyp, ist, nsnow, nsoil, &
        isnow, dt, rhoair, sfcprs, qair, &
        sfctmp, thair, lwdn, uu, vv, zref, &
        co2air, o2air, solad, solai, cosz, igs, &
        eair, tbot, zsnso, zsoil, &
        elai, esai, fwet, foln, &
        fveg, pahv, pahg, pahb, &
        qsnow, dzsnso, lat, canliq, canice, iloc, jloc, &
        z0wrf, &
        imelt, snicev, snliqv, epore, t2m, fsno, &
        sav, sag, qmelt, fsa, fsr, taux, &
        tauy, fira, fsh, fcev, fgev, fctr, &
        trad, psn, apar, ssoil, btrani, btran, &
        ponding, ts, latheav, latheag, frozen_canopy, frozen_ground, &
        tv, tg, stc, snowh, eah, tah, &
        sneqvo, sneqv, sh2o, smc, snice, snliq, &
        albold, cm, ch, dx, dz8w, q2, &
        tauss, laisun, laisha, rb, &
        qc, qsfc, psfc, &
        t2mv, t2mb, fsrv, &
        fsrg, rssun, rssha, albsnd, albsni, bgap, wgap, tgv, tgb, &
        q1, q2v, q2b, q2e, chv, chb, emissi, pah, canhs, &
        shg, shc, shb, evg, evb, ghv, ghb, irg, irc, irb, tr, evc, &
        chleaf, chuc, chv2, chb2, &
        eflxb, hcpct, acc_ssoil, &
        julian, swdown, prcp, fb, gecros1d)
    call emit_exit(icase)
  end do

  close(unit)

contains

! ---------------------------------------------------------------------------
! emitters
! ---------------------------------------------------------------------------
  subroutine emitr(ic, role, name, val)
    integer, intent(in) :: ic
    character(len=*), intent(in) :: role, name
    real, intent(in) :: val
    integer :: ib
    ib = transfer(val, ib)
    write(unit, '(A,",",A,",",A,",",Z8.8,",",ES17.9E2)') &
         trim(case_name(ic)), trim(role), trim(name), ib, val
  end subroutine emitr

  subroutine emiti(ic, role, name, val)
    integer, intent(in) :: ic
    character(len=*), intent(in) :: role, name
    integer, intent(in) :: val
    write(unit, '(A,",",A,",",A,",",Z8.8,",",I17)') &
         trim(case_name(ic)), trim(role), trim(name), val, val
  end subroutine emiti

  subroutine emitl(ic, role, name, val)
    integer, intent(in) :: ic
    character(len=*), intent(in) :: role, name
    logical, intent(in) :: val
    integer :: iv
    iv = 0
    if (val) iv = 1
    call emiti(ic, role, name, iv)
  end subroutine emitl

  subroutine emitr_idx(ic, role, name, idx, val)
    integer, intent(in) :: ic, idx
    character(len=*), intent(in) :: role, name
    real, intent(in) :: val
    character(len=48) :: nm
    write(nm, '(A,"_",SP,I0)') trim(name), idx
    call emitr(ic, role, trim(nm), val)
  end subroutine emitr_idx

  ! A slot that ENERGY declares INTENT(OUT) and then does not assign on this
  ! path.  What the Fortran run leaves there is stack residue -- at COSZ <= 0
  ! this driver observed a NaN in FSRG -- so pinning it would pin the
  ! allocator, not the physics.  The row records the *name* as undefined and
  ! carries 0.0, which is the defined behaviour the ports are held to (the
  ! same rule THERMOPROP's dead slots already follow in noahmp-leaves.csv).
  ! Nothing downstream may read an 'undef' row as a measurement.
  subroutine emit_undef(ic, name, idx, has_idx)
    integer, intent(in) :: ic, idx
    character(len=*), intent(in) :: name
    logical, intent(in) :: has_idx
    character(len=48) :: nm
    if (has_idx) then
      write(nm, '(A,"_",SP,I0)') trim(name), idx
    else
      nm = trim(name)
    end if
    write(unit, '(A,",undef,",A,",",Z8.8,",",ES17.9E2)') &
         trim(case_name(ic)), trim(nm), 0, 0.0
  end subroutine emit_undef

  subroutine emiti_idx(ic, role, name, idx, val)
    integer, intent(in) :: ic, idx, val
    character(len=*), intent(in) :: role, name
    character(len=48) :: nm
    write(nm, '(A,"_",SP,I0)') trim(name), idx
    call emiti(ic, role, trim(nm), val)
  end subroutine emiti_idx

  subroutine emit_options()
    call emiti(1, 'opt', 'DVEG',      DVEG)
    call emiti(1, 'opt', 'OPT_CRS',   OPT_CRS)
    call emiti(1, 'opt', 'OPT_BTR',   OPT_BTR)
    call emiti(1, 'opt', 'OPT_RUN',   OPT_RUN)
    call emiti(1, 'opt', 'OPT_SFC',   OPT_SFC)
    call emiti(1, 'opt', 'OPT_FRZ',   OPT_FRZ)
    call emiti(1, 'opt', 'OPT_INF',   OPT_INF)
    call emiti(1, 'opt', 'OPT_RAD',   OPT_RAD)
    call emiti(1, 'opt', 'OPT_ALB',   OPT_ALB)
    call emiti(1, 'opt', 'OPT_SNF',   OPT_SNF)
    call emiti(1, 'opt', 'OPT_TBOT',  OPT_TBOT)
    call emiti(1, 'opt', 'OPT_STC',   OPT_STC)
    call emiti(1, 'opt', 'OPT_RSF',   OPT_RSF)
    call emiti(1, 'opt', 'OPT_SOIL',  OPT_SOIL)
    call emiti(1, 'opt', 'OPT_PEDO',  OPT_PEDO)
    call emiti(1, 'opt', 'OPT_CROP',  OPT_CROP)
    call emiti(1, 'opt', 'OPT_IRR',   OPT_IRR)
    call emiti(1, 'opt', 'OPT_IRRM',  OPT_IRRM)
    call emiti(1, 'opt', 'OPT_INFDV', OPT_INFDV)
    call emiti(1, 'opt', 'OPT_TDRN',  OPT_TDRN)
    call emiti(1, 'opt', 'SOIL_UPDATE_STEPS', soil_update_steps)
    call emitl(1, 'opt', 'CALCULATE_SOIL', calculate_soil)
  end subroutine emit_options

! ---------------------------------------------------------------------------
! Column construction.  Cases 1-4 are the noahmp-sflx.csv columns verbatim.
! ---------------------------------------------------------------------------
  subroutine build_case(ic)
    integer, intent(in) :: ic

    iloc     = 1
    jloc     = 1
    yearlen  = 365
    dt       = 60.0
    dx       = 3000.0
    dz8w     = 40.0
    zref     = 20.0
    ice      = 0
    ist      = 1
    croptype = 0
    pgs      = 0
    co2air   = 40.0
    o2air    = 21200.0
    foln     = 1.0
    prcpconv = 0.0
    prcpshcv = 0.0
    prcpgrpl = 0.0
    prcphail = 0.0
    qc       = 0.0
    slopetype = 1
    soilcolor = 4
    gecros1d = 0.0
    acc_ssoil = 0.0

    ! Four-layer WRF default soil column: 0.10/0.30/0.60/1.00 m thick.
    zsoil = [-0.10, -0.40, -1.00, -2.00]

    select case (ic)
    case (1)   ! vegetated, warm, sunlit, unsaturated, snow-free
      vegtyp = 10;  soilcat = 3
      shdfac = 0.72;  shdmax = 0.80
      lat = 0.6632;  julian = 200.5;  cosz = 0.82
      sfctmp = 296.0;  sfcprs = 96500.0
      uu = 3.5;  vv = -1.2;  q2 = 0.0105
      soldn = 820.0;  lwdn = 350.0
      prcpnonc = 0.0;  prcpsnow = 0.0
      tg = 297.5;  tv = 296.5;  tah = 296.2;  eah = 1500.0
      snodep = 0.0;  sneqv = 0.0
      soilt = [294.0, 292.0, 290.0, 288.0]
      soilw = [0.220, 0.240, 0.260, 0.280]
      soilliq = soilw
      canliq = 0.15;  canice = 0.0
      lai = 2.2;  sai = 0.5;  tbot = 288.0
    case (2)   ! same vegetation, nocturnal, non-convective rain
      vegtyp = 10;  soilcat = 3
      shdfac = 0.72;  shdmax = 0.80
      lat = 0.6632;  julian = 200.9;  cosz = 0.0
      sfctmp = 291.0;  sfcprs = 96200.0
      uu = 1.8;  vv = 0.4;  q2 = 0.0122
      soldn = 0.0;  lwdn = 372.0
      prcpnonc = 1.2e-3;  prcpsnow = 0.0
      tg = 291.5;  tv = 291.0;  tah = 291.2;  eah = 1900.0
      snodep = 0.0;  sneqv = 0.0
      soilt = [292.0, 291.0, 290.0, 288.5]
      soilw = [0.300, 0.305, 0.300, 0.290]
      soilliq = soilw
      canliq = 0.05;  canice = 0.0
      lai = 2.2;  sai = 0.5;  tbot = 288.0
    case (3)   ! layered snowpack over partly frozen soil, snowing
      vegtyp = 1;  soilcat = 6
      shdfac = 0.55;  shdmax = 0.70
      lat = 0.7854;  julian = 15.4;  cosz = 0.28
      sfctmp = 263.0;  sfcprs = 92000.0
      uu = 5.5;  vv = 2.0;  q2 = 0.0016
      soldn = 190.0;  lwdn = 225.0
      prcpnonc = 6.0e-4;  prcpsnow = 6.0e-4
      tg = 264.0;  tv = 264.5;  tah = 264.2;  eah = 190.0
      snodep = 0.20;  sneqv = 50.0
      soilt = [270.5, 272.0, 274.0, 277.0]
      soilw = [0.300, 0.310, 0.320, 0.330]
      soilliq = [0.090, 0.140, 0.320, 0.330]
      canliq = 0.0;  canice = 0.6
      lai = 3.4;  sai = 0.9;  tbot = 278.0
    case (4)   ! bare ground, sub-layer snow at the melting point
      vegtyp = 16;  soilcat = 1
      shdfac = 0.02;  shdmax = 0.05
      lat = 0.6981;  julian = 60.5;  cosz = 0.62
      sfctmp = 279.0;  sfcprs = 98000.0
      uu = 2.4;  vv = 1.1;  q2 = 0.0042
      soldn = 610.0;  lwdn = 300.0
      prcpnonc = 0.0;  prcpsnow = 0.0
      tg = 274.5;  tv = 274.5;  tah = 275.0;  eah = 610.0
      snodep = 0.02;  sneqv = 5.0
      soilt = [274.6, 274.0, 275.0, 278.0]
      soilw = [0.150, 0.170, 0.190, 0.210]
      soilliq = [0.150, 0.170, 0.190, 0.210]
      canliq = 0.0;  canice = 0.0
      lai = 0.2;  sai = 0.1;  tbot = 280.0
    case (5)   ! calm desert: UR clamps to 1, SH2O(1) < 0.01 with no snow so
               ! RSURF saturates at 1.E6, and the root zone is below wilting so
               ! the BTRAN GX clamp takes its MAX(0.,.) leg.
      vegtyp = 10;  soilcat = 1
      shdfac = 0.30;  shdmax = 0.35
      lat = 0.4363;  julian = 190.5;  cosz = 0.95
      sfctmp = 312.0;  sfcprs = 90000.0
      uu = 0.25;  vv = 0.15;  q2 = 0.0030
      soldn = 980.0;  lwdn = 390.0
      prcpnonc = 0.0;  prcpsnow = 0.0
      tg = 322.0;  tv = 313.0;  tah = 312.5;  eah = 800.0
      snodep = 0.0;  sneqv = 0.0
      soilt = [318.0, 310.0, 302.0, 296.0]
      soilw = [0.008, 0.020, 0.030, 0.040]
      soilliq = soilw
      canliq = 0.0;  canice = 0.0
      lai = 0.6;  sai = 0.2;  tbot = 294.0
    case (6)   ! deep snow over saturated soil.  Three branches only this case
               ! reaches: SNOWH exceeds 0.65*HVT so ZPD is taken from the snow
               ! (:2093), SH2O(1) exceeds SMCMAX so MIN(1.,SH2O/SMCMAX) in
               ! L_RSURF takes its 1.0 leg and RSURF collapses to zero (:2185),
               ! and SH2O above SMCREF drives the BTRAN GX clamp onto its
               ! MIN(1.,.) leg (:2165).  VEGTYP 19 has HVT = 2.0 m, above the
               ! 1.0 m threshold at which PHENOLOGY switches to the exponential
               ! burial rule, so the canopy survives 1.4 m of snow.
               ! DVEG = 4 makes PHENOLOGY read LAI from the MPTABLE monthly
               ! climatology, and the tundra rows carry almost no leaf in
               ! February, so a mid-summer date is used: this is a late-lying
               ! high-latitude snowpack, not a winter one.
      vegtyp = 19;  soilcat = 3
      shdfac = 0.68;  shdmax = 0.75
      lat = 0.8727;  julian = 190.5;  cosz = 0.35
      sfctmp = 270.0;  sfcprs = 94000.0
      uu = 4.0;  vv = -3.0;  q2 = 0.0026
      soldn = 300.0;  lwdn = 250.0
      prcpnonc = 2.0e-4;  prcpsnow = 2.0e-4
      tg = 271.0;  tv = 271.5;  tah = 270.8;  eah = 300.0
      snodep = 1.40;  sneqv = 330.0
      soilt = [274.0, 276.0, 278.0, 280.0]
      soilw = [0.450, 0.440, 0.420, 0.410]
      soilliq = [0.450, 0.440, 0.420, 0.410]
      canliq = 0.0;  canice = 0.2
      lai = 1.6;  sai = 0.4;  tbot = 281.0
    case (7)   ! sub-freezing canopy over a thawed ground: the only case where
               ! frozen_canopy and frozen_ground disagree, so LATHEAV /= LATHEAG
               ! and the two psychrometric constants separate.
      vegtyp = 5;  soilcat = 8
      shdfac = 0.60;  shdmax = 0.65
      lat = 0.9599;  julian = 300.5;  cosz = 0.15
      sfctmp = 271.5;  sfcprs = 97000.0
      uu = 2.2;  vv = 1.6;  q2 = 0.0033
      soldn = 120.0;  lwdn = 265.0
      prcpnonc = 0.0;  prcpsnow = 0.0
      tg = 274.6;  tv = 271.0;  tah = 272.0;  eah = 480.0
      snodep = 0.0;  sneqv = 0.0
      soilt = [275.0, 277.0, 279.0, 281.0]
      soilw = [0.260, 0.270, 0.280, 0.290]
      soilliq = soilw
      canliq = 0.02;  canice = 0.05
      lai = 1.1;  sai = 0.3;  tbot = 282.0
    case (8)   ! urban.  parameters%URBAN_FLAG replaces Z0MG/ZPDG/Z0M/ZPD with
               ! the canopy values (:2101-2106) and, with no snow, pins RSURF at
               ! 1.E6 (:2204-2206).  PHENOLOGY zeroes LAI/SAI for urban, so VAI
               ! is zero and the bare-only leg of the tile average runs.
      vegtyp = 13;  soilcat = 5
      shdfac = 0.10;  shdmax = 0.15
      lat = 0.7330;  julian = 210.5;  cosz = 0.70
      sfctmp = 301.0;  sfcprs = 100500.0
      uu = 3.0;  vv = 2.5;  q2 = 0.0125
      soldn = 760.0;  lwdn = 400.0
      prcpnonc = 0.0;  prcpsnow = 0.0
      tg = 308.0;  tv = 305.0;  tah = 302.0;  eah = 1800.0
      snodep = 0.0;  sneqv = 0.0
      soilt = [305.0, 301.0, 297.0, 293.0]
      soilw = [0.180, 0.200, 0.220, 0.240]
      soilliq = soilw
      canliq = 0.0;  canice = 0.0
      lai = 0.4;  sai = 0.1;  tbot = 291.0
    case (9)   ! a single snow layer (SNOW_INIT gives ISNOW = -1 for 0.025 m <=
               ! SNODEP <= 0.05 m) under a canopy, with TV above TFRZ and TG
               ! below it -- the one case where frozen_ground is set and
               ! frozen_canopy is not, the mirror of case 7.
               !
               ! SNEQV is 8.75 mm rather than a round 8: at a snow density of
               ! exactly 250 kg/m3 the FSNO argument lands on 1.75, one of the
               ! FP32 values where glibc's tanhf and an FP64-then-round shim
               ! disagree.  Without that the fixture's snow columns all happen
               ! to fall on arguments where the two agree, and it could not
               ! tell a correct TANH from a plausible one.  See
               ! tests/test_noahmp_energy.py::test_gate_fails_if_tanh_*.
      vegtyp = 5;  soilcat = 8
      shdfac = 0.62;  shdmax = 0.66
      lat = 0.8378;  julian = 90.5;  cosz = 0.45
      sfctmp = 272.5;  sfcprs = 95500.0
      uu = 1.4;  vv = -2.6;  q2 = 0.0035
      soldn = 430.0;  lwdn = 275.0
      prcpnonc = 3.0e-4;  prcpsnow = 3.0e-4
      tg = 272.0;  tv = 274.5;  tah = 273.0;  eah = 520.0
      snodep = 0.035;  sneqv = 8.75
      soilt = [272.5, 274.0, 276.0, 279.0]
      soilw = [0.240, 0.250, 0.260, 0.270]
      soilliq = [0.180, 0.250, 0.260, 0.270]
      canliq = 0.03;  canice = 0.10
      lai = 1.3;  sai = 0.35;  tbot = 280.0
    end select

    soiltype = soilcat
    call transfer_mp_parameters(nsoil, vegtyp, soiltype, slopetype, &
        soilcolor, croptype, parameters)

    ! Snow-layer topology exactly as WRF initializes it.
    swexy(1, 1)    = sneqv
    snodepxy(1, 1) = snodep
    tgxy(1, 1)     = tg
    call snow_init(1, 1, 1, 1, 1, 1, 1, 1, nsnow, nsoil, zsoil, swexy, tgxy, &
        snodepxy, zsnsoxy, tsnoxy, snicexy, snliqxy, isnowxy)

    isnow = isnowxy(1, 1)
    ! SNOW_INIT assigns ZSNSOXY only from ISNOW+1 up (module_sf_noahmpdrv.F
    ! :2432-2436), so the buried slots come back as whatever was on the stack.
    ! Left alone they make the *entry state* non-reproducible: a -O0 and a
    ! FCOPTIM build of this very driver disagreed on ZSNSO(-1).  ENERGY never
    ! reads them -- THERMOPROP, TSNOSOI and PHASECHANGE all start their loops
    ! at ISNOW+1 -- so pinning them at zero fixes the fixture without changing
    ! any output, and build_energy.sh's -O0/FCOPTIM diff is the proof.
    zsnso  = 0.0
    dzsnso = 0.0
    do iz = isnow + 1, nsoil
      zsnso(iz) = zsnsoxy(1, iz, 1)
    end do
    do iz = -nsnow + 1, 0
      snice(iz) = snicexy(1, iz, 1)
      snliq(iz) = snliqxy(1, iz, 1)
      stc(iz)   = tsnoxy(1, iz, 1)
    end do
    do iz = 1, nsoil
      stc(iz)  = soilt(iz)
      smc(iz)  = soilw(iz)
      sh2o(iz) = soilliq(iz)
    end do

    snowh  = snodep
    sneqvo = sneqv
    albold = 0.65
    tauss  = 0.0
    fwet   = 0.0
    qsnow  = 0.0
    qrain  = 0.0
    qsfc   = q2
    psfc   = sfcprs
    cm     = 0.01
    ch     = 0.01
    ! INTENT(INOUT) in ENERGY but overwritten before any read: RB at
    ! module_sf_noahmplsm.F:2053, LAISUN/LAISHA by RADIATION at :2118.
    laisun = 0.0
    laisha = 0.0
    rb     = 0.0
    ! INTENT(OUT) of NOAHMP_SFLX pre-set at :806-815 before the prologue runs.
    pahv = 0.0;  pahg = 0.0;  pahb = 0.0;  pah = 0.0;  canhs = 0.0

    call sflx_prologue()
  end subroutine build_case

! ---------------------------------------------------------------------------
! The NOAHMP_SFLX code between subroutine entry and the ENERGY call
! (module_sf_noahmplsm.F:819-947), with the irrigation block omitted because
! OPT_IRR = 0 and IRRFRA = 0 make TRIGGER_IRRIGATION and SPRINKLER_IRRIGATION
! unreachable: the trigger needs IRRFRA >= parameters%IRR_FRAC, the sprinkler
! needs IRAMTSI > 0, and CROPLU is false for every VEGTYP used here
! (1, 5, 10, 13, 16, 19) because MODIS-IGBP sets it only for 12 and 14.
! Every physics evaluation here is a call into the pinned module.
! ---------------------------------------------------------------------------
  subroutine sflx_prologue()
    call atm(parameters, sfcprs, sfctmp, q2, &
        prcpconv, prcpnonc, prcpshcv, prcpsnow, prcpgrpl, prcphail, &
        soldn, cosz, thair, qair, &
        eair, rhoair, qprecc, qprecl, solad, solai, &
        swdown, bdfall, rain, snowfall, fp, fpice, prcp)

    do iz = isnow + 1, nsoil
      if (iz == isnow + 1) then
        dzsnso(iz) = -zsnso(iz)
      else
        dzsnso(iz) = zsnso(iz - 1) - zsnso(iz)
      end if
    end do

    troot = 0.0
    do iz = 1, parameters%NROOT
      troot = troot + stc(iz) * dzsnso(iz) / (-zsoil(parameters%NROOT))
    end do

    call phenology(parameters, vegtyp, croptype, snowh, tv, lat, yearlen, &
        julian, lai, sai, troot, elai, esai, igs, pgs, fb)

    ! DVEG = 4 arm of module_sf_noahmplsm.F:876-888.
    fveg = shdmax
    if (fveg <= 0.05) fveg = 0.05
    if (parameters%urban_flag .or. vegtyp == parameters%ISBARREN) fveg = 0.0
    if (elai + esai == 0.0) fveg = 0.0

    call precip_heat(parameters, iloc, jloc, vegtyp, dt, uu, vv, &
        elai, esai, fveg, ist, &
        bdfall, rain, snowfall, fp, &
        canliq, canice, tv, sfctmp, tg, &
        qintr, qdripr, qthror, qints, qdrips, qthros, &
        pahv, pahg, pahb, qrain, qsnow, snowhin, fwet, cmc)
  end subroutine sflx_prologue

! ---------------------------------------------------------------------------
  subroutine emit_entry(ic)
    integer, intent(in) :: ic
    integer :: k

    call emiti(ic, 'cfg', 'VEGTYP',    vegtyp)
    call emiti(ic, 'cfg', 'SOILTYPE',  soilcat)
    call emiti(ic, 'cfg', 'SLOPETYPE', slopetype)
    call emiti(ic, 'cfg', 'SOILCOLOR', soilcolor)
    call emiti(ic, 'cfg', 'CROPTYPE',  croptype)
    call emiti(ic, 'cfg', 'YEARLEN',   yearlen)

    ! parameters components ENERGY's own body reads, echoed so the fixture
    ! records which MPTABLE row it stands on.  The full transfer is pinned in
    ! noahmp-parameters.csv; validate_energy_oracle.py cross-checks these.
    call emitl(ic, 'par', 'URBAN_FLAG', parameters%URBAN_FLAG)
    call emiti(ic, 'par', 'NROOT',     parameters%NROOT)
    call emiti(ic, 'par', 'ISBARREN',  parameters%ISBARREN)
    call emitr(ic, 'par', 'MFSNO',     parameters%MFSNO)
    call emitr(ic, 'par', 'SCFFAC',    parameters%SCFFAC)
    call emitr(ic, 'par', 'Z0SNO',     parameters%Z0SNO)
    call emitr(ic, 'par', 'Z0MVT',     parameters%Z0MVT)
    call emitr(ic, 'par', 'HVT',       parameters%HVT)
    call emitr(ic, 'par', 'CWPVT',     parameters%CWPVT)
    call emitr(ic, 'par', 'SNOW_EMIS', parameters%SNOW_EMIS)
    call emitr(ic, 'par', 'RSURF_EXP', parameters%RSURF_EXP)
    call emitr(ic, 'par', 'RSURF_SNOW', parameters%RSURF_SNOW)
    do k = 1, 2
      call emitr_idx(ic, 'par', 'EG', k, parameters%EG(k))
    end do
    do k = 1, nsoil
      call emitr_idx(ic, 'par', 'SMCMAX', k, parameters%SMCMAX(k))
      call emitr_idx(ic, 'par', 'SMCREF', k, parameters%SMCREF(k))
      call emitr_idx(ic, 'par', 'SMCWLT', k, parameters%SMCWLT(k))
      call emitr_idx(ic, 'par', 'PSISAT', k, parameters%PSISAT(k))
      call emitr_idx(ic, 'par', 'BEXP',   k, parameters%BEXP(k))
    end do

    ! The pre-prologue column state, so the entry state below is reproducible.
    call emitr(ic, 'seed', 'SHDFAC',   shdfac)
    call emitr(ic, 'seed', 'SHDMAX',   shdmax)
    call emitr(ic, 'seed', 'LAI',      lai)
    call emitr(ic, 'seed', 'SAI',      sai)
    call emitr(ic, 'seed', 'SOLDN',    soldn)
    call emitr(ic, 'seed', 'PRCPCONV', prcpconv)
    call emitr(ic, 'seed', 'PRCPNONC', prcpnonc)
    call emitr(ic, 'seed', 'PRCPSHCV', prcpshcv)
    call emitr(ic, 'seed', 'PRCPSNOW', prcpsnow)
    call emitr(ic, 'seed', 'PRCPGRPL', prcpgrpl)
    call emitr(ic, 'seed', 'PRCPHAIL', prcphail)
    call emitr(ic, 'seed', 'SNODEP',   snodep)
    call emitr(ic, 'seed', 'SWE',      sneqv)
    call emitr(ic, 'seed', 'TROOT',    troot)
    call emitr(ic, 'seed', 'BDFALL',   bdfall)
    call emitr(ic, 'seed', 'RAIN',     rain)
    call emitr(ic, 'seed', 'SNOWFALL', snowfall)
    call emitr(ic, 'seed', 'FP',       fp)
    call emitr(ic, 'seed', 'FPICE',    fpice)
    call emitr(ic, 'seed', 'QRAIN',    qrain)
    call emitr(ic, 'seed', 'SNOWHIN',  snowhin)
    call emitr(ic, 'seed', 'CMC',      cmc)
    call emitr(ic, 'seed', 'QINTR',    qintr)
    call emitr(ic, 'seed', 'QDRIPR',   qdripr)
    call emitr(ic, 'seed', 'QTHROR',   qthror)
    call emitr(ic, 'seed', 'QINTS',    qints)
    call emitr(ic, 'seed', 'QDRIPS',   qdrips)
    call emitr(ic, 'seed', 'QTHROS',   qthros)
    call emitr(ic, 'seed', 'SWDOWN',   swdown)
    call emitr(ic, 'seed', 'QPRECC',   qprecc)
    call emitr(ic, 'seed', 'QPRECL',   qprecl)

    ! ---- ENERGY entry state ------------------------------------------------
    call emiti(ic, 'in', 'ICE',    ice)
    call emiti(ic, 'in', 'IST',    ist)
    call emiti(ic, 'in', 'ISNOW',  isnow)
    call emiti(ic, 'in', 'ILOC',   iloc)
    call emiti(ic, 'in', 'JLOC',   jloc)
    call emitr(ic, 'in', 'DT',     dt)
    call emitr(ic, 'in', 'RHOAIR', rhoair)
    call emitr(ic, 'in', 'SFCPRS', sfcprs)
    call emitr(ic, 'in', 'QAIR',   qair)
    call emitr(ic, 'in', 'SFCTMP', sfctmp)
    call emitr(ic, 'in', 'THAIR',  thair)
    call emitr(ic, 'in', 'LWDN',   lwdn)
    call emitr(ic, 'in', 'UU',     uu)
    call emitr(ic, 'in', 'VV',     vv)
    call emitr(ic, 'in', 'ZREF',   zref)
    call emitr(ic, 'in', 'CO2AIR', co2air)
    call emitr(ic, 'in', 'O2AIR',  o2air)
    do k = 1, 2
      call emitr_idx(ic, 'in', 'SOLAD', k, solad(k))
      call emitr_idx(ic, 'in', 'SOLAI', k, solai(k))
    end do
    call emitr(ic, 'in', 'COSZ',   cosz)
    call emitr(ic, 'in', 'IGS',    igs)
    call emitr(ic, 'in', 'EAIR',   eair)
    call emitr(ic, 'in', 'TBOT',   tbot)
    do k = -nsnow + 1, nsoil
      call emitr_idx(ic, 'in', 'ZSNSO',  k, zsnso(k))
      call emitr_idx(ic, 'in', 'DZSNSO', k, dzsnso(k))
      call emitr_idx(ic, 'in', 'STC',    k, stc(k))
    end do
    do k = 1, nsoil
      call emitr_idx(ic, 'in', 'ZSOIL', k, zsoil(k))
      call emitr_idx(ic, 'in', 'SH2O',  k, sh2o(k))
      call emitr_idx(ic, 'in', 'SMC',   k, smc(k))
    end do
    do k = -nsnow + 1, 0
      call emitr_idx(ic, 'in', 'SNICE', k, snice(k))
      call emitr_idx(ic, 'in', 'SNLIQ', k, snliq(k))
    end do
    call emitr(ic, 'in', 'ELAI',   elai)
    call emitr(ic, 'in', 'ESAI',   esai)
    call emitr(ic, 'in', 'FWET',   fwet)
    call emitr(ic, 'in', 'FOLN',   foln)
    call emitr(ic, 'in', 'FVEG',   fveg)
    call emitr(ic, 'in', 'PAHV',   pahv)
    call emitr(ic, 'in', 'PAHG',   pahg)
    call emitr(ic, 'in', 'PAHB',   pahb)
    call emitr(ic, 'in', 'QSNOW',  qsnow)
    call emitr(ic, 'in', 'LAT',    lat)
    call emitr(ic, 'in', 'CANLIQ', canliq)
    call emitr(ic, 'in', 'CANICE', canice)
    call emitr(ic, 'in', 'TV',     tv)
    call emitr(ic, 'in', 'TG',     tg)
    call emitr(ic, 'in', 'SNOWH',  snowh)
    call emitr(ic, 'in', 'EAH',    eah)
    call emitr(ic, 'in', 'TAH',    tah)
    call emitr(ic, 'in', 'SNEQVO', sneqvo)
    call emitr(ic, 'in', 'SNEQV',  sneqv)
    call emitr(ic, 'in', 'ALBOLD', albold)
    call emitr(ic, 'in', 'CM',     cm)
    call emitr(ic, 'in', 'CH',     ch)
    call emitr(ic, 'in', 'DX',     dx)
    call emitr(ic, 'in', 'DZ8W',   dz8w)
    call emitr(ic, 'in', 'Q2',     q2)
    call emitr(ic, 'in', 'TAUSS',  tauss)
    call emitr(ic, 'in', 'LAISUN', laisun)
    call emitr(ic, 'in', 'LAISHA', laisha)
    call emitr(ic, 'in', 'RB',     rb)
    call emitr(ic, 'in', 'QC',     qc)
    call emitr(ic, 'in', 'QSFC',   qsfc)
    call emitr(ic, 'in', 'PSFC',   psfc)
    call emitr(ic, 'in', 'ACC_SSOIL', acc_ssoil)
    call emitr(ic, 'in', 'JULIAN', julian)
    call emitr(ic, 'in', 'PRCP',   prcp)
    call emitr(ic, 'in', 'FB',     fb)
  end subroutine emit_entry

! ---------------------------------------------------------------------------
  subroutine emit_exit(ic)
    integer, intent(in) :: ic
    integer :: k

    call emitr(ic, 'out', 'Z0WRF',  z0wrf)
    ! IMELT (PHASECHANGE, :5063) and HCPCT (THERMOPROP, :2413-2427) are written
    ! only over ISNOW+1..NSOIL; the buried slots stay undefined.
    do k = -nsnow + 1, nsoil
      if (k > isnow) then
        call emiti_idx(ic, 'out', 'IMELT', k, imelt(k))
        call emitr_idx(ic, 'out', 'HCPCT', k, hcpct(k))
      else
        call emit_undef(ic, 'IMELT', k, .true.)
        call emit_undef(ic, 'HCPCT', k, .true.)
      end if
      call emitr_idx(ic, 'out', 'STC',   k, stc(k))
    end do
    ! CSNOW (:2547-2565) likewise writes only ISNOW+1..0.
    do k = -nsnow + 1, 0
      if (k > isnow) then
        call emitr_idx(ic, 'out', 'SNICEV', k, snicev(k))
        call emitr_idx(ic, 'out', 'SNLIQV', k, snliqv(k))
        call emitr_idx(ic, 'out', 'EPORE',  k, epore(k))
      else
        call emit_undef(ic, 'SNICEV', k, .true.)
        call emit_undef(ic, 'SNLIQV', k, .true.)
        call emit_undef(ic, 'EPORE',  k, .true.)
      end if
      call emitr_idx(ic, 'out', 'SNICE',  k, snice(k))
      call emitr_idx(ic, 'out', 'SNLIQ',  k, snliq(k))
    end do
    ! BTRANI is written over 1..parameters%NROOT only (:2164-2172); the layers
    ! below the root zone stay INTENT(OUT)-undefined.  NOAHMP_SFLX then hands
    ! all NSOIL of them to WATER, so WRF really does read that residue -- but
    ! that is the column's problem, not ENERGY's, and nothing here pins it.
    do k = 1, nsoil
      if (k <= parameters%NROOT) then
        call emitr_idx(ic, 'out', 'BTRANI', k, btrani(k))
      else
        call emit_undef(ic, 'BTRANI', k, .true.)
      end if
      call emitr_idx(ic, 'out', 'SH2O',   k, sh2o(k))
      call emitr_idx(ic, 'out', 'SMC',    k, smc(k))
    end do
    do k = 1, 2
      call emitr_idx(ic, 'out', 'ALBSND', k, albsnd(k))
      call emitr_idx(ic, 'out', 'ALBSNI', k, albsni(k))
    end do
    call emitr(ic, 'out', 'T2M',    t2m)
    call emitr(ic, 'out', 'FSNO',   fsno)
    call emitr(ic, 'out', 'SAV',    sav)
    call emitr(ic, 'out', 'SAG',    sag)
    call emitr(ic, 'out', 'QMELT',  qmelt)
    call emitr(ic, 'out', 'FSA',    fsa)
    call emitr(ic, 'out', 'FSR',    fsr)
    call emitr(ic, 'out', 'TAUX',   taux)
    call emitr(ic, 'out', 'TAUY',   tauy)
    call emitr(ic, 'out', 'FIRA',   fira)
    call emitr(ic, 'out', 'FSH',    fsh)
    call emitr(ic, 'out', 'FCEV',   fcev)
    call emitr(ic, 'out', 'FGEV',   fgev)
    call emitr(ic, 'out', 'FCTR',   fctr)
    call emitr(ic, 'out', 'TRAD',   trad)
    call emitr(ic, 'out', 'PSN',    psn)
    call emitr(ic, 'out', 'APAR',   apar)
    call emitr(ic, 'out', 'SSOIL',  ssoil)
    call emitr(ic, 'out', 'BTRAN',  btran)
    call emitr(ic, 'out', 'PONDING', ponding)
    call emitr(ic, 'out', 'TS',     ts)
    call emitr(ic, 'out', 'LATHEAV', latheav)
    call emitr(ic, 'out', 'LATHEAG', latheag)
    call emitl(ic, 'out', 'FROZEN_CANOPY', frozen_canopy)
    call emitl(ic, 'out', 'FROZEN_GROUND', frozen_ground)
    call emitr(ic, 'out', 'TV',     tv)
    call emitr(ic, 'out', 'TG',     tg)
    call emitr(ic, 'out', 'SNOWH',  snowh)
    call emitr(ic, 'out', 'EAH',    eah)
    call emitr(ic, 'out', 'TAH',    tah)
    call emitr(ic, 'out', 'SNEQVO', sneqvo)
    call emitr(ic, 'out', 'SNEQV',  sneqv)
    call emitr(ic, 'out', 'ALBOLD', albold)
    call emitr(ic, 'out', 'CM',     cm)
    call emitr(ic, 'out', 'CH',     ch)
    call emitr(ic, 'out', 'TAUSS',  tauss)
    call emitr(ic, 'out', 'LAISUN', laisun)
    call emitr(ic, 'out', 'LAISHA', laisha)
    call emitr(ic, 'out', 'RB',     rb)
    call emitr(ic, 'out', 'QSFC',   qsfc)
    call emitr(ic, 'out', 'T2MV',   t2mv)
    call emitr(ic, 'out', 'T2MB',   t2mb)
    ! ALBEDO's init loop (:2908-2921) zeroes ALBD/ALBI/ALBGRD/ALBGRI/ALBSND/
    ! ALBSNI/FABD/FABI/FTDD/FTID/FTII/FSUN and nothing else, then takes
    ! `IF(COSZ <= 0) GOTO 100` past TWOSTREAM.  FREVD/FREVI/FREGD/FREGI and
    ! BGAP/WGAP are therefore undefined at night, and SURRAD propagates the
    ! first four into FSRV/FSRG at :3111-3112.
    if (cosz > 0.0) then
      call emitr(ic, 'out', 'FSRV', fsrv)
      call emitr(ic, 'out', 'FSRG', fsrg)
      call emitr(ic, 'out', 'BGAP', bgap)
      call emitr(ic, 'out', 'WGAP', wgap)
    else
      call emit_undef(ic, 'FSRV', 0, .false.)
      call emit_undef(ic, 'FSRG', 0, .false.)
      call emit_undef(ic, 'BGAP', 0, .false.)
      call emit_undef(ic, 'WGAP', 0, .false.)
    end if
    call emitr(ic, 'out', 'RSSUN',  rssun)
    call emitr(ic, 'out', 'RSSHA',  rssha)
    call emitr(ic, 'out', 'TGV',    tgv)
    call emitr(ic, 'out', 'TGB',    tgb)
    call emitr(ic, 'out', 'Q1',     q1)
    call emitr(ic, 'out', 'Q2V',    q2v)
    call emitr(ic, 'out', 'Q2B',    q2b)
    call emitr(ic, 'out', 'Q2E',    q2e)
    call emitr(ic, 'out', 'CHV',    chv)
    call emitr(ic, 'out', 'CHB',    chb)
    call emitr(ic, 'out', 'EMISSI', emissi)
    call emitr(ic, 'out', 'PAH',    pah)
    call emitr(ic, 'out', 'CANHS',  canhs)
    call emitr(ic, 'out', 'SHG',    shg)
    call emitr(ic, 'out', 'SHC',    shc)
    call emitr(ic, 'out', 'SHB',    shb)
    call emitr(ic, 'out', 'EVG',    evg)
    call emitr(ic, 'out', 'EVB',    evb)
    call emitr(ic, 'out', 'GHV',    ghv)
    call emitr(ic, 'out', 'GHB',    ghb)
    call emitr(ic, 'out', 'IRG',    irg)
    call emitr(ic, 'out', 'IRC',    irc)
    call emitr(ic, 'out', 'IRB',    irb)
    call emitr(ic, 'out', 'TR',     tr)
    call emitr(ic, 'out', 'EVC',    evc)
    call emitr(ic, 'out', 'CHLEAF', chleaf)
    call emitr(ic, 'out', 'CHUC',   chuc)
    call emitr(ic, 'out', 'CHV2',   chv2)
    call emitr(ic, 'out', 'CHB2',   chb2)
    call emitr(ic, 'out', 'EFLXB',  eflxb)
    call emitr(ic, 'out', 'ACC_SSOIL', acc_ssoil)
  end subroutine emit_exit

end program run_energy
