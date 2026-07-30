program run_ruc_lsmruc_oracle
  !--------------------------------------------------------------------------
  ! Oracle for the RUC LSM DRIVER, phys/module_sf_ruclsm.F:84-1175 subroutine
  ! LSMRUC, built against the UNMODIFIED module.
  !
  ! LSMRUC is a subroutine, so every result is a returned argument and no
  ! debugger is needed: the fixture is the complete intent(inout)/intent(out)
  ! argument set on both sides of the call, plus the complete forcing.
  !
  ! Two independent RUNS, each integrated for two STEPS, because several of
  ! the driver's branches are selected by call scalars rather than by column
  ! state:
  !
  !   run 1  MODIFIED_IGBP_MODIS_NOAH (MODI-RUC), frpcpn=.true.,
  !          rdlai2d=.false., dt=60,  iswater=17, isice=15
  !   run 2  USGS (USGS-RUC),         frpcpn=.false.,
  !          rdlai2d=.true.,  dt=180, iswater=16, isice=24
  !
  ! myj is .false. in both runs.  That is not a coverage gap this file can
  ! close: gpuwm's SOIL and SNOWSOIL ports are fail-closed on myj=.true., so
  ! LSMRUC's :681-682 exchange-coefficient arm is unreachable through the
  ! supported lane and a fixture that bound it could not be replayed.
  !
  !   step 1 ktau=1 -- the first-step initialisation block (:463-566)
  !   step 2 ktau=2 -- the same state carried forward, so the accumulators
  !                    (sfcrunoff, udrunoff, acrunoff, sfcevp, snowfallac,
  !                    acsnow, snom) are pinned with a nonzero incoming value
  !
  ! mosaic_lu and mosaic_soil are 0 in both runs.  That is not a gap in the
  ! coverage of this file: SOILVEGIN's mosaic arms are outside the pinned RUC
  ! option lane (gpuwm.core.ruc.ruc_surface_parameters is fail-closed on
  ! them), and LSMRUC's irrigation block at :984-1009 is gated on the same
  ! mosaic_lu==1, so it is unreachable wherever SOILVEGIN is.
  !
  ! The module is compiled -DEM_CORE=0, exactly as build.sh compiles it for
  ! every other RUC fixture, so the EM_CORE==1 arms (SPP perturbations, the
  ! lake bypass, and the rainncv/snowncv/graupelncv precipitation partition)
  ! are not present in the object this driver links against.
  !
  ! LSMRUC calls SFCTMP once per column, and SFCTMP reads an uninitialised
  ! local `ilnb` (:1385) on the thin-snow blended path.  Inside LSMRUC that
  ! stack slot carries the PREVIOUS COLUMN's layer count.  This driver zeroes
  ! the region LSMRUC's callee frames will occupy before every call, with a
  ! recursive helper, so the fixture is reproducible; argument 2 sets the fill
  ! value and running it a second time with a nonzero fill is the negative
  ! control that proves the read is real.
  !
  ! usage: run_lsmruc OUTPUT.csv [STACK_FILL]
  !--------------------------------------------------------------------------
  use module_sf_ruclsm, only: lsmruc, ruclsminit
  implicit none

  integer, parameter :: ncol = 12, nsl = 9, nz = 2
  integer, parameter :: nlcat = 28, nscat = 19
  integer, parameter :: ids = 1, ide = ncol + 1
  integer, parameter :: jds = 1, jde = 2
  integer, parameter :: kds = 1, kde = nz + 1
  integer, parameter :: ims = 1, ime = ncol
  integer, parameter :: jms = 1, jme = 1
  integer, parameter :: kms = 1, kme = nz
  integer, parameter :: its = 1, ite = ncol
  integer, parameter :: jts = 1, jte = 1
  integer, parameter :: kts = 1, kte = nz

  character(len=24), parameter :: names1(ncol) = [character(len=24) :: &
      'warm_forest_rain', 'warm_grass_mixed', 'cold_crop_snowfall',   &
      'deep_pack_forest', 'open_water', 'snow_on_sea_ice',            &
      'bare_sea_ice', 'dry_barren', 'urban_thin_snow',                &
      'mixed_forest_melt', 'savanna_wet', 'tundra_snow_free']
  character(len=24), parameter :: names2(ncol) = [character(len=24) :: &
      'usgs_crop_rain', 'usgs_grass_allsnow', 'usgs_crop_snowpack',   &
      'usgs_forest_deep_pack', 'usgs_water', 'usgs_seaice_snow',      &
      'usgs_seaice_bare', 'usgs_desert', 'usgs_urban_thin_snow',      &
      'usgs_mixed_melt', 'usgs_wetland', 'usgs_tundra_subthresh']
  character(len=24) :: names(ncol)

  character(len=1024) :: output_path, pad_argument
  character(len=32)   :: mminlu
  integer :: i, k, unit, run, step, lutype
  real    :: stack_pad

  !--- call scalars -------------------------------------------------------
  real    :: dt, xice_threshold
  integer :: ktau, iswater, isice, mosaic_lu, mosaic_soil
  logical :: myj, frpcpn, rdlai2d
  real, parameter :: cp = 1004.5, rovcp = 287.0 / 1004.5
  real, parameter :: g0 = 9.81, lv = 2.5e6, stbolt = 5.67051e-8

  !--- forcing ------------------------------------------------------------
  integer :: ivgtyp(ims:ime,jms:jme), isltyp(ims:ime,jms:jme)
  real :: landusef(ims:ime,1:nlcat,jms:jme)
  real :: soilctop(ims:ime,1:nscat,jms:jme)
  real :: z3d(ims:ime,kms:kme,jms:jme), p8w(ims:ime,kms:kme,jms:jme)
  real :: t3d(ims:ime,kms:kme,jms:jme), qv3d(ims:ime,kms:kme,jms:jme)
  real :: qc3d(ims:ime,kms:kme,jms:jme), rho3d(ims:ime,kms:kme,jms:jme)
  real :: zs(1:nsl)
  real :: rainbl(ims:ime,jms:jme), frzfrac(ims:ime,jms:jme)
  real :: glw(ims:ime,jms:jme), gsw(ims:ime,jms:jme)
  real :: chs(ims:ime,jms:jme), flqc(ims:ime,jms:jme)
  real :: flhc(ims:ime,jms:jme), albbck(ims:ime,jms:jme)
  real :: xland(ims:ime,jms:jme), xice(ims:ime,jms:jme)
  real :: tbot(ims:ime,jms:jme)
  real :: shdmin(ims:ime,jms:jme), shdmax(ims:ime,jms:jme)

  !--- state --------------------------------------------------------------
  real :: soilmois(ims:ime,1:nsl,jms:jme), sh2o(ims:ime,1:nsl,jms:jme)
  real :: tso(ims:ime,1:nsl,jms:jme), smfr3d(ims:ime,1:nsl,jms:jme)
  real :: keepfr3dflag(ims:ime,1:nsl,jms:jme)
  real :: snow(ims:ime,jms:jme), snowh(ims:ime,jms:jme)
  real :: snowc(ims:ime,jms:jme), canwat(ims:ime,jms:jme)
  real :: snoalb(ims:ime,jms:jme), alb(ims:ime,jms:jme)
  real :: emiss(ims:ime,jms:jme), lai(ims:ime,jms:jme)
  real :: mavail(ims:ime,jms:jme), sfcexc(ims:ime,jms:jme)
  real :: z0(ims:ime,jms:jme), znt(ims:ime,jms:jme)
  real :: vegfra(ims:ime,jms:jme), soilt(ims:ime,jms:jme)
  real :: hfx(ims:ime,jms:jme), qfx(ims:ime,jms:jme), lh(ims:ime,jms:jme)
  real :: sfcevp(ims:ime,jms:jme), sfcrunoff(ims:ime,jms:jme)
  real :: udrunoff(ims:ime,jms:jme), acrunoff(ims:ime,jms:jme)
  real :: grdflx(ims:ime,jms:jme), acsnow(ims:ime,jms:jme)
  real :: snom(ims:ime,jms:jme), qvg(ims:ime,jms:jme)
  real :: qcg(ims:ime,jms:jme), dew(ims:ime,jms:jme)
  real :: qsfc(ims:ime,jms:jme), qsg(ims:ime,jms:jme)
  real :: chklowq(ims:ime,jms:jme), soilt1(ims:ime,jms:jme)
  real :: tsnav(ims:ime,jms:jme), smavail(ims:ime,jms:jme)
  real :: smmax(ims:ime,jms:jme), rhosnf(ims:ime,jms:jme)
  real :: precipfr(ims:ime,jms:jme), snowfallac(ims:ime,jms:jme)

  !--- entry snapshots ----------------------------------------------------
  real, dimension(ims:ime,1:nsl,jms:jme) :: soilmois_i, sh2o_i, tso_i
  real, dimension(ims:ime,1:nsl,jms:jme) :: smfr3d_i, keepfr_i
  real, dimension(ims:ime,jms:jme) :: snow_i, snowh_i, snowc_i, canwat_i
  real, dimension(ims:ime,jms:jme) :: snoalb_i, alb_i, emiss_i, lai_i
  real, dimension(ims:ime,jms:jme) :: mavail_i, sfcexc_i, z0_i, znt_i
  real, dimension(ims:ime,jms:jme) :: soilt_i, hfx_i, qfx_i, lh_i
  real, dimension(ims:ime,jms:jme) :: sfcevp_i, sfcrunoff_i, udrunoff_i
  real, dimension(ims:ime,jms:jme) :: acrunoff_i, grdflx_i, acsnow_i
  real, dimension(ims:ime,jms:jme) :: snom_i, qvg_i, qcg_i, dew_i
  real, dimension(ims:ime,jms:jme) :: qsfc_i, qsg_i, chklowq_i, soilt1_i
  real, dimension(ims:ime,jms:jme) :: tsnav_i, smavail_i, smmax_i
  real, dimension(ims:ime,jms:jme) :: rhosnf_i, precipfr_i, snowfallac_i

  call get_command_argument(1, output_path)
  if (len_trim(output_path) == 0) then
    write(*, '(A)') 'usage: run_lsmruc OUTPUT.csv [STACK_FILL]'
    error stop 2
  end if
  call get_command_argument(2, pad_argument)
  if (len_trim(pad_argument) == 0) then
    stack_pad = 0.0
  else
    read(pad_argument, *) stack_pad
  end if

  open(newunit=unit, file=trim(output_path), status='replace', action='write')
  write(unit, '(A)') 'run,step,case,k,dt,ktau,myj,frpcpn,mosaic_lu,'   // &
      'mosaic_soil,rdlai2d,iswater,isice,xice_threshold,nlcat,nscat,'  // &
      'lutype,cp,rovcp,g0,lv,stbolt,zs,ivgtyp,isltyp,xland,xice,tbot,' // &
      'shdmin,shdmax,vegfra,rainbl,frzfrac,glw,gsw,chs,flqc,flhc,'     // &
      'albbck,z3d,p8w,t3d,qv3d,qc3d,rho3d,'                            // &
      'snow_i,snowh_i,snowc_i,canwat_i,snoalb_i,alb_i,emiss_i,lai_i,'  // &
      'mavail_i,sfcexc_i,z0_i,znt_i,soilt_i,hfx_i,qfx_i,lh_i,'         // &
      'sfcevp_i,sfcrunoff_i,udrunoff_i,acrunoff_i,grdflx_i,acsnow_i,'  // &
      'snom_i,qvg_i,qcg_i,dew_i,qsfc_i,qsg_i,chklowq_i,soilt1_i,'      // &
      'tsnav_i,smavail_i,smmax_i,rhosnf_i,precipfr_i,snowfallac_i,'    // &
      'soilmois_i,sh2o_i,tso_i,smfr3d_i,keepfr_i,'                     // &
      'snow,snowh,snowc,canwat,snoalb,alb,emiss,lai,mavail,sfcexc,'    // &
      'z0,znt,soilt,hfx,qfx,lh,sfcevp,sfcrunoff,udrunoff,acrunoff,'    // &
      'grdflx,acsnow,snom,qvg,qcg,dew,qsfc,qsg,chklowq,soilt1,tsnav,'  // &
      'smavail,smmax,rhosnf,precipfr,snowfallac,'                      // &
      'soilmois,sh2o,tso,smfr3d,keepfr'

  zs = [0.00, 0.01, 0.04, 0.10, 0.30, 0.60, 1.00, 1.60, 3.00]

  do run = 1, 2

    !--------------------------------------------------------------------
    ! call scalars and cold-start state for this run
    !--------------------------------------------------------------------
    mosaic_lu = 0
    mosaic_soil = 0
    xice_threshold = 0.5

    if (run == 1) then
      names = names1
      mminlu = 'MODIFIED_IGBP_MODIS_NOAH'
      lutype = 0
      iswater = 17
      isice = 15
      dt = 60.0
      myj = .false.
      frpcpn = .true.
      rdlai2d = .false.

      ivgtyp(:,1) = [ 1, 10, 12,  2, 17, 15, 15, 16, 13,  5,  9, 20]
      isltyp(:,1) = [ 4,  6,  3,  8, 14, 16, 16,  1,  6,  7,  9,  2]
      xland (:,1) = [1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
      xice  (:,1) = [0.0, 0.0, 0.0, 0.0, 0.0, 0.85, 0.62, 0.0, 0.0, 0.0, 0.20, 0.0]
      vegfra(:,1) = [72.0, 55.0, 88.0, 92.0, 0.0, 0.0, 0.0, 2.0, 12.0, 64.0, 48.0, 30.0]
      shdmin(:,1) = [20.0, 15.0, 10.0, 30.0, 0.0, 0.0, 0.0, 0.0, 5.0, 20.0, 10.0, 5.0]
      shdmax(:,1) = [90.0, 85.0, 95.0, 95.0, 0.0, 0.0, 0.0, 5.0, 40.0, 90.0, 70.0, 60.0]
      rainbl(:,1) = [0.90, 1.20, 0.80, 0.00, 0.40, 0.20, 0.00, 0.00, 0.10, 0.50, 3.00, 0.00]
      frzfrac(:,1)= [0.00, 0.35, 1.00, 0.00, 0.00, 1.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00]
      glw   (:,1) = [340.0, 300.0, 250.0, 235.0, 350.0, 215.0, 225.0, 355.0, 265.0, 305.0, 360.0, 240.0]
      gsw   (:,1) = [520.0, 430.0, 150.0, 110.0, 400.0,  70.0,  95.0, 610.0, 180.0, 330.0, 470.0, 200.0]
      chs   (:,1) = [0.012, 0.014, 0.008, 0.006, 0.020, 0.005, 0.007, 0.018, 0.009, 0.013, 0.016, 0.010]
      flqc  (:,1) = [0.011, 0.013, 0.007, 0.005, 0.019, 0.004, 0.006, 0.017, 0.008, 0.012, 0.015, 0.009]
      flhc  (:,1) = [11.0, 13.0, 7.0, 5.0, 19.0, 4.0, 6.0, 17.0, 8.0, 12.0, 15.0, 9.0]
      alb   (:,1) = [0.12, 0.19, 0.18, 0.12, 0.08, 0.55, 0.55, 0.25, 0.18, 0.13, 0.20, 0.15]
      snoalb(:,1) = [0.52, 0.70, 0.66, 0.35, 0.70, 0.82, 0.82, 0.75, 0.46, 0.53, 0.50, 0.75]
      emiss (:,1) = [0.95, 0.92, 0.935, 0.95, 0.98, 0.98, 0.98, 0.85, 0.88, 0.94, 0.92, 0.90]
      lai   (:,1) = [6.40, 2.90, 5.68, 6.48, 0.01, 0.01, 0.01, 0.75, 1.00, 5.50, 3.66, 3.35]
      canwat(:,1) = [0.15, 0.08, 0.12, 0.05, 0.00, 0.00, 0.00, 0.00, 0.02, 0.10, 0.18, 0.00]
      snow  (:,1) = [0.0, 0.0, 12.0, 140.0, 0.0, 40.0, 0.0, 0.0, 3.0, 25.0, 0.0, 0.0]
      snowh (:,1) = [0.00, 0.00, 0.06, 0.45, 0.00, 0.18, 0.00, 0.00, 0.012, 0.10, 0.00, 0.00]
      snowc (:,1) = [0.0, 0.0, 0.7, 1.0, 0.0, 1.0, 0.0, 0.0, 0.3, 0.8, 0.0, 0.0]
      soilt (:,1) = [297.0, 275.5, 267.0, 262.0, 286.0, 260.0, 264.0, 301.0, 270.5, 277.0, 294.0, 265.0]
      tbot  (:,1) = [288.0, 284.0, 278.0, 276.0, 286.0, 271.0, 271.0, 292.0, 281.0, 283.0, 289.0, 277.0]

      do k = kms, kme
        t3d (:,k,1) = [298.0, 274.5, 268.0, 263.0, 286.0, 261.0, 265.0, 300.0, 270.0, 277.5, 292.0, 265.5]
        qv3d(:,k,1) = [0.0120, 0.0040, 0.0022, 0.0014, 0.00955, 0.0011, 0.0016, 0.0035, 0.0024, 0.0048, 0.0139, 0.0018]
        qc3d(:,k,1) = [0.0, 1.0e-5, 2.0e-5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0e-5, 0.0, 3.0e-5, 0.0]
        z3d (:,k,1) = 80.0
        p8w (:,k,1) = 95000.0
        rho3d(:,k,1) = 1.10
      end do

      !-- ktau==1 initialisation arms (:490-521): columns 1-4 enter with an
      !-- out-of-range soilt1 (1-2 with snowc==0, 3-4 with snowc>0), 5-6 the
      !-- same on a water and a sea-ice point, 7-12 with soilt1 in range.
      soilt1(:,1) = [-999.0, -999.0, -999.0, -999.0, 999.0, -999.0, &
                     264.0, 300.0, 270.0, 276.0, 293.0, 265.0]
      !-- qcg arm (:503): 1-4 out of range, 5-12 inside it.
      qcg   (:,1) = [-1.0, -1.0, 0.5, -1.0, 5.0e-4, 3.0e-4, 2.0e-4, &
                     1.0e-4, 5.0e-5, 6.0e-5, 1.0e-4, 0.0]
      !-- qvg arm (:512): 1-6 out of range, 7-12 inside it.
      qvg   (:,1) = [0.0, -0.5, 0.0, 0.2, 0.0, 0.0, &
                     0.0016, 0.0060, 0.0026, 0.0050, 0.0150, 0.0020]
      mavail(:,1) = [0.60, 0.55, 0.70, 0.40, 1.00, 1.00, 1.00, 0.05, 0.30, 0.65, 0.95, 0.35]

      do k = 1, nsl
        tso(1,k,1) = 296.0 - 1.00 * real(k - 1)
        tso(2,k,1) = 275.0 + 0.60 * real(k - 1)
        tso(3,k,1) = 268.0 + 0.90 * real(k - 1)
        tso(4,k,1) = 263.0 + 1.20 * real(k - 1)
        tso(5,k,1) = 286.0
        tso(6,k,1) = min(271.0, 259.0 + 1.10 * real(k - 1))
        tso(7,k,1) = min(271.0, 263.0 + 0.90 * real(k - 1))
        tso(8,k,1) = 300.0 - 0.80 * real(k - 1)
        tso(9,k,1) = 271.0 + 1.00 * real(k - 1)
        tso(10,k,1) = 277.0 + 0.70 * real(k - 1)
        tso(11,k,1) = 293.0 - 0.40 * real(k - 1)
        tso(12,k,1) = 266.0 + 1.30 * real(k - 1)
        soilmois(1,k,1) = 0.240 + 0.005 * real(k - 1)
        soilmois(2,k,1) = 0.300 + 0.004 * real(k - 1)
        soilmois(3,k,1) = 0.180 + 0.006 * real(k - 1)
        soilmois(4,k,1) = 0.330 + 0.003 * real(k - 1)
        soilmois(5,k,1) = 1.0
        soilmois(6,k,1) = 1.0
        soilmois(7,k,1) = 1.0
        soilmois(8,k,1) = 0.030 + 0.002 * real(k - 1)
        soilmois(9,k,1) = 0.220 + 0.004 * real(k - 1)
        soilmois(10,k,1) = 0.260 + 0.005 * real(k - 1)
        soilmois(11,k,1) = 0.380 + 0.004 * real(k - 1)
        soilmois(12,k,1) = 0.200 + 0.006 * real(k - 1)
      end do

    else
      names = names2
      mminlu = 'USGS'
      lutype = 1
      iswater = 16
      isice = 24
      dt = 180.0
      myj = .false.
      frpcpn = .false.
      rdlai2d = .true.

      ivgtyp(:,1) = [ 3,  7,  2, 14, 16, 24, 24, 19,  1, 15, 17, 22]
      isltyp(:,1) = [ 4,  6,  3,  8, 14, 16, 16,  1,  6,  7,  9,  2]
      xland (:,1) = [1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
      xice  (:,1) = [0.0, 0.0, 0.0, 0.0, 0.0, 0.90, 0.55, 0.0, 0.0, 0.0, 0.0, 0.30]
      vegfra(:,1) = [80.0, 60.0, 75.0, 90.0, 0.0, 0.0, 0.0, 3.0, 15.0, 70.0, 55.0, 35.0]
      shdmin(:,1) = [10.0, 12.0, 8.0, 25.0, 0.0, 0.0, 0.0, 0.0, 5.0, 18.0, 12.0, 6.0]
      shdmax(:,1) = [95.0, 88.0, 92.0, 96.0, 0.0, 0.0, 0.0, 6.0, 45.0, 92.0, 75.0, 65.0]
      !-- frpcpn=.false. splits on tabs<=273.15 alone; frzfrac is unread.
      rainbl(:,1) = [1.50, 0.70, 1.10, 0.30, 0.60, 0.40, 0.20, 0.00, 0.25, 0.80, 4.00, 0.35]
      frzfrac(:,1)= [0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20]
      glw   (:,1) = [345.0, 295.0, 245.0, 230.0, 352.0, 210.0, 222.0, 358.0, 268.0, 308.0, 362.0, 238.0]
      gsw   (:,1) = [505.0, 415.0, 140.0, 105.0, 395.0,  65.0,  90.0, 600.0, 175.0, 320.0, 460.0, 195.0]
      chs   (:,1) = [0.015, 0.011, 0.009, 0.007, 0.021, 0.004, 0.008, 0.019, 0.010, 0.014, 0.017, 0.006]
      flqc  (:,1) = [0.014, 0.010, 0.008, 0.006, 0.020, 0.003, 0.007, 0.018, 0.009, 0.013, 0.016, 0.005]
      flhc  (:,1) = [14.0, 10.0, 8.0, 6.0, 20.0, 3.0, 7.0, 18.0, 9.0, 13.0, 16.0, 5.0]
      alb   (:,1) = [0.18, 0.19, 0.17, 0.12, 0.08, 0.55, 0.55, 0.25, 0.18, 0.14, 0.14, 0.15]
      snoalb(:,1) = [0.64, 0.64, 0.64, 0.35, 0.70, 0.75, 0.75, 0.75, 0.40, 0.55, 0.59, 0.65]
      emiss (:,1) = [0.92, 0.92, 0.92, 0.92, 0.98, 0.98, 0.98, 0.88, 0.88, 0.93, 0.95, 0.92]
      lai   (:,1) = [5.68, 2.90, 5.68, 6.40, 0.01, 0.01, 0.01, 0.75, 1.00, 5.50, 5.72, 3.35]
      canwat(:,1) = [0.20, 0.06, 0.14, 0.04, 0.00, 0.00, 0.00, 0.00, 0.03, 0.11, 0.22, 0.01]
      snow  (:,1) = [0.0, 0.0, 15.0, 160.0, 0.0, 35.0, 0.0, 0.0, 4.0, 28.0, 0.0, 6.0]
      snowh (:,1) = [0.00, 0.00, 0.07, 0.50, 0.00, 0.15, 0.00, 0.00, 0.015, 0.11, 0.00, 0.03]
      snowc (:,1) = [0.0, 0.0, 0.75, 1.0, 0.0, 1.0, 0.0, 0.0, 0.35, 0.85, 0.0, 0.4]
      soilt (:,1) = [292.0, 270.0, 265.0, 261.0, 287.0, 258.0, 263.0, 303.0, 271.5, 276.0, 291.0, 264.0]
      tbot  (:,1) = [287.0, 281.0, 277.0, 275.0, 287.0, 271.0, 271.0, 293.0, 282.0, 282.0, 288.0, 276.0]

      do k = kms, kme
        t3d (:,k,1) = [291.0, 270.0, 266.0, 262.0, 287.0, 259.0, 263.5, 302.0, 271.0, 276.5, 290.0, 264.5]
        qv3d(:,k,1) = [0.0105, 0.0026, 0.0019, 0.0012, 0.00985, 0.0009, 0.0014, 0.0030, 0.0022, 0.0044, 0.0125, 0.0016]
        qc3d(:,k,1) = [1.0e-5, 0.0, 3.0e-5, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0e-5, 0.0, 4.0e-5, 0.0]
        z3d (:,k,1) = 60.0
        p8w (:,k,1) = 98000.0
        rho3d(:,k,1) = 1.18
      end do

      !-- second run repeats the three ktau==1 arms with a different pairing
      !-- so no arm is bound to one land-use dataset.
      soilt1(:,1) = [420.0, 292.0, -999.0, 264.0, -999.0, 258.0, &
                     -999.0, 303.0, 165.0, 277.0, -999.0, 265.0]
      qcg   (:,1) = [2.0e-4, -2.0, 1.0e-4, 0.3, 4.0e-4, 1.0e-4, &
                     -1.0, 5.0e-5, 2.0e-4, 8.0e-5, 6.0e-4, 0.0]
      qvg   (:,1) = [0.0090, 0.0, 0.0018, 0.5, 0.0095, 0.0012, &
                     0.0, 0.0055, 0.0024, 0.0046, 0.0, 0.0018]
      mavail(:,1) = [0.75, 0.50, 0.65, 0.45, 1.00, 1.00, 1.00, 0.03, 0.35, 0.60, 0.98, 0.40]

      do k = 1, nsl
        tso(1,k,1) = 291.0 - 0.90 * real(k - 1)
        tso(2,k,1) = 270.0 + 0.80 * real(k - 1)
        tso(3,k,1) = 265.0 + 1.00 * real(k - 1)
        tso(4,k,1) = 261.0 + 1.30 * real(k - 1)
        tso(5,k,1) = 287.0
        tso(6,k,1) = min(271.0, 258.0 + 1.20 * real(k - 1))
        tso(7,k,1) = min(271.0, 262.0 + 1.00 * real(k - 1))
        tso(8,k,1) = 302.0 - 0.70 * real(k - 1)
        tso(9,k,1) = 272.0 + 0.90 * real(k - 1)
        tso(10,k,1) = 276.0 + 0.60 * real(k - 1)
        tso(11,k,1) = 291.0 - 0.30 * real(k - 1)
        tso(12,k,1) = 265.0 + 1.20 * real(k - 1)
        soilmois(1,k,1) = 0.260 + 0.004 * real(k - 1)
        soilmois(2,k,1) = 0.320 + 0.003 * real(k - 1)
        soilmois(3,k,1) = 0.200 + 0.005 * real(k - 1)
        soilmois(4,k,1) = 0.350 + 0.002 * real(k - 1)
        soilmois(5,k,1) = 1.0
        soilmois(6,k,1) = 1.0
        soilmois(7,k,1) = 1.0
        soilmois(8,k,1) = 0.025 + 0.001 * real(k - 1)
        soilmois(9,k,1) = 0.240 + 0.003 * real(k - 1)
        soilmois(10,k,1) = 0.280 + 0.004 * real(k - 1)
        soilmois(11,k,1) = 0.400 + 0.003 * real(k - 1)
        soilmois(12,k,1) = 0.220 + 0.005 * real(k - 1)
      end do
    end if

    albbck = alb
    z0 = 0.0
    znt = 0.0
    qsg = 0.0
    qsfc = 0.0
    dew = 0.0
    tsnav = 0.0
    chklowq = 1.0
    sfcexc = 0.0
    hfx = 0.0
    qfx = 0.0
    lh = 0.0
    grdflx = 0.0
    !-- nonzero incoming accumulators so step 1 cannot hide a dropped "+="
    !-- for the fields the ktau==1 block does NOT reset (sfcevp, acsnow).
    sfcevp = 7.5
    acsnow = 3.25
    sfcrunoff = 2.0
    udrunoff = 1.0
    acrunoff = 4.0
    snowfallac = 0.5
    snom = 0.25
    smavail = 0.0
    smmax = 0.0
    rhosnf = 0.0
    precipfr = 0.0
    keepfr3dflag = 0.0
    sh2o = soilmois
    smfr3d = 0.0

    landusef = 0.0
    soilctop = 0.0
    do i = 1, ncol
      landusef(i,ivgtyp(i,1),1) = 1.0
      soilctop(i,isltyp(i,1),1) = 1.0
    end do

    call ruclsminit(sh2o, smfr3d, tso, soilmois, isltyp, ivgtyp, &
        mminlu, xice, mavail, nsl, iswater, isice, znt, .false., .true., &
        ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, 1, nsl, &
        its, ite, jts, jte, 1, nsl)

    do step = 1, 2
      ktau = step

      soilmois_i = soilmois
      sh2o_i = sh2o
      tso_i = tso
      smfr3d_i = smfr3d
      keepfr_i = keepfr3dflag
      snow_i = snow
      snowh_i = snowh
      snowc_i = snowc
      canwat_i = canwat
      snoalb_i = snoalb
      alb_i = alb
      emiss_i = emiss
      lai_i = lai
      mavail_i = mavail
      sfcexc_i = sfcexc
      z0_i = z0
      znt_i = znt
      soilt_i = soilt
      hfx_i = hfx
      qfx_i = qfx
      lh_i = lh
      sfcevp_i = sfcevp
      sfcrunoff_i = sfcrunoff
      udrunoff_i = udrunoff
      acrunoff_i = acrunoff
      grdflx_i = grdflx
      acsnow_i = acsnow
      snom_i = snom
      qvg_i = qvg
      qcg_i = qcg
      dew_i = dew
      qsfc_i = qsfc
      qsg_i = qsg
      chklowq_i = chklowq
      soilt1_i = soilt1
      tsnav_i = tsnav
      smavail_i = smavail
      smmax_i = smmax
      rhosnf_i = rhosnf
      precipfr_i = precipfr
      snowfallac_i = snowfallac

      call scrub(10, stack_pad)
      call lsmruc(0, dt, ktau, nsl, zs, rainbl, snow, snowh, snowc,  &
          frzfrac, frpcpn, rhosnf, precipfr, z3d, p8w, t3d, qv3d,    &
          qc3d, rho3d, glw, gsw, emiss, chklowq, chs, flqc, flhc,    &
          mavail, canwat, vegfra, alb, znt, z0, snoalb, albbck, lai, &
          mminlu, landusef, nlcat, mosaic_lu, mosaic_soil, soilctop, &
          nscat, qsfc, qsg, qvg, qcg, dew, soilt1, tsnav, tbot,      &
          ivgtyp, isltyp, xland, iswater, isice, xice,               &
          xice_threshold, cp, rovcp, g0, lv, stbolt, soilmois, sh2o, &
          smavail, smmax, tso, soilt, hfx, qfx, lh, sfcrunoff,       &
          udrunoff, acrunoff, sfcexc, sfcevp, grdflx, snowfallac,    &
          acsnow, snom, smfr3d, keepfr3dflag, myj, shdmin, shdmax,   &
          rdlai2d, ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, &
          kms, kme, its, ite, jts, jte, kts, kte)

      do i = 1, ncol
        do k = 1, nsl
          write(unit, '(*(g0,:,","))') run, step, trim(names(i)), k,   &
              dt, ktau, myj, frpcpn, mosaic_lu, mosaic_soil, rdlai2d,  &
              iswater, isice, xice_threshold, nlcat, nscat, lutype,    &
              cp, rovcp, g0, lv, stbolt, zs(k), ivgtyp(i,1),           &
              isltyp(i,1), xland(i,1), xice(i,1), tbot(i,1),           &
              shdmin(i,1), shdmax(i,1), vegfra(i,1), rainbl(i,1),      &
              frzfrac(i,1), glw(i,1), gsw(i,1), chs(i,1), flqc(i,1),   &
              flhc(i,1), albbck(i,1), z3d(i,kms,1), p8w(i,kms,1),      &
              t3d(i,kms,1), qv3d(i,kms,1), qc3d(i,kms,1),              &
              rho3d(i,kms,1),                                          &
              snow_i(i,1), snowh_i(i,1), snowc_i(i,1), canwat_i(i,1),  &
              snoalb_i(i,1), alb_i(i,1), emiss_i(i,1), lai_i(i,1),     &
              mavail_i(i,1), sfcexc_i(i,1), z0_i(i,1), znt_i(i,1),     &
              soilt_i(i,1), hfx_i(i,1), qfx_i(i,1), lh_i(i,1),         &
              sfcevp_i(i,1), sfcrunoff_i(i,1), udrunoff_i(i,1),        &
              acrunoff_i(i,1), grdflx_i(i,1), acsnow_i(i,1),           &
              snom_i(i,1), qvg_i(i,1), qcg_i(i,1), dew_i(i,1),         &
              qsfc_i(i,1), qsg_i(i,1), chklowq_i(i,1), soilt1_i(i,1),  &
              tsnav_i(i,1), smavail_i(i,1), smmax_i(i,1),              &
              rhosnf_i(i,1), precipfr_i(i,1), snowfallac_i(i,1),       &
              soilmois_i(i,k,1), sh2o_i(i,k,1), tso_i(i,k,1),          &
              smfr3d_i(i,k,1), keepfr_i(i,k,1),                        &
              snow(i,1), snowh(i,1), snowc(i,1), canwat(i,1),          &
              snoalb(i,1), alb(i,1), emiss(i,1), lai(i,1),             &
              mavail(i,1), sfcexc(i,1), z0(i,1), znt(i,1),             &
              soilt(i,1), hfx(i,1), qfx(i,1), lh(i,1), sfcevp(i,1),    &
              sfcrunoff(i,1), udrunoff(i,1), acrunoff(i,1),            &
              grdflx(i,1), acsnow(i,1), snom(i,1), qvg(i,1),           &
              qcg(i,1), dew(i,1), qsfc(i,1), qsg(i,1), chklowq(i,1),   &
              soilt1(i,1), tsnav(i,1), smavail(i,1), smmax(i,1),       &
              rhosnf(i,1), precipfr(i,1), snowfallac(i,1),             &
              soilmois(i,k,1), sh2o(i,k,1), tso(i,k,1),                &
              smfr3d(i,k,1), keepfr3dflag(i,k,1)
        end do
      end do
    end do
  end do

  close(unit)

contains

  !-- Overwrite the stack region LSMRUC and its callees will occupy, so the
  !-- uninitialised `ilnb` at :1385 starts from a known value on the FIRST
  !-- column of every call.  Recursion is what makes the coverage deep
  !-- enough; the sum() keeps the array live so -O0 cannot elide the stores.
  recursive subroutine scrub(depth, fill)
    integer, intent(in) :: depth
    real, intent(in) :: fill
    real :: pad(4096)
    pad = fill
    if (depth > 0) call scrub(depth - 1, fill)
    if (sum(pad) < -1.0e30) write(*, *) 'unreachable'
  end subroutine scrub

end program run_ruc_lsmruc_oracle
