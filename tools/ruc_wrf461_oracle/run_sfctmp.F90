program run_ruc_sfctmp_oracle
  ! Drives the unmodified WRF v4.6.1 module_sf_ruclsm::sfctmp and records
  ! every intent(inout)/intent(out) argument on both sides of the call.
  !
  ! sfctmp is a subroutine, so unlike its snow-preparation prologue the
  ! DISPATCH (phys/module_sf_ruclsm.F:1767-2193) needs no debugger to
  ! observe: the dispatch is the tail of the routine, so sfctmp's returned
  ! arguments ARE the dispatch's outputs.  This program therefore pins the
  ! whole routine end to end; the preparation half is already pinned
  ! independently by run_sfctmp_prep.F90 + sfctmp_prep.gdb, so any residue
  ! here localises to the dispatch.
  !
  ! Cases 1-17 are the preparation harness's forcing, spliced in verbatim
  ! so both fixtures drive sfctmp from byte-identical state.  Cases 18-24
  ! are added here for the dispatch arms that case set does not reach:
  !   18  mosaic land pack that melts out inside the step (:2077 reset
  !       after the recombination at :1979)
  !   19  non-mosaic melt-out, reached through the keep_snow_albedo latch
  !   20  bare sea ice with alb == albice, the skipped leg of :2160
  !   21  second mosaic sea-ice column at a different snow fraction
  !   22  non-mosaic snow land entered with every diagnostic nonzero
  !   23  non-mosaic snow sea ice entered with every diagnostic nonzero
  !   24  snow-free land entered with every diagnostic nonzero, USGS-RUC
  !   25  melting mosaic land, so both legs of the recombined runoff1
  !       are nonzero
  !   26  melting mosaic sea ice
  !   27  melting non-mosaic sea ice, where runoff1 IS smelt (:1968)
  !   28  saturated mosaic sand under heavy rain, the only case that
  !       drives runoff2 off zero
  !   29  warm dry grassland in the EVAPORATION regime with a nearly empty
  !       canopy: the only case where transf's transpiration is live, so pc
  !       is bound, and the only case with cn away from cfactr_data, so the
  !       canopy exponent is bound.  Also the only nonzero entry qcg and the
  !       only entry soilice/soiliqw that contradict soilm1d.
  !   30  frozen snow-free land with keepfr=1 and a second distinct
  !       smfrkeep, which is what binds the "melting and freezing is
  !       balanced" clamp at :2704-2707 as a read of smfrkeep
  !   31  non-mosaic snow on sea ice entered with a large ground flux, the
  !       second distinct nonzero entry s on a snowseaice path
  !   32  a second evaporating column at a different pc, which is what makes
  !       pc a read rather than a constant: transf only consults it when
  !       lai <= 4 (:6415-6422), so both 29 and 32 sit under that threshold
  !       with different table rows
  !   33  sublimating sea-ice pack that melts out inside the step, the
  !       sea-ice half of the :2077 reset
  !
  ! myj is .false. throughout: gpuwm's RUC lane admits myj=.false. only.
  !
  ! STACK HYGIENE.  sfctmp declares ilnb at module_sf_ruclsm.F:1385 and never
  ! initialises it.  snowsoil (:1928) and snowseaice (:1956) take it
  ! intent(inout) and only assign it when the pack clears snowtemp's snth
  ! threshold; on the thin blended path nothing assigns it and the
  ! if(ilnb.gt.1) at :4410/:5716 reads whatever gfortran left in that stack
  ! slot.  It selects the one-layer or two-layer tsnav form, so an
  ! uninitialised read is OBSERVABLE in the output.
  !
  ! This driver therefore zeroes that region of the stack before every call,
  ! with a recursive helper whose frames cover it, so the fixture pins
  ! ilnb = 0 rather than residue.  The physics module is untouched; only the
  ! driver's own stack is controlled.  Argument 2 sets the fill value and
  ! defaults to 0; running it a second time with a nonzero fill is the
  ! negative control that proves the read happens -- it moves tsnav on
  ! exactly the thin-pack cases (4, 12, 21, 26, 28) and nothing else, over
  ! all 160 columns of all 28 cases.
  use module_sf_ruclsm, only: sfctmp, ruclsm_soilvegparm, drysmc, &
      maxsmc, refsmc, wltsmc, satpsi, satdk, bb, hc, qtz, pctbl, &
      laitbl, lemitbl, z0tbl, cfactr_data
  implicit none

  integer, parameter :: ncase = 33, nzs = 9, nddzs = 14
  character(len=24), parameter :: names(ncase) = [character(len=24) :: &
      'warm_rain_canopy_drip', 'bare_soil_rain', 'deep_pack_densify', &
      'shallow_snow_mosaic', 'aged_pack_no_densify', 'new_snow_mosaic_drip', &
      'fresh_snow_keep_albedo', 'graupel_dense_new_snow', 'urban_snow_cap', &
      'sea_ice_snow_deep', 'sea_ice_snow_partial', 'sea_ice_snow_mosaic', &
      'sea_ice_bare', 'snow_water_no_depth', 'usgs_crop_rain_drip', &
      'usgs_urban_snow_cap', 'usgs_warm_pack_new_snow', &
      'melt_out_mosaic_land', 'melt_out_fresh_snow', &
      'sea_ice_bare_alb_match', 'sea_ice_snow_mosaic_thin', &
      'snow_land_entry_state', 'sea_ice_snow_entry_state', &
      'usgs_no_snow_entry_state', 'snow_land_melt_mosaic', &
      'sea_ice_snow_melt_mosaic', 'sea_ice_snow_melt', &
      'saturated_mosaic_drain', 'dry_grass_evaporating', &
      'frozen_keepfr_land', 'sea_ice_snow_flux_entry', &
      'dry_shrub_evaporating', 'sea_ice_melt_out']
  integer, parameter :: dataset(ncase) = [ &
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, &
      0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  integer, parameter :: isice_case(ncase) = [ &
      15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 24, 24, 24, &
      15, 15, 15, 15, 15, 15, 24, 15, 15, 15, 15, 15, 15, 15, 15, 15]
  integer, parameter :: veg_category(ncase) = [ &
      10, 16, 10, 10, 1, 10, 10, 12, 13, 15, 15, 15, 15, 16, 2, 1, 8, &
      10, 10, 15, 15, 5, 15, 7, 10, 15, 15, 10, 4, 9, 15, 20, 15]
  integer, parameter :: soil_category(ncase) = [ &
      6, 16, 6, 6, 4, 6, 6, 8, 6, 16, 16, 16, 16, 16, 6, 6, 6, &
      6, 6, 16, 16, 3, 16, 4, 6, 16, 16, 1, 4, 3, 16, 5, 16]
  integer, parameter :: nroot_case(ncase) = [ &
      6, 1, 6, 6, 8, 6, 6, 4, 1, 1, 1, 1, 1, 1, 4, 1, 4, &
      3, 3, 1, 1, 7, 1, 2, 6, 1, 1, 6, 5, 4, 1, 5, 1]
  real, parameter :: zsmain(nzs) = [0.0, 0.01, 0.04, 0.10, 0.30, &
      0.60, 1.00, 1.60, 3.00]
  real, parameter :: xlv = 2.5e6, cp = 1004.5, r_d = 287.0
  real, parameter :: g0_p = 9.81, cw = 4.183e6, stbolt = 5.67051e-8
  character(len=1024) :: output_path, pad_argument
  real :: stack_pad, cn_force
  logical :: force_phase
  integer :: n, k, k1, k2, unit, ktau, iland, isoil, nroot, ivgtyp, isltyp
  integer :: isice, loaded, iland_i
  real :: delt, xgeom, cq, evs, eis, r61, conflx, meltfactor
  real :: xland, prcpms, newsnms, snwe, snhei, snowfrac, snhei_force
  real :: rhosn, rhonewsn, rhosnfall, snowrat, grauprat, icerat, curat
  real :: patm, tabs, qvatm, qcatm, rho, glw, gsw, emiss, qkms, tkms, pc
  real :: mavail, cst, vegfra, alb, znt, alb_snow, alb_snow_free, lai
  real :: seaice, qwrtz, rhocs, dqm, qmin, ref, wilt, psis, bclh, ksat
  real :: sat, cn, rovcp, kqwrtz, kice, kwt, c1sn, c2sn
  real :: snweprint, snheiprint, rsm, soilt, soilt1, tsnav, dew
  real :: qvg, qsg, qcg, smelt, snoh, snflx, snom, snowfallac, acsnow
  real :: edir1, ec1, ett1, eeta, qfx, hfx, s, sublim, evapl, prcpl
  real :: fltot, runoff1, runoff2, infiltr, smf
  real :: zshalf(nzs), dtdzs(nddzs), dtdzs2(nzs), tbq(5001)
  real :: soilm1d(nzs), ts1d(nzs), smfrkeep(nzs), keepfr(nzs)
  real :: soilice(nzs), soiliqw(nzs), rstochcol(nzs), fieldcol_sf(nzs)
  real :: snwe_i, snhei_i, snowfrac_i, rhosn_i, rhonewsn_i, rhosnfall_i
  real :: snowrat_i, grauprat_i, icerat_i, curat_i, alb_snow_i, emiss_i
  real :: mavail_i, alb_i, cst_i, znt_i, soilt_i, soilt1_i
  real :: tsnav_i, dew_i, qvg_i, qsg_i, qcg_i, smelt_i
  real :: snoh_i, snflx_i, snom_i, snowfallac_i, acsnow_i, edir1_i
  real :: ec1_i, ett1_i, eeta_i, qfx_i, hfx_i, s_i
  real :: sublim_i, evapl_i, prcpl_i, fltot_i, runoff1_i, runoff2_i
  real :: infiltr_i, smf_i, rsm_i, snweprint_i, snheiprint_i
  real :: soilm1d_i(nzs)
  real :: ts1d_i(nzs)
  real :: smfrkeep_i(nzs)
  real :: keepfr_i(nzs)
  real :: soilice_i(nzs)
  real :: soiliqw_i(nzs)
  logical :: myj

  call get_command_argument(1, output_path)
  if (len_trim(output_path) == 0) then
    write(*, '(A)') 'usage: run_sfctmp OUTPUT.csv [STACK_FILL]'
    error stop 2
  end if
  call get_command_argument(2, pad_argument)
  if (len_trim(pad_argument) == 0) then
    stack_pad = 0.0
  else
    read(pad_argument, *) stack_pad
  end if

  zshalf(1) = 0.0
  do k = 2, nzs
    zshalf(k) = 0.5 * (zsmain(k - 1) + zsmain(k))
  end do

  cq = 173.15 - 0.05
  r61 = 6.1153 * 0.62198
  do k = 1, 5001
    cq = cq + 0.05
    evs = exp(17.67 * (cq - 273.15) / (cq - 29.65))
    eis = exp(22.514 - 6.15e3 / cq)
    if (cq >= 273.15) then
      tbq(k) = r61 * evs
    else
      tbq(k) = r61 * eis
    end if
  end do

  rovcp = r_d / cp
  kqwrtz = 7.7
  kice = 2.2
  kwt = 0.57
  rstochcol = 0.0
  fieldcol_sf = 0.0
  ktau = 1
  myj = .false.
  loaded = -1

  open(newunit=unit, file=trim(output_path), status='replace', action='write')
  write(unit, '(A)') 'case,case_index,k,delt,conflx,ktau,nroot,meltfactor,ivgtyp,isltyp,isice,xland,seaice,prcpms,' // &
      'newsnms,patm,tabs,qvatm,qcatm,rho,glw,gsw,qkms,tkms,pc,lai,vegfra,sat,cn,c1sn,c2sn,cw,xlv,cp,' // &
      'rovcp,g0_p,stbolt,kqwrtz,kice,kwt,qwrtz,rhocs,dqm,qmin,ref,wilt,psis,bclh,ksat,alb_snow_free,' // &
      'zsmain,zshalf,iland_before,snwe_before,snhei_before,snowfrac_before,rhosn_before,' // &
      'rhonewsn_before,rhosnfall_before,snowrat_before,grauprat_before,icerat_before,curat_before,' // &
      'alb_snow_before,emiss_before,mavail_before,alb_before,cst_before,znt_before,soilt_before,' // &
      'soilt1_before,tsnav_before,dew_before,qvg_before,qsg_before,qcg_before,smelt_before,snoh_before,' // &
      'snflx_before,snom_before,snowfallac_before,acsnow_before,edir1_before,ec1_before,ett1_before,' // &
      'eeta_before,qfx_before,hfx_before,s_before,sublim_before,evapl_before,prcpl_before,fltot_before,' // &
      'runoff1_before,runoff2_before,infiltr_before,smf_before,rsm_before,snweprint_before,' // &
      'snheiprint_before,soilm1d_before,ts1d_before,smfrkeep_before,keepfr_before,soilice_before,' // &
      'soiliqw_before,iland_after,snwe_after,snhei_after,snowfrac_after,rhosn_after,rhonewsn_after,' // &
      'rhosnfall_after,snowrat_after,grauprat_after,icerat_after,curat_after,alb_snow_after,' // &
      'emiss_after,mavail_after,alb_after,cst_after,znt_after,soilt_after,soilt1_after,tsnav_after,' // &
      'dew_after,qvg_after,qsg_after,qcg_after,smelt_after,snoh_after,snflx_after,snom_after,' // &
      'snowfallac_after,acsnow_after,edir1_after,ec1_after,ett1_after,eeta_after,qfx_after,hfx_after,' // &
      's_after,sublim_after,evapl_after,prcpl_after,fltot_after,runoff1_after,runoff2_after,' // &
      'infiltr_after,smf_after,rsm_after,snweprint_after,snheiprint_after,soilm1d_after,ts1d_after,' // &
      'smfrkeep_after,keepfr_after,soilice_after,soiliqw_after'

  do n = 1, ncase
    ! Reset ahead of the case block, not after it, so a case can enter with
    ! a saturated surface.  Cases 1-17 never assign these, so they still
    ! reach sfctmp with the preparation harness's zeros.
    qvg = 0.0
    qsg = 0.0
    qcg = 0.0
    if (dataset(n) /= loaded) then
      if (dataset(n) == 0) then
        call ruclsm_soilvegparm('MODI-RUC', 'STAS-RUC')
      else
        call ruclsm_soilvegparm('USGS-RUC', 'STAS-RUC')
      end if
      loaded = dataset(n)
    end if

    isice = isice_case(n)
    ivgtyp = veg_category(n)
    isoil = soil_category(n)
    nroot = nroot_case(n)
    iland = ivgtyp
    isltyp = isoil
    qwrtz = qtz(isoil)
    rhocs = hc(isoil) * 1.0e6
    bclh = bb(isoil)
    dqm = maxsmc(isoil) - drysmc(isoil)
    ksat = satdk(isoil)
    psis = -satpsi(isoil)
    qmin = drysmc(isoil)
    ref = refsmc(isoil)
    wilt = wltsmc(isoil)
    pc = pctbl(ivgtyp)
    lai = laitbl(ivgtyp)
    emiss = lemitbl(ivgtyp)
    ! lsmruc's snow-density constants (:744-745).  Two cases perturb them
    ! so a port that hard-codes them cannot reproduce this fixture.
    c1sn = 0.026
    c2sn = 21.0
    ! lsmruc's canopy-water capacity; varied for the same reason.
    sat = 5.0e-4
    delt = 60.0
    alb_snow_free = 0.18
    alb_snow = 0.70
    alb = alb_snow_free
    znt = z0tbl(ivgtyp)
    conflx = 40.0
    meltfactor = 1.0
    xland = 1.0
    seaice = 0.0
    patm = 0.95
    qcatm = 0.0
    rho = 1.25
    qkms = 0.010
    tkms = 0.009
    mavail = 0.60
    cst = 0.0
    prcpms = 0.0
    newsnms = 0.0
    snowrat = 0.0
    grauprat = 0.0
    icerat = 0.0
    curat = 0.0
    snwe = 0.0
    snhei_force = -1.0
    snowfrac = 0.0
    rhosn = 200.0
    rhonewsn = 100.0
    rhosnfall = 100.0
    snowfallac = 0.0
    tsnav = 0.0
    acsnow = 0.0
    dew = 0.0
    smelt = 0.0
    snoh = 0.0
    snflx = 0.0
    snom = 0.0
    edir1 = 0.0
    ec1 = 0.0
    ett1 = 0.0
    eeta = 0.0
    qfx = 0.0
    hfx = 0.0
    s = 0.0
    sublim = 0.0
    evapl = 0.0
    prcpl = 0.0
    fltot = 0.0
    runoff1 = 0.0
    runoff2 = 0.0
    infiltr = 0.0
    smf = 0.0
    rsm = 0.0
    snweprint = 0.0
    snheiprint = 0.0
    smfrkeep = 0.0
    keepfr = 0.0
    soilt1 = 0.0
    cn_force = -1.0
    force_phase = .false.

    select case (n)
    case (1)
      ! Snow-free warm grassland under rain with a nearly full canopy
      ! reservoir: the vegfrac>0.01 interception branch (:1555-1571) fills
      ! cst past sat and spills drip, rainf latches, and the snhei>0 block
      ! is skipped entirely.
      tabs = 291.0
      qvatm = 0.011
      glw = 340.0
      gsw = 520.0
      soilt = 292.0
      vegfra = 65.0
      prcpms = 4.0e-6
      cst = 4.8e-4
      ! Deliberately not alb_snow_free, so that passing alb through is
      ! distinguishable from resetting it.
      alb = 0.22
      do k = 1, nzs
        ts1d(k) = 291.0 - 0.8 * real(k - 1)
        soilm1d(k) = min(dqm, 0.24 + 0.004 * real(k - 1))
      end do
    case (2)
      ! Snow-free barren point with vegfra=0 on a half-length step: the
      ! vegfrac<=0.01 branch (:1572-1577) zeroes the canopy store and routes
      ! all rain to infwater.
      delt = 30.0
      tabs = 288.0
      qvatm = 0.008
      glw = 330.0
      gsw = 480.0
      soilt = 289.0
      vegfra = 0.0
      prcpms = 6.0e-6
      cst = 1.0e-4
      ! Deliberately not lemitbl(ivgtyp), so that passing emiss through is
      ! distinguishable from resetting it to emiss_snowfree.
      emiss = 0.93
      do k = 1, nzs
        ts1d(k) = 288.0 - 0.5 * real(k - 1)
        soilm1d(k) = min(dqm, 0.10 + 0.002 * real(k - 1))
      end do
    case (3)
      ! Deep low-density pack on a double-length step: bsn*snwe*100 clears
      ! 1.e-4 so the Koren density update at :1497-1498 actually runs,
      ! snhei>4*znt takes the third roughness branch (:1677), full cover
      ! keeps snow_mosaic=0, and a low maximum snow albedo puts albsn under
      ! 0.4 so the first leg of :1731 sets alb=albsn.
      delt = 120.0
      tabs = 271.0
      qvatm = 0.0035
      glw = 260.0
      gsw = 120.0
      soilt = 271.2
      tsnav = -2.0
      soilt1 = 271.0
      vegfra = 0.0
      alb_snow = 0.35
      rhosn = 100.0
      snwe = 0.5
      snowfrac = 1.0
      znt = 0.15
      alb = alb_snow
      do k = 1, nzs
        ts1d(k) = 272.0 + 0.3 * real(k - 1)
        soilm1d(k) = min(dqm, 0.30 + 0.003 * real(k - 1))
      end do
    case (4)
      ! Thin 0.025 m pack: snhei is below 0.0081*1.e3/rhosn so the density
      ! update is skipped, the tanh cover falls under 0.75 so :1656 turns
      ! on the mosaic, and snhei<=2*znt takes the first roughness branch.
      tabs = 266.0
      qvatm = 0.0021
      glw = 240.0
      gsw = 60.0
      soilt = 265.0
      tsnav = -8.0
      soilt1 = 265.0
      vegfra = 20.0
      rhosn = 200.0
      snwe = 0.005
      snowfrac = 0.80
      znt = 0.05
      alb = 0.40
      do k = 1, nzs
        ts1d(k) = 264.0 + 0.8 * real(k - 1)
        soilm1d(k) = min(dqm, 0.28 + 0.003 * real(k - 1))
      end do
    case (5)
      ! Aged dense forest pack: snhei clears 0.0081*1.e3/rhosn but
      ! bsn*snwe*100 stays under 1.e-4, so :1496 jumps to 777 and rhosn is
      ! left alone.  snhei lands between 2*znt and 4*znt for the middle
      ! roughness branch (:1675).  The incoming rhosnfall is a third
      ! distinct value that has to survive to the output untouched.
      tabs = 258.0
      qvatm = 0.0009
      glw = 195.0
      gsw = 25.0
      soilt = 257.0
      tsnav = -12.0
      soilt1 = 260.0
      vegfra = 45.0
      rhosn = 400.0
      rhosnfall = 250.0
      snwe = 0.036
      snowfrac = 0.90
      znt = 0.03
      alb = 0.60
      do k = 1, nzs
        ts1d(k) = 265.0 + 0.9 * real(k - 1)
        soilm1d(k) = min(dqm, 0.26 + 0.003 * real(k - 1))
      end do
    case (6)
      ! New snow and rain together on a partly covered pack: the incoming
      ! snowfrac<0.75 sets snow_mosaic at :1504, the canopy spills drip,
      ! and the mosaic split at :1585-1591 routes dripliq to infwater and
      ! dripsn onto the pack.  All four hydrometeor fractions are nonzero,
      ! including curat.  snowfallac stays small so snowfracnewsn<0.99 and
      ! the mosaic survives.
      tabs = 272.5
      qvatm = 0.0040
      glw = 290.0
      gsw = 90.0
      soilt = 271.8
      tsnav = -1.5
      soilt1 = 271.5
      vegfra = 55.0
      prcpms = 3.0e-6
      newsnms = 2.0e-6
      snowrat = 0.75
      grauprat = 0.10
      icerat = 0.10
      curat = 0.05
      rhosn = 150.0
      snwe = 0.010
      snowfrac = 0.50
      znt = 0.075
      alb = 0.45
      cst = 4.9e-4
      snowfallac = 0.02
      do k = 1, nzs
        ts1d(k) = 271.5 + 0.4 * real(k - 1)
        soilm1d(k) = min(dqm, 0.29 + 0.003 * real(k - 1))
      end do
    case (7)
      ! A full fresh-snow event over a low maximum snow albedo, with a
      ! reduced canopy capacity: snowfracnewsn saturates at 1 and rhosnfall
      ! stays under 450, so keep_snow_albedo latches at :1661, the mosaic is
      ! switched off, and the :1703-1710 correction lifts albsn to 0.7.  The
      ! incoming snowfrac>=0.75 keeps snow_mosaic=0 at :1504, so the drip is
      ! added to snwe through the non-mosaic path at :1593.
      sat = 4.0e-4
      tabs = 268.0
      qvatm = 0.0022
      glw = 235.0
      gsw = 70.0
      soilt = 267.0
      tsnav = -6.0
      soilt1 = 267.0
      vegfra = 35.0
      prcpms = 1.0e-6
      newsnms = 5.0e-6
      snowrat = 1.0
      alb_snow = 0.35
      rhosn = 120.0
      snwe = 0.020
      snowfrac = 0.85
      znt = 0.075
      alb = 0.33
      cst = 4.95e-4
      snowfallac = 8.0
      do k = 1, nzs
        ts1d(k) = 266.0 + 0.7 * real(k - 1)
        soilm1d(k) = min(dqm, 0.27 + 0.003 * real(k - 1))
      end do
    case (8)
      ! Graupel-dominated fall just under freezing: rhonewsn saturates at
      ! the 125 cap (:1520), rhonewgr at the 500 cap (:1521), and the
      ! weighted rhosnfall hits the 500 clamp at :1527, which is >=450 so
      ! keep_snow_albedo stays 0 even though snowfracnewsn saturates.
      tabs = 274.0
      qvatm = 0.0048
      glw = 300.0
      gsw = 110.0
      soilt = 272.9
      tsnav = -0.5
      soilt1 = 272.8
      vegfra = 30.0
      newsnms = 8.0e-6
      grauprat = 1.0
      rhosn = 250.0
      snwe = 0.030
      snowfrac = 0.95
      znt = 0.20
      alb = 0.55
      snowfallac = 5.0
      do k = 1, nzs
        ts1d(k) = 272.0 + 0.3 * real(k - 1)
        soilm1d(k) = min(dqm, 0.31 + 0.003 * real(k - 1))
      end do
    case (9)
      ! MODI-RUC urban snow: the tanh cover exceeds 0.75 and :1645 clamps it
      ! back to exactly 0.75, which then fails the :1656 mosaic test.
      ! znt>0.2 also blocks the roughness blend, and the incoming rhosnfall
      ! is a fourth distinct value.
      tabs = 269.0
      qvatm = 0.0025
      glw = 245.0
      gsw = 100.0
      soilt = 268.0
      tsnav = -0.5
      soilt1 = 268.0
      vegfra = 10.0
      rhosn = 180.0
      rhosnfall = 300.0
      snwe = 0.22
      snowfrac = 0.95
      znt = 0.50
      alb = 0.50
      do k = 1, nzs
        ts1d(k) = 267.0 + 0.6 * real(k - 1)
        soilm1d(k) = min(dqm, 0.20 + 0.003 * real(k - 1))
      end do
    case (10)
      ! Snowfall onto deep cold sea ice: the Zubov ice column at :1472-1478
      ! runs and the alb_snow_free floor of :1483 is the binding leg, full
      ! cover keeps snow_mosaic=0, and the very cold air pins rhonewsn on
      ! the 1000./17. floor of :1520.
      seaice = 1.0
      xland = 2.0
      tabs = 250.0
      qvatm = 0.0006
      glw = 185.0
      gsw = 30.0
      soilt = 252.0
      tsnav = -20.0
      soilt1 = 252.0
      vegfra = 0.0
      dqm = 1.0
      ref = 1.0
      qmin = 0.0
      wilt = 0.0
      newsnms = 1.0e-6
      snowrat = 0.9
      grauprat = 0.1
      rhosn = 300.0
      snwe = 0.15
      snowfrac = 1.0
      znt = 0.011
      alb = 0.70
      alb_snow_free = 0.55
      do k = 1, nzs
        ts1d(k) = min(271.4, 250.0 + 2.0 * real(k - 1))
        soilm1d(k) = 1.0
        soiliqw(k) = 0.0
        soilice(k) = 1.0
        smfrkeep(k) = 1.0
      end do
    case (11)
      ! Partly covered sea ice near the melting point: albice is pulled
      ! down to the alb_snow_free-0.05 floor by the warm ice surface,
      ! snowfrac lands between 0.75 and 1 so the non-mosaic ice blend at
      ! :1744 gives albsn<alb_snow and :1758 takes alb=albsn.
      seaice = 1.0
      xland = 2.0
      tabs = 270.0
      qvatm = 0.0040
      glw = 300.0
      gsw = 150.0
      soilt = 270.8
      tsnav = -1.0
      soilt1 = 270.8
      vegfra = 0.0
      dqm = 1.0
      ref = 1.0
      qmin = 0.0
      wilt = 0.0
      rhosn = 350.0
      snwe = 0.042
      snowfrac = 0.80
      znt = 0.011
      alb = 0.60
      alb_snow_free = 0.55
      do k = 1, nzs
        ts1d(k) = min(271.4, 270.5 + 0.1 * real(k - 1))
        soilm1d(k) = 1.0
        soiliqw(k) = 0.0
        soilice(k) = 1.0
        smfrkeep(k) = 1.0
      end do
    case (12)
      ! Thin snow on melting sea ice: the cover drops under 0.75 so the
      ! mosaic ice branch at :1740-1742 sets albsn=alb_snow and emiss=0.98,
      ! and the warm skin makes the albsn-0.1 floor of :1761 the binding
      ! leg.
      seaice = 1.0
      xland = 2.0
      tabs = 272.0
      qvatm = 0.0042
      glw = 310.0
      gsw = 70.0
      soilt = 272.9
      tsnav = -0.3
      soilt1 = 272.9
      vegfra = 0.0
      dqm = 1.0
      ref = 1.0
      qmin = 0.0
      wilt = 0.0
      rhosn = 250.0
      snwe = 0.004
      snowfrac = 0.85
      znt = 0.011
      alb = 0.65
      alb_snow_free = 0.55
      do k = 1, nzs
        ts1d(k) = min(271.4, 271.0 + 0.05 * real(k - 1))
        soilm1d(k) = 1.0
        soiliqw(k) = 0.0
        soilice(k) = 1.0
        smfrkeep(k) = 1.0
      end do
    case (13)
      ! Bare sea ice under rain: the ice column and albice are built, but
      ! snhei==0 zeroes snowfrac at :1429 and the whole snhei>0 block is
      ! skipped, so alb, emiss, znt and iland pass straight through.
      seaice = 1.0
      xland = 2.0
      tabs = 271.5
      qvatm = 0.0043
      glw = 315.0
      gsw = 95.0
      soilt = 270.9
      soilt1 = 270.9
      vegfra = 0.0
      dqm = 1.0
      ref = 1.0
      qmin = 0.0
      wilt = 0.0
      prcpms = 2.0e-6
      rhosn = 200.0
      snwe = 0.0
      snowfrac = 0.30
      znt = 0.011
      alb = 0.55
      alb_snow_free = 0.55
      do k = 1, nzs
        ts1d(k) = min(271.4, 270.8 + 0.05 * real(k - 1))
        soilm1d(k) = 1.0
        soiliqw(k) = 0.0
        soilice(k) = 1.0
        smfrkeep(k) = 1.0
      end do
    case (14)
      ! Degenerate entry state: snow water on the ground with a reported
      ! depth of exactly zero.  :1429 zeroes snowfrac, :1493 and :1600 both
      ! test snhei rather than snwe, so the whole snow block is skipped and
      ! snwe survives untouched.  A port that recomputed snhei from
      ! snwe/rhosn instead of reading the argument cannot reproduce this.
      delt = 30.0
      tabs = 270.0
      qvatm = 0.0030
      glw = 250.0
      gsw = 150.0
      soilt = 269.5
      vegfra = 0.0
      rhosn = 220.0
      rhosnfall = 400.0
      snwe = 0.012
      snhei_force = 0.0
      snowfrac = 0.60
      znt = 0.065
      alb = 0.30
      emiss = 0.87
      iland = 1
      do k = 1, nzs
        ts1d(k) = 269.0 + 0.5 * real(k - 1)
        soilm1d(k) = min(dqm, 0.15 + 0.002 * real(k - 1))
      end do
    case (15)
      ! USGS-RUC snow-free cropland under rain on a half-length step, with a
      ! reduced canopy capacity: same interception branch as case 1 but the
      ! land-use dataset, isice, z0tbl/lemitbl rows and URBAN index all
      ! differ, so a port that hard-codes any of them fails here.
      delt = 30.0
      sat = 3.5e-4
      tabs = 293.0
      qvatm = 0.012
      glw = 345.0
      gsw = 540.0
      soilt = 294.0
      vegfra = 50.0
      prcpms = 5.0e-6
      cst = 3.0e-4
      alb = 0.25
      emiss = 0.90
      do k = 1, nzs
        ts1d(k) = 293.0 - 0.9 * real(k - 1)
        soilm1d(k) = min(dqm, 0.22 + 0.004 * real(k - 1))
      end do
    case (16)
      ! USGS-RUC urban snow with non-default snow-density constants: the
      ! URBAN index is 1 here rather than 13, the compaction runs with
      ! c1sn/c2sn away from lsmruc's values, and the cover is clamped to
      ! 0.75 by :1645.
      c1sn = 0.030
      c2sn = 18.0
      tabs = 267.0
      qvatm = 0.0020
      glw = 240.0
      gsw = 95.0
      soilt = 266.5
      tsnav = -1.0
      soilt1 = 266.5
      vegfra = 8.0
      rhosn = 160.0
      snwe = 0.20
      snowfrac = 0.92
      znt = 0.45
      alb = 0.48
      alb_snow = 0.65
      alb_snow_free = 0.20
      do k = 1, nzs
        ts1d(k) = 266.0 + 0.5 * real(k - 1)
        soilm1d(k) = min(dqm, 0.18 + 0.003 * real(k - 1))
      end do
    case (17)
      ! USGS-RUC pack whose mean temperature has drifted above freezing on
      ! a 90 s step: min(0.,tsnav) at :1495 has to clamp it, otherwise the
      ! compaction runs away.  All four hydrometeor fractions are nonzero,
      ! snowfracnewsn saturates with rhosnfall<450 so keep_snow_albedo
      ! latches and the :1703-1710 correction fires again on a different
      ! dataset.
      delt = 90.0
      sat = 6.0e-4
      tabs = 265.0
      qvatm = 0.0019
      glw = 232.0
      gsw = 65.0
      soilt = 266.0
      tsnav = 1.0
      soilt1 = 266.0
      vegfra = 25.0
      newsnms = 3.0e-6
      snowrat = 0.60
      grauprat = 0.20
      icerat = 0.15
      curat = 0.05
      rhosn = 200.0
      snwe = 0.20
      snowfrac = 0.88
      znt = 0.10
      alb = 0.36
      alb_snow = 0.38
      alb_snow_free = 0.22
      cst = 2.0e-4
      snowfallac = 6.0
      do k = 1, nzs
        ts1d(k) = 265.0 + 0.6 * real(k - 1)
        soilm1d(k) = min(dqm, 0.24 + 0.003 * real(k - 1))
      end do
    case (18)
      ! The last 0.02 mm of a mosaic pack, in dry windy air: snowsoil's beta
      ! limiter sublimates the whole pack inside the step, so this is the one
      ! case that reaches :2077 "all snow is melted" -- alb=alb_snow_free and
      ! iland=ivgtyp -- AFTER the mosaic land recombination at :1979 has run.
      ! It also carries nonzero entry evapl, acsnow and hydrometeor ratios,
      ! none of which any snow path assigns.
      conflx = 55.0
      meltfactor = 2.0
      tabs = 268.0
      qvatm = 0.0002
      qcatm = 2.0e-4
      rho = 1.18
      patm = 0.92
      qkms = 0.200
      tkms = 0.150
      mavail = 0.75
      glw = 250.0
      gsw = 80.0
      soilt = 269.0
      tsnav = -3.0
      soilt1 = 269.0
      qvg = 0.0022
      qsg = 0.0022
      vegfra = 20.0
      prcpms = 3.0e-7
      rhosn = 300.0
      snwe = 2.0e-5
      snowfrac = 0.30
      znt = 0.04
      alb = 0.42
      cst = 1.0e-4
      acsnow = 3.5
      evapl = 7.0e-5
      sublim = 2.0e-5
      eeta = 1.1e-4
      qfx = 1.3e-4
      hfx = 21.0
      s = -9.0
      dew = 4.0e-6
      edir1 = 5.0e-8
      ec1 = 6.0e-8
      ett1 = 7.0e-8
      prcpl = 1.5e-5
      fltot = 0.4
      runoff1 = 1.0e-7
      runoff2 = 2.0e-7
      infiltr = 3.0e-7
      snowrat = 0.4
      grauprat = 0.3
      icerat = 0.2
      curat = 0.1
      do k = 1, nzs
        ts1d(k) = 268.0 + 0.5 * real(k - 1)
        soilm1d(k) = min(dqm, 0.20 + 0.002 * real(k - 1))
        keepfr(k) = 1.0
        smfrkeep(k) = 0.05
      end do
    case (19)
      ! The same sublimation melt-out on the NON-mosaic path.  A tiny snow
      ! fraction would normally force the mosaic on at :1656, so the route in
      ! is the keep_snow_albedo latch at :1659-1663: snowfallac is large
      ! enough that snowfracnewsn saturates and the fresh-fall density stays
      ! under 450, which sets snow_mosaic back to 0.  This reaches :2077
      ! WITHOUT the recombination having run.
      delt = 45.0
      conflx = 33.0
      meltfactor = 0.85
      tabs = 266.0
      qvatm = 0.00015
      qcatm = 1.0e-4
      rho = 1.20
      qkms = 0.220
      tkms = 0.160
      mavail = 0.55
      glw = 240.0
      gsw = 70.0
      soilt = 268.0
      tsnav = -5.0
      soilt1 = 268.0
      qvg = 0.0024
      qsg = 0.0024
      vegfra = 0.0
      newsnms = 2.0e-7
      snowrat = 1.0
      rhosn = 300.0
      snwe = 5.0e-6
      snowfrac = 0.20
      znt = 0.03
      alb = 0.44
      snowfallac = 5.0
      acsnow = 1.25
      evapl = 3.0e-5
      sublim = 1.0e-5
      eeta = 9.0e-5
      qfx = 1.0e-4
      hfx = 17.0
      s = -6.0
      smf = 12.0
      do k = 1, nzs
        ts1d(k) = 267.0 + 0.35 * real(k - 1)
        soilm1d(k) = min(dqm, 0.22 + 0.002 * real(k - 1))
      end do
    case (20)
      ! Bare sea ice whose surface albedo already equals albice, so the
      ! :2160 "if(alb.ne.albice)" rescaling of gswnew is SKIPPED.  Case 13
      ! takes the other leg.  ts1d(1) is below 263.15 K, which pins albice on
      ! the alb_snow_free leg of :1484.
      conflx = 48.0
      meltfactor = 2.0
      seaice = 1.0
      xland = 2.0
      tabs = 253.0
      qvatm = 0.0008
      qcatm = 5.0e-5
      rho = 1.35
      patm = 0.99
      qkms = 0.006
      tkms = 0.005
      glw = 190.0
      gsw = 40.0
      soilt = 255.0
      soilt1 = 255.0
      tsnav = -7.5
      snflx = -13.0
      snoh = 19.0
      snom = 6.5
      vegfra = 0.0
      dqm = 1.0
      ref = 1.0
      qmin = 0.0
      wilt = 0.0
      rhosn = 200.0
      snwe = 0.0
      snowfrac = 0.40
      znt = 0.011
      alb = 0.60
      alb_snow_free = 0.60
      eeta = 8.0e-5
      qfx = 9.0e-5
      hfx = -14.0
      s = 3.0
      evapl = 4.0e-5
      sublim = 6.0e-5
      smf = 8.0
      acsnow = 2.0
      edir1 = 1.0e-7
      ec1 = 2.0e-7
      ett1 = 3.0e-7
      runoff1 = 4.0e-7
      runoff2 = 5.0e-7
      infiltr = 6.0e-7
      cst = 2.0e-4
      mavail = 0.35
      do k = 1, nzs
        ts1d(k) = min(271.4, 254.0 + 1.5 * real(k - 1))
        soilm1d(k) = 1.0
        soiliqw(k) = 0.0
        soilice(k) = 1.0
        smfrkeep(k) = 1.0
      end do
    case (21)
      ! A second mosaic sea-ice column at a different snow fraction, entered
      ! with nonzero diagnostics.  Two independent mosaic-ice cases are what
      ! pin the (1-snowfrac)/snowfrac weights of :2044-2070 as weights rather
      ! than as either endpoint, and the entry state distinguishes the ice
      ! recombination's short output list (which leaves edir1, ec1, ett1,
      ! mavail, cst and infiltr on their forced values) from the land one.
      delt = 75.0
      conflx = 36.0
      meltfactor = 0.85
      seaice = 1.0
      xland = 2.0
      tabs = 268.0
      qvatm = 0.0031
      qcatm = 1.5e-4
      rho = 1.30
      patm = 0.97
      qkms = 0.012
      tkms = 0.011
      glw = 270.0
      gsw = 120.0
      soilt = 268.5
      tsnav = -3.0
      soilt1 = 268.5
      vegfra = 0.0
      dqm = 1.0
      ref = 1.0
      qmin = 0.0
      wilt = 0.0
      rhosn = 190.0
      snwe = 0.0080
      snowfrac = 0.45
      znt = 0.011
      alb = 0.62
      alb_snow_free = 0.52
      alb_snow = 0.68
      eeta = 5.0e-5
      qfx = 7.0e-5
      hfx = -11.0
      s = 2.5
      evapl = 2.5e-5
      sublim = 3.5e-5
      dew = 1.0e-6
      prcpl = 4.0e-6
      fltot = 0.7
      acsnow = 4.0
      edir1 = 8.0e-8
      ec1 = 9.0e-8
      ett1 = 1.0e-7
      runoff1 = 1.1e-7
      runoff2 = 1.2e-7
      infiltr = 1.3e-7
      cst = 3.0e-4
      mavail = 0.45
      do k = 1, nzs
        ts1d(k) = min(271.4, 267.5 + 0.4 * real(k - 1))
        soilm1d(k) = 1.0
        soiliqw(k) = 0.0
        soilice(k) = 1.0
        smfrkeep(k) = 1.0
      end do
    case (22)
      ! Non-mosaic snow on mixed forest, entered with every diagnostic
      ! nonzero.  snowsoil writes most of them, but evapl, acsnow and the
      ! four hydrometeor ratios are never assigned anywhere on this path, so
      ! this is the case that proves they are passed through rather than
      ! reset.  snfr = snowfrac (not 1) because the mosaic is off; the pack is
      ! deep and znt sits on the 0.2 cap of :1626 so the blended cover clears
      ! 0.75 and :1656 leaves snow_mosaic at 0.
      delt = 105.0
      conflx = 52.0
      meltfactor = 2.0
      tabs = 262.0
      qvatm = 0.0015
      qcatm = 3.0e-4
      rho = 1.28
      patm = 0.90
      qkms = 0.007
      tkms = 0.0065
      mavail = 0.85
      glw = 215.0
      gsw = 55.0
      soilt = 263.0
      tsnav = -9.0
      soilt1 = 263.0
      vegfra = 70.0
      rhosn = 240.0
      snwe = 0.300
      snowfrac = 0.90
      znt = 0.20
      alb = 0.58
      alb_snow = 0.72
      alb_snow_free = 0.16
      cst = 2.5e-4
      acsnow = 6.5
      evapl = 9.0e-5
      sublim = 4.5e-5
      eeta = 1.4e-4
      qfx = 1.6e-4
      hfx = -25.0
      s = 5.0
      dew = 3.0e-6
      edir1 = 1.4e-7
      ec1 = 1.5e-7
      ett1 = 1.6e-7
      prcpl = 2.0e-5
      fltot = 1.1
      runoff1 = 1.7e-7
      runoff2 = 1.8e-7
      infiltr = 1.9e-7
      snowrat = 0.25
      grauprat = 0.25
      icerat = 0.25
      curat = 0.25
      do k = 1, nzs
        ts1d(k) = 262.0 + 0.7 * real(k - 1)
        soilm1d(k) = min(dqm, 0.25 + 0.003 * real(k - 1))
      end do
    case (23)
      ! Non-mosaic snow on sea ice, entered with every diagnostic nonzero.
      ! The post-snowseaice block at :1965-1975 forces edir1, ec1, ett1,
      ! runoff1, runoff2, mavail, infiltr, cst and the whole soil column, so
      ! a port that instead carried the entry values through fails here.
      delt = 40.0
      conflx = 44.0
      meltfactor = 0.85
      seaice = 1.0
      xland = 2.0
      tabs = 259.0
      qvatm = 0.0011
      qcatm = 2.5e-4
      rho = 1.33
      patm = 0.98
      qkms = 0.009
      tkms = 0.008
      glw = 205.0
      gsw = 45.0
      soilt = 260.0
      tsnav = -11.0
      soilt1 = 260.0
      vegfra = 0.0
      dqm = 1.0
      ref = 1.0
      qmin = 0.0
      wilt = 0.0
      rhosn = 280.0
      snwe = 0.070
      snowfrac = 0.95
      znt = 0.011
      alb = 0.66
      alb_snow = 0.74
      alb_snow_free = 0.50
      cst = 3.5e-4
      mavail = 0.40
      acsnow = 5.5
      evapl = 6.0e-5
      sublim = 2.5e-5
      eeta = 1.0e-4
      qfx = 1.2e-4
      hfx = -19.0
      s = 4.0
      dew = 2.0e-6
      edir1 = 2.0e-7
      ec1 = 2.1e-7
      ett1 = 2.2e-7
      prcpl = 1.0e-5
      fltot = 0.9
      runoff1 = 2.3e-7
      runoff2 = 2.4e-7
      infiltr = 2.5e-7
      smf = 15.0
      do k = 1, nzs
        ts1d(k) = min(271.4, 259.5 + 1.0 * real(k - 1))
        soilm1d(k) = 0.6
        soiliqw(k) = 0.2
        soilice(k) = 0.4
        smfrkeep(k) = 0.4
        keepfr(k) = 1.0
      end do
    case (24)
      ! USGS-RUC snow-free land entered with every diagnostic nonzero, on a
      ! third distinct conflx/meltfactor and with qcatm, mavail, rho, patm,
      ! qkms and tkms all off their defaults.  soil overwrites the flux
      ! diagnostics; smelt, snweprint and snheiprint are forced to zero at
      ! :2120-2122; and soilt1, tsnav, snflx, snoh, snom, evapl, acsnow and the
      ! hydrometeor ratios survive, because nothing on the no-snow path
      ! assigns them.
      delt = 150.0
      conflx = 61.0
      meltfactor = 2.0
      tabs = 296.0
      qvatm = 0.0135
      qcatm = 4.0e-4
      rho = 1.15
      patm = 0.88
      qkms = 0.021
      tkms = 0.019
      mavail = 0.30
      glw = 350.0
      gsw = 560.0
      soilt = 297.0
      soilt1 = 295.0
      tsnav = -4.5
      snflx = 27.0
      snoh = 41.0
      snom = 8.25
      vegfra = 15.0
      prcpms = 1.0e-6
      sat = 4.5e-4
      cst = 1.2e-4
      znt = 0.02
      alb = 0.28
      emiss = 0.91
      acsnow = 7.5
      evapl = 1.0e-4
      sublim = 5.0e-5
      eeta = 1.5e-4
      qfx = 1.7e-4
      hfx = 33.0
      s = -12.0
      dew = 5.0e-6
      edir1 = 2.6e-7
      ec1 = 2.7e-7
      ett1 = 2.8e-7
      prcpl = 2.5e-5
      fltot = 1.3
      runoff1 = 2.9e-7
      runoff2 = 3.0e-7
      infiltr = 3.1e-7
      smelt = 4.0e-6
      snweprint = 0.05
      snheiprint = 0.25
      rsm = 1.0e-3
      snowrat = 0.5
      grauprat = 0.2
      icerat = 0.2
      curat = 0.1
      do k = 1, nzs
        ts1d(k) = 296.0 - 1.0 * real(k - 1)
        soilm1d(k) = min(dqm, 0.19 + 0.004 * real(k - 1))
      end do
    case (25)
      ! Melting mosaic pack over warm wet grassland.  The snow-free portion
      ! runs soil under rain, so runoff1s > 0, while snowsoil melts part of
      ! the pack, so runoff1 > 0 as well; this is the only case where both
      ! legs of the :2069 runoff1 blend are nonzero, which is what makes
      ! snowfrac there a weight rather than an on/off switch.  smelt, snoh
      ! and snom are also nonzero here for the first time.
      delt = 120.0
      conflx = 45.0
      meltfactor = 2.0
      tabs = 281.0
      qvatm = 0.0075
      rho = 1.20
      qkms = 0.020
      tkms = 0.018
      mavail = 0.65
      glw = 345.0
      gsw = 500.0
      soilt = 275.0
      tsnav = -0.1
      soilt1 = 274.0
      qvg = 0.0050
      qsg = 0.0050
      vegfra = 25.0
      prcpms = 8.0e-6
      rhosn = 250.0
      snwe = 0.004
      snowfrac = 0.30
      znt = 0.05
      alb = 0.40
      cst = 2.0e-4
      snom = 1.5
      do k = 1, nzs
        ts1d(k) = 275.0 + 0.3 * real(k - 1)
        soilm1d(k) = min(dqm, 0.34 + 0.002 * real(k - 1))
      end do
    case (26)
      ! Melting mosaic sea ice.  runoff1s stays 0 because sice never writes
      ! it, so the :2069 blend reduces to smelt*snowfrac -- which is what
      ! separates a correct recombination from one that returns smelt.
      delt = 120.0
      conflx = 42.0
      meltfactor = 2.0
      seaice = 1.0
      xland = 2.0
      tabs = 275.0
      qvatm = 0.0055
      rho = 1.26
      qkms = 0.016
      tkms = 0.014
      glw = 325.0
      gsw = 210.0
      soilt = 274.0
      tsnav = -0.1
      soilt1 = 273.5
      qvg = 0.0048
      qsg = 0.0048
      vegfra = 0.0
      dqm = 1.0
      ref = 1.0
      qmin = 0.0
      wilt = 0.0
      prcpms = 5.0e-6
      rhosn = 250.0
      snwe = 0.002
      snowfrac = 0.35
      znt = 0.011
      alb = 0.58
      alb_snow_free = 0.52
      snom = 2.5
      do k = 1, nzs
        ts1d(k) = min(271.4, 271.0 + 0.05 * real(k - 1))
        soilm1d(k) = 1.0
        soiliqw(k) = 0.0
        soilice(k) = 1.0
        smfrkeep(k) = 1.0
      end do
    case (27)
      ! Melting NON-mosaic sea ice: deep enough that the blended cover clears
      ! 0.75.  Here :1968 runoff1 = smelt reaches the output untouched by any
      ! recombination, and snom accumulates from a nonzero entry value.
      delt = 120.0
      conflx = 39.0
      meltfactor = 2.0
      seaice = 1.0
      xland = 2.0
      tabs = 274.5
      qvatm = 0.0052
      rho = 1.27
      qkms = 0.015
      tkms = 0.013
      glw = 320.0
      gsw = 200.0
      soilt = 273.6
      tsnav = -0.2
      soilt1 = 273.4
      qvg = 0.0046
      qsg = 0.0046
      vegfra = 0.0
      dqm = 1.0
      ref = 1.0
      qmin = 0.0
      wilt = 0.0
      rhosn = 250.0
      snwe = 0.050
      snowfrac = 0.95
      znt = 0.011
      alb = 0.56
      alb_snow_free = 0.52
      snom = 3.5
      do k = 1, nzs
        ts1d(k) = min(271.4, 271.0 + 0.05 * real(k - 1))
        soilm1d(k) = 1.0
        soiliqw(k) = 0.0
        soilice(k) = 1.0
        smfrkeep(k) = 1.0
      end do
    case (28)
      ! Saturated sand under 360 mm/h of rain on a five-minute step.  Every
      ! soil layer enters at porosity, so soilmoist's :6071-6073 deep-drainage
      ! term fires and runoff2 leaves zero for the only time in this fixture.
      ! Layer 1's excess goes to the surface runoff instead (:6043), so the
      ! drainage has to reach k >= 2, which needs sand's conductivity and a
      ! step this long.  This is also the fixture's only 300 s solver geometry.
      delt = 300.0
      tabs = 277.0
      qvatm = 0.0065
      rho = 1.22
      qkms = 0.014
      tkms = 0.012
      mavail = 0.95
      glw = 330.0
      gsw = 300.0
      soilt = 274.5
      tsnav = -0.2
      soilt1 = 274.0
      qvg = 0.0050
      qsg = 0.0050
      vegfra = 35.0
      prcpms = 1.0e-4
      rhosn = 250.0
      snwe = 0.006
      snowfrac = 0.30
      znt = 0.05
      alb = 0.45
      cst = 3.0e-4
      do k = 1, nzs
        ts1d(k) = 275.0 + 0.2 * real(k - 1)
        soilm1d(k) = dqm
      end do
    case (29)
      ! Warm dry grassland with a nearly empty canopy: qvatm is far below
      ! qsg, so soil takes the EVAPORATION leg of :2762 and transf's
      ! transpiration is live -- the only case in the fixture where pc is
      ! read to any effect.  cst/sat = 0.04 puts (cst/sat)**cn under the 0.25
      ! cap of :2597 as well, so cn is read too, and cn is moved off
      ! cfactr_data here and nowhere else.  Entry qcg is nonzero and entry
      ! soilice/soiliqw deliberately contradict soilm1d, so a port that
      ! carried either through instead of rebuilding it from the freezing
      ! curve fails here.  The land-use row is chosen for lai <= 4, which is
      ! what makes transf read pc at all (:6415-6422); case 32 is the same
      ! regime on a row with a different pc.
      delt = 45.0
      conflx = 37.0
      cn_force = 0.62
      force_phase = .true.
      tabs = 300.0
      qvatm = 0.0040
      glw = 350.0
      gsw = 600.0
      rho = 1.14
      qkms = 0.018
      tkms = 0.016
      mavail = 0.50
      soilt = 299.0
      vegfra = 60.0
      sat = 5.0e-4
      cst = 2.0e-5
      znt = 0.08
      alb = 0.20
      emiss = 0.94
      qcg = 3.0e-4
      smfrkeep = 0.33
      do k = 1, nzs
        ts1d(k) = 298.0 - 0.6 * real(k - 1)
        soilm1d(k) = min(dqm, 0.22 + 0.003 * real(k - 1))
        soiliqw(k) = 0.5 * soilm1d(k)
        soilice(k) = 0.1
      end do
    case (30)
      ! Frozen snow-free land entered with keepfr=1 and a second distinct
      ! smfrkeep.  :2704-2707 then clamps soilice to smfrkeep, which is the
      ! only read of smfrkeep anywhere in sfctmp's tree; case 18 supplies the
      ! other value, so neither can be hard-coded.
      delt = 60.0
      conflx = 41.0
      tabs = 266.0
      qvatm = 0.0016
      glw = 225.0
      gsw = 90.0
      rho = 1.31
      qkms = 0.011
      tkms = 0.010
      mavail = 0.70
      soilt = 269.5
      vegfra = 30.0
      prcpms = 2.0e-6
      cst = 1.5e-4
      znt = 0.06
      alb = 0.21
      emiss = 0.92
      do k = 1, nzs
        ts1d(k) = 270.6 - 0.4 * real(k - 1)
        soilm1d(k) = min(dqm, 0.30 + 0.002 * real(k - 1))
        smfrkeep(k) = 0.33
        keepfr(k) = 1.0
      end do
    case (31)
      ! Non-mosaic snow on sea ice entered with a large ground flux, so the
      ! entry s reaching snowseaice is a second distinct nonzero value.
      delt = 60.0
      conflx = 47.0
      meltfactor = 0.85
      seaice = 1.0
      xland = 2.0
      tabs = 261.0
      qvatm = 0.0013
      qcatm = 1.0e-4
      rho = 1.32
      patm = 0.96
      qkms = 0.010
      tkms = 0.009
      glw = 210.0
      gsw = 50.0
      soilt = 261.5
      tsnav = -10.0
      soilt1 = 261.5
      qcg = 1.5e-4
      vegfra = 0.0
      dqm = 1.0
      ref = 1.0
      qmin = 0.0
      wilt = 0.0
      rhosn = 260.0
      snwe = 0.090
      snowfrac = 0.97
      znt = 0.011
      alb = 0.64
      alb_snow = 0.73
      alb_snow_free = 0.48
      s = 45.0
      hfx = -22.0
      eeta = 1.1e-4
      qfx = 1.35e-4
      do k = 1, nzs
        ts1d(k) = min(271.4, 260.0 + 1.2 * real(k - 1))
        soilm1d(k) = 1.0
        soiliqw(k) = 0.0
        soilice(k) = 1.0
        smfrkeep(k) = 1.0
      end do
    case (32)
      ! A second evaporating column, on a land-use row whose lai is under the
      ! :6415 threshold with a DIFFERENT pc.  Case 29 supplies the other
      ! value; between them neither pc can be hard-coded.
      delt = 45.0
      conflx = 37.0
      cn_force = 0.62
      tabs = 301.0
      qvatm = 0.0038
      glw = 352.0
      gsw = 610.0
      rho = 1.13
      qkms = 0.019
      tkms = 0.017
      mavail = 0.50
      soilt = 300.0
      vegfra = 60.0
      sat = 5.0e-4
      cst = 2.0e-5
      znt = 0.06
      alb = 0.19
      emiss = 0.94
      do k = 1, nzs
        ts1d(k) = 299.0 - 0.6 * real(k - 1)
        soilm1d(k) = min(dqm, 0.22 + 0.003 * real(k - 1))
      end do
    case (33)
      ! The last 0.02 mm of a sea-ice pack in dry windy air: snowseaice
      ! sublimates it inside the step, so this is the sea-ice half of the
      ! :2077 "all snow is melted" reset -- alb=alb_snow_free with
      ! iland=ivgtyp already equal to isice.  It also carries a large entry
      ! ground flux s, which snowseaice leaves untouched once the pack is
      ! gone.  The air has to be warmer than the pack for the pack to go all
      ! the way: snowseaice's beta limiter (:4011-4014) is set from the
      ! ENTRY qsg, while the sublimation it subtracts at :4366 uses the qsg
      ! the surface solve leaves, so snwe only reaches amax1's zero when the
      ! skin warms during the step.
      delt = 60.0
      conflx = 51.0
      meltfactor = 2.0
      seaice = 1.0
      xland = 2.0
      tabs = 270.0
      qvatm = 0.00018
      qcatm = 1.0e-4
      rho = 1.29
      patm = 0.93
      qkms = 0.260
      tkms = 0.185
      glw = 245.0
      gsw = 75.0
      soilt = 265.0
      tsnav = -4.0
      soilt1 = 265.0
      qvg = 0.0021
      qsg = 0.0021
      vegfra = 0.0
      dqm = 1.0
      ref = 1.0
      qmin = 0.0
      wilt = 0.0
      rhosn = 300.0
      snwe = 5.0e-8
      snowfrac = 0.25
      znt = 0.011
      alb = 0.61
      alb_snow_free = 0.50
      s = 33.0
      hfx = 18.0
      eeta = 1.2e-4
      qfx = 1.4e-4
      sublim = 2.0e-5
      do k = 1, nzs
        ts1d(k) = min(271.4, 264.0 + 1.0 * real(k - 1))
        soilm1d(k) = 1.0
        soiliqw(k) = 0.0
        soilice(k) = 1.0
        smfrkeep(k) = 1.0
      end do
    end select

    if (seaice < 0.5 .and. .not. force_phase) then
      do k = 1, nzs
        soiliqw(k) = soilm1d(k)
        soilice(k) = 0.0
      end do
    end if
    ! Snow depth is the diagnostic lsmruc carries, snwe*rhowater/rhosn,
    ! unless the case deliberately forces a different entry state.
    if (snhei_force >= 0.0) then
      snhei = snhei_force
    else if (snwe > 0.0) then
      snhei = snwe * 1.0e3 / rhosn
    else
      snhei = 0.0
    end if
    dtdzs = 0.0
    dtdzs2 = 0.0
    do k = 2, nzs - 1
      k1 = 2 * k - 3
      k2 = k1 + 1
      xgeom = delt / 2.0 / (zshalf(k + 1) - zshalf(k))
      dtdzs(k1) = xgeom / (zsmain(k) - zsmain(k - 1))
      dtdzs2(k - 1) = xgeom
      dtdzs(k2) = xgeom / (zsmain(k + 1) - zsmain(k))
    end do
    cn = cfactr_data
    if (cn_force >= 0.0) cn = cn_force

    iland_i = iland
    snwe_i = snwe
    snhei_i = snhei
    snowfrac_i = snowfrac
    rhosn_i = rhosn
    rhonewsn_i = rhonewsn
    rhosnfall_i = rhosnfall
    snowrat_i = snowrat
    grauprat_i = grauprat
    icerat_i = icerat
    curat_i = curat
    alb_snow_i = alb_snow
    emiss_i = emiss
    mavail_i = mavail
    alb_i = alb
    cst_i = cst
    znt_i = znt
    soilt_i = soilt
    soilt1_i = soilt1
    tsnav_i = tsnav
    dew_i = dew
    qvg_i = qvg
    qsg_i = qsg
    qcg_i = qcg
    smelt_i = smelt
    snoh_i = snoh
    snflx_i = snflx
    snom_i = snom
    snowfallac_i = snowfallac
    acsnow_i = acsnow
    edir1_i = edir1
    ec1_i = ec1
    ett1_i = ett1
    eeta_i = eeta
    qfx_i = qfx
    hfx_i = hfx
    s_i = s
    sublim_i = sublim
    evapl_i = evapl
    prcpl_i = prcpl
    fltot_i = fltot
    runoff1_i = runoff1
    runoff2_i = runoff2
    infiltr_i = infiltr
    smf_i = smf
    rsm_i = rsm
    snweprint_i = snweprint
    snheiprint_i = snheiprint
    soilm1d_i = soilm1d
    ts1d_i = ts1d
    smfrkeep_i = smfrkeep
    keepfr_i = keepfr
    soilice_i = soilice
    soiliqw_i = soiliqw

    call scrub(8, stack_pad)
    call sfctmp(0, rstochcol, fieldcol_sf, &
        delt, ktau, conflx, n, 1, &
        nzs, nddzs, nroot, meltfactor, &
        iland, isoil, xland, ivgtyp, isltyp, prcpms, &
        newsnms, snwe, snhei, snowfrac, &
        rhosn, rhonewsn, rhosnfall, &
        snowrat, grauprat, icerat, curat, &
        patm, tabs, qvatm, qcatm, rho, &
        glw, gsw, emiss, qkms, tkms, pc, &
        mavail, cst, vegfra, alb, znt, &
        alb_snow, alb_snow_free, lai, &
        myj, seaice, isice, &
        qwrtz, rhocs, dqm, qmin, ref, wilt, psis, bclh, ksat, &
        sat, cn, zsmain, zshalf, dtdzs, dtdzs2, tbq, &
        cp, rovcp, g0_p, xlv, stbolt, cw, c1sn, c2sn, &
        kqwrtz, kice, kwt, &
        snweprint, snheiprint, rsm, &
        soilm1d, ts1d, smfrkeep, keepfr, soilt, soilt1, &
        tsnav, dew, qvg, qsg, qcg, &
        smelt, snoh, snflx, snom, snowfallac, acsnow, &
        edir1, ec1, ett1, eeta, qfx, hfx, s, sublim, &
        evapl, prcpl, fltot, runoff1, runoff2, soilice, &
        soiliqw, infiltr, smf)

    do k = 1, nzs
      write(unit, '(*(g0,:,","))') trim(names(n)), n, k, delt, conflx, ktau, nroot, meltfactor, &
          ivgtyp, isltyp, isice, xland, seaice, prcpms, newsnms, patm, tabs, &
          qvatm, qcatm, rho, glw, gsw, qkms, tkms, pc, lai, vegfra, sat, cn, &
          c1sn, c2sn, cw, xlv, cp, rovcp, g0_p, stbolt, kqwrtz, kice, kwt, &
          qwrtz, rhocs, dqm, qmin, ref, wilt, psis, bclh, ksat, &
          alb_snow_free, zsmain(k), zshalf(k), iland_i, snwe_i, snhei_i, &
          snowfrac_i, rhosn_i, rhonewsn_i, rhosnfall_i, snowrat_i, &
          grauprat_i, icerat_i, curat_i, alb_snow_i, emiss_i, mavail_i, &
          alb_i, cst_i, znt_i, soilt_i, soilt1_i, tsnav_i, dew_i, qvg_i, &
          qsg_i, qcg_i, smelt_i, snoh_i, snflx_i, snom_i, snowfallac_i, &
          acsnow_i, edir1_i, ec1_i, ett1_i, eeta_i, qfx_i, hfx_i, s_i, &
          sublim_i, evapl_i, prcpl_i, fltot_i, runoff1_i, runoff2_i, &
          infiltr_i, smf_i, rsm_i, snweprint_i, snheiprint_i, soilm1d_i(k), &
          ts1d_i(k), smfrkeep_i(k), keepfr_i(k), soilice_i(k), soiliqw_i(k), &
          iland, snwe, snhei, snowfrac, rhosn, rhonewsn, rhosnfall, snowrat, &
          grauprat, icerat, curat, alb_snow, emiss, mavail, alb, cst, znt, &
          soilt, soilt1, tsnav, dew, qvg, qsg, qcg, smelt, snoh, snflx, snom, &
          snowfallac, acsnow, edir1, ec1, ett1, eeta, qfx, hfx, s, sublim, &
          evapl, prcpl, fltot, runoff1, runoff2, infiltr, smf, rsm, &
          snweprint, snheiprint, soilm1d(k), ts1d(k), smfrkeep(k), keepfr(k), &
          soilice(k), soiliqw(k)
    end do
  end do
  close(unit)

contains

  !-- Overwrite the stack region sfctmp's frame will occupy.  Recursion is
  !-- what makes the coverage deep enough; the sum() keeps the array live so
  !-- -O0 cannot elide the stores.
  recursive subroutine scrub(depth, fill)
    integer, intent(in) :: depth
    real, intent(in) :: fill
    real :: pad(4096)
    pad = fill
    if (depth > 0) call scrub(depth - 1, fill)
    if (sum(pad) < -1.0e30) write(*, *) 'unreachable'
  end subroutine scrub

end program run_ruc_sfctmp_oracle
