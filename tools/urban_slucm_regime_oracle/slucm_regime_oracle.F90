! Regime column oracle for the single-layer urban canopy model,
! sf_urban_physics = 1 (physics verification lane, 2026-10-07).
!
! Same argument driver as tools/urban_wrf471_oracle/ucm_column_oracle.F90
! (same CSV layout, so gpuwm.verify.urban_ucm_oracle.load_variant reads it),
! with the nine branch scenarios replaced by eighteen weather regimes:
! convective afternoons, stable and arctic nights, snow-season and
! freezing-rain days, a coastal marine column, high-terrain day and night
! (air density of about 1,900 m), a desert surface, a windstorm, a dew night,
! the canopy-wind edge, a long-step irrigation night, a near-calm column and
! a first level just above the roof.  Every regime runs for every urban
! category of the table.  The regime list is REGIMES.csv beside this file.
!
! Drives the byte-unmodified WRF phys/module_sf_urban.F (v4.6.1 here): its own
! urban_param_init reads URBPARM.TBL or URBPARM_LCZ.TBL from the working
! directory, and `urban` is then called column by column, several consecutive
! steps per column, with the prognostic state carried from step to step
! exactly as module_sf_noahdrv.F:1360-1590 carries it through the *_URB2D and
! *_URB3D arrays.  Every value in the fixture is either a word this driver put
! into an argument before the call or a word `urban` put there during it.
!
! Usage:  run_ucm TABLE VARIANT OUT_PREFIX [CH TS AH ALH IMP IRI GR FGR BOUND]
!
!   TABLE    nlcd  -> use_wudapt_lcz = 0, URBPARM.TBL (3 categories)
!            lcz   -> use_wudapt_lcz = 1, URBPARM_LCZ.TBL (11 categories)
!   VARIANT  a label written into every row
!   The optional switches OVERRIDE the table's own values after
!   urban_param_init has read it.  They are the module variables
!   read_param and urban consult (CH_SCHEME_DATA, TS_SCHEME_DATA, AHOPTION,
!   ALHOPTION, IMP_SCHEME, IRI_SCHEME, GROPTION, FGR, BOUNDR/B/G_DATA), set
!   to exactly what a table carrying those rows would have set them to.
!   Any value < -8 leaves the table's own value in place.
!
! Writes OUT_PREFIX-table.csv (the per-category parameters read_param hands
! `urban`, and the scalar switches) and OUT_PREFIX.csv (one row per column
! step: inputs, state before, outputs, state after).
!
! What is NOT exercised, and why:
!   * slucm_distributed_drag (distributed_aerodynamics_option): gridded
!     Z0_URB2D/LF_URB2D_S morphology is not carried by the engine; the option
!     stays refused there.  urban_param_init is called with .false.
!   * the NUDAPT arm (mh_urb > 0, module_sf_urban.F:649-788): gridded
!     morphology again; every row passes mh_urb = 0, WRF's "USING DEFAULT
!     URBAN MORPHOLOGY" arm (urban_var_init, :2794-2809).
!   * SHADOW = .true.: hard-coded .false. at :610, the arm is dead in WRF.

program run_ucm
  use module_sf_urban
  implicit none

  integer, parameter :: nsoil = 4
  integer, parameter :: nstep = 6
  integer, parameter :: nscen = 18
  real :: dzr(nsoil), dzb(nsoil), dzg(nsoil)
  character(len=16) :: table, variant
  character(len=512) :: prefix, arg
  integer :: use_lcz, ut, sc, st, k, u, nrow, i
  integer :: ov(9)
  real :: ovfgr

  ! --- per-call arguments (module_sf_noahdrv.F:1321-1452 names) -----------
  logical :: lsolar
  integer :: utype, jmonth
  real :: ta, qa, ua, u1, v1, ssg, ssgd, ssgq, llg, rain, rhoo, za
  real :: declin, cosz, omg, xlat, delt, znt, chs, chs2
  real :: tr, tb, tg, tc, qc, uc
  real :: trl(nsoil), tbl(nsoil), tgl(nsoil)
  real :: xxxr, xxxb, xxxg, xxxc
  real :: ts, qs, sh, lh, lh_kinematic, sw, alb, lw, g, rn, psim, psih
  real :: gz1oz0, u10, v10, th2, q2, ust
  real :: cmr_urb, chr_urb, cmc_urb, chc_urb, cmgr_urb, chgr_urb
  real :: mh_urb, stdh_urb, lf_urb(4), lp_urb, hgt_urb, frc_urb, lb_urb
  real :: zo_check
  real :: cmcr, tgr, tgrl(nsoil), smr(nsoil)
  real :: drelr, drelb, drelg, flxhumr, flxhumb, flxhumg
  real :: lf_urb_s, z0_urb, vegfrac

  ! --- scenario ------------------------------------------------------------
  real :: tsurf0, tlay0(3), zr_u, zdc_u, z0c_u

  if (command_argument_count() < 3) then
    write(*, '(A)') 'usage: run_ucm TABLE VARIANT OUT_PREFIX [CH TS AH ALH IMP IRI GR FGR BOUND]'
    stop 2
  end if
  call get_command_argument(1, table)
  call get_command_argument(2, variant)
  call get_command_argument(3, prefix)
  ov = -9
  ovfgr = -9.
  do i = 1, 9
    if (command_argument_count() >= 3 + i) then
      call get_command_argument(3 + i, arg)
      if (i == 8) then
        read(arg, *) ovfgr
      else if (i == 9) then
        read(arg, *) ov(9)
      else
        read(arg, *) ov(i)
      end if
    end if
  end do

  if (trim(table) == 'nlcd') then
    use_lcz = 0
  else if (trim(table) == 'lcz') then
    use_lcz = 1
  else
    write(*, '(A)') 'TABLE must be nlcd or lcz'
    stop 2
  end if

  dzr = 0.
  dzb = 0.
  dzg = 0.
  call urban_param_init(dzr, dzb, dzg, nsoil, 1, use_lcz, .false.)

  if (ov(1) > -8) ch_scheme_data = ov(1)
  if (ov(2) > -8) ts_scheme_data = ov(2)
  if (ov(3) > -8) ahoption = ov(3)
  if (ov(4) > -8) alhoption = ov(4)
  if (ov(5) > -8) imp_scheme = ov(5)
  if (ov(6) > -8) iri_scheme = ov(6)
  if (ov(7) > -8) groption = ov(7)
  if (ovfgr > -8.) fgr = ovfgr
  if (ov(9) > -8) then
    boundr_data = ov(9)
    boundb_data = ov(9)
    boundg_data = ov(9)
  end if

  call write_table()

  open(newunit=u, file=trim(prefix)//'.csv', status='replace', action='write')
  call write_header()
  nrow = 0
  lsolar = .false.

  do ut = 1, icate
    zr_u = zr_tbl(ut)
    zdc_u = zdc_tbl(ut)
    z0c_u = z0c_tbl(ut)
    do sc = 1, nscen
      call scenario_init()
      do st = 1, nstep
        call scenario_forcing()
        call write_inputs()
        call urban(lsolar, nsoil, nsoil, nsoil, dzr, dzb, dzg,           &
                   utype, ta, qa, ua, u1, v1, ssg, ssgd, ssgq, llg, rain,  &
                   rhoo, za, declin, cosz, omg, xlat, delt, znt,           &
                   chs, chs2,                                              &
                   tr, tb, tg, tc, qc, uc,                                 &
                   trl, tbl, tgl,                                          &
                   xxxr, xxxb, xxxg, xxxc,                                 &
                   ts, qs, sh, lh, lh_kinematic,                           &
                   sw, alb, lw, g, rn, psim, psih,                         &
                   gz1oz0,                                                 &
                   cmr_urb, chr_urb, cmc_urb, chc_urb,                     &
                   u10, v10, th2, q2, ust, mh_urb, stdh_urb, lf_urb,       &
                   lp_urb, hgt_urb, frc_urb, lb_urb, zo_check,             &
                   cmcr, tgr, tgrl, smr, cmgr_urb, chgr_urb, jmonth,       &
                   drelr, drelb, drelg, flxhumr, flxhumb, flxhumg,         &
                   lf_urb_s, z0_urb, vegfrac)
        call write_outputs()
        nrow = nrow + 1
      end do
    end do
  end do
  close(u)
  write(*, '(A,A,A,I0,A)') 'run_ucm ', trim(variant), ': ', nrow, ' rows'

contains

  ! urban_var_init (module_sf_urban.F:2840-2889) for a column whose surface
  ! and soil start from tsurf0/tlay0, plus the Registry's zero start for the
  ! CM*/CH*_SFCDIF exchange coefficients (Registry.EM_COMMON:984-989).
  subroutine scenario_init()
    real :: sm0
    utype = ut
    jmonth = 7
    sm0 = 0.2
    select case (sc)
    case (1)            ! convective hot afternoon
      tsurf0 = 318.
      tlay0 = (/ 312., 305., 298. /)
    case (2)            ! convective tropical humid
      tsurf0 = 307.
      tlay0 = (/ 304., 302., 300. /)
      sm0 = 0.35
    case (3)            ! convective shower on hot pavement (25 mm/h)
      tsurf0 = 305.
      tlay0 = (/ 303., 300., 297. /)
    case (4)            ! stable night, clear and calm
      tsurf0 = 283.
      tlay0 = (/ 287., 290., 292. /)
    case (5)            ! stable night, strong cold inversion
      tsurf0 = 268.
      tlay0 = (/ 272., 276., 280. /)
      jmonth = 11
    case (6)            ! snow season day, frozen surfaces, January
      tsurf0 = 258.
      tlay0 = (/ 260., 265., 271. /)
      jmonth = 1
      sm0 = 0.3
    case (7)            ! freezing rain, December
      tsurf0 = 271.
      tlay0 = (/ 272., 274., 276. /)
      jmonth = 12
      sm0 = 0.3
    case (8)            ! arctic night, deep cold
      tsurf0 = 238.
      tlay0 = (/ 242., 250., 260. /)
      jmonth = 1
      sm0 = 0.25
    case (9)            ! coastal marine column, near saturation
      tsurf0 = 289.
      tlay0 = (/ 289., 288., 287. /)
      jmonth = 6
      sm0 = 0.3
    case (10)           ! high terrain day, about 1,900 m
      tsurf0 = 302.
      tlay0 = (/ 298., 293., 288. /)
      sm0 = 0.12
    case (11)           ! high terrain night, about 1,900 m
      tsurf0 = 266.
      tlay0 = (/ 270., 274., 278. /)
      jmonth = 10
      sm0 = 0.12
    case (12)           ! desert, hot dry surface
      tsurf0 = 335.
      tlay0 = (/ 325., 315., 305. /)
      sm0 = 0.07
    case (13)           ! windstorm, short step
      tsurf0 = 291.
      tlay0 = (/ 291., 290., 289. /)
      jmonth = 3
    case (14)           ! humid night: dew on the green roof (EPGR <= 0)
      tsurf0 = 286.
      tlay0 = (/ 288., 289., 290. /)
    case (15)           ! first level just above ZDC+Z0C+2 (canopy-wind edge)
      tsurf0 = 300.
      tlay0 = (/ 299., 297., 295. /)
    case (16)           ! night after rain, irrigation hour, 300 s step
      tsurf0 = 289.
      tlay0 = (/ 290., 291., 291. /)
      sm0 = 0.38
    case (17)           ! near calm, sunny
      tsurf0 = 310.
      tlay0 = (/ 306., 301., 297. /)
      sm0 = 0.1
    case (18)           ! first level 3 m above the roof, afternoon
      tsurf0 = 312.
      tlay0 = (/ 308., 302., 297. /)
    end select

    tr = tsurf0 + 0.
    tb = tsurf0 - 2.
    tg = tsurf0 + 1.
    tc = tsurf0 - 1.
    qc = 0.008
    uc = 0.
    xxxr = 0.
    xxxb = 0.
    xxxg = 0.
    xxxc = 0.
    trl(1) = tlay0(1) + 0.
    trl(2) = 0.5 * (tlay0(1) + tlay0(2))
    trl(3) = tlay0(2) + 0.
    trl(4) = tlay0(2) + (tlay0(3) - tlay0(2)) * 0.29
    tbl = trl
    tgl(1) = tlay0(1)
    tgl(2) = tlay0(2)
    tgl(3) = tlay0(3)
    tgl(4) = tlay0(3) + 1.
    drelr = 0.
    drelb = 0.
    drelg = 0.
    flxhumr = 0.
    flxhumb = 0.
    flxhumg = 0.
    cmcr = 0.
    tgr = tsurf0 + 0.
    tgrl = trl
    smr(1) = sm0
    smr(2) = sm0
    smr(3) = sm0 + 0.02
    smr(4) = 0.
    cmr_urb = 0.
    chr_urb = 0.
    cmc_urb = 0.
    chc_urb = 0.
    cmgr_urb = 0.
    chgr_urb = 0.
    mh_urb = 0.
    stdh_urb = 0.
    lf_urb = 0.
    lp_urb = 0.
    hgt_urb = 0.
    lb_urb = 0.
    frc_urb = frc_urb_tbl(ut)
    zo_check = 0.
    lf_urb_s = 0.
    z0_urb = 0.
    vegfrac = 0.
  end subroutine scenario_init

  ! Forcing for step st of regime sc.  Regime values are typical of the
  ! named weather; the steps drift some of them so the carried state moves.
  subroutine scenario_forcing()
    real :: s, dir_u, dir_v
    s = real(st - 1)
    delt = 60.
    declin = 0.35
    xlat = 38.5
    za = max(35., zr_u + 15.)
    chs = 0.02
    chs2 = 0.015
    rain = 0.
    dir_u = 0.8
    dir_v = -0.6
    select case (sc)
    case (1)
      ta = 305. + 0.4 * s
      qa = 0.010
      ua = 2.0
      ssg = 950. - 10. * s
      llg = 420.
      rhoo = 1.12
      omg = 0.30 + 0.004 * s
      cosz = 0.9
      chs = 0.03
      chs2 = 0.022
    case (2)
      ta = 303. + 0.1 * s
      qa = 0.021
      ua = 3.0
      ssg = 800.
      llg = 450.
      rhoo = 1.13
      omg = 0.10
      cosz = 0.95
      xlat = 1.3
      declin = 0.05
      dir_u = -0.6
      dir_v = 0.8
    case (3)
      ta = 296.
      qa = 0.017
      ua = 8.
      ssg = 150.
      llg = 420.
      rain = 25.
      rhoo = 1.16
      omg = 0.5
      cosz = 0.6
    case (4)
      ta = 285. - 0.3 * s
      qa = 0.006
      ua = 0.4
      ssg = 0.
      llg = 280.
      rhoo = 1.22
      omg = -2.6
      cosz = -0.5
      chs = 0.002
      chs2 = 0.0015
    case (5)
      ta = 271. - 0.2 * s
      qa = 0.0025
      ua = 1.2
      ssg = 0.
      llg = 220.
      rhoo = 1.30
      omg = 2.9
      cosz = -0.6
      chs = 0.003
      chs2 = 0.002
      declin = -0.25
    case (6)
      ta = 255. + 0.2 * s
      qa = 0.0008
      ua = 4.
      ssg = 350.
      llg = 190.
      rhoo = 1.37
      omg = 0.1
      cosz = 0.35
      declin = -0.38
      xlat = 46.
      dir_u = 0.
      dir_v = 1.
    case (7)
      ta = 272.5
      qa = 0.0042
      ua = 6.
      ssg = 60.
      llg = 300.
      rain = 2.0
      rhoo = 1.29
      omg = -0.4
      cosz = 0.3
      declin = -0.40
      dir_u = -1.
      dir_v = 0.
    case (8)
      ta = 235. - 0.1 * s
      qa = 0.0002
      ua = 2.
      ssg = 0.
      llg = 140.
      rhoo = 1.47
      omg = 3.0
      cosz = -0.2
      declin = -0.40
      xlat = 64.8
      chs = 0.004
      chs2 = 0.003
    case (9)
      ta = 288.
      qa = 0.0108
      ua = 9.
      ssg = 200.
      llg = 380.
      rhoo = 1.22
      omg = -1.0
      cosz = 0.4
      xlat = 37.8
      dir_u = 1.
      dir_v = 0.
    case (10)
      ta = 290. + 0.3 * s
      qa = 0.006
      ua = 5.
      ssg = 1050.
      llg = 300.
      rhoo = 0.97
      omg = 0.2
      cosz = 0.92
      xlat = 39.7
    case (11)
      ta = 268. - 0.2 * s
      qa = 0.0025
      ua = 3.
      ssg = 0.
      llg = 200.
      rhoo = 0.99
      omg = -2.4
      cosz = -0.4
      xlat = 39.7
      chs = 0.004
      chs2 = 0.003
    case (12)
      ta = 316.
      qa = 0.0025
      ua = 4.
      ssg = 1000.
      llg = 400.
      rhoo = 1.08
      omg = 0.4
      cosz = 0.9
      xlat = 33.4
    case (13)
      ta = 290.
      qa = 0.008
      ua = 28.
      ssg = 300.
      llg = 350.
      rhoo = 1.18
      omg = -0.6
      cosz = 0.5
      delt = 30.
      dir_u = 0.6
      dir_v = 0.8
    case (14)
      ta = 291.
      qa = 0.0165
      ua = 1.5
      ssg = 0.
      llg = 360.
      rhoo = 1.19
      omg = 2.6
      cosz = -0.5
    case (15)
      ta = 301.
      qa = 0.010
      ua = 5.
      ssg = 700.
      llg = 380.
      rhoo = 1.15
      omg = 0.3
      cosz = 0.8
      if (zdc_u + z0c_u + 2. + 1. < zr_u + 2.) then
        za = zdc_u + z0c_u + 3.
      else
        za = zdc_u + z0c_u + 2.5
      end if
    case (16)
      ta = 290.
      qa = 0.011
      ua = 2.
      ssg = 0.
      llg = 350.
      if (st <= 2) then
        rain = 3.0
      end if
      rhoo = 1.19
      omg = 2.4
      cosz = -0.3
      delt = 300.
    case (17)
      ta = 304.
      qa = 0.009
      ua = 0.15
      ssg = 850.
      llg = 400.
      rhoo = 1.14
      omg = 0.05
      cosz = 0.9
      dir_u = 0.6
      dir_v = 0.8
    case (18)
      ta = 306.
      qa = 0.011
      ua = 4.
      ssg = 900.
      llg = 410.
      rhoo = 1.13
      omg = 0.6
      cosz = 0.85
      za = zr_u + 3.
    end select
    u1 = dir_u * ua
    v1 = dir_v * ua
    ssgd = 0.8 * ssg
    ssgq = ssg - ssgd
    znt = 0.5
  end subroutine scenario_forcing

  subroutine write_table()
    integer :: ut2, v
    real :: zr, sigma_zed, z0c, z0hc, zdc, svf, r, rw, hgt, ah, capr, capb, capg
    real :: aksr, aksb, aksg, albr, albb, albg, epsr, epsb, epsg
    real :: z0r, z0b, z0g, z0hb, z0hg, betr, betb, betg
    real :: trlend, tblend, tglend, akanda, alh
    integer :: numdir, numhgt, boundr, boundb, boundg, chs_, tss_
    real :: street_direction(maxdirs), street_width(maxdirs), building_width(maxdirs)
    real :: height_bin(maxhgts), hpercent_bin(maxhgts)
    open(newunit=v, file=trim(prefix)//'-table.csv', status='replace', action='write')
    write(v, '(A)') 'utype,zr,sigma_zed,z0c,z0hc,zdc,svf,r,rw,hgt,ah,alh,' //         &
         'capr,capb,capg,aksr,aksb,aksg,albr,albb,albg,epsr,epsb,epsg,' //          &
         'z0r,z0b,z0g,z0hb,z0hg,betr,betb,betg,trlend,tblend,tglend,akanda_urban,' // &
         'frc_urb,building_width1,street_width1'
    do ut2 = 1, icate
      call read_param(ut2, zr, sigma_zed, z0c, z0hc, zdc, svf, r, rw, hgt,    &
                      ah, capr, capb, capg, aksr, aksb, aksg, albr, albb,     &
                      albg, epsr, epsb, epsg, z0r, z0b, z0g, z0hb, z0hg,      &
                      betr, betb, betg, trlend, tblend, tglend,               &
                      numdir, street_direction, street_width,                 &
                      building_width, numhgt, height_bin, hpercent_bin,       &
                      boundr, boundb, boundg, chs_, tss_, akanda, alh)
      write(v, '(I0)', advance='no') ut2
      write(v, '(38(",",ES16.8E3))') zr, sigma_zed, z0c, z0hc, zdc, svf, r, rw, &
           hgt, ah, alh, capr, capb, capg, aksr, aksb, aksg, albr, albb, albg,  &
           epsr, epsb, epsg, z0r, z0b, z0g, z0hb, z0hg, betr, betb, betg,       &
           trlend, tblend, tglend, akanda, frc_urb_tbl(ut2), building_width(1), &
           street_width(1)
    end do
    close(v)
    open(newunit=v, file=trim(prefix)//'-switches.csv', status='replace', action='write')
    write(v, '(A)') 'name,value'
    write(v, '(A,I0)') 'boundr,', boundr_data
    write(v, '(A,I0)') 'boundb,', boundb_data
    write(v, '(A,I0)') 'boundg,', boundg_data
    write(v, '(A,I0)') 'ch_scheme,', ch_scheme_data
    write(v, '(A,I0)') 'ts_scheme,', ts_scheme_data
    write(v, '(A,I0)') 'ahoption,', ahoption
    write(v, '(A,I0)') 'alhoption,', alhoption
    write(v, '(A,I0)') 'imp_scheme,', imp_scheme
    write(v, '(A,I0)') 'iri_scheme,', iri_scheme
    write(v, '(A,I0)') 'groption,', groption
    write(v, '(A,ES16.8E3)') 'fgr,', fgr
    write(v, '(A,ES16.8E3)') 'oasis,', oasis
    do k = 1, nsoil
      write(v, '(A,I0,A,ES16.8E3)') 'dzr', k, ',', dzr(k)
    end do
    do k = 1, nsoil
      write(v, '(A,I0,A,ES16.8E3)') 'dzb', k, ',', dzb(k)
    end do
    do k = 1, nsoil
      write(v, '(A,I0,A,ES16.8E3)') 'dzg', k, ',', dzg(k)
    end do
    do k = 1, 4
      write(v, '(A,I0,A,ES16.8E3)') 'dzgr', k, ',', dzgr(k)
    end do
    do k = 1, 3
      write(v, '(A,I0,A,ES16.8E3)') 'porimp', k, ',', porimp(k)
    end do
    do k = 1, 3
      write(v, '(A,I0,A,ES16.8E3)') 'dengimp', k, ',', dengimp(k)
    end do
    do k = 1, 24
      write(v, '(A,I0,A,ES16.8E3)') 'ahdiuprf', k, ',', ahdiuprf(k)
    end do
    do k = 1, 4
      write(v, '(A,I0,A,ES16.8E3)') 'alhseason', k, ',', alhseason(k)
    end do
    do k = 1, 48
      write(v, '(A,I0,A,ES16.8E3)') 'alhdiuprf', k, ',', alhdiuprf(k)
    end do
    close(v)
  end subroutine write_table

  subroutine write_header()
    character(len=4096) :: h
    h = 'variant,utype,scenario,step,jmonth,ta,qa,ua,u1,v1,ssg,llg,rain,rhoo,za,' // &
        'declin,cosz,omg,xlat,delt,znt_in,chs,chs2,frc_urb'
    h = trim(h) // ',tr_in,tb_in,tg_in,tc_in,qc_in,uc_in' //                    &
        ',trl1_in,trl2_in,trl3_in,trl4_in,tbl1_in,tbl2_in,tbl3_in,tbl4_in' //   &
        ',tgl1_in,tgl2_in,tgl3_in,tgl4_in,xxxr_in,xxxb_in,xxxg_in,xxxc_in' //   &
        ',cmr_in,chr_in,cmc_in,chc_in,cmgr_in,chgr_in,cmcr_in,tgr_in' //        &
        ',tgrl1_in,tgrl2_in,tgrl3_in,tgrl4_in,smr1_in,smr2_in,smr3_in,smr4_in' // &
        ',drelr_in,drelb_in,drelg_in,flxhumr_in,flxhumb_in,flxhumg_in'
    h = trim(h) // ',ts,qs,sh,lh,lh_kinematic,sw,alb,lw,g,rn,psim,psih,gz1oz0' // &
        ',u10,v10,th2,q2,ust,znt'
    h = trim(h) // ',tr,tb,tg,tc,qc,uc' //                                      &
        ',trl1,trl2,trl3,trl4,tbl1,tbl2,tbl3,tbl4' //                           &
        ',tgl1,tgl2,tgl3,tgl4,xxxr,xxxb,xxxg,xxxc' //                           &
        ',cmr,chr,cmc,chc,cmgr,chgr,cmcr,tgr' //                                &
        ',tgrl1,tgrl2,tgrl3,tgrl4,smr1,smr2,smr3,smr4' //                       &
        ',drelr,drelb,drelg,flxhumr,flxhumb,flxhumg'
    write(u, '(A)') trim(h)
  end subroutine write_header

  subroutine write_inputs()
    write(u, '(A,",",I0,",",I0,",",I0,",",I0)', advance='no') trim(variant), utype, sc, st, jmonth
    write(u, '(19(",",ES16.8E3))', advance='no') ta, qa, ua, u1, v1, ssg, llg, rain, &
         rhoo, za, declin, cosz, omg, xlat, delt, znt, chs, chs2, frc_urb
    write(u, '(6(",",ES16.8E3))', advance='no') tr, tb, tg, tc, qc, uc
    write(u, '(12(",",ES16.8E3))', advance='no') trl, tbl, tgl
    write(u, '(4(",",ES16.8E3))', advance='no') xxxr, xxxb, xxxg, xxxc
    write(u, '(8(",",ES16.8E3))', advance='no') cmr_urb, chr_urb, cmc_urb, chc_urb, &
         cmgr_urb, chgr_urb, cmcr, tgr
    write(u, '(8(",",ES16.8E3))', advance='no') tgrl, smr
    write(u, '(6(",",ES16.8E3))', advance='no') drelr, drelb, drelg, flxhumr, flxhumb, flxhumg
  end subroutine write_inputs

  subroutine write_outputs()
    write(u, '(19(",",ES16.8E3))', advance='no') ts, qs, sh, lh, lh_kinematic, sw, &
         alb, lw, g, rn, psim, psih, gz1oz0, u10, v10, th2, q2, ust, znt
    write(u, '(6(",",ES16.8E3))', advance='no') tr, tb, tg, tc, qc, uc
    write(u, '(12(",",ES16.8E3))', advance='no') trl, tbl, tgl
    write(u, '(4(",",ES16.8E3))', advance='no') xxxr, xxxb, xxxg, xxxc
    write(u, '(8(",",ES16.8E3))', advance='no') cmr_urb, chr_urb, cmc_urb, chc_urb, &
         cmgr_urb, chgr_urb, cmcr, tgr
    write(u, '(8(",",ES16.8E3))', advance='no') tgrl, smr
    write(u, '(6(",",ES16.8E3))') drelr, drelb, drelg, flxhumr, flxhumb, flxhumg
  end subroutine write_outputs

end program run_ucm
