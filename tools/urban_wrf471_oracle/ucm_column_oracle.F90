! Column oracle for the single-layer urban canopy model, sf_urban_physics = 1.
!
! Drives the byte-unmodified WRF v4.7.1 phys/module_sf_urban.F: its own
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
  integer, parameter :: nscen = 9
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
    utype = ut
    jmonth = 7
    select case (sc)
    case (1)            ! day, dry, unstable (surfaces warmer than air)
      tsurf0 = 303.
      tlay0 = (/ 301., 298., 295. /)
    case (2)            ! night, dry, stable
      tsurf0 = 281.
      tlay0 = (/ 284., 287., 289. /)
    case (3)            ! day, light rain
      tsurf0 = 296.
      tlay0 = (/ 295., 294., 293. /)
    case (4)            ! day, heavy rain (> 1 mm/h: IMP_SCHEME 1 BETR = 0.7)
      tsurf0 = 294.
      tlay0 = (/ 294., 293., 292. /)
    case (5)            ! night, rain then dry (retention decay, irrigation hour)
      tsurf0 = 289.
      tlay0 = (/ 290., 291., 291. /)
    case (6)            ! windy day (canopy wind above 5 m/s)
      tsurf0 = 299.
      tlay0 = (/ 298., 296., 294. /)
    case (7)            ! winter, cold, weak sun, January
      tsurf0 = 262.
      tlay0 = (/ 264., 268., 272. /)
      jmonth = 1
    case (8)            ! humid night: dew on the green roof (EPGR <= 0)
      tsurf0 = 286.
      tlay0 = (/ 288., 289., 290. /)
    case (9)            ! first level just above ZDC+Z0C+2 (canopy-wind branch)
      tsurf0 = 300.
      tlay0 = (/ 299., 297., 295. /)
    end select

    tr = tsurf0 + 0.
    tb = tsurf0 + 0.
    tg = tsurf0 + 0.
    tc = tsurf0 + 0.
    qc = 0.01
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
    smr(1) = 0.2
    smr(2) = 0.2
    smr(3) = 0.2
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

  ! Forcing for step st of scenario sc.  Deliberately not meteorologically
  ! tuned: the rows exist to reach branches.
  subroutine scenario_forcing()
    real :: s
    s = real(st - 1)
    delt = 60.
    declin = 0.35
    xlat = 38.5
    za = max(35., zr_u + 15.)
    chs = 0.02
    chs2 = 0.015
    select case (sc)
    case (1)
      ta = 299. + 0.3 * s
      qa = 0.012
      ua = 3.2
      ssg = 820. - 20. * s
      llg = 390.
      rain = 0.
      rhoo = 1.14
      omg = 0.21 + 0.004 * s
      cosz = 0.85
    case (2)
      ta = 286. - 0.2 * s
      qa = 0.007
      ua = 1.0
      ssg = 0.
      llg = 300.
      rain = 0.
      rhoo = 1.21
      omg = -2.9
      cosz = -0.4
      chs = 0.004
      chs2 = 0.003
    case (3)
      ta = 294.
      qa = 0.014
      ua = 4.5
      ssg = 250.
      llg = 400.
      rain = 0.5
      rhoo = 1.17
      omg = 0.6
      cosz = 0.6
    case (4)
      ta = 292.
      qa = 0.015
      ua = 6.
      ssg = 120.
      llg = 410.
      rain = 4.0
      rhoo = 1.18
      omg = -0.5
      cosz = 0.5
      delt = 180.
    case (5)
      ta = 290.
      qa = 0.011
      ua = 2.
      ssg = 0.
      llg = 350.
      if (st <= 2) then
        rain = 3.0
      else
        rain = 0.
      end if
      rhoo = 1.19
      omg = 2.4           ! local hour 21 (tloc = 21): the irrigation hour
      cosz = -0.3
    case (6)
      ta = 297.
      qa = 0.009
      ua = 14.
      ssg = 600.
      llg = 370.
      rain = 0.
      rhoo = 1.16
      omg = -0.8
      cosz = 0.7
    case (7)
      ta = 266.
      qa = 0.0015
      ua = 2.5
      ssg = 280.
      llg = 230.
      rain = 0.
      rhoo = 1.33
      omg = 0.05
      cosz = 0.3
      declin = -0.38
    case (8)
      ta = 291.
      qa = 0.0165
      ua = 1.5
      ssg = 0.
      llg = 360.
      rain = 0.
      rhoo = 1.19
      omg = 2.6
      cosz = -0.5
    case (9)
      ta = 301.
      qa = 0.010
      ua = 5.
      ssg = 700.
      llg = 380.
      rain = 0.
      rhoo = 1.15
      omg = 0.3
      cosz = 0.8
      if (zdc_u + z0c_u + 2. + 1. < zr_u + 2.) then
        za = zdc_u + z0c_u + 3.
      else
        za = zdc_u + z0c_u + 2.5
      end if
    end select
    u1 = 0.8 * ua
    v1 = -0.6 * ua
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
