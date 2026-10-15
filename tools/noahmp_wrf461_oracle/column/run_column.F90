! Column oracle for WRF v4.6.1 Noah-MP through its own driver.
!
! Calls, from unmodified pinned sources:
!   module_sf_noahmpdrv::NOAHMP_INIT  (cold start, which calls SNOW_INIT)
!   module_sf_noahmpdrv::noahmplsm    (one call per step, ITIMESTEP = 1..NSTEP)
! and through them NOAHMP_SFLX, NOAHMP_GLACIER, TRANSFER_MP_PARAMETERS and
! the table readers, exactly as module_surface_driver.F:3127-3181 reaches
! them (keyword arguments below carry the same actuals WRF passes,
! including the six optional MP_* precipitation amounts).
!
! The grid is NX columns by one row: its..ite = 1..NX, jts = jte = 1,
! ide = NX + 1 and jde = 2 (NOAHMP_INIT clips to ide-1 and jde-1).  Two
! atmospheric levels are carried because P_ML reads P8W3D(I,KTS+1,J).
!
! Option identity: WRF's Registry defaults for sf_surface_physics = 4
! (Registry.EM_COMMON:2679-2700): dveg=4 opt_crs=1 opt_btr=1 opt_run=3
! opt_sfc=1 opt_frz=1 opt_inf=1 opt_rad=3 opt_alb=2 opt_snf=1 opt_tbot=2
! opt_stc=1 opt_gla=1 opt_rsf=1 opt_soil=1 opt_pedo=1 opt_crop=0 opt_irr=0
! opt_irrm=0 opt_infdv=1 opt_tdrn=0 soiltstep=0, iz0tlnd=0,
! sf_urban_physics=0, xice_threshold=0.5, MMINLU MODIFIED_IGBP_MODIS_NOAH.
!
! Input:  the tagged binary written by gen_columns.py.
! Output: the same record format; one "step" record then every compared
!         field, for the cold-started state (step 0) and after each step.
!
! Usage:  run_column INPUT.bin OUTPUT.bin

module column_io
  implicit none
  integer, parameter :: nsoil = 4
  integer :: nx = 0, nstep = 0
  real :: dt = 0.0, dx = 0.0
  real, allocatable :: julian_s(:)
  integer, allocatable :: yr_s(:)
  integer, allocatable :: ivgtyp(:,:), isltyp(:,:)
  real, allocatable :: tslb(:,:,:), smois(:,:,:)
  real, allocatable :: xland(:,:), xice(:,:), tsk(:,:), tmn(:,:), snow(:,:), &
      snowh(:,:), lai(:,:), xlat(:,:), xlong(:,:), vegfra(:,:), shdmax(:,:)
  ! per-step forcing, (nx, nstep)
  real, allocatable :: f_coszen(:,:), f_t3d(:,:), f_qv3d(:,:), f_u(:,:), &
      f_v(:,:), f_dz8w(:,:), f_p8w1(:,:), f_p8w2(:,:), f_swdown(:,:), &
      f_glw(:,:), f_rainbl(:,:), f_sr(:,:), f_rainc(:,:), f_rainnc(:,:), &
      f_rainsh(:,:), f_snownc(:,:), f_graupnc(:,:), f_hailnc(:,:)

contains

  subroutine read_input(path)
    character(len=*), intent(in) :: path
    integer :: unit, ios, kind, cnt
    character(len=16) :: name
    real, allocatable :: r(:)
    integer, allocatable :: iv(:)

    open(newunit=unit, file=path, access='stream', form='unformatted', &
         status='old', action='read')
    do
      read(unit, iostat=ios) name
      if (ios /= 0) exit
      read(unit) kind, cnt
      if (kind == 1) then
        allocate(r(cnt))
        read(unit) r
        call store_real(trim(name), r, cnt)
        deallocate(r)
      else
        allocate(iv(cnt))
        read(unit) iv
        call store_int(trim(name), iv, cnt)
        deallocate(iv)
      end if
    end do
    close(unit)
    if (nx < 1 .or. nstep < 1) error stop 'input carries no nx/nstep'
  end subroutine read_input

  subroutine store_int(name, iv, cnt)
    character(len=*), intent(in) :: name
    integer, intent(in) :: cnt, iv(cnt)
    select case (name)
    case ('nx')
      nx = iv(1)
    case ('nstep')
      nstep = iv(1)
    case ('yr')
      allocate(yr_s(cnt)); yr_s = iv
    case ('ivgtyp')
      allocate(ivgtyp(nx, 1)); ivgtyp(:, 1) = iv
    case ('isltyp')
      allocate(isltyp(nx, 1)); isltyp(:, 1) = iv
    case default
      write(*, *) 'unknown integer record ', name
      error stop 3
    end select
  end subroutine store_int

  subroutine put2(a, r, cnt)
    real, allocatable, intent(inout) :: a(:,:)
    integer, intent(in) :: cnt
    real, intent(in) :: r(cnt)
    if (cnt /= nx) error stop 'static record length is not nx'
    allocate(a(nx, 1)); a(:, 1) = r
  end subroutine put2

  subroutine putf(a, r, cnt)
    real, allocatable, intent(inout) :: a(:,:)
    integer, intent(in) :: cnt
    real, intent(in) :: r(cnt)
    if (cnt /= nx * nstep) error stop 'forcing record length is not nx*nstep'
    allocate(a(nx, nstep)); a = reshape(r, [nx, nstep])
  end subroutine putf

  subroutine store_real(name, r, cnt)
    character(len=*), intent(in) :: name
    integer, intent(in) :: cnt
    real, intent(in) :: r(cnt)
    select case (name)
    case ('dt');     dt = r(1)
    case ('dx');     dx = r(1)
    case ('julian'); allocate(julian_s(cnt)); julian_s = r
    case ('tslb')
      allocate(tslb(nx, nsoil, 1)); tslb(:, :, 1) = reshape(r, [nx, nsoil])
    case ('smois')
      allocate(smois(nx, nsoil, 1)); smois(:, :, 1) = reshape(r, [nx, nsoil])
    case ('xland');  call put2(xland, r, cnt)
    case ('xice');   call put2(xice, r, cnt)
    case ('tsk');    call put2(tsk, r, cnt)
    case ('tmn');    call put2(tmn, r, cnt)
    case ('snow');   call put2(snow, r, cnt)
    case ('snowh');  call put2(snowh, r, cnt)
    case ('lai');    call put2(lai, r, cnt)
    case ('xlat');   call put2(xlat, r, cnt)
    case ('xlong');  call put2(xlong, r, cnt)
    case ('vegfra'); call put2(vegfra, r, cnt)
    case ('shdmax'); call put2(shdmax, r, cnt)
    case ('coszen'); call putf(f_coszen, r, cnt)
    case ('t3d');    call putf(f_t3d, r, cnt)
    case ('qv3d');   call putf(f_qv3d, r, cnt)
    case ('u');      call putf(f_u, r, cnt)
    case ('v');      call putf(f_v, r, cnt)
    case ('dz8w');   call putf(f_dz8w, r, cnt)
    case ('p8w1');   call putf(f_p8w1, r, cnt)
    case ('p8w2');   call putf(f_p8w2, r, cnt)
    case ('swdown'); call putf(f_swdown, r, cnt)
    case ('glw');    call putf(f_glw, r, cnt)
    case ('rainbl'); call putf(f_rainbl, r, cnt)
    case ('sr');     call putf(f_sr, r, cnt)
    case ('rainc');  call putf(f_rainc, r, cnt)
    case ('rainnc'); call putf(f_rainnc, r, cnt)
    case ('rainsh'); call putf(f_rainsh, r, cnt)
    case ('snownc'); call putf(f_snownc, r, cnt)
    case ('graupnc'); call putf(f_graupnc, r, cnt)
    case ('hailnc'); call putf(f_hailnc, r, cnt)
    case default
      write(*, *) 'unknown real record ', name
      error stop 3
    end select
  end subroutine store_real

end module column_io


program run_noahmp_column
  use column_io
  use module_sf_noahmpdrv, only: noahmplsm, noahmp_init
  implicit none

  character(len=1024) :: in_path, out_path
  character(len=256) :: llanduse
  integer :: unit, step, ims, ime, jms, jme, kms, kme
  integer :: ids, ide, jds, jde, kds, kde, its, ite, jts, jte, kts, kte
  real :: dzs(nsoil)

  ! ---- the WRF grid arrays noahmplsm and NOAHMP_INIT take -----------------
  real, allocatable, dimension(:,:) :: coszin, swddir, swddif, swdown, glw, &
      precip_in, sr, mp_rainc, mp_rainnc, mp_shcv, mp_snow, mp_graup, mp_hail
  real, allocatable, dimension(:,:,:) :: dz8w, t3d, qv3d, u_phy, v_phy, p8w3d
  real, allocatable, dimension(:,:) :: soilcl1, soilcl2, soilcl3, soilcl4, &
      planting, harvest, season_gdd, td_fraction, irfract, sifract, mifract, &
      fifract
  real, allocatable :: soilcomp(:,:,:), gecros_state(:,:,:), croptype(:,:,:)
  integer, allocatable, dimension(:,:) :: cropcat, irnumsi, irnummi, irnumfi, &
      isnowxy, pgsxy
  real, allocatable, dimension(:,:) :: hfx, qfx, lh, grdflx, smstav, smstot, &
      sfcrunoff, udrunoff, albedo, snowc, canwat, acsnom, acsnow, emiss, qsfc, &
      z0, znt, rs, irwatsi, irwatmi, irwatfi, ireloss, irsivol, irmivol, &
      irfivol, irrsplh, tvxy, tgxy, canicexy, canliqxy, eahxy, tahxy, cmxy, &
      chxy, fwetxy, sneqvoxy, alboldxy, qsnowxy, qrainxy, wslakexy, zwtxy, &
      waxy, wtxy, lfmassxy, rtmassxy, stmassxy, woodxy, stblcpxy, fastcpxy, &
      xsaixy, taussxy, smcwtdxy, deeprechxy, rechxy, grainxy, gddxy, qtdrain, &
      t2mvxy, t2mbxy, q2mvxy, q2mbxy, tradxy, neexy, gppxy, nppxy, fvegxy, &
      runsfxy, runsbxy, ecanxy, edirxy, etranxy, fsaxy, firaxy, aparxy, &
      psnxy, savxy, sagxy, rssunxy, rsshaxy, bgapxy, wgapxy, tgvxy, tgbxy, &
      chvxy, chbxy, shgxy, shcxy, shbxy, evgxy, evbxy, ghvxy, ghbxy, irgxy, &
      ircxy, irbxy, trxy, evcxy, chleafxy, chucxy, chv2xy, chb2xy, qintsxy, &
      qintrxy, qdripsxy, qdriprxy, qthrosxy, qthrorxy, qsnsubxy, qsnfroxy, &
      qsubcxy, qfrocxy, qevacxy, qdewcxy, qfrzcxy, qmeltcxy, qsnbotxy, &
      qmeltxy, pondingxy, pahxy, pahgxy, pahvxy, pahbxy, fpicexy, rainlsm, &
      snowlsm, forctlsm, forcqlsm, forcplsm, forczlsm, forcwlsm, acc_ssoilxy, &
      acc_qinsurxy, acc_qsevaxy, eflxbxy, soilenergy, snowenergy, canhsxy, &
      acc_dwaterxy, acc_prcpxy, acc_ecanxy, acc_etranxy, acc_edirxy, chstarxy
  real, allocatable :: sh2o(:,:,:), smoiseq(:,:,:), acc_etranixy(:,:,:)
  real, allocatable :: tsnoxy(:,:,:), snicexy(:,:,:), snliqxy(:,:,:), &
      zsnsoxy(:,:,:)

  call get_command_argument(1, in_path)
  call get_command_argument(2, out_path)
  if (len_trim(in_path) == 0 .or. len_trim(out_path) == 0) then
    write(*, '(A)') 'usage: run_column INPUT.bin OUTPUT.bin'
    error stop 2
  end if
  call read_input(trim(in_path))

  ims = 1; ime = nx; its = 1; ite = nx; ids = 1; ide = nx + 1
  jms = 1; jme = 1;  jts = 1; jte = 1;  jds = 1; jde = 2
  kms = 1; kme = 2;  kts = 1; kte = 2;  kds = 1; kde = 3
  dzs = [0.1, 0.3, 0.6, 1.0]
  llanduse = 'MODIFIED_IGBP_MODIS_NOAH'

  allocate(coszin(nx,1), swddir(nx,1), swddif(nx,1), swdown(nx,1), glw(nx,1), &
      precip_in(nx,1), sr(nx,1), mp_rainc(nx,1), mp_rainnc(nx,1), &
      mp_shcv(nx,1), mp_snow(nx,1), mp_graup(nx,1), mp_hail(nx,1))
  allocate(dz8w(nx,2,1), t3d(nx,2,1), qv3d(nx,2,1), u_phy(nx,2,1), &
      v_phy(nx,2,1), p8w3d(nx,2,1))
  allocate(soilcl1(nx,1), soilcl2(nx,1), soilcl3(nx,1), soilcl4(nx,1), &
      planting(nx,1), harvest(nx,1), season_gdd(nx,1), td_fraction(nx,1), &
      irfract(nx,1), sifract(nx,1), mifract(nx,1), fifract(nx,1))
  allocate(soilcomp(nx,2*nsoil,1), gecros_state(nx,60,1), croptype(nx,5,1))
  allocate(cropcat(nx,1), irnumsi(nx,1), irnummi(nx,1), irnumfi(nx,1), &
      isnowxy(nx,1), pgsxy(nx,1))
  allocate(hfx(nx,1), qfx(nx,1), lh(nx,1), grdflx(nx,1), smstav(nx,1), &
      smstot(nx,1), sfcrunoff(nx,1), udrunoff(nx,1), albedo(nx,1), &
      snowc(nx,1), canwat(nx,1), acsnom(nx,1), acsnow(nx,1), emiss(nx,1), &
      qsfc(nx,1), z0(nx,1), znt(nx,1), rs(nx,1), irwatsi(nx,1), &
      irwatmi(nx,1), irwatfi(nx,1), ireloss(nx,1), irsivol(nx,1), &
      irmivol(nx,1), irfivol(nx,1), irrsplh(nx,1), tvxy(nx,1), tgxy(nx,1), &
      canicexy(nx,1), canliqxy(nx,1), eahxy(nx,1), tahxy(nx,1), cmxy(nx,1), &
      chxy(nx,1), fwetxy(nx,1), sneqvoxy(nx,1), alboldxy(nx,1), &
      qsnowxy(nx,1), qrainxy(nx,1), wslakexy(nx,1), zwtxy(nx,1), waxy(nx,1), &
      wtxy(nx,1), lfmassxy(nx,1), rtmassxy(nx,1), stmassxy(nx,1), &
      woodxy(nx,1), stblcpxy(nx,1), fastcpxy(nx,1), xsaixy(nx,1), &
      taussxy(nx,1), smcwtdxy(nx,1), deeprechxy(nx,1), rechxy(nx,1), &
      grainxy(nx,1), gddxy(nx,1), qtdrain(nx,1), t2mvxy(nx,1), t2mbxy(nx,1), &
      q2mvxy(nx,1), q2mbxy(nx,1), tradxy(nx,1), neexy(nx,1), gppxy(nx,1), &
      nppxy(nx,1), fvegxy(nx,1), runsfxy(nx,1), runsbxy(nx,1), ecanxy(nx,1), &
      edirxy(nx,1), etranxy(nx,1), fsaxy(nx,1), firaxy(nx,1), aparxy(nx,1), &
      psnxy(nx,1), savxy(nx,1), sagxy(nx,1), rssunxy(nx,1), rsshaxy(nx,1), &
      bgapxy(nx,1), wgapxy(nx,1), tgvxy(nx,1), tgbxy(nx,1), chvxy(nx,1), &
      chbxy(nx,1), shgxy(nx,1), shcxy(nx,1), shbxy(nx,1), evgxy(nx,1), &
      evbxy(nx,1), ghvxy(nx,1), ghbxy(nx,1), irgxy(nx,1), ircxy(nx,1), &
      irbxy(nx,1), trxy(nx,1), evcxy(nx,1), chleafxy(nx,1), chucxy(nx,1), &
      chv2xy(nx,1), chb2xy(nx,1), qintsxy(nx,1), qintrxy(nx,1), &
      qdripsxy(nx,1), qdriprxy(nx,1), qthrosxy(nx,1), qthrorxy(nx,1), &
      qsnsubxy(nx,1), qsnfroxy(nx,1), qsubcxy(nx,1), qfrocxy(nx,1), &
      qevacxy(nx,1), qdewcxy(nx,1), qfrzcxy(nx,1), qmeltcxy(nx,1), &
      qsnbotxy(nx,1), qmeltxy(nx,1), pondingxy(nx,1), pahxy(nx,1), &
      pahgxy(nx,1), pahvxy(nx,1), pahbxy(nx,1), fpicexy(nx,1), rainlsm(nx,1), &
      snowlsm(nx,1), forctlsm(nx,1), forcqlsm(nx,1), forcplsm(nx,1), &
      forczlsm(nx,1), forcwlsm(nx,1), acc_ssoilxy(nx,1), acc_qinsurxy(nx,1), &
      acc_qsevaxy(nx,1), eflxbxy(nx,1), soilenergy(nx,1), snowenergy(nx,1), &
      canhsxy(nx,1), acc_dwaterxy(nx,1), acc_prcpxy(nx,1), acc_ecanxy(nx,1), &
      acc_etranxy(nx,1), acc_edirxy(nx,1), chstarxy(nx,1))
  allocate(sh2o(nx,nsoil,1), smoiseq(nx,nsoil,1), acc_etranixy(nx,nsoil,1))
  allocate(tsnoxy(nx,-2:0,1), snicexy(nx,-2:0,1), snliqxy(nx,-2:0,1), &
      zsnsoxy(nx,-2:nsoil,1))

  ! Registry cold state: every carrier starts at zero, as WRF allocates it.
  coszin = 0.; swddir = 0.; swddif = 0.; swdown = 0.; glw = 0.
  precip_in = 0.; sr = 0.; mp_rainc = 0.; mp_rainnc = 0.; mp_shcv = 0.
  mp_snow = 0.; mp_graup = 0.; mp_hail = 0.
  dz8w = 0.; t3d = 0.; qv3d = 0.; u_phy = 0.; v_phy = 0.; p8w3d = 0.
  soilcl1 = 0.; soilcl2 = 0.; soilcl3 = 0.; soilcl4 = 0.
  planting = 0.; harvest = 0.; season_gdd = 0.; td_fraction = 0.
  irfract = 0.; sifract = 0.; mifract = 0.; fifract = 0.
  soilcomp = 0.; gecros_state = 0.; croptype = 0.
  cropcat = 0; irnumsi = 0; irnummi = 0; irnumfi = 0; isnowxy = 0; pgsxy = 0
  hfx = 0.; qfx = 0.; lh = 0.; grdflx = 0.; smstav = 0.; smstot = 0.
  sfcrunoff = 0.; udrunoff = 0.; albedo = 0.; snowc = 0.; canwat = 0.
  acsnom = 0.; acsnow = 0.; emiss = 0.; qsfc = 0.; z0 = 0.; znt = 0.; rs = 0.
  irwatsi = 0.; irwatmi = 0.; irwatfi = 0.; ireloss = 0.; irsivol = 0.
  irmivol = 0.; irfivol = 0.; irrsplh = 0.
  tvxy = 0.; tgxy = 0.; canicexy = 0.; canliqxy = 0.; eahxy = 0.; tahxy = 0.
  cmxy = 0.; chxy = 0.; fwetxy = 0.; sneqvoxy = 0.; alboldxy = 0.
  qsnowxy = 0.; qrainxy = 0.; wslakexy = 0.; zwtxy = 0.; waxy = 0.; wtxy = 0.
  lfmassxy = 0.; rtmassxy = 0.; stmassxy = 0.; woodxy = 0.; stblcpxy = 0.
  fastcpxy = 0.; xsaixy = 0.; taussxy = 0.; smcwtdxy = 0.; deeprechxy = 0.
  rechxy = 0.; grainxy = 0.; gddxy = 0.; qtdrain = 0.
  t2mvxy = 0.; t2mbxy = 0.; q2mvxy = 0.; q2mbxy = 0.; tradxy = 0.; neexy = 0.
  gppxy = 0.; nppxy = 0.; fvegxy = 0.; runsfxy = 0.; runsbxy = 0.
  ecanxy = 0.; edirxy = 0.; etranxy = 0.; fsaxy = 0.; firaxy = 0.
  aparxy = 0.; psnxy = 0.; savxy = 0.; sagxy = 0.; rssunxy = 0.; rsshaxy = 0.
  bgapxy = 0.; wgapxy = 0.; tgvxy = 0.; tgbxy = 0.; chvxy = 0.; chbxy = 0.
  shgxy = 0.; shcxy = 0.; shbxy = 0.; evgxy = 0.; evbxy = 0.; ghvxy = 0.
  ghbxy = 0.; irgxy = 0.; ircxy = 0.; irbxy = 0.; trxy = 0.; evcxy = 0.
  chleafxy = 0.; chucxy = 0.; chv2xy = 0.; chb2xy = 0.; qintsxy = 0.
  qintrxy = 0.; qdripsxy = 0.; qdriprxy = 0.; qthrosxy = 0.; qthrorxy = 0.
  qsnsubxy = 0.; qsnfroxy = 0.; qsubcxy = 0.; qfrocxy = 0.; qevacxy = 0.
  qdewcxy = 0.; qfrzcxy = 0.; qmeltcxy = 0.; qsnbotxy = 0.; qmeltxy = 0.
  pondingxy = 0.; pahxy = 0.; pahgxy = 0.; pahvxy = 0.; pahbxy = 0.
  fpicexy = 0.; rainlsm = 0.; snowlsm = 0.; forctlsm = 0.; forcqlsm = 0.
  forcplsm = 0.; forczlsm = 0.; forcwlsm = 0.; acc_ssoilxy = 0.
  acc_qinsurxy = 0.; acc_qsevaxy = 0.; eflxbxy = 0.; soilenergy = 0.
  snowenergy = 0.; canhsxy = 0.; acc_dwaterxy = 0.; acc_prcpxy = 0.
  acc_ecanxy = 0.; acc_etranxy = 0.; acc_edirxy = 0.; chstarxy = 0.
  sh2o = 0.; smoiseq = 0.; acc_etranixy = 0.
  tsnoxy = 0.; snicexy = 0.; snliqxy = 0.; zsnsoxy = 0.

  ! ---- NOAHMP_INIT: module_physics_init's cold start ----------------------
  call noahmp_init(llanduse, snow, snowh, canwat, isltyp, ivgtyp, xlat, &
       tslb, smois, sh2o, dzs, .true., .true., &
       tsk, isnowxy, tvxy, tgxy, canicexy, tmn, xice, &
       canliqxy, eahxy, tahxy, cmxy, chxy, &
       fwetxy, sneqvoxy, alboldxy, qsnowxy, qrainxy, wslakexy, zwtxy, waxy, &
       wtxy, tsnoxy, zsnsoxy, snicexy, snliqxy, lfmassxy, rtmassxy, &
       stmassxy, woodxy, stblcpxy, fastcpxy, xsaixy, lai, &
       grainxy, gddxy, &
       croptype, cropcat, &
       irnumsi, irnummi, irnumfi, irwatsi, &
       irwatmi, irwatfi, ireloss, irsivol, &
       irmivol, irfivol, irrsplh, &
       t2mvxy, t2mbxy, chstarxy, &
       nsoil, .false., &
       .true., 3, 0, 0, 0, &
       0, &
       ids, ide, jds, jde, kds, kde, &
       ims, ime, jms, jme, kms, kme, &
       its, ite, jts, jte, kts, kte, &
       smoiseq=smoiseq, smcwtdxy=smcwtdxy, rechxy=rechxy, &
       deeprechxy=deeprechxy, qtdrain=qtdrain)

  open(newunit=unit, file=trim(out_path), access='stream', &
       form='unformatted', status='replace', action='write')
  call dump(0)

  do step = 1, nstep
    coszin(:, 1)  = f_coszen(:, step)
    swdown(:, 1)  = f_swdown(:, step)
    glw(:, 1)     = f_glw(:, step)
    precip_in(:, 1) = f_rainbl(:, step)
    sr(:, 1)      = f_sr(:, step)
    mp_rainc(:, 1)  = f_rainc(:, step)
    mp_rainnc(:, 1) = f_rainnc(:, step)
    mp_shcv(:, 1)   = f_rainsh(:, step)
    mp_snow(:, 1)   = f_snownc(:, step)
    mp_graup(:, 1)  = f_graupnc(:, step)
    mp_hail(:, 1)   = f_hailnc(:, step)
    t3d(:, 1, 1)  = f_t3d(:, step);  t3d(:, 2, 1) = f_t3d(:, step)
    qv3d(:, 1, 1) = f_qv3d(:, step); qv3d(:, 2, 1) = f_qv3d(:, step)
    u_phy(:, 1, 1) = f_u(:, step);   u_phy(:, 2, 1) = f_u(:, step)
    v_phy(:, 1, 1) = f_v(:, step);   v_phy(:, 2, 1) = f_v(:, step)
    dz8w(:, 1, 1) = f_dz8w(:, step); dz8w(:, 2, 1) = f_dz8w(:, step)
    p8w3d(:, 1, 1) = f_p8w1(:, step)
    p8w3d(:, 2, 1) = f_p8w2(:, step)

    call noahmplsm(ITIMESTEP=step, YR=yr_s(step), JULIAN=julian_s(step), &
        COSZIN=coszin, XLAT=xlat, XLONG=xlong, &
        DZ8W=dz8w, DT=dt, DZS=dzs, NSOIL=nsoil, DX=dx, &
        IVGTYP=ivgtyp, ISLTYP=isltyp, VEGFRA=vegfra, VEGMAX=shdmax, TMN=tmn, &
        XLAND=xland, XICE=xice, XICE_THRES=0.5, CROPCAT=cropcat, &
        PLANTING=planting, HARVEST=harvest, SEASON_GDD=season_gdd, &
        IDVEG=4, IOPT_CRS=1, IOPT_BTR=1, IOPT_RUN=3, IOPT_SFC=1, IOPT_FRZ=1, &
        IOPT_INF=1, IOPT_RAD=3, IOPT_ALB=2, IOPT_SNF=1, IOPT_TBOT=2, &
        IOPT_STC=1, IOPT_GLA=1, IOPT_RSF=1, IOPT_SOIL=1, IOPT_PEDO=1, &
        IOPT_CROP=0, IOPT_IRR=0, IOPT_IRRM=0, IOPT_INFDV=1, IOPT_TDRN=0, &
        SOILTSTEP=0.0, IZ0TLND=0, SF_URBAN_PHYSICS=0, &
        SOILCOMP=soilcomp, SOILCL1=soilcl1, SOILCL2=soilcl2, &
        SOILCL3=soilcl3, SOILCL4=soilcl4, &
        T3D=t3d, QV3D=qv3d, U_PHY=u_phy, V_PHY=v_phy, SWDOWN=swdown, &
        SWDDIR=swddir, SWDDIF=swddif, GLW=glw, &
        P8W3D=p8w3d, PRECIP_IN=precip_in, SR=sr, &
        IRFRACT=irfract, SIFRACT=sifract, MIFRACT=mifract, FIFRACT=fifract, &
        TSK=tsk, HFX=hfx, QFX=qfx, LH=lh, GRDFLX=grdflx, SMSTAV=smstav, &
        SMSTOT=smstot, SFCRUNOFF=sfcrunoff, UDRUNOFF=udrunoff, &
        ALBEDO=albedo, SNOWC=snowc, SMOIS=smois, &
        SH2O=sh2o, TSLB=tslb, SNOW=snow, SNOWH=snowh, CANWAT=canwat, &
        ACSNOM=acsnom, ACSNOW=acsnow, EMISS=emiss, QSFC=qsfc, &
        Z0=z0, ZNT=znt, &
        IRNUMSI=irnumsi, IRNUMMI=irnummi, IRNUMFI=irnumfi, IRWATSI=irwatsi, &
        IRWATMI=irwatmi, IRWATFI=irwatfi, IRELOSS=ireloss, IRSIVOL=irsivol, &
        IRMIVOL=irmivol, IRFIVOL=irfivol, IRRSPLH=irrsplh, LLANDUSE=llanduse, &
        ISNOWXY=isnowxy, TVXY=tvxy, TGXY=tgxy, CANICEXY=canicexy, &
        CANLIQXY=canliqxy, EAHXY=eahxy, TAHXY=tahxy, CMXY=cmxy, CHXY=chxy, &
        FWETXY=fwetxy, SNEQVOXY=sneqvoxy, ALBOLDXY=alboldxy, &
        QSNOWXY=qsnowxy, QRAINXY=qrainxy, WSLAKEXY=wslakexy, ZWTXY=zwtxy, &
        WAXY=waxy, WTXY=wtxy, TSNOXY=tsnoxy, ZSNSOXY=zsnsoxy, &
        SNICEXY=snicexy, SNLIQXY=snliqxy, LFMASSXY=lfmassxy, &
        RTMASSXY=rtmassxy, STMASSXY=stmassxy, WOODXY=woodxy, &
        STBLCPXY=stblcpxy, FASTCPXY=fastcpxy, XLAIXY=lai, XSAIXY=xsaixy, &
        TAUSSXY=taussxy, SMOISEQ=smoiseq, SMCWTDXY=smcwtdxy, &
        DEEPRECHXY=deeprechxy, RECHXY=rechxy, GRAINXY=grainxy, GDDXY=gddxy, &
        PGSXY=pgsxy, GECROS_STATE=gecros_state, QTDRAIN=qtdrain, &
        TD_FRACTION=td_fraction, &
        T2MVXY=t2mvxy, T2MBXY=t2mbxy, Q2MVXY=q2mvxy, Q2MBXY=q2mbxy, &
        TRADXY=tradxy, NEEXY=neexy, GPPXY=gppxy, NPPXY=nppxy, &
        FVEGXY=fvegxy, RUNSFXY=runsfxy, RUNSBXY=runsbxy, ECANXY=ecanxy, &
        EDIRXY=edirxy, ETRANXY=etranxy, FSAXY=fsaxy, FIRAXY=firaxy, &
        APARXY=aparxy, PSNXY=psnxy, SAVXY=savxy, SAGXY=sagxy, &
        RSSUNXY=rssunxy, RSSHAXY=rsshaxy, BGAPXY=bgapxy, WGAPXY=wgapxy, &
        TGVXY=tgvxy, TGBXY=tgbxy, CHVXY=chvxy, CHBXY=chbxy, SHGXY=shgxy, &
        SHCXY=shcxy, SHBXY=shbxy, EVGXY=evgxy, EVBXY=evbxy, GHVXY=ghvxy, &
        GHBXY=ghbxy, IRGXY=irgxy, IRCXY=ircxy, IRBXY=irbxy, TRXY=trxy, &
        EVCXY=evcxy, CHLEAFXY=chleafxy, CHUCXY=chucxy, CHV2XY=chv2xy, &
        CHB2XY=chb2xy, RS=rs, &
        QINTSXY=qintsxy, QINTRXY=qintrxy, QDRIPSXY=qdripsxy, &
        QDRIPRXY=qdriprxy, QTHROSXY=qthrosxy, QTHRORXY=qthrorxy, &
        QSNSUBXY=qsnsubxy, QSNFROXY=qsnfroxy, QSUBCXY=qsubcxy, &
        QFROCXY=qfrocxy, QEVACXY=qevacxy, QDEWCXY=qdewcxy, QFRZCXY=qfrzcxy, &
        QMELTCXY=qmeltcxy, QSNBOTXY=qsnbotxy, QMELTXY=qmeltxy, &
        PONDINGXY=pondingxy, PAHXY=pahxy, PAHGXY=pahgxy, PAHVXY=pahvxy, &
        PAHBXY=pahbxy, FPICEXY=fpicexy, RAINLSM=rainlsm, SNOWLSM=snowlsm, &
        FORCTLSM=forctlsm, FORCQLSM=forcqlsm, FORCPLSM=forcplsm, &
        FORCZLSM=forczlsm, FORCWLSM=forcwlsm, &
        ACC_SSOILXY=acc_ssoilxy, ACC_QINSURXY=acc_qinsurxy, &
        ACC_QSEVAXY=acc_qsevaxy, ACC_ETRANIXY=acc_etranixy, &
        EFLXBXY=eflxbxy, SOILENERGY=soilenergy, SNOWENERGY=snowenergy, &
        CANHSXY=canhsxy, ACC_DWATERXY=acc_dwaterxy, ACC_PRCPXY=acc_prcpxy, &
        ACC_ECANXY=acc_ecanxy, ACC_ETRANXY=acc_etranxy, &
        ACC_EDIRXY=acc_edirxy, &
        ids=ids, ide=ide, jds=jds, jde=jde, kds=kds, kde=kde, &
        ims=ims, ime=ime, jms=jms, jme=jme, kms=kms, kme=kme, &
        its=its, ite=ite, jts=jts, jte=jte, kts=kts, kte=kte, &
        MP_RAINC=mp_rainc, MP_RAINNC=mp_rainnc, MP_SHCV=mp_shcv, &
        MP_SNOW=mp_snow, MP_GRAUP=mp_graup, MP_HAIL=mp_hail)

    call dump(step)
  end do
  close(unit)

contains

  subroutine rec2(name, a)
    character(len=*), intent(in) :: name
    real, intent(in) :: a(:,:)
    character(len=16) :: tag
    tag = name
    write(unit) tag, 1, size(a)
    write(unit) a
  end subroutine rec2

  subroutine reci(name, a)
    character(len=*), intent(in) :: name
    integer, intent(in) :: a(:,:)
    character(len=16) :: tag
    tag = name
    write(unit) tag, 2, size(a)
    write(unit) a
  end subroutine reci

  subroutine rec3(name, a)
    character(len=*), intent(in) :: name
    real, intent(in) :: a(:,:,:)
    character(len=16) :: tag
    tag = name
    write(unit) tag, 1, size(a)
    write(unit) a
  end subroutine rec3

  ! The comparison set: every carrier WOOF's noahmp_lsm_step writes, in
  ! WOOF's names.  The order is the order compare_column.py reports in.
  subroutine dump(istep)
    integer, intent(in) :: istep
    character(len=16) :: tag
    tag = 'step'
    write(unit) tag, 2, 1
    write(unit) istep
    call rec2('tsk', tsk);           call rec2('hfx', hfx)
    call rec2('qfx', qfx);           call rec2('lh', lh)
    call rec2('grdflx', grdflx);     call rec2('smstav', smstav)
    call rec2('smstot', smstot);     call rec2('sfcrunoff', sfcrunoff)
    call rec2('udrunoff', udrunoff); call rec2('albedo', albedo)
    call rec2('snowc', snowc);       call rec2('snow', snow)
    call rec2('snowh', snowh);       call rec2('canwat', canwat)
    call rec2('acsnom', acsnom);     call rec2('acsnow', acsnow)
    call rec2('emiss', emiss);       call rec2('qsfc', qsfc)
    call rec2('z0', z0);             call rec2('znt', znt)
    call rec2('lai', lai)
    call rec2('tvxy', tvxy);         call rec2('tgxy', tgxy)
    call rec2('canicexy', canicexy); call rec2('canliqxy', canliqxy)
    call rec2('eahxy', eahxy);       call rec2('tahxy', tahxy)
    call rec2('cmxy', cmxy);         call rec2('chxy', chxy)
    call rec2('fwetxy', fwetxy);     call rec2('sneqvoxy', sneqvoxy)
    call rec2('alboldxy', alboldxy); call rec2('qsnowxy', qsnowxy)
    call rec2('qrainxy', qrainxy);   call rec2('wslakexy', wslakexy)
    call rec2('waxy', waxy);         call rec2('xsaixy', xsaixy)
    call rec2('taussxy', taussxy)
    call rec2('t2mvxy', t2mvxy);     call rec2('t2mbxy', t2mbxy)
    call rec2('q2mvxy', q2mvxy);     call rec2('q2mbxy', q2mbxy)
    call rec2('tradxy', tradxy);     call rec2('fsaxy', fsaxy)
    call rec2('firaxy', firaxy);     call rec2('ecanxy', ecanxy)
    call rec2('edirxy', edirxy);     call rec2('etranxy', etranxy)
    call rec2('runsfxy', runsfxy);   call rec2('runsbxy', runsbxy)
    call rec2('fvegxy', fvegxy);     call rec2('rssunxy', rssunxy)
    call rec2('rsshaxy', rsshaxy);   call rec2('chvxy', chvxy)
    call rec2('chbxy', chbxy);       call rec2('fpicexy', fpicexy)
    call rec2('qsnbotxy', qsnbotxy); call rec2('qmeltxy', qmeltxy)
    call rec2('pondingxy', pondingxy); call rec2('canhsxy', canhsxy)
    call rec2('eflxbxy', eflxbxy);   call rec2('soilenergy', soilenergy)
    call rec2('snowenergy', snowenergy); call rec2('rs', rs)
    call reci('isnowxy', isnowxy);   call reci('pgsxy', pgsxy)
    call rec3('tslb', tslb);         call rec3('smois', smois)
    call rec3('sh2o', sh2o)
    call rec3('tsnoxy', tsnoxy);     call rec3('snicexy', snicexy)
    call rec3('snliqxy', snliqxy);   call rec3('zsnsoxy', zsnsoxy)
  end subroutine dump

end program run_noahmp_column
