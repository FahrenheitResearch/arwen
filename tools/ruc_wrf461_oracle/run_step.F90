program run_ruc_step_oracle
  use module_sf_ruclsm, only: lsmruc, ruclsminit
  implicit none

  integer, parameter :: ncol = 4, nsl = 9, nz = 2
  integer, parameter :: nlcat = 21, nscat = 19
  integer, parameter :: ids = 1, ide = ncol + 1
  integer, parameter :: jds = 1, jde = 2
  integer, parameter :: kds = 1, kde = nz + 1
  integer, parameter :: ims = 1, ime = ncol
  integer, parameter :: jms = 1, jme = 1
  integer, parameter :: kms = 1, kme = nz
  integer, parameter :: its = 1, ite = ncol
  integer, parameter :: jts = 1, jte = 1
  integer, parameter :: kts = 1, kte = nz
  character(len=16), parameter :: names(ncol) = [character(len=16) :: &
      'warm_rain', 'cold_snow', 'water', 'sea_ice']
  character(len=1024) :: output_path
  character(len=32) :: mminlu
  integer :: i, k, unit
  integer :: ivgtyp(ims:ime,jms:jme), isltyp(ims:ime,jms:jme)
  real :: landusef(ims:ime,1:nlcat,jms:jme)
  real :: soilctop(ims:ime,1:nscat,jms:jme)
  real :: z3d(ims:ime,kms:kme,jms:jme), p8w(ims:ime,kms:kme,jms:jme)
  real :: t3d(ims:ime,kms:kme,jms:jme), qv3d(ims:ime,kms:kme,jms:jme)
  real :: qc3d(ims:ime,kms:kme,jms:jme), rho3d(ims:ime,kms:kme,jms:jme)
  real :: soilmois(ims:ime,1:nsl,jms:jme), sh2o(ims:ime,1:nsl,jms:jme)
  real :: tso(ims:ime,1:nsl,jms:jme), smfr3d(ims:ime,1:nsl,jms:jme)
  real :: keepfr3dflag(ims:ime,1:nsl,jms:jme)
  real :: soilmois_before(ims:ime,1:nsl,jms:jme)
  real :: sh2o_before(ims:ime,1:nsl,jms:jme)
  real :: tso_before(ims:ime,1:nsl,jms:jme)
  real :: smfr_before(ims:ime,1:nsl,jms:jme)
  real :: zs(1:nsl)

  real :: rainbl(ims:ime,jms:jme), snow(ims:ime,jms:jme)
  real :: snowh(ims:ime,jms:jme), snowc(ims:ime,jms:jme)
  real :: frzfrac(ims:ime,jms:jme), rhosnf(ims:ime,jms:jme)
  real :: precipfr(ims:ime,jms:jme), glw(ims:ime,jms:jme)
  real :: gsw(ims:ime,jms:jme), emiss(ims:ime,jms:jme)
  real :: chklowq(ims:ime,jms:jme), chs(ims:ime,jms:jme)
  real :: flqc(ims:ime,jms:jme), flhc(ims:ime,jms:jme)
  real :: mavail(ims:ime,jms:jme), canwat(ims:ime,jms:jme)
  real :: vegfra(ims:ime,jms:jme), alb(ims:ime,jms:jme)
  real :: znt(ims:ime,jms:jme), z0(ims:ime,jms:jme)
  real :: snoalb(ims:ime,jms:jme), albbck(ims:ime,jms:jme)
  real :: lai(ims:ime,jms:jme), qsfc(ims:ime,jms:jme)
  real :: qsg(ims:ime,jms:jme), qvg(ims:ime,jms:jme)
  real :: qcg(ims:ime,jms:jme), dew(ims:ime,jms:jme)
  real :: soilt1(ims:ime,jms:jme), tsnav(ims:ime,jms:jme)
  real :: tbot(ims:ime,jms:jme), xland(ims:ime,jms:jme)
  real :: xice(ims:ime,jms:jme), smavail(ims:ime,jms:jme)
  real :: smmax(ims:ime,jms:jme), soilt(ims:ime,jms:jme)
  real :: hfx(ims:ime,jms:jme), qfx(ims:ime,jms:jme)
  real :: lh(ims:ime,jms:jme), sfcrunoff(ims:ime,jms:jme)
  real :: udrunoff(ims:ime,jms:jme), acrunoff(ims:ime,jms:jme)
  real :: sfcexc(ims:ime,jms:jme), sfcevp(ims:ime,jms:jme)
  real :: grdflx(ims:ime,jms:jme), snowfallac(ims:ime,jms:jme)
  real :: acsnow(ims:ime,jms:jme), snom(ims:ime,jms:jme)
  real :: shdmin(ims:ime,jms:jme), shdmax(ims:ime,jms:jme)

  call get_command_argument(1, output_path)
  if (len_trim(output_path) == 0) then
    write(*, '(A)') 'usage: run_step OUTPUT.csv'
    error stop 2
  end if

  mminlu = 'MODIFIED_IGBP_MODIS_NOAH'
  zs = [0.00, 0.01, 0.04, 0.10, 0.30, 0.60, 1.00, 1.60, 3.00]
  ivgtyp(:,1) = [1, 10, 17, 15]
  isltyp(:,1) = [1, 4, 14, 16]
  xland(:,1) = [1.0, 1.0, 2.0, 1.0]
  xice(:,1) = [0.0, 0.0, 0.0, 0.8]

  landusef = 0.0
  soilctop = 0.0
  do i = 1, ncol
    landusef(i,ivgtyp(i,1),1) = 1.0
    soilctop(i,isltyp(i,1),1) = 1.0
  end do

  tso = 285.0
  soilmois = 0.25
  do k = 1, nsl
    tso(1,k,1) = 295.0 - 1.25 * real(k - 1)
    soilmois(1,k,1) = 0.24 + 0.005 * real(k - 1)
    tso(2,k,1) = 270.0 + 0.50 * real(k - 1)
    soilmois(2,k,1) = 0.30 + 0.004 * real(k - 1)
    tso(3,k,1) = 285.0
    soilmois(3,k,1) = 1.0
    tso(4,k,1) = min(271.0, 266.0 + 0.75 * real(k - 1))
    soilmois(4,k,1) = 1.0
  end do
  tbot(:,1) = tso(:,nsl,1)

  soilt(:,1) = [296.0, 269.0, 285.0, 267.0]
  snow(:,1) = [0.0, 20.0, 0.0, 50.0]
  snowh(:,1) = [0.0, 0.08, 0.0, 0.20]
  snowc(:,1) = [0.0, 0.5, 0.0, 1.0]
  rainbl(:,1) = [0.5, 0.3, 0.0, 0.1]
  frzfrac(:,1) = [0.0, 1.0, 0.0, 1.0]
  glw(:,1) = [340.0, 240.0, 350.0, 220.0]
  gsw(:,1) = [520.0, 120.0, 400.0, 80.0]
  emiss(:,1) = [0.95, 0.95, 0.98, 0.98]
  alb(:,1) = [0.12, 0.35, 0.08, 0.55]
  albbck = alb
  snoalb(:,1) = [0.52, 0.70, 0.70, 0.82]
  canwat(:,1) = [0.10, 0.05, 0.0, 0.0]
  vegfra(:,1) = [70.0, 45.0, 0.0, 0.0]
  shdmin(:,1) = [20.0, 20.0, 0.0, 0.0]
  shdmax(:,1) = [90.0, 85.0, 0.0, 0.0]
  lai = 0.0
  z0 = 0.0
  znt = 0.0
  mavail = 0.5

  z3d = 80.0
  p8w = 95000.0
  rho3d = 1.10
  qc3d = 0.0
  do k = kms, kme
    t3d(:,k,1) = [298.0, 268.0, 286.0, 266.0]
    qv3d(:,k,1) = [0.012, 0.003, 0.008, 0.002]
  end do

  chs = 0.010
  flqc = 0.010
  flhc = 10.0
  hfx = 0.0
  qfx = 0.0
  lh = 0.0
  qsfc = 0.0
  qsg = 0.0
  qvg = 0.0
  qcg = -1.0
  dew = 0.0
  soilt1 = -999.0
  tsnav = 0.0
  chklowq = 1.0
  sfcexc = 0.0
  sfcevp = 0.0
  sfcrunoff = 0.0
  udrunoff = 0.0
  acrunoff = 0.0
  grdflx = 0.0
  snowfallac = 0.0
  acsnow = 0.0
  snom = 0.0
  smavail = 0.0
  smmax = 0.0
  rhosnf = 0.0
  precipfr = 0.0
  keepfr3dflag = 0.0
  sh2o = soilmois
  smfr3d = 0.0

  call ruclsminit(sh2o, smfr3d, tso, soilmois, isltyp, ivgtyp, &
      mminlu, xice, mavail, nsl, 17, 15, znt, .false., .true., &
      ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, 1, nsl, &
      its, ite, jts, jte, 1, nsl)

  tso_before = tso
  soilmois_before = soilmois
  sh2o_before = sh2o
  smfr_before = smfr3d

  call lsmruc(0, 60.0, 1, nsl, zs, rainbl, snow, snowh, snowc, &
      frzfrac, .true., rhosnf, precipfr, z3d, p8w, t3d, qv3d, &
      qc3d, rho3d, glw, gsw, emiss, chklowq, chs, flqc, flhc, &
      mavail, canwat, vegfra, alb, znt, z0, snoalb, albbck, lai, &
      mminlu, landusef, nlcat, 0, 0, soilctop, nscat, qsfc, qsg, &
      qvg, qcg, dew, soilt1, tsnav, tbot, ivgtyp, isltyp, xland, &
      17, 15, xice, 0.5, 1004.5, 287.0/1004.5, 9.81, 2.5e6, &
      5.67051e-8, soilmois, sh2o, smavail, smmax, tso, soilt, hfx, &
      qfx, lh, sfcrunoff, udrunoff, acrunoff, sfcexc, sfcevp, &
      grdflx, snowfallac, acsnow, snom, smfr3d, keepfr3dflag, &
      .true., shdmin, shdmax, .false., ids, ide, jds, jde, kds, kde, &
      ims, ime, jms, jme, kms, kme, its, ite, jts, jte, kts, kte)

  open(newunit=unit, file=trim(output_path), status='replace', action='write')
  write(unit, '(A)') 'case,k,tso_before,soilmois_before,sh2o_before,' // &
      'smfr_before,tso_after,soilmois_after,sh2o_after,smfr_after,' // &
      'tsk,hfx,qfx,lh,grdflx,snow,snowh,snowc,mavail,smavail,smmax,' // &
      'sfcrunoff,udrunoff,acrunoff,sfcevp,qsfc,qvg,emiss,alb,znt,' // &
      'rhosnf,precipfr,snowfallac'
  do i = 1, ncol
    do k = 1, nsl
      write(unit, '(*(g0,:,","))') trim(names(i)), k, tso_before(i,k,1), &
          soilmois_before(i,k,1), sh2o_before(i,k,1), smfr_before(i,k,1), &
          tso(i,k,1), soilmois(i,k,1), sh2o(i,k,1), smfr3d(i,k,1), &
          soilt(i,1), hfx(i,1), qfx(i,1), lh(i,1), grdflx(i,1), &
          snow(i,1), snowh(i,1), snowc(i,1), mavail(i,1), smavail(i,1), &
          smmax(i,1), sfcrunoff(i,1), udrunoff(i,1), acrunoff(i,1), &
          sfcevp(i,1), qsfc(i,1), qvg(i,1), emiss(i,1), alb(i,1), &
          znt(i,1), rhosnf(i,1), precipfr(i,1), snowfallac(i,1)
    end do
  end do
  close(unit)
end program run_ruc_step_oracle
