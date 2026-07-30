program nssl2_composed_column
  use module_mp_nssl_2mom, only: nssl_2mom_init, nssl_2mom_driver
  implicit none

  integer :: input_unit, output_unit, nz, k
  integer :: ids, ide, jds, jde, kds, kde
  integer :: ims, ime, jms, jme, kms, kme
  integer :: its, ite, jts, jte, kts, kte
  real :: dt_s, nssl_params(20)
  real, allocatable :: theta(:,:,:), qv(:,:,:), qc(:,:,:), qr(:,:,:)
  real, allocatable :: qi(:,:,:), qs(:,:,:), qg(:,:,:), qh(:,:,:)
  real, allocatable :: qndrop(:,:,:), qnr(:,:,:), qni(:,:,:)
  real, allocatable :: qns(:,:,:), qng(:,:,:), qnh(:,:,:), qnn(:,:,:)
  real, allocatable :: qvolg(:,:,:), qvolh(:,:,:)
  real, allocatable :: pressure(:,:,:), exner(:,:,:), rho(:,:,:)
  real, allocatable :: dz(:,:,:), w(:,:,:), w_lower(:), w_upper(:)
  real, allocatable :: dbz(:,:,:), re_cloud(:,:,:), re_ice(:,:,:)
  real, allocatable :: re_snow(:,:,:)
  real :: rainnc(1,1), rainncv(1,1), snownc(1,1), snowncv(1,1)
  real :: grplnc(1,1), grplncv(1,1), hailnc(1,1), hailncv(1,1)
  real :: sr(1,1)
  character(len=512) :: input_path, output_path

  call get_command_argument(1, input_path)
  call get_command_argument(2, output_path)
  if (len_trim(input_path) == 0 .or. len_trim(output_path) == 0) then
     error stop 'usage: nssl2_composed_column INPUT.txt OUTPUT.csv'
  endif

  open(newunit=input_unit, file=trim(input_path), status='old', action='read')
  read(input_unit,*) nz, dt_s
  if (nz < 3 .or. dt_s <= 0.0) error stop 'invalid composed-column header'

  allocate(theta(1,nz,1), qv(1,nz,1), qc(1,nz,1), qr(1,nz,1))
  allocate(qi(1,nz,1), qs(1,nz,1), qg(1,nz,1), qh(1,nz,1))
  allocate(qndrop(1,nz,1), qnr(1,nz,1), qni(1,nz,1))
  allocate(qns(1,nz,1), qng(1,nz,1), qnh(1,nz,1), qnn(1,nz,1))
  allocate(qvolg(1,nz,1), qvolh(1,nz,1))
  allocate(pressure(1,nz,1), exner(1,nz,1), rho(1,nz,1))
  allocate(dz(1,nz,1), w(1,nz,1), w_lower(nz), w_upper(nz))
  allocate(dbz(1,nz,1), re_cloud(1,nz,1), re_ice(1,nz,1))
  allocate(re_snow(1,nz,1))

  do k = 1, nz
     read(input_unit,*) theta(1,k,1), qv(1,k,1), qc(1,k,1), &
          qr(1,k,1), qi(1,k,1), qs(1,k,1), qg(1,k,1), &
          qh(1,k,1), qndrop(1,k,1), qnr(1,k,1), qni(1,k,1), &
          qns(1,k,1), qng(1,k,1), qnh(1,k,1), qnn(1,k,1), &
          qvolg(1,k,1), qvolh(1,k,1), pressure(1,k,1), &
          exner(1,k,1), rho(1,k,1), dz(1,k,1), &
          w_lower(k), w_upper(k)
     w(1,k,1) = 0.5 * (w_lower(k) + w_upper(k))
  enddo
  close(input_unit)

  nssl_params = 0.0
  nssl_params(1:10) = [0.5e9, 0.0, 1.0, 4.0e5, 4.0e4, &
                       8.0e5, 3.0e6, 500.0, 900.0, 100.0]
  call nssl_2mom_init(nssl_params=nssl_params, ipctmp=5, mixphase=0, &
       nssl_density_on=.true., nssl_hail_on=.true.,                 &
       nssl_ccn_on=.true., nssl_icdx=6, nssl_icdxhl=6)

  rainnc = 0.0
  rainncv = 0.0
  snownc = 0.0
  snowncv = 0.0
  grplnc = 0.0
  grplncv = 0.0
  hailnc = 0.0
  hailncv = 0.0
  sr = 0.0
  dbz = -999.0
  re_cloud = -999.0
  re_ice = -999.0
  re_snow = -999.0

  ids = 1; ide = 1; jds = 1; jde = 1; kds = 1; kde = nz
  ims = 1; ime = 1; jms = 1; jme = 1; kms = 1; kme = nz
  its = 1; ite = 1; jts = 1; jte = 1; kts = 1; kte = nz

  call nssl_2mom_driver( &
       qv=qv, qc=qc, qr=qr, qi=qi, qs=qs, qh=qg, qhl=qh, &
       ccw=qndrop, crw=qnr, cci=qni, csw=qns, chw=qng, chl=qnh, &
       cn=qnn, f_cn=.true., vhw=qvolg, vhl=qvolh, &
       f_vhw=.true., f_vhl=.true., &
       th=theta, pii=exner, p=pressure, w=w, dn=rho, dz=dz, &
       dtp=dt_s, itimestep=2, &
       rainnc=rainnc, rainncv=rainncv, &
       snownc=snownc, snowncv=snowncv, &
       grplnc=grplnc, grplncv=grplncv, &
       hailnc=hailnc, hailncv=hailncv, sr=sr, &
       dx=500.0, dy=500.0, dbz=dbz, diagflag=.true., &
       re_cloud=re_cloud, re_ice=re_ice, re_snow=re_snow, &
       has_reqc=1, has_reqi=1, has_reqs=1, &
       nssl_progn=.false., &
       ids=ids, ide=ide, jds=jds, jde=jde, kds=kds, kde=kde, &
       ims=ims, ime=ime, jms=jms, jme=jme, kms=kms, kme=kme, &
       its=its, ite=ite, jts=jts, jte=jte, kts=kts, kte=kte)

  open(newunit=output_unit, file=trim(output_path), status='replace', &
       action='write', form='formatted')
  write(output_unit,'(A)') &
       'engine,k,theta,qv,qc,qr,qi,qs,qg,qh,qndrop,qnr,qni,qns,' // &
       'qng,qnh,qnn,qvolg,qvolh,refl_10cm,effc_m,effi_m,effs_m,' // &
       'rainnc,rainncv,snownc,snowncv,graupelnc,graupelncv,' // &
       'hailnc,hailncv,sr'
  do k = 1, nz
     write(output_unit,'(A,",",I0,30(",",ES24.16E3))') &
          'wrf', k, theta(1,k,1), qv(1,k,1), qc(1,k,1), qr(1,k,1), &
          qi(1,k,1), qs(1,k,1), qg(1,k,1), qh(1,k,1), &
          qndrop(1,k,1), qnr(1,k,1), qni(1,k,1), qns(1,k,1), &
          qng(1,k,1), qnh(1,k,1), qnn(1,k,1), &
          qvolg(1,k,1), qvolh(1,k,1), dbz(1,k,1), &
          re_cloud(1,k,1), re_ice(1,k,1), re_snow(1,k,1), &
          rainnc(1,1), rainncv(1,1), snownc(1,1), snowncv(1,1), &
          grplnc(1,1), grplncv(1,1), hailnc(1,1), hailncv(1,1), &
          sr(1,1)
  enddo
  close(output_unit)
  print '(A,1X,A)', 'NSSL2_COMPOSED_WRF_COMPLETE', trim(output_path)
end program nssl2_composed_column
