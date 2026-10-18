program run_morrison_columns
  ! Drive WRF's public MP_MORR_TWO_MOMENT over many columns read from a
  ! binary file, for stress sets beyond the 28 committed fixture columns.
  ! Every output written here is a value WRF returned; this program never
  ! reimplements Morrison arithmetic.
  !
  ! usage: run_morrison_columns IN.bin OUT.bin
  ! IN  (stream, little endian): int32 ncol, nz, mode; real32 dt; then the
  !     18 level inputs th qv qc qr qi qs qg ni ns nr ng rho pii p dz
  !     qrcuten qscuten qicuten, each (ncol, nz) Fortran order; then the
  !     7 surface inputs rainnc rainncv snownc snowncv graupelnc graupelncv
  !     sr, each (ncol).
  ! OUT: the 12 level outputs th qv qc qr qi qs qg ni ns nr ng refl_10cm,
  !     then the 7 surface outputs, same layouts.
  use module_mp_morr_two_moment, only: morr_two_moment_init, &
                                        mp_morr_two_moment
  implicit none
  integer :: ncol, nz, mode, uin, uout
  real :: dt
  real, allocatable, dimension(:,:,:) :: th, qv, qc, qr, qi, qs, qg, ni, &
       ns, nr, ng, rho, pii, p, dz, qrcu, qscu, qicu, w, refl
  real, allocatable, dimension(:,:) :: ht, rainnc, rainncv, snownc, &
       snowncv, graupelnc, graupelncv, sr
  character(len=1024) :: inpath, outpath

  call get_command_argument(1, inpath)
  call get_command_argument(2, outpath)
  open(newunit=uin, file=trim(inpath), access='stream', form='unformatted', &
       status='old', action='read')
  read(uin) ncol, nz, mode, dt
  allocate(th(ncol,nz,1), qv(ncol,nz,1), qc(ncol,nz,1), qr(ncol,nz,1),     &
           qi(ncol,nz,1), qs(ncol,nz,1), qg(ncol,nz,1), ni(ncol,nz,1),     &
           ns(ncol,nz,1), nr(ncol,nz,1), ng(ncol,nz,1), rho(ncol,nz,1),    &
           pii(ncol,nz,1), p(ncol,nz,1), dz(ncol,nz,1), qrcu(ncol,nz,1),   &
           qscu(ncol,nz,1), qicu(ncol,nz,1), w(ncol,nz,1), refl(ncol,nz,1))
  allocate(ht(ncol,1), rainnc(ncol,1), rainncv(ncol,1), snownc(ncol,1),    &
           snowncv(ncol,1), graupelnc(ncol,1), graupelncv(ncol,1),         &
           sr(ncol,1))
  read(uin) th, qv, qc, qr, qi, qs, qg, ni, ns, nr, ng, rho, pii, p, dz,   &
            qrcu, qscu, qicu
  read(uin) rainnc, rainncv, snownc, snowncv, graupelnc, graupelncv, sr
  close(uin)
  w = 0.0
  ht = 0.0
  refl = sign(0.0, -1.0)

  call morr_two_moment_init(mode)
  call mp_morr_two_moment(                                                 &
       itimestep=1, th=th, qv=qv, qc=qc, qr=qr, qi=qi, qs=qs, qg=qg,       &
       ni=ni, ns=ns, nr=nr, ng=ng, rho=rho, pii=pii, p=p,                  &
       dt_in=dt, dz=dz, ht=ht, w=w, rainnc=rainnc, rainncv=rainncv,        &
       sr=sr, snownc=snownc, snowncv=snowncv, graupelnc=graupelnc,         &
       graupelncv=graupelncv, refl_10cm=refl, diagflag=.true.,             &
       do_radar_ref=1, qrcuten=qrcu, qscuten=qscu, qicuten=qicu,           &
       ids=1, ide=ncol, jds=1, jde=1, kds=1, kde=nz,                       &
       ims=1, ime=ncol, jms=1, jme=1, kms=1, kme=nz,                       &
       its=1, ite=ncol, jts=1, jte=1, kts=1, kte=nz)

  open(newunit=uout, file=trim(outpath), access='stream',                 &
       form='unformatted', status='replace', action='write')
  write(uout) th, qv, qc, qr, qi, qs, qg, ni, ns, nr, ng, refl
  write(uout) rainnc, rainncv, snownc, snowncv, graupelnc, graupelncv, sr
  close(uout)
end program run_morrison_columns
