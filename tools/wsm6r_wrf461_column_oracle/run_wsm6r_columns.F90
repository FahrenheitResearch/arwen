program run_wsm6r_columns
  ! Drive WRF v4.6.1's own WSM6R entry point -- phys/module_mp_wsm6r.F
  ! subroutine wsm6r, the routine module_microphysics_driver.F:2362 calls for
  ! mp_physics = 106 -- over the column set make_columns.py wrote, and dump
  ! every output word after every step.
  !
  ! The call passes exactly the arguments the driver passes (th, the six
  ! moist species, rho, pi_phy, p, dz8w, dt, rainnc, rainncv and the tile
  ! bounds).  The columns form one WRF tile row: its..ite.  module_mp_wsm6r.F
  ! is self-contained: it has no USE and takes no physical constant from
  ! the driver.
  !
  ! RADII.  WRF's wsm6r computes no effective radius (WSM6R is not in
  ! module_physics_init.F's use_mp_re list).  gpuwm diagnoses WSM6's radii
  ! from the WSM6R state for RTE+RRTMGP cloud optics, because the two schemes
  ! share their particle-size PARAMETERs; that diagnosis is graded here
  ! against WRF's OWN mp_wsm6_effectRad_run (phys/physics_mmm), called after
  ! each wsm6r step exactly as module_mp_wsm6.F:215-226 calls it after
  ! mp_wsm6_run: t = th*pii, with mp_wsm6_init's words (pidnc, pidn0s) from
  ! module_physics_init.F's WSM6 arguments and module_model_constants.F's
  ! background and maximum radii.
  !
  ! RAINNCV.  WRF's wsm6r writes RAINNCV only when the minor loop's fallout
  ! exceeds qmin, and OVERWRITES it every minor loop.  The driver never
  ! zeroes it, so a dry step keeps the previous step's value.  This harness
  ! zeroes RAINNCV before every call (it is an input condition, like the
  ! column set); gpuwm's own RAINNCV is the step total, the declared
  ! divergence at the head of gpuwm/core/kernels/wsm6r.cu.
  !
  ! usage: run_wsm6r_columns COLUMNS.BIN OUT.BIN DT NSTEPS
  !
  ! OUT.BIN, little-endian:  int32 ncol, nz, nsteps; float32 dt;
  ! then per step: 8 level fields (nz*ncol, column index fastest)
  !   th, qv, qc, qi, qr, qs, qg, t (= th*pii, the wrapper's own product)
  ! then 3 more level fields: re_cloud, re_ice, re_snow (metres)
  ! and 2 surface fields (ncol): rain, rainncv.
  use module_mp_wsm6r, only: wsm6r
  use ccpp_kind_types, only: kind_phys
  use module_model_constants, only: rhoair0, rhowater, rhosnow, cliq, cpv, &
       svpt0, epsilon, re_qc_bg, re_qi_bg, re_qs_bg, re_qc_max, re_qi_max, re_qs_max
  use mp_wsm6, only: mp_wsm6_init
  use mp_wsm6_effectrad, only: mp_wsm6_effectRad_run
  implicit none

  integer :: ncol, nz, nsteps, n, k, uin, uout, hdr(2), errflg
  character(len=256) :: errmsg
  real(kind_phys), allocatable, dimension(:,:) :: thv, qcv, qiv, qsv, denv, rec, rei, res
  real :: dt
  character(len=1024) :: inpath, outpath, arg
  real, allocatable, dimension(:,:,:) :: th, q, qc, qr, qi, qs, qg, &
       den, pii, p, dz
  real, allocatable, dimension(:,:) :: rain, rainncv, buf
  real, allocatable :: sbuf(:)

  if (kind(1.0) /= 4) error stop 'default real is not real(4)'
  if (command_argument_count() /= 4) error stop 'usage: run_wsm6r_columns IN OUT DT NSTEPS'
  call get_command_argument(1, inpath)
  call get_command_argument(2, outpath)
  call get_command_argument(3, arg); read(arg, *) dt
  call get_command_argument(4, arg); read(arg, *) nsteps

  open(newunit=uin, file=trim(inpath), access='stream', form='unformatted', status='old')
  read(uin) hdr
  ncol = hdr(1); nz = hdr(2)
  allocate(th(ncol,nz+1,1), q(ncol,nz+1,1), qc(ncol,nz+1,1), qr(ncol,nz+1,1), &
       qi(ncol,nz+1,1), qs(ncol,nz+1,1), qg(ncol,nz+1,1), den(ncol,nz+1,1), &
       pii(ncol,nz+1,1), p(ncol,nz+1,1), dz(ncol,nz+1,1))
  allocate(rain(ncol,1), rainncv(ncol,1), buf(ncol,nz), sbuf(ncol))
  allocate(thv(ncol,nz), qcv(ncol,nz), qiv(ncol,nz), qsv(ncol,nz), denv(ncol,nz), &
       rec(ncol,nz), rei(ncol,nz), res(ncol,nz))
  if (kind(1.0_kind_phys) /= 4) error stop 'kind_phys is not real(4): build with -DRWORDSIZE=4'
  ! The staggered top row (k = nz+1) is never read (kte = nz); keep it finite.
  th = 300.; q = 0.; qc = 0.; qr = 0.; qi = 0.; qs = 0.; qg = 0.
  den = 1.; pii = 1.; p = 1.e5; dz = 100.
  read(uin) buf; th(:,1:nz,1) = buf
  read(uin) buf; q(:,1:nz,1) = buf
  read(uin) buf; qc(:,1:nz,1) = buf
  read(uin) buf; qr(:,1:nz,1) = buf
  read(uin) buf; qi(:,1:nz,1) = buf
  read(uin) buf; qs(:,1:nz,1) = buf
  read(uin) buf; qg(:,1:nz,1) = buf
  read(uin) buf; den(:,1:nz,1) = buf
  read(uin) buf; pii(:,1:nz,1) = buf
  read(uin) buf; p(:,1:nz,1) = buf
  read(uin) buf; dz(:,1:nz,1) = buf
  read(uin) sbuf; rain(:,1) = sbuf
  close(uin)

  call mp_wsm6_init(rhoair0, rhowater, rhosnow, cliq, cpv, 0, errmsg, errflg)
  if (errflg /= 0) error stop 'mp_wsm6_init failed'

  open(newunit=uout, file=trim(outpath), access='stream', form='unformatted', status='replace')
  write(uout) ncol, nz, nsteps, dt

  do n = 1, nsteps
    rainncv = 0.
    call wsm6r(th=th, q=q, qc=qc, qr=qr, qi=qi, qs=qs, qg=qg,              &
               den=den, pii=pii, p=p, delz=dz, delt=dt,                    &
               rain=rain, rainncv=rainncv,                                 &
               ids=1, ide=ncol+1, jds=1, jde=2, kds=1, kde=nz+1,           &
               ims=1, ime=ncol, jms=1, jme=1, kms=1, kme=nz+1,             &
               its=1, ite=ncol, jts=1, jte=1, kts=1, kte=nz)
    write(uout) th(:,1:nz,1)
    write(uout) q(:,1:nz,1)
    write(uout) qc(:,1:nz,1)
    write(uout) qi(:,1:nz,1)
    write(uout) qr(:,1:nz,1)
    write(uout) qs(:,1:nz,1)
    write(uout) qg(:,1:nz,1)
    do k = 1, nz
      buf(:,k) = th(:,k,1)*pii(:,k,1)
    enddo
    write(uout) buf
    thv = buf
    qcv = qc(:,1:nz,1); qiv = qi(:,1:nz,1); qsv = qs(:,1:nz,1); denv = den(:,1:nz,1)
    rec = re_qc_bg; rei = re_qi_bg; res = re_qs_bg
    call mp_wsm6_effectRad_run(.true., thv, qcv, qiv, qsv, denv, epsilon, svpt0, &
         re_qc_bg, re_qi_bg, re_qs_bg, re_qc_max, re_qi_max, re_qs_max, &
         rec, rei, res, 1, ncol, 1, nz, errmsg, errflg)
    if (errflg /= 0) error stop 'mp_wsm6_effectRad_run failed'
    write(uout) rec
    write(uout) rei
    write(uout) res
    write(uout) rain(:,1), rainncv(:,1)
  enddo
  close(uout)
end program run_wsm6r_columns
