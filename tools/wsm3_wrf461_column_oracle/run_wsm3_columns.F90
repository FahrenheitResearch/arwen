program run_wsm3_columns
  ! Drive WRF v4.6.1's own WSM3 entry point -- phys/module_mp_wsm3.F
  ! subroutine wsm3, the routine module_microphysics_driver.F:2230 calls for
  ! mp_physics = 3 -- over the column set make_columns.py wrote, and dump
  ! every output word after every step.
  !
  ! Everything below is WRF's: wsm3init with the arguments
  ! module_physics_init.F:4483 passes, the wsm3 wrapper with the arguments
  ! module_microphysics_driver.F:2230-2255 passes (constants from WRF's own
  ! share/module_model_constants.F, compiled from the pinned tree), and the
  ! radiation effective radii requested (has_reqc = has_reqi = has_reqs = 1,
  ! as module_physics_init.F:1004-1024 sets them for mp_physics = 3 under
  ! RRTMG or Goddard radiation with use_mp_re = 1).  WRF computes no
  ! REFL_10CM for WSM3 (the driver passes none; module_diag_nwp.F:549 has
  ! "nothing to do here"), so none is dumped.  The columns form one WRF tile
  ! row: its..ite.
  !
  ! usage: run_wsm3_columns COLUMNS.BIN OUT.BIN DT NSTEPS
  !
  ! OUT.BIN, little-endian:  int32 ncol, nz, nsteps; float32 dt;
  ! then per step: 8 level fields (nz*ncol, column index fastest)
  !   th, qv, qc, qr, re_cloud, re_ice, re_snow,
  !   t (= th*pii, the temperature the wrapper hands effectRad)
  ! and 5 surface fields (ncol): rain, rainncv, snow, snowncv, sr.
  use module_model_constants, only: g, cp, cpv, r_d, r_v, svpt0, ep_1, ep_2, &
       epsilon, xls, xlv, xlf, rhoair0, rhowater, rhosnow, cliq, cice, psat, &
       re_qc_bg, re_qi_bg, re_qs_bg
  use module_mp_wsm3, only: wsm3, wsm3init
  implicit none

  integer :: ncol, nz, nsteps, n, k, uin, uout, hdr(2)
  real :: dt
  character(len=1024) :: inpath, outpath, arg
  real, allocatable, dimension(:,:,:) :: th, q, qc, qr, w, den, pii, p, dz, &
       rec, rei, res
  real, allocatable, dimension(:,:) :: rain, rainncv, snow, snowncv, sr, buf, wbuf
  real, allocatable :: sbuf(:)

  if (kind(1.0) /= 4) error stop 'default REAL is not real(4)'
  if (command_argument_count() /= 4) error stop 'usage: run_wsm3_columns IN OUT DT NSTEPS'
  call get_command_argument(1, inpath)
  call get_command_argument(2, outpath)
  call get_command_argument(3, arg); read(arg, *) dt
  call get_command_argument(4, arg); read(arg, *) nsteps

  open(newunit=uin, file=trim(inpath), access='stream', form='unformatted', status='old')
  read(uin) hdr
  ncol = hdr(1); nz = hdr(2)
  allocate(th(ncol,nz+1,1), q(ncol,nz+1,1), qc(ncol,nz+1,1), qr(ncol,nz+1,1), &
       w(ncol,nz+1,1), den(ncol,nz+1,1), pii(ncol,nz+1,1), p(ncol,nz+1,1), &
       dz(ncol,nz+1,1), rec(ncol,nz+1,1), rei(ncol,nz+1,1), res(ncol,nz+1,1))
  allocate(rain(ncol,1), rainncv(ncol,1), snow(ncol,1), snowncv(ncol,1), &
       sr(ncol,1), buf(ncol,nz), wbuf(ncol,nz+1), sbuf(ncol))
  ! The staggered top row (k = nz+1) of the mass fields is never read by the
  ! wrapper (kte = nz); give it finite values anyway.  w is a full-level
  ! field and carries its own top face.
  th = 300.; q = 0.; qc = 0.; qr = 0.; w = 0.
  den = 1.; pii = 1.; p = 1.e5; dz = 100.
  read(uin) buf; th(:,1:nz,1) = buf
  read(uin) buf; q(:,1:nz,1) = buf
  read(uin) buf; qc(:,1:nz,1) = buf
  read(uin) buf; qr(:,1:nz,1) = buf
  read(uin) buf; den(:,1:nz,1) = buf
  read(uin) buf; pii(:,1:nz,1) = buf
  read(uin) buf; p(:,1:nz,1) = buf
  read(uin) buf; dz(:,1:nz,1) = buf
  read(uin) wbuf; w(:,1:nz+1,1) = wbuf
  read(uin) sbuf; rain(:,1) = sbuf
  read(uin) sbuf; snow(:,1) = sbuf
  close(uin)
  rainncv = 0.; snowncv = 0.; sr = 0.
  rec = re_qc_bg; rei = re_qi_bg; res = re_qs_bg

  call wsm3init(rhoair0, rhowater, rhosnow, cliq, cpv, .true.)

  open(newunit=uout, file=trim(outpath), access='stream', form='unformatted', status='replace')
  write(uout) ncol, nz, nsteps, dt

  do n = 1, nsteps
    call wsm3(th=th, q=q, qci=qc, qrs=qr,                                &
              w=w, den=den, pii=pii, p=p, delz=dz,                       &
              delt=dt, g=g, cpd=cp, cpv=cpv, rd=r_d, rv=r_v, t0c=svpt0,  &
              ep1=ep_1, ep2=ep_2, qmin=epsilon,                          &
              xls=xls, xlv0=xlv, xlf0=xlf, den0=rhoair0, denr=rhowater,  &
              cliq=cliq, cice=cice, psat=psat,                           &
              rain=rain, rainncv=rainncv, snow=snow, snowncv=snowncv,    &
              sr=sr,                                                     &
              has_reqc=1, has_reqi=1, has_reqs=1,                        &
              re_cloud=rec, re_ice=rei, re_snow=res,                     &
              ids=1, ide=ncol+1, jds=1, jde=2, kds=1, kde=nz+1,          &
              ims=1, ime=ncol, jms=1, jme=1, kms=1, kme=nz+1,            &
              its=1, ite=ncol, jts=1, jte=1, kts=1, kte=nz)
    write(uout) th(:,1:nz,1)
    write(uout) q(:,1:nz,1)
    write(uout) qc(:,1:nz,1)
    write(uout) qr(:,1:nz,1)
    write(uout) rec(:,1:nz,1)
    write(uout) rei(:,1:nz,1)
    write(uout) res(:,1:nz,1)
    do k = 1, nz
      buf(:,k) = th(:,k,1)*pii(:,k,1)
    enddo
    write(uout) buf
    write(uout) rain(:,1), rainncv(:,1), snow(:,1), snowncv(:,1), sr(:,1)
  enddo
  close(uout)
end program run_wsm3_columns
