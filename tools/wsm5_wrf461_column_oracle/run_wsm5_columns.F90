program run_wsm5_columns
  ! Drive WRF v4.6.1's own WSM5 entry point -- phys/module_mp_wsm5.F
  ! subroutine wsm5, the routine module_microphysics_driver.F:2268 calls for
  ! mp_physics = 4 -- over a column set make_columns.py wrote, and dump
  ! every output word after every step.
  !
  ! Everything below is WRF's: wsm5init with the arguments
  ! module_physics_init.F:4485 passes, the wsm5 wrapper with the arguments
  ! module_microphysics_driver.F:2268-2300 passes (constants from WRF's own
  ! share/module_model_constants.F, the build tree's module), radar
  ! reflectivity requested (diagflag, do_radar_ref = 1) and the radiation
  ! effective radii requested (has_reqc = has_reqi = has_reqs = 1, as an
  ! RRTMG run sets them).  The columns form one WRF tile row: its..ite.
  !
  ! The arm it links against is chosen at link time (build.sh): the scalar
  ! or strict WRF build tree's own module_mp_wsm5.o, or a defined arm.
  ! Built with -DWSM5_PROBE it also writes the defined arm's per-column
  ! probe counters after every step.
  !
  ! usage: run_wsm5_columns COLUMNS.BIN OUT.BIN DT NSTEPS
  !
  ! OUT.BIN, little-endian:  int32 ncol, nz, nsteps, probe; float32 dt;
  ! then per step: 11 level fields (nz*ncol, column index fastest)
  !   th, qv, qc, qi, qr, qs, re_cloud, re_ice, re_snow, refl_10cm,
  !   t (= th*pii, the temperature the wrapper hands effectRad)
  ! and 5 surface fields (ncol): rain, rainncv, snow, snowncv, sr;
  ! with probe = 1, then two int32 ncol arrays: the cumulative remap
  ! fall-through and active-psevp counters of the defined arm.
  use module_model_constants, only: g, cp, cpv, r_d, r_v, svpt0, ep_1, ep_2, &
       epsilon, xls, xlv, xlf, rhoair0, rhowater, rhosnow, cliq, cice, psat, &
       re_qc_bg, re_qi_bg, re_qs_bg
  use module_mp_wsm5, only: wsm5init, wsm5
#ifdef WSM5_PROBE
  use module_mp_wsm5, only: wsm5_probe_remap, wsm5_probe_psevp
#endif
  implicit none

  integer :: ncol, nz, nsteps, n, k, uin, uout, hdr(2), probe
  real :: dt
  character(len=1024) :: inpath, outpath, arg
  real, allocatable, dimension(:,:,:) :: th, q, qc, qr, qi, qs, &
       den, pii, p, dz, rec, rei, res, refl
  real, allocatable, dimension(:,:) :: rain, rainncv, snow, snowncv, sr, buf
  real, allocatable :: sbuf(:)

  if (kind(1.0) /= 4) error stop 'default REAL is not real(4)'
  if (command_argument_count() /= 4) error stop 'usage: run_wsm5_columns IN OUT DT NSTEPS'
  call get_command_argument(1, inpath)
  call get_command_argument(2, outpath)
  call get_command_argument(3, arg); read(arg, *) dt
  call get_command_argument(4, arg); read(arg, *) nsteps
  probe = 0
#ifdef WSM5_PROBE
  probe = 1
#endif

  open(newunit=uin, file=trim(inpath), access='stream', form='unformatted', status='old')
  read(uin) hdr
  ncol = hdr(1); nz = hdr(2)
#ifdef WSM5_PROBE
  if (ncol > size(wsm5_probe_remap)) error stop 'more columns than probe slots'
#endif
  allocate(th(ncol,nz+1,1), q(ncol,nz+1,1), qc(ncol,nz+1,1), qr(ncol,nz+1,1), &
       qi(ncol,nz+1,1), qs(ncol,nz+1,1), den(ncol,nz+1,1), &
       pii(ncol,nz+1,1), p(ncol,nz+1,1), dz(ncol,nz+1,1), rec(ncol,nz+1,1), &
       rei(ncol,nz+1,1), res(ncol,nz+1,1), refl(ncol,nz+1,1))
  allocate(rain(ncol,1), rainncv(ncol,1), snow(ncol,1), snowncv(ncol,1), &
       sr(ncol,1), buf(ncol,nz), sbuf(ncol))
  ! The staggered top row (k = nz+1) is never read by the wrapper (kte = nz);
  ! give it finite values anyway.
  th = 300.; q = 0.; qc = 0.; qr = 0.; qi = 0.; qs = 0.
  den = 1.; pii = 1.; p = 1.e5; dz = 100.
  read(uin) buf; th(:,1:nz,1) = buf
  read(uin) buf; q(:,1:nz,1) = buf
  read(uin) buf; qc(:,1:nz,1) = buf
  read(uin) buf; qr(:,1:nz,1) = buf
  read(uin) buf; qi(:,1:nz,1) = buf
  read(uin) buf; qs(:,1:nz,1) = buf
  read(uin) buf; den(:,1:nz,1) = buf
  read(uin) buf; pii(:,1:nz,1) = buf
  read(uin) buf; p(:,1:nz,1) = buf
  read(uin) buf; dz(:,1:nz,1) = buf
  read(uin) sbuf; rain(:,1) = sbuf
  read(uin) sbuf; snow(:,1) = sbuf
  close(uin)
  rainncv = 0.; snowncv = 0.; sr = 0.
  rec = re_qc_bg; rei = re_qi_bg; res = re_qs_bg; refl = -35.

  call wsm5init(rhoair0, rhowater, rhosnow, cliq, cpv, .true.)

  open(newunit=uout, file=trim(outpath), access='stream', form='unformatted', status='replace')
  write(uout) ncol, nz, nsteps, probe, dt

  do n = 1, nsteps
    call wsm5(th=th, q=q, qc=qc, qr=qr, qi=qi, qs=qs,                      &
              den=den, pii=pii, p=p, delz=dz,                              &
              delt=dt, g=g, cpd=cp, cpv=cpv, rd=r_d, rv=r_v, t0c=svpt0,     &
              ep1=ep_1, ep2=ep_2, qmin=epsilon,                            &
              xls=xls, xlv0=xlv, xlf0=xlf, den0=rhoair0, denr=rhowater,     &
              cliq=cliq, cice=cice, psat=psat,                             &
              rain=rain, rainncv=rainncv, snow=snow, snowncv=snowncv,      &
              sr=sr,                                                       &
              refl_10cm=refl, diagflag=.true., do_radar_ref=1,             &
              has_reqc=1, has_reqi=1, has_reqs=1,                          &
              re_cloud=rec, re_ice=rei, re_snow=res,                       &
              ids=1, ide=ncol+1, jds=1, jde=2, kds=1, kde=nz+1,            &
              ims=1, ime=ncol, jms=1, jme=1, kms=1, kme=nz+1,              &
              its=1, ite=ncol, jts=1, jte=1, kts=1, kte=nz)
    write(uout) th(:,1:nz,1)
    write(uout) q(:,1:nz,1)
    write(uout) qc(:,1:nz,1)
    write(uout) qi(:,1:nz,1)
    write(uout) qr(:,1:nz,1)
    write(uout) qs(:,1:nz,1)
    write(uout) rec(:,1:nz,1)
    write(uout) rei(:,1:nz,1)
    write(uout) res(:,1:nz,1)
    write(uout) refl(:,1:nz,1)
    do k = 1, nz
      buf(:,k) = th(:,k,1)*pii(:,k,1)
    enddo
    write(uout) buf
    write(uout) rain(:,1), rainncv(:,1), snow(:,1), snowncv(:,1), sr(:,1)
#ifdef WSM5_PROBE
    write(uout) wsm5_probe_remap(1:ncol), wsm5_probe_psevp(1:ncol)
#endif
  enddo
  close(uout)
end program run_wsm5_columns
