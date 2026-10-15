program run_wsm7_columns
  ! Drive WRF v4.6.1's own WSM7 entry point -- phys/module_mp_wsm7.F
  ! subroutine wsm7, the routine module_microphysics_driver.F:2385 calls for
  ! mp_physics = 24 -- over the column set make_columns.py wrote, and dump
  ! every output word after every step.
  !
  ! Everything below is WRF's: wsm7init with the arguments
  ! module_physics_init.F:4489 passes, the wsm7 wrapper with the arguments
  ! module_microphysics_driver.F:2385-2418 passes (constants from WRF's own
  ! share/module_model_constants.F, compiled from the pinned tree), radar
  ! reflectivity requested (diagflag, do_radar_ref = 1) and the radiation
  ! effective radii requested (has_reqc = has_reqi = has_reqs = 1, as an
  ! RRTMG run sets them, module_physics_init.F:988-1024).  The columns form
  ! one WRF tile row: its..ite.
  !
  ! usage: run_wsm7_columns COLUMNS.BIN OUT.BIN DT NSTEPS [PROBE.TXT]
  !
  ! OUT.BIN, little-endian:  int32 ncol, nz, nsteps; float32 dt; then per
  ! step 13 level fields (nz*ncol, column index fastest)
  !   th, qv, qc, qi, qr, qs, qg, qh, re_cloud, re_ice, re_snow, refl_10cm,
  !   t (= th*pii, the temperature the wrapper hands effectRad and refl)
  ! and 9 surface fields (ncol): rain, rainncv, snow, snowncv, graupel,
  ! graupelncv, hail, hailncv, sr.
  !
  ! PROBE.TXT (the probe arm only, built with -DWSM7_PROBE): per column the
  ! number of PLM remap target layers whose find_kt search fell through with
  ! arrival mass above kb (tools/wsm7_wrf461_column_oracle/defined_arm.py).
  use module_model_constants, only: g, cp, cpv, r_d, r_v, svpt0, ep_1, ep_2, &
       epsilon, xls, xlv, xlf, rhoair0, rhowater, rhosnow, cliq, cice, psat, &
       re_qc_bg, re_qi_bg, re_qs_bg
  use module_mp_wsm7, only: wsm7, wsm7init
#ifdef WSM7_PROBE
  use module_mp_wsm7, only: wsm7probe_ft
#endif
  implicit none

  integer :: ncol, nz, nsteps, n, k, uin, uout, hdr(2), i
  real :: dt
  character(len=1024) :: inpath, outpath, arg, probepath
  real, allocatable, dimension(:,:,:) :: th, q, qc, qr, qi, qs, qg, qh, &
       den, pii, p, dz, rec, rei, res, refl
  real, allocatable, dimension(:,:) :: rain, rainncv, snow, snowncv, &
       graupel, graupelncv, hail, hailncv, sr, buf
  real, allocatable :: sbuf(:)

  ! WRF's -DRWORDSIZE=4 configuration: default REAL is binary32.
  if (kind(1.0) /= 4 .or. digits(1.0) /= 24) error stop 'default REAL is not binary32'
  if (command_argument_count() < 4) error stop 'usage: run_wsm7_columns IN OUT DT NSTEPS [PROBE]'
  call get_command_argument(1, inpath)
  call get_command_argument(2, outpath)
  call get_command_argument(3, arg); read(arg, *) dt
  call get_command_argument(4, arg); read(arg, *) nsteps
  probepath = ''
  if (command_argument_count() >= 5) call get_command_argument(5, probepath)

  open(newunit=uin, file=trim(inpath), access='stream', form='unformatted', &
       status='old', convert='little_endian')
  read(uin) hdr
  ncol = hdr(1); nz = hdr(2)
  allocate(th(ncol,nz+1,1), q(ncol,nz+1,1), qc(ncol,nz+1,1), qr(ncol,nz+1,1), &
       qi(ncol,nz+1,1), qs(ncol,nz+1,1), qg(ncol,nz+1,1), qh(ncol,nz+1,1), &
       den(ncol,nz+1,1), pii(ncol,nz+1,1), p(ncol,nz+1,1), dz(ncol,nz+1,1), &
       rec(ncol,nz+1,1), rei(ncol,nz+1,1), res(ncol,nz+1,1), refl(ncol,nz+1,1))
  allocate(rain(ncol,1), rainncv(ncol,1), snow(ncol,1), snowncv(ncol,1), &
       graupel(ncol,1), graupelncv(ncol,1), hail(ncol,1), hailncv(ncol,1), &
       sr(ncol,1), buf(ncol,nz), sbuf(ncol))
  ! The staggered top row (k = nz+1) is never read by the wrapper (kte = nz);
  ! give it finite values anyway.
  th = 300.; q = 0.; qc = 0.; qr = 0.; qi = 0.; qs = 0.; qg = 0.; qh = 0.
  den = 1.; pii = 1.; p = 1.e5; dz = 100.
  read(uin) buf; th(:,1:nz,1) = buf
  read(uin) buf; q(:,1:nz,1) = buf
  read(uin) buf; qc(:,1:nz,1) = buf
  read(uin) buf; qr(:,1:nz,1) = buf
  read(uin) buf; qi(:,1:nz,1) = buf
  read(uin) buf; qs(:,1:nz,1) = buf
  read(uin) buf; qg(:,1:nz,1) = buf
  read(uin) buf; qh(:,1:nz,1) = buf
  read(uin) buf; den(:,1:nz,1) = buf
  read(uin) buf; pii(:,1:nz,1) = buf
  read(uin) buf; p(:,1:nz,1) = buf
  read(uin) buf; dz(:,1:nz,1) = buf
  read(uin) sbuf; rain(:,1) = sbuf
  read(uin) sbuf; snow(:,1) = sbuf
  read(uin) sbuf; graupel(:,1) = sbuf
  read(uin) sbuf; hail(:,1) = sbuf
  close(uin)
  rainncv = 0.; snowncv = 0.; graupelncv = 0.; hailncv = 0.; sr = 0.
  rec = re_qc_bg; rei = re_qi_bg; res = re_qs_bg; refl = -35.

  call wsm7init(rhoair0, rhowater, rhosnow, cliq, cpv, .true.)

  open(newunit=uout, file=trim(outpath), access='stream', form='unformatted', &
       status='replace', convert='little_endian')
  write(uout) ncol, nz, nsteps, dt

  do n = 1, nsteps
    call wsm7(th=th, q=q, qc=qc, qr=qr, qi=qi, qs=qs, qg=qg, qh=qh,       &
              den=den, pii=pii, p=p, delz=dz,                              &
              delt=dt, g=g, cpd=cp, cpv=cpv, rd=r_d, rv=r_v, t0c=svpt0,     &
              ep1=ep_1, ep2=ep_2, qmin=epsilon,                            &
              xls=xls, xlv0=xlv, xlf0=xlf, den0=rhoair0, denr=rhowater,     &
              cliq=cliq, cice=cice, psat=psat,                             &
              rain=rain, rainncv=rainncv, snow=snow, snowncv=snowncv,      &
              sr=sr,                                                       &
              refl_10cm=refl, diagflag=.true., do_radar_ref=1,             &
              graupel=graupel, graupelncv=graupelncv,                      &
              hail=hail, hailncv=hailncv,                                  &
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
    write(uout) qg(:,1:nz,1)
    write(uout) qh(:,1:nz,1)
    write(uout) rec(:,1:nz,1)
    write(uout) rei(:,1:nz,1)
    write(uout) res(:,1:nz,1)
    write(uout) refl(:,1:nz,1)
    do k = 1, nz
      buf(:,k) = th(:,k,1)*pii(:,k,1)
    enddo
    write(uout) buf
    write(uout) rain(:,1), rainncv(:,1), snow(:,1), snowncv(:,1), &
                graupel(:,1), graupelncv(:,1), hail(:,1), hailncv(:,1), sr(:,1)
  enddo
  close(uout)

#ifdef WSM7_PROBE
  if (len_trim(probepath) > 0) then
    open(newunit=uout, file=trim(probepath), status='replace', action='write')
    do i = 1, ncol
      write(uout, '(i0,1x,i0)') i, wsm7probe_ft(i)
    enddo
    close(uout)
  endif
#endif
end program run_wsm7_columns
