! Column driver for the byte-unmodified WRF v4.6.1 Purdue Lin scheme.
!
! Reads lin-inputs.bin (written by make_lin_inputs.py), calls the public
! lin_et_al once per column exactly as phys/module_microphysics_driver.F
! does for mp_physics = 2 (F_QG = .true., F_QNDROP = .false., the qndrop and
! WRF-Chem diagnostic arguments absent, diagflag = .true.), and writes every
! state and surface output to lin-outputs.bin.  Each column is its own call
! (its = ite = column) so each can carry its own time step.
!
! Physical constants come from WRF's own module_model_constants, the values
! the microphysics driver hands the scheme.
program run_lin
  use module_model_constants, only: g, cp, r_d, r_v, xls, xlv, xlf, &
       rhowater, rhosnow, ep_2, svp1, svp2, svp3, svpt0
  use module_mp_lin, only: lin_et_al
  implicit none

  integer :: ncol, nz, ic, u, ios
  real, allocatable :: th(:,:,:), qv(:,:,:), qc(:,:,:), qr(:,:,:), &
       qi(:,:,:), qs(:,:,:), qg(:,:,:), rho(:,:,:), pii(:,:,:), &
       p(:,:,:), z(:,:,:), dz8w(:,:,:), refl(:,:,:)
  real, allocatable :: dt(:), ht(:,:), rainnc(:,:), rainncv(:,:), &
       snownc(:,:), snowncv(:,:), graupelnc(:,:), graupelncv(:,:), sr(:,:)
  character(len=512) :: inpath, outpath
  logical :: f_qg, f_qndrop, diagflag

  if (command_argument_count() /= 2) then
     print *, 'usage: run_lin lin-inputs.bin lin-outputs.bin'
     stop 2
  end if
  call get_command_argument(1, inpath)
  call get_command_argument(2, outpath)

  open(newunit=u, file=trim(inpath), access='stream', form='unformatted', &
       status='old', iostat=ios)
  if (ios /= 0) stop 3
  read(u) ncol, nz
  allocate(th(ncol,nz,1), qv(ncol,nz,1), qc(ncol,nz,1), qr(ncol,nz,1), &
       qi(ncol,nz,1), qs(ncol,nz,1), qg(ncol,nz,1), rho(ncol,nz,1), &
       pii(ncol,nz,1), p(ncol,nz,1), z(ncol,nz,1), dz8w(ncol,nz,1), &
       refl(ncol,nz,1))
  allocate(dt(ncol), ht(ncol,1), rainnc(ncol,1), rainncv(ncol,1), &
       snownc(ncol,1), snowncv(ncol,1), graupelnc(ncol,1), &
       graupelncv(ncol,1), sr(ncol,1))
  read(u) dt, ht, rainnc, snownc, graupelnc
  read(u) th, qv, qc, qr, qi, qs, qg, rho, pii, p, z, dz8w
  close(u)

  rainncv = -999.
  snowncv = -999.
  graupelncv = -999.
  sr = -999.
  refl = -35.
  f_qg = .true.
  f_qndrop = .false.
  diagflag = .true.

  do ic = 1, ncol
     call lin_et_al(th=th, qv=qv, ql=qc, qr=qr, qi=qi, qs=qs,         &
          rho=rho, pii=pii, p=p, dt_in=dt(ic), z=z, ht=ht, dz8w=dz8w, &
          grav=g, cp=cp, rair=r_d, rvapor=r_v,                        &
          xls=xls, xlv=xlv, xlf=xlf, rhowater=rhowater,               &
          rhosnow=rhosnow, ep2=ep_2, svp1=svp1, svp2=svp2,            &
          svp3=svp3, svpt0=svpt0,                                     &
          rainnc=rainnc, rainncv=rainncv,                             &
          snownc=snownc, snowncv=snowncv,                             &
          graupelnc=graupelnc, graupelncv=graupelncv, sr=sr,          &
          refl_10cm=refl, diagflag=diagflag, do_radar_ref=1,          &
          ids=1, ide=ncol+1, jds=1, jde=2, kds=1, kde=nz+1,           &
          ims=1, ime=ncol, jms=1, jme=1, kms=1, kme=nz,               &
          its=ic, ite=ic, jts=1, jte=1, kts=1, kte=nz,                &
          f_qg=f_qg, f_qndrop=f_qndrop, qg=qg)
  end do

  open(newunit=u, file=trim(outpath), access='stream', &
       form='unformatted', status='replace')
  write(u) ncol, nz
  write(u) th, qv, qc, qr, qi, qs, qg
  write(u) rainnc, rainncv, snownc, snowncv, graupelnc, graupelncv, sr
  write(u) refl
  close(u)
end program run_lin
