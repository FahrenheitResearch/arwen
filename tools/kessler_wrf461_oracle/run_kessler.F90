program run_kessler_oracle
  ! Drive SUBROUTINE kessler from the byte-unmodified WRF v4.6.1
  ! phys/module_mp_kessler.F.  Every output word written here was returned
  ! by WRF's routine; this program reads inputs and copies arrays, it never
  ! does Kessler arithmetic.  Constants are WRF's own module_model_constants
  ! parameters, passed exactly as module_microphysics_driver.F:983-996 does.
  !
  ! usage: run_kessler INPUTS.bin OUTPUTS.bin
  ! Each case restarts from the inputs and calls kessler nsteps times with
  ! its dt (free running: step n reads step n-1's outputs, as in WRF).  After
  ! every call it writes t, qv, qc, qr (each ncol x nz, i fastest) and
  ! RAINNC, RAINNCV (ncol).
  use module_model_constants, only: xlv, cp, ep_2, svp1, svp2, svp3, &
                                    svpt0, rhowater
  use module_mp_kessler, only: kessler
  implicit none

  integer :: ncol, nz, ncase, kme, ic, istep, uin, uout
  real, allocatable :: dts(:)
  integer, allocatable :: nsteps(:)
  real, allocatable, dimension(:, :, :) :: t0, qv0, qc0, qr0, rho, pii, &
                                           z, dz8w, t, qv, qc, qr
  real, allocatable :: rainnc0(:, :), rainnc(:, :), rainncv(:, :)
  character(len=1024) :: fin, fout

  call get_command_argument(1, fin)
  call get_command_argument(2, fout)
  if (len_trim(fin) == 0 .or. len_trim(fout) == 0) then
    write(*, '(A)') 'usage: run_kessler INPUTS.bin OUTPUTS.bin'
    error stop 2
  end if
  if (storage_size(1.0) /= 32) error stop 3

  open(newunit=uin, file=trim(fin), access='stream', form='unformatted', &
       status='old', action='read')
  read(uin) ncol, nz, ncase
  allocate(dts(ncase), nsteps(ncase))
  read(uin) dts
  read(uin) nsteps
  ! WRF memory dims: kme = kde = the staggered top (nz + 1); the scheme's
  ! tile is kts..kte = 1..nz half levels.  Level nz+1 is never read.
  kme = nz + 1
  allocate(t0(ncol, kme, 1), qv0(ncol, kme, 1), qc0(ncol, kme, 1), &
           qr0(ncol, kme, 1), rho(ncol, kme, 1), pii(ncol, kme, 1), &
           z(ncol, kme, 1), dz8w(ncol, kme, 1), t(ncol, kme, 1), &
           qv(ncol, kme, 1), qc(ncol, kme, 1), qr(ncol, kme, 1))
  allocate(rainnc0(ncol, 1), rainnc(ncol, 1), rainncv(ncol, 1))
  t0 = 0.; qv0 = 0.; qc0 = 0.; qr0 = 0.; rho = 1.; pii = 1.
  z = 0.; dz8w = 1.
  read(uin) t0(:, 1:nz, 1)
  read(uin) qv0(:, 1:nz, 1)
  read(uin) qc0(:, 1:nz, 1)
  read(uin) qr0(:, 1:nz, 1)
  read(uin) rho(:, 1:nz, 1)
  read(uin) pii(:, 1:nz, 1)
  read(uin) z(:, 1:nz, 1)
  read(uin) dz8w(:, 1:nz, 1)
  read(uin) rainnc0(:, 1)
  close(uin)

  open(newunit=uout, file=trim(fout), access='stream', form='unformatted', &
       status='replace', action='write')
  do ic = 1, ncase
    t = t0; qv = qv0; qc = qc0; qr = qr0
    rainnc = rainnc0; rainncv = 0.
    do istep = 1, nsteps(ic)
      call kessler(t=t, qv=qv, qc=qc, qr=qr, rho=rho, pii=pii,          &
                   dt_in=dts(ic), z=z, xlv=xlv, cp=cp,                   &
                   ep2=ep_2, svp1=svp1, svp2=svp2, svp3=svp3,            &
                   svpt0=svpt0, rhowater=rhowater, dz8w=dz8w,            &
                   rainnc=rainnc, rainncv=rainncv,                       &
                   ids=1, ide=ncol + 1, jds=1, jde=2, kds=1, kde=kme,    &
                   ims=1, ime=ncol, jms=1, jme=1, kms=1, kme=kme,        &
                   its=1, ite=ncol, jts=1, jte=1, kts=1, kte=nz)
      write(uout) t(:, 1:nz, 1)
      write(uout) qv(:, 1:nz, 1)
      write(uout) qc(:, 1:nz, 1)
      write(uout) qr(:, 1:nz, 1)
      write(uout) rainnc(:, 1)
      write(uout) rainncv(:, 1)
    end do
  end do
  close(uout)
end program run_kessler_oracle
