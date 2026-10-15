program run_swrad_oracle
  ! Drive SUBROUTINE SWRAD (and swinit for CSSCA) from the byte-unmodified
  ! WRF v4.6.1 phys/module_ra_sw.F.  Every RTHRATEN and GSW word written
  ! here was returned by WRF's routine; this program reads inputs and copies
  ! arrays.  The one line of arithmetic it does is the radiation driver's
  ! own SWDOWN for sw_physics = 1, module_radiation_driver.F:2877:
  !     SWDOWN(I,J)=GSW(I,J)/(1.-ALBEDO(I,J))
  ! Constants are WRF's own module_model_constants parameters, passed as
  ! module_radiation_driver.F:2343-2369 passes them (R=r_d, CP=cp, G=g).
  ! RTHRATEN enters zeroed, so on return it holds SWRAD's TTEN/pi exactly.
  ! Every species flag is .TRUE. (an ice-bearing scheme); no WRF-Chem
  ! aerosol (pm2_5 absent), no eclipse (obscur = 0).
  !
  ! usage: run_swrad INPUTS.bin OUTPUTS.bin
  ! For each case: swinit(swrad_scat), one SWRAD call over every column,
  ! then write rthraten (ncol x nz, i fastest), gsw (ncol), swdown (ncol).
  use module_model_constants, only: r_d, cp, g
  use module_ra_sw, only: swrad, swinit
  implicit none

  integer :: ncol, nz, ncase, kme, ic, uin, uout
  real, allocatable :: solcon(:), scat(:)
  integer, allocatable :: icloud(:)
  real, allocatable, dimension(:, :, :) :: t, p, qv, qc, qr, qi, qs, qg, &
                                           dz8w, pii, rho, rthraten
  real, allocatable, dimension(:, :) :: coszen, albedo, xlat, xlong, gsw, &
                                        swdown, obscur
  character(len=1024) :: fin, fout

  call get_command_argument(1, fin)
  call get_command_argument(2, fout)
  if (len_trim(fin) == 0 .or. len_trim(fout) == 0) then
    write(*, '(A)') 'usage: run_swrad INPUTS.bin OUTPUTS.bin'
    error stop 2
  end if
  if (storage_size(1.0) /= 32) error stop 3

  open(newunit=uin, file=trim(fin), access='stream', form='unformatted', &
       status='old', action='read')
  read(uin) ncol, nz, ncase
  allocate(solcon(ncase), scat(ncase), icloud(ncase))
  read(uin) solcon
  read(uin) scat
  read(uin) icloud
  ! WRF memory dims: kme = kde = the staggered top (nz + 1); the scheme's
  ! tile is kts..kte = 1..nz half levels.  SWRAD reverses with
  ! NK = kme-1-K+kms, so level nz+1 is never read.
  kme = nz + 1
  allocate(t(ncol, kme, 1), p(ncol, kme, 1), qv(ncol, kme, 1), &
           qc(ncol, kme, 1), qr(ncol, kme, 1), qi(ncol, kme, 1), &
           qs(ncol, kme, 1), qg(ncol, kme, 1), dz8w(ncol, kme, 1), &
           pii(ncol, kme, 1), rho(ncol, kme, 1), rthraten(ncol, kme, 1))
  allocate(coszen(ncol, 1), albedo(ncol, 1), xlat(ncol, 1), &
           xlong(ncol, 1), gsw(ncol, 1), swdown(ncol, 1), obscur(ncol, 1))
  t = 250.; p = 50000.; qv = 0.; qc = 0.; qr = 0.; qi = 0.; qs = 0.
  qg = 0.; dz8w = 1.; pii = 1.; rho = 1.
  read(uin) t(:, 1:nz, 1)
  read(uin) p(:, 1:nz, 1)
  read(uin) qv(:, 1:nz, 1)
  read(uin) qc(:, 1:nz, 1)
  read(uin) qr(:, 1:nz, 1)
  read(uin) qi(:, 1:nz, 1)
  read(uin) qs(:, 1:nz, 1)
  read(uin) qg(:, 1:nz, 1)
  read(uin) dz8w(:, 1:nz, 1)
  read(uin) pii(:, 1:nz, 1)
  read(uin) rho(:, 1:nz, 1)
  read(uin) coszen(:, 1)
  read(uin) albedo(:, 1)
  close(uin)
  xlat = 0.; xlong = 0.; obscur = 0.

  open(newunit=uout, file=trim(fout), access='stream', form='unformatted', &
       status='replace', action='write')
  do ic = 1, ncase
    call swinit(scat(ic), .true.,                                       &
                1, ncol + 1, 1, 2, 1, kme,                              &
                1, ncol, 1, 1, 1, kme,                                  &
                1, ncol, 1, 1, 1, nz)
    rthraten = 0.
    gsw = -999.
    call swrad(dt=60., rthraten=rthraten, gsw=gsw, xlat=xlat,           &
               xlong=xlong, albedo=albedo, rho_phy=rho, t3d=t,          &
               qv3d=qv, qc3d=qc, qr3d=qr, qi3d=qi, qs3d=qs, qg3d=qg,    &
               p3d=p, pi3d=pii, dz8w=dz8w, gmt=0., r=r_d, cp=cp, g=g,   &
               julday=1, ghg_input=0, xtime=0., declin=0.,              &
               solcon=solcon(ic),                                       &
               f_qv=.true., f_qc=.true., f_qr=.true., f_qi=.true.,      &
               f_qs=.true., f_qg=.true.,                                &
               radfrq=30., icloud=icloud(ic), degrad=0.017453293,       &
               warm_rain=.false.,                                       &
               ids=1, ide=ncol + 1, jds=1, jde=2, kds=1, kde=kme,       &
               ims=1, ime=ncol, jms=1, jme=1, kms=1, kme=kme,           &
               its=1, ite=ncol, jts=1, jte=1, kts=1, kte=nz,            &
               coszen=coszen, julian=0., obscur=obscur)
    ! module_radiation_driver.F:2877, the driver's SWDOWN for Dudhia.
    swdown(:, 1) = gsw(:, 1) / (1. - albedo(:, 1))
    write(uout) rthraten(:, 1:nz, 1)
    write(uout) gsw(:, 1)
    write(uout) swdown(:, 1)
  end do
  close(uout)
end program run_swrad_oracle
