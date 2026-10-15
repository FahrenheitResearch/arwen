! Column runner for the Milbrandt-Yau (mp_physics=9) WRF v4.6.1 oracle.
!
! Calls the public WRF wrapper mp_milbrandt2mom_driver from the pinned,
! byte-unmodified phys/module_mp_milbrandt2mom.F, on one j-row of NCOL
! columns, NSTEPS times in a row (each call starts from the previous call's
! output, as in a model run), and writes every INOUT field after every call.
!
! Input  (stream, little-endian, see gen_columns.py):
!   int32 ncol, nz, nsteps; real32 dt;
!   18 (ncol,nz) fields: qv qc qr qi qs qg qh nc nr ni ns ng nh th pii p w dz
!   5 (ncol) fields: p8w(k=1) rainnc snownc grplnc hailnc
! Output (stream): for each step, in this order,
!   15 (ncol,nz) fields: qv qc qr qi qs qg qh nc nr ni ns ng nh th zet
!   9 (ncol) fields: rainnc rainncv snownc snowncv grplnc grplncv
!                    hailnc hailncv sr
program run_my2
  use module_mp_milbrandt2mom, only: mp_milbrandt2mom_driver
  implicit none
  integer :: ncol, nz, nsteps, step, k
  real :: dt
  real, allocatable, dimension(:,:,:) :: qv, qc, qr, qi, qs, qg, qh, &
       nc, nr, ni, ns, ng, nh, th, pii, p, w, dz, p8w, zet
  real, allocatable, dimension(:,:) :: rainnc, rainncv, snownc, snowncv, &
       grplnc, grplncv, hailnc, hailncv, sr
  real, allocatable :: buf(:,:), sbuf(:)
  character(len=512) :: fin, fout
  integer :: u, v

  if (kind(1.0) /= 4) stop 'default REAL must be 4 bytes'
  call get_command_argument(1, fin)
  call get_command_argument(2, fout)
  open(newunit=u, file=trim(fin), access='stream', form='unformatted', &
       status='old', convert='little_endian')
  read(u) ncol, nz, nsteps
  read(u) dt

  ! Memory dims ims:ime = 1:ncol, kms:kme = 1:nz+1 (staggered top, as WRF),
  ! jms:jme = 1:1.  Tile kts:kte = 1:nz (mass levels).
  allocate(qv(ncol,nz+1,1), qc(ncol,nz+1,1), qr(ncol,nz+1,1), qi(ncol,nz+1,1), &
       qs(ncol,nz+1,1), qg(ncol,nz+1,1), qh(ncol,nz+1,1), nc(ncol,nz+1,1), &
       nr(ncol,nz+1,1), ni(ncol,nz+1,1), ns(ncol,nz+1,1), ng(ncol,nz+1,1), &
       nh(ncol,nz+1,1), th(ncol,nz+1,1), pii(ncol,nz+1,1), p(ncol,nz+1,1), &
       w(ncol,nz+1,1), dz(ncol,nz+1,1), p8w(ncol,nz+1,1), zet(ncol,nz+1,1))
  allocate(rainnc(ncol,1), rainncv(ncol,1), snownc(ncol,1), snowncv(ncol,1), &
       grplnc(ncol,1), grplncv(ncol,1), hailnc(ncol,1), hailncv(ncol,1), sr(ncol,1))
  allocate(buf(ncol,nz), sbuf(ncol))

  ! Fill the unused staggered top with a signalling value; the wrapper
  ! never reads it.
  call rd(qv); call rd(qc); call rd(qr); call rd(qi); call rd(qs); call rd(qg)
  call rd(qh); call rd(nc); call rd(nr); call rd(ni); call rd(ns); call rd(ng)
  call rd(nh); call rd(th); call rd(pii); call rd(p); call rd(w); call rd(dz)
  p8w = -1.0e30
  read(u) sbuf; p8w(:,1,1) = sbuf
  read(u) sbuf; rainnc(:,1) = sbuf
  read(u) sbuf; snownc(:,1) = sbuf
  read(u) sbuf; grplnc(:,1) = sbuf
  read(u) sbuf; hailnc(:,1) = sbuf
  close(u)
  zet = 0.0
  rainncv = 0.0; snowncv = 0.0; grplncv = 0.0; hailncv = 0.0; sr = 0.0

  open(newunit=v, file=trim(fout), access='stream', form='unformatted', &
       status='replace', convert='little_endian')
  do step = 1, nsteps
     call mp_milbrandt2mom_driver(qv, qc, qr, qi, qs, qg, qh, nc, nr, ni, ns, ng, &
          nh, th, pii, p, w, dz, dt, step, p8w,                               &
          rainnc, rainncv, snownc, snowncv, grplnc, grplncv,                  &
          hailnc, hailncv, sr, zet,                                           &
          1, ncol, 1, 1, 1, nz+1,                                             &
          1, ncol, 1, 1, 1, nz+1,                                             &
          1, ncol, 1, 1, 1, nz)
     call wr(qv); call wr(qc); call wr(qr); call wr(qi); call wr(qs); call wr(qg)
     call wr(qh); call wr(nc); call wr(nr); call wr(ni); call wr(ns); call wr(ng)
     call wr(nh); call wr(th); call wr(zet)
     write(v) rainnc(:,1), rainncv(:,1), snownc(:,1), snowncv(:,1), &
          grplnc(:,1), grplncv(:,1), hailnc(:,1), hailncv(:,1), sr(:,1)
  end do
  close(v)

contains

  subroutine rd(a)
    real, intent(inout) :: a(:,:,:)
    read(u) buf
    a = -1.0e30
    a(:,1:nz,1) = buf
  end subroutine rd

  subroutine wr(a)
    real, intent(in) :: a(:,:,:)
    buf = a(:,1:nz,1)
    write(v) buf
  end subroutine wr

end program run_my2
