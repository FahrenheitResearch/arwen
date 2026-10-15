program remap_probe
  ! Lane 073's transport-only remap control, run through WSM5's own
  ! nislfv_rain_plm (module_mp_wsm5, whichever arm build.sh links): uniform
  ! condensate density 0.001 kg m-3 (den 1 kg m-3, so q = 1 g/kg), 100 m
  ! cells, a uniform 1 m/s fall speed, no velocity iteration (iter = 0) and
  ! no microphysical source or sink.  The final column plus the surface
  ! fallout must equal the initial inventory.
  !
  ! usage: remap_probe OUT.BIN
  ! OUT.BIN, little-endian, per case (nz in 2, 50, 80 x dt in 0.001, 1,
  ! 20): int32 nz; float32 dt; float32 rq(nz) after the call; float32
  ! precip.
  use module_mp_wsm5, only: nislfv_rain_plm
  implicit none
  integer, parameter :: nlev(3) = (/ 2, 50, 80 /)
  real, parameter :: dts(3) = (/ 0.001, 1.0, 20.0 /)
  integer :: a, b, nz, u
  character(len=1024) :: outpath
  real, allocatable :: den(:,:), denfac(:,:), tk(:,:), dz(:,:), ww(:,:), rq(:,:)
  real :: precip(1)

  call get_command_argument(1, outpath)
  open(newunit=u, file=trim(outpath), access='stream', form='unformatted', status='replace')
  do a = 1, 3
    nz = nlev(a)
    allocate(den(1,nz), denfac(1,nz), tk(1,nz), dz(1,nz), ww(1,nz), rq(1,nz))
    do b = 1, 3
      den = 1.0; denfac = 1.0; tk = 280.0; dz = 100.0; ww = 1.0; rq = 0.001
      call nislfv_rain_plm(1, nz, den, denfac, tk, dz, ww, rq, precip, dts(b), 1, 0)
      write(u) nz, dts(b), rq(1,:), precip(1)
    enddo
    deallocate(den, denfac, tk, dz, ww, rq)
  enddo
  close(u)
end program remap_probe
