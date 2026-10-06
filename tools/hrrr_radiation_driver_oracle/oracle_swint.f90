! swint_opt = 1 oracle: the fork's update_swinterp_parameters,
! interp_sw_radiation and calc_coszen/radconst driven over a column set.
!
! swint_in.bin (stream, little-endian):
!   int32 ncol, ncall, nloc
!   real32 coszen(ncol, ncall)        radiation-step coszen per call
!   real32 czcall(ncol, ncall)        coszen_loc at the radiation step
!   real32 swddir(ncol, ncall)        the scheme's SWDDIR per call
!   real32 swdown(ncol, ncall)        the scheme's SWDOWN per call
!   real32 czloc(ncol, nloc, ncall)   coszen_loc of the steps after each call
!   real32 albedo(ncol)
! swint_out.bin: per call, after update_swinterp_parameters:
!   bb, bx, gg, gx, coszen_ref, swdown_ref, swddir_ref (ncol each), then the
!   radiation-step interp (swdown, swddir, swddni, swddif, gsw), then nloc
!   between-call interps of the same five arrays.
! coszen_in.bin: int32 n, nset; real32 xlat(n), xlon(n); then per set
!   real32 julian, xtime, gmt
! coszen_out.bin: per set real32 declin, solcon, coszen(n), hrang(n)
program oracle_swint
  use hrrr_radiation_driver_oracle
  implicit none
  integer :: ncol, ncall, nloc, ic, il, n, nset, is
  real, allocatable :: coszen(:,:,:), czcall(:,:,:), swddir_in(:,:,:), swdown_in(:,:,:)
  real, allocatable :: czloc(:,:,:,:), albedo(:,:)
  real, allocatable :: bb(:,:), bx(:,:), gg(:,:), gx(:,:)
  real, allocatable :: coszen_ref(:,:), swdown_ref(:,:), swddir_ref(:,:)
  real, allocatable :: swdown(:,:), swddir(:,:), swddni(:,:), swddif(:,:), gsw(:,:)
  real, allocatable :: xlat(:,:), xlon(:,:), cz(:,:), hr(:,:)
  real :: julian, xtime, gmt, declin, solcon
  real, parameter :: degrad = 3.1415926535897932384626433 / 180.
  real, parameter :: dpd = 360. / 365.

  open(10, file='swint_in.bin', access='stream', form='unformatted', status='old')
  read(10) ncol, ncall, nloc
  allocate(coszen(ncol,1,ncall), czcall(ncol,1,ncall), swddir_in(ncol,1,ncall), &
           swdown_in(ncol,1,ncall), czloc(ncol,1,nloc,ncall), albedo(ncol,1))
  read(10) coszen, czcall, swddir_in, swdown_in, czloc, albedo
  close(10)
  allocate(bb(ncol,1), bx(ncol,1), gg(ncol,1), gx(ncol,1), coszen_ref(ncol,1), &
           swdown_ref(ncol,1), swddir_ref(ncol,1), swdown(ncol,1), swddir(ncol,1), &
           swddni(ncol,1), swddif(ncol,1), gsw(ncol,1))
  bb = 0.; bx = 0.; gg = 0.; gx = 0.
  coszen_ref = 0.; swdown_ref = 0.; swddir_ref = 0.
  swdown = 0.; swddir = 0.; swddni = 0.; swddif = 0.; gsw = 0.
  open(11, file='swint_out.bin', access='stream', form='unformatted', status='replace')
  do ic = 1, ncall
     swddir = swddir_in(:,:,ic)
     swdown = swdown_in(:,:,ic)
     call update_swinterp_parameters(1, ncol, 1, 1, 1, ncol, 1, 1, &
          coszen(:,:,ic), czcall(:,:,ic), swddir, swdown, &
          swddir_ref, bb, bx, swdown_ref, gg, gx, coszen_ref)
     write(11) bb, bx, gg, gx, coszen_ref, swdown_ref, swddir_ref
     call interp_sw_radiation(1, ncol, 1, 1, 1, ncol, 1, 1, &
          coszen_ref, czcall(:,:,ic), swddir_ref, bb, bx, swdown_ref, gg, gx, albedo, &
          swdown, swddir, swddni, swddif, gsw)
     write(11) swdown, swddir, swddni, swddif, gsw
     do il = 1, nloc
        call interp_sw_radiation(1, ncol, 1, 1, 1, ncol, 1, 1, &
             coszen_ref, czloc(:,:,il,ic), swddir_ref, bb, bx, swdown_ref, gg, gx, albedo, &
             swdown, swddir, swddni, swddif, gsw)
        write(11) swdown, swddir, swddni, swddif, gsw
     end do
  end do
  close(11)

  open(12, file='coszen_in.bin', access='stream', form='unformatted', status='old')
  read(12) n, nset
  allocate(xlat(n,1), xlon(n,1), cz(n,1), hr(n,1))
  read(12) xlat, xlon
  open(13, file='coszen_out.bin', access='stream', form='unformatted', status='replace')
  do is = 1, nset
     read(12) julian, xtime, gmt
     call radconst(xtime, declin, solcon, julian, degrad, dpd)
     cz = 0.; hr = 0.
     call calc_coszen(1, n, 1, 1, 1, n, 1, 1, julian, xtime, gmt, declin, degrad, xlon, xlat, cz, hr)
     write(13) declin, solcon, cz, hr
  end do
  close(12)
  close(13)
end program oracle_swint
