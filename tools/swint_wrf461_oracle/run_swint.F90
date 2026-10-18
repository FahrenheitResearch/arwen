program run_swint_oracle
  ! Drive WRF v4.6.1's swint_opt = 1 routines (radconst, calc_coszen,
  ! update_swinterp_parameters, interp_sw_radiation; text cut unchanged by
  ! extract.py) through a sequence of model steps, in the order
  ! module_radiation_driver.F runs them:
  !
  !   every step   radconst(XTIME) + calc_coszen(xtime)          :1158-1170
  !   rad. step    radconst(XTIME) + calc_coszen(xtime+radt*0.5) :1188-1206
  !                (the scheme's SWDOWN/SWDDIR are inputs here)
  !                update_swinterp_parameters                     :2932-2938
  !   every step   interp_sw_radiation with the live ALBEDO       :2964-2980
  !
  ! Bx, bb, Gx, gg and the three *_ref arrays start at zero, as WRF's
  ! itimestep = 1 block (:1708-1717) and the Registry misc state leave them.
  ! DEGRAD and DPD are WRF's own module_model_constants parameters, the
  ! values the driver passes.  This program does no arithmetic of its own
  ! beyond xtime+radt*0.5, which is the driver's own argument expression.
  !
  ! usage: run_swint IN.bin OUT.bin
  ! IN (stream, native little-endian):
  !   int32 ncol, nstep, ncall
  !   real32 radt                       minutes
  !   real32 xlat(ncol), xlon(ncol)
  !   real32 julian(nstep), xtime(nstep), gmt(nstep)
  !   int32  radflag(nstep)             1 on a radiation step
  !   real32 swdown(ncol,ncall), swddir(ncol,ncall)
  !   real32 albedo(ncol,nstep)         the live ALBEDO each step
  ! OUT: per step, ncol words each, in this order:
  !   coszen_loc, coszen (radiation-step cosine, last value held),
  !   bb, bx, gg, gx, coszen_ref, swdown_ref, swddir_ref,
  !   swdown, swddir, swddni, swddif, gsw
  use module_model_constants, only: degrad, dpd
  use swint_wrf461
  implicit none
  integer :: ncol, nstep, ncall, s, icall
  real :: radt, declin, solcon
  real, allocatable :: xlat(:,:), xlon(:,:), julian(:), xtime(:), gmt(:)
  integer, allocatable :: radflag(:)
  real, allocatable :: swdown_in(:,:), swddir_in(:,:), albedo_in(:,:)
  real, allocatable, dimension(:,:) :: coszen_loc, hrang_loc, coszen, hrang, &
       bb, bx, gg, gx, coszen_ref, swdown_ref, swddir_ref, &
       swdown, swddir, swddni, swddif, gsw, albedo
  character(len=1024) :: fin, fout

  call get_command_argument(1, fin)
  call get_command_argument(2, fout)
  open(10, file=trim(fin), access='stream', form='unformatted', status='old')
  read(10) ncol, nstep, ncall
  read(10) radt
  allocate(xlat(ncol,1), xlon(ncol,1), julian(nstep), xtime(nstep), gmt(nstep), &
           radflag(nstep), swdown_in(ncol,ncall), swddir_in(ncol,ncall), &
           albedo_in(ncol,nstep))
  read(10) xlat, xlon
  read(10) julian, xtime, gmt
  read(10) radflag
  read(10) swdown_in, swddir_in
  read(10) albedo_in
  close(10)

  allocate(coszen_loc(ncol,1), hrang_loc(ncol,1), coszen(ncol,1), hrang(ncol,1), &
           bb(ncol,1), bx(ncol,1), gg(ncol,1), gx(ncol,1), coszen_ref(ncol,1), &
           swdown_ref(ncol,1), swddir_ref(ncol,1), swdown(ncol,1), swddir(ncol,1), &
           swddni(ncol,1), swddif(ncol,1), gsw(ncol,1), albedo(ncol,1))
  coszen_loc = 0.; hrang_loc = 0.; coszen = 0.; hrang = 0.
  bb = 0.; bx = 0.; gg = 0.; gx = 0.
  coszen_ref = 0.; swdown_ref = 0.; swddir_ref = 0.
  swdown = 0.; swddir = 0.; swddni = 0.; swddif = 0.; gsw = 0.

  open(11, file=trim(fout), access='stream', form='unformatted', status='replace')
  icall = 0
  do s = 1, nstep
     call radconst(xtime(s), declin, solcon, julian(s), degrad, dpd)
     call calc_coszen(1, ncol, 1, 1, 1, ncol, 1, 1, julian(s), xtime(s), gmt(s), &
                      declin, degrad, xlon, xlat, coszen_loc, hrang_loc)
     if (radflag(s) == 1) then
        icall = icall + 1
        call radconst(xtime(s), declin, solcon, julian(s), degrad, dpd)
        call calc_coszen(1, ncol, 1, 1, 1, ncol, 1, 1, julian(s), xtime(s)+radt*0.5, &
                         gmt(s), declin, degrad, xlon, xlat, coszen, hrang)
        swdown(:,1) = swdown_in(:,icall)
        swddir(:,1) = swddir_in(:,icall)
        call update_swinterp_parameters(1, ncol, 1, 1, 1, ncol, 1, 1, &
             coszen, coszen_loc, swddir, swdown, &
             swddir_ref, bb, bx, swdown_ref, gg, gx, coszen_ref)
     end if
     albedo(:,1) = albedo_in(:,s)
     call interp_sw_radiation(1, ncol, 1, 1, 1, ncol, 1, 1, &
          coszen_ref, coszen_loc, swddir_ref, bb, bx, swdown_ref, gg, gx, albedo, &
          swdown, swddir, swddni, swddif, gsw)
     write(11) coszen_loc, coszen, bb, bx, gg, gx, coszen_ref, swdown_ref, &
               swddir_ref, swdown, swddir, swddni, swddif, gsw
  end do
  close(11)
  if (icall /= ncall) then
     print *, 'radiation calls', icall, '/= ncall', ncall
     stop 3
  end if
end program run_swint_oracle
