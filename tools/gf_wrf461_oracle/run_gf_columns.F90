program run_gf_columns
  ! Drive the byte-unmodified WRF v4.6.1 Grell-Freitas driver GFDRV over an
  ! arbitrary list of columns read from a file, one GFDRV call per column.
  !
  ! run_cu_gf.F90 beside this program drives the 18 branch-targeted cases
  ! of gf_cases.F90.  This one drives whatever columns gf_regime_columns.py
  ! wrote -- realistic soundings across regimes, a clos_choice sweep and the
  ! edge cases -- so the case table lives in Python where it is easy to
  ! read, and the Fortran side stays a pure pass-through to WRF's own entry
  ! point.  Nothing here computes physics.
  !
  ! Input (stream, little-endian, written by gf_regime_columns.py):
  !   int32 magic = 47011, ncol, nz
  !   float32 sc(6, ncol)          dt, dx, ht, hfx, qfx, xland
  !   int32   ii(3, ncol)          kpbl, ishallow, ichoice
  !   float32 lev(nz, 14, ncol)    u v w t qv p pi rho dz8w rthften rqvften
  !                                rthraten rthblten rqvblten (mass levels)
  !   float32 p8w(nz + 1, ncol)    interface pressure
  ! Output (stream):
  !   int32 magic = 47012, ncol, nz, kte_mode
  !   float32 lo(nz, 8, ncol)      RTHCUTEN RQVCUTEN RQCCUTEN RQICUTEN
  !                                DUDT_PHY DVDT_PHY GDC GDC2
  !   float32 so(5, ncol)          RAINCV PRATEC HTOP HBOT xmb_shallow
  !   int32   io(4, ncol)          ktop_deep k22_shallow kbcon_shallow
  !                                ktop_shallow
  !
  ! kte_mode selects the vertical tile convention:
  !   0  kte = nz, kde = nz + 1 -- what WRF itself passes: the solver calls
  !      cumulus_driver with KTE=min(k_end,kde-1)
  !      (dyn_em/module_first_rk_step_part1.F), and what the kernel's nz
  !      means.  THIS is the oracle of record.
  !   1  kte = kde = nz + 1 -- a probe, not WRF's convention: GFDRV's local
  !      arrays carry one level above ktf = nz, and the mass-level inputs at
  !      nz + 1 are a quiet NaN, so a read of that level would show in the
  !      answer.  The only expected difference from mode 0 is HBOT's
  !      initial value, REAL(KTE) (module_cu_gf_wrfdrv.F:349).
  use module_cu_gf_wrfdrv, only: gfdrv
  implicit none

  integer, parameter :: nlev_in = 14
  integer :: ncol, nz, magic, kte_mode, ios
  real, allocatable :: sc(:, :), lev(:, :, :), p8w_in(:, :)
  integer, allocatable :: ii(:, :)
  real, allocatable :: lo(:, :, :), so(:, :)
  integer, allocatable :: io(:, :)
  character(len=1024) :: in_path, out_path, mode_arg, only_arg
  integer :: uin, uout, ic, only

  call get_command_argument(1, in_path)
  call get_command_argument(2, out_path)
  call get_command_argument(3, mode_arg)
  if (len_trim(in_path) == 0 .or. len_trim(out_path) == 0 .or. &
      len_trim(mode_arg) == 0) then
    write(*, '(A)') 'usage: run_gf_columns INPUT.bin OUTPUT.bin KTE_MODE [COLUMN]'
    error stop 2
  end if
  read(mode_arg, *) kte_mode
  if (kte_mode /= 0 .and. kte_mode /= 1) error stop 3
  ! Optional 4th argument: run only this (1-based) column, leaving the other
  ! outputs zero.  build_regime.sh uses it to localise a bounds-check abort.
  only = 0
  call get_command_argument(4, only_arg)
  if (len_trim(only_arg) > 0) read(only_arg, *) only

  open(newunit=uin, file=trim(in_path), access='stream', form='unformatted', &
       status='old', action='read')
  read(uin) magic, ncol, nz
  if (magic /= 47011) error stop 4
  allocate(sc(6, ncol), ii(3, ncol), lev(nz, nlev_in, ncol), p8w_in(nz + 1, ncol))
  read(uin, iostat=ios) sc, ii, lev, p8w_in
  if (ios /= 0) error stop 5
  close(uin)

  allocate(lo(nz, 8, ncol), so(5, ncol), io(4, ncol))
  lo = 0.0
  so = 0.0
  io = 0
  do ic = 1, ncol
    if (only == 0 .or. only == ic) call one_column(ic)
  end do

  open(newunit=uout, file=trim(out_path), access='stream', form='unformatted', &
       status='replace', action='write')
  write(uout) 47012, ncol, nz, kte_mode
  write(uout) lo, so, io
  close(uout)
  write(*, '(A,I0,A,I0)') 'gf regime columns written: ', ncol, ' kte_mode ', kte_mode

contains

  subroutine one_column(ic)
    integer, intent(in) :: ic
    ! GFDRV's write window is its..ite inset four cells from ids and five
    ! from ide (module_cu_gf_wrfdrv.F:264-277); one active column sits at
    ! i = j = 5 of a 10 x 10 domain, the geometry run_cu_gf.F90 uses.
    integer, parameter :: ids = 1, ide = 10, jds = 1, jde = 10
    integer, parameter :: its = 5, ite = 5, jts = 5, jte = 5
    integer :: kds, kde, kts, kte, kms, kme
    real, allocatable, dimension(:, :, :) :: u3, v3, w3, t3, q3, p3, pi3
    real, allocatable, dimension(:, :, :) :: rho3, dz3, p8w3
    real, allocatable, dimension(:, :, :) :: thf, qvf, thra, thbl, qvbl
    real, allocatable, dimension(:, :, :) :: thcu, qvcu, qccu, qicu
    real, allocatable, dimension(:, :, :) :: dudt, dvdt, gdc, gdc2, pat, fld
    real, dimension(ids:ide, jds:jde) :: raincv, pratec, htop, hbot
    real, dimension(ids:ide, jds:jde) :: ht2, hfx2, qfx2, xland2, xmbsh
    integer, dimension(ids:ide, jds:jde) :: kpbl2, ktopd, k22s, kbcs, ktps
    real :: dtstep, dxv, qnan
    integer :: k, ichoice, ishallow, spp

    kds = 1
    kts = 1
    if (kte_mode == 0) then
      kte = nz
      kde = nz + 1
    else
      kte = nz + 1
      kde = nz + 1
    end if
    kms = kds
    kme = kde
    qnan = transfer(int(z'7FC00000'), 0.0)

    allocate(u3(ids:ide, kms:kme, jds:jde))
    ! mold= gives the bounds; every array is filled explicitly below.
    allocate(v3, w3, t3, q3, p3, pi3, rho3, dz3, p8w3, thf, qvf, thra,    &
             thbl, qvbl, thcu, qvcu, qccu, qicu, dudt, dvdt, gdc, gdc2,   &
             pat, fld, mold=u3)
    u3 = 0.0; v3 = 0.0; w3 = 0.0; t3 = 0.0; q3 = 0.0; p3 = 0.0; pi3 = 1.0
    rho3 = 0.0; dz3 = 0.0; p8w3 = 0.0
    thf = 0.0; qvf = 0.0; thra = 0.0; thbl = 0.0; qvbl = 0.0
    thcu = 0.0; qvcu = 0.0; qccu = 0.0; qicu = 0.0
    dudt = 0.0; dvdt = 0.0; gdc = 0.0; gdc2 = 0.0; pat = 0.0; fld = 0.0
    raincv = 0.0; pratec = 0.0; htop = 0.0; hbot = 0.0
    ht2 = 0.0; hfx2 = 0.0; qfx2 = 0.0; xland2 = 1.0; xmbsh = 0.0
    kpbl2 = 1; ktopd = 0; k22s = 0; kbcs = 0; ktps = 0

    ! Fill every (i, j) of the domain with the column, so nothing GFDRV can
    ! touch outside the write window is a zero-pressure column.
    do k = 1, nz
      u3(:, k, :) = lev(k, 1, ic)
      v3(:, k, :) = lev(k, 2, ic)
      w3(:, k, :) = lev(k, 3, ic)
      t3(:, k, :) = lev(k, 4, ic)
      q3(:, k, :) = lev(k, 5, ic)
      p3(:, k, :) = lev(k, 6, ic)
      pi3(:, k, :) = lev(k, 7, ic)
      rho3(:, k, :) = lev(k, 8, ic)
      dz3(:, k, :) = lev(k, 9, ic)
      thf(:, k, :) = lev(k, 10, ic)
      qvf(:, k, :) = lev(k, 11, ic)
      thra(:, k, :) = lev(k, 12, ic)
      thbl(:, k, :) = lev(k, 13, ic)
      qvbl(:, k, :) = lev(k, 14, ic)
    end do
    do k = 1, nz + 1
      p8w3(:, k, :) = p8w_in(k, ic)
    end do
    if (kte_mode == 1) then
      ! Level nz + 1 of the mass-level inputs: WRF never defines it for
      ! GFDRV.  A NaN here makes any read visible.
      u3(:, nz + 1, :) = qnan; v3(:, nz + 1, :) = qnan
      w3(:, nz + 1, :) = qnan; t3(:, nz + 1, :) = qnan
      q3(:, nz + 1, :) = qnan; p3(:, nz + 1, :) = qnan
      pi3(:, nz + 1, :) = qnan; rho3(:, nz + 1, :) = qnan
      dz3(:, nz + 1, :) = qnan
      thf(:, nz + 1, :) = qnan; qvf(:, nz + 1, :) = qnan
      thra(:, nz + 1, :) = qnan; thbl(:, nz + 1, :) = qnan
      qvbl(:, nz + 1, :) = qnan
    end if
    dtstep = sc(1, ic)
    dxv = sc(2, ic)
    ht2 = sc(3, ic)
    hfx2 = sc(4, ic)
    qfx2 = sc(5, ic)
    xland2 = sc(6, ic)
    kpbl2 = ii(1, ic)
    ishallow = ii(2, ic)
    ichoice = ii(3, ic)
    spp = 0

    call gfdrv(spp, pat, fld,                                               &
               DT=dtstep, DX=dxv,                                           &
               rho=rho3, RAINCV=raincv, PRATEC=pratec,                      &
               U=u3, V=v3, t=t3, W=w3, q=q3, p=p3, pi=pi3,                  &
               dz8w=dz3, p8w=p8w3,                                          &
               htop=htop, hbot=hbot, ktop_deep=ktopd,                       &
               HT=ht2, hfx=hfx2, qfx=qfx2, XLAND=xland2,                    &
               GDC=gdc, GDC2=gdc2, kpbl=kpbl2,                              &
               k22_shallow=k22s, kbcon_shallow=kbcs,                        &
               ktop_shallow=ktps, xmb_shallow=xmbsh,                        &
               ichoice=ichoice, ishallow_g3=ishallow,                       &
               ids=ids, ide=ide, jds=jds, jde=jde, kds=kds, kde=kde,        &
               ims=ids, ime=ide, jms=jds, jme=jde, kms=kms, kme=kme,        &
               its=its, ite=ite, jts=jts, jte=jte, kts=kts, kte=kte,        &
               periodic_x=.false., periodic_y=.false.,                      &
               RQVCUTEN=qvcu, RQCCUTEN=qccu, RQICUTEN=qicu,                 &
               RQVFTEN=qvf, RTHFTEN=thf,                                    &
               RTHCUTEN=thcu, RTHRATEN=thra,                                &
               rqvblten=qvbl, rthblten=thbl,                                &
               dudt_phy=dudt, dvdt_phy=dvdt)

    do k = 1, nz
      lo(k, 1, ic) = thcu(its, k, jts)
      lo(k, 2, ic) = qvcu(its, k, jts)
      lo(k, 3, ic) = qccu(its, k, jts)
      lo(k, 4, ic) = qicu(its, k, jts)
      lo(k, 5, ic) = dudt(its, k, jts)
      lo(k, 6, ic) = dvdt(its, k, jts)
      lo(k, 7, ic) = gdc(its, k, jts)
      lo(k, 8, ic) = gdc2(its, k, jts)
    end do
    so(1, ic) = raincv(its, jts)
    so(2, ic) = pratec(its, jts)
    so(3, ic) = htop(its, jts)
    so(4, ic) = hbot(its, jts)
    so(5, ic) = xmbsh(its, jts)
    io(1, ic) = ktopd(its, jts)
    io(2, ic) = k22s(its, jts)
    io(3, ic) = kbcs(its, jts)
    io(4, ic) = ktps(its, jts)
  end subroutine one_column

end program run_gf_columns
