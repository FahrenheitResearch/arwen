program run_shinhong_columns
  ! Drive the byte-unmodified WRF v4.6.1 phys/module_bl_shinhong.F through its
  ! own 3-D entry `shinhong`, one column per call, over every record of the
  ! binary column file make_columns.py writes, and dump every output as raw
  ! float32 (no text round trip, so NaN payloads and signed zeros survive).
  !
  ! The call is the call module_pbl_driver.F:1302 makes in a WRF run:
  !   * t3d is t_phy = th_phy*pi_phy, the float32 product phy_prep forms
  !     (module_big_step_utilities_em.F:4855); the file carries it;
  !   * psfc = p8w(kms) = p_interface(1) (module_pbl_driver.F:1033);
  !   * rublten..rqiblten are zeroed first (module_pbl_driver.F:1037-1047),
  !     because shinhong2d accumulates into them (:1205, :1331, :1458);
  !   * flag_qi = .true. and rqiblten present (the driver requires qi_curr);
  !   * ctopo = ctopo2 = 1 (Registry topo_wind = 0, the default and the arm
  !     the WOOF kernel implements);
  !   * every optional argument present, as the driver passes them.
  ! exch_h, el_pbl, hpbl, wstar, delta, regime and kpbl start at zero; tke_pbl
  ! starts at the record's TKE.  exch_h(1) is never written by the scheme,
  ! so it reads back the zero it started with on both sides.
  !
  ! Input record (little endian, stream):
  !   int32[4]      nz, tke_diag, family, base
  !   float32[16]   dt dx dy psfc znt ust hfx qfx wspd br psim psih xland
  !                 u10 v10 corf
  !   float32[nz]   u v theta pi t qv qc qi p dz tke
  !   float32[nz+1] p_interface
  ! Output record:
  !   int32[4]      nz, kpbl, family, base
  !   float32[nz]   rublten rvblten rthblten rqvblten rqcblten rqiblten
  !                 exch_h tke el
  !   float32[9]    hpbl wstar delta u10 v10 ust znt wspd regime
  use module_bl_shinhong, only: shinhong
  implicit none

  ! WRF module_model_constants, default single-precision build.
  real, parameter :: g = 9.81
  real, parameter :: r_d = 287.0
  real, parameter :: r_v = 461.6
  real, parameter :: cp = 7.0 * r_d / 2.0
  real, parameter :: rovcp = r_d / cp
  real, parameter :: rovg = r_d / g
  real, parameter :: xlv = 2.5e6
  real, parameter :: ep1 = r_v / r_d - 1.0
  real, parameter :: ep2 = r_d / r_v
  real, parameter :: karman = 0.4

  integer :: hdr(4), nz, tkediag, ios, uin, uout, k, ncol
  real :: sc(16)
  real, allocatable :: lv(:, :), pint(:)
  real, allocatable, dimension(:, :, :) :: u3d, v3d, th3d, t3d, qv3d, qc3d, &
       qi3d, p3d, p3di, pi3d, dz8w, rublten, rvblten, rthblten, rqvblten, &
       rqcblten, rqiblten, exch_h, tke_pbl, el_pbl
  real, allocatable :: znu(:), znw(:)
  real, dimension(1, 1) :: psfc, znt, ust, hpbl, psim, psih, xland, hfx, &
       qfx, wspd, br, u10, v10, ctopo, ctopo2, corf, wstar, delta, regime
  integer, dimension(1, 1) :: kpbl2d
  real :: p_top, dtstep, dxv, dyv
  character(len=1024) :: in_path, out_path

  call get_command_argument(1, in_path)
  call get_command_argument(2, out_path)
  if (len_trim(in_path) == 0 .or. len_trim(out_path) == 0) then
    write(*, '(A)') 'usage: run_shinhong_columns IN.bin OUT.bin'
    error stop 2
  end if
  open(newunit=uin, file=trim(in_path), access='stream', form='unformatted', &
       status='old', action='read')
  open(newunit=uout, file=trim(out_path), access='stream', &
       form='unformatted', status='replace', action='write')

  ncol = 0
  do
    read(uin, iostat=ios) hdr
    if (ios /= 0) exit
    nz = hdr(1)
    tkediag = hdr(2)
    read(uin) sc
    allocate(lv(nz, 11), pint(nz + 1))
    read(uin) lv
    read(uin) pint

    allocate(u3d(1, nz + 1, 1), v3d(1, nz + 1, 1), th3d(1, nz + 1, 1), &
             t3d(1, nz + 1, 1), qv3d(1, nz + 1, 1), qc3d(1, nz + 1, 1), &
             qi3d(1, nz + 1, 1), p3d(1, nz + 1, 1), p3di(1, nz + 1, 1), &
             pi3d(1, nz + 1, 1), dz8w(1, nz + 1, 1), rublten(1, nz + 1, 1), &
             rvblten(1, nz + 1, 1), rthblten(1, nz + 1, 1), &
             rqvblten(1, nz + 1, 1), rqcblten(1, nz + 1, 1), &
             rqiblten(1, nz + 1, 1), exch_h(1, nz + 1, 1), &
             tke_pbl(1, nz + 1, 1), el_pbl(1, nz + 1, 1), &
             znu(nz + 1), znw(nz + 1))

    ! level nz+1 of every mass array is outside kts:kte and never read
    u3d = 0.0; v3d = 0.0; th3d = 0.0; t3d = 0.0
    qv3d = 0.0; qc3d = 0.0; qi3d = 0.0; p3d = 0.0; pi3d = 1.0; dz8w = 0.0
    do k = 1, nz
      u3d(1, k, 1) = lv(k, 1)
      v3d(1, k, 1) = lv(k, 2)
      th3d(1, k, 1) = lv(k, 3)
      pi3d(1, k, 1) = lv(k, 4)
      t3d(1, k, 1) = lv(k, 5)
      qv3d(1, k, 1) = lv(k, 6)
      qc3d(1, k, 1) = lv(k, 7)
      qi3d(1, k, 1) = lv(k, 8)
      p3d(1, k, 1) = lv(k, 9)
      dz8w(1, k, 1) = lv(k, 10)
    end do
    do k = 1, nz + 1
      p3di(1, k, 1) = pint(k)
    end do

    rublten = 0.0; rvblten = 0.0; rthblten = 0.0
    rqvblten = 0.0; rqcblten = 0.0; rqiblten = 0.0
    exch_h = 0.0; el_pbl = 0.0; tke_pbl = 0.0
    do k = 1, nz
      tke_pbl(1, k, 1) = lv(k, 11)
    end do

    dtstep = sc(1); dxv = sc(2); dyv = sc(3)
    psfc = sc(4); znt = sc(5); ust = sc(6); hfx = sc(7); qfx = sc(8)
    wspd = sc(9); br = sc(10); psim = sc(11); psih = sc(12)
    xland = sc(13); u10 = sc(14); v10 = sc(15); corf = sc(16)
    hpbl = 0.0; kpbl2d = 0; wstar = 0.0; delta = 0.0; regime = 0.0
    ctopo = 1.0; ctopo2 = 1.0

    ! znu/znw/p_top are declared by the wrapper and never referenced.
    p_top = pint(nz + 1)
    do k = 1, nz + 1
      znw(k) = (pint(k) - p_top) / (pint(1) - p_top)
    end do
    do k = 1, nz
      znu(k) = 0.5 * (znw(k) + znw(k + 1))
    end do
    znu(nz + 1) = 0.0

    call shinhong(u3d=u3d, v3d=v3d, th3d=th3d, t3d=t3d,                       &
                  qv3d=qv3d, qc3d=qc3d, qi3d=qi3d,                            &
                  p3d=p3d, p3di=p3di, pi3d=pi3d,                              &
                  rublten=rublten, rvblten=rvblten, rthblten=rthblten,        &
                  rqvblten=rqvblten, rqcblten=rqcblten, rqiblten=rqiblten,    &
                  flag_qi=.true.,                                             &
                  cp=cp, g=g, rovcp=rovcp, rd=r_d, rovg=rovg,                 &
                  ep1=ep1, ep2=ep2, karman=karman, xlv=xlv, rv=r_v,           &
                  dz8w=dz8w, psfc=psfc,                                       &
                  znu=znu, znw=znw, p_top=p_top,                              &
                  znt=znt, ust=ust, hpbl=hpbl, psim=psim, psih=psih,          &
                  xland=xland, hfx=hfx, qfx=qfx, wspd=wspd, br=br,            &
                  dt=dtstep, kpbl2d=kpbl2d,                                   &
                  exch_h=exch_h,                                              &
                  u10=u10, v10=v10,                                           &
                  shinhong_tke_diag=tkediag,                                  &
                  tke_pbl=tke_pbl, el_pbl=el_pbl, corf=corf,                  &
                  dx=dxv, dy=dyv,                                             &
                  ids=1, ide=1, jds=1, jde=1, kds=1, kde=nz + 1,              &
                  ims=1, ime=1, jms=1, jme=1, kms=1, kme=nz + 1,              &
                  its=1, ite=1, jts=1, jte=1, kts=1, kte=nz,                  &
                  ctopo=ctopo, ctopo2=ctopo2,                                 &
                  wstar=wstar, delta=delta,                                   &
                  regime=regime)

    write(uout) nz, kpbl2d(1, 1), hdr(3), hdr(4)
    write(uout) rublten(1, 1:nz, 1)
    write(uout) rvblten(1, 1:nz, 1)
    write(uout) rthblten(1, 1:nz, 1)
    write(uout) rqvblten(1, 1:nz, 1)
    write(uout) rqcblten(1, 1:nz, 1)
    write(uout) rqiblten(1, 1:nz, 1)
    write(uout) exch_h(1, 1:nz, 1)
    write(uout) tke_pbl(1, 1:nz, 1)
    write(uout) el_pbl(1, 1:nz, 1)
    write(uout) hpbl(1, 1), wstar(1, 1), delta(1, 1), u10(1, 1), v10(1, 1), &
                ust(1, 1), znt(1, 1), wspd(1, 1), regime(1, 1)

    deallocate(lv, pint, u3d, v3d, th3d, t3d, qv3d, qc3d, qi3d, p3d, p3di, &
               pi3d, dz8w, rublten, rvblten, rthblten, rqvblten, rqcblten, &
               rqiblten, exch_h, tke_pbl, el_pbl, znu, znw)
    ncol = ncol + 1
  end do
  close(uin)
  close(uout)
  write(*, '(A,I0,A)') 'ran ', ncol, ' columns'
end program run_shinhong_columns
