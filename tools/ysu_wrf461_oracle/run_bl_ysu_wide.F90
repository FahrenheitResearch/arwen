program run_bl_ysu_wide
  ! The wide YSU oracle: drive bl_ysu_run from the byte-unmodified WRF v4.6.1
  ! phys/physics_mmm/bl_ysu.F90 on every column of a binary input file
  ! written by make_wide_columns.py, and write every word it returns.
  !
  ! Each column is ONE call with its=ite=1, made with exactly the argument
  ! set module_bl_ysu.F:404-435 passes in a WRF run:
  !   * uox/vox (WRF's ocean currents, uoce/voce; zero without a coupled
  !     ocean) are PRESENT;
  !   * ctopo/ctopo2 are PRESENT (1.0 unless topo_wind is on);
  !   * the BEP arrays are PRESENT and flag_bep is .false., so bepswitch = 0;
  !   * dusfc/dvsfc/dtsfc/dqsfc are absent, as the driver leaves them;
  !   * f_qc = f_qi = .true., and nmix = 1 with qmix = qv/2 so the scalar
  !     arm runs (its output is written but the port has no tracer arm).
  ! tx is the float32 product theta*pi the input file carries, which is
  ! phy_prep's t_phy.  The theta tendency the WRF driver hands the solver is
  ! module_bl_ysu.F:452's ttnp/pi2d (NEED_B4B_DURING_CCPP_TESTING == 1), so
  ! that division is made here, in float32, exactly as there.
  !
  ! Output record per column (little endian):
  !   int32    kpbl
  !   float32  hpbl, wstar, delta, u10 (after the call), v10 (after)
  !   float32[nz] utnp, vtnp, ttnp, rthblten, qvtnp, qctnp, qitnp,
  !               exch_hx, exch_mx, qmixtnp
  use ccpp_kind_types, only: kind_phys
  use bl_ysu, only: bl_ysu_run
  implicit none

  ! WRF module_model_constants values for the default single-precision build.
  real(kind=kind_phys), parameter :: g = 9.81
  real(kind=kind_phys), parameter :: r_d = 287.0
  real(kind=kind_phys), parameter :: r_v = 461.6
  real(kind=kind_phys), parameter :: cp = 7.0 * r_d / 2.0
  real(kind=kind_phys), parameter :: rovcp = r_d / cp
  real(kind=kind_phys), parameter :: rovg = r_d / g
  real(kind=kind_phys), parameter :: xlv = 2.5e6
  real(kind=kind_phys), parameter :: ep1 = r_v / r_d - 1.0
  real(kind=kind_phys), parameter :: ep2 = r_d / r_v
  real(kind=kind_phys), parameter :: karman = 0.4

  character(len=1024) :: in_path, out_path
  character(len=256) :: errmsg
  integer :: errflg, uin, uout, ios, ncol, k
  integer :: hdr(4), nz
  real(kind=kind_phys) :: sc(18)
  logical :: topdown, with_bep

  real(kind=kind_phys), allocatable :: ux(:,:), vx(:,:), th(:,:), pi2d(:,:)
  real(kind=kind_phys), allocatable :: tx(:,:), qvx(:,:), qcx(:,:), qix(:,:)
  real(kind=kind_phys), allocatable :: p2d(:,:), dz8w2d(:,:), rthraten(:,:)
  real(kind=kind_phys), allocatable :: p2di(:,:), qmix(:,:,:), qmixtnp(:,:,:)
  real(kind=kind_phys), allocatable :: utnp(:,:), vtnp(:,:), ttnp(:,:)
  real(kind=kind_phys), allocatable :: qvtnp(:,:), qctnp(:,:), qitnp(:,:)
  real(kind=kind_phys), allocatable :: exch_hx(:,:), exch_mx(:,:)
  real(kind=kind_phys), allocatable :: bep0(:,:), bep1(:,:), rthblten(:)
  real(kind=kind_phys), allocatable :: bep_drag(:,:), bep_heat(:,:), bep_q(:,:)
  real(kind=kind_phys), allocatable :: bep_momsrc(:,:), bep_tsrc(:,:), bep_qsrc(:,:)
  real(kind=kind_phys) :: dt, psfcpa(1), znt(1), ust(1), hfx(1), qfx(1)
  real(kind=kind_phys) :: wspd(1), br(1), psim(1), psih(1), xland(1)
  real(kind=kind_phys) :: u10(1), v10(1), ctopo(1), ctopo2(1)
  real(kind=kind_phys) :: uox(1), vox(1), hpbl(1), wstar(1), delta(1)
  real(kind=kind_phys) :: frcurb(1)
  integer :: kpbl1d(1)

  call get_command_argument(1, in_path)
  call get_command_argument(2, out_path)
  if (len_trim(in_path) == 0 .or. len_trim(out_path) == 0) then
    write(*, '(A)') 'usage: run_bl_ysu_wide INPUT.bin OUTPUT.bin'
    error stop 2
  end if

  open(newunit=uin, file=trim(in_path), access='stream', form='unformatted', &
       status='old', action='read', convert='little_endian')
  open(newunit=uout, file=trim(out_path), access='stream', &
       form='unformatted', status='replace', action='write', &
       convert='little_endian')

  ncol = 0
  do
    read(uin, iostat=ios) hdr
    if (ios /= 0) exit
    nz = hdr(1)
    topdown = hdr(2) == 1
    with_bep = hdr(4) == 2
    read(uin) sc
    call alloc(nz)
    read(uin) ux(1, :), vx(1, :), th(1, :), pi2d(1, :), tx(1, :), &
              qvx(1, :), qcx(1, :), qix(1, :), p2d(1, :), dz8w2d(1, :), &
              rthraten(1, :)
    read(uin) p2di(1, :)
    dt = sc(1);  psfcpa(1) = sc(2);  znt(1) = sc(3);  ust(1) = sc(4)
    hfx(1) = sc(5);  qfx(1) = sc(6);  wspd(1) = sc(7);  br(1) = sc(8)
    psim(1) = sc(9);  psih(1) = sc(10);  xland(1) = sc(11)
    u10(1) = sc(12);  v10(1) = sc(13);  ctopo(1) = sc(14)
    ctopo2(1) = sc(15);  uox(1) = sc(16);  vox(1) = sc(17)
    frcurb(1) = sc(18)

    qmix(1, :, 1) = qvx(1, :) * 0.5
    utnp = 0.0; vtnp = 0.0; ttnp = 0.0
    qvtnp = 0.0; qctnp = 0.0; qitnp = 0.0
    exch_hx = 0.0; exch_mx = 0.0; qmixtnp = 0.0
    hpbl = 0.0; kpbl1d = 0; wstar = 0.0; delta = 0.0
    bep0 = 0.0; bep1 = 1.0
    bep_drag = 0.0; bep_heat = 0.0; bep_q = 0.0
    bep_momsrc = 0.0; bep_tsrc = 0.0; bep_qsrc = 0.0
    if (with_bep) then
      bep_drag(:,1:min(4,nz)) = -0.002
      bep_heat(:,1:min(4,nz)) = -0.00001
      bep_q(:,1:min(4,nz)) = -0.0001
      bep_momsrc(:,1:min(4,nz)) = 0.0002
      bep_tsrc(:,1:min(4,nz)) = 0.003
      bep_qsrc(:,1:min(4,nz)) = 0.00000002
    end if
    errmsg = ''; errflg = -1

    write(*, '(A,I0)') 'YSUCOL ', ncol
    call bl_ysu_run(ux=ux, vx=vx, tx=tx, qvx=qvx, qcx=qcx, qix=qix,          &
                    nmix=1, qmix=qmix, p2d=p2d, p2di=p2di, pi2d=pi2d,        &
                    f_qc=.true., f_qi=.true.,                               &
                    utnp=utnp, vtnp=vtnp, ttnp=ttnp, qvtnp=qvtnp,            &
                    qctnp=qctnp, qitnp=qitnp, qmixtnp=qmixtnp,               &
                    cp=cp, g=g, rovcp=rovcp, rd=r_d, rovg=rovg,              &
                    ep1=ep1, ep2=ep2, karman=karman, xlv=xlv, rv=r_v,        &
                    dz8w2d=dz8w2d, psfcpa=psfcpa,                            &
                    znt=znt, ust=ust, hpbl=hpbl,                             &
                    psim=psim, psih=psih, xland=xland,                       &
                    hfx=hfx, qfx=qfx, wspd=wspd, br=br,                      &
                    dt=dt, kpbl1d=kpbl1d,                                    &
                    exch_hx=exch_hx, exch_mx=exch_mx,                        &
                    wstar=wstar, delta=delta, u10=u10, v10=v10,              &
                    uox=uox, vox=vox,                                        &
                    rthraten=rthraten, ysu_topdown_pblmix=topdown,           &
                    ctopo=ctopo, ctopo2=ctopo2,                              &
                    a_u=bep_drag, a_v=bep_drag, a_t=bep_heat, a_q=bep_q, a_e=bep0, &
                    b_u=bep_momsrc, b_v=bep_momsrc, b_t=bep_tsrc, b_q=bep_qsrc, b_e=bep0, &
                    sfk=bep1, vlk=bep1, dlu=bep0, dlg=bep0, frcurb=frcurb,   &
                    flag_bep=with_bep,                                       &
                    its=1, ite=1, kte=nz, kme=nz + 1,                        &
                    errmsg=errmsg, errflg=errflg)
    if (errflg /= 0) then
      write(*, '(A,I0,A,A)') 'column ', ncol + 1, ' errmsg: ', trim(errmsg)
      error stop 3
    end if

    do k = 1, nz
      ! module_bl_ysu.F:452
      rthblten(k) = ttnp(1, k) / pi2d(1, k)
    end do

    write(uout) kpbl1d(1)
    write(uout) hpbl(1), wstar(1), delta(1), u10(1), v10(1)
    write(uout) utnp(1, :), vtnp(1, :), ttnp(1, :), rthblten(1:nz), &
                qvtnp(1, :), qctnp(1, :), qitnp(1, :), exch_hx(1, :), &
                exch_mx(1, :), qmixtnp(1, :, 1)
    ncol = ncol + 1
  end do

  close(uin)
  close(uout)
  write(*, '(A,I0,A)') 'wrote ', ncol, ' columns'

contains

  subroutine alloc(n)
    integer, intent(in) :: n
    if (allocated(ux)) then
      deallocate(ux, vx, th, pi2d, tx, qvx, qcx, qix, p2d, dz8w2d, rthraten, &
                 p2di, qmix, qmixtnp, utnp, vtnp, ttnp, qvtnp, qctnp, qitnp, &
                 exch_hx, exch_mx, bep0, bep1, rthblten, bep_drag, bep_heat, bep_q, &
                 bep_momsrc, bep_tsrc, bep_qsrc)
    end if
    allocate(ux(1, n), vx(1, n), th(1, n), pi2d(1, n), tx(1, n), qvx(1, n), &
             qcx(1, n), qix(1, n), p2d(1, n), dz8w2d(1, n), rthraten(1, n), &
             p2di(1, n + 1), qmix(1, n, 1), qmixtnp(1, n, 1), utnp(1, n), &
             vtnp(1, n), ttnp(1, n), qvtnp(1, n), qctnp(1, n), qitnp(1, n), &
             exch_hx(1, n), exch_mx(1, n), bep0(1, n), bep1(1, n), &
             rthblten(n), bep_drag(1,n), bep_heat(1,n), bep_q(1,n), &
             bep_momsrc(1,n), bep_tsrc(1,n), bep_qsrc(1,n))
  end subroutine alloc

end program run_bl_ysu_wide
