! Official WRF v4.6.1 NSSL option-18 driver over a set of columns.
!
! Usage: nssl2_column_oracle INPUT.bin OUTPUT.bin DT ITIMESTEP MODE
!   MODE = column : one nssl_2mom_driver call per column (its = ite = i)
!   MODE = slab   : one call over all columns (its = 1, ite = ncol)
!
! The call mirrors phys/module_microphysics_driver.F:1952 for the resolved
! mp_physics=18 defaults (module_check_a_mundo.F: 2-moment, hail, CCN and
! graupel+hail density on) and phys/module_physics_init.F:4646 for init.
! Memory and domain k extents are the staggered nz+1 (WRF's kme = kde),
! tile extents the nz mass levels (kte = kde-1), so W reaches the driver
! staggered, exactly as grid%w_2 does.  dx/dy are not passed (WRF does not
! pass them).  diagflag=.true. and ke_diag=nz are a history-output step
! (dyn_em/solve_em.F:363-369).  has_reqc/i/s=1 as with RRTMG use_mp_re.
!
! Output stream (little-endian float32): for each 3-D field in OUT3 order
! ncol*nz values (k fastest), then each 2-D field in OUT2 order, ncol values.
program nssl2_column_oracle
  use module_mp_nssl_2mom, only: nssl_2mom_init, nssl_2mom_driver
  implicit none

  integer, parameter :: nin3 = 21
  integer :: u, nz, ncol, nzs, i, k, itimestep, ilo, ihi, ccn_is_ccna
  real :: dt_s, nssl_params(20)
  character(len=512) :: input_path, output_path, arg
  character(len=16) :: mode
  real, allocatable :: f3(:,:,:), wv(:,:)
  real, allocatable :: theta(:,:,:), qv(:,:,:), qc(:,:,:), qr(:,:,:)
  real, allocatable :: qi(:,:,:), qs(:,:,:), qg(:,:,:), qh(:,:,:)
  real, allocatable :: qndrop(:,:,:), qnr(:,:,:), qni(:,:,:)
  real, allocatable :: qns(:,:,:), qng(:,:,:), qnh(:,:,:), qnn(:,:,:)
  real, allocatable :: qvolg(:,:,:), qvolh(:,:,:)
  real, allocatable :: pressure(:,:,:), exner(:,:,:), rho(:,:,:)
  real, allocatable :: dz(:,:,:), w(:,:,:)
  real, allocatable :: dbz(:,:,:), re_cloud(:,:,:), re_ice(:,:,:), re_snow(:,:,:)
  real, allocatable :: rainnc(:,:), rainncv(:,:), snownc(:,:), snowncv(:,:)
  real, allocatable :: grplnc(:,:), grplncv(:,:), hailnc(:,:), hailncv(:,:)
  real, allocatable :: sr(:,:), hail_maxk1(:,:), hail_max2d(:,:)

  call get_command_argument(1, input_path)
  call get_command_argument(2, output_path)
  call get_command_argument(3, arg); read(arg, *) dt_s
  call get_command_argument(4, arg); read(arg, *) itimestep
  call get_command_argument(5, mode)
  if (len_trim(output_path) == 0 .or. (mode /= 'column' .and. mode /= 'slab')) &
       error stop 'usage: nssl2_column_oracle IN.bin OUT.bin DT ITIMESTEP column|slab'

  open(newunit=u, file=trim(input_path), access='stream', form='unformatted', &
       status='old', action='read', convert='little_endian')
  read(u) nz, ncol
  nzs = nz + 1
  allocate(f3(nz, ncol, nin3), wv(nzs, ncol))
  read(u) f3
  read(u) wv

#define ALLOC3(x) allocate(x(ncol, nzs, 1)); x = 0.0
  ALLOC3(theta); ALLOC3(qv); ALLOC3(qc); ALLOC3(qr); ALLOC3(qi); ALLOC3(qs)
  ALLOC3(qg); ALLOC3(qh); ALLOC3(qndrop); ALLOC3(qnr); ALLOC3(qni); ALLOC3(qns)
  ALLOC3(qng); ALLOC3(qnh); ALLOC3(qnn); ALLOC3(qvolg); ALLOC3(qvolh)
  ALLOC3(pressure); ALLOC3(exner); ALLOC3(rho); ALLOC3(dz); ALLOC3(w)
  ALLOC3(dbz); ALLOC3(re_cloud); ALLOC3(re_ice); ALLOC3(re_snow)
  allocate(rainnc(ncol,1), rainncv(ncol,1), snownc(ncol,1), snowncv(ncol,1))
  allocate(grplnc(ncol,1), grplncv(ncol,1), hailnc(ncol,1), hailncv(ncol,1))
  allocate(sr(ncol,1), hail_maxk1(ncol,1), hail_max2d(ncol,1))
  read(u) rainnc(:,1)
  read(u) snownc(:,1)
  read(u) grplnc(:,1)
  read(u) hailnc(:,1)
  close(u)

  do i = 1, ncol
     do k = 1, nz
        theta(i,k,1) = f3(k,i,1);   qv(i,k,1) = f3(k,i,2)
        qc(i,k,1) = f3(k,i,3);      qr(i,k,1) = f3(k,i,4)
        qi(i,k,1) = f3(k,i,5);      qs(i,k,1) = f3(k,i,6)
        qg(i,k,1) = f3(k,i,7);      qh(i,k,1) = f3(k,i,8)
        qndrop(i,k,1) = f3(k,i,9);  qnr(i,k,1) = f3(k,i,10)
        qni(i,k,1) = f3(k,i,11);    qns(i,k,1) = f3(k,i,12)
        qng(i,k,1) = f3(k,i,13);    qnh(i,k,1) = f3(k,i,14)
        qnn(i,k,1) = f3(k,i,15);    qvolg(i,k,1) = f3(k,i,16)
        qvolh(i,k,1) = f3(k,i,17);  pressure(i,k,1) = f3(k,i,18)
        exner(i,k,1) = f3(k,i,19);  rho(i,k,1) = f3(k,i,20)
        dz(i,k,1) = f3(k,i,21)
     end do
     do k = 1, nzs
        w(i,k,1) = wv(k,i)
     end do
  end do

  ! module_physics_init.F:4621-4651 with the Registry defaults.
  nssl_params = 0.0
  nssl_params(1)  = 0.5e9    ! nssl_cccn
  nssl_params(2)  = 0.0      ! nssl_alphah
  nssl_params(3)  = 1.0      ! nssl_alphahl
  nssl_params(4)  = 4.0e5    ! nssl_cnoh
  nssl_params(5)  = 4.0e4    ! nssl_cnohl
  nssl_params(6)  = 8.0e5    ! nssl_cnor
  nssl_params(7)  = 3.0e6    ! nssl_cnos
  nssl_params(8)  = 500.0    ! nssl_rho_qh
  nssl_params(9)  = 900.0    ! nssl_rho_qhl
  nssl_params(10) = 100.0    ! nssl_rho_qs
  nssl_params(11) = 0.0      ! nssl_ipelec (elec_physics = 0)
  nssl_params(12) = 12.0     ! nssl_isaund
  ccn_is_ccna = 0
  call nssl_2mom_init(nssl_params=nssl_params, ipctmp=5, mixphase=0, &
       nssl_density_on=.true., nssl_hail_on=.true., nssl_ccn_on=.true., &
       nssl_icdx=6, nssl_icdxhl=6, ccn_is_ccna=ccn_is_ccna)

  rainncv = 0.0; snowncv = 0.0; grplncv = 0.0; hailncv = 0.0; sr = 0.0
  hail_maxk1 = 0.0; hail_max2d = 0.0
  dbz = 0.0; re_cloud = 0.0; re_ice = 0.0; re_snow = 0.0

  if (mode == 'slab') then
     call run(1, ncol)
  else
     do i = 1, ncol
        call run(i, i)
     end do
  end if

  open(newunit=u, file=trim(output_path), access='stream', form='unformatted', &
       status='replace', action='write', convert='little_endian')
  call put3(theta); call put3(qv); call put3(qc); call put3(qr); call put3(qi)
  call put3(qs); call put3(qg); call put3(qh); call put3(qndrop); call put3(qnr)
  call put3(qni); call put3(qns); call put3(qng); call put3(qnh); call put3(qnn)
  call put3(qvolg); call put3(qvolh); call put3(dbz); call put3(re_cloud)
  call put3(re_ice); call put3(re_snow)
  write(u) rainnc(:,1), rainncv(:,1), snownc(:,1), snowncv(:,1), &
       grplnc(:,1), grplncv(:,1), hailnc(:,1), hailncv(:,1), sr(:,1)
  close(u)
  print '(A,1X,A,1X,I0,1X,A)', 'NSSL2_COLUMN_ORACLE_COMPLETE', trim(mode), ncol, &
       trim(output_path)

contains

  subroutine put3(x)
    real, intent(in) :: x(:,:,:)
    integer :: ii
    do ii = 1, ncol
       write(u) x(ii, 1:nz, 1)
    end do
  end subroutine put3

  subroutine run(ilo, ihi)
    integer, intent(in) :: ilo, ihi
    call nssl_2mom_driver( &
         itimestep=itimestep, th=theta, &
         qv=qv, qc=qc, qr=qr, qi=qi, qs=qs, qh=qg, qhl=qh, &
         ccw=qndrop, crw=qnr, cci=qni, csw=qns, chw=qng, chl=qnh, &
         vhw=qvolg, f_vhw=.true., vhl=qvolh, f_vhl=.true., &
         cn=qnn, f_cn=.true., &
         pii=exner, p=pressure, w=w, dz=dz, dtp=dt_s, dn=rho, &
         rainnc=rainnc, rainncv=rainncv, snownc=snownc, snowncv=snowncv, &
         hailnc=hailnc, hailncv=hailncv, grplnc=grplnc, grplncv=grplncv, &
         sr=sr, dbz=dbz, &
         nssl_progn=.false., diagflag=.true., ke_diag=nz, &
         cu_used=0, &
         re_cloud=re_cloud, re_ice=re_ice, re_snow=re_snow, &
         has_reqc=1, has_reqi=1, has_reqs=1, &
         hail_maxk1=hail_maxk1, hail_max2d=hail_max2d, nwp_diagnostics=0, &
         ids=1, ide=ncol + 1, jds=1, jde=2, kds=1, kde=nzs, &
         ims=1, ime=ncol, jms=1, jme=1, kms=1, kme=nzs, &
         its=ilo, ite=ihi, jts=1, jte=1, kts=1, kte=nz)
  end subroutine run

end program nssl2_column_oracle
