! Full-driver column oracle for WRF v4.7.1 bl_pbl_physics = 9 (CAMUWPBL).
!
! Reads a case file written by make_cases.py, initializes exactly as WRF
! does for this scheme, and calls the byte-unmodified camuwpbl for NSTEPS
! consecutive steps on every column, writing each step's inputs and outputs
! through uwio.  Usage:  run_camuwpbl CASES.bin OUTSTEM
!
! Initialization, in WRF's order:
!   module_physics_init.F CAM_INIT (lines 1359-1382 call it; body at the
!   SUBROUTINE CAM_INIT): pcnst_runtime = 5 and pcnst_mp = 5 (EM core, no
!   chemistry, mp_physics /= CAMMGMP), esinti (the saturation table), then
!   ALLOCATE_CAM_ARRAYS' constituent arrays and the five cnst_add calls
!   (Q, CLDLIQ, CLDICE, NUMLIQ, NUMICE).  The modal-aerosol arrays that
!   ALLOCATE_CAM_ARRAYS also allocates belong to CAMMGMP and are never read
!   by this scheme, so they are not allocated here.
!   module_physics_init.F:3825-3832: camuwpblinit(..., restart=.false.,
!   is_CAMMGMP_used=.false.) with the full-level tile (kte = kde), exactly
!   as phy_init passes it.
! Step: module_pbl_driver.F:1928-1983 with KTE = min(k_end, kde-1) = nk.
!   is_CAMMGMP_used is .false. in every ArWen configuration (no CAMMGMP
!   microphysics), so CLDFRA_OLD_MP is never read and is passed as zeros.
!
! Between steps the harness advances the state with the step's own
! tendencies in single precision (th, qv, qc, qi, qni, u, v; t = th*exner).
! That update is harness bookkeeping, not WRF: every step's exact inputs are
! written beside its outputs, so a port is graded step by step and the update
! rule never enters a comparison.
program run_camuwpbl
  use shr_kind_mod, only: r8 => shr_kind_r8
  use module_cam_support, only: pcnst_runtime, pcnst_mp
  use module_cam_esinti, only: esinti
  use module_cam_gffgch, only: gffgch
  use physconst, only: mwh2o, cpwv, epsilo, latvap, latice, rh2o, cpair, &
       tmelt, mwdry, gravit, rair, zvir, karman
  use constituents, only: cnst_add, cnst_name, cnst_longname, cnst_cp, &
       cnst_cv, cnst_mw, cnst_type, cnst_rgas, qmin, qmincg, &
       cnst_fixed_ubc, apcnst, bpcnst, hadvnam, vadvnam, dcconnam, &
       fixcnam, tendnam, ptendnam, dmetendnam, sflxnam, tottnam
  use wv_saturation, only: estblf
  use module_bl_camuwpbl_driver, only: camuwpblinit, camuwpbl
  use uwio
#ifdef UWSPY
  use uwspy
#endif
  implicit none

  character(len=512) :: casefile, outstem
  integer :: u_in, magic, version, ncol, nk, nsteps
  logical :: has_step2
  real(4), allocatable :: carry2(:,:)
  real(4) :: dt
  integer :: ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme
  integer :: its, ite, jts, jte, kts, kte
  integer :: i, k, n, step, mm, ixcldliq, ixcldice, ixnumliq, ixnumice
  character(len=16) :: tag
  real(r8) :: t8, es8
  integer :: itype
  real(r8) :: table(250)

  real(4), allocatable :: u(:,:,:), v(:,:,:), th(:,:,:), rho(:,:,:), qv(:,:,:), &
       qc(:,:,:), qi(:,:,:), qnc(:,:,:), qni(:,:,:), p(:,:,:), z(:,:,:), &
       t(:,:,:), cldfra(:,:,:), cldfra_old_mp(:,:,:), exner(:,:,:), &
       rthratenlw(:,:,:), wsedl3d(:,:,:), p8w(:,:,:), z_at_w(:,:,:)
  real(4), allocatable :: hfx(:,:), qfx(:,:), ust(:,:), ht(:,:)
  real(4), allocatable :: rublten(:,:,:), rvblten(:,:,:), rthblten(:,:,:), &
       rqvblten(:,:,:), rqcblten(:,:,:), rqiblten(:,:,:), rqniblten(:,:,:), &
       kvm3d(:,:,:), kvh3d(:,:,:), tke_pbl(:,:,:), turbtype3d(:,:,:), smaw3d(:,:,:)
  real(4), allocatable :: tauresx2d(:,:), tauresy2d(:,:), tpert2d(:,:), &
       qpert2d(:,:), wpert2d(:,:), pblh2d(:,:)
  integer, allocatable :: kpbl2d(:,:)
  real(4), allocatable :: buf(:,:)

  call get_command_argument(1, casefile)
  call get_command_argument(2, outstem)

  open(newunit=u_in, file=trim(casefile), access='stream', form='unformatted', &
       status='old', action='read')
  read(u_in) magic, version, ncol, nk, nsteps, dt
  if (magic /= 1431785538 .or. version /= 1) error stop 'bad case file'  ! tag 0x55575042

  ids = 1; ide = ncol + 1; jds = 1; jde = 2; kds = 1; kde = nk + 1
  ims = 1; ime = ncol;     jms = 1; jme = 1; kms = 1; kme = nk + 1
  its = 1; ite = ncol;     jts = 1; jte = 1; kts = 1

  allocate(u(ims:ime,kms:kme,jms:jme), v(ims:ime,kms:kme,jms:jme), &
       th(ims:ime,kms:kme,jms:jme), rho(ims:ime,kms:kme,jms:jme), &
       qv(ims:ime,kms:kme,jms:jme), qc(ims:ime,kms:kme,jms:jme), &
       qi(ims:ime,kms:kme,jms:jme), qnc(ims:ime,kms:kme,jms:jme), &
       qni(ims:ime,kms:kme,jms:jme), p(ims:ime,kms:kme,jms:jme), &
       z(ims:ime,kms:kme,jms:jme), t(ims:ime,kms:kme,jms:jme), &
       cldfra(ims:ime,kms:kme,jms:jme), cldfra_old_mp(ims:ime,kms:kme,jms:jme), &
       exner(ims:ime,kms:kme,jms:jme), rthratenlw(ims:ime,kms:kme,jms:jme), &
       wsedl3d(ims:ime,kms:kme,jms:jme), p8w(ims:ime,kms:kme,jms:jme), &
       z_at_w(ims:ime,kms:kme,jms:jme))
  allocate(hfx(ims:ime,jms:jme), qfx(ims:ime,jms:jme), ust(ims:ime,jms:jme), &
       ht(ims:ime,jms:jme))
  allocate(rublten(ims:ime,kms:kme,jms:jme), rvblten(ims:ime,kms:kme,jms:jme), &
       rthblten(ims:ime,kms:kme,jms:jme), rqvblten(ims:ime,kms:kme,jms:jme), &
       rqcblten(ims:ime,kms:kme,jms:jme), rqiblten(ims:ime,kms:kme,jms:jme), &
       rqniblten(ims:ime,kms:kme,jms:jme), kvm3d(ims:ime,kms:kme,jms:jme), &
       kvh3d(ims:ime,kms:kme,jms:jme), tke_pbl(ims:ime,kms:kme,jms:jme), &
       turbtype3d(ims:ime,kms:kme,jms:jme), smaw3d(ims:ime,kms:kme,jms:jme))
  allocate(tauresx2d(ims:ime,jms:jme), tauresy2d(ims:ime,jms:jme), &
       tpert2d(ims:ime,jms:jme), qpert2d(ims:ime,jms:jme), &
       wpert2d(ims:ime,jms:jme), pblh2d(ims:ime,jms:jme), kpbl2d(ims:ime,jms:jme))

  ! Everything WRF would hand the scheme that this harness does not set is
  ! zero, as the Registry initializes it.
  u = 0.; v = 0.; th = 0.; rho = 0.; qv = 0.; qc = 0.; qi = 0.; qnc = 0.
  qni = 0.; p = 0.; z = 0.; t = 0.; cldfra = 0.; cldfra_old_mp = 0.
  exner = 0.; rthratenlw = 0.; wsedl3d = 0.; p8w = 0.; z_at_w = 0.
  rublten = 0.; rvblten = 0.; rthblten = 0.; rqvblten = 0.; rqcblten = 0.
  rqiblten = 0.; rqniblten = 0.; kvm3d = 0.; kvh3d = 0.; tke_pbl = 0.
  turbtype3d = 0.; smaw3d = 0.; tauresx2d = 0.; tauresy2d = 0.
  tpert2d = 0.; qpert2d = 0.; wpert2d = 0.; pblh2d = 0.; kpbl2d = 0

  call read_mass(u); call read_mass(v); call read_mass(th); call read_mass(rho)
  call read_mass(qv); call read_mass(qc); call read_mass(qi); call read_mass(qnc)
  call read_mass(qni); call read_mass(p); call read_mass(z); call read_mass(t)
  call read_mass(cldfra); call read_mass(exner); call read_mass(rthratenlw)
  call read_mass(wsedl3d)
  call read_full(p8w); call read_full(z_at_w)
  call read_sfc(hfx); call read_sfc(qfx); call read_sfc(ust); call read_sfc(ht)
  close(u_in)

  ! ---- CAM_INIT (module_physics_init.F), EM core, no chemistry ----
  pcnst_runtime = 5
  pcnst_mp = pcnst_runtime
  call esinti(epsilo, latvap, latice, rh2o, cpair, tmelt)
  allocate(cnst_name(pcnst_runtime), cnst_longname(pcnst_runtime), &
       cnst_cp(pcnst_runtime), cnst_cv(pcnst_runtime), cnst_mw(pcnst_runtime), &
       cnst_type(pcnst_runtime), cnst_rgas(pcnst_runtime), qmin(pcnst_runtime), &
       qmincg(pcnst_runtime), cnst_fixed_ubc(pcnst_runtime), apcnst(pcnst_runtime), &
       bpcnst(pcnst_runtime), hadvnam(pcnst_runtime), vadvnam(pcnst_runtime), &
       dcconnam(pcnst_runtime), fixcnam(pcnst_runtime), tendnam(pcnst_runtime), &
       ptendnam(pcnst_runtime), dmetendnam(pcnst_runtime), sflxnam(pcnst_runtime), &
       tottnam(pcnst_runtime))
  cnst_fixed_ubc(:) = .false.
  call cnst_add('Q', mwh2o, cpwv, 1.E-12_r8, mm, longname='Specific humidity', readiv=.true.)
  call cnst_add('CLDLIQ', mwdry, cpair, 0._r8, ixcldliq, longname='Grid box averaged cloud liquid amount')
  call cnst_add('CLDICE', mwdry, cpair, 0._r8, ixcldice, longname='Grid box averaged cloud ice amount')
  call cnst_add('NUMLIQ', mwdry, cpair, 0._r8, ixnumliq, longname='Grid box averaged cloud liquid number')
  call cnst_add('NUMICE', mwdry, cpair, 0._r8, ixnumice, longname='Grid box averaged cloud ice number')

  ! ---- phy_init -> camuwpblinit, full-level tile ----
  call camuwpblinit(rublten, rvblten, rthblten, rqvblten, .false., tke_pbl, .false., &
       ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
       its, ite, jts, jte, kts, kde)
  kte = nk

  call uwio_open(trim(outstem))
#ifdef UWSPY
  call uwspy_open(trim(outstem)//'-spy')
#endif
  uwio_prefix = 'const/'
  call uwio_put1_r8('cpair', cpair); call uwio_put1_r8('gravit', gravit)
  call uwio_put1_r8('rair', rair); call uwio_put1_r8('zvir', zvir)
  call uwio_put1_r8('latvap', latvap); call uwio_put1_r8('latice', latice)
  call uwio_put1_r8('karman', karman); call uwio_put1_r8('epsilo', epsilo)
  call uwio_put1_r8('rh2o', rh2o); call uwio_put1_r8('tmelt', tmelt)
  call uwio_put1_r8('mwh2o', mwh2o); call uwio_put1_r8('cpwv', cpwv)
  call uwio_put1_r8('mwdry', mwdry)
  call uwio_put1_r8('qmin1', qmin(1)); call uwio_put1_r8('qmin2', qmin(2))
  call uwio_put1_r8('qmin3', qmin(3))
  call uwio_put1_r8('b123', 5.8_r8**(2._r8/3._r8))
  call uwio_put1_r4('dt', dt)
  call uwio_put1_i4('ncol', ncol); call uwio_put1_i4('nk', nk)
  call uwio_put1_i4('nsteps', nsteps)
  call uwio_put1_i4('ixcldliq', ixcldliq); call uwio_put1_i4('ixcldice', ixcldice)
  call uwio_put1_i4('ixnumliq', ixnumliq); call uwio_put1_i4('ixnumice', ixnumice)
  ! gestbl's own loop (module_cam_wv_saturation.F gestbl, esinti's arguments
  ! tmn=173.16, tmx=375.16, trice=20, ip=.true.), replayed through the public
  ! gffgch with gestbl's own arithmetic (t = tmin - 1, then t = t + 1 per
  ! entry, itype = -ttrice), so the whole table (204 entries, the rest
  ! -99999) is a fixture.  estblf evaluated at tmin + n - 1 is written beside
  ! it as a probe of the lookup itself (e - tmin is not exact there, so the
  ! probe interpolates; it is a function fixture, not a table read-back).
  t8 = 173.16_r8 - 1.0_r8
  table = -99999.0_r8
  do n = 1, int(375.16_r8 - 173.16_r8 + 2.000001_r8)
     t8 = t8 + 1.0_r8
     itype = -20
     call gffgch(t8, es8, itype)
     table(n) = es8
  end do
  call uwio_put_r8('estbl', table, 250)
  do n = 1, 203
     table(n) = estblf(173.16_r8 + real(n - 1, r8))
  end do
  call uwio_put_r8('estblf_probe', table, 203)

  ! Optional subnormal probe sidecar. WRF sources and arithmetic are unchanged.
  ! Two (nk+1,ncol) diffusivity blocks then two (1,ncol) stress blocks.
  ! NaN is a sentinel meaning retain the preceding step's value.
  inquire(file=trim(casefile)//'.step2', exist=has_step2)
  do step = 1, nsteps
     if (step == 2 .and. has_step2) then
        open(newunit=u_in, file=trim(casefile)//'.step2', access='stream', &
             form='unformatted', status='old', action='read')
        allocate(carry2(nk+1,ncol))
        read(u_in) carry2
        do i=1,ncol
           do k=1,nk+1
              if (carry2(k,i) == carry2(k,i)) kvm3d(i,k,1)=carry2(k,i)
           end do
        end do
        read(u_in) carry2
        do i=1,ncol
           do k=1,nk+1
              if (carry2(k,i) == carry2(k,i)) kvh3d(i,k,1)=carry2(k,i)
           end do
        end do
        deallocate(carry2)
        allocate(carry2(1,ncol))
        read(u_in) carry2
        do i=1,ncol
           if (carry2(1,i) == carry2(1,i)) tauresx2d(i,1)=carry2(1,i)
        end do
        read(u_in) carry2
        do i=1,ncol
           if (carry2(1,i) == carry2(1,i)) tauresy2d(i,1)=carry2(1,i)
        end do
        deallocate(carry2)
        close(u_in)
     end if
     write(tag, '(a,i0,a)') 's', step, '/'
     uwio_prefix = tag
     call put_mass('u', u); call put_mass('v', v); call put_mass('th', th)
     call put_mass('rho', rho); call put_mass('qv', qv); call put_mass('qc', qc)
     call put_mass('qi', qi); call put_mass('qnc', qnc); call put_mass('qni', qni)
     call put_mass('p', p); call put_mass('z', z); call put_mass('t', t)
     call put_mass('cldfra', cldfra); call put_mass('exner', exner)
     call put_mass('rthratenlw', rthratenlw); call put_mass('wsedl3d', wsedl3d)
     call put_full('p8w', p8w); call put_full('z_at_w', z_at_w)
     call put_sfc('hfx', hfx); call put_sfc('qfx', qfx); call put_sfc('ust', ust)
     call put_sfc('ht', ht)
     call put_full('kvm3d_in', kvm3d); call put_full('kvh3d_in', kvh3d)
     call put_sfc('tauresx2d_in', tauresx2d); call put_sfc('tauresy2d_in', tauresy2d)
     call uwio_put1_i4('itimestep', step)

     call camuwpbl(dt, u, v, th, rho, qv, hfx, qfx, ust, p8w, &
          p, z, t, qc, qi, z_at_w, cldfra_old_mp, cldfra, ht, &
          rthratenlw, exner, .false., &
          step, qnc, qni, wsedl3d, &
          ids, ide, jds, jde, kds, kde, &
          ims, ime, jms, jme, kms, kme, &
          its, ite, jts, jte, kts, kte, &
          tauresx2d, tauresy2d, &
          rublten, rvblten, rthblten, rqiblten, rqniblten, rqvblten, rqcblten, &
          kvm3d, kvh3d, &
          tpert2d, qpert2d, wpert2d, smaw3d, turbtype3d, &
          tke_pbl, pblh2d, kpbl2d)

     call put_mass('rublten', rublten); call put_mass('rvblten', rvblten)
     call put_mass('rthblten', rthblten); call put_mass('rqvblten', rqvblten)
     call put_mass('rqcblten', rqcblten); call put_mass('rqiblten', rqiblten)
     call put_mass('rqniblten', rqniblten)
     call put_full('kvm3d', kvm3d); call put_full('kvh3d', kvh3d)
     call put_full('tke_pbl', tke_pbl); call put_full('turbtype3d', turbtype3d)
     call put_full('smaw3d', smaw3d)
     call put_sfc('tauresx2d', tauresx2d); call put_sfc('tauresy2d', tauresy2d)
     call put_sfc('tpert2d', tpert2d); call put_sfc('qpert2d', qpert2d)
     call put_sfc('wpert2d', wpert2d); call put_sfc('pblh2d', pblh2d)
     call uwio_put_i4('kpbl2d', kpbl2d(its:ite, 1), ncol)

     ! harness bookkeeping only (see header)
     do k = kts, kte
        do i = its, ite
           th(i,k,1) = th(i,k,1) + dt * rthblten(i,k,1)
           t(i,k,1) = th(i,k,1) * exner(i,k,1)
           qv(i,k,1) = max(0., qv(i,k,1) + dt * rqvblten(i,k,1))
           qc(i,k,1) = max(0., qc(i,k,1) + dt * rqcblten(i,k,1))
           qi(i,k,1) = max(0., qi(i,k,1) + dt * rqiblten(i,k,1))
           qni(i,k,1) = max(0., qni(i,k,1) + dt * rqniblten(i,k,1))
           u(i,k,1) = u(i,k,1) + dt * rublten(i,k,1)
           v(i,k,1) = v(i,k,1) + dt * rvblten(i,k,1)
        end do
     end do
  end do
  call uwio_close()
#ifdef UWSPY
  call uwspy_close()
#endif

contains

  subroutine uwio_put1_r4(name, x)
    character(len=*), intent(in) :: name
    real(4), intent(in) :: x
    real(4) :: a(1)
    a(1) = x
    call uwio_put_r4(name, a, 1)
  end subroutine uwio_put1_r4

  subroutine read_mass(a)
    real(4), intent(inout) :: a(ims:ime,kms:kme,jms:jme)
    allocate(buf(nk, ncol))
    read(u_in) buf
    do i = 1, ncol
       a(i, 1:nk, 1) = buf(:, i)
    end do
    deallocate(buf)
  end subroutine read_mass

  subroutine read_full(a)
    real(4), intent(inout) :: a(ims:ime,kms:kme,jms:jme)
    allocate(buf(nk + 1, ncol))
    read(u_in) buf
    do i = 1, ncol
       a(i, 1:nk + 1, 1) = buf(:, i)
    end do
    deallocate(buf)
  end subroutine read_full

  subroutine read_sfc(a)
    real(4), intent(inout) :: a(ims:ime,jms:jme)
    allocate(buf(1, ncol))
    read(u_in) buf
    a(1:ncol, 1) = buf(1, :)
    deallocate(buf)
  end subroutine read_sfc

  subroutine put_mass(name, a)
    character(len=*), intent(in) :: name
    real(4), intent(in) :: a(ims:ime,kms:kme,jms:jme)
    real(4) :: o(nk, ncol)
    do i = 1, ncol
       o(:, i) = a(i, 1:nk, 1)
    end do
    call uwio_put_r4(name, o, nk * ncol)
  end subroutine put_mass

  subroutine put_full(name, a)
    character(len=*), intent(in) :: name
    real(4), intent(in) :: a(ims:ime,kms:kme,jms:jme)
    real(4) :: o(nk + 1, ncol)
    do i = 1, ncol
       o(:, i) = a(i, 1:nk + 1, 1)
    end do
    call uwio_put_r4(name, o, (nk + 1) * ncol)
  end subroutine put_full

  subroutine put_sfc(name, a)
    character(len=*), intent(in) :: name
    real(4), intent(in) :: a(ims:ime,jms:jme)
    call uwio_put_r4(name, a(1:ncol, 1), ncol)
  end subroutine put_sfc
end program run_camuwpbl
