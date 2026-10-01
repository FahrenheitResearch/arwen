! Drive WRF 4.7.1's own compiled IEVA routines (module_ieva_em.o and
! calc_mu_uv_1 out of main/libwrflib.a) on arrays captured from a gpuwm
! forecast at an IEVA substep, and write what WRF computes, so capture.py
! can compare gpuwm's CUDA output word for word.
!
! Input: <dir>/meta.txt and <dir>/<name>.bin (float32, C order (k, j, i)),
! written by capture.py.  Output: <dir>/wrf_<name>.bin in the same layout.
!
! The arrays are placed on WRF's (ims:ime, kms:kme, jms:jme) memory with a
! three-point halo filled by edge replication; ids = its = 1, ide = ite =
! nx + 1 (the u staggering), kds = kts = 1, kde = kte = nz + 1, as one tile
! of one patch.  config_flags: rk_ord 3, zadvect_implicit 1, specified.
!
! Built two ways (build.sh): against WRF 4.7.1's own module_ieva_em, and,
! with -DA179, against that module with A179's two advect_w_implicit
! corrections (wrf_a179.py), which is what gpuwm's ieva_solve_w computes.
program ieva_oracle
  use module_configure, only: grid_config_rec_type
  use module_ieva_em
  use module_big_step_utilities_em, only: calc_mu_uv_1
  implicit none

  type(grid_config_rec_type) :: cf
  character(len=512) :: dir, mode
  logical :: chain
  integer :: nx, ny, nz, has_msf
  real :: dt, dx, dy, cf1, cf2, cf3, dt_s, rdx, rdy
  integer :: ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme
  integer :: its, ite, jts, jte, kts, kte
  integer :: i, j, k

  real, allocatable, dimension(:,:,:) :: u, v, ww, u_old, v_old, w_old, &
       ph, ph_old, phb, s_old, wwE, wwI, ru_t, rv_t, rw_t, rth_t, rph_t, &
       ru, rv, dum3, ww_m, q_old, q_tend, wwE_m, wwI_m
  real, allocatable, dimension(:,:) :: mut, mut_old, mut_new, muu, muv, &
       muu_old, muv_old, muu_new, muv_new, msft, msfu, msfv, msfv_inv, ht, &
       muts, mu0s
  real, allocatable, dimension(:) :: rdnw, rdn, c1f, c2f, c1h, c2h, fnm, fnp

  call get_command_argument(1, dir)
  call get_command_argument(2, mode)
  ! ``chain``: every solve reads WRF's own upstream outputs (a synthetic
  ! fixture); otherwise each solve reads gpuwm's, so it is graded alone.
  chain = trim(mode) == 'chain'
  open(10, file=trim(dir)//'/meta.txt', status='old')
  read(10, *) nx, ny, nz, has_msf
  read(10, *) dt, dx, dy, dt_s
  read(10, *) cf1, cf2, cf3
  close(10)

  ids = 1; ide = nx + 1; jds = 1; jde = ny + 1; kds = 1; kde = nz + 1
  its = ids; ite = ide; jts = jds; jte = jde; kts = kds; kte = kde
  ims = -2; ime = nx + 4; jms = -2; jme = ny + 4; kms = 1; kme = nz + 1

  cf%rk_ord = 3
  cf%zadvect_implicit = 1
  cf%specified = .true.
  cf%nested = .false.
  cf%open_xs = .false.; cf%open_xe = .false.
  cf%open_ys = .false.; cf%open_ye = .false.
  cf%periodic_x = .false.; cf%periodic_y = .false.
  cf%polar = .false.

  rdx = 1./dx
  rdy = 1./dy

  call alloc3(u);  call alloc3(v); call alloc3(ww); call alloc3(u_old)
  call alloc3(v_old); call alloc3(w_old); call alloc3(ph); call alloc3(ph_old)
  call alloc3(phb); call alloc3(s_old); call alloc3(wwE); call alloc3(wwI)
  call alloc3(ru_t); call alloc3(rv_t); call alloc3(rw_t); call alloc3(rth_t)
  call alloc3(rph_t); call alloc3(ru); call alloc3(rv); call alloc3(dum3)
  call alloc3(ww_m); call alloc3(q_old); call alloc3(q_tend)
  call alloc3(wwE_m); call alloc3(wwI_m)
  call alloc2(mut); call alloc2(mut_old); call alloc2(mut_new); call alloc2(muu)
  call alloc2(muv); call alloc2(muu_old); call alloc2(muv_old)
  call alloc2(muu_new); call alloc2(muv_new); call alloc2(msft); call alloc2(msfu)
  call alloc2(msfv); call alloc2(msfv_inv); call alloc2(ht); call alloc2(muts)
  call alloc2(mu0s)
  allocate(rdnw(kms:kme), rdn(kms:kme), c1f(kms:kme), c2f(kms:kme), &
           c1h(kms:kme), c2h(kms:kme), fnm(kms:kme), fnp(kms:kme))
  rdnw = 0.; rdn = 0.; c1f = 0.; c2f = 0.; c1h = 0.; c2h = 0.; fnm = 0.; fnp = 0.

  ! (nlev, nyx, nxx) C-order extents per staggering.
  call read3('u', u, nz, ny, nx + 1)
  call read3('v', v, nz, ny + 1, nx)
  call read3('ww', ww, nz + 1, ny, nx)
  call read3('u0', u_old, nz, ny, nx + 1)
  call read3('v0', v_old, nz, ny + 1, nx)
  call read3('w0', w_old, nz + 1, ny, nx)
  call read3('php', ph, nz + 1, ny, nx)
  call read3('php0', ph_old, nz + 1, ny, nx)
  call read3('phb', phb, nz + 1, ny, nx)
  call read3('theta_old', s_old, nz, ny, nx)
  call read3('ru_t_explicit', ru_t, nz, ny, nx + 1)
  call read3('rv_t_explicit', rv_t, nz, ny + 1, nx)
  call read3('rth_t_explicit', rth_t, nz, ny, nx)
  call read3('rph_t_explicit', rph_t, nz + 1, ny, nx)
  call read3('rw_t_explicit', rw_t, nz + 1, ny, nx)
  call read2('mut', mut, ny, nx)
  call read2('mut_old', mut_old, ny, nx)
  call read2('mux', muu, ny, nx + 1)
  call read2('muy', muv, ny + 1, nx)
  call read2('msft', msft, ny, nx)
  call read2('msfu', msfu, ny, nx + 1)
  call read2('msfv', msfv, ny + 1, nx)
  call read2('ht', ht, ny, nx)
  call read1('rdnw', rdnw, nz)
  call read1('rdn', rdn, nz)
  call read1('c1f', c1f, nz + 1)
  call read1('c2f', c2f, nz + 1)
  call read1('c1h', c1h, nz)
  call read1('c2h', c2h, nz)
  call read1('fnm', fnm, nz)
  call read1('fnp', fnp, nz)
  do j = jms, jme
    do i = ims, ime
      msfv_inv(i, j) = 1./msfv(i, j)      ! real.exe: msfvx_inv = 1./msfvx
    end do
  end do

  ! ---- rk_tendency, module_em.F:438-487 -------------------------------
  call WW_SPLIT(wwE, wwI, u, v, ww, mut, rdnw, msft, c1f, c2f, rdx, rdy, &
                msfu, msfu, msfv, msfv, dt, cf, 3, &
                ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
                its, ite, jts, jte, kts, kte)
  call write3('wwE', wwE, nz + 1, ny, nx)
  call write3('wwI', wwI, nz + 1, ny, nx)
  call CALC_MUT_NEW(u, v, c1h, c2h, mut_old, muu, muv, mut_new, dt, rdx, rdy, &
                    msft, msft, msfu, msfu, msfv, msfv_inv, msfv, rdnw, &
                    ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
                    its, ite, jts, jte, kts, kte)
  call write2('mut_new', mut_new, ny, nx)
  call halo2(mut_new, nx, ny)
  call calc_mu_uv_1(cf, mut_old, muu_old, muv_old, &
                    ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
                    its, ite, jts, jte, kts, kte)
  call calc_mu_uv_1(cf, mut_new, muu_new, muv_new, &
                    ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
                    its, ite, jts, jte, kts, kte)
  ! wwI is read back from gpuwm so each solve below is graded on its own
  ! input; the split itself is graded above.
  if (.not. chain) then
    call read3('wwI', wwI, nz + 1, ny, nx)
    call read3('wwE', wwE, nz + 1, ny, nx)
    call read2('mut_new', mut_new, ny, nx)
    call calc_mu_uv_1(cf, mut_new, muu_new, muv_new, &
                      ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
                      its, ite, jts, jte, kts, kte)
  end if

  call advect_u_implicit(u, u_old, ru_t, ru, rv, wwI, c1h, c2h, &
                         muu_old, muu, muu_new, cf, msfu, msfu, msfv, msfv, &
                         msft, msft, fnm, fnp, dt, rdx, rdy, rdnw, &
                         ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
                         its, ite, jts, jte, kts, kte)
  call write3('ru_t', ru_t, nz, ny, nx + 1)
  call advect_v_implicit(v, v_old, rv_t, ru, rv, wwI, c1h, c2h, &
                         muv_old, muv, muv_new, cf, msfu, msfu, msfv, msfv, &
                         msft, msft, fnm, fnp, dt, rdx, rdy, rdnw, &
                         ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
                         its, ite, jts, jte, kts, kte)
  call write3('rv_t', rv_t, nz, ny + 1, nx)
  call advect_s_implicit(dum3, s_old, rth_t, ru, rv, wwI, c1h, c2h, &
                         mut_old, mut, mut_new, cf, msfu, msfu, msfv, msfv, &
                         msft, msft, fnm, fnp, dt, rdx, rdy, rdnw, &
                         ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
                         its, ite, jts, jte, kts, kte)
  call write3('rth_t', rth_t, nz, ny, nx)
  call advect_ph_implicit(ph, ph_old, rph_t, phb, ru, rv, wwE, wwI, dum3, &
                          c1f, c2f, mut, cf, msfu, msfu, msfv, msfv, msft, msft, &
                          fnm, fnp, dt, rdx, rdy, rdnw, &
                          ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
                          its, ite, jts, jte, kts, kte)
  call write3('rph_t', rph_t, nz + 1, ny, nx)
  ! advect_w_implicit reads the u/v/ph tendencies after their own solves;
  ! each is read back from gpuwm so the w solve is graded on its own input.
  if (.not. chain) then
    call read3('ru_t_ieva', ru_t, nz, ny, nx + 1)
    call read3('rv_t_ieva', rv_t, nz, ny + 1, nx)
    call read3('rph_t_ieva', rph_t, nz + 1, ny, nx)
  end if
#ifdef A179
  ! WRF's routine with A179's two corrections (wrf_a179.py): the lower
  ! boundary uncouples the u/v tendencies with the stage face masses muu
  ! and muv, so it takes those and c1h/c2h.
  call advect_w_implicit(dum3, w_old, rw_t, ru_t, rv_t, ht, wwI, &
                         ph, ph_old, rph_t, c1f, c2f, cf1, cf2, cf3, &
                         c1h, c2h, muu, muv, &
                         mut_old, mut, mut_new, cf, msfu, msfu, msfv, msfv, &
                         msft, msft, fnm, fnp, dt, rdx, rdy, rdn, &
                         ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
                         its, ite, jts, jte, kts, kte)
#else
  call advect_w_implicit(dum3, w_old, rw_t, ru_t, rv_t, ht, wwI, &
                         ph, ph_old, rph_t, c1f, c2f, cf1, cf2, cf3, &
                         mut_old, mut, mut_new, cf, msfu, msfu, msfv, msfv, &
                         msft, msft, fnm, fnp, dt, rdx, rdy, rdn, &
                         ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
                         its, ite, jts, jte, kts, kte)
#endif
  call write3('rw_t', rw_t, nz + 1, ny, nx)

  ! ---- rk_scalar_tend, module_em.F:1216-1364 (first moist species) -----
  call read3('ww_m', ww_m, nz + 1, ny, nx)
  call read2('muts', muts, ny, nx)
  call read2('mu0s', mu0s, ny, nx)
  call WW_SPLIT(wwE_m, wwI_m, u_old, v_old, ww_m, muts, rdnw, msft, c1f, c2f, &
                rdx, rdy, msfu, msfu, msfv, msfv, dt_s, cf, 3, &
                ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
                its, ite, jts, jte, kts, kte)
  call write3('wwE_m', wwE_m, nz + 1, ny, nx)
  call write3('wwI_m', wwI_m, nz + 1, ny, nx)
  if (.not. chain) call read3('wwI_m', wwI_m, nz + 1, ny, nx)
  call read3('q_old', q_old, nz, ny, nx)
  call read3('q_tend_explicit', q_tend, nz, ny, nx)
  call advect_s_implicit(dum3, q_old, q_tend, ru, rv, wwI_m, c1h, c2h, &
                         mu0s, muts, muts, cf, msfu, msfu, msfv, msfv, &
                         msft, msft, fnm, fnp, dt_s, rdx, rdy, rdnw, &
                         ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
                         its, ite, jts, jte, kts, kte)
  call write3('q_tend', q_tend, nz, ny, nx)
  print *, 'IEVA_ORACLE_DONE'

contains

  subroutine alloc3(a)
    real, allocatable, intent(inout) :: a(:,:,:)
    allocate(a(ims:ime, kms:kme, jms:jme)); a = 0.
  end subroutine

  subroutine alloc2(a)
    real, allocatable, intent(inout) :: a(:,:)
    allocate(a(ims:ime, jms:jme)); a = 0.
  end subroutine

  subroutine read3(name, a, nl, nyy, nxx)
    character(len=*), intent(in) :: name
    real, intent(inout) :: a(ims:, kms:, jms:)
    integer, intent(in) :: nl, nyy, nxx
    real, allocatable :: t(:,:,:)
    integer :: ii, jj, kk, ic, jc
    allocate(t(nxx, nyy, nl))
    open(11, file=trim(dir)//'/'//name//'.bin', access='stream', &
         form='unformatted', status='old')
    read(11) t
    close(11)
    do jj = jms, jme
      jc = max(1, min(nyy, jj))
      do kk = 1, nl
        do ii = ims, ime
          ic = max(1, min(nxx, ii))
          a(ii, kk, jj) = t(ic, jc, kk)
        end do
      end do
    end do
    deallocate(t)
  end subroutine

  subroutine read2(name, a, nyy, nxx)
    character(len=*), intent(in) :: name
    real, intent(inout) :: a(ims:, jms:)
    integer, intent(in) :: nyy, nxx
    real, allocatable :: t(:,:)
    integer :: ii, jj
    allocate(t(nxx, nyy))
    open(11, file=trim(dir)//'/'//name//'.bin', access='stream', &
         form='unformatted', status='old')
    read(11) t
    close(11)
    do jj = jms, jme
      do ii = ims, ime
        a(ii, jj) = t(max(1, min(nxx, ii)), max(1, min(nyy, jj)))
      end do
    end do
    deallocate(t)
  end subroutine

  subroutine halo2(a, nxx, nyy)
    real, intent(inout) :: a(ims:, jms:)
    integer, intent(in) :: nxx, nyy
    integer :: ii, jj
    do jj = jms, jme
      do ii = ims, ime
        a(ii, jj) = a(max(1, min(nxx, ii)), max(1, min(nyy, jj)))
      end do
    end do
  end subroutine

  subroutine read1(name, a, n)
    character(len=*), intent(in) :: name
    real, intent(inout) :: a(kms:)
    integer, intent(in) :: n
    open(11, file=trim(dir)//'/'//name//'.bin', access='stream', &
         form='unformatted', status='old')
    read(11) a(1:n)
    close(11)
  end subroutine

  subroutine write3(name, a, nl, nyy, nxx)
    character(len=*), intent(in) :: name
    real, intent(in) :: a(ims:, kms:, jms:)
    integer, intent(in) :: nl, nyy, nxx
    real, allocatable :: t(:,:,:)
    integer :: ii, jj, kk
    allocate(t(nxx, nyy, nl))
    do kk = 1, nl
      do jj = 1, nyy
        do ii = 1, nxx
          t(ii, jj, kk) = a(ii, kk, jj)
        end do
      end do
    end do
    open(12, file=trim(dir)//'/wrf_'//name//'.bin', access='stream', &
         form='unformatted', status='replace')
    write(12) t
    close(12)
    deallocate(t)
  end subroutine

  subroutine write2(name, a, nyy, nxx)
    character(len=*), intent(in) :: name
    real, intent(in) :: a(ims:, jms:)
    integer, intent(in) :: nyy, nxx
    open(12, file=trim(dir)//'/wrf_'//name//'.bin', access='stream', &
         form='unformatted', status='replace')
    write(12) a(1:nxx, 1:nyy)
    close(12)
  end subroutine

end program ieva_oracle
