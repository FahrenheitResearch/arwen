! Drive WRF 4.7.1's own compiled w_damp (module_big_step_utilities_em.o
! out of main/libwrflib.a) on the arrays synth.py writes, once per
! (w_crit_cfl, zadvect_implicit) case, and write the rw_tend WRF computes
! so tests/test_w_crit_cfl.py can compare gpuwm's CUDA kernel word for word.
!
! Input: <dir>/meta.txt and <dir>/<name>.bin (float32, C order (k, j, i)).
! Output: <dir>/wrf_rw_t_<case>.bin in the same layout, and
! <dir>/wrf_max_cfl.txt (case, max_vert_cfl, max_horiz_cfl).
!
! One tile of one patch: ids = its = 1, ide = ite = nx + 1, kds = kts = 1,
! kde = kte = nz + 1, memory = domain (w_damp reads no neighbour).
! config_flags: w_damping 1, polar false; w_crit_cfl and zadvect_implicit
! per case.
program w_damp_oracle
  use module_configure, only: grid_config_rec_type
  use module_big_step_utilities_em, only: w_damp
  implicit none

  type(grid_config_rec_type) :: cf
  character(len=512) :: dir
  character(len=32) :: tag
  integer :: nx, ny, nz, ncase, icase, ieva
  real :: dt, dx, dy, rdx, rdy, crit, max_vert_cfl, max_horiz_cfl
  integer :: ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme
  integer :: its, ite, jts, jte, kts, kte

  real, allocatable, dimension(:,:,:) :: u, v, ww, w, rw0, rw
  real, allocatable, dimension(:,:) :: mut, msfux, msfuy, msfvx, msfvy
  real, allocatable, dimension(:) :: rdnw, c1f, c2f

  call get_command_argument(1, dir)
  open(10, file=trim(dir)//'/meta.txt', status='old')
  read(10, *) nx, ny, nz, ncase
  read(10, *) dt, dx, dy
  ids = 1; ide = nx + 1; jds = 1; jde = ny + 1; kds = 1; kde = nz + 1
  its = ids; ite = ide; jts = jds; jte = jde; kts = kds; kte = kde
  ims = ids; ime = ide; jms = jds; jme = jde; kms = kds; kme = kde
  rdx = 1./dx
  rdy = 1./dy

  allocate(u(ims:ime, kms:kme, jms:jme), v(ims:ime, kms:kme, jms:jme), &
           ww(ims:ime, kms:kme, jms:jme), w(ims:ime, kms:kme, jms:jme), &
           rw0(ims:ime, kms:kme, jms:jme), rw(ims:ime, kms:kme, jms:jme))
  allocate(mut(ims:ime, jms:jme), msfux(ims:ime, jms:jme), &
           msfuy(ims:ime, jms:jme), msfvx(ims:ime, jms:jme), &
           msfvy(ims:ime, jms:jme))
  allocate(rdnw(kms:kme), c1f(kms:kme), c2f(kms:kme))
  u = 0.; v = 0.; ww = 0.; w = 0.; rw0 = 0.; mut = 0.
  msfux = 1.; msfuy = 1.; msfvx = 1.; msfvy = 1.
  rdnw = 0.; c1f = 0.; c2f = 0.

  call read3('u', u, nz, ny, nx + 1)
  call read3('v', v, nz, ny + 1, nx)
  call read3('ww', ww, nz + 1, ny, nx)
  call read3('w', w, nz + 1, ny, nx)
  call read3('rw_t', rw0, nz + 1, ny, nx)
  call read2('mut', mut, ny, nx)
  call read1('rdnw', rdnw, nz)
  call read1('c1f', c1f, nz + 1)
  call read1('c2f', c2f, nz + 1)

  cf%w_damping = 1
  cf%polar = .false.
  cf%fft_filter_lat = 45.
  open(13, file=trim(dir)//'/wrf_max_cfl.txt', status='replace')
  do icase = 1, ncase
    read(10, *) tag, crit, ieva
    cf%w_crit_cfl = crit
    cf%zadvect_implicit = ieva
    rw = rw0
    call w_damp(rw, max_vert_cfl, max_horiz_cfl, u, v, ww, w, mut, &
                c1f, c2f, rdnw, rdx, rdy, msfux, msfuy, msfvx, msfvy, dt, &
                cf, ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, &
                kms, kme, its, ite, jts, jte, kts, kte)
    call write3('rw_t_'//trim(tag), rw, nz + 1, ny, nx)
    write(13, '(a, 2(1x, es24.16))') trim(tag), max_vert_cfl, max_horiz_cfl
  end do
  close(13)
  close(10)
  print *, 'W_DAMP_ORACLE_DONE'

contains

  subroutine read3(name, a, nl, nyy, nxx)
    character(len=*), intent(in) :: name
    real, intent(inout) :: a(ims:, kms:, jms:)
    integer, intent(in) :: nl, nyy, nxx
    real, allocatable :: t(:,:,:)
    integer :: ii, jj, kk
    allocate(t(nxx, nyy, nl))
    open(11, file=trim(dir)//'/'//name//'.bin', access='stream', &
         form='unformatted', status='old')
    read(11) t
    close(11)
    do jj = 1, nyy
      do kk = 1, nl
        do ii = 1, nxx
          a(ii, kk, jj) = t(ii, jj, kk)
        end do
      end do
    end do
    deallocate(t)
  end subroutine

  subroutine read2(name, a, nyy, nxx)
    character(len=*), intent(in) :: name
    real, intent(inout) :: a(ims:, jms:)
    integer, intent(in) :: nyy, nxx
    open(11, file=trim(dir)//'/'//name//'.bin', access='stream', &
         form='unformatted', status='old')
    read(11) a(1:nxx, 1:nyy)
    close(11)
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

end program w_damp_oracle
