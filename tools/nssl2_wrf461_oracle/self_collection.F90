program nssl2_self_collection_oracle
  use module_mp_nssl_2mom, only: nssl_2mom_init, nssl_2mom_gs
  implicit none

  integer, parameter :: nx = 4, nz = 12, na = 40, nxtra = 20
  integer, parameter :: lc = 3, lr = 4, li = 5, ls = 6, lh = 7
  integer, parameter :: lhl = 8, lqmx = 30, lt = 1, lnr = 11
  integer :: i, k, unit, case_id
  real :: nssl_params(20), dt, target_diameter, volume
  real :: density2d(nx,nz+1), qr(nx,nz), nr(nx,nz)
  real :: xdnmx(lc:lhl), xdnmn(lc:lhl), xdn0(lc:lhl)
  real :: cdx(lc:lhl), time_seconds(nx,nz)
  integer :: ido(lc:lqmx)
  double precision :: timevtcalc
  character(len=512) :: output_path

  call get_command_argument(1, output_path)
  if (len_trim(output_path) == 0) then
     error stop 'usage: nssl2_self_collection_oracle OUTPUT.csv'
  endif

  nssl_params = 0.0
  nssl_params(1:10) = [0.5e9, 0.0, 1.0, 4.0e5, 4.0e4, &
                       8.0e5, 3.0e6, 500.0, 900.0, 100.0]
  call nssl_2mom_init(nssl_params=nssl_params, ipctmp=5, mixphase=0, &
       nssl_density_on=.true., nssl_hail_on=.true.,                 &
       nssl_ccn_on=.true., nssl_icdx=6, nssl_icdxhl=6)

  qr = 0.0
  nr = 0.0
  density2d = 1.0
  time_seconds = 0.0
  do k = 1, nz + 1
     do i = 1, nx
        density2d(i,k) = 1.30 - 0.075*real(k-1) + 0.01*real(i-1)
     enddo
  enddo
  do k = 1, nz
     do i = 1, nx
        case_id = mod((k-1)*nx+i-1, 12)
        select case (case_id)
        case (0)
           target_diameter = 79.0e-6
           dt = 0.05
        case (1)
           target_diameter = 80.0e-6
           dt = 0.05
        case (2)
           target_diameter = 99.0e-6
           dt = 0.10
        case (3)
           target_diameter = 100.0e-6
           dt = 0.10
        case (4)
           target_diameter = 250.0e-6
           ! Exercise WRF's process-level number-depletion limiter.
           dt = 2000.0
        case (5)
           target_diameter = 599.0e-6
           dt = 0.10
        case (6)
           target_diameter = 610.0e-6
           dt = 0.10
        case (7)
           target_diameter = 900.0e-6
           dt = 0.25
        case (8)
           target_diameter = 1.50e-3
           dt = 0.50
        case (9)
           target_diameter = 1.99e-3
           dt = 1.00
        case (10)
           target_diameter = 2.01e-3
           dt = 10.0
        case default
           target_diameter = 2.50e-3
           dt = 60.0
        end select
        qr(i,k) = 2.5e-5 * 1.55**real(mod(case_id,6))
        volume = 0.523599 * target_diameter**3
        nr(i,k) = qr(i,k)/(1000.0*volume)
        time_seconds(i,k) = dt
     enddo
  enddo

  ! nssl_2mom_gs accepts one timestep for the full slab.  Run each fixture
  ! cell as its own one-column slab below so dt remains part of the vector.
  open(newunit=unit, file=trim(output_path), status='replace', action='write')
  write(unit,'(A)') 'case,i,k,dt_s,rho_kg_m3,qr_before,nr_before_per_kg,qr_after,nr_after_per_kg'
  do k = 1, nz
     do i = 1, nx
        call run_cell(i, k, unit)
     enddo
  enddo
  close(unit)

  print '(A,1X,A)', 'NSSL2_SELF_COLLECTION_ORACLE_COMPLETE', trim(output_path)

contains

  subroutine run_cell(ii, kk, output_unit)
    integer, intent(in) :: ii, kk, output_unit
    real :: a1(1,1,3,na)
    real :: u0(1,1,3), u1(1,1,3), u2(1,1,3), u3(1,1,3)
    real :: u4(1,1,3), u5(1,1,3), u6(1,1,3), u7(1,1,3)
    real :: u8(1,1,3), u9(1,1,3), uu0(1,1,3), uu7(1,1,3)
    real :: zg(1,1,3), dd(1,1,3), pp2(1,1,3), ppn(1,1,3)
    real :: ww(1,1,3), tt3(1,1,3), tee(1,3), aa(1,1,3,nxtra)
    real :: rp(1,3), ep(1,3), alp(1,3,3), el(1,1,3), th(3,1)
    real :: rho, before_qr, before_nr, step
    integer :: c

    c = mod((kk-1)*nx+ii-1, 12)
    rho = density2d(ii,kk)
    before_qr = qr(ii,kk)
    before_nr = nr(ii,kk)
    step = time_seconds(ii,kk)

    a1 = 0.0
    a1(:,:,:,lt) = 300.0
    a1(:,:,:,2) = 0.1
    a1(:,:,:,lr) = before_qr
    ! This is the driver's exact Registry-mixing-ratio to internal-number-
    ! concentration conversion at module_mp_nssl_2mom.F:2933-2944.
    a1(:,:,:,lnr) = before_nr*rho

    u0 = 300.0
    u1 = 0.0
    u2 = 0.0
    u3 = 0.0
    u4 = 0.0
    u5 = 0.0
    u6 = 0.0
    u7 = 0.0
    u8 = 0.0
    u9 = 0.0
    uu0 = 380.0/100000.0
    uu7 = 1.0
    zg = 1000.0
    dd = rho
    pp2 = 1.0
    ppn = 100000.0
    ww = 0.0
    tt3 = 0.0
    tee = 0.0
    aa = 0.0
    rp = 0.0
    ep = 0.0
    alp = 0.0
    el = 0.0
    th = 0.0
    timevtcalc = 0.0d0

    xdnmx = 900.0
    xdnmx(lc) = 1000.0
    xdnmx(lr) = 1000.0
    xdnmx(li) = 917.0
    xdnmx(ls) = 300.0
    xdnmn = 900.0
    xdnmn(lc) = 1000.0
    xdnmn(lr) = 1000.0
    xdnmn(li) = 100.0
    xdnmn(ls) = 100.0
    xdnmn(lh) = 170.0
    xdnmn(lhl) = 500.0
    xdn0 = 900.0
    xdn0(lc) = 1000.0
    xdn0(lr) = 1000.0
    xdn0(li) = 900.0
    xdn0(ls) = 100.0
    xdn0(lh) = 500.0
    xdn0(lhl) = 900.0
    cdx = 0.6
    cdx(ls) = 2.0
    cdx(lh) = 0.8
    cdx(lhl) = 0.45
    ido = 1

    call nssl_2mom_gs(1,1,3,na,1,0,0,step,zg,              &
         u0,u1,u2,u3,u4,u5,u6,u7,u8,u9,a1,dd,pp2,ppn,ww,0, &
         uu0,uu7,1.0,1.0,1.0,1,ido,xdnmx,xdnmn,cdx,xdn0,  &
         tt3,tee,th,1,1000.0,1000.0,3,timevtcalc,aa,.false.,&
         .false.,rp,ep,alp,el,1,1,1,1,1)

    write(output_unit,'(I0,2(",",I0),6(",",ES24.16E3))') &
         c, ii, kk, step, rho, before_qr, before_nr,         &
         a1(1,1,2,lr), a1(1,1,2,lnr)/rho
  end subroutine run_cell
end program nssl2_self_collection_oracle
