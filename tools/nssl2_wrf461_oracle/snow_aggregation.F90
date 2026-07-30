program nssl2_snow_aggregation_oracle
  use module_mp_nssl_2mom, only: nssl_2mom_init, nssl_2mom_gs
  implicit none

  integer, parameter :: nx = 4, nz = 12, na = 40, nxtra = 20
  integer, parameter :: lc = 3, lr = 4, li = 5, ls = 6, lh = 7
  integer, parameter :: lhl = 8, lqmx = 30, lt = 1, lv = 2, lns = 13
  integer :: i, k, unit, nml_unit, case_id, repetition
  real :: nssl_params(20), dt, temperature, target_diameter
  real :: snow_volume, rho, before_qs, before_ns
  real :: xdnmx(lc:lhl), xdnmn(lc:lhl), xdn0(lc:lhl), cdx(lc:lhl)
  integer :: ido(lc:lqmx)
  double precision :: timevtcalc
  character(len=512) :: output_path

  call get_command_argument(1, output_path)
  if (len_trim(output_path) == 0) then
     error stop 'usage: nssl2_snow_aggregation_oracle OUTPUT.csv'
  endif

  nssl_params = 0.0
  nssl_params(1:10) = [0.5e9, 0.0, 1.0, 4.0e5, 4.0e4, &
                       8.0e5, 3.0e6, 500.0, 900.0, 100.0]

  ! Preserve the default aggregation coefficients while disabling the only
  ! snow-only neighboring tendencies: deposition/sublimation, collisional
  ! fragmentation, and primary ice nucleation.  Other species remain empty.
  open(newunit=nml_unit, file='namelist.input', status='replace', action='write')
  write(nml_unit,'(A)') '&nssl_mp_params'
  write(nml_unit,'(A)') '  depfac = 0.0,'
  write(nml_unit,'(A)') '  isnwfrac = 0,'
  write(nml_unit,'(A)') '  icenucopt = 0,'
  write(nml_unit,'(A)') '/'
  close(nml_unit)

  call nssl_2mom_init(nssl_params=nssl_params, ipctmp=5, mixphase=0, &
       nssl_density_on=.true., nssl_hail_on=.true.,                 &
       nssl_ccn_on=.true., nssl_icdx=6, nssl_icdxhl=6)

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

  open(newunit=unit, file=trim(output_path), status='replace', action='write')
  write(unit,'(A)') 'case,i,k,dt_s,rho_kg_m3,temperature_k,target_snow_diameter_m,qs_before,qns_before_per_kg,qs_after,qns_after_per_kg'
  do k = 1, nz
     do i = 1, nx
        case_id = mod((k-1)*nx+i-1, 12)
        repetition = ((k-1)*nx+i-1)/12
        select case (case_id)
        case (0)
           temperature = 258.14; target_diameter = 0.30e-3; dt = 10.0
        case (1)
           temperature = 258.15; target_diameter = 0.30e-3; dt = 10.0
        case (2)
           temperature = 258.16; target_diameter = 0.30e-3; dt = 10.0
        case (3)
           temperature = 263.14; target_diameter = 0.50e-3; dt = 10.0
        case (4)
           temperature = 263.15; target_diameter = 0.50e-3; dt = 10.0
        case (5)
           temperature = 263.16; target_diameter = 0.50e-3; dt = 10.0
        case (6)
           temperature = 273.14; target_diameter = 1.00e-3; dt = 10.0
        case (7)
           temperature = 273.15; target_diameter = 1.00e-3; dt = 10.0
        case (8)
           temperature = 260.00; target_diameter = 9.00e-6; dt = 5.0
        case (9)
           temperature = 268.00; target_diameter = 9.50e-3; dt = 30.0
        case (10)
           temperature = 268.00; target_diameter = 10.10e-3; dt = 60.0
        case default
           temperature = 268.00; target_diameter = 1.00e-3; dt = 1000.0
        end select

        select case (repetition)
        case (0)
           before_qs = 0.0
        case (1)
           before_qs = 1.0e-13
        case (2)
           before_qs = 2.5e-5
        case default
           before_qs = 2.5e-3
        end select
        rho = 1.30 - 0.018*real(k-1) + 0.007*real(i-1)
        snow_volume = 0.523599*target_diameter**3
        before_ns = before_qs/(100.0*snow_volume)
        call run_cell(unit)
     enddo
  enddo
  close(unit)

  print '(A,1X,A)', 'NSSL2_SNOW_AGGREGATION_ORACLE_COMPLETE', &
       trim(output_path)

contains

  subroutine run_cell(output_unit)
    integer, intent(in) :: output_unit
    real :: a1(1,1,3,na)
    real :: u0(1,1,3), u1(1,1,3), u2(1,1,3), u3(1,1,3)
    real :: u4(1,1,3), u5(1,1,3), u6(1,1,3), u7(1,1,3)
    real :: u8(1,1,3), u9(1,1,3), uu0(1,1,3), uu7(1,1,3)
    real :: zg(1,1,3), dd(1,1,3), pp2(1,1,3), ppn(1,1,3)
    real :: ww(1,1,3), tt3(1,1,3), tee(1,3), aa(1,1,3,nxtra)
    real :: rp(1,3), ep(1,3), alp(1,3,3), el(1,1,3), th(3,1)

    a1 = 0.0
    a1(:,:,:,lt) = temperature
    a1(:,:,:,lv) = 0.0
    a1(:,:,:,ls) = before_qs
    ! Registry #/kg -> internal #/m3 for nssl_2mom_gs.
    a1(:,:,:,lns) = before_ns*rho
    u0 = temperature
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

    call nssl_2mom_gs(1,1,3,na,1,0,0,dt,zg,                &
         u0,u1,u2,u3,u4,u5,u6,u7,u8,u9,a1,dd,pp2,ppn,ww,0, &
         uu0,uu7,1.0,1.0,1.0,1,ido,xdnmx,xdnmn,cdx,xdn0,  &
         tt3,tee,th,1,1000.0,1000.0,3,timevtcalc,aa,.false.,&
         .false.,rp,ep,alp,el,1,1,1,1,1)

    write(output_unit,'(3(I0,","),7(ES24.16E3,","),ES24.16E3)') &
         case_id, i, k, dt, rho, temperature, target_diameter,      &
         before_qs, before_ns, a1(1,1,2,ls), a1(1,1,2,lns)/rho
  end subroutine run_cell
end program nssl2_snow_aggregation_oracle
