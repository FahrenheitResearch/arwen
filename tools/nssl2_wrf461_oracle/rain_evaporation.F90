program nssl2_rain_evaporation_oracle
  use module_mp_nssl_2mom, only: nssl_2mom_init, nssl_2mom_gs
  implicit none

  integer, parameter :: nx = 4, nz = 12, na = 40, nxtra = 20
  integer, parameter :: lc = 3, lr = 4, li = 5, ls = 6, lh = 7
  integer, parameter :: lhl = 8, lqmx = 30, lt = 1, lv = 2
  integer, parameter :: lnr = 11
  integer :: i, k, unit, nml_unit, case_id, ltemq
  real :: nssl_params(20), dt, pressure, exner, rho, temperature
  real :: relative_humidity, table_temperature, qvs
  real :: rain_diameter, rain_volume, before_qv, before_theta
  real :: before_qr, before_nr, after_qv, after_theta, after_qr, after_nr
  real :: xdnmx(lc:lhl), xdnmn(lc:lhl), xdn0(lc:lhl), cdx(lc:lhl)
  integer :: ido(lc:lqmx)
  double precision :: timevtcalc
  character(len=512) :: output_path

  call get_command_argument(1, output_path)
  if (len_trim(output_path) == 0) then
     error stop 'usage: nssl2_rain_evaporation_oracle OUTPUT.csv'
  endif

  nssl_params = 0.0
  nssl_params(1:10) = [0.5e9, 0.0, 1.0, 4.0e5, 4.0e4, &
                       8.0e5, 3.0e6, 500.0, 900.0, 100.0]

  ! With only warm rain present, these switches remove the remaining
  ! neighboring warm-rain sources without changing default evaporation.
  open(newunit=nml_unit, file='namelist.input', status='replace', action='write')
  write(nml_unit,'(A)') '&nssl_mp_params'
  write(nml_unit,'(A)') '  dmrauto = -2,'
  write(nml_unit,'(A)') '  icracr = 0,'
  write(nml_unit,'(A)') '  evapfac = 1.0,'
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
  write(unit,'(A)') 'case,i,k,dt_s,pressure_pa,exner,rho_kg_m3,temperature_k,relative_humidity,qv_before,theta_before_k,qr_before,nr_before_per_kg,qv_after,theta_after_k,qr_after,nr_after_per_kg'

  do k = 1, nz
     do i = 1, nx
        case_id = mod((k-1)*nx+i-1, 12)
        select case (case_id)
        case (0)
           temperature = 280.0; pressure = 100000.0
           relative_humidity = 0.40; rain_diameter = 0.30e-3
           before_qr = 0.0; dt = 1.0
        case (1)
           temperature = 275.0; pressure = 90000.0
           relative_humidity = 0.20; rain_diameter = 0.080e-3
           before_qr = 5.0e-6; dt = 0.1
        case (2)
           temperature = 282.0; pressure = 80000.0
           relative_humidity = 0.55; rain_diameter = 0.10e-3
           before_qr = 2.0e-5; dt = 1.0
        case (3)
           temperature = 288.0; pressure = 95000.0
           relative_humidity = 0.80; rain_diameter = 0.30e-3
           before_qr = 1.0e-4; dt = 5.0
        case (4)
           temperature = 295.0; pressure = 100000.0
           relative_humidity = 0.95; rain_diameter = 0.70e-3
           before_qr = 5.0e-4; dt = 10.0
        case (5)
           temperature = 305.0; pressure = 90000.0
           relative_humidity = 0.30; rain_diameter = 1.50e-3
           before_qr = 2.0e-3; dt = 60.0
        case (6)
           temperature = 300.0; pressure = 70000.0
           relative_humidity = 0.99; rain_diameter = 2.65e-3
           before_qr = 1.0e-3; dt = 30.0
        case (7)
           temperature = 278.0; pressure = 65000.0
           relative_humidity = 0.10; rain_diameter = 0.15e-3
           before_qr = 5.0e-5; dt = 20.0
        case (8)
           temperature = 285.0; pressure = 60000.0
           relative_humidity = 0.65; rain_diameter = 0.50e-3
           before_qr = 8.0e-4; dt = 2.0
        case (9)
           temperature = 292.0; pressure = 85000.0
           relative_humidity = 0.50; rain_diameter = 1.00e-3
           before_qr = 1.0e-4; dt = 0.5
        case (10)
           temperature = 298.0; pressure = 75000.0
           relative_humidity = 0.75; rain_diameter = 2.00e-3
           before_qr = 3.0e-3; dt = 120.0
        case default
           temperature = 310.0; pressure = 100000.0
           relative_humidity = 0.01; rain_diameter = 2.70e-3
           before_qr = 1.0e-2; dt = 300.0
        end select

        ! Repeat each branch at four nearby thermodynamic states while
        ! retaining the exact 0.002-K table quantization used by WRF.
        temperature = temperature - 0.137*real((k-1)/3)
        pressure = pressure - 317.0*real((k-1)/3)
        ltemq = int((temperature-163.15)/0.002 + 1.5)
        ltemq = min(1000001, max(1, ltemq))
        table_temperature = 163.15 + real(ltemq-1)*0.002
        qvs = (380.0/pressure)*exp(17.2693882* &
             (table_temperature-273.15)/(table_temperature-35.86))
        before_qv = relative_humidity*qvs
        exner = (pressure/100000.0)**(287.04/1004.0)
        before_theta = temperature/exner
        rho = pressure/(287.04*temperature*(1.0 + 0.608*before_qv))
        rain_volume = 0.523599*rain_diameter**3
        if (before_qr > 0.0) then
           before_nr = before_qr/(1000.0*rain_volume)
        else
           before_nr = 0.0
        endif

        call run_case(after_qv, after_theta, after_qr, after_nr)

        write(unit,'(3(I0,","),13(ES24.16E3,","),ES24.16E3)') &
             case_id, i, k, dt, pressure, exner, rho, temperature, &
             relative_humidity, before_qv, before_theta, before_qr, &
             before_nr, after_qv, after_theta, after_qr, after_nr
     enddo
  enddo
  close(unit)

  print '(A,1X,A)', 'NSSL2_RAIN_EVAPORATION_ORACLE_COMPLETE', &
       trim(output_path)

contains

  subroutine run_case(output_qv, output_theta, output_qr, output_nr)
    real, intent(out) :: output_qv, output_theta, output_qr, output_nr
    real :: a1(1,1,3,na)
    real :: u0(1,1,3), u1(1,1,3), u2(1,1,3), u3(1,1,3)
    real :: u4(1,1,3), u5(1,1,3), u6(1,1,3), u7(1,1,3)
    real :: u8(1,1,3), u9(1,1,3), uu0(1,1,3), uu7(1,1,3)
    real :: zg(1,1,3), dd(1,1,3), pp2(1,1,3), ppn(1,1,3)
    real :: ww(1,1,3), tt3(1,1,3), tee(1,3), aa(1,1,3,nxtra)
    real :: rp(1,3), ep(1,3), alp(1,3,3), el(1,1,3), th(3,1)

    a1 = 0.0
    a1(:,:,:,lt) = before_theta
    a1(:,:,:,lv) = before_qv
    a1(:,:,:,lr) = before_qr
    a1(:,:,:,lnr) = before_nr*rho
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
    uu0 = 380.0/pressure
    uu7 = exner
    zg = 1000.0
    dd = rho
    pp2 = exner
    ppn = pressure
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

    output_theta = a1(1,1,2,lt)
    output_qv = a1(1,1,2,lv)
    output_qr = a1(1,1,2,lr)
    output_nr = a1(1,1,2,lnr)/rho
  end subroutine run_case
end program nssl2_rain_evaporation_oracle
