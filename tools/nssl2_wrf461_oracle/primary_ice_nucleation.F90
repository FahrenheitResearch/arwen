program nssl2_primary_ice_nucleation_oracle
  use module_mp_nssl_2mom, only: nssl_2mom_init, nssl_2mom_gs
  implicit none

  integer, parameter :: nx = 4, nz = 12, na = 40, nxtra = 20
  integer, parameter :: lc = 3, lr = 4, li = 5, ls = 6, lh = 7
  integer, parameter :: lhl = 8, lqmx = 30, lt = 1, lv = 2, lni = 12
  integer :: i, k, unit, nml_unit, case_id, repetition, table_index
  real :: nssl_params(20), dt, temperature, pressure, exner, rho
  real :: rh_ice, table_temperature, qsi, before_qv, vertical_velocity
  real :: layer_depth, nuclei_minus, nuclei_center, nuclei_plus
  real, parameter :: steps(4) = [0.1, 1.0, 10.0, 60.0]
  real, parameter :: depths(4) = [100.0, 500.0, 1000.0, 2000.0]
  real :: xdnmx(lc:lhl), xdnmn(lc:lhl), xdn0(lc:lhl), cdx(lc:lhl)
  integer :: ido(lc:lqmx)
  double precision :: timevtcalc
  character(len=512) :: output_path

  call get_command_argument(1, output_path)
  if (len_trim(output_path) == 0) then
     error stop 'usage: nssl2_primary_ice_nucleation_oracle OUTPUT.csv'
  endif

  nssl_params = 0.0
  nssl_params(1:10) = [0.5e9, 0.0, 1.0, 4.0e5, 4.0e4, &
                       8.0e5, 3.0e6, 500.0, 900.0, 100.0]
  open(newunit=nml_unit, file='namelist.input', status='replace', action='write')
  write(nml_unit,'(A)') '&nssl_mp_params'
  write(nml_unit,'(A)') '  icenucopt = 1,'
  write(nml_unit,'(A)') '  isnwfrac = 0,'
  write(nml_unit,'(A)') '/'
  close(nml_unit)
  call nssl_2mom_init(nssl_params=nssl_params, ipctmp=5, mixphase=0, &
       nssl_density_on=.true., nssl_hail_on=.true.,                 &
       nssl_ccn_on=.true., nssl_icdx=6, nssl_icdxhl=6)

  xdnmx = 900.0; xdnmx(lc) = 1000.0; xdnmx(lr) = 1000.0
  xdnmx(li) = 917.0; xdnmx(ls) = 300.0
  xdnmn = 900.0; xdnmn(lc) = 1000.0; xdnmn(lr) = 1000.0
  xdnmn(li) = 100.0; xdnmn(ls) = 100.0
  xdnmn(lh) = 170.0; xdnmn(lhl) = 500.0
  xdn0 = 900.0; xdn0(lc) = 1000.0; xdn0(lr) = 1000.0
  xdn0(li) = 900.0; xdn0(ls) = 100.0
  xdn0(lh) = 500.0; xdn0(lhl) = 900.0
  cdx = 0.6; cdx(ls) = 2.0; cdx(lh) = 0.8; cdx(lhl) = 0.45
  ido = 1

  open(newunit=unit, file=trim(output_path), status='replace', action='write')
  write(unit,'(A)') 'case,repetition,dt_s,rho_kg_m3,pressure_pa,exner,temperature_k,rh_ice,w_m_s,dz_m,nuclei_minus_m3,nuclei_center_m3,nuclei_plus_m3,theta_before_k,qv_before,qi_before,qni_before_per_kg,theta_after_k,qv_after,qi_after,qni_after_per_kg,qs_after,qns_after_per_kg'

  do k = 1, nz
     do i = 1, nx
        case_id = mod((k-1)*nx+i-1, 12)
        repetition = ((k-1)*nx+i-1)/12
        dt = steps(repetition+1)
        pressure = 95000.0 - 8000.0*real(repetition) - 300.0*real(case_id)
        rho = 1.20 - 0.13*real(repetition) + 0.002*real(case_id)
        layer_depth = depths(repetition+1)

        temperature = 250.0
        rh_ice = 1.05
        vertical_velocity = 1.0
        nuclei_minus = 1.0e3
        nuclei_center = 2.0e3
        nuclei_plus = 3.0e3 + 1.0e3*real(repetition)
        select case (case_id)
        case (0)
           temperature = 250.0
        case (1)
           temperature = 260.0; vertical_velocity = 0.0
        case (2)
           temperature = 260.0; vertical_velocity = -2.0
        case (3)
           temperature = 255.0
           nuclei_minus = 5.0e3; nuclei_center = 3.0e3; nuclei_plus = 1.0e3
        case (4)
           temperature = 268.16
        case (5)
           temperature = 245.0; rh_ice = 0.99
        case (6)
           temperature = 267.9; rh_ice = 1.0001; vertical_velocity = 0.2
        case (7)
           temperature = 240.0; vertical_velocity = 10.0
           nuclei_minus = 1.0e3; nuclei_center = 5.0e8; nuclei_plus = 1.0e9
        case (8)
           temperature = 235.0; vertical_velocity = 5.0
           nuclei_minus = 0.0; nuclei_center = 2.5e5; nuclei_plus = 5.0e5
        case (9)
           temperature = 265.0; vertical_velocity = 3.0; rh_ice = 1.001
           nuclei_minus = 100.0; nuclei_center = 1.0e5; nuclei_plus = 2.0e5
        case (10)
           temperature = 255.0; vertical_velocity = 0.05; rh_ice = 1.10
           nuclei_minus = 0.0; nuclei_center = 1.0e4; nuclei_plus = 1.0e6
        case default
           temperature = 268.149; vertical_velocity = 1.0; rh_ice = 1.00001
           nuclei_minus = 1000.0; nuclei_center = 1001.0; nuclei_plus = 1002.0
        end select

        exner = (pressure/100000.0)**(287.04/1004.0)
        table_index = int((temperature-163.15)/0.002 + 1.5)
        table_index = min(1000001, max(1, table_index))
        table_temperature = 163.15 + real(table_index-1)*0.002
        qsi = (380.0/pressure)*exp(21.87455*(table_temperature-273.15) &
             /(table_temperature-7.66))
        before_qv = rh_ice*qsi
        call run_cell(unit)
     enddo
  enddo
  close(unit)
  print '(A,1X,A)', 'NSSL2_PRIMARY_ICE_NUCLEATION_ORACLE_COMPLETE', &
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
    real :: theta_before

    theta_before = temperature/exner
    a1 = 0.0
    a1(:,:,:,lt) = theta_before
    a1(:,:,:,lv) = before_qv
    u0 = temperature
    u1 = 0.0; u2 = 0.0; u3 = 0.0; u4 = 0.0; u5 = 0.0
    u6 = 0.0; u8 = 0.0; u9 = 0.0
    u7(1,1,1) = nuclei_minus
    u7(1,1,2) = nuclei_center
    u7(1,1,3) = nuclei_plus
    uu0 = 380.0/pressure; uu7 = 1.0; zg = layer_depth; dd = rho
    pp2 = exner; ppn = pressure; ww = vertical_velocity; tt3 = 0.0
    tee = 0.0; aa = 0.0; rp = 0.0; ep = 0.0; alp = 0.0
    el = 0.0; th = 0.0; timevtcalc = 0.0d0

    call nssl_2mom_gs(1,1,3,na,1,0,0,dt,zg,                &
         u0,u1,u2,u3,u4,u5,u6,u7,u8,u9,a1,dd,pp2,ppn,ww,0, &
         uu0,uu7,1.0,1.0,1.0,1,ido,xdnmx,xdnmn,cdx,xdn0,  &
         tt3,tee,th,1,1000.0,1000.0,3,timevtcalc,aa,.false.,&
         .false.,rp,ep,alp,el,1,1,1,1,1)

    write(output_unit,'(2(I0,","),20(ES24.16E3,","),ES24.16E3)') &
         case_id, repetition, dt, rho, pressure, exner, temperature, rh_ice, &
         vertical_velocity, layer_depth, nuclei_minus, nuclei_center, &
         nuclei_plus, theta_before, before_qv, 0.0, 0.0,                &
         a1(1,1,2,lt), a1(1,1,2,lv), a1(1,1,2,li),                    &
         a1(1,1,2,lni)/rho, a1(1,1,2,ls), a1(1,1,2,13)/rho
  end subroutine run_cell
end program nssl2_primary_ice_nucleation_oracle
