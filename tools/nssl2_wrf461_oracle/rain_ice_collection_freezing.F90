program nssl2_rain_ice_collection_freezing_oracle
  use module_mp_nssl_2mom, only: nssl_2mom_init, nssl_2mom_gs
  implicit none

  integer, parameter :: nx = 4, nz = 12, na = 40, nxtra = 20
  integer, parameter :: lc = 3, lr = 4, li = 5, ls = 6, lh = 7
  integer, parameter :: lhl = 8, lqmx = 30, lt = 1, lv = 2
  integer, parameter :: lnc = 10, lnr = 11, lni = 12, lns = 13
  integer, parameter :: lnh = 14, lnhl = 15, lvh = 16, lvhl = 17
  integer :: i, k, unit, nml_unit, case_id, repetition, table_index
  real :: nssl_params(20), dt, temperature, pressure, exner, rho
  real :: rain_diameter, ice_diameter, rain_mass, ice_mass
  real :: before_qv, before_qr, before_nr, before_qi, before_ni
  real :: table_temperature, qvs
  real, parameter :: steps(4) = [0.1, 1.0, 10.0, 60.0]
  real :: xdnmx(lc:lhl), xdnmn(lc:lhl), xdn0(lc:lhl), cdx(lc:lhl)
  integer :: ido(lc:lqmx)
  double precision :: timevtcalc
  character(len=512) :: output_path

  call get_command_argument(1, output_path)
  if (len_trim(output_path) == 0) then
     error stop 'usage: nssl2_rain_ice_collection_freezing_oracle OUTPUT.csv'
  endif

  nssl_params = 0.0
  nssl_params(1:10) = [0.5e9, 0.0, 1.0, 4.0e5, 4.0e4, &
                       8.0e5, 3.0e6, 500.0, 900.0, 100.0]
  open(newunit=nml_unit, file='namelist.input', status='replace', action='write')
  write(nml_unit,'(A)') '&nssl_mp_params'
  write(nml_unit,'(A)') '  icenucopt = 0,'
  write(nml_unit,'(A)') '  icfn = 0,'
  write(nml_unit,'(A)') '  ibfc = 0,'
  write(nml_unit,'(A)') '  iacr = 2,'
  write(nml_unit,'(A)') '  iacrsize = 5,'
  write(nml_unit,'(A)') '  icracr = 0,'
  write(nml_unit,'(A)') '  ibiggopt = 0,'
  write(nml_unit,'(A)') '  ibiggsnow = 0,'
  write(nml_unit,'(A)') '  nsplinter = 0,'
  write(nml_unit,'(A)') '  iscni = 0,'
  write(nml_unit,'(A)') '  dmrauto = -2,'
  write(nml_unit,'(A)') '  depfac = 0.0,'
  write(nml_unit,'(A)') '  iglcnvi = 0,'
  write(nml_unit,'(A)') '  iglcnvs = 0,'
  write(nml_unit,'(A)') '  isnwfrac = 0,'
  write(nml_unit,'(A)') '  ihlcnh = 0,'
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
  write(unit,'(A)') 'case,repetition,dt_s,rho_kg_m3,pressure_pa,exner,temperature_k,rain_diameter_m,ice_diameter_m,theta_before_k,qv_before,qr_before,qnr_before_per_kg,qi_before,qni_before_per_kg,qg_before,qng_before_per_kg,qvolg_before_m3_per_kg,theta_after_k,qv_after,qr_after,qnr_after_per_kg,qi_after,qni_after_per_kg,qg_after,qng_after_per_kg,qvolg_after_m3_per_kg,qc_after,qs_after,qh_after'

  do k = 1, nz
     do i = 1, nx
        case_id = mod((k-1)*nx+i-1, 12)
        repetition = ((k-1)*nx+i-1)/12
        dt = steps(repetition+1)
        pressure = 96000.0 - 8000.0*real(repetition) - 350.0*real(case_id)
        rho = 1.22 - 0.13*real(repetition) + 0.002*real(case_id)
        temperature = 265.0
        rain_diameter = 0.50e-3
        ice_diameter = 100.0e-6
        before_qr = 5.0e-4
        before_qi = 2.0e-5
        select case (case_id)
        case (0)
           before_qr = 0.0
        case (1)
           before_qi = 0.0
        case (2)
           rain_diameter = 80.0e-6
        case (3)
           ice_diameter = 8.0e-6
        case (4)
           temperature = 271.0; rain_diameter = 0.30e-3
           ice_diameter = 40.0e-6
        case (5)
           temperature = 269.0; rain_diameter = 0.30e-3
           ice_diameter = 40.0e-6
        case (6)
           temperature = 267.0; rain_diameter = 0.50e-3
           ice_diameter = 80.0e-6
        case (7)
           temperature = 263.0; rain_diameter = 0.80e-3
           ice_diameter = 150.0e-6
        case (8)
           temperature = 258.0; rain_diameter = 1.20e-3
           ice_diameter = 250.0e-6; before_qr = 1.0e-3
           before_qi = 5.0e-5
        case (9)
           temperature = 248.0; rain_diameter = 2.0e-3
           ice_diameter = 500.0e-6; before_qr = 2.0e-3
           before_qi = 1.0e-4
        case (10)
           temperature = 240.0; rain_diameter = 4.0e-3
           ice_diameter = 1.0e-3; before_qr = 2.0e-3
           before_qi = 2.0e-4
        case default
           temperature = 265.0; rain_diameter = 0.12e-3
           ice_diameter = 40.0e-6; before_qr = 5.0e-3
           before_qi = 2.0e-3
        end select

        exner = (pressure/100000.0)**(287.04/1004.0)
        table_index = int((temperature-163.15)/0.002 + 1.5)
        table_index = min(1000001, max(1, table_index))
        table_temperature = 163.15 + real(table_index-1)*0.002
        qvs = (380.0/pressure)*exp(17.2693882*(table_temperature-273.15) &
             /(table_temperature-35.86))
        before_qv = qvs
        rain_mass = 1000.0*(3.141592653589793/6.0)*rain_diameter**3
        ice_mass = (ice_diameter/0.1871)**(1.0/0.3429)
        if (before_qr > 0.0) then
           before_nr = before_qr/rain_mass
        else
           before_nr = 0.0
        endif
        if (before_qi > 0.0) then
           before_ni = before_qi/ice_mass
        else
           before_ni = 0.0
        endif
        call run_cell(unit)
     enddo
  enddo
  close(unit)
  print '(A,1X,A)', 'NSSL2_RAIN_ICE_COLLECTION_FREEZING_ORACLE_COMPLETE', &
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
    a1(:,:,:,lr) = before_qr
    a1(:,:,:,li) = before_qi
    a1(:,:,:,lnr) = before_nr*rho
    a1(:,:,:,lni) = before_ni*rho
    u0 = temperature
    u1 = 0.0; u2 = 0.0; u3 = 0.0; u4 = 0.0; u5 = 0.0
    u6 = 0.0; u7 = 0.0; u8 = 0.0; u9 = 0.0
    uu0 = 380.0/pressure; uu7 = 1.0; zg = 1000.0; dd = rho
    pp2 = exner; ppn = pressure; ww = 0.0; tt3 = 0.0; tee = 0.0
    aa = 0.0; rp = 0.0; ep = 0.0; alp = 0.0; el = 0.0; th = 0.0
    timevtcalc = 0.0d0

    call nssl_2mom_gs(1,1,3,na,1,0,0,dt,zg,                &
         u0,u1,u2,u3,u4,u5,u6,u7,u8,u9,a1,dd,pp2,ppn,ww,0, &
         uu0,uu7,1.0,1.0,1.0,1,ido,xdnmx,xdnmn,cdx,xdn0,  &
         tt3,tee,th,1,1000.0,1000.0,3,timevtcalc,aa,.false.,&
         .false.,rp,ep,alp,el,1,1,1,1,1)

    write(output_unit,'(2(I0,","),27(ES24.16E3,","),ES24.16E3)') &
         case_id, repetition, dt, rho, pressure, exner, temperature, &
         rain_diameter, ice_diameter, theta_before, before_qv, before_qr, &
         before_nr, before_qi, before_ni, 0.0, 0.0, 0.0, &
         a1(1,1,2,lt), a1(1,1,2,lv), a1(1,1,2,lr), &
         a1(1,1,2,lnr)/rho, a1(1,1,2,li), a1(1,1,2,lni)/rho, &
         a1(1,1,2,lh), a1(1,1,2,lnh)/rho, a1(1,1,2,lvh)/rho, &
         a1(1,1,2,lc), a1(1,1,2,ls), a1(1,1,2,lhl)
  end subroutine run_cell
end program nssl2_rain_ice_collection_freezing_oracle
