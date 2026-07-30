program nssl2_graupel_hail_vapor_exchange_oracle
  use module_mp_nssl_2mom, only: nssl_2mom_init, nssl_2mom_gs
  implicit none

  integer, parameter :: nx = 4, nz = 12, na = 40, nxtra = 20
  integer, parameter :: lc = 3, lr = 4, li = 5, ls = 6, lh = 7
  integer, parameter :: lhl = 8, lqmx = 30, lt = 1, lv = 2
  integer, parameter :: lnh = 14, lnhl = 15, lvh = 16, lvhl = 17
  integer :: i, k, unit, nml_unit, case_id, repetition, table_index
  real :: nssl_params(20), dt, temperature, pressure, exner, rho
  real :: rh_ice, graupel_diameter, hail_diameter
  real :: graupel_density, hail_density, graupel_volume, hail_volume
  real :: before_qv, before_qg, before_ng, before_vg
  real :: before_qh, before_nh, before_vh
  real :: table_temperature, qsi, scale
  real :: xdnmx(lc:lhl), xdnmn(lc:lhl), xdn0(lc:lhl), cdx(lc:lhl)
  integer :: ido(lc:lqmx)
  double precision :: timevtcalc
  character(len=512) :: output_path

  call get_command_argument(1, output_path)
  if (len_trim(output_path) == 0) then
     error stop 'usage: nssl2_graupel_hail_vapor_exchange_oracle OUTPUT.csv'
  endif

  nssl_params = 0.0
  nssl_params(1:10) = [0.5e9, 0.0, 1.0, 4.0e5, 4.0e4, &
                       8.0e5, 3.0e6, 500.0, 900.0, 100.0]

  ! Retain default deposition/sublimation, density prediction, and Milbrandt-
  ! Morrison terminal-speed/ventilation parameters.  Empty neighboring
  ! categories remove collection/freezing, and the explicit conversion gate
  ! keeps the hail and graupel outputs a direct vapor-exchange oracle.
  open(newunit=nml_unit, file='namelist.input', status='replace', action='write')
  write(nml_unit,'(A)') '&nssl_mp_params'
  write(nml_unit,'(A)') '  icenucopt = 0,'
  write(nml_unit,'(A)') '  ess0 = 0.0,'
  write(nml_unit,'(A)') '  esi0 = 0.0,'
  write(nml_unit,'(A)') '  isnwfrac = 0,'
  write(nml_unit,'(A)') '  icvhl2h = 0,'
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
  write(unit,'(A)') 'case,i,k,dt_s,rho_kg_m3,pressure_pa,exner,temperature_k,rh_ice,target_graupel_diameter_m,target_hail_diameter_m,graupel_density_kg_m3,hail_density_kg_m3,theta_before_k,qv_before,qg_before,qng_before_per_kg,qvolg_before_m3_per_kg,qh_before,qnh_before_per_kg,qvolh_before_m3_per_kg,theta_after_k,qv_after,qg_after,qng_after_per_kg,qvolg_after_m3_per_kg,qh_after,qnh_after_per_kg,qvolh_after_m3_per_kg'
  do k = 1, nz
     do i = 1, nx
        case_id = mod((k-1)*nx+i-1, 12)
        repetition = ((k-1)*nx+i-1)/12
        pressure = 100000.0 - 10500.0*real(repetition) - 350.0*real(case_id)
        rho = 1.28 - 0.16*real(repetition) + 0.002*real(case_id)
        scale = 10.0**real(repetition-2)
        before_qg = 0.0
        before_qh = 0.0
        graupel_density = 500.0
        hail_density = 900.0

        select case (case_id)
        case (0)
           temperature = 238.0; rh_ice = 1.05; dt = 0.1
           graupel_diameter = 0.20e-3; hail_diameter = 0.50e-3
           graupel_density = 170.0; before_qg = 2.5e-5*scale
        case (1)
           temperature = 245.0; rh_ice = 1.05; dt = 1.0
           graupel_diameter = 0.50e-3; hail_diameter = 0.50e-3
           hail_density = 900.0; before_qh = 2.5e-5*scale
        case (2)
           temperature = 250.0; rh_ice = 1.20; dt = 5.0
           graupel_diameter = 1.0e-3; hail_diameter = 1.0e-3
           graupel_density = 300.0; hail_density = 750.0
           before_qg = 1.25e-5*scale; before_qh = 1.25e-5*scale
        case (3)
           temperature = 255.0; rh_ice = 1.50; dt = 30.0
           graupel_diameter = 3.0e-3; hail_diameter = 2.0e-3
           graupel_density = 500.0; before_qg = 2.5e-5*scale
        case (4)
           temperature = 240.0; rh_ice = 0.95; dt = 1.0
           graupel_diameter = 0.25e-3; hail_diameter = 0.50e-3
           graupel_density = 250.0; before_qg = 2.5e-5*scale
        case (5)
           temperature = 248.0; rh_ice = 0.90; dt = 5.0
           graupel_diameter = 0.50e-3; hail_diameter = 1.0e-3
           hail_density = 800.0; before_qh = 2.5e-5*scale
        case (6)
           temperature = 258.0; rh_ice = 0.75; dt = 15.0
           graupel_diameter = 1.0e-3; hail_diameter = 3.0e-3
           graupel_density = 400.0; hail_density = 700.0
           before_qg = 1.25e-5*scale; before_qh = 1.25e-5*scale
        case (7)
           temperature = 265.0; rh_ice = 0.20; dt = 60.0
           graupel_diameter = 5.0e-3; hail_diameter = 8.0e-3
           graupel_density = 700.0; hail_density = 850.0
           before_qg = 5.0e-6*scale; before_qh = 2.0e-5*scale
        case (8)
           temperature = 268.0; rh_ice = 0.999; dt = 300.0
           graupel_diameter = 0.10e-3; hail_diameter = 0.20e-3
           graupel_density = 170.0; hail_density = 900.0
           before_qg = 1.0e-5*scale; before_qh = 1.5e-5*scale
        case (9)
           temperature = 250.0; rh_ice = 1.01; dt = 10.0
           graupel_diameter = 8.0e-3; hail_diameter = 1.0e-3
           graupel_density = 800.0; before_qg = 2.5e-5*scale
        case (10)
           temperature = 260.0; rh_ice = 1.10; dt = 60.0
           graupel_diameter = 2.0e-3; hail_diameter = 20.0e-3
           graupel_density = 500.0; hail_density = 900.0
           before_qg = 5.0e-6*scale; before_qh = 2.0e-5*scale
        case default
           temperature = 272.0; rh_ice = 1.001; dt = 1000.0
           graupel_diameter = 10.0e-3; hail_diameter = 30.0e-3
           graupel_density = 800.0; hail_density = 900.0
           if (repetition == 0) then
              before_qg = 0.0; before_qh = 0.0
           else if (repetition == 1) then
              before_qg = 1.0e-13; before_qh = 1.0e-13
           else
              before_qg = 1.0e-5*scale; before_qh = 1.5e-5*scale
           endif
        end select

        exner = (pressure/100000.0)**(287.04/1004.0)
        table_index = int((temperature-163.15)/0.002 + 1.5)
        table_index = min(1000001, max(1, table_index))
        table_temperature = 163.15 + real(table_index-1)*0.002
        qsi = (380.0/pressure)*exp(21.87455*(table_temperature-273.15) &
             /(table_temperature-7.66))
        before_qv = rh_ice*qsi
        graupel_volume = 0.523599*graupel_diameter**3
        hail_volume = 0.523599*hail_diameter**3
        if (before_qg > 0.0) then
           before_ng = before_qg/(graupel_density*graupel_volume)
           before_vg = before_qg/graupel_density
        else
           before_ng = 0.0; before_vg = 0.0
        endif
        if (before_qh > 0.0) then
           before_nh = before_qh/(hail_density*hail_volume)
           before_vh = before_qh/hail_density
        else
           before_nh = 0.0; before_vh = 0.0
        endif
        call run_cell(unit)
     enddo
  enddo
  close(unit)
  print '(A,1X,A)', 'NSSL2_GRAUPEL_HAIL_VAPOR_EXCHANGE_ORACLE_COMPLETE', &
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
    a1(:,:,:,lh) = before_qg
    a1(:,:,:,lhl) = before_qh
    a1(:,:,:,lnh) = before_ng*rho
    a1(:,:,:,lnhl) = before_nh*rho
    a1(:,:,:,lvh) = before_vg*rho
    a1(:,:,:,lvhl) = before_vh*rho
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

    write(output_unit,'(3(I0,","),25(ES24.16E3,","),ES24.16E3)') &
         case_id, i, k, dt, rho, pressure, exner, temperature, rh_ice, &
         graupel_diameter, hail_diameter, graupel_density, hail_density, &
         theta_before, before_qv, before_qg, before_ng, before_vg, &
         before_qh, before_nh, before_vh, a1(1,1,2,lt), a1(1,1,2,lv), &
         a1(1,1,2,lh), a1(1,1,2,lnh)/rho, a1(1,1,2,lvh)/rho, &
         a1(1,1,2,lhl), a1(1,1,2,lnhl)/rho, a1(1,1,2,lvhl)/rho
  end subroutine run_cell
end program nssl2_graupel_hail_vapor_exchange_oracle
