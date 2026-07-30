program nssl2_secondary_ice_conversions_oracle
  use module_mp_nssl_2mom, only: nssl_2mom_init, nssl_2mom_gs
  implicit none

  integer, parameter :: nx = 38, nz = 4, na = 40, nxtra = 20
  integer, parameter :: lc = 3, lr = 4, li = 5, ls = 6, lh = 7
  integer, parameter :: lhl = 8, lqmx = 30, lt = 1, lv = 2
  integer, parameter :: lnc = 10, lnr = 11, lni = 12, lns = 13
  integer, parameter :: lnh = 14, lnhl = 15, lvh = 16, lvhl = 17
  real, parameter :: steps(nz) = (/0.1, 1.0, 10.0, 60.0/)
  integer :: i, k, unit, nml_unit, case_id, repetition, table_index
  integer :: cfg_ibfc, cfg_icfn, cfg_itype2, cfg_iglcnvi, cfg_iglcnvs
  integer :: cfg_ihlcnh
  real :: nssl_params(20), xdnmx(lc:lhl), xdnmn(lc:lhl)
  real :: cdx(lc:lhl), xdn0(lc:lhl)
  real :: dt, cell_depth, rho, pressure, exner, temperature
  real :: table_temperature, qvs, cloud_diameter, rain_diameter, ice_diameter
  real :: snow_diameter, graupel_diameter, hail_diameter
  real :: graupel_density, hail_density
  real :: cloud_mass, rain_mass, ice_mass, snow_mass, graupel_mass, hail_mass
  real :: before_qv, before_qc, before_nc, before_qr, before_nr
  real :: before_qi, before_ni
  real :: before_qs, before_ns, before_qg, before_ng, before_vg
  real :: before_qh, before_nh, before_vh
  real :: ventr_default, ventc_default, c1sw_default
  integer :: ido(lc:lqmx)
  double precision :: timevtcalc
  character(len=512) :: output_path
  character(len=16) :: mode

  call get_command_argument(1, output_path)
  call get_command_argument(2, mode)
  if (len_trim(output_path) == 0) then
     error stop 'usage: nssl2_secondary_ice_conversions_oracle OUTPUT.csv full|baseline'
  endif
  ! ibfc=0 activates WRF's alternate post-assembly cloud-freezing path and
  ! ihlcnh=0 activates a legacy wet-growth conversion.  The true isolated
  ! baselines are therefore ibfc=3 and post-namelist ihlcnh=-1.
  cfg_ibfc = 3; cfg_icfn = 0; cfg_itype2 = 0
  cfg_iglcnvi = 0; cfg_iglcnvs = 0; cfg_ihlcnh = -1
  select case (trim(mode))
  case ('baseline')
  case ('contact')
     cfg_icfn = 2
  case ('homogeneous')
     cfg_ibfc = 1
  case ('hm')
     cfg_itype2 = 2
  case ('ice_to_g')
     cfg_iglcnvi = 1
  case ('snow_to_g')
     cfg_iglcnvs = 2
  case ('g_to_h')
     cfg_ihlcnh = 3
  case ('all')
     cfg_ibfc = 1; cfg_icfn = 2; cfg_itype2 = 2
     cfg_iglcnvi = 1; cfg_iglcnvs = 2; cfg_ihlcnh = 3
  case default
     error stop 'second argument must name a supported isolation mode'
  end select

  nssl_params = 0.0
  nssl_params(1) = 1.0e9
  nssl_params(2) = 0.0
  nssl_params(3) = 1.0
  nssl_params(6) = 0.0
  nssl_params(7) = -0.8
  nssl_params(8) = 500.0
  nssl_params(9) = 900.0
  nssl_params(10) = 100.0

  open(newunit=nml_unit, file='namelist.input', status='replace', action='write')
  write(nml_unit,'(A)') '&nssl_mp_params'
  write(nml_unit,'(A)') '  icenucopt = 0,'
  write(nml_unit,'(A)') '  iacr = 0,'
  write(nml_unit,'(A)') '  eri0 = 0.0,'
  write(nml_unit,'(A)') '  icracr = 0,'
  write(nml_unit,'(A)') '  ibiggopt = 0,'
  write(nml_unit,'(A)') '  ibiggsnow = 0,'
  write(nml_unit,'(A)') '  nsplinter = 0,'
  write(nml_unit,'(A)') '  iscni = 0,'
  write(nml_unit,'(A)') '  ess0 = 0.0,'
  write(nml_unit,'(A)') '  ehs0 = 0.0,'
  write(nml_unit,'(A)') '  dmrauto = -2,'
  write(nml_unit,'(A)') '  evapfac = 0.0,'
  write(nml_unit,'(A)') '  depfac = 0.0,'
  write(nml_unit,'(A)') '  isnwfrac = 0,'
  write(nml_unit,'(A)') '  icvhl2h = 0,'
  write(nml_unit,'(A,I0,A)') '  ibfc = ', cfg_ibfc, ','
  write(nml_unit,'(A,I0,A)') '  icfn = ', cfg_icfn, ','
  write(nml_unit,'(A)') '  itype1 = 0,'
  write(nml_unit,'(A,I0,A)') '  itype2 = ', cfg_itype2, ','
  write(nml_unit,'(A,I0,A)') '  iglcnvi = ', cfg_iglcnvi, ','
  write(nml_unit,'(A,I0,A)') '  iglcnvs = ', cfg_iglcnvs, ','
  ! ipconc=5 resolves the module default ihlcnh=-1 to option 3 before
  ! namelist parsing.  The g_to_h/all modes pin that resolved default; the
  ! isolated baseline writes -1 after resolution so no conversion branch runs.
  write(nml_unit,'(A,I0,A)') '  ihlcnh = ', cfg_ihlcnh, ','
  write(nml_unit,'(A)') '/'
  close(nml_unit)
  call nssl_2mom_init(nssl_params=nssl_params, ipctmp=5, mixphase=0, &
       nssl_density_on=.true., nssl_hail_on=.true.,                 &
       nssl_ccn_on=.true., nssl_icdx=6, nssl_icdxhl=6)
  ventr_default = gamma(2.0)
  ventc_default = gamma(4.0/3.0)
  c1sw_default = gamma(-0.8 + 4.0/3.0) * 0.2**(-1.0/3.0) &
       / gamma(0.2)

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
  write(unit,'(A)') 'case,repetition,dt_s,cell_depth_m,rho_kg_m3,pressure_pa,exner,temperature_k,cloud_diameter_m,rain_diameter_m,ice_diameter_m,snow_diameter_m,graupel_diameter_m,hail_diameter_m,graupel_density_kg_m3,hail_density_kg_m3,theta_before_k,qv_before,qc_before,qnc_before_per_kg,qr_before,qnr_before_per_kg,qi_before,qni_before_per_kg,qs_before,qns_before_per_kg,qg_before,qng_before_per_kg,qvolg_before_m3_per_kg,qh_before,qnh_before_per_kg,qvolh_before_m3_per_kg,theta_after_k,qv_after,qc_after,qnc_after_per_kg,qr_after,qnr_after_per_kg,qi_after,qni_after_per_kg,qs_after,qns_after_per_kg,qg_after,qng_after_per_kg,qvolg_after_m3_per_kg,qh_after,qnh_after_per_kg,qvolh_after_m3_per_kg'

  do k = 1, nz
     do i = 1, nx
        case_id = mod((k-1)*nx+i-1, nx)
        repetition = ((k-1)*nx+i-1)/nx
        dt = steps(repetition+1)
        cell_depth = 350.0 + 65.0*real(mod(case_id, 9))
        pressure = 98000.0 - 8000.0*real(repetition) - 175.0*real(case_id)
        rho = 1.24 - 0.14*real(repetition) + 0.0015*real(case_id)
        temperature = 260.0
        cloud_diameter = 30.0e-6
        rain_diameter = 1.0e-3
        ice_diameter = 100.0e-6
        snow_diameter = 500.0e-6
        graupel_diameter = 3.0e-3
        hail_diameter = 5.0e-3
        graupel_density = 500.0
        hail_density = 700.0
        before_qc = 0.0; before_qr = 0.0; before_qi = 0.0; before_qs = 0.0
        before_qg = 0.0; before_qh = 0.0
        select case (case_id)
        case (0)
           temperature = 260.0
        case (1)
           temperature = 271.15; before_qc = 8.0e-4
        case (2)
           temperature = 271.14; before_qc = 8.0e-4
        case (3)
           temperature = 268.0; before_qc = 8.0e-4
           cloud_diameter = 8.0e-6
        case (4)
           temperature = 260.0; before_qc = 8.0e-4
           cloud_diameter = 60.0e-6
        case (5)
           temperature = 260.0; before_qc = 2.0e-7
           cloud_diameter = 30.0e-6
        case (6)
           temperature = 240.0; before_qc = 8.0e-4
        case (7)
           temperature = 238.0; before_qc = 8.0e-4
        case (8)
           temperature = 235.0; before_qc = 8.0e-4
        case (9)
           temperature = 230.0; before_qc = 8.0e-4
        case (10)
           temperature = 230.0; before_qc = 8.0e-4
           cloud_diameter = 8.0e-6
        case (11)
           temperature = 230.0; before_qc = 8.0e-4
           cloud_diameter = 60.0e-6
        case (12)
           temperature = 225.0; before_qc = 5.0e-3
           cloud_diameter = 80.0e-6
        case (13)
           temperature = 265.15; before_qc = 1.0e-3; before_qg = 2.0e-3
           cloud_diameter = 60.0e-6
        case (14)
           temperature = 266.0; before_qc = 1.0e-3; before_qg = 2.0e-3
           cloud_diameter = 60.0e-6
        case (15)
           temperature = 268.15; before_qc = 1.0e-3; before_qg = 2.0e-3
           cloud_diameter = 24.0e-6
        case (16)
           temperature = 268.15; before_qc = 1.0e-3; before_qg = 2.0e-3
           cloud_diameter = 60.0e-6
        case (17)
           temperature = 270.0; before_qc = 1.0e-3; before_qg = 2.0e-3
           cloud_diameter = 60.0e-6
        case (18)
           temperature = 271.15; before_qc = 1.0e-3; before_qg = 2.0e-3
           cloud_diameter = 60.0e-6
        case (19)
           temperature = 271.16; before_qc = 1.0e-3; before_qg = 2.0e-3
           cloud_diameter = 60.0e-6
        case (20)
           temperature = 268.15; before_qc = 1.0e-3; before_qh = 2.0e-3
           cloud_diameter = 60.0e-6; hail_density = 900.0
        case (21)
           temperature = 268.15; before_qc = 2.0e-3
           before_qg = 2.0e-3; before_qh = 2.0e-3
           cloud_diameter = 60.0e-6; graupel_density = 800.0
        case (22)
           temperature = 260.0; before_qc = 1.0e-3; before_qi = 1.0e-3
           cloud_diameter = 60.0e-6; ice_diameter = 100.0e-6
        case (23)
           temperature = 260.0; before_qc = 1.0e-3; before_qi = 1.0e-3
           cloud_diameter = 16.0e-6; ice_diameter = 40.0e-6
        case (24)
           temperature = 273.0; before_qc = 1.0e-3; before_qi = 1.0e-3
           cloud_diameter = 60.0e-6; ice_diameter = 100.0e-6
        case (25)
           temperature = 272.99; before_qc = 1.0e-3; before_qi = 1.0e-3
           cloud_diameter = 60.0e-6; ice_diameter = 100.0e-6
        case (26)
           temperature = 260.0; before_qc = 1.0e-3; before_qs = 2.0e-3
           cloud_diameter = 60.0e-6; snow_diameter = 500.0e-6
        case (27)
           temperature = 260.0; before_qc = 1.0e-3; before_qs = 2.0e-3
           cloud_diameter = 16.0e-6; snow_diameter = 100.0e-6
        case (28)
           temperature = 273.0; before_qc = 1.0e-3; before_qs = 2.0e-3
           cloud_diameter = 60.0e-6; snow_diameter = 500.0e-6
        case (29)
           temperature = 260.0; before_qc = 1.5e-3; before_qg = 3.0e-3
           cloud_diameter = 60.0e-6; graupel_diameter = 10.0e-3
           graupel_density = 800.0
        case (30)
           temperature = 271.14; before_qc = 1.5e-3; before_qg = 3.0e-3
           cloud_diameter = 60.0e-6; graupel_diameter = 8.0e-3
           graupel_density = 800.0
        case (31)
           temperature = 271.15; before_qc = 1.5e-3; before_qg = 3.0e-3
           cloud_diameter = 60.0e-6; graupel_diameter = 8.0e-3
           graupel_density = 800.0
        case (32)
           temperature = 260.0; before_qc = 1.5e-3; before_qg = 3.0e-3
           cloud_diameter = 60.0e-6; graupel_diameter = 2.0e-3
           graupel_density = 500.0
        case (33)
           temperature = 260.0; before_qc = 1.5e-3; before_qg = 8.0e-5
           cloud_diameter = 60.0e-6; graupel_diameter = 8.0e-3
           graupel_density = 800.0
        case (34)
           temperature = 260.0; before_qc = 1.5e-3
           before_qg = 3.0e-3; before_qh = 2.0e-3
           cloud_diameter = 60.0e-6; graupel_diameter = 10.0e-3
           graupel_density = 800.0; hail_diameter = 8.0e-3
        case (35)
           temperature = 268.15; before_qc = 4.0e-3
           before_qi = 1.0e-3; before_qs = 2.0e-3
           before_qg = 3.0e-3; before_qh = 2.0e-3
           cloud_diameter = 60.0e-6; ice_diameter = 100.0e-6
           snow_diameter = 500.0e-6; graupel_diameter = 8.0e-3
           graupel_density = 800.0; hail_diameter = 8.0e-3
        case (36)
           temperature = 268.15; before_qc = 2.0e-5
           before_qi = 1.0e-3; before_qs = 2.0e-3
           before_qg = 3.0e-3; before_qh = 2.0e-3
           cloud_diameter = 60.0e-6; ice_diameter = 100.0e-6
           snow_diameter = 500.0e-6; graupel_diameter = 8.0e-3
           graupel_density = 800.0; hail_diameter = 8.0e-3
        case default
           temperature = 260.0; before_qh = 2.0e-3
           hail_diameter = 8.0e-3; hail_density = 500.0
        end select

        exner = (pressure/100000.0)**(287.04/1004.0)
        table_index = int((temperature-163.15)/0.002 + 1.5)
        table_index = min(1000001, max(1, table_index))
        table_temperature = 163.15 + real(table_index-1)*0.002
        qvs = (380.0/pressure)*exp(17.2693882*(table_temperature-273.15) &
             /(table_temperature-35.86))
        before_qv = qvs
        cloud_mass = 1000.0*(3.141592653589793/6.0)*cloud_diameter**3
        rain_mass = 1000.0*(3.141592653589793/6.0)*rain_diameter**3
        ice_mass = (ice_diameter/0.1871)**(1.0/0.3429)
        snow_mass = 100.0*(3.141592653589793/6.0)*snow_diameter**3
        graupel_mass = graupel_density*(3.141592653589793/6.0) &
             *graupel_diameter**3
        hail_mass = hail_density*(3.141592653589793/6.0) &
             *hail_diameter**3
        before_nc = 0.0; before_nr = 0.0; before_ni = 0.0; before_ns = 0.0
        before_ng = 0.0; before_vg = 0.0
        before_nh = 0.0; before_vh = 0.0
        if (before_qc > 0.0) before_nc = before_qc/cloud_mass
        if (before_qr > 0.0) before_nr = before_qr/rain_mass
        if (before_qi > 0.0) before_ni = before_qi/ice_mass
        if (before_qs > 0.0) before_ns = before_qs/snow_mass
        if (before_qg > 0.0) then
           before_ng = before_qg/graupel_mass
           before_vg = before_qg/graupel_density
        endif
        if (before_qh > 0.0) then
           before_nh = before_qh/hail_mass
           before_vh = before_qh/hail_density
        endif
        call run_cell(unit)
     enddo
  enddo
  close(unit)
  print '(A,1X,A)', 'NSSL2_SECONDARY_ICE_CONVERSIONS_ORACLE_COMPLETE', &
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
    a1(:,:,:,lc) = before_qc
    a1(:,:,:,lr) = before_qr
    a1(:,:,:,li) = before_qi
    a1(:,:,:,ls) = before_qs
    a1(:,:,:,lh) = before_qg
    a1(:,:,:,lhl) = before_qh
    a1(:,:,:,lnc) = before_nc*rho
    a1(:,:,:,lnr) = before_nr*rho
    a1(:,:,:,lni) = before_ni*rho
    a1(:,:,:,lns) = before_ns*rho
    a1(:,:,:,lnh) = before_ng*rho
    a1(:,:,:,lnhl) = before_nh*rho
    a1(:,:,:,lvh) = before_vg*rho
    a1(:,:,:,lvhl) = before_vh*rho
    before_nc = a1(1,1,2,lnc)/rho
    before_nr = a1(1,1,2,lnr)/rho
    before_ni = a1(1,1,2,lni)/rho
    before_ns = a1(1,1,2,lns)/rho
    before_ng = a1(1,1,2,lnh)/rho
    before_nh = a1(1,1,2,lnhl)/rho
    before_vg = a1(1,1,2,lvh)/rho
    before_vh = a1(1,1,2,lvhl)/rho
    u0 = temperature
    u1 = 0.0; u2 = 0.0; u3 = 0.0; u4 = 0.0; u5 = 0.0
    u6 = 0.0; u7 = 0.0; u8 = 0.0; u9 = 0.0
    uu0 = 380.0/pressure; uu7 = 1.0; zg = cell_depth; dd = rho
    pp2 = exner; ppn = pressure; ww = 0.0; tt3 = 0.0; tee = 0.0
    aa = 0.0; rp = 0.0; ep = 0.0; alp = 0.0; el = 0.0; th = 0.0
    timevtcalc = 0.0d0

    call nssl_2mom_gs(1,1,3,na,1,0,0,dt,zg,                &
         u0,u1,u2,u3,u4,u5,u6,u7,u8,u9,a1,dd,pp2,ppn,ww,0, &
         uu0,uu7,ventr_default,ventc_default,c1sw_default, &
         1,ido,xdnmx,xdnmn,cdx,xdn0,                       &
         tt3,tee,th,1,1000.0,1000.0,3,timevtcalc,aa,.false.,&
         .false.,rp,ep,alp,el,1,1,1,1,1)

    write(output_unit,'(2(I0,","),45(ES24.16E3,","),ES24.16E3)') &
         case_id, repetition, dt, cell_depth, rho, pressure, exner, &
         temperature, cloud_diameter, rain_diameter, ice_diameter, snow_diameter, &
         graupel_diameter, hail_diameter, graupel_density, hail_density, &
         theta_before, before_qv, before_qc, before_nc, before_qr, &
         before_nr, before_qi, before_ni, before_qs, before_ns, before_qg, before_ng, before_vg, &
         before_qh, before_nh, before_vh, a1(1,1,2,lt), a1(1,1,2,lv), &
         a1(1,1,2,lc), a1(1,1,2,lnc)/rho, a1(1,1,2,lr), &
         a1(1,1,2,lnr)/rho, a1(1,1,2,li), &
         a1(1,1,2,lni)/rho, a1(1,1,2,ls), a1(1,1,2,lns)/rho, &
         a1(1,1,2,lh), a1(1,1,2,lnh)/rho, a1(1,1,2,lvh)/rho, &
         a1(1,1,2,lhl), a1(1,1,2,lnhl)/rho, a1(1,1,2,lvhl)/rho
  end subroutine run_cell
end program nssl2_secondary_ice_conversions_oracle
