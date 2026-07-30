program nssl2_frozen_cross_collection_oracle
  use module_mp_nssl_2mom, only: nssl_2mom_init, nssl_2mom_gs
  implicit none

  integer, parameter :: nx = 4, nz = 20, na = 40, nxtra = 20
  integer, parameter :: lc = 3, lr = 4, li = 5, ls = 6, lh = 7
  integer, parameter :: lhl = 8, lqmx = 30, lt = 1, lv = 2
  integer, parameter :: lnc = 10, lnr = 11, lni = 12, lns = 13
  integer, parameter :: lnh = 14, lnhl = 15, lvh = 16, lvhl = 17
  integer :: i, k, unit, nml_unit, case_id, repetition, table_index
  real :: nssl_params(20), dt, temperature, pressure, exner, rho
  real :: rain_diameter, ice_diameter, snow_diameter
  real :: graupel_diameter, hail_diameter, graupel_density, hail_density
  real :: cloud_mass, rain_mass, ice_mass, snow_mass
  real :: graupel_mass, hail_mass, before_qv, before_qc, before_nc
  real :: before_qr, before_nr, before_qi, before_ni
  real :: before_qs, before_ns, before_qg, before_ng, before_vg
  real :: before_qh, before_nh, before_vh, table_temperature, qvs
  real, parameter :: steps(4) = [0.1, 1.0, 10.0, 60.0]
  real :: xdnmx(lc:lhl), xdnmn(lc:lhl), xdn0(lc:lhl), cdx(lc:lhl)
  integer :: ido(lc:lqmx)
  double precision :: timevtcalc
  character(len=512) :: output_path

  call get_command_argument(1, output_path)
  if (len_trim(output_path) == 0) then
     error stop 'usage: nssl2_frozen_cross_collection_oracle OUTPUT.csv'
  endif

  nssl_params = 0.0
  nssl_params(1:10) = [0.5e9, 0.0, 1.0, 4.0e5, 4.0e4, &
                       8.0e5, 3.0e6, 500.0, 900.0, 100.0]
  open(newunit=nml_unit, file='namelist.input', status='replace', action='write')
  write(nml_unit,'(A)') '&nssl_mp_params'
  write(nml_unit,'(A)') '  icenucopt = 0,'
  write(nml_unit,'(A)') '  icfn = 0,'
  write(nml_unit,'(A)') '  ibfc = 0,'
  write(nml_unit,'(A)') '  iacr = 0,'
  write(nml_unit,'(A)') '  eri0 = 0.0,'
  write(nml_unit,'(A)') '  icracr = 0,'
  write(nml_unit,'(A)') '  ibiggopt = 0,'
  write(nml_unit,'(A)') '  ibiggsnow = 0,'
  write(nml_unit,'(A)') '  itype1 = 0,'
  write(nml_unit,'(A)') '  itype2 = 0,'
  write(nml_unit,'(A)') '  nsplinter = 0,'
  write(nml_unit,'(A)') '  iscni = 0,'
  write(nml_unit,'(A)') '  ess0 = 0.0,'
  write(nml_unit,'(A)') '  ehw0 = 0.0,'
  write(nml_unit,'(A)') '  ehlw0 = 0.0,'
  write(nml_unit,'(A)') '  dmrauto = -2,'
  write(nml_unit,'(A)') '  evapfac = 0.0,'
  write(nml_unit,'(A)') '  depfac = 0.0,'
  write(nml_unit,'(A)') '  iglcnvi = 0,'
  write(nml_unit,'(A)') '  iglcnvs = 0,'
  write(nml_unit,'(A)') '  isnwfrac = 0,'
  write(nml_unit,'(A)') '  ihlcnh = 0,'
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
  write(unit,'(A)') 'case,repetition,dt_s,rho_kg_m3,pressure_pa,exner,temperature_k,rain_diameter_m,ice_diameter_m,snow_diameter_m,graupel_diameter_m,hail_diameter_m,graupel_density_kg_m3,hail_density_kg_m3,theta_before_k,qv_before,qc_before,qr_before,qnr_before_per_kg,qi_before,qni_before_per_kg,qs_before,qns_before_per_kg,qg_before,qng_before_per_kg,qvolg_before_m3_per_kg,qh_before,qnh_before_per_kg,qvolh_before_m3_per_kg,theta_after_k,qv_after,qc_after,qr_after,qnr_after_per_kg,qi_after,qni_after_per_kg,qs_after,qns_after_per_kg,qg_after,qng_after_per_kg,qvolg_after_m3_per_kg,qh_after,qnh_after_per_kg,qvolh_after_m3_per_kg'

  do k = 1, nz
     do i = 1, nx
        case_id = mod((k-1)*nx+i-1, 20)
        repetition = ((k-1)*nx+i-1)/20
        dt = steps(repetition+1)
        pressure = 96000.0 - 8000.0*real(repetition) - 250.0*real(case_id)
        rho = 1.22 - 0.13*real(repetition) + 0.0015*real(case_id)
        ! Stay below the native 242 K graupel-to-hail conversion floor so
        ! process 26 measures only dry cross-collection (conversion is #28).
        temperature = 238.0 + 0.2*real(mod(case_id, 20))
        rain_diameter = 1.0e-3
        ice_diameter = 100.0e-6
        snow_diameter = 300.0e-6
        graupel_diameter = 1.5e-3
        hail_diameter = 4.0e-3
        graupel_density = 500.0
        hail_density = 900.0
        before_qc = 0.0
        before_qr = 0.0
        before_qi = 0.0
        before_qs = 0.0
        before_qg = 0.0
        before_qh = 0.0
        select case (case_id)
        case (0)
           continue
        case (1)
           before_qi = 2.0e-4; before_qs = 5.0e-4
        case (2)
           before_qr = 5.0e-4; before_qs = 5.0e-4
        case (3)
           before_qi = 2.0e-4; before_qg = 1.0e-3
        case (4)
           before_qc = 1.0e-13; before_qs = 5.0e-4
           before_qg = 1.0e-3; snow_diameter = 30.0e-6
        case (5)
           before_qc = 1.0e-13; before_qs = 5.0e-4
           before_qg = 1.0e-3; snow_diameter = 80.0e-6
        case (6)
           before_qc = 1.0e-13; before_qs = 5.0e-4
           before_qg = 1.0e-3; snow_diameter = 200.0e-6
           graupel_density = 290.0
        case (7)
           before_qc = 1.0e-13; before_qs = 5.0e-4
           before_qg = 1.0e-3; snow_diameter = 200.0e-6
           graupel_density = 450.0
        case (8)
           before_qr = 5.0e-4; before_qg = 1.0e-3
        case (9)
           before_qi = 2.0e-4; before_qh = 1.0e-3
        case (10)
           before_qc = 1.0e-13; before_qi = 2.0e-4
           before_qh = 1.0e-3
        case (11)
           before_qs = 5.0e-4; before_qh = 1.0e-3
        case (12)
           before_qr = 5.0e-4; before_qh = 1.0e-3
        case (13)
           before_qc = 1.0e-13; before_qr = 5.0e-4
           before_qi = 2.0e-4; before_qs = 5.0e-4
           before_qg = 1.0e-3; before_qh = 1.0e-3
        case (14)
           before_qc = 1.0e-13; before_qr = 5.0e-6
           before_qi = 3.0e-6; before_qs = 5.0e-3
           before_qg = 5.0e-2; before_qh = 5.0e-2
           graupel_diameter = 1.50e-3; hail_diameter = 1.50e-3
           snow_diameter = 150.0e-6
        case (15)
           before_qc = 1.0e-13; before_qr = 8.0e-4
           before_qi = 3.0e-4; before_qs = 8.0e-4
           before_qg = 2.0e-3; before_qh = 2.0e-3
           graupel_density = 170.0; hail_density = 500.0
           graupel_diameter = 3.0e-3; hail_diameter = 8.0e-3
           snow_diameter = 500.0e-6
        case (16)
           ! Dedicated three-collector ice cap: the tiny snow diameter makes
           ! snow<-ice hit its independent 10-percent cap, while interior
           ! dense moments make graupel<-ice and hail<-ice hit theirs.
           before_qc = 1.0e-13; before_qi = 3.0e-6
           before_qs = 5.0e-4; before_qg = 5.0e-2; before_qh = 5.0e-2
           snow_diameter = 18.0e-6
           graupel_density = 170.0; hail_density = 500.0
           graupel_diameter = 0.50e-3; hail_diameter = 0.50e-3
        case (17)
           ! Dedicated two-collector snow cap.  Graupel's cloud/density
           ! gates are deliberately open and both dense means are interior.
           before_qc = 1.0e-13; before_qs = 5.0e-6
           before_qg = 1.0e-1; before_qh = 5.0e-2
           snow_diameter = 150.0e-6
           graupel_density = 900.0; hail_density = 500.0
           graupel_diameter = 0.35e-3; hail_diameter = 0.50e-3
        case (18)
           ! Dedicated two-collector rain cap.
           before_qr = 5.0e-6; before_qg = 5.0e-2; before_qh = 5.0e-2
           graupel_diameter = 1.50e-3; hail_diameter = 1.50e-3
        case default
           ! High-count, interior dense moments with no donors must be an
           ! exact no-op (including number and predicted-volume moments).
           before_qg = 1.0e-1; before_qh = 5.0e-2
           graupel_density = 900.0; hail_density = 700.0
           graupel_diameter = 0.35e-3; hail_diameter = 0.50e-3
        end select

        exner = (pressure/100000.0)**(287.04/1004.0)
        table_index = int((temperature-163.15)/0.002 + 1.5)
        table_index = min(1000001, max(1, table_index))
        table_temperature = 163.15 + real(table_index-1)*0.002
        ! Use the exact native liquid-saturation lookup node so NUCOND is
        ! inert.  With the now-fully-read depfac=0 namelist control, frozen
        ! vapor exchange is independently disabled.
        qvs = (380.0/pressure)*exp(17.2693882*(table_temperature-273.15) &
             /(table_temperature-35.86))
        before_qv = qvs
        cloud_mass = 1000.0*(3.141592653589793/6.0)*(2.0e-6)**3
        rain_mass = 1000.0*(3.141592653589793/6.0)*rain_diameter**3
        ice_mass = (ice_diameter/0.1871)**(1.0/0.3429)
        snow_mass = 100.0*(3.141592653589793/6.0)*snow_diameter**3
        graupel_mass = graupel_density*(3.141592653589793/6.0) &
             *graupel_diameter**3
        hail_mass = hail_density*(3.141592653589793/6.0) &
             *hail_diameter**3
        if (before_qc > 0.0) then
           before_nc = before_qc/cloud_mass
        else
           before_nc = 0.0
        endif
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
        if (before_qs > 0.0) then
           before_ns = before_qs/snow_mass
        else
           before_ns = 0.0
        endif
        if (before_qg > 0.0) then
           before_ng = before_qg/graupel_mass
           before_vg = before_qg/graupel_density
        else
           before_ng = 0.0
           before_vg = 0.0
        endif
        if (before_qh > 0.0) then
           before_nh = before_qh/hail_mass
           before_vh = before_qh/hail_density
        else
           before_nh = 0.0
           before_vh = 0.0
        endif
        call run_cell(unit)
     enddo
  enddo
  close(unit)
  print '(A,1X,A)', 'NSSL2_FROZEN_CROSS_COLLECTION_ORACLE_COMPLETE', &
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
    ! Report the exact packed/unpacked REAL values used by the native call.
    ! This keeps true no-op rows bitwise stable instead of counting harmless
    ! host-side multiply/divide roundoff as microphysical number change.
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
    uu0 = 380.0/pressure; uu7 = 1.0; zg = 1000.0; dd = rho
    pp2 = exner; ppn = pressure; ww = 0.0; tt3 = 0.0; tee = 0.0
    aa = 0.0; rp = 0.0; ep = 0.0; alp = 0.0; el = 0.0; th = 0.0
    timevtcalc = 0.0d0

    call nssl_2mom_gs(1,1,3,na,1,0,0,dt,zg,                &
         u0,u1,u2,u3,u4,u5,u6,u7,u8,u9,a1,dd,pp2,ppn,ww,0, &
         uu0,uu7,1.0,1.0,1.0,1,ido,xdnmx,xdnmn,cdx,xdn0,  &
         tt3,tee,th,1,1000.0,1000.0,3,timevtcalc,aa,.false.,&
         .false.,rp,ep,alp,el,1,1,1,1,1)

    write(output_unit,'(2(I0,","),41(ES24.16E3,","),ES24.16E3)') &
         case_id, repetition, dt, rho, pressure, exner, temperature, &
         rain_diameter, ice_diameter, snow_diameter, graupel_diameter, &
         hail_diameter, graupel_density, hail_density, theta_before, &
         before_qv, before_qc, before_qr, before_nr, before_qi, before_ni, &
         before_qs, before_ns, before_qg, before_ng, before_vg, before_qh, &
         before_nh, before_vh, a1(1,1,2,lt), a1(1,1,2,lv), &
         a1(1,1,2,lc), a1(1,1,2,lr), a1(1,1,2,lnr)/rho, &
         a1(1,1,2,li), a1(1,1,2,lni)/rho, a1(1,1,2,ls), &
         a1(1,1,2,lns)/rho, a1(1,1,2,lh), a1(1,1,2,lnh)/rho, &
         a1(1,1,2,lvh)/rho, a1(1,1,2,lhl), a1(1,1,2,lnhl)/rho, &
         a1(1,1,2,lvhl)/rho
  end subroutine run_cell
end program nssl2_frozen_cross_collection_oracle
