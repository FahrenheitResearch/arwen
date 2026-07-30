program nssl2_hail_cloud_riming_oracle
  use module_mp_nssl_2mom, only: nssl_2mom_init, nssl_2mom_gs
  implicit none

  integer, parameter :: nx = 4, nz = 12, na = 40, nxtra = 20
  integer, parameter :: lc = 3, lr = 4, li = 5, ls = 6, lh = 7
  integer, parameter :: lhl = 8, lqmx = 30, lt = 1, lv = 2
  integer, parameter :: lnc = 10, lnh = 14, lnhl = 15
  integer, parameter :: lvh = 16, lvhl = 17
  integer :: i, k, unit, nml_unit, case_id, repetition, table_index
  real :: nssl_params(20), dt, cell_depth, temperature, pressure, exner, rho
  real :: cloud_diameter, hail_diameter, hail_density
  real :: cloud_mass, hail_mass, before_qv, before_qc, before_nc
  real :: before_qh, before_nh, before_vh, table_temperature, qvs
  real, parameter :: steps(4) = [0.1, 1.0, 10.0, 60.0]
  real :: xdnmx(lc:lhl), xdnmn(lc:lhl), xdn0(lc:lhl), cdx(lc:lhl)
  integer :: ido(lc:lqmx)
  double precision :: timevtcalc
  character(len=512) :: output_path

  call get_command_argument(1, output_path)
  if (len_trim(output_path) == 0) then
     error stop 'usage: nssl2_hail_cloud_riming_oracle OUTPUT.csv'
  endif

  nssl_params = 0.0
  nssl_params(1:10) = [0.5e9, 0.0, 1.0, 4.0e5, 4.0e4, &
                       8.0e5, 3.0e6, 500.0, 900.0, 100.0]
  open(newunit=nml_unit, file='namelist.input', status='replace', action='write')
  write(nml_unit,'(A)') '&nssl_mp_params'
  write(nml_unit,'(A)') '  icenucopt = 0,'
  write(nml_unit,'(A)') '  icfn = 0,'
  write(nml_unit,'(A)') '  iacr = 0,'
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
  write(unit,'(A)') 'case,repetition,dt_s,cell_depth_m,rho_kg_m3,pressure_pa,exner,temperature_k,cloud_diameter_m,hail_diameter_m,hail_density_kg_m3,theta_before_k,qv_before,qc_before,qnc_before_per_kg,qh_before,qnh_before_per_kg,qvolh_before_m3_per_kg,theta_after_k,qv_after,qc_after,qnc_after_per_kg,qh_after,qnh_after_per_kg,qvolh_after_m3_per_kg,qg_after,qng_after_per_kg,qvolg_after_m3_per_kg,qi_after,qs_after,qr_after'

  do k = 1, nz
     do i = 1, nx
        case_id = mod((k-1)*nx+i-1, 12)
        repetition = ((k-1)*nx+i-1)/12
        dt = steps(repetition+1)
        cell_depth = 250.0 + 100.0*real(case_id)
        pressure = 96000.0 - 8000.0*real(repetition) - 350.0*real(case_id)
        rho = 1.22 - 0.13*real(repetition) + 0.002*real(case_id)
        temperature = 245.0 + 0.5*real(mod(case_id, 8))
        cloud_diameter = 20.0e-6
        hail_diameter = 2.0e-3
        hail_density = 700.0
        before_qc = 5.0e-4
        before_qh = 2.0e-4
        select case (case_id)
        case (0)
           before_qh = 0.0
        case (1)
           before_qc = 0.0
        case (2)
           cloud_diameter = 4.0e-6; hail_diameter = 0.30e-3
           hail_density = 500.0
        case (3)
           cloud_diameter = 8.0e-6; hail_diameter = 0.50e-3
           hail_density = 600.0
        case (4)
           cloud_diameter = 15.0e-6; hail_diameter = 1.0e-3
           hail_density = 700.0; before_qh = 2.0e-3
        case (5)
           cloud_diameter = 30.0e-6; hail_diameter = 2.0e-3
           hail_density = 800.0
        case (6)
           cloud_diameter = 50.0e-6; hail_diameter = 3.0e-3
           hail_density = 900.0
        case (7)
           cloud_diameter = 60.0e-6; hail_diameter = 5.0e-3
           hail_density = 500.0
        case (8)
           cloud_diameter = 70.0e-6; hail_diameter = 10.0e-3
           hail_density = 700.0
        case (9)
           cloud_diameter = 25.0e-6; hail_diameter = 20.0e-3
           hail_density = 800.0; before_qc = 1.0e-6; before_qh = 2.0e-3
        case (10)
           cloud_diameter = 50.0e-6; hail_diameter = 30.0e-3
           hail_density = 900.0; before_qc = 1.0e-3; before_qh = 1.0e-6
        case default
           cloud_diameter = 80.0e-6; hail_diameter = 40.0e-3
           hail_density = 600.0; before_qc = 2.0e-3; before_qh = 5.0e-3
        end select

        exner = (pressure/100000.0)**(287.04/1004.0)
        table_index = int((temperature-163.15)/0.002 + 1.5)
        table_index = min(1000001, max(1, table_index))
        table_temperature = 163.15 + real(table_index-1)*0.002
        qvs = (380.0/pressure)*exp(17.2693882*(table_temperature-273.15) &
             /(table_temperature-35.86))
        before_qv = qvs
        cloud_mass = 1000.0*(3.141592653589793/6.0)*cloud_diameter**3
        hail_mass = hail_density*(3.141592653589793/6.0) &
             *hail_diameter**3
        if (before_qc > 0.0) then
           before_nc = before_qc/cloud_mass
        else
           before_nc = 0.0
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
  print '(A,1X,A)', 'NSSL2_HAIL_CLOUD_RIMING_ORACLE_COMPLETE', &
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
    a1(:,:,:,lhl) = before_qh
    a1(:,:,:,lnc) = before_nc*rho
    a1(:,:,:,lnhl) = before_nh*rho
    a1(:,:,:,lvhl) = before_vh*rho
    u0 = temperature
    u1 = 0.0; u2 = 0.0; u3 = 0.0; u4 = 0.0; u5 = 0.0
    u6 = 0.0; u7 = 0.0; u8 = 0.0; u9 = 0.0
    uu0 = 380.0/pressure; uu7 = 1.0; zg = cell_depth; dd = rho
    pp2 = exner; ppn = pressure; ww = 0.0; tt3 = 0.0; tee = 0.0
    aa = 0.0; rp = 0.0; ep = 0.0; alp = 0.0; el = 0.0; th = 0.0
    timevtcalc = 0.0d0

    call nssl_2mom_gs(1,1,3,na,1,0,0,dt,zg,                &
         u0,u1,u2,u3,u4,u5,u6,u7,u8,u9,a1,dd,pp2,ppn,ww,0, &
         uu0,uu7,1.0,1.0,1.0,1,ido,xdnmx,xdnmn,cdx,xdn0,  &
         tt3,tee,th,1,1000.0,1000.0,3,timevtcalc,aa,.false.,&
         .false.,rp,ep,alp,el,1,1,1,1,1)

    write(output_unit,'(2(I0,","),28(ES24.16E3,","),ES24.16E3)') &
         case_id, repetition, dt, cell_depth, rho, pressure, exner, &
         temperature, cloud_diameter, hail_diameter, hail_density, &
         theta_before, before_qv, before_qc, before_nc, before_qh, &
         before_nh, before_vh, a1(1,1,2,lt), a1(1,1,2,lv), &
         a1(1,1,2,lc), a1(1,1,2,lnc)/rho, a1(1,1,2,lhl), &
         a1(1,1,2,lnhl)/rho, a1(1,1,2,lvhl)/rho, a1(1,1,2,lh), &
         a1(1,1,2,lnh)/rho, a1(1,1,2,lvh)/rho, a1(1,1,2,li), &
         a1(1,1,2,ls), a1(1,1,2,lr)
  end subroutine run_cell
end program nssl2_hail_cloud_riming_oracle
