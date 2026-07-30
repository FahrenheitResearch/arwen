program nssl2_nucond_production_oracle
  use module_mp_nssl_2mom, only: nssl_2mom_init, NUCOND
  implicit none

  integer, parameter :: nx = 4, nz = 16, na = 40, nxtra = 20
  integer, parameter :: lt = 1, lv = 2, lc = 3, lr = 4
  integer, parameter :: lccn = 9, lnc = 10, lnr = 11
  integer :: i, k, unit, case_id, ltemq
  real :: nssl_params(20), dt, pressure, exner, rho, temperature
  real :: theta_before, qvs, qv_before, qc_before, qr_before
  real :: nc_before, nr_before, qnn_before, supersaturation_percent
  real :: vertical_velocity, cloud_diameter, rain_diameter
  real :: qv_after, theta_after, qc_after, qr_after
  real :: nc_after, nr_after, qnn_after, table_temperature
  character(len=512) :: output_path

  call get_command_argument(1, output_path)
  if (len_trim(output_path) == 0) then
     error stop 'usage: nssl2_nucond_production_oracle OUTPUT.csv'
  endif

  nssl_params = 0.0
  nssl_params(1:10) = [0.5e9, 0.0, 1.0, 4.0e5, 4.0e4, &
                       8.0e5, 3.0e6, 500.0, 900.0, 100.0]
  call nssl_2mom_init(nssl_params=nssl_params, ipctmp=5, mixphase=0, &
       nssl_density_on=.true., nssl_hail_on=.true.,                 &
       nssl_ccn_on=.true., nssl_icdx=6, nssl_icdxhl=6)

  open(newunit=unit, file=trim(output_path), status='replace', action='write')
  write(unit,'(A)') 'case,i,k,dt_s,pressure_pa,exner,rho_kg_m3,temperature_k,supersaturation_percent,w_m_s,qv_before,theta_before_k,qc_before,qr_before,nc_before_per_kg,nr_before_per_kg,qnn_before_per_kg,qv_after,theta_after_k,qc_after,qr_after,nc_after_per_kg,nr_after_per_kg,qnn_after_per_kg'

  do k = 1, nz
     do i = 1, nx
        case_id = mod((k-1)*nx+i-1, 16)
        dt = 5.0
        supersaturation_percent = 1.0
        vertical_velocity = 3.0
        qc_before = 2.0e-4
        qr_before = 3.0e-4
        cloud_diameter = 16.0e-6
        rain_diameter = 800.0e-6
        qnn_before = 0.65*(0.5e9/1.225)

        select case (case_id)
        case (0)
           supersaturation_percent = 0.10; dt = 1.0
        case (1)
           supersaturation_percent = 0.49; dt = 10.0
           qr_before = 8.0e-4; rain_diameter = 2.0e-3
        case (2)
           supersaturation_percent = 1.5; dt = 30.0
           vertical_velocity = 10.0; qnn_before = 0.20*(0.5e9/1.225)
        case (3)
           supersaturation_percent = 95.0; dt = 5.0
           qnn_before = 0.80*(0.5e9/1.225)
        case (4)
           supersaturation_percent = 250.0; dt = 1.0
           vertical_velocity = 20.0
        case (5)
           supersaturation_percent = -1.0; dt = 10.0
        case (6)
           supersaturation_percent = -25.0; dt = 60.0
           qc_before = 1.0e-7; cloud_diameter = 20.0e-6
        case (7)
           supersaturation_percent = 5.0; dt = 30.0
           qc_before = 0.0; qr_before = 0.0
           cloud_diameter = 8.0e-6; rain_diameter = 100.0e-6
        case (8)
           supersaturation_percent = 0.39; dt = 5.0
           qc_before = 0.0; qr_before = 0.0
        case (9)
           supersaturation_percent = 190.5; dt = 10.0
           qc_before = 0.0; qr_before = 5.0e-4
           qnn_before = 0.95*(0.5e9/1.225)
        case (10)
           supersaturation_percent = 0.8; dt = 5.0
           cloud_diameter = 1.0e-6; qnn_before = 0.05*(0.5e9/1.225)
        case (11)
           supersaturation_percent = 2.0; dt = 30.0
           rain_diameter = 20.0e-6
        case (12)
           supersaturation_percent = 0.0; dt = 1.0
           vertical_velocity = 0.0
        case (13)
           supersaturation_percent = 10.0; dt = 60.0
           vertical_velocity = 25.0; qnn_before = 0.01*(0.5e9/1.225)
        case (14)
           supersaturation_percent = 20.0; dt = 5.0
           vertical_velocity = -3.0
        case default
           supersaturation_percent = -0.1; dt = 10.0
           qc_before = 0.5e-13; qr_before = 0.5e-12
           qnn_before = 0.30*(0.5e9/1.225)
        end select

        temperature = 244.0 + 13.0*real(mod(k-1,4)) + 0.21*real(i-1)
        pressure = 100000.0 - 11000.0*real(mod(k-1,4)) - 173.0*real(i-1)
        exner = (pressure/100000.0)**(287.04/1004.0)
        theta_before = temperature/exner
        temperature = theta_before*exner
        ltemq = int((temperature-163.15)/0.002 + 1.5)
        ltemq = min(1000001, max(1, ltemq))
        table_temperature = 163.15 + real(ltemq-1)*0.002
        qvs = (380.0/pressure)*exp(17.2693882* &
             (table_temperature-273.15)/(table_temperature-35.86))
        qv_before = (1.0 + 0.01*supersaturation_percent)*qvs
        rho = pressure/(287.04*temperature*(1.0 + 0.608*qv_before))
        if (qc_before > 0.0) then
           nc_before = qc_before/(1000.0*0.523599*cloud_diameter**3)
        else
           nc_before = 0.0
        endif
        if (qr_before > 0.0) then
           nr_before = qr_before/(1000.0*0.523599*rain_diameter**3)
        else
           nr_before = 0.0
        endif

        call run_case(qv_after, theta_after, qc_after, qr_after, &
             nc_after, nr_after, qnn_after)

        write(unit,'(3(I0,","),20(ES24.16E3,","),ES24.16E3)') &
             case_id, i, k, dt, pressure, exner, rho, temperature, &
             supersaturation_percent, vertical_velocity, qv_before, &
             theta_before, qc_before, qr_before, nc_before, nr_before, &
             qnn_before, qv_after, theta_after, qc_after, qr_after, &
             nc_after, nr_after, qnn_after
     enddo
  enddo
  close(unit)

  print '(A,1X,A)', 'NSSL2_NUCOND_PRODUCTION_ORACLE_COMPLETE', &
       trim(output_path)

contains

  subroutine run_case(output_qv, output_theta, output_qc, output_qr, &
       output_nc, output_nr, output_qnn)
    real, intent(out) :: output_qv, output_theta, output_qc, output_qr
    real, intent(out) :: output_nc, output_nr, output_qnn
    real :: a1(1,1,5,na), t0(1,1,5), t9(1,1,5)
    real :: dz3d(1,1,5), density(1,1,5), pp2(1,1,5), ppn(1,1,5)
    real :: ww(1,1,5), aa(1,1,5,nxtra), ssfilt(1,1,5)
    real :: t00(1,1,5), t77(1,1,5)

    a1 = 0.0
    a1(:,:,:,lt) = theta_before
    a1(:,:,:,lv) = qv_before
    a1(:,:,:,lc) = qc_before
    a1(:,:,:,lr) = qr_before
    a1(:,:,:,lnc) = nc_before*rho
    a1(:,:,:,lnr) = nr_before*rho
    a1(:,:,:,lccn) = qnn_before*rho
    t0 = temperature
    t9 = 0.0
    dz3d = 1000.0
    density = rho
    pp2 = exner
    ppn = pressure
    ww = vertical_velocity
    aa = 0.0
    ssfilt = 0.0
    t00 = 380.0/pressure
    t77 = exner

    call NUCOND(1,1,5,na,1,0,0,dt,1,dz3d,t0,t9,a1,density, &
         pp2,ppn,ww,1,aa,.false.,ssfilt,t00,t77,.false.)

    output_theta = a1(1,1,3,lt)
    output_qv = a1(1,1,3,lv)
    output_qc = a1(1,1,3,lc)
    output_qr = a1(1,1,3,lr)
    output_nc = a1(1,1,3,lnc)/rho
    output_nr = a1(1,1,3,lnr)/rho
    output_qnn = a1(1,1,3,lccn)/rho
  end subroutine run_case
end program nssl2_nucond_production_oracle
