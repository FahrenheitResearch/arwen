program nssl2_cloud_interior_renucleation_oracle
  use module_mp_nssl_2mom, only: nssl_2mom_init, NUCOND
  implicit none

  integer, parameter :: nx = 4, nz = 12, na = 40, nxtra = 20
  integer, parameter :: lt = 1, lv = 2, lc = 3, lccn = 9, lnc = 10
  integer :: i, k, unit, case_id, ltemq
  real :: nssl_params(20), dt, pressure, exner, rho, temperature
  real :: theta_before, qvs, qv_before, qc_before, nc_before, qnn_before
  real :: supersaturation_percent, vertical_velocity
  real :: qv_after, theta_after, qc_after, nc_after, qnn_after
  real :: table_temperature
  character(len=512) :: output_path

  call get_command_argument(1, output_path)
  if (len_trim(output_path) == 0) then
     error stop 'usage: nssl2_cloud_interior_renucleation_oracle OUTPUT.csv'
  endif

  nssl_params = 0.0
  nssl_params(1:10) = [0.5e9, 0.0, 1.0, 4.0e5, 4.0e4, &
                       8.0e5, 3.0e6, 500.0, 900.0, 100.0]
  call nssl_2mom_init(nssl_params=nssl_params, ipctmp=5, mixphase=0, &
       nssl_density_on=.true., nssl_hail_on=.true.,                 &
       nssl_ccn_on=.true., nssl_icdx=6, nssl_icdxhl=6)

  open(newunit=unit, file=trim(output_path), status='replace', action='write')
  write(unit,'(A)') 'case,i,k,dt_s,pressure_pa,exner,rho_kg_m3,temperature_k,supersaturation_percent,w_m_s,qv_before,theta_before_k,qc_before,nc_before_per_kg,qnn_before_per_kg,qv_after,theta_after_k,qc_after,nc_after_per_kg,qnn_after_per_kg'

  do k = 1, nz
     do i = 1, nx
        case_id = mod((k-1)*nx+i-1, 12)
        select case (case_id)
        case (0)
           supersaturation_percent = 0.51
           vertical_velocity = 0.05
           qc_before = 1.0e-4; nc_before = 1.0e8
           qnn_before = 0.50*(0.5e9/1.225); dt = 1.0
        case (1)
           supersaturation_percent = 0.75
           vertical_velocity = 0.5
           qc_before = 5.0e-5; nc_before = 2.0e8
           qnn_before = 0.10*(0.5e9/1.225); dt = 5.0
        case (2)
           supersaturation_percent = 1.0
           vertical_velocity = 1.0
           qc_before = 2.0e-4; nc_before = 3.0e8
           qnn_before = 0.75*(0.5e9/1.225); dt = 10.0
        case (3)
           supersaturation_percent = 2.0
           vertical_velocity = 5.0
           qc_before = 5.0e-4; nc_before = 5.0e7
           qnn_before = 1.00*(0.5e9/1.225); dt = 30.0
        case (4)
           supersaturation_percent = 5.0
           vertical_velocity = 20.0
           qc_before = 1.0e-3; nc_before = 8.0e8
           qnn_before = 0.02*(0.5e9/1.225); dt = 60.0
        case (5)
           supersaturation_percent = 20.0
           vertical_velocity = 0.0
           qc_before = 3.0e-4; nc_before = 1.5e8
           qnn_before = 0.80*(0.5e9/1.225); dt = 1.0
        case (6)
           supersaturation_percent = 50.0
           vertical_velocity = -2.0
           qc_before = 7.5e-4; nc_before = 4.0e8
           qnn_before = 0.60*(0.5e9/1.225); dt = 5.0
        case (7)
           supersaturation_percent = 89.0
           vertical_velocity = 2.0
           qc_before = 2.5e-4; nc_before = 2.0e7
           qnn_before = 0.30*(0.5e9/1.225); dt = 10.0
        case (8)
           supersaturation_percent = 0.60
           vertical_velocity = 10.0
           qc_before = 1.0e-6; nc_before = 1.0e12
           qnn_before = 0.90*(0.5e9/1.225); dt = 30.0
        case (9)
           supersaturation_percent = 1.5
           vertical_velocity = 0.1
           qc_before = 1.0e-3; nc_before = 1.0e3
           qnn_before = 1.20*(0.5e9/1.225); dt = 60.0
        case (10)
           supersaturation_percent = 3.0
           vertical_velocity = 3.0
           qc_before = 1.0e-5; nc_before = 5.0e8
           qnn_before = 0.001*(0.5e9/1.225); dt = 1.0
        case default
           supersaturation_percent = 10.0
           vertical_velocity = 30.0
           qc_before = 5.0e-4; nc_before = 2.5e8
           qnn_before = 0.0; dt = 5.0
        end select

        temperature = 262.0 + 7.5*real(mod(k-1,4)) + 0.19*real(i-1)
        pressure = 100000.0 - 9000.0*real(mod(k-1,4)) - 173.0*real(i-1)
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

        call run_case(qv_after, theta_after, qc_after, nc_after, qnn_after)

        write(unit,'(3(I0,","),16(ES24.16E3,","),ES24.16E3)') &
             case_id, i, k, dt, pressure, exner, rho, temperature, &
             supersaturation_percent, vertical_velocity, qv_before, &
             theta_before, qc_before, nc_before, qnn_before, qv_after, &
             theta_after, qc_after, nc_after, qnn_after
     enddo
  enddo
  close(unit)

  print '(A,1X,A)', 'NSSL2_CLOUD_INTERIOR_RENUCLEATION_ORACLE_COMPLETE', &
       trim(output_path)

contains

  subroutine run_case(output_qv, output_theta, output_qc, output_nc, output_qnn)
    real, intent(out) :: output_qv, output_theta, output_qc, output_nc
    real, intent(out) :: output_qnn
    real :: a1(1,1,3,na), t0(1,1,3), t9(1,1,3)
    real :: dz3d(1,1,3), density(1,1,3), pp2(1,1,3), ppn(1,1,3)
    real :: ww(1,1,3), aa(1,1,3,nxtra), ssfilt(1,1,3)
    real :: t00(1,1,3), t77(1,1,3)

    a1 = 0.0
    a1(:,:,:,lt) = theta_before
    a1(:,:,:,lv) = qv_before
    a1(:,:,:,lc) = qc_before
    a1(:,:,:,lnc) = nc_before*rho
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

    call NUCOND(1,1,3,na,1,0,0,dt,1,dz3d,t0,t9,a1,density, &
         pp2,ppn,ww,1,aa,.false.,ssfilt,t00,t77,.false.)

    output_theta = a1(1,1,2,lt)
    output_qv = a1(1,1,2,lv)
    output_qc = a1(1,1,2,lc)
    output_nc = a1(1,1,2,lnc)/rho
    output_qnn = a1(1,1,2,lccn)/rho
  end subroutine run_case
end program nssl2_cloud_interior_renucleation_oracle
