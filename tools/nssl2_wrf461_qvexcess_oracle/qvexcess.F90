program nssl2_qvexcess_oracle
  use iso_fortran_env, only: int32
  use module_mp_nssl_2mom, only: QVEXCESS
  implicit none

  integer, parameter :: ncases = 128, nqsat = 1000001
  integer, parameter :: ngs = 1, ngscnt = 1
  real, parameter :: fqsat = 0.002, cbw = 35.86
  real, parameter :: caw = 17.2693882, cpi = 1.0/1004.0
  real, parameter :: cwmasn = 1000.0*0.523599*(4.0e-6)**3
  real, parameter :: cwmasn5 = 1000.0*0.523599*(10.0e-6)**3
  real, parameter :: cwmas20 = 1000.0*0.523599*(40.0e-6)**3
  real, allocatable :: tabqvs(:)
  real :: qv0(ngs), qwvp0(ngs), qcw1(ngscnt), pres(ngs)
  real :: thetap0(ngs), theta0(ngs), pi0(ngs), pk(ngs)
  real :: qv_workspace(ngs), qwvp_workspace(ngs)
  real :: qc_workspace(ngscnt)
  real :: thetap_workspace(ngs), theta_workspace(ngs)
  real :: fcqv1(ngs), felvcp(ngs)
  real :: trace_qss(2), trace_qwv(2), trace_qcw(2), trace_thetap(2)
  integer :: trace_branch(2)
  real :: temperature, table_temperature, target_ss, qss_initial
  real :: vapor_total, vapor_ratio, cloud, temp_limited, latent_heat
  real :: rho, cloud_number, ccn_number, background_ccn, cloud_mean_mass
  real :: qvex, qvex_workspace, trace_qvex, theta_before, theta_delta
  real :: theta_direct_after, theta_split_after
  real :: qv_direct_after, qv_split_after, qc_after
  real :: cloud_number_after, ccn_after, new_cloud_number
  real :: vapor_split, theta_split, deficit
  integer :: case_id, index, ltemq, unit, couple_number
  integer(int32) :: official_bits, trace_bits
  character(len=512) :: output_path

  call get_command_argument(1, output_path)
  if (len_trim(output_path) == 0) then
     error stop 'usage: nssl2_qvexcess_oracle OUTPUT.csv'
  endif

  allocate(tabqvs(nqsat))
  do index = 1, nqsat
     table_temperature = 163.15 + real(index-1)*fqsat
     tabqvs(index) = exp(caw*(table_temperature-273.15)/ &
          (table_temperature-cbw))
  enddo

  open(newunit=unit, file=trim(output_path), status='replace', action='write')
  write(unit,'(A)') 'case,temperature_initial_k,pressure_pa,exner,theta_base_k,theta_perturbation_k,qv_base,qv_perturbation,qc_initial,target_supersaturation_percent,latent_over_cp_k,condensation_factor_k,rho_kg_m3,cloud_number_initial_m3,ccn_initial_m3,background_ccn_m3,cloud_mean_mass_kg,couple_number,iteration1_branch,iteration1_target_qv,iteration1_qv,iteration1_qc,iteration1_theta_perturbation_k,iteration2_branch,iteration2_target_qv,iteration2_qv,iteration2_qc,iteration2_theta_perturbation_k,qvex,theta_direct_after_k,theta_split_after_k,qv_direct_after,qv_split_after,qc_after,cloud_number_after_m3,ccn_after_m3,new_cloud_number_m3,workspace_qvex'

  do case_id = 0, ncases-1
     select case (mod(case_id, 16))
     case (0);  temperature = 163.149
     case (1);  temperature = 163.150
     case (2);  temperature = 180.0
     case (3);  temperature = 233.15
     case (4);  temperature = 244.123
     case (5);  temperature = 263.15
     case (6);  temperature = 273.149
     case (7);  temperature = 273.15
     case (8);  temperature = 300.0
     case (9);  temperature = 313.15
     case (10); temperature = 330.0
     case (11); temperature = 350.0
     case (12); temperature = 2163.149
     case (13); temperature = 2163.150
     case (14); temperature = 2163.151
     case default; temperature = 280.001
     end select

     select case (mod(case_id/2, 8))
     case (0); pres(1) = 15000.0
     case (1); pres(1) = 30000.0
     case (2); pres(1) = 50000.0
     case (3); pres(1) = 70000.0
     case (4); pres(1) = 85000.0
     case (5); pres(1) = 100000.0
     case (6); pres(1) = 105000.0
     case default; pres(1) = 250000.0
     end select
     pi0(1) = (pres(1)/100000.0)**(287.04/1004.0)
     pk(1) = pi0(1)
     theta_before = temperature/pi0(1)

     select case (mod(case_id/4, 4))
     case (0); theta_split = 0.0
     case (1); theta_split = -20.0
     case (2); theta_split = 5.25
     case default; theta_split = 100.0
     end select
     thetap0(1) = theta_split
     theta0(1) = theta_before-thetap0(1)
     theta_before = theta0(1)+thetap0(1)
     temperature = theta_before*pi0(1)

     temp_limited = min(temperature, 313.15)
     temp_limited = max(temp_limited, 233.15)
     latent_heat = 2500837.367*(273.15/temp_limited)** &
          (0.167 + 3.67e-4*temp_limited)
     felvcp(1) = latent_heat*cpi
     fcqv1(1) = 4098.0258*latent_heat*cpi

     select case (mod(case_id, 4))
     case (0); target_ss = 0.0
     case (1); target_ss = 0.4
     case (2); target_ss = 90.0
     case default; target_ss = 250.0
     end select
     ltemq = int((temperature-163.15)/fqsat+1.5)
     ltemq = min(nqsat, max(1, ltemq))
     qss_initial = (0.01*target_ss+1.0)*(380.0/pres(1))*tabqvs(ltemq)

     select case (mod(case_id, 12))
     case (0); vapor_ratio = 0.0
     case (1); vapor_ratio = 0.5
     case (2); vapor_ratio = 1.0-2.0**(-20)
     case (3); vapor_ratio = 1.0
     case (4); vapor_ratio = 1.0+2.0**(-20)
     case (5); vapor_ratio = 1.0001
     case (6); vapor_ratio = 1.02
     case (7); vapor_ratio = 1.5
     case (8); vapor_ratio = 1.9
     case (9); vapor_ratio = 2.5
     case (10); vapor_ratio = 10.0
     case default; vapor_ratio = 0.99
     end select
     vapor_total = qss_initial*vapor_ratio

     select case (mod(case_id/3, 4))
     case (0); vapor_split = 1.0
     case (1); vapor_split = 0.75
     case (2); vapor_split = 1.25
     case default; vapor_split = 0.01
     end select
     qv0(1) = vapor_total*vapor_split
     qwvp0(1) = vapor_total-qv0(1)

     deficit = max(qss_initial-vapor_total, 0.0)
     select case (mod(case_id, 8))
     case (0); cloud = 0.0
     case (1); cloud = 1.0e-15
     case (2); cloud = 1.0e-12
     case (3); cloud = 1.0e-8
     case (4); cloud = max(1.0e-12, 0.5*deficit)
     case (5); cloud = max(1.0e-12, 2.0*deficit)
     case (6); cloud = 5.0e-3
     case default; cloud = 0.2
     end select
     qcw1(1) = cloud

     rho = pres(1)/(287.04*max(temperature, 100.0)* &
          (1.0+0.608*min(vapor_total, 0.1)))
     select case (mod(case_id, 4))
     case (0); cloud_number = 0.0
     case (1); cloud_number = 1.0e6
     case (2); cloud_number = 2.0e8
     case default; cloud_number = 1.0e9
     end select
     select case (mod(case_id/2, 4))
     case (0); ccn_number = 0.0
     case (1); ccn_number = 5.0e6
     case (2); ccn_number = 2.0e8
     case default; ccn_number = 8.0e8
     end select
     background_ccn = rho*408163264.0
     select case (mod(case_id, 6))
     case (0); cloud_mean_mass = cwmasn
     case (1); cloud_mean_mass = cwmasn5
     case (2); cloud_mean_mass = 0.5*cwmas20
     case (3); cloud_mean_mass = cwmas20
     case (4); cloud_mean_mass = 2.0*cwmas20
     case default; cloud_mean_mass = 1.0e-8
     end select
     couple_number = mod(case_id/8, 2)

     call QVEXCESS(ngs,1,qwvp0,qv0,qcw1,pres,thetap0,theta0, &
          qvex,pi0,tabqvs,nqsat,fqsat,cbw,fcqv1,felvcp,target_ss, &
          pk,ngscnt)
     call trace_qvexcess(qwvp0(1),qv0(1),qcw1(1),pres(1), &
          thetap0(1),theta0(1),trace_qvex,pi0(1),tabqvs,nqsat, &
          fqsat,cbw,fcqv1(1),felvcp(1),target_ss,pk(1), &
          trace_branch,trace_qss,trace_qwv,trace_qcw,trace_thetap)
     official_bits = transfer(qvex, official_bits)
     trace_bits = transfer(trace_qvex, trace_bits)
     if (official_bits /= trace_bits) then
        error stop 'diagnostic trace diverged from official QVEXCESS'
     endif

     ! The production workspace API receives already-combined theta and qv.
     ! Compile that representation through the official routine separately;
     ! pre-summing FP32 split inputs can legitimately change the final ulps.
     theta_workspace(1) = theta0(1)+thetap0(1)
     thetap_workspace(1) = 0.0
     qv_workspace(1) = qv0(1)+qwvp0(1)
     qwvp_workspace(1) = 0.0
     qc_workspace(1) = qcw1(1)
     call QVEXCESS(ngs,1,qwvp_workspace,qv_workspace,qc_workspace,pres, &
          thetap_workspace,theta_workspace,qvex_workspace,pi0,tabqvs, &
          nqsat,fqsat,cbw,fcqv1,felvcp,target_ss,pk,ngscnt)

     theta_delta = (felvcp(1)*qvex)/pi0(1)
     theta_direct_after = theta_before
     theta_split_after = theta_before
     qv_direct_after = qv0(1)+qwvp0(1)
     qv_split_after = qv_direct_after
     qc_after = qcw1(1)
     cloud_number_after = cloud_number
     ccn_after = ccn_number
     new_cloud_number = 0.0
     if (qvex > 0.0) then
        theta_direct_after = theta_direct_after+theta_delta
        theta_split_after = theta0(1)+(thetap0(1)+theta_delta)
        qv_direct_after = qv_direct_after-qvex
        qv_split_after = qv0(1)+(qwvp0(1)-qvex)
        qc_after = qc_after+qvex
        if (couple_number == 1) then
           new_cloud_number = min(max(ccn_number,background_ccn), &
                rho*qvex/max(cwmasn5,max(cwmas20,cloud_mean_mass)))
           cloud_number_after = cloud_number_after+new_cloud_number
           ccn_after = max(0.0,ccn_after-new_cloud_number)
        endif
     endif

     write(unit,'(I0,37(",",ES24.16E3))') case_id, temperature, pres(1), &
          pi0(1), theta0(1), thetap0(1), qv0(1), qwvp0(1), qcw1(1), &
          target_ss, felvcp(1), fcqv1(1), rho, cloud_number, ccn_number, &
          background_ccn, cloud_mean_mass, real(couple_number), &
          real(trace_branch(1)), trace_qss(1), trace_qwv(1), trace_qcw(1), &
          trace_thetap(1), real(trace_branch(2)), trace_qss(2), &
          trace_qwv(2), trace_qcw(2), trace_thetap(2), qvex, &
          theta_direct_after, theta_split_after, qv_direct_after, &
          qv_split_after, qc_after, cloud_number_after, ccn_after, &
          new_cloud_number, qvex_workspace
  enddo
  close(unit)

  print '(A,1X,A)', 'NSSL2_QVEXCESS_ORACLE_COMPLETE', trim(output_path)

contains

  subroutine trace_qvexcess(qwvp_input,qv_input,qc_input,pressure, &
       thetap_input,theta_input,qvex_output,exner,table,n_table, &
       table_step,water_offset,condensation_factor,latent_over_cp, &
       target_ss,temperature_exner,branch,target_qv,trial_qv,trial_qc, &
       trial_thetap)
    integer, intent(in) :: n_table
    real, intent(in) :: qwvp_input,qv_input,qc_input,pressure
    real, intent(in) :: thetap_input,theta_input,exner
    real, intent(in) :: table(n_table),table_step,water_offset
    real, intent(in) :: condensation_factor,latent_over_cp,target_ss
    real, intent(in) :: temperature_exner
    integer, intent(out) :: branch(2)
    real, intent(out) :: qvex_output,target_qv(2),trial_qv(2)
    real, intent(out) :: trial_qc(2),trial_thetap(2)
    integer :: iteration, table_index
    real :: pqs,thetap,theta,qwvp,qvap,temperature
    real :: qwv,qcw,qvs,qss,dqcw,dqwv,dqvcnd

    pqs = 380.0/pressure
    thetap = thetap_input
    theta = thetap+theta_input
    qwvp = qwvp_input
    qvap = max(qwvp_input+qv_input,0.0)
    temperature = theta*temperature_exner
    qwv = max(0.0,qvap)
    qcw = max(0.0,qc_input)
    table_index = int((temperature-163.15)/table_step+1.5)
    table_index = min(n_table,max(1,table_index))
    qvs = pqs*table(table_index)
    qss = (0.01*target_ss+1.0)*qvs

    do iteration = 1,2
       dqcw = 0.0
       dqwv = qwv-qss
       branch(iteration) = 0
       if (dqwv < 0.0) then
          if (qcw > -dqwv) then
             dqcw = dqwv
             dqwv = 0.0
             branch(iteration) = -1
          else
             dqcw = -qcw
             dqwv = dqwv+qcw
             branch(iteration) = -2
          endif
          qwvp = qwvp-dqcw
          qcw = qcw+dqcw
          thetap = thetap+(1.0/exner)*(latent_over_cp*dqcw)
       endif
       if (dqwv >= 0.0) then
          dqvcnd = dqwv/(1.0+condensation_factor*qss/ &
               ((temperature-water_offset)**2))
          dqcw = dqvcnd
          if (dqwv > 0.0) branch(iteration) = 1
          thetap = thetap+(latent_over_cp*dqcw)/exner
          qwvp = qwvp-dqvcnd
          qcw = qcw+dqcw
       endif
       theta = thetap+theta_input
       temperature = theta*temperature_exner
       qvap = max(qwvp+qv_input,0.0)
       table_index = int((temperature-163.15)/table_step+1.5)
       table_index = min(n_table,max(1,table_index))
       qvs = pqs*table(table_index)
       qcw = max(0.0,qcw)
       qwv = max(0.0,qvap)
       qss = (0.01*target_ss+1.0)*qvs
       target_qv(iteration) = qss
       trial_qv(iteration) = qwv
       trial_qc(iteration) = qcw
       trial_thetap(iteration) = thetap
    enddo
    qvex_output = max(0.0,qcw-qc_input)
  end subroutine trace_qvexcess
end program nssl2_qvexcess_oracle
