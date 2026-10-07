! GSL 3e6660c6df54e95a0871e990c2294dd397ae3860
! physics/smoke_dust/module_plumerise.F90:143-166, copied by the harness.
! This oracle isolates driver overrides and injection, not MAKEPLUME.
program gsl_run_ebu_distribute
  use machine, only: kind_phys
  use oracle_io
  implicit none
  integer, parameter :: ims=1, ime=1, jms=1, jme=1, kts=1, kte=120
  integer :: i=1,j=1,k,kp1,kp2,c,wind_eff_opt,kpbl(1,1),k_min(1,1),k_max(1,1)
  real(kind_phys) :: frp_inst(1,1),uspdavg2d(1,1),hpbl2d(1,1)
  real(kind_phys) :: frp_min=1.e7,frp_wthreshold=1.e9,zpbl_lim=2.e3,uspd_lim=5.
  real(kind_phys) :: flam_frac(1,1),ebu_in(1,1),z_at_w(1,kte+1,1),ebu(1,kte,1),dz_plume
  character(1024) :: root
  character(32) :: case
  call oracle_root(root)
  do c=1,12
    do k=1,kte+1
      z_at_w(1,k,1)=real(k-1,kind_phys)*37.25+real((k-1)**2,kind_phys)*2.125
    enddo
    frp_inst=1.e8; uspdavg2d=5.; hpbl2d=2001.; kpbl=20; wind_eff_opt=1
    kp1=4; kp2=18; ebu_in=1.234567; ebu=0.
    select case(c)
    case(1); frp_inst=0.
    case(2); frp_inst=9999999.
    case(3); frp_inst=1.e7
    case(4); frp_inst=10000001.
    case(5); uspdavg2d=4.999
    case(6); hpbl2d=2000.
    case(7); wind_eff_opt=0
    case(8); frp_inst=1.e9
    case(9); frp_inst=1000000064.
    case(10); kpbl=120; kp1=51; kp2=52
    case(11); kpbl=1; ebu_in=0.
    case(12); frp_inst=2.e10; kp1=119; kp2=121
    end select
    write(case,'("case_",I2.2)') c
    call oracle_open(trim(case))
    call oracle_put('frp_inst',real(frp_inst(1,1),4))
    call oracle_put('uspdavg2d',real(uspdavg2d(1,1),4))
    call oracle_put('hpbl2d',real(hpbl2d(1,1),4))
    call oracle_put('kpbl',kpbl(1,1))
    call oracle_put('wind_eff_opt',wind_eff_opt)
    call oracle_put('plume_k_min',kp1)
    call oracle_put('plume_k_max',kp2)
    call oracle_put('ebu_in',real(ebu_in(1,1),4))
    call oracle_put('z_at_w',real(z_at_w(1,:,1),4))
    include 'plume_override.inc'
    call oracle_put('k_min',k_min(1,1))
    call oracle_put('k_max',k_max(1,1))
    call oracle_put('flam_frac',real(flam_frac(1,1),4))
    call oracle_put('ebu',real(ebu(1,:,1),4))
    call oracle_close()
  enddo
end program gsl_run_ebu_distribute
