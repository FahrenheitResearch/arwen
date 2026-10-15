module module_state_description
 integer, parameter :: ltng_crm_PR92w=1,ltng_crm_PR92z=2,ltng_lpi=3,ltng_cpm_PR92z=11
 integer, parameter :: GDSCHEME=93,G3SCHEME=5,GFSCHEME=3
 logical, parameter :: F_QG=.true.
 integer, parameter :: num_moist=6,P_QV=1,P_QC=2,P_QR=3,P_QI=4,P_QS=5,P_QG=6
end module
module module_model_constants
end module
module module_wrf_error
 character(512) :: wrf_err_message
contains
 subroutine wrf_debug(level,message)
 integer :: level
 character(*) :: message
 end subroutine
 subroutine wrf_error_fatal(message)
 character(*) :: message
 print *, trim(message)
 error stop 1
 end subroutine
end module
module module_dm
contains
 real function wrf_dm_sum_real(x)
 real :: x
 wrf_dm_sum_real=x
 end function
 real function wrf_dm_max_real(x)
 real :: x
 wrf_dm_max_real=x
 end function
end module
module module_utility
 type WRFU_Time
 integer :: month=1
 end type
contains
 subroutine WRFU_TimeGet(time,mm)
 type(WRFU_Time) :: time
 integer :: mm
 mm=time%month
 end subroutine
end module
subroutine nl_set_lightning_dt(id,dt)
 integer :: id
 real :: dt
end subroutine
