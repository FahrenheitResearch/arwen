use module_model_constants
integer, parameter :: P_QV=1
! chem_prep never reads config_flags (module_chem_utilities.F:8-181).
type grid_config_rec_type
 integer :: unused=0
end type
