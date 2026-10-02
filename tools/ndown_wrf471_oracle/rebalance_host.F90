! WRF v4.7.1's ndown rebalance, byte-unmodified.  build.sh writes
! rebalance_wrf471.inc as lines 4982-5266 of dyn_em/module_initialize_real.F
! (SUBROUTINE rebalance ... END SUBROUTINE rebalance) after checking that
! file's sha256 against SOURCES.sha256, and this module is the scope the
! subroutine sat in: the modules module_initialize_real USEs for the names
! the subroutine reads, and the module variable hold_ups it assigns.
! dummy_new_args.inc / dummy_new_decl.inc stand in for the Registry's
! generated argument list: the one 4-D array the routine reads, moist.
MODULE ndown_rebalance_host
   USE module_configure
   USE module_domain
   USE module_model_constants
   USE module_state_description
   USE module_wrf_error
   IMPLICIT NONE
   LOGICAL :: hold_ups
CONTAINS
#include "rebalance_wrf471.inc"
END MODULE ndown_rebalance_host
