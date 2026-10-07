! The emiss_opt == 6 block of WRF v4.7.1 gocart emissions
! (chem/emissions_driver.F:1597-1619), included byte for byte by
! build_inventory.sh, inside a subroutine whose declarations are those of
! emissions_driver for the names the block reads (chem, emis_ant, rho_phy,
! alt, dz8w REAL; dtstep REAL; conv REAL; ksub = 0 as at :747).
module inventory_harness
  use module_configure
  use module_state_description
  implicit none
contains
  subroutine gocart_emiss_opt6(config_flags, chem, emis_ant, rho_phy, alt,   &
                               dz8w, dtstep,                                 &
                               ims, ime, jms, jme, kms, kme,                 &
                               its, ite, jts, jte, kts, kte)
    type(grid_config_rec_type), intent(in) :: config_flags
    integer, intent(in) :: ims, ime, jms, jme, kms, kme,                     &
                           its, ite, jts, jte, kts, kte
    real, dimension(ims:ime, kms:kme, jms:jme, num_chem), intent(inout) :: chem
    real, dimension(ims:ime, 1:config_flags%kemit, jms:jme, num_emis_ant),  &
          intent(in) :: emis_ant
    real, dimension(ims:ime, kms:kme, jms:jme), intent(in) :: rho_phy, alt, dz8w
    real, intent(in) :: dtstep
    integer :: i, j, k, ksub
    real :: conv
    ksub = 0
#include "emiss_opt6.inc"
  end subroutine gocart_emiss_opt6
end module inventory_harness
