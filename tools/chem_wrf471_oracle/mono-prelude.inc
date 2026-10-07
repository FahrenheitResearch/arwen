! Only configuration storage and the error buffer, no physical computation.
! Fields read by dyn_em/module_advect_em.F:6069-7885,9495-10560.
implicit none
type grid_config_rec_type
  integer :: h_sca_adv_order=5, v_sca_adv_order=3
  logical :: periodic_x=.false., periodic_y=.false., polar=.false.
  logical :: symmetric_xs=.false., symmetric_xe=.false.
  logical :: symmetric_ys=.false., symmetric_ye=.false.
  logical :: open_xs=.false., open_xe=.false.
  logical :: open_ys=.false., open_ye=.false.
  logical :: specified=.false., nested=.false.
end type
character(len=1024) :: wrf_err_message
