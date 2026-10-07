
! numgas is read only by unsupported methods, all of which stop below.
! Returning zero here cannot affect methods 5, 6 or 7.
integer function get_last_gas(opt)
 integer :: opt
 get_last_gas=0
end function

subroutine PV2O3_CONST()
 error stop "unreachable boundary branch: PV2O3_CONST"
end subroutine

subroutine PVS()
 error stop "unreachable boundary branch: PVS"
end subroutine

subroutine bdy_chem_value()
 error stop "unreachable boundary branch: bdy_chem_value"
end subroutine

subroutine bdy_chem_value_cam_mam()
 error stop "unreachable boundary branch: bdy_chem_value_cam_mam"
end subroutine

subroutine bdy_chem_value_cb05()
 error stop "unreachable boundary branch: bdy_chem_value_cb05"
end subroutine

subroutine bdy_chem_value_cb05_vbs()
 error stop "unreachable boundary branch: bdy_chem_value_cb05_vbs"
end subroutine

subroutine bdy_chem_value_cbmz()
 error stop "unreachable boundary branch: bdy_chem_value_cbmz"
end subroutine

subroutine bdy_chem_value_ghg()
 error stop "unreachable boundary branch: bdy_chem_value_ghg"
end subroutine

subroutine bdy_chem_value_mosaic()
 error stop "unreachable boundary branch: bdy_chem_value_mosaic"
end subroutine

subroutine bdy_chem_value_racm()
 error stop "unreachable boundary branch: bdy_chem_value_racm"
end subroutine

subroutine bdy_chem_value_sorgam()
 error stop "unreachable boundary branch: bdy_chem_value_sorgam"
end subroutine
