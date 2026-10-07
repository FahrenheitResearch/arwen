module module_state_description
  implicit none
  integer, parameter :: param_first_scalar = 2
  ! chem, gocart_simple package order (registry.chem:4022)
  integer, parameter :: p_so2 = 2, p_sulf = 3, p_dms = 4, p_msa = 5,     &
                        p_p25 = 6, p_bc1 = 7, p_bc2 = 8, p_oc1 = 9,       &
                        p_oc2 = 10, p_dust_1 = 11, p_dust_2 = 12,         &
                        p_dust_3 = 13, p_dust_4 = 14, p_dust_5 = 15,      &
                        p_seas_1 = 16, p_seas_2 = 17, p_seas_3 = 18,      &
                        p_seas_4 = 19, p_p10 = 20
  integer, parameter :: num_chem = 20
  ! species of other packages the GOCART files name but never reach here
  integer, parameter :: p_ac0 = 1, p_corn = 1, p_p25i = 1, p_p25j = 1,    &
                        p_soila = 1, p_naai = 1, p_naaj = 1, p_seas = 1,  &
                        p_h2o2 = 1, p_ho = 1, p_no3 = 1
  ! moist (Thompson-like set; only the indices are used)
  integer, parameter :: p_qv = 2, p_qc = 3, p_qr = 4, p_qi = 5,           &
                        p_qs = 6, p_qg = 7, num_moist = 7
  ! emis_dust (dustgocart / dustgocartafwa, registry.chem:4127-4128)
  integer, parameter :: p_edust1 = 2, p_edust2 = 3, p_edust3 = 4,         &
                        p_edust4 = 5, p_edust5 = 6, num_emis_dust = 6
  ! emis_seas (seasgocart, registry.chem:4133)
  integer, parameter :: p_eseas1 = 2, p_eseas2 = 3, p_eseas3 = 4,         &
                        p_eseas4 = 5, num_emis_seas = 5
  ! chem_opt package values (registry.chem package lines)
  integer, parameter :: gocart_simple = 300, gocartracm_kpp = 301,        &
                        gocartradm2 = 303, chem_vash = 400, dust = 401,   &
                        chem_volc = 402, mozcart_kpp = 112,               &
                        t1_mozcart_kpp = 114, cb05_sorg_aq_kpp = 131,     &
                        cb05_sorg_vbs_aq_kpp = 132
  ! sf_surface_physics package values (Registry.EM_COMMON:3156-3161)
  integer, parameter :: lsmscheme = 2, ruclsmscheme = 3, pxlsmscheme = 7
  ! dust_opt / seas_opt / dmsemis_opt values (registry.chem:4127-4136)
  integer, parameter :: dustgocart = 1, dustgocartafwa = 3, dustuoc = 4,  &
                        seasgocart = 1, dmsgocart = 1
end module module_state_description
