! Registry/registry.chem:3940,4022. Unused Registry indices equal 1.
implicit none
type grid_config_rec_type
 integer :: chem_opt
 logical :: do_pvozone=.false.
end type
integer, parameter :: param_first_scalar=2
! chem/module_data_radm2.F:5; module_data_sorgam.F:949-950.
real, parameter :: epsilc=1.e-16, conmin=1.e-16
! chem/module_data_sorgam.F:557-558; only unsupported branches read it.
real, parameter :: rgasuniv=8.314510
real :: fracref(100)=1.
integer, external :: get_last_gas
integer, parameter :: p_bc1=7
integer, parameter :: p_bc2=8
integer, parameter :: p_co=7
integer, parameter :: p_co2=1
integer, parameter :: p_corn=1
integer, parameter :: p_dms=4
integer, parameter :: p_dust_1=11
integer, parameter :: p_dust_2=12
integer, parameter :: p_dust_3=13
integer, parameter :: p_dust_4=14
integer, parameter :: p_dust_5=15
integer, parameter :: p_hcl=1
integer, parameter :: p_ho2=1
integer, parameter :: p_isopo2=1
integer, parameter :: p_msa=5
integer, parameter :: p_mtf=1
integer, parameter :: p_nu0=1
integer, parameter :: p_num_a01=1
integer, parameter :: p_num_a02=1
integer, parameter :: p_num_a03=1
integer, parameter :: p_num_a04=1
integer, parameter :: p_num_a05=1
integer, parameter :: p_num_a06=1
integer, parameter :: p_num_a07=1
integer, parameter :: p_num_a08=1
integer, parameter :: p_num_a3=1
integer, parameter :: p_num_a7=1
integer, parameter :: p_o3=1
integer, parameter :: p_oc1=9
integer, parameter :: p_oc2=10
integer, parameter :: p_p10=20
integer, parameter :: p_p25=6
integer, parameter :: p_seas_1=16
integer, parameter :: p_seas_2=17
integer, parameter :: p_seas_3=18
integer, parameter :: p_seas_4=19
integer, parameter :: p_so2=2
integer, parameter :: p_so4_a01=1
integer, parameter :: p_so4_a02=1
integer, parameter :: p_so4_a03=1
integer, parameter :: p_so4_a04=1
integer, parameter :: p_so4_a05=1
integer, parameter :: p_so4_a06=1
integer, parameter :: p_so4_a07=1
integer, parameter :: p_so4_a08=1
integer, parameter :: p_so4_a1=1
integer, parameter :: p_so4aj=1
integer, parameter :: p_sulf=3
integer, parameter :: p_tracer_1=1
integer, parameter :: CB05_SORG_AQ_KPP=1000
integer, parameter :: CB05_SORG_VBS_AQ_KPP=1001
integer, parameter :: CHEM_TRACE2=1002
integer, parameter :: CHEM_TRACER=13
integer, parameter :: CHEM_VASH=1004
integer, parameter :: CHEM_VOLC=1005
integer, parameter :: CHEM_VOLC_4BIN=1006
integer, parameter :: CO2_TRACER=1007
integer, parameter :: DUST=1008
integer, parameter :: GHG_TRACER=1009
integer, parameter :: GOCARTRACM_KPP=1010
integer, parameter :: GOCART_SIMPLE=300
integer, parameter :: RACMPM_KPP=1012
integer, parameter :: RACMSORG_AQCHEM_KPP=1013
integer, parameter :: RACMSORG_KPP=1014
integer, parameter :: RACM_ESRLSORG_AQCHEM_KPP=1015
integer, parameter :: RACM_ESRLSORG_KPP=1016
integer, parameter :: RACM_KPP=1017
integer, parameter :: RACM_MIM_KPP=1018
integer, parameter :: RACM_SOA_VBS_AQCHEM_KPP=1019
integer, parameter :: RACM_SOA_VBS_HET_KPP=1020
integer, parameter :: RACM_SOA_VBS_KPP=1021
integer, parameter :: cbmz_cam_mam3_aq=1022
integer, parameter :: cbmz_cam_mam3_noaq=1023
integer, parameter :: cbmz_cam_mam7_aq=1024
integer, parameter :: cbmz_cam_mam7_noaq=1025
