"""WRF registry state contract for FAST_SBM (mp_physics=30).

Source: WRF 4.6.1 Registry/registry.sbm:318. Bin state is prognostic,
not reconstructible from bulk mixing ratios at a restart boundary.
"""
BIN_GROUPS = (("ff1",33),("ff5",33),("ff6",33),("ff8",43))
BIN_SPECIES = tuple(f"{group}i{i:02d}" for group,count in BIN_GROUPS for i in range(1,count+1))
MASS_SPECIES = ("qv","qc","qr","qi","qs","qg")
NUMBER_SPECIES = ("nn","nc","nr","ni","ns","ng")
EXTRA_SPECIES = MASS_SPECIES[3:] + BIN_SPECIES + NUMBER_SPECIES
FLAG_FIELDS = ("hail", "rainnc_own_column", "rainnc_bottom_outflow",
               "nucleation_water_sink", "dtcond_carry", "breakup_retry_cap",
               "cloud_base_first_level", "fall_speed_own_bin",
               "snow_clip_qi_plus_qs", "melt_exponential", "fusion_heat_334",
               "homfrz_any_saturation", "dsupice_w_own_point",
               "dsupice_wind_own_point", "supsat_rederived", "sedimentation_cell_depth", "collision_mass_conserved")
TABLE_ORDER = ("xl", "xi", "xs", "xg", "xh", "vr1", "vr2", "vr3", "vr4", "vr5",
               "rlec", "riec", "rsec", "rgec", "rhec", "ro1bl", "ro2bl", "ro3bl",
               "ro4bl", "ro5bl", "ywll", "ywls", "ywlg", "ywlh", "ywss", "pkij", "qkj")
STATE_FIELDS = ("w","u","v","th_old","chem","dz8w","rho","p","pii","th",
                "qv","qv_old","qc","qr","qi","qs","qg","qnc","qnr","qni",
                "qns","qng","qna","xland","rainnc","rainncv","snownc",
                "snowncv","graupelnc","graupelncv")
EXTRA_TABLES = ("xl_mg", "xs_mg", "xg_mg", "xh_mg", "ima", "chucm",
                "ecoalmassm", "prob", "gain_var_new", "dropradii",
                "fccnr_con", "fccnr_mar", "rccn")
CUDA_OPTIONS = ("--std=c++17", "--fmad=false", "--ftz=false",
                "--prec-div=true", "--prec-sqrt=true")


def corrected_flags(hail=1):
    f={name:1 for name in FLAG_FIELDS}
    f.update(hail=hail,dtcond_carry=0,breakup_retry_cap=64)
    return f
