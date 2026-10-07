"""Source-pinned WRF v4.7.1 SFIRE history metadata and dimension classes."""

# name: (Registry type, stagger, description, units, citation, history, layout)
SFIRE_REGISTRY_FIELDS = {
    'LFN_HIST': ('real', 'Z', 'level function history', '1', 'WRF-v4.7.1/Registry/registry.fire:27', True, 'fine'),
    'LFN_TIME': ('real', '', 'level function history time', 's', 'WRF-v4.7.1/Registry/registry.fire:28', True, 'ignition'),
    'NFUEL_CAT': ('real', 'Z', 'fuel data', '', 'WRF-v4.7.1/Registry/registry.fire:33', True, 'fine'),
    'ZSF': ('real', 'Z', 'height of surface above sea level', 'm', 'WRF-v4.7.1/Registry/registry.fire:34', True, 'fine'),
    'DZDXF': ('real', 'Z', 'surface gradient x', '1', 'WRF-v4.7.1/Registry/registry.fire:35', True, 'fine'),
    'DZDYF': ('real', 'Z', 'surface gradient y', '1', 'WRF-v4.7.1/Registry/registry.fire:36', True, 'fine'),
    'TIGN_G': ('real', 'Z', 'ignition time on ground', 's', 'WRF-v4.7.1/Registry/registry.fire:37', True, 'fine'),
    'RTHFRTEN': ('real', 'Z', 'temperature tendency', 'K/s', 'WRF-v4.7.1/Registry/registry.fire:44', True, 'volume'),
    'RQVFRTEN': ('real', 'Z', 'humidity tendency', '', 'WRF-v4.7.1/Registry/registry.fire:45', True, 'volume'),
    'AVG_FUEL_FRAC': ('real', 'Z', 'fuel remaining averaged to atmospheric grid', '1', 'WRF-v4.7.1/Registry/registry.fire:50', True, 'mass'),
    'GRNHFX': ('real', 'Z', 'heat flux from ground fire', 'W/m^2', 'WRF-v4.7.1/Registry/registry.fire:51', True, 'mass'),
    'GRNQFX': ('real', 'Z', 'moisture flux from ground fire', 'W/m^2', 'WRF-v4.7.1/Registry/registry.fire:52', True, 'mass'),
    'CANHFX': ('real', 'Z', 'heat flux from crown fire', 'W/m^2', 'WRF-v4.7.1/Registry/registry.fire:53', True, 'mass'),
    'CANQFX': ('real', 'Z', 'moisture flux from crown fire', 'W/m^2', 'WRF-v4.7.1/Registry/registry.fire:54', True, 'mass'),
    'UAH': ('real', 'X', 'wind at fire_wind_height', 'm/s', 'WRF-v4.7.1/Registry/registry.fire:55', True, 'mass'),
    'VAH': ('real', 'Y', 'wind at fire_wind_height', 'm/s', 'WRF-v4.7.1/Registry/registry.fire:56', True, 'mass'),
    'GRNHFX_FU': ('real', 'Z', 'heat flux from ground fire (feedback unsensitive)', 'W/m^2', 'WRF-v4.7.1/Registry/registry.fire:57', False, 'mass'),
    'GRNQFX_FU': ('real', 'Z', 'moisture flux from ground fire (feedback unsensitive)', 'W/m^2', 'WRF-v4.7.1/Registry/registry.fire:58', False, 'mass'),
    'LFN': ('real', 'Z', 'level function', '1', 'WRF-v4.7.1/Registry/registry.fire:66', True, 'fine'),
    'LFN_0': ('real', 'Z', 'level function for time integration, step 0', '1', 'WRF-v4.7.1/Registry/registry.fire:67', False, 'fine'),
    'LFN_1': ('real', 'Z', 'level function for time integration, step 1', '1', 'WRF-v4.7.1/Registry/registry.fire:68', False, 'fine'),
    'LFN_2': ('real', 'Z', 'level function for time integration, step 2', '1', 'WRF-v4.7.1/Registry/registry.fire:69', False, 'fine'),
    'LFN_S0': ('real', 'Z', 'level set sign function from LSM integration', '1', 'WRF-v4.7.1/Registry/registry.fire:70', False, 'fine'),
    'LFN_S1': ('real', 'Z', 'level set function for reinitialization integration', '1', 'WRF-v4.7.1/Registry/registry.fire:71', False, 'fine'),
    'LFN_S2': ('real', 'Z', 'level set function for reinitialization integration', '1', 'WRF-v4.7.1/Registry/registry.fire:72', False, 'fine'),
    'LFN_S3': ('real', 'Z', 'level set function for reinitialization integration', '1', 'WRF-v4.7.1/Registry/registry.fire:73', False, 'fine'),
    'FUEL_FRAC': ('real', 'Z', 'fuel remaining', '1', 'WRF-v4.7.1/Registry/registry.fire:75', True, 'fine'),
    'FIRE_AREA': ('real', 'Z', 'fraction of cell area on fire', '1', 'WRF-v4.7.1/Registry/registry.fire:76', True, 'fine'),
    'UF': ('real', 'Z', 'fire wind', 'm/s', 'WRF-v4.7.1/Registry/registry.fire:77', True, 'fine'),
    'VF': ('real', 'Z', 'fire wind', 'm/s', 'WRF-v4.7.1/Registry/registry.fire:78', True, 'fine'),
    'FGRNHFX': ('real', 'Z', 'heat flux from ground fire', 'W/m^2', 'WRF-v4.7.1/Registry/registry.fire:79', True, 'fine'),
    'FGRNQFX': ('real', 'Z', 'moisture flux from ground fire', 'W/m^2', 'WRF-v4.7.1/Registry/registry.fire:80', True, 'fine'),
    'FCANHFX': ('real', 'Z', 'heat flux from crown fire', 'W/m^2', 'WRF-v4.7.1/Registry/registry.fire:81', True, 'fine'),
    'FCANQFX': ('real', 'Z', 'moisture flux from crown fire', 'W/m^2', 'WRF-v4.7.1/Registry/registry.fire:82', True, 'fine'),
    'ROS': ('real', 'Z', 'rate of spread', 'm/s', 'WRF-v4.7.1/Registry/registry.fire:83', False, 'fine'),
    'BURNT_AREA_DT': ('real', 'Z', 'fraction of cell area burnt on current dt', '-', 'WRF-v4.7.1/Registry/registry.fire:84', True, 'fine'),
    'FLAME_LENGTH': ('real', 'Z', 'fire flame length', 'm', 'WRF-v4.7.1/Registry/registry.fire:85', True, 'fine'),
    'ROS_FRONT': ('real', 'Z', 'rate of spread at fire front', 'm/s', 'WRF-v4.7.1/Registry/registry.fire:86', True, 'fine'),
    'FMC_G': ('real', 'Z', 'ground fuel moisture contents', '1', 'WRF-v4.7.1/Registry/registry.fire:91', True, 'fine'),
    'FMC_GC': ('real', 'Z', 'fuel moisture contents by class', '1', 'WRF-v4.7.1/Registry/registry.fire:98', True, 'classes'),
    'FMEP': ('real', 'Z', 'fuel moisture extended model parameters', '1', 'WRF-v4.7.1/Registry/registry.fire:99', True, 'extended'),
    'FMC_EQUI': ('real', 'Z', 'fuel moisture contents by class equilibrium (diagnostics only)', '1', 'WRF-v4.7.1/Registry/registry.fire:100', True, 'classes'),
    'FMC_TEND': ('real', 'Z', 'fuel moisture contents by class time lag (diagnostics only)', 'h', 'WRF-v4.7.1/Registry/registry.fire:101', True, 'classes'),
    'RAIN_OLD': ('real', 'Z', 'previous value of accumulated rain', 'mm', 'WRF-v4.7.1/Registry/registry.fire:102', True, 'mass'),
    'T2_OLD': ('real', 'Z', 'previous value of air temperature at 2m', 'K', 'WRF-v4.7.1/Registry/registry.fire:103', True, 'mass'),
    'Q2_OLD': ('real', 'Z', 'previous value of 2m specific humidity', 'kg/kg', 'WRF-v4.7.1/Registry/registry.fire:104', True, 'mass'),
    'PSFC_OLD': ('real', 'Z', 'previous value of surface pressure', 'Pa', 'WRF-v4.7.1/Registry/registry.fire:105', True, 'mass'),
    'RH_FIRE': ('real', 'Z', 'relative humidity at the surface', '1', 'WRF-v4.7.1/Registry/registry.fire:106', True, 'mass'),
    'FMOIST_LASTTIME': ('real', '', 'last time the moisture model was run', 's', 'WRF-v4.7.1/Registry/registry.fire:107', True, 'scalar'),
    'FMOIST_NEXTTIME': ('real', '', 'next time the moisture model will run', 's', 'WRF-v4.7.1/Registry/registry.fire:108', True, 'scalar'),
    'FXLONG': ('real', 'Z', 'longitude of midpoints of fire cells', 'degrees', 'WRF-v4.7.1/Registry/registry.fire:125', True, 'fine'),
    'FXLAT': ('real', 'Z', 'latitude of midpoints of fire cells', 'degrees', 'WRF-v4.7.1/Registry/registry.fire:126', True, 'fine'),
    'FUEL_TIME': ('real', 'Z', 'fuel', '', 'WRF-v4.7.1/Registry/registry.fire:127', True, 'fine'),
    'BBB': ('real', 'Z', 'fuel', '', 'WRF-v4.7.1/Registry/registry.fire:128', True, 'fine'),
    'BETAFL': ('real', 'Z', 'fuel', '', 'WRF-v4.7.1/Registry/registry.fire:129', True, 'fine'),
    'PHIWC': ('real', 'Z', 'fuel', '', 'WRF-v4.7.1/Registry/registry.fire:130', True, 'fine'),
    'R_0': ('real', 'Z', 'fuel', '', 'WRF-v4.7.1/Registry/registry.fire:131', True, 'fine'),
    'FGIP': ('real', 'Z', 'fuel', '', 'WRF-v4.7.1/Registry/registry.fire:132', True, 'fine'),
    'ISCHAP': ('real', 'Z', 'fuel', '', 'WRF-v4.7.1/Registry/registry.fire:133', True, 'fine'),
    'FZ0': ('real', 'Z', 'roughness length of fire cells', 'm', 'WRF-v4.7.1/Registry/registry.fire:134', True, 'fine'),
    'IBOROS': ('real', 'Z', 'fire intensity over rate of spread', 'kJ/m^2', 'WRF-v4.7.1/Registry/registry.fire:135', True, 'fine'),
    'FS_FIRE_ROSDT': ('real', 'Z', 'fire rate of spread (on fire refined grid) between generation cycles', '', 'WRF-v4.7.1/Registry/registry.fire:384', False, 'fine'),
    "FLINEINT": ("real", "Z", "fireline intensity", "kW/m", "", True, "fine"),
    "LFN_OUT": ("real", "Z", "level function before prescribed ignition", "1", "", True, "fine"),
}

SFIRE_REGISTRY_FIELDS.update({
    "FS_LAST_GEN_DT":("integer","","cycles since last firebrand generation","count","WRF-v4.7.1/Registry/registry.fire:381",False,"scalar"),
    "FS_GEN_IDMAX":("integer","","highest ID number assigned to particle","ID","WRF-v4.7.1/Registry/registry.fire:382",False,"scalar"),
    "FS_COUNT_RESET":("integer","","flag to reset deposit count","1","WRF-v4.7.1/Registry/registry.fire:383",False,"scalar"),
    "FS_FIRE_AREA":("real","","fire area on meteorological grid","1","WRF-v4.7.1/Registry/registry.fire:385",True,"mass"),
    "FS_FUEL_SPOTTING_RISK":("real","","fuel risk for spotting likelihood","1","WRF-v4.7.1/Registry/registry.fire:386",True,"mass"),
    "FS_COUNT_LANDED_ALL":("real","","firebrand count: landed since sim start","count","WRF-v4.7.1/Registry/registry.fire:387",True,"mass"),
    "FS_COUNT_LANDED_HIST":("real","","firebrand count: landed during history interval","count","WRF-v4.7.1/Registry/registry.fire:388",True,"mass"),
    "FS_LANDING_MASK":("integer","","valid landing gridpoints","1","WRF-v4.7.1/Registry/registry.fire:389",True,"mass"),
    "FS_GEN_INST":("integer","","firebrand number generation - instantaneous","count","WRF-v4.7.1/Registry/registry.fire:390",True,"mass"),
    "FS_FRAC_LANDED":("real","","fraction of firebrand landed during history interval","1","WRF-v4.7.1/Registry/registry.fire:391",True,"mass"),
    "FS_SPOTTING_LKHD":("real","","fire spotting likelihood","1","WRF-v4.7.1/Registry/registry.fire:392",True,"mass"),
    "FS_P_ID":("integer","","particle unique ID","ID","WRF-v4.7.1/Registry/registry.fire:397",False,"particles"),
    "FS_P_SRC":("integer","","particle source ID","src","WRF-v4.7.1/Registry/registry.fire:398",False,"particles"),
    "FS_P_DT":("integer","","particle active time steps","count","WRF-v4.7.1/Registry/registry.fire:399",False,"particles"),
    "FS_P_X":("real","","particle x position","index","WRF-v4.7.1/Registry/registry.fire:400",False,"particles"),
    "FS_P_Y":("real","","particle y position","index","WRF-v4.7.1/Registry/registry.fire:401",False,"particles"),
    "FS_P_Z":("real","","particle height above ground","m","WRF-v4.7.1/Registry/registry.fire:402; actual height carried by module_firebrand_spotting.F",False,"particles"),
    "FS_P_MASS":("real","","firebrand mass","g","WRF-v4.7.1/Registry/registry.fire:403",False,"particles"),
    "FS_P_DIAM":("real","","firebrand diameter","mm","WRF-v4.7.1/Registry/registry.fire:404",False,"particles"),
    "FS_P_EFFD":("real","","firebrand effective diameter","mm","WRF-v4.7.1/Registry/registry.fire:405",False,"particles"),
    "FS_P_TEMP":("real","","firebrand temperature","K","WRF-v4.7.1/Registry/registry.fire:406",False,"particles"),
    "FS_P_TVEL":("real","","firebrand terminal velocity","m/s","WRF-v4.7.1/Registry/registry.fire:407",False,"particles"),
})

SFIRE_SPOTTING_COARSE_FIELDS=("FS_FIRE_AREA","FS_FUEL_SPOTTING_RISK","FS_COUNT_LANDED_ALL",
    "FS_COUNT_LANDED_HIST","FS_LANDING_MASK","FS_GEN_INST","FS_FRAC_LANDED","FS_SPOTTING_LKHD")
SFIRE_PARTICLE_FIELDS=frozenset(name for name,row in SFIRE_REGISTRY_FIELDS.items() if row[-1]=="particles")

SFIRE_FINE_FIELDS = frozenset(name for name, row in SFIRE_REGISTRY_FIELDS.items() if row[-1] == "fine")
SFIRE_CLASS_FIELDS = frozenset(name for name, row in SFIRE_REGISTRY_FIELDS.items() if row[-1] == "classes")
SFIRE_VOLUME_FIELDS = frozenset(name for name, row in SFIRE_REGISTRY_FIELDS.items() if row[-1] == "volume")

# The public fire inventory has one spelling table for the producer and
# the disk planner. Stage fields are retained in checkpoints; history
# publishes every native diagnostic and fuel parameter available in state.
SFIRE_GRID_FIELD_MAP = {
    "FIRE_AREA": "fire_area", "ROS": "ros", "ROS_FRONT": "ros_front",
    "FLINEINT": "flineint", "FLAME_LENGTH": "flame_length",
    "FGRNHFX": "fgrnhfx", "FGRNQFX": "fgrnqfx",
    "FCANHFX": "fcanhfx", "FCANQFX": "fcanqfx",
    "LFN": "lfn", "LFN_OUT": "lfn_out", "TIGN_G": "tign",
    "FUEL_FRAC": "fuel_frac", "NFUEL_CAT": "nfuel_cat",
    "ZSF": "zsf", "DZDXF": "dzdxf", "DZDYF": "dzdyf", "FMC_G": "fmc_g",
    "BURNT_AREA_DT": "burnt_area_dt", "FUEL_TIME": "fuel_time",
    "UF": "vx", "VF": "vy", "FGIP": "fgip", "BBB": "bbb",
    "BETAFL": "betafl", "PHIWC": "phiwc", "R_0": "r_0",
    "ISCHAP": "ischap", "IBOROS": "iboros",
}
SFIRE_COARSE_FIELD_NAMES = (
    "GRNHFX", "GRNQFX", "GRNHFX_FU", "GRNQFX_FU", "CANHFX", "CANQFX",
    "AVG_FUEL_FRAC", "UAH", "VAH", "RTHFRTEN", "RQVFRTEN",
)
SFIRE_MOISTURE_FIELD_MAP = {
    "RAIN_OLD": "rain_old", "T2_OLD": "t2_old", "Q2_OLD": "q2_old",
    "PSFC_OLD": "psfc_old", "FMC_GC": "fmc_gc", "FMEP": "fmep",
    "FMC_EQUI": "fmc_equi", "FMC_TEND": "fmc_lag", "RH_FIRE": "rh_fire",
}


def sfire_history_shapes(cfg):
    """Raw producer shapes without allocating arrays or importing CUDA."""
    if int(getattr(cfg, "ifire", 0)) != 2:
        return {}
    ny, nx, nz = int(cfg.ny), int(cfg.nx), int(cfg.nz)
    fine = (ny * int(cfg.sr_y), nx * int(cfg.sr_x))
    result = {name: fine for name in SFIRE_GRID_FIELD_MAP}
    result.update({name: (ny, nx) for name in SFIRE_COARSE_FIELD_NAMES})
    result.update(FZ0=fine, FXLAT=fine, FXLONG=fine,
                  UAH=(ny, nx+1), VAH=(ny+1, nx),
                  RTHFRTEN=(nz, ny, nx), RQVFRTEN=(nz, ny, nx))
    # These registry history planes exist even when no brands are generated.
    result.update({name: (ny, nx) for name in SFIRE_SPOTTING_COARSE_FIELDS})
    result.update(LFN_HIST=fine, LFN_TIME=(1,))
    if int(getattr(cfg, "ifire", 0)) == 2:
        result.update(FMOIST_LASTTIME=(), FMOIST_NEXTTIME=())
        for name in SFIRE_MOISTURE_FIELD_MAP:
            result[name] = ((int(cfg.nfmc), ny, nx) if name in SFIRE_CLASS_FIELDS else
                            (2, ny, nx) if name == "FMEP" else (ny, nx))
    if cfg.fs_firebrand_gen_lim>0:
        result.update({name:(ny,nx) for name in SFIRE_SPOTTING_COARSE_FIELDS})
        result.update(FS_FIRE_ROSDT=fine,FS_LAST_GEN_DT=(),FS_GEN_IDMAX=(),FS_COUNT_RESET=())
        if cfg.trackember:
            result.update({name:(int(cfg.fs_array_maxsize),) for name in SFIRE_PARTICLE_FIELDS})
    return result
