use crate::compute::ComputeOpts;
use crate::error::WrfResult;
use crate::file::WrfFile;

/// Output dimensionality.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VarDim {
    /// `[ny, nx]`
    TwoD,
    /// `[nz, ny, nx]`
    ThreeD,
}

/// Definition of a computed WRF variable.
pub struct VarDef {
    pub name: &'static str,
    pub aliases: &'static [&'static str],
    pub description: &'static str,
    pub default_units: &'static str,
    pub dim: VarDim,
    pub compute: fn(&WrfFile, usize, &ComputeOpts) -> WrfResult<Vec<f64>>,
}

// ── Compute function imports ──
use crate::diag::cape as dcape;
use crate::diag::cloud as dcloud;
use crate::diag::ecape as decape;
use crate::diag::extra as dextra;
use crate::diag::helicity as dhel;
use crate::diag::moisture as dmoist;
use crate::diag::pressure as dpres;
use crate::diag::radar as dradar;
use crate::diag::severe as dsevere;
use crate::diag::srh as dsrh;
use crate::diag::thermo as dthermo;
use crate::diag::vorticity as dvort;
use crate::diag::wind as dwind;

/// Master registry of all supported variables.
pub static VARS: &[VarDef] = &[
    // ── Phase 1: Foundation ──
    VarDef {
        name: "pressure",
        aliases: &[],
        description: "Full model pressure",
        default_units: "hPa",
        dim: VarDim::ThreeD,
        compute: dpres::compute_pressure_hpa,
    },
    VarDef {
        name: "pres",
        aliases: &["p", "pressure_pa"],
        description: "Full model pressure",
        default_units: "Pa",
        dim: VarDim::ThreeD,
        compute: dpres::compute_pressure,
    },
    VarDef {
        name: "height",
        aliases: &["z", "height_msl"],
        description: "Model height MSL",
        default_units: "m",
        dim: VarDim::ThreeD,
        compute: dpres::compute_height,
    },
    VarDef {
        name: "height_agl",
        aliases: &["z_agl"],
        description: "Model height AGL",
        default_units: "m",
        dim: VarDim::ThreeD,
        compute: dpres::compute_height_agl,
    },
    // NOTE: zstag and geopt_stag are omitted from the registry because they
    // return nz_stag levels but VarDim::ThreeD implies [nz, ny, nx], causing
    // reshape failures.  They can still be accessed via the raw variable path.
    VarDef {
        name: "geopt",
        aliases: &["geopotential"],
        description: "Full geopotential",
        default_units: "m2/s2",
        dim: VarDim::ThreeD,
        compute: dpres::compute_geopt,
    },
    VarDef {
        name: "terrain",
        aliases: &["ter", "hgt"],
        description: "Terrain height",
        default_units: "m",
        dim: VarDim::TwoD,
        compute: dpres::compute_terrain,
    },
    VarDef {
        name: "t2",
        aliases: &["t2m", "temperature_2m"],
        description: "2-m temperature",
        default_units: "K",
        dim: VarDim::TwoD,
        compute: dthermo::compute_t2,
    },
    VarDef {
        name: "tv2m",
        aliases: &["virtual_temperature_2m"],
        description: "2-m virtual temperature",
        default_units: "K",
        dim: VarDim::TwoD,
        compute: dthermo::compute_tv2m,
    },
    VarDef {
        name: "temp",
        aliases: &["tk"],
        description: "Temperature",
        default_units: "K",
        dim: VarDim::ThreeD,
        compute: dthermo::compute_temp,
    },
    VarDef {
        name: "tc",
        aliases: &["temp_c"],
        description: "Temperature in Celsius",
        default_units: "degC",
        dim: VarDim::ThreeD,
        compute: dthermo::compute_tc,
    },
    VarDef {
        name: "theta",
        aliases: &["th", "potential_temperature"],
        description: "Potential temperature",
        default_units: "K",
        dim: VarDim::ThreeD,
        compute: dthermo::compute_theta,
    },
    VarDef {
        name: "ua",
        aliases: &["u_destag"],
        description: "U-wind (destaggered)",
        default_units: "m/s",
        dim: VarDim::ThreeD,
        compute: dwind::compute_ua,
    },
    VarDef {
        name: "va",
        aliases: &["v_destag"],
        description: "V-wind (destaggered)",
        default_units: "m/s",
        dim: VarDim::ThreeD,
        compute: dwind::compute_va,
    },
    VarDef {
        name: "wa",
        aliases: &["w_destag"],
        description: "W-wind (destaggered)",
        default_units: "m/s",
        dim: VarDim::ThreeD,
        compute: dwind::compute_wa,
    },
    VarDef {
        name: "lat",
        aliases: &["xlat", "latitude"],
        description: "Latitude",
        default_units: "degrees",
        dim: VarDim::TwoD,
        compute: dwind::compute_lat,
    },
    VarDef {
        name: "lon",
        aliases: &["xlong", "longitude"],
        description: "Longitude",
        default_units: "degrees",
        dim: VarDim::TwoD,
        compute: dwind::compute_lon,
    },
    // ── Phase 2: Thermodynamics ──
    VarDef {
        name: "theta_e",
        aliases: &["eth"],
        description: "Equivalent potential temperature",
        default_units: "K",
        dim: VarDim::ThreeD,
        compute: dthermo::compute_theta_e,
    },
    VarDef {
        name: "tv",
        aliases: &["virtual_temperature"],
        description: "Virtual temperature",
        default_units: "K",
        dim: VarDim::ThreeD,
        compute: dthermo::compute_tv,
    },
    VarDef {
        name: "twb",
        aliases: &["wet_bulb"],
        description: "Wet-bulb temperature",
        default_units: "K",
        dim: VarDim::ThreeD,
        compute: dthermo::compute_twb,
    },
    VarDef {
        name: "td",
        aliases: &["dp", "dewpoint"],
        description: "Dewpoint temperature",
        default_units: "degC",
        dim: VarDim::ThreeD,
        compute: dthermo::compute_td,
    },
    VarDef {
        name: "rh",
        aliases: &["relative_humidity"],
        description: "Relative humidity",
        default_units: "%",
        dim: VarDim::ThreeD,
        compute: dthermo::compute_rh,
    },
    VarDef {
        name: "slp",
        aliases: &["sea_level_pressure", "mslp"],
        description: "Sea-level pressure",
        default_units: "hPa",
        dim: VarDim::TwoD,
        compute: dpres::compute_slp,
    },
    VarDef {
        name: "omega",
        aliases: &["vertical_velocity_pressure"],
        description: "Vertical velocity in pressure coordinates",
        default_units: "Pa/s",
        dim: VarDim::ThreeD,
        compute: dpres::compute_omega,
    },
    // ── Phase 2: Moisture ──
    VarDef {
        name: "pw",
        aliases: &["precipitable_water"],
        description: "Precipitable water",
        default_units: "mm",
        dim: VarDim::TwoD,
        compute: dmoist::compute_pw,
    },
    VarDef {
        name: "rh2m",
        aliases: &["rh2", "relative_humidity_2m"],
        description: "2-m relative humidity",
        default_units: "%",
        dim: VarDim::TwoD,
        compute: dmoist::compute_rh2m,
    },
    VarDef {
        name: "dp2m",
        aliases: &["td2", "td2m", "dewpoint_2m"],
        description: "2-m dewpoint temperature",
        default_units: "degC",
        dim: VarDim::TwoD,
        compute: dmoist::compute_dp2m,
    },
    VarDef {
        name: "mixing_ratio",
        aliases: &["qvapor", "water_vapor_mixing_ratio"],
        description: "Water vapor mixing ratio",
        default_units: "kg/kg",
        dim: VarDim::ThreeD,
        compute: dmoist::compute_mixing_ratio,
    },
    VarDef {
        name: "specific_humidity",
        aliases: &["q"],
        description: "Specific humidity",
        default_units: "kg/kg",
        dim: VarDim::ThreeD,
        compute: dmoist::compute_specific_humidity,
    },
    // ── Phase 3: CAPE ──
    VarDef {
        name: "sbcape",
        aliases: &["surface_based_cape"],
        description: "Surface-based CAPE",
        default_units: "J/kg",
        dim: VarDim::TwoD,
        compute: dcape::compute_sbcape,
    },
    VarDef {
        name: "sbcin",
        aliases: &["surface_based_cin"],
        description: "Surface-based CIN",
        default_units: "J/kg",
        dim: VarDim::TwoD,
        compute: dcape::compute_sbcin,
    },
    VarDef {
        name: "mlcape",
        aliases: &["mixed_layer_cape"],
        description: "Mixed-layer CAPE",
        default_units: "J/kg",
        dim: VarDim::TwoD,
        compute: dcape::compute_mlcape,
    },
    VarDef {
        name: "mlcin",
        aliases: &["mixed_layer_cin"],
        description: "Mixed-layer CIN",
        default_units: "J/kg",
        dim: VarDim::TwoD,
        compute: dcape::compute_mlcin,
    },
    VarDef {
        name: "mucape",
        aliases: &["most_unstable_cape"],
        description: "Most-unstable CAPE",
        default_units: "J/kg",
        dim: VarDim::TwoD,
        compute: dcape::compute_mucape,
    },
    VarDef {
        name: "mucin",
        aliases: &["most_unstable_cin"],
        description: "Most-unstable CIN",
        default_units: "J/kg",
        dim: VarDim::TwoD,
        compute: dcape::compute_mucin,
    },
    VarDef {
        name: "lcl",
        aliases: &["lcl_height"],
        description: "Lifted condensation level height AGL",
        default_units: "m",
        dim: VarDim::TwoD,
        compute: dcape::compute_lcl,
    },
    VarDef {
        name: "lfc",
        aliases: &["lfc_height"],
        description: "Level of free convection height AGL",
        default_units: "m",
        dim: VarDim::TwoD,
        compute: dcape::compute_lfc,
    },
    VarDef {
        name: "el",
        aliases: &["el_height", "equilibrium_level"],
        description: "Equilibrium level height AGL",
        default_units: "m",
        dim: VarDim::TwoD,
        compute: dcape::compute_el,
    },
    VarDef {
        name: "cape2d",
        aliases: &["cape_2d"],
        description: "CAPE/CIN/LCL/LFC (backward-compat tuple)",
        default_units: "J/kg",
        dim: VarDim::TwoD,
        compute: dcape::compute_cape2d,
    },
    VarDef {
        name: "cape3d",
        aliases: &["cape_3d"],
        description: "3-D CAPE field",
        default_units: "J/kg",
        dim: VarDim::ThreeD,
        compute: dcape::compute_cape3d,
    },
    // ── Phase 4: Wind & SRH ──
    VarDef {
        name: "wspd",
        aliases: &["wind_speed"],
        description: "Wind speed",
        default_units: "m/s",
        dim: VarDim::ThreeD,
        compute: dwind::compute_wspd,
    },
    VarDef {
        name: "wdir",
        aliases: &["wind_direction"],
        description: "Wind direction",
        default_units: "degrees",
        dim: VarDim::ThreeD,
        compute: dwind::compute_wdir,
    },
    VarDef {
        name: "uvmet",
        aliases: &["earth_rotated_wind"],
        description: "U/V wind rotated to earth coordinates",
        default_units: "m/s",
        dim: VarDim::ThreeD,
        compute: dwind::compute_uvmet,
    },
    VarDef {
        name: "uvmet10",
        aliases: &["earth_rotated_wind_10m"],
        description: "10-m U/V wind rotated to earth coordinates",
        default_units: "m/s",
        dim: VarDim::TwoD,
        compute: dwind::compute_uvmet10,
    },
    VarDef {
        name: "wspd10",
        aliases: &["wind_speed_10m"],
        description: "10-m wind speed",
        default_units: "m/s",
        dim: VarDim::TwoD,
        compute: dwind::compute_wspd10,
    },
    VarDef {
        name: "wdir10",
        aliases: &["wind_direction_10m"],
        description: "10-m wind direction",
        default_units: "degrees",
        dim: VarDim::TwoD,
        compute: dwind::compute_wdir10,
    },
    VarDef {
        name: "srh1",
        aliases: &["srh_0_1km"],
        description: "0-1 km storm-relative helicity (Bunkers)",
        default_units: "m2/s2",
        dim: VarDim::TwoD,
        compute: dsrh::compute_srh1,
    },
    VarDef {
        name: "srh3",
        aliases: &["srh_0_3km"],
        description: "0-3 km storm-relative helicity (Bunkers)",
        default_units: "m2/s2",
        dim: VarDim::TwoD,
        compute: dsrh::compute_srh3,
    },
    VarDef {
        name: "srh",
        aliases: &["storm_relative_helicity"],
        description: "Storm-relative helicity (configurable depth)",
        default_units: "m2/s2",
        dim: VarDim::TwoD,
        compute: dsrh::compute_srh,
    },
    VarDef {
        name: "shear_0_1km",
        aliases: &["bulk_shear_1km", "shr01"],
        description: "0-1 km bulk wind shear",
        default_units: "m/s",
        dim: VarDim::TwoD,
        compute: dsrh::compute_shear_0_1km,
    },
    VarDef {
        name: "shear_0_6km",
        aliases: &["bulk_shear_6km", "shr06"],
        description: "0-6 km bulk wind shear",
        default_units: "m/s",
        dim: VarDim::TwoD,
        compute: dsrh::compute_shear_0_6km,
    },
    VarDef {
        name: "bunkers_rm",
        aliases: &["bunkers_right"],
        description: "Bunkers right-moving supercell motion",
        default_units: "m/s",
        dim: VarDim::TwoD,
        compute: dsrh::compute_bunkers_rm,
    },
    VarDef {
        name: "bunkers_lm",
        aliases: &["bunkers_left"],
        description: "Bunkers left-moving supercell motion",
        default_units: "m/s",
        dim: VarDim::TwoD,
        compute: dsrh::compute_bunkers_lm,
    },
    VarDef {
        name: "mean_wind_0_6km",
        aliases: &["mean_wind_6km"],
        description: "0-6 km mean wind",
        default_units: "m/s",
        dim: VarDim::TwoD,
        compute: dsrh::compute_mean_wind_0_6km,
    },
    // ── Phase 6: Remaining diagnostics ──
    VarDef {
        name: "avo",
        aliases: &["absolute_vorticity"],
        description: "Absolute vorticity",
        default_units: "s-1",
        dim: VarDim::ThreeD,
        compute: dvort::compute_avo,
    },
    VarDef {
        name: "pvo",
        aliases: &["potential_vorticity"],
        description: "Potential vorticity",
        default_units: "PVU",
        dim: VarDim::ThreeD,
        compute: dvort::compute_pvo,
    },
    VarDef {
        name: "dbz",
        aliases: &["reflectivity"],
        description: "Simulated reflectivity",
        default_units: "dBZ",
        dim: VarDim::ThreeD,
        compute: dradar::compute_dbz,
    },
    VarDef {
        name: "maxdbz",
        aliases: &["max_reflectivity", "composite_reflectivity", "mdbz"],
        description: "Maximum (composite) reflectivity",
        default_units: "dBZ",
        dim: VarDim::TwoD,
        compute: dradar::compute_maxdbz,
    },
    VarDef {
        name: "ctt",
        aliases: &["cloud_top_temperature"],
        description: "Cloud-top temperature",
        default_units: "degC",
        dim: VarDim::TwoD,
        compute: dcloud::compute_ctt,
    },
    VarDef {
        name: "cloudfrac",
        aliases: &["cloud_fraction"],
        description: "Cloud fraction (low/mid/high)",
        default_units: "%",
        dim: VarDim::TwoD,
        compute: dcloud::compute_cloudfrac,
    },
    VarDef {
        name: "uhel",
        aliases: &["updraft_helicity", "helicity"],
        description: "Updraft helicity",
        default_units: "m2/s2",
        dim: VarDim::TwoD,
        compute: dhel::compute_uhel,
    },
    // ── Phase 7: Severe & Extras ──
    VarDef {
        name: "stp",
        aliases: &["significant_tornado_parameter"],
        description: "Significant Tornado Parameter (fixed or effective via layer_type)",
        default_units: "dimensionless",
        dim: VarDim::TwoD,
        compute: dsevere::compute_stp_generic,
    },
    VarDef {
        name: "stp_fixed",
        aliases: &[],
        description: "STP fixed-layer (Thompson et al. 2003)",
        default_units: "dimensionless",
        dim: VarDim::TwoD,
        compute: dsevere::compute_stp,
    },
    VarDef {
        name: "stp_effective",
        aliases: &["stp_eff"],
        description: "STP effective-layer",
        default_units: "dimensionless",
        dim: VarDim::TwoD,
        compute: dsevere::compute_stp_effective,
    },
    VarDef {
        name: "scp",
        aliases: &["supercell_composite_parameter"],
        description: "Supercell Composite Parameter (effective SRH + EBWD)",
        default_units: "dimensionless",
        dim: VarDim::TwoD,
        compute: dsevere::compute_scp,
    },
    VarDef {
        name: "ehi",
        aliases: &["energy_helicity_index"],
        description: "Energy-Helicity Index",
        default_units: "dimensionless",
        dim: VarDim::TwoD,
        compute: dsevere::compute_ehi,
    },
    VarDef {
        name: "ecape_scp",
        aliases: &["entraining_scp", "experimental_ecape_scp"],
        description: "Experimental SCP analog using ECAPE for the buoyancy term",
        default_units: "dimensionless",
        dim: VarDim::TwoD,
        compute: dsevere::compute_ecape_scp,
    },
    VarDef {
        name: "ecape_ehi",
        aliases: &["entraining_ehi", "experimental_ecape_ehi"],
        description: "Experimental EHI analog using ECAPE for the buoyancy term",
        default_units: "dimensionless",
        dim: VarDim::TwoD,
        compute: dsevere::compute_ecape_ehi,
    },
    VarDef {
        name: "tehi",
        aliases: &["tornadic_0_1km_ehi"],
        description: "SPC beta Tornadic 0-1 km EHI",
        default_units: "dimensionless",
        dim: VarDim::TwoD,
        compute: dsevere::compute_tehi,
    },
    VarDef {
        name: "tts",
        aliases: &[
            "tornadic_tilting_stretching",
            "tornadic_tilting_and_stretching",
        ],
        description: "SPC beta Tornadic Tilting and Stretching",
        default_units: "dimensionless",
        dim: VarDim::TwoD,
        compute: dsevere::compute_tts,
    },
    VarDef {
        name: "vtp_mod",
        aliases: &["modified_violent_tornado_parameter"],
        description: "Modified Violent Tornado Parameter",
        default_units: "dimensionless",
        dim: VarDim::TwoD,
        compute: dsevere::compute_vtp_mod,
    },
    VarDef {
        name: "critical_angle",
        aliases: &["crit_angle"],
        description: "Critical angle",
        default_units: "degrees",
        dim: VarDim::TwoD,
        compute: dsevere::compute_critical_angle,
    },
    VarDef {
        name: "ship",
        aliases: &["significant_hail_parameter"],
        description: "Significant Hail Parameter",
        default_units: "dimensionless",
        dim: VarDim::TwoD,
        compute: dsevere::compute_ship,
    },
    VarDef {
        name: "bri",
        aliases: &["bulk_richardson_number"],
        description: "Bulk Richardson Number",
        default_units: "dimensionless",
        dim: VarDim::TwoD,
        compute: dsevere::compute_bri,
    },
    VarDef {
        name: "lapse_rate_700_500",
        aliases: &["lr75"],
        description: "700-500 hPa lapse rate",
        default_units: "degC/km",
        dim: VarDim::TwoD,
        compute: dextra::compute_lapse_rate_700_500,
    },
    VarDef {
        name: "lapse_rate_0_3km",
        aliases: &["lr03"],
        description: "0-3 km AGL lapse rate",
        default_units: "degC/km",
        dim: VarDim::TwoD,
        compute: dextra::compute_lapse_rate_0_3km,
    },
    VarDef {
        name: "freezing_level",
        aliases: &["fzlev"],
        description: "Freezing level height AGL",
        default_units: "m",
        dim: VarDim::TwoD,
        compute: dextra::compute_freezing_level,
    },
    VarDef {
        name: "wet_bulb_0",
        aliases: &["wb0", "wet_bulb_zero"],
        description: "Wet-bulb zero height AGL",
        default_units: "m",
        dim: VarDim::TwoD,
        compute: dextra::compute_wet_bulb_0,
    },
    VarDef {
        name: "theta_w",
        aliases: &["wet_bulb_potential_temperature"],
        description: "Wet-bulb potential temperature",
        default_units: "K",
        dim: VarDim::ThreeD,
        compute: dextra::compute_theta_w,
    },
    VarDef {
        name: "fosberg",
        aliases: &["fwi", "fosberg_fire_weather_index"],
        description: "Fosberg Fire Weather Index",
        default_units: "dimensionless",
        dim: VarDim::TwoD,
        compute: dextra::compute_fosberg,
    },
    VarDef {
        name: "haines",
        aliases: &["haines_index"],
        description: "Haines Index",
        default_units: "dimensionless",
        dim: VarDim::TwoD,
        compute: dextra::compute_haines,
    },
    VarDef {
        name: "hdw",
        aliases: &["hot_dry_windy"],
        description: "Hot-Dry-Windy Index",
        default_units: "dimensionless",
        dim: VarDim::TwoD,
        compute: dextra::compute_hdw,
    },
    // ── Configurable / generic variables ──
    VarDef {
        name: "cape",
        aliases: &[],
        description: "CAPE (configurable parcel_type or custom parcel, supports top_m)",
        default_units: "J/kg",
        dim: VarDim::TwoD,
        compute: dcape::compute_cape_generic,
    },
    VarDef {
        name: "cin",
        aliases: &[],
        description: "CIN (configurable parcel_type or custom parcel, supports top_m)",
        default_units: "J/kg",
        dim: VarDim::TwoD,
        compute: dcape::compute_cin_generic,
    },
    VarDef {
        name: "effective_inflow",
        aliases: &["eff_inflow", "effective_inflow_layer"],
        description: "Effective inflow layer base/top heights AGL",
        default_units: "m",
        dim: VarDim::TwoD,
        compute: dcape::compute_effective_inflow_layer,
    },
    VarDef {
        name: "effective_cape",
        aliases: &["eff_cape"],
        description: "MUCAPE within effective inflow layer",
        default_units: "J/kg",
        dim: VarDim::TwoD,
        compute: dcape::compute_effective_inflow_cape,
    },
    VarDef {
        name: "ecape",
        aliases: &["entraining_cape"],
        description: "Entraining CAPE from ecape-rs",
        default_units: "J/kg",
        dim: VarDim::TwoD,
        compute: decape::compute_ecape,
    },
    VarDef {
        name: "ncape",
        aliases: &["normalized_cape"],
        description: "Normalized CAPE from ecape-rs",
        default_units: "J/kg",
        dim: VarDim::TwoD,
        compute: decape::compute_ncape,
    },
    VarDef {
        name: "ecape_cape",
        aliases: &["entraining_parcel_cape"],
        description: "Entraining parcel CAPE from ecape-rs",
        default_units: "J/kg",
        dim: VarDim::TwoD,
        compute: decape::compute_ecape_cape,
    },
    VarDef {
        name: "ecape_cin",
        aliases: &["ecin", "entraining_parcel_cin"],
        description: "Entraining parcel CIN from ecape-rs",
        default_units: "J/kg",
        dim: VarDim::TwoD,
        compute: decape::compute_ecape_cin,
    },
    VarDef {
        name: "ecape_lfc",
        aliases: &["entraining_parcel_lfc"],
        description: "Entraining parcel LFC from ecape-rs",
        default_units: "m",
        dim: VarDim::TwoD,
        compute: decape::compute_ecape_lfc,
    },
    VarDef {
        name: "ecape_el",
        aliases: &["entraining_parcel_el"],
        description: "Entraining parcel EL from ecape-rs",
        default_units: "m",
        dim: VarDim::TwoD,
        compute: decape::compute_ecape_el,
    },
    VarDef {
        name: "effective_srh",
        aliases: &["srh_eff", "eff_srh"],
        description: "SRH over effective inflow layer (Bunkers)",
        default_units: "m2/s2",
        dim: VarDim::TwoD,
        compute: dsrh::compute_effective_srh,
    },
    VarDef {
        name: "ebwd",
        aliases: &["effective_bulk_wind_difference", "effective_bulk_shear"],
        description: "Effective bulk wind difference",
        default_units: "m/s",
        dim: VarDim::TwoD,
        compute: dsevere::compute_effective_bulk_wind_difference,
    },
    VarDef {
        name: "bulk_shear",
        aliases: &["shear"],
        description: "Bulk wind shear (configurable bottom_m / top_m)",
        default_units: "m/s",
        dim: VarDim::TwoD,
        compute: dsrh::compute_bulk_shear,
    },
    VarDef {
        name: "mean_wind",
        aliases: &[],
        description: "Mean wind (configurable bottom_m / top_m)",
        default_units: "m/s",
        dim: VarDim::TwoD,
        compute: dsrh::compute_mean_wind,
    },
    VarDef {
        name: "lapse_rate",
        aliases: &["lr"],
        description: "Lapse rate (configurable bottom_m / top_m, use_virtual)",
        default_units: "degC/km",
        dim: VarDim::TwoD,
        compute: dextra::compute_lapse_rate,
    },
];

/// Look up a variable definition by name or alias (case-insensitive).
pub fn get_var_def(name: &str) -> Option<&'static VarDef> {
    let lower = name.to_lowercase();
    VARS.iter()
        .find(|v| v.name == lower || v.aliases.iter().any(|a| *a == lower))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pressure_defaults_match_wrf_python_names() {
        assert_eq!(get_var_def("pressure").unwrap().default_units, "hPa");
        assert_eq!(get_var_def("pres").unwrap().default_units, "Pa");
        assert_eq!(get_var_def("p").unwrap().default_units, "Pa");
    }

    #[test]
    fn common_wrf_python_aliases_resolve() {
        assert_eq!(get_var_def("cape_2d").unwrap().name, "cape2d");
        assert_eq!(get_var_def("cape_3d").unwrap().name, "cape3d");
        assert_eq!(get_var_def("mdbz").unwrap().name, "maxdbz");
        assert_eq!(get_var_def("helicity").unwrap().name, "uhel");
        assert_eq!(get_var_def("entraining_cape").unwrap().name, "ecape");
        assert_eq!(get_var_def("ecin").unwrap().name, "ecape_cin");
    }
}
