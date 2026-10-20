//! The parameter table (`data/params.v1.json`, schema
//! `arwen.hydro-world.params.v1`). Every input source and every recipe is a
//! row: a different soil product, land-cover product or DEM is a new row, not
//! new code. Recipe numbers carry their published source in the row.
use crate::Result;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ParamTable {
    pub schema: String,
    pub dem_sources: BTreeMap<String, DemSource>,
    pub soil_sources: BTreeMap<String, SoilSource>,
    pub land_cover_sources: BTreeMap<String, LandCoverSource>,
    pub soil_recipes: BTreeMap<String, SoilRecipe>,
    pub lgar_recipes: BTreeMap<String, LgarRecipe>,
    pub constants: BTreeMap<String, Constant>,
    pub slop: SlopRecipe,
    pub giuh: GiuhRecipe,
    pub bounds: BTreeMap<String, [f64; 2]>,
    /// CFE subsurface recipes (K_lf, K_nash, alpha_fc); `cfe_recipe_default` names the
    /// row a run uses unless its config names another
    #[serde(default)]
    pub cfe_recipes: BTreeMap<String, CfeRecipe>,
    #[serde(default)]
    pub cfe_recipe_default: Option<String>,
}

/// A CFE subsurface recipe row. `method`:
/// - `constants`: K_lf, K_nash and alpha_fc from `constants` (the NextGen example-packet priors);
/// - `hillslope_kinematic`: regionalised from soil hydraulics and the DEM hillslope geometry,
///   K_lf = 3600 * 2 F Ks sin(b) D / L (m/h) and K_nash = N 3600 F Ks sin(b) / (theta_d L) (1/h),
///   alpha_fc by soil class. Relations and their sources are cited in the row and at the code.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CfeRecipe {
    pub method: String,
    pub source: String,
    #[serde(default)]
    pub lateral_ksat_factor: f64,
    #[serde(default)]
    pub lateral_ksat_factor_source: String,
    #[serde(default)]
    pub n_nash: f64,
    /// pressure head of one atmosphere (m of water): alpha_fc is a fraction of it (CFE convention)
    #[serde(default)]
    pub atmosphere_head_m: f64,
    /// STAS class id (as a string) -> alpha_fc; classes not listed take `alpha_fc_default`
    #[serde(default)]
    pub alpha_fc_by_soil_class: BTreeMap<String, f64>,
    #[serde(default)]
    pub alpha_fc_default: f64,
    #[serde(default)]
    pub alpha_fc_source: String,
    #[serde(default)]
    pub min_hillslope_length_m: f64,
    #[serde(default)]
    pub min_sin_slope: f64,
    #[serde(default)]
    pub min_drainable_porosity: f64,
    #[serde(default)]
    pub k_lf_range: [f64; 2],
    #[serde(default)]
    pub k_nash_range: [f64; 2],
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Licence {
    pub name: String,
    pub attribution: String,
    pub verdict_ref: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DemSource {
    pub dataset: String,
    pub url_pattern: String,
    /// file name with {ns} {lat2} {ew} {lon3} placeholders (1 degree tiles,
    /// named by the south-west corner)
    pub tile_pattern: String,
    pub vertical_datum: String,
    pub licence: Licence,
}

fn fill_tile(p: &str, lat: i32, lon: i32) -> String {
    p.replace("{ns}", if lat >= 0 { "N" } else { "S" })
        .replace("{lat2}", &format!("{:02}", lat.abs()))
        .replace("{ew}", if lon >= 0 { "E" } else { "W" })
        .replace("{lon3}", &format!("{:03}", lon.abs()))
}

impl DemSource {
    pub fn tile_name(&self, lat: i32, lon: i32) -> String {
        fill_tile(&self.tile_pattern, lat, lon)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SoilProp {
    /// multiply the stored integer by this to get `unit`
    pub scale: f64,
    pub unit: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SoilSource {
    pub dataset: String,
    pub url_pattern: String,
    /// {prefix} {prop} {depth}
    pub file_pattern: String,
    pub depth_labels: Vec<String>,
    pub depths_cm: Vec<[f64; 2]>,
    pub props: BTreeMap<String, SoilProp>,
    pub nodata: f64,
    /// which props hold sand, silt, clay (percent after scale) and organic carbon (g/kg)
    pub sand: String,
    pub silt: String,
    pub clay: String,
    pub organic_carbon: String,
    pub licence: Licence,
}

impl SoilSource {
    pub fn file_name(&self, prefix: &str, prop: &str, depth: &str) -> String {
        self.file_pattern.replace("{prefix}", prefix).replace("{prop}", prop).replace("{depth}", depth)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LandCoverClass {
    pub code: u32,
    pub name: String,
    /// Noah-OWP vegtyp per vegetation class table (USGS, MODIFIED_IGBP_MODIS_NOAH)
    pub veg: BTreeMap<String, i64>,
    /// vegtyp between 23.5 S and 23.5 N, where it differs (tree cover -> evergreen broadleaf)
    #[serde(default)]
    pub veg_tropical: BTreeMap<String, i64>,
    /// fraction of the pixel counted impervious (CFE urban_decimal_fraction)
    pub impervious: f64,
    /// NRCS velocity coefficient k (ft/s) for V = k * S^0.5 and the NEH row it comes from
    pub neh_k_ft_s: f64,
    pub neh_row: String,
    pub water: bool,
    pub ice: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LandCoverSource {
    pub dataset: String,
    pub url_pattern: String,
    /// {ns}{lat2}{ew}{lon3}, tiles named by the south-west corner
    pub tile_pattern: String,
    pub tile_degrees: u32,
    pub classes: Vec<LandCoverClass>,
    pub licence: Licence,
}

impl LandCoverSource {
    pub fn tile_name(&self, lat: i32, lon: i32) -> String {
        fill_tile(&self.tile_pattern, lat, lon)
    }
}

/// One STAS soil class row (Noah SOILPARM.TBL).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SoilClassRow {
    pub id: i64,
    pub name: String,
    pub bb: f64,
    pub maxsmc: f64,
    pub satpsi: f64,
    pub satdk: f64,
    pub wltsmc: f64,
    pub refsmc: f64,
    /// true for classes that carry soil hydraulics (not water/ice)
    pub hydraulic: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SoilRecipe {
    /// "class_lookup" or "cosby1984"
    pub method: String,
    pub soil_class_table: String,
    pub source: String,
    /// depth range (cm) whose texture sets the class / the parameters
    pub texture_depth_cm: [f64; 2],
    /// organic carbon (g/kg) at or above which a cell is ORGANIC MATERIAL
    pub organic_carbon_min_g_kg: f64,
    pub organic_class: i64,
    /// nearest-neighbour passes that fill SoilGrids gaps (built-up land); one soil cell of reach per pass
    pub gap_fill_cells: usize,
    pub water_class: i64,
    pub ice_class: i64,
    /// USDA texture class name -> STAS class id
    pub usda_to_class: BTreeMap<String, i64>,
    pub classes: Vec<SoilClassRow>,
    /// how each CFE soil parameter is aggregated over a catchment and why
    pub aggregation: BTreeMap<String, String>,
    /// Cosby et al. (1984) regression coefficients (method cosby1984)
    pub cosby: Option<BTreeMap<String, f64>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VgRow {
    pub usda: String,
    pub theta_r: f64,
    pub theta_e: f64,
    pub alpha_per_cm: f64,
    pub n: f64,
    pub ks_cm_h: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LgarRecipe {
    pub source: String,
    pub rows: Vec<VgRow>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Constant {
    pub value: f64,
    pub unit: String,
    pub source: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SlopRecipe {
    /// "terrain_tan_coarse" (mean tan(slope) of the DEM block-averaged to
    /// `coarse_m`) or "constant"
    pub method: String,
    pub coarse_m: f64,
    pub min: f64,
    pub max: f64,
    pub source: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GiuhRecipe {
    pub dt_s: f64,
    pub slope_floor: f64,
    pub channel_init_area_km2: f64,
    pub channel_neh_k_ft_s: f64,
    pub channel_neh_row: String,
    pub max_ordinates: usize,
    pub source: String,
}

impl ParamTable {
    pub fn load(path: &str) -> Result<ParamTable> {
        let s = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
        let t: ParamTable = serde_json::from_str(&s).map_err(|e| format!("{path}: {e}"))?;
        if t.schema != "arwen.hydro-world.params.v1" {
            return Err(format!("{path}: schema {} is not arwen.hydro-world.params.v1", t.schema).into());
        }
        for (k, r) in &t.soil_recipes {
            for (usda, id) in &r.usda_to_class {
                if !r.classes.iter().any(|c| c.id == *id) {
                    return Err(format!("soil recipe {k}: {usda} -> class {id} has no row").into());
                }
            }
        }
        for name in ["soil_depth_m", "refkdt", "alpha_fc", "max_gw_storage_m", "gw_coeff_m3_s", "gw_expon", "k_lf", "k_nash", "soil_color"] {
            if !t.constants.contains_key(name) {
                return Err(format!("{path}: constants.{name} missing").into());
            }
        }
        // every number carries its published source; a row still holding the
        // generator placeholder was built without its source file
        for (name, c) in &t.constants {
            if c.source.contains("PLACEHOLDER") || c.source.trim().is_empty() {
                return Err(format!("{path}: constants.{name} has no source").into());
            }
        }
        if let Some(d) = &t.cfe_recipe_default {
            if !t.cfe_recipes.contains_key(d) {
                return Err(format!("{path}: cfe_recipe_default {d} has no cfe_recipes row").into());
            }
        }
        for (k, r) in &t.cfe_recipes {
            if r.source.trim().is_empty() {
                return Err(format!("{path}: cfe_recipes.{k} has no source").into());
            }
            match r.method.as_str() {
                "constants" => {}
                "hillslope_kinematic" => {
                    if !(r.lateral_ksat_factor > 0.0 && r.n_nash >= 1.0 && r.atmosphere_head_m > 0.0 && r.min_hillslope_length_m > 0.0
                        && r.k_lf_range[1] > r.k_lf_range[0] && r.k_nash_range[1] > r.k_nash_range[0]
                        && !r.lateral_ksat_factor_source.trim().is_empty() && !r.alpha_fc_source.trim().is_empty())
                    {
                        return Err(format!("{path}: cfe_recipes.{k} (hillslope_kinematic) is incomplete").into());
                    }
                }
                m => return Err(format!("{path}: cfe_recipes.{k} has unknown method {m}").into()),
            }
        }
        Ok(t)
    }
    pub fn c(&self, name: &str) -> f64 {
        self.constants[name].value
    }
}
