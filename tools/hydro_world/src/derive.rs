//! Engine packets derived from a contract fabric, so the existing engine
//! preparers consume a HYDRO-WORLD fabric unchanged:
//! - `cfe_ini`: one CFE INI per divide for `nextgen-cfe-prepare`
//!   (request schema `arwen.cfe.prepare/v1`); keys, order and unit annotations
//!   follow the ngen example packet (data/gauge_01073000/CFE at be11a180).
//! - `noahowp_overrides`: the per-catchment override map `arwen-noahowp-init`
//!   accepts (lat, lon, terrain_slope, azimuth, isltyp, vegtyp, soilcolor).
//! - `canonicalize`: the row order every builder writes.
use crate::check::canonical_key;
use crate::contract::{Fabric, ModelAttributes};
use crate::Result;
use serde_json::{json, Value};

/// Initial storage fractions come from the initial-state provider (reanalysis
/// or spin-up), never from the static fabric.
#[derive(Clone, Copy, Debug)]
pub struct CfeInitial {
    /// soil_storage, fraction of soil capacity (m/m), 0..1
    pub soil_storage: f64,
    /// gw_storage, fraction of max_gw_storage (m/m), 0..1
    pub gw_storage: f64,
}

fn need(v: Option<f64>, name: &str, id: &str) -> Result<f64> {
    v.filter(|x| x.is_finite()).ok_or_else(|| format!("{id}: model_attributes.{name} missing").into())
}

pub fn cfe_ini(m: &ModelAttributes, init: CfeInitial) -> Result<String> {
    let id = m.divide_id.as_deref().unwrap_or("?");
    for (name, v) in [("soil_storage", init.soil_storage), ("gw_storage", init.gw_storage)] {
        if !(0.0..=1.0).contains(&v) {
            return Err(format!("{id}: initial {name}={v} is not a fraction").into());
        }
    }
    let giuh = m.giuh_ordinates.as_deref().ok_or_else(|| format!("{id}: giuh_ordinates missing"))?;
    let ords = giuh.split(',').map(|s| s.trim().parse::<f64>()).collect::<std::result::Result<Vec<_>, _>>()?;
    if ords.is_empty() || ords.iter().any(|x| !x.is_finite() || *x < 0.0) || (ords.iter().sum::<f64>() - 1.0).abs() > 1e-6 {
        return Err(format!("{id}: giuh_ordinates must be non-negative and sum to 1").into());
    }
    let mut s = String::new();
    let mut kv = |k: &str, v: String| {
        s.push_str(k);
        s.push('=');
        s.push_str(&v);
        s.push('\n');
    };
    kv("forcing_file", "BMI".into());
    kv("surface_partitioning_scheme", "Schaake".into());
    kv("soil_params.depth", format!("{}[m]", need(m.soil_depth_m, "soil_depth_m", id)?));
    kv("soil_params.b", format!("{}[]", need(m.soil_b, "soil_b", id)?));
    kv("soil_params.satdk", format!("{}[m s-1]", need(m.soil_satdk_m_s, "soil_satdk_m_s", id)?));
    kv("soil_params.satpsi", format!("{}[m]", need(m.soil_satpsi_m, "soil_satpsi_m", id)?));
    kv("soil_params.slop", format!("{}[m/m]", need(m.soil_slop, "soil_slop", id)?));
    kv("soil_params.smcmax", format!("{}[m/m]", need(m.soil_smcmax, "soil_smcmax", id)?));
    kv("soil_params.wltsmc", format!("{}[m/m]", need(m.soil_wltsmc, "soil_wltsmc", id)?));
    kv("soil_params.expon", "1.0[]".into());
    kv("soil_params.expon_secondary", "1.0[]".into());
    kv("refkdt", format!("{}", need(m.refkdt, "refkdt", id)?));
    kv("max_gw_storage", format!("{}[m]", need(m.max_gw_storage_m, "max_gw_storage_m", id)?));
    kv("Cgw", format!("{}[m h-1]", need(m.cgw_m_h, "cgw_m_h", id)?));
    kv("expon", format!("{}[]", need(m.gw_expon, "gw_expon", id)?));
    kv("gw_storage", format!("{}[m/m]", init.gw_storage));
    kv("alpha_fc", format!("{}", need(m.alpha_fc, "alpha_fc", id)?));
    kv("soil_storage", format!("{}[m/m]", init.soil_storage));
    kv("K_nash", format!("{}[]", need(m.k_nash, "k_nash", id)?));
    kv("K_lf", format!("{}[]", need(m.k_lf, "k_lf", id)?));
    kv("nash_storage_subsurface", "0.0,0.0".into());
    kv("num_timesteps", "1".into());
    kv("verbosity", "1".into());
    kv("DEBUG", "0".into());
    kv("giuh_ordinates", ords.iter().map(|x| format!("{x}")).collect::<Vec<_>>().join(","));
    Ok(s)
}

pub fn noahowp_overrides(m: &ModelAttributes) -> Result<Value> {
    let id = m.divide_id.as_deref().unwrap_or("?");
    let int = |v: Option<i64>, n: &str| v.ok_or_else(|| -> crate::Error { format!("{id}: model_attributes.{n} missing").into() });
    Ok(json!({
        "lat": need(m.centroid_lat_deg, "centroid_lat_deg", id)?,
        "lon": need(m.centroid_lon_deg, "centroid_lon_deg", id)?,
        "terrain_slope": need(m.slope_mean_deg, "slope_mean_deg", id)?,
        "azimuth": need(m.aspect_mean_deg, "aspect_mean_deg", id)?,
        "isltyp": int(m.soil_class, "soil_class")?,
        "vegtyp": int(m.veg_class, "veg_class")?,
        "soilcolor": int(m.soil_color, "soil_color")?,
    }))
}

/// Sort every table into canonical order and renumber fid = rank + 1.
/// network sorts by (id, hf_id, toid): v2.0.1 lists a flowpath once per
/// source member reach, and the order must still be deterministic.
pub fn canonicalize(f: &mut Fabric) {
    macro_rules! sort {
        ($v:expr, $key:expr) => {{
            $v.sort_by(|a, b| $key(a).cmp(&$key(b)));
            for (i, r) in $v.iter_mut().enumerate() {
                r.fid = i as i64 + 1;
            }
        }};
    }
    let k = |s: &Option<String>| (canonical_key(s.as_deref().unwrap_or("")), s.clone());
    sort!(f.flowpaths, |r: &crate::contract::Flowpath| k(&r.id));
    sort!(f.flowpath_attributes, |r: &crate::contract::FlowpathAttributes| k(&r.id));
    sort!(f.divides, |r: &crate::contract::Divide| k(&r.divide_id));
    sort!(f.nexus, |r: &crate::contract::Nexus| k(&r.id));
    sort!(f.network, |r: &crate::contract::NetworkRow| (k(&r.id), r.hf_id.map(f64::to_bits), r.toid.clone()));
    sort!(f.hydrolocations, |r: &crate::contract::Hydrolocation| (k(&r.id), r.hl_uri.clone()));
    sort!(f.lakes, |r: &crate::contract::Lake| (k(&r.id), r.hl_link.map(f64::to_bits)));
    if let Some(m) = f.model_attributes.as_mut() {
        sort!(m, |r: &ModelAttributes| k(&r.divide_id));
    }
}
