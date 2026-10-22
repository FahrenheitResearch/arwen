//! The init-source table (`arwen.hydro-world.init-sources.v1`). A source is a
//! row; the code below reads any row with the same shape.
use crate::Result;
use serde::Deserialize;
use serde_json::Value;
use std::collections::BTreeMap;

pub const SCHEMA: &str = "arwen.hydro-world.init-sources.v1";

#[derive(Deserialize, Debug, Clone)]
pub struct Table {
    pub schema: String,
    pub target: Target,
    pub rows: BTreeMap<String, Row>,
}

#[derive(Deserialize, Debug, Clone)]
pub struct Target {
    pub noah_soil_layer_bounds_m: Vec<f64>,
    pub freezing_point_k: f64,
    pub latent_heat_fusion_j_kg: f64,
    pub gravity_m_s2: f64,
}

#[derive(Deserialize, Debug, Clone)]
pub struct Row {
    pub status: String,
    #[serde(default)]
    pub reason: Option<String>,
    pub licence: Licence,
    pub grid: Grid,
    pub state: State,
    #[serde(default)]
    pub soil_moisture_transfer: Option<Transfer>,
    #[serde(default)]
    pub forcing: BTreeMap<String, Forcing>,
}

#[derive(Deserialize, Debug, Clone)]
pub struct Licence {
    pub spdx: String,
    pub status: String,
    pub attribution: String,
}

#[derive(Deserialize, Debug, Clone)]
pub struct Grid {
    pub kind: String,
    #[serde(default)]
    pub latitude: Option<String>,
    #[serde(default)]
    pub longitude: Option<String>,
    /// Projected grid: x/y coordinate variables (cell centres, metres) and
    /// the PROJ definition of their CRS.
    #[serde(default)]
    pub x: Option<String>,
    #[serde(default)]
    pub y: Option<String>,
    #[serde(default)]
    pub crs: Option<String>,
}

#[derive(Deserialize, Debug, Clone)]
pub struct Layer {
    pub variable: String,
    pub top_m: f64,
    pub bottom_m: f64,
}

#[derive(Deserialize, Debug, Clone)]
pub struct Layered {
    pub units: String,
    pub scale: Value,
    pub layers: Vec<Layer>,
}

#[derive(Deserialize, Debug, Clone)]
pub struct Scalar {
    pub variable: String,
    #[serde(default = "one")]
    pub scale: f64,
}
fn one() -> f64 {
    1.0
}

#[derive(Deserialize, Debug, Clone)]
pub struct State {
    pub soil_water: Layered,
    pub soil_temperature: Layered,
    pub snow_water_equivalent: Scalar,
    #[serde(default)]
    pub snow_density: Option<Scalar>,
    #[serde(default)]
    pub snow_depth: Option<Scalar>,
    #[serde(default)]
    pub snow_temperature: Option<Scalar>,
    #[serde(default)]
    pub land_fraction: Option<Value>,
    #[serde(default)]
    pub soil_class: Option<Scalar>,
}

#[derive(Deserialize, Debug, Clone)]
pub struct SoilClass {
    pub name: String,
    pub theta_sat: f64,
    #[serde(default)]
    pub theta_cap: Option<f64>,
    pub theta_pwp: f64,
}

#[derive(Deserialize, Debug, Clone)]
pub struct Transfer {
    pub method: String,
    #[serde(default)]
    pub classes: BTreeMap<String, SoilClass>,
    #[serde(default)]
    pub fallback_class: Option<String>,
    #[serde(default)]
    pub same_classes_as: Option<String>,
}

#[derive(Deserialize, Debug, Clone)]
pub struct Term {
    pub variable: String,
    pub sign: f64,
    /// Per-term overrides of the binding's temporal kind and scale.
    #[serde(default)]
    pub temporal: Option<String>,
    #[serde(default)]
    pub scale: Option<f64>,
}

#[derive(Deserialize, Debug, Clone)]
pub struct Forcing {
    pub units: String,
    pub temporal: String,
    pub interval_seconds: i64,
    pub time_offset_seconds: i64,
    #[serde(default)]
    pub clamp_min: Option<f64>,
    #[serde(default)]
    pub scale: Option<f64>,
    pub terms: Vec<Term>,
}

pub fn load(path: &str) -> Result<Table> {
    let t: Table = serde_json::from_str(&std::fs::read_to_string(path)?)?;
    if t.schema != SCHEMA {
        return Err(format!("{path}: schema {} is not {SCHEMA}", t.schema).into());
    }
    Ok(t)
}

/// The row if it may be used, else the refusal the row itself states.
/// Prevents a product built on an input whose licence was not read at the
/// source, that needs an account not yet ruled on, or that has no reader.
pub fn admit<'a>(t: &'a Table, name: &str) -> Result<&'a Row> {
    let row = t.rows.get(name).ok_or_else(|| format!("no init-source row {name}"))?;
    if row.status != "admitted" {
        return Err(format!("init-source row {name} is {}: {}", row.status, row.reason.as_deref().unwrap_or("")).into());
    }
    if row.licence.status != "verified" {
        return Err(format!("init-source row {name}: licence not verified at the source").into());
    }
    Ok(row)
}

/// Soil classes for the wetness-index transfer, following same_classes_as.
pub fn classes<'a>(t: &'a Table, row: &'a Row) -> Result<Option<&'a Transfer>> {
    let Some(tr) = row.soil_moisture_transfer.as_ref() else { return Ok(None) };
    if let Some(other) = tr.same_classes_as.as_deref() {
        let o = t.rows.get(other).and_then(|r| r.soil_moisture_transfer.as_ref()).ok_or("same_classes_as names no row")?;
        return Ok(Some(o));
    }
    Ok(Some(tr))
}
