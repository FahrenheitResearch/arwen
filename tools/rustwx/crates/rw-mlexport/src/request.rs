//! The request `gpuwm ml-export` writes and this binary carries out.
//!
//! Schema `ml-export.request/v1`.  The front door resolves its four tables
//! (variables, level sets, lattice spacings, naming schemes) into the rows a
//! run needs and hands them over whole, so a new variable or level set is a
//! table row on the Python side and nothing here changes.  Nothing in the
//! request names an engine except the provenance strings, which are copied
//! into the dataset as text.

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::error::{refuse, Result};

/// The request schema this binary speaks.
pub const SCHEMA: &str = "ml-export.request/v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Mode {
    /// Every frame, then finalize: the whole export in one call.
    Run,
    /// Add the request's frames to the export folder's datasets.
    Append,
    /// Close the datasets in the export folder (metadata, README, receipt,
    /// optional ZIP).
    Finalize,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Request {
    pub schema: String,
    pub mode: Mode,
    /// History files, `.gz` history files, or ZIPs whose `wrfout_dNN_*`
    /// members are the history files.
    #[serde(default)]
    pub inputs: Vec<PathBuf>,
    /// The export folder: `<out>/dNN.zarr`, `README.txt`,
    /// `ml-export-receipt.json`; with `zip`, `<out>-ml.zip` beside it.
    pub out: PathBuf,
    #[serde(default)]
    pub overwrite: bool,
    #[serde(default)]
    pub zip: bool,
    #[serde(default)]
    pub variables: Vec<VariableRow>,
    pub levels: LevelSpec,
    pub grid: GridRequest,
    /// The naming scheme: a key into each row's `names`.
    pub names: String,
    #[serde(default = "default_layout")]
    pub layout: Layout,
    #[serde(default)]
    pub domains: Option<Vec<String>>,
    #[serde(default)]
    pub every_hours: Option<f64>,
    #[serde(default)]
    pub start: Option<String>,
    #[serde(default)]
    pub end: Option<String>,
    #[serde(default)]
    pub skip_unavailable: bool,
    #[serde(default)]
    pub threads: Option<usize>,
    /// Lattice spacings (degrees) `--grid latlon` picks from when no
    /// spacing is given.
    #[serde(default)]
    pub spacings_deg: Vec<f64>,
    pub provenance: Provenance,
}

fn default_layout() -> Layout {
    Layout::Analysis
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Layout {
    /// `time` is the valid time (ERA5's shape).
    Analysis,
    /// `time` is the one initialization time and `prediction_timedelta` the
    /// lead (WeatherBench 2's forecast shape).
    Forecast,
}

/// Provenance strings copied into the dataset as given.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct Provenance {
    /// Source metadata aliases, supplied as table rows by the front door.
    #[serde(default)]
    pub history_attributes: BTreeMap<String, Vec<String>>,
    /// Engine identity keyed by its source version attribute, as table data.
    #[serde(default)]
    pub history_engines: BTreeMap<String, String>,
    /// Title-prefix identity for histories carrying a legacy version attribute.
    #[serde(default)]
    pub history_engine_titles: BTreeMap<String, String>,
    /// The engine's name as the front door spells it (`source` attribute).
    pub engine: String,
    /// The front door's own version (`exporter` attribute).
    pub exporter_version: String,
    /// SHA-256 of the run's configuration file, when the caller had one.
    #[serde(default)]
    pub config_sha256: Option<String>,
    /// UTC time of the request (`history` attribute).  Carried in the
    /// request rather than read from the clock so two exports of the same
    /// input with the same request are byte-identical.
    #[serde(default)]
    pub created_utc: String,
    /// The options as the user gave them (`history` attribute).
    #[serde(default)]
    pub options: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum LevelKind {
    Pressure,
    Model,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LevelSpec {
    /// The level set's id (`wb13`, `era5-37`, `model`, `custom`).
    pub set: String,
    pub kind: LevelKind,
    /// Pressure levels (hPa), any order; written in the order given.
    #[serde(default)]
    pub hpa: Vec<u32>,
    /// Model levels to keep, 1 at the bottom; empty keeps every level.
    #[serde(default)]
    pub model_levels: Vec<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum GridKind {
    Native,
    Latlon,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum RegridMethod {
    Bilinear,
    AreaMean,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct GridRequest {
    pub kind: GridKind,
    #[serde(default)]
    pub deg: Option<f64>,
    #[serde(default = "default_method")]
    pub method: RegridMethod,
}

fn default_method() -> RegridMethod {
    RegridMethod::Bilinear
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum VariableKind {
    /// `(time, level, y, x)`.
    Level,
    /// `(time, y, x)`.
    Surface,
    /// `(y, x)`, from the first frame.
    Static,
}

/// One row of the variables table, as the front door resolved it.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct VariableRow {
    pub id: String,
    pub kind: VariableKind,
    /// Output name per naming scheme (`wb2`, `era5`, ...).
    pub names: BTreeMap<String, String>,
    /// The operator that makes it (see `ops.rs`).
    pub op: String,
    /// History-file fields it cannot be made without.
    #[serde(default)]
    pub fields: Vec<String>,
    /// Fields it reads when present.
    #[serde(default)]
    pub optional_fields: Vec<String>,
    /// The below-ground rule for a level variable.
    #[serde(default)]
    pub below_ground: Option<String>,
    /// Multiplier applied to a `raw:` operator's value.
    #[serde(default)]
    pub scale: Option<f64>,
    pub units: String,
    #[serde(default)]
    pub long_name: String,
    #[serde(default)]
    pub standard_name: Option<String>,
    #[serde(default)]
    pub short_name: Option<String>,
    #[serde(default)]
    pub param_id: Option<i64>,
    #[serde(default)]
    pub comment: Option<String>,
}

impl VariableRow {
    pub fn out_name(&self, scheme: &str) -> Result<String> {
        self.names.get(scheme).cloned().ok_or_else(|| {
            refuse(format!(
                "variable '{}' has no name under the '{scheme}' naming scheme, so it would be written under a name no loader expects",
                self.id
            ))
        })
    }
}

impl Request {
    /// Structural checks the front door also makes, repeated here because
    /// the binary is a door of its own.
    pub fn validate(&self) -> Result<()> {
        if self.schema != SCHEMA {
            return Err(refuse(format!(
                "request schema '{}' is not {SCHEMA}; a request this binary does not speak would be read wrongly rather than refused later",
                self.schema
            )));
        }
        if self.mode != Mode::Finalize && self.inputs.is_empty() {
            return Err(refuse(
                "no input history files were given, so there is nothing to export",
            ));
        }
        if self.mode != Mode::Finalize && self.variables.is_empty() {
            return Err(refuse("no variables were selected, so the dataset would be empty"));
        }
        let mut seen = BTreeMap::new();
        for row in &self.variables {
            let name = row.out_name(&self.names)?;
            if let Some(other) = seen.insert(name.clone(), row.id.clone()) {
                return Err(refuse(format!(
                    "variables '{other}' and '{}' are both named '{name}' under the '{}' scheme, so one would overwrite the other",
                    row.id, self.names
                )));
            }
        }
        if self.levels.kind == LevelKind::Pressure {
            if self.levels.hpa.is_empty()
                && self.variables.iter().any(|r| r.kind == VariableKind::Level)
            {
                return Err(refuse("the level set is empty, so no level variable could be written"));
            }
            let mut sorted = self.levels.hpa.clone();
            sorted.sort_unstable();
            sorted.dedup();
            if sorted.len() != self.levels.hpa.len() {
                return Err(refuse(
                    "a pressure level is listed twice, so two slices of one array would hold the same level",
                ));
            }
            if let Some(bad) = self.levels.hpa.iter().find(|&&p| p == 0 || p > 1100) {
                return Err(refuse(format!(
                    "pressure level {bad} hPa is outside 1 to 1100 hPa, so it cannot be a level of any atmosphere this model runs"
                )));
            }
        }
        if let Some(deg) = self.grid.deg {
            if !(deg.is_finite() && deg > 0.0 && deg <= 10.0) {
                return Err(refuse(format!(
                    "lattice spacing {deg} degrees is outside (0, 10], so the regular grid would be empty or meaningless"
                )));
            }
        }
        if let Some(every) = self.every_hours {
            if !(every.is_finite() && every > 0.0) {
                return Err(refuse(format!(
                    "--every {every} is not a positive number of hours, so no frame could be selected"
                )));
            }
        }
        Ok(())
    }
}
