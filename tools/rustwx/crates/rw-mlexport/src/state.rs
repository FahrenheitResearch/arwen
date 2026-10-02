//! What an unfinished export remembers between `append` calls, kept in
//! `<out>/.ml-export-state/` and deleted when the export is finalized.
//!
//! `run` goes through the same state (append every frame, then finalize),
//! so a frame-by-frame export and a one-call export of the same frames are
//! the same bytes.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::{fail, refuse, Result};
use crate::frame::FileMeta;
use crate::grid::NativeGrid;
use crate::request::Request;

pub const STATE_DIR: &str = ".ml-export-state";
const STATE_FILE: &str = "state.json";
pub const STATE_SCHEMA: &str = "ml-export.state/v3";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputRecord {
    pub name: String,
    pub sha256: String,
    pub time_index: usize,
    pub valid: i64,
    /// Seconds making and writing the frame's variables.
    pub seconds: f64,
    /// Seconds unpacking, opening and hashing its input (shared equally by
    /// the frames one input holds).
    #[serde(default)]
    pub read_seconds: f64,
    pub bytes_written: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VarState {
    pub id: String,
    pub name: String,
    /// Why the variable is not in the dataset, when it is not.
    pub omitted: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegridInfo {
    pub lat: Vec<f64>,
    pub lon: Vec<f64>,
    pub deg: f64,
    pub description: String,
    pub boundary_rows: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainState {
    pub id: String,
    pub meta: FileMeta,
    pub native: NativeGrid,
    pub regrid: Option<RegridInfo>,
    /// Output grid size (after any regrid).
    pub out_ny: usize,
    pub out_nx: usize,
    pub init_time: i64,
    pub times: Vec<i64>,
    pub inputs: Vec<InputRecord>,
    /// Pressure levels kept and dropped (hPa), or model levels kept.
    pub levels_kept: Vec<u32>,
    pub levels_dropped: Vec<u32>,
    pub model_levels: Vec<usize>,
    /// ZNU at the kept model levels, when the file carries it.
    pub eta: Option<Vec<f64>>,
    /// The lid the levels were decided against (Pa), and whether the file
    /// stated it (P_TOP) or it is the top mass level.
    pub lid_pa: f64,
    pub lid_stated: bool,
    pub levels_per_chunk: usize,
    pub variables: Vec<VarState>,
    /// For each time, which stored latitude/longitude it has (0 = the
    /// first frame's).  More than one version is a moving nest.
    pub latlon_version: Vec<usize>,
    pub latlon_versions: usize,
    /// For each time, whether an accumulation period found its partner.
    pub period_partners: BTreeMap<String, Vec<bool>>,
    /// SHA-256 standing for the run's configuration, and what it is of.
    pub config_digest: String,
    pub config_digest_kind: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct State {
    pub schema: String,
    /// The options every append must repeat (see `options_digest`).
    pub options_digest: String,
    pub request: Request,
    pub domains: BTreeMap<String, DomainState>,
}

pub fn state_dir(out: &Path) -> PathBuf {
    out.join(STATE_DIR)
}

impl State {
    pub fn load(out: &Path) -> Result<Option<State>> {
        let path = state_dir(out).join(STATE_FILE);
        if !path.is_file() {
            return Ok(None);
        }
        let text = fs::read_to_string(&path)?;
        let state: State = serde_json::from_str(&text)
            .map_err(|e| fail(format!("the export state in {} is unreadable: {e}", path.display())))?;
        if state.schema != STATE_SCHEMA {
            return Err(refuse(format!(
                "{} was started by a different exporter version ({}), so appending to it would mix two layouts in one dataset",
                out.display(),
                state.schema
            )));
        }
        Ok(Some(state))
    }

    pub fn save(&self, out: &Path) -> Result<()> {
        let dir = state_dir(out);
        fs::create_dir_all(&dir)?;
        let text = serde_json::to_string(self).map_err(|e| fail(format!("{e}")))?;
        let partial = dir.join("state.json.partial");
        fs::write(&partial, text)?;
        fs::rename(&partial, dir.join(STATE_FILE))?;
        Ok(())
    }
}

/// Store and load f64 planes in the state folder.
pub fn save_plane(out: &Path, name: &str, values: &[f64]) -> Result<()> {
    let dir = state_dir(out);
    fs::create_dir_all(&dir)?;
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    fs::write(dir.join(name), bytes)?;
    Ok(())
}

pub fn load_plane(out: &Path, name: &str) -> Result<Option<Vec<f64>>> {
    let path = state_dir(out).join(name);
    if !path.is_file() {
        return Ok(None);
    }
    let bytes = fs::read(&path)?;
    Ok(Some(
        bytes
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
            .collect(),
    ))
}

pub fn remove_plane(out: &Path, name: &str) {
    let _ = fs::remove_file(state_dir(out).join(name));
}
