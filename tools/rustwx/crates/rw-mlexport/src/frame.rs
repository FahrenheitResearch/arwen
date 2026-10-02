//! One history file opened through wrf-core's pure-Rust reader: its grid,
//! its projection, its provenance attributes and its valid times.

use std::io::Read;
use std::path::Path;

use sha2::{Digest, Sha256};
use wrf_core::WrfFile;

use crate::error::{fail, refuse, Result};
use crate::times;
use crate::request::Provenance;

/// What a history file says about itself.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct FileMeta {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub domain: String,
    pub parent: Option<String>,
    pub map_proj: i32,
    pub truelat1: f64,
    pub truelat2: f64,
    pub stand_lon: f64,
    pub cen_lat: f64,
    pub cen_lon: f64,
    pub moad_cen_lat: f64,
    pub pole_lat: f64,
    pub pole_lon: f64,
    pub dx: f64,
    pub dy: f64,
    /// The run's start (`SIMULATION_START_DATE`), the head grid's start.
    pub simulation_start: Option<i64>,
    pub gpuwm_version: Option<String>,
    #[serde(default)]
    pub source_engine: Option<String>,
    pub ic_source: Option<String>,
    pub ic_cycle: Option<String>,
    pub history_preset: Option<String>,
    pub spec_bdy_width: Option<i64>,
    pub bucket_mm: Option<f64>,
}

impl FileMeta {
    /// The attributes that decide whether two files are the same grid.
    ///
    /// Floating attributes agree to a part in 1e9 rather than bit for bit:
    /// the first frame's are read back from the export's saved state, and a
    /// JSON round trip may move a float by its last bit, which made a
    /// 967.383 m hex window refuse its own second frame.
    pub fn same_grid(&self, other: &FileMeta) -> bool {
        let close = |a: f64, b: f64| (a - b).abs() <= 1e-9 * a.abs().max(b.abs()).max(1.0);
        self.nx == other.nx
            && self.ny == other.ny
            && self.nz == other.nz
            && self.map_proj == other.map_proj
            && close(self.truelat1, other.truelat1)
            && close(self.truelat2, other.truelat2)
            && close(self.stand_lon, other.stand_lon)
            && close(self.dx, other.dx)
            && close(self.dy, other.dy)
    }
}

pub struct Opened {
    pub file: WrfFile,
    pub meta: FileMeta,
    pub times: Vec<i64>,
}

fn attr_f64(file: &WrfFile, name: &str) -> Option<f64> {
    file.global_attr_f64(name).ok().filter(|v| v.is_finite())
}

fn attr_str(file: &WrfFile, name: &str) -> Option<String> {
    file.global_attr_str(name)
        .ok()
        .map(|s| s.trim_end_matches('\0').trim().to_string())
        .filter(|s| !s.is_empty())
}

/// Open `path` and read what the export needs to place its frames.
/// `name` is the history file's base name, `hint` its name-derived domain.
pub fn open(path: &Path, name: &str, hint_domain: Option<&str>, provenance: &Provenance) -> Result<Opened> {
    let file = WrfFile::open(path).map_err(|e| {
        refuse(format!(
            "{name} cannot be read as a WRF history file ({e}), so none of its fields could be exported"
        ))
    })?;
    let grid_id = file.global_attr_i32("GRID_ID").ok();
    let domain = match (grid_id, hint_domain) {
        (Some(id), _) if id > 0 => format!("d{id:02}"),
        (_, Some(d)) => d.to_string(),
        _ => "d01".to_string(),
    };
    let parent = file
        .global_attr_i32("PARENT_ID")
        .ok()
        .filter(|&p| p > 0 && Some(p) != grid_id)
        .map(|p| format!("d{p:02}"));
    let map_proj = file.global_attr_i32("MAP_PROJ").map_err(|_| {
        refuse(format!(
            "{name} carries no MAP_PROJ attribute, so its grid cannot be placed on the earth"
        ))
    })?;
    let dx = attr_f64(&file, "DX").unwrap_or(file.dx);
    let dy = attr_f64(&file, "DY").unwrap_or(file.dy);
    let simulation_start = attr_str(&file, "SIMULATION_START_DATE").and_then(|s| times::parse(&s));
    let meta = FileMeta {
        nx: file.nx,
        ny: file.ny,
        nz: file.nz,
        domain,
        parent,
        map_proj,
        truelat1: attr_f64(&file, "TRUELAT1").unwrap_or(0.0),
        truelat2: attr_f64(&file, "TRUELAT2").unwrap_or(0.0),
        stand_lon: attr_f64(&file, "STAND_LON").unwrap_or(0.0),
        cen_lat: attr_f64(&file, "CEN_LAT").unwrap_or(0.0),
        cen_lon: attr_f64(&file, "CEN_LON").unwrap_or(0.0),
        moad_cen_lat: attr_f64(&file, "MOAD_CEN_LAT").unwrap_or(0.0),
        pole_lat: attr_f64(&file, "POLE_LAT").unwrap_or(90.0),
        pole_lon: attr_f64(&file, "POLE_LON").unwrap_or(0.0),
        dx,
        dy,
        simulation_start,
        gpuwm_version: source_attr(&file, provenance, "version", "GPUWM_VERSION")?,
        source_engine: source_engine(&file, provenance)?,
        ic_source: source_attr(&file, provenance, "ic_source", "GPUWM_INITIAL_CONDITION_SOURCE")?,
        ic_cycle: source_attr(&file, provenance, "ic_cycle", "GPUWM_INITIAL_CONDITION_CYCLE")?,
        history_preset: source_attr(&file, provenance, "history_preset", "GPUWM_HISTORY_PRESET")?,
        spec_bdy_width: file.global_attr_i32("SPEC_BDY_WIDTH").ok().map(i64::from),
        bucket_mm: attr_f64(&file, "BUCKET_MM"),
    };
    let labels = file.times().map_err(|e| {
        refuse(format!(
            "{name} has no readable Times variable ({e}), so its frames cannot be placed on the time axis"
        ))
    })?;
    let mut valid = Vec::with_capacity(labels.len());
    for label in &labels {
        valid.push(times::parse_named(label, &format!("{name}: Times entry"))?);
    }
    if valid.is_empty() {
        return Err(refuse(format!("{name} holds no time records, so there is nothing to export from it")));
    }
    Ok(Opened { file, meta, times: valid })
}

fn source_attr(file: &WrfFile, provenance: &Provenance, key: &str, legacy: &str) -> Result<Option<String>> {
    let aliases = provenance.history_attributes.get(key);
    let names: Vec<&str> = aliases.map(|v| v.iter().map(String::as_str).collect()).unwrap_or_else(|| vec![legacy]);
    let values: std::collections::BTreeSet<String> = names.into_iter().filter_map(|n| attr_str(file, n)).collect();
    if values.len() > 1 {
        return Err(refuse(format!("the history file carries conflicting {key} metadata aliases, so its source provenance cannot be assigned correctly")));
    }
    Ok(values.into_iter().next())
}

fn source_engine(file: &WrfFile, provenance: &Provenance) -> Result<Option<String>> {
    if let Some(title) = attr_str(file, "TITLE") {
        let engines: std::collections::BTreeSet<String> = provenance.history_engine_titles.iter()
            .filter(|(prefix, _)| title.starts_with(prefix.as_str()))
            .map(|(_, engine)| engine.clone()).collect();
        if engines.len() > 1 {
            return Err(refuse("the history title matches different source engine identities, so its source cannot be labeled correctly"));
        }
        if let Some(engine) = engines.into_iter().next() { return Ok(Some(engine)); }
    }
    let engines: std::collections::BTreeSet<String> = provenance.history_engines.iter()
        .filter(|(attribute, _)| attr_str(file, attribute).is_some())
        .map(|(_, engine)| engine.clone()).collect();
    if engines.len() > 1 {
        return Err(refuse("the history file carries version attributes for different source engines, so its source cannot be labeled correctly"));
    }
    Ok(engines.into_iter().next())
}

/// SHA-256 of a file's bytes, hex.
pub fn sha256_file(path: &Path) -> Result<String> {
    let mut file = std::fs::File::open(path).map_err(|e| fail(format!("could not hash {}: {e}", path.display())))?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 8 << 20];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }
    Ok(hex(&hasher.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_grid_read_back_from_json_is_still_the_same_grid() {
        let meta = FileMeta {
            nx: 248, ny: 248, nz: 55, domain: "d01".into(), parent: None, map_proj: 1,
            truelat1: f64::from(32.67113f32), truelat2: f64::from(42.67113f32),
            stand_lon: f64::from(-122.2584f32), cen_lat: 37.6755, cen_lon: -122.26,
            moad_cen_lat: 37.6755, pole_lat: 90.0, pole_lon: 0.0,
            dx: f64::from(967.383f32), dy: f64::from(967.383f32),
            simulation_start: None, gpuwm_version: None, ic_source: None, ic_cycle: None,
            source_engine: None,
            history_preset: None, spec_bdy_width: None, bucket_mm: None,
        };
        let back: FileMeta = serde_json::from_str(&serde_json::to_string(&meta).unwrap()).unwrap();
        assert!(meta.same_grid(&back));
        let mut moved = meta.clone();
        moved.dx = 1000.0;
        assert!(!meta.same_grid(&moved));
    }
}

pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

pub fn sha256_bytes(bytes: &[u8]) -> String {
    hex(&Sha256::digest(bytes))
}
