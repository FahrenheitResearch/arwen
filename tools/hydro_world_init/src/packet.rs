//! Sealed output packets (little-endian raw arrays + manifest with SHA-256)
//! and CFE parameter rows.
use crate::cfe::{self, Params};
use crate::Result;
use rayon::prelude::*;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::path::Path;

pub fn sha256_hex(b: &[u8]) -> String {
    Sha256::digest(b).iter().map(|x| format!("{x:02x}")).collect()
}

pub struct Array {
    pub name: String,
    pub shape: Vec<usize>,
    pub data: Vec<f64>,
    pub units: String,
    pub meaning: String,
}

/// Write arrays as <name>.f64 (little-endian) in parallel and return their
/// manifest entries. The directory must be new: evidence is never overwritten.
pub fn write_arrays(dir: &Path, arrays: &[Array]) -> Result<Vec<Value>> {
    let entries: Vec<Result<Value>> = arrays
        .par_iter()
        .map(|a| {
            let n: usize = a.shape.iter().product();
            if n != a.data.len() {
                return Err(format!("{}: shape {:?} != {} values", a.name, a.shape, a.data.len()).into());
            }
            let mut b = Vec::with_capacity(8 * n);
            for v in &a.data {
                b.extend_from_slice(&v.to_le_bytes());
            }
            let file = format!("{}.f64", a.name);
            std::fs::write(dir.join(&file), &b)?;
            Ok(json!({"name": a.name, "file": file, "dtype": "<f8", "shape": a.shape, "bytes": b.len(),
                      "sha256": sha256_hex(&b), "units": a.units, "meaning": a.meaning}))
        })
        .collect();
    entries.into_iter().collect()
}

pub fn new_dir(path: &str) -> Result<&Path> {
    let p = Path::new(path);
    std::fs::create_dir(p).map_err(|e| format!("{path}: output directory must be new ({e})"))?;
    Ok(p)
}

pub fn write_json(path: &Path, v: &Value) -> Result<String> {
    let s = serde_json::to_string_pretty(v)? + "\n";
    std::fs::write(path, &s)?;
    Ok(sha256_hex(s.as_bytes()))
}

/// CFE parameters per catchment, in divide order: the fabric's own
/// model_attributes when every divide has a row, else one parameter row file
/// (schema arwen.hydro-world.cfe-params-row.v1) for all catchments, which the
/// receipt names. Both go through hydro_world::derive::cfe_ini, the packet the
/// engine's nextgen-cfe-prepare reads, then the same INI parser as the port.
pub fn params_for(fabric: &hydro_world::contract::Fabric, ids: &[String], row_file: Option<&str>) -> Result<(Vec<Params>, Vec<String>, Value)> {
    use hydro_world::derive::{cfe_ini, CfeInitial};
    let init = CfeInitial { soil_storage: 0.5, gw_storage: 0.5 };
    if let Some(m) = fabric.model_attributes.as_ref().filter(|m| m.len() == ids.len()) {
        let by_id: std::collections::BTreeMap<&str, &hydro_world::contract::ModelAttributes> =
            m.iter().filter_map(|r| r.divide_id.as_deref().map(|d| (d, r))).collect();
        // One INI per catchment, built and parsed on all cores, in id order.

        let pairs = ids
            .par_iter()
            .map(|id| -> Result<(Params, String)> {
                let r = by_id.get(id.as_str()).ok_or_else(|| format!("{id}: no model_attributes row"))?;
                let ini = cfe_ini(r, init)?;
                Ok((cfe::parse_ini(&ini)?, ini))
            })
            .collect::<Result<Vec<_>>>()?;
        let (ps, inis): (Vec<Params>, Vec<String>) = pairs.into_iter().unzip();
        return Ok((ps, inis, json!({"source": "fabric model_attributes", "gw_storage_cold_start": init.gw_storage})));
    }
    let path = row_file.ok_or("fabric has no model_attributes; pass a CFE parameter row file (params)")?;
    let v: Value = serde_json::from_str(&std::fs::read_to_string(path)?)?;
    if v["schema"] != "arwen.hydro-world.cfe-params-row.v1" {
        return Err(format!("{path}: not an arwen.hydro-world.cfe-params-row.v1 file").into());
    }
    let m: hydro_world::contract::ModelAttributes = serde_json::from_value(v["model_attributes"].clone())?;
    let gw0 = v["gw_storage_cold_start"].as_f64().ok_or("params row lacks gw_storage_cold_start")?;
    let ini = cfe_ini(&m, CfeInitial { soil_storage: 0.5, gw_storage: gw0 })?;
    let mut p = cfe::parse_ini(&ini)?;
    p.gw_storage = gw0;
    let bytes = std::fs::read(path)?;
    Ok((vec![p; ids.len()], vec![ini; ids.len()],
        json!({"source": "parameter row file", "file": path, "sha256": sha256_hex(&bytes),
               "provenance": v["provenance"], "gw_storage_cold_start": gw0})))
}
