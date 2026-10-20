//! Complete physical CFE initialization, not a restart of native budget counters.
use crate::{cfe, packet, read_engine_array, Result};
use rayon::prelude::*;
use serde::Deserialize;
use serde_json::{json, Value};
use std::path::Path;

const SCHEMA: &str = "arwen.hydro-world.cfe-raw-initial.v1";
const FILE: &str = "cfe-parameters.json";

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Column { id: String, ini: String }

/// Seal the actual INIs used by spin-up, including column-specific counts.
pub fn write(dir: &Path, columns: Vec<Value>) -> Result<Value> {
    let count = columns.len();
    let sha = packet::write_json(&dir.join(FILE), &json!(columns))?;
    Ok(json!({"schema": SCHEMA, "parameters": {"file": FILE, "sha256": sha, "count": count}}))
}

fn initial_ini(ini: &str, expected: &cfe::Params, soil: f64, gw: f64, nash: &[f64], queue: &[f64], perc: f64) -> Result<String> {
    let p = cfe::parse_ini(ini)?;
    let mut bound = expected.clone();
    // Initial fractions/Nash seeds describe the start of spin-up, not static
    // parameters. All static parameters and both carrier counts must agree.
    bound.soil_storage = p.soil_storage;
    bound.gw_storage = p.gw_storage;
    bound.nash_storage_subsurface = p.nash_storage_subsurface.clone();
    if bound != p { return Err("CFE raw state: spin-up parameters differ from fabric parameters".into()); }
    let m = cfe::Cfe::new(&p);
    let nn = m.nash.len();
    let ng = m.giuh.len();
    if nn == 0 || ng == 0 || nash.len() < nn || queue.len() < ng
        || nash[nn..].iter().chain(&queue[ng..]).any(|x| *x != 0.0) {
        return Err("CFE raw state: carrier counts/padding differ from spin-up parameters".into());
    }
    if !soil.is_finite() || soil < 0.0 || soil > m.soil.storage_max_m
        || !gw.is_finite() || gw < 0.0 || gw > m.gw.storage_max_m
        || !perc.is_finite() || perc < 0.0
        || nash.iter().chain(queue).any(|x| !x.is_finite() || *x < 0.0)
        || !(soil + gw + nash[..nn].iter().sum::<f64>() + queue[..ng].iter().sum::<f64>()).is_finite() {
        return Err("CFE raw state: nonfinite or out-of-bounds physical storage/carry".into());
    }
    let csv = |v: &[f64]| v.iter().map(f64::to_string).collect::<Vec<_>>().join(",");
    Ok(format!("{ini}initial_soil_storage_m={soil}\ninitial_gw_storage_m={gw}\ninitial_nash_subsurface_m={}\ninitial_giuh_queue_m={}\ninitial_flux_perc_m={perc}\n", csv(&nash[..nn]), csv(&queue[..ng])))
}

/// Descriptor-absent v1 packets retain legacy decoding. A present malformed
/// descriptor never falls back to fractions and silently loses carried water.
pub fn read(dir: &str, man: &Value, ids: &[String], expected: &[cfe::Params], expected_inis: &[String]) -> Result<Option<Vec<String>>> {
    let Some(raw) = man.get("cfe_raw_initial") else { return Ok(None); };
    let seal = &raw["parameters"];
    if raw["schema"] != SCHEMA || seal["file"] != FILE || seal["count"].as_u64() != Some(ids.len() as u64) {
        return Err("CFE raw state: malformed parameter descriptor/schema/count".into());
    }
    let bytes = std::fs::read(Path::new(dir).join(FILE))?;
    if seal["sha256"].as_str() != Some(packet::sha256_hex(&bytes).as_str()) {
        return Err("CFE raw state: parameter payload SHA mismatch".into());
    }
    let cols: Vec<Column> = serde_json::from_slice(&bytes)?;
    if cols.len() != ids.len() || expected.len() != ids.len() || expected_inis.len() != ids.len() || cols.iter().zip(ids).any(|(c, id)| &c.id != id) {
        return Err("CFE raw state: ordered parameter IDs differ from fabric/state".into());
    }
    // Do not let a key unknown to the scoped spin-up parser silently change a
    // formulation in the native packet consumer. Only initial fractions vary.
    let static_lines = |ini: &str| ini.lines().filter(|l| !l.starts_with("soil_storage=") && !l.starts_with("gw_storage=")).map(str::to_owned).collect::<Vec<_>>();
    for (col, ini) in cols.iter().zip(expected_inis) {
        if static_lines(&col.ini) != static_lines(ini) {
            return Err(format!("{}: spin-up INI static parameters/formulation differ from fabric", col.id).into());
        }
    }
    let nn = expected.iter().map(|p| p.n_nash_subsurface).max().unwrap_or(0);
    let ng = expected.iter().map(|p| p.giuh.len()).max().unwrap_or(0);
    let soil = read_engine_array(dir, "cfe_soil_storage_m", &[ids.len()], "m")?;
    let gw = read_engine_array(dir, "cfe_gw_storage_m", &[ids.len()], "m")?;
    let nash = read_engine_array(dir, "cfe_nash_storage_subsurface", &[ids.len(), nn], "m")?;
    let queue = read_engine_array(dir, "cfe_giuh_queue", &[ids.len(), ng], "m")?;
    let perc = read_engine_array(dir, "cfe_flux_perc_carry", &[ids.len()], "m")?;
    (0..ids.len()).into_par_iter().map(|k| {
        initial_ini(&cols[k].ini, &expected[k], soil[k], gw[k], &nash[k*nn..(k+1)*nn], &queue[k*ng..(k+1)*ng], perc[k])
            .map_err(|e| format!("{}: {e}", ids[k]).into())
    }).collect::<Result<Vec<_>>>().map(Some)
}
