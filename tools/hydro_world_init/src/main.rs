//! rw_hydro_init: initial water state for gpuwm hydro on any HYDRO-WORLD fabric.
//!
//! Commands (each takes one REQUEST.json, prints a JSON receipt on stdout and
//! keeps it in its output directory):
//!   tessellate  proof fixture: equal-area squares over a lon/lat box (not hydrography)
//!   bbox        lon/lat box of a fabric's divides
//!   weights     catchment area weights on a source grid (all cores)
//!   state       source state at one time -> per-catchment initial state packet
//!   spinup      CFE spin-up from a state packet over the row's forcing (all cores)
//!   forcing-gaps  records of fetched source files holding non-finite values (all cores)
//!   forcing-fill  patched copies of those files from the row's fill source (all cores)
//!   cfe-trace   one CFE column over a forcing table, every state word in hex
//!   engine-packets  CFE INIs + nextgen-cfe-prepare request and arwen-noahowp-init request
mod cfe;
mod channel;
mod forcing;
mod gapfill;
mod geom;
mod grid;
mod ncread;
mod packet;
mod proj;
mod raw_cfe;
mod state;
mod table;
mod tessellate;
mod time;

use packet::{new_dir, write_arrays, write_json, Array};
use rayon::prelude::*;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::time::Instant;

pub type Error = Box<dyn std::error::Error + Send + Sync>;
pub type Result<T> = std::result::Result<T, Error>;

fn s<'a>(v: &'a Value, k: &str) -> Result<&'a str> {
    v[k].as_str().ok_or_else(|| format!("request lacks string field {k}").into())
}
fn f(v: &Value, k: &str) -> Result<f64> {
    v[k].as_f64().ok_or_else(|| format!("request lacks number field {k}").into())
}

struct Clock {
    t0: Instant,
    last: Instant,
    stages: Vec<Value>,
}
impl Clock {
    fn new() -> Self {
        let n = Instant::now();
        Clock { t0: n, last: n, stages: vec![] }
    }
    fn stage(&mut self, name: &str, parallelism: &str) {
        let n = Instant::now();
        self.stages.push(json!({"stage": name, "seconds": (n - self.last).as_secs_f64(), "parallelism": parallelism}));
        self.last = n;
    }
    fn done(&self) -> Value {
        json!({"stages": self.stages, "total_seconds": self.t0.elapsed().as_secs_f64()})
    }
}

fn cores() -> String {
    format!("{} threads (rayon, all cores)", rayon::current_num_threads())
}

fn read_fabric(path: &str) -> Result<(hydro_world::contract::Fabric, String)> {
    // The file hash and the table read run side by side.
    let (fab, sha) = rayon::join(
        || hydro_world::gpkg::read(path).map(|(fab, _report)| fab),
        || std::fs::read(path).map(|bytes| packet::sha256_hex(&bytes)).map_err(|e| format!("{path}: {e}")),
    );
    Ok((fab?, sha?))
}

fn cmd_tessellate(r: &Value) -> Result<Value> {
    let mut c = Clock::new();
    let fab = tessellate::build(s(r, "name")?, f(r, "south")?, f(r, "west")?, f(r, "north")?, f(r, "east")?, f(r, "cell_m")?)?;
    let out = s(r, "output")?;
    hydro_world::gpkg::write(&fab, out)?;
    c.stage("tessellate+write", "1 thread (fixture generation, < 1 s)");
    Ok(json!({"schema": "arwen.hydro-world.init-tessellate.v1", "output": out, "divides": fab.divides.len(),
              "crs": fab.srs.srs_name, "note": "proof fixture: equal-area squares, not hydrography", "timing": c.done()}))
}

fn cmd_bbox(r: &Value) -> Result<Value> {
    let (fab, _) = read_fabric(s(r, "fabric")?)?;
    let crs = proj::fabric_crs(&fab.srs);
    let d = grid::read_divides(&fab)?;
    let pad = r["pad_degrees"].as_f64().unwrap_or(0.5);
    let chunk = d.polys.len().div_ceil(4 * rayon::current_num_threads()).max(1);
    let ext: Vec<Result<[f64; 4]>> = d
        .polys
        .par_chunks(chunk)
        .map(|part| {
            let t = proj::Transform::new(&crs, "EPSG:4326")?;
            let mut b = [f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY];
            for p in part.iter().flatten().flatten().flatten() {
                let q = t.apply(p[0], p[1])?;
                b = [b[0].min(q[1]), b[1].min(q[0]), b[2].max(q[1]), b[3].max(q[0])];
            }
            Ok(b)
        })
        .collect();
    let mut b = [f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY];
    for e in ext {
        let e = e?;
        b = [b[0].min(e[0]), b[1].min(e[1]), b[2].max(e[2]), b[3].max(e[3])];
    }
    Ok(json!({"area_south_west_north_east": [b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad], "divides": d.ids.len()}))
}

fn cmd_weights(r: &Value) -> Result<Value> {
    let mut c = Clock::new();
    let tbl = table::load(s(r, "table")?)?;
    let row = tbl.rows.get(s(r, "row")?).ok_or("unknown row")?;
    let fabric_path = s(r, "fabric")?;
    let (fab, sha) = read_fabric(fabric_path)?;
    let divides = grid::read_divides(&fab)?;
    c.stage("read fabric", "1 thread (SQLite read)");
    let src = ncread::open(s(r, "grid_file")?)?;
    let g = ncread::grid(&src, row)?;
    c.stage("read grid", "1 thread (coordinates only)");
    let crs = proj::fabric_crs(&fab.srs);
    let densify = r["densify_m"].as_f64().unwrap_or(1000.0);
    let (mut rows, threads) = grid::compute(&divides, &crs, &g, densify)?;
    c.stage("intersect divides with cells", &cores());
    // Optional valid-cell mask (valid_from: {file, variable, record}): a source
    // that leaves cells missing (ERA5-Land over sea) keeps only overlaps with
    // cells holding a finite value; weights are renormalised over them. A
    // catchment with no valid overlap takes the nearest valid cell centre (in
    // the grid's coordinates) and is counted in the receipt.
    let mut mask_receipt = Value::Null;
    if let Some(v) = r.get("valid_from").filter(|v| !v.is_null()) {
        let msrc = ncread::open(s(v, "file")?)?;
        let mg = ncread::grid(&msrc, row)?;
        if mg.x != g.x || mg.y != g.y {
            return Err("valid_from grid differs from the weight grid".into());
        }
        let values = ncread::field(&msrc, s(v, "variable")?, v["record"].as_u64().unwrap_or(0) as usize)?;
        let valid: Vec<bool> = values.iter().map(|x| x.is_finite()).collect();
        let valid_cells: Vec<usize> = (0..valid.len()).filter(|&i| valid[i]).collect();
        if valid_cells.is_empty() {
            return Err("valid_from field has no finite cell".into());
        }
        let nx = g.nx();
        let fallbacks: Vec<usize> = rows
            .par_iter_mut()
            .enumerate()
            .filter_map(|(k, (idx, wt, ov, cen))| {
                let keep: Vec<usize> = (0..idx.len()).filter(|&j| valid[idx[j] as usize]).collect();
                if keep.is_empty() {
                    let d2 = |cell: usize| {
                        let (yy, xx) = (cell / nx, cell % nx);
                        (g.x[xx] - cen[0]).powi(2) + (g.y[yy] - cen[1]).powi(2)
                    };
                    let best = *valid_cells.iter().min_by(|a, b| d2(**a).total_cmp(&d2(**b))).unwrap();
                    let total: f64 = ov.iter().sum();
                    *idx = vec![best as u32];
                    *wt = vec![1.0];
                    *ov = vec![total];
                    return Some(k);
                }
                let total: f64 = keep.iter().map(|&j| ov[j]).sum();
                *idx = keep.iter().map(|&j| idx[j]).collect();
                *ov = keep.iter().map(|&j| ov[j]).collect();
                *wt = ov.iter().map(|a| a / total).collect();
                None
            })
            .collect();
        mask_receipt = json!({"file": s(v, "file")?, "variable": s(v, "variable")?, "valid_cells": valid_cells.len(),
            "cells": valid.len(), "nearest_valid_fallback_catchments": fallbacks.len(),
            "fallback_ids": fallbacks.iter().take(50).map(|&k| divides.ids[k].clone()).collect::<Vec<_>>()});
        c.stage("valid-cell mask", &cores());
    }
    let grid_shape = vec![g.ny(), g.nx()];
    let mut w = grid::Weights {
        schema: grid::WEIGHTS_SCHEMA.into(),
        fabric: fabric_path.into(),
        fabric_sha256: sha,
        grid: g,
        catchment_ids: divides.ids.clone(),
        grid_shape,
        offsets: vec![0],
        indices: vec![],
        weights: vec![],
        overlap_m2: vec![],
        catchment_area_m2: divides.area_m2.clone(),
        coverage_fraction: vec![],
        centroid: vec![],
    };
    for (k, (idx, wt, ov, cen)) in rows.into_iter().enumerate() {
        let total: f64 = ov.iter().sum();
        w.coverage_fraction.push(total / divides.area_m2[k]);
        w.indices.extend(idx);
        w.weights.extend(wt);
        w.overlap_m2.extend(ov);
        w.offsets.push(w.indices.len() as u64);
        w.centroid.push(cen);
    }
    let dir = new_dir(s(r, "output")?)?;
    let wsha = write_json(&dir.join("weights.json"), &serde_json::to_value(&w)?)?;
    c.stage("write", "1 thread");
    let cov = state::stat(w.coverage_fraction.iter().copied());
    let receipt = json!({"schema": "arwen.hydro-world.init-weights-receipt.v1", "fabric": fabric_path, "fabric_sha256": w.fabric_sha256,
        "fabric_crs": fab.srs.srs_name, "grid": {"kind": w.grid.kind, "ny": w.grid.ny(), "nx": w.grid.nx()},
        "catchments": w.catchment_ids.len(), "edges": w.indices.len(), "coverage_fraction": cov, "valid_from": mask_receipt,
        "weights_sha256": wsha, "threads": threads, "timing": c.done()});
    write_json(&dir.join("receipt.json"), &receipt)?;
    Ok(receipt)
}

fn load_weights(dir: &str) -> Result<grid::Weights> {
    Ok(serde_json::from_str(&std::fs::read_to_string(std::path::Path::new(dir).join("weights.json"))?)?)
}

fn cmd_state(r: &Value) -> Result<Value> {
    let mut c = Clock::new();
    let tbl = table::load(s(r, "table")?)?;
    let row_name = s(r, "row")?;
    let row = table::admit(&tbl, row_name)?;
    let (fab, fsha) = read_fabric(s(r, "fabric")?)?;
    let w = load_weights(s(r, "weights")?)?;
    if w.fabric_sha256 != fsha {
        return Err("weights were built for a different fabric file".into());
    }
    let (params, _inis, pinfo) = packet::params_for(&fab, &w.catchment_ids, r["params"].as_str())?;
    c.stage("read fabric, weights, parameters", "1 thread");
    let src = ncread::open(s(r, "source_file")?)?;
    let valid = s(r, "valid_time")?;
    let rec = ncread::record_index(&src, time::parse(valid)?)?;
    let partial = r["partial_coverage"].as_bool().unwrap_or(false);
    let m = state::map_state_cover(&tbl, row, &src, rec, &w, &params, partial)?;
    c.stage("map source state onto catchments", &cores());
    let n = m.states.len();
    let nl = tbl.target.noah_soil_layer_bounds_m.len() - 1;
    let ks = row.state.soil_water.layers.len();
    let flat = |g: &dyn Fn(&state::CatchmentState) -> Vec<f64>| m.states.iter().flat_map(|x| g(x)).collect::<Vec<f64>>();
    let one = |g: &dyn Fn(&state::CatchmentState) -> f64| m.states.iter().map(|x| g(x)).collect::<Vec<f64>>();
    let arrays = vec![
        Array { name: "soil_moisture_noah".into(), shape: vec![n, nl], data: flat(&|x| x.smc.clone()), units: "m3 m-3".into(), meaning: "total soil water per Noah layer (0-0.1, 0.1-0.4, 0.4-1.0, 1.0-2.0 m)".into() },
        Array { name: "soil_liquid_noah".into(), shape: vec![n, nl], data: flat(&|x| x.sh2o.clone()), units: "m3 m-3".into(), meaning: "liquid soil water per Noah layer (Noah-OWP sh2o)".into() },
        Array { name: "soil_ice_noah".into(), shape: vec![n, nl], data: flat(&|x| x.sice.clone()), units: "m3 m-3".into(), meaning: "frozen soil water per Noah layer (Noah-OWP sice)".into() },
        Array { name: "soil_temperature_noah".into(), shape: vec![n, nl], data: flat(&|x| x.stc.clone()), units: "K".into(), meaning: "soil temperature per Noah layer".into() },
        Array { name: "soil_quantity_source_layers".into(), shape: vec![n, ks], data: flat(&|x| x.soil_moisture_source.clone()), units: "row transfer quantity".into(), meaning: "catchment mean on the source layers: relative wetness (wetness_index), 0 wilting / 1 field capacity / 2 saturation (field_capacity_anchored), or m3 m-3 (volumetric)".into() },
        Array { name: "snow_water_equivalent".into(), shape: vec![n], data: one(&|x| x.swe_m), units: "m".into(), meaning: "snow water equivalent".into() },
        Array { name: "snow_depth".into(), shape: vec![n], data: one(&|x| x.snow_depth_m), units: "m".into(), meaning: "snow depth".into() },
        Array { name: "snow_temperature".into(), shape: vec![n], data: one(&|x| x.snow_temperature_k), units: "K".into(), meaning: "snow temperature (top soil layer, capped at freezing, where there is no snow)".into() },
        Array { name: "cfe_soil_storage".into(), shape: vec![n], data: one(&|x| x.cfe_soil_storage), units: "1".into(), meaning: "CFE soil_storage, fraction of smcmax*depth".into() },
        Array { name: "cfe_gw_storage".into(), shape: vec![n], data: one(&|x| x.cfe_gw_storage), units: "1".into(), meaning: "CFE gw_storage, fraction of max_gw_storage (cold start unless spun up)".into() },
    ];
    // bounds
    let mut bad = BTreeMap::<&str, usize>::new();
    for (k, x) in m.states.iter().enumerate() {
        let p = &params[k];
        if partial && !x.cfe_soil_storage.is_finite() {
            continue; // not covered by this source: NaN by design, counted in the map
        }
        for j in 0..nl {
            let ok = x.smc[j].is_finite() && x.smc[j] >= 0.0 && x.smc[j] <= p.smcmax + 1e-12;
            *bad.entry("soil_moisture_outside_0_smcmax").or_default() += usize::from(!ok);
            *bad.entry("liquid_or_ice_negative").or_default() += usize::from(x.sh2o[j] < 0.0 || x.sice[j] < 0.0);
            *bad.entry("liquid_plus_ice_not_total").or_default() += usize::from((x.sh2o[j] + x.sice[j] - x.smc[j]).abs() > 1e-12);
            *bad.entry("soil_temperature_outside_200_330K").or_default() += usize::from(!(x.stc[j] > 200.0 && x.stc[j] < 330.0));
        }
        *bad.entry("snow_negative_or_nonfinite").or_default() += usize::from(!(x.swe_m >= 0.0 && x.snow_depth_m >= 0.0));
        *bad.entry("snow_temperature_outside_200K_freezing").or_default() += usize::from(!(x.snow_temperature_k > 200.0 && x.snow_temperature_k <= tbl.target.freezing_point_k + 1e-9));
        *bad.entry("cfe_fraction_outside_0_1").or_default() += usize::from(!((0.0..=1.0).contains(&x.cfe_soil_storage) && (0.0..=1.0).contains(&x.cfe_gw_storage)));
    }
    let violations: usize = bad.values().sum();
    let dir = new_dir(s(r, "output")?)?;
    let entries = write_arrays(dir, &arrays)?;
    let cfe_init: serde_json::Map<String, Value> = w.catchment_ids.iter().zip(&m.states).map(|(id, x)| (id.clone(), json!({"soil_storage": x.cfe_soil_storage, "gw_storage": x.cfe_gw_storage}))).collect();
    let noah: serde_json::Map<String, Value> = w.catchment_ids.iter().zip(&m.states).map(|(id, x)| (id.clone(), json!({"sh2o": x.sh2o, "sice": x.sice}))).collect();
    write_json(&dir.join("cfe-initial.json"), &Value::Object(cfe_init))?;
    write_json(&dir.join("noahowp-overrides.json"), &Value::Object(noah))?;
    let stats: serde_json::Map<String, Value> = arrays.iter().map(|a| (a.name.clone(), serde_json::to_value(state::stat(a.data.iter().copied())).unwrap())).collect();
    let layer_stats: Vec<Value> = (0..nl).map(|j| json!({"layer": j, "smc": state::stat(m.states.iter().map(|x| x.smc[j])), "sice": state::stat(m.states.iter().map(|x| x.sice[j])), "stc": state::stat(m.states.iter().map(|x| x.stc[j]))})).collect();
    let manifest = json!({"schema": "arwen.hydro-world.init-state.v1", "row": row_name, "licence": row.licence.spdx, "attribution": row.licence.attribution,
        "valid_time": valid, "source_file": s(r, "source_file")?, "fabric": s(r, "fabric")?, "fabric_sha256": fsha,
        "catchment_ids": w.catchment_ids, "noah_soil_layer_bounds_m": tbl.target.noah_soil_layer_bounds_m, "parameters": pinfo, "arrays": entries,
        "engine_handoff": {"cfe": "cfe-initial.json gives soil_storage/gw_storage per catchment for hydro_world::derive::cfe_ini (CfeInitial) -> nextgen-cfe-prepare",
                           "noahowp": "noahowp-overrides.json gives sh2o/sice per catchment as arwen-noahowp-init overrides (namelist keys)",
                           "thermal_and_snow_state": "engine-packets binds soil_temperature_noah and aggregate SWE/depth/temperature to arwen-noahowp-init; snow recipe aggregate-all-ice-divide-v1"}});
    c.stage("write packet", "rayon over arrays");
    let msha = write_json(&dir.join("manifest.json"), &manifest)?;
    let receipt = json!({"schema": "arwen.hydro-world.init-state-receipt.v1", "row": row_name, "valid_time": valid, "catchments": n,
        "counts": m.counts, "bounds_violations": violations, "bounds_checks": bad, "stats": stats, "noah_layers": layer_stats,
        "manifest_sha256": msha, "timing": c.done()});
    write_json(&dir.join("receipt.json"), &receipt)?;
    Ok(receipt)
}

fn read_state_packet(dir: &str, name: &str) -> Result<Vec<f64>> {
    let p = std::path::Path::new(dir);
    let man: Value = serde_json::from_str(&std::fs::read_to_string(p.join("manifest.json"))?)?;
    let e = man["arrays"].as_array().and_then(|a| a.iter().find(|e| e["name"] == name)).ok_or_else(|| format!("{dir}: no array {name}"))?;
    let b = std::fs::read(p.join(e["file"].as_str().unwrap()))?;
    if packet::sha256_hex(&b) != e["sha256"].as_str().unwrap_or("") {
        return Err(format!("{dir}/{name}: payload hash differs from its manifest").into());
    }
    Ok(b.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect())
}

// Engine transport cannot interpret a transposed array or another unit as native state.
fn read_engine_array(dir: &str, name: &str, shape: &[usize], units: &str) -> Result<Vec<f64>> {
    let man: Value = serde_json::from_str(&std::fs::read_to_string(std::path::Path::new(dir).join("manifest.json"))?)?;
    let a = man["arrays"].as_array().and_then(|a| a.iter().find(|a| a["name"] == name)).ok_or_else(|| format!("{dir}: no array {name}"))?;
    if a["dtype"] != "<f8" || a["shape"] != json!(shape) || a["units"] != units
        || a["bytes"].as_u64() != Some((shape.iter().product::<usize>() * 8) as u64) {
        return Err(format!("{dir}/{name}: layout or units differ from the engine-state contract").into());
    }
    let file = a["file"].as_str().ok_or_else(|| format!("{dir}/{name}: array has no payload file"))?;
    let bytes = std::fs::read(std::path::Path::new(dir).join(file))?;
    if bytes.len() != shape.iter().product::<usize>() * 8
        || packet::sha256_hex(&bytes) != a["sha256"].as_str().unwrap_or("") {
        return Err(format!("{dir}/{name}: payload bytes/hash differ from the engine-state contract").into());
    }
    let values: Vec<f64> = bytes.chunks_exact(8).map(|b| f64::from_le_bytes(b.try_into().unwrap())).collect();
    if values.len() != shape.iter().product::<usize>() || values.iter().any(|x| !x.is_finite()) {
        return Err(format!("{dir}/{name}: nonfinite state or payload extent differs from its declared layout").into());
    }
    Ok(values)
}

fn cmd_spinup(r: &Value) -> Result<Value> {
    let mut c = Clock::new();
    let tbl = table::load(s(r, "table")?)?;
    let row_name = s(r, "row")?;
    let row = table::admit(&tbl, row_name)?;
    let (fab, fsha) = read_fabric(s(r, "fabric")?)?;
    c.stage("read fabric", "table read beside the file hash (2 tasks)");
    let w = load_weights(s(r, "weights")?)?;
    if w.fabric_sha256 != fsha {
        return Err("weights were built for a different fabric file".into());
    }
    c.stage("read weights", "1 thread");
    let init_dir = s(r, "initial_state")?;
    let mut soil0 = read_state_packet(init_dir, "cfe_soil_storage")?;
    let mut gw0 = read_state_packet(init_dir, "cfe_gw_storage")?;
    // Convergence check arm: start every catchment from given fractions instead.
    if let Some(o) = r["initial_override"].as_object() {
        if let Some(x) = o.get("soil_storage").and_then(Value::as_f64) {
            soil0.iter_mut().for_each(|v| *v = x);
        }
        if let Some(x) = o.get("gw_storage").and_then(Value::as_f64) {
            gw0.iter_mut().for_each(|v| *v = x);
        }
    }
    let (mut params, inis, pinfo) = packet::params_for(&fab, &w.catchment_ids, r["params"].as_str())?;
    for (k, p) in params.iter_mut().enumerate() {
        p.soil_storage = soil0[k];
        p.gw_storage = gw0[k];
    }
    // The INI each catchment runs with: the engine packet text with the
    // initial storages written in shortest round-trip form (same f64 on parse).
    let ini_for = |k: usize| -> String {
        inis[k]
            .lines()
            .map(|l| {
                if l.starts_with("soil_storage=") {
                    format!("soil_storage={}[m/m]", soil0[k])
                } else if l.starts_with("gw_storage=") {
                    format!("gw_storage={}[m/m]", gw0[k])
                } else {
                    l.to_string()
                }
            })
            .collect::<Vec<_>>()
            .join("\n")
            + "\n"
    };
    // Every catchment's INI round trip, checked on all cores.
    params.par_iter().enumerate().try_for_each(|(k, p)| -> Result<()> {
        if cfe::parse_ini(&ini_for(k))? != *p {
            return Err(format!("{}: INI text does not reproduce the spin-up parameters", w.catchment_ids[k]).into());
        }
        Ok(())
    })?;
    c.stage("initial state, parameters", "reads 1 thread; INI round-trip check on all cores");
    // land weights from the state file's land fraction, if the row has one
    let land = match row.state.land_fraction.as_ref().and_then(|v| v.get("variable")).and_then(|v| v.as_str()) {
        Some(var) => {
            let lf = r["land_file"].as_str().ok_or("spinup needs land_file (the state file) when the row has a land fraction")?;
            let ls = ncread::open(lf)?;
            Some(ncread::field(&ls, var, 0)?)
        }
        None => None,
    };
    let lw = state::land_weights(&w, land.as_deref())?;
    let files: Vec<String> = r["forcing_files"].as_array().ok_or("forcing_files")?.iter().map(|v| v.as_str().unwrap_or("").to_string()).collect();
    let start = time::parse(s(r, "start")?)?;
    let hours = r["hours"].as_u64().ok_or("hours")? as usize;
    let days = hours / 24;
    struct Out {
        cfe: cfe::Cfe,
        s0: f64,
        inp: f64,
        out_et: f64,
        runoff: Vec<f64>,
        daily: Vec<[f64; 6]>,
        resid: f64,
        q_last_day_m_h: f64,
        totals: [f64; 3],
    }
    // Catchments traced for the native comparison: their exact per-hour inputs.
    let trace_k: Vec<usize> = r["trace"].as_array().cloned().unwrap_or_default().iter()
        .map(|v| v.as_u64().map(|k| k as usize).ok_or("trace entries are catchment indices"))
        .collect::<std::result::Result<_, _>>()?;
    let mut trace_csv: Vec<String> = vec![String::new(); trace_k.len()];
    let mut outs: Vec<Out> = params
        .par_iter()
        .map(|p| {
            let m = cfe::Cfe::new(p);
            let s0 = m.storage();
            Out { cfe: m, s0, inp: 0.0, out_et: 0.0, runoff: Vec::with_capacity(hours), daily: vec![[0.0f64; 6]; days.max(1)],
                  resid: 0.0, q_last_day_m_h: 0.0, totals: [0.0; 3] }
        })
        .collect();
    // Forcing and CFE advance together in 24 h chunks, so only the files a
    // chunk reads are resident (forcing::ForcingStream). Each catchment takes
    // the same inputs in the same order as the whole-series loop did.
    let mut stream = forcing::ForcingStream::open(row, &files, &lw, start)?;
    // A carried spin-up (carry_to: the forecast starts past the source's
    // record) ends at the last complete day: a preliminary stream (ERA5T)
    // can publish an hour before every variable is filled (potential
    // evaporation was NaN from 2026-10-10 while the store declared 10-15).
    let carrying = r["carry_to"].as_str().is_some();
    let mut truncated = Value::Null;
    let mut hours = hours;
    for h0 in (0..hours).step_by(24) {
        let h1 = (h0 + 24).min(hours);
        let (water_m, pet_m) = match stream.chunk(h0, h1) {
            Ok(chunk) => chunk,
            Err(e) if carrying && h0 > 0 && e.to_string().contains(forcing::MISSING) => {
                truncated = json!({"requested_hours": hours, "ran_hours": h0, "first_incomplete": time::iso(start + 3600 * h0 as i64),
                                   "reason": e.to_string()});
                hours = h0;
                break;
            }
            Err(e) => return Err(e),
        };
        outs.par_iter_mut().enumerate().for_each(|(k, o)| {
            let m = &mut o.cfe;
            for h in h0..h1 {
                let precip_mm_h = water_m[h - h0][k] * 1000.0;
                let pet_m_s = pet_m[h - h0][k] / 3600.0;
                m.update(precip_mm_h, pet_m_s);
                let rain = precip_mm_h / 1000.0;
                o.inp += rain;
                o.out_et += m.actual_et_m;
                o.totals[2] += m.qout_m;
                o.runoff.push(m.qout_m);
                let d = (h / 24).min(o.daily.len() - 1);
                o.daily[d][0] += m.soil.storage_m / m.soil.storage_max_m / 24.0;
                o.daily[d][1] += m.gw.storage_m / m.gw.storage_max_m / 24.0;
                o.daily[d][2] += m.storage() / 24.0;
                o.daily[d][3] += m.qout_m;
                o.daily[d][4] += m.actual_et_m;
                o.daily[d][5] += rain;
                if h + 24 >= hours {
                    o.q_last_day_m_h += m.qout_m / 24.0;
                }
            }
        });
        for (slot, &k) in trace_csv.iter_mut().zip(&trace_k) {
            if k < params.len() {
                for h in h0..h1 {
                    slot.push_str(&format!("{:e},{:e}\n", water_m[h - h0][k] * 1000.0, pet_m[h - h0][k] / 3600.0));
                }
            }
        }
    }
    if !truncated.is_null() {
        // The last-day mean is over the last 24 hours actually run.
        outs.par_iter_mut().for_each(|o| {
            let tail = &o.runoff[o.runoff.len().saturating_sub(24)..];
            o.q_last_day_m_h = tail.iter().sum::<f64>() / tail.len().max(1) as f64;
        });
    }
    let days = hours / 24;
    outs.par_iter_mut().for_each(|o| {
        let out_q = o.totals[2];
        o.resid = o.inp - o.out_et - out_q - (o.cfe.storage() - o.s0);
        o.totals = [o.inp, o.out_et, out_q];
    });
    c.stage("read forcing + CFE spin-up, streamed in 24 h chunks",
            &format!("forcing: files, records and catchments in parallel (rayon); CFE: {} (one catchment per task)", cores()));
    // Live analysis soil at the forecast start (soil_override: a state packet
    // for this fabric whose cfe_soil_storage is NaN where the analysis does
    // not cover a catchment). Groundwater, Nash, GIUH and the channels keep
    // the spun-up state. The mass-balance residual above is the spin-up's own.
    let mut soil_override = Value::Null;
    if let Some(dir) = r["soil_override"].as_str() {
        let man: Value = serde_json::from_str(&std::fs::read_to_string(std::path::Path::new(dir).join("manifest.json"))?)?;
        if man["fabric_sha256"] != fsha {
            return Err(format!("{dir}: soil override was made for another fabric").into());
        }
        if man["catchment_ids"] != json!(w.catchment_ids) {
            return Err(format!("{dir}: soil override catchment order differs from the spin-up").into());
        }
        let frac = read_state_packet(dir, "cfe_soil_storage")?;
        if frac.len() != outs.len() {
            return Err(format!("{dir}: soil override has {} catchments, spin-up {}", frac.len(), outs.len()).into());
        }
        let changes: Vec<Option<f64>> = outs.par_iter_mut().zip(&frac).map(|(o, f)| {
            if !(f.is_finite() && (0.0..=1.0).contains(f)) {
                return None;
            }
            let before = o.cfe.soil.storage_m;
            o.cfe.soil.storage_m = f * o.cfe.soil.storage_max_m;
            Some(o.cfe.soil.storage_m - before)
        }).collect();
        let applied: Vec<f64> = changes.iter().flatten().copied().collect();
        soil_override = json!({"state_packet": dir, "valid_time": man["valid_time"], "row": man["row"],
            "catchments_overridden": applied.len(), "catchments_kept_spun_up": outs.len() - applied.len(),
            "soil_storage_change_m": state::stat(applied.iter().copied())});
        c.stage("live analysis soil override", &cores());
    }
    let n = outs.len();
    let finite = outs.iter().all(|o| o.cfe.storage().is_finite() && o.daily.iter().flatten().all(|x| x.is_finite()));
    let inside = outs.iter().all(|o| {
        o.cfe.soil.storage_m >= 0.0 && o.cfe.soil.storage_m <= o.cfe.soil.storage_max_m + 1e-12 && o.cfe.gw.storage_m >= 0.0 && o.cfe.gw.storage_m <= o.cfe.gw.storage_max_m + 1e-12
    });
    let max_resid = outs.iter().map(|o| o.resid.abs()).fold(0.0, f64::max);
    let overflow: u64 = outs.iter().map(|o| o.cfe.perc_overflow_events).sum();
    // domain daily series (area-weighted by catchment area)
    let area: f64 = w.catchment_area_m2.iter().sum();
    let daily_domain: Vec<Value> = (0..days.max(1))
        .map(|d| {
            let mut a = [0.0; 6];
            for (k, o) in outs.iter().enumerate() {
                for i in 0..6 {
                    a[i] += o.daily[d][i] * w.catchment_area_m2[k] / area;
                }
            }
            json!({"day": d + 1, "date_end": time::iso(start + 86_400 * (d as i64 + 1)), "soil_storage_fraction": a[0], "gw_storage_fraction": a[1],
                   "total_storage_m": a[2], "runoff_m": a[3], "aet_m": a[4], "water_input_m": a[5]})
        })
        .collect();
    // drift over the last 7 days, per catchment, of daily mean total storage relative to that week's water input
    let last = days.saturating_sub(1);
    let first_of_week = days.saturating_sub(7);
    let drift: Vec<f64> = outs.iter().map(|o| (o.daily[last][2] - o.daily[first_of_week][2]) * 1000.0).collect();
    let mut sorted = drift.iter().map(|x| x.abs()).collect::<Vec<_>>();
    sorted.sort_by(f64::total_cmp);
    let p95 = sorted.get(((sorted.len() as f64) * 0.95) as usize).copied().unwrap_or(0.0);
    let gw_drift: Vec<f64> = outs.iter().map(|o| o.daily[last][1] - o.daily[first_of_week][1]).collect();
    // A replay seed is tied to the first forcing hour, not the final day's
    // equilibrium. The coupled GPU initializer then routes EVERY hour through
    // MC and level-pool reservoirs before accepting the forecast initial state.
    let continuous = r["retain_runoff_replay"].as_bool().unwrap_or(false);
    let lateral: BTreeMap<String, f64> = w.catchment_ids.iter().zip(&outs).zip(&w.catchment_area_m2).map(|((id, o), a)| (id.clone(), (if continuous { o.runoff[0] } else { o.q_last_day_m_h }) * a / 3600.0)).collect();
    let channel = if fab.flowpaths.is_empty() { None } else { Some(channel::steady_state(&fab, &lateral)?) };
    c.stage("channel steady state", "1 thread (topological accumulation, < 1 s)");
    let dir = new_dir(s(r, "output")?)?;
    // Native-comparison inputs for chosen catchments: the exact INI and the
    // exact per-hour BMI inputs (shortest round-trip text) the spin-up used.
    let mut traced = Vec::new();
    for (&k, csv) in trace_k.iter().zip(&trace_csv) {
        let id = w.catchment_ids.get(k).ok_or("trace index out of range")?;
        std::fs::write(dir.join(format!("trace-{id}.ini")), ini_for(k))?;
        std::fs::write(dir.join(format!("trace-{id}.csv")), csv)?;
        traced.push(id.clone());
    }
    let soil_frac: Vec<f64> = outs.iter().map(|o| o.cfe.soil.storage_m / o.cfe.soil.storage_max_m).collect();
    let gw_frac: Vec<f64> = outs.iter().map(|o| o.cfe.gw.storage_m / o.cfe.gw.storage_max_m).collect();
    let nn = outs.iter().map(|o| o.cfe.nash.len()).max().unwrap_or(0);
    let ng = outs.iter().map(|o| o.cfe.giuh.len()).max().unwrap_or(0);
    let mut arrays = vec![
        Array { name: "cfe_soil_storage_m".into(), shape: vec![n], data: outs.iter().map(|o| o.cfe.soil.storage_m).collect(), units: "m".into(), meaning: "raw CFE soil reservoir storage after spin-up, without a fraction round trip".into() },
        Array { name: "cfe_gw_storage_m".into(), shape: vec![n], data: outs.iter().map(|o| o.cfe.gw.storage_m).collect(), units: "m".into(), meaning: "raw CFE groundwater reservoir storage after spin-up, without a fraction round trip".into() },
        Array { name: "cfe_soil_storage".into(), shape: vec![n], data: soil_frac.clone(), units: "1".into(), meaning: "CFE soil_storage after spin-up, fraction of smcmax*depth".into() },
        Array { name: "cfe_gw_storage".into(), shape: vec![n], data: gw_frac.clone(), units: "1".into(), meaning: "CFE gw_storage after spin-up, fraction of max_gw_storage".into() },
        Array { name: "cfe_nash_storage_subsurface".into(), shape: vec![n, nn], data: outs.iter().flat_map(|o| { let mut v = o.cfe.nash.clone(); v.resize(nn, 0.0); v }).collect(), units: "m".into(), meaning: "lateral-flow Nash cascade storages".into() },
        Array { name: "cfe_giuh_queue".into(), shape: vec![n, ng], data: outs.iter().flat_map(|o| { let mut v = o.cfe.queue[..o.cfe.giuh.len()].to_vec(); v.resize(ng, 0.0); v }).collect(), units: "m".into(), meaning: "GIUH runoff queue".into() },
        Array { name: "cfe_flux_perc_carry".into(), shape: vec![n], data: outs.iter().map(|o| o.cfe.flux_perc_m).collect(), units: "m".into(), meaning: "percolation flux the next CFE step reads (C carries it between calls)".into() },
        Array { name: "runoff_last_day_mean".into(), shape: vec![n], data: outs.iter().map(|o| o.q_last_day_m_h).collect(), units: "m h-1".into(), meaning: "mean CFE Qout over the last 24 h of spin-up".into() },
        Array { name: "mass_balance_residual".into(), shape: vec![n], data: outs.iter().map(|o| o.resid).collect(), units: "m".into(), meaning: "input - AET - Qout - storage change over the spin-up".into() },
    ];
    if continuous {
        arrays.push(Array { name: "runoff_replay".into(), shape: vec![n, hours],
            data: outs.iter().flat_map(|o| o.runoff.iter().copied()).collect(),
            units: "m per hour".into(), meaning: "CFE Qout at every replay hour; catchment-major, chronological within each catchment".into() });
    }
    let routing_ids = if let Some(ch) = &channel {
        arrays.push(Array { name: "routing_state".into(), shape: vec![ch.state.len(), 3], data: ch.state.iter().flatten().copied().collect(), units: "m3 s-1, m s-1, m".into(), meaning: "steady-state (Q, velocity, depth) per flowpath, flowpaths table order (routing_ids.json)".into() });
        // Bind the ordered ID axis and the state array in the same manifest.
        // A state hash alone cannot detect values being joined to swapped IDs.
        let sha = write_json(&dir.join("routing_ids.json"), &json!(ch.ids))?;
        Some(json!({"file": "routing_ids.json", "sha256": sha, "count": ch.ids.len()}))
    } else {
        None
    };
    let entries = write_arrays(dir, &arrays)?;
    let cfe_init: serde_json::Map<String, Value> = w.catchment_ids.iter().zip(soil_frac.iter().zip(&gw_frac)).map(|(id, (s, g))| (id.clone(), json!({"soil_storage": s, "gw_storage": g}))).collect();
    write_json(&dir.join("cfe-initial.json"), &Value::Object(cfe_init))?;
    let raw_initial = raw_cfe::write(dir, w.catchment_ids.iter().enumerate()
        .map(|(k, id)| json!({"id": id, "ini": ini_for(k)})).collect())?;
    // carry_to: the forecast starts after the source's last hour. The state
    // spun up to that hour is the state at carry_to (held, persistence), and
    // the packet says so; routing binds it at the forecast origin.
    let spin_end = start + 3600 * hours as i64;
    let (end, shift, carry) = match r["carry_to"].as_str() {
        Some(to) => {
            let t = time::parse(to)?;
            if t < spin_end || (t - spin_end) % 3600 != 0 {
                return Err(format!("carry_to {to} must be whole hours at or after the spin-up end {}", time::iso(spin_end)).into());
            }
            (t, t - spin_end, json!({"method": "persistence: the state spun up to the source's last hour is held to the forecast start",
                "spinup_end": time::iso(spin_end), "carried_to": to, "hours": (t - spin_end) / 3600,
                "reason": r["carry_reason"].clone(), "spinup_truncated": truncated.clone()}))
        }
        None => (spin_end, 0, Value::Null),
    };
    let mut manifest = json!({"schema": "arwen.hydro-world.init-spinup.v1", "row": row_name, "licence": row.licence.spdx, "attribution": row.licence.attribution,
        "start": s(r, "start")?, "end": time::iso(end), "hours": hours, "fabric_sha256": fsha, "catchment_ids": w.catchment_ids,
        "parameters": pinfo, "initial_state": init_dir, "arrays": entries, "cfe_raw_initial": raw_initial,
        "cfe_port": "NOAA-OWP/cfe f9182dfb3d81c407c66c5b782114173893c30cf2 (Apache-2.0), CPU Rust port, Schaake + GIUH"});
    if let Some(ids) = routing_ids {
        manifest["routing_ids"] = ids;
    }
    if !carry.is_null() {
        manifest["carry_over"] = carry.clone();
    }
    if !soil_override.is_null() {
        manifest["soil_override"] = soil_override.clone();
    }
    if continuous {
        manifest["routing_replay"] = json!({"array": "runoff_replay", "start": time::iso(start + shift),
            "end": time::iso(end), "hours": hours, "interval_seconds": 3600,
            "seed": "steady first-hour flow/Manning channels; reservoir level-pool equilibrium at that first-hour flow; not the accepted final state"});
        if shift != 0 {
            manifest["routing_replay"]["source_window"] = json!({"start": s(r, "start")?, "end": time::iso(spin_end),
                "note": "the replay's hours are the spin-up's own; carry_over places their end at the forecast origin"});
        }
    }
    let msha = write_json(&dir.join("manifest.json"), &manifest)?;
    let tot = outs.iter().zip(&w.catchment_area_m2).fold([0.0; 3], |mut a, (o, ar)| {
        for i in 0..3 {
            a[i] += o.totals[i] * ar / area;
        }
        a
    });
    c.stage("write packet", "rayon over arrays");
    let receipt = json!({"schema": "arwen.hydro-world.init-spinup-receipt.v1", "row": row_name, "catchments": n, "hours": hours, "end": time::iso(end), "carry_over": carry, "soil_override": soil_override,
        "forcing_records_read": stream.records_read, "all_finite": finite, "storages_inside_bounds": inside,
        "max_abs_mass_balance_residual_m": max_resid, "percolation_overflow_branch_events": overflow,
        "domain_totals_m": {"water_input": tot[0], "aet": tot[1], "runoff": tot[2]},
        "last_week_total_storage_drift_mm": {"stat": state::stat(drift.iter().copied()), "p95_abs": p95},
        "last_week_gw_fraction_drift": state::stat(gw_drift.iter().copied()),
        "final": {"soil_storage_fraction": state::stat(soil_frac.iter().copied()), "gw_storage_fraction": state::stat(gw_frac.iter().copied())},
        "daily_domain": daily_domain,
        "channel": channel.as_ref().map(|ch| json!({"flowpaths": ch.ids.len(), "terminal": ch.terminal, "levels": ch.levels, "missing_geometry": ch.missing_geometry,
            "q_m3_s": state::stat(ch.state.iter().map(|x| x[0])), "depth_m": state::stat(ch.state.iter().map(|x| x[2]))})),
        "traced_catchments": traced, "initial_override": r["initial_override"], "manifest_sha256": msha, "timing": c.done()});
    write_json(&dir.join("receipt.json"), &receipt)?;
    Ok(receipt)
}

fn cmd_cfe_trace(r: &Value) -> Result<Value> {
    let ini = std::fs::read_to_string(s(r, "ini")?)?;
    let p = cfe::parse_ini(&ini)?;
    let mut m = cfe::Cfe::new(&p);
    let mut lines = String::new();
    let hex = |x: f64| format!("{:016x}", x.to_bits());
    for (i, line) in std::fs::read_to_string(s(r, "forcing_csv")?)?.lines().enumerate() {
        let mut it = line.split(',');
        let pr: f64 = it.next().ok_or("forcing line")?.trim().parse()?;
        let pet: f64 = it.next().ok_or("forcing line")?.trim().parse()?;
        m.update(pr, pet);
        let mut words = vec![hex(m.soil.storage_m), hex(m.gw.storage_m), hex(m.qout_m), hex(m.direct_runoff_m), hex(m.nash_lateral_m), hex(m.flux_from_deep_gw_m), hex(m.flux_perc_m), hex(m.flux_lat_m), hex(m.actual_et_m), hex(m.infiltration_excess_m)];
        for q in &m.queue {
            words.push(hex(*q));
        }
        for v in &m.nash {
            words.push(hex(*v));
        }
        lines.push_str(&format!("{i} {}\n", words.join(" ")));
    }
    std::fs::write(s(r, "output")?, &lines)?;
    Ok(json!({"steps": lines.lines().count(), "output": s(r, "output")?}))
}

/// Engine packets for a standalone run: one CFE INI per divide (fabric
/// model_attributes through hydro_world::derive::cfe_ini, with the spun-up
/// soil, groundwater and subsurface Nash storages) plus the nextgen-cfe-prepare
/// request, and the arwen-noahowp-init request (derive::noahowp_overrides plus
/// the state packet's sh2o/sice, four soil temperatures and aggregate snow). Request keys:
/// fabric, state (state-at-date dir), spinup (spinup dir), noah_namelist,
/// time_origin, output (new dir).
fn cmd_engine_packets(r: &Value) -> Result<Value> {
    use hydro_world::derive::{cfe_ini, noahowp_overrides, CfeInitial};
    let mut c = Clock::new();
    let (fab, fsha) = read_fabric(s(r, "fabric")?)?;
    let ids: Vec<String> = fab.divides.iter().map(|d| d.divide_id.clone().ok_or("divide without divide_id")).collect::<std::result::Result<_, _>>()?;
    let attrs = fab.model_attributes.as_ref().ok_or("fabric has no model_attributes (run hydro-world-params first)")?;
    let by_id: BTreeMap<&str, &hydro_world::contract::ModelAttributes> =
        attrs.iter().filter_map(|m| m.divide_id.as_deref().map(|d| (d, m))).collect();
    c.stage("read fabric", "1 thread (SQLite read)");
    let (state_dir, spin_dir) = (s(r, "state")?, s(r, "spinup")?);
    let mut layer_bounds = Vec::new();
    let mut spin_manifest = Value::Null;
    for (dir, epoch_key) in [(state_dir, "valid_time"), (spin_dir, "end")] {
        let man: Value = serde_json::from_str(&std::fs::read_to_string(std::path::Path::new(dir).join("manifest.json"))?)?;
        if man["fabric_sha256"] != fsha {
            return Err(format!("{dir}: packet was made for another fabric").into());
        }
        if man[epoch_key].as_str() != Some(s(r, "time_origin")?) {
            return Err(format!("{dir}: state epoch differs from the requested model start").into());
        }
        if dir == state_dir {
            layer_bounds = serde_json::from_value::<Vec<f64>>(man["noah_soil_layer_bounds_m"].clone())?;
            if layer_bounds.len() != 5 || layer_bounds[0] != 0.0 || layer_bounds.iter().any(|x| !x.is_finite())
                || layer_bounds.windows(2).any(|b| b[1] <= b[0]) {
                return Err("Noah state requires four ordered finite soil layers starting at zero depth".into());
            }
        }
        if dir == spin_dir { spin_manifest = man.clone(); }
        {
            let order = man["catchment_ids"].as_array().ok_or("state packet lacks catchment IDs; positional state could bind to the wrong divide")?;
            if order.len() != ids.len() || order.iter().zip(&ids).any(|(a, b)| a.as_str() != Some(b)) {
                return Err(format!("{dir}: catchment order differs from the fabric").into());
            }
        }
    }
    let (expected, expected_inis, _) = packet::params_for(&fab, &ids, None)?;
    let raw_inis = raw_cfe::read(spin_dir, &spin_manifest, &ids, &expected, &expected_inis)?;
    let legacy = if raw_inis.is_none() {
        Some((read_engine_array(spin_dir, "cfe_soil_storage", &[ids.len()], "1")?,
              read_engine_array(spin_dir, "cfe_gw_storage", &[ids.len()], "1")?,
              read_engine_array(spin_dir, "cfe_nash_storage_subsurface", &[ids.len(), 2], "m")?))
    } else { None };
    let sh2o = read_engine_array(state_dir, "soil_liquid_noah", &[ids.len(), 4], "m3 m-3")?;
    let sice = read_engine_array(state_dir, "soil_ice_noah", &[ids.len(), 4], "m3 m-3")?;
    let stc = read_engine_array(state_dir, "soil_temperature_noah", &[ids.len(), 4], "K")?;
    let snow_swe = read_engine_array(state_dir, "snow_water_equivalent", &[ids.len()], "m")?;
    let snow_depth = read_engine_array(state_dir, "snow_depth", &[ids.len()], "m")?;
    let snow_temp = read_engine_array(state_dir, "snow_temperature", &[ids.len()], "K")?;
    let n = ids.len();
    if stc.len() != 4 * n {
        return Err("state/spin-up array extents differ from the fabric".into());
    }
    c.stage("read state packets", "1 thread");
    let out = new_dir(s(r, "output")?)?;
    let ini_dir = out.join("cfe-ini");
    std::fs::create_dir(&ini_dir)?;
    let namelist = std::fs::canonicalize(s(r, "noah_namelist")?)?;
    // Optional explicit substitute for catchments whose majority cover is open
    // water: the Noah-OWP tables give water NROOT 0 and the resident allocation
    // runs NROOT 1..4 only, and the STAS WATER soil row is not a soil. Request:
    // "water_catchments": {"water_vegtyp":16,"water_isltyp":14,"vegtyp":7,"isltyp":6}.
    // Without it such catchments refuse in arwen-noahowp-init. Counted in the receipt.
    let water = r.get("water_catchments").filter(|v| !v.is_null()).cloned();
    let substituted = std::sync::atomic::AtomicUsize::new(0);
    let rows: Vec<(Value, Value)> = (0..n)
        .into_par_iter()
        .map(|k| -> Result<(Value, Value)> {
            let id = &ids[k];
            let m = by_id.get(id.as_str()).ok_or_else(|| format!("{id}: no model_attributes row"))?;
            let ini = if let Some(raw) = &raw_inis { raw[k].clone() } else {
                let (soil, gw, nash) = legacy.as_ref().unwrap();
                let ini = cfe_ini(m, CfeInitial { soil_storage: soil[k].clamp(0.0, 1.0), gw_storage: gw[k].clamp(0.0, 1.0) })?;
                let nash_line = format!("nash_storage_subsurface={},{}", nash[2 * k].max(0.0), nash[2 * k + 1].max(0.0));
                if !ini.contains("nash_storage_subsurface=0.0,0.0\n") {
                    return Err(format!("{id}: cfe_ini layout changed (nash_storage_subsurface line)").into());
                }
                ini.replace("nash_storage_subsurface=0.0,0.0", &nash_line)
            };
            let path = ini_dir.join(format!("{id}.ini"));
            std::fs::write(&path, &ini)?;
            let mut ov = noahowp_overrides(m)?;
            if let Some(w) = &water {
                let mut hit = false;
                for (from, key) in [("water_vegtyp", "vegtyp"), ("water_isltyp", "isltyp")] {
                    if w[from].is_i64() && ov[key] == w[from] {
                        ov[key] = w[key].clone();
                        hit = true;
                    }
                }
                if hit {
                    substituted.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
            }
            ov["sh2o"] = json!(sh2o[4 * k..4 * k + 4]);
            ov["sice"] = json!(sice[4 * k..4 * k + 4]);
            ov["soil_temperature_k"] = json!(stc[4 * k..4 * k + 4]);
            // Keep the declared soil-layer mapping, then convert metres SWE to
            // kg/m2 (native mm water) in f64, then round to native REAL before
            // JSON transport: a shortest f64 decimal can straddle an f32 tie.
            let mut dz = vec![0.0; 3];
            dz.extend(layer_bounds.windows(2).map(|b| b[1] - b[0]));
            ov["dzsnso"] = json!(dz);
            ov["snow_water_equivalent_mm"] = json!((snow_swe[k] * 1000.0) as f32);
            ov["snow_depth_m"] = json!(snow_depth[k] as f32);
            ov["snow_temperature_k"] = json!(snow_temp[k] as f32);
            Ok((
                json!({"id": id, "config": {"path": path, "sha256": packet::sha256_hex(ini.as_bytes())}}),
                json!({"id": id, "namelist": namelist, "overrides": ov}),
            ))
        })
        .collect::<Result<_>>()?;
    c.stage("CFE INIs and Noah-OWP rows", &cores());
    let (cfe_cols, noah_rows): (Vec<Value>, Vec<Value>) = rows.into_iter().unzip();
    let cfe_req = json!({"schema": "arwen.cfe.prepare/v1", "time_origin": s(r, "time_origin")?, "columns": cfe_cols,
        "output": out.join("cfe-packet")});
    let cfe_sha = write_json(&out.join("cfe-request.json"), &cfe_req)?;
    let noah_sha = write_json(&out.join("noah-request.json"), &json!({"catchments": noah_rows}))?;
    c.stage("write requests", "1 thread");
    let receipt = json!({"schema": "arwen.hydro-world.engine-packets.v1", "fabric": s(r, "fabric")?, "fabric_sha256": fsha,
        "catchments": n, "cfe_request_sha256": cfe_sha, "noah_request_sha256": noah_sha,
        "noah_namelist": namelist, "state": state_dir, "spinup": spin_dir,
        "water_catchments": {"rule": water, "substituted_catchments": substituted.into_inner()},
        "cfe_initialization": if raw_inis.is_some() {"complete raw stores, GIUH queue and percolation carry; exact spin-up INIs/ordered IDs sealed and static parameters checked against fabric; fresh native budget epoch, not a restart"} else {"legacy fractions plus Nash only: raw soil/GW, GIUH queue and percolation carry unbound; regenerate spin-up for complete raw initialization"},
        "spinup_manifest_sha256": packet::sha256_hex(&std::fs::read(std::path::Path::new(spin_dir).join("manifest.json"))?),
        "initial_storage": "CFE initialization as declared above; Noah-OWP sh2o/sice, declared soil layers, soil temperature and aggregate SWE/depth/temperature from the state packet; aggregate snow uses the explicit aggregate-all-ice-divide-v1 recipe (uniform temperature, no inferred liquid water or age)",
        "timing": c.done()});
    write_json(&out.join("receipt.json"), &receipt)?;
    Ok(receipt)
}

/// The commit this binary was built from; the release pin step reads it from the bytes.
const SOURCE_REV_STAMP: &str = concat!("GPUWM_BRIDGE_SOURCE_REV=", env!("GPUWM_BRIDGE_SOURCE_REV"));

fn main() {
    std::hint::black_box(SOURCE_REV_STAMP);
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: rw_hydro_init <tessellate|bbox|weights|state|spinup|forcing-gaps|forcing-fill|cfe-trace|engine-packets> REQUEST.json");
        std::process::exit(2);
    }
    let run = || -> Result<Value> {
        let r: Value = serde_json::from_str(&std::fs::read_to_string(&args[2])?)?;
        match args[1].as_str() {
            "tessellate" => cmd_tessellate(&r),
            "bbox" => cmd_bbox(&r),
            "weights" => cmd_weights(&r),
            "state" => cmd_state(&r),
            "spinup" => cmd_spinup(&r),
            "forcing-gaps" => gapfill::cmd_forcing_gaps(&r),
            "forcing-fill" => gapfill::cmd_forcing_fill(&r),
            "cfe-trace" => cmd_cfe_trace(&r),
            "engine-packets" => cmd_engine_packets(&r),
            c => Err(format!("unknown command {c}").into()),
        }
    };
    match run() {
        Ok(v) => println!("{}", serde_json::to_string_pretty(&v).unwrap()),
        Err(e) => {
            eprintln!("rw_hydro_init {}: {e}", args[1]);
            std::process::exit(1);
        }
    }
}
