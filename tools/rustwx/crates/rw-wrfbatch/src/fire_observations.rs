//! Same-valid-time observed perimeter overlay and coupling evidence.
use netcrust::File;
use rustwx_render::{Color, ContourLayer};
use serde::Deserialize;
use serde_json::{Value, json};
use static_fields::{
    fire_ignition::{FeatureSource, verification},
    projection::GridSpec,
};
use std::path::Path;

#[derive(Deserialize)]
struct Request {
    source: FeatureSource,
    incident_id: String,
    observed_utc_epoch_ms: i64,
    grid_spec: GridSpec,
}

/// Explicit verification input is applied to every requested frame. A time
/// mismatch fails instead of drawing an unrelated later perimeter.
pub fn overlay(
    file: &File,
    record: usize,
    time: &str,
    lat: &[f32],
    lon: &[f32],
    lfn: &[f32],
    shape: [usize; 2],
    dx: f64,
    dy: f64,
    output: &Path,
) -> Result<Option<ContourLayer>, String> {
    let Some(path) = std::env::var_os("GPUWM_FIRE_OBSERVATIONS") else {
        return Ok(None);
    };
    let path = std::path::PathBuf::from(path);
    let mut r: Request = serde_json::from_slice(&std::fs::read(&path).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    if r.source.input.path.is_relative() {
        r.source.input.path = path
            .parent()
            .unwrap_or(Path::new("."))
            .join(&r.source.input.path);
    }
    let model_ms = chrono::NaiveDateTime::parse_from_str(time, "%Y-%m-%d_%H:%M:%S")
        .or_else(|_| chrono::NaiveDateTime::parse_from_str(time, "%Y-%m-%dT%H:%M:%S"))
        .map_err(|e| format!("Fire verification valid UTC time: {e}"))?
        .and_utc()
        .timestamp_millis();
    let sr_x = file
        .attribute("SR_X")
        .and_then(|a| a.as_f64())
        .ok_or("Fire overlay needs SR_X")? as usize;
    let sr_y = file
        .attribute("SR_Y")
        .and_then(|a| a.as_f64())
        .ok_or("Fire overlay needs SR_Y")? as usize;
    if r.grid_spec.e_we < 2 || r.grid_spec.e_sn < 2 || sr_x == 0 || sr_y == 0 {
        return Err("Observation grid needs physical dimensions and positive refinement".into());
    }
    let nx = (r.grid_spec.e_we as usize - 1)
        .checked_mul(sr_x)
        .ok_or("Observation grid size overflow")?;
    let ny = (r.grid_spec.e_sn as usize - 1)
        .checked_mul(sr_y)
        .ok_or("Observation grid size overflow")?;
    if nx == 0 || ny == 0 || nx > shape[1] || ny > shape[0] {
        return Err("Observation grid differs from physical fire output".into());
    }
    let crop = |data: &[f32]| -> Vec<f32> {
        (0..ny)
            .flat_map(|j| data[j * shape[1]..j * shape[1] + nx].iter().copied())
            .collect()
    };
    let (observed, mut receipt) = verification(
        &r.source,
        &r.incident_id,
        r.observed_utc_epoch_ms,
        model_ms,
        &r.grid_spec,
        &crop(lat),
        &crop(lon),
        &crop(lfn),
        dx * dy,
    )
    .map_err(|e| e.to_string())?;
    let mut full = vec![f32::NAN; lfn.len()];
    for j in 0..ny {
        full[j * shape[1]..j * shape[1] + nx].copy_from_slice(&observed[j * nx..(j + 1) * nx]);
    }
    let mut coupling = serde_json::Map::new();
    for name in [
        "FGRNHFX", "FGRNQFX", "GRNHFX", "GRNQFX", "RTHFRTEN", "RQVFRTEN",
    ] {
        let values = file
            .read_array_f64_record_or_all(name, record as u64)
            .map_err(|e| format!("Coupling receipt needs {name}: {e}"))?;
        let data = values.values();
        if data.is_empty() || data.iter().any(|v| !v.is_finite()) {
            return Err(format!("Coupling receipt has invalid {name}"));
        }
        let lo = data.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        coupling.insert(name.into(), json!({"min":lo,"max":hi,"samples":data.len()}));
    }
    let pyro = file
        .attribute("FIRE_PYROCONVECTION_ENABLED")
        .and_then(|a| a.as_f64())
        == Some(1.);
    receipt["coupling"] = Value::Object(coupling);
    receipt["heat_and_moisture_feedback_enabled"] = json!(pyro);
    receipt["pyroconvection_meaning"] =
        json!("thermodynamic feedback enabled; pyroCb occurrence not implied");
    receipt["contours"] = json!({"model":"dark red","observed":"cyan"});
    receipt["history"] = json!({"record":record});
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    std::fs::write(
        output.with_extension("fire-verification.json"),
        serde_json::to_vec_pretty(&receipt).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    Ok(Some(ContourLayer {
        data: full,
        levels: vec![0.],
        color: Color::rgba(0, 170, 225, 255),
        width: 3,
        labels: false,
        show_extrema: false,
        pattern: Default::default(),
        major_every: None,
        major_width: None,
    }))
}
