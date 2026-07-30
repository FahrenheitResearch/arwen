use ecape_rs::{CapeType, ParcelOptions, StormMotionType, calc_ecape_parcel};
use serde::{Deserialize, Serialize};
use std::io::{self, Read};
use std::time::Instant;

#[derive(Debug, Deserialize)]
struct OptionsPayload {
    cape_type: Option<String>,
    storm_motion_type: Option<String>,
    origin_pressure_pa: Option<f64>,
    origin_height_m: Option<f64>,
    mixed_layer_depth_pa: Option<f64>,
    inflow_layer_bottom_m: Option<f64>,
    inflow_layer_top_m: Option<f64>,
    storm_motion_u_ms: Option<f64>,
    storm_motion_v_ms: Option<f64>,
    entrainment_rate: Option<f64>,
    pseudoadiabatic: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct InputPayload {
    pressure_hpa: Vec<f64>,
    height_m: Vec<f64>,
    temperature_k: Vec<f64>,
    dewpoint_k: Vec<f64>,
    u_wind_ms: Vec<f64>,
    v_wind_ms: Vec<f64>,
    options: Option<OptionsPayload>,
    reps: Option<usize>,
}

#[derive(Debug, Serialize)]
struct OutputPayload {
    reps: usize,
    elapsed_ms: f64,
    per_call_ms: f64,
    ecape_jkg: f64,
    ncape_jkg: f64,
    cape_jkg: f64,
    cin_jkg: f64,
    lfc_m: Option<f64>,
    el_m: Option<f64>,
    storm_motion_u_ms: f64,
    storm_motion_v_ms: f64,
    parcel_pressure_pa: Vec<f64>,
    parcel_height_m: Vec<f64>,
    parcel_temperature_k: Vec<f64>,
    parcel_qv_kgkg: Vec<f64>,
    parcel_qt_kgkg: Vec<f64>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input)?;
    let payload: InputPayload = serde_json::from_str(&input)?;
    let reps = payload.reps.unwrap_or(1).max(1);

    let options_payload = payload.options;
    let options = ParcelOptions {
        cape_type: CapeType::parse_or_default(
            options_payload
                .as_ref()
                .and_then(|o| o.cape_type.as_deref()),
        ),
        storm_motion_type: StormMotionType::parse_or_default(
            options_payload
                .as_ref()
                .and_then(|o| o.storm_motion_type.as_deref()),
        ),
        origin_pressure_pa: options_payload.as_ref().and_then(|o| o.origin_pressure_pa),
        origin_height_m: options_payload.as_ref().and_then(|o| o.origin_height_m),
        mixed_layer_depth_pa: options_payload
            .as_ref()
            .and_then(|o| o.mixed_layer_depth_pa)
            .or(Some(10000.0)),
        inflow_layer_bottom_m: options_payload
            .as_ref()
            .and_then(|o| o.inflow_layer_bottom_m)
            .or(Some(0.0)),
        inflow_layer_top_m: options_payload
            .as_ref()
            .and_then(|o| o.inflow_layer_top_m)
            .or(Some(1000.0)),
        storm_motion_u_ms: options_payload.as_ref().and_then(|o| o.storm_motion_u_ms),
        storm_motion_v_ms: options_payload.as_ref().and_then(|o| o.storm_motion_v_ms),
        entrainment_rate: options_payload.as_ref().and_then(|o| o.entrainment_rate),
        pseudoadiabatic: options_payload
            .as_ref()
            .and_then(|o| o.pseudoadiabatic)
            .or(Some(true)),
    };

    let pressure_pa: Vec<f64> = payload.pressure_hpa.iter().map(|v| v * 100.0).collect();

    let start = Instant::now();
    let mut result = calc_ecape_parcel(
        &payload.height_m,
        &pressure_pa,
        &payload.temperature_k,
        &payload.dewpoint_k,
        &payload.u_wind_ms,
        &payload.v_wind_ms,
        &options,
    )?;
    for _ in 1..reps {
        result = calc_ecape_parcel(
            &payload.height_m,
            &pressure_pa,
            &payload.temperature_k,
            &payload.dewpoint_k,
            &payload.u_wind_ms,
            &payload.v_wind_ms,
            &options,
        )?;
    }
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

    let output = OutputPayload {
        reps,
        elapsed_ms,
        per_call_ms: elapsed_ms / reps as f64,
        ecape_jkg: result.ecape_jkg,
        ncape_jkg: result.ncape_jkg,
        cape_jkg: result.cape_jkg,
        cin_jkg: result.cin_jkg,
        lfc_m: result.lfc_m,
        el_m: result.el_m,
        storm_motion_u_ms: result.storm_motion_u_ms,
        storm_motion_v_ms: result.storm_motion_v_ms,
        parcel_pressure_pa: result.parcel_profile.pressure_pa,
        parcel_height_m: result.parcel_profile.height_m,
        parcel_temperature_k: result.parcel_profile.temperature_k,
        parcel_qv_kgkg: result.parcel_profile.qv_kgkg,
        parcel_qt_kgkg: result.parcel_profile.qt_kgkg,
    };

    println!("{}", serde_json::to_string(&output)?);
    Ok(())
}
