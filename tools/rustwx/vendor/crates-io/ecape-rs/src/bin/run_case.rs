use ecape_rs::{
    CapeType, ParcelOptions, ParcelProfile, StormMotionType, calc_ecape_ncape,
    calc_ecape_ncape_from_reference, calc_ecape_parcel, summarize_parcel_profile,
};
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
    reference_ecape_jkg: f64,
    reference_ncape_jkg: f64,
    reference_cape_jkg: f64,
    reference_lfc_m: Option<f64>,
    reference_el_m: Option<f64>,
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

fn interp_desc(x_desc: &[f64], y: &[f64], target: f64) -> f64 {
    if target >= x_desc[0] {
        return y[0];
    }
    if target <= x_desc[x_desc.len() - 1] {
        return y[y.len() - 1];
    }
    for i in 1..x_desc.len() {
        if x_desc[i - 1] >= target && target >= x_desc[i] {
            let x0 = x_desc[i - 1];
            let x1 = x_desc[i];
            let y0 = y[i - 1];
            let y1 = y[i];
            if (x1 - x0).abs() < 1e-12 {
                return y0;
            }
            return y0 + (y1 - y0) * (target - x0) / (x1 - x0);
        }
    }
    y[y.len() - 1]
}

fn align_parcel_profile(
    input_pressure_pa: &[f64],
    input_height_m: &[f64],
    env_temperature_k: &[f64],
    env_qv_kgkg: &[f64],
    raw: &ParcelProfile,
) -> ParcelProfile {
    let mut pressure_pa = Vec::with_capacity(input_pressure_pa.len());
    let mut height_m = Vec::with_capacity(input_pressure_pa.len());
    let mut temperature_k = Vec::with_capacity(input_pressure_pa.len());
    let mut qv_kgkg = Vec::with_capacity(input_pressure_pa.len());
    let mut qt_kgkg = Vec::with_capacity(input_pressure_pa.len());

    for i in 0..input_pressure_pa.len() {
        let p = input_pressure_pa[i];
        pressure_pa.push(p);
        height_m.push(input_height_m[i]);
        if p <= raw.pressure_pa[0] && p >= raw.pressure_pa[raw.pressure_pa.len() - 1] {
            temperature_k.push(interp_desc(&raw.pressure_pa, &raw.temperature_k, p));
            qv_kgkg.push(interp_desc(&raw.pressure_pa, &raw.qv_kgkg, p));
            qt_kgkg.push(interp_desc(&raw.pressure_pa, &raw.qt_kgkg, p));
        } else {
            temperature_k.push(env_temperature_k[i]);
            qv_kgkg.push(env_qv_kgkg[i]);
            qt_kgkg.push(env_qv_kgkg[i]);
        }
    }

    ParcelProfile {
        pressure_pa,
        height_m,
        temperature_k,
        qv_kgkg,
        qt_kgkg,
        buoyancy_ms2: Vec::new(),
    }
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
    let qv: Vec<f64> = pressure_pa
        .iter()
        .zip(payload.dewpoint_k.iter())
        .map(|(p, td)| {
            let es = 611.2 * ((17.67 * (td - 273.15)) / ((td - 273.15) + 243.5)).exp();
            0.62197 * es / (p - 0.37803 * es)
        })
        .collect();

    let summary = calc_ecape_ncape(
        &payload.height_m,
        &pressure_pa,
        &payload.temperature_k,
        &qv,
        &payload.u_wind_ms,
        &payload.v_wind_ms,
        &options,
    )?;

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

    let aligned_profile = align_parcel_profile(
        &pressure_pa,
        &payload.height_m,
        &payload.temperature_k,
        &qv,
        &result.parcel_profile,
    );
    let aligned_summary = summarize_parcel_profile(
        &aligned_profile.height_m,
        &aligned_profile.temperature_k,
        &aligned_profile.qv_kgkg,
        &aligned_profile.qt_kgkg,
        &payload.height_m,
        &payload.temperature_k,
        &qv,
    );
    let aligned_ecape = calc_ecape_ncape_from_reference(
        &payload.height_m,
        &pressure_pa,
        &payload.temperature_k,
        &qv,
        &payload.u_wind_ms,
        &payload.v_wind_ms,
        &options,
        aligned_summary.cape_jkg,
        aligned_summary.lfc_m,
        aligned_summary.el_m,
    );

    let output = OutputPayload {
        reps,
        elapsed_ms,
        per_call_ms: elapsed_ms / reps as f64,
        reference_ecape_jkg: summary.ecape_jkg,
        reference_ncape_jkg: summary.ncape_jkg,
        reference_cape_jkg: summary.cape_jkg,
        reference_lfc_m: summary.lfc_m,
        reference_el_m: summary.el_m,
        ecape_jkg: aligned_ecape.ecape_jkg,
        ncape_jkg: aligned_ecape.ncape_jkg,
        cape_jkg: aligned_summary.cape_jkg,
        cin_jkg: aligned_summary.cin_jkg,
        lfc_m: aligned_summary.lfc_m,
        el_m: aligned_summary.el_m,
        storm_motion_u_ms: aligned_ecape.storm_motion_u_ms,
        storm_motion_v_ms: aligned_ecape.storm_motion_v_ms,
        parcel_pressure_pa: aligned_profile.pressure_pa,
        parcel_height_m: aligned_profile.height_m,
        parcel_temperature_k: aligned_profile.temperature_k,
        parcel_qv_kgkg: aligned_profile.qv_kgkg,
        parcel_qt_kgkg: aligned_profile.qt_kgkg,
    };

    println!("{}", serde_json::to_string(&output)?);
    Ok(())
}
