//! One native command for verification receipts, scorecards and field sheets.

use rw_wrfbatch::verification::{self, Artifact, VerificationReceipt, VerificationRequest};
use rw_wrfbatch::verification_io::{self, load_arm, load_radar, source_identity, ArmData};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

/// Embed the build revision for release artifact verification.
pub static GPUWM_BRIDGE_SOURCE_REV_STAMP: &str =
    concat!("GPUWM_BRIDGE_SOURCE_REV=", env!("GPUWM_BRIDGE_SOURCE_REV"));

const ABI:&str="gpuwm.verify-visuals.request.v1\tgpuwm.verify-visuals.receipt.v1\tgpuwm.verify-visuals.inventory.v1\tstations\tradar\tartifacts\tnative-scoring\tnative-rendering";
const USAGE:&str="rw_verify --request REQUEST.json --json RECEIPT.json [--image CARD.png]\nrw_verify --inventory REQUEST.json\nrw_verify --prepare REQUEST.json --json PREPARED_REQUEST.json\nrw_verify --help | --abi";

fn prepare(request: &VerificationRequest) -> Result<Value, String> {
    let root = request
        .out_root
        .as_ref()
        .ok_or("--prepare requires out_root")?;
    let folder = root
        .join(rw_wrfbatch::panel::safe_component(
            &request.domain,
            "native_grid",
        ))
        .join("verification_inputs")
        .join(&request.valid_time[..10]);
    std::fs::create_dir_all(&folder).map_err(|e| e.to_string())?;
    let stamp = request.valid_time.replace(['-', ':', 'T', 'Z'], "");
    let mut prepared = request.clone();
    for spec in &mut prepared.arms {
        if !spec.path.is_file() {
            continue;
        }
        let arm = load_arm(spec)?;
        verification_io::validate_arm_time(spec, &arm, &request.valid_time)?;
        if let Some((dx, dy)) = verification_io::arm_spacing(&arm) {
            if prepared.dx_km == 0.0 {
                prepared.dx_km = dx;
                prepared.dy_km = Some(dy);
            }
        }
        if spec.kind == "npz"
            && arm.provenance["metadata"]["schema"].as_str()
                == Some("gpuwm.verify-visuals.inputs.v1")
            && spec.path.parent() == Some(folder.as_path())
        {
            continue;
        }
        let name = rw_wrfbatch::panel::safe_component(&spec.label, "forecast");
        let point_source = if spec.kind == "points" {
            Some(spec.path.clone())
        } else {
            spec.points_path.clone()
        };
        let retained_points = if let Some(source) = point_source {
            let destination = folder.join(format!("{name}-{stamp}.points.json"));
            if source != destination {
                std::fs::copy(&source, &destination).map_err(|e| e.to_string())?;
            }
            Some(destination)
        } else {
            None
        };
        spec.source_identity = Some(arm.provenance.clone());
        if arm.grid.is_some() {
            let destination = folder.join(format!("{name}-{stamp}.npz"));
            verification_io::write_snapshot(&destination, &arm)?;
            let metadata = json!({"schema":"gpuwm.verify-visuals.inputs.v1","artifact":source_identity(&destination)?,"valid_time":request.valid_time,"init_time":arm.provenance["metadata"]["init_time"],"dx_km":verification_io::arm_spacing(&arm).map(|v|v.0),"dy_km":verification_io::arm_spacing(&arm).map(|v|v.1),"source_identity":arm.provenance,"projection":arm.grid.as_ref().and_then(|g|g.projection.as_ref()),"fields":arm.fields.keys().collect::<Vec<_>>()});
            std::fs::write(
                destination.with_extension("metadata.json"),
                serde_json::to_vec_pretty(&metadata).map_err(|e| e.to_string())?,
            )
            .map_err(|e| e.to_string())?;
            spec.kind = "npz".into();
            spec.path = destination;
            spec.grid_path = None;
            spec.previous_path = None;
            spec.points_path = retained_points;
            spec.fields = arm.fields.keys().map(|q| (q.clone(), q.clone())).collect();
        } else if let Some(destination) = retained_points {
            spec.path = destination;
        }
    }
    serde_json::to_value(prepared).map_err(|e| e.to_string())
}

fn inventory(request: &VerificationRequest) -> Result<Value, String> {
    let mut arm_rows = Vec::new();
    let mut bbox = None;
    let mut spacing = None;
    let mut init_time = None;
    for spec in &request.arms {
        if !spec.path.is_file() {
            arm_rows.push(json!({"label":spec.label,"quantities":[],"missing":true}));
            continue;
        }
        let arm = load_arm(spec)?;
        verification_io::validate_arm_time(spec, &arm, &request.valid_time)?;
        spacing = spacing.or_else(|| verification_io::arm_spacing(&arm));
        init_time = init_time.or_else(|| {
            arm.provenance["metadata"]["init_time"]
                .as_str()
                .or_else(|| arm.provenance["point_extract"]["init_time"].as_str())
                .map(str::to_string)
        });
        let mut supported: BTreeSet<String> = arm.fields.keys().cloned().collect();
        for point in arm.points.values() {
            supported.extend(
                point
                    .values
                    .keys()
                    .filter(|q| {
                        verification::STATION_QUANTITIES
                            .iter()
                            .any(|(name, _, _)| name == q)
                    })
                    .cloned(),
            );
        }
        if let Some(grid) = arm.grid.as_ref() {
            bbox.get_or_insert_with(|| grid.bbox());
        }
        arm_rows.push(json!({"label":spec.label,"quantities":supported,"missing":false}));
    }
    if bbox.is_none() {
        if let Some(table) = &request.station_table_path {
            let meta = verification_io::read_json(table)?;
            let mut bounds = [180.0_f64, 90.0_f64, -180.0_f64, -90.0_f64];
            for s in meta["stations"]
                .as_array()
                .ok_or("station table has no stations")?
            {
                let lat = s["latitude"]
                    .as_f64()
                    .or_else(|| s["lat"].as_f64())
                    .ok_or("station table latitude absent")?;
                let lon = s["longitude"]
                    .as_f64()
                    .or_else(|| s["lon"].as_f64())
                    .ok_or("station table longitude absent")?;
                bounds[0] = bounds[0].min(lon);
                bounds[1] = bounds[1].min(lat);
                bounds[2] = bounds[2].max(lon);
                bounds[3] = bounds[3].max(lat);
            }
            bbox = Some(bounds);
        }
    }
    Ok(
        json!({"schema":verification::INVENTORY_SCHEMA,"bbox":bbox,"dx_km":spacing.map(|v|v.0).unwrap_or(request.dx_km),"dy_km":spacing.map(|v|v.1).or(request.dy_km),"init_time":init_time,"valid_time":request.valid_time,"domain":request.domain,"arms":arm_rows}),
    )
}

fn score(
    request: &VerificationRequest,
    request_bytes: &[u8],
    card: Option<&Path>,
) -> Result<VerificationReceipt, String> {
    let arms = request
        .arms
        .iter()
        .map(load_arm)
        .collect::<Result<Vec<ArmData>, _>>()?;
    for (spec, arm) in request.arms.iter().zip(&arms) {
        verification_io::validate_arm_time(spec, arm, &request.valid_time)?;
    }
    let mut request = request.clone();
    if let Some(spacing) = arms.first().and_then(verification_io::arm_spacing) {
        request.dx_km = spacing.0;
        request.dy_km = Some(spacing.1);
    }
    request.validate()?;
    let observations = if let Some(path) = &request.stations_path {
        verification_io::load_station_observations(path, &request.valid_time)?
    } else {
        Vec::new()
    };
    let (stations, station_samples, station_drops) =
        verification::paired_station_scores(&request, &arms, &observations)?;
    let mut receipt=VerificationReceipt{schema:verification::RECEIPT_SCHEMA.into(),valid_time:request.valid_time.clone(),domain:request.domain.clone(),request_sha256:format!("{:x}",Sha256::digest(request_bytes)),error_convention:"forecast minus observation".into(),station_interpolation:"bilinear native coordinate lattice; supplied point extracts preserve their extraction method".into(),temperature_method:"raw model temperature; no elevation adjustment; temperature_2m_raw preferred when present".into(),wind_metric:"scalar 10 m wind speed error, not vector wind error".into(),neighborhood_support:"complete square neighbourhoods on common observed and forecast finite coverage, native-grid nearest sampling".into(),sources:arms.iter().map(|a|a.provenance.clone()).collect(),stations,station_samples,station_drops,radar:Vec::new(),artifacts:Vec::new(),skipped:Vec::new()};
    if let Some(path) = &request.stations_path {
        receipt.sources.push(source_identity(path)?);
    } else {
        receipt
            .skipped
            .push("station observations not supplied".into());
    }
    for row in &receipt.stations {
        if row.status == "missing" {
            receipt
                .skipped
                .push(format!("{} has no paired station samples", row.quantity));
        }
    }
    let mut radars = Vec::new();
    for spec in &request.radar {
        let radar = load_radar(spec)?;
        let offset = (verification::parse_time(&radar.valid_time)?
            - verification::parse_time(&request.valid_time)?)
        .num_seconds()
        .abs();
        let window = if spec.quantity == "precipitation_1h"
            || spec.quantity == "precipitation_accumulation"
        {
            0
        } else {
            request.radar_window_seconds
        };
        if offset > window {
            return Err(format!(
                "{} observation is {offset} seconds from valid time, outside {window} seconds",
                spec.quantity
            ));
        }
        let Some((_, _, units, thresholds)) = verification::RADAR_QUANTITIES
            .iter()
            .find(|(q, _, _, _)| *q == spec.quantity)
        else {
            return Err(format!(
                "no verification threshold metadata for {}",
                spec.quantity
            ));
        };
        let Some(grid) = arms.first().and_then(|a| a.grid.as_ref()) else {
            receipt
                .skipped
                .push(format!("{} forecast grids unavailable", spec.quantity));
            continue;
        };
        if !arms
            .iter()
            .all(|a| a.grid.is_some() && a.fields.contains_key(&spec.quantity))
        {
            receipt.skipped.push(format!(
                "{} field is unavailable in a forecast arm",
                spec.quantity
            ));
            continue;
        }
        let observed = radar
            .values
            .iter()
            .zip(&radar.valid)
            .map(|(&v, &ok)| if ok { v } else { f64::NAN })
            .collect::<Vec<_>>();
        let observed = verification::mapped_fields(grid, &radar.grid, &observed)?;
        let valid = observed.iter().map(|v| v.is_finite()).collect::<Vec<_>>();
        let forecasts = arms
            .iter()
            .map(|a| {
                Ok((
                    a.label.clone(),
                    verification::mapped_fields(
                        grid,
                        a.grid.as_ref().unwrap(),
                        &a.fields[&spec.quantity],
                    )?,
                ))
            })
            .collect::<Result<Vec<_>, String>>()?;
        for &threshold in *thresholds {
            for &width in &request.widths_km {
                receipt.radar.push(verification::paired_fss_rect(
                    &spec.quantity,
                    units,
                    threshold,
                    width,
                    request.dx_km,
                    request.dy_km.unwrap_or(request.dx_km),
                    grid.ny,
                    grid.nx,
                    &observed,
                    &valid,
                    &forecasts,
                )?);
            }
        }
        receipt.sources.push(source_identity(&spec.path)?);
        receipt.sources.push(source_identity(&spec.grid_path)?);
        radars.push(radar);
    }
    receipt.artifacts =
        rw_wrfbatch::verification_render::map_sheets(&request, &receipt, &arms, &radars)?;
    if let Some(path) = card {
        rw_wrfbatch::verification_render::scorecard(&receipt, path)?;
        receipt
            .artifacts
            .push(Artifact::new("verification_scorecard".into(), path.into())?);
    }
    Ok(receipt)
}

fn main() -> ExitCode {
    let _ = std::hint::black_box(GPUWM_BRIDGE_SOURCE_REV_STAMP);
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.iter().any(|a| a == "--abi") {
        println!("{ABI}");
        return ExitCode::SUCCESS;
    }
    if args.is_empty() || args.iter().any(|a| a == "--help" || a == "-h") {
        println!("{USAGE}");
        return ExitCode::SUCCESS;
    }
    let run = || -> Result<(), String> {
        let mut request_path = None;
        let mut output = None;
        let mut card = None;
        let mut inventory_only = false;
        let mut prepare_only = false;
        let mut i = 0;
        while i < args.len() {
            let key = &args[i];
            i += 1;
            let value = args.get(i).ok_or_else(|| format!("{key} needs a value"))?;
            i += 1;
            match key.as_str() {
                "--request" => request_path = Some(PathBuf::from(value)),
                "--inventory" => {
                    request_path = Some(PathBuf::from(value));
                    inventory_only = true
                }
                "--json" => output = Some(PathBuf::from(value)),
                "--prepare" => {
                    request_path = Some(PathBuf::from(value));
                    prepare_only = true;
                }
                "--image" => card = Some(PathBuf::from(value)),
                other => return Err(format!("unknown option {other}")),
            }
        }
        let path = request_path.ok_or("--request or --inventory is required")?;
        let bytes = std::fs::read(&path).map_err(|e| e.to_string())?;
        let request: VerificationRequest =
            serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
        let value = if inventory_only {
            inventory(&request)?
        } else if prepare_only {
            prepare(&request)?
        } else {
            serde_json::to_value(score(&request, &bytes, card.as_deref())?)
                .map_err(|e| e.to_string())?
        };
        let text = serde_json::to_string_pretty(&value).map_err(|e| e.to_string())?;
        if let Some(path) = output {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
            }
            std::fs::write(&path, format!("{text}\n")).map_err(|e| e.to_string())?;
        }
        println!("{text}");
        Ok(())
    };
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("rw_verify: {e}");
            ExitCode::FAILURE
        }
    }
}
