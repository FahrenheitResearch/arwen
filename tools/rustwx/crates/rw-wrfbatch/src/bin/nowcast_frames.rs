//! `rw_nowcast_frames` -- a 2D reflectivity NetCDF file written out as the
//! `gpuwm-obs.nowcast-frames.v1` contract (`docs/nowcast-frames.md`).
//!
//! The renderer (`rw_compare --panel "LABEL=frames:ROOT@VALID"`) and the
//! radar-heating adapter (`rw_nexrad grid-composite`) read reflectivity
//! frames in one documented byte format and nothing else.  A nowcast
//! runner that writes its output as NetCDF instead -- a 2D `refc` variable
//! over time, optional member, y and x, with the cell-centre latitude and
//! longitude beside it -- has no way onto a sheet or into the heating
//! until something writes those files.  This binary is that something,
//! in Rust, so no Python decodes the frames on the way.
//!
//! ```text
//! rw_nowcast_frames from-netcdf --netcdf FILE --out ROOT --source-id ID
//!     --lattice lambert:truelat1=38.5,truelat2=38.5,stand_lon=-97.5,ref_lat=38.5,earth_radius_m=6371229
//!   | --lattice latlon
//!     [--variable refc] [--lat lat] [--lon lon]
//!     [--issue TIME] [--causal true|false] [--receipt FILE.json]
//!     [--source-kind nowcast|observed] [--package TEXT] [--revision TEXT]
//! rw_nowcast_frames --help | --abi
//! ```
//!
//! What is read off the file: the variable's dimensions name the time axis
//! (the dimension whose coordinate variable carries CF `units` of the form
//! `hours since ...`) and, when a fourth dimension is present, the member
//! axis; the last two dimensions are rows and columns.  The lattice is
//! fitted from the latitude and longitude variables in the projection the
//! caller names (`lambert:` with its five numbers) or as a regular
//! latitude-longitude grid (`latlon`), and refused when the grid is not
//! rectilinear in that projection or its corners do not reproduce.
//!
//! What the file cannot say is taken from a runner receipt (`--receipt`,
//! a JSON object with `issue_time`, `causal`, `latest_input_time`,
//! `objects[]`, `seed`, `variant`, `step_s`, `load_s`), from the file's
//! own global attributes (`init`, `causal`, `seed`, `variant`, `package`),
//! or from the flags, in that order of precedence from last to first.  A
//! frame set with no issue time, or one that does not say whether it saw
//! the future, is refused: the sheet would be labelled wrong and the
//! heating could be scored as a forecast.  A receipt whose issue time or
//! causal flag disagrees with the file's own attributes is refused too:
//! one of them describes another run.
//!
//! The one line on stdout is a JSON record naming the root, the frames
//! written and the lattice.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use chrono::{DateTime, Duration, NaiveDateTime, Utc};
use rw_wrfbatch::nowcast_frames::{
    FrameData, FramesProvenance, LambertProjection, Lattice, lambert_lattice_from_latlon, lattice_json,
    latlon_lattice_from_latlon, parse_utc, write_frames_root,
};
use serde_json::{Value, json};

/// Embed the build revision for release artifact verification.
pub static GPUWM_BRIDGE_SOURCE_REV_STAMP: &str =
    concat!("GPUWM_BRIDGE_SOURCE_REV=", env!("GPUWM_BRIDGE_SOURCE_REV"));

const ABI_MARKER: &str = "gpuwm-rw-nowcast-frames-v1\tfrom-netcdf\t--netcdf\t--out\t--source-id\t--lattice\t\
--variable\t--lat\t--lon\t--issue\t--causal\t--receipt\t--source-kind\t--package\t--revision";

const USAGE: &str = "usage: rw_nowcast_frames from-netcdf --netcdf FILE --out ROOT --source-id ID \
--lattice lambert:truelat1=..,truelat2=..,stand_lon=..,ref_lat=..,earth_radius_m=.. | --lattice latlon \
[--variable refc] [--lat lat] [--lon lon] [--issue TIME] [--causal true|false] [--receipt FILE.json] \
[--source-kind nowcast|observed] [--package TEXT] [--revision TEXT]
       rw_nowcast_frames --help | --abi

from-netcdf writes a gpuwm-obs.nowcast-frames.v1 root (nowcast.json, inputs.json, <stamp>/refc.f32) from a
NetCDF file holding a 2D reflectivity variable over [time, (member,) y, x] with cell-centre latitude and
longitude variables.  --lattice names the projection the grid is fitted in.  The issue time and the causal
flag come from --issue / --causal, else the runner receipt (--receipt), else the file's global attributes
(init, causal); a frame set that cannot say both is refused.";

#[derive(Clone, Debug)]
enum LatticeRequest {
    Lambert(LambertProjection),
    LatLon,
}

impl LatticeRequest {
    fn parse(text: &str) -> Result<Self, String> {
        let text = text.trim();
        if text == "latlon" {
            return Ok(Self::LatLon);
        }
        match text.split_once(':') {
            Some(("lambert", rest)) => Ok(Self::Lambert(LambertProjection::parse(rest)?)),
            _ => Err(format!("--lattice {text:?}: lambert:KEY=VALUE,... or latlon")),
        }
    }
}

struct Args {
    netcdf: PathBuf,
    out: PathBuf,
    source_id: String,
    lattice: LatticeRequest,
    variable: String,
    lat: String,
    lon: String,
    issue: Option<DateTime<Utc>>,
    causal: Option<bool>,
    receipt: Option<PathBuf>,
    source_kind: String,
    package: Option<String>,
    revision: Option<String>,
}

fn parse_args(args: &[String]) -> Result<Args, String> {
    let mut netcdf = None;
    let mut out = None;
    let mut source_id = None;
    let mut lattice = None;
    let mut variable = "refc".to_string();
    let mut lat = "lat".to_string();
    let mut lon = "lon".to_string();
    let mut issue = None;
    let mut causal = None;
    let mut receipt = None;
    let mut source_kind = "nowcast".to_string();
    let mut package = None;
    let mut revision = None;
    let mut index = 0;
    while index < args.len() {
        let flag = args[index].as_str();
        let mut value = || -> Result<String, String> {
            index += 1;
            args.get(index).cloned().ok_or_else(|| format!("{flag} needs a value"))
        };
        match flag {
            "--netcdf" => netcdf = Some(PathBuf::from(value()?)),
            "--out" => out = Some(PathBuf::from(value()?)),
            "--source-id" => source_id = Some(value()?),
            "--lattice" => lattice = Some(LatticeRequest::parse(&value()?)?),
            "--variable" => variable = value()?,
            "--lat" => lat = value()?,
            "--lon" => lon = value()?,
            "--issue" => issue = Some(parse_utc(&value()?)?),
            "--causal" => {
                causal = Some(match value()?.to_ascii_lowercase().as_str() {
                    "true" | "yes" | "1" => true,
                    "false" | "no" | "0" => false,
                    other => return Err(format!("--causal {other:?}: true or false")),
                })
            }
            "--receipt" => receipt = Some(PathBuf::from(value()?)),
            "--source-kind" => {
                let kind = value()?;
                if kind != "nowcast" && kind != "observed" {
                    return Err(format!("--source-kind {kind:?}: nowcast or observed"));
                }
                source_kind = kind;
            }
            "--package" => package = Some(value()?),
            "--revision" => revision = Some(value()?),
            other => return Err(format!("unknown argument {other:?}\n{USAGE}")),
        }
        index += 1;
    }
    let source_id = source_id.ok_or("--source-id names the frames' source")?;
    if source_id.trim().is_empty() || source_id.contains(char::is_whitespace) {
        return Err("--source-id is one token with no spaces".into());
    }
    Ok(Args {
        netcdf: netcdf.ok_or("--netcdf names the file to convert")?,
        out: out.ok_or("--out names the frames root to write")?,
        source_id,
        lattice: lattice.ok_or("--lattice names the projection the grid is fitted in (lambert:... or latlon)")?,
        variable,
        lat,
        lon,
        issue,
        causal,
        receipt,
        source_kind,
        package,
        revision,
    })
}

fn open(path: &Path) -> Result<netcrust::File, String> {
    match netcrust::File::open(path) {
        Ok(file) => Ok(file),
        Err(strict) => {
            let options = netcrust::NcOpenOptions {
                metadata_mode: netcrust::NcMetadataMode::Lossy,
                ..Default::default()
            };
            let file = netcrust::File::open_with_options(path, options)
                .map_err(|error| format!("open {}: {error} (strict open: {strict})", path.display()))?;
            eprintln!("NOTE\t{}: metadata read in lossy mode ({strict})", path.display());
            Ok(file)
        }
    }
}

/// A global attribute as text, whatever its stored type.
fn global_text(file: &netcrust::File, name: &str) -> Option<String> {
    let attribute = file.attribute(name)?;
    if let Some(text) = attribute.as_string() {
        return Some(text.trim().to_string());
    }
    attribute.as_f64().map(|number| {
        if number.fract() == 0.0 { format!("{}", number as i64) } else { number.to_string() }
    })
}

/// `init` as runners spell it: `2024-05-21 18:00`, `2024-05-21T18:00`, or any contract spelling.
fn parse_loose_utc(text: &str) -> Result<DateTime<Utc>, String> {
    if let Ok(time) = parse_utc(text) {
        return Ok(time);
    }
    for format in ["%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%MZ"] {
        if let Ok(naive) = NaiveDateTime::parse_from_str(text.trim(), format) {
            return Ok(naive.and_utc());
        }
    }
    Err(format!("{text:?} is not a UTC time"))
}

/// CF `units`: `<unit> since <epoch>`, as a (seconds per unit, epoch) pair.
fn cf_time_units(units: &str) -> Option<(f64, DateTime<Utc>)> {
    let (unit, epoch) = units.trim().split_once(" since ")?;
    let seconds = match unit.trim().to_ascii_lowercase().as_str() {
        "seconds" | "second" | "s" => 1.0,
        "minutes" | "minute" | "min" => 60.0,
        "hours" | "hour" | "h" => 3600.0,
        "days" | "day" | "d" => 86_400.0,
        _ => return None,
    };
    let epoch = epoch.trim().trim_end_matches(" UTC").trim_end_matches('Z');
    let epoch = ["%Y-%m-%d %H:%M:%S%.f", "%Y-%m-%dT%H:%M:%S%.f", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M"]
        .iter()
        .find_map(|format| NaiveDateTime::parse_from_str(epoch, format).ok())
        .or_else(|| chrono::NaiveDate::parse_from_str(epoch, "%Y-%m-%d").ok().map(|day| day.and_hms_opt(0, 0, 0).unwrap()))?;
    Some((seconds, epoch.and_utc()))
}

fn variable_text_attribute(variable: &netcrust::Variable, name: &str) -> Option<String> {
    variable.attribute(name).and_then(|attribute| attribute.as_string().map(|text| text.trim().to_string()))
}

/// A text attribute of a coordinate variable.  In a NetCDF-4 file a
/// coordinate variable is an HDF5 dimension scale, and the NetCDF index
/// netcrust answers `variable()` from omits those datasets; their data
/// already reads through netcrust's raw-HDF5 by-name fallback, and this is
/// the metadata half of the same fallback.  Without it a StormScope file's
/// `valid_time:units` is invisible and the frames are refused as having no
/// time dimension.
fn coordinate_text_attribute(file: &netcrust::File, variable: &str, name: &str) -> Option<String> {
    if let Some(coordinate) = file.variable(variable) {
        return variable_text_attribute(&coordinate, name);
    }
    if file.has_hdf5_dataset(variable) {
        return file.hdf5_dataset_attribute_string(variable, name).map(|text| text.trim().to_string());
    }
    None
}

/// A runner receipt's `objects[]` as the input ledger's rows: the bucket
/// split off the key, every other field carried as it was.
fn ledger_objects(receipt: &Value) -> Vec<Value> {
    receipt
        .get("objects")
        .and_then(Value::as_array)
        .map(|objects| {
            objects
                .iter()
                .map(|object| {
                    let mut row = object.clone();
                    if let Some(key) = object.get("key").and_then(Value::as_str) {
                        if object.get("bucket").is_none() {
                            if let Some((bucket, rest)) = key.split_once('/') {
                                row["bucket"] = json!(bucket);
                                row["key"] = json!(rest);
                            }
                        }
                    }
                    row
                })
                .collect()
        })
        .unwrap_or_default()
}

fn run(args: &Args) -> Result<Value, String> {
    let file = open(&args.netcdf)?;
    let variable = file
        .variable(&args.variable)
        .ok_or_else(|| format!("{} has no variable {:?}", args.netcdf.display(), args.variable))?;
    let units = variable_text_attribute(&variable, "units").unwrap_or_default();
    if !units.eq_ignore_ascii_case("dbz") {
        return Err(format!(
            "{} {} is in {units:?}; frames are dBZ and nothing is converted here",
            args.netcdf.display(),
            args.variable
        ));
    }
    let dims: Vec<(String, usize)> =
        variable.dimensions().iter().map(|dim| (dim.name().to_string(), dim.len())).collect();
    if dims.len() != 3 && dims.len() != 4 {
        return Err(format!(
            "{} has dimensions {:?}; frames come from [time, y, x] or [time, member, y, x]",
            args.variable,
            dims.iter().map(|(name, _)| name.as_str()).collect::<Vec<_>>()
        ));
    }
    let (ny, nx) = (dims[dims.len() - 2].1, dims[dims.len() - 1].1);
    // The time axis is the leading dimension whose coordinate variable has CF time units.
    let mut time_axis = None;
    for (axis, (name, _)) in dims.iter().enumerate().take(dims.len() - 2) {
        if let Some(units) = coordinate_text_attribute(&file, name, "units") {
            if let Some(parsed) = cf_time_units(&units) {
                if time_axis.replace((axis, parsed)).is_some() {
                    return Err(format!("{} has two time-like dimensions", args.variable));
                }
            }
        }
    }
    let (time_axis, (seconds_per_unit, epoch)) = time_axis.ok_or_else(|| {
        format!(
            "{} has no time dimension: none of its leading dimensions has a coordinate variable with CF units \
             like \"hours since 2024-05-21 19:00:00\"",
            args.variable
        )
    })?;
    let member_axis = (0..dims.len() - 2).find(|axis| *axis != time_axis);
    let members = member_axis.map_or(1, |axis| dims[axis].1);
    let nt = dims[time_axis].1;
    if nt == 0 || members == 0 || ny < 2 || nx < 2 {
        return Err(format!("{} is {:?}; nothing to write", args.variable, dims));
    }
    let time_values = file
        .read_array_f64(&dims[time_axis].0)
        .map_err(|error| format!("read {}: {error}", dims[time_axis].0))?
        .into_values();
    let valid_times: Vec<DateTime<Utc>> = time_values
        .iter()
        .map(|value| {
            let seconds = value * seconds_per_unit;
            if !seconds.is_finite() || seconds.fract() != 0.0 {
                return Err(format!("time value {value} is not a whole number of seconds"));
            }
            Ok(epoch + Duration::seconds(seconds as i64))
        })
        .collect::<Result<_, _>>()?;

    let read_plane = |name: &str| -> Result<Vec<f64>, String> {
        let array = file.read_array_f64(name).map_err(|error| format!("read {name}: {error}"))?;
        let shape = array.shape().to_vec();
        let values = array.into_values();
        match shape.as_slice() {
            [rows, columns] if *rows == ny && *columns == nx => Ok(values),
            [count] if name == args.lat && *count == ny => {
                Ok((0..ny).flat_map(|j| std::iter::repeat_n(values[j], nx)).collect())
            }
            [count] if name == args.lon && *count == nx => Ok((0..ny).flat_map(|_| values.iter().copied()).collect()),
            other => Err(format!("{name} is {other:?}; the grid is {ny}x{nx}")),
        }
    };
    let lat = read_plane(&args.lat)?;
    let lon: Vec<f64> = read_plane(&args.lon)?.into_iter().map(|lon| (lon + 180.0).rem_euclid(360.0) - 180.0).collect();
    let lattice = match &args.lattice {
        LatticeRequest::Lambert(projection) => {
            Lattice::Lambert(lambert_lattice_from_latlon(&lat, &lon, ny, nx, projection)?)
        }
        LatticeRequest::LatLon => Lattice::LatLon(latlon_lattice_from_latlon(&lat, &lon, ny, nx)?),
    };

    let receipt: Option<Value> = match &args.receipt {
        Some(path) => Some(
            serde_json::from_slice(&std::fs::read(path).map_err(|error| format!("read {}: {error}", path.display()))?)
                .map_err(|error| format!("{}: {error}", path.display()))?,
        ),
        None => None,
    };
    let receipt_text = |key: &str| receipt.as_ref().and_then(|doc| doc.get(key)).and_then(Value::as_str).map(str::to_string);
    // The receipt and the file's own attributes may each say the issue time
    // and the causal flag; when both do, they must agree, or one of them
    // describes another run and the frames would carry the wrong lead
    // times or be scored as a forecast.  A flag is the operator's word.
    let receipt_issue = match receipt_text("issue_time") {
        Some(text) => Some(parse_loose_utc(&text)?),
        None => None,
    };
    let file_issue = match global_text(&file, "init") {
        Some(text) => Some(parse_loose_utc(&text)?),
        None => None,
    };
    if let (Some(from_receipt), Some(from_file)) = (receipt_issue, file_issue) {
        if from_receipt != from_file {
            return Err(format!(
                "the receipt's issue_time {} is not the file's init {}; one of them describes another run and \
                 every lead time would be labelled from the wrong issue",
                from_receipt.to_rfc3339(),
                from_file.to_rfc3339()
            ));
        }
    }
    let issue = match args.issue.or(receipt_issue).or(file_issue) {
        Some(issue) => issue,
        None => return Err("no issue time: give --issue, a receipt with issue_time, or an init attribute".into()),
    };
    let receipt_causal = receipt.as_ref().and_then(|doc| doc.get("causal")).and_then(Value::as_bool);
    let file_causal = match global_text(&file, "causal").map(|text| text.to_ascii_lowercase()) {
        Some(text) if text == "true" => Some(true),
        Some(text) if text == "false" => Some(false),
        _ => None,
    };
    if let (Some(from_receipt), Some(from_file)) = (receipt_causal, file_causal) {
        if from_receipt != from_file {
            return Err(format!(
                "the receipt says causal {from_receipt} and the file's causal attribute says {from_file}; a frame \
                 set that cannot say whether it saw the future could be scored as a forecast"
            ));
        }
    }
    let causal = match args.causal.or(receipt_causal).or(file_causal) {
        Some(causal) => causal,
        None => {
            return Err("no causal flag: a frame set must say whether it saw the future (--causal, a receipt \
                        with causal, or a causal attribute)"
                .into());
        }
    };
    let latest_input_time = match receipt_text("latest_input_time") {
        Some(text) => Some(parse_loose_utc(&text)?),
        None => None,
    };
    if let Some(declared) = receipt.as_ref().and_then(|doc| doc.get("members")).and_then(Value::as_u64) {
        if declared as usize != members {
            return Err(format!("the receipt says {declared} members and the file holds {members}"));
        }
    }

    let array = file
        .read_array::<f32>(&args.variable)
        .map_err(|error| format!("read {}: {error}", args.variable))?;
    let shape = array.shape().to_vec();
    if shape != dims.iter().map(|(_, len)| *len).collect::<Vec<_>>() {
        return Err(format!("{} read back as {shape:?}, not {dims:?}", args.variable));
    }
    let data: Vec<f32> = array.iter().copied().collect();
    drop(array);
    let fill: Vec<f64> = ["_FillValue", "missing_value"]
        .iter()
        .filter_map(|name| variable.attribute(name).and_then(|attribute| attribute.as_f64()))
        .filter(|value| value.is_finite())
        .collect();
    let mut strides = vec![1usize; shape.len()];
    for axis in (0..shape.len() - 1).rev() {
        strides[axis] = strides[axis + 1] * shape[axis + 1];
    }
    let plane = ny * nx;
    let mut planes: Vec<Vec<f32>> = Vec::with_capacity(nt);
    for t in 0..nt {
        let mut block = Vec::with_capacity(members * plane);
        for m in 0..members {
            let mut offset = t * strides[time_axis];
            if let Some(axis) = member_axis {
                offset += m * strides[axis];
            }
            block.extend(data[offset..offset + plane].iter().map(|value| {
                if fill.iter().any(|flag| (f64::from(*value) - flag).abs() <= 1.0e-3) { f32::NAN } else { *value }
            }));
        }
        planes.push(block);
    }
    drop(data);
    let frames: Vec<FrameData<'_>> =
        valid_times.iter().zip(&planes).map(|(valid, values)| FrameData { valid: *valid, values }).collect();

    let variant = receipt_text("variant").or_else(|| global_text(&file, "variant"));
    let package = args.package.clone().or_else(|| global_text(&file, "package")).or_else(|| variant.clone());
    let revision = args.revision.clone().or_else(|| receipt_text("revision"));
    let timing = json!({
        "step_s": receipt.as_ref().and_then(|doc| doc.get("step_s")).cloned(),
        "load_s": receipt.as_ref().and_then(|doc| doc.get("load_s")).cloned(),
    });
    let peak = receipt
        .as_ref()
        .and_then(|doc| doc.get("torch_peak_reserved_gib"))
        .and_then(Value::as_f64)
        .map(|gib| json!((gib * 1024.0 * 1024.0 * 1024.0).round() as u64))
        .unwrap_or(Value::Null);
    let provenance = FramesProvenance {
        source: json!({
            "kind": args.source_kind,
            "id": args.source_id,
            "package": package,
            "revision": revision,
            "code": {
                "variant": variant,
                "precision": receipt_text("precision").or_else(|| global_text(&file, "precision")),
            },
        }),
        sampler: Value::Null,
        seed: receipt
            .as_ref()
            .and_then(|doc| doc.get("seed"))
            .cloned()
            .or_else(|| global_text(&file, "seed").and_then(|text| text.parse::<i64>().ok()).map(Value::from))
            .unwrap_or(Value::Null),
        issue_time: issue,
        latest_input_time,
        causal,
        input_objects: receipt.as_ref().map(ledger_objects).unwrap_or_default(),
        history: json!({
            "no_coverage": {"applied": false,
                "note": "cells are as the source file holds them; a runner that masks unobserved history cells writes NaN there itself"},
            "causal_rule": receipt_text("causal_rule"),
            "violations": receipt.as_ref().and_then(|doc| doc.get("violations")).cloned(),
        }),
        timing,
        peak_device_bytes: peak,
        converter: json!({
            "tool": "rw_nowcast_frames from-netcdf",
            "source_rev": env!("GPUWM_BRIDGE_SOURCE_REV"),
            "netcdf": args.netcdf,
            "variable": args.variable,
            "dimensions": dims.iter().map(|(name, len)| json!({"name": name, "len": len})).collect::<Vec<_>>(),
            "receipt": args.receipt,
            "lattice_request": match &args.lattice {
                LatticeRequest::Lambert(_) => "lambert",
                LatticeRequest::LatLon => "latlon",
            },
        }),
    };
    let receipt_path = write_frames_root(&args.out, &lattice, members, &provenance, &frames)?;
    Ok(json!({
        "schema": "gpuwm-obs.nowcast-frames-convert.v1",
        "root": args.out,
        "receipt": receipt_path,
        "frames": frames.len(),
        "members": members,
        "issue_time": issue.to_rfc3339(),
        "first_valid": valid_times.first().map(|time| time.to_rfc3339()),
        "last_valid": valid_times.last().map(|time| time.to_rfc3339()),
        "causal": causal,
        "lattice": lattice_json(&lattice),
    }))
}

fn main() -> ExitCode {
    // Keep the stamp in the binary so a bundle cut can prove its revision.
    let _ = std::hint::black_box(GPUWM_BRIDGE_SOURCE_REV_STAMP);
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("--help") | Some("-h") | None => {
            println!("{USAGE}");
            return if args.is_empty() { ExitCode::from(2) } else { ExitCode::SUCCESS };
        }
        Some("--abi") => {
            println!("{ABI_MARKER}");
            return ExitCode::SUCCESS;
        }
        Some("from-netcdf") => {}
        Some(other) => {
            eprintln!("rw_nowcast_frames: unknown command {other:?}\n{USAGE}");
            return ExitCode::from(2);
        }
    }
    let parsed = match parse_args(&args[1..]) {
        Ok(parsed) => parsed,
        Err(error) => {
            eprintln!("rw_nowcast_frames: {error}");
            return ExitCode::from(2);
        }
    };
    match run(&parsed) {
        Ok(record) => {
            println!("{record}");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("rw_nowcast_frames: refused: {error}");
            ExitCode::from(3)
        }
    }
}
