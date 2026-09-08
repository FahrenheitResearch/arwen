//! Generic regular-grid Zarr extraction. Provider names belong in request metadata.
//! zarrs owns codecs/array decoding; Drew's NetCDF writer owns serialization and
//! mapped-engine owns the existing WRF humidity derivations.
use chrono::{DateTime, NaiveDate, NaiveDateTime};
use netcdf_writer::{AttrValue, NcFormat, NcType, NcWriter, Schema, VarData};
use serde::Deserialize;
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fs,
    io::{BufWriter, Write},
    ops::Range,
    path::Path,
    sync::Arc,
};
use zarrs::{
    array::Array,
    group::Group,
    storage::{ReadableStorage, ReadableStorageTraits},
};

type Result<T> = std::result::Result<T, Box<dyn Error>>;
type SourceArray = Array<dyn ReadableStorageTraits>;
const SCHEMA: &str = "arwen.regular-forcing.v1";

#[used]
pub static GPUWM_BRIDGE_SOURCE_REV_STAMP: &str =
    concat!("GPUWM_BRIDGE_SOURCE_REV=", env!("GPUWM_BRIDGE_SOURCE_REV"));

#[derive(Deserialize)]
struct Request {
    store: String,
    times: Vec<String>,
    area: [f64; 4], // south, west, north, east
    fields: Vec<Field>,
    expected_levels_hpa: Vec<f64>,
    #[serde(default)]
    derived: Vec<Derived>,
    #[serde(default)]
    specific_humidity_authority: bool,
}
#[derive(Deserialize)]
struct Field {
    source: String,
    output: String,
    units: String,
    pressure: bool,
}
#[derive(Deserialize)]
struct Derived {
    kind: String,
    output: String,
    units: String,
    #[serde(default)]
    inputs: Vec<String>,
}
fn fail<T>(message: impl Into<String>) -> Result<T> {
    Err(message.into().into())
}
fn moment(text: &str, end_of_day: bool) -> Result<NaiveDateTime> {
    if text.len() == 10 {
        let day = NaiveDate::parse_from_str(text, "%Y-%m-%d")?;
        return Ok(day
            .and_hms_opt(
                if end_of_day { 23 } else { 0 },
                if end_of_day { 59 } else { 0 },
                if end_of_day { 59 } else { 0 },
            )
            .unwrap());
    }
    let iso = text.trim().replace(' ', "T");
    if let Ok(t) = DateTime::parse_from_rfc3339(&iso) {
        return Ok(t.naive_utc());
    }
    Ok(NaiveDateTime::parse_from_str(
        iso.trim_end_matches('Z'),
        "%Y-%m-%dT%H:%M:%S%.f",
    )?)
}
fn open(store: &ReadableStorage, name: &str) -> Result<SourceArray> {
    Ok(Array::open(store.clone(), &format!("/{name}"))?)
}
fn numbers(array: &SourceArray, ranges: &[Range<u64>]) -> Result<Vec<f64>> {
    let metadata = serde_json::to_value(array.metadata())?;
    let dtype = metadata
        .get("dtype")
        .or_else(|| metadata.get("data_type"))
        .and_then(Value::as_str)
        .ok_or("array has no supported numeric dtype")?;
    let mut values: Vec<f64> = match dtype {
        "<f4" | ">f4" | "float32" => array
            .retrieve_array_subset::<Vec<f32>>(&ranges)?
            .into_iter()
            .map(f64::from)
            .collect(),
        "<f8" | ">f8" | "float64" => array.retrieve_array_subset::<Vec<f64>>(&ranges)?,
        "<i8" | ">i8" | "int64" => array
            .retrieve_array_subset::<Vec<i64>>(&ranges)?
            .into_iter()
            .map(|x| x as f64)
            .collect(),
        "<i4" | ">i4" | "int32" => array
            .retrieve_array_subset::<Vec<i32>>(&ranges)?
            .into_iter()
            .map(f64::from)
            .collect(),
        other => return fail(format!("unsupported source dtype {other}")),
    };
    let attrs = array.attributes();
    let number = |name: &str| attrs.get(name).and_then(Value::as_f64);
    let scale = number("scale_factor").unwrap_or(1.0);
    let offset = number("add_offset").unwrap_or(0.0);
    if !scale.is_finite() || scale == 0.0 || !offset.is_finite() {
        return fail("invalid source CF packing transform");
    }
    for value in &mut values {
        if !value.is_finite()
            || number("_FillValue").is_some_and(|f| f == *value)
            || number("missing_value").is_some_and(|f| f == *value)
        {
            *value = f64::NAN;
        } else {
            *value = *value * scale + offset;
        }
    }
    Ok(values)
}
fn units_match(array: &SourceArray, expected: &str, name: &str) -> Result<()> {
    // Early archive arrays omit some units; the provider's versioned mapping
    // supplies those. When present, source units must agree before relabeling.
    let Some(actual) = array.attributes().get("units") else {
        return Ok(());
    };
    let actual = actual
        .as_str()
        .ok_or("source units attribute must be text")?;
    let normal = actual
        .trim()
        .to_lowercase()
        .replace("**", "")
        .replace(['^', ' '], "");
    let aliases: &[&str] = match expected {
        "K" => &["k", "kelvin"],
        "Pa" => &["pa", "pascal", "pascals"],
        "hPa" => &["hpa", "millibar", "millibars", "mbar", "mb"],
        "m s-1" => &["ms-1", "m/s"],
        "m2 s-2" => &["m2s-2", "m2/s2"],
        "kg kg-1" => &["kgkg-1", "kg/kg", "1"],
        "m3 m-3" => &["m3m-3", "m3/m3"],
        "1" => &["1", "dimensionless", "(0-1)", "0-1", "proportion"],
        "m" => &[
            "m",
            "mofwaterequivalent",
            "mwaterequivalent",
            "metres",
            "meters",
        ],
        "degrees_north" => &["degrees_north", "degree_north", "degreesnorth"],
        "degrees_east" => &["degrees_east", "degree_east", "degreeseast"],
        _ => return fail(format!("unsupported declared quantity unit {expected}")),
    };
    if !aliases.contains(&normal.as_str()) {
        return fail(format!(
            "{name}: source units {actual:?} disagree with declared units {expected:?}"
        ));
    }
    Ok(())
}
fn coordinate(store: &ReadableStorage, name: &str) -> Result<(SourceArray, Vec<f64>)> {
    let array = open(store, name)?;
    if array.shape().len() != 1 {
        return fail(format!("{name} coordinate must be one dimensional"));
    }
    let values = numbers(&array, &[0..array.shape()[0]])?;
    if values.is_empty() || values.iter().any(|x| !x.is_finite()) {
        return fail(format!("{name} coordinate is empty or nonfinite"));
    }
    Ok((array, values))
}
fn dimensions(array: &SourceArray) -> Result<Vec<String>> {
    let metadata = serde_json::to_value(array.metadata())?;
    let names = array
        .attributes()
        .get("_ARRAY_DIMENSIONS")
        .or_else(|| metadata.get("dimension_names"))
        .and_then(Value::as_array)
        .ok_or("source array lacks explicit dimension names")?;
    names
        .iter()
        .map(|v| {
            v.as_str()
                .map(str::to_owned)
                .ok_or_else(|| "source dimension name is not a string".into())
        })
        .collect()
}
fn regular(values: &[f64], name: &str) -> Result<f64> {
    if values.len() < 2 {
        return fail(format!("{name} needs at least two coordinates"));
    }
    let step = values[1] - values[0];
    if step == 0.0
        || values
            .windows(2)
            .any(|x| ((x[1] - x[0]) - step).abs() > step.abs() * 1e-5)
    {
        return fail(format!("{name} is not a strictly regular coordinate"));
    }
    Ok(step)
}
fn write_field(
    writer: &mut NcWriter,
    record: u64,
    ids: (usize, usize),
    values: &[f64],
) -> Result<()> {
    writer.write_record(record, ids.0, VarData::F64(values))?;
    let missing: Vec<i8> = values.iter().map(|x| i8::from(!x.is_finite())).collect();
    writer.write_record(record, ids.1, VarData::I8(&missing))?;
    Ok(())
}
fn define_field(
    schema: &mut Schema,
    name: &str,
    units: &str,
    source: &str,
    dims: &[usize],
) -> Result<(usize, usize)> {
    let id = schema.def_var(name, NcType::Double, dims)?;
    schema.put_var_attr(id, "units", AttrValue::Text(units.into()))?;
    schema.put_var_attr(id, "source_variable", AttrValue::Text(source.into()))?;
    schema.put_var_attr(id, "arwen_field", AttrValue::Text(name.into()))?;
    let mask = schema.def_var(&format!("missing__{name}"), NcType::Byte, dims)?;
    schema.put_var_attr(mask, "units", AttrValue::Text("1".into()))?;
    schema.put_var_attr(
        id,
        "arwen_missing_mask",
        AttrValue::Text(format!("missing__{name}")),
    )?;
    Ok((id, mask))
}
fn extract(request: Request, output: &Path) -> Result<Value> {
    if output.exists() {
        return fail(format!("output already exists: {}", output.display()));
    }
    if request.times.is_empty() || request.fields.is_empty() {
        return fail("request needs times and fields");
    }
    let times: Vec<_> = request
        .times
        .iter()
        .map(|t| moment(t, false))
        .collect::<Result<_>>()?;
    if times.windows(2).any(|t| t[0] >= t[1]) {
        return fail("requested times must be unique and increasing");
    }
    let store: ReadableStorage = if request.store.starts_with("https://") {
        Arc::new(zarrs_http::HTTPStore::new(&request.store)?)
    } else {
        Arc::new(zarrs::filesystem::FilesystemStore::new(&request.store)?)
    };
    let group = Group::open(store.clone(), "/")?;
    let valid_start = group
        .attributes()
        .get("valid_time_start")
        .and_then(Value::as_str)
        .ok_or("store lacks valid_time_start authority")?;
    let valid_stop = group
        .attributes()
        .get("valid_time_stop")
        .and_then(Value::as_str)
        .ok_or("store lacks valid_time_stop authority")?;
    let start = moment(valid_start, false)?;
    let finalized_stop = moment(valid_stop, true)?;
    let preliminary_stop = group
        .attributes()
        .get("valid_time_stop_era5t")
        .and_then(Value::as_str)
        .map(|text| moment(text, true))
        .transpose()?
        .unwrap_or(finalized_stop);
    let stop = finalized_stop.max(preliminary_stop);
    let provisional_times: Vec<_> = times
        .iter()
        .filter(|t| **t > finalized_stop)
        .map(ToString::to_string)
        .collect();
    if !provisional_times.is_empty() {
        eprintln!(
            "Zarr: {} requested time(s) use preliminary ERA5T data after {valid_stop}",
            provisional_times.len()
        );
    }
    if times.iter().any(|t| *t < start || *t > stop) {
        return fail(format!(
            "requested times exceed declared source coverage {start} through {stop}"
        ));
    }
    let (time_array, source_time) = coordinate(&store, "time")?;
    let units = time_array
        .attributes()
        .get("units")
        .and_then(Value::as_str)
        .ok_or("time lacks CF units")?;
    if let Some(calendar) = time_array
        .attributes()
        .get("calendar")
        .and_then(Value::as_str)
    {
        if !["standard", "gregorian", "proleptic_gregorian"].contains(&calendar) {
            return fail("unsupported CF calendar");
        }
    }
    let (unit, epoch) = units
        .split_once(" since ")
        .ok_or("unsupported CF time units")?;
    let factor = match unit {
        "hours" | "hour" => 3600.0,
        "seconds" | "second" => 1.0,
        "days" | "day" => 86400.0,
        _ => return fail("unsupported CF time unit"),
    };
    let epoch = moment(epoch, false)?.and_utc().timestamp();
    let mut time_indices = Vec::new();
    for t in &times {
        let expected = (t.and_utc().timestamp() - epoch) as f64 / factor;
        let matches: Vec<_> = source_time
            .iter()
            .enumerate()
            .filter(|(_, x)| (**x - expected).abs() < 1e-8)
            .map(|(i, _)| i as u64)
            .collect();
        if matches.len() != 1 {
            return fail(format!(
                "requested time {t} does not resolve to one source coordinate"
            ));
        }
        time_indices.push(matches[0]);
    }
    let (lat_array, latitude) = coordinate(&store, "latitude")?;
    let (lon_array, longitude) = coordinate(&store, "longitude")?;
    let (level_array, levels) = coordinate(&store, "level")?;
    units_match(&lat_array, "degrees_north", "latitude")?;
    units_match(&lon_array, "degrees_east", "longitude")?;
    units_match(&level_array, "hPa", "level")?;
    let dy = regular(&latitude, "latitude")?.abs();
    let dx = regular(&longitude, "longitude")?;
    if dx <= 0.0 || (dx * longitude.len() as f64 - 360.0).abs() > 1e-4 {
        return fail("longitude must be an increasing global cyclic axis");
    }
    let mut sorted_levels = levels.clone();
    sorted_levels.sort_by(f64::total_cmp);
    let mut expected = request.expected_levels_hpa.clone();
    expected.sort_by(f64::total_cmp);
    if sorted_levels != expected || sorted_levels.windows(2).any(|p| p[0] == p[1]) {
        return fail("source pressure levels differ from the declared complete inventory");
    }
    let mut level_indices: Vec<_> = (0..levels.len()).collect();
    level_indices.sort_by(|a, b| levels[*b].total_cmp(&levels[*a]));
    let out_levels: Vec<_> = level_indices.iter().map(|i| levels[*i]).collect();
    let [south, west, north, east] = request.area;
    if request.area.iter().any(|x| !x.is_finite())
        || south < -90.0
        || north > 90.0
        || south >= north
    {
        return fail("invalid requested area");
    }
    let ys: Vec<_> = latitude
        .iter()
        .enumerate()
        .filter(|(_, y)| **y >= south - dy && **y <= north + dy)
        .map(|(i, _)| i)
        .collect();
    if ys.len() < 2 {
        return fail("requested latitude window has fewer than two source rows");
    }
    let y0 = *ys.first().unwrap();
    let y1 = *ys.last().unwrap() + 1;
    let out_lat = latitude[y0..y1].to_vec();
    if out_lat.iter().copied().fold(f64::INFINITY, f64::min) > south
        || out_lat.iter().copied().fold(f64::NEG_INFINITY, f64::max) < north
    {
        return fail("source latitude does not cover requested area");
    }
    let mut width = (east - west).rem_euclid(360.0);
    if width == 0.0 && east != west {
        width = 360.0;
    }
    if width <= 0.0 {
        return fail("requested longitude span is empty");
    }
    let x_start = ((west - longitude[0]) / dx).floor() as i64 - 1;
    let nx = (((west + width - longitude[0]) / dx).ceil() as i64 + 1 - x_start + 1)
        .min(longitude.len() as i64) as usize;
    let xs: Vec<_> = (0..nx)
        .map(|i| (x_start + i as i64).rem_euclid(longitude.len() as i64) as usize)
        .collect();
    let out_lon: Vec<_> = (0..nx)
        .map(|i| longitude[0] + (x_start + i as i64) as f64 * dx)
        .collect();
    let ny = out_lat.len();
    let plane = ny * nx;
    let mut schema = Schema::new(NcFormat::Offset64);
    schema.put_global_attr("Conventions", AttrValue::Text("CF-1.8".into()))?;
    schema.put_global_attr(
        "arwen_regular_forcing_schema",
        AttrValue::Text(SCHEMA.into()),
    )?;
    schema.put_global_attr(
        "specific_humidity_authority",
        AttrValue::Text(
            if request.specific_humidity_authority {
                "direct"
            } else {
                "unspecified"
            }
            .into(),
        ),
    )?;
    schema.put_global_attr("source_store", AttrValue::Text(request.store.clone()))?;
    schema.put_global_attr(
        "source_valid_time_start",
        AttrValue::Text(valid_start.into()),
    )?;
    schema.put_global_attr("source_valid_time_stop", AttrValue::Text(valid_stop.into()))?;
    schema.put_global_attr(
        "source_provisional_times",
        AttrValue::Text(serde_json::to_string(&provisional_times)?),
    )?;
    let td = schema.def_dim("time", 0, true)?;
    let pd = schema.def_dim("level", levels.len(), false)?;
    let yd = schema.def_dim("latitude", ny, false)?;
    let xd = schema.def_dim("longitude", nx, false)?;
    let mut coordinates = Vec::new();
    for (name, dim, units) in [
        ("time", td, "seconds since 1970-01-01 00:00:00"),
        ("level", pd, "hPa"),
        ("latitude", yd, "degrees_north"),
        ("longitude", xd, "degrees_east"),
    ] {
        let id = schema.def_var(name, NcType::Double, &[dim])?;
        schema.put_var_attr(id, "units", AttrValue::Text(units.into()))?;
        coordinates.push(id);
    }
    let mut arrays = Vec::new();
    let mut field_ids = BTreeMap::new();
    for field in &request.fields {
        let array = open(&store, &field.source)?;
        units_match(&array, &field.units, &field.source)?;
        let dims = dimensions(&array)?;
        let expected: Vec<&str> = if field.pressure {
            vec!["time", "level", "latitude", "longitude"]
        } else if dims.len() == 2 {
            vec!["latitude", "longitude"]
        } else {
            vec!["time", "latitude", "longitude"]
        };
        if dims.iter().map(String::as_str).collect::<Vec<_>>() != expected {
            return fail(format!(
                "{} has unexpected dimension order {:?}",
                field.source, dims
            ));
        }
        let expected_shape: Vec<u64> = expected
            .iter()
            .map(|d| match *d {
                "time" => source_time.len() as u64,
                "level" => levels.len() as u64,
                "latitude" => latitude.len() as u64,
                _ => longitude.len() as u64,
            })
            .collect();
        if array.shape() != expected_shape {
            return fail(format!(
                "{} shape does not match its coordinate axes",
                field.source
            ));
        }
        let dims = if field.pressure {
            vec![td, pd, yd, xd]
        } else {
            vec![td, yd, xd]
        };
        let ids = define_field(
            &mut schema,
            &field.output,
            &field.units,
            &field.source,
            &dims,
        )?;
        if field_ids.insert(field.output.clone(), ids).is_some() {
            return fail("duplicate output field");
        }
        arrays.push(array);
    }
    for item in &request.derived {
        let dims = match item.kind.as_str() {
            "pressure_from_levels" => vec![td, pd, yd, xd],
            "specific_humidity_from_dewpoint" => vec![td, yd, xd],
            _ => return fail(format!("unknown native derivation {}", item.kind)),
        };
        let ids = define_field(&mut schema, &item.output, &item.units, &item.kind, &dims)?;
        if field_ids.insert(item.output.clone(), ids).is_some() {
            return fail("duplicate derived output field");
        }
    }
    let dependencies: BTreeSet<_> = request
        .derived
        .iter()
        .flat_map(|d| d.inputs.iter().cloned())
        .collect();
    let mut writer = NcWriter::create(output, schema)?;
    writer.write_var(coordinates[1], VarData::F64(&out_levels))?;
    writer.write_var(coordinates[2], VarData::F64(&out_lat))?;
    writer.write_var(coordinates[3], VarData::F64(&out_lon))?;
    for (record, ti) in time_indices.iter().enumerate() {
        writer.write_record(
            record as u64,
            coordinates[0],
            VarData::F64(&[times[record].and_utc().timestamp() as f64]),
        )?;
        let mut retained = BTreeMap::new();
        for (field, array) in request.fields.iter().zip(&arrays) {
            eprintln!(
                "Zarr: {} {}/{} {}",
                times[record],
                record + 1,
                times.len(),
                field.source
            );
            let mut ranges = Vec::new();
            if array.shape().len() > 2 {
                ranges.push(*ti..*ti + 1);
            }
            if field.pressure {
                ranges.push(0..levels.len() as u64);
            }
            ranges.extend([y0 as u64..y1 as u64, 0..longitude.len() as u64]);
            let source = numbers(array, &ranges)?;
            let mut values =
                Vec::with_capacity(plane * if field.pressure { levels.len() } else { 1 });
            for zi in if field.pressure {
                level_indices.clone()
            } else {
                vec![0]
            } {
                for yi in 0..ny {
                    for xi in &xs {
                        values.push(source[(zi * ny + yi) * longitude.len() + xi]);
                    }
                }
            }
            write_field(
                &mut writer,
                record as u64,
                field_ids[&field.output],
                &values,
            )?;
            if dependencies.contains(&field.output) {
                retained.insert(field.output.clone(), values);
            }
        }
        for item in &request.derived {
            let values: Vec<f64> = match item.kind.as_str() {
                "pressure_from_levels" => out_levels
                    .iter()
                    .flat_map(|p| std::iter::repeat_n(*p * 100.0, plane))
                    .collect(),
                "specific_humidity_from_dewpoint" => {
                    if item.inputs.len() != 3 {
                        return fail("dewpoint humidity needs ordered dewpoint, temperature, pressure inputs");
                    }
                    let input: Vec<_> = item
                        .inputs
                        .iter()
                        .map(|n| {
                            retained
                                .get(n)
                                .ok_or_else(|| format!("missing derivation input {n}"))
                        })
                        .collect::<std::result::Result<_, _>>()?;
                    if input
                        .iter()
                        .any(|x| x.len() != plane || x.iter().any(|v| !v.is_finite()))
                    {
                        return fail(
                            "surface humidity inputs must be complete finite surface fields",
                        );
                    }
                    let rh = mapped_engine::derive::surface_relative_humidity(input[0], input[1]);
                    mapped_engine::derive::saturation_mixing_ratio(input[1], input[2], &rh)
                        .into_iter()
                        .map(|r| r / (1.0 + r))
                        .collect()
                }
                _ => unreachable!(),
            };
            write_field(&mut writer, record as u64, field_ids[&item.output], &values)?;
        }
    }
    writer.finish()?;
    Ok(
        json!({"schema": SCHEMA, "times": request.times, "levels_hpa":out_levels,
        "latitude_bounds":[out_lat.iter().copied().fold(f64::INFINITY,f64::min),out_lat.iter().copied().fold(f64::NEG_INFINITY,f64::max)],
        "longitude_bounds":[out_lon[0],out_lon[nx-1]], "shape":[times.len(),levels.len(),ny,nx],
        "fields":field_ids.keys().collect::<Vec<_>>(), "provisional_times":provisional_times, "source_metadata":group.attributes()}),
    )
}
fn dump_record(args: &[String]) -> Result<()> {
    if args.len() < 6 { return fail("usage: rw_zarr dump-record FILE INDEX OUTDIR VARIABLE..."); }
    let file = netcrust::File::open(&args[2])?;
    if file.attribute("arwen_regular_forcing_schema").and_then(|a| a.as_string().map(str::to_owned)).as_deref() != Some(SCHEMA) {
        return fail("native record read requires the regular forcing schema");
    }
    let index: u64 = args[3].parse()?;
    let out = Path::new(&args[4]);
    fs::create_dir_all(out)?;
    let mut variables = Vec::new();
    for (slot, name) in args[5..].iter().enumerate() {
        let values = file.read_array_f64_record_or_all(name,index)?;
        let leaf = format!("variable-{slot:04}.bin");
        let mut writer = BufWriter::new(fs::OpenOptions::new().create_new(true).write(true).open(out.join(&leaf))?);
        for value in values.values() { writer.write_all(&value.to_le_bytes())?; }
        writer.flush()?;
        variables.push(json!({"name":name,"file":leaf,"shape":values.shape(),"dtype":"<f8"}));
    }
    println!("{}",json!({"schema":"arwen.regular-forcing-record.v1","index":index,"variables":variables}));
    Ok(())
}
fn run() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if args.get(1).map(String::as_str)==Some("dump-record") { return dump_record(&args); }
    if args.len() != 4 || args[1] != "extract" {
        return fail("usage: rw_zarr extract REQUEST.json OUTPUT.nc");
    }
    let request: Request = serde_json::from_slice(&fs::read(&args[2])?)?;
    let result = extract(request, Path::new(&args[3]))?;
    println!("{}", serde_json::to_string(&result)?);
    Ok(())
}
fn main() {
    let _ = std::hint::black_box(GPUWM_BRIDGE_SOURCE_REV_STAMP);
    if let Err(error) = run() {
        eprintln!("rw_zarr: {error}");
        std::process::exit(1);
    }
}
